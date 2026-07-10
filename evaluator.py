#!/usr/bin/env python3
"""
NNS Competition Evaluator
=========================
Evaluates student submissions for the nearest neighbor search competition.

Each student provides a Docker image that inherits from nns-competition/base.
Students implement algorithm.py (fit / query / get_n_distances) and
scenarios.yaml (one entry per parameter configuration to evaluate).

Usage
-----
  python evaluator.py evaluate --team alice --image alice/nns:latest \\
                                --dataset data/sift-128-euclidean.hdf5
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sqlite3
import tarfile
import tempfile
import threading
import time
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from icecream import ic

import sys
import docker
import docker.errors
import h5py
import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DB      = "results.db"
DEFAULT_TIMEOUT = 1800          # seconds per container run
CONTAINER_DATA_MOUNT    = "/competition/data"
CONTAINER_RESULTS_MOUNT = "/competition/results"
MEM_POLL_INTERVAL = 0.5        # seconds between memory stat polls

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

_CREATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    team_name           TEXT    NOT NULL,
    docker_image        TEXT    NOT NULL,
    dataset             TEXT    NOT NULL,
    scenario            TEXT    NOT NULL,
    timestamp           TEXT    NOT NULL,
    status              TEXT    NOT NULL,   -- 'success' | 'failed' | 'timeout'
    error_message       TEXT,
    build_time_s        REAL,
    total_query_time_s  REAL,
    qps                 REAL,
    peak_mem_mb         REAL,    -- container-level peak RSS (cgroup)
    index_mem_mb        REAL,    -- difference post index - pre index peak RSS 
    n_dist_queries      INTEGER,
    avg_recall          REAL,
    extra_metrics       TEXT     -- JSON blob
);

CREATE TABLE IF NOT EXISTS detail (
    run_id              INTEGER NOT NULL,
    query_index         INTEGER NOT NULL,
    query_time_s        REAL,
    query_recall        REAL,

    FOREIGN KEY(run_id) REFERENCES runs(id)
);
"""


def open_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(_CREATE_SCHEMA)
    conn.commit()
    return conn


def insert_run(conn: sqlite3.Connection, row: dict) -> int:
    cur = conn.execute(
        """
        INSERT INTO runs (
            team_name, docker_image, dataset, scenario, timestamp, status, error_message,
            build_time_s, total_query_time_s, qps,
            peak_mem_mb, index_mem_mb, n_dist_queries,
            avg_recall,
            extra_metrics
        ) VALUES (
            :team_name, :docker_image, :dataset, :scenario, :timestamp, :status, :error_message,
            :build_time_s, :total_query_time_s, :qps,
            :peak_mem_mb, :index_mem_mb, :n_dist_queries,
            :avg_recall,
            :extra_metrics
        )
        """,
        row,
    )
    conn.commit()
    return cur.lastrowid


def run_exists(
    conn: sqlite3.Connection,
    team_name: str,
    docker_image: str,
    dataset: str,
    scenario: str,
) -> bool:
    """Return True if a successful run for this configuration is already recorded."""
    cur = conn.execute(
        """
        SELECT 1 FROM runs
        WHERE team_name = ? AND docker_image = ? AND dataset = ?
              AND scenario = ? AND status = 'success'
        LIMIT 1
        """,
        (team_name, docker_image, dataset, scenario),
    )
    return cur.fetchone() is not None


def insert_detail(conn: sqlite3.Connection, run_id: int, times: np.ndarray, all_recalls: np.ndarray) -> None:
    rows = [
        {"run_id": run_id, "query_index": i, "query_time_s": times[i], "query_recall": all_recalls[i]}
        for i in range(times.shape[0])
    ]
    conn.executemany(
        """
        INSERT INTO detail (
            run_id, query_index, query_time_s, query_recall
        ) VALUES (
            :run_id, :query_index, :query_time_s, :query_recall
        )
        """,
        rows
    )
    conn.commit()


def _empty_row(team, image, dataset, scenario, timestamp) -> dict:
    return dict(
        team_name=team, docker_image=image, dataset=dataset,
        scenario=scenario, timestamp=timestamp,
        status="failed", error_message=None,
        build_time_s=None, total_query_time_s=None, qps=None,
        peak_mem_mb=None, index_mem_mb=None, n_dist_queries=None,
        avg_recall=None,
        extra_metrics=None,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def recalls(true_distances: np.ndarray, predicted_distances: np.ndarray, k: int) -> np.ndarray:
    def compute_recall(td, pd):
        pd = pd[:k]
        threshold = td[k-1]
        return np.mean(pd <= threshold)
    return np.array([
        compute_recall(true_distances[i], predicted_distances[i])
        for i in range(true_distances.shape[0])
    ])


# ---------------------------------------------------------------------------
# Scenario discovery
# ---------------------------------------------------------------------------

def extract_scenarios_yaml(client: docker.DockerClient, image: str) -> list[str]:
    """
    Extract /app/scenarios.yaml from the image without starting a container.

    Returns a list of scenario names only – params are resolved at runtime by
    the harness once it knows the dataset name.  Validates overall structure
    so badly-formed submissions fail fast before any containers are started.

    Expected schema:
      scenarios:
        <scenario_name>:
          default:             # required fallback
            index_params: {}
            query_params: {}
          <dataset_name>:      # optional dataset-specific override
            index_params: {}
            query_params: {}
    """
    log = logging.getLogger(__name__)
    container = client.containers.create(image)
    try:
        stream, _ = container.get_archive("/app/scenarios.yaml")
        buf = io.BytesIO(b"".join(stream))
        with tarfile.open(fileobj=buf) as tar:
            raw = tar.extractfile(tar.getmembers()[0]).read()
        parsed = yaml.safe_load(raw)
    finally:
        container.remove(force=True)

    if "scenarios" not in parsed or not isinstance(parsed["scenarios"], dict):
        raise ValueError(f"scenarios.yaml must contain a top-level 'scenarios' mapping. Got instead:\n {raw.decode()}")

    scenario_names = []
    for sname, sval in parsed["scenarios"].items():
        if not isinstance(sname, str) or not sname.isidentifier():
            raise ValueError(
                f"Scenario name {sname!r} is invalid; "
                "must be a non-empty string usable as a Python identifier."
            )
        sval = sval or {}
        if not isinstance(sval, dict):
            raise ValueError(f"Scenario {sname!r} must be a mapping.")
        if "default" not in sval:
            raise ValueError(
                f"Scenario {sname!r} is missing a 'default' block. "
                "Every scenario must have a 'default' fallback."
            )
        for block_name, block in sval.items():
            block = block or {}
            for key in ("index_params", "query_params"):
                if key in block and not isinstance(block[key], (dict, type(None))):
                    raise ValueError(
                        f"Scenario {sname!r}, block {block_name!r}: "
                        f"'{key}' must be a mapping."
                    )
        scenario_names.append(sname)

    if not scenario_names:
        raise ValueError("scenarios.yaml defines no scenarios.")

    log.info("Discovered %d scenario(s): %s", len(scenario_names), ", ".join(scenario_names))
    return scenario_names


# ---------------------------------------------------------------------------
# Memory polling
# ---------------------------------------------------------------------------

class PeakMemoryMonitor:
    """
    Polls container memory stats in a background thread.
    Uses Docker's cgroup-based max_usage (true peak RSS, includes C heap).
    """

    def __init__(self, container, interval: float = MEM_POLL_INTERVAL):
        self._container = container
        self._interval  = interval
        self._peak_mb   = 0.0
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self) -> float:
        self._stop.set()
        self._thread.join()
        return self._peak_mb

    def _run(self):
        while not self._stop.is_set():
            try:
                stats = self._container.stats(stream=False)
                usage = stats.get("memory_stats", {}).get("usage", 0)
                self._peak_mb = max(self._peak_mb, usage / (1024 ** 2))
            except Exception:
                pass
            self._stop.wait(self._interval)


# ---------------------------------------------------------------------------
# Single-scenario container run
# ---------------------------------------------------------------------------

def run_scenario_container(
    client: docker.DockerClient,
    image: str,
    data_dir: str,
    results_dir: str,
    dataset_filename: str,
    dataset_name: str,
    scenario_name: str,
    k: int,
    timeout: int,
) -> dict:
    """
    Run one container for one scenario.
    Returns: {status, peak_mem_mb, wall_time, error (optional)}
    """
    log = logging.getLogger(__name__)

    volumes = {
        str(Path(data_dir).resolve()): {
            "bind": CONTAINER_DATA_MOUNT, "mode": "ro",
        },
        str(Path(results_dir).resolve()): {
            "bind": CONTAINER_RESULTS_MOUNT, "mode": "rw",
        },
    }
    environment = {
        "DATASET_PATH":  f"{CONTAINER_DATA_MOUNT}/{dataset_filename}",
        "RESULTS_PATH":  f"{CONTAINER_RESULTS_MOUNT}/results.hdf5",
        "SCENARIO_NAME": scenario_name,
        "DATASET_NAME":  dataset_name,
        "QUERY_K":       str(k),
    }

    container = None
    t0 = time.monotonic()
    try:
        container = client.containers.run(
            image,
            detach=True,
            volumes=volumes,
            environment=environment,
            # mem_limit=MEMORY_LIMIT,
            network_disabled=True,
            remove=False,
        )
        log.info("Container %s started  [scenario=%s]", container.short_id, scenario_name)

        monitor = PeakMemoryMonitor(container)
        monitor.start()

        try:
            result = container.wait(timeout=timeout)
            exit_code = result["StatusCode"]
        except Exception as exc:
            container.kill()
            peak_mb = monitor.stop()
            wall = time.monotonic() - t0
            log.warning("Container timed out after %.0fs", wall)
            return {"status": "timeout", "wall_time": wall,
                    "peak_mem_mb": peak_mb, "error": str(exc)}

        peak_mb = monitor.stop()
        wall = time.monotonic() - t0

        logs_tail = container.logs(
            stdout=True, stderr=True
        ).decode("utf-8", errors="replace")[-3000:]
        print(logs_tail)

        if exit_code != 0:
            log.error("Container exited %d\n%s", exit_code, logs_tail)
            return {
                "status": "failed", "wall_time": wall, "peak_mem_mb": peak_mb,
                "error": f"Exit code {exit_code}\n---\n{logs_tail}",
            }

        log.info("Container done in %.1fs  peak_mem=%.0fMB", wall, peak_mb)
        return {"status": "success", "wall_time": wall, "peak_mem_mb": peak_mb}

    finally:
        if container:
            try:
                container.remove(force=True)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Core evaluation pipeline
# ---------------------------------------------------------------------------

def evaluate(
    conn: sqlite3.Connection,
    client: docker.DockerClient,
    team_name: str,
    docker_image: str,
    dataset_path: str,
    k: int = 100,
    timeout: int = DEFAULT_TIMEOUT,
    scenarios: list[str] | None = None,
) -> list[dict]:
    log = logging.getLogger(__name__)
    dataset_path = Path(dataset_path)
    dataset_name = dataset_path.stem
    timestamp    = datetime.now(timezone.utc).isoformat()

    log.info("=" * 60)
    log.info("Team: %s | Image: %s | Dataset: %s", team_name, docker_image, dataset_name)

    # -- Discover scenarios from the image --
    try:
        available = extract_scenarios_yaml(client, docker_image)
    except Exception as exc:
        log.warning(f"Could not read scenarios.yaml from image: {exc}")
        row = _empty_row(team_name, docker_image, dataset_name, "__no_scenarios__", timestamp)
        row["error_message"] = f"Could not read scenarios.yaml from image: {exc}"
        insert_run(conn, row)
        return [row]

    results = []

    # -- Restrict to the requested scenarios, if any --
    if scenarios is None:
        scenarios = available
    else:
        selected = []
        for scenario in scenarios:
            if scenario not in available:
                log.warning(
                    "Scenario %r not defined by image %s (available: %s)",
                    scenario, docker_image, ", ".join(available),
                )
                row = _empty_row(team_name, docker_image, dataset_name, scenario, timestamp)
                row["error_message"] = (
                    f"Scenario {scenario!r} not defined by image. "
                    f"Available scenarios: {', '.join(available)}"
                )
                insert_run(conn, row)
                results.append(row)
            else:
                selected.append(scenario)
        scenarios = selected

    # -- Load ground truth once --
    log.info("Loading ground-truth from %s ...", dataset_path)
    try:
        with h5py.File(dataset_path, "r") as f:
            true_neighbors = f["neighbors"][:]
            n_test = f["test"].shape[0]
    except Exception as exc:
        row = _empty_row(team_name, docker_image, dataset_name, "__load_failed__", timestamp)
        row["error_message"] = f"Failed to load dataset: {exc}"
        insert_run(conn, row)
        results.append(row)
        return results

    for scenario_name in scenarios:
        if run_exists(conn, team_name, docker_image, dataset_name, scenario_name):
            log.info(
                "--- Scenario: %s --- already in database, skipping",
                scenario_name,
            )
            continue

        log.info("--- Scenario: %s ---", scenario_name)
        row = _empty_row(team_name, docker_image, dataset_name, scenario_name, timestamp)

        with tempfile.TemporaryDirectory(prefix="nns_eval_") as tmpdir:
            run_result = run_scenario_container(
                client=client,
                image=docker_image,
                data_dir=str(dataset_path.parent),
                results_dir=tmpdir,
                dataset_filename=dataset_path.name,
                dataset_name=dataset_name,
                scenario_name=scenario_name,
                k=k,
                timeout=timeout,
            )

            row["peak_mem_mb"] = run_result.get("peak_mem_mb")

            if run_result["status"] != "success":
                row["status"]        = run_result["status"]
                row["error_message"] = run_result.get("error")
                run_id = insert_run(conn, row)
                log.warning("Scenario %s: %s (id=%d)", scenario_name, row["status"], run_id)
                results.append(row)
                continue

            # -- Parse results.hdf5 --
            results_file = Path(tmpdir) / "results.hdf5"
            if not results_file.exists():
                row["error_message"] = (
                    "results.hdf5 was not written. "
                    "Ensure the harness entrypoint is not overridden."
                )
                insert_run(conn, row)
                results.append(row)
                continue

            try:
                with h5py.File(results_file, "r") as f:
                    pred_neighbors  = f["neighbors"][:]
                    build_time      = float(f["build_time"][()])
                    query_times_s   = f["query_times"][:]
                    n_dist_queries  = int(f["n_dist_queries"][()])
                    index_mem_mb  = int(f["index_mem_mb"][()])
            except Exception as exc:
                row["error_message"] = f"Failed to parse results.hdf5: {exc}"
                insert_run(conn, row)
                results.append(row)
                continue

            # Compute distances
            with h5py.File(dataset_path) as f:
                data = f["train"][:]
                queries = f["test"][:]
                true_distances = f["distances"][:]
                predicted_distances = np.array(
                    [
                        np.linalg.norm(data[pred_neighbors[i]] - queries[i], axis=1)
                        for i in range(queries.shape[0])
                    ]
                )
            
            # Shape checks
            if pred_neighbors.shape[0] != n_test:
                row["error_message"] = (
                    f"'neighbors' has {pred_neighbors.shape[0]} rows, expected {n_test}."
                )
                insert_run(conn, row)
                results.append(row)
                continue
            if predicted_distances.shape[0] != n_test:
                row["error_message"] = (
                    f"'predicted_distances' has {predicted_distances.shape[0]} rows, expected {n_test}."
                )
                insert_run(conn, row)
                results.append(row)
                continue
            if query_times_s.shape[0] != n_test:
                row["error_message"] = (
                    f"'query_times' has {query_times_s.shape[0]} entries, expected {n_test}."
                )
                insert_run(conn, row)
                results.append(row)
                continue

            # -- Metrics --
            all_recalls = recalls(true_distances, predicted_distances, k)
            avg_recall = all_recalls.mean()
            total_qt_s = float(query_times_s.sum())
            qps      = n_test / total_qt_s
            lat_ms   = query_times_s * 1e3
            ic(total_qt_s, qps, n_test)

            row.update(
                status="success",
                build_time_s=build_time,
                total_query_time_s=total_qt_s,
                qps=qps,
                n_dist_queries=n_dist_queries,
                index_mem_mb=index_mem_mb,
                avg_recall=avg_recall,
                extra_metrics=json.dumps({
                    "latency_ms": {
                        "mean":   float(np.mean(lat_ms)),
                        "median": float(np.median(lat_ms)),
                        "p95":    float(np.percentile(lat_ms, 95)),
                        "p99":    float(np.percentile(lat_ms, 99)),
                        "p999":   float(np.percentile(lat_ms, 99.9)),
                        "max":    float(np.max(lat_ms)),
                    },
                }),
            )

            log.info(
                "RESULT [%s]  QPS=%.1f  build=%.2fs  "
                "peak=%.0fMB  index_mem=%dMB "
                "dist_query=%d  avg_recall=%s",
                scenario_name, qps, build_time,
                row["peak_mem_mb"] or 0, index_mem_mb,
                n_dist_queries, avg_recall,
            )

        run_id = insert_run(conn, row)
        insert_detail(conn, run_id, query_times_s, all_recalls)
        log.info("Saved run id=%d  scenario=%s", run_id, scenario_name)
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Config-file driven evaluation
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """
    Load and validate a TOML evaluation config.

    Expected schema:
      scenarios = ["batch", "streaming"]    # scenario names to run for every team
      datasets = [                          # list of dataset .hdf5 paths
        "data/sift-128-euclidean.hdf5",
        "data/glove-100-angular.hdf5",
      ]

      # optional overrides (defaults shown)
      db      = "results.db"
      timeout = 1800
      k       = 100

      [[teams]]
      name  = "alice"
      image = "alice/nns:latest"

      [[teams]]
      name  = "bob"
      image = "bob/nns:latest"
    """
    with open(path, "rb") as f:
        cfg = tomllib.load(f)

    datasets = cfg.get("datasets")
    if not isinstance(datasets, list) or not datasets or not all(isinstance(d, str) for d in datasets):
        raise ValueError("config: 'datasets' must be a non-empty list of dataset paths.")

    scenarios = cfg.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios or not all(isinstance(s, str) for s in scenarios):
        raise ValueError("config: 'scenarios' must be a non-empty list of scenario names.")

    teams = cfg.get("teams")
    if not isinstance(teams, list) or not teams:
        raise ValueError("config: 'teams' must be a non-empty array of tables.")
    for i, team in enumerate(teams):
        if not isinstance(team, dict) or not isinstance(team.get("name"), str) \
                or not isinstance(team.get("image"), str):
            raise ValueError(
                f"config: teams[{i}] must be a table with string 'name' and 'image' fields."
            )

    return {
        "datasets": datasets,
        "scenarios": scenarios,
        "teams": teams,
        "db":      cfg.get("db", DEFAULT_DB),
        "timeout": int(cfg.get("timeout", DEFAULT_TIMEOUT)),
        "k":       int(cfg.get("k", 100)),
    }


def run_config(client: docker.DockerClient, config_path: str) -> None:
    """Run every (team, dataset) pair for the scenario named in the config."""
    log = logging.getLogger(__name__)
    cfg = load_config(config_path)
    conn = open_db(cfg["db"])

    log.info(
        "Config: %d team(s) x %d dataset(s) x %d scenario(s)=[%s], db=%s",
        len(cfg["teams"]), len(cfg["datasets"]), len(cfg["scenarios"]),
        ", ".join(cfg["scenarios"]), cfg["db"],
    )

    for team in cfg["teams"]:
        for dataset_path in cfg["datasets"]:
            evaluate(
                conn=conn,
                client=client,
                team_name=team["name"],
                docker_image=team["image"],
                dataset_path=dataset_path,
                k=cfg["k"],
                timeout=cfg["timeout"],
                scenarios=cfg["scenarios"],
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(
        description="NNS Competition Evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("evaluate", help="Evaluate a single submission")
    p.add_argument("--team",    required=True)
    p.add_argument("--image",   required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--db",      default=DEFAULT_DB)
    p.add_argument("--timeout", default=DEFAULT_TIMEOUT, type=int)
    p.add_argument("--k", type=int, default=100)

    r = sub.add_parser("run", help="Evaluate submissions described in a TOML config file")
    r.add_argument("--config", required=True, help="Path to the TOML config file")

    return parser


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()

    # TODO: print leaderbords

    client = docker.from_env()

    if args.command == "evaluate":
        conn = open_db(args.db)
        evaluate(conn=conn, client=client, team_name=args.team,
                 docker_image=args.image, dataset_path=args.dataset,
                 k=args.k, timeout=args.timeout)
    elif args.command == "run":
        run_config(client=client, config_path=args.config)

if __name__ == "__main__":
    main()

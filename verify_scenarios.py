#!/usr/bin/env python3
"""
Verify competitors/faiss-hnsw/scenarios.yaml against the PRIVATE eval datasets.

For each (scenario, dataset) it resolves params exactly like the harness, builds
the FAISS HNSW index and computes distance-based recall @k=100 exactly like
evaluator.py, then checks it against the scenario's target.
"""
import glob
import time
import numpy as np
import h5py
import faiss
import yaml

K = 100
TARGETS = {"high_recall": 0.95, "fast": 0.80, "memory": 0.95}
SCEN_FILE = "competitors/faiss-hnsw/scenarios.yaml"


def resolve(scen_cfg, dataset_name):
    block = scen_cfg.get(dataset_name, scen_cfg.get("default"))
    return dict(block["index_params"]), dict(block["query_params"])


def compute_recall(pred, data, queries, true_distances, k=K):
    n = queries.shape[0]
    recs = np.empty(n)
    for i in range(n):
        pd = np.linalg.norm(data[pred[i]] - queries[i], axis=1)
        recs[i] = np.mean(pd[:k] <= true_distances[i][k - 1])
    return recs.mean()


def main():
    scenarios = yaml.safe_load(open(SCEN_FILE))["scenarios"]
    paths = sorted(glob.glob("datasets/*-private.hdf5"))

    rows = []
    for path in paths:
        name = path.split("/")[-1].replace(".hdf5", "")
        with h5py.File(path, "r") as f:
            data = np.ascontiguousarray(f["train"][:], dtype=np.float32)
            queries = np.ascontiguousarray(f["test"][:], dtype=np.float32)
            true_distances = f["distances"][:]
        print(f"\n=== {name}  train={data.shape[0]} dim={data.shape[1]} ===", flush=True)

        # cache indexes by (M, efC) so we build each graph once per dataset
        cache = {}
        for scen in ("high_recall", "fast", "memory"):
            ip, qp = resolve(scenarios[scen], name)
            key = (ip["M"], ip["efConstruction"])
            if key not in cache:
                faiss.omp_set_num_threads(16)
                idx = faiss.IndexHNSWFlat(data.shape[1], ip["M"])
                idx.hnsw.efConstruction = ip["efConstruction"]
                t0 = time.perf_counter()
                idx.add(data)
                print(f"  built M={ip['M']} efC={ip['efConstruction']} in "
                      f"{time.perf_counter()-t0:.1f}s", flush=True)
                cache[key] = idx
            idx = cache[key]
            idx.hnsw.efSearch = qp["ef"]
            _, pred = idx.search(queries, K)
            rec = compute_recall(pred, data, queries, true_distances)
            tgt = TARGETS[scen]
            ok = "PASS" if rec >= tgt else "FAIL"
            print(f"  {scen:11s} M={ip['M']} efC={ip['efConstruction']} ef={qp['ef']:4d}"
                  f"  recall={rec:.4f}  target>={tgt}  [{ok}]", flush=True)
            rows.append((name, scen, ip["M"], ip["efConstruction"], qp["ef"], rec, tgt, ok))
        del cache

    print(f"\n\n{'#'*74}\nVERIFICATION SUMMARY (private datasets)\n{'#'*74}")
    print(f"{'dataset':26s} {'scenario':11s} {'M':>3s} {'efC':>4s} {'ef':>4s} "
          f"{'recall':>7s} {'target':>7s}  result")
    allpass = True
    for (name, scen, M, efC, ef, rec, tgt, ok) in rows:
        if ok == "FAIL":
            allpass = False
        print(f"{name:26s} {scen:11s} {M:>3d} {efC:>4d} {ef:>4d} "
              f"{rec:>7.4f} {tgt:>7.2f}  {ok}")
    print(f"\n{'ALL PASS' if allpass else 'SOME FAILED — adjust ef'}")


if __name__ == "__main__":
    main()

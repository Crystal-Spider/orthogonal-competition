#!/usr/bin/env python3
"""Turn competition results into Formula 1 style race animations.

Each dataset becomes a racing circuit and each team a car.  The cars are driven
by the per-query timings recorded in the ``detail`` table of ``results.db``: the
race clock is wall clock, so at any simulated time ``t`` a car sits at

    position = (number of queries it had answered by t) / (total queries)

around the track.  A car that answers queries faster covers more ground per
tick, pulls away, and crosses the line first.

Teams that did not reach the scenario's recall target still race but are flagged
DNF -- the trophy goes to the first car that finishes *and* met the target, the
same rule the Sherlock Holmes board in ``evaluator.py`` applies.

A car also retires the instant the target becomes *mathematically* out of reach.
After ``k`` queries with cumulative recall ``S_k`` the best final average still
achievable is ``(S_k + (N - k)) / N`` -- every remaining query scoring a perfect
1.0.  The first ``k`` where that upper bound drops below the threshold is the
moment the run is doomed, and the car spins off into the run-off area there
rather than circulating to the end only to be labelled DNF.  Because the bound
is monotone and ends at the run's own average, every run finishing below the
target retires somewhere (at worst on the line itself).

The unscored ``faiss-hnsw-baseline`` reference run is left off the grid by
default; ``--pace-car`` puts it back as a grey pace car (always DNF, since its
recall sits below the target).

Every race opens with the F1 start lights: five lamps fill over three seconds
while the field waits on a staggered grid, then all five go out at once and the
clock starts.  ``--no-countdown`` starts racing on load instead.

The infield of each circuit holds a 2D UMAP of that dataset, sampled down to
``--umap-sample`` points (100000 by default) and turning slowly counterclockwise,
so a circuit shows the shape of the vectors that were actually searched on it.
Embeddings are cached as ``.npy`` under ``--umap-cache``, keyed by sample size
and seed, so only the first build pays for them.  This is the one feature that
wants third party packages (numpy, h5py, umap-learn); they are imported lazily
and a missing one costs you the scatter, not the deck.

Output is one self-contained HTML file per dataset (inline SVG + inline JS, no
external assets), plus the pages tying them together: ``index.html`` (the
circuits), ``paddock.html`` (every car, its colour and badge), and the standings
-- ``standings.html`` listing the five championships, with one page each behind
it (``standings-sherlock-holmes.html`` and friends).  The boards are declared in
``BOARDS`` and scored the way ``evaluator.print_boards`` scores them; each has
its own scenario, metric and recall bar, so all five are rendered whatever
``--scenario`` is being raced, and ``--recall-threshold`` only moves the racing.
Dory and Marie Kondo additionally apply the README's rule that a run must stay
within twice the baseline's query time, which ``evaluator.py`` does not.
The hub page names no winner on purpose: watch the races first.

Every page is drawn for a projector.  The layout is a 1600 x 900 slide: a race
is that frame exactly, centred and letterboxed in whatever window it is opened
in, while the document pages take their width from it and flow downwards.  One
rem is 16px at that size and the stylesheets are written in rem throughout, so
the type, the standings board and the chrome all scale with the frame instead of
staying at desk-reading size on a lecture hall wall.  Below 900px wide there is
nothing left to letterbox and the race falls back to a plain scrolling page.

    python animate.py
    python animate.py --pace-car
    python animate.py --scenario fast --recall-threshold 0.8 --laps 3
    python animate.py --dataset imagenet-clip-private --no-countdown --open
    python animate.py --umap-sample 2000        # sparser infields, faster build
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import random
import sqlite3
import time
import webbrowser
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("animate")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DB        = "results.db"
DEFAULT_SCENARIO  = "high_recall"
DEFAULT_OUT       = "races"
DEFAULT_THRESHOLD = 0.95        # recall target of the high_recall scenario
DEFAULT_LAPS      = 1
DEFAULT_DURATION  = 20.0        # seconds of playback for the whole race

BASELINE_TEAM = "faiss-hnsw-baseline"   # not scored; off the grid unless --pace-car

# Slack on the "is the target still reachable" test.  The bound is exact maths,
# but it is evaluated on a running float sum, so a run whose best case lands
# *exactly* on the threshold must not be retired by a 1e-16 rounding crumb.
RECALL_EPS = 1e-9

VIEW_W, VIEW_H = 1000, 640      # SVG viewBox of the circuit
TRACK_MARGIN   = 86             # keeps the asphalt stroke inside the viewBox
CIRCUIT_POINTS = 420            # polyline resolution of the generated circuit
# How far inside the midline the infield starts: the widest track stroke is the
# run-off at 76, half of which (38) paints inwards, plus 10 of breathing room.
TRACK_CLEAR    = 48

UMAP_SAMPLE = 100000            # points embedded per dataset
UMAP_SEED   = 42                # seeds both the row sample and UMAP itself
UMAP_CACHE  = ".umap-cache"     # where the .npy embeddings live
INDEX_DOTS  = 600               # dots kept for the (much smaller) index thumbnails

# Categorical palette for the racing teams.
#
# Cars can end up next to *any* other car on track, so this palette is held to
# the all-pairs gate rather than the (weaker) adjacent-pair one.  Both columns
# were validated as a set with the data-viz validator:
#
#   light: CVD dE 9.4 (deutan), normal-vision dE 17.7   -- all pairs, surface #fcfcfb
#   dark : CVD dE 9.5 (deutan), normal-vision dE 17.8   -- all pairs, surface #1a1a19
#
# A few slots sit below 3:1 against the surface, which obliges the relief rule:
# every car carries a permanent name tag, and the standings table view spells
# out identity in text.  Both are always on -- colour is never the only cue.
#
# Slots are assigned in this fixed order to teams sorted by name, so a team
# keeps its colour across every circuit.  Never cycle or generate a 7th hue:
# overflow teams fall back to the neutral, and the run is logged.
TEAM_COLORS = [
    ("blue",    "#5199f5", "#3490fe"),
    ("red",     "#ff343d", "#cd0522"),
    ("aqua",    "#39c78f", "#127752"),
    ("violet",  "#5f58b7", "#5d58a9"),
    ("green",   "#259121", "#33ac2e"),
    ("magenta", "#a83768", "#c47591"),
]
NEUTRAL      = ("#6b6a66", "#9b9a92")   # pace car / overflow
NEUTRAL_DNS  = ("#8f8e88", "#6f6e68")   # never made it to the grid

# Top-down open wheel car, nose pointing +x, roughly 38 x 18 user units.
CAR_SVG = (
    '<rect class="wheel" x="6.5" y="-9.5" width="7.5" height="5" rx="1.6"/>'
    '<rect class="wheel" x="6.5" y="4.5" width="7.5" height="5" rx="1.6"/>'
    '<rect class="wheel" x="-12" y="-9.5" width="8" height="5" rx="1.6"/>'
    '<rect class="wheel" x="-12" y="4.5" width="8" height="5" rx="1.6"/>'
    '<rect class="wing" x="14" y="-7" width="4" height="14" rx="1.5"/>'
    '<rect class="wing" x="-19" y="-7.5" width="4.5" height="15" rx="1.5"/>'
    '<path class="body" d="M 17,0 L 10,-3.2 L 2,-4.4 L -4,-5.4 L -13,-5.4 '
    'L -15,-3 L -15,3 L -13,5.4 L -4,5.4 L 2,4.4 L 10,3.2 Z"/>'
    '<ellipse class="cockpit" cx="-2" cy="0" rx="3.6" ry="2.6"/>'
)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class TeamRun:
    """One team's run on one dataset, ready to be raced."""

    team:       str
    status:     str                       # 'success' | 'failed' | 'timeout'
    qps:        float | None = None
    recall:     float | None = None
    total_time: float | None = None
    build_time: float | None = None
    cum:        list[float] = field(default_factory=list)   # cumulative query times
    color:      tuple[str, str] = NEUTRAL                   # (light, dark)
    tag:        str = ""
    qualified:  bool = False              # met the scenario's recall target
    baseline:   bool = False
    error:      str | None = None
    # Retirement: the 1-based query at which the recall target became
    # unreachable, and the wall clock moment that query completed.  None when
    # the run stayed in contention all the way to the flag.
    out_index:  int | None = None
    out_time:   float | None = None

    @property
    def racing(self) -> bool:
        return bool(self.cum)

    @property
    def retired(self) -> bool:
        return self.out_index is not None


def tag_candidates(team: str) -> list[str]:
    """Badge spellings for a car, shortest first, e.g. 'kinda-neighbors' -> KIN."""
    parts = [p for p in team.replace("_", "-").split("-") if p]
    if len(parts) >= 2:
        return [(parts[0][:2] + parts[1][:1]).upper(),
                (parts[0][:2] + parts[1][:2]).upper(),
                (parts[0][:3] + parts[1][:2]).upper()]
    return [team[:3].upper(), team[:4].upper(), team[:5].upper()]


def assign_tags(teams: Iterable[str]) -> dict[str, str]:
    """Give every car a badge no other car shares.

    Resolved over the whole scenario rather than per dataset, so a car keeps the
    same badge on every circuit.  Teams whose names share a prefix -- 'annvedi'
    and 'annarchy' both want ANN -- fall through to a longer spelling until they
    separate, giving ANNA and ANNV.
    """
    order = sorted(teams)
    if not order:
        return {}
    cands = {t: tag_candidates(t) for t in order}
    level = dict.fromkeys(order, 0)
    tags = {t: cands[t][0] for t in order}

    # Promote *every* member of a colliding group, not just the loser: ANNA and
    # ANNV read apart at a glance on a moving car, ANN and ANNV do not.
    for _ in range(max(len(c) for c in cands.values())):
        counts = Counter(tags.values())
        stuck = [t for t in order
                 if (counts[tags[t]] > 1 or tags[t] == "PACE")
                 and level[t] + 1 < len(cands[t])]
        if not stuck:
            break
        for t in stuck:
            level[t] += 1
            tags[t] = cands[t][level[t]]

    taken = {"PACE"}                      # reserved for the baseline car
    for team in order:                    # names identical even at full length
        while tags[team] in taken:
            stem, n = tags[team][:3], 2
            while f"{stem}{n}" in taken:
                n += 1
            tags[team] = f"{stem}{n}"
        taken.add(tags[team])
    return tags


def load_races(conn: sqlite3.Connection, scenario: str, threshold: float,
               pace_car: bool = False) -> dict[str, list[TeamRun]]:
    """Return {dataset: [TeamRun, ...]} for one scenario, fastest run first.

    ``pace_car`` puts the unscored ``BASELINE_TEAM`` reference run on the grid
    as a grey car.  It is off by default, and dropping its rows here rather
    than at render time is what keeps it out of everything downstream in one
    move: colour assignment, the standings, the table and the DNS list all
    derive from the list this returns.
    """
    rows = conn.execute(
        """
        select id, dataset, team_name, status, qps, avg_recall,
               total_query_time_s, build_time_s, error_message
        from runs
        where scenario = ?
        order by dataset, team_name
        """,
        (scenario,),
    ).fetchall()
    if not pace_car:
        rows = [r for r in rows if r[2] != BASELINE_TEAM]
    if not rows:
        return {}

    # One pass over `detail` for every run of interest, rather than a query per
    # run -- the table has no index on run_id.  Both the timing and the recall
    # of each query come out of this single scan: the timings drive the car
    # round the circuit, the recalls decide when it is mathematically out.
    wanted = {r[0] for r in rows if r[3] == "success"}
    traces: dict[int, list[tuple[float, float]]] = {rid: [] for rid in wanted}
    if wanted:
        marks = ",".join("?" * len(wanted))
        for run_id, qt, qr in conn.execute(
            f"select run_id, query_time_s, query_recall from detail "
            f"where run_id in ({marks}) order by run_id, query_index",
            tuple(wanted),
        ):
            traces[run_id].append((qt or 0.0, qr or 0.0))

    # Stable colour assignment: sorted over every team that actually gets on a
    # grid somewhere in this scenario, so a team wears the same colour on every
    # circuit.  Teams that only ever failed or timed out never appear on track
    # and take the reserved "did not start" neutral instead of a hue.
    racing_teams = sorted({r[2] for r in rows
                           if r[3] == "success" and r[2] != BASELINE_TEAM})
    palette: dict[str, tuple[str, str]] = {}
    for i, team in enumerate(racing_teams):
        if i < len(TEAM_COLORS):
            palette[team] = TEAM_COLORS[i][1:]
        else:
            log.warning("more than %d racing teams: %r falls back to the neutral "
                        "colour (never generate a new hue)", len(TEAM_COLORS), team)
            palette[team] = NEUTRAL

    # Badges follow the same rule as colours: resolved once over the scenario so
    # they are stable across circuits, and unique so no two cars share one.
    tags = assign_tags({r[2] for r in rows if r[2] != BASELINE_TEAM})

    races: dict[str, list[TeamRun]] = {}
    for run_id, dataset, team, status, qps, recall, total, build, err in rows:
        if scenario.startswith("__"):        # sentinel scenarios, never raced
            continue
        trace = traces.get(run_id, ())
        n = len(trace)
        cum, acc = [], 0.0
        got, out_index, out_time = 0.0, None, None
        for k, (qt, qr) in enumerate(trace, 1):
            acc += qt
            cum.append(round(acc, 6))
            got += qr
            # Best final average still on the table: every remaining query a 1.0.
            # Monotone non-increasing, so the first k below the bar is the one.
            if out_index is None and (got + (n - k)) / n < threshold - RECALL_EPS:
                out_index, out_time = k, cum[-1]
        run = TeamRun(
            team=team, status=status, qps=qps, recall=recall,
            total_time=total, build_time=build, cum=cum,
            out_index=out_index, out_time=out_time,
            color=NEUTRAL if team == BASELINE_TEAM
                  else palette.get(team, NEUTRAL_DNS),
            tag="PACE" if team == BASELINE_TEAM else tags[team],
            qualified=status == "success" and (recall or 0.0) >= threshold,
            baseline=team == BASELINE_TEAM,
            error=err,
        )
        races.setdefault(dataset, [])
        # Keep the fastest successful run when a team has several rows.
        prev = next((r for r in races[dataset] if r.team == team), None)
        if prev is None:
            races[dataset].append(run)
        elif run.racing and (not prev.racing or (run.total_time or math.inf)
                             < (prev.total_time or math.inf)):
            races[dataset][races[dataset].index(prev)] = run

    for dataset, runs in races.items():
        runs.sort(key=lambda r: (not r.racing, r.total_time or math.inf, r.team))
    return {d: r for d, r in races.items() if any(x.racing for x in r)}


# ---------------------------------------------------------------------------
# Circuit geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Circuit:
    """A circuit's SVG path plus the empty disc its infield offers."""

    d:  str            # the closed path, ready for a `d` attribute
    cx: float          # centre of the infield, in view coordinates
    cy: float
    r:  float          # largest disc around (cx, cy) that clears the asphalt


def circuit(dataset: str) -> Circuit:
    """A deterministic, distinct closed circuit for a dataset, as an SVG path.

    The shape is a polar Fourier curve ``r(t) = 1 + sum a_k cos(k t + p_k)``
    squashed along y.  Because ``sum |a_k|`` is capped well below 1 the radius
    stays positive, so the curve is star shaped about the origin and can never
    self-intersect -- and squash plus rotation are linear, so they preserve
    that.  Seeding from the dataset name keeps the circuit stable across runs.

    Star shaped about the origin is also what makes the infield well defined:
    the disc of radius ``min_t |p(t)|`` about the polar origin is entirely
    inside the curve, so anything drawn within it (minus the width the track
    strokes paint *inwards*) cannot end up on the racing line.
    """
    seed = int.from_bytes(hashlib.sha256(dataset.encode()).digest()[:8], "big")
    rng = random.Random(seed)

    ks = sorted(rng.sample([2, 3, 4, 5, 6], k=rng.choice([2, 2, 3])))
    weights = [rng.uniform(0.35, 1.0) for _ in ks]
    scale = rng.uniform(0.26, 0.36) / sum(weights)
    amps = [w * scale for w in weights]
    phases = [rng.uniform(0, 2 * math.pi) for _ in ks]
    rot = rng.uniform(0, 2 * math.pi)
    squash = rng.uniform(0.58, 0.84)

    pts = []
    for i in range(CIRCUIT_POINTS):
        t = 2 * math.pi * i / CIRCUIT_POINTS
        r = 1.0 + sum(a * math.cos(k * t + p) for a, k, p in zip(amps, ks, phases))
        pts.append((r * math.cos(t + rot), r * math.sin(t + rot) * squash))

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    k = min((VIEW_W - 2 * TRACK_MARGIN) / w, (VIEW_H - 2 * TRACK_MARGIN) / h)
    ox = (VIEW_W - w * k) / 2 - min(xs) * k
    oy = (VIEW_H - h * k) / 2 - min(ys) * k

    d = [f"M {pts[0][0] * k + ox:.1f},{pts[0][1] * k + oy:.1f}"]
    d += [f"L {x * k + ox:.1f},{y * k + oy:.1f}" for x, y in pts[1:]]
    d.append("Z")

    inradius = min(math.hypot(x, y) for x, y in pts) * k
    return Circuit(" ".join(d), ox, oy, max(0.0, inradius - TRACK_CLEAR))


def circuit_path(dataset: str) -> str:
    """Just the path of :func:`circuit`, for callers that want nothing else."""
    return circuit(dataset).d


# ---------------------------------------------------------------------------
# Dataset embedding
#
# The one part of this script that wants third party packages.  Everything here
# imports numpy / h5py / umap-learn *locally* and returns None when they are
# missing, so `python animate.py` still builds the whole deck on a bare stdlib
# interpreter -- it just leaves the infields empty.
# ---------------------------------------------------------------------------


def dataset_file(dataset: str, datasets_dir: Path) -> Path | None:
    """The HDF5 file holding the vectors a circuit was raced on, if it is here.

    Runs are recorded under names like ``agnews-mxbai-private``, but only the
    ``-public`` twin is usually kept on disk.  Falling back to it is exact, not
    an approximation: ``prepare_data.py`` cuts both files from one shuffled
    array and asserts that their ``/train`` datasets are identical -- only the
    held-out ``/test`` queries differ, and those are not what we embed.
    """
    direct = datasets_dir / f"{dataset}.hdf5"
    if direct.exists():
        return direct
    for suffix, twin in (("-private", "-public"), ("-public", "-private")):
        if dataset.endswith(suffix):
            alt = datasets_dir / f"{dataset[: -len(suffix)]}{twin}.hdf5"
            if alt.exists():
                return alt
    stem = dataset.rsplit("-", 1)[0]
    return next(iter(sorted(datasets_dir.glob(f"{stem}-*.hdf5"))), None)


def umap_points(dataset: str, datasets_dir: Path, cache_dir: Path,
                sample: int = UMAP_SAMPLE, seed: int = UMAP_SEED
                ) -> list[tuple[float, float]] | None:
    """A 2D UMAP of ``sample`` points of a dataset, or None if it cannot be had.

    Cached as ``<dataset>.n<sample>.s<seed>.npy``.  Sample size and seed are in
    the name on purpose: changing either must not quietly reuse an embedding of
    a different point set.  On a cache hit neither h5py nor umap-learn is even
    imported, which is what keeps rebuilds of the pages instant.
    """
    if sample <= 0:
        return None

    try:
        import numpy as np
    except ImportError:
        log.warning("numpy is not available -- circuits will have empty infields "
                    "(the flake.nix devshell has everything; try `direnv reload`)")
        return None

    cache = cache_dir / f"{dataset}.n{sample}.s{seed}.npy"
    if cache.is_file():
        log.info("%-32s embedding: cached %s", dataset, cache.name)
        return [(float(x), float(y)) for x, y in np.load(cache)]

    path = dataset_file(dataset, datasets_dir)
    if path is None:
        log.warning("%-32s embedding: no HDF5 file under %s/", dataset, datasets_dir)
        return None
    try:
        import h5py
        import umap  # type: ignore[import-not-found]
    except ImportError as exc:
        log.warning("%-32s embedding: %s -- try `direnv reload`", dataset, exc)
        return None

    t0 = time.perf_counter()
    with h5py.File(path, "r") as hfp:
        train: Any = hfp["/train"]
        n = train.shape[0]
        if sample >= n:
            rows = train[:]
        else:
            # Sorted indices, so h5py walks the (contiguous, unchunked) dataset
            # forwards once instead of seeking backwards over gigabytes.
            idx = np.sort(np.random.default_rng(seed).choice(n, sample, replace=False))
            rows = train[idx]
    rows = np.asarray(rows, dtype=np.float32)

    # Euclidean, whatever the upstream dataset called itself: prepare_data.py
    # computes every ground truth with an L2 norm, so L2 neighbourhoods are the
    # ones the teams were actually scored on.
    emb = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                    metric="euclidean", random_state=seed).fit_transform(rows)
    emb = np.asarray(emb, dtype=np.float32)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    log.info("%-32s embedding: %d x %d points from %s in %.1fs -> %s",
             dataset, rows.shape[0], rows.shape[1], path.name,
             time.perf_counter() - t0, cache.name)
    return [(float(x), float(y)) for x, y in emb]


def _median(sorted_values: list[float]) -> float:
    n = len(sorted_values)
    return (sorted_values[n // 2] if n % 2
            else (sorted_values[n // 2 - 1] + sorted_values[n // 2]) / 2)


def dots_path(points: list[tuple[float, float]], cx: float, cy: float, r: float,
              keep: int | None = None) -> str:
    """An embedding as one SVG path of zero-length subpaths -- a dot per point.

    A single element instead of ten thousand ``<circle>``s: the browser keeps
    the whole cloud as one paint, so rotating it stays cheap.  Zero-length
    subpaths render as dots under ``stroke-linecap: round``.

    The cloud is centred on its *median* and scaled so its 99.5th percentile
    radius lands on ``r``, with the stragglers beyond that pulled onto the rim.
    One stray filament must not shrink everything else to a speck, and nothing
    may reach the asphalt.  Both the centring and the scaling are radial, so a
    rotation about (cx, cy) sweeps out exactly the same disc.
    """
    if not points:
        return ""
    if keep is not None and keep < len(points):
        step = len(points) / keep          # deterministic thinning, no reshuffle
        points = [points[int(i * step)] for i in range(keep)]

    mx = _median(sorted(p[0] for p in points))
    my = _median(sorted(p[1] for p in points))
    radii = sorted(math.hypot(x - mx, y - my) for x, y in points)
    scale = radii[min(len(radii) - 1, int(len(radii) * 0.995))] or 1.0

    placed = []
    for x, y in points:
        dx, dy = (x - mx) / scale, (y - my) / scale
        d = math.hypot(dx, dy)
        if d > 1.0:
            dx, dy = dx / d, dy / d
        # Tenths of a user unit: finer than any dot we draw, and the deltas
        # below are differences of the *quantised* values, so they never drift.
        placed.append((round((cx + dx * r) * 10), round((cy + dy * r) * 10)))

    placed.sort(key=lambda p: (p[1], p[0]))       # scanline order: tiny deltas
    parts, px, py = [], 0, 0
    for i, (x, y) in enumerate(placed):
        cmd = "M" if i == 0 else "m"
        a, b = (x, y) if i == 0 else (x - px, y - py)
        parts.append(f"{cmd}{_tenths(a)},{_tenths(b)}h0")
        px, py = x, y
    return "".join(parts)


def _tenths(v: int) -> str:
    """Tenths of a unit as the shortest SVG number: 120 -> '12', 3 -> '.3'."""
    s = f"{v / 10:.1f}".rstrip("0").rstrip(".") or "0"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def umap_svg(points: list[tuple[float, float]] | None, circ: Circuit,
             keep: int | None = None) -> str:
    """The infield scatter for a circuit, or nothing at all."""
    d = dots_path(points or [], circ.cx, circ.cy, circ.r, keep)
    if not d:
        return ""
    # transform-origin in absolute user units.  Percentages would be a trap:
    # under `transform-box: view-box` Chrome resolves them against the element's
    # own origin, which is why the paddock cars had to use `fill-box`.  Here the
    # root viewBox starts at 0 0, so an absolute origin is unambiguous.
    return (f'<g class="umap" style="transform-origin:{circ.cx:.0f}px {circ.cy:.0f}px" '
            f'aria-hidden="true"><path d="{d}"/></g>')


def build_roster(races: dict[str, list[TeamRun]], winners: dict[str, str | None]
                 ) -> list[dict]:
    """One entry per team in the scenario, folded over every circuit.

    Teams that only ever failed or timed out are kept -- a team that entered and
    never made a grid should be visible, not silently absent -- and carry the
    reserved "did not start" grey they already have from ``load_races``.
    """
    roster: dict[str, dict] = {}
    for dataset, runs in races.items():
        for r in runs:
            e = roster.setdefault(r.team, {
                "team": r.team, "tag": r.tag, "light": r.color[0], "dark": r.color[1],
                "baseline": r.baseline, "entries": 0, "starts": 0, "finishes": 0,
                "retirements": 0, "wins": 0, "best_qps": None, "best_on": None,
            })
            e["entries"] += 1
            if not r.racing:
                continue
            e["starts"] += 1
            if r.retired:
                e["retirements"] += 1
            else:
                e["finishes"] += 1
            if winners.get(dataset) == r.team:
                e["wins"] += 1
            if r.qps is not None and (e["best_qps"] is None or r.qps > e["best_qps"]):
                e["best_qps"], e["best_on"] = r.qps, dataset

    # Winners first, then the rest of the runners, then whoever never started.
    return sorted(roster.values(),
                  key=lambda e: (e["baseline"], not e["starts"], -e["wins"],
                                 -(e["best_qps"] or 0.0), e["team"]))


# ---------------------------------------------------------------------------
# Championships
#
# The competition is not one title but five, each scoring the same seven
# circuits by a different measure.  They are declared once, here, and every
# standings page is rendered from this table -- adding a sixth trophy is a row,
# not a code path.  The first five mirror `evaluator.print_boards`, in its order.
# ---------------------------------------------------------------------------

# Columns of `runs` a board may be scored by.  `championship` interpolates the
# metric straight into its SQL (there is no way to parameterise a column name),
# so the name has to come from a closed set rather than from anything a caller
# can invent -- this frozenset is what makes that f-string safe.
METRIC_COLUMNS = frozenset({"qps", "peak_mem_mb", "build_time_s", "n_dist_queries"})

# How a metric is named in prose, on the cards and in the page headers.
METRIC_LABELS = {
    "qps":            "queries per second",
    "peak_mem_mb":    "peak memory",
    "build_time_s":   "build time",
    "n_dist_queries": "distance computations",
}


@dataclass(frozen=True)
class Board:
    """One championship: which runs are eligible, and what ranks them."""

    title:    str                    # "Marie Kondo"
    slug:     str                    # -> standings-marie-kondo.html
    scenario: str
    metric:   str                    # a column of `runs`, from METRIC_COLUMNS
    descending: bool                 # True when a bigger number is better
    threshold:  float                # minimum average recall to be eligible
    unit:     str                    # rendered after the metric value
    rule:     str                    # one line of prose for the card and header
    # Cap on query time as a multiple of the faiss-hnsw baseline's on the same
    # dataset.  README.md attaches it to the two prizes that would otherwise
    # reward an index that is cheap to hold or to build and hopeless to search.
    baseline_cap: float | None = None

    @property
    def file(self) -> str:
        return f"standings-{self.slug}.html"


BOARDS = [
    Board("Sherlock Holmes", "sherlock-holmes", "high_recall", "qps", True, 0.95,
          "qps", "The fastest approach that still finds what it was sent for."),
    Board("Bianconiglio", "bianconiglio", "fast", "qps", True, 0.80,
          "qps", "Always late, always running: the fastest approach at recall 0.8."),
    Board("Dory", "dory", "memory", "peak_mem_mb", False, 0.95,
          "MB", "The smallest memory footprint, without forgetting the neighbours.",
          baseline_cap=2.0),
    Board("Marie Kondo", "marie-kondo", "high_recall", "build_time_s", False, 0.95,
          "s", "The quickest to tidy a dataset into an index.",
          baseline_cap=2.0),
    Board("Paperone", "paperone", "high_recall", "n_dist_queries", False, 0.95,
          "dists", "The stingiest with full distance computations.",
          baseline_cap=2.0),
]


def format_metric(board: Board, value: float | None) -> str:
    """A board's metric as a short, readable string with its unit."""
    if value is None:
        return "&mdash;"
    if board.metric == "n_dist_queries":
        for cut, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
            if abs(value) >= cut:
                # 256000 reads as 256k, not 256.0k -- the tenth is only there
                # for the values that actually need it.
                return f"{value / cut:.1f}".removesuffix(".0") + suffix
        return f"{value:.0f}"
    if board.metric == "build_time_s":
        return f"{value:.1f} {board.unit}"
    return f"{value:,.0f} {board.unit}".replace(",", "&thinsp;")


def baseline_times(conn: sqlite3.Connection, scenario: str) -> dict[str, float]:
    """Query time of the reference run per dataset, for the 2x eligibility cap."""
    return {d: t for d, t in conn.execute(
        "select dataset, total_query_time_s from runs "
        "where scenario = ? and team_name = ? and status = 'success' "
        "and total_query_time_s is not null",
        (scenario, BASELINE_TEAM))}


def championship(conn: sqlite3.Connection, board: Board,
                 datasets: set[str] | None = None) -> list[dict]:
    """Season points for one board, scored the way ``evaluator.print_boards`` does.

    Deliberately queries ``runs`` itself instead of scoring off the loaded grid:
    ``load_races`` drops the baseline rows up front unless ``--pace-car`` is on,
    and scoring off that would quietly change which points slots get consumed.

    Two rules are easy to get subtly wrong, and both match evaluator.py: a run
    that is not eligible consumes *no* points slot (the next eligible run takes
    the one it would have had), while the baseline consumes the slot it earned
    and scores nothing -- it is simply absent from the points dict.

    A third rule is this file's own: runs that tie on the metric all score the
    slot the first of them landed on, and the tie consumes one slot per run, so
    a two-way tie for pole reads 10, 10, 6.  The Paperone board needs it -- an
    approach that computes no distances at all cannot be said to have beaten
    another that also computed none -- but it applies to every board.

    Three deliberate departures from ``evaluator.py leaderboard``, then: ties,
    ``Board.baseline_cap`` (README.md makes Dory, Marie Kondo and Paperone
    conditional on a run staying within twice the baseline's query time, which
    the CLI does not implement), and zero: the CLI still drops a run whose
    metric is 0.0, which is exactly the Paperone result this board rewards.
    Only Sherlock Holmes and Bianconiglio must still agree with it exactly.
    """
    if board.metric not in METRIC_COLUMNS:      # see METRIC_COLUMNS
        raise ValueError(f"not a scoreable column: {board.metric!r}")

    points = [10, 8, 6, 4, 3, 2, 1, 0, 0, 0, 0, 0]
    rows = conn.execute(
        f"""
        select dataset, team_name, {board.metric}, status, avg_recall,
               total_query_time_s
        from runs where scenario = ?
        order by dataset, {board.metric} {"desc" if board.descending else "asc"}
        """, (board.scenario,)).fetchall()

    caps: dict[str, float] = {}
    if board.baseline_cap is not None:
        caps = baseline_times(conn, board.scenario)
        missing = {d for d, *_ in rows if d not in caps}
        if datasets is not None:
            missing &= datasets
        for d in sorted(missing):
            log.warning("%s: no %s run on %s -- the %gx query-time cap cannot be "
                        "applied there", board.title, BASELINE_TEAM, d,
                        board.baseline_cap)

    teams = {r[1] for r in rows if r[1] != BASELINE_TEAM}
    scored = {t: {"team": t, "points": 0, "per_dataset": {}} for t in teams}
    # `idx` is the slot the next eligible run consumes; `slot` is the one the
    # current run scores, which lags behind it for the second and later run of a
    # tie.  `last_metric` is what tells them apart -- a sentinel, not None, since
    # None is a metric value the rows can legitimately carry.
    unmeasured = object()
    last_dataset, idx, slot, last_metric = None, 0, 0, unmeasured
    for dataset, team, metric, status, recall, query_time in rows:
        if datasets is not None and dataset not in datasets:
            continue
        if dataset != last_dataset:
            last_dataset, idx, last_metric = dataset, 0, unmeasured
        # A NULL metric is a run that was never measured, not a run that scored
        # zero: on an ascending board it would otherwise sort to the front and
        # walk off with pole.  (A real 0 is eligible -- see the docstring.)
        if (status != "success" or metric is None
                or (recall or 0.0) < board.threshold):
            continue
        if (board.baseline_cap is not None and dataset in caps
                and (query_time or 0.0) > board.baseline_cap * caps[dataset]):
            continue
        if metric != last_metric:
            slot, last_metric = idx, metric
        if team in scored:
            got = points[slot] if slot < len(points) else 0
            scored[team]["points"] += got
            scored[team]["per_dataset"][dataset] = (got, metric)
        idx += 1

    return sorted(scored.values(), key=lambda e: (-e["points"], e["team"]))


def rank_groups(scored: list[dict]) -> list[tuple[int, list[dict]]]:
    """Season standings grouped by equal points, with competition ranks.

    Two teams level on points share a rank and the next team takes the one
    after both of them -- 1, 1, 3 -- the same way the per-circuit points slots
    are consumed.  The podium and the table both count off this, so they cannot
    disagree about who is a champion.
    """
    groups, rank = [], 1
    for _, entries in itertools.groupby(scored, key=lambda e: e["points"]):
        entries = list(entries)
        groups.append((rank, entries))
        rank += len(entries)
    return groups


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def _payload(dataset: str, runs: list[TeamRun], scenario: str, threshold: float,
             laps: int, duration: float, countdown: bool = True) -> dict:
    racers = [r for r in runs if r.racing]
    n_queries = max(len(r.cum) for r in racers)

    # The clock runs until the last car still in the race takes the flag -- a
    # car that spun off is not worth waiting for.  It is never cut shorter than
    # the last retirement though, or that spin-off would happen off screen
    # (the case where every single car retires, and the two coincide).
    survivors = [r for r in racers if not r.retired]
    race_time = max([r.cum[-1] for r in survivors]
                    + [r.out_time or 0.0 for r in racers])

    finishers = sorted(racers, key=lambda r: r.cum[-1])
    winner = next((r.team for r in finishers if r.qualified), None)
    # "First across the line" only counts cars that were still running: a
    # retirement never crosses it, so it can never take that mention either.
    fastest = next((r.team for r in finishers if not r.retired), None)

    return {
        "dataset": dataset,
        "scenario": scenario,
        "threshold": threshold,
        "laps": laps,
        "n_queries": n_queries,
        "race_time": round(race_time, 6),
        "play_time": duration,
        "countdown": countdown,
        "winner": winner,
        "fastest": fastest,
        "solo": len(racers) == 1,
        "teams": [
            {
                "team": r.team,
                "tag": r.tag,
                "status": r.status,
                "light": r.color[0],
                "dark": r.color[1],
                "qps": r.qps,
                "recall": r.recall,
                "total": r.total_time,
                "build": r.build_time,
                "qualified": r.qualified,
                "baseline": r.baseline,
                "error": (r.error or "").strip()[:160] or None,
                "out_index": r.out_index,
                "out_time": r.out_time,
                "cum": r.cum,
            }
            for r in runs
        ],
    }


def render_race(dataset: str, runs: list[TeamRun], scenario: str, threshold: float,
                laps: int, duration: float, countdown: bool = True,
                dots: list[tuple[float, float]] | None = None) -> str:
    data = _payload(dataset, runs, scenario, threshold, laps, duration, countdown)
    blob = json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")
    circ = circuit(dataset)
    return (
        RACE_TEMPLATE
        .replace("__TITLE__", f"{dataset} &middot; {scenario}")
        .replace("__DATASET__", dataset)
        .replace("__SCENARIO__", scenario)
        .replace("<!--__NAV__-->", nav_html(""))
        .replace("__TRACK_D__", circ.d)
        .replace("__CAR_SVG__", CAR_SVG)
        .replace("__VIEWBOX__", f"0 0 {VIEW_W} {VIEW_H}")
        .replace("/*__PAYLOAD__*/null", blob)
        # Last, like the index cards: the scatter is data, and a token that
        # happened to appear inside it must not be substituted.
        .replace("<!--__UMAP__-->", umap_svg(dots, circ))
    )


_PAGES = [("index.html", "Circuits"), ("paddock.html", "Paddock"),
          ("standings.html", "Standings")]


def nav_html(current: str) -> str:
    """The top level nav, with the page you are on marked.

    Every ``standings-*.html`` board lights the one Standings entry: the five
    trophies are one destination up here, and are told apart in the sub-nav.
    """
    links = []
    for href, label in _PAGES:
        on = (current.startswith("standings") if href == "standings.html"
              else href == current)
        here = ' aria-current="page"' if on else ""
        links.append('<a href="' + href + '"' + here + '>' + label + '</a>')
    return '<nav aria-label="Pages">' + "".join(links) + "</nav>"


def subnav_html(current: str) -> str:
    """The trophy strip shown on the standings hub and on every board page."""
    links = ['<a href="standings.html"'
             + (' aria-current="page"' if current == "standings.html" else "")
             + ">All five</a>"]
    for b in BOARDS:
        here = ' aria-current="page"' if b.file == current else ""
        links.append(f'<a href="{b.file}"{here}>{b.title}</a>')
    return '<nav class="sub" aria-label="Championships">' + "".join(links) + "</nav>"


def dataset_codes(datasets: Iterable[str]) -> dict[str, str]:
    """Short column headers, e.g. 'agnews-mxbai-private' -> AGN.

    Reuses the car-badge allocator, so the codes are unique by the same rule.
    """
    heads = {d: d.split("-")[0] for d in datasets}
    if len(set(heads.values())) < len(heads):    # two circuits share a first word
        heads = {d: d for d in heads}
    codes = assign_tags(set(heads.values()))
    return {d: codes[h] for d, h in heads.items()}


def _car_svg(size: float, delay: str = "", spin: bool = True) -> str:
    """The shared car silhouette, tinted through the --car custom property.

    The race page has to tint its cars from JS, because the fill is a
    presentation attribute it rewrites on every repaint; a static page can let
    the stylesheet resolve --car per theme instead, which is why these pages
    stay theme-aware without a line of script.  The viewBox is centred on the
    car's own origin so it can spin without clipping.
    """
    # Sized in rem, like every other length on these pages: the whole document
    # scales with the slide, and a car pinned to a pixel size would shrink
    # against the card around it on a projector.
    return (f'<svg class="carart" style="width:{size}rem;height:{size}rem" '
            f'viewBox="-23 -23 46 46" aria-hidden="true">'
            f'<g class="{"spin" if spin else "parked"}"{delay}>{CAR_SVG}</g></svg>')


def render_paddock(roster: list[dict], scenario: str) -> str:
    cards = []
    for i, e in enumerate(roster):
        started = e["starts"] > 0
        klass = "card" + ("" if started else " dns") + (" pace" if e["baseline"] else "")
        bits = [f'{e["starts"]} start' + ("" if e["starts"] == 1 else "s")]
        if e["finishes"]:
            bits.append(f'{e["finishes"]} finish' + ("" if e["finishes"] == 1 else "es"))
        if e["retirements"]:
            bits.append(f'{e["retirements"]} out')
        record = " &middot; ".join(bits) if started else \
            f'no start in {e["entries"]} entr' + ("y" if e["entries"] == 1 else "ies")
        cards.append(
            f'<article class="{klass}" '
            f'style="--car-l:{e["light"]};--car-d:{e["dark"]}">'
            + _car_svg(12.5, f' style="animation-delay:-{i * 2.6:.1f}s"')
            + f'<div class="card-body"><div class="badge">{e["tag"]}</div>'
            f'<h2>{e["team"]}</h2>'
            f'<p class="record">{record}</p></div>'
            + "</article>"
        )
    return (
        PADDOCK_TEMPLATE
        .replace("__SCENARIO__", scenario)
        .replace("<!--__NAV__-->", nav_html("paddock.html"))
        .replace("<!--__CARDS__-->", "\n".join(cards))
    )


def board_look(roster: list[dict], teams: Iterable[str]) -> dict[str, dict]:
    """Colour and badge per team, for every team that scores on *any* board.

    The roster only knows the scenario that was raced, so a team that scored on
    ``fast`` or ``memory`` but never made the rendered grid would otherwise show
    up on its standings page as an anonymous grey row with an em dash for a
    badge.  Those teams get a badge from the same allocator the cars use --
    resolved over the union, so it cannot collide with a racer's -- and the
    reserved "did not start" grey, which is exactly what they are.
    """
    look = {e["team"]: e for e in roster}
    missing = sorted(set(teams) - look.keys())
    if missing:
        tags = assign_tags(look.keys() | set(missing))
        for team in missing:
            look[team] = {"team": team, "tag": tags[team],
                          "light": NEUTRAL_DNS[0], "dark": NEUTRAL_DNS[1]}
    return look


def render_standings(board: Board, scored: list[dict], look: dict[str, dict],
                     datasets: list[str], total: int) -> str:
    codes = dataset_codes(datasets)

    def style(team: str) -> str:
        e = look.get(team)
        if e is None:                        # scored, but never reached a grid
            return "--car-l:var(--dns);--car-d:var(--dns)"
        return f'--car-l:{e["light"]};--car-d:{e["dark"]}'

    # One step per group of teams level on points, so a shared title is a step
    # with two cars on it rather than a car that has to be left off.  The three
    # steps are the three competition *ranks*, not the top three teams: a
    # two-way tie for the title fills P1 twice and leaves P2 empty, because the
    # team behind them finished third and gets the P3 block it earned.
    standing = rank_groups(scored)
    groups = {rank: entries for rank, entries in standing if rank <= 3}
    podium = []
    for rank in (2, 1, 3):                   # P2 on the left, P1 centre, P3 right
        entries = groups.get(rank)           # a tie for the title leaves no P2
        if entries is None:
            continue
        who = []
        for e in entries:
            tag = look.get(e["team"], {}).get("tag", "&mdash;")
            who.append(
                f'<div class="who" style="{style(e["team"])}">'
                f'<div class="crown">{"&#127942;" if rank == 1 else ""}</div>'
                + _car_svg(6.5, spin=False)  # the podium is parc fermé, not a spin
                + f'<div class="badge">{tag}</div><div class="name">{e["team"]}</div>'
                f'<div class="pts">{e["points"]} pts</div></div>'
            )
        podium.append(
            f'<div class="step p{rank}" style="--n:{len(entries)}">'
            f'<div class="cars">{"".join(who)}</div>'
            f'<div class="block">{rank}</div></div>'
        )

    # Teams level on points share the # column too -- 1, 1, 3 -- so the table
    # never quietly promotes one of two champions above the other.
    ranks = {e["team"]: rank for rank, entries in standing for e in entries}
    head = "".join(f'<th class="num" title="{d}">{codes[d]}</th>' for d in datasets)
    rows = []
    for e in scored:
        tag = look.get(e["team"], {}).get("tag", "&mdash;")
        cells = []
        for d in datasets:
            got = e["per_dataset"].get(d)
            if got is None:                  # not eligible on that circuit
                cells.append('<td class="num"><span class="pts">&middot;</span></td>')
                continue
            pts, metric = got
            cells.append(f'<td class="num"><span class="pts">{pts}</span>'
                         f'<span class="met">{format_metric(board, metric)}</span></td>')
        rows.append(
            f'<tr style="{style(e["team"])}">'
            f'<td class="num rank">{ranks[e["team"]]}</td>'
            f'<td><span class="swatch"></span><span class="badge sm">{tag}</span>'
            f'{e["team"]}</td>{"".join(cells)}'
            f'<td class="num total">{e["points"]}</td></tr>'
        )

    legend = " &middot; ".join(f"<strong>{codes[d]}</strong> {d}" for d in datasets)
    partial = ("" if len(datasets) == total else
               f'<p class="note">Partial season &mdash; {len(datasets)} of {total} '
               f'circuits rendered.</p>')
    return (
        STANDINGS_TEMPLATE
        .replace("__TITLE__", board.title)
        .replace("__SCENARIO__", board.scenario)
        .replace("__RULE__", board.rule)
        .replace("__DIRECTION__", "highest" if board.descending else "lowest")
        .replace("__METRIC__", METRIC_LABELS[board.metric])
        .replace("__THRESHOLD__", f"{board.threshold:g}")
        .replace("<!--__CAP__-->", cap_note(board))
        .replace("<!--__NAV__-->", nav_html(board.file))
        .replace("<!--__SUBNAV__-->", subnav_html(board.file))
        .replace("<!--__PODIUM__-->", "\n".join(podium))
        .replace("<!--__HEAD__-->", head)
        .replace("<!--__ROWS__-->", "\n".join(rows))
        .replace("<!--__LEGEND__-->", legend)
        .replace("<!--__PARTIAL__-->", partial)
    )


def cap_note(board: Board) -> str:
    """The eligibility caveat for a board that carries a baseline query-time cap."""
    if board.baseline_cap is None:
        return ""
    return (f'<p class="note">Eligibility also requires finishing the queries in '
            f'no more than {board.baseline_cap:g}&times; the time the '
            f'<code>{BASELINE_TEAM}</code> reference run took on the same '
            f'circuit &mdash; a small index nobody can search is not a prize.</p>')


def render_standings_hub(counts: dict[str, int], total: int) -> str:
    """The Standings landing page: the five trophies, and not a hint of who won.

    Deliberately says nothing about standings.  The whole deck is meant to be
    watched before it is read, so this page names the contests and the rules and
    lets the board pages do the reveal.
    """
    cards = []
    for i, b in enumerate(BOARDS):
        n = counts.get(b.slug, 0)
        cards.append(
            f'<a class="card" href="{b.file}">'
            + _car_svg(9.5, f' style="animation-delay:-{i * 3.1:.1f}s"')
            + f'<div class="card-body"><h2>{b.title}</h2>'
            f'<p class="rule">{b.rule}</p>'
            f'<p class="terms"><span class="chip">{b.scenario}</span>'
            f'{"highest" if b.descending else "lowest"} '
            f'{METRIC_LABELS[b.metric]} &middot; recall &ge; {b.threshold:g}'
            + (f' &middot; &le;{b.baseline_cap:g}&times; baseline time'
               if b.baseline_cap is not None else "")
            + f'</p><p class="muted">{n} of {total} circuits scored</p>'
            '</div></a>'
        )
    return (
        HUB_TEMPLATE
        .replace("<!--__NAV__-->", nav_html("standings.html"))
        .replace("<!--__SUBNAV__-->", subnav_html("standings.html"))
        .replace("<!--__CARDS__-->", "\n".join(cards))
    )


def render_index(entries: list[dict], scenario: str) -> str:
    cards = []
    for e in entries:
        cards.append(
            f'<a class="card" href="{e["file"]}">'
            # Substituted here, not through the template: the cards are spliced
            # in after the template's own tokens are resolved, so a token left
            # inside a card would survive into the page.
            f'<svg viewBox="0 0 {VIEW_W} {VIEW_H}" preserveAspectRatio="xMidYMid meet">'
            f'{e["umap"]}'
            f'<path class="mini-asphalt" d="{e["d"]}"/>'
            f'<path class="mini-line" d="{e["d"]}"/></svg>'
            f'<div class="card-body"><h2>{e["dataset"]}</h2>'
            f'<p class="muted">{e["n_teams"]} on the grid &middot; '
            f'{e["n_queries"]} queries</p></div></a>'
        )
    return (
        INDEX_TEMPLATE
        .replace("__SCENARIO__", scenario)
        .replace("<!--__NAV__-->", nav_html("index.html"))
        .replace("<!--__CARDS__-->", "\n".join(cards))
    )


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

_THEME_CSS = """
:root {
  color-scheme: light;
  --surface-0:#f2f1ec; --surface-1:#fcfcfb; --surface-2:#e8e7e1;
  --text-1:#0b0b0b; --text-2:#52514e; --text-3:#84837c;
  --border:#dcdbd3;
  --grass:#e4e8dc; --asphalt:#4a4a47; --asphalt-2:#5a5a56;
  --kerb-a:#d94a45; --kerb-b:#fbfbf9; --line:#f4f3ee;
  --neutral:#6b6a66; --dns:#8f8e88; --umap-dot:rgba(34,40,28,.52);
  --shadow:0 1px 2px rgba(0,0,0,.08), 0 8px 24px rgba(0,0,0,.06);
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --surface-0:#111110; --surface-1:#1a1a19; --surface-2:#242422;
    --text-1:#ffffff; --text-2:#c3c2b7; --text-3:#8d8c83;
    --border:#343431;
    --grass:#1f231c; --asphalt:#3a3a37; --asphalt-2:#2e2e2b;
    --kerb-a:#b83b37; --kerb-b:#d8d7cf; --line:#6e6d66;
    --neutral:#9b9a92; --dns:#6f6e68; --umap-dot:rgba(210,220,194,.40);
    --shadow:0 1px 2px rgba(0,0,0,.5), 0 8px 24px rgba(0,0,0,.4);
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --surface-0:#111110; --surface-1:#1a1a19; --surface-2:#242422;
  --text-1:#ffffff; --text-2:#c3c2b7; --text-3:#8d8c83;
  --border:#343431;
  --grass:#1f231c; --asphalt:#3a3a37; --asphalt-2:#2e2e2b;
  --kerb-a:#b83b37; --kerb-b:#d8d7cf; --line:#6e6d66;
  --neutral:#9b9a92; --dns:#6f6e68; --umap-dot:rgba(210,220,194,.40);
  --shadow:0 1px 2px rgba(0,0,0,.5), 0 8px 24px rgba(0,0,0,.4);
}
"""

# The infield scatter.  Shared by the race pages and the index thumbnails, which
# otherwise have no CSS in common.
#
# The rotation is negative because SVG's y axis points down, so a negative angle
# is what reads as counterclockwise on screen.  120s per turn: fast enough to
# notice if you look, slow enough never to pull the eye off the racing.
_UMAP_CSS = """
.umap {
  pointer-events: none; animation: umapspin 120s linear infinite;
  /* Ten thousand subpaths share the SVG with cars that move every frame.  Ask
     for a layer of its own so the cloud is rasterised once and then spun,
     instead of being re-drawn behind every repaint of the race. */
  will-change: transform;
}
.umap path {
  fill: none; stroke: var(--umap-dot); stroke-width: 0.2; stroke-linecap: round;
}
@keyframes umapspin { to { transform: rotate(-360deg); } }
@media (prefers-reduced-motion: reduce) { .umap { animation: none; } }
"""

# Shared chrome for the three static pages (index, paddock, standings).  The
# race page keeps its own layout: it is a full-height app, these are documents.
_PAGE_CSS = """
/* These pages are read off a projector, so they are laid out on the same 1600px
   wide slide as the races: 1rem is 16px at that width and every length below is
   in rem, which makes the whole document scale as one.  Unlike a race, a
   standings table can run past the bottom of the screen, so the height is left
   to flow and the page scrolls; only the width drives the scale.  The clamp
   keeps a narrow laptop window legible and stops a 4K panel from turning the
   body text into a billboard. */
html { font-size: clamp(12px, 1vw, 26px); }
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2.4rem 2.2rem 3.75rem; background: var(--surface-0); color: var(--text-1);
  font: 1.375rem/1.5 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 88rem; margin: 0 auto; }
header.page {
  max-width: 88rem; margin: 0 auto 2rem; display: flex; align-items: flex-end;
  gap: 1.25rem; flex-wrap: wrap;
  /* At slide sizes the title can push the nav or the toggle onto a second row;
     ending the rows keeps whatever wraps tucked under the nav on the right
     rather than stranded on the left margin. */
  justify-content: flex-end;
}
h1 { margin: 0 0 .3rem; font-size: 2.4rem; letter-spacing: -.02em; }
header.page p { margin: 0; color: var(--text-2); }
.spacer { flex: 1; }
nav { display: flex; gap: .25rem; }
nav a {
  text-decoration: none; color: var(--text-2); font-size: 1.2rem; padding: .4rem .8rem;
  border-radius: 999px; border: 1px solid transparent;
}
nav a:hover { background: var(--surface-2); color: var(--text-1); }
nav a[aria-current="page"] {
  background: var(--surface-1); border-color: var(--border);
  color: var(--text-1); font-weight: 600;
}
/* The trophy strip under the header: the same pills one level quieter, and it
   wraps onto a second line rather than pushing the header wide on a phone. */
nav.sub { max-width: 88rem; margin: -1rem auto 1.8rem; flex-wrap: wrap; gap: .3rem; }
nav.sub a {
  font-size: 1.3rem; padding: .4rem .85rem; border-color: var(--border);
  background: var(--surface-1);
}
nav.sub a[aria-current="page"] { background: var(--surface-2); }
button.theme {
  font: inherit; font-size: 1.5rem; line-height: 1; cursor: pointer; padding: .45rem .7rem;
  background: var(--surface-1); color: var(--text-2);
  border: 1px solid var(--border); border-radius: .6rem;
}
button.theme:hover { color: var(--text-1); }
.muted { color: var(--text-3); font-size: 1.25rem; }
.badge {
  display: inline-block; font-size: 1.15rem; font-weight: 800; letter-spacing: .07em;
  padding: .2rem .5rem; border-radius: .45rem; background: var(--car); color: #fff;
  text-shadow: 0 1px 2px rgba(0,0,0,.35);
}

/* Cars on the static pages resolve their colour straight from the stylesheet,
   so light/dark and the theme toggle both work without a line of script. */
[style*="--car-l"] { --car: var(--car-l); }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) [style*="--car-l"] { --car: var(--car-d); }
}
:root[data-theme="dark"] [style*="--car-l"] { --car: var(--car-d); }
/* Tyres get their own token: the race page can hardcode near-black because its
   cars sit on asphalt, but on a dark card that reads as a hole in the car. */
:root { --tyre: #232320; }
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) { --tyre: #5c5b55; }
}
:root[data-theme="dark"] { --tyre: #5c5b55; }
.carart .body, .carart .wing { fill: var(--car); stroke: var(--surface-1); stroke-width: 2; }
.carart .wheel { fill: var(--tyre); }
.carart .cockpit { fill: rgba(0,0,0,.45); }
/* fill-box, not view-box: the car spins about its own bounding-box centre.
   With view-box, percentages resolve from the user-space origin rather than the
   viewBox corner, and a centred viewBox sends the car orbiting off the card. */
.spin, .parked { transform-box: fill-box; transform-origin: 50% 50%; }
.spin { animation: spin 16s linear infinite; }
.parked { transform: rotate(-90deg); }        /* nose up, as if in parc ferme */
@keyframes spin { to { transform: rotate(360deg); } }
@media (prefers-reduced-motion: reduce) {
  .spin { animation: none; transform: rotate(-24deg); }
}
""" + _UMAP_CSS + """
/* A thumbnail is about a fifth of the size of a race page, so a 1.5-unit dot
   would land well under one device pixel and simply vanish. */
.card .umap path { stroke-width: 5; }
"""

# The three static pages share one toggle: flip the explicit theme, letting the
# :root[data-theme] blocks in _THEME_CSS take over from the media query.
_THEME_JS = """
document.querySelector("button.theme").onclick = () => {
  const r = document.documentElement, t = r.getAttribute("data-theme");
  const dark = t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  r.setAttribute("data-theme", dark ? "light" : "dark");
};
"""

PADDOCK_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paddock &middot; __SCENARIO__</title>
<style>
__THEME_CSS__
__PAGE_CSS__
.grid {
  max-width: 88rem; margin: 0 auto; display: grid; gap: 1.4rem;
  grid-template-columns: repeat(auto-fill, minmax(24rem, 1fr));
}
.card {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 1rem;
  overflow: hidden; display: flex; flex-direction: column; align-items: center;
  padding: 1.1rem 1.25rem 1.25rem;
}
.card .carart { display: block; margin: .4rem 0 .15rem; }
.card-body { text-align: center; width: 100%; }
.card h2 {
  margin: .6rem 0 .4rem; font-size: 1.55rem; font-weight: 650; overflow-wrap: anywhere;
}
.card .record { margin: 0; font-size: 1.375rem; color: var(--text-2); }
.card .muted { margin: .35rem 0 0; }
.wins { display: block; margin-top: .3rem; font-weight: 600; color: var(--text-1); }
.card.dns { opacity: .72; border-style: dashed; }
.card.dns .carart { opacity: .55; }
.card.pace .badge { letter-spacing: .04em; }
</style>
</head>
<body>
<header class="page">
  <div>
    <h1>The Paddock</h1>
    <p>Every entry in the <strong>__SCENARIO__</strong> scenario. A car keeps its
       colour and badge on every circuit.</p>
  </div>
  <span class="spacer"></span>
  <!--__NAV__-->
  <button class="theme" title="Toggle light / dark">&#9681;</button>
</header>
<div class="grid">
<!--__CARDS__-->
</div>
<script>__THEME_JS__</script>
</body>
</html>
""".replace("__THEME_CSS__", _THEME_CSS).replace("__PAGE_CSS__", _PAGE_CSS) \
   .replace("__THEME_JS__", _THEME_JS)

STANDINGS_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__ &middot; Championship</title>
<style>
__THEME_CSS__
__PAGE_CSS__
.podium {
  display: flex; align-items: flex-end; justify-content: center; gap: 1.1rem;
  margin: .6rem auto 2.6rem; max-width: 60rem; flex-wrap: wrap;
}
/* A step is as wide as the number of cars standing on it: --n is the size of
   the group of teams level on points that share this rank. */
.step {
  flex: 1 1 calc(15rem * var(--n, 1)); max-width: calc(19rem * var(--n, 1));
  text-align: center;
}
.step .cars { display: flex; justify-content: center; gap: 1.1rem; }
/* Column + `margin-top: auto` on the points line, so two cars sharing a step
   keep their points on the same line however many rows their names take. */
.step .who {
  flex: 1 1 0; min-width: 0; padding-bottom: .8rem;
  display: flex; flex-direction: column;
}
/* The badge is the one child of that column that must not stretch to the full
   width of the step -- it is a pill, not a banner. */
.step .badge { align-self: center; }
.step .carart { display: block; margin: 0 auto .15rem; }
/* The crown row and the two-line name slot are reserved on every step, so the
   three cars line up by podium height instead of by how long a team name is. */
.step .crown { height: 2.2rem; font-size: 2.2rem; line-height: 1.15; }
.step .name {
  font-size: 1.5rem; font-weight: 650; margin-top: .45rem; overflow-wrap: anywhere;
  min-height: 2.6em;
}
.step .pts {
  margin-top: auto; font-size: 1.375rem; color: var(--text-2);
  font-variant-numeric: tabular-nums;
}
.step .block {
  border: 1px solid var(--border); border-bottom: 0; border-radius: .8rem .8rem 0 0;
  background: var(--surface-1); color: var(--text-3);
  font-size: 2.1rem; font-weight: 800; display: flex; align-items: flex-end;
  justify-content: center; padding-bottom: .6rem;
}
.step.p1 .block { height: 7rem; background: var(--surface-2); color: var(--text-2); }
.step.p2 .block { height: 5rem; }
.step.p3 .block { height: 3.8rem; }
.tablewrap {
  max-width: 88rem; margin: 0 auto; overflow-x: auto;
  border: 1px solid var(--border); border-radius: .9rem; background: var(--surface-1);
}
table { border-collapse: collapse; width: 100%; font-size: 1.375rem; }
th, td { padding: .65rem .75rem; text-align: left; white-space: nowrap; }
thead th {
  font-size: 1.1rem; text-transform: uppercase; letter-spacing: .07em;
  color: var(--text-3); font-weight: 600; border-bottom: 1px solid var(--border);
}
tbody tr + tr td { border-top: 1px solid var(--border); }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.rank { color: var(--text-3); width: 1%; }
.total { font-weight: 700; }
.swatch {
  display: inline-block; width: .9rem; height: .9rem; border-radius: .25rem;
  background: var(--car); margin-right: .6rem; vertical-align: middle;
}
.badge.sm { margin-right: .6rem; padding: .15rem .45rem; font-size: 1.05rem; }
/* Stacked cell: the points a run scored, and the measurement that earned them.
   Scoped to the table -- .step .pts is the podium's own points line. */
td .pts { display: block; }
td .met { display: block; margin-top: .1rem; font-size: 1.1rem; color: var(--text-3); }
.legend { max-width: 88rem; margin: 1rem auto 0; color: var(--text-3); font-size: 1.25rem; }
.note { max-width: 88rem; margin: 0 auto 1.4rem; color: var(--text-2); }
.note code { font-size: 1.25rem; }
</style>
</head>
<body>
<header class="page">
  <div>
    <h1>__TITLE__</h1>
    <p>__RULE__<br>
       Points across every circuit of the <strong>__SCENARIO__</strong> scenario:
       10-8-6-4-3-2-1 by __DIRECTION__ __METRIC__, among runs that reached recall
       __THRESHOLD__.</p>
  </div>
  <span class="spacer"></span>
  <!--__NAV__-->
  <button class="theme" title="Toggle light / dark">&#9681;</button>
</header>
<!--__SUBNAV__-->
<!--__CAP__-->
<!--__PARTIAL__-->
<div class="podium">
<!--__PODIUM__-->
</div>
<div class="tablewrap">
  <table>
    <thead><tr><th class="num">#</th><th>Team</th><!--__HEAD__--><th class="num">Pts</th></tr></thead>
    <tbody>
<!--__ROWS__-->
    </tbody>
  </table>
</div>
<p class="legend"><!--__LEGEND__--></p>
<script>__THEME_JS__</script>
</body>
</html>
""".replace("__THEME_CSS__", _THEME_CSS).replace("__PAGE_CSS__", _PAGE_CSS) \
   .replace("__THEME_JS__", _THEME_JS)

HUB_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Championships</title>
<style>
__THEME_CSS__
__PAGE_CSS__
.grid {
  max-width: 88rem; margin: 0 auto; display: grid; gap: 1.4rem;
  grid-template-columns: repeat(auto-fill, minmax(26rem, 1fr));
}
.card {
  display: flex; flex-direction: column; align-items: center; text-decoration: none;
  color: inherit; background: var(--surface-1); border: 1px solid var(--border);
  border-radius: 1rem; padding: 1.25rem 1.4rem 1.4rem;
  transition: transform .18s, border-color .18s, box-shadow .18s;
}
.card:hover { transform: translateY(-.2rem); border-color: var(--text-3); box-shadow: var(--shadow); }
/* Every trophy car is the neutral grey: a coloured one would give the game away
   before the visitor has opened a single board. */
.card { --car: var(--neutral); }
.card .carart { display: block; margin: .3rem 0 .15rem; }
.card-body { text-align: center; width: 100%; }
.card h2 { margin: .8rem 0 .45rem; font-size: 1.75rem; font-weight: 650; }
.card .rule { margin: 0 0 .8rem; font-size: 1.375rem; color: var(--text-2); }
.card .terms { margin: 0; font-size: 1.25rem; color: var(--text-3); }
.chip {
  display: inline-block; margin-right: .45rem; padding: .15rem .55rem; border-radius: .45rem;
  background: var(--surface-2); color: var(--text-2); font-size: 1.15rem;
  font-weight: 600; letter-spacing: .02em;
}
.card .muted { margin: .6rem 0 0; }
</style>
</head>
<body>
<header class="page">
  <div>
    <h1>Championships</h1>
    <p>Five titles over the same seven circuits, each scoring a different
       virtue. Pick one &mdash; no spoilers on this page.</p>
  </div>
  <span class="spacer"></span>
  <!--__NAV__-->
  <button class="theme" title="Toggle light / dark">&#9681;</button>
</header>
<!--__SUBNAV__-->
<div class="grid">
<!--__CARDS__-->
</div>
<script>__THEME_JS__</script>
</body>
</html>
""".replace("__THEME_CSS__", _THEME_CSS).replace("__PAGE_CSS__", _PAGE_CSS) \
   .replace("__THEME_JS__", _THEME_JS)

RACE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
__THEME_CSS__
__UMAP_CSS__
/* A race is a slide: a fixed 1600 x 900 (16:9) frame, centred and letterboxed
   in whatever window or projector it lands in.  1rem is 16px at that size, and
   min(1vw, 1.7778vh) holds that ratio as the frame grows, so every length below
   is written in rem and the page scales as one piece -- text included, which is
   the whole point on a projector at the back of a lecture hall.  Lengths inside
   the track SVG stay in user units: the viewBox already scales them. */
* { box-sizing: border-box; }
html {
  height: 100%; font-size: min(1vw, 1.7778vh); background: var(--surface-0);
  display: flex; align-items: center; justify-content: center;
}
body {
  margin: 0; width: 100rem; height: 56.25rem;
  background: var(--surface-0); color: var(--text-1);
  font: 1.5rem/1.45 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  display: flex; flex-direction: column; overflow: hidden;
}
/* Everything here has to sit on one line of a 100rem slide -- title, both
   chips, the nav, the clock and the toggle.  A wrapped header steals height
   from the track, so the sizes below are budgeted against the longest dataset
   name in the field and still leave a few rem of slack. */
header {
  display: flex; align-items: center; gap: .9rem; flex-wrap: wrap;
  padding: .95rem 1.5rem; border-bottom: 1px solid var(--border);
  background: var(--surface-1);
}
header h1 { margin: 0; font-size: 1.75rem; font-weight: 650; letter-spacing: -.01em; }
.chip {
  font-size: 1rem; text-transform: uppercase; letter-spacing: .06em;
  padding: .25rem .6rem; border-radius: 999px; border: 1px solid var(--border);
  color: var(--text-2); background: var(--surface-2);
}
.spacer { flex: 1; }
nav { display: flex; gap: .25rem; }
nav a {
  text-decoration: none; color: var(--text-2); font-size: 1.15rem; padding: .35rem .7rem;
  border-radius: 999px; border: 1px solid transparent;
}
nav a:hover { background: var(--surface-2); color: var(--text-1); }
.readout { display: flex; gap: 1.4rem; align-items: baseline; }
.readout div { text-align: right; }
.readout .k { font-size: 1.05rem; text-transform: uppercase; letter-spacing: .07em; color: var(--text-3); }
.readout .v { font-variant-numeric: tabular-nums; font-size: 2rem; font-weight: 600; }
main { flex: 1; display: flex; min-height: 0; }
#stage { flex: 1; position: relative; min-width: 0; background: var(--grass); }
#track-svg { width: 100%; height: 100%; display: block; }

.runoff   { fill: none; stroke: var(--surface-2); stroke-width: 76; stroke-linejoin: round; opacity: .55; }
.kerb-w   { fill: none; stroke: var(--kerb-b); stroke-width: 62; stroke-linejoin: round; }
.kerb-r   { fill: none; stroke: var(--kerb-a); stroke-width: 62; stroke-linejoin: round;
            stroke-dasharray: 13 13; }
.asphalt  { fill: none; stroke: var(--asphalt); stroke-width: 52; stroke-linejoin: round; }
.asphalt2 { fill: none; stroke: var(--asphalt-2); stroke-width: 46; stroke-linejoin: round; }
.midline  { fill: none; stroke: var(--line); stroke-width: 2; stroke-dasharray: 14 18; opacity: .5; }

.car .body    { stroke: var(--surface-1); stroke-width: 2; }
.car .wing    { stroke: var(--surface-1); stroke-width: 1.2; }
.car .wheel   { fill: #1b1b19; }
.car .cockpit { fill: rgba(0,0,0,.45); }
/* Sized in track user units, not rem: these ride inside the circuit's viewBox,
   which scales them with the track rather than with the slide.  Big enough to
   read from the back of a room without swamping the car underneath. */
.car .tag {
  font-size: 17px; font-weight: 700; letter-spacing: .04em; text-anchor: middle;
  fill: var(--text-1); paint-order: stroke; stroke: var(--surface-1); stroke-width: 5;
  stroke-linejoin: round;
}
.car.dnf .body, .car.dnf .wing { opacity: .55; stroke-dasharray: 3 2.5; }
.car.done .tag { fill: var(--text-1); }

/* Retired: beached in the run-off, colour drained out of it.  These rules beat
   the per-car fill presentation attribute, so no repaint bookkeeping needed. */
.car.out .body, .car.out .wing { fill: var(--neutral); stroke-dasharray: 3 2.5; }
.car.out .wheel   { fill: var(--neutral); }
.car.out .cockpit { fill: rgba(0,0,0,.22); }
.car.out .tag     { fill: var(--text-3); }
.outmark {
  display: none; font-size: 15px; font-weight: 800; letter-spacing: .12em;
  text-anchor: middle; fill: var(--kerb-a); paint-order: stroke;
  stroke: var(--surface-1); stroke-width: 5; stroke-linejoin: round;
}
.car.out .outmark { display: block; }

aside {
  width: 29.4rem; flex: none; border-left: 1px solid var(--border);
  background: var(--surface-1); display: flex; flex-direction: column; min-height: 0;
}
aside h2 {
  margin: 0; padding: .95rem 1.3rem .6rem; font-size: 1.2rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: .08em; color: var(--text-3);
}
#standings { position: relative; margin: 0 .8rem; flex: none; }
.row {
  position: absolute; left: 0; right: 0; height: 4.625rem; display: flex;
  align-items: center; gap: .8rem; padding: 0 .5rem; border-radius: .65rem;
  transition: transform .45s cubic-bezier(.22,1,.36,1);
}
.row.lead { background: var(--surface-2); }
.pos {
  width: 1.7rem; text-align: right; font-variant-numeric: tabular-nums;
  font-weight: 700; color: var(--text-3); font-size: 1.45rem;
}
.swatch { width: .85rem; height: 2.1rem; border-radius: .25rem; flex: none; }
.who { flex: 1; min-width: 0; }
.who .nm {
  font-size: 1.5rem; font-weight: 600; white-space: nowrap;
  overflow: hidden; text-overflow: ellipsis;
}
.bar { height: .35rem; border-radius: .2rem; background: var(--surface-2); margin-top: .35rem; overflow: hidden; }
.bar i { display: block; height: 100%; border-radius: .2rem; width: 0; }
.gap {
  font-variant-numeric: tabular-nums; font-size: 1.3rem; color: var(--text-2);
  text-align: right; min-width: 6.4rem;
}
.gap .flag { font-size: 1.05rem; letter-spacing: .05em; color: var(--text-3); text-transform: uppercase; }
.tagline { font-size: 1.2rem; color: var(--text-3); }

.row.retired .swatch { opacity: .4; }
.row.retired .nm { color: var(--text-3); font-weight: 500; }
.row.retired .pos { color: var(--text-3); }
.row.retired .gap .flag { color: var(--kerb-a); }

.notes { padding: .5rem 1.3rem 1.1rem; border-top: 1px solid var(--border); margin-top: auto; }
.notes .dns { display: flex; align-items: center; gap: .6rem; font-size: 1.3rem; color: var(--text-2); padding: .15rem 0; }
.dot { width: .7rem; height: .7rem; border-radius: 50%; flex: none; display: inline-block; }

footer {
  display: flex; align-items: center; gap: 1rem; padding: .8rem 1.5rem;
  border-top: 1px solid var(--border); background: var(--surface-1);
}
button {
  font: inherit; font-size: 1.375rem; color: var(--text-1); background: var(--surface-2);
  border: 1px solid var(--border); border-radius: .55rem; padding: .4rem .9rem; cursor: pointer;
}
button:hover { border-color: var(--text-3); }
button[aria-pressed="true"] { background: var(--text-1); color: var(--surface-1); border-color: var(--text-1); }
.grp { display: flex; gap: .3rem; }
/* The native slider draws itself in device pixels, so it is the one control
   that will not scale on its own; the explicit height keeps it in proportion
   with the buttons beside it. */
#scrub { flex: 1; min-width: 6rem; height: 1.8rem; accent-color: var(--text-2); }

/* Five columns have to fit the width of the standings column, so this one
   table stays a size below the rest of the slide. */
table { border-collapse: collapse; width: 100%; font-size: 1rem; }
th, td { text-align: left; padding: .3rem .45rem; border-bottom: 1px solid var(--border); }
th { font-size: .95rem; text-transform: uppercase; letter-spacing: .06em; color: var(--text-3); font-weight: 600; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
#table-view { display: none; padding: 0 1rem 1rem; overflow: auto; }
#table-view.on { display: block; }
.muted { color: var(--text-3); }

#confetti { position: absolute; inset: 0; pointer-events: none; }
#sweep {
  position: absolute; inset: 0; pointer-events: none; opacity: 0;
  background-image:
    repeating-conic-gradient(#111 0 25%, #fff 0 50%);
  background-size: 3.6rem 3.6rem;
  mix-blend-mode: normal;
}
#sweep.go { animation: sweep 1.15s cubic-bezier(.5,0,.5,1) 1; }
@keyframes sweep {
  0%   { opacity: 0; transform: translateX(-105%) skewX(-12deg); }
  22%  { opacity: .92; }
  70%  { opacity: .92; }
  100% { opacity: 0; transform: translateX(105%) skewX(-12deg); }
}
#toast {
  position: absolute; left: 50%; bottom: 1.8rem; transform: translateX(-50%) translateY(.6rem);
  background: var(--surface-1); border: 1px solid var(--border); border-radius: .8rem;
  padding: .7rem 1.1rem; font-size: 1.4rem; color: var(--text-2); box-shadow: var(--shadow);
  opacity: 0; transition: opacity .3s, transform .3s; pointer-events: none; max-width: 90%;
}
#toast.on { opacity: 1; transform: translateX(-50%) translateY(0); }

/* Start lights.  The gantry sits over the grid while the pre-roll runs; it is
   pointer-transparent so nothing under it becomes unclickable mid-countdown. */
/* Not centred: a real gantry hangs above the track, and the middle of the stage
   belongs to the dataset's embedding.  Sitting the lights in the upper third
   keeps the countdown off the point cloud. */
#lights {
  position: absolute; inset: 0; display: none; pointer-events: none;
  flex-direction: column; align-items: center; justify-content: flex-start;
  gap: 1.1rem; padding-top: 7%;
}
#lights.on { display: flex; }
#gantry {
  display: flex; gap: .95rem; padding: 1.1rem 1.4rem; border-radius: 1.1rem;
  background: rgba(12,12,11,.82); box-shadow: var(--shadow);
}
#gantry i {
  width: 2.875rem; height: 2.875rem; border-radius: 50%; display: block;
  background: #2b2b28; box-shadow: inset 0 .15rem .3rem rgba(0,0,0,.6);
  transition: background .12s ease-out, box-shadow .12s ease-out;
}
#gantry i.lit { background: #e5262c; box-shadow: 0 0 1.25rem .25rem rgba(229,38,44,.55); }
#cd-num {
  font-size: 7.5rem; font-weight: 800; letter-spacing: -.04em; line-height: 1;
  color: var(--text-1); text-shadow: 0 .15rem 1.4rem var(--surface-0);
  font-variant-numeric: tabular-nums;
}
#cd-num.go { color: #1f9d3f; animation: gopop .5s ease-out; }
@keyframes gopop {
  0%   { transform: scale(.7); opacity: 0; }
  35%  { transform: scale(1.12); opacity: 1; }
  100% { transform: scale(1); opacity: 1; }
}
@media (prefers-reduced-motion: reduce) {
  #sweep.go { animation-duration: .01s; }
  .row { transition: none; }
  #gantry i { transition: none; }
  #cd-num.go { animation: none; }
}
/* Below the slide's own width there is nothing to letterbox: fall back to a
   plain scrolling page at a fixed, readable size rather than shrinking the
   frame until nothing on it can be read. */
@media (max-width: 900px) {
  html { height: auto; display: block; font-size: 16px; }
  body { width: auto; height: auto; min-height: 100%; overflow: auto; }
  main { flex-direction: column; }
  #stage { min-height: 60vh; }
  aside { width: auto; border-left: 0; border-top: 1px solid var(--border); }
}
</style>
</head>
<body>

<header>
  <h1>__DATASET__</h1>
  <span class="chip">__SCENARIO__</span>
  <span class="chip" id="grid-note"></span>
  <span class="spacer"></span>
  <!--__NAV__-->
  <div class="readout">
    <div><div class="k">lap</div><div class="v" id="lap">1/1</div></div>
    <div><div class="k">race clock</div><div class="v" id="clock">0.000s</div></div>
  </div>
  <button id="theme" title="Toggle light / dark">&#9681;</button>
</header>

<main>
  <div id="stage">
    <svg id="track-svg" viewBox="__VIEWBOX__" preserveAspectRatio="xMidYMid meet"
         role="img" aria-label="Race circuit for __DATASET__">
      <defs>
        <pattern id="checker" width="16" height="16" patternUnits="userSpaceOnUse">
          <rect width="16" height="16" fill="#fbfbf9"/>
          <rect width="8" height="8" fill="#17171a"/>
          <rect x="8" y="8" width="8" height="8" fill="#17171a"/>
        </pattern>
      </defs>
      <!--__UMAP__-->
      <path class="runoff"   d="__TRACK_D__"/>
      <path class="kerb-w"   d="__TRACK_D__"/>
      <path class="kerb-r"   d="__TRACK_D__"/>
      <path class="asphalt"  d="__TRACK_D__"/>
      <path class="asphalt2" d="__TRACK_D__"/>
      <path class="midline" id="track" d="__TRACK_D__"/>
      <g id="startline"></g>
      <g id="cars"></g>
    </svg>
    <canvas id="confetti"></canvas>
    <div id="sweep"></div>
    <div id="toast" role="status" aria-live="polite"></div>
    <div id="lights" role="status" aria-live="assertive">
      <div id="gantry"><i></i><i></i><i></i><i></i><i></i></div>
      <div id="cd-num"></div>
    </div>
  </div>

  <aside>
    <h2>Standings</h2>
    <div id="standings"></div>
    <div id="table-view">
      <table>
        <thead><tr>
          <th>Team</th><th class="num">QPS</th><th class="num">Recall</th>
          <th class="num">Time</th><th>Result</th>
        </tr></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
    <div class="notes" id="notes"></div>
  </aside>
</main>

<footer>
  <button id="play">Pause</button>
  <button id="replay">Replay</button>
  <div class="grp" id="speeds"></div>
  <input id="scrub" type="range" min="0" max="1000" value="0" step="1" aria-label="Race position">
  <button id="toggle-table" aria-pressed="false">Table</button>
</footer>

<script>
const RACE = /*__PAYLOAD__*/null;

// ---- theme -----------------------------------------------------------------
const root = document.documentElement;
function isDark() {
  const t = root.getAttribute("data-theme");
  if (t) return t === "dark";
  return matchMedia("(prefers-color-scheme: dark)").matches;
}
document.getElementById("theme").onclick = () => {
  root.setAttribute("data-theme", isDark() ? "light" : "dark");
  paint();
};
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", paint);
const colorOf = (t) => isDark() ? t.dark : t.light;

// ---- setup -----------------------------------------------------------------
const SVGNS = "http://www.w3.org/2000/svg";
const track = document.getElementById("track");
const LEN = track.getTotalLength();
const N = RACE.n_queries, LAPS = RACE.laps;
const racers = RACE.teams.filter(t => t.cum.length > 0);
const dns = RACE.teams.filter(t => t.cum.length === 0);
const OFF = 9;                       // lateral spacing between cars, user units
const TAIL = 1.6;                    // extra seconds of race clock after the last car
const SLIDE_PLAY = 0.85;             // playback seconds a spin-off takes to settle
const OUT_DIST = 44;                 // how far off the racing line a car ends up
                                     // (past the kerb at 31 and the run-off edge
                                     //  at 38, so it beaches clear of the track)
const OUT_CREEP = 24;                // how far it carries on along the track first
const OUT_SLEW = 118;                // degrees of slew as it lets go
const ROW_GAP = 46;                  // spacing between starting grid rows
const GRID_OFF = 24;                 // lateral spacing of the two grid columns
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const CD = RACE.countdown ? 3.0 : 0;  // seconds of start lights before the flag drops

document.getElementById("grid-note").textContent =
  racers.length + (racers.length === 1 ? " car" : " cars") + " \\u00b7 " + N + " queries";

function pointAt(s) {
  const p = track.getPointAtLength(s);
  const q = track.getPointAtLength((s + 1.5) % LEN);
  return { x: p.x, y: p.y, a: Math.atan2(q.y - p.y, q.x - p.x) };
}

// A coarse sample of the centreline: the average of it is the middle of the
// circuit, which is what "outwards" means for a spin-off, and the individual
// points let us check that a car thrown that way does not land on some other
// part of the track where the circuit doubles back on itself.
const SAMPLES = [];
for (let i = 0; i < 360; i++) SAMPLES.push(track.getPointAtLength(LEN * i / 360));
const HUB = {
  x: SAMPLES.reduce((a, p) => a + p.x, 0) / SAMPLES.length,
  y: SAMPLES.reduce((a, p) => a + p.y, 0) / SAMPLES.length,
};
function clearance(x, y) {
  let d = Infinity;
  for (const p of SAMPLES) {
    const q = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
    if (q < d) d = q;
  }
  return Math.sqrt(d);
}

// start / finish line, laid across the track at s = 0
(function () {
  const p = pointAt(0), g = document.getElementById("startline");
  const r = document.createElementNS(SVGNS, "rect");
  r.setAttribute("x", -7); r.setAttribute("y", -26);
  r.setAttribute("width", 14); r.setAttribute("height", 52);
  r.setAttribute("fill", "url(#checker)");
  r.setAttribute("transform", "translate(" + p.x + "," + p.y + ") rotate(" +
                 (p.a * 180 / Math.PI) + ")");
  g.appendChild(r);
})();

// ---- cars ------------------------------------------------------------------
const carsG = document.getElementById("cars");
racers.forEach((t, i) => {
  const g = document.createElementNS(SVGNS, "g");
  // A car that retires earns its dashed, drained look on the spot.  Only a run
  // that is short of the target yet never provably doomed (which the maths
  // rules out, but the data could still contradict) wears it from the start.
  g.setAttribute("class", "car" + (t.qualified || t.out_index != null ? "" : " dnf"));
  const rot = document.createElementNS(SVGNS, "g");
  rot.innerHTML = '__CAR_SVG__';
  const tag = document.createElementNS(SVGNS, "text");
  // Alternate rows of the field carry their name a little higher, so two cars
  // running abreast -- or the pile-up of finishers parked on the line -- do not
  // print their tags on top of each other.
  tag.setAttribute("class", "tag"); tag.setAttribute("y", i % 2 ? -37 : -21);
  tag.textContent = t.tag;
  const out = document.createElementNS(SVGNS, "text");
  out.setAttribute("class", "outmark"); out.setAttribute("y", 28);
  out.textContent = "OUT";
  g.appendChild(rot); g.appendChild(tag); g.appendChild(out);
  carsG.appendChild(g);
  t._g = g; t._rot = rot; t._tag = tag; t._lane = i - (racers.length - 1) / 2;
  // Starting grid slot: two columns, staggered back from the line, in the
  // payload's own order (fastest run first) -- a pretend qualifying result.
  t._row2 = Math.floor(i / 2); t._slot = (i % 2) ? .5 : -.5;
  // Where it was standing when the target went out of reach -- a pure function
  // of the payload, so the run-off spot never depends on how we got there.
  t._outP = t.out_index == null ? null : Math.min(1, t.out_index / N);
});

// Where a retiring car comes to rest.  It depends only on the query it went out
// on, so it is settled once here and the spin-off is then a plain interpolation
// towards it -- which is what keeps the whole thing a function of simT alone.
const CLEAR_OK = 34;                 // clear of the kerb (31) and reads as "off"
for (const t of racers) {
  if (t._outP == null) continue;
  const s = t._outP >= 1 ? 0 : ((t._outP * LAPS) % 1) * LEN;
  const pt = pointAt(s);
  const ux = -Math.sin(pt.a), uy = Math.cos(pt.a);
  const spot = (sg) => ({
    sg,
    x: pt.x + ux * sg * OUT_DIST + Math.cos(pt.a) * OUT_CREEP,
    y: pt.y + uy * sg * OUT_DIST + Math.sin(pt.a) * OUT_CREEP,
  });
  // Outwards, into the outside of the corner -- unless the circuit folds back
  // there and that would beach the car on another straight, in which case the
  // inside run-off is the roomier one.
  const away = spot(((pt.x - HUB.x) * ux + (pt.y - HUB.y) * uy) >= 0 ? 1 : -1);
  const back = spot(-away.sg);
  const room = clearance(away.x, away.y);
  const rest = (room >= CLEAR_OK || room >= clearance(back.x, back.y)) ? away : back;
  t._park = { x: rest.x, y: rest.y, a: pt.a * 180 / Math.PI + rest.sg * OUT_SLEW };
}

// ---- standings rows --------------------------------------------------------
const ROWH = 4.625;                  // rem, matching .row's height in the CSS
const board = document.getElementById("standings");
board.style.height = (racers.length * ROWH) + "rem";
racers.forEach(t => {
  const row = document.createElement("div");
  row.className = "row";
  row.innerHTML =
    '<div class="pos"></div><div class="swatch"></div>' +
    '<div class="who"><div class="nm"></div><div class="bar"><i></i></div></div>' +
    '<div class="gap"></div>';
  board.appendChild(row);
  t._row = row;
  t._pos = row.querySelector(".pos");
  t._sw = row.querySelector(".swatch");
  t._nm = row.querySelector(".nm");
  t._fill = row.querySelector(".bar i");
  t._gap = row.querySelector(".gap");
  t._nm.textContent = t.team;
});

// teams that never made the grid
const notes = document.getElementById("notes");
if (dns.length) {
  notes.innerHTML = '<div class="tagline" style="margin-bottom:.45rem">Did not start</div>' +
    dns.map(t => '<div class="dns"><span class="dot" style="background:var(--dns)"></span>' +
      t.team + ' \\u2014 ' + t.status + '</div>').join("");
}

// results table (accessible alternative to the colour encoding)
document.getElementById("tbody").innerHTML = RACE.teams.map(t => {
  const res = t.cum.length === 0 ? t.status.toUpperCase()
            : t.team === RACE.winner ? "WINNER"
            : t.out_index != null
              ? "OUT \\u00b7 recall unreachable @ " + t.out_index + " q"
            : t.qualified ? "finished" : "DNF (recall)";
  const f = (v, d) => v == null ? "\\u2014" : v.toFixed(d);
  return "<tr><td>" + t.team + "</td><td class='num'>" + f(t.qps, 1) +
         "</td><td class='num'>" + f(t.recall, 4) + "</td><td class='num'>" +
         f(t.total, 3) + "s</td><td>" + res + "</td></tr>";
}).join("");
document.getElementById("toggle-table").onclick = (e) => {
  const on = document.getElementById("table-view").classList.toggle("on");
  e.currentTarget.setAttribute("aria-pressed", String(on));
};

// ---- race model ------------------------------------------------------------
const RATE = RACE.race_time / RACE.play_time;   // race seconds per playback second
// The spin-off is measured in playback time so it reads the same on a 0.4s
// circuit and a 7s one; converting it to race seconds keeps the whole thing a
// pure function of simT.  The tail is stretched if a car retires on the line,
// so its slide is never cut off by the end of the clock.
const SLIDE = Math.max(1e-9, SLIDE_PLAY * RATE);
const RACE_T = RACE.race_time + Math.max(TAIL, SLIDE * 1.3);
const clamp01 = (v) => v < 0 ? 0 : v > 1 ? 1 : v;
const easeOut = (u) => 1 - Math.pow(1 - u, 3);  // quick break away, then settles

function queriesAt(cum, t) {                    // count of cum[i] <= t
  let lo = 0, hi = cum.length;
  while (lo < hi) { const m = (lo + hi) >> 1; if (cum[m] <= t) lo = m + 1; else hi = m; }
  return lo;
}
function progressAt(t, cum) {
  const k = queriesAt(cum, t);
  if (k >= cum.length) return 1;
  const prev = k ? cum[k - 1] : 0;
  const span = cum[k] - prev;
  const frac = span > 0 ? (t - prev) / span : 0;
  return Math.min(1, (k + frac) / N);
}
const fmt = (s) => s.toFixed(3) + "s";

let simT = 0, playing = true, speed = 1, last = null, celebrated = false, stamped = false;
// Teams whose retirement has already been announced, plus the little queue that
// stops five simultaneous spin-offs firing five toasts.  Like `celebrated` and
// `stamped` this is a latch over a value derived from simT, never a state of
// its own: rearm() recomputes all three whenever simT jumps.
let announced = new Set(), outQueue = [], outWindow = 0, outHold = 0;
// Seconds of countdown still to run, or null once the race owns the clock.
let preroll = null, goTimer = null;

// ---- painting --------------------------------------------------------------
function paint() {
  for (const t of racers) {
    const c = colorOf(t);
    t._rot.querySelectorAll(".body,.wing").forEach(e => e.setAttribute("fill", c));
    t._sw.style.background = c;
    t._fill.style.background = c;
  }
  if (RACE.winner) {
    const w = racers.find(t => t.team === RACE.winner);
    if (w) document.documentElement.style.setProperty("--win", colorOf(w));
  }
}

function frame() {
  for (const t of racers) {
    // Retirement is read off simT exactly like position is, so scrubbing back
    // puts the car on track again and scrubbing forward beaches it once more.
    t._out = t.out_time != null && simT >= t.out_time;
    if (t._out) {
      t._p = t._outP;                           // frozen where it spun off
      t._q = t.out_index;
      t._slide = REDUCED ? 1 : easeOut(clamp01((simT - t.out_time) / SLIDE));
    } else {
      t._p = progressAt(simT, t.cum);
      t._q = Math.min(N, queriesAt(t.cum, simT));
      t._slide = 0;
    }
  }
  const order = racers.slice().sort((a, b) => {
    // Retirements drop to the bottom, the ones that got furthest first.
    if (a._out !== b._out) return a._out ? 1 : -1;
    if (a._out) return (b.out_index - a.out_index) || (a.out_time - b.out_time);
    return (a._p >= 1 && b._p >= 1)
      ? a.cum[a.cum.length - 1] - b.cum[b.cum.length - 1]
      : b._p - a._p;
  });
  const lead = order[0];

  order.forEach((t, i) => {
    // Position on track: a finished car parks on the line, everyone else sits
    // at the fractional part of (progress x laps) around the circuit.
    // While the lights are up the cars form a staggered starting grid behind
    // the line instead of stacking on it -- the countdown is the one moment
    // the whole field is meant to be readable.
    const grid = preroll !== null;
    const s = grid ? LEN - ((t._row2 + (t._slot > 0 ? .5 : 0)) * ROW_GAP + ROW_GAP * .5)
                   : (t._p >= 1 ? 0 : ((t._p * LAPS) % 1) * LEN);
    const pt = pointAt(s);
    const ux = -Math.sin(pt.a), uy = Math.cos(pt.a);          // unit track normal
    const lane = grid ? t._slot * GRID_OFF : t._lane * OFF;
    let x = pt.x + ux * lane, y = pt.y + uy * lane;
    let ang = pt.a * 180 / Math.PI;
    if (t._out) {
      // Let go of the racing line and slew towards the resting spot; the grid
      // lane goes with it, so every retirement ends the same distance off the
      // track rather than at the luck of its slot.
      const u = t._slide;
      x += (t._park.x - x) * u;
      y += (t._park.y - y) * u;
      ang += (t._park.a - ang) * u;
    }
    t._g.setAttribute("transform", "translate(" + x + "," + y + ")");
    t._rot.setAttribute("transform", "rotate(" + ang + ")");
    t._g.setAttribute("opacity", t._out ? (1 - .42 * t._slide).toFixed(3) : 1);
    t._g.classList.toggle("out", t._out);
    t._g.classList.toggle("done", !t._out && t._p >= 1);

    // standings row
    t._row.style.transform = "translateY(" + (i * ROWH) + "rem)";
    t._row.classList.toggle("lead", i === 0 && !t._out);
    t._row.classList.toggle("retired", t._out);
    t._pos.textContent = t._out ? "\\u2014" : (i + 1);
    t._fill.style.width = (t._p * 100).toFixed(2) + "%";
    t._fill.style.background = t._out ? "var(--neutral)" : colorOf(t);
    if (t._out) {
      t._gap.innerHTML = "<span class='flag'>out \\u00b7 recall unreachable</span><br>@ " +
        t.out_index + " q";
    } else if (t._p >= 1) {
      t._gap.innerHTML = "<span class='flag'>" +
        (t.qualified ? "finished" : "dnf recall " + t.recall.toFixed(3)) +
        "</span><br>" + fmt(t.cum[t.cum.length - 1]);
    } else if (i === 0) {
      t._gap.innerHTML = "<span class='flag'>leader</span><br>" + t._q + "/" + N;
    } else {
      // Interval to the leader, the way timing screens show it: how long ago
      // the leader passed the point this car has just reached.
      const dq = lead._q - t._q;
      const when = t._q > 0 ? (lead._p >= 1 ? lead.cum[Math.min(t._q, N) - 1]
                                            : lead.cum[t._q - 1]) : 0;
      t._gap.innerHTML = "<span class='flag'>+" + dq + " q</span><br>+" +
        Math.max(0, simT - when).toFixed(2) + "s";
    }
  });

  document.getElementById("clock").textContent = fmt(Math.min(simT, RACE.race_time));
  const lapNow = Math.min(LAPS, Math.floor((lead ? lead._p : 0) * LAPS) + 1);
  document.getElementById("lap").textContent = lapNow + "/" + LAPS;
  document.getElementById("scrub").value = Math.round(1000 * simT / RACE_T);

  // retirements that have just become true at this simT
  for (const t of racers) {
    if (t._out && !announced.has(t.team)) { announced.add(t.team); outQueue.push(t); }
  }
  flushOut();

  // first across the line, but short of the recall target
  if (!stamped && RACE.fastest !== RACE.winner) {
    const f = racers.find(t => t.team === RACE.fastest);
    if (f && !f._out && f._p >= 1) {
      stamped = true;
      toast("\\u26a0\\ufe0f <strong>" + f.team + "</strong> is first across the line \\u2014 but recall " +
            f.recall.toFixed(3) + " &lt; " + RACE.threshold +
            ", so the win is not awarded.");
    }
  }
  if (!celebrated && RACE.winner) {
    const w = racers.find(t => t.team === RACE.winner);
    if (w && w._p >= 1) { celebrated = true; celebrate(w, order); }
  }
}

// ---- start lights ----------------------------------------------------------
// The countdown is deliberately *not* part of simT.  Everything else on this
// page -- position, retirement, the celebration latches, the scrubber -- is a
// pure function of the race clock, and a pre-roll that moved it would break
// that.  It is its own small phase that holds the clock at 0 until lights out,
// so frame() keeps drawing the cars on the grid and nothing else notices.
const lightsEl = document.getElementById("lights");
const gantry = [...document.getElementById("gantry").children];
const cdNum = document.getElementById("cd-num");

function showLights() {
  lightsEl.classList.add("on");
  const lit = Math.min(5, Math.floor((CD - preroll) / CD * 5) + 1);
  gantry.forEach((l, i) => l.classList.toggle("lit", i < lit));
  cdNum.classList.remove("go");
  cdNum.textContent = String(Math.max(1, Math.ceil(preroll)));
}
function goFlash() {                       // all five out at once, then race
  gantry.forEach(l => l.classList.remove("lit"));
  cdNum.textContent = "GO!";
  cdNum.classList.add("go");
  goTimer = setTimeout(() => lightsEl.classList.remove("on"), 520);
}
function armCountdown() {
  clearTimeout(goTimer);
  preroll = CD > 0 ? CD : null;
  if (preroll === null) lightsEl.classList.remove("on"); else showLights();
}
function cancelCountdown() {               // a scrub asked for a position, not a start
  clearTimeout(goTimer);
  preroll = null;
  lightsEl.classList.remove("on");
}

function tick(now) {
  if (last === null) last = now;
  const dt = Math.min(0.25, (now - last) / 1000);
  last = now;
  if (preroll !== null) {
    if (playing) {
      preroll -= dt * speed;
      if (preroll <= 0) { preroll = null; goFlash(); } else showLights();
    }
  } else if (playing) {
    simT += dt * RATE * speed;
    if (simT >= RACE_T) { simT = RACE_T; playing = false; setPlay(); }
  }
  frame();
  requestAnimationFrame(tick);
}

// ---- celebration -----------------------------------------------------------
let toastTimer = null;
function toast(html) {
  const el = document.getElementById("toast");
  el.innerHTML = html; el.classList.add("on");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("on"), 5200);
}

// A pile-up can take several cars out within a handful of frames, and one
// toast each would be a strobe.  Queued retirements are held for a short
// window so simultaneous ones merge into a single line, and consecutive
// batches are spaced out; the queue itself is rebuilt by rearm() on a scrub,
// so a jump through half the race never dumps a backlog on screen.
function flushOut() {
  if (!outQueue.length) return;
  const now = performance.now();
  if (!outWindow) { outWindow = now; return; }
  if (now - outWindow < 200 || now < outHold) return;

  const batch = outQueue.sort((a, b) => a.out_index - b.out_index);
  outQueue = []; outWindow = 0; outHold = now + 2800;

  const boom = "\\ud83d\\udca5 ";
  const named = (t) => "<strong>" + t.team + "</strong>";
  if (batch.length === 1) {
    toast(boom + named(batch[0]) + " is out on query " + batch[0].out_index +
          " \\u2014 recall " + RACE.threshold + " can no longer be reached.");
  } else if (batch.length <= 3) {
    toast(boom + batch.map(t => named(t) + " (" + t.out_index + " q)").join(", ") +
          " \\u2014 out, the recall target is beyond all of them.");
  } else {
    toast(boom + batch.length + " cars out between query " + batch[0].out_index +
          " and " + batch[batch.length - 1].out_index +
          " \\u2014 recall " + RACE.threshold + " unreachable.");
  }
}

function celebrate(w, order) {
  const sweep = document.getElementById("sweep");
  sweep.classList.remove("go"); void sweep.offsetWidth; sweep.classList.add("go");

  const second = order.find(t => t !== w && t.qualified);
  const margin = second ? (second.cum[second.cum.length - 1] - w.cum[w.cum.length - 1]) : null;

  w._g.animate(
    [{ transform: "scale(1)" }, { transform: "scale(1.45)" }, { transform: "scale(1)" }],
    { duration: 700, iterations: 3, easing: "ease-in-out" }
  );
  confetti(colorOf(w));
}

const cv = document.getElementById("confetti"), ctx = cv.getContext("2d");
let bits = [];
// The canvas is the one part of the stage measured in device pixels rather than
// in the slide's rem, so the paper has to be scaled by hand: how many pixels a
// rem is worth right now is exactly how much bigger a projected frame is than
// the 1600 x 900 design one.
const remPx = () => parseFloat(getComputedStyle(document.documentElement).fontSize) / 16;
function confetti(color) {
  const stage = document.getElementById("stage");
  cv.width = stage.clientWidth; cv.height = stage.clientHeight;
  const hues = [color].concat(racers.map(colorOf));
  const S = remPx();
  bits = [];
  for (let i = 0; i < 220; i++) {
    bits.push({
      x: cv.width * (.2 + .6 * Math.random()), y: cv.height * .55,
      vx: (Math.random() - .5) * 12 * S, vy: (-6 - Math.random() * 11) * S,
      w: (4 + Math.random() * 6) * S, h: (3 + Math.random() * 5) * S,
      rot: Math.random() * 6.28, vr: (Math.random() - .5) * .35,
      c: i % 3 === 0 ? hues[1 + (i % (hues.length - 1))] : color,
      life: 1
    });
  }
  if (bits.length) requestAnimationFrame(drawConfetti);
}
function drawConfetti() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  const g = .32 * remPx();
  let alive = 0;
  for (const b of bits) {
    b.vy += g; b.x += b.vx; b.y += b.vy; b.vx *= .992; b.rot += b.vr;
    if (b.y > cv.height + 30) b.life = 0;
    if (b.life <= 0) continue;
    alive++;
    ctx.save(); ctx.translate(b.x, b.y); ctx.rotate(b.rot);
    ctx.fillStyle = b.c; ctx.fillRect(-b.w / 2, -b.h / 2, b.w, b.h); ctx.restore();
  }
  if (alive) requestAnimationFrame(drawConfetti); else ctx.clearRect(0, 0, cv.width, cv.height);
}

// ---- controls --------------------------------------------------------------
const playBtn = document.getElementById("play");
const setPlay = () => playBtn.textContent = playing ? "Pause" : "Play";
playBtn.onclick = () => {
  // Play at the end of a race rewinds, and a rewind gets its lights back.
  if (!playing && preroll === null && simT >= RACE_T) { simT = 0; rearm(); armCountdown(); }
  playing = !playing; setPlay();
};
document.getElementById("replay").onclick = () => {
  simT = 0; rearm(); armCountdown(); playing = true; setPlay();
  bits = []; ctx.clearRect(0, 0, cv.width, cv.height);
  document.getElementById("toast").classList.remove("on");
};
const speeds = document.getElementById("speeds");
[0.5, 1, 2, 4].forEach(s => {
  const b = document.createElement("button");
  b.textContent = s + "\\u00d7";
  b.setAttribute("aria-pressed", String(s === 1));
  b.onclick = () => {
    speed = s;
    [...speeds.children].forEach(c => c.setAttribute("aria-pressed", String(c === b)));
  };
  speeds.appendChild(b);
});
const crossed = (name) => {
  const t = racers.find(x => x.team === name);
  return !!t && simT >= t.cum[t.cum.length - 1];
};
// Every one-shot announcement in the page is a latch over something simT
// already decides, so moving the clock just recomputes the latches: scrubbing
// past an event must not re-fire it, scrubbing back before it must arm it
// again, and neither may leave a queued toast behind.
function rearm() {
  celebrated = crossed(RACE.winner);
  stamped = crossed(RACE.fastest);
  announced = new Set(racers.filter(t => t.out_time != null && simT >= t.out_time)
                            .map(t => t.team));
  outQueue = []; outWindow = 0; outHold = 0;
}
document.getElementById("scrub").oninput = (e) => {
  cancelCountdown();
  simT = RACE_T * e.target.value / 1000;
  rearm();
};
addEventListener("keydown", (e) => {
  if (e.key === " ") { e.preventDefault(); playBtn.click(); }
  if (e.key.toLowerCase() === "r") document.getElementById("replay").click();
});
addEventListener("resize", () => {
  const stage = document.getElementById("stage");
  cv.width = stage.clientWidth; cv.height = stage.clientHeight;
});

paint();
frame();
armCountdown();
requestAnimationFrame(tick);
</script>
</body>
</html>
""".replace("__THEME_CSS__", _THEME_CSS).replace("__UMAP_CSS__", _UMAP_CSS)


INDEX_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Orthogonal Grand Prix &middot; __SCENARIO__</title>
<style>
__THEME_CSS__
__PAGE_CSS__
.grid {
  max-width: 88rem; margin: 0 auto; display: grid; gap: 1.4rem;
  grid-template-columns: repeat(auto-fill, minmax(25rem, 1fr));
}
.card {
  display: block; text-decoration: none; color: inherit; background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 1rem; overflow: hidden;
  transition: transform .18s, border-color .18s, box-shadow .18s;
}
.card:hover { transform: translateY(-.2rem); border-color: var(--text-3); box-shadow: var(--shadow); }
.card svg { display: block; width: 100%; height: 12rem; background: var(--grass); }
.mini-asphalt { fill: none; stroke: var(--asphalt); stroke-width: 46; stroke-linejoin: round; }
.mini-line { fill: none; stroke: var(--line); stroke-width: 3; stroke-dasharray: 16 20; opacity: .55; }
.card-body { padding: 1rem 1.2rem 1.2rem; }
.card h2 { margin: 0 0 .45rem; font-size: 1.6rem; font-weight: 650; }
.card p { margin: 0; font-size: 1.375rem; color: var(--text-2); }
.dot { width: .8rem; height: .8rem; border-radius: 50%; display: inline-block; margin-right: .4rem; }
.muted { color: var(--text-3); font-size: 1.25rem; margin-top: .3rem !important; }
</style>
</head>
<body>
<header class="page">
  <div>
    <h1>Orthogonal Grand Prix</h1>
    <p>One circuit per dataset, driven by the per-query timings of the
       <strong>__SCENARIO__</strong> scenario.</p>
  </div>
  <span class="spacer"></span>
  <!--__NAV__-->
  <button class="theme" title="Toggle light / dark">&#9681;</button>
</header>
<div class="grid">
<!--__CARDS__-->
</div>
<script>__THEME_JS__</script>
</body>
</html>
""".replace("__THEME_CSS__", _THEME_CSS).replace("__PAGE_CSS__", _PAGE_CSS) \
   .replace("__THEME_JS__", _THEME_JS)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Render Formula 1 style race animations from results.db",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--db", default=DEFAULT_DB, help="SQLite results database")
    p.add_argument("--scenario", default=DEFAULT_SCENARIO,
                   help="scenario to race (default: %(default)s)")
    p.add_argument("--out", default=DEFAULT_OUT, help="output directory")
    p.add_argument("--dataset", action="append", metavar="NAME",
                   help="only render this dataset (repeatable)")
    p.add_argument("--recall-threshold", type=float, default=DEFAULT_THRESHOLD,
                   help="recall a run must reach to be eligible to win the race "
                        "on a circuit, and below which a car retires; the "
                        "championship pages ignore it and use each trophy's own "
                        "bar (default: %(default)s)")
    p.add_argument("--laps", type=int, default=DEFAULT_LAPS,
                   help="times round the circuit for a full run")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION,
                   help="playback seconds for the whole race")
    p.add_argument("--pace-car", action="store_true",
                   help=f"put the unscored {BASELINE_TEAM} reference run on the "
                        "grid as a grey pace car (default: off, it is left out "
                        "of the race entirely)")
    p.add_argument("--no-countdown", action="store_false", dest="countdown",
                   help="start racing on load instead of running the F1 start "
                        "lights first")
    p.add_argument("--umap-sample", type=int, default=UMAP_SAMPLE, metavar="N",
                   help="points to embed in each circuit's infield, 0 to leave "
                        "the infields empty (default: %(default)s)")
    p.add_argument("--umap-seed", type=int, default=UMAP_SEED, metavar="N",
                   help="seeds the row sample and UMAP itself (default: %(default)s)")
    p.add_argument("--umap-cache", default=UMAP_CACHE, metavar="DIR",
                   help="directory of cached .npy embeddings (default: %(default)s)")
    p.add_argument("--datasets-dir", default="datasets", metavar="DIR",
                   help="where the dataset HDF5 files live (default: %(default)s)")
    p.add_argument("--open", action="store_true",
                   help="open the index page in a browser when done")
    return p


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = build_parser().parse_args()

    if args.laps < 1:
        args.laps = 1

    db = Path(args.db)
    if not db.exists():
        raise SystemExit(f"no such database: {db}")

    # Read only: unlike evaluator.open_db(), never create or migrate tables here.
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        races = load_races(conn, args.scenario, args.recall_threshold,
                           pace_car=args.pace_car)
        if not races:
            raise SystemExit(
                f"no successful runs for scenario {args.scenario!r} in {db}")

        season = len(races)                  # circuits before --dataset narrows it
        if args.dataset:
            missing = set(args.dataset) - races.keys()
            for m in sorted(missing):
                log.warning("dataset %r has no raceable run in scenario %r",
                            m, args.scenario)
            races = {d: r for d, r in races.items() if d in set(args.dataset)}
            if not races:
                raise SystemExit("none of the requested datasets can be raced")

        # Every championship, not just the one that was raced: the boards carry
        # their own scenario and recall bar, so `fast` and `memory` get scored
        # here even when the circuits on screen are the `high_recall` ones.
        boards = {b.slug: championship(conn, b, set(races)) for b in BOARDS}
    finally:
        conn.close()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    entries, winners = [], {}
    for dataset in sorted(races):
        runs = races[dataset]
        dots = umap_points(dataset, Path(args.datasets_dir), Path(args.umap_cache),
                           args.umap_sample, args.umap_seed)
        html = render_race(dataset, runs, args.scenario, args.recall_threshold,
                           args.laps, args.duration, args.countdown, dots)
        path = out / f"{dataset}.html"
        path.write_text(html, encoding="utf-8")

        data = _payload(dataset, runs, args.scenario, args.recall_threshold,
                        args.laps, args.duration, args.countdown)
        winner = next((r for r in runs if r.team == data["winner"]), None)
        winners[dataset] = data["winner"]
        circ = circuit(dataset)
        entries.append({
            "dataset": dataset,
            "file": path.name,
            "d": circ.d,
            "umap": umap_svg(dots, circ, keep=INDEX_DOTS),
            "winner": data["winner"],
            "color": (winner.color[0] if winner else "var(--dns)"),
            "n_teams": sum(1 for r in runs if r.racing),
            "n_queries": data["n_queries"],
        })
        log.info("%-32s %d cars, %d queries, winner=%s",
                 dataset, entries[-1]["n_teams"], data["n_queries"],
                 data["winner"] or "-- (nobody met the recall target)")

    roster = build_roster(races, winners)
    (out / "paddock.html").write_text(
        render_paddock(roster, args.scenario), encoding="utf-8")
    log.info("paddock: %d cars (%d never started)",
             len(roster), sum(1 for e in roster if not e["starts"]))

    look = board_look(roster, {e["team"] for b in boards.values() for e in b})
    circuits = sorted(races)
    counts = {}
    for b in BOARDS:
        scored = boards[b.slug]
        counts[b.slug] = len({d for e in scored for d in e["per_dataset"]})
        (out / b.file).write_text(
            render_standings(b, scored, look, circuits, season), encoding="utf-8")
        # Deliberately no leader in this line: the terminal should not spoil the
        # hub page, which goes to the same lengths to stay quiet.
        log.info("%-16s %d of %d circuits scored, %d teams on the board",
                 b.title, counts[b.slug], len(circuits),
                 sum(1 for e in scored if e["points"]))
    (out / "standings.html").write_text(
        render_standings_hub(counts, len(circuits)), encoding="utf-8")

    index = out / "index.html"
    index.write_text(render_index(entries, args.scenario), encoding="utf-8")
    log.info("wrote %d races + index, paddock and %d standings pages to %s/",
             len(entries), len(BOARDS) + 1, out)

    if args.open:
        webbrowser.open(index.resolve().as_uri())


if __name__ == "__main__":
    main()

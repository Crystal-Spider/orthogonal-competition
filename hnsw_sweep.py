#!/usr/bin/env python3
"""
Sweep FAISS HNSW parameters to find configs reaching >=0.95 avg recall per dataset.

Recall is computed exactly as the competition evaluator does:
  predicted_distances = ||data[pred_neighbors] - query||
  threshold           = true_distances[k-1]
  recall              = mean(predicted_distances[:k] <= threshold)
then averaged over all queries.
"""
import sys
import time
import glob
import numpy as np
import h5py
import faiss

K = 100
TARGET = 0.95

# (M, efConstruction) index configs to build
INDEX_CONFIGS = [
    (16, 100),   # competition baseline
    (32, 200),   # robust high-recall config
]
# efSearch values to sweep (ascending; recall is monotone increasing in ef)
EF_GRID = [50, 80, 120, 160, 200, 260, 320, 400, 500, 650, 800, 1000]


def compute_recall(pred_neighbors, data, queries, true_distances, k=K):
    n = queries.shape[0]
    recs = np.empty(n)
    for i in range(n):
        pd = np.linalg.norm(data[pred_neighbors[i]] - queries[i], axis=1)
        threshold = true_distances[i][k - 1]
        recs[i] = np.mean(pd[:k] <= threshold)
    return recs.mean()


def run_dataset(path):
    name = path.split("/")[-1].replace("-public.hdf5", "")
    with h5py.File(path, "r") as f:
        data = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        queries = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        true_distances = f["distances"][:]
    n, dim = data.shape
    print(f"\n{'='*70}\n{name}: train={n} dim={dim} queries={queries.shape[0]}", flush=True)

    results = []
    for M, efC in INDEX_CONFIGS:
        faiss.omp_set_num_threads(16)  # threads don't affect recall, only speed
        index = faiss.IndexHNSWFlat(dim, M)
        index.hnsw.efConstruction = efC
        t0 = time.perf_counter()
        index.add(data)
        build_t = time.perf_counter() - t0
        print(f"  [M={M}, efC={efC}] built in {build_t:.1f}s", flush=True)

        found_ef = None
        for ef in EF_GRID:
            index.hnsw.efSearch = ef
            t0 = time.perf_counter()
            _, pred = index.search(queries, K)
            search_t = time.perf_counter() - t0
            rec = compute_recall(pred, data, queries, true_distances)
            mark = ""
            if rec >= TARGET and found_ef is None:
                found_ef = ef
                mark = "  <-- reaches target"
            print(f"      ef={ef:5d}  recall={rec:.4f}  "
                  f"({search_t*1000/queries.shape[0]:.2f} ms/query, 16 thr){mark}",
                  flush=True)
            if rec >= TARGET:
                break
        results.append((M, efC, found_ef, build_t))
        del index
    return name, results


def main():
    paths = sorted(glob.glob("datasets/*-public.hdf5"))
    if len(sys.argv) > 1:
        paths = [p for p in paths if any(a in p for a in sys.argv[1:])]
    summary = []
    for p in paths:
        summary.append(run_dataset(p))

    print(f"\n\n{'#'*70}\nSUMMARY: minimal efSearch reaching >={TARGET} avg recall @k={K}\n{'#'*70}")
    print(f"{'dataset':28s} {'M=16,efC=100':>14s} {'M=32,efC=200':>14s}")
    for name, results in summary:
        cells = []
        for (M, efC, found_ef, bt) in results:
            cells.append(f"ef={found_ef}" if found_ef else ">1000")
        print(f"{name:28s} {cells[0]:>14s} {cells[1]:>14s}")


if __name__ == "__main__":
    main()

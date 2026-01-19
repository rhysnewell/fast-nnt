#! /usr/bin/env python3

import argparse
import csv
import os
import resource
import subprocess
import sys
import tempfile
import time

import pandas as pd

import fastnntpy as fn


def max_rss_mib():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS ru_maxrss is bytes; on Linux it's kilobytes.
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def run_once(threads, data, labels):
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    rss0 = max_rss_mib()

    fn.set_fastnnt_threads(threads)
    fn.run_neighbour_net(
        data,
        max_iterations=5000,
        ordering_method="splitstree4",
        labels=labels,
    )

    elapsed = time.perf_counter() - t0
    cpu = time.process_time() - cpu0
    rss1 = max_rss_mib()
    peak_rss = max(rss0, rss1)

    return {
        "threads": threads,
        "elapsed_sec": elapsed,
        "cpu_sec": cpu,
        "peak_ram_mib": peak_rss,
    }


def single_mode(args):
    data = pd.read_csv("test/data/large/large_dist_matrix.csv")
    labels = list(data.columns)
    result = run_once(args.threads, data, labels)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writeheader()
        writer.writerow(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true", help="Run a single benchmark in-process.")
    parser.add_argument("--threads", type=int, help="Thread count for --single.")
    parser.add_argument("--out-csv", help="Output CSV path for --single.")
    args = parser.parse_args()

    if args.single:
        if args.threads is None or args.out_csv is None:
            raise SystemExit("--single requires --threads and --out-csv")
        single_mode(args)
        return

    thread_counts = [1, 2, 4, 8]
    repeats = 3
    out_path = "test/python/fastnnt_threads_benchmark.csv"

    results = []
    for threads in thread_counts:
        for _ in range(repeats):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp_path = tmp.name
            try:
                subprocess.run(
                    [sys.executable, __file__, "--single", "--threads", str(threads), "--out-csv", tmp_path],
                    check=True,
                )
                with open(tmp_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        results.append(row)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["threads", "elapsed_sec", "cpu_sec", "peak_ram_mib"])
        writer.writeheader()
        writer.writerows(results)

    for row in results:
        print(row)


if __name__ == "__main__":
    main()

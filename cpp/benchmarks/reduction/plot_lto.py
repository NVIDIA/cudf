# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plot fixed-loop C++ reduction results as SVG."""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LABELS = {"cudf_ast": "cuDF AST", "cudf_ast_jit": "cuDF AST JIT", "lto": "LTO"}


def plot(records, destination, subtitle):
    destination.mkdir(parents=True, exist_ok=True)
    q13 = [row for row in records if row["workload"] == "tpcds_q13_post_join" and int(row["rows"]) > 0]
    for nulls in sorted({row["null_percent"] for row in q13}, key=int):
        points = [row for row in q13 if row["null_percent"] == nulls]
        for throughput in (False, True):
            fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
            for engine, label in LABELS.items():
                group = sorted((row for row in points if row["engine"] == engine),
                               key=lambda row: int(row["rows"]))
                x, y, lower, upper = [], [], [], []
                for row in group:
                    median, minimum, maximum = (float(row[key]) for key in
                                                 ("median_ms", "min_ms", "max_ms"))
                    value = float(row["logical_bytes"]) / (median * 1e6) if throughput else median
                    low = float(row["logical_bytes"]) / (maximum * 1e6) if throughput else minimum
                    high = float(row["logical_bytes"]) / (minimum * 1e6) if throughput else maximum
                    x.append(int(row["rows"]))
                    y.append(value)
                    lower.append(value-low)
                    upper.append(high-value)
                ax.errorbar(x, y, yerr=[lower, upper], marker="o", capsize=3, label=label)
            metric = "Throughput" if throughput else "Latency"
            ax.set(xscale="log", xlabel="Input rows",
                   ylabel="Logical GB/s" if throughput else "Milliseconds",
                   title=f"TPC-DS Q13 Aggregation {metric}\n{subtitle}")
            ax.legend(title=f"{nulls}% null aggregate inputs")
            fig.savefig(destination / f"q13_{metric.lower()}_null{nulls}.svg")
            plt.close(fig)

    for workload in ("segmented_sum", "segmented_sum_max"):
        for nulls in sorted({row["null_percent"] for row in records
                             if row["workload"] == workload}, key=int):
            points = [row for row in records if row["workload"] == workload
                      and row["null_percent"] == nulls]
            shapes = ("small", "large", "uneven")
            methods = (("cudf", "none"), ("lto", "unknown"), ("lto", "known"))
            fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
            width = 0.25
            for index, (engine, hint) in enumerate(methods):
                group = [next(row for row in points
                              if row["shape"] == shape and row["engine"] == engine
                              and row["hint"] == hint) for shape in shapes]
                medians = [float(row["median_ms"]) for row in group]
                ranges = [[float(row["median_ms"])-float(row["min_ms"]) for row in group],
                          [float(row["max_ms"])-float(row["median_ms"]) for row in group]]
                label = "cuDF" if engine == "cudf" else f"LTO ({hint} length)"
                ax.bar([pos+(index-1)*width for pos in range(len(shapes))], medians,
                       width, yerr=ranges, capsize=3, label=label)
            labels = []
            for shape in shapes:
                row = next(row for row in points if row["shape"] == shape)
                labels.append(f"{shape}\n{int(row['rows']):,} rows / {row['segments']} segments")
            ax.set_xticks(range(len(shapes)), labels)
            ax.set(yscale="log", ylabel="Milliseconds (log scale)",
                   title=f"{workload.replace('_', ' ').title()} Latency\n{subtitle}\n{nulls}% null inputs")
            ax.legend()
            fig.savefig(destination / f"{workload}_latency_null{nulls}.svg")
            plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--subtitle", default="")
    args = parser.parse_args()
    with args.csv.open() as source:
        plot(list(csv.DictReader(source)), args.output_dir, args.subtitle)

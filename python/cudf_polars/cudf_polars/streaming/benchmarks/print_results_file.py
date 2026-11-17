# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Read a saved benchmark results file and print it in human-readable form.

The benchmark runners write one JSON object per line to ``--output``, appending
across runs. This reads such a file back and prints the query timings, and the
per-rank I/O summaries when the run was made with ``--rapidsmpf-statistics``.

    python -m cudf_polars.streaming.benchmarks.print_results_file results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import TYPE_CHECKING, Any

from rapidsmpf.utils.string import format_bytes

from cudf_polars.streaming.benchmarks.utils import SuccessRecord, record_from_dict

_DESCRIPTION = (
    "Read a saved benchmark results file and print the query timings, and the "
    "per-rank I/O summaries when the run was made with --rapidsmpf-statistics."
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cudf_polars.streaming.benchmarks.utils import FailedRecord


def load_runs(path: Path) -> list[dict[str, Any]]:
    """
    Read every run from a results file, without interpreting it.

    Parameters
    ----------
    path
        Path to a file written by a benchmark runner's ``--output``.

    Returns
    -------
    One decoded line per run, in the order they were appended.

    Raises
    ------
    ValueError
        If the file holds no runs.
    """
    with path.open() as f:
        runs = [json.loads(line) for line in f if line.strip()]
    if not runs:
        raise ValueError(f"{path} contains no runs")
    return runs


def iter_records(run: dict[str, Any]) -> Iterator[SuccessRecord | FailedRecord]:
    """
    Yield every per-iteration record of a run, ordered by query then iteration.

    Parameters
    ----------
    run
        A single run, as returned by :func:`load_runs`.

    Yields
    ------
    The run's per-iteration records.
    """
    for _, records in sorted(run["records"].items(), key=lambda kv: int(kv[0])):
        yield from map(record_from_dict, records)


def print_header(run: dict[str, Any]) -> None:
    """
    Print the run's identifying configuration.

    Parameters
    ----------
    run
        A single run, as returned by :func:`load_runs`.
    """
    print(f"run       : {run.get('run_id')}  ({run.get('timestamp')})")
    print(f"engine    : {run.get('engine_name')}  frontend={run.get('frontend')}")
    print(f"dataset   : {run.get('dataset_path')}  scale={run.get('scale_factor')}")
    print(f"workers   : {run.get('n_workers')}  iterations={run.get('iterations')}")


def print_timings(run: dict[str, Any]) -> None:
    """
    Print min, max and mean duration for each query.

    Parameters
    ----------
    run
        A single run, as returned by :func:`load_runs`.
    """
    print("\nTimings")
    print(f"  {'query':>6}  {'iters':>5}  {'min':>9}  {'max':>9}  {'mean':>9}")
    total = 0.0
    for query, records in sorted(run["records"].items(), key=lambda kv: int(kv[0])):
        durations = [
            r.duration
            for r in map(record_from_dict, records)
            if isinstance(r, SuccessRecord)
        ]
        if not durations:
            print(f"  {query:>6}  {'-':>5}  {'no successful iterations':>31}")
            continue
        total += mean(durations)
        print(
            f"  {query:>6}  {len(durations):>5}  {min(durations):>8.4f}s  "
            f"{max(durations):>8.4f}s  {mean(durations):>8.4f}s"
        )
    if total > 0:
        print(f"  {'total':>6}  {'':>5}  {'':>9}  {'':>9}  {total:>8.4f}s")


def _io_summaries(
    record: SuccessRecord | FailedRecord,
) -> dict[int, dict[str, Any]]:
    """
    Return a record's I/O summaries keyed by rank.

    Parameters
    ----------
    record
        One per-iteration record.

    Returns
    -------
    The summaries, empty if the iteration recorded none.
    """
    if not isinstance(record, SuccessRecord) or record.io_summaries is None:
        return {}
    return {int(rank): s for rank, s in record.io_summaries.items()}


def print_io_summaries(run: dict[str, Any]) -> None:
    """
    Print the per-rank I/O summaries, one row per rank per iteration.

    Kept per rank rather than totalled, since the point of per-rank I/O
    statistics is to expose skew that a total would hide.

    Parameters
    ----------
    run
        A single run, as returned by :func:`load_runs`.
    """
    rows = [
        (r.query, r.iteration, rank, s)
        for r in iter_records(run)
        for rank, s in sorted(_io_summaries(r).items())
    ]
    if not rows:
        # The run was made without `--rapidsmpf-statistics`, so no rank counted.
        print("\nI/O: not collected (run with --rapidsmpf-statistics)")
        return

    print("\nI/O per rank")
    print(
        f"  {'query':>6}  {'iter':>4}  {'rank':>4}  {'ops':>7}  {'read':>11}  "
        f"{'busy':>9}  {'busy%':>6}  {'bandwidth':>11}  backends"
    )
    for query, iteration, rank, s in rows:
        backends = (
            ",".join(
                name
                for name, totals in s.get("by_backend", {}).items()
                if totals.get("num_ops")
            )
            or "-"
        )
        print(
            f"  {query:>6}  {iteration:>4}  {rank:>4}  {s['num_ops']:>7}  "
            f"{format_bytes(s['bytes_read']):>11}  {s['busy_ns'] / 1e6:>7.1f}ms  "
            f"{s['busy_fraction'] * 100:>5.1f}%  "
            f"{s['busy_bytes_per_sec'] / 1e9:>7.2f}GB/s  {backends}"
        )

    # Skew is what the per-rank view is for, so say it outright rather than
    # leaving it to be eyeballed across rows.
    by_iteration: dict[tuple[Any, Any], list[int]] = {}
    for query, iteration, _, s in rows:
        by_iteration.setdefault((query, iteration), []).append(s["bytes_read"])
    shared = {k: v for k, v in by_iteration.items() if len(v) > 1}

    # A rank that read nothing while its peers did is the most extreme skew there
    # is, and it has no finite ratio, so it is reported rather than skipped.
    starved = sorted(k for k, v in shared.items() if min(v) == 0 < max(v))
    if starved:
        where = ", ".join(f"q{q} iter {i}" for q, i in starved)
        print(f"\n  ranks that read nothing while peers did: {where}")

    worst = max(
        ((max(v) / min(v), k) for k, v in shared.items() if min(v) > 0),
        default=None,
    )
    if worst is not None:
        ratio, (query, iteration) = worst
        label = "widest finite read skew" if starved else "widest read skew"
        print(f"\n  {label}: {ratio:.2f}x (query {query}, iteration {iteration})")


def main(args: argparse.Namespace) -> None:
    """
    Print the requested runs.

    Parameters
    ----------
    args
        Parsed command-line arguments.
    """
    runs = load_runs(args.path)
    selected = runs if args.all else runs[-1:]
    if not args.all and len(runs) > 1:
        print(f"({len(runs)} runs in {args.path}, showing the last, --all for every)\n")

    for i, run in enumerate(selected):
        if i:
            print()
        print("=" * 78)
        print_header(run)
        print("=" * 78)
        print_timings(run)
        if not args.no_io:
            print_io_summaries(run)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Parse command-line arguments.

    Parameters
    ----------
    argv
        Argument list, or ``None`` to read ``sys.argv``.

    Returns
    -------
    The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=_DESCRIPTION)
    parser.add_argument(
        "path", type=Path, help="Results file written by a runner's --output."
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print every run in the file, not just the last one.",
    )
    parser.add_argument(
        "--no-io",
        action="store_true",
        help="Skip the per-rank I/O summaries.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        main(parse_args())
    except ValueError as e:
        raise SystemExit(f"error: {e}") from None

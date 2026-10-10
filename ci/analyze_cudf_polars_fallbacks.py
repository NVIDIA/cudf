#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize JSON GPU execution diagnostics from upstream Polars CI runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def build_report(paths: list[Path]) -> dict[str, Any]:
    """Combine available shard/configuration reports without losing run identity."""
    tests = []
    runs = []
    unreadable = []
    for path in paths:
        try:
            run = json.loads(path.read_text())
            engine = run["engine"]
            if engine == "spmd":
                engine += f"-{run['blocksize']}"
            run["exitstatus"]
            run["collected"]
            run_tests = run.pop("tests")
            # Equal node IDs in different matrix jobs are separate executions,
            # not duplicates: fallback can depend on the tested configuration.
            run_tests = [
                {
                    "nodeid": test["nodeid"],
                    "outcome": test["outcome"],
                    "fallback": test["fallback"],
                    "gpu_attempted": test.get("gpu_attempted", "unknown"),
                    "gpu_executed": test.get("gpu_executed", "unknown"),
                    "engine": engine,
                    "run": path.name,
                }
                for test in run_tests
            ]
            tests.extend(run_tests)
            runs.append({**run, "engine": engine, "run": path.name})
        except (OSError, ValueError, KeyError, TypeError) as error:
            unreadable.append({"run": path.name, "error": str(error)})

    grouped = defaultdict(list)
    for test in tests:
        grouped[test["engine"]].append(test)
    summaries = {}
    for engine in sorted({run["engine"] for run in runs}):
        engine_runs = [run for run in runs if run["engine"] == engine]
        engine_tests = grouped[engine]
        summaries[engine] = {
            "runs": len(engine_runs),
            "nonzero_exitstatus": sum(
                run["exitstatus"] != 0 for run in engine_runs
            ),
            "collected": sum(run["collected"] for run in engine_runs),
            "total": len(engine_tests),
            "fallback": sum(
                test["fallback"] == "true" for test in engine_tests
            ),
            "no_fallback_observed": sum(
                test["fallback"] == "false" for test in engine_tests
            ),
            "unknown": sum(
                "unknown"
                in (
                    test["fallback"],
                    test["gpu_attempted"],
                    test["gpu_executed"],
                )
                for test in engine_tests
            ),
            "gpu_attempted": sum(
                test["gpu_attempted"] == "true" for test in engine_tests
            ),
            "gpu_executed": sum(
                test["gpu_executed"] == "true" for test in engine_tests
            ),
            "execution": dict(
                Counter(execution_category(test) for test in engine_tests)
            ),
            "outcomes": dict(
                sorted(
                    Counter(test["outcome"] for test in engine_tests).items()
                )
            ),
        }
    return {
        "summary": summaries,
        "runs": runs,
        "unreadable_reports": unreadable,
        "tests": sorted(
            tests,
            key=lambda test: (test["engine"], test["nodeid"], test["run"]),
        ),
    }


def execution_category(test: dict[str, str]) -> str:
    """Distinguish observed GPU success from merely avoiding CPU fallback."""
    if "unknown" in (
        test["fallback"],
        test["gpu_attempted"],
        test["gpu_executed"],
    ):
        return "unknown"
    if test["gpu_executed"] == "true":
        return "mixed" if test["fallback"] == "true" else "gpu_only"
    if test["gpu_attempted"] == "true":
        return "attempted_without_success"
    return "not_attempted"


def summary_markdown(report: dict[str, Any]) -> str:
    """Describe observed executions, not an assumed complete baseline."""
    lines = [
        "## Upstream Polars GPU execution diagnostics",
        "",
        "Counts include executions across the available shards and matrix configurations. "
        "Missing jobs are not represented; failed or partial runs are not a complete baseline.",
        "",
        "GPU success means the GPU executor returned a result, not merely that translation succeeded. "
        "Mixed tests observed both GPU success and CPU fallback. Tests with no successful GPU query "
        "may fall back or raise; per-test signals and outcomes are in the JSON artifact.",
        "",
        "| Engine | Reports | GPU success, no fallback | Mixed GPU / CPU | Attempted, no GPU success | Never attempted | Missing telemetry | Reported / collected | Nonzero runs |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for engine, summary in report["summary"].items():
        execution = summary["execution"]
        lines.append(
            f"| {engine} | {summary['runs']} | {execution.get('gpu_only', 0)} | "
            f"{execution.get('mixed', 0)} | {execution.get('attempted_without_success', 0)} | "
            f"{execution.get('not_attempted', 0)} | {summary['unknown']} | "
            f"{summary['total']} / {summary['collected']} | {summary['nonzero_exitstatus']} |"
        )
    if not report["runs"]:
        lines.extend(
            ["", "No readable diagnostics were uploaded by the upstream jobs."]
        )
    if report["unreadable_reports"]:
        lines.extend(
            [
                "",
                f"Unreadable reports: {len(report['unreadable_reports'])}; see the JSON artifact.",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path)
    args = parser.parse_args()
    report = build_report(sorted(args.reports_dir.glob("*.json")))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fallback-diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if args.summary_file:
        with args.summary_file.open("a") as summary:
            summary.write(summary_markdown(report))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize JSON GPU execution diagnostics from upstream Polars CI runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PR_SELECTION_ENGINE = "spmd-default"
PR_PRODUCER_PREFIX = "cudf-polars-polars-tests (spmd, default,"


def configuration_key(run: dict[str, Any]) -> str | tuple[Any, ...]:
    """Keep GPU/driver/OS matrix variants distinct when combining shards."""
    return run.get("configuration") or tuple(
        run.get(name) for name in ("python", "arch", "cuda", "dependencies")
    )


def validate_report(run: dict[str, Any], tests: list[dict[str, Any]]) -> None:
    """Reject malformed reports before calculating summaries or candidates."""
    if run["engine"] not in ("in-memory", "spmd"):
        raise ValueError(f"Invalid injected engine: {run['engine']!r}")
    for name in (
        "python",
        "arch",
        "cuda",
        "dependencies",
        "configuration",
        "polars_version",
    ):
        if (value := run.get(name)) is not None and not isinstance(value, str):
            raise ValueError(f"Invalid {name}: {value!r}")
    if any(
        name in run
        and (
            not isinstance(run[name], list)
            or not all(isinstance(node, str) for node in run[name])
        )
        for name in (
            "collected_nodeids",
            "collection_nodeids",
            "collection_skips",
        )
    ):
        raise ValueError("Invalid collection inventory")
    outcomes = {
        "incomplete",
        "passed",
        "xpassed",
        "xfailed",
        "skipped",
        "failed",
        "error",
    }
    for test in tests:
        if (
            not isinstance(test["nodeid"], str)
            or test["outcome"] not in outcomes
        ):
            raise ValueError(f"Invalid pytest item: {test!r}")
        for name in ("fallback", "gpu_attempted", "gpu_executed"):
            if test[name] not in ("true", "false", "unknown"):
                raise ValueError(
                    f"Invalid {name} for {test['nodeid']}: {test[name]!r}"
                )


def build_report(paths: list[Path]) -> dict[str, Any]:
    """Combine available shard/configuration reports without losing run identity."""
    tests = []
    runs = []
    unreadable = []
    for path in paths:
        try:
            raw_report = path.read_bytes()
            run = json.loads(raw_report)
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
                    **test,
                    "gpu_attempted": test.get("gpu_attempted", "unknown"),
                    "gpu_executed": test.get("gpu_executed", "unknown"),
                    "engine": engine,
                    "run": path.name,
                }
                for test in run_tests
            ]
            validate_report(run, run_tests)
            tests.extend(run_tests)
            runs.append(
                {
                    **run,
                    "engine": engine,
                    "run": path.name,
                    "source_artifact": path.as_posix(),
                    "report_sha256": hashlib.sha256(raw_report).hexdigest(),
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            unreadable.append({"run": path.name, "error": str(error)})

    grouped = defaultdict(list)
    for test in tests:
        grouped[test["engine"]].append(test)
    summaries = {}
    for engine in sorted({run["engine"] for run in runs}):
        engine_runs = [run for run in runs if run["engine"] == engine]
        engine_tests = grouped[engine]
        unfiltered = defaultdict(set)
        for run in engine_runs:
            configuration = (run.get("polars_version"), configuration_key(run))
            unfiltered[configuration].update(run.get("collection_nodeids", []))
        summaries[engine] = {
            "runs": len(engine_runs),
            "nonzero_exitstatus": sum(
                run["exitstatus"] != 0 for run in engine_runs
            ),
            "collected": sum(run["collected"] for run in engine_runs),
            "total": len(engine_tests),
            "unfiltered_collected": sum(
                len(nodes) for nodes in unfiltered.values()
            ),
            "selection_modes": sorted(
                {run.get("selection_mode", "unknown") for run in engine_runs}
            ),
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


def selection_key(run: dict[str, Any]) -> str:
    """Keep support observations tied to their upstream version and engine."""
    engine = run["engine"]
    variant, blocksize = (
        engine.rsplit("-", 1)
        if engine.startswith("spmd-")
        else (engine, "default")
    )
    return f"{run['polars_version']}|{variant}|{blocksize}"


def audit_report(
    report: dict[str, Any],
    maintained: dict[str, Any],
    jobs: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Generate reviewed-list candidates only from complete full-suite evidence."""
    # The other nightly configurations remain diagnostic; only the PR
    # configuration has a maintained selection whose drift can be gated.
    errors = (
        [
            f"Unreadable report: {run['run']}"
            for run in report["unreadable_reports"]
        ]
        if jobs is None
        else []
    )
    runs = [
        run
        for run in report["runs"]
        if run.get("engine") == PR_SELECTION_ENGINE
    ]
    selected_runs = {run["run"] for run in runs}
    by_run = defaultdict(list)
    for test in report["tests"]:
        if test.get("run") in selected_runs:
            by_run[test["run"]].append(test)
    cohorts = defaultdict(list)
    for run in runs:
        tests = by_run[run["run"]]
        try:
            key = selection_key(run)
            if (
                any(
                    not isinstance(run[name], str) or not run[name]
                    for name in ("polars_version", "cudf_sha", "upstream_sha")
                )
                or run["shard_id"] not in range(run["num_shards"])
                or any(
                    not isinstance(run[name], list)
                    or not all(isinstance(node, str) for node in run[name])
                    for name in (
                        "collected_nodeids",
                        "collection_nodeids",
                        "collection_skips",
                    )
                )
            ):
                errors.append(f"Invalid collection metadata: {run['run']}")
                continue
            inventory = set(run["collected_nodeids"])
            observed = {test["nodeid"] for test in tests}
            if (
                run["selection_mode"] != "full"
                or run["exitstatus"] != 0
                or run["consistent"] is not True
                or len(tests) != run["collected"]
                or len(observed) != len(tests)
                or observed != inventory
                or len(inventory) != len(run["collected_nodeids"])
                or not all(test.get("complete") is True for test in tests)
                or any(test["outcome"] == "incomplete" for test in tests)
                or any(execution_category(test) == "unknown" for test in tests)
            ):
                errors.append(f"Incomplete full-suite evidence: {run['run']}")
            cohort = (key, configuration_key(run))
            cohorts[cohort].append(run)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"Missing audit metadata in {run['run']}: {error}")
    for cohort, shards in cohorts.items():
        counts = {run["num_shards"] for run in shards}
        ids = [run["shard_id"] for run in shards]
        if len(counts) != 1 or sorted(ids) != list(range(next(iter(counts)))):
            errors.append(f"Missing or duplicated shards: {cohort}")
        if len({tuple(run["collection_nodeids"]) for run in shards}) != 1:
            errors.append(f"Inconsistent collection across shards: {cohort}")
        if (
            len({tuple(run.get("collection_skips", [])) for run in shards})
            != 1
        ):
            errors.append(
                f"Inconsistent collection skips across shards: {cohort}"
            )
    validated_runs = [run for shards in cohorts.values() for run in shards]
    if len({run["cudf_sha"] for run in validated_runs}) != 1:
        errors.append("Discovery reports span multiple cuDF revisions")
    upstream_revisions = defaultdict(set)
    for run in validated_runs:
        upstream_revisions[run["polars_version"]].add(run["upstream_sha"])
    if any(len(revisions) != 1 for revisions in upstream_revisions.values()):
        errors.append(
            "A Polars version has conflicting upstream checkout revisions"
        )
    configurations = defaultdict(set)
    for key, _ in cohorts:
        version, configuration = key.split("|", 1)
        configurations[version].add(configuration)
    if not configurations or any(
        observed != {"spmd|default"} for observed in configurations.values()
    ):
        errors.append(
            "Missing full default-SPMD discovery runs for a Polars version"
        )
    if jobs is not None:
        if isinstance(jobs, list):
            jobs = [
                job
                for job in jobs
                if isinstance(job, dict)
                and isinstance(job.get("name"), str)
                and job["name"].startswith(PR_PRODUCER_PREFIX)
            ]
        if not isinstance(jobs, list) or not all(
            isinstance(job, dict)
            and isinstance(job.get("name"), str)
            and isinstance(job.get("conclusion"), str)
            and isinstance(job.get("id"), int)
            and not isinstance(job["id"], bool)
            and job["id"] >= 0
            for job in jobs
        ):
            errors.append("Invalid GitHub producer-job inventory")
            jobs = []
        if len({job["id"] for job in jobs}) != len(jobs):
            errors.append("Duplicated GitHub producer jobs")
        # Cancelled producers may have no steps. Identify helpers by name rather
        # than requiring a test step that might never have been scheduled.
        helpers = [
            job for job in jobs if job["name"].endswith(" / compute-matrix")
        ]
        if any(job["conclusion"] != "success" for job in helpers):
            errors.append(
                "A GitHub producer matrix did not finish successfully"
            )
        jobs = [job for job in jobs if job not in helpers]
        expected = {str(job["id"]) for job in jobs}
        producers = [run.get("producer") for run in runs]
        if (
            not expected
            or set(producers) != expected
            or len(producers) != len(expected)
            or any(job["conclusion"] != "success" for job in jobs)
        ):
            errors.append(
                "Reports do not match all successful GitHub producer jobs"
            )
        else:
            job_names = {str(job["id"]): job["name"] for job in jobs}
            for run in runs:
                if (
                    run.get("configuration")
                    != job_names[run["producer"]].rsplit(" / ", 1)[-1]
                ):
                    errors.append(
                        f"Incorrect producer configuration: {run['run']}"
                    )
    if errors:
        return {
            "status": "incomplete",
            "errors": errors,
            "differences": {},
        }, {}

    generated: dict[str, Any] = {"schema_version": 1, "selections": {}}
    differences = {}
    for key in sorted({cohort[0] for cohort in cohorts}):
        key_runs = [run for run in runs if selection_key(run) == key]
        tests = [test for run in key_runs for test in by_run[run["run"]]]
        gpu = {
            test["nodeid"] for test in tests if test["gpu_executed"] == "true"
        }
        collected = set().union(
            *(set(run["collection_nodeids"]) for run in key_runs)
        )
        exercised = {
            test["nodeid"] for test in tests if test["outcome"] != "skipped"
        }
        entry = maintained.get("selections", {}).get(key, {})
        previous = set(entry.get("nodes", []))
        skipped_modules = set().union(
            *(set(run.get("collection_skips", [])) for run in key_runs)
        )
        stale = {
            node
            for node in previous - collected
            if not any(node.startswith(module) for module in skipped_modules)
        }
        lost = (previous & exercised) - gpu
        unobserved = previous - exercised - stale
        generated["selections"][key] = {
            "nodes": sorted(gpu | unobserved),
            "baseline": [
                {
                    name: run.get(name)
                    for name in (
                        "upstream_sha",
                        "run",
                        "source_artifact",
                        "producer",
                        "configuration",
                        "workflow_run",
                        "run_attempt",
                        "report_sha256",
                        "python",
                        "arch",
                        "cuda",
                        "dependencies",
                    )
                    if name not in {"workflow_run", "run_attempt"}
                    or run.get(name) is not None
                }
                for run in sorted(key_runs, key=lambda run: run["run"])
            ],
        }
        differences[key] = {
            "added": sorted(gpu - previous),
            "lost_gpu_execution": sorted(lost),
            "stale": sorted(stale),
            "unobserved": sorted(unobserved),
            "configuration_dependent": sorted(
                {
                    test["nodeid"]
                    for test in tests
                    if test["nodeid"] in gpu
                    and test["gpu_executed"] == "false"
                    and test["outcome"] != "skipped"
                }
            ),
        }
    for key in sorted(
        set(maintained.get("selections", {})) - set(generated["selections"])
    ):
        # Retain older Polars versions only for the configuration PR CI uses.
        if key.endswith("|spmd|default"):
            generated["selections"][key] = maintained["selections"][key]
    drift = any(
        row[field]
        for row in differences.values()
        for field in ("added", "lost_gpu_execution", "stale")
    )
    return {
        "status": "drift" if drift else "current",
        "errors": [],
        "differences": differences,
    }, generated


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
        "| Engine | Selection | Reports | GPU success, no fallback | Mixed GPU / CPU | Attempted, no GPU success | No GPU attempt observed | Missing telemetry | Reported / selected / unfiltered | Nonzero runs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for engine, summary in report["summary"].items():
        execution = summary["execution"]
        lines.append(
            f"| {engine} | {', '.join(summary.get('selection_modes', ['unknown']))} | {summary['runs']} | {execution.get('gpu_only', 0)} | "
            f"{execution.get('mixed', 0)} | {execution.get('attempted_without_success', 0)} | "
            f"{execution.get('not_attempted', 0)} | {summary['unknown']} | "
            f"{summary['total']} / {summary['collected']} / {summary.get('unfiltered_collected', 0)} | {summary['nonzero_exitstatus']} |"
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
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--reports-dir", type=Path)
    inputs.add_argument("--diagnostics-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-file", type=Path)
    parser.add_argument("--audit-run-list", type=Path)
    parser.add_argument("--expected-jobs", type=Path)
    parser.add_argument("--generate-run-list", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.audit_run_list or args.generate_run_list:
        # A parsing failure must not leave a previous successful audit or
        # removal proposal available to the publication step.
        (args.output_dir / "gpu-run-list-candidate.json").unlink(
            missing_ok=True
        )
        (args.output_dir / "run-list-drift.json").write_text(
            json.dumps(
                {
                    "status": "incomplete",
                    "errors": ["Audit did not finish"],
                    "differences": {},
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    report = (
        json.loads(args.diagnostics_file.read_text())
        if args.diagnostics_file
        else build_report(sorted(args.reports_dir.glob("*.json")))
    )
    (args.output_dir / "fallback-diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    if args.summary_file:
        with args.summary_file.open("a") as summary:
            summary.write(summary_markdown(report))
    if args.audit_run_list or args.generate_run_list:
        maintained = (
            json.loads(args.audit_run_list.read_text())
            if args.audit_run_list
            else {}
        )
        jobs = (
            json.loads(args.expected_jobs.read_text())
            if args.expected_jobs
            else None
        )
        audit, generated = audit_report(report, maintained, jobs)
        print(f"GPU run-list audit: {audit['status']}")
        for key, row in audit["differences"].items():
            print(
                f"{key}: {len(row['added'])} added, "
                f"{len(row['lost_gpu_execution'])} lost GPU execution, "
                f"{len(row['stale'])} stale"
            )
        for error in audit["errors"]:
            print(f"Incomplete evidence: {error}")
        print(f"Exact differences: {args.output_dir / 'run-list-drift.json'}")
        (args.output_dir / "run-list-drift.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n"
        )
        if generated:
            (args.output_dir / "gpu-run-list-candidate.json").write_text(
                json.dumps(generated, indent=2, sort_keys=True) + "\n"
            )
            if args.generate_run_list:
                args.generate_run_list.write_text(
                    json.dumps(generated, indent=2, sort_keys=True) + "\n"
                )
        if args.summary_file:
            with args.summary_file.open("a") as summary:
                summary.write(
                    f"\n## GPU run-list audit: {audit['status']}\n\n"
                )
                for key, row in audit["differences"].items():
                    summary.write(
                        f"- {key}: {len(row['added'])} additions, {len(row['lost_gpu_execution'])} lost GPU observations, {len(row['stale'])} stale nodes.\n"
                    )
                for error in audit["errors"]:
                    summary.write(f"- {error}\n")
        if audit["status"] == "incomplete" or (
            args.audit_run_list and audit["status"] == "drift"
        ):
            raise SystemExit(1)


if __name__ == "__main__":
    main()

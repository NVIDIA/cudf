#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export local-only, upstream-only, and shared cudf-polars coverage."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from coverage import CoverageData

MAX_CONTEXTS_PER_LINE = 20


def _read_data(path: Path) -> CoverageData:
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f"Missing or empty coverage data: {path}")
    data = CoverageData(basename=str(path))
    data.read()
    if not any(
        context.endswith("|run") for context in data.measured_contexts()
    ):
        raise ValueError(
            f"No test-body coverage contexts in {path}; collect with "
            "--cov-context=test using pytest-xdist."
        )
    return data


def build_report(
    local: CoverageData, upstream: list[CoverageData]
) -> dict[str, Any]:
    """Compare source lines without inferring equivalent test assertions."""
    files = {}
    totals = {"local_only": 0, "upstream_only": 0, "both": 0}
    filenames = local.measured_files().union(
        *(data.measured_files() for data in upstream)
    )
    for filename in sorted(filenames):
        if "/cudf_polars/" not in filename.replace("\\", "/"):
            continue
        local_lines = set(local.lines(filename) or ())
        upstream_lines = set().union(
            *(data.lines(filename) or () for data in upstream)
        )
        local_contexts = local.contexts_by_lineno(filename)
        upstream_contexts: dict[int, set[str]] = defaultdict(set)
        for data in upstream:
            for line, contexts in data.contexts_by_lineno(filename).items():
                upstream_contexts[line].update(contexts)
        categories: dict[str, list[Any]] = {
            "local_only": [],
            "upstream_only": [],
            "both": [],
        }
        for line in sorted(local_lines | upstream_lines):
            entry: dict[str, Any] = {"line": line}
            for suite, contexts in (
                ("local", local_contexts),
                ("upstream", upstream_contexts),
            ):
                values = sorted(contexts.get(line, ()))
                # Common paths have thousands of contexts; the raw databases
                # remain the complete export without duplicating them in JSON.
                entry[f"{suite}_contexts"] = values[:MAX_CONTEXTS_PER_LINE]
                entry[f"{suite}_context_count"] = len(values)
                entry[f"{suite}_contexts_truncated"] = (
                    len(values) > MAX_CONTEXTS_PER_LINE
                )
            if line in local_lines and line in upstream_lines:
                category = "both"
            else:
                category = (
                    "local_only" if line in local_lines else "upstream_only"
                )
            categories[category].append(entry)
            totals[category] += 1
        if any(categories.values()):
            files[filename] = categories
    return {"files": files, "summary": {**totals, "files": len(files)}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-data", type=Path, required=True)
    parser.add_argument(
        "--upstream-data", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cudf-sha", required=True)
    parser.add_argument("--polars-sha", required=True)
    parser.add_argument("--polars-version", required=True)
    parser.add_argument("--collection-metadata", type=json.loads)
    parser.add_argument("--local-exit-status", type=int)
    parser.add_argument("--upstream-in-memory-exit-status", type=int)
    parser.add_argument("--upstream-spmd-exit-status", type=int)
    args = parser.parse_args()
    try:
        local = _read_data(args.local_data)
        upstream = [_read_data(path) for path in args.upstream_data]
    except ValueError as error:
        parser.error(str(error))
    report = build_report(local, upstream)
    report["provenance"] = {
        "cudf_sha": args.cudf_sha,
        "polars_sha": args.polars_sha,
        "polars_version": args.polars_version,
    }
    report["collection"] = args.collection_metadata
    report["context_data"] = {
        "local": os.path.relpath(args.local_data.resolve(), args.output_dir),
        "upstream": [
            os.path.relpath(path.resolve(), args.output_dir)
            for path in args.upstream_data
        ],
    }
    report["suite_exit_statuses"] = {
        "local": args.local_exit_status,
        "upstream_in_memory": args.upstream_in_memory_exit_status,
        "upstream_spmd": args.upstream_spmd_exit_status,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "coverage-comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

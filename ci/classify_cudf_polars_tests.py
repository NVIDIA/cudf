#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify local cudf-polars tests using broad and optional strict coverage."""

from __future__ import annotations

import argparse
import ast
import html
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from coverage import CoverageData
from coverage.numbits import numbits_to_nums

SOURCE_MARKER = "/cudf_polars/"
INTROSPECTIVE_TEST_FILES = frozenset(
    {
        "tests/dsl/test_nodebase.py",
        "tests/dsl/test_serialization.py",
        "tests/dsl/test_to_ast.py",
        "tests/dsl/test_traversal.py",
        "tests/test_repr.py",
    }
)
SCAN_IO_TERMS = ("scan", "/io", "parquet", "sink")
INTERNAL_TEST_PREFIXES = (
    "tests/containers/",
    "tests/dsl/",
    "tests/quent/",
    "tests/testing/",
    "tests/utils/",
)
INTERNAL_TEST_FILES = frozenset(
    {
        "tests/test_cache.py",
        "tests/test_config.py",
        "tests/test_engine_execute.py",
        "tests/test_executors.py",
        "tests/test_hconcat.py",
        "tests/test_mapfunction.py",
        "tests/test_profile.py",
        "tests/test_tracing.py",
        "tests/test_tracing_disabled.py",
        "tests/test_unstable.py",
    }
)
INTROSPECTIVE_NAME_TERMS = (
    "hash",
    "metadata",
    "repr",
    "signature",
    "sorted_flags",
    "stable_id",
)
NUMERIC_DTYPE_TERMS = ("decimal", "dtype", "float", "nan", "null")
RANK_AWARE_SOURCE_TEST = "tests/test_rank_aware_source.py"

NEXT_ACTION = {
    "defer_scan_io": "defer_until_upstream_scan_io_is_practical",
    "deletion_review_candidate": "manually_compare_public_contract_then_delete_or_retain",
    "retain_internal_contract": "retain",
    "retain_no_source_coverage": "inspect_if_a_future_pruning_batch_targets_this_module",
    "retain_not_strictly_covered": "seek_strict_upstream_coverage_before_pruning",
    "retain_numeric_or_dtype": "retain",
    "retain_streaming_or_engine": "retain",
    "retain_unique_local_coverage": "retain_or_upstream_missing_behavior",
    "retain_unsupported_or_fallback": "retain",
    "rewrite_introspective": "rewrite_as_public_behavioral_test",
}


@dataclass
class TestCoverage:
    """Coverage evidence accumulated for one pytest node ID."""

    nodeid: str
    total_lines: int = 0
    local_only_lines: int = 0
    strict_lines: int = 0
    source_files: set[str] = field(default_factory=set)


def _read_data(path: Path) -> CoverageData:
    data = CoverageData(basename=str(path))
    data.read()
    return data


def _source_lines(data: CoverageData) -> dict[str, set[int]]:
    return {
        filename: set(data.lines(filename) or ())
        for filename in data.measured_files()
        if SOURCE_MARKER in filename.replace("\\", "/")
    }


def _nodeid(context: str) -> str:
    return context.rsplit("|", 1)[0]


def _test_file(nodeid: str) -> str:
    return nodeid.split("::", 1)[0]


def _test_name(nodeid: str) -> str:
    return nodeid.split("::", 1)[1].split("[", 1)[0]


def _unsupported_test_names(tests_root: Path) -> set[tuple[str, str]]:
    """Return local test functions that explicitly assert GPU unavailability."""
    result: set[tuple[str, str]] = set()
    for path in tests_root.rglob("test_*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        relative = f"tests/{path.relative_to(tests_root).as_posix()}"
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            calls = {
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
            }
            calls.update(
                child.func.attr
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
            )
            if {
                "assert_ir_translation_raises",
                "assert_gpu_raises",
                "raises",
            } & calls:
                result.add((relative, node.name))
    return result


def _classification(
    coverage: TestCoverage, unsupported: set[tuple[str, str]]
) -> str:
    """Apply the conservative pruning policy in precedence order."""
    path = _test_file(coverage.nodeid)
    if any(term in path.lower() for term in SCAN_IO_TERMS):
        return "defer_scan_io"
    if path == RANK_AWARE_SOURCE_TEST:
        return "defer_scan_io"
    if path.startswith("tests/streaming/"):
        return "retain_streaming_or_engine"
    if path in INTROSPECTIVE_TEST_FILES:
        return "rewrite_introspective"
    if path.startswith(INTERNAL_TEST_PREFIXES) or path in INTERNAL_TEST_FILES:
        return "retain_internal_contract"
    if any(
        term in _test_name(coverage.nodeid)
        for term in INTROSPECTIVE_NAME_TERMS
    ):
        return "retain_internal_contract"
    if (path, _test_name(coverage.nodeid)) in unsupported:
        return "retain_unsupported_or_fallback"
    if any(term in coverage.nodeid.lower() for term in NUMERIC_DTYPE_TERMS):
        return "retain_numeric_or_dtype"
    if coverage.local_only_lines:
        return "retain_unique_local_coverage"
    if coverage.total_lines and coverage.strict_lines == coverage.total_lines:
        return "deletion_review_candidate"
    return "retain_not_strictly_covered"


def classify(
    local_data: Path,
    broad_data: Path,
    strict_data: Path | None,
    tests_root: Path,
    collected_nodeids: set[str] | None = None,
) -> dict[str, Any]:
    """Classify every local pytest context that executes cudf-polars source."""
    broad_lines = _source_lines(_read_data(broad_data))
    strict_lines = (
        _source_lines(_read_data(strict_data))
        if strict_data is not None
        else {}
    )
    test_coverage: dict[str, TestCoverage] = {}
    connection = sqlite3.connect(local_data)
    rows = connection.execute(
        """
        SELECT file.path, context.context, line_bits.numbits
        FROM line_bits
        JOIN file ON file.id = line_bits.file_id
        JOIN context ON context.id = line_bits.context_id
        WHERE file.path LIKE ? AND context.context LIKE 'tests/%'
        """,
        (f"%{SOURCE_MARKER}%",),
    )
    for filename, context, numbits in rows:
        nodeid = _nodeid(context)
        covered = set(numbits_to_nums(numbits))
        if not covered:
            continue
        record = test_coverage.setdefault(nodeid, TestCoverage(nodeid))
        record.total_lines += len(covered)
        record.local_only_lines += len(
            covered - broad_lines.get(filename, set())
        )
        record.strict_lines += len(covered & strict_lines.get(filename, set()))
        record.source_files.add(filename)
    connection.close()

    unsupported = _unsupported_test_names(tests_root)
    tests = []
    for record in test_coverage.values():
        classification = _classification(record, unsupported)
        tests.append(
            {
                "nodeid": record.nodeid,
                "classification": classification,
                "next_action": NEXT_ACTION[classification],
                "covered_line_count": record.total_lines,
                "local_only_line_count": record.local_only_lines,
                "strict_line_count": record.strict_lines,
                "source_files": sorted(record.source_files),
            }
        )
    for nodeid in sorted((collected_nodeids or set()) - set(test_coverage)):
        tests.append(
            {
                "nodeid": nodeid,
                "classification": "retain_no_source_coverage",
                "next_action": NEXT_ACTION["retain_no_source_coverage"],
                "covered_line_count": 0,
                "local_only_line_count": 0,
                "strict_line_count": 0,
                "source_files": [],
            }
        )
    tests.sort(key=lambda test: (test["classification"], test["nodeid"]))
    summary = Counter(test["classification"] for test in tests)
    return {"summary": dict(sorted(summary.items())), "tests": tests}


def _write_html(report: dict[str, Any], output: Path) -> None:
    rows = []
    for test in report["tests"]:
        source_files = "<br>".join(
            html.escape(filename.rsplit(SOURCE_MARKER, 1)[-1])
            for filename in test["source_files"]
        )
        rows.append(
            "<tr>"
            f"<td>{html.escape(test['classification'])}</td>"
            f"<td>{html.escape(test['next_action'])}</td>"
            f"<td><code>{html.escape(test['nodeid'])}</code></td>"
            f"<td>{test['covered_line_count']}</td>"
            f"<td>{test['local_only_line_count']}</td>"
            f"<td>{test['strict_line_count']}</td>"
            f"<td>{source_files}</td></tr>"
        )
    summary = " ".join(
        f"{html.escape(category)}: {count};"
        for category, count in report["summary"].items()
    )
    output.write_text(
        "<!doctype html><meta charset=utf-8>"
        "<title>cudf-polars local-test classification</title>"
        "<h1>cudf-polars local-test classification</h1>"
        f"<p>{summary}</p>"
        "<p>Deletion-review candidates have no local-only covered lines and "
        "all their covered lines are exercised by the strict no-fallback "
        "upstream suite. If strict coverage was not supplied, no test is "
        "classified as a deletion-review candidate. They still require "
        "semantic review.</p>"
        "<table><thead><tr><th>Classification</th><th>Next action</th><th>Local node</th>"
        "<th>Covered lines</th><th>Local-only</th><th>Strict</th>"
        "<th>Source files</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-data", type=Path, required=True)
    parser.add_argument("--broad-data", type=Path, required=True)
    parser.add_argument(
        "--strict-data",
        type=Path,
        help=(
            "coverage data from a no-fallback upstream run; omitted until the "
            "strict GPU policy is available"
        ),
    )
    parser.add_argument("--tests-root", type=Path, required=True)
    parser.add_argument(
        "--collected-nodeids",
        type=Path,
        help="pytest --collect-only -q output for tests without source coverage",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    collected_nodeids = None
    if args.collected_nodeids is not None:
        collected_nodeids = {
            line.strip()
            for line in args.collected_nodeids.read_text().splitlines()
            if line.startswith("tests/") and "::" in line
        }
    report = classify(
        args.local_data,
        args.broad_data,
        args.strict_data,
        args.tests_root,
        collected_nodeids,
    )
    (args.output_dir / "local-test-classification.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _write_html(report, args.output_dir / "local-test-classification.html")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify cudf-polars source lines covered by local and upstream tests."""

from __future__ import annotations

import argparse
import html
import json
from itertools import islice
from pathlib import Path
from typing import Any

from coverage import CoverageData

MAX_CONTEXTS_PER_LINE = 20
MAX_CANDIDATE_EXAMPLES = 3
MAX_CANDIDATE_CONTEXTS_PER_LINE = 5

# Candidate discovery deliberately focuses on ordinary, non-streaming public
# behavior. The report is a review aid, not an automatic admission mechanism.
STRICT_CANDIDATE_SOURCE_SUFFIXES = (
    "/dsl/translate.py",
    "/dsl/expressions/aggregation.py",
    "/dsl/expressions/boolean.py",
    "/containers/column.py",
)
STRICT_CANDIDATE_EXCLUDED_PATHS = (
    "/datatypes/",
    "/io/",
    "/streaming/",
    "/testing/",
)
STRICT_CANDIDATE_EXCLUDED_TESTS = (
    "test_describe.py",
    "test_null_count.py",
    "test_cse.py",
    "test_optimizations.py",
    "test_statistics.py",
)
STRICT_CANDIDATE_EXCLUDED_TERMS = (
    "decimal",
    "dtype",
    "kurtosis",
    "quantile",
    "skew",
)


def _read_data(path: Path) -> CoverageData:
    data = CoverageData(basename=str(path))
    data.read()
    return data


def _nodeid(context: str) -> str:
    """Remove coverage.py's pytest phase suffix from a test context."""
    return context.rsplit("|", 1)[0]


def _is_strict_candidate_source(filename: str) -> bool:
    normalized = filename.replace("\\", "/")
    return normalized.endswith(STRICT_CANDIDATE_SOURCE_SUFFIXES)


def _is_strict_candidate_context(context: str) -> bool:
    """Whether a context belongs in the focused upstream-unit review pool."""
    nodeid = _nodeid(context)
    return (
        nodeid.startswith("tests/unit/")
        and not any(path in nodeid for path in STRICT_CANDIDATE_EXCLUDED_PATHS)
        and not any(test in nodeid for test in STRICT_CANDIDATE_EXCLUDED_TESTS)
        and not any(
            term in nodeid.lower() for term in STRICT_CANDIDATE_EXCLUDED_TERMS
        )
    )


def build_report(
    local: CoverageData, upstream: CoverageData
) -> dict[str, Any]:
    """Return a line-level classification, retaining pytest test contexts."""
    files = sorted(
        {
            *local.measured_files(),
            *upstream.measured_files(),
        }
    )
    result: dict[str, Any] = {
        "files": {},
        "summary": {},
        "strict_candidates": [],
    }
    totals = {"local_only": 0, "upstream_only": 0, "both": 0}
    candidates: dict[str, dict[str, Any]] = {}
    for filename in files:
        if "/cudf_polars/" not in filename.replace("\\", "/"):
            continue
        local_lines = set(local.lines(filename) or ())
        upstream_lines = set(upstream.lines(filename) or ())
        local_contexts = local.contexts_by_lineno(filename)
        upstream_contexts = upstream.contexts_by_lineno(filename)
        classifications: dict[str, list[Any]] = {
            "local_only": [],
            "upstream_only": [],
            "both": [],
        }
        for line in sorted(local_lines | upstream_lines):
            local_line_contexts = sorted(local_contexts.get(line, set()))
            upstream_line_contexts = sorted(upstream_contexts.get(line, set()))
            entry = {
                "line": line,
                "local_contexts": local_line_contexts[:MAX_CONTEXTS_PER_LINE],
                "local_context_count": len(local_line_contexts),
                "local_contexts_truncated": len(local_line_contexts)
                > MAX_CONTEXTS_PER_LINE,
                "upstream_contexts": upstream_line_contexts[
                    :MAX_CONTEXTS_PER_LINE
                ],
                "upstream_context_count": len(upstream_line_contexts),
                "upstream_contexts_truncated": len(upstream_line_contexts)
                > MAX_CONTEXTS_PER_LINE,
            }
            if line in local_lines and line in upstream_lines:
                classifications["both"].append(entry)
                if _is_strict_candidate_source(filename):
                    candidate_contexts = (
                        context
                        for context in upstream_line_contexts
                        if _is_strict_candidate_context(context)
                    )
                    for context in islice(
                        candidate_contexts, MAX_CANDIDATE_CONTEXTS_PER_LINE
                    ):
                        nodeid = _nodeid(context)
                        candidate = candidates.setdefault(
                            nodeid,
                            {
                                "nodeid": nodeid,
                                "shared_lines": set(),
                                "source_modules": set(),
                                "examples": [],
                            },
                        )
                        candidate["shared_lines"].add((filename, line))
                        candidate["source_modules"].add(filename)
                        if len(candidate["examples"]) < MAX_CANDIDATE_EXAMPLES:
                            candidate["examples"].append(
                                {
                                    "source_file": filename,
                                    "line": line,
                                    "local_contexts": local_line_contexts[
                                        :MAX_CONTEXTS_PER_LINE
                                    ],
                                    "upstream_context": context,
                                }
                            )
            elif line in local_lines:
                classifications["local_only"].append(entry)
            else:
                classifications["upstream_only"].append(entry)
        if any(classifications.values()):
            result["files"][filename] = classifications
            for category, lines in classifications.items():
                totals[category] += len(lines)
    result["summary"] = {**totals, "files": len(result["files"])}
    for candidate in candidates.values():
        source_modules = sorted(candidate["source_modules"])
        result["strict_candidates"].append(
            {
                "nodeid": candidate["nodeid"],
                "sampled_shared_line_count": len(candidate["shared_lines"]),
                "local_only_line_count": sum(
                    len(result["files"][filename]["local_only"])
                    for filename in source_modules
                ),
                "source_modules": source_modules,
                "representative_contexts": candidate["examples"],
            }
        )
    result["strict_candidates"].sort(
        key=lambda candidate: (
            -candidate["sampled_shared_line_count"],
            candidate["nodeid"],
        )
    )
    return result


def _write_html(report: dict[str, Any], output: Path) -> None:
    rows = []
    for filename, categories in report["files"].items():
        rows.append(
            "<tr>"
            f"<td>{html.escape(filename)}</td>"
            f"<td>{len(categories['local_only'])}</td>"
            f"<td>{len(categories['upstream_only'])}</td>"
            f"<td>{len(categories['both'])}</td>"
            "</tr>"
        )
    summary = report["summary"]
    output.write_text(
        "<!doctype html><meta charset=utf-8>"
        "<title>cudf-polars coverage comparison</title>"
        "<h1>cudf-polars coverage comparison</h1>"
        f"<p>Files: {summary['files']}; local-only: {summary['local_only']}; "
        f"upstream-only: {summary['upstream_only']}; both: {summary['both']}.</p>"
        "<p>See <code>coverage-comparison.json</code> for line-level pytest "
        "contexts.</p><table><thead><tr><th>Source file</th><th>Local only</th>"
        "<th>Upstream only</th><th>Both</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _write_candidate_html(report: dict[str, Any], output: Path) -> None:
    """Write a compact review table for prospective strict GPU nodes."""
    rows = []
    for candidate in report["strict_candidates"]:
        modules = "<br>".join(
            html.escape(module.rsplit("/cudf_polars/", 1)[-1])
            for module in candidate["source_modules"]
        )
        examples = "<br>".join(
            html.escape(
                f"{example['source_file']}:{example['line']} ← "
                f"{example['upstream_context']}"
            )
            for example in candidate["representative_contexts"]
        )
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(candidate['nodeid'])}</code></td>"
            f"<td>{candidate['sampled_shared_line_count']}</td>"
            f"<td>{candidate['local_only_line_count']}</td>"
            f"<td>{modules}</td><td><code>{examples}</code></td>"
            "</tr>"
        )
    output.write_text(
        "<!doctype html><meta charset=utf-8>"
        "<title>cudf-polars strict GPU candidates</title>"
        "<h1>cudf-polars strict GPU candidate review</h1>"
        "<p>These nodes are coverage-discovery candidates only. Admit a node "
        "only after strict in-memory and SPMD runs pass.</p>"
        "<table><thead><tr><th>Upstream node</th><th>Sampled shared lines</th>"
        "<th>Local-only lines in modules</th><th>Source modules</th>"
        "<th>Representative upstream contexts</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-data", type=Path, required=True)
    parser.add_argument("--upstream-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        _read_data(args.local_data), _read_data(args.upstream_data)
    )
    (args.output_dir / "coverage-comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _write_html(report, args.output_dir / "index.html")
    _write_candidate_html(
        report, args.output_dir / "strict-gpu-candidates.html"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Classify cudf-polars source lines covered by local and upstream tests."""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from coverage import CoverageData

MIN_SHARED_LINES = 4
MAX_SHARED_LINE_EXAMPLES = 10
MAX_LOCAL_ONLY_LINE_EXAMPLES = 10
MAX_UPSTREAM_CONTEXTS_PER_LINE = 100
EXCLUDED_LOCAL_PATHS = (
    "tests/containers/",
    "tests/dsl/",
    "tests/io/",
    "tests/streaming/",
    "tests/testing/",
    "tests/utils/",
)
EXCLUDED_LOCAL_TERMS = ("dtype", "decimal", "fallback", "unsupported", "scan")
GENERIC_NAME_TOKENS = frozenset({"test", "expr", "lazy", "frame", "polars"})

# The queue is deliberately limited to ordinary, non-streaming public behavior;
# overlap elsewhere is still visible in the line-level report.
REVIEW_EXCLUDED_PATHS = (
    "/datatypes/",
    "/io/",
    "/streaming/",
    "/testing/",
)
REVIEW_EXCLUDED_TESTS = (
    "test_describe.py",
    "test_null_count.py",
    "test_cse.py",
    "test_optimizations.py",
    "test_statistics.py",
)
REVIEW_EXCLUDED_TERMS = (
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


def _is_review_context(context: str) -> bool:
    """Whether a context belongs in the focused upstream-unit review pool."""
    nodeid = _nodeid(context)
    return (
        nodeid.startswith("tests/unit/")
        and not any(path in nodeid for path in REVIEW_EXCLUDED_PATHS)
        and not any(test in nodeid for test in REVIEW_EXCLUDED_TESTS)
        and not any(term in nodeid.lower() for term in REVIEW_EXCLUDED_TERMS)
    )


def _behavior_tokens(nodeid: str) -> set[str]:
    path, _, test_name = nodeid.partition("::")
    behavior = f"{path.rsplit('/', 1)[-1]}_{test_name}"
    return {
        token
        for token in re.split(r"[^a-z0-9]+", behavior.lower())
        if len(token) > 2 and token not in GENERIC_NAME_TOKENS
    }


def _review_pairs(
    local_tests: dict[str, set[tuple[str, int]]],
    upstream_tests: dict[str, set[tuple[str, int]]],
    upstream_lines: dict[str, set[int]],
) -> list[dict[str, Any]]:
    upstream_by_line: dict[tuple[str, int], set[str]] = defaultdict(set)
    for nodeid, lines in upstream_tests.items():
        for line in lines:
            upstream_by_line[line].add(nodeid)

    pairs = []
    for local_nodeid, lines in local_tests.items():
        if local_nodeid.startswith(EXCLUDED_LOCAL_PATHS) or any(
            term in local_nodeid.lower() for term in EXCLUDED_LOCAL_TERMS
        ):
            continue
        matches = Counter(
            upstream_nodeid
            for line in lines
            if len(upstream_by_line.get(line, ()))
            <= MAX_UPSTREAM_CONTEXTS_PER_LINE
            for upstream_nodeid in upstream_by_line.get(line, ())
        )
        name_tokens = _behavior_tokens(local_nodeid)
        matches = Counter(
            {
                nodeid: count
                for nodeid, count in matches.items()
                if count >= MIN_SHARED_LINES
                and name_tokens & _behavior_tokens(nodeid)
            }
        )
        if not matches:
            continue
        upstream_nodeid = min(
            matches,
            key=lambda nodeid: (-matches[nodeid], nodeid),
        )
        shared = sorted(lines & upstream_tests[upstream_nodeid])
        local_only = sorted(
            line
            for line in lines
            if line[1] not in upstream_lines.get(line[0], set())
        )
        pairs.append(
            {
                "local_nodeid": local_nodeid,
                "upstream_nodeid": upstream_nodeid,
                "shared_line_count": len(shared),
                "local_line_count": len(lines),
                "shared_fraction": round(len(shared) / len(lines), 3),
                "local_only_line_count": len(local_only),
                "shared_lines": [
                    {"source_file": filename, "line": line}
                    for filename, line in shared[:MAX_SHARED_LINE_EXAMPLES]
                ],
                "local_only_lines": [
                    {"source_file": filename, "line": line}
                    for filename, line in local_only[
                        :MAX_LOCAL_ONLY_LINE_EXAMPLES
                    ]
                ],
                "other_upstream_matches": [
                    {"nodeid": nodeid, "shared_line_count": count}
                    for nodeid, count in sorted(
                        matches.items(),
                        key=lambda match: (-match[1], match[0]),
                    )
                    if nodeid != upstream_nodeid
                ][:5],
                "status": "review_candidate",
            }
        )
    return sorted(
        pairs,
        key=lambda pair: (
            -pair["shared_line_count"] / pair["local_line_count"],
            -pair["shared_line_count"],
            pair["local_nodeid"],
        ),
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
        "review_pairs": [],
    }
    totals = {"local_only": 0, "upstream_only": 0, "both": 0}
    local_tests: dict[str, set[tuple[str, int]]] = defaultdict(set)
    upstream_tests: dict[str, set[tuple[str, int]]] = defaultdict(set)
    all_upstream_lines: dict[str, set[int]] = {}
    context_maps: dict[
        str, tuple[dict[int, set[str]], dict[int, set[str]]]
    ] = {}
    for filename in files:
        if "/cudf_polars/" not in filename.replace("\\", "/"):
            continue
        local_lines = set(local.lines(filename) or ())
        upstream_lines = set(upstream.lines(filename) or ())
        all_upstream_lines[filename] = upstream_lines
        local_contexts = local.contexts_by_lineno(filename)
        upstream_contexts = upstream.contexts_by_lineno(filename)
        context_maps[filename] = (local_contexts, upstream_contexts)
        classifications: dict[str, list[Any]] = {
            "local_only": [],
            "upstream_only": [],
            "both": [],
        }
        for line in sorted(local_lines | upstream_lines):
            local_line_contexts = sorted(local_contexts.get(line, set()))
            upstream_line_contexts = sorted(upstream_contexts.get(line, set()))
            for context in local_line_contexts:
                if _nodeid(context).startswith("tests/"):
                    local_tests[_nodeid(context)].add((filename, line))
            for context in upstream_line_contexts:
                if _is_review_context(context):
                    upstream_tests[_nodeid(context)].add((filename, line))
            entry = {
                "line": line,
                "local_contexts": local_line_contexts,
                "upstream_contexts": upstream_line_contexts,
            }
            if line in local_lines and line in upstream_lines:
                classifications["both"].append(entry)
            elif line in local_lines:
                classifications["local_only"].append(entry)
            else:
                classifications["upstream_only"].append(entry)
        if any(classifications.values()):
            result["files"][filename] = classifications
            for category, lines in classifications.items():
                totals[category] += len(lines)
    result["summary"] = {**totals, "files": len(result["files"])}
    result["review_pairs"] = _review_pairs(
        local_tests, upstream_tests, all_upstream_lines
    )
    for pair in result["review_pairs"]:
        for example in pair["shared_lines"]:
            local_contexts, upstream_contexts = context_maps[
                example["source_file"]
            ]
            line = example["line"]
            example["local_context"] = next(
                context
                for context in sorted(local_contexts[line])
                if _nodeid(context) == pair["local_nodeid"]
            )
            example["upstream_context"] = next(
                context
                for context in sorted(upstream_contexts[line])
                if _nodeid(context) == pair["upstream_nodeid"]
            )
        for example in pair["local_only_lines"]:
            local_contexts, _ = context_maps[example["source_file"]]
            example["local_context"] = next(
                context
                for context in sorted(local_contexts[example["line"]])
                if _nodeid(context) == pair["local_nodeid"]
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
    """Write a compact table of local/upstream pairs for behavioral review."""
    rows = []
    for pair in report["review_pairs"]:
        examples = "<br>".join(
            html.escape(f"{line['source_file']}:{line['line']}")
            for line in pair["shared_lines"]
        )
        local_only_examples = "<br>".join(
            html.escape(f"{line['source_file']}:{line['line']}")
            for line in pair["local_only_lines"]
        )
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(pair['local_nodeid'])}</code></td>"
            f"<td><code>{html.escape(pair['upstream_nodeid'])}</code></td>"
            f"<td>{'yes' if pair['upstream_gpu_observed'] else 'no'}</td>"
            f"<td>{pair['shared_line_count']}</td>"
            f"<td>{pair['shared_fraction']:.1%}</td>"
            f"<td>{pair['local_only_line_count']}</td>"
            f"<td><code>{examples}</code></td>"
            f"<td><code>{local_only_examples}</code></td>"
            "</tr>"
        )
    output.write_text(
        "<!doctype html><meta charset=utf-8>"
        "<title>cudf-polars upstream review candidates</title>"
        "<h1>cudf-polars upstream review candidates</h1>"
        "<p>Coverage overlap is a review signal, not proof that upstream "
        "tests assert the same behavior. GPU observation comes from the "
        "maintained SPMD-default run list.</p>"
        "<table><thead><tr><th>Local node</th><th>Upstream node</th>"
        "<th>GPU observed</th>"
        "<th>Shared lines</th><th>Shared fraction</th>"
        "<th>Local-only lines</th>"
        "<th>Shared source lines</th><th>Local-only source lines</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-data", type=Path, required=True)
    parser.add_argument("--upstream-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cudf-sha", required=True)
    parser.add_argument("--polars-sha", required=True)
    parser.add_argument("--polars-version", required=True)
    parser.add_argument("--gpu-run-list", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(
        _read_data(args.local_data), _read_data(args.upstream_data)
    )
    report["provenance"] = {
        "cudf_sha": args.cudf_sha,
        "polars_sha": args.polars_sha,
        "polars_version": args.polars_version,
    }
    selections = json.loads(args.gpu_run_list.read_text())["selections"]
    selection_key = f"{args.polars_version}|spmd|default"
    if selection_key not in selections:
        parser.error(f"No GPU-execution selection for {selection_key}")
    gpu_nodes = set(selections[selection_key]["nodes"])
    for pair in report["review_pairs"]:
        pair["upstream_gpu_observed"] = pair["upstream_nodeid"] in gpu_nodes
    (args.output_dir / "coverage-comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _write_html(report, args.output_dir / "index.html")
    _write_candidate_html(
        report, args.output_dir / "upstream-review-candidates.html"
    )


if __name__ == "__main__":
    main()

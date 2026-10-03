#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize cudf-polars fallback telemetry from upstream Polars JUnit XML."""

from __future__ import annotations

import argparse
import html
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any


def _outcome(testcase: ET.Element) -> str:
    if testcase.find("failure") is not None:
        return "failed"
    if testcase.find("error") is not None:
        return "error"
    if testcase.find("skipped") is not None:
        return "skipped"
    return "passed"


def _report_argument(value: str) -> tuple[str, Path]:
    """Parse an ENGINE=JUNIT_XML command-line argument."""
    engine, separator, path = value.partition("=")
    if not separator or not engine or not path:
        raise argparse.ArgumentTypeError(
            "reports must have the form ENGINE=JUNIT_XML"
        )
    return engine, Path(path)


def read_report(path: Path, engine: str) -> list[dict[str, str]]:
    """Read one engine's JUnit report into node-level fallback diagnostics."""
    root = ET.parse(path).getroot()
    tests: list[dict[str, str]] = []
    for testcase in root.iter("testcase"):
        properties = {
            property_.get("name"): property_.get("value", "")
            for property_ in testcase.findall("properties/property")
        }
        nodeid = properties.get("cudf_polars_nodeid")
        fallback = properties.get("cudf_polars_fallback")
        if nodeid is None or fallback is None:
            continue
        tests.append(
            {
                "nodeid": nodeid,
                "engine": engine,
                "outcome": _outcome(testcase),
                "fallback": fallback,
            }
        )
    return sorted(tests, key=lambda test: test["nodeid"])


def build_report(reports: dict[str, Path]) -> dict[str, Any]:
    """Build a compact per-engine summary and a per-node diagnostic table."""
    tests = [
        test
        for engine, path in reports.items()
        for test in read_report(path, engine)
    ]
    summaries = {}
    for engine in reports:
        engine_tests = [test for test in tests if test["engine"] == engine]
        summaries[engine] = {
            "total": len(engine_tests),
            "fallback": sum(
                test["fallback"] == "true" for test in engine_tests
            ),
            "no_fallback_observed": sum(
                test["fallback"] == "false" for test in engine_tests
            ),
            "outcomes": dict(
                sorted(
                    Counter(test["outcome"] for test in engine_tests).items()
                )
            ),
        }
    return {"summary": summaries, "tests": tests}


def write_html(report: dict[str, Any], output: Path) -> None:
    """Write a readable version of the JSON diagnostic data."""
    summaries = "".join(
        "<li>"
        f"{html.escape(engine)}: {summary['fallback']} fallback; "
        f"{summary['no_fallback_observed']} no fallback observed; "
        f"{summary['total']} total.</li>"
        for engine, summary in report["summary"].items()
    )
    rows = "".join(
        "<tr>"
        f"<td><code>{html.escape(test['nodeid'])}</code></td>"
        f"<td>{html.escape(test['engine'])}</td>"
        f"<td>{html.escape(test['outcome'])}</td>"
        f"<td>{html.escape(test['fallback'])}</td>"
        "</tr>"
        for test in report["tests"]
    )
    output.write_text(
        "<!doctype html><meta charset=utf-8>"
        "<title>cudf-polars fallback diagnostics</title>"
        "<h1>cudf-polars fallback diagnostics</h1>"
        "<p><code>false</code> means no CPU fallback was observed; it does not "
        "by itself prove that a test invoked LazyFrame.collect.</p>"
        f"<ul>{summaries}</ul>"
        "<table><thead><tr><th>Upstream node</th><th>Engine</th>"
        "<th>Outcome</th><th>CPU fallback observed</th>"
        "</tr></thead><tbody>"
        f"{rows}</tbody></table>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        type=_report_argument,
        metavar="ENGINE=JUNIT_XML",
        required=True,
        help="JUnit XML report from one injected-engine run",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    reports = dict(args.report)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(reports)
    (args.output_dir / "fallback-diagnostics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    write_html(report, args.output_dir / "index.html")


if __name__ == "__main__":
    main()

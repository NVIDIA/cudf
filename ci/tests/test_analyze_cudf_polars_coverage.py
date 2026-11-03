# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cudf-polars coverage comparison analysis."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from coverage import CoverageData

if TYPE_CHECKING:
    from types import ModuleType


def _load_analyzer() -> ModuleType:
    path = Path(__file__).parents[1] / "analyze_cudf_polars_coverage.py"
    spec = importlib.util.spec_from_file_location("coverage_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _CoverageData:
    def __init__(self, data: dict[str, dict[int, set[str]]]) -> None:
        self.data = data

    def measured_files(self) -> set[str]:
        return set(self.data)

    def lines(self, filename: str) -> list[int]:
        return list(self.data.get(filename, {}))

    def contexts_by_lineno(self, filename: str) -> dict[int, set[str]]:
        return self.data.get(filename, {})


def test_build_report_emits_focused_upstream_unit_candidates() -> None:
    analyzer = _load_analyzer()
    source = "/work/cudf_polars/dsl/translate.py"
    ignored_source = "/work/cudf_polars/streaming/io.py"
    local = _CoverageData(
        {
            source: {
                10: {"tests/dsl/test_to_ast.py::test_translate|run"},
                11: set(),
            },
            ignored_source: {20: {"tests/test_io.py::test_scan|run"}},
        }
    )
    upstream = _CoverageData(
        {
            source: {
                10: {"tests/unit/lazyframe/test_lazy.py::test_translate|run"},
                12: {
                    "tests/unit/lazyframe/test_lazy.py::test_upstream_only|run"
                },
            },
            ignored_source: {
                20: {"tests/unit/test_streaming.py::test_io|run"}
            },
        }
    )

    report = analyzer.build_report(local, upstream)

    assert report["summary"] == {
        "both": 2,
        "files": 2,
        "local_only": 1,
        "upstream_only": 1,
    }
    assert report["review_pairs"] == []
    assert report["files"][source]["both"][0]["local_contexts"] == [
        "tests/dsl/test_to_ast.py::test_translate|run"
    ]
    assert report["files"][source]["both"][0]["upstream_contexts"] == [
        "tests/unit/lazyframe/test_lazy.py::test_translate|run"
    ]


def test_review_pairs_need_shared_behavior_and_remain_candidates() -> None:
    analyzer = _load_analyzer()
    source = "/work/cudf_polars/dsl/translate.py"
    local = _CoverageData(
        {
            source: {
                line: {"tests/test_join.py::test_join|run"}
                for line in range(10, 15)
            }
        }
    )
    upstream = _CoverageData(
        {
            source: {
                line: {
                    "tests/unit/operations/test_join.py::test_join|run",
                    "tests/unit/operations/test_filter.py::test_filter|run",
                }
                for line in range(10, 14)
            }
        }
    )

    pairs = analyzer.build_report(local, upstream)["review_pairs"]

    assert len(pairs) == 1
    assert pairs[0]["local_nodeid"] == "tests/test_join.py::test_join"
    assert pairs[0]["upstream_nodeid"] == (
        "tests/unit/operations/test_join.py::test_join"
    )
    assert pairs[0]["shared_line_count"] == 4
    assert pairs[0]["local_only_line_count"] == 1
    assert pairs[0]["shared_lines"][0] == {
        "source_file": source,
        "line": 10,
        "local_context": "tests/test_join.py::test_join|run",
        "upstream_context": "tests/unit/operations/test_join.py::test_join|run",
    }
    assert pairs[0]["status"] == "review_candidate"


def test_review_pairs_rank_by_overlap_fraction() -> None:
    analyzer = _load_analyzer()
    source = "/work/cudf_polars/dsl/translate.py"
    upstream_lines = {line for line in range(10, 18)}
    local_tests = {
        "tests/test_join.py::test_join": {
            (source, line) for line in range(10, 15)
        },
        "tests/test_filter.py::test_filter": {
            (source, line) for line in range(14, 20)
        },
    }
    upstream_tests = {
        "tests/unit/test_join.py::test_join": {
            (source, line) for line in range(10, 14)
        },
        "tests/unit/test_filter.py::test_filter": {
            (source, line) for line in range(14, 18)
        },
    }

    pairs = analyzer._review_pairs(
        local_tests, upstream_tests, {source: upstream_lines}
    )

    assert [pair["local_nodeid"] for pair in pairs] == [
        "tests/test_join.py::test_join",
        "tests/test_filter.py::test_filter",
    ]
    assert [pair["local_only_line_count"] for pair in pairs] == [0, 2]


def test_candidate_context_filter_excludes_non_ordinary_behavior() -> None:
    analyzer = _load_analyzer()

    assert analyzer._is_review_context(
        "tests/unit/operations/test_filter.py::test_filter|run"
    )
    assert not analyzer._is_review_context(
        "tests/unit/datatypes/test_decimal.py::test_decimal_aggregations|run"
    )
    assert not analyzer._is_review_context(
        "tests/unit/lazyframe/test_cse.py::test_cse_10452|run"
    )
    assert not analyzer._is_review_context(
        "tests/unit/operations/test_group_by.py::test_group_by_skew_kurtosis|run"
    )


def test_main_writes_provenance_and_contexts(tmp_path, monkeypatch) -> None:
    analyzer = _load_analyzer()
    source = str(tmp_path / "cudf_polars" / "dsl" / "translate.py")
    local_path = tmp_path / ".coverage.local"
    upstream_path = tmp_path / ".coverage.upstream"
    local = CoverageData(basename=str(local_path))
    local.set_context("tests/test_join.py::test_join|run")
    local.add_lines({source: {10, 11}})
    local.write()
    upstream = CoverageData(basename=str(upstream_path))
    upstream.set_context("tests/unit/test_join.py::test_join|run")
    upstream.add_lines({source: {10, 12}})
    upstream.write()
    output = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_cudf_polars_coverage.py",
            "--local-data",
            str(local_path),
            "--upstream-data",
            str(upstream_path),
            "--output-dir",
            str(output),
            "--cudf-sha",
            "cudf-sha",
            "--polars-sha",
            "polars-sha",
            "--polars-version",
            "2.1.0",
        ],
    )

    analyzer.main()

    report = json.loads((output / "coverage-comparison.json").read_text())
    assert report["provenance"] == {
        "cudf_sha": "cudf-sha",
        "polars_sha": "polars-sha",
        "polars_version": "2.1.0",
    }
    assert report["files"][source]["both"][0]["local_contexts"] == [
        "tests/test_join.py::test_join|run"
    ]
    assert (output / "index.html").is_file()
    assert (output / "upstream-review-candidates.html").is_file()

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the cudf-polars coverage comparison analysis."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

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
    assert report["strict_candidates"] == [
        {
            "nodeid": "tests/unit/lazyframe/test_lazy.py::test_translate",
            "sampled_shared_line_count": 1,
            "local_only_line_count": 1,
            "source_modules": [source],
            "representative_contexts": [
                {
                    "source_file": source,
                    "line": 10,
                    "local_contexts": [
                        "tests/dsl/test_to_ast.py::test_translate|run"
                    ],
                    "upstream_context": "tests/unit/lazyframe/test_lazy.py::test_translate|run",
                }
            ],
        }
    ]


def test_candidate_context_filter_excludes_non_ordinary_behavior() -> None:
    analyzer = _load_analyzer()

    assert analyzer._is_strict_candidate_context(
        "tests/unit/operations/test_filter.py::test_filter|run"
    )
    assert not analyzer._is_strict_candidate_context(
        "tests/unit/datatypes/test_decimal.py::test_decimal_aggregations|run"
    )
    assert not analyzer._is_strict_candidate_context(
        "tests/unit/lazyframe/test_cse.py::test_cse_10452|run"
    )
    assert not analyzer._is_strict_candidate_context(
        "tests/unit/operations/test_group_by.py::test_group_by_skew_kurtosis|run"
    )

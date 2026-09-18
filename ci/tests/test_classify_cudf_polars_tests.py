# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for local cudf-polars test classification."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from coverage import CoverageData

if TYPE_CHECKING:
    from types import ModuleType


def _load_classifier() -> ModuleType:
    path = Path(__file__).parents[1] / "classify_cudf_polars_tests.py"
    spec = importlib.util.spec_from_file_location("test_classifier", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _coverage(
    module: ModuleType, nodeid: str, *, unique: int = 0, strict: int = 1
):
    return module.TestCoverage(
        nodeid=nodeid,
        total_lines=1,
        local_only_lines=unique,
        strict_lines=strict,
    )


def test_classification_precedence() -> None:
    classifier = _load_classifier()
    assert (
        classifier._classification(
            _coverage(classifier, "tests/test_scan.py::test_scan"), set()
        )
        == "defer_scan_io"
    )
    assert (
        classifier._classification(
            _coverage(
                classifier, "tests/streaming/test_select.py::test_select"
            ),
            set(),
        )
        == "retain_streaming_or_engine"
    )
    assert (
        classifier._classification(
            _coverage(classifier, "tests/dsl/test_to_ast.py::test_ast"), set()
        )
        == "rewrite_introspective"
    )
    assert (
        classifier._classification(
            _coverage(classifier, "tests/test_filter.py::test_unsupported"),
            {("tests/test_filter.py", "test_unsupported")},
        )
        == "retain_unsupported_or_fallback"
    )
    assert (
        classifier._classification(
            _coverage(
                classifier, "tests/test_filter.py::test_query", unique=1
            ),
            set(),
        )
        == "retain_unique_local_coverage"
    )
    assert (
        classifier._classification(
            _coverage(classifier, "tests/test_filter.py::test_query"), set()
        )
        == "deletion_review_candidate"
    )
    assert (
        classifier._classification(
            _coverage(
                classifier, "tests/test_filter.py::test_query", strict=0
            ),
            set(),
        )
        == "retain_not_strictly_covered"
    )
    assert (
        classifier._classification(
            _coverage(classifier, "tests/test_distinct.py::test_distinct"),
            set(),
        )
        == "deletion_review_candidate"
    )


def test_missing_strict_coverage_is_conservative(tmp_path: Path) -> None:
    classifier = _load_classifier()
    local_data = CoverageData(basename=str(tmp_path / "local.coverage"))
    local_data.set_context("tests/test_query.py::test_query|run")
    local_data.add_lines({"/tmp/cudf_polars/dsl/translate.py": {1}})
    local_data.write()
    upstream_data = CoverageData(basename=str(tmp_path / "upstream.coverage"))
    upstream_data.set_context("tests/unit/test_query.py::test_query|run")
    upstream_data.add_lines({"/tmp/cudf_polars/dsl/translate.py": {1}})
    upstream_data.write()
    report = classifier.classify(
        tmp_path / "local.coverage",
        tmp_path / "upstream.coverage",
        None,
        tmp_path,
    )
    assert report["tests"] == [
        {
            "classification": "retain_not_strictly_covered",
            "covered_line_count": 1,
            "local_only_line_count": 0,
            "next_action": "seek_strict_upstream_coverage_before_pruning",
            "nodeid": "tests/test_query.py::test_query",
            "source_files": ["/tmp/cudf_polars/dsl/translate.py"],
            "strict_line_count": 0,
        }
    ]

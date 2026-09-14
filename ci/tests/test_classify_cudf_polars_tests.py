# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for local cudf-polars test classification."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

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

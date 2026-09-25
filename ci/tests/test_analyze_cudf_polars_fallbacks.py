# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the upstream Polars fallback diagnostic report."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from types import ModuleType


def _load_analyzer() -> ModuleType:
    path = Path(__file__).parents[1] / "analyze_cudf_polars_fallbacks.py"
    spec = importlib.util.spec_from_file_location("fallback_analysis", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_junit(path: Path, testcases: str) -> None:
    path.write_text(f"<testsuite>{testcases}</testsuite>")


def test_build_report_classifies_fallback_per_engine(tmp_path: Path) -> None:
    analyzer = _load_analyzer()
    in_memory = tmp_path / "in-memory.xml"
    spmd = tmp_path / "spmd.xml"
    _write_junit(
        in_memory,
        """
        <testcase><properties>
          <property name="cudf_polars_nodeid" value="tests/unit/test_a.py::test_a" />
          <property name="cudf_polars_fallback" value="true" />
        </properties></testcase>
        <testcase><properties>
          <property name="cudf_polars_nodeid" value="tests/unit/test_b.py::test_b" />
          <property name="cudf_polars_fallback" value="false" />
        </properties><failure /></testcase>
        """,
    )
    _write_junit(
        spmd,
        """
        <testcase><properties>
          <property name="cudf_polars_nodeid" value="tests/unit/test_a.py::test_a" />
          <property name="cudf_polars_fallback" value="false" />
        </properties></testcase>
        <testcase name="missing-telemetry" />
        """,
    )

    report = analyzer.build_report({"in-memory": in_memory, "spmd": spmd})

    assert report["summary"] == {
        "in-memory": {
            "total": 2,
            "fallback": 1,
            "no_fallback_observed": 1,
            "outcomes": {"failed": 1, "passed": 1},
        },
        "spmd": {
            "total": 1,
            "fallback": 0,
            "no_fallback_observed": 1,
            "outcomes": {"passed": 1},
        },
    }
    assert report["tests"] == [
        {
            "nodeid": "tests/unit/test_a.py::test_a",
            "engine": "in-memory",
            "outcome": "passed",
            "fallback": "true",
        },
        {
            "nodeid": "tests/unit/test_b.py::test_b",
            "engine": "in-memory",
            "outcome": "failed",
            "fallback": "false",
        },
        {
            "nodeid": "tests/unit/test_a.py::test_a",
            "engine": "spmd",
            "outcome": "passed",
            "fallback": "false",
        },
    ]

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check coverage exports against real coverage.py context databases."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from coverage import CoverageData

_spec = importlib.util.spec_from_file_location(
    "coverage_analysis",
    Path(__file__).parents[1] / "analyze_cudf_polars_coverage.py",
)
analyzer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyzer)


def _data(path, contexts):
    data = CoverageData(basename=str(path))
    for context, files in contexts.items():
        data.set_context(context)
        data.add_lines(files)
    data.write()
    return data


def test_line_categories_retain_test_phases(tmp_path):
    source = "/work/cudf_polars/dsl/translate.py"
    engine = "/work/cudf_polars/engine/core.py"
    local = _data(
        tmp_path / ".coverage.local",
        {
            "tests/test_filter.py::test_filter|run": {source: {10, 11}},
            "tests/test_filter.py::test_filter|setup": {engine: {20}},
        },
    )
    upstream = _data(
        tmp_path / ".coverage.upstream",
        {
            "tests/unit/test_join.py::test_join|run": {source: {10, 12}},
            "tests/unit/test_join.py::test_join|teardown": {engine: {21}},
        },
    )

    report = analyzer.build_report(local, [upstream])

    assert report["summary"] == {
        "both": 1,
        "local_only": 2,
        "upstream_only": 2,
        "files": 2,
    }
    assert [
        line["line"] for line in report["files"][source]["local_only"]
    ] == [11]
    assert [
        line["line"] for line in report["files"][source]["upstream_only"]
    ] == [12]
    shared = report["files"][source]["both"][0]
    assert shared["local_contexts"] == [
        "tests/test_filter.py::test_filter|run"
    ]
    assert shared["upstream_contexts"] == [
        "tests/unit/test_join.py::test_join|run"
    ]
    assert report["files"][engine]["local_only"][0]["local_contexts"] == [
        "tests/test_filter.py::test_filter|setup"
    ]
    assert report["files"][engine]["upstream_only"][0][
        "upstream_contexts"
    ] == ["tests/unit/test_join.py::test_join|teardown"]
    assert set(report) == {"files", "summary"}


def test_truncated_samples_keep_complete_raw_contexts(tmp_path):
    source = "/work/cudf_polars/dsl/translate.py"
    contexts = {
        f"tests/test_expr.py::test_case_{i}|run": {source: {10}}
        for i in range(25)
    }
    path = tmp_path / ".coverage.local"
    local = _data(path, contexts)
    report = analyzer.build_report(local, [])

    line = report["files"][source]["local_only"][0]
    assert line["local_contexts"] == sorted(contexts)[:20]
    assert line["local_context_count"] == 25
    assert line["local_contexts_truncated"]
    assert set(
        analyzer._read_data(path).contexts_by_lineno(source)[10]
    ) == set(contexts)


@pytest.mark.parametrize("kind", ["missing", "empty", "no_contexts"])
def test_unusable_context_data_is_rejected(tmp_path, kind):
    path = tmp_path / ".coverage.local"
    if kind == "empty":
        path.touch()
    elif kind == "no_contexts":
        _data(path, {"": {"/work/cudf_polars/callback.py": {10}}})
    with pytest.raises(
        ValueError, match=r"coverage data|No test-body coverage contexts"
    ):
        analyzer._read_data(path)


def test_export_combines_engines_and_records_partial_runs(
    tmp_path, monkeypatch
):
    source = "/work/cudf_polars/dsl/translate.py"
    local = tmp_path / ".coverage.local"
    memory = tmp_path / ".coverage.upstream-in-memory"
    spmd = tmp_path / ".coverage.upstream-spmd"
    _data(
        local, {"tests/test_expr.py::test_local|run": {source: {10, 11, 12}}}
    )
    _data(memory, {"tests/unit/test_expr.py::test_memory|run": {source: {10}}})
    _data(spmd, {"tests/unit/test_expr.py::test_spmd|run": {source: {10, 11}}})
    output = tmp_path / "reports"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_cudf_polars_coverage.py",
            "--local-data",
            str(local),
            "--upstream-data",
            str(memory),
            "--upstream-data",
            str(spmd),
            "--output-dir",
            str(output),
            "--cudf-sha",
            "cudf-sha",
            "--polars-sha",
            "polars-sha",
            "--polars-version",
            "2.1.0",
            "--collection-metadata",
            json.dumps(
                {"upstream": {"marker": "not slow", "gpu_run_list": None}}
            ),
            "--local-exit-status",
            "0",
            "--upstream-in-memory-exit-status",
            "1",
            "--upstream-spmd-exit-status",
            "0",
        ],
    )

    analyzer.main()

    report = json.loads((output / "coverage-comparison.json").read_text())
    assert report["provenance"] == {
        "cudf_sha": "cudf-sha",
        "polars_sha": "polars-sha",
        "polars_version": "2.1.0",
    }
    assert report["collection"] == {
        "upstream": {"marker": "not slow", "gpu_run_list": None}
    }
    assert report["suite_exit_statuses"] == {
        "local": 0,
        "upstream_in_memory": 1,
        "upstream_spmd": 0,
    }
    assert report["context_data"] == {
        "local": "../.coverage.local",
        "upstream": [
            "../.coverage.upstream-in-memory",
            "../.coverage.upstream-spmd",
        ],
    }
    assert [line["line"] for line in report["files"][source]["both"]] == [
        10,
        11,
    ]
    assert report["files"][source]["both"][0]["upstream_contexts"] == [
        "tests/unit/test_expr.py::test_memory|run",
        "tests/unit/test_expr.py::test_spmd|run",
    ]
    assert [
        line["line"] for line in report["files"][source]["local_only"]
    ] == [12]

# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the benchmark results interchange format."""

from __future__ import annotations

import json
import uuid
from typing import TYPE_CHECKING, Any

import pytest

from cudf_polars.streaming.benchmarks.utils import (
    CPUInfo,
    FailedRecord,
    GPUInfo,
    HardwareInfo,
    PackageVersions,
    QueryMode,
    RunConfig,
    RunOptions,
    SuccessRecord,
    ValidationMethod,
    ValidationResult,
    VersionInfo,
    _finalize_benchmark_run,
    parse_args,
)

if TYPE_CHECKING:
    from pathlib import Path


def _run_config(**overrides: Any) -> RunConfig:
    values: dict[str, Any] = {
        "engine_name": "cudf-polars",
        "queries": [1],
        "query_set": "pdsh",
        "dataset_path": "/data/scale-10",
        "scale_factor": 10,
        "suffix": ".parquet",
        "query_mode": QueryMode.DATAFRAME,
        "frontend": "spmd",
        "iterations": 1,
        "n_workers": 2,
        "extra_info": {
            "environment": {"POLARS_MAX_THREADS": "1", "UNSET": None},
            "campaign": "nightly",
        },
        "versions": PackageVersions(
            cudf_polars=VersionInfo(version="test-version", commit="abc123"),
            polars="1.35.0",
            python="3.13.0",
            rapidsmpf=VersionInfo(version="0.3.0", commit="def456"),
            duckdb="1.4.0",
        ),
        "records": {
            1: [
                SuccessRecord(
                    query=1,
                    iteration=0,
                    duration=0.25,
                    statistics={"alloc": {"count": 2, "value": 1.5}},
                    io_summaries={"0": {"bytes_read": 10}},
                    validation_result=ValidationResult(status="Passed", message=None),
                )
            ]
        },
        "hardware": HardwareInfo(
            gpus=[
                GPUInfo(
                    name="GPU",
                    index=0,
                    free_memory=1,
                    used_memory=2,
                    total_memory=3,
                )
            ],
            cpu=CPUInfo(model="CPU", physical_cores=4, logical_cores=8),
        ),
        "run_id": uuid.UUID("00000000-0000-0000-0000-000000000001"),
        "timestamp": "2026-11-10T12:00:00+00:00",
        "command_line": "python -m benchmark",
        "capture_env_vars": "",
        "validation_method": ValidationMethod(
            expected_source="polars-cpu",
            comparison_method="polars",
            comparison_options={},
            expected_location=None,
        ),
    }
    values.update(overrides)
    return RunConfig(**values)


def test_schema_v2_preserves_run_and_iteration_details(tmp_path: Path) -> None:
    output = tmp_path / "results" / "out.jsonl"
    archive = tmp_path / "logs" / "trace.zip"
    run = _run_config(
        sku_name="h100",
        storage_configuration_name="lustre",
        benchmark_definition_name="tpch-sf10",
        node_count=1,
    )

    result = run.serialize(None, archive, output_path=output)

    assert result["schema_version"] == 2
    assert result["query_engine"] == {
        "engine_name": "cudf-polars",
        "version": "test-version",
        "commit_hash": "abc123",
    }
    assert result["gpu_count"] == 2
    assert result["validation_status"] == "passed"
    assert result["sku_name"] == "h100"
    assert result["storage_configuration_name"] == "lustre"
    assert result["benchmark_definition_name"] == "tpch-sf10"
    assert result["node_count"] == 1
    assert result["environment"] == [
        {"key": "POLARS_MAX_THREADS", "value": "1"},
        {"key": "UNSET", "value": ""},
    ]
    assert result["extra_info"]["user"] == {"campaign": "nightly"}
    assert result["extra_info"]["hardware"]["gpus"][0]["name"] == "GPU"

    query_log = result["query_logs"][0]
    assert query_log["query_name"] == "1"
    assert query_log["execution_order"] == 0
    assert query_log["runtime_ms"] == 250
    assert query_log["validation_result"]["status"] == "passed"
    assert query_log["engine_metrics"] == {
        "alloc.count": 2,
        "alloc.value": 1.5,
    }
    assert query_log["extra_info"]["statistics"] == {
        "alloc": {"count": 2, "value": 1.5}
    }
    assert query_log["extra_info"]["io_summaries"] == {"0": {"bytes_read": 10}}

    assert result["roles"][-1] == {"type": "quent", "filename": "trace.zip"}
    assert result["artifacts"] == [
        {
            "path": "../logs/trace.zip",
            "filename": "trace.zip",
            "title": "Quent Archive",
            "description": "ZIP archive containing Quent execution traces.",
            "media_type": "application/zip",
        }
    ]


def test_schema_v2_failure_is_not_validated() -> None:
    result = _run_config(
        records={
            1: [FailedRecord(query=1, iteration=0, traceback="failed to execute query")]
        },
        validation_method=None,
    ).serialize(None, None)

    assert result["validation_status"] == "not-validated"
    assert result["query_logs"] == [
        {
            "query_name": "1",
            "execution_order": 0,
            "status": "error",
            "extra_info": {
                "iteration": 0,
                "traceback": "failed to execute query",
            },
            "validation_result": None,
        }
    ]


def test_partial_validation_is_not_marked_passed() -> None:
    result = _run_config(
        records={
            1: [
                SuccessRecord(
                    query=1,
                    iteration=0,
                    duration=1,
                    validation_result=ValidationResult(status="Passed", message=None),
                ),
                SuccessRecord(query=1, iteration=1, duration=2),
            ]
        }
    ).serialize(None, None)

    assert result["validation_status"] == "not-validated"


def test_catalog_cli_fields_are_optional() -> None:
    args = parse_args(
        [
            "1",
            "--frontend",
            "polars-cpu",
            "--path",
            "/data",
            "--scale",
            "1",
        ]
    )
    assert args.sku_name is None
    assert args.storage_configuration_name is None
    assert args.benchmark_definition_name is None
    assert args.node_count is None

    result = _run_config(
        frontend="polars-cpu",
        engine_name="polars-cpu",
    ).serialize(None, None)
    assert result["gpu_count"] == 0
    assert "sku_name" not in result
    assert "storage_configuration_name" not in result
    assert "benchmark_definition_name" not in result
    assert "node_count" not in result


def test_finalization_appends_lifecycle_timings(tmp_path: Path) -> None:
    output = tmp_path / "results.jsonl"
    run = _run_config(validation_method=None)
    options = RunOptions(output=output)

    for startup, shutdown in ((10.5, 5.25), (20.5, 6.25)):
        with pytest.raises(SystemExit, match="0"):
            _finalize_benchmark_run(
                options,
                run,
                validation_failures=[],
                query_failures=[],
                serializable_engine_config=run.serialize(
                    None, None, output_path=output
                ),
                startup_duration_ms=startup,
                shutdown_duration_ms=shutdown,
            )

    results = [json.loads(line) for line in output.read_text().splitlines()]
    assert [
        (result["startup_duration_ms"], result["shutdown_duration_ms"])
        for result in results
    ] == [(10.5, 5.25), (20.5, 6.25)]

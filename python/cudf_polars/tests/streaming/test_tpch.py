# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-H validation tests for the streaming GPU engine."""

from __future__ import annotations

import contextlib
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from cudf_polars.streaming.benchmarks.pdsh import PDSHQueries
from cudf_polars.streaming.benchmarks.utils import (
    FailedRecord,
    RunConfig,
    check_input_data_type,
    run_polars_query,
)
from cudf_polars.testing.engine_utils import warns_on_spmd

if TYPE_CHECKING:
    from pytest_subtests import SubTests

    from cudf_polars.engine.spmd import SPMDEngine
    from cudf_polars.streaming.benchmarks.utils import RunOptions, ValidationMethod

TPCH_SUFFIX = "/*.parquet"

EXPECTED_WARNINGS: dict[int, str] = {
    11: "ConditionalJoin not supported for multiple partitions.",
    22: "ConditionalJoin not supported for multiple partitions.",
}


@pytest.fixture(scope="session")
def tpch_data_dir(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> str:
    path = request.config.getoption("path")
    if path is not None:
        return path
    data_dir = tmp_path_factory.mktemp("tpch")
    try:
        subprocess.run(
            [
                "tpchgen-cli",
                "parquet",
                "-s",
                str(request.config.getoption("scale") or 1.0),
                "--parts=4",
                f"--output-dir={data_dir}",
            ],
            check=True,
            timeout=1800,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            "tpchgen-cli is not installed. Install it to generate TPC-H test data."
        ) from e
    return str(data_dir)


@pytest.fixture(scope="session")
def tpch_run_config(
    request: pytest.FixtureRequest,
    tpch_data_dir: str,
    tpc_iterations: int,
    tpc_validation_method: ValidationMethod,
) -> RunConfig:
    return RunConfig(
        engine_name="cudf-polars",
        queries=list(range(1, PDSHQueries.num_queries + 1)),
        query_set="pdsh",
        dataset_path=Path(tpch_data_dir),
        scale_factor=request.config.getoption("scale") or 1.0,
        suffix=request.config.getoption("suffix") or TPCH_SUFFIX,
        frontend="spmd",
        iterations=tpc_iterations,
        io_mode=request.config.getoption("io_mode"),
        validation_method=tpc_validation_method,
        command_line="",
        capture_env_vars="",
    )


@pytest.fixture(scope="session")
def tpch_numeric_type(tpch_run_config: RunConfig) -> str:
    numeric_type, _ = check_input_data_type(tpch_run_config)
    return numeric_type


@pytest.fixture(scope="session")
def tpch_date_type(tpch_run_config: RunConfig) -> str:
    _, date_type = check_input_data_type(tpch_run_config)
    return date_type


def test_tpch_query(
    request: pytest.FixtureRequest,
    subtests: SubTests,
    q_id: int,
    tpch_run_config: RunConfig,
    tpch_numeric_type: str,
    tpch_date_type: str,
    tpc_spmd_engine: SPMDEngine,
    tpc_run_options: RunOptions,
) -> None:
    reason = PDSHQueries.EXPECTED_FAILURES_TPCH.get(q_id)
    if reason is not None:
        request.applymarker(pytest.mark.xfail(reason=reason))

    warning = EXPECTED_WARNINGS.get(q_id)
    ctx = (
        warns_on_spmd(tpc_spmd_engine, UserWarning, match=warning)
        if warning is not None
        else contextlib.nullcontext()
    )

    query_result = getattr(PDSHQueries, f"q{q_id}")(tpch_run_config)

    with ctx:
        qr = run_polars_query(
            q_id=q_id,
            query_result=query_result,
            benchmark=PDSHQueries,
            run_config=tpch_run_config,
            run_options=tpc_run_options,
            engine=tpc_spmd_engine,
            numeric_type=tpch_numeric_type,
            date_type=tpch_date_type,
        )

    if reason is not None:
        record = qr.query_records[0]
        if isinstance(record, FailedRecord):
            raise RuntimeError(record.traceback)
        if (
            record.validation_result is not None
            and record.validation_result.status == "Failed"
        ):
            raise RuntimeError(record.validation_result.message or "Validation failed")
    else:
        for record in qr.query_records:
            with subtests.test(msg=f"iter{record.iteration}"):
                if isinstance(record, FailedRecord):
                    pytest.fail(record.traceback)
                elif (
                    record.validation_result is not None
                    and record.validation_result.status == "Failed"
                ):
                    pytest.fail(record.validation_result.message or "Validation failed")

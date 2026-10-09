# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS validation tests for the streaming GPU engine."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pytest

from cudf_polars.streaming.benchmarks.pdsds import PDSDSPolarsQueries
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

TPCDS_SUFFIX = ".parquet"

CONDITIONAL_JOIN_NOT_SUPPORTED = (
    "ConditionalJoin not supported for multiple partitions."
)
SORT_NOT_SUPPORTED = "sort currently only supports column names as `by` keys."

EXPECTED_WARNINGS: dict[int, str] = {
    14: CONDITIONAL_JOIN_NOT_SUPPORTED,
    23: CONDITIONAL_JOIN_NOT_SUPPORTED,
    24: CONDITIONAL_JOIN_NOT_SUPPORTED,
    36: SORT_NOT_SUPPORTED,
    70: SORT_NOT_SUPPORTED,
    86: SORT_NOT_SUPPORTED,
}


@pytest.fixture(scope="session")
def tpcds_data_dir(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> str:
    path = request.config.getoption("path") or os.environ.get("TPCDS_DATA_DIR")
    if path is not None:
        return path
    scale = request.config.getoption("scale") or 1.0
    data_dir = tmp_path_factory.mktemp("tpcds")
    with duckdb.connect() as conn:
        conn.execute("INSTALL tpcds")
        conn.execute("LOAD tpcds")
        conn.execute("CALL dsdgen(sf=$1)", [scale])
        for table in conn.execute("SHOW TABLES").df()["name"]:
            conn.execute(
                f"COPY {table} TO $1 (FORMAT PARQUET)",
                [f"{data_dir}/{table}.parquet"],
            )
    return str(data_dir)


@pytest.fixture(scope="session")
def tpcds_run_config(
    request: pytest.FixtureRequest,
    tpcds_data_dir: str,
    tpc_iterations: int,
    tpc_validation_method: ValidationMethod,
) -> RunConfig:
    return RunConfig(
        engine_name="cudf-polars",
        queries=list(range(1, PDSDSPolarsQueries.num_queries + 1)),
        query_set="pdsds",
        dataset_path=Path(tpcds_data_dir),
        scale_factor=request.config.getoption("scale") or 1.0,
        suffix=request.config.getoption("suffix") or TPCDS_SUFFIX,
        qualification=request.config.getoption("qualification"),
        frontend="spmd",
        iterations=tpc_iterations,
        io_mode=request.config.getoption("io_mode"),
        validation_method=tpc_validation_method,
        command_line="",
        capture_env_vars="",
    )


@pytest.fixture(scope="session")
def tpcds_numeric_type(tpcds_run_config: RunConfig) -> str:
    numeric_type, _ = check_input_data_type(tpcds_run_config)
    return numeric_type


@pytest.fixture(scope="session")
def tpcds_date_type(tpcds_run_config: RunConfig) -> str:
    _, date_type = check_input_data_type(tpcds_run_config)
    return date_type


def test_tpcds_query(
    request: pytest.FixtureRequest,
    subtests: SubTests,
    q_id: int,
    tpcds_run_config: RunConfig,
    tpcds_numeric_type: str,
    tpcds_date_type: str,
    tpc_spmd_engine: SPMDEngine,
    tpc_run_options: RunOptions,
) -> None:
    reason = PDSDSPolarsQueries.EXPECTED_FAILURES_TPCDS.get(q_id)
    if reason is not None:
        request.applymarker(pytest.mark.xfail(reason=reason))

    warning = EXPECTED_WARNINGS.get(q_id)
    ctx = (
        warns_on_spmd(tpc_spmd_engine, UserWarning, match=warning)
        if warning is not None
        else contextlib.nullcontext()
    )

    query_result = getattr(PDSDSPolarsQueries, f"q{q_id}")(tpcds_run_config)

    with ctx:
        qr = run_polars_query(
            q_id=q_id,
            query_result=query_result,
            benchmark=PDSDSPolarsQueries,
            run_config=tpcds_run_config,
            run_options=tpc_run_options,
            engine=tpc_spmd_engine,
            numeric_type=tpcds_numeric_type,
            date_type=tpcds_date_type,
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

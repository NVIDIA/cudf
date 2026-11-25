# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TPC-DS validation tests for the streaming GPU engine."""

from __future__ import annotations

import contextlib
import os
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pytest

from cudf_polars.streaming.benchmarks.pdsds import PDSDSPolarsQueries
from cudf_polars.streaming.benchmarks.utils import (
    FailedRecord,
    RunConfig,
    SQLQueries,
    check_input_data_type,
    run_polars_query,
)
from cudf_polars.testing.engine_utils import warns_on_spmd

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_subtests import SubTests

    import polars as pl

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
    36: SORT_NOT_SUPPORTED,
    70: SORT_NOT_SUPPORTED,
    86: SORT_NOT_SUPPORTED,
}

# Whether these warn depends on how many partitions the inputs have, which
# varies with the scale factor and hardware. Tolerate the warning but don't
# require it.
OPTIONAL_WARNINGS: dict[int, str] = {
    24: CONDITIONAL_JOIN_NOT_SUPPORTED,
}

DECIMAL_DIVISOR = "`decimal_div` with an integer divisor, #25396"
DECIMAL_SCALE = (
    "Comparing Decimals whose common scale does not fit in 38 digits, #25395"
)
INT_RANGE = "RangeFunction.IntRange is not supported, #25397"

# The Polars SQL queries that do not yet run on the GPU, with the reason. The
# tests are strict: fixing one fails its test until the entry is removed.
EXPECTED_SQL_FAILURES: dict[int, str] = {
    4: DECIMAL_DIVISOR,
    21: DECIMAL_DIVISOR,
    23: f"{DECIMAL_SCALE}; broadcasting aggregated expressions in groupby, #20565",
    34: DECIMAL_DIVISOR,
    45: "ListFunction.Contains and ListFunction.Length are not supported, #25398",
    47: INT_RANGE,
    51: "fill_null over(...) of a non-pointwise expression other than cum_sum, #23606",
    54: DECIMAL_DIVISOR,
    57: INT_RANGE,
    58: f"{DECIMAL_DIVISOR}; {DECIMAL_SCALE}",
    66: DECIMAL_DIVISOR,
    72: "Adding an integer number of days to a Date is not supported, #25400",
    73: DECIMAL_DIVISOR,
    78: DECIMAL_DIVISOR,
    83: DECIMAL_DIVISOR,
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
        request.applymarker(pytest.mark.xfail(reason=reason, strict=False))

    warning = EXPECTED_WARNINGS.get(q_id)
    optional_warning = OPTIONAL_WARNINGS.get(q_id)
    ctx: contextlib.AbstractContextManager[object]
    if warning is not None:
        ctx = warns_on_spmd(tpc_spmd_engine, UserWarning, match=warning)
    elif optional_warning is not None:
        ctx = warnings.catch_warnings()
    else:
        ctx = contextlib.nullcontext()

    query_result = getattr(PDSDSPolarsQueries, f"q{q_id}")(tpcds_run_config)

    with ctx:
        if optional_warning is not None:
            warnings.filterwarnings(
                "ignore", message=optional_warning, category=UserWarning
            )
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
        for record in qr.query_records:
            if isinstance(record, FailedRecord):
                message = record.traceback
            elif (
                record.validation_result is not None
                and record.validation_result.status == "Failed"
            ):
                message = record.validation_result.message or "Validation failed"
            else:
                continue
            raise RuntimeError(message)
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


def test_tpcds_sql_query(
    request: pytest.FixtureRequest,
    q_id: int,
    tpcds_run_config: RunConfig,
    tpc_sql_directory: Path,
    tpc_assert_sql_matches_cpu: Callable[[pl.LazyFrame], None],
) -> None:
    reason = EXPECTED_SQL_FAILURES.get(q_id)
    if reason is not None:
        request.applymarker(pytest.mark.xfail(reason=reason))

    queries = SQLQueries("pdsds", tpc_sql_directory / "tpcds")
    frame = getattr(queries, f"q{q_id}")(tpcds_run_config).frame
    tpc_assert_sql_matches_cpu(frame)

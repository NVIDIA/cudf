# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import os
import tarfile
import urllib.request
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from polars.testing import assert_frame_equal

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.engine.spmd import SPMDEngine
from cudf_polars.streaming.benchmarks.pdsds import PDSDSPolarsQueries
from cudf_polars.streaming.benchmarks.pdsh import PDSHQueries
from cudf_polars.streaming.benchmarks.utils import (
    POLARS_VALIDATION_OPTIONS,
    RunOptions,
    ValidationMethod,
    _add_dataset_args,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

    import polars as pl

# The SQL queries of https://github.com/pola-rs/polars-2.0-benchmark, which has
# no license to vendor them under. Fetched at this revision, or read from the
# directory in TPC_SQL_DIR, which holds the ``tpch`` and ``tpcds`` directories.
TPC_SQL_REVISION = "313d104e2d871c9a36229417ac06c8668ff842ae"
TPC_SQL_URL = f"https://codeload.github.com/pola-rs/polars-2.0-benchmark/tar.gz/{TPC_SQL_REVISION}"

# Warnings about partitioning rather than fallback to the CPU.
TPC_SQL_ALLOWED_WARNINGS = (
    "ConditionalJoin not supported for multiple partitions.",
    "sort currently only supports column names as `by` keys.",
)


class PytestGroupAdapter:
    """Adapts a pytest option group to the argparse group interface."""

    def __init__(self, group: Any) -> None:
        self._group = group

    def add_argument(self, *args: Any, **kwargs: Any) -> None:
        if kwargs.get("action") is argparse.BooleanOptionalAction:
            dest = kwargs.get("dest") or args[0].lstrip("-").replace("-", "_")
            self._group.addoption(
                *args,
                dest=dest,
                action="store_true",
                default=None,
                help=kwargs.get("help", ""),
            )
            for opt in args:
                self._group.addoption(
                    f"--no-{opt.lstrip('-')}",
                    dest=dest,
                    action="store_false",
                )
        else:
            kwargs.pop("metavar", None)
            self._group.addoption(*args, **kwargs)

    def add_argument_group(self, *args: Any, **kwargs: Any) -> PytestGroupAdapter:
        return self


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("TPC benchmark options")
    adapter = PytestGroupAdapter(group)
    StreamingOptions._add_cli_args(adapter)  # type: ignore[arg-type]
    _add_dataset_args(adapter)  # type: ignore[arg-type]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "q_id" not in metafunc.fixturenames:
        return
    name = metafunc.function.__name__
    if "tpch" in name:
        num_queries = PDSHQueries.num_queries
    elif "tpcds" in name:
        num_queries = PDSDSPolarsQueries.num_queries
    else:
        return
    metafunc.parametrize(
        "q_id",
        range(1, num_queries + 1),
        ids=[f"q{i:02d}" for i in range(1, num_queries + 1)],
    )


@pytest.fixture(scope="session")
def tpc_streaming_options(request: pytest.FixtureRequest) -> StreamingOptions:
    ref = argparse.ArgumentParser()
    StreamingOptions._add_cli_args(ref)
    _add_dataset_args(ref)
    ns: dict[str, Any] = {}
    for action in ref._actions:
        if not action.option_strings or action.dest == "help":
            continue
        ns[action.dest] = request.config.getoption(action.dest)
    # Raise on fallback unless --no-raise-on-fail was given.
    if ns["raise_on_fail"] is None:
        ns["raise_on_fail"] = True
    return StreamingOptions._from_argparse(argparse.Namespace(**ns))


@pytest.fixture(scope="session")
def tpc_run_options(request: pytest.FixtureRequest) -> RunOptions:
    return RunOptions(
        validation_abs_tol=request.config.getoption("validation_abs_tol"),
    )


@pytest.fixture(scope="session")
def tpc_validation_method(tpc_run_options: RunOptions) -> ValidationMethod:
    return ValidationMethod(
        expected_source="duckdb",
        comparison_method="polars",
        comparison_options={
            **POLARS_VALIDATION_OPTIONS,
            "abs_tol": tpc_run_options.validation_abs_tol,
        },
        expected_location=None,
    )


@pytest.fixture(scope="session")
def tpc_spmd_engine(
    tpc_streaming_options: StreamingOptions,
) -> Generator[SPMDEngine, None, None]:
    with SPMDEngine(
        rapidsmpf_options=tpc_streaming_options.to_rapidsmpf_options(),
        executor_options=tpc_streaming_options.to_executor_options(),
        engine_options=tpc_streaming_options.to_engine_options(),
    ) as engine:
        yield engine


@pytest.fixture(scope="session")
def tpc_sql_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Directory with the ``tpch`` and ``tpcds`` SQL query files."""
    if (path := os.environ.get("TPC_SQL_DIR")) is not None:
        return Path(path)
    root = tmp_path_factory.mktemp("tpc_sql")
    try:
        with (
            urllib.request.urlopen(TPC_SQL_URL, timeout=300) as response,
            tarfile.open(fileobj=response, mode="r|gz") as archive,
        ):
            for member in archive:
                parts = Path(member.name).parts
                # <repository>/queries/sql/<suite>/<query>.sql
                if len(parts) != 5 or parts[1:3] != ("queries", "sql"):
                    continue
                if not member.isfile() or not parts[4].endswith(".sql"):
                    continue
                target = root / parts[3] / parts[4]
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())  # type: ignore[union-attr]
    except OSError as e:
        raise RuntimeError(
            f"Could not fetch the SQL queries from {TPC_SQL_URL}. Download them "
            "and set TPC_SQL_DIR to a directory with `tpch` and `tpcds` inside."
        ) from e
    return root


@pytest.fixture(scope="session")
def tpc_assert_sql_matches_cpu(
    tpc_spmd_engine: SPMDEngine, tpc_run_options: RunOptions
) -> Callable[[pl.LazyFrame], None]:
    """Check that a SQL query runs on the GPU and gives the CPU result."""

    def check(frame: pl.LazyFrame) -> None:
        with warnings.catch_warnings():
            for message in TPC_SQL_ALLOWED_WARNINGS:
                warnings.filterwarnings("ignore", message=message, category=UserWarning)
            gpu = frame.collect(engine=tpc_spmd_engine)
        # Ties at a LIMIT can differ between the engines, see #24883.
        assert_frame_equal(
            gpu,
            frame.collect(),
            check_row_order=False,
            check_exact=False,
            rel_tol=POLARS_VALIDATION_OPTIONS["rel_tol"],
            abs_tol=tpc_run_options.validation_abs_tol,
        )

    return check

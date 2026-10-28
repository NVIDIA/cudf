# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Any

import pytest

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
    from collections.abc import Generator


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
    ns: dict[str, Any] = {"raise_on_fail": True}
    for action in ref._actions:
        if not action.option_strings or action.dest == "help":
            continue
        ns[action.dest] = request.config.getoption(action.dest)
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

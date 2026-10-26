# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the TPC benchmark CLI argument parsing."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cudf_polars.streaming.benchmarks.pdsds import PDSDSPolarsQueries
from cudf_polars.streaming.benchmarks.pdsh import PDSHQueries
from cudf_polars.streaming.benchmarks.utils import RunConfig, parse_args


@pytest.mark.parametrize("query_id", [1, PDSHQueries.num_queries])
def test_tpch_cli_parse(query_id: int) -> None:
    args = parse_args(
        [
            str(query_id),
            "--path",
            "/data/tpch",
            "--scale",
            "1",
            "--suffix",
            "/*.parquet",
            "--frontend",
            "spmd",
            "--validate-against",
            "duckdb",
            "--iterations",
            "2",
            "--io-mode",
            "lukewarm",
        ],
        num_queries=PDSHQueries.num_queries,
    )
    assert args.query == [query_id]
    assert args.iterations == 2
    assert args.frontend == "spmd"
    assert args.io_mode == "lukewarm"
    assert args.suffix == "/*.parquet"


@pytest.mark.parametrize("query_id", [1, PDSDSPolarsQueries.num_queries])
def test_tpcds_cli_parse(query_id: int, tmp_path: Path) -> None:
    args = parse_args(
        [
            str(query_id),
            "--path",
            str(tmp_path),
            "--scale",
            "1",
            "--qualification",
            "--frontend",
            "spmd",
            "--validate-against",
            "duckdb",
            "--iterations",
            "2",
            "--io-mode",
            "lukewarm",
        ],
        num_queries=PDSDSPolarsQueries.num_queries,
    )
    vars(args).update({"query_set": PDSDSPolarsQueries.name})
    run_config = RunConfig.from_args(args)
    assert run_config.queries == [query_id]
    assert run_config.iterations == 2
    assert run_config.qualification
    assert run_config.frontend == "spmd"
    assert run_config.io_mode == "lukewarm"

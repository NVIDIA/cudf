# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for the TPC benchmark CLI argument parsing."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cudf_polars.streaming.benchmarks.pdsds import PDSDSPolarsQueries
from cudf_polars.streaming.benchmarks.pdsh import PDSHQueries
from cudf_polars.streaming.benchmarks.utils import (
    QueryMode,
    RunConfig,
    SQLQueries,
    SQLQueryFiles,
    drop_file_page_cache_recursively,
    parse_args,
)


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


@pytest.mark.parametrize("io_mode", ["cold", "lukewarm"])
def test_remote_path_single_iteration(
    io_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = parse_args(
        [
            "1",
            "--path",
            "s3://bucket/tpcds/scale-1",
            "--scale",
            "1",
            "--frontend",
            "duckdb",
            "--iterations",
            "1",
            "--io-mode",
            io_mode,
        ],
        num_queries=PDSDSPolarsQueries.num_queries,
    )
    vars(args).update({"query_set": PDSDSPolarsQueries.name})
    run_config = RunConfig.from_args(args)
    assert run_config.iterations == 1
    assert run_config.io_mode == io_mode
    monkeypatch.setitem(sys.modules, "kvikio", None)
    drop_file_page_cache_recursively(run_config.dataset_path)


def test_sql_cli_parse(tmp_path: Path) -> None:
    (tmp_path / "q01.sql").write_text("SELECT 1")
    args = parse_args(
        [
            "1",
            "--frontend",
            "in-memory",
            "--query-mode",
            "sql",
            "--sql-directory",
            str(tmp_path),
        ]
    )
    assert args.query_mode is QueryMode.SQL
    assert args.sql_directory == tmp_path


@pytest.mark.parametrize(
    "extra_args,message",
    [
        (("--query-mode", "sql"), "--query-mode sql requires --sql-directory"),
        (
            ("--sql-directory", "{sql_directory}"),
            "--sql-directory is only supported with --query-mode sql",
        ),
        (
            (
                "--query-mode",
                "sql",
                "--sql-directory",
                "{sql_directory}",
                "--qualification",
            ),
            "--qualification is not supported with --query-mode sql",
        ),
        (
            ("--query-mode", "sql", "--sql-directory", "{missing_directory}"),
            "is not a local directory",
        ),
        (
            ("--query-mode", "sql", "--sql-directory", "{sql_directory}"),
            "has no SQL file for the requested queries: q02.sql or q2.sql.",
        ),
    ],
)
def test_sql_cli_parse_raises(
    extra_args: tuple[str, ...],
    message: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "q01.sql").write_text("SELECT 1")
    formatted_args = (
        arg.format(sql_directory=tmp_path, missing_directory=tmp_path / "missing")
        for arg in extra_args
    )
    with pytest.raises(SystemExit):
        parse_args(["1,2", "--frontend", "in-memory", *formatted_args])
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("file_name", ["q01.sql", "q1.sql"])
def test_sql_query_files_path(file_name: str, tmp_path: Path) -> None:
    (tmp_path / file_name).write_text("SELECT 1")
    sql_files = SQLQueryFiles(name="pdsh", directory=tmp_path)
    assert sql_files.path(1) == tmp_path / file_name


def test_sql_query_files_path_prefers_padded(tmp_path: Path) -> None:
    (tmp_path / "q01.sql").write_text("SELECT 1")
    (tmp_path / "q1.sql").write_text("SELECT 2")
    sql_files = SQLQueryFiles(name="pdsh", directory=tmp_path)
    assert sql_files.path(1) == tmp_path / "q01.sql"


def test_sql_queries_duckdb_queries(tmp_path: Path) -> None:
    benchmark = SQLQueries("pdsh", tmp_path)
    assert benchmark().duckdb_queries == SQLQueryFiles(name="pdsh", directory=tmp_path)

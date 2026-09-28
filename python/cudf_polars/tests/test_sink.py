# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

import polars as pl

from cudf_polars.testing.asserts import (
    assert_sink_ir_translation_raises,
    assert_sink_result_equal,
)
from cudf_polars.testing.engine_utils import get_blocksize_mode
from cudf_polars.utils.versions import POLARS_VERSION_LT_138


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "a": [1, 2, 3, None, 4, 5],
            "b": ["ẅ", "x", "y", "z", "123", "abcd"],
        }
    )


@pytest.mark.parametrize("include_header", [True, False])
@pytest.mark.parametrize("null_value", [None, "NA"])
@pytest.mark.parametrize("line_terminator", ["\n", "\n\n"])
@pytest.mark.parametrize("separator", [",", "|"])
def test_sink_csv(
    engine: pl.GPUEngine,
    df,
    tmp_path,
    include_header,
    null_value,
    line_terminator,
    separator,
):
    if line_terminator == "\n\n" and get_blocksize_mode(engine) == "small":
        # We end up with an extra row per partition.
        pytest.skip("Multi-line terminator not supported with small blocksize")
    assert_sink_result_equal(
        df,
        tmp_path / "out.csv",
        engine=engine,
        write_kwargs={
            "include_header": include_header,
            "null_value": null_value,
            "line_terminator": line_terminator,
            "separator": separator,
        },
        read_kwargs={
            "has_header": include_header,
        },
    )


@pytest.mark.parametrize(
    "kwarg, value",
    [
        ("include_bom", True),
        ("date_format", "%Y-%m-%d"),
        ("time_format", "%H:%M:%S"),
        ("datetime_format", "%Y-%m-%dT%H:%M:%S"),
        ("float_scientific", True),
        ("float_precision", 10),
        ("quote_style", "non_numeric"),
        ("quote_char", "`"),
    ],
)
def test_sink_csv_unsupported_kwargs(engine: pl.GPUEngine, df, tmp_path, kwarg, value):
    assert_sink_ir_translation_raises(
        df,
        tmp_path / "unsupported.csv",
        engine,
        {kwarg: value},
        NotImplementedError,
    )


def test_sink_ndjson(engine: pl.GPUEngine, df, tmp_path):
    assert_sink_result_equal(
        df,
        tmp_path / "out.ndjson",
        engine=engine,
    )


@pytest.mark.parametrize("mkdir", [True, False])
@pytest.mark.parametrize("data_page_size", [None, 256_000])
@pytest.mark.parametrize("row_group_size", [None, 1_000])
@pytest.mark.parametrize("is_chunked", [False, True])
@pytest.mark.parametrize("n_output_chunks", [1, 4, 8])
def test_sink_parquet(
    df, tmp_path, mkdir, data_page_size, row_group_size, is_chunked, n_output_chunks
):
    assert_sink_result_equal(
        df,
        tmp_path / "out.parquet",
        write_kwargs={
            "mkdir": mkdir,
            "data_page_size": data_page_size,
            "row_group_size": row_group_size,
        },
        engine=pl.GPUEngine(
            executor="in-memory",
            raise_on_fail=True,
            parquet_options={"chunked": is_chunked, "n_output_chunks": n_output_chunks},
        ),
    )


def test_sink_parquet_array_falls_back(in_memory_engine, tmp_path):
    df = pl.LazyFrame({"a": pl.Series([[1, 2], [3, 4]], dtype=pl.Array(pl.Int8, 2))})

    assert_sink_ir_translation_raises(
        df,
        tmp_path / "array.parquet",
        in_memory_engine,
        {},
        NotImplementedError,
    )


@pytest.mark.parametrize("compression_level", [9, None])
@pytest.mark.parametrize(
    "compression", ["zstd", "gzip", "brotli", "snappy", "lz4", "uncompressed"]
)
def test_sink_parquet_compression_type(
    engine: pl.GPUEngine, df, tmp_path, compression, compression_level
):
    # LZO compression not supported in polars
    # Only zstd and gzip take a compression level, which libcudf cannot set
    if compression in {"zstd", "gzip"} and compression_level is None:
        assert_sink_result_equal(
            df,
            tmp_path / "compression.parquet",
            write_kwargs={
                "compression": compression,
                "compression_level": compression_level,
            },
            engine=pl.GPUEngine(executor="in-memory", raise_on_fail=True),
        )
    elif compression in {"snappy", "lz4", "uncompressed"}:
        assert_sink_result_equal(
            df,
            tmp_path / "compression.parquet",
            write_kwargs={"compression": compression},
            engine=pl.GPUEngine(executor="in-memory", raise_on_fail=True),
        )
    else:
        assert_sink_ir_translation_raises(
            df,
            tmp_path / "unsupported_compression.parquet",
            engine,
            {"compression": compression, "compression_level": compression_level},
            NotImplementedError,
        )


def test_sink_csv_nested_data(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    path = tmp_path / "data.csv"

    lf = pl.LazyFrame({"list": [[1, 2, 3, 4, 5]]})
    with pytest.raises(
        pl.exceptions.ComputeError, match="CSV format does not support nested data"
    ):
        lf.sink_csv(path, engine=pl.GPUEngine())


def test_chunked_sink_empty_table_to_parquet(tmp_path):
    assert_sink_result_equal(
        pl.LazyFrame(),
        tmp_path / "out.parquet",
        engine=pl.GPUEngine(
            executor="in-memory",
            raise_on_fail=True,
            parquet_options={"chunked": True, "n_output_chunks": 2},
        ),
    )


@pytest.mark.parametrize("file_type", ["csv", "ndjson"])
def test_sink_in_memory_executor(df, tmp_path, file_type):
    assert_sink_result_equal(
        df,
        tmp_path / f"out.{file_type}",
        engine=pl.GPUEngine(raise_on_fail=True, executor="in-memory"),
    )


@pytest.mark.parametrize("compression", ["gzip", "zstd"])
@pytest.mark.parametrize("file_type", ["csv", "ndjson"])
@pytest.mark.skipif(
    POLARS_VERSION_LT_138,
    reason="compression parameter added in Polars 1.38",
)
def test_sink_compression_raises(
    engine: pl.GPUEngine, df, tmp_path, compression, file_type
):
    path = tmp_path / f"out.{file_type}"
    assert_sink_ir_translation_raises(
        df,
        path,
        engine,
        {"compression": compression, "check_extension": False},
        NotImplementedError,
    )

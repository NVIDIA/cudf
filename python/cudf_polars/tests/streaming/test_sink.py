# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl

from cudf_polars.engine.options import StreamingOptions
from cudf_polars.testing.asserts import assert_sink_result_equal


@pytest.fixture(scope="module")
def df():
    return pl.LazyFrame(
        {
            "x": range(30),
            "y": [1, 2, None] * 10,
            "z": ["ẅ", "a", "z", "123", "abcd"] * 6,
        }
    )


PARQUET_SINK_CASES = [
    pytest.param(10, None, None, True, id="multipart-default"),
    pytest.param(1_000_000, None, None, True, id="single-partition"),
    pytest.param(10, None, None, False, id="mkdir-disabled"),
    pytest.param(10, 1024, None, True, id="data-page-size"),
    pytest.param(10, None, 10, True, id="row-group-size"),
    pytest.param(10, 1024, 10, True, id="encoding-interaction"),
]


@pytest.mark.parametrize(
    "max_rows_per_partition,data_page_size,row_group_size,mkdir", PARQUET_SINK_CASES
)
def test_sink_parquet_single_file(
    df,
    streaming_engine_factory,
    tmp_path,
    mkdir,
    data_page_size,
    row_group_size,
    max_rows_per_partition,
):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(max_rows_per_partition=max_rows_per_partition),
    )
    assert_sink_result_equal(
        df,
        tmp_path / "test_sink.parquet",
        write_kwargs={
            "mkdir": mkdir,
            "data_page_size": data_page_size,
            "row_group_size": row_group_size,
        },
        engine=streaming_engine,
    )


@pytest.mark.parametrize(
    "max_rows_per_partition,data_page_size,row_group_size,mkdir", PARQUET_SINK_CASES
)
def test_sink_parquet_directory(
    df,
    streaming_engine_factory,
    tmp_path,
    mkdir,
    data_page_size,
    row_group_size,
    max_rows_per_partition,
):
    streaming_engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=max_rows_per_partition,
            sink_to_directory=True,
        ),
    )
    assert_sink_result_equal(
        df,
        tmp_path / "test_sink.parquet",
        write_kwargs={
            "mkdir": mkdir,
            "data_page_size": data_page_size,
            "row_group_size": row_group_size,
        },
        engine=streaming_engine,
    )

    check_path = Path(tmp_path / "test_sink_gpu.parquet")
    expected_file_count = (
        df.collect(engine=streaming_engine).height // max_rows_per_partition
    )
    assert check_path.is_dir()
    if expected_file_count > 1:
        assert len(list(check_path.iterdir())) == expected_file_count


def test_sink_parquet_raises(df: pl.LazyFrame, tmp_path, streaming_engine_factory):
    """No streaming-engine cluster supports ``sink_to_directory=False``."""
    engine = streaming_engine_factory(StreamingOptions(sink_to_directory=False))
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=r"ValueError: The [^ ]+ cluster requires sink_to_directory=True",
    ):
        df.sink_parquet(tmp_path / "test_sink_gpu.parquet", engine=engine)


@pytest.mark.parametrize(
    "max_rows_per_partition,separator,null_value,include_header",
    [
        pytest.param(10, ",", None, True, id="multipart-default"),
        pytest.param(1_000_000, ",", None, True, id="single-partition"),
        pytest.param(10, ",", None, False, id="without-header"),
        pytest.param(10, ",", "NA", True, id="null-value"),
        pytest.param(10, "|", None, True, id="separator"),
    ],
)
def test_sink_csv(
    df,
    streaming_engine_factory,
    tmp_path,
    include_header,
    null_value,
    separator,
    max_rows_per_partition,
):
    engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=max_rows_per_partition,
            raise_on_fail=True,
        ),
    )
    assert_sink_result_equal(
        df,
        tmp_path / "out.csv",
        write_kwargs={
            "include_header": include_header,
            "null_value": null_value,
            "separator": separator,
        },
        read_kwargs={
            "has_header": include_header,
        },
        engine=engine,
    )


@pytest.mark.parametrize("max_rows_per_partition", [10, 1_000_000])
def test_sink_ndjson(df, streaming_engine_factory, tmp_path, max_rows_per_partition):
    engine = streaming_engine_factory(
        StreamingOptions(
            max_rows_per_partition=max_rows_per_partition,
            raise_on_fail=True,
        ),
    )
    assert_sink_result_equal(
        df,
        tmp_path / "out.ndjson",
        engine=engine,
    )

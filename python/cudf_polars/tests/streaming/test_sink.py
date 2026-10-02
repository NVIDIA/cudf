# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import pytest

import polars as pl

from cudf_polars.dsl.ir import CallbackSink
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


@pytest.mark.parametrize("mkdir", [True, False])
@pytest.mark.parametrize("data_page_size", [None, 1024])
@pytest.mark.parametrize("row_group_size", [None, 10])
@pytest.mark.parametrize("max_rows_per_partition", [10, 1_000_000])
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


@pytest.mark.parametrize("mkdir", [True, False])
@pytest.mark.parametrize("data_page_size", [None, 1024])
@pytest.mark.parametrize("row_group_size", [None, 10])
@pytest.mark.parametrize("max_rows_per_partition", [10, 1_000_000])
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


@pytest.mark.parametrize("include_header", [True, False])
@pytest.mark.parametrize("null_value", [None, "NA"])
@pytest.mark.parametrize("separator", [",", "|"])
@pytest.mark.parametrize("max_rows_per_partition", [10, 1_000_000])
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


def test_callback_sink_batches(df, spmd_engine_factory, tmp_path):
    engine = spmd_engine_factory(
        StreamingOptions(max_rows_per_partition=10, raise_on_fail=True)
    )
    output = tmp_path / "batches.jsonl"

    def write_batch(batch: pl.DataFrame) -> None:
        with output.open("a") as file:
            file.write(json.dumps(batch.to_dict(as_series=False)) + "\n")

    sink = df.sink_batches(write_batch, chunk_size=7, lazy=True)
    assert not output.exists()
    assert sink.collect(engine=engine).shape == (0, 0)

    batches = [json.loads(line) for line in output.read_text().splitlines()]
    assert [len(batch["x"]) for batch in batches] == [7, 7, 7, 7, 2]
    assert [x for batch in batches for x in batch["x"]] == list(range(30))


def test_callback_sink_stops_after_true(df, spmd_engine_factory, tmp_path):
    engine = spmd_engine_factory(
        StreamingOptions(max_rows_per_partition=10, raise_on_fail=True)
    )
    output = tmp_path / "seen.txt"

    def stop_after_first(batch: pl.DataFrame) -> bool:
        with output.open("a") as file:
            file.write(f"{batch.height}\n")
        return True

    assert df.sink_batches(stop_after_first, chunk_size=7, engine=engine) is None
    assert output.read_text().splitlines() == ["7"]


@pytest.mark.spmd
def test_callback_sink_rejects_multiple_ranks(spmd_engine_factory):
    engine = spmd_engine_factory(StreamingOptions(raise_on_fail=True))
    if engine.comm.nranks == 1:
        pytest.skip("requires multiple SPMD ranks")

    sink = pl.LazyFrame({"x": [1]}).sink_batches(lambda batch: None, lazy=True)
    with pytest.raises(
        NotImplementedError,
        match="Callback sinks are not yet supported for multiple ranks",
    ):
        sink.collect(engine=engine)


def test_callback_sink_pickle_metadata():
    payload = pickle.dumps(bool)
    version = bytes(sys.version_info[1:3])
    assert CallbackSink.load_function(b"\x01" + version + payload) is bool
    assert CallbackSink.load_function(b"\x00\x00\x00" + payload) is bool

    other_minor = (sys.version_info.minor + 1) % 256
    with pytest.raises(ValueError, match="different Python version"):
        CallbackSink.load_function(bytes((1, other_minor, 0)) + payload)
    with pytest.raises(ValueError, match="Invalid Polars callback serialization"):
        CallbackSink.load_function(b"\x02" + version + payload)

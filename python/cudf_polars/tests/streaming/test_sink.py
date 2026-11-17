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


@pytest.mark.parametrize(
    "max_rows_per_partition,chunk_size,expected_sizes",
    [
        (10, None, [10, 10, 10]),
        (10, 7, [7, 7, 7, 7, 2]),
        (1, 7, [7, 7, 7, 7, 2]),
        (1, 100, [30]),
    ],
)
def test_callback_sink_batches(
    df,
    spmd_engine_factory,
    tmp_path,
    max_rows_per_partition,
    chunk_size,
    expected_sizes,
):
    engine = spmd_engine_factory(
        StreamingOptions(
            max_rows_per_partition=max_rows_per_partition, raise_on_fail=True
        )
    )
    output = tmp_path / "batches.jsonl"

    def write_batch(batch: pl.DataFrame) -> None:
        with output.open("a") as file:
            file.write(json.dumps(batch.to_dict(as_series=False)) + "\n")

    sink = df.sink_batches(write_batch, chunk_size=chunk_size, lazy=True)
    assert not output.exists()
    assert sink.collect(engine=engine).shape == (0, 0)

    batches = [json.loads(line) for line in output.read_text().splitlines()]
    assert [len(batch["x"]) for batch in batches] == expected_sizes
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
@pytest.mark.parametrize("chunk_size", [None, 3])
@pytest.mark.parametrize("stop_after_first", [False, True])
def test_callback_sink_multiple_ranks(
    spmd_engine_factory, tmp_path, chunk_size, stop_after_first
):
    engine = spmd_engine_factory(
        StreamingOptions(max_rows_per_partition=2, raise_on_fail=True)
    )
    rank = engine.comm.rank
    output = tmp_path / "rank_batches.jsonl"

    def write_batch(batch: pl.DataFrame) -> bool:
        with output.open("a") as file:
            file.write(json.dumps(batch.to_dict(as_series=False)) + "\n")
        return stop_after_first

    df = pl.LazyFrame({"x": range(rank * 5, rank * 5 + 5)})
    sink = df.sink_batches(write_batch, chunk_size=chunk_size, lazy=True)
    assert sink.collect(engine=engine).shape == (0, 0)
    if rank == 0:
        batches = [json.loads(line) for line in output.read_text().splitlines()]
        expected = list(range(engine.comm.nranks * 5))
        if stop_after_first:
            expected = expected[: chunk_size or 2]
        assert [x for batch in batches for x in batch["x"]] == expected
        if stop_after_first:
            assert len(batches) == 1
        elif chunk_size is not None:
            assert all(len(batch["x"]) == chunk_size for batch in batches[:-1])
    else:
        assert not output.exists()


@pytest.mark.spmd
@pytest.mark.parametrize("maintain_order", [False, True])
def test_callback_sink_parallel_option(spmd_engine_factory, tmp_path, maintain_order):
    engine = spmd_engine_factory(
        StreamingOptions(
            max_rows_per_partition=2,
            parallel_sink_batches=True,
            raise_on_fail=True,
        )
    )
    rank = engine.comm.rank
    output = tmp_path / f"parallel_rank_{rank}.jsonl"

    def write_batch(batch: pl.DataFrame) -> None:
        with output.open("a") as file:
            file.write(json.dumps(batch.to_dict(as_series=False)) + "\n")

    df = pl.LazyFrame({"x": range(rank * 5, rank * 5 + 5)})
    df.sink_batches(
        write_batch,
        chunk_size=3,
        maintain_order=maintain_order,
        engine=engine,
    )

    if maintain_order and rank != 0:
        assert not output.exists()
    else:
        batches = [json.loads(line) for line in output.read_text().splitlines()]
        expected = (
            list(range(engine.comm.nranks * 5))
            if maintain_order
            else list(range(rank * 5, rank * 5 + 5))
        )
        assert [x for batch in batches for x in batch["x"]] == expected
        assert all(len(batch["x"]) == 3 for batch in batches[:-1])


@pytest.mark.spmd
def test_callback_sink_empty_rank_zero(spmd_engine_factory, tmp_path):
    engine = spmd_engine_factory(StreamingOptions(raise_on_fail=True))
    output = tmp_path / "empty_rank_zero.jsonl"

    def write_batch(batch: pl.DataFrame) -> None:
        with output.open("a") as file:
            file.write(json.dumps(batch.to_dict(as_series=False)) + "\n")

    values = [] if engine.comm.rank == 0 else [engine.comm.rank]
    df = pl.LazyFrame({"x": pl.Series(values, dtype=pl.Int64)})
    df.sink_batches(write_batch, engine=engine)
    if engine.comm.rank == 0 and engine.comm.nranks > 1:
        batches = [json.loads(line) for line in output.read_text().splitlines()]
        assert [x for batch in batches for x in batch["x"]] == list(
            range(1, engine.comm.nranks)
        )
    else:
        assert not output.exists()


@pytest.mark.spmd
@pytest.mark.parametrize("parallel_sink_batches", [False, True])
def test_callback_sink_duplicated_input(
    spmd_engine_factory, tmp_path, parallel_sink_batches
):
    engine = spmd_engine_factory(
        StreamingOptions(
            parallel_sink_batches=parallel_sink_batches,
            raise_on_fail=True,
        )
    )
    output = tmp_path / "duplicated_batches.jsonl"

    def write_batch(batch: pl.DataFrame) -> None:
        with output.open("a") as file:
            file.write(json.dumps(batch.to_dict(as_series=False)) + "\n")

    df = pl.LazyFrame({"x": [engine.comm.rank + 1]}).select(pl.col("x").sum())
    df.sink_batches(
        write_batch,
        maintain_order=not parallel_sink_batches,
        engine=engine,
    )
    if engine.comm.rank == 0:
        batches = [json.loads(line) for line in output.read_text().splitlines()]
        assert batches == [{"x": [sum(range(1, engine.comm.nranks + 1))]}]
    else:
        assert not output.exists()


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

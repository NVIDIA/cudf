# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal

import pylibcudf as plc

from cudf_polars.containers import DataFrame, DataType
from cudf_polars.dsl.utils.per_path import PerPathValues
from cudf_polars.utils.cuda_stream import get_cuda_stream
from cudf_polars.utils.versions import POLARS_VERSION_LT_138


@pytest.fixture
def per_path() -> PerPathValues:
    return PerPathValues(pl.DataFrame({"part": [1, 2, 3], "cat": ["u", "u", "v"]}))


def test_from_polars() -> None:
    df = pl.DataFrame(
        {"part": [1, 2], "cat": ["u", "v"]},
        schema={"part": pl.Int32, "cat": pl.String},
    )
    got = PerPathValues.from_polars(df)
    assert got == PerPathValues(df)
    assert got is not None
    assert got.names == ("part", "cat")
    assert got.dtypes == (DataType(pl.Int32()), DataType(pl.String()))


@pytest.mark.skipif(
    POLARS_VERSION_LT_138,
    reason="height parameter added in Polars 1.38",
)
def test_from_polars_zero_width() -> None:
    assert PerPathValues.from_polars(pl.DataFrame(height=3)) is None


def test_hashable(per_path: PerPathValues) -> None:
    assert hash(per_path) == hash(
        PerPathValues(pl.DataFrame({"part": [1, 2, 3], "cat": ["u", "u", "v"]}))
    )


def test_dtypes_distinguish_identical_values() -> None:
    values = {"part": [1, 2]}
    narrow = PerPathValues(pl.DataFrame(values, schema={"part": pl.Int32}))
    wide = PerPathValues(pl.DataFrame(values, schema={"part": pl.Int64}))
    assert narrow != wide
    assert hash(narrow) != hash(wide)


def test_names_distinguish_identical_values() -> None:
    # hash_rows digests the values but not the column they sit in, so the
    # schema is what tells these two apart.
    values = [1, 2]
    part = PerPathValues(pl.DataFrame({"part": values}))
    cat = PerPathValues(pl.DataFrame({"cat": values}))
    assert part != cat
    assert hash(part) != hash(cat)


def test_path_order_matters() -> None:
    forwards = PerPathValues(pl.DataFrame({"part": [1, 2]}))
    backwards = PerPathValues(pl.DataFrame({"part": [2, 1]}))
    assert forwards != backwards
    assert hash(forwards) != hash(backwards)


def test_tall_partitions_are_distinguished() -> None:
    # Polars elides the middle of a tall frame's repr, so identity cannot
    # rest on it. hash_rows sees every row.
    tall = PerPathValues(pl.DataFrame({"part": range(40)}))
    other = PerPathValues(pl.DataFrame({"part": [*range(20), 999, *range(21, 40)]}))
    assert repr(tall.df) == repr(other.df)
    assert tall != other
    assert hash(tall) != hash(other)


def test_not_equal_to_other_types(per_path: PerPathValues) -> None:
    assert per_path != per_path.df


def test_repr(per_path: PerPathValues) -> None:
    assert repr(per_path) == f"PerPathValues(df={per_path.df!r})"


def test_num_paths(per_path: PerPathValues) -> None:
    assert per_path.num_paths == 3


@pytest.mark.parametrize(
    "values,expected",
    [
        ({"part": [1, 2, 3], "cat": ["u", "u", "v"]}, False),
        ({"part": [1, 1, 1], "cat": ["u", "u", "v"]}, False),
        ({"part": [1, 1, 1], "cat": ["u", "u", "u"]}, True),
        ({"part": [1], "cat": ["u"]}, True),
        ({"part": [None, None], "cat": [None, None]}, True),
    ],
)
def test_is_uniform(values, expected) -> None:
    assert PerPathValues(pl.DataFrame(values)).is_uniform is expected


def test_slice(per_path: PerPathValues) -> None:
    assert per_path.slice(1, 3) == PerPathValues(
        pl.DataFrame({"part": [2, 3], "cat": ["u", "v"]})
    )


def test_broadcast() -> None:
    stream = get_cuda_stream()
    per_path = PerPathValues(pl.DataFrame({"part": [7], "cat": ["u"]}))
    got = DataFrame(per_path.broadcast(3, stream=stream), stream=stream)
    assert_frame_equal(
        got.to_polars(), pl.DataFrame({"part": [7, 7, 7], "cat": ["u", "u", "u"]})
    )


def test_broadcast_uses_first_path() -> None:
    stream = get_cuda_stream()
    per_path = PerPathValues(pl.DataFrame({"part": [7, 7]}))
    got = DataFrame(per_path.broadcast(2, stream=stream), stream=stream)
    assert_frame_equal(got.to_polars(), pl.DataFrame({"part": [7, 7]}))


def test_broadcast_null() -> None:
    stream = get_cuda_stream()
    per_path = PerPathValues(pl.DataFrame({"part": [None]}, schema={"part": pl.Int64}))
    got = DataFrame(per_path.broadcast(2, stream=stream), stream=stream)
    assert_frame_equal(
        got.to_polars(), pl.DataFrame({"part": [None, None]}, schema={"part": pl.Int64})
    )


def test_repeat(per_path: PerPathValues) -> None:
    stream = get_cuda_stream()
    got = DataFrame(per_path.repeat([2, 0, 1], stream=stream), stream=stream)
    assert_frame_equal(
        got.to_polars(),
        pl.DataFrame({"part": [1, 1, 3], "cat": ["u", "u", "v"]}),
    )


def test_repeat_empty(per_path: PerPathValues) -> None:
    stream = get_cuda_stream()
    got = DataFrame(per_path.repeat([0, 0, 0], stream=stream), stream=stream)
    assert got.num_rows == 0


def test_gather(per_path: PerPathValues) -> None:
    stream = get_cuda_stream()
    source_index = plc.Column.from_arrow(
        pl.Series(values=[0, 0, 2, 1], dtype=pl.Int32()), stream=stream
    )
    got = DataFrame(per_path.gather(source_index, stream=stream), stream=stream)
    assert_frame_equal(
        got.to_polars(),
        pl.DataFrame({"part": [1, 1, 3, 2], "cat": ["u", "u", "v", "u"]}),
    )


def test_gather_preserves_nulls() -> None:
    stream = get_cuda_stream()
    per_path = PerPathValues(
        pl.DataFrame({"part": [1, None]}, schema={"part": pl.Int64})
    )
    source_index = plc.Column.from_arrow(
        pl.Series(values=[1, 0], dtype=pl.Int32()), stream=stream
    )
    got = DataFrame(per_path.gather(source_index, stream=stream), stream=stream)
    assert_frame_equal(got.to_polars(), pl.DataFrame({"part": [None, 1]}))


@pytest.mark.parametrize(
    "dtype",
    [pl.Int32, pl.Int64, pl.String, pl.Float64, pl.Boolean, pl.Date, pl.Datetime("us")],
)
def test_dtypes_survive_conversion(dtype: pl.DataType) -> None:
    stream = get_cuda_stream()
    series = pl.Series("part", [0, 1], dtype=pl.Int64).cast(dtype, strict=False)
    per_path = PerPathValues(series.to_frame())
    source_index = plc.Column.from_arrow(
        pl.Series(values=[1, 0], dtype=pl.Int32()), stream=stream
    )
    got = DataFrame(per_path.gather(source_index, stream=stream), stream=stream)
    assert_frame_equal(got.to_polars(), series.reverse().to_frame())

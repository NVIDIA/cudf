# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("level", [None, 0, "l0", 1, ["l0", 1]])
@pytest.mark.parametrize(
    "column_names",
    [
        ["v0", "v1"],
        ["v0", "index"],
        pd.MultiIndex.from_tuples([("x0", "x1"), ("y0", "y1")]),
        pd.MultiIndex.from_tuples([(1, 2), (10, 11)], names=["ABC", "DEF"]),
    ],
)
@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index(level, drop, column_names, inplace, col_level, col_fill):
    midx = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=["l0", None]
    )
    pdf = pd.DataFrame(
        [[1, 2], [3, 4], [5, 6], [7, 8]], index=midx, columns=column_names
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.reset_index(
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )
    got = gdf.reset_index(
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )
    if inplace:
        expect = pdf
        got = gdf

    assert_eq(expect, got)


@pytest.mark.parametrize("level", [None, 0, 1, [None]])
@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index_dup_level_name(level, drop, inplace, col_level, col_fill):
    # midx levels are named [None, None]
    midx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    pdf = pd.DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]], index=midx)
    gdf = cudf.from_pandas(pdf)
    if level == [None]:
        assert_exceptions_equal(
            lfunc=pdf.reset_index,
            rfunc=gdf.reset_index,
            lfunc_args_and_kwargs=(
                [],
                {"level": level, "drop": drop, "inplace": inplace},
            ),
            rfunc_args_and_kwargs=(
                [],
                {"level": level, "drop": drop, "inplace": inplace},
            ),
        )
        return

    expect = pdf.reset_index(
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )
    got = gdf.reset_index(
        level=level,
        drop=drop,
        inplace=inplace,
        col_level=col_level,
        col_fill=col_fill,
    )
    if inplace:
        expect = pdf
        got = gdf

    assert_eq(expect, got)


@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index_named(drop, inplace, col_level, col_fill):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pdf.index.name = "cudf"
    gdf.index.name = "cudf"

    expect = pdf.reset_index(
        drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill
    )
    got = gdf.reset_index(
        drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill
    )
    if inplace:
        expect = pdf
        got = gdf
    assert_eq(expect, got)


@pytest.mark.parametrize("column_names", [["x", "y"], ["index", "y"]])
@pytest.mark.parametrize("col_level", [0, 1])
@pytest.mark.parametrize("col_fill", ["", "some_lv"])
def test_reset_index_unnamed(drop, inplace, column_names, col_level, col_fill):
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    pdf.columns = column_names
    gdf.columns = column_names

    expect = pdf.reset_index(
        drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill
    )
    got = gdf.reset_index(
        drop=drop, inplace=inplace, col_level=col_level, col_fill=col_fill
    )
    if inplace:
        expect = pdf
        got = gdf
    assert_eq(expect, got)


def test_reset_index_invalid_level():
    with pytest.raises(IndexError):
        cudf.DataFrame([1]).reset_index(level=2)

    with pytest.raises(IndexError):
        pd.DataFrame([1]).reset_index(level=2)


@pytest.mark.parametrize(
    "level", [-1, -2, -3, [-1], [-2], [0, -1], [-2, -1], ["l0", -1]]
)
def test_reset_index_negative_level_multiindex(level, drop):
    # Negative levels count from the end of the MultiIndex.
    midx = pd.MultiIndex.from_tuples(
        [("a", 1, "x"), ("b", 2, "y")], names=["l0", "l1", "l2"]
    )
    pdf = pd.DataFrame({"v": [1, 2]}, index=midx)
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.reset_index(level=level, drop=drop),
        gdf.reset_index(level=level, drop=drop),
    )


@pytest.mark.parametrize("level", [-1, [-1]])
def test_reset_index_negative_level_flat(level, drop):
    pdf = pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="x"))
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.reset_index(level=level, drop=drop),
        gdf.reset_index(level=level, drop=drop),
    )


@pytest.mark.parametrize("level", [np.int64(-1), np.int32(0)])
def test_reset_index_numpy_integer_level(level, drop):
    midx = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["l0", "l1"])
    pdf = pd.DataFrame({"v": [1, 2]}, index=midx)
    gdf = cudf.from_pandas(pdf)
    assert_eq(
        pdf.reset_index(level=level, drop=drop),
        gdf.reset_index(level=level, drop=drop),
    )


@pytest.mark.parametrize("level", [-2, -5])
def test_reset_index_level_underflow_flat(level):
    pdf = pd.DataFrame({"a": [1, 2]}, index=pd.Index([10, 20], name="x"))
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        lfunc=pdf.reset_index,
        rfunc=gdf.reset_index,
        lfunc_args_and_kwargs=([], {"level": level}),
        rfunc_args_and_kwargs=([], {"level": level}),
    )
    with pytest.raises(
        IndexError, match=f"{level} is not a valid level number"
    ):
        gdf.reset_index(level=level)


@pytest.mark.parametrize("level", [-3, -4])
def test_reset_index_level_underflow_multiindex(level):
    midx = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["l0", "l1"])
    pdf = pd.DataFrame({"v": [1, 2]}, index=midx)
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        lfunc=pdf.reset_index,
        rfunc=gdf.reset_index,
        lfunc_args_and_kwargs=([], {"level": level}),
        rfunc_args_and_kwargs=([], {"level": level}),
    )
    with pytest.raises(
        IndexError, match=f"{level} is not a valid level number"
    ):
        gdf.reset_index(level=level)


def test_reset_index_unknown_multiindex_level_name():
    midx = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["l0", "l1"])
    pdf = pd.DataFrame({"v": [1, 2]}, index=midx)
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        lfunc=pdf.reset_index,
        rfunc=gdf.reset_index,
        lfunc_args_and_kwargs=([], {"level": "nope"}),
        rfunc_args_and_kwargs=([], {"level": "nope"}),
    )
    with pytest.raises(KeyError, match="Level nope not found"):
        gdf.reset_index(level="nope")


def test_reset_index_ambiguous_duplicate_level_name():
    midx = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["d", "d"])
    pdf = pd.DataFrame({"v": [1, 2]}, index=midx)
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        lfunc=pdf.reset_index,
        rfunc=gdf.reset_index,
        lfunc_args_and_kwargs=([], {"level": "d"}),
        rfunc_args_and_kwargs=([], {"level": "d"}),
    )
    with pytest.raises(ValueError, match="occurs multiple times"):
        gdf.reset_index(level="d")
    # Duplicate names are still addressable by level number.
    assert_eq(pdf.reset_index(level=0), gdf.reset_index(level=0))
    assert_eq(pdf.reset_index(level=-1), gdf.reset_index(level=-1))

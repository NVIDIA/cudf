# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.api.extensions import no_default
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


@pytest.fixture(params=[True, False])
def drop(request):
    """Param for `drop` argument"""
    return request.param


@pytest.mark.parametrize("level", [None, 0, "l0", 1, ["l0", 1]])
@pytest.mark.parametrize("original_name", [None, "original_ser"])
@pytest.mark.parametrize("name", [None, "ser", no_default])
def test_reset_index(level, drop, inplace, original_name, name):
    if not drop and inplace:
        pytest.skip(
            "For exception checks, see "
            "test_reset_index_dup_level_name_exceptions"
        )

    midx = pd.MultiIndex.from_tuples(
        [("a", 1), ("a", 2), ("b", 1), ("b", 2)], names=["l0", None]
    )
    ps = pd.Series(range(4), index=midx, name=original_name)
    gs = cudf.from_pandas(ps)

    expect = ps.reset_index(level=level, drop=drop, name=name, inplace=inplace)

    got = gs.reset_index(level=level, drop=drop, name=name, inplace=inplace)
    if inplace:
        expect = ps
        got = gs
    if isinstance(expect, pd.DataFrame) and expect.columns.dtype == "object":
        # pandas preserves None as a column label, returning Index[object]
        # which cuDF does not support
        expect.columns = expect.columns.astype(got.columns.dtype)
    assert_eq(expect, got)


@pytest.mark.parametrize("level", [None, 0, 1, [None]])
@pytest.mark.parametrize("original_name", [None, "original_ser"])
@pytest.mark.parametrize("name", [None, "ser"])
def test_reset_index_dup_level_name(level, drop, inplace, original_name, name):
    # midx levels are named [None, None]
    if level == [None] or not drop and inplace:
        pytest.skip(
            "For exception checks, see "
            "test_reset_index_dup_level_name_exceptions"
        )

    midx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    ps = pd.Series(range(4), index=midx, name=original_name)
    gs = cudf.from_pandas(ps)

    expect = ps.reset_index(level=level, drop=drop, inplace=inplace, name=name)
    got = gs.reset_index(level=level, drop=drop, inplace=inplace, name=name)
    if inplace:
        expect = ps
        got = gs
    if isinstance(expect, pd.DataFrame) and expect.columns.dtype == "object":
        # pandas preserves None as a column label, returning Index[object]
        # which cuDF does not support
        expect.columns = expect.columns.astype(got.columns.dtype)

    assert_eq(expect, got)


@pytest.mark.parametrize("original_name", [None, "original_ser"])
@pytest.mark.parametrize("name", [None, "ser"])
def test_reset_index_named(drop, inplace, original_name, name):
    if not drop and inplace:
        pytest.skip(
            "For exception checks, see "
            "test_reset_index_dup_level_name_exceptions"
        )

    ps = pd.Series(range(4), index=["x", "y", "z", "w"], name=original_name)
    gs = cudf.from_pandas(ps)

    ps.index.name = "cudf"
    gs.index.name = "cudf"

    expect = ps.reset_index(drop=drop, inplace=inplace, name=name)
    got = gs.reset_index(drop=drop, inplace=inplace, name=name)

    if inplace:
        expect = ps
        got = gs
    if isinstance(expect, pd.DataFrame) and expect.columns.dtype == "object":
        # pandas preserves None as a column label, returning Index[object]
        # which cuDF does not support
        expect.columns = expect.columns.astype(got.columns.dtype)

    assert_eq(expect, got)


def test_reset_index_dup_level_name_exceptions():
    midx = pd.MultiIndex.from_tuples([("a", 1), ("a", 2), ("b", 1), ("b", 2)])
    ps = pd.Series(range(4), index=midx)
    gs = cudf.from_pandas(ps)

    # Should specify duplicate level names with level number.
    assert_exceptions_equal(
        lfunc=ps.reset_index,
        rfunc=gs.reset_index,
        lfunc_args_and_kwargs=(
            [],
            {"level": [None]},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"level": [None]},
        ),
    )

    # Cannot use drop=False and inplace=True to turn a series into dataframe.
    assert_exceptions_equal(
        lfunc=ps.reset_index,
        rfunc=gs.reset_index,
        lfunc_args_and_kwargs=(
            [],
            {"drop": False, "inplace": True},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"drop": False, "inplace": True},
        ),
    )

    # Pandas raises the above exception should these two inputs crosses.
    assert_exceptions_equal(
        lfunc=ps.reset_index,
        rfunc=gs.reset_index,
        lfunc_args_and_kwargs=(
            [],
            {"level": [None], "drop": False, "inplace": True},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"level": [None], "drop": False, "inplace": True},
        ),
    )


@pytest.mark.parametrize("level", [-1, -2, [-1], [0, -1]])
def test_reset_index_negative_level_multiindex(level, drop):
    # Negative levels count from the end of the MultiIndex.
    midx = pd.MultiIndex.from_tuples([("a", 1), ("b", 2)], names=["l0", "l1"])
    ps = pd.Series([1, 2], index=midx)
    gs = cudf.from_pandas(ps)
    assert_eq(
        ps.reset_index(level=level, drop=drop),
        gs.reset_index(level=level, drop=drop),
    )


@pytest.mark.parametrize("level", [-2, -5])
def test_reset_index_level_underflow(level, drop):
    ps = pd.Series([1, 2], index=pd.Index([10, 20], name="x"))
    gs = cudf.from_pandas(ps)
    assert_exceptions_equal(
        lfunc=ps.reset_index,
        rfunc=gs.reset_index,
        lfunc_args_and_kwargs=([], {"level": level, "drop": drop}),
        rfunc_args_and_kwargs=([], {"level": level, "drop": drop}),
    )

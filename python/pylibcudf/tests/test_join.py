# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable

import numpy as np
import pyarrow as pa
import pytest
from utils import assert_table_eq

import pylibcudf as plc


@pytest.fixture
def left():
    return pa.Table.from_arrays(
        [[0, 1, 2, 100], [3, 4, 5, None]],
        schema=pa.schema({"a": pa.int32(), "b": pa.int32()}),
    )


@pytest.fixture
def right():
    return pa.Table.from_arrays(
        [[-1, -2, 0, 1, -3], [10, 3, 4, 5, None]],
        schema=pa.schema({"c": pa.int32(), "d": pa.int32()}),
    )


@pytest.fixture
def expr():
    return plc.expressions.Operation(
        plc.expressions.ASTOperator.LESS,
        plc.expressions.ColumnReference(
            0, plc.expressions.TableReference.LEFT
        ),
        plc.expressions.ColumnReference(
            0, plc.expressions.TableReference.RIGHT
        ),
    )


def test_cross_join(left, right):
    # Remove the nulls so the calculation of the expected result works
    left = left[:-1]
    right = right[:-1]
    pleft = plc.Table.from_arrow(left)
    pright = plc.Table.from_arrow(right)

    expect = pa.Table.from_arrays(
        [
            *(np.repeat(c.to_numpy(), len(right)) for c in left.columns),
            *(np.tile(c.to_numpy(), len(left)) for c in right.columns),
        ],
        names=["a", "b", "c", "d"],
    )

    got = plc.join.cross_join(pleft, pright)

    assert_table_eq(expect, got)


sentinel = np.iinfo(np.int32).min


@pytest.mark.parametrize(
    "join_type,expect_left,expect_right",
    [
        (plc.join.HashJoin.inner_join, [0, 1], [2, 3]),
        (
            plc.join.HashJoin.left_join,
            [0, 1, 2, 3],
            [sentinel, sentinel, 2, 3],
        ),
        (
            plc.join.HashJoin.full_join,
            [sentinel, sentinel, sentinel, 0, 1, 2, 3],
            [sentinel, sentinel, 0, 1, 2, 3, 4],
        ),
    ],
    ids=["inner", "left", "full"],
)
def test_hash_join(
    left: pa.Table,
    right: pa.Table,
    join_type: Callable[
        [plc.join.HashJoin, plc.Table], tuple[plc.Column, plc.Column]
    ],
    expect_left: list[int | None],
    expect_right: list[int | None],
):
    d_left = plc.Table.from_arrow(left)
    d_right = plc.Table.from_arrow(right)

    joiner = plc.join.HashJoin(
        plc.Table(d_right.columns()[:1]),
        has_nulls=plc.join.NullableJoin.YES,
        compare_nulls=plc.types.NullEquality.EQUAL,
    )

    lg, rg = join_type(joiner, plc.Table(d_left.columns()[:1]))
    got_left = sorted(lg.to_pylist())
    got_right = sorted(rg.to_pylist())
    assert got_left == expect_left
    assert got_right == expect_right


@pytest.mark.parametrize(
    "join_type,expect",
    [
        (plc.join.FilteredJoin.semi_join, [0, 1]),
        (plc.join.FilteredJoin.anti_join, [2, 3]),
    ],
    ids=["semi", "anti"],
)
def test_filtered_join(
    left: pa.Table,
    right: pa.Table,
    join_type: Callable[[plc.join.FilteredJoin, plc.Table], plc.Column],
    expect: list[int | None],
):
    d_left = plc.Table.from_arrow(left)
    d_right = plc.Table.from_arrow(right)

    joiner = plc.join.FilteredJoin(
        plc.Table(d_right.columns()[:1]),
        compare_nulls=plc.types.NullEquality.EQUAL,
    )

    lg = join_type(joiner, plc.Table(d_left.columns()[:1]))
    got = sorted(lg.to_pylist())
    assert got == expect


@pytest.mark.parametrize(
    "join_type,expect_left,expect_right",
    [
        (plc.join.conditional_inner_join, {0}, {3}),
        (plc.join.conditional_left_join, {0, 1, 2, 3}, {3, sentinel}),
        (
            plc.join.conditional_full_join,
            {0, 1, 2, 3, sentinel},
            {0, 1, 2, 3, 4, sentinel},
        ),
    ],
    ids=["inner", "left", "full"],
)
def test_conditional_join(
    left, right, expr, join_type, expect_left, expect_right
):
    pleft = plc.Table.from_arrow(left)
    pright = plc.Table.from_arrow(right)

    g_left, g_right = map(
        lambda self: self.to_arrow(), join_type(pleft, pright, expr)
    )

    assert set(g_left.to_pylist()) == expect_left
    assert set(g_right.to_pylist()) == expect_right


@pytest.mark.parametrize(
    "join_type,expect",
    [
        (plc.join.conditional_left_semi_join, {0}),
        (plc.join.conditional_left_anti_join, {1, 2, 3}),
    ],
    ids=["semi", "anti"],
)
def test_conditional_semianti_join(left, right, expr, join_type, expect):
    pleft = plc.Table.from_arrow(left)
    pright = plc.Table.from_arrow(right)

    g_left = join_type(pleft, pright, expr).to_arrow()

    assert set(g_left.to_pylist()) == expect


@pytest.mark.parametrize(
    "join_type,expect_left,expect_right",
    [
        (plc.join.mixed_inner_join, set(), set()),
        (plc.join.mixed_left_join, {0, 1, 2, 3}, {sentinel}),
        (
            plc.join.mixed_full_join,
            {0, 1, 2, 3, sentinel},
            {0, 1, 2, 3, 4, sentinel},
        ),
    ],
    ids=["inner", "left", "full"],
)
@pytest.mark.parametrize(
    "null_equality",
    [plc.types.NullEquality.EQUAL, plc.types.NullEquality.UNEQUAL],
    ids=["nulls_equal", "nulls_not_equal"],
)
def test_mixed_join(
    left, right, expr, join_type, expect_left, expect_right, null_equality
):
    pleft = plc.Table.from_arrow(left)
    pright = plc.Table.from_arrow(right)

    g_left, g_right = map(
        lambda self: self.to_arrow(),
        join_type(
            plc.Table(pleft.columns()[1:]),
            plc.Table(pright.columns()[1:]),
            pleft,
            pright,
            expr,
            null_equality,
        ),
    )

    assert set(g_left.to_pylist()) == expect_left
    assert set(g_right.to_pylist()) == expect_right


@pytest.mark.parametrize(
    "join_type,expect",
    [
        (plc.join.mixed_left_semi_join, set()),
        (plc.join.mixed_left_anti_join, {0, 1, 2, 3}),
    ],
    ids=["semi", "anti"],
)
@pytest.mark.parametrize(
    "null_equality",
    [plc.types.NullEquality.EQUAL, plc.types.NullEquality.UNEQUAL],
    ids=["nulls_equal", "nulls_not_equal"],
)
def test_mixed_semianti_join(
    left, right, expr, join_type, expect, null_equality
):
    pleft = plc.Table.from_arrow(left)
    pright = plc.Table.from_arrow(right)

    g_left = join_type(
        plc.Table(pleft.columns()[1:]),
        plc.Table(pright.columns()[1:]),
        pleft,
        pright,
        expr,
        null_equality,
    ).to_arrow()

    assert set(g_left.to_pylist()) == expect

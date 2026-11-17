# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import weakref

import numpy as np
import pyarrow as pa
import pytest
from utils import assert_table_eq

from rmm.pylibrmm.memory_resource import CudaMemoryResource
from rmm.pylibrmm.stream import Stream

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


@pytest.mark.uses_custom_stream
def test_asof_join():
    left_on = plc.Column.from_arrow(
        pa.array([4, 0, 3, None, 1], type=pa.int64())
    )
    right_on = plc.Column.from_arrow(pa.array([1, 3, 3, 7], type=pa.int64()))
    joiner = plc.join.AsofJoin(plc.Table([]), right_on)
    del right_on
    assert joiner.join(plc.Table([]), left_on).to_arrow().to_pylist() == [
        2,
        sentinel,
        2,
        sentinel,
        0,
    ]
    probe_stream = Stream()
    other_left = plc.Column.from_arrow(
        pa.array([2, 8], type=pa.int64()), stream=probe_stream
    )
    result = joiner.join(
        plc.Table([]),
        other_left,
        strategy=plc.join.AsofJoinStrategy.BACKWARD,
        exact_matches=plc.join.AsofJoinExactMatches.YES,
        stream=probe_stream,
    )
    del joiner
    assert result.to_arrow(stream=probe_stream).to_pylist() == [0, 3]


def test_asof_join_grouped():
    left_by = plc.Table([plc.Column.from_arrow(pa.array(["a", "b", "a"]))])
    left_on = plc.Column.from_arrow(pa.array([2, 2, 4]))
    right_by = plc.Table([plc.Column.from_arrow(pa.array(["a", "a", "b"]))])
    right_on = plc.Column.from_arrow(pa.array([1, 3, 1]))

    result = plc.join.AsofJoin(right_by, right_on).join(left_by, left_on)
    assert result.to_arrow().to_pylist() == [0, 2, 1]


@pytest.mark.uses_custom_stream
def test_asof_join_right_key_lifetime_across_streams():
    input_stream, build_stream, probe_stream = Stream(), Stream(), Stream()
    right_on = plc.Column.from_arrow(
        pa.array([1, 3], type=pa.int64()), stream=input_stream
    )
    plc.experimental.join_streams([input_stream], build_stream)
    joiner = plc.join.AsofJoin(plc.Table([]), right_on, stream=build_stream)
    left_on = plc.Column.from_arrow(
        pa.array([2], type=pa.int64()), stream=probe_stream
    )
    result = joiner.join(plc.Table([]), left_on, stream=probe_stream)
    del right_on, joiner
    assert result.to_arrow(stream=probe_stream).to_pylist() == [0]


@pytest.mark.parametrize(
    "left_values,right_values,expected",
    [
        ([1, 2], [], [sentinel, sentinel]),
        ([], [1, 2], []),
        ([1, 2], [None, None], [sentinel, sentinel]),
    ],
)
def test_asof_join_empty_or_null(left_values, right_values, expected):
    left_on = plc.Column.from_arrow(pa.array(left_values, type=pa.int64()))
    right_on = plc.Column.from_arrow(pa.array(right_values, type=pa.int64()))
    result = plc.join.AsofJoin(plc.Table([]), right_on).join(
        plc.Table([]), left_on
    )
    assert result.to_arrow().to_pylist() == expected


@pytest.fixture(
    params=[plc.types.NullEquality.EQUAL, plc.types.NullEquality.UNEQUAL]
)
def compare_nulls(request):
    return request.param


@pytest.fixture(
    params=[
        pytest.param(
            ([0, 1, 2, 100], [-1, -2, 0, 1, -3], [(0, 2), (1, 3)], []),
            id="no_nulls",
        ),
        pytest.param(
            (
                [0, None, 2, None],
                [None, 2, 0, None],
                [(0, 2), (2, 1)],
                [(1, 0), (1, 3), (3, 0), (3, 3)],
            ),
            id="nulls_both_sides",
        ),
        pytest.param(
            ([None, None], [None], [], [(0, 0), (1, 0)]),
            id="all_nulls",
        ),
        pytest.param(
            ([0, None], [0, 1], [(0, 0)], []),
            id="probe_nulls",
        ),
        pytest.param(
            ([0, 1], [0, None], [(0, 0)], []),
            id="build_nulls",
        ),
    ]
)
def join_keys(request, compare_nulls):
    left, right, matches, null_matches = request.param
    if compare_nulls == plc.types.NullEquality.EQUAL:
        matches = matches + null_matches
    return (
        plc.Table([plc.Column.from_arrow(pa.array(left, type=pa.int32()))]),
        plc.Table([plc.Column.from_arrow(pa.array(right, type=pa.int32()))]),
        matches,
    )


@pytest.mark.parametrize("join_type", ["inner", "left", "full"])
def test_hash_join(join_keys, compare_nulls, join_type):
    left, right, matches = join_keys
    joiner = plc.join.HashJoin(right, compare_nulls=compare_nulls)
    lg, rg = getattr(joiner, f"{join_type}_join")(left)

    expected = matches.copy()
    if join_type in {"left", "full"}:
        matched_left = {i for i, _ in matches}
        expected.extend(
            (i, sentinel)
            for i in range(left.num_rows())
            if i not in matched_left
        )
    if join_type == "full":
        matched_right = {j for _, j in matches}
        expected.extend(
            (sentinel, j)
            for j in range(right.num_rows())
            if j not in matched_right
        )
    assert sorted(zip(lg.to_pylist(), rg.to_pylist(), strict=True)) == sorted(
        expected
    )


@pytest.mark.parametrize("join_type", ["semi", "anti"])
def test_filtered_join(join_keys, compare_nulls, join_type):
    left, right, matches = join_keys
    joiner = plc.join.FilteredJoin(right, compare_nulls=compare_nulls)
    result = getattr(joiner, f"{join_type}_join")(left)
    matched_left = {i for i, _ in matches}
    expected = (
        matched_left
        if join_type == "semi"
        else set(range(left.num_rows())) - matched_left
    )
    assert sorted(result.to_pylist()) == sorted(expected)


@pytest.mark.parametrize("join_type", ["inner", "left", "full"])
def test_hash_join_non_nullable(left, right, join_type):
    left = plc.Table.from_arrow(left.select(["a"]))
    right = plc.Table.from_arrow(right.select(["c"]))
    nullable = plc.join.HashJoin(right, has_nulls=plc.join.NullableJoin.YES)
    non_nullable = plc.join.HashJoin(right, has_nulls=plc.join.NullableJoin.NO)
    expected = getattr(nullable, f"{join_type}_join")(left)
    result = getattr(non_nullable, f"{join_type}_join")(left)
    assert sorted(
        zip(*(col.to_pylist() for col in result), strict=True)
    ) == sorted(zip(*(col.to_pylist() for col in expected), strict=True))


@pytest.mark.parametrize(
    "join_class", [plc.join.HashJoin, plc.join.FilteredJoin]
)
def test_join_owns_build_inputs(join_class):
    class Resource(CudaMemoryResource):
        pass

    right = plc.Table([plc.Column.from_arrow(pa.array([1, 2]))])
    mr = Resource()
    mr_ref = weakref.ref(mr)
    joiner = join_class(right, mr=mr)
    del right, mr
    # Probing and destruction must remain safe after the caller drops its owners.
    assert mr_ref() is not None
    left = plc.Table([plc.Column.from_arrow(pa.array([2, 3]))])
    if join_class is plc.join.HashJoin:
        lg, rg = joiner.inner_join(left)
        assert list(zip(lg.to_pylist(), rg.to_pylist(), strict=True)) == [
            (0, 1)
        ]
    else:
        assert joiner.semi_join(left).to_pylist() == [0]
    del joiner
    assert mr_ref() is None


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

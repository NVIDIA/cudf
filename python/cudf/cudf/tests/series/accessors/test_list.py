# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import operator
import re

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.api.types import is_scalar
from cudf.testing import assert_eq
from cudf.utils.dtypes import cudf_dtype_to_pa_type


@pytest.mark.parametrize(
    "data",
    [
        [[]],
        [[[]]],
        [[0]],
        [[0, 1]],
        [[0, 1], [2, 3]],
        [[[0, 1], [2]], [[3, 4]]],
        [[[0, 1, None], None], None, [[3, 2, None], None]],
        [[["a", "c", None], None], None, [["b", "d", None], None]],
    ],
)
def test_leaves(data):
    pa_array = pa.array(data)
    while hasattr(pa_array, "flatten"):
        pa_array = pa_array.flatten()

    expect = cudf.Series(pa_array)
    got = cudf.Series(data).list.leaves
    assert_eq(
        expect,
        got,
        check_dtype=not isinstance(pa_array, pa.NullArray),
    )


@pytest.mark.parametrize(
    "data",
    [
        [[]],
        [[1, 2, 3], [4, 5]],
        [[1, 2, 3], [], [4, 5]],
        [[1, 2, 3], None, [4, 5]],
        [[None, None], [None]],
        [[[[[[1, 2, 3]]]]]],
        cudf.Series([[1, 2]]).iloc[0:0],
        cudf.Series([None, [1, 2]]).iloc[0:1],
    ],
)
def test_len(data):
    gsr = cudf.Series(data)
    psr = gsr.to_pandas(arrow_type=True)
    if pa.types.is_null(psr.dtype.pyarrow_dtype):
        psr = psr.astype(pd.ArrowDtype(gsr.dtype.to_arrow()))
    expect = psr.list.len()
    got = gsr.list.len().to_pandas(arrow_type=True)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    ("data", "idx"),
    [
        ([[1, 2, 3], [3, 4, 5], [4, 5, 6]], [[0, 1], [2], [1, 2]]),
        ([[1, 2, 3], [3, 4, 5], [4, 5, 6]], [[1, 2, 0], [1, 0, 2], [0, 1, 2]]),
        ([[1, 2, 3], []], [[0, 1], []]),
        ([[1, 2, 3], [None]], [[0, 1], []]),
        ([[1, None, 3], None], [[0, 1], []]),
    ],
)
def test_take(data, idx):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.Series(zip(ps, idx, strict=True)).map(
        lambda x: [x[0][i] for i in x[1]] if x[0] is not None else None
    )
    got = gs.list.take(idx)
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "invalid",
    [
        [[0]],
        [1, 2, 3, 4],
        [["a", "b"], ["c"]],
        [[[1], [0]], [[0]]],
        [[0, 1], None],
    ],
)
def test_take_invalid(invalid):
    gs = cudf.Series([[0, 1], [2, 3]])
    with pytest.raises(ValueError, match="input must be list type"):
        gs.list.take(invalid)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ([[1, 1, 2, 2], [], None, [3, 4, 5]], [[1, 2], [], None, [3, 4, 5]]),
        (
            [[1.233, np.nan, 1.234, 3.141, np.nan, 1.234]],
            [[1.233, 1.234, np.nan, 3.141]],
        ),  # duplicate nans
        ([[1, 1, 2, 2, None, None]], [[1, 2, None]]),  # duplicate nulls
        (
            [[1.233, np.nan, None, 1.234, 3.141, np.nan, 1.234, None]],
            [[1.233, 1.234, np.nan, None, 3.141]],
        ),  # duplicate nans and nulls
        ([[2, None, 1, None, 2]], [[1, 2, None]]),
        ([[], []], [[], []]),
        ([[], None], [[], None]),
    ],
)
def test_unique(data, expected):
    """
    Pandas de-duplicates nans and nulls respectively in Series.unique.
    `expected` is setup to mimic such behavior
    """
    gs = cudf.Series(data, nan_as_null=False)

    got = gs.list.unique().list.sort_values()
    expected = cudf.Series(expected, nan_as_null=False).list.sort_values()

    assert_eq(expected, got)


def key_func_builder(x, na_position):
    return x if x is not None else -1e8 if na_position == "first" else 1e8


@pytest.mark.parametrize(
    "data",
    [
        [[4, 2, None, 9], [8, 8, 2], [2, 1]],
        [[4, 2, None, 9], [8, 8, 2], None],
        [[4, 2, None, 9], [], None],
    ],
)
@pytest.mark.parametrize(
    "index",
    [
        None,
        pd.Index(["a", "b", "c"]),
        pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (1, "a")], names=["l0", "l1"]
        ),
    ],
)
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
@pytest.mark.parametrize("ignore_index", [True, False])
def test_sort_values(data, index, ascending, na_position, ignore_index):
    key_func = functools.partial(key_func_builder, na_position=na_position)

    ps = pd.Series(data, index=index)
    gs = cudf.from_pandas(ps)

    expected = ps.apply(
        lambda x: (
            sorted(x, key=key_func, reverse=not ascending)
            if x is not None
            else None
        )
    )
    if ignore_index:
        expected.reset_index(drop=True, inplace=True)
    got = gs.list.sort_values(
        ascending=ascending, na_position=na_position, ignore_index=ignore_index
    )

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data, index, expect",
    [
        ([[None, None], [None, None]], 0, [None, None]),
        ([[1, 2], [3, 4]], 0, [1, 3]),
        ([["a", "b"], ["c", "d"]], 1, ["b", "d"]),
        ([[1, None], [None, 2]], 1, [None, 2]),
        ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], 1, [[3, 4], [7, 8]]),
    ],
)
def test_get(data, index, expect):
    sr = cudf.Series(data)
    expect = cudf.Series(expect)
    got = sr.list.get(index)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [{"k": "v1"}, {"k": "v2"}],
        [[{"k": "v1", "b": "v2"}], [{"k": "v3", "b": "v4"}]],
        [
            [{"k": "v1", "b": [{"c": 10, "d": "v5"}]}],
            [{"k": "v3", "b": [{"c": 14, "d": "v6"}]}],
        ],
    ],
)
@pytest.mark.parametrize("index", [0, 1])
def test_get_nested_struct_dtype_transfer(data, index):
    sr = cudf.Series([data])
    expect = cudf.Series(data[index : index + 1])
    assert_eq(expect, sr.list.get(index))


def test_get_nested_lists():
    sr = cudf.Series(
        [
            [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [], [[3, 4], [7, 8]]],
            [[], [[9, 10]], [[11, 12], [13, 14]]],
        ]
    )
    expect = cudf.Series([[[1, 2], [3, 4]], []])
    got = sr.list.get(0)
    assert_eq(expect, got)


def test_get_default():
    sr = cudf.Series([[1, 2], [3, 4, 5], [6, 7, 8, 9]])

    assert_eq(cudf.Series([cudf.NA, 5, 8]), sr.list.get(2))
    assert_eq(cudf.Series([cudf.NA, 5, 8]), sr.list.get(2, default=cudf.NA))
    assert_eq(cudf.Series([0, 5, 8]), sr.list.get(2, default=0))
    assert_eq(cudf.Series([0, 3, 7]), sr.list.get(-3, default=0))
    assert_eq(cudf.Series([2, 5, 9]), sr.list.get(-1))

    string_sr = cudf.Series(
        [["apple", "banana"], ["carrot", "daffodil", "elephant"]]
    )
    assert_eq(
        cudf.Series(["default", "elephant"]),
        string_sr.list.get(2, default="default"),
    )

    sr_with_null = cudf.Series([[0, cudf.NA], [1]])
    assert_eq(cudf.Series([cudf.NA, 0]), sr_with_null.list.get(1, default=0))

    sr_nested = cudf.Series([[[1, 2], [3, 4], [5, 6]], [[5, 6], [7, 8]]])
    assert_eq(cudf.Series([[3, 4], [7, 8]]), sr_nested.list.get(1))
    assert_eq(cudf.Series([[5, 6], cudf.NA]), sr_nested.list.get(2))
    assert_eq(
        cudf.Series([[5, 6], [0, 0]]), sr_nested.list.get(2, default=[0, 0])
    )


def test_get_ind_sequence():
    # test .list.get() when `index` is a sequence
    sr = cudf.Series([[1, 2], [3, 4, 5], [6, 7, 8, 9]])
    assert_eq(cudf.Series([1, 4, 8]), sr.list.get([0, 1, 2]))
    assert_eq(cudf.Series([1, 4, 8]), sr.list.get(cudf.Series([0, 1, 2])))
    assert_eq(cudf.Series([cudf.NA, 5, cudf.NA]), sr.list.get([2, 2, -5]))
    assert_eq(cudf.Series([0, 5, 0]), sr.list.get([2, 2, -5], default=0))
    sr_nested = cudf.Series([[[1, 2], [3, 4], [5, 6]], [[5, 6], [7, 8]]])
    assert_eq(cudf.Series([[1, 2], [7, 8]]), sr_nested.list.get([0, 1]))


@pytest.mark.parametrize(
    "data, scalar, expect",
    [
        (
            [[1, 2, 3], []],
            1,
            [True, False],
        ),
        (
            [[1, 2, 3], [], [3, 4, 5]],
            6,
            [False, False, False],
        ),
        (
            [[1.0, 2.0, 3.0], None, []],
            2.0,
            [True, None, False],
        ),
        (
            [[None, "b", "c"], [], ["b", "e", "f"]],
            "b",
            [True, False, True],
        ),
        ([[None, 2, 3], None, []], 1, [False, None, False]),
        (
            [[None, "b", "c"], [], ["b", "e", "f"]],
            "d",
            [False, False, False],
        ),
    ],
)
def test_contains_scalar(data, scalar, expect):
    sr = cudf.Series(data)
    expect = cudf.Series(expect)
    got = sr.list.contains(
        pa.scalar(scalar, type=cudf_dtype_to_pa_type(sr.dtype.element_type))
    )
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data, expect",
    [
        (
            [[1, 2, 3], []],
            [None, None],
        ),
        (
            [[1.0, 2.0, 3.0], None, []],
            [None, None, None],
        ),
        (
            [[None, 2, 3], [], None],
            [None, None, None],
        ),
        (
            [[1, 2, 3], [3, 4, 5]],
            [None, None],
        ),
        (
            [[], [], []],
            [None, None, None],
        ),
    ],
)
def test_contains_null_search_key(data, expect):
    sr = cudf.Series(data)
    expect = cudf.Series(expect, dtype="bool")
    got = sr.list.contains(
        pa.scalar(None, type=cudf_dtype_to_pa_type(sr.dtype.element_type))
    )
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data, scalar",
    [
        (
            [[9, 0, 2], [], [1, None, 0]],
            "x",
        ),
        (
            [["z", "y", None], None, [None, "x"]],
            5,
        ),
    ],
)
def test_contains_invalid(data, scalar):
    sr = cudf.Series(data)
    with pytest.raises(
        TypeError,
        match=r"Type/Scale of search key does not "
        "match list column element type.",
    ):
        sr.list.contains(scalar)


@pytest.mark.parametrize(
    "data, search_key, expect",
    [
        (
            [[1, 2, 3], [], [3, 4, 5]],
            3,
            [2, -1, 0],
        ),
        (
            [[1.0, 2.0, 3.0], None, [2.0, 5.0]],
            2.0,
            [1, None, 0],
        ),
        (
            [[None, "b", "c"], [], ["b", "e", "f"]],
            "f",
            [-1, -1, 2],
        ),
        ([[-5, None, 8], None, []], -5, [0, None, -1]),
        (
            [[None, "x", None, "y"], ["z", "i", "j"]],
            "y",
            [3, -1],
        ),
        (
            [["h", "a", None], ["t", "g"]],
            ["a", "b"],
            [1, -1],
        ),
        (
            [None, ["h", "i"], ["p", "k", "z"]],
            ["x", None, "z"],
            [None, None, 2],
        ),
        (
            [["d", None, "e"], [None, "f"], []],
            pa.scalar(None, type=pa.string()),
            [None, None, None],
        ),
        (
            [None, [10, 9, 8], [5, 8, None]],
            pa.scalar(None, type=pa.int64()),
            [None, None, None],
        ),
    ],
)
def test_index(data, search_key, expect):
    sr = cudf.Series(data)
    expect = cudf.Series(expect, dtype="int32")
    if is_scalar(search_key):
        got = sr.list.index(
            pa.scalar(
                search_key, type=cudf_dtype_to_pa_type(sr.dtype.element_type)
            )
        )
    else:
        got = sr.list.index(
            cudf.Series(search_key, dtype=sr.dtype.element_type)
        )

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data, search_key",
    [
        (
            [[9, None, 8], [], [7, 6, 5]],
            "c",
        ),
        (
            [["a", "b", "c"], None, [None, "d"]],
            2,
        ),
        (
            [["e", "s"], ["t", "w"]],
            [5, 6],
        ),
    ],
)
def test_index_invalid_type(data, search_key):
    sr = cudf.Series(data)
    with pytest.raises(
        TypeError,
        match=r"Type/Scale of search key does not "
        "match list column element type.",
    ):
        sr.list.index(search_key)


@pytest.mark.parametrize(
    "data, search_key",
    [
        (
            [[5, 8], [2, 6]],
            [8, 2, 4],
        ),
        (
            [["h", "j"], ["p", None], ["t", "z"]],
            ["j", "a"],
        ),
    ],
)
def test_index_invalid_length(data, search_key):
    sr = cudf.Series(data)
    with pytest.raises(
        RuntimeError,
        match=r"Number of search keys must match list column size.",
    ):
        sr.list.index(search_key)


@pytest.mark.parametrize(
    "row",
    [
        [[]],
        [[1]],
        [[1, 2]],
        [[1, 2], [3, 4, 5]],
        [[1, 2], [], [3, 4, 5]],
        [[1, 2, None], [3, 4, 5]],
        [[1, 2, None], None, [3, 4, 5]],
        [[1, 2, None], None, [], [3, 4, 5]],
        [[[1, 2], [3, 4]], [[5, 6, 7], [8, 9]]],
        [[["a", "c", "de", None], None, ["fg"]], [["abc", "de"], None]],
    ],
)
@pytest.mark.parametrize("dropna", [True, False])
def test_concat_elements(row, dropna):
    if any(x is None for x in row):
        if dropna:
            row = [x for x in row if x is not None]
            result = functools.reduce(operator.add, row)
        else:
            result = None
    else:
        result = functools.reduce(operator.add, row)

    expect = pd.Series([result])
    got = cudf.Series([row]).list.concat(dropna=dropna)
    assert_eq(expect, got)


def test_concat_elements_raise():
    s = cudf.Series([[1, 2, 3]])  # no nesting
    with pytest.raises(
        ValueError,
        match=r".*Child of the input lists column must also be a lists column",
    ):
        s.list.concat()


def test_list_iterate_error():
    s = cudf.Series([[[[1, 2]], [[2], [3]]], [[[2]]], [[[3]]]])
    with pytest.raises(TypeError, match="ListMethods object is not iterable"):
        iter(s.list)


def test_list_methods_setattr():
    ser = cudf.Series([["a", "b", "c"], ["d", "e", "f"]])

    with pytest.raises(AttributeError):
        ser.list.a = "b"


def test_lists_contains(numeric_types_as_str):
    inner_data = np.array([1, 2, 3], dtype=numeric_types_as_str)

    data = cudf.Series([inner_data])

    contained_scalar = inner_data.dtype.type(2)
    not_contained_scalar = inner_data.dtype.type(42)

    assert data.list.contains(contained_scalar)[0]
    assert not data.list.contains(not_contained_scalar)[0]


def test_lists_contains_datetime(temporal_types_as_str):
    inner_data = np.array([1, 2, 3], dtype=temporal_types_as_str)

    unit, _ = np.datetime_data(inner_data.dtype)

    data = cudf.Series([inner_data])

    contained_scalar = inner_data.dtype.type(2, unit)
    not_contained_scalar = inner_data.dtype.type(42, unit)

    assert data.list.contains(contained_scalar)[0]
    assert not data.list.contains(not_contained_scalar)[0]


def test_lists_contains_bool():
    data = cudf.Series([[True, True, True]])

    assert data.list.contains(True)[0]
    assert not data.list.contains(False)[0]


def test_list_accessor_invalid_dtype_message():
    # The .list accessor error message names the required 'list[pyarrow]'
    # dtype and reports the offending dtype, matching pandas.
    gs = cudf.Series([1, 2, 3])
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "Can only use the '.list' accessor with 'list[pyarrow]' dtype, "
            "not int64."
        ),
    ):
        gs.list

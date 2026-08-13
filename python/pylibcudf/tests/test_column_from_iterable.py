# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import array
import operator
from functools import reduce

import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc

DTYPES = [
    ("int", int, pa.int64()),
    ("float", float, pa.float64()),
    ("bool", bool, pa.bool_()),
]

SHAPES = [
    (6,),
    (2, 3),
    (2, 2, 3),
    (2, 2, 2, 3),
]


@pytest.fixture(params=SHAPES, ids=lambda x: f"shape={x}")
def shape(request):
    return request.param


@pytest.fixture(params=DTYPES, ids=lambda x: x[0])
def dtype_info(request):
    return request.param


@pytest.fixture
def dummy_large_string_type():
    class Bar(bytes):
        def __len__(self):
            return 2**31 + 1

    class Foo(str):
        def encode(self):
            return Bar(b"x")

    return Foo


def generate_list_data(shape, dtype):
    def gen_values(n):
        if dtype is bool:
            return [(i % 2) == 0 for i in range(n)]
        else:
            return [dtype(i) for i in range(n)]

    def nest(values, shape):
        if not shape:
            return values[0]
        size = len(values) // shape[0]
        return [
            nest(values[i * size : (i + 1) * size], shape[1:])
            for i in range(shape[0])
        ]

    return nest(gen_values(reduce(operator.mul, shape)), shape)


def test_from_list(shape, dtype_info):
    _, py_type, pa_type = dtype_info
    data = generate_list_data(shape, py_type)
    expect_type = pa_type
    for _ in range(len(shape) - 1):
        expect_type = pa.list_(expect_type)

    expect = pa.array(data, type=expect_type)

    got = plc.Column.from_iterable_of_py(
        data, dtype=plc.DataType.from_arrow(pa_type)
    )

    assert_column_eq(expect, got)


def test_from_list_empty_without_dtype_raises():
    with pytest.raises(
        ValueError, match="Cannot infer dtype from empty iterable object"
    ):
        plc.Column.from_iterable_of_py([])


def test_from_list_irregular_shapes_raises():
    data = [[1], [2, 3], [], [4]]
    with pytest.raises(ValueError, match="Inconsistent inner list shapes"):
        plc.Column.from_iterable_of_py(data)


def test_from_list_nested_dicts_raises():
    data = [[{"a": 1}], [{"b": 2}]]
    with pytest.raises(TypeError, match="Unsupported scalar type"):
        plc.Column.from_iterable_of_py(data)


def test_from_list_nested_none_raises():
    data = [[None], [None]]
    with pytest.raises(TypeError, match="Unsupported scalar type"):
        plc.Column.from_iterable_of_py(data)


def test_from_zero_dimensional_list_with_dtype():
    got = plc.Column.from_iterable_of_py(
        [], dtype=plc.DataType(plc.TypeId.STRING)
    )
    expect = pa.array([], type=pa.string())
    assert_column_eq(expect, got)


@pytest.mark.parametrize(
    "factory, pa_type",
    [
        (lambda: range(5), pa.int64()),
        (lambda: (x for x in range(5)), pa.float32()),
        (lambda: array.array("q", range(5)), pa.int64()),
        (lambda: filter(lambda x: x % 2 == 0, range(10)), pa.int16()),
        (lambda: map(int, [1.1, 2.2, 3.3]), pa.int64()),
        (lambda: reversed([1, 2, 3, 4]), pa.int64()),
        (lambda: tuple(range(4)), pa.int32()),
    ],
)
def test_from_builtin_iterables(factory, pa_type):
    materialized = list(factory())
    got = plc.Column.from_iterable_of_py(
        iter(materialized), dtype=plc.DataType.from_arrow(pa_type)
    )
    expect = pa.array(materialized, type=pa_type)
    assert_column_eq(expect, got)


def test_from_custom_iterable():
    class Foo:
        def __init__(self, values):
            self.values = values

        def __iter__(self):
            return iter(self.values)

    list_data = [1, 2, 3]
    got = plc.Column.from_iterable_of_py(
        Foo(list_data), dtype=plc.DataType(plc.TypeId.INT64)
    )
    expect = pa.array(list_data, type=pa.int64())
    assert_column_eq(expect, got)


def test_from_custom_generator():
    class Foo:
        def __init__(self, stop):
            self.i = 0
            self.stop = stop

        def __iter__(self):
            return self

        def __next__(self):
            if self.i < self.stop:
                val = self.i
                self.i += 1
                return val
            raise StopIteration

    got = plc.Column.from_iterable_of_py(
        Foo(4), dtype=plc.DataType(plc.TypeId.INT64)
    )
    expect = pa.array([0, 1, 2, 3], type=pa.int64())
    assert_column_eq(expect, got)


@pytest.mark.parametrize(
    "data, pa_type",
    [
        (["a", "b", "c"], pa.string()),
        ([["a"], ["b"], ["c"]], pa.list_(pa.string())),
        ([["foo"], ["bar"], ["baz"]], pa.list_(pa.string())),
        ([["a", "b"], ["c", "d"]], pa.list_(pa.string())),
        ([["a", "b", "c"], ["d", "e", "f"]], pa.list_(pa.string())),
        ([[["a"]], [["b"]], [["c"]]], pa.list_(pa.list_(pa.string()))),
        (lambda: map(str, ("x", "y", "z")), pa.string()),
    ],
)
def test_from_iterable_of_str(data, pa_type):
    if callable(data):
        data = list(data())
    got = plc.Column.from_iterable_of_py(data)
    expect = pa.array(data, type=pa_type)
    assert_column_eq(expect, got)


def test_from_list_of_large_strings(dummy_large_string_type):
    Foo = dummy_large_string_type
    data = [Foo(), Foo()]
    col = plc.Column.from_iterable_of_py(
        data, dtype=plc.DataType(plc.TypeId.STRING)
    )

    assert col.type().id() == plc.TypeId.STRING
    assert col.children()[0].type().id() == plc.TypeId.INT64


def test_from_nested_list_of_large_strings(dummy_large_string_type):
    Foo = dummy_large_string_type
    nested_data = [[Foo(), Foo()], [Foo(), Foo()]]
    col = plc.Column.from_iterable_of_py(
        nested_data, dtype=plc.DataType(plc.TypeId.STRING)
    )

    assert col.type().id() == plc.TypeId.LIST
    assert col.children()[1].type().id() == plc.TypeId.STRING
    assert col.children()[1].children()[0].type().id() == plc.TypeId.INT64


@pytest.mark.parametrize(
    "arr",
    [pa.array([1, 2], type=pa.int32()), np.array([1, 2], dtype=np.int32)],
)
def test_from_iterable_pyarrow_or_numpy_array(arr):
    with pytest.raises(TypeError, match="Please call"):
        plc.Column.from_iterable_of_py(arr)


def test_from_iterable_plc_column():
    with pytest.raises(ValueError, match="is not iterable"):
        plc.Column.from_iterable_of_py(plc.Column.from_iterable_of_py([1]))


@pytest.mark.parametrize(
    "arr",
    [
        pa.array([-1, 2, 3], type=pa.int64()),
        pa.array([2, None, 3, 1, None, 5, None, 7, 8, None], type=pa.int16()),
        pa.array([1, None, 3], type=pa.int32()),
        pa.array([1, 2, 3], type=pa.uint8()),
        pa.array([1, None, 3], type=pa.uint32()),
        pa.array([1.25, 4.5, 7.75], type=pa.float32()),
        pa.array([1.25, None, 7.75], type=pa.float64()),
        pa.array([True, False, True], type=pa.bool_()),
        pa.array([True, None, False, True, None, False], type=pa.bool_()),
        pa.array(["fóó", "bår", "bäz"], type=pa.string()),
        pa.array([], type=pa.string()),
        pa.array(["hello", None, "wörld", "🚀"], type=pa.large_string()),
    ],
)
def test_to_pylist(arr):
    col = plc.Column.from_arrow(arr)
    got = col.to_pylist()
    expect = arr.to_pylist()
    assert got == expect


def test_to_pylist_unsupported():
    with pytest.raises(
        NotImplementedError,
        match="Column with dtype",
    ):
        plc.Column.from_iterable_of_py(
            [[1, 2], [3, 4]], dtype=plc.DataType(plc.TypeId.INT32)
        ).to_pylist()

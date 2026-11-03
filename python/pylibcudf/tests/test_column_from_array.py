# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc

np = pytest.importorskip("numpy")
cp = pytest.importorskip("cupy")

SIZE_TYPE_LIMIT = 2**31 - 1

CUPY_DTYPES = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
    np.bool_,
]

NUMPY_DTYPES = [
    *CUPY_DTYPES,
    *[
        np.dtype("datetime64[s]"),
        np.dtype("datetime64[ms]"),
        np.dtype("datetime64[us]"),
        np.dtype("datetime64[ns]"),
        np.dtype("timedelta64[s]"),
        np.dtype("timedelta64[ms]"),
        np.dtype("timedelta64[us]"),
        np.dtype("timedelta64[ns]"),
    ],
]


@pytest.fixture(
    params=[(6,), (2, 3), (2, 2, 3), (2, 2, 2, 3)],
    ids=lambda x: f"ndim={len(x)}",
)
def shape(request):
    return request.param


@pytest.fixture(params=CUPY_DTYPES, ids=repr)
def cp_array(request, shape, patch_cupy_stream):
    dtype = request.param
    size = np.prod(shape)
    if dtype == np.bool_:
        np_arr = np.array(
            [True, False] * ((size + 1) // 2), dtype=dtype
        ).reshape(shape)
    else:
        np_arr = np.arange(size, dtype=dtype).reshape(shape)
    with patch_cupy_stream:
        return cp.asarray(np_arr), np_arr


@pytest.fixture(params=NUMPY_DTYPES, ids=repr)
def np_array(request, shape):
    dtype = request.param
    size = np.prod(shape)
    if dtype == np.bool_:
        arr = np.array([True, False] * ((size + 1) // 2), dtype=dtype).reshape(
            shape
        )
    elif np.issubdtype(dtype, np.datetime64):
        unit = np.datetime_data(dtype)[0]
        start = np.datetime64("2000-01-01", unit)
        step = np.timedelta64(1, unit)
        arr = np.arange(start, start + size * step, step, dtype=dtype).reshape(
            shape
        )
    else:
        arr = np.arange(size, dtype=dtype).reshape(shape)
    return arr


def test_from_cupy_array(cp_array):
    cp_arr, np_arr = cp_array
    arrow_type = pa.from_numpy_dtype(np_arr.dtype)
    for _ in range(len(np_arr.shape) - 1):
        arrow_type = pa.list_(arrow_type)
    expected = pa.array(np_arr.tolist(), type=arrow_type)

    got = plc.Column.from_array(cp_arr)
    assert_column_eq(expected, got)


def test_from_numpy_array(np_array):
    arr = np_array
    arrow_type = pa.from_numpy_dtype(arr.dtype)
    for _ in range(len(arr.shape) - 1):
        arrow_type = pa.list_(arrow_type)
    expected = pa.array(arr.tolist(), type=arrow_type)

    got = plc.Column.from_array(arr)
    assert_column_eq(expected, got)


def test_non_c_contiguous_raises(cp_array, patch_cupy_stream):
    cp_arr = cp_array[0]
    if len(cp_arr.shape) == 1:
        return

    with patch_cupy_stream:
        fortran_arr = cp.asfortranarray(cp_arr)
    with pytest.raises(
        ValueError,
        match="Data must be C-contiguous",
    ):
        plc.Column.from_array(fortran_arr)


def test_row_limit_exceed_raises():
    class Foo:
        def __init__(self, shape):
            self.shape = shape
            self.__cuda_array_interface__ = {
                "shape": shape,
                "typestr": "<f8",
                "data": (0, False),
                "version": 3,
            }

    with pytest.raises(
        ValueError,
        match="Number of rows exceeds size_type limit",
    ):
        plc.Column.from_array(Foo((SIZE_TYPE_LIMIT, 1)))


def test_flat_size_exceeds_size_type_limit(patch_cupy_stream):
    nrows = 2**16
    ncols = (SIZE_TYPE_LIMIT // nrows) + 1

    with patch_cupy_stream:
        arr = cp.zeros((nrows, ncols), dtype=np.int32)

    with pytest.raises(
        ValueError,
        match="Flat size exceeds size_type limit",
    ):
        plc.Column.from_array(arr)


@pytest.mark.parametrize("obj", [None, "str"])
def test_from_ndarray_invalid_obj(obj):
    with pytest.raises(
        TypeError,
        match=r"Cannot convert object of type .* to a pylibcudf Column",
    ):
        plc.Column.from_array(obj)


def test_array_interface_with_data_none():
    class ArrayInterfaceWithNone:
        @property
        def __array_interface__(self):
            return {
                "shape": (4,),
                "typestr": "<i4",
                "data": None,
                "version": 3,
            }

    with pytest.raises(
        ValueError,
        match=r"Expected a data field .* the array interface.",
    ):
        plc.Column.from_array(ArrayInterfaceWithNone())


def test_array_interface_with_mask_raises():
    class ArrayInterfaceWithMask:
        def __init__(self):
            self.values = np.array([1, 2, 3], dtype=np.int32)

        @property
        def __array_interface__(self):
            return {**self.values.__array_interface__, "mask": object()}

    with pytest.raises(
        NotImplementedError,
        match="Masked array-interface inputs are not supported",
    ):
        plc.Column.from_array_interface(ArrayInterfaceWithMask())


def test_from_zero_dimensional_array():
    arr = np.array(0)
    with pytest.raises(
        ValueError,
        match="shape must be a non-empty tuple",
    ):
        plc.Column.from_array(arr)


@pytest.mark.parametrize("np_or_cp_array", [np.array, cp.array])
@pytest.mark.parametrize(
    "arr, dtype, expect",
    [
        ([], np.int64, pa.array([], type=pa.int64())),
        ([[], []], np.int32, pa.array([[], []], type=pa.list_(pa.int32()))),
    ],
)
def test_empty_array(np_or_cp_array, arr, dtype, expect, patch_cupy_stream):
    with patch_cupy_stream:
        arr = np_or_cp_array(arr, dtype=dtype)
    col = plc.Column.from_array(arr)

    assert_column_eq(expect, col)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import itertools
import weakref

import numpy as np
import pytest

import rmm

import pylibcudf as plc

DTYPES = [
    "u1",
    "i2",
    "f4",
    "f8",
    "f16",
]
SIZES = [
    0,
    1,
    1000,
    1024,
    10000,
]


@pytest.fixture(params=tuple(itertools.product(SIZES, DTYPES)), ids=repr)
def np_array(request):
    size, dtype = request.param
    return np.empty((size,), dtype=dtype)


@pytest.mark.parametrize("stream", [None, rmm.pylibrmm.stream.Stream()])
def test_cuda_array_interface(np_array, stream):
    buf = rmm.DeviceBuffer(
        ptr=np_array.__array_interface__["data"][0],
        size=np_array.nbytes,
        stream=plc.utils._get_stream(stream),
    )
    gpumemview = plc.gpumemoryview(buf)

    np_array_view = np_array.view("u1")

    ai = np_array_view.__array_interface__
    cai = gpumemview.__cuda_array_interface__
    assert cai["shape"] == ai["shape"]
    assert cai["strides"] == ai["strides"]
    assert cai["typestr"] == ai["typestr"]


@pytest.mark.parametrize("stream", [None, rmm.pylibrmm.stream.Stream()])
def test_len(np_array, stream):
    buf = rmm.DeviceBuffer(
        ptr=np_array.__array_interface__["data"][0],
        size=np_array.nbytes,
        stream=plc.utils._get_stream(stream),
    )
    gpumemview = plc.gpumemoryview(buf)

    np_array_view = np_array.view("u1")

    assert len(gpumemview) == len(np_array_view)
    assert gpumemview.nbytes == np_array.nbytes


@pytest.mark.parametrize(
    "s",
    [
        slice(1, 3),
        slice(None, 2),
        slice(3, None),
        slice(2, 2),
        slice(0, 10000),
    ],
)
def test_slice(np_array, s):
    gv = plc.Column.from_array(np_array.view("u1")).data()
    result = plc.Column.from_array(gv.byte_slice(s)).to_pylist()
    assert result == np_array.view("u1")[s].tolist()


def test_slice_fails(np_array):
    gv = plc.Column.from_array(np_array.view("u1")).data()
    with pytest.raises(TypeError, match="requires a slice"):
        gv.byte_slice(0)
    with pytest.raises(ValueError, match="step=1"):
        gv.byte_slice(slice(None, None, 2))


def test_slice_keeps_parent_alive():
    col = plc.Column.from_array(np.arange(10, dtype="u1"))
    gv = col.data()
    col_ref = weakref.ref(col)
    gv_ref = weakref.ref(gv)
    s = gv.byte_slice(slice(2, 5))
    del col, gv
    assert col_ref() is None
    assert gv_ref() is not None
    del s
    assert gv_ref() is None

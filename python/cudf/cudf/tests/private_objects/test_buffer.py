# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import pytest

from cudf.core.buffer import Buffer, as_buffer

pytestmark = pytest.mark.spilling


@pytest.fixture
def arr_len():
    return 10


@pytest.mark.parametrize(
    "data, expect_success",
    [
        (lambda arr_len: cp.zeros(arr_len), True),
        (lambda arr_len: cp.zeros((1, arr_len)), True),
        (lambda arr_len: cp.zeros((1, arr_len, 1)), True),
        (lambda arr_len: cp.zeros((arr_len, arr_len)), True),
        (
            lambda arr_len: cp.zeros((arr_len, arr_len)).reshape(
                arr_len * arr_len
            ),
            True,
        ),
        (lambda arr_len: cp.zeros((arr_len, arr_len))[:, 0], False),
    ],
)
def test_buffer_from_cuda_iface_contiguous(data, expect_success, arr_len):
    data = data(arr_len)
    if expect_success:
        as_buffer(data.view("|u1"))
    else:
        with pytest.raises(ValueError):
            as_buffer(data.view("|u1"))


@pytest.mark.parametrize(
    "data",
    [
        lambda arr_len: cp.arange(arr_len),
        lambda arr_len: cp.arange(arr_len).reshape(1, arr_len),
        lambda arr_len: cp.arange(arr_len).reshape(1, arr_len, 1),
        lambda arr_len: cp.arange(arr_len**2).reshape(arr_len, arr_len),
    ],
)
def test_buffer_from_cuda_iface_dtype(data, arr_len):
    data = data(arr_len)
    for dtype in ("uint8", "int8", "float32", "int32"):
        typed_data = data.astype(dtype)
        buf = as_buffer(typed_data)
        got = cp.array(buf).reshape(-1).view("uint8")
        expect = typed_data.reshape(-1).view("uint8")
        assert (expect == got).all()


def test_buffer_creation_from_any(arr_len):
    ary = cp.arange(arr_len)
    b = as_buffer(ary)
    assert isinstance(b, Buffer)
    assert ary.data.ptr == b.ptr
    assert ary.nbytes == b.size


@pytest.mark.parametrize("size", [10, 2**10 + 500, 2**20])
def test_buffer_str(size):
    ary = cp.arange(size, dtype="uint8")
    buf = as_buffer(ary)
    assert f"size={size}" in repr(buf)


@pytest.mark.parametrize(
    "size,expect", [(10, "10B"), (2**10 + 500, "1.49KiB"), (2**20, "1MiB")]
)
def test_buffer_repr(size, expect):
    ary = cp.arange(size, dtype="uint8")
    buf = as_buffer(ary)
    assert f"size={expect}" in str(buf)


@pytest.mark.parametrize(
    "idx",
    [
        slice(0, 0),
        slice(0, 1),
        slice(-2, -1),
        slice(0, 10),
        slice(2, 3),
        slice(2, -1),
    ],
)
def test_buffer_slice(idx, arr_len):
    ary = cp.arange(arr_len, dtype="uint8")
    buf = as_buffer(ary)
    expect = ary[idx]
    got = cp.array(buf[idx])
    assert (expect == got).all()


@pytest.mark.parametrize(
    "idx, err_type, err_msg",
    [
        (1, TypeError, "Argument 'key' has incorrect type"),
        (slice(3, 2), ValueError, "size cannot be negative"),
        (slice(1, 2, 2), ValueError, "slice must be C-contiguous"),
        (slice(1, 2, -1), ValueError, "slice must be C-contiguous"),
        (slice(3, 2, -1), ValueError, "slice must be C-contiguous"),
    ],
)
def test_buffer_slice_fail(idx, err_type, err_msg, arr_len):
    ary = cp.arange(arr_len, dtype="uint8")
    buf = as_buffer(ary)

    with pytest.raises(err_type, match=err_msg):
        buf[idx]

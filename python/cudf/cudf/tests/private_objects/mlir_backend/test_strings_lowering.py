# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import operator

import cupy as cp
import numpy as np
import pytest
from numba_cuda_mlir import cuda, types

import cudf.core.udf.mlir_backend.strings_lowering  # noqa: F401  registers len
from cudf.core.udf.api import Masked
from cudf.core.udf.mlir_backend.strings_typing import (
    ManagedStrArrayWrapper,
    mlir_string,
    mlir_string_arg_handler,
)
from cudf.core.udf.utils import DEPRECATED_SM_REGEX

from .utils import MLIRNumbaCudaConfig

pytestmark = [
    pytest.mark.filterwarnings(f"ignore:{DEPRECATED_SM_REGEX}:UserWarning"),
    pytest.mark.filterwarnings(
        "ignore:Grid size:"
        "numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


class _DeviceBuf:
    """Minimal ``.ptr``-exposing wrapper over a cupy array (test marshaller)."""

    def __init__(self, arr):
        self._arr = arr

    @property
    def ptr(self):
        return int(self._arr.data.ptr)


def _make_mlir_string_array(pystrings):
    """Build a device ``mlir_string`` array (borrowed data, null meminfo).

    ``None`` entries produce null rows: a null ``data`` pointer with
    ``nbytes == 0`` (mirroring how a null string row is marshalled), which the
    length loop must never dereference.

    Returns the ``ManagedStrArrayWrapper`` plus the backing device arrays, which
    must be kept alive for the duration of the kernel launch.
    """
    encoded = [None if s is None else s.encode("utf-8") for s in pystrings]
    chars = b"".join(e for e in encoded if e) or b"\x00"
    chars_dev = cp.asarray(np.frombuffer(chars, dtype=np.uint8))
    base = int(chars_dev.data.ptr)
    # struct layout {u64 meminfo, u64 data, i64 nbytes} == 3 x 8 bytes
    structs = np.zeros(len(pystrings) * 3, dtype=np.uint64)
    offset = 0
    for i, e in enumerate(encoded):
        if e is None:
            # null row: data==0 (null ptr), nbytes==0 (already zeroed)
            continue
        structs[i * 3 + 1] = base + offset  # data
        structs[i * 3 + 2] = len(e)  # nbytes
        offset += len(e)
    structs_dev = cp.asarray(structs)
    wrapper = ManagedStrArrayWrapper(_DeviceBuf(structs_dev))
    return wrapper, (chars_dev, structs_dev)


@pytest.mark.parametrize(
    "value,expected",
    [
        ("", 0),
        ("a", 1),
        ("abc", 3),
        ("h\u00e9llo", 5),  # é is 2 UTF-8 bytes, 1 char
        ("\U0001f600x", 2),  # emoji is 4 UTF-8 bytes, 1 char
        ("na\u00efve", 5),
    ],
)
def test_len_counts_characters(value, expected):
    """``len(mlir_string)`` returns the UTF-8 character count, not byte count."""
    arr, _keep = _make_mlir_string_array([value])
    out = cp.zeros(1, dtype=np.int64)

    @cuda.jit(
        types.void(types.int64[::1], types.CPointer(mlir_string)),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, s):
        o[0] = len(s[0])

    with MLIRNumbaCudaConfig():
        k[1, 1](out, arr)
    cuda.synchronize()
    assert int(out.get()[0]) == expected


def test_len_over_array():
    """``len`` over a multi-element ``mlir_string`` array, one thread per row."""
    strings = ["", "a", "abc", "h\u00e9llo", "\U0001f600x"]
    arr, _keep = _make_mlir_string_array(strings)
    out = cp.zeros(len(strings), dtype=np.int64)

    @cuda.jit(
        types.void(types.int64[::1], types.CPointer(mlir_string)),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, s):
        i = cuda.grid(1)
        if i < o.size:
            o[i] = len(s[i])

    with MLIRNumbaCudaConfig():
        k[1, len(strings)](out, arr)
    cuda.synchronize()
    assert out.get().tolist() == [len(s) for s in strings]


@pytest.mark.parametrize("valid", [True, False])
def test_masked_len_propagates_validity(valid):
    """``len(Masked(mlir_string))`` -> ``Masked(int64)`` carrying validity.

    The ``valid=False`` case uses a genuine null-row payload (null ``data``,
    ``nbytes == 0``) so the scan is exercised against the null representation,
    not a stand-in valid payload.
    """
    payload = "abc" if valid else None
    arr, _keep = _make_mlir_string_array([payload])
    out = cp.zeros(1, dtype=np.int64)
    out_valid = cp.zeros(1, dtype=np.bool_)

    @cuda.jit(
        types.void(
            types.int64[::1],
            types.boolean[::1],
            types.CPointer(mlir_string),
            types.boolean[::1],
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, ov, s, sv):
        m = len(Masked(s[0], sv[0]))
        o[0] = m.value
        ov[0] = m.valid

    with MLIRNumbaCudaConfig():
        k[1, 1](out, out_valid, arr, cp.array([valid], dtype=np.bool_))
    cuda.synchronize()
    assert bool(out_valid.get()[0]) is valid
    # Payload of an invalid Masked is not an API guarantee; only assert it when
    # the result is valid.
    if valid:
        assert int(out.get()[0]) == 3


def test_masked_len_mixed_null_rows():
    """``len`` over a ``Masked(mlir_string)`` array with interleaved null rows.

    Null rows carry a null ``data`` pointer with ``nbytes == 0``; the kernel
    must not crash on them and must propagate ``valid=False``.
    """
    strings = ["abc", None, "h\u00e9llo", None, ""]
    valid = [s is not None for s in strings]
    arr, _keep = _make_mlir_string_array(strings)
    n = len(strings)
    out = cp.zeros(n, dtype=np.int64)
    out_valid = cp.zeros(n, dtype=np.bool_)

    @cuda.jit(
        types.void(
            types.int64[::1],
            types.boolean[::1],
            types.CPointer(mlir_string),
            types.boolean[::1],
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, ov, s, sv):
        i = cuda.grid(1)
        if i < o.size:
            m = len(Masked(s[i], sv[i]))
            o[i] = m.value
            ov[i] = m.valid

    with MLIRNumbaCudaConfig():
        k[1, n](out, out_valid, arr, cp.array(valid, dtype=np.bool_))
    cuda.synchronize()
    got_valid = out_valid.get().tolist()
    got_value = out.get().tolist()
    assert got_valid == valid
    for s, v, value in zip(strings, got_valid, got_value, strict=True):
        if v:
            assert value == len(s)


_CMP_OPS = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]

# Comparison operands are independent of the operator under test, so build the
# device ``mlir_string`` arrays once per module and reuse across parametrizations.
_CMP_LEFT = ["abc", "abc", "abd", "ab", "abcd", "", "h\u00e9llo"]
_CMP_RIGHT = ["abc", "abd", "abc", "abc", "abc", "", "h\u00e9llo"]
_CMP_LITERAL_STRINGS = ["foo", "fop", "fon", "fo", "foobar", ""]


@pytest.fixture(scope="module")
def cmp_array_operands():
    """``(left, right, left_arr, right_arr)`` for element-wise comparisons."""
    la, keep_l = _make_mlir_string_array(_CMP_LEFT)
    ra, keep_r = _make_mlir_string_array(_CMP_RIGHT)
    # keep_l/keep_r must outlive the kernel launches that read the arrays.
    yield _CMP_LEFT, _CMP_RIGHT, la, ra, (keep_l, keep_r)


@pytest.fixture(scope="module")
def cmp_literal_operand():
    """``(strings, arr)`` for comparisons against a string literal."""
    arr, keep = _make_mlir_string_array(_CMP_LITERAL_STRINGS)
    yield _CMP_LITERAL_STRINGS, arr, keep


@pytest.mark.parametrize("op", _CMP_OPS)
def test_string_comparison_arrays(op, cmp_array_operands):
    """``str <cmp> str`` element-wise over two ``mlir_string`` arrays."""
    left, right, la, ra, _keep = cmp_array_operands
    n = len(left)
    out = cp.zeros(n, dtype=np.bool_)

    @cuda.jit(
        types.void(
            types.boolean[::1],
            types.CPointer(mlir_string),
            types.CPointer(mlir_string),
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, a, b):
        i = cuda.grid(1)
        if i < o.size:
            o[i] = op(a[i], b[i])

    with MLIRNumbaCudaConfig():
        k[1, n](out, la, ra)
    cuda.synchronize()
    assert out.get().tolist() == [
        op(x, y) for x, y in zip(left, right, strict=True)
    ]


@pytest.mark.parametrize("op", _CMP_OPS)
def test_string_comparison_literal(op, cmp_literal_operand):
    """``str <cmp> "literal"`` and the reflected form over an array."""
    strings, arr, _keep = cmp_literal_operand
    n = len(strings)
    out = cp.zeros(n, dtype=np.bool_)
    out_r = cp.zeros(n, dtype=np.bool_)

    @cuda.jit(
        types.void(
            types.boolean[::1],
            types.boolean[::1],
            types.CPointer(mlir_string),
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, orev, s):
        i = cuda.grid(1)
        if i < o.size:
            o[i] = op(s[i], "foo")
            orev[i] = op("foo", s[i])

    with MLIRNumbaCudaConfig():
        k[1, n](out, out_r, arr)
    cuda.synchronize()
    assert out.get().tolist() == [op(s, "foo") for s in strings]
    assert out_r.get().tolist() == [op("foo", s) for s in strings]


def test_masked_string_comparison_literal():
    """``Masked(str) == literal`` -> ``Masked(bool)``, validity from the operand."""
    strings = ["foo", "bar", "foo"]
    valid = [True, False, True]
    arr, _k = _make_mlir_string_array(strings)
    n = len(strings)
    out = cp.zeros(n, dtype=np.bool_)
    out_valid = cp.zeros(n, dtype=np.bool_)

    @cuda.jit(
        types.void(
            types.boolean[::1],
            types.boolean[::1],
            types.CPointer(mlir_string),
            types.boolean[::1],
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, ov, s, sv):
        i = cuda.grid(1)
        if i < o.size:
            r = Masked(s[i], sv[i]) == "foo"
            o[i] = r.value
            ov[i] = r.valid

    with MLIRNumbaCudaConfig():
        k[1, n](out, out_valid, arr, cp.array(valid, dtype=np.bool_))
    cuda.synchronize()
    assert out_valid.get().tolist() == valid
    for i in range(n):
        if valid[i]:
            assert bool(out.get()[i]) is (strings[i] == "foo")


def test_masked_string_comparison_propagates_validity():
    """``Masked(str) == Masked(str)`` -> ``Masked(bool)``; valid iff both valid.

    The result value is only asserted on rows whose result is valid.
    """
    a = ["abc", "abd", "xyz", "ab"]
    av = [True, False, True, True]
    b = ["abc", "abd", "xyw", "abc"]
    bv = [True, True, False, True]
    arr_a, _ka = _make_mlir_string_array(a)
    arr_b, _kb = _make_mlir_string_array(b)
    n = len(a)
    out = cp.zeros(n, dtype=np.bool_)
    out_valid = cp.zeros(n, dtype=np.bool_)

    @cuda.jit(
        types.void(
            types.boolean[::1],
            types.boolean[::1],
            types.CPointer(mlir_string),
            types.boolean[::1],
            types.CPointer(mlir_string),
            types.boolean[::1],
        ),
        extensions=[mlir_string_arg_handler],
    )
    def k(o, ov, sa, sav, sb, sbv):
        i = cuda.grid(1)
        if i < o.size:
            r = Masked(sa[i], sav[i]) == Masked(sb[i], sbv[i])
            o[i] = r.value
            ov[i] = r.valid

    with MLIRNumbaCudaConfig():
        k[1, n](
            out,
            out_valid,
            arr_a,
            cp.array(av, dtype=np.bool_),
            arr_b,
            cp.array(bv, dtype=np.bool_),
        )
    cuda.synchronize()
    got_valid = out_valid.get().tolist()
    assert got_valid == [x and y for x, y in zip(av, bv, strict=True)]
    for i in range(n):
        if got_valid[i]:
            assert bool(out.get()[i]) is (a[i] == b[i])

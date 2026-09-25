# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-MLIR building blocks composed by the ``mlir_string`` lowerings."""

from __future__ import annotations

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, llvm, scf
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.lowering_utilities import GEP_DYNAMIC_INDEX


def _is_begin_utf8_char(byte_val: ir.Value) -> ir.Value:
    """i1: whether ``byte_val`` is a UTF-8 start byte (top 2 bits != ``10``)."""
    i32 = ir.IntegerType.get_signless(32)
    masked = arith.andi(arith.extui(i32, byte_val), arith.constant(i32, 0xC0))
    return arith.cmpi(
        arith.CmpIPredicate.ne, masked, arith.constant(i32, 0x80)
    )


def _lower_len(ms_val: ir.Value) -> ir.Value:
    """``len(mlir_string) -> i64`` character count (pure MLIR).

    Counts UTF-8 start bytes over the ``nbytes``-long data buffer, so the result
    is the number of characters (code points), matching cuDF ``str.len``. The
    count is accumulated directly in ``i64`` to match the declared result type.
    The ``mlir_string`` layout is ``{ptr meminfo, ptr data, i64 nbytes}`` --
    fields 1 (``data``) and 2 (``nbytes``) are read below.
    """
    data = llvm.extractvalue(llvm.PointerType.get(), ms_val, [1])
    nb = llvm.extractvalue(T.i64(), ms_val, [2])
    zero_i64 = arith.constant(T.i64(), 0)
    one_i64 = arith.constant(T.i64(), 1)

    loop = scf.ForOp(zero_i64, nb, one_i64, [zero_i64])
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc = loop.inner_iter_args[0]
        # data[idx] -- GEP a dynamic byte offset into the i8* buffer
        byte_ptr = llvm.getelementptr(
            llvm.PointerType.get(),
            data,
            [idx],
            [GEP_DYNAMIC_INDEX],
            T.i8(),
            None,
        )
        b = llvm.load(T.i8(), byte_ptr)
        inc = arith.select(_is_begin_utf8_char(b), one_i64, zero_i64)
        scf.yield_([arith.addi(acc, inc)])

    return loop.results[0]

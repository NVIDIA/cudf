# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-MLIR building blocks composed by the ``mlir_string`` lowerings.

Helpers work on raw MLIR SSA values (no cuDF/NRT dependency). ``mlir_string``
layout is ``{ptr meminfo, ptr data, i64 nbytes}``; ops run over an internal
*view* value ``{ptr data, i32 nbytes, i32 length}`` built by
:func:`_mlir_string_to_view`.
"""

from __future__ import annotations

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, llvm, scf
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.lowering_utilities import GEP_DYNAMIC_INDEX


def _byte_at(data: ir.Value, idx: ir.Value) -> ir.Value:
    """Load ``data[idx]`` as ``i8`` via a dynamic GEP on the ``i8*`` buffer."""
    return llvm.load(
        T.i8(),
        llvm.getelementptr(
            llvm.PointerType.get(),
            data,
            [idx],
            [GEP_DYNAMIC_INDEX],
            T.i8(),
            None,
        ),
    )


# --- mlir_string / view field accessors ------------------------------------
# view layout: {ptr data, i32 nbytes, i32 length}; only data + nbytes are used.
def _view_data(view_val: ir.Value) -> ir.Value:
    return llvm.extractvalue(llvm.PointerType.get(), view_val, [0])


def _view_nbytes(view_val: ir.Value) -> ir.Value:
    return llvm.extractvalue(ir.IntegerType.get_signless(32), view_val, [1])


def _ms_extract_nbytes(ms_val: ir.Value) -> ir.Value:
    """Extract ``nbytes`` (i64, field 2) from an ``mlir_string`` value."""
    return llvm.extractvalue(T.i64(), ms_val, [2])


def _mlir_string_to_view(ms_val: ir.Value) -> ir.Value:
    """Build an internal view ``{ptr data, i32 nbytes, i32 length}``.

    ``length`` is set equal to ``nbytes`` here; the character count is computed
    on demand by :func:`_lower_len`.
    """
    i32 = ir.IntegerType.get_signless(32)
    ptr = llvm.PointerType.get()
    data = llvm.extractvalue(ptr, ms_val, [1])
    nbytes_i32 = arith.trunci(i32, _ms_extract_nbytes(ms_val))
    view_ty = llvm.StructType.get_literal([ptr, i32, i32])
    val = llvm.UndefOp(view_ty)
    val = llvm.insertvalue(
        container=val, value=data, position=ir.DenseI64ArrayAttr.get([0])
    )
    val = llvm.insertvalue(
        container=val, value=nbytes_i32, position=ir.DenseI64ArrayAttr.get([1])
    )
    return llvm.insertvalue(
        container=val, value=nbytes_i32, position=ir.DenseI64ArrayAttr.get([2])
    )


def _is_begin_utf8_char(byte_val: ir.Value) -> ir.Value:
    """i1: whether ``byte_val`` is a UTF-8 start byte (top 2 bits != ``10``)."""
    i32 = ir.IntegerType.get_signless(32)
    masked = arith.andi(arith.extui(i32, byte_val), arith.constant(i32, 0xC0))
    return arith.cmpi(
        arith.CmpIPredicate.ne, masked, arith.constant(i32, 0x80)
    )


def _lower_len(str_view: ir.Value) -> ir.Value:
    """``len`` -> ``i32`` character count (number of UTF-8 start bytes).

    ``int32`` matches libcudf's ``size_type`` used for string lengths.
    """
    i32 = ir.IntegerType.get_signless(32)
    data = _view_data(str_view)
    nb = llvm.zext(T.i64(), _view_nbytes(str_view))
    zero_i32 = arith.constant(i32, 0)
    loop = scf.ForOp(
        arith.constant(T.i64(), 0),
        nb,
        arith.constant(T.i64(), 1),
        [zero_i32],
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc = loop.inner_iter_args[0]
        inc = arith.select(
            _is_begin_utf8_char(_byte_at(data, idx)),
            arith.constant(i32, 1),
            zero_i32,
        )
        scf.yield_([arith.addi(acc, inc)])
    return loop.results[0]

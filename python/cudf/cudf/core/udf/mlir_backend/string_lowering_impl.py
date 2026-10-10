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


def _lower_compare(lhs: ir.Value, rhs: ir.Value) -> ir.Value:
    """Byte-wise lexicographic compare of two views -> ``i32`` (sign gives order).

    Matches ``cudf::string_view::compare``: the signed difference of the first
    differing byte, else a length comparison. Callers turn the sign into the
    boolean result for a specific comparison operator. The scan early-exits at
    the first differing byte (``scf.while``).
    """
    i32 = ir.IntegerType.get_signless(32)
    lhs_data = _view_data(lhs)
    rhs_data = _view_data(rhs)
    lhs_nb = llvm.zext(T.i64(), _view_nbytes(lhs))
    rhs_nb = llvm.zext(T.i64(), _view_nbytes(rhs))
    zero_i32 = arith.constant(i32, 0)
    zero_i64 = arith.constant(T.i64(), 0)

    min_len = arith.select(
        arith.cmpi(arith.CmpIPredicate.slt, lhs_nb, rhs_nb), lhs_nb, rhs_nb
    )

    # Walk the shared prefix carrying (byte_index, diff). ``diff`` stays 0 until
    # the first differing byte; a non-zero ``diff`` is the signed ordering value
    # and ends the loop (so we never scan past the first mismatch).
    loop = scf.WhileOp([T.i64(), i32], [zero_i64, zero_i32])
    before = loop.before.blocks.append(T.i64(), i32)
    with ir.InsertionPoint(before):
        idx, diff = before.arguments
        in_bounds = arith.cmpi(arith.CmpIPredicate.ult, idx, min_len)
        no_diff = arith.cmpi(arith.CmpIPredicate.eq, diff, zero_i32)
        scf.condition(arith.andi(in_bounds, no_diff), [idx, diff])
    after = loop.after.blocks.append(T.i64(), i32)
    with ir.InsertionPoint(after):
        idx, _diff = after.arguments
        b1 = arith.extui(i32, _byte_at(lhs_data, idx))
        b2 = arith.extui(i32, _byte_at(rhs_data, idx))
        next_idx = arith.addi(idx, arith.constant(T.i64(), 1))
        scf.YieldOp([next_idx, arith.subi(b1, b2)])

    byte_diff = loop.results[1]
    found = arith.cmpi(arith.CmpIPredicate.ne, byte_diff, zero_i32)
    lhs_longer = arith.cmpi(arith.CmpIPredicate.sgt, lhs_nb, rhs_nb)
    rhs_longer = arith.cmpi(arith.CmpIPredicate.slt, lhs_nb, rhs_nb)
    len_cmp = arith.select(
        lhs_longer,
        arith.constant(i32, 1),
        arith.select(rhs_longer, arith.constant(i32, -1), zero_i32),
    )
    return arith.select(found, byte_diff, len_cmp)


def _bytes_equal(
    a_data: ir.Value, a_off: ir.Value, b_data: ir.Value, n: ir.Value
) -> ir.Value:
    """i1: whether ``a_data[a_off:a_off+n] == b_data[0:n]`` byte-for-byte.

    Early-exits (``scf.while``) at the first differing byte; callers must pass
    ``n == 0`` when the range would read out of bounds.
    """
    i1 = ir.IntegerType.get_signless(1)
    zero_i64 = arith.constant(T.i64(), 0)
    loop = scf.WhileOp([T.i64(), i1], [zero_i64, arith.constant(i1, 1)])
    before = loop.before.blocks.append(T.i64(), i1)
    with ir.InsertionPoint(before):
        idx, eq = before.arguments
        in_bounds = arith.cmpi(arith.CmpIPredicate.ult, idx, n)
        scf.condition(arith.andi(in_bounds, eq), [idx, eq])
    after = loop.after.blocks.append(T.i64(), i1)
    with ir.InsertionPoint(after):
        idx, _eq = after.arguments
        av = _byte_at(a_data, arith.addi(a_off, idx))
        bv = _byte_at(b_data, idx)
        next_idx = arith.addi(idx, arith.constant(T.i64(), 1))
        scf.YieldOp([next_idx, arith.cmpi(arith.CmpIPredicate.eq, av, bv)])
    return loop.results[1]


def _lower_startswith(str_view: ir.Value, prefix: ir.Value) -> ir.Value:
    """``startswith(prefix)`` -> ``i1`` (byte prefix match)."""
    i1 = ir.IntegerType.get_signless(1)
    src_nb = llvm.zext(T.i64(), _view_nbytes(str_view))
    pfx_nb = llvm.zext(T.i64(), _view_nbytes(prefix))
    too_long = arith.cmpi(arith.CmpIPredicate.ugt, pfx_nb, src_nb)
    # Compare 0 bytes when the prefix cannot fit, so the scan never over-reads
    # the source; the result is false anyway.
    n = arith.select(too_long, arith.constant(T.i64(), 0), pfx_nb)
    eq = _bytes_equal(
        _view_data(str_view), arith.constant(T.i64(), 0), _view_data(prefix), n
    )
    return arith.andi(arith.xori(too_long, arith.constant(i1, 1)), eq)


def _lower_endswith(str_view: ir.Value, suffix: ir.Value) -> ir.Value:
    """``endswith(suffix)`` -> ``i1`` (byte suffix match)."""
    i1 = ir.IntegerType.get_signless(1)
    src_nb = llvm.zext(T.i64(), _view_nbytes(str_view))
    sfx_nb = llvm.zext(T.i64(), _view_nbytes(suffix))
    too_long = arith.cmpi(arith.CmpIPredicate.ugt, sfx_nb, src_nb)
    zero_i64 = arith.constant(T.i64(), 0)
    # Clamp both the start offset and the length when the suffix cannot fit so
    # the scan stays in bounds; the result is false anyway.
    n = arith.select(too_long, zero_i64, sfx_nb)
    offset = arith.select(too_long, zero_i64, arith.subi(src_nb, sfx_nb))
    eq = _bytes_equal(_view_data(str_view), offset, _view_data(suffix), n)
    return arith.andi(arith.xori(too_long, arith.constant(i1, 1)), eq)


def _lower_find(str_view: ir.Value, target: ir.Value) -> ir.Value:
    """``find(target)`` -> ``i32`` character position of the first match, else -1.

    Byte-level substring search; the returned index counts UTF-8 characters (not
    bytes) before the match, matching cuDF ``str.find``.
    """
    i32 = ir.IntegerType.get_signless(32)
    i1 = ir.IntegerType.get_signless(1)
    src_data = _view_data(str_view)
    src_nb = llvm.zext(T.i64(), _view_nbytes(str_view))
    tgt_data = _view_data(target)
    tgt_nb = llvm.zext(T.i64(), _view_nbytes(target))
    zero_i32 = arith.constant(i32, 0)
    neg_one = arith.constant(i32, -1)
    one_i64 = arith.constant(T.i64(), 1)

    tgt_empty = arith.cmpi(
        arith.CmpIPredicate.eq, tgt_nb, arith.constant(T.i64(), 0)
    )
    too_long = arith.cmpi(arith.CmpIPredicate.sgt, tgt_nb, src_nb)
    # number of candidate start offsets = src_nb - tgt_nb + 1 (0 if target longer)
    search_len = arith.select(
        too_long,
        arith.constant(T.i64(), 0),
        arith.addi(arith.subi(src_nb, tgt_nb), one_i64),
    )

    loop = scf.ForOp(
        arith.constant(T.i64(), 0),
        search_len,
        one_i64,
        [neg_one, zero_i32, arith.constant(i1, 0)],
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc_result = loop.inner_iter_args[0]
        acc_char_pos = loop.inner_iter_args[1]
        acc_found = loop.inner_iter_args[2]
        matched = _bytes_equal(src_data, idx, tgt_data, tgt_nb)
        first_match = arith.andi(
            matched, arith.xori(acc_found, arith.constant(i1, 1))
        )
        new_result = arith.select(first_match, acc_char_pos, acc_result)
        is_start = _is_begin_utf8_char(_byte_at(src_data, idx))
        new_char_pos = arith.addi(
            acc_char_pos,
            arith.select(is_start, arith.constant(i32, 1), zero_i32),
        )
        scf.yield_([new_result, new_char_pos, arith.ori(acc_found, matched)])

    # An empty target matches at character position 0.
    return arith.select(tgt_empty, zero_i32, loop.results[0])


def _lower_rfind(str_view: ir.Value, target: ir.Value) -> ir.Value:
    """``rfind(target)`` -> ``i32`` character position of the last match, else -1."""
    i32 = ir.IntegerType.get_signless(32)
    src_data = _view_data(str_view)
    src_nb = llvm.zext(T.i64(), _view_nbytes(str_view))
    tgt_data = _view_data(target)
    tgt_nb = llvm.zext(T.i64(), _view_nbytes(target))
    zero_i32 = arith.constant(i32, 0)
    neg_one = arith.constant(i32, -1)
    one_i64 = arith.constant(T.i64(), 1)

    tgt_empty = arith.cmpi(
        arith.CmpIPredicate.eq, tgt_nb, arith.constant(T.i64(), 0)
    )
    too_long = arith.cmpi(arith.CmpIPredicate.sgt, tgt_nb, src_nb)
    search_len = arith.select(
        too_long,
        arith.constant(T.i64(), 0),
        arith.addi(arith.subi(src_nb, tgt_nb), one_i64),
    )

    # Scan forward keeping the most recent match, so the last match wins.
    loop = scf.ForOp(
        arith.constant(T.i64(), 0), search_len, one_i64, [neg_one, zero_i32]
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        acc_result = loop.inner_iter_args[0]
        acc_char_pos = loop.inner_iter_args[1]
        matched = _bytes_equal(src_data, idx, tgt_data, tgt_nb)
        new_result = arith.select(matched, acc_char_pos, acc_result)
        is_start = _is_begin_utf8_char(_byte_at(src_data, idx))
        new_char_pos = arith.addi(
            acc_char_pos,
            arith.select(is_start, arith.constant(i32, 1), zero_i32),
        )
        scf.yield_([new_result, new_char_pos])

    # An empty target matches at the end -> len(str).
    return arith.select(
        tgt_empty,
        _lower_len(str_view),
        arith.select(too_long, neg_one, loop.results[0]),
    )


def _lower_count(str_view: ir.Value, target: ir.Value) -> ir.Value:
    """``count(target)`` -> ``i32`` non-overlapping occurrence count."""
    i32 = ir.IntegerType.get_signless(32)
    src_data = _view_data(str_view)
    src_nb = llvm.zext(T.i64(), _view_nbytes(str_view))
    tgt_data = _view_data(target)
    tgt_nb = llvm.zext(T.i64(), _view_nbytes(target))
    zero_i32 = arith.constant(i32, 0)
    zero_i64 = arith.constant(T.i64(), 0)
    one_i64 = arith.constant(T.i64(), 1)

    tgt_empty = arith.cmpi(arith.CmpIPredicate.eq, tgt_nb, zero_i64)
    too_long = arith.cmpi(arith.CmpIPredicate.sgt, tgt_nb, src_nb)

    loop = scf.ForOp(zero_i64, src_nb, one_i64, [zero_i64, zero_i32])
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        cur_pos = loop.inner_iter_args[0]
        cur_count = loop.inner_iter_args[1]
        at_pos = arith.cmpi(arith.CmpIPredicate.eq, idx, cur_pos)
        remaining = arith.subi(src_nb, cur_pos)
        enough = arith.cmpi(arith.CmpIPredicate.sge, remaining, tgt_nb)
        should_check = arith.andi(at_pos, enough)
        # Compare 0 bytes when the check does not apply so _bytes_equal never
        # reads past the source buffer.
        cmp_len = arith.select(should_check, tgt_nb, zero_i64)
        matched = arith.andi(
            should_check, _bytes_equal(src_data, cur_pos, tgt_data, cmp_len)
        )
        new_count = arith.select(
            matched, arith.addi(cur_count, arith.constant(i32, 1)), cur_count
        )
        # On a match advance past it (non-overlapping); otherwise step one byte.
        next_pos = arith.select(
            matched, arith.addi(cur_pos, tgt_nb), arith.addi(cur_pos, one_i64)
        )
        final_pos = arith.select(at_pos, next_pos, cur_pos)
        final_count = arith.select(at_pos, new_count, cur_count)
        scf.yield_([final_pos, final_count])

    # Python counts an empty target len(str)+1 times.
    src_len_plus_one = arith.addi(_lower_len(str_view), arith.constant(i32, 1))
    return arith.select(
        tgt_empty,
        src_len_plus_one,
        arith.select(too_long, zero_i32, loop.results[1]),
    )


def _lower_contains(container: ir.Value, target: ir.Value) -> ir.Value:
    """``target in container`` -> ``i1`` (substring membership)."""
    return arith.cmpi(
        arith.CmpIPredicate.sge,
        _lower_find(container, target),
        arith.constant(ir.IntegerType.get_signless(32), 0),
    )

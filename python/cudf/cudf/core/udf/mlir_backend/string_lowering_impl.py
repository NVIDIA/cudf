# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pure-MLIR building blocks composed by the ``mlir_string`` lowerings.

Most helpers work on raw MLIR SSA values with no runtime dependency. The
allocation helpers (:func:`allocate_mlir_string`, :func:`lower_concat`)
additionally emit calls into the Numba Runtime (NRT) to create *owned* strings;
see the developer guide "String UDF memory management" for the ownership model.
``mlir_string`` layout is ``{ptr meminfo, ptr data, i64 nbytes}``; ops run over
an internal *view* value ``{ptr data, i32 nbytes, i32 length}`` built by
:func:`_mlir_string_to_view`.
"""

from __future__ import annotations

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, func, llvm, scf
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.lowering_utilities import (
    GEP_DYNAMIC_INDEX,
    get_or_insert_function,
)

from pylibcudf.strings.char_types import StringCharacterTypes


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


# --- NRT-managed allocation (creates new owned strings) --------------------
# Unlike the accessors above, these helpers emit calls into the Numba Runtime
# to allocate device memory for *new* strings, so the result carries a non-null
# ``meminfo`` that numba's reference counting will free. See the developer guide
# "String UDF memory management".
def _call_nrt(
    gpu_module: ir.Module,
    name: str,
    arg_types: list,
    result_types: list,
    args: list,
) -> ir.Value:
    """Emit a call to NRT device function ``name`` declared in ``gpu_module``."""
    ft = ir.FunctionType.get(arg_types, result_types)
    callee = get_or_insert_function(name, ft, gpu_module)
    return func.call(
        result=result_types, callee=callee.name.value, operands_=args
    )


def build_mlir_string(
    meminfo: ir.Value, data: ir.Value, nbytes_i64: ir.Value
) -> ir.Value:
    """Assemble an ``mlir_string`` value ``{meminfo, data, nbytes}``.

    ``meminfo`` lands in field 0, which is where numba's reference counting
    looks for the ``MemInfo`` (see ``MLIRStringModel.get_nrt_meminfo``).

    Parameters
    ----------
    meminfo : ir.Value
        The NRT ``MemInfo`` pointer (null for a borrowed string).
    data : ir.Value
        Pointer to the UTF-8 byte buffer.
    nbytes_i64 : ir.Value
        Byte length of the string (``i64``).

    Returns
    -------
    ir.Value
        The assembled ``mlir_string`` SSA value.
    """
    ptr = llvm.PointerType.get()
    ty = llvm.StructType.get_literal([ptr, ptr, T.i64()])
    val = llvm.UndefOp(ty)
    val = llvm.insertvalue(
        container=val, value=meminfo, position=ir.DenseI64ArrayAttr.get([0])
    )
    val = llvm.insertvalue(
        container=val, value=data, position=ir.DenseI64ArrayAttr.get([1])
    )
    return llvm.insertvalue(
        container=val, value=nbytes_i64, position=ir.DenseI64ArrayAttr.get([2])
    )


def allocate_mlir_string(
    gpu_module: ir.Module, nbytes_i64: ir.Value
) -> tuple[ir.Value, ir.Value]:
    """Allocate an NRT-managed byte buffer; return ``(meminfo, data_ptr)``.

    ``NRT_MemInfo_new_varsize`` installs the varsize destructor so the buffer is
    freed when the refcount drops to zero (a no-op under the arena allocator).
    At least one byte is requested to avoid a device ``malloc(0)``.

    Parameters
    ----------
    gpu_module : ir.Module
        The GPU module the NRT functions are declared in.
    nbytes_i64 : ir.Value
        Requested buffer size in bytes (``i64``).

    Returns
    -------
    tuple of ir.Value
        ``(meminfo, data_ptr)`` for the newly allocated buffer.
    """
    ptr = llvm.PointerType.get()
    alloc_size = arith.maxui(nbytes_i64, arith.constant(T.i64(), 1))
    mi = _call_nrt(
        gpu_module, "NRT_MemInfo_new_varsize", [T.i64()], [ptr], [alloc_size]
    )
    data = _call_nrt(gpu_module, "NRT_MemInfo_data_fast", [ptr], [ptr], [mi])
    return mi, data


def lower_concat(
    gpu_module: ir.Module, lhs_view: ir.Value, rhs_view: ir.Value
) -> ir.Value:
    """Concatenate two views into a newly allocated owned ``mlir_string``.

    Computes ``total = lhs.nbytes + rhs.nbytes``, allocates an NRT buffer of
    that size, copies the ``lhs`` then the ``rhs`` bytes, and returns
    ``{meminfo, data, total}``.

    Parameters
    ----------
    gpu_module : ir.Module
        The GPU module the NRT functions are declared in.
    lhs_view : ir.Value
        View of the left operand's bytes.
    rhs_view : ir.Value
        View of the right operand's bytes.

    Returns
    -------
    ir.Value
        The concatenated owned ``mlir_string``.
    """
    ptr = llvm.PointerType.get()
    lhs_nbytes = llvm.zext(T.i64(), _view_nbytes(lhs_view))
    rhs_nbytes = llvm.zext(T.i64(), _view_nbytes(rhs_view))
    total = arith.addi(lhs_nbytes, rhs_nbytes)
    mi, data = allocate_mlir_string(gpu_module, total)
    llvm.intr_memcpy(data, _view_data(lhs_view), lhs_nbytes, False)
    dst = llvm.getelementptr(
        ptr, data, [lhs_nbytes], [GEP_DYNAMIC_INDEX], T.i8(), None
    )
    llvm.intr_memcpy(dst, _view_data(rhs_view), rhs_nbytes, False)
    return build_mlir_string(mi, data, total)


def _gep_byte_offset(base: ir.Value, offset: ir.Value) -> ir.Value:
    """Pointer to ``base + offset`` bytes (GEP on an ``i8*`` buffer)."""
    return llvm.getelementptr(
        llvm.PointerType.get(),
        base,
        [offset],
        [GEP_DYNAMIC_INDEX],
        T.i8(),
        None,
    )


def _empty_view() -> ir.Value:
    """View with a null data pointer and ``nbytes == 0`` (whitespace mode)."""
    i32 = ir.IntegerType.get_signless(32)
    ptr = llvm.PointerType.get()
    zero = arith.constant(i32, 0)
    null = llvm.inttoptr(ptr, arith.constant(T.i64(), 0))
    view_ty = llvm.StructType.get_literal([ptr, i32, i32])
    val = llvm.UndefOp(view_ty)
    val = llvm.insertvalue(
        container=val, value=null, position=ir.DenseI64ArrayAttr.get([0])
    )
    val = llvm.insertvalue(
        container=val, value=zero, position=ir.DenseI64ArrayAttr.get([1])
    )
    return llvm.insertvalue(
        container=val, value=zero, position=ir.DenseI64ArrayAttr.get([2])
    )


def _is_strip_char(
    byte_val: ir.Value, strip_data: ir.Value, strip_nb: ir.Value
) -> ir.Value:
    """i1: whether ``byte_val`` should be stripped.

    Mirrors libcudf ``detail::strip``: with an empty strip set, strip bytes with
    code ``<= 0x20`` (ASCII whitespace); otherwise strip any byte present in the
    strip set.
    """
    i1 = ir.IntegerType.get_signless(1)
    zero = arith.constant(T.i64(), 0)
    one = arith.constant(T.i64(), 1)
    strip_empty = arith.cmpi(arith.CmpIPredicate.eq, strip_nb, zero)
    is_ws = arith.cmpi(
        arith.CmpIPredicate.ule, byte_val, arith.constant(T.i8(), 0x20)
    )
    loop = scf.ForOp(zero, strip_nb, one, [arith.constant(i1, 0)])
    si = loop.body.arguments[0]
    found = loop.body.arguments[1]
    with ir.InsertionPoint(loop.body):
        sb = _byte_at(strip_data, si)
        eq = arith.cmpi(arith.CmpIPredicate.eq, byte_val, sb)
        scf.YieldOp([arith.ori(found, eq)])
    return arith.select(strip_empty, is_ws, loop.results[0])


def _strip_core(
    gpu_module: ir.Module,
    str_view: ir.Value,
    strip_view: ir.Value,
    do_left: bool,
    do_right: bool,
) -> ir.Value:
    """Shared strip/lstrip/rstrip: trim the matching run from each requested end.

    Scans from the left (and/or right) while the current byte is a strip char
    and no non-strip byte has yet been seen on that side, then allocates and
    copies the surviving ``[left_idx, right_idx)`` byte range.
    """
    data = _view_data(str_view)
    nb = llvm.zext(T.i64(), _view_nbytes(str_view))
    strip_data = _view_data(strip_view)
    strip_nb = llvm.zext(T.i64(), _view_nbytes(strip_view))
    zero = arith.constant(T.i64(), 0)
    one = arith.constant(T.i64(), 1)

    if do_left:
        loop_l = scf.ForOp(zero, nb, one, [zero])
        i = loop_l.body.arguments[0]
        left = loop_l.body.arguments[1]
        with ir.InsertionPoint(loop_l.body):
            should_strip = _is_strip_char(
                _byte_at(data, i), strip_data, strip_nb
            )
            still = arith.cmpi(arith.CmpIPredicate.eq, left, i)
            advance = arith.andi(should_strip, still)
            scf.YieldOp([arith.select(advance, arith.addi(left, one), left)])
        left_idx = loop_l.results[0]
    else:
        left_idx = zero

    if do_right:
        loop_r = scf.ForOp(zero, nb, one, [nb])
        j = loop_r.body.arguments[0]
        right = loop_r.body.arguments[1]
        with ir.InsertionPoint(loop_r.body):
            ridx = arith.subi(arith.subi(nb, j), one)
            should_strip = _is_strip_char(
                _byte_at(data, ridx), strip_data, strip_nb
            )
            expected = arith.subi(nb, j)
            still = arith.cmpi(arith.CmpIPredicate.eq, right, expected)
            advance = arith.andi(should_strip, still)
            scf.YieldOp([arith.select(advance, ridx, right)])
        right_idx = loop_r.results[0]
    else:
        right_idx = nb

    nonempty = arith.cmpi(arith.CmpIPredicate.ult, left_idx, right_idx)
    result_len = arith.select(nonempty, arith.subi(right_idx, left_idx), zero)
    mi, out_data = allocate_mlir_string(gpu_module, result_len)
    llvm.intr_memcpy(
        out_data, _gep_byte_offset(data, left_idx), result_len, False
    )
    return build_mlir_string(mi, out_data, result_len)


def lower_strip(
    gpu_module: ir.Module, str_view: ir.Value, strip_view: ir.Value
) -> ir.Value:
    """``s.strip(chars)`` -> trimmed ``mlir_string`` (both ends).

    Parameters
    ----------
    gpu_module : ir.Module
        The GPU module the NRT functions are declared in.
    str_view : ir.Value
        View of the source string's bytes.
    strip_view : ir.Value
        View of the strip-character set (empty for whitespace mode).

    Returns
    -------
    ir.Value
        The trimmed owned ``mlir_string``.
    """
    return _strip_core(gpu_module, str_view, strip_view, True, True)


def lower_lstrip(
    gpu_module: ir.Module, str_view: ir.Value, strip_view: ir.Value
) -> ir.Value:
    """``s.lstrip(chars)`` -> left-trimmed ``mlir_string``.

    Parameters
    ----------
    gpu_module : ir.Module
        The GPU module the NRT functions are declared in.
    str_view : ir.Value
        View of the source string's bytes.
    strip_view : ir.Value
        View of the strip-character set (empty for whitespace mode).

    Returns
    -------
    ir.Value
        The left-trimmed owned ``mlir_string``.
    """
    return _strip_core(gpu_module, str_view, strip_view, True, False)


def lower_rstrip(
    gpu_module: ir.Module, str_view: ir.Value, strip_view: ir.Value
) -> ir.Value:
    """``s.rstrip(chars)`` -> right-trimmed ``mlir_string``.

    Parameters
    ----------
    gpu_module : ir.Module
        The GPU module the NRT functions are declared in.
    str_view : ir.Value
        View of the source string's bytes.
    strip_view : ir.Value
        View of the strip-character set (empty for whitespace mode).

    Returns
    -------
    ir.Value
        The right-trimmed owned ``mlir_string``.
    """
    return _strip_core(gpu_module, str_view, strip_view, False, True)


def lower_replace(
    gpu_module: ir.Module,
    src: ir.Value,
    old: ir.Value,
    new: ir.Value,
) -> ir.Value:
    """``s.replace(old, new)`` -> every non-overlapping ``old`` replaced by ``new``.

    Byte-level substring matching (equivalent to character matching for
    well-formed UTF-8). Two passes: count occurrences to size the result, then
    scan-and-copy emitting replacement or source bytes. An empty ``old`` inserts
    ``new`` around every byte, matching libcudf's replace.

    Parameters
    ----------
    gpu_module : ir.Module
        The GPU module the NRT functions are declared in.
    src : ir.Value
        View of the source string's bytes.
    old : ir.Value
        View of the substring to replace.
    new : ir.Value
        View of the replacement substring.

    Returns
    -------
    ir.Value
        The owned ``mlir_string`` with replacements applied.
    """
    i1 = ir.IntegerType.get_signless(1)
    s_data, o_data, n_data = _view_data(src), _view_data(old), _view_data(new)
    s_nb = llvm.zext(T.i64(), _view_nbytes(src))
    o_nb = llvm.zext(T.i64(), _view_nbytes(old))
    n_nb = llvm.zext(T.i64(), _view_nbytes(new))

    zero = arith.constant(T.i64(), 0)
    one = arith.constant(T.i64(), 1)
    false_val = arith.constant(i1, 0)
    eq = arith.CmpIPredicate.eq
    ult = arith.CmpIPredicate.ult
    uge = arith.CmpIPredicate.uge

    target_empty = arith.cmpi(eq, o_nb, zero)

    # --- Pass 1: count non-overlapping occurrences of ``old`` ---
    long_enough = arith.cmpi(uge, s_nb, o_nb)
    diff = arith.subi(s_nb, o_nb)
    search_range = arith.select(long_enough, arith.addi(diff, one), zero)

    count_loop = scf.ForOp(zero, search_range, one, [zero, zero])
    ci = count_loop.body.arguments[0]
    c_count = count_loop.body.arguments[1]
    c_skip = count_loop.body.arguments[2]
    with ir.InsertionPoint(count_loop.body):
        c_should_skip = arith.cmpi(ult, ci, c_skip)
        c_inner = scf.ForOp(zero, o_nb, one, [zero])
        cj = c_inner.body.arguments[0]
        cmm = c_inner.body.arguments[1]
        with ir.InsertionPoint(c_inner.body):
            c_ne = arith.cmpi(
                arith.CmpIPredicate.ne,
                _byte_at(s_data, arith.addi(ci, cj)),
                _byte_at(o_data, cj),
            )
            scf.YieldOp([arith.addi(cmm, arith.extui(T.i64(), c_ne))])
        c_matches = arith.cmpi(eq, c_inner.results[0], zero)
        c_valid = arith.andi(
            c_matches, arith.cmpi(eq, c_should_skip, false_val)
        )
        c_new_count = arith.select(c_valid, arith.addi(c_count, one), c_count)
        c_new_skip = arith.select(c_valid, arith.addi(ci, o_nb), c_skip)
        scf.YieldOp([c_new_count, c_new_skip])

    occ_count = count_loop.results[0]

    # result_len = s_nb - occ_count * o_nb + occ_count * n_nb
    result_len = arith.addi(
        arith.subi(s_nb, arith.muli(occ_count, o_nb)),
        arith.muli(occ_count, n_nb),
    )

    mi, out_data = allocate_mlir_string(gpu_module, result_len)
    scratch_one = llvm.alloca(
        llvm.PointerType.get(), arith.constant(T.i64(), 1), elem_type=T.i8()
    )

    # --- Pass 2: scan and copy, emitting replacement and/or source bytes ---
    build_loop = scf.ForOp(zero, s_nb, one, [zero, zero])
    si = build_loop.body.arguments[0]
    out_off = build_loop.body.arguments[1]
    b_skip = build_loop.body.arguments[2]
    with ir.InsertionPoint(build_loop.body):
        active = arith.cmpi(eq, arith.cmpi(ult, si, b_skip), false_val)
        can_match = arith.cmpi(uge, arith.subi(s_nb, si), o_nb)
        b_inner = scf.ForOp(zero, o_nb, one, [zero])
        bj = b_inner.body.arguments[0]
        bmm = b_inner.body.arguments[1]
        with ir.InsertionPoint(b_inner.body):
            # Clamp the source index to the last byte: near the end ``si + bj``
            # can exceed ``s_nb``; ``can_match`` already gates such positions
            # out, so the clamped compare only avoids the out-of-bounds read.
            b_hidx = arith.minui(arith.addi(si, bj), arith.subi(s_nb, one))
            b_ne = arith.cmpi(
                arith.CmpIPredicate.ne,
                _byte_at(s_data, b_hidx),
                _byte_at(o_data, bj),
            )
            scf.YieldOp([arith.addi(bmm, arith.extui(T.i64(), b_ne))])
        b_match_valid = arith.andi(
            arith.andi(can_match, arith.cmpi(eq, b_inner.results[0], zero)),
            active,
        )
        nonempty_match = arith.andi(
            b_match_valid, arith.cmpi(eq, target_empty, false_val)
        )
        write_repl = arith.ori(
            arith.andi(target_empty, active), nonempty_match
        )
        write_byte = arith.ori(
            arith.andi(target_empty, active),
            arith.andi(arith.cmpi(eq, nonempty_match, false_val), active),
        )

        len_a = arith.select(write_repl, n_nb, zero)
        llvm.intr_memcpy(
            _gep_byte_offset(out_data, out_off), n_data, len_a, False
        )
        off_after_a = arith.addi(out_off, len_a)

        llvm.store(_byte_at(s_data, si), scratch_one)
        len_b = arith.select(write_byte, one, zero)
        llvm.intr_memcpy(
            _gep_byte_offset(out_data, off_after_a), scratch_one, len_b, False
        )
        new_out_off = arith.addi(off_after_a, len_b)
        new_skip = arith.select(nonempty_match, arith.addi(si, o_nb), b_skip)
        scf.YieldOp([new_out_off, new_skip])

    # Empty target: ``new`` is also inserted after the last character.
    trailing_len = arith.select(target_empty, n_nb, zero)
    llvm.intr_memcpy(
        _gep_byte_offset(out_data, build_loop.results[0]),
        n_data,
        trailing_len,
        False,
    )

    return build_mlir_string(mi, out_data, result_len)


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
    # Skip the scan entirely for an empty target (handled by the result select
    # below); this also avoids the char-position load reading one byte past the
    # source when ``search_len`` would otherwise be ``src_nb + 1``.
    search_len = arith.select(
        arith.ori(too_long, tgt_empty),
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
    # Skip the scan entirely for an empty target (handled by the result select
    # below); this also avoids the char-position load reading one byte past the
    # source when ``search_len`` would otherwise be ``src_nb + 1``.
    search_len = arith.select(
        arith.ori(too_long, tgt_empty),
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


# --- character-class predicates (libcudf flags table) ----------------------
# Flag bits sourced directly from the public ``cudf::strings`` enum so the
# values stay in sync with libcudf rather than being duplicated here.
_CT_DECIMAL = int(StringCharacterTypes.DECIMAL)
_CT_NUMERIC = int(StringCharacterTypes.NUMERIC)
_CT_DIGIT = int(StringCharacterTypes.DIGIT)
_CT_ALPHA = int(StringCharacterTypes.ALPHA)
_CT_SPACE = int(StringCharacterTypes.SPACE)
_CT_UPPER = int(StringCharacterTypes.UPPER)
_CT_LOWER = int(StringCharacterTypes.LOWER)
_CT_ALPHANUM = int(StringCharacterTypes.ALPHANUM)
_CT_CASE_TYPES = int(StringCharacterTypes.CASE_TYPES)
_CT_ALL_TYPES = int(StringCharacterTypes.ALL_TYPES)
# The SPECIAL bit (0x80) is not part of the public StringCharacterTypes enum;
# it marks characters whose case conversion lives in the special-case table.
_CT_SPECIAL = 1 << 7
# Prime used to index the special-case mapping table (cudf::strings::detail).
_SPECIAL_CASE_PRIME = 499


def _utf8_byte_width(first_byte: ir.Value) -> ir.Value:
    """Width in ``[1, 4]`` (i64) of the UTF-8 char starting at ``first_byte``."""
    fb = arith.extui(T.i64(), first_byte)

    def _is(mask: int) -> ir.Value:
        return arith.cmpi(
            arith.CmpIPredicate.eq,
            arith.andi(fb, arith.constant(T.i64(), mask)),
            arith.constant(T.i64(), mask),
        )

    width = arith.constant(T.i64(), 1)
    width = arith.select(_is(0xC0), arith.constant(T.i64(), 2), width)
    width = arith.select(_is(0xE0), arith.constant(T.i64(), 3), width)
    return arith.select(_is(0xF0), arith.constant(T.i64(), 4), width)


def _decode_utf8_to_codepoint(
    data_ptr: ir.Value, byte_offset: ir.Value, nbytes_i64: ir.Value
) -> tuple[ir.Value, ir.Value]:
    """Decode the UTF-8 char at ``data_ptr[byte_offset]``.

    Returns ``(codepoint_i32, width_i64)``. Every byte load (including the first)
    is clamped to ``nbytes - 1`` so decoding never reads past the buffer, even
    for a truncated final character.
    """
    i32 = ir.IntegerType.get_signless(32)
    last_valid = arith.subi(nbytes_i64, arith.constant(T.i64(), 1))
    byte_offset = arith.minui(byte_offset, last_valid)
    b0 = _byte_at(data_ptr, byte_offset)
    width = _utf8_byte_width(b0)
    b0_32 = arith.extui(i32, b0)

    def _cont(step: int) -> ir.Value:
        off = arith.minui(
            arith.addi(byte_offset, arith.constant(T.i64(), step)), last_valid
        )
        return arith.extui(i32, _byte_at(data_ptr, off))

    def _c(val: int) -> ir.Value:
        return arith.constant(i32, val)

    b1, b2, b3 = _cont(1), _cont(2), _cont(3)
    cp1 = arith.andi(b0_32, _c(0x7F))
    cp2 = arith.ori(
        arith.shli(arith.andi(b0_32, _c(0x1F)), _c(6)),
        arith.andi(b1, _c(0x3F)),
    )
    cp3 = arith.ori(
        arith.ori(
            arith.shli(arith.andi(b0_32, _c(0x0F)), _c(12)),
            arith.shli(arith.andi(b1, _c(0x3F)), _c(6)),
        ),
        arith.andi(b2, _c(0x3F)),
    )
    cp4 = arith.ori(
        arith.ori(
            arith.shli(arith.andi(b0_32, _c(0x07)), _c(18)),
            arith.shli(arith.andi(b1, _c(0x3F)), _c(12)),
        ),
        arith.ori(
            arith.shli(arith.andi(b2, _c(0x3F)), _c(6)),
            arith.andi(b3, _c(0x3F)),
        ),
    )

    is1 = arith.cmpi(arith.CmpIPredicate.eq, width, arith.constant(T.i64(), 1))
    is2 = arith.cmpi(arith.CmpIPredicate.eq, width, arith.constant(T.i64(), 2))
    is3 = arith.cmpi(arith.CmpIPredicate.eq, width, arith.constant(T.i64(), 3))
    cp = cp4
    cp = arith.select(is3, cp3, cp)
    cp = arith.select(is2, cp2, cp)
    cp = arith.select(is1, cp1, cp)
    return cp, width


def _all_characters_of_type(
    str_view: ir.Value,
    flags_table_ptr: ir.Value,
    types_i32: ir.Value,
    verify_types_i32: ir.Value,
) -> ir.Value:
    """i1: mirror ``cudf::strings::udf::all_characters_of_type``.

    ``flags_table_ptr`` is a device pointer to the ``uint8`` character-flags
    table; ``types``/``verify_types`` are character-class bit masks.
    """
    i32 = ir.IntegerType.get_signless(32)
    i1 = ir.IntegerType.get_signless(1)
    data = _view_data(str_view)
    nb = llvm.zext(T.i64(), _view_nbytes(str_view))
    zero_i32 = arith.constant(i32, 0)
    true_i1 = arith.constant(i1, 1)
    max_cp = arith.constant(i32, 0xFFFF)
    all_types = arith.constant(i32, _CT_ALL_TYPES)
    is_empty = arith.cmpi(
        arith.CmpIPredicate.eq, nb, arith.constant(T.i64(), 0)
    )

    # Loop state: [byte_offset, check, check_count].
    loop = scf.ForOp(
        arith.constant(T.i64(), 0),
        nb,
        arith.constant(T.i64(), 1),
        [arith.constant(T.i64(), 0), true_i1, zero_i32],
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        cur_off = loop.inner_iter_args[0]
        cur_check = loop.inner_iter_args[1]
        cur_count = loop.inner_iter_args[2]

        at_boundary = arith.cmpi(arith.CmpIPredicate.eq, idx, cur_off)
        cp, char_width = _decode_utf8_to_codepoint(data, cur_off, nb)
        in_range = arith.cmpi(arith.CmpIPredicate.ule, cp, max_cp)
        # Clamp the table index to the BMP before loading; out-of-range code
        # points get flag 0 (so the load never runs past the 64K table).
        safe_cp = arith.select(in_range, cp, zero_i32)
        flag = arith.extui(
            i32, _byte_at(flags_table_ptr, arith.extui(T.i64(), safe_cp))
        )
        flag = arith.select(in_range, flag, zero_i32)

        should_verify = arith.ori(
            arith.cmpi(
                arith.CmpIPredicate.ne,
                arith.andi(verify_types_i32, flag),
                zero_i32,
            ),
            arith.andi(
                arith.cmpi(arith.CmpIPredicate.eq, flag, zero_i32),
                arith.cmpi(
                    arith.CmpIPredicate.eq, verify_types_i32, all_types
                ),
            ),
        )
        type_match = arith.cmpi(
            arith.CmpIPredicate.ne, arith.andi(types_i32, flag), zero_i32
        )
        new_check = arith.select(
            should_verify, arith.andi(cur_check, type_match), cur_check
        )
        new_count = arith.select(
            should_verify,
            arith.addi(cur_count, arith.constant(i32, 1)),
            cur_count,
        )
        final_check = arith.select(at_boundary, new_check, cur_check)
        final_count = arith.select(at_boundary, new_count, cur_count)
        next_off = arith.select(
            at_boundary, arith.addi(cur_off, char_width), cur_off
        )
        scf.yield_([next_off, final_check, final_count])

    has_chars = arith.cmpi(arith.CmpIPredicate.sgt, loop.results[2], zero_i32)
    return arith.andi(
        arith.andi(loop.results[1], has_chars),
        arith.xori(is_empty, true_i1),
    )


def _lower_isalpha(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isalpha`` -> ``i1``."""
    i32 = ir.IntegerType.get_signless(32)
    return _all_characters_of_type(
        str_view,
        flags,
        arith.constant(i32, _CT_ALPHA),
        arith.constant(i32, _CT_ALL_TYPES),
    )


def _lower_isalnum(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isalnum`` -> ``i1``."""
    i32 = ir.IntegerType.get_signless(32)
    return _all_characters_of_type(
        str_view,
        flags,
        arith.constant(i32, _CT_ALPHANUM),
        arith.constant(i32, _CT_ALL_TYPES),
    )


def _lower_isdigit(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isdigit`` -> ``i1``."""
    i32 = ir.IntegerType.get_signless(32)
    return _all_characters_of_type(
        str_view,
        flags,
        arith.constant(i32, _CT_DIGIT),
        arith.constant(i32, _CT_ALL_TYPES),
    )


def _lower_isdecimal(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isdecimal`` -> ``i1``."""
    i32 = ir.IntegerType.get_signless(32)
    return _all_characters_of_type(
        str_view,
        flags,
        arith.constant(i32, _CT_DECIMAL),
        arith.constant(i32, _CT_ALL_TYPES),
    )


def _lower_isnumeric(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isnumeric`` -> ``i1``."""
    i32 = ir.IntegerType.get_signless(32)
    return _all_characters_of_type(
        str_view,
        flags,
        arith.constant(i32, _CT_NUMERIC),
        arith.constant(i32, _CT_ALL_TYPES),
    )


def _lower_isspace(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isspace`` -> ``i1``."""
    i32 = ir.IntegerType.get_signless(32)
    return _all_characters_of_type(
        str_view,
        flags,
        arith.constant(i32, _CT_SPACE),
        arith.constant(i32, _CT_ALL_TYPES),
    )


def _lower_isupper(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``isupper`` -> ``i1``."""
    i32 = ir.IntegerType.get_signless(32)
    return _all_characters_of_type(
        str_view,
        flags,
        arith.constant(i32, _CT_UPPER),
        arith.constant(i32, _CT_CASE_TYPES),
    )


def _lower_islower(str_view: ir.Value, flags: ir.Value) -> ir.Value:
    """``islower`` -> ``i1``."""
    i32 = ir.IntegerType.get_signless(32)
    return _all_characters_of_type(
        str_view,
        flags,
        arith.constant(i32, _CT_LOWER),
        arith.constant(i32, _CT_CASE_TYPES),
    )


def _lower_istitle(str_view: ir.Value, flags_table_ptr: ir.Value) -> ir.Value:
    """``istitle`` -> ``i1``. Mirrors ``cudf::strings::udf::is_title``."""
    i32 = ir.IntegerType.get_signless(32)
    i1 = ir.IntegerType.get_signless(1)
    data = _view_data(str_view)
    nb = llvm.zext(T.i64(), _view_nbytes(str_view))
    zero_i32 = arith.constant(i32, 0)
    true_i1 = arith.constant(i1, 1)
    false_i1 = arith.constant(i1, 0)
    max_cp = arith.constant(i32, 0xFFFF)
    upper_mask = arith.constant(i32, _CT_UPPER)
    case_mask = arith.constant(i32, _CT_CASE_TYPES)

    # Loop state: [byte_offset, valid (saw a cased char), should_be_cap, ok].
    loop = scf.ForOp(
        arith.constant(T.i64(), 0),
        nb,
        arith.constant(T.i64(), 1),
        [arith.constant(T.i64(), 0), false_i1, true_i1, true_i1],
    )
    with ir.InsertionPoint(loop.body):
        idx = loop.induction_variable
        cur_off = loop.inner_iter_args[0]
        cur_valid = loop.inner_iter_args[1]
        cur_should_cap = loop.inner_iter_args[2]
        cur_ok = loop.inner_iter_args[3]

        at_boundary = arith.cmpi(arith.CmpIPredicate.eq, idx, cur_off)
        cp, char_width = _decode_utf8_to_codepoint(data, cur_off, nb)
        in_range = arith.cmpi(arith.CmpIPredicate.ule, cp, max_cp)
        safe_cp = arith.select(in_range, cp, zero_i32)
        flag = arith.extui(
            i32, _byte_at(flags_table_ptr, arith.extui(T.i64(), safe_cp))
        )
        flag = arith.select(in_range, flag, zero_i32)

        is_upper = arith.cmpi(
            arith.CmpIPredicate.ne, arith.andi(flag, upper_mask), zero_i32
        )
        is_cased = arith.cmpi(
            arith.CmpIPredicate.ne, arith.andi(flag, case_mask), zero_i32
        )
        wrong_case = arith.xori(cur_should_cap, is_upper)
        new_ok = arith.select(
            arith.andi(is_cased, wrong_case), false_i1, cur_ok
        )
        new_valid = arith.select(is_cased, true_i1, cur_valid)
        new_should_cap = arith.select(is_cased, false_i1, true_i1)

        final_ok = arith.select(at_boundary, new_ok, cur_ok)
        final_valid = arith.select(at_boundary, new_valid, cur_valid)
        final_should_cap = arith.select(
            at_boundary, new_should_cap, cur_should_cap
        )
        next_off = arith.select(
            at_boundary, arith.addi(cur_off, char_width), cur_off
        )
        scf.yield_([next_off, final_valid, final_should_cap, final_ok])

    return arith.andi(loop.results[1], loop.results[3])


# --- getitem (character-indexed substring; allocates via NRT) ---------------
def _count_chars(data: ir.Value, nbytes_i64: ir.Value) -> ir.Value:
    """Number of UTF-8 characters (start bytes) in ``data[:nbytes]`` as i64."""
    zero = arith.constant(T.i64(), 0)
    one = arith.constant(T.i64(), 1)
    loop = scf.ForOp(zero, nbytes_i64, one, [zero])
    i = loop.body.arguments[0]
    count = loop.body.arguments[1]
    with ir.InsertionPoint(loop.body):
        is_start = _is_begin_utf8_char(_byte_at(data, i))
        inc = arith.select(is_start, one, zero)
        scf.YieldOp([arith.addi(count, inc)])
    return loop.results[0]


def _char_to_byte(
    data: ir.Value, nbytes_i64: ir.Value, char_pos: ir.Value
) -> ir.Value:
    """Byte offset where the ``char_pos``-th character starts.

    Returns ``nbytes`` when ``char_pos`` is at or past the end, so a
    one-past-the-end position maps to the buffer length (used as a slice stop).
    """
    zero = arith.constant(T.i64(), 0)
    one = arith.constant(T.i64(), 1)
    loop = scf.ForOp(zero, nbytes_i64, one, [zero, nbytes_i64])
    i = loop.body.arguments[0]
    count = loop.body.arguments[1]
    result = loop.body.arguments[2]
    with ir.InsertionPoint(loop.body):
        is_start = _is_begin_utf8_char(_byte_at(data, i))
        at_target = arith.andi(
            is_start, arith.cmpi(arith.CmpIPredicate.eq, count, char_pos)
        )
        new_result = arith.select(at_target, i, result)
        new_count = arith.select(is_start, arith.addi(count, one), count)
        scf.YieldOp([new_count, new_result])
    return loop.results[1]


def lower_getitem_int(
    gpu_module: ir.Module, str_view: ir.Value, idx_i64: ir.Value
) -> ir.Value:
    """``s[i]`` -> the single character at character position ``i``.

    Negative indices count from the end; out-of-range indices yield an empty
    string (UDFs cannot raise ``IndexError``). Allocates and copies the byte
    range of the selected UTF-8 character.

    Parameters
    ----------
    gpu_module : ir.Module
        The GPU module the NRT functions are declared in.
    str_view : ir.Value
        View of the source string's bytes.
    idx_i64 : ir.Value
        Character index (``i64``), possibly negative.

    Returns
    -------
    ir.Value
        The selected single-character owned ``mlir_string``.
    """
    zero = arith.constant(T.i64(), 0)
    one = arith.constant(T.i64(), 1)
    data = _view_data(str_view)
    nbytes = llvm.zext(T.i64(), _view_nbytes(str_view))
    nchars = _count_chars(data, nbytes)

    neg = arith.cmpi(arith.CmpIPredicate.slt, idx_i64, zero)
    idx = arith.select(neg, arith.addi(idx_i64, nchars), idx_i64)
    # An index still out of range after normalization (below -nchars or
    # >= nchars) yields an empty string; only an in-range index selects a
    # character. Clamping with max/min alone would wrongly map a very negative
    # index to the first character.
    in_range = arith.andi(
        arith.cmpi(arith.CmpIPredicate.sge, idx, zero),
        arith.cmpi(arith.CmpIPredicate.slt, idx, nchars),
    )
    start = arith.select(in_range, idx, nchars)
    stop = arith.select(in_range, arith.addi(idx, one), nchars)

    byte_start = _char_to_byte(data, nbytes, start)
    byte_stop = _char_to_byte(data, nbytes, stop)
    result_len = arith.subi(byte_stop, byte_start)

    mi, out_data = allocate_mlir_string(gpu_module, result_len)
    llvm.intr_memcpy(
        out_data, _gep_byte_offset(data, byte_start), result_len, False
    )
    return build_mlir_string(mi, out_data, result_len)


def lower_getitem_slice(
    gpu_module: ir.Module,
    str_view: ir.Value,
    start_i64: ir.Value,
    stop_i64: ir.Value,
) -> ir.Value:
    """``s[a:b]`` -> the substring over character positions ``[a, b)``.

    Negative bounds count from the end; both bounds are normalized and clamped
    to ``[0, nchars]`` (with ``stop >= start``), so the copied byte range is
    always in bounds. ``step`` is not supported (handled at the typing layer).

    Parameters
    ----------
    gpu_module : ir.Module
        The GPU module the NRT functions are declared in.
    str_view : ir.Value
        View of the source string's bytes.
    start_i64 : ir.Value
        Start character position (``i64``), possibly negative.
    stop_i64 : ir.Value
        Stop character position (``i64``), possibly negative.

    Returns
    -------
    ir.Value
        The substring owned ``mlir_string``.
    """
    zero = arith.constant(T.i64(), 0)
    data = _view_data(str_view)
    nbytes = llvm.zext(T.i64(), _view_nbytes(str_view))
    nchars = _count_chars(data, nbytes)

    def _norm(pos: ir.Value) -> ir.Value:
        neg = arith.cmpi(arith.CmpIPredicate.slt, pos, zero)
        return arith.select(neg, arith.addi(pos, nchars), pos)

    start = arith.maxsi(zero, arith.minsi(_norm(start_i64), nchars))
    stop = arith.maxsi(start, arith.minsi(_norm(stop_i64), nchars))

    byte_start = _char_to_byte(data, nbytes, start)
    byte_stop = _char_to_byte(data, nbytes, stop)
    result_len = arith.subi(byte_stop, byte_start)

    mi, out_data = allocate_mlir_string(gpu_module, result_len)
    llvm.intr_memcpy(
        out_data, _gep_byte_offset(data, byte_start), result_len, False
    )
    return build_mlir_string(mi, out_data, result_len)


# --- case conversion (upper/lower via the Unicode cases tables) ------------
def _codepoint_to_utf8_bytes(cp_i32: ir.Value) -> ir.Value:
    """UTF-8 byte width (i64) of code point ``cp_i32``."""
    i32 = ir.IntegerType.get_signless(32)
    is1 = arith.cmpi(
        arith.CmpIPredicate.ult, cp_i32, arith.constant(i32, 0x80)
    )
    is2 = arith.cmpi(
        arith.CmpIPredicate.ult, cp_i32, arith.constant(i32, 0x800)
    )
    is3 = arith.cmpi(
        arith.CmpIPredicate.ult, cp_i32, arith.constant(i32, 0x10000)
    )
    n = arith.constant(T.i64(), 4)
    n = arith.select(is3, arith.constant(T.i64(), 3), n)
    n = arith.select(is2, arith.constant(T.i64(), 2), n)
    return arith.select(is1, arith.constant(T.i64(), 1), n)


def _write_codepoint_utf8(
    data_ptr: ir.Value, byte_offset: ir.Value, cp_i32: ir.Value
) -> ir.Value:
    """Encode ``cp_i32`` as UTF-8 at ``data_ptr[byte_offset]``; return width (i64).

    All four byte positions are written unconditionally (so there is no
    conditional side effect); only the returned width reflects the real length,
    and the caller copies exactly that many bytes.
    """
    i32 = ir.IntegerType.get_signless(32)

    def _c(v: int) -> ir.Value:
        return arith.constant(i32, v)

    nbytes = _codepoint_to_utf8_bytes(cp_i32)
    is1 = arith.cmpi(
        arith.CmpIPredicate.eq, nbytes, arith.constant(T.i64(), 1)
    )
    is2 = arith.cmpi(
        arith.CmpIPredicate.eq, nbytes, arith.constant(T.i64(), 2)
    )
    is3 = arith.cmpi(
        arith.CmpIPredicate.eq, nbytes, arith.constant(T.i64(), 3)
    )

    def _t(v: ir.Value) -> ir.Value:
        return arith.trunci(T.i8(), v)

    b1_0 = _t(cp_i32)
    b2_0 = _t(arith.ori(arith.shrui(cp_i32, _c(6)), _c(0xC0)))
    b2_1 = _t(arith.ori(arith.andi(cp_i32, _c(0x3F)), _c(0x80)))
    b3_0 = _t(arith.ori(arith.shrui(cp_i32, _c(12)), _c(0xE0)))
    b3_1 = _t(
        arith.ori(arith.andi(arith.shrui(cp_i32, _c(6)), _c(0x3F)), _c(0x80))
    )
    b3_2 = _t(arith.ori(arith.andi(cp_i32, _c(0x3F)), _c(0x80)))
    b4_0 = _t(arith.ori(arith.shrui(cp_i32, _c(18)), _c(0xF0)))
    b4_1 = _t(
        arith.ori(arith.andi(arith.shrui(cp_i32, _c(12)), _c(0x3F)), _c(0x80))
    )
    b4_2 = _t(
        arith.ori(arith.andi(arith.shrui(cp_i32, _c(6)), _c(0x3F)), _c(0x80))
    )
    b4_3 = _t(arith.ori(arith.andi(cp_i32, _c(0x3F)), _c(0x80)))

    byte0 = arith.select(
        is1, b1_0, arith.select(is2, b2_0, arith.select(is3, b3_0, b4_0))
    )
    byte1 = arith.select(is2, b2_1, arith.select(is3, b3_1, b4_1))
    byte2 = arith.select(is3, b3_2, b4_2)
    byte3 = b4_3

    def _dst(step: int) -> ir.Value:
        return _gep_byte_offset(
            data_ptr, arith.addi(byte_offset, arith.constant(T.i64(), step))
        )

    llvm.store(byte0, _gep_byte_offset(data_ptr, byte_offset))
    llvm.store(byte1, _dst(1))
    llvm.store(byte2, _dst(2))
    llvm.store(byte3, _dst(3))
    return nbytes


def _case_convert_char_loop(
    data_ptr: ir.Value,
    nbytes_i64: ir.Value,
    flags_ptr: ir.Value,
    cases_ptr: ir.Value,
    special_ptr: ir.Value,
    case_flag_i8: ir.Value,
    is_upper: bool,
    out_ptr: ir.Value | None,
) -> ir.Value:
    """Per-character case-conversion scan; returns total output bytes (i64).

    With ``out_ptr is None`` this counts output bytes (sizing pass); otherwise it
    writes converted bytes. Characters flagged with ``case_flag_i8`` convert via
    the cases table; characters flagged SPECIAL (and not upper/lower) use the
    special-case mapping table (one code point to several). Each path writes into
    its own scratch buffer and a single clamped ``memcpy`` emits the chosen
    result, so there is no conditional side effect.
    """
    i32 = ir.IntegerType.get_signless(32)
    i16 = ir.IntegerType.get_signless(16)
    zero = arith.constant(T.i64(), 0)
    one = arith.constant(T.i64(), 1)
    three = arith.constant(T.i64(), 3)
    false_i1 = arith.constant(ir.IntegerType.get_signless(1), 0)

    def _i8(v: int) -> ir.Value:
        return arith.constant(T.i8(), v)

    if out_ptr is not None:
        scratch_normal = llvm.alloca(
            llvm.PointerType.get(),
            arith.constant(T.i64(), 4),
            elem_type=T.i8(),
        )
        scratch_special = llvm.alloca(
            llvm.PointerType.get(),
            arith.constant(T.i64(), 12),
            elem_type=T.i8(),
        )

    loop = scf.ForOp(zero, nbytes_i64, one, [zero, zero])
    pos = loop.body.arguments[0]
    out_off = loop.body.arguments[1]
    skip_until = loop.body.arguments[2]
    with ir.InsertionPoint(loop.body):
        should_skip = arith.cmpi(arith.CmpIPredicate.ult, pos, skip_until)
        cp, width = _decode_utf8_to_codepoint(data_ptr, pos, nbytes_i64)
        cp_i64 = arith.extui(T.i64(), cp)

        in_range = arith.cmpi(
            arith.CmpIPredicate.ule, cp, arith.constant(i32, 0xFFFF)
        )
        flag_raw = llvm.load(T.i8(), _gep_byte_offset(flags_ptr, cp_i64))
        flag = arith.select(in_range, flag_raw, _i8(0))

        matches_case = arith.cmpi(
            arith.CmpIPredicate.ne, arith.andi(flag, case_flag_i8), _i8(0)
        )
        is_special = arith.cmpi(
            arith.CmpIPredicate.ne, arith.andi(flag, _i8(_CT_SPECIAL)), _i8(0)
        )
        is_upper_or_lower = arith.cmpi(
            arith.CmpIPredicate.ne,
            arith.andi(flag, _i8(_CT_UPPER | _CT_LOWER)),
            _i8(0),
        )
        special_only = arith.andi(
            is_special,
            arith.cmpi(arith.CmpIPredicate.eq, is_upper_or_lower, false_i1),
        )
        needs_convert = arith.ori(matches_case, special_only)

        # non-special path: cases_table[cp] (u16 entries)
        case_cp_i16 = llvm.load(
            i16, _gep_byte_offset(cases_ptr, arith.shli(cp_i64, one))
        )
        case_cp_i32 = arith.extui(i32, case_cp_i16)
        normal_out_bytes = _codepoint_to_utf8_bytes(case_cp_i32)
        normal_new_off = arith.addi(out_off, normal_out_bytes)

        # special path: special_table[cp % PRIME], 16-byte entries
        hash_i64 = arith.extui(
            T.i64(),
            arith.remui(cp, arith.constant(i32, _SPECIAL_CASE_PRIME)),
        )
        entry_ptr = _gep_byte_offset(
            special_ptr, arith.muli(hash_i64, arith.constant(T.i64(), 16))
        )
        count_offset = (
            arith.constant(T.i64(), 0)
            if is_upper
            else arith.constant(T.i64(), 8)
        )
        num_chars_raw = arith.extui(
            T.i64(), llvm.load(i16, _gep_byte_offset(entry_ptr, count_offset))
        )
        num_chars = arith.select(
            arith.cmpi(arith.CmpIPredicate.ugt, num_chars_raw, three),
            three,
            num_chars_raw,
        )
        chars_base = _gep_byte_offset(
            entry_ptr,
            arith.constant(T.i64(), 2 if is_upper else 10),
        )

        sc_count = scf.ForOp(zero, num_chars, one, [zero])
        sc_ci = sc_count.body.arguments[0]
        sc_cacc = sc_count.body.arguments[1]
        with ir.InsertionPoint(sc_count.body):
            sc_cp = arith.extui(
                i32,
                llvm.load(
                    i16,
                    _gep_byte_offset(chars_base, arith.shli(sc_ci, one)),
                ),
            )
            scf.YieldOp([arith.addi(sc_cacc, _codepoint_to_utf8_bytes(sc_cp))])
        special_out_bytes = sc_count.results[0]

        converted_off = arith.select(
            is_special,
            arith.addi(out_off, special_out_bytes),
            normal_new_off,
        )
        new_off = arith.select(
            needs_convert, converted_off, arith.addi(out_off, width)
        )
        final_out_off = arith.select(should_skip, out_off, new_off)
        new_skip = arith.select(
            should_skip, skip_until, arith.addi(pos, width)
        )

        if out_ptr is not None:
            _write_codepoint_utf8(scratch_normal, zero, case_cp_i32)
            sc_wr = scf.ForOp(zero, num_chars, one, [zero])
            sc_wi = sc_wr.body.arguments[0]
            sc_wacc = sc_wr.body.arguments[1]
            with ir.InsertionPoint(sc_wr.body):
                sc_wcp = arith.extui(
                    i32,
                    llvm.load(
                        i16,
                        _gep_byte_offset(chars_base, arith.shli(sc_wi, one)),
                    ),
                )
                scf.YieldOp(
                    [
                        arith.addi(
                            sc_wacc,
                            _write_codepoint_utf8(
                                scratch_special, sc_wacc, sc_wcp
                            ),
                        )
                    ]
                )
            conv_src = arith.select(
                is_special, scratch_special, scratch_normal
            )
            chosen_src = arith.select(
                needs_convert, conv_src, _gep_byte_offset(data_ptr, pos)
            )
            chosen_len = arith.select(
                needs_convert,
                arith.select(is_special, special_out_bytes, normal_out_bytes),
                width,
            )
            safe_len = arith.select(should_skip, zero, chosen_len)
            llvm.intr_memcpy(
                _gep_byte_offset(out_ptr, out_off), chosen_src, safe_len, False
            )

        scf.YieldOp([final_out_off, new_skip])

    return loop.results[0]


def _lower_case(
    gpu_module: ir.Module,
    str_view: ir.Value,
    flags_ptr: ir.Value,
    cases_ptr: ir.Value,
    special_ptr: ir.Value,
    case_flag: int,
    is_upper: bool,
) -> ir.Value:
    """Shared upper/lower: size, allocate, then convert.

    ``flags_ptr``/``cases_ptr``/``special_ptr`` are device pointers to the
    libcudf flags, cases, and special-case tables.
    """
    data = _view_data(str_view)
    nb = llvm.zext(T.i64(), _view_nbytes(str_view))
    case_flag_i8 = arith.constant(T.i8(), case_flag)
    out_nbytes = _case_convert_char_loop(
        data,
        nb,
        flags_ptr,
        cases_ptr,
        special_ptr,
        case_flag_i8,
        is_upper,
        None,
    )
    mi, out_data = allocate_mlir_string(gpu_module, out_nbytes)
    _case_convert_char_loop(
        data,
        nb,
        flags_ptr,
        cases_ptr,
        special_ptr,
        case_flag_i8,
        is_upper,
        out_data,
    )
    return build_mlir_string(mi, out_data, out_nbytes)


def lower_upper(
    gpu_module: ir.Module,
    str_view: ir.Value,
    flags_ptr: ir.Value,
    cases_ptr: ir.Value,
    special_ptr: ir.Value,
) -> ir.Value:
    """``s.upper()`` -> uppercased ``mlir_string`` (converts lower-case chars).

    Parameters
    ----------
    gpu_module : ir.Module
        The GPU module the NRT functions are declared in.
    str_view : ir.Value
        View of the source string's bytes.
    flags_ptr : ir.Value
        Device pointer to the character-flags table.
    cases_ptr : ir.Value
        Device pointer to the character-cases table.
    special_ptr : ir.Value
        Device pointer to the special-case mapping table.

    Returns
    -------
    ir.Value
        The uppercased owned ``mlir_string``.
    """
    return _lower_case(
        gpu_module,
        str_view,
        flags_ptr,
        cases_ptr,
        special_ptr,
        _CT_LOWER,
        True,
    )


def lower_lower(
    gpu_module: ir.Module,
    str_view: ir.Value,
    flags_ptr: ir.Value,
    cases_ptr: ir.Value,
    special_ptr: ir.Value,
) -> ir.Value:
    """``s.lower()`` -> lowercased ``mlir_string`` (converts upper-case chars).

    Parameters
    ----------
    gpu_module : ir.Module
        The GPU module the NRT functions are declared in.
    str_view : ir.Value
        View of the source string's bytes.
    flags_ptr : ir.Value
        Device pointer to the character-flags table.
    cases_ptr : ir.Value
        Device pointer to the character-cases table.
    special_ptr : ir.Value
        Device pointer to the special-case mapping table.

    Returns
    -------
    ir.Value
        The lowercased owned ``mlir_string``.
    """
    return _lower_case(
        gpu_module,
        str_view,
        flags_ptr,
        cases_ptr,
        special_ptr,
        _CT_UPPER,
        False,
    )

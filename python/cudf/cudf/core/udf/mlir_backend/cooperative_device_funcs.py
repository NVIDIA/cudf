# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compile the cooperative reduction device functions to a single PTX "shim".

Every (reduction, dtype) combination is emitted as a public ``func.func``
(``numba_cuda_mlir._mlir.dialects.func``) into *one* ``gpu.module``, compiled
through the full MLIR->LLVM->PTX pipeline once per compute capability (cached),
and exposed as a :class:`PTXSource` via ``cooperative_reductions_linkable()``.
Callers link it into a kernel by passing
``link=[cooperative_reductions_linkable()]`` to ``cuda.jit`` -- mirroring how the
legacy backend linked its block-reduction shim. The kernel module only contains
declarations (no bodies), so the MLIR inliner cannot inline them and the linker
resolves the calls against this shim.

Shared-memory contract: the outlined functions receive the dynamic shared memory
base pointer as their last argument. gpu.dynamic_shared_memory only works inside
kernels; device functions must receive it externally.
"""

from contextlib import contextmanager
from functools import lru_cache

from numba_cuda_mlir import cuda
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import (
    arith,
    func,
    gpu,
    llvm,
    math as mlir_math,
    scf,
)
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir.passmanager import PassManager
from numba_cuda_mlir.lowering_utilities import context
from numba_cuda_mlir.numba_cuda import types as nb_types
from numba_cuda_mlir.numba_cuda.cudadrv.linkable_code import PTXSource

# ---------------------------------------------------------------------------
# Shared-memory layout. This is the single source of truth for the reduction
# scratch offset; the lowering launches each cooperative kernel with at least
# _SHM_REDUCTION_OFFSET + 2 * _MAX_BLOCK_SIZE * 8 bytes of dynamic shared memory
# (two f64 scratch arrays: value/sum + index/sum-of-squares).
# ---------------------------------------------------------------------------

_SHM_REDUCTION_OFFSET = 16  # bytes 16+:   reduction scratch (0..15 reserved)
_MAX_BLOCK_SIZE = 1024

# Dtypes with cooperative reduction support.
_DTYPES = (nb_types.float64, nb_types.float32, nb_types.int64, nb_types.int32)


# ---------------------------------------------------------------------------
# Helpers (self-contained — no builder dependency)
# ---------------------------------------------------------------------------


def _ptr():
    return llvm.PointerType.get()


def _ptr3():
    """Pointer in address space 3 (shared memory)."""
    return ir.Type.parse("!llvm.ptr<3>")


def _const_i64(val):
    return arith.constant(T.i64(), val)


def _dtype_mlir_type(dtype):
    if dtype == nb_types.float64:
        return ir.F64Type.get()
    if dtype == nb_types.float32:
        return ir.F32Type.get()
    if dtype == nb_types.int64:
        return T.i64()
    if dtype == nb_types.int32:
        return ir.IntegerType.get_signless(32)
    raise NotImplementedError(f"Unsupported dtype {dtype}")


def _dtype_tag(dtype):
    return {
        nb_types.float64: "f64",
        nb_types.float32: "f32",
        nb_types.int64: "i64",
        nb_types.int32: "i32",
    }[dtype]


def _zero_val(mlir_ty):
    if isinstance(mlir_ty, ir.F64Type):
        return arith.constant(mlir_ty, 0.0)
    if isinstance(mlir_ty, ir.F32Type):
        return arith.constant(mlir_ty, 0.0)
    if isinstance(mlir_ty, ir.IntegerType):
        return arith.constant(mlir_ty, 0)
    raise NotImplementedError


# INT32_MIN: the sentinel MLIR's ``llvm.getelementptr`` API uses in its indices
# list to mark a *dynamic* (SSA value) index rather than a constant one.
GEP_DYNAMIC_INDEX = -2147483648


def _gep_typed(base_ptr, index_i64, elem_mlir_type, ptr_ty=None):
    """Typed GEP by a dynamic ``index_i64``. ``ptr_ty`` defaults to the generic
    address space; pass ``_ptr3()`` for shared-memory (addr-space-3) pointers.
    """
    return llvm.getelementptr(
        ptr_ty if ptr_ty is not None else _ptr(),
        base_ptr,
        [index_i64],
        [GEP_DYNAMIC_INDEX],
        elem_mlir_type,
        None,
    )


def _gep_typed3(base_ptr, index_i64, elem_mlir_type):
    """GEP on address-space-3 pointers (shared memory)."""
    return _gep_typed(base_ptr, index_i64, elem_mlir_type, _ptr3())


def _tid_x():
    return arith.index_castui(T.i64(), gpu.thread_id("x"))


def _bdim_x():
    return arith.index_castui(T.i64(), gpu.block_dim("x"))


def _widen_to_f64(val, elem_ty, f64_ty):
    """Widen a loaded element to f64 (extend floats, signed-convert ints)."""
    if isinstance(elem_ty, ir.F64Type):
        return val
    if isinstance(elem_ty, ir.F32Type):
        return arith.extf(f64_ty, val)
    return arith.sitofp(f64_ty, val)


# ---------------------------------------------------------------------------
# Function name convention
# ---------------------------------------------------------------------------


def func_name(op, dtype):
    """Outlined device-function symbol, e.g. ``_coop_sum_i64``."""
    return f"_coop_{op}_{_dtype_tag(dtype)}"


# ---------------------------------------------------------------------------
# Shared-memory helpers for address-space-3 access in device functions
# ---------------------------------------------------------------------------


def _shm3_base(shm_generic_ptr, byte_offset):
    """Cast generic shm ptr to addr-space-3, offset by bytes, return ptr<3>."""
    shm3 = llvm.addrspacecast(_ptr3(), shm_generic_ptr)
    return llvm.getelementptr(
        _ptr3(),
        shm3,
        [_const_i64(byte_offset)],
        [GEP_DYNAMIC_INDEX],
        T.i8(),
        None,
    )


# ---------------------------------------------------------------------------
# Reduction primitives shared by every outlined function
# ---------------------------------------------------------------------------


@contextmanager
def _device_reduction(name, input_types, result_type):
    """Emit the boilerplate for one cooperative device function.

    Creates the public ``func.func`` named ``name`` taking ``input_types`` plus a
    trailing shared-memory base pointer, opens its entry block, and yields
    ``(inputs, tid, bdim, scratch)`` where ``inputs`` are the non-shm arguments
    and ``scratch`` is the (addr-space-3) reduction scratch region. The caller
    emits the body and the ``func.ReturnOp`` inside the ``with`` block.
    """
    function_type = ir.FunctionType.get([*input_types, _ptr()], [result_type])
    fn = func.FuncOp(name=name, type=function_type, visibility="public")
    entry = fn.add_entry_block()
    with ir.InsertionPoint(entry):
        inputs = list(fn.arguments[:-1])
        shm_base = fn.arguments[-1]
        tid = _tid_x()
        bdim = _bdim_x()
        scratch = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)
        yield inputs, tid, bdim, scratch


def _emit_thread_reduce(inputs, size, tid, bdim, initial_values, update):
    """Strided per-thread partial reduction.

    ``inputs`` is a sequence of ``(data_ptr, elem_ty)``; each thread strides over
    ``[tid, tid+bdim, ...)`` up to ``size``. ``update(i, accumulators, elements)``
    returns the new accumulators. Returns the per-thread accumulators (one per
    ``initial_values`` entry).
    """
    loop = scf.ForOp(tid, size, bdim, list(initial_values))
    with ir.InsertionPoint(loop.body):
        i = loop.induction_variable
        accumulators = list(loop.body.arguments[1:])
        elements = [
            llvm.load(ty, _gep_typed(ptr, i, ty)) for ptr, ty in inputs
        ]
        scf.YieldOp(list(update(i, accumulators, elements)))
    return list(loop.results)


def _emit_tree_reduce(bdim, tid, combine_fn):
    """Emit an in-place shared-memory tree reduction over ``bdim`` slots.

    Each thread must have already written its partial into ``shm[tid]`` and
    issued a ``gpu.barrier()``. ``combine_fn(partner)`` is invoked only for the
    active threads; it must read ``shm[tid]`` and ``shm[partner]``, combine
    them, and write the result back to ``shm[tid]``. On return ``shm[0]`` holds
    the fully reduced value and a trailing barrier has run so every thread may
    read it.

    The active count ``m`` starts at ``bdim`` and shrinks by ceil-halving
    (``m -> ceil(m/2)``); the first ``floor(m/2)`` threads fold in the upper
    half. This is correct for *any* block size, not only powers of two -- the
    JIT GroupBy launcher rounds the block size up to a multiple of 32, so groups
    routinely run on 96, 160, ... threads.
    """
    one = _const_i64(1)
    tree = scf.WhileOp([T.i64()], [bdim])
    bb = tree.before.blocks.append(T.i64())
    with ir.InsertionPoint(bb):
        m_b = bb.arguments[0]
        scf.ConditionOp(arith.cmpi(arith.CmpIPredicate.ugt, m_b, one), [m_b])
    ab = tree.after.blocks.append(T.i64())
    with ir.InsertionPoint(ab):
        m = ab.arguments[0]
        half = arith.shrui(arith.addi(m, one), one)  # ceil(m / 2)
        limit = arith.subi(m, half)  # floor(m / 2) active
        in_range = arith.cmpi(arith.CmpIPredicate.ult, tid, limit)
        rif = scf.IfOp(in_range, [], has_else=False)
        with ir.InsertionPoint(rif.then_block):
            partner = arith.addi(tid, half)
            combine_fn(partner)
            scf.YieldOp([])
        gpu.barrier()
        scf.YieldOp([half])


def _emit_block_reduce(shm, value, value_ty, bdim, tid, combine):
    """Store each thread's ``value`` to ``shm[tid]``, tree-reduce with the binary
    ``combine(lhs, rhs)``, and return the result broadcast to all threads.

    Ends with a barrier so ``shm`` may be reused by a subsequent pass.
    """
    zero = _const_i64(0)
    llvm.store(value, _gep_typed3(shm, tid, value_ty))
    gpu.barrier()

    def _combine(partner):
        mv = llvm.load(value_ty, _gep_typed3(shm, tid, value_ty))
        pv = llvm.load(value_ty, _gep_typed3(shm, partner, value_ty))
        llvm.store(combine(mv, pv), _gep_typed3(shm, tid, value_ty))

    _emit_tree_reduce(bdim, tid, _combine)
    result = llvm.load(value_ty, _gep_typed3(shm, zero, value_ty))
    gpu.barrier()
    return result


def _emit_tree_sum_f64(shm, value, bdim, tid):
    """Block-reduce an f64 ``value`` by addition; broadcast the total."""
    return _emit_block_reduce(
        shm, value, ir.F64Type.get(), bdim, tid, arith.addf
    )


# ---------------------------------------------------------------------------
# Function emitters: each appends one public func.func to the current
# gpu.module body (the caller owns the MLIR context / insertion point).
# ---------------------------------------------------------------------------


def _emit_sum(dtype):
    """_coop_sum_{dtype}(ptr data, i64 size, ptr shm) -> scalar.

    Shared-memory tree reduction. Floats skip NaN (pandas skipna=True, all-NaN
    sum is 0.0); integers accumulate in int64 (overflow-safe promotion).
    """
    elem_ty = _dtype_mlir_type(dtype)
    is_float = isinstance(elem_ty, (ir.F64Type, ir.F32Type))
    acc_ty = elem_ty if is_float else T.i64()
    add = arith.addf if is_float else arith.addi

    with _device_reduction(
        func_name("sum", dtype), [_ptr(), T.i64()], acc_ty
    ) as (inputs, tid, bdim, scratch):
        data, size = inputs
        zero_acc = _zero_val(acc_ty)

        def update(i, acc, elems):
            elem = elems[0]
            if is_float:
                is_nan = arith.cmpf(arith.CmpFPredicate.UNO, elem, elem)
                return [add(acc[0], arith.select(is_nan, zero_acc, elem))]
            if elem_ty != acc_ty:
                elem = arith.extsi(acc_ty, elem)
            return [add(acc[0], elem)]

        (partial,) = _emit_thread_reduce(
            [(data, elem_ty)], size, tid, bdim, [zero_acc], update
        )
        result = _emit_block_reduce(scratch, partial, acc_ty, bdim, tid, add)
        func.ReturnOp([result])


def _emit_mean(dtype):
    """_coop_mean_{dtype}(ptr data, i64 size, ptr shm) -> f64."""
    elem_ty = _dtype_mlir_type(dtype)
    f64_ty = ir.F64Type.get()

    with _device_reduction(
        func_name("mean", dtype), [_ptr(), T.i64()], f64_ty
    ) as (inputs, tid, bdim, scratch):
        data, size = inputs
        zero = arith.constant(f64_ty, 0.0)

        def update(i, acc, elems):
            return [
                arith.addf(acc[0], _widen_to_f64(elems[0], elem_ty, f64_ty))
            ]

        (partial,) = _emit_thread_reduce(
            [(data, elem_ty)], size, tid, bdim, [zero], update
        )
        total = _emit_tree_sum_f64(scratch, partial, bdim, tid)
        func.ReturnOp([arith.divf(total, arith.uitofp(f64_ty, size))])


def _emit_minmax(dtype, *, is_max):
    """_coop_{min,max}_{dtype}(ptr data, i64 size, ptr shm) -> scalar.

    Shared-memory tree reduction. Floats use minnumf/maxnumf, which skip NaN
    operands (pandas skipna=True; see #14860); integers use minsi/maxsi.
    """
    elem_ty = _dtype_mlir_type(dtype)
    is_float = isinstance(elem_ty, (ir.F64Type, ir.F32Type))

    def mm(a, b):
        if is_float:
            return arith.maxnumf(a, b) if is_max else arith.minnumf(a, b)
        return arith.maxsi(a, b) if is_max else arith.minsi(a, b)

    name = func_name("max" if is_max else "min", dtype)
    with _device_reduction(name, [_ptr(), T.i64()], elem_ty) as (
        inputs,
        tid,
        bdim,
        scratch,
    ):
        data, size = inputs
        # Threads with no elements seed from data[0], a real element, so they
        # never skew the min/max.
        first_elem = llvm.load(
            elem_ty, _gep_typed(data, _const_i64(0), elem_ty)
        )

        def update(i, acc, elems):
            return [mm(acc[0], elems[0])]

        (partial,) = _emit_thread_reduce(
            [(data, elem_ty)], size, tid, bdim, [first_elem], update
        )
        result = _emit_block_reduce(scratch, partial, elem_ty, bdim, tid, mm)
        func.ReturnOp([result])


def _emit_std(dtype, *, take_sqrt, name):
    """_coop_std_/_coop_var_{dtype}(ptr, i64, ptr) -> f64.

    Two-pass sample variance (ddof=1), which avoids the catastrophic
    cancellation of the one-pass sum(x^2) - sum(x)^2/n form for large-magnitude
    inputs:
      pass 1: mean = sum(x) / n
      pass 2: var  = sum((x - mean)^2) / (n - 1)
    Returns sqrt(variance) when take_sqrt (std), else variance (var). For
    n == 1 this is 0 / 0 -> NaN, matching pandas.
    """
    elem_ty = _dtype_mlir_type(dtype)
    f64_ty = ir.F64Type.get()

    with _device_reduction(name, [_ptr(), T.i64()], f64_ty) as (
        inputs,
        tid,
        bdim,
        scratch,
    ):
        data, size = inputs
        zero = arith.constant(f64_ty, 0.0)
        n_f64 = arith.uitofp(f64_ty, size)

        # Pass 1: grand mean.
        def upd_sum(i, acc, elems):
            return [
                arith.addf(acc[0], _widen_to_f64(elems[0], elem_ty, f64_ty))
            ]

        (p1,) = _emit_thread_reduce(
            [(data, elem_ty)], size, tid, bdim, [zero], upd_sum
        )
        mean = arith.divf(_emit_tree_sum_f64(scratch, p1, bdim, tid), n_f64)

        # Pass 2: sum of squared deviations.
        def upd_ss(i, acc, elems):
            d = arith.subf(_widen_to_f64(elems[0], elem_ty, f64_ty), mean)
            return [arith.addf(acc[0], arith.mulf(d, d))]

        (p2,) = _emit_thread_reduce(
            [(data, elem_ty)], size, tid, bdim, [zero], upd_ss
        )
        total_ss = _emit_tree_sum_f64(scratch, p2, bdim, tid)

        one = arith.constant(f64_ty, 1.0)
        variance = arith.divf(total_ss, arith.subf(n_f64, one))
        func.ReturnOp([mlir_math.sqrt(variance) if take_sqrt else variance])


def _emit_corr(dtype):
    """_coop_corr_{dtype}(ptr lhs, ptr rhs, i64 size, ptr shm) -> f64.

    Pearson correlation = covar(l, r) / (std(l) * std(r)), computed two-pass
    (means first, then centered sums) to match pandas and avoid catastrophic
    cancellation. The ddof factor cancels, so correlation reduces to
    S_lr / sqrt(S_ll * S_rr). Returns NaN when either centered sum of squares is
    zero. Integer inputs only (float corr is rejected in typing).
    """
    elem_ty = _dtype_mlir_type(dtype)
    f64_ty = ir.F64Type.get()

    def w(elem):
        return _widen_to_f64(elem, elem_ty, f64_ty)

    with _device_reduction(
        func_name("corr", dtype), [_ptr(), _ptr(), T.i64()], f64_ty
    ) as (inputs, tid, bdim, scratch):
        lhs, rhs, size = inputs
        zero = arith.constant(f64_ty, 0.0)
        n_f64 = arith.uitofp(f64_ty, size)

        # Pass 1: means.
        def upd_means(i, acc, elems):
            return [
                arith.addf(acc[0], w(elems[0])),
                arith.addf(acc[1], w(elems[1])),
            ]

        s_l, s_r = _emit_thread_reduce(
            [(lhs, elem_ty), (rhs, elem_ty)],
            size,
            tid,
            bdim,
            [zero, zero],
            upd_means,
        )
        mean_l = arith.divf(_emit_tree_sum_f64(scratch, s_l, bdim, tid), n_f64)
        mean_r = arith.divf(_emit_tree_sum_f64(scratch, s_r, bdim, tid), n_f64)

        # Pass 2: centered sums S_lr, S_ll, S_rr.
        def upd_centered(i, acc, elems):
            dl = arith.subf(w(elems[0]), mean_l)
            dr = arith.subf(w(elems[1]), mean_r)
            return [
                arith.addf(acc[0], arith.mulf(dl, dr)),
                arith.addf(acc[1], arith.mulf(dl, dl)),
                arith.addf(acc[2], arith.mulf(dr, dr)),
            ]

        p_lr, p_ll, p_rr = _emit_thread_reduce(
            [(lhs, elem_ty), (rhs, elem_ty)],
            size,
            tid,
            bdim,
            [zero, zero, zero],
            upd_centered,
        )
        s_lr = _emit_tree_sum_f64(scratch, p_lr, bdim, tid)
        s_ll = _emit_tree_sum_f64(scratch, p_ll, bdim, tid)
        s_rr = _emit_tree_sum_f64(scratch, p_rr, bdim, tid)

        denom = arith.mulf(mlir_math.sqrt(s_ll), mlir_math.sqrt(s_rr))
        is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, denom, zero)
        nan_f64 = arith.constant(f64_ty, float("nan"))
        func.ReturnOp(
            [arith.select(is_zero, nan_f64, arith.divf(s_lr, denom))]
        )


def _emit_idx(dtype, *, is_max):
    """_coop_idx{max,min}_{dtype}(ptr data, ptr index, i64 size, ptr shm) -> i64.

    Returns the int64 index label of the max/min element in the group, with ties
    broken by the smallest index (pandas semantics). NaN values are skipped; an
    all-NaN group falls back to the first index label (index[0]), matching the
    legacy BlockIdxMax/BlockIdxMin. Uses two shared-memory scratch arrays (value
    + index) and a tree reduction that carries the (value, index) pair.
    """
    elem_ty = _dtype_mlir_type(dtype)
    is_float = isinstance(elem_ty, (ir.F64Type, ir.F32Type))
    i64 = T.i64()
    name = func_name("idxmax" if is_max else "idxmin", dtype)

    with _device_reduction(name, [_ptr(), _ptr(), i64], i64) as (
        inputs,
        tid,
        bdim,
        shm_val,
    ):
        data, index, size = inputs
        # Second scratch region (index payloads) just past the value region.
        shm_idx = _gep_typed(
            shm_val, _const_i64(_MAX_BLOCK_SIZE * 8), T.i8(), _ptr3()
        )
        zero = _const_i64(0)

        # Sentinel "worst" value so empty/NaN-only threads always lose.
        if is_float:
            sentinel = arith.constant(
                elem_ty, float("-inf") if is_max else float("inf")
            )
        else:
            width = elem_ty.width
            sentinel_int = (
                -(2 ** (width - 1)) if is_max else (2 ** (width - 1) - 1)
            )
            sentinel = arith.constant(elem_ty, sentinel_int)
        idx_sentinel = arith.constant(i64, 2**63 - 1)

        def _is_better(cand, best, cand_i, best_i):
            # cand strictly beats best, or ties and has a smaller index.
            if is_float:
                strict = arith.cmpf(
                    arith.CmpFPredicate.OGT
                    if is_max
                    else arith.CmpFPredicate.OLT,
                    cand,
                    best,
                )
                eq = arith.cmpf(arith.CmpFPredicate.OEQ, cand, best)
            else:
                strict = arith.cmpi(
                    arith.CmpIPredicate.sgt
                    if is_max
                    else arith.CmpIPredicate.slt,
                    cand,
                    best,
                )
                eq = arith.cmpi(arith.CmpIPredicate.eq, cand, best)
            i_smaller = arith.cmpi(arith.CmpIPredicate.slt, cand_i, best_i)
            return arith.ori(strict, arith.andi(eq, i_smaller))

        def update(i, acc, elems):
            elem, idx = elems
            best_v, best_i = acc
            better = _is_better(elem, best_v, idx, best_i)
            if is_float:
                not_nan = arith.cmpf(arith.CmpFPredicate.ORD, elem, elem)
                better = arith.andi(better, not_nan)
            return [
                arith.select(better, elem, best_v),
                arith.select(better, idx, best_i),
            ]

        best_v, best_i = _emit_thread_reduce(
            [(data, elem_ty), (index, i64)],
            size,
            tid,
            bdim,
            [sentinel, idx_sentinel],
            update,
        )

        llvm.store(best_v, _gep_typed3(shm_val, tid, elem_ty))
        llvm.store(best_i, _gep_typed3(shm_idx, tid, i64))
        gpu.barrier()

        def _combine(partner):
            mv = llvm.load(elem_ty, _gep_typed3(shm_val, tid, elem_ty))
            mi = llvm.load(i64, _gep_typed3(shm_idx, tid, i64))
            pv = llvm.load(elem_ty, _gep_typed3(shm_val, partner, elem_ty))
            pi = llvm.load(i64, _gep_typed3(shm_idx, partner, i64))
            take_p = _is_better(pv, mv, pi, mi)
            llvm.store(
                arith.select(take_p, pv, mv),
                _gep_typed3(shm_val, tid, elem_ty),
            )
            llvm.store(
                arith.select(take_p, pi, mi),
                _gep_typed3(shm_idx, tid, i64),
            )

        _emit_tree_reduce(bdim, tid, _combine)

        result_idx = llvm.load(i64, _gep_typed3(shm_idx, zero, i64))
        # All elements skipped (all-NaN group): no thread beat the sentinel, so
        # fall back to the first index label like the legacy BlockIdxMax/Min.
        all_skipped = arith.cmpi(
            arith.CmpIPredicate.eq, result_idx, idx_sentinel
        )
        idx0 = llvm.load(i64, _gep_typed(index, zero, i64))
        func.ReturnOp([arith.select(all_skipped, idx0, result_idx)])


# ---------------------------------------------------------------------------
# The reduction shim: one gpu.module with every (reduction, dtype) function
# ---------------------------------------------------------------------------


def _build_all_reductions_module():
    """Build one MLIR module (one gpu.module) with every reduction function."""
    with context.get_context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp(sym_name="coop_reductions_module")

        body = gpu_mod.bodyRegion.blocks.append()
        with ir.InsertionPoint(body):
            for dt in _DTYPES:
                _emit_sum(dt)
                _emit_mean(dt)
                _emit_minmax(dt, is_max=False)
                _emit_minmax(dt, is_max=True)
                _emit_std(dt, take_sqrt=True, name=func_name("std", dt))
                _emit_std(dt, take_sqrt=False, name=func_name("var", dt))
                _emit_corr(dt)
                _emit_idx(dt, is_max=True)
                _emit_idx(dt, is_max=False)

        return str(module)


# ---------------------------------------------------------------------------
# Compilation: MLIR module string -> PTX bytes
# ---------------------------------------------------------------------------


def _compile_mlir_module_to_ptx(mlir_str, cc):
    """Compile a standalone MLIR module string through the full pipeline to PTX."""
    from numba_cuda_mlir.mlir_optimization import (
        _call_llvm70_capi,
        _compile_to_ptx,
        _needs_llvm70_path,
        _nvvm_options,
        _prepare_llvm_ir,
        get_base_pipeline,
        run_pre_codegen_patterns,
    )
    from numba_cuda_mlir.numba_cuda.cudadrv.nvvm import LibDevice

    with context.get_context():
        module = ir.Module.parse(mlir_str)
        pm = PassManager.parse(get_base_pipeline())
        pm.run(module.operation)
        run_pre_codegen_patterns(module)

        use_llvm70 = _needs_llvm70_path(cc)

        if use_llvm70:
            ptx = _call_llvm70_capi(module, {"chip": f"sm_{cc}"})
        else:
            llvm_ir = _prepare_llvm_ir(module)
            libdevice = LibDevice()
            nvvm_opts = _nvvm_options(cc)
            ptx = _compile_to_ptx(llvm_ir, cc, libdevice, nvvm_opts)

        return ptx


# ---------------------------------------------------------------------------
# Public: the linkable reduction shim (compiled once per compute capability)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _reductions_linkable_for_cc(cc_str):
    ptx = _compile_mlir_module_to_ptx(_build_all_reductions_module(), cc_str)
    # Keep PTX as bytes: cuda.core's ObjectCode.from_ptx treats a str as a file
    # path (-> "no such file") but bytes as inline data.
    if isinstance(ptx, str):
        ptx = ptx.encode()
    return PTXSource(ptx, name="coop_reductions")


def cooperative_reductions_linkable(cc=None):
    """Return a :class:`PTXSource` with every cooperative reduction function.

    Pass it to ``cuda.jit(..., link=[cooperative_reductions_linkable()])`` on any
    kernel that calls a ``CooperativeArray`` reduction. ``cc`` is a
    ``(major, minor)`` tuple; when omitted it is taken from the current device.
    The PTX is built and compiled once per compute capability and cached.
    """
    if cc is None:
        cc = cuda.get_current_device().compute_capability
    return _reductions_linkable_for_cc(f"{cc[0]}{cc[1]}")

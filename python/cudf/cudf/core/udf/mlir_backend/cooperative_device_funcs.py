"""
Compile the cooperative reduction device functions to a single PTX "shim".

Every (reduction, dtype) combination is emitted as a public ``func.func`` into
*one* ``gpu.module``, compiled through the full MLIR->LLVM->PTX pipeline once per
compute capability (cached), and exposed as a :class:`PTXSource` via
``cooperative_reductions_linkable()``. Callers link it into a kernel by passing
``link=[cooperative_reductions_linkable()]`` to ``cuda.jit`` -- mirroring how the
legacy numba-cuda backend linked its ``udf_cpp`` block-reduction shim. The kernel
module only contains declarations (no bodies), so the MLIR inliner cannot inline
them and the linker resolves the calls against this shim.

Shared-memory contract: the outlined functions receive the dynamic shared memory
base pointer as their last argument. gpu.dynamic_shared_memory only works inside
kernels; device functions must receive it externally.
"""

from functools import lru_cache

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir.dialects import arith, func, gpu, llvm, scf
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
    if dtype == nb_types.float64: return ir.F64Type.get()
    if dtype == nb_types.float32: return ir.F32Type.get()
    if dtype == nb_types.int64: return T.i64()
    if dtype == nb_types.int32: return ir.IntegerType.get_signless(32)
    raise NotImplementedError(f"Unsupported dtype {dtype}")


def _dtype_tag(dtype):
    return {nb_types.float64: "f64", nb_types.float32: "f32",
            nb_types.int64: "i64", nb_types.int32: "i32"}[dtype]


def _zero_val(mlir_ty):
    if isinstance(mlir_ty, ir.F64Type): return arith.constant(mlir_ty, 0.0)
    if isinstance(mlir_ty, ir.F32Type): return arith.constant(mlir_ty, 0.0)
    if isinstance(mlir_ty, ir.IntegerType): return arith.constant(mlir_ty, 0)
    raise NotImplementedError


GEP_DYNAMIC_INDEX = -2147483648


def _gep_typed(base_ptr, index_i64, elem_mlir_type):
    return llvm.getelementptr(
        _ptr(), base_ptr, [index_i64], [GEP_DYNAMIC_INDEX],
        elem_mlir_type, None,
    )


def _gep_typed3(base_ptr, index_i64, elem_mlir_type):
    """GEP on address-space-3 pointers (shared memory)."""
    return llvm.getelementptr(
        _ptr3(), base_ptr, [index_i64], [GEP_DYNAMIC_INDEX],
        elem_mlir_type, None,
    )


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

def sum_func_name(dtype):
    return f"_coop_sum_{_dtype_tag(dtype)}"


def mean_func_name(dtype):
    return f"_coop_mean_{_dtype_tag(dtype)}"


def min_func_name(dtype):
    return f"_coop_min_{_dtype_tag(dtype)}"


def max_func_name(dtype):
    return f"_coop_max_{_dtype_tag(dtype)}"


def std_func_name(dtype):
    return f"_coop_std_{_dtype_tag(dtype)}"


def var_func_name(dtype):
    return f"_coop_var_{_dtype_tag(dtype)}"


def corr_func_name(dtype):
    return f"_coop_corr_{_dtype_tag(dtype)}"


def idxmax_func_name(dtype):
    return f"_coop_idxmax_{_dtype_tag(dtype)}"


def idxmin_func_name(dtype):
    return f"_coop_idxmin_{_dtype_tag(dtype)}"


# ---------------------------------------------------------------------------
# Shared-memory helpers for address-space-3 access in device functions
# ---------------------------------------------------------------------------

def _shm3_base(shm_generic_ptr, byte_offset):
    """Cast generic shm ptr to addr-space-3, offset by bytes, return ptr<3>."""
    shm3 = llvm.addrspacecast(_ptr3(), shm_generic_ptr)
    return llvm.getelementptr(
        _ptr3(), shm3, [_const_i64(byte_offset)],
        [GEP_DYNAMIC_INDEX], T.i8(), None,
    )


# ---------------------------------------------------------------------------
# Block tree-reduction primitive (correct for non-power-of-two block sizes)
# ---------------------------------------------------------------------------

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
        scf.ConditionOp(
            arith.cmpi(arith.CmpIPredicate.ugt, m_b, one), [m_b])
    ab = tree.after.blocks.append(T.i64())
    with ir.InsertionPoint(ab):
        m = ab.arguments[0]
        half = arith.shrui(arith.addi(m, one), one)   # ceil(m / 2)
        limit = arith.subi(m, half)                   # floor(m / 2) active
        in_range = arith.cmpi(arith.CmpIPredicate.ult, tid, limit)
        rif = scf.IfOp(in_range, [], has_else=False)
        with ir.InsertionPoint(rif.then_block):
            partner = arith.addi(tid, half)
            combine_fn(partner)
            scf.YieldOp([])
        gpu.barrier()
        scf.YieldOp([half])


def _emit_tree_sum_f64(shm3, value, bdim, tid):
    """Block-reduce an f64 ``value`` by addition; broadcast the total.

    Stores ``value`` to ``shm[tid]``, tree-reduces, and returns the grand total
    (identical on every thread). Ends with a barrier so the same scratch may be
    reused by a subsequent pass.
    """
    f64_ty = ir.F64Type.get()
    zero = _const_i64(0)
    llvm.store(value, _gep_typed3(shm3, tid, f64_ty))
    gpu.barrier()

    def _combine(partner):
        mv = llvm.load(f64_ty, _gep_typed3(shm3, tid, f64_ty))
        pv = llvm.load(f64_ty, _gep_typed3(shm3, partner, f64_ty))
        llvm.store(arith.addf(mv, pv), _gep_typed3(shm3, tid, f64_ty))

    _emit_tree_reduce(bdim, tid, _combine)
    total = llvm.load(f64_ty, _gep_typed3(shm3, zero, f64_ty))
    gpu.barrier()
    return total


# ---------------------------------------------------------------------------
# Function emitters: each appends one public func.func to the current
# gpu.module body (the caller owns the MLIR context / insertion point).
# ---------------------------------------------------------------------------

def _emit_sum(dtype):
    """_coop_sum_{dtype}(ptr data, i64 size, ptr shm) -> scalar.

    Shared-memory tree reduction. Floats skip NaN (pandas skipna=True, all-NaN
    sum is 0.0); integers accumulate in int64 (overflow-safe promotion).
    """
    name = sum_func_name(dtype)
    elem_ty = _dtype_mlir_type(dtype)
    is_float = isinstance(elem_ty, (ir.F64Type, ir.F32Type))
    acc_ty = elem_ty if is_float else T.i64()
    ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [acc_ty])
    fn = func.FuncOp(name=name, type=ft, visibility="public")
    entry = fn.add_entry_block()

    with ir.InsertionPoint(entry):
        data, size, shm_base = fn.arguments
        shm3 = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)

        tid = _tid_x()
        bdim = _bdim_x()
        zero_acc = _zero_val(acc_ty)
        zero = _const_i64(0)

        partial = scf.ForOp(tid, size, bdim, [zero_acc])
        with ir.InsertionPoint(partial.body):
            i = partial.induction_variable
            acc = partial.body.arguments[1]
            elem = llvm.load(elem_ty, _gep_typed(data, i, elem_ty))
            if is_float:
                is_nan = arith.cmpf(arith.CmpFPredicate.UNO, elem, elem)
                elem = arith.select(is_nan, zero_acc, elem)
                new_acc = arith.addf(acc, elem)
            else:
                if elem_ty != acc_ty:
                    elem = arith.extsi(acc_ty, elem)
                new_acc = arith.addi(acc, elem)
            scf.YieldOp([new_acc])

        llvm.store(partial.results[0], _gep_typed3(shm3, tid, acc_ty))
        gpu.barrier()

        def _combine(partner):
            mv = llvm.load(acc_ty, _gep_typed3(shm3, tid, acc_ty))
            pv = llvm.load(acc_ty, _gep_typed3(shm3, partner, acc_ty))
            r = arith.addf(mv, pv) if is_float else arith.addi(mv, pv)
            llvm.store(r, _gep_typed3(shm3, tid, acc_ty))

        _emit_tree_reduce(bdim, tid, _combine)
        result = llvm.load(acc_ty, _gep_typed3(shm3, zero, acc_ty))
        func.ReturnOp([result])


def _emit_mean(dtype):
    """_coop_mean_{dtype}(ptr data, i64 size, ptr shm) -> f64."""
    name = mean_func_name(dtype)
    elem_ty = _dtype_mlir_type(dtype)
    f64_ty = ir.F64Type.get()
    ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [f64_ty])
    fn = func.FuncOp(name=name, type=ft, visibility="public")
    entry = fn.add_entry_block()

    with ir.InsertionPoint(entry):
        data, size, shm_base = fn.arguments
        shm3 = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)

        tid = _tid_x()
        bdim = _bdim_x()
        zero_f64 = arith.constant(f64_ty, 0.0)

        partial = scf.ForOp(tid, size, bdim, [zero_f64])
        with ir.InsertionPoint(partial.body):
            i = partial.induction_variable
            acc = partial.body.arguments[1]
            elem = llvm.load(elem_ty, _gep_typed(data, i, elem_ty))
            e64 = _widen_to_f64(elem, elem_ty, f64_ty)
            scf.YieldOp([arith.addf(acc, e64)])

        total = _emit_tree_sum_f64(shm3, partial.results[0], bdim, tid)
        mean_val = arith.divf(total, arith.uitofp(f64_ty, size))
        func.ReturnOp([mean_val])


def _emit_minmax(dtype, *, is_max):
    """_coop_{min,max}_{dtype}(ptr data, i64 size, ptr shm) -> scalar.

    Shared-memory tree reduction. Floats use minnumf/maxnumf, which skip NaN
    operands (pandas skipna=True; see #14860); integers use minsi/maxsi.
    """
    name = max_func_name(dtype) if is_max else min_func_name(dtype)
    elem_ty = _dtype_mlir_type(dtype)
    is_float = isinstance(elem_ty, (ir.F64Type, ir.F32Type))
    ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [elem_ty])
    fn = func.FuncOp(name=name, type=ft, visibility="public")
    entry = fn.add_entry_block()

    with ir.InsertionPoint(entry):
        data, size, shm_base = fn.arguments
        shm3 = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)

        tid = _tid_x()
        bdim = _bdim_x()
        zero = _const_i64(0)

        def _mm(a, b):
            if is_float:
                return arith.maxnumf(a, b) if is_max else arith.minnumf(a, b)
            return arith.maxsi(a, b) if is_max else arith.minsi(a, b)

        # Threads with no elements seed from data[0], a real element, so they
        # never skew the min/max.
        first_elem = llvm.load(elem_ty, _gep_typed(data, zero, elem_ty))

        partial = scf.ForOp(tid, size, bdim, [first_elem])
        with ir.InsertionPoint(partial.body):
            i = partial.induction_variable
            acc = partial.body.arguments[1]
            elem = llvm.load(elem_ty, _gep_typed(data, i, elem_ty))
            scf.YieldOp([_mm(acc, elem)])

        llvm.store(partial.results[0], _gep_typed3(shm3, tid, elem_ty))
        gpu.barrier()

        def _combine(partner):
            mv = llvm.load(elem_ty, _gep_typed3(shm3, tid, elem_ty))
            pv = llvm.load(elem_ty, _gep_typed3(shm3, partner, elem_ty))
            llvm.store(_mm(mv, pv), _gep_typed3(shm3, tid, elem_ty))

        _emit_tree_reduce(bdim, tid, _combine)
        result = llvm.load(elem_ty, _gep_typed3(shm3, zero, elem_ty))
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
    from numba_cuda_mlir._mlir.dialects import math as mlir_math

    elem_ty = _dtype_mlir_type(dtype)
    f64_ty = ir.F64Type.get()
    ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [f64_ty])
    fn = func.FuncOp(name=name, type=ft, visibility="public")
    entry = fn.add_entry_block()

    with ir.InsertionPoint(entry):
        data, size, shm_base = fn.arguments
        shm3 = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)

        tid = _tid_x()
        bdim = _bdim_x()
        zero_f64 = arith.constant(f64_ty, 0.0)

        # Pass 1: grand mean.
        p1 = scf.ForOp(tid, size, bdim, [zero_f64])
        with ir.InsertionPoint(p1.body):
            i = p1.induction_variable
            acc = p1.body.arguments[1]
            elem = llvm.load(elem_ty, _gep_typed(data, i, elem_ty))
            scf.YieldOp([arith.addf(acc, _widen_to_f64(elem, elem_ty, f64_ty))])
        n_f64 = arith.uitofp(f64_ty, size)
        mean = arith.divf(
            _emit_tree_sum_f64(shm3, p1.results[0], bdim, tid), n_f64)

        # Pass 2: sum of squared deviations.
        p2 = scf.ForOp(tid, size, bdim, [zero_f64])
        with ir.InsertionPoint(p2.body):
            i = p2.induction_variable
            acc = p2.body.arguments[1]
            elem = llvm.load(elem_ty, _gep_typed(data, i, elem_ty))
            d = arith.subf(_widen_to_f64(elem, elem_ty, f64_ty), mean)
            scf.YieldOp([arith.addf(acc, arith.mulf(d, d))])
        total_ss = _emit_tree_sum_f64(shm3, p2.results[0], bdim, tid)

        one_f64 = arith.constant(f64_ty, 1.0)
        variance = arith.divf(total_ss, arith.subf(n_f64, one_f64))

        result = mlir_math.sqrt(variance) if take_sqrt else variance
        func.ReturnOp([result])


def _emit_corr(dtype):
    """_coop_corr_{dtype}(ptr lhs, ptr rhs, i64 size, ptr shm) -> f64.

    Pearson correlation = covar(l, r) / (std(l) * std(r)), computed two-pass
    (means first, then centered sums) to match pandas and avoid catastrophic
    cancellation. The ddof factor cancels, so correlation reduces to
    S_lr / sqrt(S_ll * S_rr). Returns NaN when either centered sum of squares is
    zero. Integer inputs only (float corr is rejected in typing).
    """
    from numba_cuda_mlir._mlir.dialects import math as mlir_math

    name = corr_func_name(dtype)
    elem_ty = _dtype_mlir_type(dtype)
    f64_ty = ir.F64Type.get()
    ft = ir.FunctionType.get([_ptr(), _ptr(), T.i64(), _ptr()], [f64_ty])
    fn = func.FuncOp(name=name, type=ft, visibility="public")
    entry = fn.add_entry_block()

    with ir.InsertionPoint(entry):
        lhs, rhs, size, shm_base = fn.arguments
        shm3 = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)

        tid = _tid_x()
        bdim = _bdim_x()
        zero_f64 = arith.constant(f64_ty, 0.0)

        def _load(ptr, i):
            return _widen_to_f64(
                llvm.load(elem_ty, _gep_typed(ptr, i, elem_ty)),
                elem_ty, f64_ty,
            )

        # Pass 1: means.
        p1 = scf.ForOp(tid, size, bdim, [zero_f64, zero_f64])
        with ir.InsertionPoint(p1.body):
            i = p1.induction_variable
            a_l, a_r = p1.body.arguments[1:3]
            scf.YieldOp([
                arith.addf(a_l, _load(lhs, i)),
                arith.addf(a_r, _load(rhs, i)),
            ])
        n_f64 = arith.uitofp(f64_ty, size)
        mean_l = arith.divf(
            _emit_tree_sum_f64(shm3, p1.results[0], bdim, tid), n_f64)
        mean_r = arith.divf(
            _emit_tree_sum_f64(shm3, p1.results[1], bdim, tid), n_f64)

        # Pass 2: centered sums S_lr, S_ll, S_rr.
        p2 = scf.ForOp(tid, size, bdim, [zero_f64, zero_f64, zero_f64])
        with ir.InsertionPoint(p2.body):
            i = p2.induction_variable
            a_lr, a_ll, a_rr = p2.body.arguments[1:4]
            dl = arith.subf(_load(lhs, i), mean_l)
            dr = arith.subf(_load(rhs, i), mean_r)
            scf.YieldOp([
                arith.addf(a_lr, arith.mulf(dl, dr)),
                arith.addf(a_ll, arith.mulf(dl, dl)),
                arith.addf(a_rr, arith.mulf(dr, dr)),
            ])
        s_lr = _emit_tree_sum_f64(shm3, p2.results[0], bdim, tid)
        s_ll = _emit_tree_sum_f64(shm3, p2.results[1], bdim, tid)
        s_rr = _emit_tree_sum_f64(shm3, p2.results[2], bdim, tid)

        denom = arith.mulf(mlir_math.sqrt(s_ll), mlir_math.sqrt(s_rr))
        is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, denom, zero_f64)
        nan_f64 = arith.constant(f64_ty, float("nan"))
        result = arith.select(is_zero, nan_f64, arith.divf(s_lr, denom))
        func.ReturnOp([result])


def _emit_idx(dtype, *, is_max):
    """_coop_idx{max,min}_{dtype}(ptr data, ptr index, i64 size, ptr shm) -> i64.

    Returns the int64 index label of the max/min element in the group, with ties
    broken by the smallest index (pandas semantics). NaN values are skipped; an
    all-NaN group falls back to the first index label (index[0]), matching the
    legacy BlockIdxMax/BlockIdxMin. Uses two shared-memory scratch arrays (value
    + index) and a tree reduction that carries the (value, index) pair.
    """
    name = idxmax_func_name(dtype) if is_max else idxmin_func_name(dtype)
    _SHM_IDX_OFFSET = _SHM_REDUCTION_OFFSET + _MAX_BLOCK_SIZE * 8

    elem_ty = _dtype_mlir_type(dtype)
    is_float = isinstance(elem_ty, (ir.F64Type, ir.F32Type))
    i64 = T.i64()
    ft = ir.FunctionType.get([_ptr(), _ptr(), i64, _ptr()], [i64])
    fn = func.FuncOp(name=name, type=ft, visibility="public")
    entry = fn.add_entry_block()

    with ir.InsertionPoint(entry):
        data, index, size, shm_base = fn.arguments
        shm_val = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)
        shm_idx = _shm3_base(shm_base, _SHM_IDX_OFFSET)

        tid = _tid_x()
        bdim = _bdim_x()
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
                    arith.CmpFPredicate.OGT if is_max
                    else arith.CmpFPredicate.OLT,
                    cand, best,
                )
                eq = arith.cmpf(arith.CmpFPredicate.OEQ, cand, best)
            else:
                strict = arith.cmpi(
                    arith.CmpIPredicate.sgt if is_max
                    else arith.CmpIPredicate.slt,
                    cand, best,
                )
                eq = arith.cmpi(arith.CmpIPredicate.eq, cand, best)
            i_smaller = arith.cmpi(arith.CmpIPredicate.slt, cand_i, best_i)
            return arith.ori(strict, arith.andi(eq, i_smaller))

        partial = scf.ForOp(tid, size, bdim, [sentinel, idx_sentinel])
        with ir.InsertionPoint(partial.body):
            i = partial.induction_variable
            best_v = partial.body.arguments[1]
            best_i = partial.body.arguments[2]
            elem = llvm.load(elem_ty, _gep_typed(data, i, elem_ty))
            idx = llvm.load(i64, _gep_typed(index, i, i64))
            better = _is_better(elem, best_v, idx, best_i)
            if is_float:
                not_nan = arith.cmpf(arith.CmpFPredicate.ORD, elem, elem)
                better = arith.andi(better, not_nan)
            new_v = arith.select(better, elem, best_v)
            new_i = arith.select(better, idx, best_i)
            scf.YieldOp([new_v, new_i])

        llvm.store(partial.results[0], _gep_typed3(shm_val, tid, elem_ty))
        llvm.store(partial.results[1], _gep_typed3(shm_idx, tid, i64))
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
            arith.CmpIPredicate.eq, result_idx, idx_sentinel)
        idx0 = llvm.load(i64, _gep_typed(index, zero, i64))
        result = arith.select(all_skipped, idx0, result_idx)
        func.ReturnOp([result])


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
                _emit_std(dt, take_sqrt=True, name=std_func_name(dt))
                _emit_std(dt, take_sqrt=False, name=var_func_name(dt))
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
        get_base_pipeline,
        run_pre_codegen_patterns,
        _prepare_llvm_ir,
        _compile_to_ptx,
        _nvvm_options,
        _needs_llvm70_path,
        _call_llvm70_capi,
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
        from numba_cuda_mlir import cuda

        cc = cuda.get_current_device().compute_capability
    return _reductions_linkable_for_cc(f"{cc[0]}{cc[1]}")

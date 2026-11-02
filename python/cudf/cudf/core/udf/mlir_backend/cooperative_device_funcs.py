"""
Compile cooperative device functions to PTX for separate linking.

Each (op, dtype) combination produces a standalone gpu.module that is
compiled through the full MLIR->LLVM->PTX pipeline independently from
the kernel.  The kernel module only contains declarations (no bodies),
so the MLIR inliner cannot inline them.  The NVVM linker resolves
cross-module calls at link time.

Shared-memory contract: the outlined functions receive the dynamic
shared memory base pointer as their last argument.  gpu.dynamic_shared_memory
only works inside kernels; device functions must receive it externally.
"""

import operator
from functools import lru_cache

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir.dialects import arith, func, gpu, llvm, scf
from numba_cuda_mlir._mlir.passmanager import PassManager
from numba_cuda_mlir.lowering_utilities import context, get_or_insert_function
from numba_cuda_mlir.numba_cuda import types as nb_types


# ---------------------------------------------------------------------------
# Shared-memory layout (must match cooperative.py)
# ---------------------------------------------------------------------------

_SHM_BROADCAST_OFFSET = 0   # bytes 0..7:  data ptr slot
_SHM_MI_SLOT = 8            # bytes 8..15: meminfo ptr slot
_SHM_REDUCTION_OFFSET = 16  # bytes 16+:   reduction scratch
_MAX_BLOCK_SIZE = 1024


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


def _const_index(val):
    return arith.constant(T.index(), val)


def _dtype_mlir_type(dtype):
    if dtype == nb_types.float64: return ir.F64Type.get()
    if dtype == nb_types.float32: return ir.F32Type.get()
    if dtype == nb_types.int64: return T.i64()
    if dtype == nb_types.int32: return ir.IntegerType.get_signless(32)
    raise NotImplementedError(f"Unsupported dtype {dtype}")


def _dtype_byte_size(dtype):
    if dtype in (nb_types.float64, nb_types.int64): return 8
    if dtype in (nb_types.float32, nb_types.int32): return 4
    raise NotImplementedError(f"Unsupported dtype {dtype}")


def _dtype_tag(dtype):
    return {nb_types.float64: "f64", nb_types.float32: "f32",
            nb_types.int64: "i64", nb_types.int32: "i32"}[dtype]


def _op_tag(op):
    return {operator.add: "add", operator.sub: "sub", operator.mul: "mul",
            operator.truediv: "div"}[op]


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


def _call_nrt(gpu_mod, name, arg_types, result_types, args):
    ft = ir.FunctionType.get(arg_types, result_types)
    callee = get_or_insert_function(name, ft, gpu_mod)
    return func.call(
        result=result_types, callee=callee.name.value, operands_=args,
    )


def _shm_ptr_at(base_ptr, byte_offset):
    """GEP from raw shm base to a specific byte offset (generic addr space)."""
    return llvm.getelementptr(
        _ptr(), base_ptr, [_const_i64(byte_offset)],
        [GEP_DYNAMIC_INDEX], T.i8(), None,
    )


def _arith_op_emit(op, lhs, rhs, mlir_ty):
    if isinstance(mlir_ty, (ir.F64Type, ir.F32Type)):
        if op is operator.add: return arith.addf(lhs, rhs)
        if op is operator.sub: return arith.subf(lhs, rhs)
        if op is operator.mul: return arith.mulf(lhs, rhs)
        if op is operator.truediv: return arith.divf(lhs, rhs)
    elif isinstance(mlir_ty, ir.IntegerType):
        if op is operator.add: return arith.addi(lhs, rhs)
        if op is operator.sub: return arith.subi(lhs, rhs)
        if op is operator.mul: return arith.muli(lhs, rhs)
        if op is operator.truediv: return arith.divsi(lhs, rhs)
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Function name convention
# ---------------------------------------------------------------------------

def binop_func_name(op, dtype):
    return f"_coop_{_op_tag(op)}_{_dtype_tag(dtype)}"


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


def exp_func_name(dtype):
    return f"_coop_exp_{_dtype_tag(dtype)}"


def scalar_binop_func_name(op, dtype):
    return f"_coop_s{_op_tag(op)}_{_dtype_tag(dtype)}"


def scalar_rbinop_func_name(op, dtype):
    return f"_coop_rs{_op_tag(op)}_{_dtype_tag(dtype)}"


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
# Module builders: each returns a complete MLIR module string
# ---------------------------------------------------------------------------

def _build_binop_module(op, dtype):
    """Build MLIR module for _coop_{op}_{dtype}(ptr lhs, ptr rhs, i64 size, ptr shm) -> void.

    Results written to shm:
        shm[0..7]  = data pointer (broadcast from thread 0)
        shm[8..15] = meminfo pointer (thread 0 only)
    """
    elem_bytes = _dtype_byte_size(dtype)
    name = binop_func_name(op, dtype)

    with context.get_context(), ir.Location.unknown():
        elem_ty = _dtype_mlir_type(dtype)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp(sym_name=f"{name}_module")

        body = gpu_mod.bodyRegion.blocks.append()
        ft = ir.FunctionType.get([_ptr(), _ptr(), T.i64(), _ptr()], [])
        with ir.InsertionPoint(body):
            fn = func.FuncOp(name=name, type=ft, visibility="public")
            entry = fn.add_entry_block()

        with ir.InsertionPoint(entry):
            lhs_data, rhs_data, size, shm_base = fn.arguments

            data_slot = _shm_ptr_at(shm_base, _SHM_BROADCAST_OFFSET)
            mi_slot = _shm_ptr_at(shm_base, _SHM_MI_SLOT)

            tid = _tid_x()
            is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, _const_i64(0))
            alloc_bytes = arith.muli(size, _const_i64(elem_bytes))
            alloc_bytes = arith.maxui(alloc_bytes, _const_i64(1))

            t0_if = scf.IfOp(is_t0, results_=[], has_else=False)
            with ir.InsertionPoint(t0_if.then_block):
                mi_val = _call_nrt(gpu_mod, "NRT_MemInfo_new_varsize",
                                   [T.i64()], [_ptr()], [alloc_bytes])
                data_val = _call_nrt(gpu_mod, "NRT_MemInfo_data_fast",
                                     [_ptr()], [_ptr()], [mi_val])
                llvm.store(data_val, data_slot)
                llvm.store(mi_val, mi_slot)
                scf.YieldOp([])
            gpu.barrier()

            out_data = llvm.load(_ptr(), data_slot)
            bdim = _bdim_x()
            loop = scf.ForOp(tid, size, bdim, [])
            with ir.InsertionPoint(loop.body):
                i = loop.induction_variable
                a = llvm.load(elem_ty, _gep_typed(lhs_data, i, elem_ty))
                b = llvm.load(elem_ty, _gep_typed(rhs_data, i, elem_ty))
                r = _arith_op_emit(op, a, b, elem_ty)
                llvm.store(r, _gep_typed(out_data, i, elem_ty))
                scf.YieldOp([])
            gpu.barrier()

            func.ReturnOp([])

        return str(module)


def _build_sum_module(dtype):
    """Build MLIR module for _coop_sum_{dtype}(ptr data, i64 size, ptr shm) -> scalar.

    Uses shared-memory tree reduction via address-space-3 pointers.
    """
    name = sum_func_name(dtype)

    with context.get_context(), ir.Location.unknown():
        elem_ty = _dtype_mlir_type(dtype)
        is_float = isinstance(elem_ty, (ir.F64Type, ir.F32Type))
        # pandas promotes integer sums to int64 (overflow-safe); floats keep
        # their width. Accumulate (and store partials) in the accumulator type.
        acc_ty = elem_ty if is_float else T.i64()
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp(sym_name=f"{name}_module")

        body = gpu_mod.bodyRegion.blocks.append()
        ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [acc_ty])
        with ir.InsertionPoint(body):
            fn = func.FuncOp(name=name, type=ft, visibility="public")
            entry = fn.add_entry_block()

        with ir.InsertionPoint(entry):
            data, size, shm_base = fn.arguments
            shm3 = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)

            tid = _tid_x()
            bdim = _bdim_x()
            zero_acc = _zero_val(acc_ty)
            zero = _const_i64(0)
            one = _const_i64(1)

            partial = scf.ForOp(tid, size, bdim, [zero_acc])
            with ir.InsertionPoint(partial.body):
                i = partial.induction_variable
                acc = partial.body.arguments[1]
                elem = llvm.load(elem_ty, _gep_typed(data, i, elem_ty))
                if is_float:
                    # Skip NaN (pandas skipna=True): sum of all-NaN is 0.0.
                    is_nan = arith.cmpf(
                        arith.CmpFPredicate.UNO, elem, elem
                    )
                    elem = arith.select(is_nan, zero_acc, elem)
                    new_acc = arith.addf(acc, elem)
                else:
                    if elem_ty != acc_ty:
                        elem = arith.extsi(acc_ty, elem)
                    new_acc = arith.addi(acc, elem)
                scf.YieldOp([new_acc])

            llvm.store(partial.results[0], _gep_typed3(shm3, tid, acc_ty))
            gpu.barrier()

            half_bdim = arith.shrui(bdim, one)
            tree = scf.WhileOp([T.i64()], [half_bdim])
            bb = tree.before.blocks.append(T.i64())
            with ir.InsertionPoint(bb):
                s_b = bb.arguments[0]
                scf.ConditionOp(
                    arith.cmpi(arith.CmpIPredicate.ugt, s_b, zero), [s_b])
            ab = tree.after.blocks.append(T.i64())
            with ir.InsertionPoint(ab):
                s = ab.arguments[0]
                in_range = arith.cmpi(arith.CmpIPredicate.ult, tid, s)
                rif = scf.IfOp(in_range, [], has_else=False)
                with ir.InsertionPoint(rif.then_block):
                    partner = arith.addi(tid, s)
                    my_v = llvm.load(acc_ty, _gep_typed3(shm3, tid, acc_ty))
                    p_v = llvm.load(acc_ty, _gep_typed3(shm3, partner, acc_ty))
                    if is_float:
                        summed = arith.addf(my_v, p_v)
                    else:
                        summed = arith.addi(my_v, p_v)
                    llvm.store(summed, _gep_typed3(shm3, tid, acc_ty))
                    scf.YieldOp([])
                gpu.barrier()
                scf.YieldOp([arith.shrui(s, one)])

            result = llvm.load(acc_ty, _gep_typed3(shm3, zero, acc_ty))
            func.ReturnOp([result])

        return str(module)


def _build_mean_module(dtype):
    """Build MLIR module for _coop_mean_{dtype}(ptr data, i64 size, ptr shm) -> f64."""
    name = mean_func_name(dtype)

    with context.get_context(), ir.Location.unknown():
        elem_ty = _dtype_mlir_type(dtype)
        f64_ty = ir.F64Type.get()
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp(sym_name=f"{name}_module")

        body = gpu_mod.bodyRegion.blocks.append()
        ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [f64_ty])
        with ir.InsertionPoint(body):
            fn = func.FuncOp(name=name, type=ft, visibility="public")
            entry = fn.add_entry_block()

        with ir.InsertionPoint(entry):
            data, size, shm_base = fn.arguments
            shm3 = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)

            tid = _tid_x()
            bdim = _bdim_x()
            zero_f64 = arith.constant(f64_ty, 0.0)
            zero = _const_i64(0)
            one = _const_i64(1)

            partial = scf.ForOp(tid, size, bdim, [zero_f64])
            with ir.InsertionPoint(partial.body):
                i = partial.induction_variable
                acc = partial.body.arguments[1]
                elem = llvm.load(elem_ty, _gep_typed(data, i, elem_ty))
                if isinstance(elem_ty, (ir.F64Type, ir.F32Type)):
                    e64 = elem if isinstance(elem_ty, ir.F64Type) else arith.extf(f64_ty, elem)
                else:
                    e64 = arith.sitofp(f64_ty, elem)
                scf.YieldOp([arith.addf(acc, e64)])

            llvm.store(partial.results[0], _gep_typed3(shm3, tid, f64_ty))
            gpu.barrier()

            half_bdim = arith.shrui(bdim, one)
            tree = scf.WhileOp([T.i64()], [half_bdim])
            bb = tree.before.blocks.append(T.i64())
            with ir.InsertionPoint(bb):
                s_b = bb.arguments[0]
                scf.ConditionOp(
                    arith.cmpi(arith.CmpIPredicate.ugt, s_b, zero), [s_b])
            ab = tree.after.blocks.append(T.i64())
            with ir.InsertionPoint(ab):
                s = ab.arguments[0]
                in_range = arith.cmpi(arith.CmpIPredicate.ult, tid, s)
                rif = scf.IfOp(in_range, [], has_else=False)
                with ir.InsertionPoint(rif.then_block):
                    partner = arith.addi(tid, s)
                    my_v = llvm.load(f64_ty, _gep_typed3(shm3, tid, f64_ty))
                    p_v = llvm.load(f64_ty, _gep_typed3(shm3, partner, f64_ty))
                    llvm.store(arith.addf(my_v, p_v), _gep_typed3(shm3, tid, f64_ty))
                    scf.YieldOp([])
                gpu.barrier()
                scf.YieldOp([arith.shrui(s, one)])

            total = llvm.load(f64_ty, _gep_typed3(shm3, zero, f64_ty))
            mean_val = arith.divf(total, arith.uitofp(f64_ty, size))
            func.ReturnOp([mean_val])

        return str(module)


def _build_min_module(dtype):
    """Build MLIR module for _coop_min_{dtype}(ptr data, i64 size, ptr shm) -> scalar.

    Uses shared-memory tree reduction with min instead of add.
    """
    name = min_func_name(dtype)

    with context.get_context(), ir.Location.unknown():
        elem_ty = _dtype_mlir_type(dtype)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp(sym_name=f"{name}_module")

        body = gpu_mod.bodyRegion.blocks.append()
        ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [elem_ty])
        with ir.InsertionPoint(body):
            fn = func.FuncOp(name=name, type=ft, visibility="public")
            entry = fn.add_entry_block()

        with ir.InsertionPoint(entry):
            data, size, shm_base = fn.arguments
            shm3 = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)

            tid = _tid_x()
            bdim = _bdim_x()
            zero = _const_i64(0)
            one = _const_i64(1)

            first_elem = llvm.load(elem_ty, _gep_typed(data, zero, elem_ty))

            partial = scf.ForOp(tid, size, bdim, [first_elem])
            with ir.InsertionPoint(partial.body):
                i = partial.induction_variable
                acc = partial.body.arguments[1]
                elem = llvm.load(elem_ty, _gep_typed(data, i, elem_ty))
                if isinstance(elem_ty, (ir.F64Type, ir.F32Type)):
                    # minnumf: IEEE minimumNumber, skips NaN operands so that
                    # NaN values are ignored (pandas skipna=True). See #14860.
                    new_acc = arith.minnumf(acc, elem)
                else:
                    new_acc = arith.minsi(acc, elem)
                scf.YieldOp([new_acc])

            llvm.store(partial.results[0], _gep_typed3(shm3, tid, elem_ty))
            gpu.barrier()

            half_bdim = arith.shrui(bdim, one)
            tree = scf.WhileOp([T.i64()], [half_bdim])
            bb = tree.before.blocks.append(T.i64())
            with ir.InsertionPoint(bb):
                s_b = bb.arguments[0]
                scf.ConditionOp(
                    arith.cmpi(arith.CmpIPredicate.ugt, s_b, zero), [s_b])
            ab = tree.after.blocks.append(T.i64())
            with ir.InsertionPoint(ab):
                s = ab.arguments[0]
                in_range = arith.cmpi(arith.CmpIPredicate.ult, tid, s)
                rif = scf.IfOp(in_range, [], has_else=False)
                with ir.InsertionPoint(rif.then_block):
                    partner = arith.addi(tid, s)
                    my_v = llvm.load(elem_ty, _gep_typed3(shm3, tid, elem_ty))
                    p_v = llvm.load(elem_ty, _gep_typed3(shm3, partner, elem_ty))
                    if isinstance(elem_ty, (ir.F64Type, ir.F32Type)):
                        reduced = arith.minnumf(my_v, p_v)
                    else:
                        reduced = arith.minsi(my_v, p_v)
                    llvm.store(reduced, _gep_typed3(shm3, tid, elem_ty))
                    scf.YieldOp([])
                gpu.barrier()
                scf.YieldOp([arith.shrui(s, one)])

            result = llvm.load(elem_ty, _gep_typed3(shm3, zero, elem_ty))
            func.ReturnOp([result])

        return str(module)


def _build_max_module(dtype):
    """Build MLIR module for _coop_max_{dtype}(ptr data, i64 size, ptr shm) -> scalar.

    Shared-memory tree reduction with max (mirror of _build_min_module).
    """
    name = max_func_name(dtype)

    with context.get_context(), ir.Location.unknown():
        elem_ty = _dtype_mlir_type(dtype)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp(sym_name=f"{name}_module")

        body = gpu_mod.bodyRegion.blocks.append()
        ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [elem_ty])
        with ir.InsertionPoint(body):
            fn = func.FuncOp(name=name, type=ft, visibility="public")
            entry = fn.add_entry_block()

        with ir.InsertionPoint(entry):
            data, size, shm_base = fn.arguments
            shm3 = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)

            tid = _tid_x()
            bdim = _bdim_x()
            zero = _const_i64(0)
            one = _const_i64(1)

            first_elem = llvm.load(elem_ty, _gep_typed(data, zero, elem_ty))

            partial = scf.ForOp(tid, size, bdim, [first_elem])
            with ir.InsertionPoint(partial.body):
                i = partial.induction_variable
                acc = partial.body.arguments[1]
                elem = llvm.load(elem_ty, _gep_typed(data, i, elem_ty))
                if isinstance(elem_ty, (ir.F64Type, ir.F32Type)):
                    # maxnumf: IEEE maximumNumber, skips NaN operands (pandas
                    # skipna=True). See #14860.
                    new_acc = arith.maxnumf(acc, elem)
                else:
                    new_acc = arith.maxsi(acc, elem)
                scf.YieldOp([new_acc])

            llvm.store(partial.results[0], _gep_typed3(shm3, tid, elem_ty))
            gpu.barrier()

            half_bdim = arith.shrui(bdim, one)
            tree = scf.WhileOp([T.i64()], [half_bdim])
            bb = tree.before.blocks.append(T.i64())
            with ir.InsertionPoint(bb):
                s_b = bb.arguments[0]
                scf.ConditionOp(
                    arith.cmpi(arith.CmpIPredicate.ugt, s_b, zero), [s_b])
            ab = tree.after.blocks.append(T.i64())
            with ir.InsertionPoint(ab):
                s = ab.arguments[0]
                in_range = arith.cmpi(arith.CmpIPredicate.ult, tid, s)
                rif = scf.IfOp(in_range, [], has_else=False)
                with ir.InsertionPoint(rif.then_block):
                    partner = arith.addi(tid, s)
                    my_v = llvm.load(elem_ty, _gep_typed3(shm3, tid, elem_ty))
                    p_v = llvm.load(elem_ty, _gep_typed3(shm3, partner, elem_ty))
                    if isinstance(elem_ty, (ir.F64Type, ir.F32Type)):
                        reduced = arith.maxnumf(my_v, p_v)
                    else:
                        reduced = arith.maxsi(my_v, p_v)
                    llvm.store(reduced, _gep_typed3(shm3, tid, elem_ty))
                    scf.YieldOp([])
                gpu.barrier()
                scf.YieldOp([arith.shrui(s, one)])

            result = llvm.load(elem_ty, _gep_typed3(shm3, zero, elem_ty))
            func.ReturnOp([result])

        return str(module)


def _build_std_module(dtype, *, take_sqrt=True, name=None):
    """Build MLIR module for _coop_std_/_coop_var_{dtype}(ptr, i64, ptr) -> f64.

    Computes population variance = mean(x²) - mean(x)² using two parallel
    reductions in shared memory:
      shm[_SHM_REDUCTION_OFFSET ..] for partial sums
      shm[_SHM_REDUCTION_OFFSET + 1024*8 ..] for partial sum-of-squares
    Returns sqrt(variance) when take_sqrt (std), else variance (var).
    """
    name = name or std_func_name(dtype)
    _SHM_SQ_OFFSET = _SHM_REDUCTION_OFFSET + _MAX_BLOCK_SIZE * 8

    with context.get_context(), ir.Location.unknown():
        elem_ty = _dtype_mlir_type(dtype)
        f64_ty = ir.F64Type.get()
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp(sym_name=f"{name}_module")

        body = gpu_mod.bodyRegion.blocks.append()
        ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [f64_ty])
        with ir.InsertionPoint(body):
            fn = func.FuncOp(name=name, type=ft, visibility="public")
            entry = fn.add_entry_block()

        with ir.InsertionPoint(entry):
            data, size, shm_base = fn.arguments
            shm3_sum = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)
            shm3_sq = _shm3_base(shm_base, _SHM_SQ_OFFSET)

            tid = _tid_x()
            bdim = _bdim_x()
            zero_f64 = arith.constant(f64_ty, 0.0)
            zero = _const_i64(0)
            one = _const_i64(1)

            partial = scf.ForOp(tid, size, bdim, [zero_f64, zero_f64])
            with ir.InsertionPoint(partial.body):
                i = partial.induction_variable
                acc_sum = partial.body.arguments[1]
                acc_sq = partial.body.arguments[2]
                elem = llvm.load(elem_ty, _gep_typed(data, i, elem_ty))
                if isinstance(elem_ty, (ir.F64Type, ir.F32Type)):
                    e64 = elem if isinstance(elem_ty, ir.F64Type) else arith.extf(f64_ty, elem)
                else:
                    e64 = arith.sitofp(f64_ty, elem)
                new_sum = arith.addf(acc_sum, e64)
                new_sq = arith.addf(acc_sq, arith.mulf(e64, e64))
                scf.YieldOp([new_sum, new_sq])

            llvm.store(partial.results[0], _gep_typed3(shm3_sum, tid, f64_ty))
            llvm.store(partial.results[1], _gep_typed3(shm3_sq, tid, f64_ty))
            gpu.barrier()

            half_bdim = arith.shrui(bdim, one)
            tree = scf.WhileOp([T.i64()], [half_bdim])
            bb = tree.before.blocks.append(T.i64())
            with ir.InsertionPoint(bb):
                s_b = bb.arguments[0]
                scf.ConditionOp(
                    arith.cmpi(arith.CmpIPredicate.ugt, s_b, zero), [s_b])
            ab = tree.after.blocks.append(T.i64())
            with ir.InsertionPoint(ab):
                s = ab.arguments[0]
                in_range = arith.cmpi(arith.CmpIPredicate.ult, tid, s)
                rif = scf.IfOp(in_range, [], has_else=False)
                with ir.InsertionPoint(rif.then_block):
                    partner = arith.addi(tid, s)
                    my_s = llvm.load(f64_ty, _gep_typed3(shm3_sum, tid, f64_ty))
                    p_s = llvm.load(f64_ty, _gep_typed3(shm3_sum, partner, f64_ty))
                    llvm.store(arith.addf(my_s, p_s), _gep_typed3(shm3_sum, tid, f64_ty))
                    my_q = llvm.load(f64_ty, _gep_typed3(shm3_sq, tid, f64_ty))
                    p_q = llvm.load(f64_ty, _gep_typed3(shm3_sq, partner, f64_ty))
                    llvm.store(arith.addf(my_q, p_q), _gep_typed3(shm3_sq, tid, f64_ty))
                    scf.YieldOp([])
                gpu.barrier()
                scf.YieldOp([arith.shrui(s, one)])

            total_sum = llvm.load(f64_ty, _gep_typed3(shm3_sum, zero, f64_ty))
            total_sq = llvm.load(f64_ty, _gep_typed3(shm3_sq, zero, f64_ty))
            n_f64 = arith.uitofp(f64_ty, size)
            one_f64 = arith.constant(f64_ty, 1.0)
            # Sample variance (ddof=1) to match pandas groupby var/std:
            #   (sum(x^2) - sum(x)^2 / n) / (n - 1)
            # For n == 1 this is 0/0 -> NaN, matching pandas.
            sum_sq_over_n = arith.divf(
                arith.mulf(total_sum, total_sum), n_f64
            )
            numerator = arith.subf(total_sq, sum_sq_over_n)
            n_minus_1 = arith.subf(n_f64, one_f64)
            variance = arith.divf(numerator, n_minus_1)

            if take_sqrt:
                from numba_cuda_mlir._mlir.dialects import math as mlir_math

                result = mlir_math.sqrt(variance)
            else:
                result = variance
            func.ReturnOp([result])

        return str(module)


def _build_var_module(dtype):
    """Population variance: std without the final sqrt."""
    return _build_std_module(
        dtype, take_sqrt=False, name=var_func_name(dtype)
    )


def _build_corr_module(dtype):
    """Build ``_coop_corr_{dtype}(ptr lhs, ptr rhs, i64 size, ptr shm) -> f64``.

    Pearson correlation = covar(l, r) / (std(l) * std(r)), with sample
    (ddof=1) covariance/variance to match pandas. Returns NaN when either
    variance is zero. Integer inputs only (float corr is rejected in typing).
    """
    from numba_cuda_mlir._mlir.dialects import math as mlir_math

    name = corr_func_name(dtype)

    with context.get_context(), ir.Location.unknown():
        elem_ty = _dtype_mlir_type(dtype)
        f64_ty = ir.F64Type.get()
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp(sym_name=f"{name}_module")

        body = gpu_mod.bodyRegion.blocks.append()
        ft = ir.FunctionType.get([_ptr(), _ptr(), T.i64(), _ptr()], [f64_ty])
        with ir.InsertionPoint(body):
            fn = func.FuncOp(name=name, type=ft, visibility="public")
            entry = fn.add_entry_block()

        with ir.InsertionPoint(entry):
            lhs, rhs, size, shm_base = fn.arguments
            shm3 = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)

            tid = _tid_x()
            bdim = _bdim_x()
            zero = _const_i64(0)
            one = _const_i64(1)
            zero_f64 = arith.constant(f64_ty, 0.0)

            def _to_f64(val):
                if isinstance(elem_ty, ir.F64Type):
                    return val
                if isinstance(elem_ty, ir.F32Type):
                    return arith.extf(f64_ty, val)
                return arith.sitofp(f64_ty, val)

            # One pass: accumulate S_l, S_r, S_ll, S_rr, S_lr per thread.
            partial = scf.ForOp(
                tid, size, bdim,
                [zero_f64, zero_f64, zero_f64, zero_f64, zero_f64],
            )
            with ir.InsertionPoint(partial.body):
                i = partial.induction_variable
                a_l, a_r, a_ll, a_rr, a_lr = partial.body.arguments[1:6]
                lv = _to_f64(llvm.load(elem_ty, _gep_typed(lhs, i, elem_ty)))
                rv = _to_f64(llvm.load(elem_ty, _gep_typed(rhs, i, elem_ty)))
                scf.YieldOp([
                    arith.addf(a_l, lv),
                    arith.addf(a_r, rv),
                    arith.addf(a_ll, arith.mulf(lv, lv)),
                    arith.addf(a_rr, arith.mulf(rv, rv)),
                    arith.addf(a_lr, arith.mulf(lv, rv)),
                ])

            def _tree_sum(value):
                """Block tree-reduce ``value`` (f64); broadcast total to all threads."""
                llvm.store(value, _gep_typed3(shm3, tid, f64_ty))
                gpu.barrier()
                half = arith.shrui(bdim, one)
                tree = scf.WhileOp([T.i64()], [half])
                bb = tree.before.blocks.append(T.i64())
                with ir.InsertionPoint(bb):
                    s_b = bb.arguments[0]
                    scf.ConditionOp(
                        arith.cmpi(arith.CmpIPredicate.ugt, s_b, zero), [s_b])
                ab = tree.after.blocks.append(T.i64())
                with ir.InsertionPoint(ab):
                    s = ab.arguments[0]
                    in_range = arith.cmpi(arith.CmpIPredicate.ult, tid, s)
                    rif = scf.IfOp(in_range, [], has_else=False)
                    with ir.InsertionPoint(rif.then_block):
                        partner = arith.addi(tid, s)
                        mv = llvm.load(f64_ty, _gep_typed3(shm3, tid, f64_ty))
                        pv = llvm.load(
                            f64_ty, _gep_typed3(shm3, partner, f64_ty))
                        llvm.store(
                            arith.addf(mv, pv), _gep_typed3(shm3, tid, f64_ty))
                        scf.YieldOp([])
                    gpu.barrier()
                    scf.YieldOp([arith.shrui(s, one)])
                total = llvm.load(f64_ty, _gep_typed3(shm3, zero, f64_ty))
                gpu.barrier()
                return total

            s_l = _tree_sum(partial.results[0])
            s_r = _tree_sum(partial.results[1])
            s_ll = _tree_sum(partial.results[2])
            s_rr = _tree_sum(partial.results[3])
            s_lr = _tree_sum(partial.results[4])

            n_f64 = arith.uitofp(f64_ty, size)
            one_f64 = arith.constant(f64_ty, 1.0)
            n_minus_1 = arith.subf(n_f64, one_f64)
            # Sample covariance/variance (ddof=1).
            covar = arith.divf(
                arith.subf(s_lr, arith.divf(arith.mulf(s_l, s_r), n_f64)),
                n_minus_1,
            )
            var_l = arith.divf(
                arith.subf(s_ll, arith.divf(arith.mulf(s_l, s_l), n_f64)),
                n_minus_1,
            )
            var_r = arith.divf(
                arith.subf(s_rr, arith.divf(arith.mulf(s_r, s_r), n_f64)),
                n_minus_1,
            )
            denom = arith.mulf(mlir_math.sqrt(var_l), mlir_math.sqrt(var_r))
            is_zero = arith.cmpf(arith.CmpFPredicate.OEQ, denom, zero_f64)
            nan_f64 = arith.constant(f64_ty, float("nan"))
            result = arith.select(is_zero, nan_f64, arith.divf(covar, denom))
            func.ReturnOp([result])

        return str(module)


def _build_idx_module(dtype, *, is_max):
    """Build ``_coop_idx{max,min}_{dtype}(ptr data, ptr index, i64 size, ptr shm) -> i64``.

    Returns the int64 index label of the max/min element in the group, with
    ties broken by the smallest index (pandas semantics). NaN values are
    skipped. Uses two shared-memory scratch arrays (value + index) and a
    tree reduction that carries the (value, index) pair.
    """
    name = idxmax_func_name(dtype) if is_max else idxmin_func_name(dtype)
    _SHM_IDX_OFFSET = _SHM_REDUCTION_OFFSET + _MAX_BLOCK_SIZE * 8

    with context.get_context(), ir.Location.unknown():
        elem_ty = _dtype_mlir_type(dtype)
        is_float = isinstance(elem_ty, (ir.F64Type, ir.F32Type))
        i64 = T.i64()
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp(sym_name=f"{name}_module")

        body = gpu_mod.bodyRegion.blocks.append()
        ft = ir.FunctionType.get([_ptr(), _ptr(), i64, _ptr()], [i64])
        with ir.InsertionPoint(body):
            fn = func.FuncOp(name=name, type=ft, visibility="public")
            entry = fn.add_entry_block()

        with ir.InsertionPoint(entry):
            data, index, size, shm_base = fn.arguments
            shm_val = _shm3_base(shm_base, _SHM_REDUCTION_OFFSET)
            shm_idx = _shm3_base(shm_base, _SHM_IDX_OFFSET)

            tid = _tid_x()
            bdim = _bdim_x()
            zero = _const_i64(0)
            one = _const_i64(1)

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
                    not_nan = arith.cmpf(
                        arith.CmpFPredicate.ORD, elem, elem
                    )
                    better = arith.andi(better, not_nan)
                new_v = arith.select(better, elem, best_v)
                new_i = arith.select(better, idx, best_i)
                scf.YieldOp([new_v, new_i])

            llvm.store(partial.results[0], _gep_typed3(shm_val, tid, elem_ty))
            llvm.store(partial.results[1], _gep_typed3(shm_idx, tid, i64))
            gpu.barrier()

            half_bdim = arith.shrui(bdim, one)
            tree = scf.WhileOp([i64], [half_bdim])
            bb = tree.before.blocks.append(i64)
            with ir.InsertionPoint(bb):
                s_b = bb.arguments[0]
                scf.ConditionOp(
                    arith.cmpi(arith.CmpIPredicate.ugt, s_b, zero), [s_b])
            ab = tree.after.blocks.append(i64)
            with ir.InsertionPoint(ab):
                s = ab.arguments[0]
                in_range = arith.cmpi(arith.CmpIPredicate.ult, tid, s)
                rif = scf.IfOp(in_range, [], has_else=False)
                with ir.InsertionPoint(rif.then_block):
                    partner = arith.addi(tid, s)
                    mv = llvm.load(elem_ty, _gep_typed3(shm_val, tid, elem_ty))
                    mi = llvm.load(i64, _gep_typed3(shm_idx, tid, i64))
                    pv = llvm.load(
                        elem_ty, _gep_typed3(shm_val, partner, elem_ty))
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
                    scf.YieldOp([])
                gpu.barrier()
                scf.YieldOp([arith.shrui(s, one)])

            result = llvm.load(i64, _gep_typed3(shm_idx, zero, i64))
            func.ReturnOp([result])

        return str(module)


def _build_idxmax_module(dtype):
    return _build_idx_module(dtype, is_max=True)


def _build_idxmin_module(dtype):
    return _build_idx_module(dtype, is_max=False)


def _build_exp_module(dtype):
    """Build MLIR module for _coop_exp_{dtype}(ptr data, i64 size, ptr shm) -> void.

    Element-wise: out[i] = exp(in[i]).
    Results written to shm (same layout as binop).
    """
    elem_bytes = _dtype_byte_size(dtype)
    name = exp_func_name(dtype)

    with context.get_context(), ir.Location.unknown():
        from numba_cuda_mlir._mlir.dialects import math as mlir_math

        elem_ty = _dtype_mlir_type(dtype)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp(sym_name=f"{name}_module")

        body = gpu_mod.bodyRegion.blocks.append()
        ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [])
        with ir.InsertionPoint(body):
            fn = func.FuncOp(name=name, type=ft, visibility="public")
            entry = fn.add_entry_block()

        with ir.InsertionPoint(entry):
            in_data, size, shm_base = fn.arguments

            data_slot = _shm_ptr_at(shm_base, _SHM_BROADCAST_OFFSET)
            mi_slot = _shm_ptr_at(shm_base, _SHM_MI_SLOT)

            tid = _tid_x()
            is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, _const_i64(0))
            alloc_bytes = arith.muli(size, _const_i64(elem_bytes))
            alloc_bytes = arith.maxui(alloc_bytes, _const_i64(1))

            t0_if = scf.IfOp(is_t0, results_=[], has_else=False)
            with ir.InsertionPoint(t0_if.then_block):
                mi_val = _call_nrt(gpu_mod, "NRT_MemInfo_new_varsize",
                                   [T.i64()], [_ptr()], [alloc_bytes])
                data_val = _call_nrt(gpu_mod, "NRT_MemInfo_data_fast",
                                     [_ptr()], [_ptr()], [mi_val])
                llvm.store(data_val, data_slot)
                llvm.store(mi_val, mi_slot)
                scf.YieldOp([])
            gpu.barrier()

            out_data = llvm.load(_ptr(), data_slot)
            bdim = _bdim_x()
            loop = scf.ForOp(tid, size, bdim, [])
            with ir.InsertionPoint(loop.body):
                i = loop.induction_variable
                a = llvm.load(elem_ty, _gep_typed(in_data, i, elem_ty))
                if isinstance(elem_ty, (ir.F64Type, ir.F32Type)):
                    r = mlir_math.exp(a)
                else:
                    a_f = arith.sitofp(elem_ty, a)
                    r = mlir_math.exp(a_f)
                llvm.store(r, _gep_typed(out_data, i, elem_ty))
                scf.YieldOp([])
            gpu.barrier()

            func.ReturnOp([])

        return str(module)


def _build_scalar_binop_module(op, dtype):
    """Build MLIR module for _coop_s{op}_{dtype}(ptr arr, scalar, i64 size, ptr shm) -> void.

    Element-wise: out[i] = arr[i] op scalar.
    Results written to shm (same layout as binop).
    """
    elem_bytes = _dtype_byte_size(dtype)
    name = scalar_binop_func_name(op, dtype)

    with context.get_context(), ir.Location.unknown():
        elem_ty = _dtype_mlir_type(dtype)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp(sym_name=f"{name}_module")

        body = gpu_mod.bodyRegion.blocks.append()
        ft = ir.FunctionType.get([_ptr(), elem_ty, T.i64(), _ptr()], [])
        with ir.InsertionPoint(body):
            fn = func.FuncOp(name=name, type=ft, visibility="public")
            entry = fn.add_entry_block()

        with ir.InsertionPoint(entry):
            arr_data, scalar_val, size, shm_base = fn.arguments

            data_slot = _shm_ptr_at(shm_base, _SHM_BROADCAST_OFFSET)
            mi_slot = _shm_ptr_at(shm_base, _SHM_MI_SLOT)

            tid = _tid_x()
            is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, _const_i64(0))
            alloc_bytes = arith.muli(size, _const_i64(elem_bytes))
            alloc_bytes = arith.maxui(alloc_bytes, _const_i64(1))

            t0_if = scf.IfOp(is_t0, results_=[], has_else=False)
            with ir.InsertionPoint(t0_if.then_block):
                mi_val = _call_nrt(gpu_mod, "NRT_MemInfo_new_varsize",
                                   [T.i64()], [_ptr()], [alloc_bytes])
                data_val = _call_nrt(gpu_mod, "NRT_MemInfo_data_fast",
                                     [_ptr()], [_ptr()], [mi_val])
                llvm.store(data_val, data_slot)
                llvm.store(mi_val, mi_slot)
                scf.YieldOp([])
            gpu.barrier()

            out_data = llvm.load(_ptr(), data_slot)
            bdim = _bdim_x()
            loop = scf.ForOp(tid, size, bdim, [])
            with ir.InsertionPoint(loop.body):
                i = loop.induction_variable
                a = llvm.load(elem_ty, _gep_typed(arr_data, i, elem_ty))
                r = _arith_op_emit(op, a, scalar_val, elem_ty)
                llvm.store(r, _gep_typed(out_data, i, elem_ty))
                scf.YieldOp([])
            gpu.barrier()

            func.ReturnOp([])

        return str(module)


def _build_scalar_rbinop_module(op, dtype):
    """Build MLIR module for _coop_rs{op}_{dtype}(ptr arr, scalar, i64 size, ptr shm) -> void.

    Reversed element-wise: out[i] = scalar op arr[i].
    Needed for non-commutative ops (sub, div) when scalar is on the LHS.
    """
    elem_bytes = _dtype_byte_size(dtype)
    name = scalar_rbinop_func_name(op, dtype)

    with context.get_context(), ir.Location.unknown():
        elem_ty = _dtype_mlir_type(dtype)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            gpu_mod = gpu.GPUModuleOp(sym_name=f"{name}_module")

        body = gpu_mod.bodyRegion.blocks.append()
        ft = ir.FunctionType.get([_ptr(), elem_ty, T.i64(), _ptr()], [])
        with ir.InsertionPoint(body):
            fn = func.FuncOp(name=name, type=ft, visibility="public")
            entry = fn.add_entry_block()

        with ir.InsertionPoint(entry):
            arr_data, scalar_val, size, shm_base = fn.arguments

            data_slot = _shm_ptr_at(shm_base, _SHM_BROADCAST_OFFSET)
            mi_slot = _shm_ptr_at(shm_base, _SHM_MI_SLOT)

            tid = _tid_x()
            is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, _const_i64(0))
            alloc_bytes = arith.muli(size, _const_i64(elem_bytes))
            alloc_bytes = arith.maxui(alloc_bytes, _const_i64(1))

            t0_if = scf.IfOp(is_t0, results_=[], has_else=False)
            with ir.InsertionPoint(t0_if.then_block):
                mi_val = _call_nrt(gpu_mod, "NRT_MemInfo_new_varsize",
                                   [T.i64()], [_ptr()], [alloc_bytes])
                data_val = _call_nrt(gpu_mod, "NRT_MemInfo_data_fast",
                                     [_ptr()], [_ptr()], [mi_val])
                llvm.store(data_val, data_slot)
                llvm.store(mi_val, mi_slot)
                scf.YieldOp([])
            gpu.barrier()

            out_data = llvm.load(_ptr(), data_slot)
            bdim = _bdim_x()
            loop = scf.ForOp(tid, size, bdim, [])
            with ir.InsertionPoint(loop.body):
                i = loop.induction_variable
                a = llvm.load(elem_ty, _gep_typed(arr_data, i, elem_ty))
                r = _arith_op_emit(op, scalar_val, a, elem_ty)
                llvm.store(r, _gep_typed(out_data, i, elem_ty))
                scf.YieldOp([])
            gpu.barrier()

            func.ReturnOp([])

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
# Cached compilation per (func_key, cc)
# ---------------------------------------------------------------------------

_TAG_TO_DTYPE = {"f64": nb_types.float64, "f32": nb_types.float32,
                  "i64": nb_types.int64, "i32": nb_types.int32}


@lru_cache(maxsize=128)
def _compile_cooperative_func(func_key, cc):
    """Compile a single cooperative device function to PTX.

    func_key is a hashable identifier like ("binop", "add", "i64").
    Returns PTX bytes.
    """
    kind = func_key[0]
    if kind == "binop":
        _, op_name, dtype_name = func_key
        op = {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
              "div": operator.truediv}[op_name]
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_binop_module(op, dtype)
    elif kind == "sum":
        _, dtype_name = func_key
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_sum_module(dtype)
    elif kind == "mean":
        _, dtype_name = func_key
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_mean_module(dtype)
    elif kind == "min":
        _, dtype_name = func_key
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_min_module(dtype)
    elif kind == "max":
        _, dtype_name = func_key
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_max_module(dtype)
    elif kind == "std":
        _, dtype_name = func_key
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_std_module(dtype)
    elif kind == "var":
        _, dtype_name = func_key
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_var_module(dtype)
    elif kind == "corr":
        _, dtype_name = func_key
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_corr_module(dtype)
    elif kind == "idxmax":
        _, dtype_name = func_key
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_idxmax_module(dtype)
    elif kind == "idxmin":
        _, dtype_name = func_key
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_idxmin_module(dtype)
    elif kind == "exp":
        _, dtype_name = func_key
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_exp_module(dtype)
    elif kind == "scalar_binop":
        _, op_name, dtype_name = func_key
        op = {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
              "div": operator.truediv}[op_name]
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_scalar_binop_module(op, dtype)
    elif kind == "scalar_rbinop":
        _, op_name, dtype_name = func_key
        op = {"add": operator.add, "sub": operator.sub, "mul": operator.mul,
              "div": operator.truediv}[op_name]
        dtype = _TAG_TO_DTYPE[dtype_name]
        mlir_str = _build_scalar_rbinop_module(op, dtype)
    else:
        raise ValueError(f"Unknown func_key kind: {kind}")

    return _compile_mlir_module_to_ptx(mlir_str, cc)


# ---------------------------------------------------------------------------
# Linker integration: add required PTX to linker before complete()
# ---------------------------------------------------------------------------

_NEEDED_FUNCS_KEY = "_cooperative_device_funcs_needed"


def register_needed_func(cres_metadata, func_key):
    """Mark a cooperative device function as needed for linking."""
    needed = cres_metadata.setdefault(_NEEDED_FUNCS_KEY, set())
    needed.add(func_key)


def link_cooperative_funcs(linker, cres_metadata):
    """Add all needed cooperative device function PTX to the linker."""
    needed = cres_metadata.get(_NEEDED_FUNCS_KEY, set())
    if not needed:
        return
    cc = linker.cc
    cc_str = f"{cc[0]}{cc[1]}"
    for func_key in needed:
        ptx = _compile_cooperative_func(func_key, cc_str)
        linker.add_ptx(ptx)

"""
Lowerings for CooperativeArrayType operations.

Element-wise binops, .sum(), and .mean() are compiled as separate device
functions (see cooperative_device_funcs.py) and linked at link time.
This module emits only declarations + func.call sites — no bodies — so
the MLIR inliner cannot inline them.  This keeps kernel compilation
O(N) instead of O(N²) for N binops.

The struct layout mirrors mlir_string: {ptr meminfo, ptr data, i64 size}.
"""

import operator

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir.dialects import arith, func, gpu, llvm, memref, scf

from numba_cuda_mlir.lowering_utilities import (
    DeferredMethodCall,
    GEP_DYNAMIC_INDEX,
    get_or_insert_function,
)
from numba_cuda_mlir.extending import lowering_registry as registry
from cudf.core.udf.mlir_backend.cooperative_types import CooperativeArrayType
from numba_cuda_mlir.numba_cuda import types as nb_types


# ---------------------------------------------------------------------------
# MLIR type / constant helpers
# ---------------------------------------------------------------------------

def _ptr():
    return llvm.PointerType.get()


def _i64():
    return ir.IntegerType.get_signless(64)


def _const_i64(val):
    return arith.constant(T.i64(), val)


def _const_index(val):
    return arith.constant(T.index(), val)


_MAX_BLOCK_SIZE = 1024


def _dtype_mlir_type(dtype):
    if dtype == nb_types.float64:
        return ir.F64Type.get()
    elif dtype == nb_types.float32:
        return ir.F32Type.get()
    elif dtype == nb_types.int64:
        return T.i64()
    elif dtype == nb_types.int32:
        return ir.IntegerType.get_signless(32)
    raise NotImplementedError(f"Unsupported dtype {dtype}")


def _dtype_byte_size(dtype):
    if dtype in (nb_types.float64, nb_types.int64):
        return 8
    elif dtype in (nb_types.float32, nb_types.int32):
        return 4
    raise NotImplementedError(f"Unsupported dtype {dtype}")


def _call_nrt(gpu_module, name, arg_types, result_types, args):
    ft = ir.FunctionType.get(arg_types, result_types)
    callee = get_or_insert_function(name, ft, gpu_module)
    return func.call(
        result=result_types, callee=callee.name.value, operands_=args,
    )


# ---------------------------------------------------------------------------
# CooperativeArray struct accessors
# ---------------------------------------------------------------------------

def _ca_struct_type():
    return llvm.StructType.get_literal([_ptr(), _ptr(), T.i64(), _ptr()])


def build_cooperative_array(meminfo, data, size_i64, index_data=None):
    ty = _ca_struct_type()
    if index_data is None:
        index_data = llvm.ZeroOp(_ptr())
    undef = llvm.UndefOp(ty)
    with_mi = llvm.insertvalue(
        container=undef, value=meminfo,
        position=ir.DenseI64ArrayAttr.get([0]),
    )
    with_data = llvm.insertvalue(
        container=with_mi, value=data,
        position=ir.DenseI64ArrayAttr.get([1]),
    )
    with_size = llvm.insertvalue(
        container=with_data, value=size_i64,
        position=ir.DenseI64ArrayAttr.get([2]),
    )
    return llvm.insertvalue(
        container=with_size, value=index_data,
        position=ir.DenseI64ArrayAttr.get([3]),
    )


def ca_extract_meminfo(val):
    return llvm.extractvalue(_ptr(), val, [0])


def ca_extract_data(val):
    return llvm.extractvalue(_ptr(), val, [1])


def ca_extract_size(val):
    return llvm.extractvalue(T.i64(), val, [2])


def ca_extract_index(val):
    return llvm.extractvalue(_ptr(), val, [3])


# ---------------------------------------------------------------------------
# Helpers: thread/block IDs, GEP for typed arrays
# ---------------------------------------------------------------------------

def _tid_x():
    return arith.index_castui(T.i64(), gpu.thread_id("x"))


def _bdim_x():
    return arith.index_castui(T.i64(), gpu.block_dim("x"))


def _gep_typed(base_ptr, index_i64, elem_mlir_type):
    return llvm.getelementptr(
        _ptr(), base_ptr, [index_i64], [GEP_DYNAMIC_INDEX],
        elem_mlir_type, None,
    )


# ---------------------------------------------------------------------------
# Block-cooperative NRT allocation (thread 0 allocates, broadcasts)
#
# Uses 2 pointer-sized slots in shared memory to broadcast
# (meminfo, data) from thread 0 to all threads.
# ---------------------------------------------------------------------------

# Shared memory layout (fixed offsets):
#   bytes 0..7:    data pointer slot (broadcast from thread 0)
#   bytes 8..15:   meminfo pointer slot (thread 0 writes)
#   bytes 16..8207: reduction scratch (1024 x f64)
_SHM_BROADCAST_OFFSET = 0
_SHM_MI_SLOT = 8
_SHM_REDUCTION_OFFSET = 16


_MEMINFO_SIZEOF = 40  # must match nrt_mlir._MEMINFO_SIZEOF


def _shm_base_ptr(builder):
    """Extract the dynamic shared memory base as a valid generic llvm.ptr.

    Uses addrspacecast (cvta.shared) so the resulting pointer is a
    proper generic virtual address, not a raw shared-memory offset.
    """
    shm_base = builder._get_shared_memory_base()
    base_idx = memref.extract_aligned_pointer_as_index(shm_base)
    base_i64 = arith.index_cast(T.i64(), base_idx)
    ptr3_ty = ir.Type.parse("!llvm.ptr<3>")
    ptr3 = llvm.inttoptr(ptr3_ty, base_i64)
    return llvm.addrspacecast(_ptr(), ptr3)


def _shm_ptr_at_offset(builder, byte_offset):
    """Get a raw generic llvm.ptr to dynamic shared memory + byte offset."""
    base = _shm_base_ptr(builder)
    return llvm.getelementptr(
        _ptr(), base, [_const_i64(byte_offset)],
        [GEP_DYNAMIC_INDEX], T.i8(), None,
    )


# ---------------------------------------------------------------------------
# Element-wise binary ops: thin call to separately-compiled device function
# ---------------------------------------------------------------------------

def _lower_binop(builder, target, args, kwargs, op):
    """Emit a call to the outlined _coop_{op}_{dtype} device function.

    The function is compiled to a separate PTX module and linked at
    link time.  Only a declaration is inserted into the kernel's
    gpu.module — no body for the inliner to see.
    """
    from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
        binop_func_name, register_needed_func, _op_tag, _dtype_tag,
    )

    lhs_val = builder.load_var(args[0])
    rhs_val = builder.load_var(args[1])
    target_type = builder.get_numba_type(target.name)
    dtype = target_type.dtype

    lhs_data = ca_extract_data(lhs_val)
    rhs_data = ca_extract_data(rhs_val)
    size = ca_extract_size(lhs_val)

    shm = _shm_base_ptr(builder)
    fname = binop_func_name(op, dtype)
    ft = ir.FunctionType.get([_ptr(), _ptr(), T.i64(), _ptr()], [])
    gm = builder.mlir_gpu_module
    callee = get_or_insert_function(fname, ft, gm)
    func.call(result=[], callee=callee.name.value,
              operands_=[lhs_data, rhs_data, size, shm])

    # The outlined function writes data ptr to shm[0] and meminfo to shm[8].
    # Read them back here.
    data_slot = _shm_ptr_at_offset(builder, _SHM_BROADCAST_OFFSET)
    mi_slot = _shm_ptr_at_offset(builder, _SHM_MI_SLOT)
    out_data = llvm.load(_ptr(), data_slot)

    tid = _tid_x()
    is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, _const_i64(0))
    null_ptr = llvm.ZeroOp(_ptr())
    mi_if = scf.IfOp(is_t0, results_=[_ptr()], has_else=True)
    with ir.InsertionPoint(mi_if.then_block):
        scf.YieldOp([llvm.load(_ptr(), mi_slot)])
    with ir.InsertionPoint(mi_if.else_block):
        scf.YieldOp([null_ptr])
    out_mi = mi_if.results[0]

    result = build_cooperative_array(out_mi, out_data, size)
    builder.store_var(target, result)

    register_needed_func(builder.metadata,
                         ("binop", _op_tag(op), _dtype_tag(dtype)))


_SUPPORTED_DTYPES = (nb_types.float64, nb_types.float32, nb_types.int64, nb_types.int32)

for _dt in _SUPPORTED_DTYPES:
    _ca = CooperativeArrayType(_dt)
    for _op in (operator.add, operator.sub, operator.mul, operator.truediv):
        def _make_lower(op, ca_type):
            @registry.lower(op, ca_type, ca_type)
            def _lower(builder, target, args, kwargs, _op=op):
                _lower_binop(builder, target, args, kwargs, _op)
        _make_lower(_op, _ca)


# ---------------------------------------------------------------------------
# .size attribute
# ---------------------------------------------------------------------------

for _dt in _SUPPORTED_DTYPES:
    _ca = CooperativeArrayType(_dt)

    def _make_size_lower(ca_type):
        @registry.lower_getattr(ca_type, "size")
        def _lower_size(context, builder, target, value, attr=None):
            ca_val = builder.load_var(value)
            builder.store_var(target, ca_extract_size(ca_val))
    _make_size_lower(_ca)


# ---------------------------------------------------------------------------
# .sum() reduction: thin call to separately-compiled device function
# ---------------------------------------------------------------------------

def _lower_sum(builder, target, args, kwargs):
    """Emit a call to the outlined _coop_sum_{dtype} device function."""
    from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
        sum_func_name, register_needed_func, _dtype_tag,
    )

    self_var = args[0]
    ca_val = builder.load_var(self_var)
    ca_type = builder.get_numba_type(self_var.name)
    dtype = ca_type.dtype
    elem_ty = _dtype_mlir_type(dtype)

    data = ca_extract_data(ca_val)
    size = ca_extract_size(ca_val)

    shm = _shm_base_ptr(builder)
    # pandas promotes integer sums to int64; floats keep their width. The
    # device function returns this accumulator type.
    acc_ty = T.i64() if isinstance(dtype, nb_types.Integer) else elem_ty
    fname = sum_func_name(dtype)
    ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [acc_ty])
    gm = builder.mlir_gpu_module
    callee = get_or_insert_function(fname, ft, gm)
    result = func.call(
        result=[acc_ty], callee=callee.name.value,
        operands_=[data, size, shm],
    )
    builder.store_var(target, result)

    register_needed_func(builder.metadata, ("sum", _dtype_tag(dtype)))


def _lower_sum_with_release(builder, target, args, kwargs):
    """Run the sum, then release the extra reference taken at getattr time."""
    self_var = args[0]
    _lower_sum(builder, target, args, kwargs)
    ca_val = builder.load_var(self_var)
    mi = ca_extract_meminfo(ca_val)
    _call_nrt(builder.mlir_gpu_module, "NRT_decref", [_ptr()], [], [mi])


for _dt in _SUPPORTED_DTYPES:
    def _make_sum_lower(ca_type):
        @registry.lower_getattr(ca_type, "sum")
        def _lower_ca_sum_attr(context, builder, target, value, attr=None):
            ca_val = builder.load_var(value)
            mi = ca_extract_meminfo(ca_val)
            _call_nrt(builder.mlir_gpu_module, "NRT_incref", [_ptr()], [], [mi])
            builder.store_var(target, DeferredMethodCall(value, _lower_sum_with_release))
    _make_sum_lower(CooperativeArrayType(_dt))


# ---------------------------------------------------------------------------
# .mean() reduction: thin call to separately-compiled device function
# ---------------------------------------------------------------------------

def _lower_mean(builder, target, args, kwargs):
    """Emit a call to the outlined _coop_mean_{dtype} device function."""
    from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
        mean_func_name, register_needed_func, _dtype_tag,
    )

    self_var = args[0]
    ca_val = builder.load_var(self_var)
    ca_type = builder.get_numba_type(self_var.name)
    dtype = ca_type.dtype

    data = ca_extract_data(ca_val)
    size = ca_extract_size(ca_val)

    shm = _shm_base_ptr(builder)
    f64_ty = ir.F64Type.get()
    fname = mean_func_name(dtype)
    ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [f64_ty])
    gm = builder.mlir_gpu_module
    callee = get_or_insert_function(fname, ft, gm)
    result = func.call(
        result=[f64_ty], callee=callee.name.value,
        operands_=[data, size, shm],
    )
    # pandas mean of float32 stays float32; truncate the f64 accumulator.
    if dtype == nb_types.float32:
        result = arith.truncf(ir.F32Type.get(), result)
    builder.store_var(target, result)

    register_needed_func(builder.metadata, ("mean", _dtype_tag(dtype)))


def _lower_mean_with_release(builder, target, args, kwargs):
    """Run the mean, then release the extra reference taken at getattr time."""
    self_var = args[0]
    _lower_mean(builder, target, args, kwargs)
    ca_val = builder.load_var(self_var)
    mi = ca_extract_meminfo(ca_val)
    _call_nrt(builder.mlir_gpu_module, "NRT_decref", [_ptr()], [], [mi])


for _dt in _SUPPORTED_DTYPES:
    def _make_mean_lower(ca_type):
        @registry.lower_getattr(ca_type, "mean")
        def _lower_ca_mean_attr(context, builder, target, value, attr=None):
            ca_val = builder.load_var(value)
            mi = ca_extract_meminfo(ca_val)
            _call_nrt(builder.mlir_gpu_module, "NRT_incref", [_ptr()], [], [mi])
            builder.store_var(target, DeferredMethodCall(value, _lower_mean_with_release))
    _make_mean_lower(CooperativeArrayType(_dt))


# ---------------------------------------------------------------------------
# .min() reduction: thin call to separately-compiled device function
# ---------------------------------------------------------------------------

def _lower_min(builder, target, args, kwargs):
    """Emit a call to the outlined _coop_min_{dtype} device function."""
    from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
        min_func_name, register_needed_func, _dtype_tag,
    )

    self_var = args[0]
    ca_val = builder.load_var(self_var)
    ca_type = builder.get_numba_type(self_var.name)
    dtype = ca_type.dtype
    elem_ty = _dtype_mlir_type(dtype)

    data = ca_extract_data(ca_val)
    size = ca_extract_size(ca_val)

    shm = _shm_base_ptr(builder)
    fname = min_func_name(dtype)
    ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [elem_ty])
    gm = builder.mlir_gpu_module
    callee = get_or_insert_function(fname, ft, gm)
    result = func.call(
        result=[elem_ty], callee=callee.name.value,
        operands_=[data, size, shm],
    )
    builder.store_var(target, result)

    register_needed_func(builder.metadata, ("min", _dtype_tag(dtype)))


def _lower_min_with_release(builder, target, args, kwargs):
    """Run the min, then release the extra reference taken at getattr time."""
    self_var = args[0]
    _lower_min(builder, target, args, kwargs)
    ca_val = builder.load_var(self_var)
    mi = ca_extract_meminfo(ca_val)
    _call_nrt(builder.mlir_gpu_module, "NRT_decref", [_ptr()], [], [mi])


for _dt in _SUPPORTED_DTYPES:
    def _make_min_lower(ca_type):
        @registry.lower_getattr(ca_type, "min")
        def _lower_ca_min_attr(context, builder, target, value, attr=None):
            ca_val = builder.load_var(value)
            mi = ca_extract_meminfo(ca_val)
            _call_nrt(builder.mlir_gpu_module, "NRT_incref", [_ptr()], [], [mi])
            builder.store_var(target, DeferredMethodCall(value, _lower_min_with_release))
    _make_min_lower(CooperativeArrayType(_dt))


def _lower_max(builder, target, args, kwargs):
    """Emit a call to the outlined _coop_max_{dtype} device function."""
    from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
        max_func_name, register_needed_func, _dtype_tag,
    )

    self_var = args[0]
    ca_val = builder.load_var(self_var)
    ca_type = builder.get_numba_type(self_var.name)
    dtype = ca_type.dtype
    elem_ty = _dtype_mlir_type(dtype)

    data = ca_extract_data(ca_val)
    size = ca_extract_size(ca_val)

    shm = _shm_base_ptr(builder)
    fname = max_func_name(dtype)
    ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [elem_ty])
    gm = builder.mlir_gpu_module
    callee = get_or_insert_function(fname, ft, gm)
    result = func.call(
        result=[elem_ty], callee=callee.name.value,
        operands_=[data, size, shm],
    )
    builder.store_var(target, result)

    register_needed_func(builder.metadata, ("max", _dtype_tag(dtype)))


def _lower_max_with_release(builder, target, args, kwargs):
    """Run the max, then release the extra reference taken at getattr time."""
    self_var = args[0]
    _lower_max(builder, target, args, kwargs)
    ca_val = builder.load_var(self_var)
    mi = ca_extract_meminfo(ca_val)
    _call_nrt(builder.mlir_gpu_module, "NRT_decref", [_ptr()], [], [mi])


for _dt in _SUPPORTED_DTYPES:
    def _make_max_lower(ca_type):
        @registry.lower_getattr(ca_type, "max")
        def _lower_ca_max_attr(context, builder, target, value, attr=None):
            ca_val = builder.load_var(value)
            mi = ca_extract_meminfo(ca_val)
            _call_nrt(builder.mlir_gpu_module, "NRT_incref", [_ptr()], [], [mi])
            builder.store_var(target, DeferredMethodCall(value, _lower_max_with_release))
    _make_max_lower(CooperativeArrayType(_dt))


# ---------------------------------------------------------------------------
# .std() reduction: thin call to separately-compiled device function
# ---------------------------------------------------------------------------

def _lower_std(builder, target, args, kwargs):
    """Emit a call to the outlined _coop_std_{dtype} device function."""
    from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
        std_func_name, register_needed_func, _dtype_tag,
    )

    self_var = args[0]
    ca_val = builder.load_var(self_var)
    ca_type = builder.get_numba_type(self_var.name)
    dtype = ca_type.dtype

    data = ca_extract_data(ca_val)
    size = ca_extract_size(ca_val)

    shm = _shm_base_ptr(builder)
    f64_ty = ir.F64Type.get()
    fname = std_func_name(dtype)
    ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [f64_ty])
    gm = builder.mlir_gpu_module
    callee = get_or_insert_function(fname, ft, gm)
    result = func.call(
        result=[f64_ty], callee=callee.name.value,
        operands_=[data, size, shm],
    )
    if dtype == nb_types.float32:
        result = arith.truncf(ir.F32Type.get(), result)
    builder.store_var(target, result)

    register_needed_func(builder.metadata, ("std", _dtype_tag(dtype)))


def _lower_std_with_release(builder, target, args, kwargs):
    """Run the std, then release the extra reference taken at getattr time."""
    self_var = args[0]
    _lower_std(builder, target, args, kwargs)
    ca_val = builder.load_var(self_var)
    mi = ca_extract_meminfo(ca_val)
    _call_nrt(builder.mlir_gpu_module, "NRT_decref", [_ptr()], [], [mi])


for _dt in _SUPPORTED_DTYPES:
    def _make_std_lower(ca_type):
        @registry.lower_getattr(ca_type, "std")
        def _lower_ca_std_attr(context, builder, target, value, attr=None):
            ca_val = builder.load_var(value)
            mi = ca_extract_meminfo(ca_val)
            _call_nrt(builder.mlir_gpu_module, "NRT_incref", [_ptr()], [], [mi])
            builder.store_var(target, DeferredMethodCall(value, _lower_std_with_release))
    _make_std_lower(CooperativeArrayType(_dt))


def _lower_var(builder, target, args, kwargs):
    """Emit a call to the outlined _coop_var_{dtype} device function."""
    from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
        var_func_name, register_needed_func, _dtype_tag,
    )

    self_var = args[0]
    ca_val = builder.load_var(self_var)
    ca_type = builder.get_numba_type(self_var.name)
    dtype = ca_type.dtype

    data = ca_extract_data(ca_val)
    size = ca_extract_size(ca_val)

    shm = _shm_base_ptr(builder)
    f64_ty = ir.F64Type.get()
    fname = var_func_name(dtype)
    ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [f64_ty])
    gm = builder.mlir_gpu_module
    callee = get_or_insert_function(fname, ft, gm)
    result = func.call(
        result=[f64_ty], callee=callee.name.value,
        operands_=[data, size, shm],
    )
    if dtype == nb_types.float32:
        result = arith.truncf(ir.F32Type.get(), result)
    builder.store_var(target, result)

    register_needed_func(builder.metadata, ("var", _dtype_tag(dtype)))


def _lower_var_with_release(builder, target, args, kwargs):
    """Run the var, then release the extra reference taken at getattr time."""
    self_var = args[0]
    _lower_var(builder, target, args, kwargs)
    ca_val = builder.load_var(self_var)
    mi = ca_extract_meminfo(ca_val)
    _call_nrt(builder.mlir_gpu_module, "NRT_decref", [_ptr()], [], [mi])


for _dt in _SUPPORTED_DTYPES:
    def _make_var_lower(ca_type):
        @registry.lower_getattr(ca_type, "var")
        def _lower_ca_var_attr(context, builder, target, value, attr=None):
            ca_val = builder.load_var(value)
            mi = ca_extract_meminfo(ca_val)
            _call_nrt(builder.mlir_gpu_module, "NRT_incref", [_ptr()], [], [mi])
            builder.store_var(target, DeferredMethodCall(value, _lower_var_with_release))
    _make_var_lower(CooperativeArrayType(_dt))


# ---------------------------------------------------------------------------
# .corr(other) reduction: thin call to separately-compiled device function
# ---------------------------------------------------------------------------

def _lower_corr(builder, target, args, kwargs):
    """Emit a call to the outlined _coop_corr_{dtype} device function."""
    from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
        corr_func_name, register_needed_func, _dtype_tag,
    )

    self_var = args[0]
    other_var = args[1]
    lhs_val = builder.load_var(self_var)
    rhs_val = builder.load_var(other_var)
    ca_type = builder.get_numba_type(self_var.name)
    dtype = ca_type.dtype

    lhs_data = ca_extract_data(lhs_val)
    rhs_data = ca_extract_data(rhs_val)
    size = ca_extract_size(lhs_val)

    shm = _shm_base_ptr(builder)
    f64_ty = ir.F64Type.get()
    fname = corr_func_name(dtype)
    ft = ir.FunctionType.get([_ptr(), _ptr(), T.i64(), _ptr()], [f64_ty])
    gm = builder.mlir_gpu_module
    callee = get_or_insert_function(fname, ft, gm)
    result = func.call(
        result=[f64_ty], callee=callee.name.value,
        operands_=[lhs_data, rhs_data, size, shm],
    )
    builder.store_var(target, result)

    register_needed_func(builder.metadata, ("corr", _dtype_tag(dtype)))


def _lower_corr_with_release(builder, target, args, kwargs):
    """Run corr, then release the extra reference taken at getattr time."""
    self_var = args[0]
    _lower_corr(builder, target, args, kwargs)
    ca_val = builder.load_var(self_var)
    mi = ca_extract_meminfo(ca_val)
    _call_nrt(builder.mlir_gpu_module, "NRT_decref", [_ptr()], [], [mi])


for _dt in _SUPPORTED_DTYPES:
    def _make_corr_lower(ca_type):
        @registry.lower_getattr(ca_type, "corr")
        def _lower_ca_corr_attr(context, builder, target, value, attr=None):
            ca_val = builder.load_var(value)
            mi = ca_extract_meminfo(ca_val)
            _call_nrt(builder.mlir_gpu_module, "NRT_incref", [_ptr()], [], [mi])
            builder.store_var(target, DeferredMethodCall(value, _lower_corr_with_release))
    _make_corr_lower(CooperativeArrayType(_dt))


# ---------------------------------------------------------------------------
# .idxmax()/.idxmin() reductions: return the int64 index label of the extremum
# ---------------------------------------------------------------------------

def _lower_idx(builder, target, args, kwargs, *, is_max):
    from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
        idxmax_func_name, idxmin_func_name, register_needed_func, _dtype_tag,
    )

    self_var = args[0]
    ca_val = builder.load_var(self_var)
    ca_type = builder.get_numba_type(self_var.name)
    dtype = ca_type.dtype

    data = ca_extract_data(ca_val)
    index = ca_extract_index(ca_val)
    size = ca_extract_size(ca_val)

    shm = _shm_base_ptr(builder)
    i64_ty = T.i64()
    fname = idxmax_func_name(dtype) if is_max else idxmin_func_name(dtype)
    ft = ir.FunctionType.get([_ptr(), _ptr(), T.i64(), _ptr()], [i64_ty])
    gm = builder.mlir_gpu_module
    callee = get_or_insert_function(fname, ft, gm)
    result = func.call(
        result=[i64_ty], callee=callee.name.value,
        operands_=[data, index, size, shm],
    )
    builder.store_var(target, result)

    kind = "idxmax" if is_max else "idxmin"
    register_needed_func(builder.metadata, (kind, _dtype_tag(dtype)))


def _make_idx_with_release(is_max):
    def _impl(builder, target, args, kwargs):
        self_var = args[0]
        _lower_idx(builder, target, args, kwargs, is_max=is_max)
        ca_val = builder.load_var(self_var)
        mi = ca_extract_meminfo(ca_val)
        _call_nrt(builder.mlir_gpu_module, "NRT_decref", [_ptr()], [], [mi])
    return _impl


_lower_idxmax_with_release = _make_idx_with_release(True)
_lower_idxmin_with_release = _make_idx_with_release(False)


for _dt in _SUPPORTED_DTYPES:
    def _make_idx_lower(ca_type):
        @registry.lower_getattr(ca_type, "idxmax")
        def _lower_ca_idxmax_attr(context, builder, target, value, attr=None):
            ca_val = builder.load_var(value)
            mi = ca_extract_meminfo(ca_val)
            _call_nrt(builder.mlir_gpu_module, "NRT_incref", [_ptr()], [], [mi])
            builder.store_var(
                target,
                DeferredMethodCall(value, _lower_idxmax_with_release),
            )

        @registry.lower_getattr(ca_type, "idxmin")
        def _lower_ca_idxmin_attr(context, builder, target, value, attr=None):
            ca_val = builder.load_var(value)
            mi = ca_extract_meminfo(ca_val)
            _call_nrt(builder.mlir_gpu_module, "NRT_incref", [_ptr()], [], [mi])
            builder.store_var(
                target,
                DeferredMethodCall(value, _lower_idxmin_with_release),
            )
    _make_idx_lower(CooperativeArrayType(_dt))


# ---------------------------------------------------------------------------
# .exp() element-wise: thin call to separately-compiled device function
# ---------------------------------------------------------------------------

def _lower_exp(builder, target, args, kwargs):
    """Emit a call to the outlined _coop_exp_{dtype} device function."""
    from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
        exp_func_name, register_needed_func, _dtype_tag,
    )

    self_var = args[0]
    ca_val = builder.load_var(self_var)
    ca_type = builder.get_numba_type(self_var.name)
    dtype = ca_type.dtype

    data = ca_extract_data(ca_val)
    size = ca_extract_size(ca_val)

    shm = _shm_base_ptr(builder)
    fname = exp_func_name(dtype)
    ft = ir.FunctionType.get([_ptr(), T.i64(), _ptr()], [])
    gm = builder.mlir_gpu_module
    callee = get_or_insert_function(fname, ft, gm)
    func.call(result=[], callee=callee.name.value,
              operands_=[data, size, shm])

    data_slot = _shm_ptr_at_offset(builder, _SHM_BROADCAST_OFFSET)
    mi_slot = _shm_ptr_at_offset(builder, _SHM_MI_SLOT)
    out_data = llvm.load(_ptr(), data_slot)

    tid = _tid_x()
    is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, _const_i64(0))
    null_ptr = llvm.ZeroOp(_ptr())
    mi_if = scf.IfOp(is_t0, results_=[_ptr()], has_else=True)
    with ir.InsertionPoint(mi_if.then_block):
        scf.YieldOp([llvm.load(_ptr(), mi_slot)])
    with ir.InsertionPoint(mi_if.else_block):
        scf.YieldOp([null_ptr])
    out_mi = mi_if.results[0]

    result = build_cooperative_array(out_mi, out_data, size)
    builder.store_var(target, result)

    register_needed_func(builder.metadata, ("exp", _dtype_tag(dtype)))


def _lower_exp_with_release(builder, target, args, kwargs):
    self_var = args[0]
    _lower_exp(builder, target, args, kwargs)
    ca_val = builder.load_var(self_var)
    mi = ca_extract_meminfo(ca_val)
    _call_nrt(builder.mlir_gpu_module, "NRT_decref", [_ptr()], [], [mi])


for _dt in _SUPPORTED_DTYPES:
    def _make_exp_lower(ca_type):
        @registry.lower_getattr(ca_type, "exp")
        def _lower_ca_exp_attr(context, builder, target, value, attr=None):
            ca_val = builder.load_var(value)
            mi = ca_extract_meminfo(ca_val)
            _call_nrt(builder.mlir_gpu_module, "NRT_incref", [_ptr()], [], [mi])
            builder.store_var(target, DeferredMethodCall(value, _lower_exp_with_release))
    _make_exp_lower(CooperativeArrayType(_dt))


# ---------------------------------------------------------------------------
# Scalar broadcast binops: CooperativeArray op scalar -> CooperativeArray
# ---------------------------------------------------------------------------

def _lower_scalar_binop(builder, target, args, kwargs, op, arr_is_lhs):
    """Emit a call to the outlined scalar binop device function."""
    from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
        scalar_binop_func_name, scalar_rbinop_func_name,
        register_needed_func, _op_tag, _dtype_tag,
    )

    if arr_is_lhs:
        arr_val = builder.load_var(args[0])
        scalar_val = builder.load_var(args[1])
    else:
        scalar_val = builder.load_var(args[0])
        arr_val = builder.load_var(args[1])

    target_type = builder.get_numba_type(target.name)
    dtype = target_type.dtype
    elem_ty = _dtype_mlir_type(dtype)

    arr_data = ca_extract_data(arr_val)
    size = ca_extract_size(arr_val)

    from numba_cuda_mlir.lowering_utilities import convert
    scalar_cast = convert(scalar_val, elem_ty)

    use_reversed = (not arr_is_lhs) and op in (operator.sub, operator.truediv)

    shm = _shm_base_ptr(builder)
    if use_reversed:
        fname = scalar_rbinop_func_name(op, dtype)
        func_key = ("scalar_rbinop", _op_tag(op), _dtype_tag(dtype))
    else:
        fname = scalar_binop_func_name(op, dtype)
        func_key = ("scalar_binop", _op_tag(op), _dtype_tag(dtype))
    ft = ir.FunctionType.get([_ptr(), elem_ty, T.i64(), _ptr()], [])
    gm = builder.mlir_gpu_module
    callee = get_or_insert_function(fname, ft, gm)
    func.call(result=[], callee=callee.name.value,
              operands_=[arr_data, scalar_cast, size, shm])

    data_slot = _shm_ptr_at_offset(builder, _SHM_BROADCAST_OFFSET)
    mi_slot = _shm_ptr_at_offset(builder, _SHM_MI_SLOT)
    out_data = llvm.load(_ptr(), data_slot)

    tid = _tid_x()
    is_t0 = arith.cmpi(arith.CmpIPredicate.eq, tid, _const_i64(0))
    null_ptr = llvm.ZeroOp(_ptr())
    mi_if = scf.IfOp(is_t0, results_=[_ptr()], has_else=True)
    with ir.InsertionPoint(mi_if.then_block):
        scf.YieldOp([llvm.load(_ptr(), mi_slot)])
    with ir.InsertionPoint(mi_if.else_block):
        scf.YieldOp([null_ptr])
    out_mi = mi_if.results[0]

    result = build_cooperative_array(out_mi, out_data, size)
    builder.store_var(target, result)

    register_needed_func(builder.metadata, func_key)


for _dt in _SUPPORTED_DTYPES:
    _ca = CooperativeArrayType(_dt)
    for _op in (operator.add, operator.sub, operator.mul, operator.truediv):
        for _scalar_ty in (nb_types.int64, nb_types.int32, nb_types.float64, nb_types.float32):
            def _make_scalar_lower(op, ca_type, scalar_type):
                @registry.lower(op, ca_type, scalar_type)
                def _lower_arr_op_scalar(builder, target, args, kwargs, _op=op):
                    _lower_scalar_binop(builder, target, args, kwargs, _op, arr_is_lhs=True)

                @registry.lower(op, scalar_type, ca_type)
                def _lower_scalar_op_arr(builder, target, args, kwargs, _op=op):
                    _lower_scalar_binop(builder, target, args, kwargs, _op, arr_is_lhs=False)
            _make_scalar_lower(_op, _ca, _scalar_ty)


# ---------------------------------------------------------------------------
# Factory: cooperative_array_from_ptr(data_ptr, size) -> CooperativeArray(f64)
# ---------------------------------------------------------------------------

def _register_cooperative_array_from_ptr():
    from cudf.core.udf.mlir_backend.cooperative_typing import cooperative_array_from_ptr as _stub

    @registry.lower(_stub, nb_types.int64, nb_types.int64)
    def _lower_from_ptr(builder, target, args, kwargs):
        data_i64 = builder.load_var(args[0])
        size_i64 = builder.load_var(args[1])
        data_ptr = llvm.inttoptr(_ptr(), data_i64)
        null_ptr = llvm.ZeroOp(_ptr())
        result = build_cooperative_array(null_ptr, data_ptr, size_i64)
        builder.store_var(target, result)


_register_cooperative_array_from_ptr()


# ---------------------------------------------------------------------------
# Constructor: CooperativeArray(Array, int64) -> CooperativeArrayType
#
# Extracts the base pointer from the memref (including slice offset)
# and builds a non-owning view (meminfo=null).
# ---------------------------------------------------------------------------

def _get_base_ptr_from_memref(array_val):
    """Extract the data pointer from a (possibly sliced) memref."""
    if not isinstance(array_val.type, ir.MemRefType):
        return array_val
    md = memref.extract_strided_metadata(array_val)
    base_memref = md[0]
    offset = md[1]
    base_idx = memref.extract_aligned_pointer_as_index(base_memref)
    base_i64 = arith.index_cast(T.i64(), base_idx)
    elem_ty = ir.MemRefType(array_val.type).element_type
    if isinstance(elem_ty, ir.IntegerType):
        elem_bytes = elem_ty.width // 8
    elif isinstance(elem_ty, ir.FloatType):
        elem_bytes = elem_ty.width // 8
    else:
        elem_bytes = 8
    offset_i64 = arith.index_cast(T.i64(), offset)
    offset_bytes = arith.muli(offset_i64, _const_i64(elem_bytes))
    data_i64 = arith.addi(base_i64, offset_bytes)
    return llvm.inttoptr(_ptr(), data_i64)


def _register_cooperative_array_constructor():
    from cudf.core.udf.mlir_backend.cooperative_typing import CooperativeArray as _ctor

    @registry.lower(_ctor, nb_types.Array, nb_types.Integer)
    def _lower_ctor(builder, target, args, kwargs):
        array_val = builder.load_var(args[0])
        size_val = builder.load_var(args[1])
        from numba_cuda_mlir.lowering_utilities import convert
        size_i64 = convert(size_val, T.i64())
        data_ptr = _get_base_ptr_from_memref(array_val)
        null_ptr = llvm.ZeroOp(_ptr())
        result = build_cooperative_array(null_ptr, data_ptr, size_i64)
        builder.store_var(target, result)

    @registry.lower(_ctor, nb_types.Array, nb_types.Integer, nb_types.Array)
    def _lower_ctor_with_index(builder, target, args, kwargs):
        array_val = builder.load_var(args[0])
        size_val = builder.load_var(args[1])
        index_val = builder.load_var(args[2])
        from numba_cuda_mlir.lowering_utilities import convert
        size_i64 = convert(size_val, T.i64())
        data_ptr = _get_base_ptr_from_memref(array_val)
        index_ptr = _get_base_ptr_from_memref(index_val)
        null_ptr = llvm.ZeroOp(_ptr())
        result = build_cooperative_array(
            null_ptr, data_ptr, size_i64, index_ptr
        )
        builder.store_var(target, result)


_register_cooperative_array_constructor()


def _install_cooperative_link_patch():
    """Link cooperative device-function PTX into the kernel at complete().

    WORKAROUND (released numba-cuda-mlir): the fork wired
    ``link_cooperative_funcs`` into ``mlir_optimization.optimize`` just before
    ``linker.complete()``; the release has no such hook. Wrap ``optimize`` so
    that, for kernels that requested cooperative device functions, the linker's
    ``complete`` first adds the needed PTX. See mlir_upstream_notes.md item 4.
    """
    import numba_cuda_mlir.mlir_optimization as _opt
    from numba_cuda_mlir.numba_cuda.cudadrv.driver import _Linker

    from cudf.core.udf.mlir_backend.cooperative_device_funcs import (
        _NEEDED_FUNCS_KEY,
        link_cooperative_funcs,
    )

    if getattr(_opt.optimize, "_cudf_coop_patched", False):
        return

    _orig_optimize = _opt.optimize

    def _optimize(cres):
        if not cres.metadata.get(_NEEDED_FUNCS_KEY):
            return _orig_optimize(cres)
        _orig_complete = _Linker.complete

        def _complete(self):
            link_cooperative_funcs(self, cres.metadata)
            return _orig_complete(self)

        _Linker.complete = _complete
        try:
            return _orig_optimize(cres)
        finally:
            _Linker.complete = _orig_complete

    _optimize._cudf_coop_patched = True
    _opt.optimize = _optimize


_install_cooperative_link_patch()

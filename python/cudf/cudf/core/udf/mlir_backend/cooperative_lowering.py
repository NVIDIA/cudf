# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lowerings for CooperativeArrayType operations.

Cooperative block reductions (.sum(), .mean(), .min(), .max(), .std(),
.var(), .corr(), .idxmin(), .idxmax()) are compiled once into a single PTX
reduction shim (see cooperative_device_funcs.py). This module emits only
declarations + func.call sites — no bodies — so the MLIR inliner cannot inline
them. Callers link the shim into the kernel by passing
``link=[cooperative_reductions_linkable()]`` to ``cuda.jit``.

Every CooperativeArray is a non-owning view: meminfo (bytes 0..7 of the
{ptr meminfo, ptr data, i64 size, ptr index} struct) is null and no
NRT-managed, array-returning operations are implemented, so cooperative
reductions pull in no NRT runtime.

Call ``register_cooperative_lowerings()`` to install the lowerings; it is invoked
once at import, but is a single entry point so the registration cost can be
measured or deferred.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, func, llvm, memref
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.extending import lowering_registry as registry
from numba_cuda_mlir.lowering_utilities import (
    DeferredMethodCall,
    get_or_insert_function,
)
from numba_cuda_mlir.numba_cuda import types as nb_types

from cudf.core.udf.mlir_backend.cooperative_device_funcs import func_name
from cudf.core.udf.mlir_backend.cooperative_types import CooperativeArrayType

if TYPE_CHECKING:
    from collections.abc import Callable

    from numba_cuda_mlir.mlir_lowering import MLIRLower
    from numba_cuda_mlir.numba_cuda.core.ir import Var


# ---------------------------------------------------------------------------
# MLIR type / constant helpers
# ---------------------------------------------------------------------------


def _ptr() -> ir.Type:
    return llvm.PointerType.get()


def _const_i64(val: int) -> ir.Value:
    return arith.constant(T.i64(), val)


def _dtype_mlir_type(dtype) -> ir.Type:
    if dtype == nb_types.float64:
        return ir.F64Type.get()
    elif dtype == nb_types.float32:
        return ir.F32Type.get()
    elif dtype == nb_types.int64:
        return T.i64()
    elif dtype == nb_types.int32:
        return ir.IntegerType.get_signless(32)
    raise NotImplementedError(f"Unsupported dtype {dtype}")


# ---------------------------------------------------------------------------
# CooperativeArray struct accessors
# ---------------------------------------------------------------------------


def _ca_struct_type() -> ir.Type:
    return llvm.StructType.get_literal([_ptr(), _ptr(), T.i64(), _ptr()])


def build_cooperative_array(
    meminfo: ir.Value,
    data: ir.Value,
    size_i64: ir.Value,
    index_data: ir.Value | None = None,
) -> ir.Value:
    ty = _ca_struct_type()
    if index_data is None:
        index_data = llvm.ZeroOp(_ptr())
    undef = llvm.UndefOp(ty)
    with_mi = llvm.insertvalue(
        container=undef,
        value=meminfo,
        position=ir.DenseI64ArrayAttr.get([0]),
    )
    with_data = llvm.insertvalue(
        container=with_mi,
        value=data,
        position=ir.DenseI64ArrayAttr.get([1]),
    )
    with_size = llvm.insertvalue(
        container=with_data,
        value=size_i64,
        position=ir.DenseI64ArrayAttr.get([2]),
    )
    return llvm.insertvalue(
        container=with_size,
        value=index_data,
        position=ir.DenseI64ArrayAttr.get([3]),
    )


def ca_extract_data(val: ir.Value) -> ir.Value:
    return llvm.extractvalue(_ptr(), val, [1])


def ca_extract_size(val: ir.Value) -> ir.Value:
    return llvm.extractvalue(T.i64(), val, [2])


def ca_extract_index(val: ir.Value) -> ir.Value:
    return llvm.extractvalue(_ptr(), val, [3])


def _shm_base_ptr(builder: MLIRLower) -> ir.Value:
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


# Dtypes with cooperative reduction support.
_SUPPORTED_DTYPES = (
    nb_types.float64,
    nb_types.float32,
    nb_types.int64,
    nb_types.int32,
)


# ---------------------------------------------------------------------------
# Reduction call sites: a thin declaration + func.call to the outlined shim
# function _coop_<op>_<dtype>. The shared memory base is always the last operand.
# ---------------------------------------------------------------------------


def _call_reduction(
    builder: MLIRLower,
    op: str,
    dtype,
    operands: list[ir.Value],
    result_ty: ir.Type,
) -> ir.Value:
    """Insert a declaration of ``_coop_<op>_<dtype>`` and call it.

    ``operands`` are the already-materialized MLIR values (data pointer(s), size,
    shared-memory base); the callee signature is inferred from their types.
    """
    ft = ir.FunctionType.get([o.type for o in operands], [result_ty])
    callee = get_or_insert_function(
        func_name(op, dtype), ft, builder.mlir_gpu_module
    )
    return func.call(
        result=[result_ty], callee=callee.name.value, operands_=operands
    )


def _reduction_operands(
    builder: MLIRLower, ca_val: ir.Value
) -> list[ir.Value]:
    """(data, size, shm) \u2014 operands shared by the single-array reductions."""
    return [
        ca_extract_data(ca_val),
        ca_extract_size(ca_val),
        _shm_base_ptr(builder),
    ]


def _self_dtype(builder: MLIRLower, self_var: Var):
    return builder.get_numba_type(self_var.name).dtype


def _lower_sum(
    builder: MLIRLower, target: Var, args: list[Var], kwargs
) -> None:
    ca_val = builder.load_var(args[0])
    dtype = _self_dtype(builder, args[0])
    # pandas promotes integer sums to int64 (overflow-safe); floats keep width.
    acc_ty = (
        T.i64()
        if isinstance(dtype, nb_types.Integer)
        else _dtype_mlir_type(dtype)
    )
    result = _call_reduction(
        builder, "sum", dtype, _reduction_operands(builder, ca_val), acc_ty
    )
    builder.store_var(target, result)


def _lower_mean(
    builder: MLIRLower, target: Var, args: list[Var], kwargs
) -> None:
    ca_val = builder.load_var(args[0])
    dtype = _self_dtype(builder, args[0])
    result = _call_reduction(
        builder,
        "mean",
        dtype,
        _reduction_operands(builder, ca_val),
        ir.F64Type.get(),
    )
    # pandas mean of float32 stays float32; truncate the f64 accumulator.
    if dtype == nb_types.float32:
        result = arith.truncf(ir.F32Type.get(), result)
    builder.store_var(target, result)


def _lower_minmax(
    builder: MLIRLower, target: Var, args: list[Var], op: str
) -> None:
    ca_val = builder.load_var(args[0])
    dtype = _self_dtype(builder, args[0])
    result = _call_reduction(
        builder,
        op,
        dtype,
        _reduction_operands(builder, ca_val),
        _dtype_mlir_type(dtype),
    )
    builder.store_var(target, result)


def _lower_min(
    builder: MLIRLower, target: Var, args: list[Var], kwargs
) -> None:
    _lower_minmax(builder, target, args, "min")


def _lower_max(
    builder: MLIRLower, target: Var, args: list[Var], kwargs
) -> None:
    _lower_minmax(builder, target, args, "max")


def _lower_std_var(
    builder: MLIRLower, target: Var, args: list[Var], op: str
) -> None:
    ca_val = builder.load_var(args[0])
    dtype = _self_dtype(builder, args[0])
    result = _call_reduction(
        builder,
        op,
        dtype,
        _reduction_operands(builder, ca_val),
        ir.F64Type.get(),
    )
    if dtype == nb_types.float32:
        result = arith.truncf(ir.F32Type.get(), result)
    builder.store_var(target, result)


def _lower_std(
    builder: MLIRLower, target: Var, args: list[Var], kwargs
) -> None:
    _lower_std_var(builder, target, args, "std")


def _lower_var(
    builder: MLIRLower, target: Var, args: list[Var], kwargs
) -> None:
    _lower_std_var(builder, target, args, "var")


def _lower_corr(
    builder: MLIRLower, target: Var, args: list[Var], kwargs
) -> None:
    lhs_val = builder.load_var(args[0])
    rhs_val = builder.load_var(args[1])
    dtype = _self_dtype(builder, args[0])
    operands = [
        ca_extract_data(lhs_val),
        ca_extract_data(rhs_val),
        ca_extract_size(lhs_val),
        _shm_base_ptr(builder),
    ]
    result = _call_reduction(
        builder, "corr", dtype, operands, ir.F64Type.get()
    )
    builder.store_var(target, result)


def _lower_idx(
    builder: MLIRLower, target: Var, args: list[Var], op: str
) -> None:
    ca_val = builder.load_var(args[0])
    dtype = _self_dtype(builder, args[0])
    operands = [
        ca_extract_data(ca_val),
        ca_extract_index(ca_val),
        ca_extract_size(ca_val),
        _shm_base_ptr(builder),
    ]
    result = _call_reduction(builder, op, dtype, operands, T.i64())
    builder.store_var(target, result)


def _lower_idxmax(
    builder: MLIRLower, target: Var, args: list[Var], kwargs
) -> None:
    _lower_idx(builder, target, args, "idxmax")


def _lower_idxmin(
    builder: MLIRLower, target: Var, args: list[Var], kwargs
) -> None:
    _lower_idx(builder, target, args, "idxmin")


# (attribute name -> lowering) for every CooperativeArray reduction.
_REDUCTIONS: tuple[tuple[str, Callable], ...] = (
    ("sum", _lower_sum),
    ("mean", _lower_mean),
    ("min", _lower_min),
    ("max", _lower_max),
    ("std", _lower_std),
    ("var", _lower_var),
    ("corr", _lower_corr),
    ("idxmax", _lower_idxmax),
    ("idxmin", _lower_idxmin),
)


# ---------------------------------------------------------------------------
# Constructor: CooperativeArray(Array, int64) -> CooperativeArrayType
#
# Extracts the base pointer from the memref (including slice offset)
# and builds a non-owning view (meminfo=null).
# ---------------------------------------------------------------------------


def _get_base_ptr_from_memref(array_val: ir.Value) -> ir.Value:
    """Extract the data pointer from a (possibly sliced) memref.

    Strides are intentionally ignored: the group/row columns handed to these
    kernels are backed by libcudf column views of fixed-width types, which have
    no stride concept (element i is always at data + (offset+i)*itemsize), and
    the kernel arg is typed C-contiguous ([::1]) accordingly. So the element
    stride is always 1 and only the slice's start offset (md[1]) matters.
    """
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


def _register_cooperative_array_constructor() -> None:
    from cudf.core.udf.mlir_backend.cooperative_typing import (
        CooperativeArray as _ctor,
    )

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


# ---------------------------------------------------------------------------
# Registration (single entry point so its cost can be measured/deferred)
# ---------------------------------------------------------------------------


def _register_size(ca_type: CooperativeArrayType) -> None:
    @registry.lower_getattr(ca_type, "size")
    def _lower_size(context, builder, target, value, attr=None):
        builder.store_var(target, ca_extract_size(builder.load_var(value)))


def _register_reduction(
    ca_type: CooperativeArrayType, name: str, lower_fn: Callable
) -> None:
    @registry.lower_getattr(ca_type, name)
    def _lower_attr(context, builder, target, value, attr=None):
        builder.store_var(target, DeferredMethodCall(value, lower_fn))


def register_cooperative_lowerings() -> None:
    """Install .size / reduction getattr lowerings and the constructor."""
    for dt in _SUPPORTED_DTYPES:
        ca_type = CooperativeArrayType(dt)
        _register_size(ca_type)
        for name, lower_fn in _REDUCTIONS:
            _register_reduction(ca_type, name, lower_fn)
    _register_cooperative_array_constructor()


register_cooperative_lowerings()

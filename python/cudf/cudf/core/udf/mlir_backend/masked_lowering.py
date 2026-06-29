# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import operator
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
from numba_cuda_mlir import types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import arith, linalg, llvm, tensor
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.extending import lower_cast, lowering_registry
from numba_cuda_mlir.lowering_utilities import (
    bool_of,
    coerce_numpy_scalars_for_binary_op,
    concretize_tuple_to_tensor,
    convert,
    equal,
    false,
    float_of,
    int_of,
    try_extract_constant,
)
from numba_cuda_mlir.models import PrimitiveModel, register_model
from numba_cuda_mlir.numba_cuda import typing as nb_typing
from numba_cuda_mlir.numba_cuda.core import ir as numba_ir
from numba_cuda_mlir.numba_cuda.types.misc import unliteral

from cudf.core.udf._ops import (
    arith_ops,
    bitwise_ops,
    comparison_ops,
    unary_ops,
)
from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.mlir_backend.masked_typing import (
    MaskedType,
    NAType,
    na_type,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numba_cuda_mlir.mlir_lowering import MLIRLower
    from numba_cuda_mlir.numba_cuda.core.ir import Var
    from numba_cuda_mlir.numba_cuda.datamodel.manager import (
        DataModelManager,
    )


def _pack_masked(
    builder: MLIRLower,
    target_type: MaskedType,
    value: mlir_ir.Value,
    valid: mlir_ir.Value,
) -> mlir_ir.Value:
    """Build a ``MaskedType`` struct value from a ``value`` and a ``valid`` bit.

    Used by the constructor lowering and by ``pack_return`` for plain
    scalars.
    """
    struct_ty = builder.get_mlir_type(target_type)
    undef = llvm.UndefOp(struct_ty)
    with_value = llvm.insertvalue(
        container=undef,
        value=value,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    with_valid = llvm.insertvalue(
        container=with_value,
        value=valid,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    return with_valid


class MaskedTypeModel(PrimitiveModel):
    def __init__(
        self, data_model_manager: DataModelManager, masked_type: MaskedType
    ) -> None:
        value_mlir = data_model_manager.lookup(
            masked_type.value_type
        ).get_value_type()
        valid_mlir = data_model_manager.lookup(types.boolean).get_value_type()
        struct_type = llvm.StructType.new_identified(
            masked_type.name, [value_mlir, valid_mlir]
        )
        super().__init__(data_model_manager, masked_type, struct_type)


def _extract_masked_value_valid(struct_val, value_mlir_ty, valid_ty):
    """Pull the ``(value, valid)`` SSA values out of a ``Masked`` struct."""
    v = llvm.extractvalue(
        res=value_mlir_ty,
        container=struct_val,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    valid = llvm.extractvalue(
        res=valid_ty,
        container=struct_val,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    return v, valid


def _lower_masked_constructor(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    if kwargs:
        raise TypeError(
            "Masked(value, valid) does not accept keyword arguments; "
            f"got {kwargs!r}"
        )
    target_type = builder.get_numba_type(target.name)
    val_var, valid_var = args

    # Coerce the raw value to the target's declared MLIR value type
    # (handles e.g. Literal[bool](True) flowing in for ``valid``).
    value_mlir_ty = builder.get_mlir_type(target_type.value_type)
    valid_mlir_ty = builder.get_mlir_type(types.boolean)
    value_mlir = convert(builder.load_var(val_var), value_mlir_ty)
    valid_mlir = convert(builder.load_var(valid_var), valid_mlir_ty)

    struct_val = _pack_masked(builder, target_type, value_mlir, valid_mlir)
    builder.store_var(target, struct_val)


def _lower_masked_getattr(
    context, builder: MLIRLower, target: Var, value: Var, attr: str
) -> None:
    struct_value = builder.load_var(value)
    if attr == "value":
        field_index = 0
    elif attr == "valid":
        field_index = 1
    else:
        raise AttributeError(f"MaskedType has no attribute {attr!r}")
    struct_ty = llvm.StructType(struct_value.type)
    field_mlir_ty = struct_ty.body[field_index]
    field_value = llvm.extractvalue(
        res=field_mlir_ty,
        container=struct_value,
        position=mlir_ir.DenseI64ArrayAttr.get([field_index]),
    )
    target_mlir_ty = builder.get_mlir_type(builder.get_numba_type(target.name))
    builder.store_var(target, convert(field_value, target_mlir_ty))


# ``cast(NA -> Masked)`` and ``cast(scalar -> Masked)``. Both build a Masked
# struct for the target's value type; they differ only in the payload (NA has
# none, so use undef) and the validity bit (NA -> invalid, scalar -> valid).
# Triggered by branch unification, e.g. ``return x if cond else cudf.NA`` or
# ``return 5``.
def _cast_to_masked(context, builder, from_ty, to_ty, val):
    value_mlir_ty = builder.get_mlir_type(to_ty.value_type)
    if isinstance(from_ty, NAType):
        value = llvm.UndefOp(value_mlir_ty)
        valid = 0
    else:
        value = convert(val, value_mlir_ty)
        valid = 1
    valid_const = arith.constant(
        result=builder.get_mlir_type(types.boolean), value=valid
    )
    return _pack_masked(builder, to_ty, value, valid_const)


# ``cast(Masked -> Masked)``: branch unification across different
# inner widths (e.g. one branch returns Masked(int32), another
# returns Masked(float64); Numba unifies to Masked(float64)).
# Promote the payload, preserve the validity bit.
def _cast_masked_to_masked(context, builder, from_ty, to_ty, val):
    if from_ty.value_type == to_ty.value_type:
        return val
    st = llvm.StructType(val.type)
    m_val, m_valid = _extract_masked_value_valid(val, st.body[0], st.body[1])
    value_mlir_ty = builder.get_mlir_type(to_ty.value_type)
    casted = convert(m_val, value_mlir_ty)
    return _pack_masked(builder, to_ty, casted, m_valid)


# ``is``/``is not`` against NA are registered for both operand orders
# (``m is NA`` and ``NA is m``), so find the MaskedType operand rather than
# assuming which position it is in.
def _masked_operand(builder, args):
    """Return the ``MaskedType`` operand of a ``Masked``/``NA`` comparison."""
    for arg in args:
        if isinstance(builder.get_numba_type(arg.name), MaskedType):
            return arg
    raise TypeError("expected a MaskedType operand")


# ``is``/``is not`` against NA both reduce to the validity bit: ``m is NA`` ->
# ``not m.valid`` and ``m is not NA`` -> ``m.valid``. Registered for both
# operators (and operand orders) via partials below.
def _lower_masked_na_compare(builder, target, args, kwargs, *, is_null):
    m = builder.load_var(_masked_operand(builder, args))
    st = llvm.StructType(m.type)
    _, valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
    if is_null:
        one = arith.constant(valid.type, 1)
        valid = arith.xori(valid, one)
    builder.store_var(target, valid)


# datetime64 / timedelta64 ``+``/``-`` need numba_cuda_mlir's ``datetime``
# lowering (which scales by unit), not a raw i64 op.
def _needs_datetimelike_delegate(op, ty1, ty2):
    if op not in (operator.add, operator.sub):
        return False
    return isinstance(
        ty1, (types.NPDatetime, types.NPTimedelta)
    ) or isinstance(ty2, (types.NPDatetime, types.NPTimedelta))


def _apply_masked_datetimelike_binary(
    builder, target, target_type, v1, v2, result_valid, op, ty1, ty2,
    ref_var,
):
    """TODO: write docstring."""
    ret_ty = target_type.value_type
    nb_sig = nb_typing.signature(ret_ty, ty1, ty2)
    cg = builder.get_registered_builder(op, nb_sig)
    if cg is None:
        raise NotImplementedError(
            f"No MLIR lowering for masked {op!r} with {ty1}, {ty2}; "
            f"signature {nb_sig}"
        )
    in1 = _make_temp_var(builder, ref_var, "mdt_l", ty1)
    in2 = _make_temp_var(builder, ref_var, "mdt_r", ty2)
    outv = _make_temp_var(builder, ref_var, "mdt_o", ret_ty)
    builder.store_var(in1, convert(v1, builder.get_mlir_type(ty1)))
    builder.store_var(in2, convert(v2, builder.get_mlir_type(ty2)))
    cg(builder, outv, [in1, in2], ())
    result_val = convert(
        builder.load_var(outv), builder.get_mlir_type(ret_ty)
    )
    packed = _pack_masked(
        builder, target_type, result_val, result_valid
    )
    builder.store_var(target, packed)


def _apply_masked_binary_op(
    builder: MLIRLower,
    target: Var,
    target_type: MaskedType,
    v1: mlir_ir.Value,
    v2: mlir_ir.Value,
    result_valid: mlir_ir.Value,
    op: Callable,
    *,
    inner_ty1: types.Type | None = None,
    inner_ty2: types.Type | None = None,
    ref_var: Var | None = None,
) -> None:
    """Apply ``op(v1, v2)`` to two scalar MLIR values, convert the result to
    the target Masked's value type, and pack it with the given validity bit.
    Numeric/boolean only at this layer.
    """
    # datetime/timedelta add/sub: delegate to the unit-aware scalar
    # lowering when we know the operand inner types.
    if (
        inner_ty1 is not None
        and inner_ty2 is not None
        and ref_var is not None
        and _needs_datetimelike_delegate(op, inner_ty1, inner_ty2)
    ):
        _apply_masked_datetimelike_binary(
            builder, target, target_type, v1, v2, result_valid, op,
            inner_ty1, inner_ty2, ref_var,
        )
        return

    target_value_mlir_ty = builder.get_mlir_type(target_type.value_type)
    v1, v2 = coerce_numpy_scalars_for_binary_op(v1, v2)
    # Comparisons compute on the (already coerced) operand type and
    # produce i1; arithmetic/bitwise compute on the target value type.
    is_cmp = op in comparison_ops
    operand_ty = v1.type if is_cmp else target_value_mlir_ty
    v1 = convert(v1, operand_ty)
    v2 = convert(v2, operand_ty)
    result_val = convert(op(v1, v2), target_value_mlir_ty)
    packed = _pack_masked(builder, target_type, result_val, result_valid)
    builder.store_var(target, packed)


def _make_lower_masked_binary(op: Callable) -> Callable:
    """``Masked <op> Masked``: AND the validity bits."""

    def _lower(
        builder: MLIRLower, target: Var, args: list[Var], kwargs: list
    ) -> None:
        target_type = builder.get_numba_type(target.name)
        m1 = builder.load_var(args[0])
        m2 = builder.load_var(args[1])
        st1 = llvm.StructType(m1.type)
        st2 = llvm.StructType(m2.type)
        v1, valid1 = _extract_masked_value_valid(m1, st1.body[0], st1.body[1])
        v2, valid2 = _extract_masked_value_valid(m2, st2.body[0], st2.body[1])
        result_valid = arith.andi(valid1, valid2)
        ty1 = builder.get_numba_type(args[0].name).value_type
        ty2 = builder.get_numba_type(args[1].name).value_type
        _apply_masked_binary_op(
            builder, target, target_type, v1, v2, result_valid, op,
            inner_ty1=ty1, inner_ty2=ty2, ref_var=args[0],
        )

    return _lower


def _scalar_value_from_var(
    builder: MLIRLower,
    s_var: Var,
) -> mlir_ir.Value:
    """Resolve the scalar operand for the Masked-vs-scalar path.

    A ``Literal`` operand carries its value in the type rather than as a
    distinct runtime register, so materialize it directly as a constant;
    genuine runtime scalars are loaded from their variable.
    """
    s_ty = builder.get_numba_type(s_var.name)
    if isinstance(s_ty, types.Literal):
        py_val = s_ty.literal_value
        base_ty = unliteral(s_ty)
        mlir_ty = builder.get_mlir_type(base_ty)
        if isinstance(py_val, (bool, np.bool_)) or (
            hasattr(mlir_ty, "width") and mlir_ty.width == 1
        ):
            py_val = 1 if py_val else 0
        return arith.constant(mlir_ty, py_val)
    return builder.load_var(s_var)


def _make_lower_masked_binary_scalar(
    op: Callable, masked_first: bool
) -> Callable:
    """``Masked <op> scalar`` and ``scalar <op> Masked``: carry the Masked
    operand's validity.
    """

    def _lower(
        builder: MLIRLower, target: Var, args: list[Var], kwargs: list
    ) -> None:
        target_type = builder.get_numba_type(target.name)
        m_var, s_var = (
            (args[0], args[1]) if masked_first else (args[1], args[0])
        )
        m = builder.load_var(m_var)
        st = llvm.StructType(m.type)
        m_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
        s_val = _scalar_value_from_var(builder, s_var)
        m_inner_ty = builder.get_numba_type(m_var.name).value_type
        s_ty = builder.get_numba_type(s_var.name)
        s_inner_ty = (
            unliteral(s_ty) if isinstance(s_ty, types.Literal) else s_ty
        )
        if masked_first:
            _apply_masked_binary_op(
                builder, target, target_type, m_val, s_val, m_valid, op,
                inner_ty1=m_inner_ty, inner_ty2=s_inner_ty, ref_var=m_var,
            )
        else:
            _apply_masked_binary_op(
                builder, target, target_type, s_val, m_val, m_valid, op,
                inner_ty1=s_inner_ty, inner_ty2=m_inner_ty, ref_var=m_var,
            )

    return _lower


def _lower_masked_binary_null(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``Masked <op> NA`` / ``NA <op> Masked``: result is invalid."""
    target_type = builder.get_numba_type(target.name)
    value_mlir_ty = builder.get_mlir_type(target_type.value_type)
    undef_val = llvm.UndefOp(value_mlir_ty)
    valid_zero = arith.constant(
        result=builder.get_mlir_type(types.boolean), value=0
    )
    packed = _pack_masked(builder, target_type, undef_val, valid_zero)
    builder.store_var(target, packed)


def _make_temp_var(builder, base_var, name_suffix, numba_type):
    """TODO: write docstring."""
    scope = getattr(base_var, "scope", None)
    loc = getattr(base_var, "loc", None)
    name = f"$masked_uop_{base_var.name}_{name_suffix}"
    temp = numba_ir.Var(scope=scope, name=name, loc=loc)
    builder.fndesc.typemap[temp.name] = numba_type
    return temp


# Generic unary: delegate the scalar op to the registered numba_cuda_mlir
# scalar lowering (``math.sin`` -> math dialect, ``operator.neg`` -> arith,
# etc.), then re-wrap with the operand's validity.
def _make_lower_masked_unary(op):
    def _lower(builder, target, args, kwargs):
        target_type = builder.get_numba_type(target.name)
        result_inner_ty = target_type.value_type
        operand_inner_ty = builder.get_numba_type(
            args[0].name
        ).value_type

        m = builder.load_var(args[0])
        st = llvm.StructType(m.type)
        m_val, m_valid = _extract_masked_value_valid(
            m, st.body[0], st.body[1]
        )
        m_val = convert(m_val, builder.get_mlir_type(operand_inner_ty))

        sig = result_inner_ty(operand_inner_ty)
        cg = builder.get_registered_builder(op, sig)
        if cg is None:
            raise NotImplementedError(
                "No MLIR lowering for unary "
                f"{getattr(op, '__name__', op)!r} on {operand_inner_ty}; "
                f"signature {sig}"
            )
        # The same operand var can feed multiple unary calls in one
        # expression (e.g. ``sin(x) + lgamma(x)``); suffix the temp var
        # name by op so typemap keys stay unique.
        op_tag = getattr(op, "__name__", "op")
        op_var = _make_temp_var(
            builder, args[0], f"{op_tag}_in", operand_inner_ty
        )
        out_var = _make_temp_var(
            builder, args[0], f"{op_tag}_out", result_inner_ty
        )
        builder.store_var(op_var, m_val)
        cg(builder, out_var, [op_var], [])
        result_val = convert(
            builder.load_var(out_var),
            builder.get_mlir_type(result_inner_ty),
        )
        packed = _pack_masked(
            builder, target_type, result_val, m_valid
        )
        builder.store_var(target, packed)

    return _lower


# ``operator.invert`` (bitwise ~) on Masked integers: there is no scalar
# @lower for invert, so do ``xori(x, -1)``. The all-ones mask is the
# signed constant -1 (two's complement); ``(1<<width)-1`` would overflow
# the signed IntegerAttr range for i64.
def _lower_masked_invert(builder, target, args, kwargs):
    target_type = builder.get_numba_type(target.name)
    result_inner_ty = target_type.value_type
    operand_inner_ty = builder.get_numba_type(args[0].name).value_type
    if not isinstance(operand_inner_ty, types.Integer):
        raise NotImplementedError(
            "operator.invert on Masked is only supported for integer "
            f"payloads, not {operand_inner_ty}"
        )
    m = builder.load_var(args[0])
    st = llvm.StructType(m.type)
    m_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
    m_val = convert(m_val, builder.get_mlir_type(operand_inner_ty))
    mask = arith.constant(result=m_val.type, value=-1)
    result_val = convert(
        arith.xori(m_val, mask), builder.get_mlir_type(result_inner_ty)
    )
    packed = _pack_masked(builder, target_type, result_val, m_valid)
    builder.store_var(target, packed)


# bool(m) / truth: ``m.valid and bool(m.value)``.
def _lower_masked_truth(builder, target, args, kwargs):
    m = builder.load_var(args[0])
    st = llvm.StructType(m.type)
    m_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
    bool_mlir_ty = builder.get_mlir_type(types.boolean)
    payload_as_bool = bool_of(convert(m_val, bool_mlir_ty))
    result = arith.select(m_valid, payload_as_bool, false())
    builder.store_var(target, result)


# int(m) -> Masked(int64); float(m) -> Masked(float64).
def _make_lower_masked_numeric_cast():
    def _lower(builder, target, args, kwargs):
        target_type = builder.get_numba_type(target.name)
        target_value_mlir_ty = builder.get_mlir_type(
            target_type.value_type
        )
        m = builder.load_var(args[0])
        st = llvm.StructType(m.type)
        m_val, m_valid = _extract_masked_value_valid(
            m, st.body[0], st.body[1]
        )
        casted = builder.mlir_convert(m_val, target_value_mlir_ty)
        packed = _pack_masked(
            builder, target_type, casted, m_valid
        )
        builder.store_var(target, packed)

    return _lower


def _const_mlir_for_membership(py_const, mlir_ty):
    if isinstance(py_const, float):
        return float_of(py_const, mlir_ty)
    if isinstance(py_const, bool):
        return int_of(int(py_const), mlir_ty)
    return int_of(py_const, mlir_ty)


# ``value in (c0, c1, ...)`` literal tuple: OR of equality vs each const.
def _lower_masked_literal_tuple_contains(builder, target, args, kwargs):
    tup = builder.load_var(args[0])
    m = builder.load_var(args[1])
    st = llvm.StructType(m.type)
    m_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])

    constant_values = []
    for x in tup:
        cv = try_extract_constant(x)
        if cv is None:
            raise NotImplementedError(
                "Masked membership in a tuple is only implemented for "
                f"constant tuple elements, got {x!r}"
            )
        constant_values.append(cv)

    result = false()
    for const_val in constant_values:
        c = _const_mlir_for_membership(const_val, m_val.type)
        m_v, c_v = coerce_numpy_scalars_for_binary_op(m_val, c)
        result = arith.ori(result, equal(m_v, c_v))

    bool_mlir_ty = builder.get_mlir_type(types.boolean)
    undef_bool = llvm.UndefOp(bool_mlir_ty)
    final_bool = arith.select(m_valid, result, undef_bool)
    target_type = builder.get_numba_type(target.name)
    packed = _pack_masked(builder, target_type, final_bool, m_valid)
    builder.store_var(target, packed)


# ``value in homogeneous_tuple``: reduce equality across the tuple.
def _lower_masked_unittuple_contains(builder, target, args, kwargs):
    tup = builder.load_var(args[0])
    if not isinstance(tup, tuple):
        raise NotImplementedError(
            f"UniTuple contains expects a lowered tuple, got {type(tup)}"
        )
    tup_t = concretize_tuple_to_tensor(tup)

    m = builder.load_var(args[1])
    st = llvm.StructType(m.type)
    m_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
    elem_ty = tup_t.type.element_type
    m_cmp = convert(m_val, elem_ty)

    def body(_op, element, accumulator):
        found = equal(element, m_cmp)
        found = arith.ori(found, accumulator)
        linalg.yield_([found])

    result_type = mlir_ir.RankedTensorType.get((), T.bool())
    init = tensor.splat(result_type, false(), [])
    dims_attr = mlir_ir.DenseI64ArrayAttr.get([0])
    reduce_op = linalg.ReduceOp(
        result=[result_type],
        inputs=[tup_t],
        inits=[init],
        dimensions=dims_attr,
    )
    block = reduce_op.combiner.blocks.append(
        tup_t.type.element_type, result_type.element_type
    )
    with mlir_ir.InsertionPoint(block):
        body(reduce_op, *block.arguments)
    combined = bool_of(tensor.extract(reduce_op.results[0], []))

    bool_mlir_ty = builder.get_mlir_type(types.boolean)
    undef_bool = llvm.UndefOp(bool_mlir_ty)
    final_bool = arith.select(m_valid, combined, undef_bool)
    target_type = builder.get_numba_type(target.name)
    packed = _pack_masked(builder, target_type, final_bool, m_valid)
    builder.store_var(target, packed)


# ``pack_return(masked)`` -> identity.
def _lower_pack_return_masked(builder, target, args, kwargs):
    builder.store_var(target, builder.load_var(args[0]))


# ``pack_return(scalar)`` -> Masked(scalar, True).
def _lower_pack_return_scalar(builder, target, args, kwargs):
    target_type = builder.get_numba_type(target.name)
    value_mlir_ty = builder.get_mlir_type(target_type.value_type)
    scalar_val = convert(builder.load_var(args[0]), value_mlir_ty)
    valid_one = arith.constant(
        result=builder.get_mlir_type(types.boolean), value=1
    )
    packed = _pack_masked(
        builder, target_type, scalar_val, valid_one
    )
    builder.store_var(target, packed)


def _register() -> None:
    """Register the data model and lowerings with ``numba_cuda_mlir``.

    Called once at module import; deferred from module top-level so the
    definitions above remain available for tests/imports without needing
    the registry to be ready.
    """
    lower = lowering_registry.lower

    register_model(MaskedType)(MaskedTypeModel)

    lower(Masked, types.Any, types.boolean)(_lower_masked_constructor)
    # Row apply may pass a literal True/False for ``valid``, typed as
    # ``Literal[bool]``; accept that shape too.
    lower(Masked, types.Any, types.Literal)(_lower_masked_constructor)

    lowering_registry.lower_getattr_generic(MaskedType)(_lower_masked_getattr)

    lower_cast(na_type, MaskedType)(_cast_to_masked)
    for _scalar_cls in (types.Integer, types.Float, types.Boolean):
        lower_cast(_scalar_cls, MaskedType)(_cast_to_masked)
    lower_cast(MaskedType, MaskedType)(_cast_masked_to_masked)

    is_na = partial(_lower_masked_na_compare, is_null=True)
    is_not_na = partial(_lower_masked_na_compare, is_null=False)
    lower(operator.is_, MaskedType, NAType)(is_na)
    lower(operator.is_, NAType, MaskedType)(is_na)
    lower(operator.is_not, MaskedType, NAType)(is_not_na)
    lower(operator.is_not, NAType, MaskedType)(is_not_na)

    for binary_op in arith_ops + bitwise_ops + comparison_ops:
        lower(binary_op, MaskedType, MaskedType)(
            _make_lower_masked_binary(binary_op)
        )
        lower(binary_op, MaskedType, types.Number)(
            _make_lower_masked_binary_scalar(binary_op, True)
        )
        lower(binary_op, types.Number, MaskedType)(
            _make_lower_masked_binary_scalar(binary_op, False)
        )
        lower(binary_op, MaskedType, types.Boolean)(
            _make_lower_masked_binary_scalar(binary_op, True)
        )
        lower(binary_op, types.Boolean, MaskedType)(
            _make_lower_masked_binary_scalar(binary_op, False)
        )
        lower(binary_op, MaskedType, NAType)(_lower_masked_binary_null)
        lower(binary_op, NAType, MaskedType)(_lower_masked_binary_null)

    for unary_op in unary_ops:
        if unary_op is operator.invert:
            continue
        lower(unary_op, MaskedType)(_make_lower_masked_unary(unary_op))
    lower(abs, MaskedType)(_make_lower_masked_unary(abs))
    lower(operator.invert, MaskedType)(_lower_masked_invert)

    lower(operator.truth, MaskedType)(_lower_masked_truth)
    lower(bool, MaskedType)(_lower_masked_truth)

    lower(float, MaskedType)(_make_lower_masked_numeric_cast())
    lower(int, MaskedType)(_make_lower_masked_numeric_cast())

    lower(operator.contains, types.Tuple, MaskedType)(
        _lower_masked_literal_tuple_contains
    )
    lower(operator.contains, types.UniTuple, MaskedType)(
        _lower_masked_unittuple_contains
    )

    lower(pack_return, MaskedType)(_lower_pack_return_masked)
    # Register per concrete scalar shape so the dispatcher matches exactly
    # rather than falling through ``types.Number`` (which would shadow
    # Boolean).
    for scalar_ty in (
        types.Integer,
        types.int8,
        types.int16,
        types.int32,
        types.int64,
        types.uint8,
        types.uint16,
        types.uint32,
        types.uint64,
        types.float32,
        types.float64,
        types.boolean,
    ):
        lower(pack_return, scalar_ty)(_lower_pack_return_scalar)


_register()

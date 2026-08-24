# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import operator
from functools import partial
from typing import TYPE_CHECKING

from numba_cuda_mlir import types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import arith, llvm
from numba_cuda_mlir.extending import lower_cast, lowering_registry
from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.models import PrimitiveModel, register_model

from cudf.core.udf.api import Masked
from cudf.core.udf.mlir_backend.masked_typing import (
    MaskedType,
    NAType,
    na_type,
)

if TYPE_CHECKING:
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


_register()

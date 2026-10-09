# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing and lowering for ``mlir_string`` operations.

Covers ``len`` (UTF-8 character count), the string comparison operators
(``==``/``!=``/``<``/``<=``/``>``/``>=``), and the affix methods
``startswith``/``endswith`` over ``mlir_string``, ``Masked(mlir_string)`` and
string-literal operands; masked operands yield a ``Masked`` result whose validity
is the AND of the operand validity bits. All lowerings are pure MLIR (see
:mod:`cudf.core.udf.mlir_backend.string_lowering_impl`). Registered with
``numba_cuda_mlir`` at import time via :func:`_register`.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING

from numba_cuda_mlir import types
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, llvm
from numba_cuda_mlir.extending import lowering_registry, typing_registry
from numba_cuda_mlir.lowering_utilities import DeferredMethodCall
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    Signature,
)
from numba_cuda_mlir.typing import signature as nb_signature

from cudf.core.udf.mlir_backend import string_lowering_impl as _impl
from cudf.core.udf.mlir_backend.masked_lowering import (
    _extract_masked_value_valid,
    _pack_masked,
)
from cudf.core.udf.mlir_backend.masked_typing import MaskedType
from cudf.core.udf.mlir_backend.strings_typing import (
    MLIRStringType,
    mlir_string,
)

if TYPE_CHECKING:
    from numba_cuda_mlir.mlir_lowering import MLIRLower
    from numba_cuda_mlir.numba_cuda.core.ir import Var

# libcudf size_type; the width of a string length result.
size_type = types.int32

# The single ``Masked(mlir_string)`` instance used as a typing/lowering key.
_masked_string = MaskedType(mlir_string)

# comparison operator -> the arith.cmpi predicate applied to the signed
# ``_impl._lower_compare`` result against zero.
_CMPOP_PREDICATES = {
    operator.eq: arith.CmpIPredicate.eq,
    operator.ne: arith.CmpIPredicate.ne,
    operator.lt: arith.CmpIPredicate.slt,
    operator.le: arith.CmpIPredicate.sle,
    operator.gt: arith.CmpIPredicate.sgt,
    operator.ge: arith.CmpIPredicate.sge,
}

# Operand-type combinations each comparison lowering is registered for
# (plain-plain, plain-literal and the masked variants).
_STR_COMBOS = (
    (MLIRStringType, MLIRStringType),
    (MLIRStringType, types.StringLiteral),
    (types.StringLiteral, MLIRStringType),
    (_masked_string, _masked_string),
    (_masked_string, types.StringLiteral),
    (types.StringLiteral, _masked_string),
    (_masked_string, MLIRStringType),
    (MLIRStringType, _masked_string),
)

# affix method name -> its pure-MLIR impl (one string arg -> boolean).
_BOOL_METHODS = {
    "startswith": _impl._lower_startswith,
    "endswith": _impl._lower_endswith,
}


def _is_plain_string(ty: types.Type) -> bool:
    """Whether ``ty`` is an ``mlir_string`` or a compile-time string literal."""
    return isinstance(ty, (MLIRStringType, types.StringLiteral))


def _is_masked_string(ty: types.Type) -> bool:
    """Whether ``ty`` is a ``Masked(mlir_string)``."""
    return isinstance(ty, MaskedType) and isinstance(
        ty.value_type, MLIRStringType
    )


def _is_string_arg(ty: types.Type) -> bool:
    """Whether ``ty`` is any string operand: plain, literal or masked."""
    return _is_plain_string(ty) or _is_masked_string(ty)


class LenMLIRStringTemplate(AbstractTemplate):
    """``len`` over strings.

    ``len(mlir_string)`` -> ``int32`` (UTF-8 character count), and
    ``len(Masked(mlir_string))`` -> ``Masked(int32)`` (validity carried from the
    operand). ``int32`` matches libcudf's ``size_type`` for string lengths.
    """

    key = len

    def generic(
        self, args: tuple[types.Type, ...], kws: dict
    ) -> Signature | None:
        """Resolve ``len`` over a (masked) ``mlir_string``.

        Parameters
        ----------
        args : tuple of types.Type
            Positional argument types.
        kws : dict
            Keyword argument types (must be empty).

        Returns
        -------
        Signature or None
            ``int32`` for a bare ``mlir_string``, ``Masked(int32)`` for a
            ``Masked(mlir_string)``, else ``None``.
        """
        if len(args) != 1 or kws:
            return None
        arg = args[0]
        if isinstance(arg, MLIRStringType):
            return nb_signature(size_type, mlir_string)
        if _is_masked_string(arg):
            return nb_signature(MaskedType(size_type), arg)
        return None


def _make_cmpop_template(cmpop: object) -> type[AbstractTemplate]:
    """Build the typing template for a string comparison operator.

    Parameters
    ----------
    cmpop : object
        The comparison operator (e.g. ``operator.eq``).

    Returns
    -------
    type
        An ``AbstractTemplate`` subclass resolving the operator to ``boolean``
        (all-plain operands) or ``Masked(boolean)`` (any masked operand).
    """

    class _CmpOpTemplate(AbstractTemplate):
        key = cmpop

        def generic(self, args, kws):
            """Resolve ``a <cmp> b`` for string operands."""
            if len(args) != 2 or kws:
                return None
            a, b = args
            if not (_is_string_arg(a) and _is_string_arg(b)):
                return None
            if _is_masked_string(a) or _is_masked_string(b):
                return nb_signature(MaskedType(types.boolean), a, b)
            if isinstance(a, MLIRStringType) or isinstance(b, MLIRStringType):
                return nb_signature(types.boolean, a, b)
            return None

    return _CmpOpTemplate


def _make_method_attr(
    attrname: str, retty: types.Type, masked: bool
) -> object:
    """Build a ``resolve_<method>`` for a one-string-arg method.

    Parameters
    ----------
    attrname : str
        Method name (e.g. ``"startswith"``).
    retty : types.Type
        The unmasked return type.
    masked : bool
        Whether the receiver is ``Masked(mlir_string)`` (wraps the result).

    Returns
    -------
    callable
        A ``resolve_`` function for the attribute template.
    """
    recvr = _masked_string if masked else mlir_string

    class _MethodTemplate(AbstractTemplate):
        key = f"MLIRString.{attrname}.{'m' if masked else 's'}"

        def generic(self, args, kws):
            """Resolve ``s.<method>(other)`` for a string ``other``.

            The result is ``Masked`` when *either* the receiver or the argument
            is masked, so an invalid argument never yields a valid plain result.
            The argument's string-ness is not gated here: a ``StringLiteral``
            argument isn't recognized by ``isinstance`` inside a ``BoundFunction``
            template, and the lowering handles the operand kinds.
            """
            if len(args) == 1 and not kws:
                result_ty = (
                    MaskedType(retty)
                    if masked or _is_masked_string(args[0])
                    else retty
                )
                return nb_signature(result_ty, args[0], recvr=self.this)
            return None

    def resolve(self, mod):
        return types.BoundFunction(_MethodTemplate, recvr)

    return resolve


@typing_registry.register_attr
class MLIRStringAttrs(AttributeTemplate):
    """Attribute typing for ``mlir_string`` methods."""

    key = mlir_string


@typing_registry.register_attr
class MaskedMLIRStringAttrs(AttributeTemplate):
    """Attribute typing for ``Masked(mlir_string)``.

    An instance-keyed attribute template shadows the class-keyed
    :class:`MaskedTypeAttrs`, and numba resolves ``resolve_*`` only from the
    template's own class, so ``.value``/``.valid`` are redefined here alongside
    the string methods.
    """

    key = _masked_string

    def resolve_value(self, typ: MaskedType) -> types.Type:
        """Resolve ``Masked(mlir_string).value``.

        Parameters
        ----------
        typ : MaskedType
            The ``Masked(mlir_string)`` receiver type.

        Returns
        -------
        types.Type
            The wrapped value type (``mlir_string``).
        """
        return typ.value_type

    def resolve_valid(self, typ: MaskedType) -> types.Type:
        """Resolve ``Masked(mlir_string).valid``.

        Parameters
        ----------
        typ : MaskedType
            The ``Masked(mlir_string)`` receiver type.

        Returns
        -------
        types.Type
            ``boolean``.
        """
        return types.boolean


for _name in _BOOL_METHODS:
    setattr(
        MLIRStringAttrs,
        f"resolve_{_name}",
        _make_method_attr(_name, types.boolean, masked=False),
    )
    setattr(
        MaskedMLIRStringAttrs,
        f"resolve_{_name}",
        _make_method_attr(_name, types.boolean, masked=True),
    )


def _lower_len(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``len(mlir_string)``: count UTF-8 characters, returned as ``int32``."""
    view = _impl._mlir_string_to_view(builder.load_var(args[0]))
    builder.store_var(target, _impl._lower_len(view))


def _lower_masked_len(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``len(Masked(mlir_string))``: character count packed with the operand's
    validity bit (``Masked(int32)``).

    The payload is scanned unconditionally; this is safe because null rows carry
    ``nbytes == 0`` (the marshaller leaves a null ``data`` pointer with zero
    length), so the count loop never dereferences ``data`` for a null row. The
    resulting count is discarded anyway when ``m_valid`` is false.
    """
    m = builder.load_var(args[0])
    st = llvm.StructType(m.type)
    ms_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
    count = _impl._lower_len(_impl._mlir_string_to_view(ms_val))
    target_type = builder.get_numba_type(target.name)
    packed = _pack_masked(builder, target_type, count, m_valid)
    builder.store_var(target, packed)


def _literal_to_view(builder: MLIRLower, literal_var: Var) -> ir.Value:
    """Materialize a ``StringLiteral`` as a view ``{ptr, i32 nbytes, i32 len}``.

    ``builder.load_var`` yields numba-cuda-mlir's materialized unicode struct
    (``{ptr data, i64 length, ...}``); field 0 is the byte-buffer pointer. Only
    ASCII literals are currently supported: for ASCII that buffer is
    byte-identical to UTF-8, so it compares correctly against ``mlir_string``
    (which stores UTF-8). Non-ASCII string literals are presently rejected
    upstream by numba-cuda-mlir's string-constant materialization, so no explicit
    guard is added here.
    """
    literal_value = builder.get_numba_type(literal_var.name).literal_value
    data_struct = builder.load_var(literal_var)
    ptr_ty = llvm.PointerType.get()
    i32 = ir.IntegerType.get_signless(32)
    view_ty = llvm.StructType.get_literal([ptr_ty, i32, i32])
    val = llvm.UndefOp(view_ty)
    val = llvm.insertvalue(
        container=val,
        value=llvm.extractvalue(ptr_ty, data_struct, [0]),
        position=ir.DenseI64ArrayAttr.get([0]),
    )
    val = llvm.insertvalue(
        container=val,
        value=arith.constant(i32, len(literal_value.encode("utf-8"))),
        position=ir.DenseI64ArrayAttr.get([1]),
    )
    return llvm.insertvalue(
        container=val,
        value=arith.constant(i32, len(literal_value)),
        position=ir.DenseI64ArrayAttr.get([2]),
    )


def _view_and_valid(
    builder: MLIRLower, var: Var, true_val: ir.Value
) -> tuple[ir.Value, ir.Value]:
    """``(view, valid)`` for a string operand var (plain, literal or masked).

    Non-masked operands report validity ``true_val``; a ``Masked(mlir_string)``
    reports its stored validity bit.
    """
    ty = builder.get_numba_type(var.name)
    if isinstance(ty, types.StringLiteral):
        return _literal_to_view(builder, var), true_val
    val = builder.load_var(var)
    if _is_masked_string(ty):
        st = llvm.StructType(val.type)
        ms_val, valid = _extract_masked_value_valid(
            val, st.body[0], st.body[1]
        )
        return _impl._mlir_string_to_view(ms_val), valid
    return _impl._mlir_string_to_view(val), true_val


def _store_maybe_masked(
    builder: MLIRLower, target: Var, result: ir.Value, valid: ir.Value
) -> None:
    """Store ``result`` at ``target``, packing a ``Masked`` value if required."""
    target_type = builder.get_numba_type(target.name)
    if isinstance(target_type, MaskedType):
        builder.store_var(
            target, _pack_masked(builder, target_type, result, valid)
        )
    else:
        builder.store_var(target, result)


def _make_lower_cmpop(predicate: arith.CmpIPredicate) -> object:
    """Build a comparison lowering for the given ``cmpi`` predicate.

    The lowering handles plain, literal and masked operands uniformly and packs
    a ``Masked`` result when the target is masked.

    Parameters
    ----------
    predicate : arith.CmpIPredicate
        Predicate applied to the signed compare result against zero.

    Returns
    -------
    callable
        A lowering ``(builder, target, args, kwargs) -> None``.
    """

    def _lower(builder, target, args, kwargs):
        valid_ty = builder.get_mlir_type(types.boolean)
        true_val = arith.constant(valid_ty, 1)
        lhs, vl = _view_and_valid(builder, args[0], true_val)
        rhs, vr = _view_and_valid(builder, args[1], true_val)
        zero = arith.constant(ir.IntegerType.get_signless(32), 0)
        result = arith.cmpi(predicate, _impl._lower_compare(lhs, rhs), zero)
        _store_maybe_masked(builder, target, result, arith.andi(vl, vr))

    return _lower


def _make_lower_binary(impl_fn: object) -> object:
    """Build the ``lower_getattr`` for a one-string-arg method (startswith/endswith).

    The returned getattr defers to a method lowering that handles plain, literal
    and masked operands uniformly and packs a ``Masked`` result when required.
    """

    def _impl_lower(builder, target, args, kwargs):
        valid_ty = builder.get_mlir_type(types.boolean)
        true_val = arith.constant(valid_ty, 1)
        recv, vr = _view_and_valid(builder, args[0], true_val)
        other, vo = _view_and_valid(builder, args[1], true_val)
        result = impl_fn(recv, other)
        _store_maybe_masked(builder, target, result, arith.andi(vr, vo))

    def _getattr(context, builder, target, value, attr=None):
        builder.store_var(target, DeferredMethodCall(value, _impl_lower))

    return _getattr


def _register() -> None:
    """Register ``len`` + comparison + affix typing/lowering with numba_cuda_mlir."""
    typing_registry.register_global(len)(LenMLIRStringTemplate)
    lowering_registry.lower(len, mlir_string)(_lower_len)
    lowering_registry.lower(len, MaskedType)(_lower_masked_len)

    for cmp_op, predicate in _CMPOP_PREDICATES.items():
        typing_registry.register_global(cmp_op)(_make_cmpop_template(cmp_op))
        impl = _make_lower_cmpop(predicate)
        for lhs_ty, rhs_ty in _STR_COMBOS:
            lowering_registry.lower(cmp_op, lhs_ty, rhs_ty)(impl)

    # startswith / endswith methods on mlir_string and Masked(mlir_string)
    for name, impl_fn in _BOOL_METHODS.items():
        getattr_fn = _make_lower_binary(impl_fn)
        lowering_registry.lower_getattr(mlir_string, name)(getattr_fn)
        lowering_registry.lower_getattr(_masked_string, name)(getattr_fn)


_register()

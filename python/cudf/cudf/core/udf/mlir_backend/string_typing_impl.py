# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typing for mlir_string: method signatures and operator overloads.

numba_cuda_mlir owns the MLIRStringType and all its methods.  cudf (or any other
consumer) can register *additional* typing for its own wrapper types
(MaskedType, etc.) that delegate to these.

The data model for MLIRStringType lives in numba_cuda_mlir.models.
"""

from __future__ import annotations

import operator

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.typing import signature as nb_signature
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    Registry,
)

from cudf.core.udf.mlir_backend.string_types import MLIRStringType

registry = Registry()

mlir_string = MLIRStringType()
size_type = types.int32


# ---------------------------------------------------------------------------
# Attribute typing: mlir_string.len, .find, .upper, …
# ---------------------------------------------------------------------------

_ID_UNARY_FUNCS = [
    "isalpha",
    "isalnum",
    "isdecimal",
    "isdigit",
    "isupper",
    "islower",
    "isspace",
    "isnumeric",
    "istitle",
]


def _make_is_attr(attrname):
    class CusimtStringIsTemplate(AbstractTemplate):
        key = f"CusimtString.{attrname}"

        def generic(self, args, kws):
            return nb_signature(types.boolean, recvr=self.this)

    def resolve_attr(self, mod):
        return types.BoundFunction(CusimtStringIsTemplate, mlir_string)

    return resolve_attr


def _make_transform_attr(attrname):
    class TransformTemplate(AbstractTemplate):
        key = f"CusimtString.{attrname}"

        def generic(self, args, kws):
            return nb_signature(mlir_string, recvr=self.this)

    def resolve_attr(self, mod):
        return types.BoundFunction(TransformTemplate, mlir_string)

    return resolve_attr


class ReplaceTemplate(AbstractTemplate):
    key = "CusimtString.replace"

    def generic(self, args, kws):
        if len(args) == 2 and not kws:
            return nb_signature(mlir_string, args[0], args[1], recvr=self.this)


def _resolve_replace(self, mod):
    return types.BoundFunction(ReplaceTemplate, mlir_string)


def _make_strip_attr(attrname):
    class StripTemplate(AbstractTemplate):
        key = f"CusimtString.{attrname}"

        def generic(self, args, kws):
            if len(args) == 1 and not kws:
                return nb_signature(mlir_string, args[0], recvr=self.this)

    def resolve_attr(self, mod):
        return types.BoundFunction(StripTemplate, mlir_string)

    return resolve_attr


def _make_binary_attr(attrname, retty):
    class BinaryAttrTemplate(AbstractTemplate):
        key = f"CusimtString.{attrname}"

        def generic(self, args, kws):
            if (
                len(args) == 1
                and not kws
                and (
                    isinstance(args[0], MLIRStringType)
                    or isinstance(args[0], types.StringLiteral)
                )
            ):
                return nb_signature(retty, args[0], recvr=self.this)

    def resolve_attr(self, mod):
        return types.BoundFunction(BinaryAttrTemplate, mlir_string)

    return resolve_attr


@registry.register_attr
class CusimtStringAttrs(AttributeTemplate):
    key = mlir_string

    def resolve_len(self, mod):
        return size_type

    def resolve_nbytes(self, mod):
        return types.int64

    def resolve_data_ptr(self, mod):
        return types.int64

    resolve_replace = _resolve_replace


for _attr in _ID_UNARY_FUNCS:
    setattr(CusimtStringAttrs, f"resolve_{_attr}", _make_is_attr(_attr))

for _attr in ("upper", "lower"):
    setattr(CusimtStringAttrs, f"resolve_{_attr}", _make_transform_attr(_attr))

for _attr in ("strip", "lstrip", "rstrip"):
    setattr(CusimtStringAttrs, f"resolve_{_attr}", _make_strip_attr(_attr))

for _attr in ("find", "rfind", "count"):
    setattr(
        CusimtStringAttrs,
        f"resolve_{_attr}",
        _make_binary_attr(_attr, size_type),
    )
for _attr in ("startswith", "endswith"):
    setattr(
        CusimtStringAttrs,
        f"resolve_{_attr}",
        _make_binary_attr(_attr, types.boolean),
    )


# ---------------------------------------------------------------------------
# Global operator typing: len(mlir_string), cmpops, add, contains
# ---------------------------------------------------------------------------


@registry.register_global(len)
class LenMLIRStringTemplate(AbstractTemplate):
    key = len

    def generic(self, args, kws):
        if len(args) == 1 and not kws and isinstance(args[0], MLIRStringType):
            return nb_signature(size_type, mlir_string)
        return None


_CMPOPS = (
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
)


def _make_cmpop_template(cmpop):
    class CmpOpTemplate(AbstractTemplate):
        key = cmpop

        def generic(self, args, kws):
            if (
                len(args) == 2
                and not kws
                and isinstance(args[0], MLIRStringType)
                and isinstance(args[1], MLIRStringType)
            ):
                return nb_signature(types.boolean, mlir_string, mlir_string)

    return CmpOpTemplate


for _op in _CMPOPS:
    registry.register_global(_op)(_make_cmpop_template(_op))


@registry.register_global(operator.add)
class AddMLIRStringTemplate(AbstractTemplate):
    key = operator.add

    def generic(self, args, kws):
        if (
            len(args) == 2
            and not kws
            and isinstance(args[0], (MLIRStringType, types.StringLiteral))
            and isinstance(args[1], (MLIRStringType, types.StringLiteral))
        ):
            return nb_signature(mlir_string, mlir_string, mlir_string)


# ---------------------------------------------------------------------------
# getitem: mlir_string[int] -> mlir_string (single char)
#          mlir_string[a:b] -> mlir_string (substring)
# ---------------------------------------------------------------------------


@registry.register_global(operator.getitem)
class GetitemMLIRStringIntTemplate(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        if (
            len(args) == 2
            and not kws
            and isinstance(args[0], MLIRStringType)
            and isinstance(args[1], types.Integer)
        ):
            return nb_signature(mlir_string, mlir_string, args[1])


@registry.register_global(operator.getitem)
class GetitemMLIRStringSliceTemplate(AbstractTemplate):
    key = operator.getitem

    def generic(self, args, kws):
        if (
            len(args) == 2
            and not kws
            and isinstance(args[0], MLIRStringType)
            and isinstance(args[1], types.SliceType)
        ):
            return nb_signature(mlir_string, mlir_string, args[1])


@registry.register_global(operator.contains)
class ContainsMLIRStringTemplate(AbstractTemplate):
    key = operator.contains

    def generic(self, args, kws):
        if len(args) != 2 or kws:
            return None
        a, b = args
        if isinstance(a, (MLIRStringType, types.StringLiteral)) and isinstance(
            b, (MLIRStringType, types.StringLiteral)
        ):
            return nb_signature(types.boolean, a, b)


# ---------------------------------------------------------------------------
# mlir_string_from_ptr(data_ptr: int64, nbytes: int64) -> mlir_string
# ---------------------------------------------------------------------------


def _mlir_string_from_ptr_stub(data_ptr, nbytes):
    """Sentinel: construct a non-owning mlir_string from a raw pointer and length."""
    raise NotImplementedError("device-only")


@registry.register_global(_mlir_string_from_ptr_stub)
class MLIRStringFromPtrTemplate(AbstractTemplate):
    key = _mlir_string_from_ptr_stub

    def generic(self, args, kws):
        if len(args) == 2 and not kws:
            if isinstance(args[0], types.Integer) and isinstance(
                args[1], types.Integer
            ):
                return nb_signature(mlir_string, types.int64, types.int64)


mlir_string_from_ptr = _mlir_string_from_ptr_stub

# NRT_decref typing is registered by the consumer (e.g. cudf) since the
# sentinel function object must be shared between typing and lowering.

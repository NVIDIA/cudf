# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import operator

from numba_cuda_mlir import models, types
from numba_cuda_mlir.extending import typing_registry
from numba_cuda_mlir.models import register_model
from numba_cuda_mlir.numba_cuda import types as nb_types
from numba_cuda_mlir.numba_cuda.extending import typeof_impl
from numba_cuda_mlir.numba_cuda.types.misc import unliteral
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    ConcreteTemplate,
)
from numba_cuda_mlir.typing import signature as nb_signature

from cudf.core.missing import NA
from cudf.core.udf.api import Masked

_SUPPORTED_MASKED_VALUE_TYPE_CLASSES = (
    types.Number,
    types.Boolean,
)


_supported_value_type_instances = (
    nb_types.integer_domain | nb_types.real_domain | {nb_types.boolean}
)


class MaskedType(types.Type):
    """Logical struct type used for propagation of nulls. Semantically carries
    the column value and corresponding validity bit from the columns bitmask.
    Operations over this type such as arithmetic are implemented to be sensitive
    to the nullity of the output value.

    Instances are parameterized by value type mapping to the type of the source
    column.
    """

    def __init__(self, value: types.Type) -> None:
        if isinstance(value, types.Literal):
            value = unliteral(value)
        if isinstance(value, _SUPPORTED_MASKED_VALUE_TYPE_CLASSES):
            self.value_type = value
        else:
            self.value_type = types.Poison(value)
        super().__init__(name=f"Masked({self.value_type})")

    def __hash__(self) -> int:
        # Two MaskedType instances compare equal (and hash equal) iff their
        # parameter ``value_type`` matches, so numba can cache them by repr.
        return hash(repr(self))

    def unify(self, context, other):
        """Pick a common type when branches return different shapes.

        ``return x if cond else cudf.NA`` unifies ``MaskedType`` and
        ``NAType``; ``return x if cond else 5`` unifies ``MaskedType``
        and a scalar; two branches returning different ``Masked``
        widths unify their inner value types. Returning ``None``
        signals "no unifier" and lets numba fall through.
        See https://numba.pydata.org/numba-doc/dev/user/troubleshoot.html#my-code-has-a-type-unification-problem
        """
        if isinstance(other, NAType):
            return self
        other_value_type = (
            other.value_type if isinstance(other, MaskedType) else other
        )
        unified = context.unify_pairs(self.value_type, other_value_type)
        return MaskedType(unified) if unified is not None else None


class NAType(types.Type):
    """Type for ``cudf.NA`` -- the missing-value sentinel that can flow
    through a UDF branch and unify into a ``MaskedType``.
    """

    def __init__(self):
        super().__init__(name="NA")

    def unify(self, context, other):
        # See https://numba.pydata.org/numba-doc/dev/user/troubleshoot.html#my-code-has-a-type-unification-problem
        # NA + Masked is delegated to MaskedType.unify (see above) so we
        # only need to handle NA + (NA | scalar) here.
        if isinstance(other, MaskedType):
            return None
        if isinstance(other, NAType):
            return self
        return MaskedType(other)


na_type = NAType()


@typeof_impl.register(type(NA))
def _typeof_na(val, c):
    return na_type


# NAType has no payload; OpaqueModel is the right data model.
register_model(NAType)(models.OpaqueModel)


# ``Masked(value, valid)`` constructor: produces a ``Masked(value_ty)``.
class MaskedConstructor(ConcreteTemplate):
    key = Masked
    cases = [
        nb_signature(MaskedType(t), t, types.boolean)
        for t in _supported_value_type_instances
    ]


# ``m.value`` -> inner value type; ``m.valid`` -> boolean. ``AttributeTemplate``
# dispatches ``resolve_<attr>`` methods automatically.
class MaskedTypeAttrs(AttributeTemplate):
    key = MaskedType

    def resolve_value(self, typ: MaskedType) -> types.Type:
        return typ.value_type

    def resolve_valid(self, typ: MaskedType) -> types.Type:
        return types.boolean


# ``is``/``is not`` between a ``MaskedType`` and ``NA`` (either operand order)
# both type to boolean; the is/is-not distinction is handled in lowering, so a
# single template serves both operators.
class MaskedNAComparison(AbstractTemplate):
    def generic(self, args, kws):
        if len(args) != 2 or kws:
            return None
        lhs, rhs = args
        if {type(lhs), type(rhs)} == {MaskedType, NAType}:
            return nb_signature(types.boolean, lhs, rhs)
        return None


def _register() -> None:
    """Register typing for ``Masked`` and ``MaskedType`` attributes with
    ``numba_cuda_mlir``. Called once at module import.
    """
    typing_registry.register_global(Masked, types.Function(MaskedConstructor))
    typing_registry.register_attr(MaskedTypeAttrs)
    typing_registry.register_global(operator.is_)(MaskedNAComparison)
    typing_registry.register_global(operator.is_not)(MaskedNAComparison)


_register()

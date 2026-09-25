# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing and lowering for ``mlir_string`` operations.

Currently only ``len`` (UTF-8 character count). Registered with
``numba_cuda_mlir`` at import time via :func:`_register`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba_cuda_mlir import types
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.extending import lowering_registry, typing_registry
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
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


class LenMLIRStringTemplate(AbstractTemplate):
    """``len`` over strings.

    ``len(mlir_string)`` -> ``int64`` (UTF-8 character count), and
    ``len(Masked(mlir_string))`` -> ``Masked(int64)`` (validity carried from the
    operand).
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
            ``int64`` for a bare ``mlir_string``, ``Masked(int64)`` for a
            ``Masked(mlir_string)``, else ``None``.
        """
        if len(args) != 1 or kws:
            return None
        arg = args[0]
        if isinstance(arg, MLIRStringType):
            return nb_signature(types.int64, mlir_string)
        if isinstance(arg, MaskedType) and isinstance(
            arg.value_type, MLIRStringType
        ):
            return nb_signature(MaskedType(types.int64), arg)
        return None


def _lower_len(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``len(mlir_string)``: count UTF-8 characters, returned as ``int64``."""
    ms_val = builder.load_var(args[0])
    builder.store_var(target, _impl._lower_len(ms_val))


def _lower_masked_len(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``len(Masked(mlir_string))``: character count packed with the operand's
    validity bit (``Masked(int64)``).

    The payload is scanned unconditionally; this is safe because null rows carry
    ``nbytes == 0`` (the marshaller leaves a null ``data`` pointer with zero
    length), so the count loop never dereferences ``data`` for a null row. The
    resulting count is discarded anyway when ``m_valid`` is false.
    """
    m = builder.load_var(args[0])
    st = llvm.StructType(m.type)
    ms_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
    count_i64 = _impl._lower_len(ms_val)
    target_type = builder.get_numba_type(target.name)
    packed = _pack_masked(builder, target_type, count_i64, m_valid)
    builder.store_var(target, packed)


def _register() -> None:
    """Register ``len`` typing and lowering with ``numba_cuda_mlir``."""
    typing_registry.register_global(len)(LenMLIRStringTemplate)
    lowering_registry.lower(len, mlir_string)(_lower_len)
    lowering_registry.lower(len, MaskedType)(_lower_masked_len)


_register()

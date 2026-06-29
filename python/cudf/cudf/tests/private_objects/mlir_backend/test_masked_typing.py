# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types

from cudf.core.udf.mlir_backend.masked_typing import (
    MaskedType,
)


def test_masked_type_repr():
    assert repr(MaskedType(types.int64)) == "Masked(int64)"
    assert repr(MaskedType(types.float64)) == "Masked(float64)"
    assert repr(MaskedType(types.boolean)) == "Masked(bool)"


def test_masked_type_equality_by_value_type():
    """Two MaskedType instances are equal iff their value_type is equal."""
    a = MaskedType(types.int64)
    b = MaskedType(types.int64)
    c = MaskedType(types.float64)
    # Numba ``Type`` equality is structural: same name -> equal.
    assert a == b
    assert a != c


def test_masked_type_hash_keys_by_value_type():
    """Numba caches typed signatures by hash; matching params must hash equal."""
    assert hash(MaskedType(types.int64)) == hash(MaskedType(types.int64))
    assert hash(MaskedType(types.int64)) != hash(MaskedType(types.float64))


def test_masked_type_unliterals_value():
    """A literal value type is unliteralled before being stored."""
    lit = types.IntegerLiteral(7)
    masked = MaskedType(lit)
    # The stored value_type is the underlying int64 (or whatever the
    # literal unwraps to), not the literal itself.
    assert not isinstance(masked.value_type, types.Literal)


def test_masked_type_unsupported_value_becomes_poison():
    """Unsupported value types are wrapped in Poison (not raised here).

    The typing pass may try to construct ``MaskedType`` for unsupported
    column dtypes; we want that to succeed at construction so a
    descriptive error can surface later when the user actually performs
    an op on the value. The poison sentinel is the carrier.
    """
    masked = MaskedType(types.unicode_type)
    assert isinstance(masked.value_type, types.Poison)


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_masked_datetime_timedelta_not_poisoned(unit):
    """TODO: write docstring."""
    dt = MaskedType(types.NPDatetime(unit))
    td = MaskedType(types.NPTimedelta(unit))
    assert dt.value_type == types.NPDatetime(unit)
    assert td.value_type == types.NPTimedelta(unit)
    assert not isinstance(dt.value_type, types.Poison)
    assert not isinstance(td.value_type, types.Poison)


def test_na_type_singleton_repr():
    """``NAType`` repr is ``"NA"``."""
    from cudf.core.udf.mlir_backend.masked_typing import NAType, na_type

    assert isinstance(na_type, NAType)
    assert repr(na_type) == "NA"


def test_typeof_cudf_na_returns_na_type():
    """``typeof(cudf.NA)`` resolves to the ``NAType`` singleton."""
    from numba_cuda_mlir.numba_cuda.typing.typeof import typeof

    from cudf.core.missing import NA
    from cudf.core.udf.mlir_backend.masked_typing import na_type

    assert typeof(NA) is na_type


def test_masked_type_unify_with_na_returns_self():
    """``Masked(t).unify(NA)`` -> ``Masked(t)``: NA carries no width."""
    from cudf.core.udf.mlir_backend.masked_typing import (
        MaskedType,
        na_type,
    )

    m = MaskedType(types.int64)
    # ``context`` is unused on the NA branch; passing None is fine.
    assert m.unify(None, na_type) == m


def test_na_type_unify_with_masked_defers_to_masked():
    """``NA.unify(Masked(...))`` -> None: tells numba to use
    ``MaskedType.unify`` instead, which gives the right answer.
    """
    from cudf.core.udf.mlir_backend.masked_typing import (
        MaskedType,
        na_type,
    )

    m = MaskedType(types.int64)
    assert na_type.unify(None, m) is None


def test_na_type_unify_with_scalar_promotes_to_masked():
    """``NA.unify(scalar)`` -> ``Masked(scalar)``: scalar branches that
    join an NA branch get wrapped.
    """
    from cudf.core.udf.mlir_backend.masked_typing import (
        MaskedType,
        na_type,
    )

    result = na_type.unify(None, types.int64)
    assert isinstance(result, MaskedType)
    assert result.value_type == types.int64

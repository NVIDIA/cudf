# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Name generation utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cudf_polars.dsl.expr import Col, NamedExpr

if TYPE_CHECKING:
    from collections.abc import Generator, Iterable

    from cudf_polars.typing import Schema


__all__ = ["names_to_indices", "unique_names"]


def unique_names(names: Iterable[str]) -> Generator[str, None, None]:
    """
    Generate unique names relative to some known names.

    Parameters
    ----------
    names
        Names we should be unique with respect to.

    Yields
    ------
    Unique names (just using sequence numbers)
    """
    prefix = "_" * max(map(len, names))
    i = 0
    while True:
        yield f"{prefix}{i}"
        i += 1


def _concrete_prefix(names: Iterable[str | NamedExpr]) -> tuple[str, ...]:
    # Exclude NamedExprs that are not concrete Col references.
    # We don't throw out the entire NamedExpr tuple if a prefix
    # of the tuple is concrete.
    prefix: list[str] = []
    for name in names:
        if isinstance(name, str):
            prefix.append(name)
        elif isinstance(name.value, Col):
            prefix.append(name.value.name)
        else:
            break
    return tuple(prefix)


def names_to_indices(
    names: Iterable[str | NamedExpr],
    schema: Schema,
    *,
    concrete_prefix: bool = False,
) -> tuple[int, ...]:
    """
    Return column indices for the given names in schema order.

    Accepts either column names (str) or NamedExpr, so it can be used with
    e.g. ir.left_on, ir.right_on as well as plain name tuples.

    Parameters
    ----------
    names
        The names to get indices for.
    schema
        The schema to get indices from.
    concrete_prefix
        If True, use only the prefix of names corresponding
        to concrete column references. If False (default),
        use all names.

    Returns
    -------
    The column indices for each name in schema order.
    """
    keys = {name: i for i, name in enumerate(schema.keys())}
    if concrete_prefix:
        names = _concrete_prefix(names)
    return tuple(
        keys[n] for n in (n.name if isinstance(n, NamedExpr) else n for n in names)
    )


def indices_to_names(indices: Iterable[int], schema: Schema) -> tuple[str, ...]:
    """
    Return column names for the given column indices in schema order.

    Parameters
    ----------
    indices
        The indices to get names for.
    schema
        The schema to get names from.

    Returns
    -------
    The column names for each index in schema order.
    """
    keys = list(schema.keys())
    return tuple(keys[i] for i in indices)

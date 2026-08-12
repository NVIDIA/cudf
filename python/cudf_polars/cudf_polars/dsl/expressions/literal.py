# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Literal DSL nodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from collections.abc import Hashable

    from cudf_polars.containers import DataFrame

__all__ = ["Literal", "LiteralColumn"]


class Literal(Expr):
    __slots__ = ("value",)
    _non_child = ("dtype", "value")
    value: Any  # Python scalar

    def __init__(self, dtype: DataType, value: Any) -> None:
        if value is None and dtype.id() == plc.TypeId.EMPTY:
            # TypeId.EMPTY not supported by libcudf
            # cuDF Python also maps EMPTY to INT8
            dtype = DataType(pl.datatypes.Int8())
        self.dtype = dtype
        self.value = value
        self.children = ()
        self.is_pointwise = True

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return Column(
            plc.Column.from_scalar(
                plc.Scalar.from_py(self.value, self.dtype.plc_type, stream=df.stream),
                1,
                stream=df.stream,
            ),
            dtype=self.dtype,
        )

    @property
    def agg_request(self) -> NoReturn:  # noqa: D102
        raise NotImplementedError(
            "Not expecting to require agg request of literal"
        )  # pragma: no cover

    def get_hashable(self) -> Hashable:
        """Get the hash of the literal."""
        return (type(self), self.dtype.plc_type, id(self.value))

    def astype(self, dtype: DataType) -> Literal:
        """Cast self to dtype."""
        if dtype.id() == plc.TypeId.STRUCT:
            raise NotImplementedError("Struct literals are not supported")
        if self.value is None:
            return Literal(dtype, self.value)
        else:
            # Use polars to cast instead of pylibcudf
            # since there are just Python scalars
            casted = pl.Series(values=[self.value], dtype=self.dtype.polars_type).cast(
                dtype.polars_type
            )[0]
            return Literal(dtype, casted)


class LiteralColumn(Expr):
    __slots__ = ("is_scalar", "value")
    _non_child = ("dtype", "value", "is_scalar")
    value: pl.Series
    is_scalar: bool

    def __init__(
        self, dtype: DataType, value: pl.Series, *, is_scalar: bool = False
    ) -> None:
        # is_scalar records whether this originated from a scalar literal
        # rather than a column. In a groupby aggregation a column literal is
        # imploded into a list per group, while a scalar is broadcast.
        self.dtype = dtype
        self.value = value
        self.is_scalar = is_scalar
        self.children = ()
        self.is_pointwise = True

    def get_hashable(self) -> Hashable:
        """Compute a hash of the column."""
        # This is stricter than necessary, but we only need this hash
        # for identity in groupby replacements so it's OK. And this
        # way we avoid doing potentially expensive compute.
        return (type(self), self.dtype.plc_type, id(self.value))

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        return Column(
            plc.Column.from_arrow(self.value, stream=df.stream), dtype=self.dtype
        )

    @property
    def agg_request(self) -> NoReturn:  # noqa: D102
        raise NotImplementedError(
            "Not expecting to require agg request of literal"
        )  # pragma: no cover

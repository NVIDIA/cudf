# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Base and common classes for expression DSL nodes."""

from __future__ import annotations

import enum
from enum import IntEnum
from typing import TYPE_CHECKING, Any, ClassVar

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.nodebase import Node
from cudf_polars.dsl.traversal import traversal

if TYPE_CHECKING:
    from typing import Self

    from cudf_polars.containers import Column, DataFrame, DataType

__all__ = ["Col", "ColRef", "ExecutionContext", "Expr", "NamedExpr"]


class ExecutionContext(IntEnum):
    FRAME = enum.auto()
    GROUPBY = enum.auto()
    ROLLING = enum.auto()
    # Follows GROUPBY semantics but useful
    # to differentiate from GROUPBY so we can
    # implement agg/per-row ops independently
    WINDOW = enum.auto()


class Expr(Node["Expr"]):
    """An abstract expression object."""

    __slots__ = ("dtype", "is_pointwise")
    dtype: DataType
    """Data type of the expression."""
    is_pointwise: bool
    """Whether this expression acts pointwise on its inputs."""
    # This annotation is needed because of https://github.com/python/mypy/issues/17981
    _non_child: ClassVar[tuple[str, ...]] = ("dtype",)
    """Names of non-child data (not Exprs) for reconstruction."""

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """
        Evaluate this expression given a dataframe for context.

        Parameters
        ----------
        df
            DataFrame that will provide columns.
        context
            What context are we performing this evaluation in?

        Notes
        -----
        Do not call this function directly, but rather :meth:`evaluate`.

        Returns
        -------
        Column representing the evaluation of the expression.

        Raises
        ------
        NotImplementedError
            If we couldn't evaluate the expression. Ideally all these
            are returned during translation to the IR, but for now we
            are not perfect.
        """
        raise NotImplementedError(
            f"Evaluation of expression {type(self).__name__}"
        )  # pragma: no cover; translation of unimplemented nodes trips first

    def evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """
        Evaluate this expression given a dataframe for context.

        Parameters
        ----------
        df
            DataFrame that will provide columns.
        context
            What context are we performing this evaluation in?

        Notes
        -----
        Individual subclasses should implement :meth:`do_evaluate`,
        this method provides logic to handle lookups in the
        substitution mapping.

        Returns
        -------
        Column representing the evaluation of the expression.

        Raises
        ------
        NotImplementedError
            If we couldn't evaluate the expression. Ideally all these
            are returned during translation to the IR, but for now we
            are not perfect.
        """
        return self.do_evaluate(df, context=context)

    def all_pointwise(self) -> bool:
        """Return True when this expression and all descendants are pointwise."""
        return all(e.is_pointwise for e in traversal([self]))

    @property
    def agg_request(self) -> plc.aggregation.Aggregation:
        """
        The aggregation for this expression in a grouped aggregation.

        Returns
        -------
        Aggregation request. Default is to collect the expression.

        Notes
        -----
        This presumes that the IR translation has decomposed groupby
        reductions only into cases we can handle.

        Raises
        ------
        NotImplementedError
            If requesting an aggregation from an unexpected expression.
        """
        return plc.aggregation.collect_list()


class ErrorExpr(Expr):
    __slots__ = ("error",)
    _non_child = ("dtype", "error")
    error: str

    def __init__(self, dtype: DataType, error: str) -> None:
        self.dtype = dtype
        self.error = error
        self.children = ()
        self.is_pointwise = False


class NamedExpr:
    # NamedExpr does not inherit from Expr since it does not appear
    # when evaluating expressions themselves, only when constructing
    # named return values in dataframe (IR) nodes.
    __slots__ = ("name", "value")
    value: Expr
    name: str

    def __init__(self, name: str, value: Expr) -> None:
        self.name = name
        self.value = value

    def __hash__(self) -> int:
        """Hash of the expression."""
        return hash((type(self), self.name, self.value))

    def __repr__(self) -> str:
        """Repr of the expression."""
        return f"NamedExpr({self.name}, {self.value})"

    def __eq__(self, other: Any) -> bool:
        """Equality of two expressions."""
        return (
            type(self) is type(other)
            and self.name == other.name
            and self.value == other.value
        )

    def __ne__(self, other: Any) -> bool:
        """Inequality of expressions."""
        return not self.__eq__(other)

    def evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """
        Evaluate this expression given a dataframe for context.

        Parameters
        ----------
        df
            DataFrame providing context
        context
            Execution context

        Returns
        -------
        Evaluated Column with name attached.

        See Also
        --------
        :meth:`Expr.evaluate` for details, this function just adds the
        name to a column produced from an expression.
        """
        return self.value.evaluate(df, context=context).rename(self.name)

    def all_pointwise(self) -> bool:
        """Return True when the underlying expression tree is pointwise."""
        return self.value.all_pointwise()

    def reconstruct(self, expr: Expr) -> Self:
        """
        Rebuild with a new `Expr` value.

        Parameters
        ----------
        expr
            New `Expr` value

        Returns
        -------
        New `NamedExpr` with `expr` as the underlying expression.
        The name of the original `NamedExpr` is preserved.
        """
        if expr is self.value:
            return self
        return type(self)(self.name, expr)


class Col(Expr):
    __slots__ = ("name",)
    _non_child = ("dtype", "name")
    name: str

    def __init__(self, dtype: DataType, name: str) -> None:
        self.dtype = dtype
        self.name = name
        self.is_pointwise = True
        self.children = ()

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        # Deliberately remove the name here so that we guarantee
        # evaluation of the IR produces names.
        return df.column_map[self.name].rename(None)


class ColRef(Expr):
    __slots__ = ("index", "table_ref")
    _non_child = ("dtype", "index", "table_ref")
    index: int
    table_ref: plc.expressions.TableReference

    def __init__(
        self,
        dtype: DataType,
        index: int,
        table_ref: plc.expressions.TableReference,
        column: Expr,
    ) -> None:
        if not isinstance(column, Col):
            raise TypeError("Column reference should only apply to columns")
        self.dtype = dtype
        self.index = index
        self.table_ref = table_ref
        self.is_pointwise = True
        self.children = (column,)

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        raise NotImplementedError(
            "Only expect this node as part of an expression translated to libcudf AST."
        )

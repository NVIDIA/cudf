# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversion of expression nodes to libcudf AST nodes."""

from __future__ import annotations

from functools import partial, reduce, singledispatch
from typing import TYPE_CHECKING, TypeAlias, TypedDict, cast

import polars as pl  # noqa: TC002 (used at runtime for pl.datatypes, pl.Series etc.)

import pylibcudf as plc
from pylibcudf import expressions as plc_expr

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.traversal import CachingVisitor, reuse_if_unchanged
from cudf_polars.typing import GenericTransformer

if TYPE_CHECKING:
    from collections.abc import Mapping

    from rmm.pylibrmm.stream import Stream


# Can't merge these op-mapping dictionaries because scoped enum values
# are exposed by cython with equality/hash based one their underlying
# representation type. So in a dict they are just treated as integers.
BINOP_TO_ASTOP = {
    plc.binaryop.BinaryOperator.EQUAL: plc_expr.ASTOperator.EQUAL,
    plc.binaryop.BinaryOperator.NULL_EQUALS: plc_expr.ASTOperator.NULL_EQUAL,
    plc.binaryop.BinaryOperator.NOT_EQUAL: plc_expr.ASTOperator.NOT_EQUAL,
    plc.binaryop.BinaryOperator.LESS: plc_expr.ASTOperator.LESS,
    plc.binaryop.BinaryOperator.LESS_EQUAL: plc_expr.ASTOperator.LESS_EQUAL,
    plc.binaryop.BinaryOperator.GREATER: plc_expr.ASTOperator.GREATER,
    plc.binaryop.BinaryOperator.GREATER_EQUAL: plc_expr.ASTOperator.GREATER_EQUAL,
    plc.binaryop.BinaryOperator.ADD: plc_expr.ASTOperator.ADD,
    plc.binaryop.BinaryOperator.SUB: plc_expr.ASTOperator.SUB,
    plc.binaryop.BinaryOperator.MUL: plc_expr.ASTOperator.MUL,
    plc.binaryop.BinaryOperator.DIV: plc_expr.ASTOperator.DIV,
    plc.binaryop.BinaryOperator.TRUE_DIV: plc_expr.ASTOperator.TRUE_DIV,
    plc.binaryop.BinaryOperator.FLOOR_DIV: plc_expr.ASTOperator.FLOOR_DIV,
    plc.binaryop.BinaryOperator.PYMOD: plc_expr.ASTOperator.PYMOD,
    plc.binaryop.BinaryOperator.BITWISE_AND: plc_expr.ASTOperator.BITWISE_AND,
    plc.binaryop.BinaryOperator.BITWISE_OR: plc_expr.ASTOperator.BITWISE_OR,
    plc.binaryop.BinaryOperator.BITWISE_XOR: plc_expr.ASTOperator.BITWISE_XOR,
    plc.binaryop.BinaryOperator.LOGICAL_AND: plc_expr.ASTOperator.LOGICAL_AND,
    plc.binaryop.BinaryOperator.LOGICAL_OR: plc_expr.ASTOperator.LOGICAL_OR,
    plc.binaryop.BinaryOperator.NULL_LOGICAL_AND: plc_expr.ASTOperator.NULL_LOGICAL_AND,
    plc.binaryop.BinaryOperator.NULL_LOGICAL_OR: plc_expr.ASTOperator.NULL_LOGICAL_OR,
}

UOP_TO_ASTOP = {
    plc.unary.UnaryOperator.SIN: plc_expr.ASTOperator.SIN,
    plc.unary.UnaryOperator.COS: plc_expr.ASTOperator.COS,
    plc.unary.UnaryOperator.TAN: plc_expr.ASTOperator.TAN,
    plc.unary.UnaryOperator.ARCSIN: plc_expr.ASTOperator.ARCSIN,
    plc.unary.UnaryOperator.ARCCOS: plc_expr.ASTOperator.ARCCOS,
    plc.unary.UnaryOperator.ARCTAN: plc_expr.ASTOperator.ARCTAN,
    plc.unary.UnaryOperator.SINH: plc_expr.ASTOperator.SINH,
    plc.unary.UnaryOperator.COSH: plc_expr.ASTOperator.COSH,
    plc.unary.UnaryOperator.TANH: plc_expr.ASTOperator.TANH,
    plc.unary.UnaryOperator.ARCSINH: plc_expr.ASTOperator.ARCSINH,
    plc.unary.UnaryOperator.ARCCOSH: plc_expr.ASTOperator.ARCCOSH,
    plc.unary.UnaryOperator.ARCTANH: plc_expr.ASTOperator.ARCTANH,
    plc.unary.UnaryOperator.EXP: plc_expr.ASTOperator.EXP,
    plc.unary.UnaryOperator.LOG: plc_expr.ASTOperator.LOG,
    plc.unary.UnaryOperator.SQRT: plc_expr.ASTOperator.SQRT,
    plc.unary.UnaryOperator.CBRT: plc_expr.ASTOperator.CBRT,
    plc.unary.UnaryOperator.CEIL: plc_expr.ASTOperator.CEIL,
    plc.unary.UnaryOperator.FLOOR: plc_expr.ASTOperator.FLOOR,
    plc.unary.UnaryOperator.ABS: plc_expr.ASTOperator.ABS,
    plc.unary.UnaryOperator.RINT: plc_expr.ASTOperator.RINT,
    plc.unary.UnaryOperator.BIT_INVERT: plc_expr.ASTOperator.BIT_INVERT,
    plc.unary.UnaryOperator.NOT: plc_expr.ASTOperator.NOT,
}

_DECIMAL_IDS = {plc.TypeId.DECIMAL32, plc.TypeId.DECIMAL64, plc.TypeId.DECIMAL128}


class ASTState(TypedDict):
    """
    State for AST transformations.

    Parameters
    ----------
    for_parquet
        Indicator for whether this transformation should provide an expression
        suitable for use in parquet filters.
    """

    for_parquet: bool
    stream: Stream


class ExprTransformerState(TypedDict):
    """
    State used for AST transformation when inserting column references.

    Parameters
    ----------
    name_to_index
        Mapping from column names to column indices in the table
        eventually used for evaluation.
    table_ref
        pylibcudf `TableReference` indicating whether column
        references are coming from the left or right table.
    """

    name_to_index: Mapping[str, int]
    table_ref: plc.expressions.TableReference


Transformer: TypeAlias = GenericTransformer[expr.Expr, plc_expr.Expression, ASTState]
ExprTransformer: TypeAlias = GenericTransformer[
    expr.Expr, expr.Expr, ExprTransformerState
]
"""Protocol for transformation of Expr nodes."""


@singledispatch
def _to_ast(node: expr.Expr, self: Transformer) -> plc_expr.Expression:
    """
    Translate an expression to a pylibcudf Expression.

    Parameters
    ----------
    node
        Expression to translate.
    self
        Recursive transformer. The state dictionary is an instance of
        :class:`ASTState`.

    Returns
    -------
    pylibcudf Expression.

    Raises
    ------
    NotImplementedError or KeyError if the expression cannot be translated.
    """
    raise NotImplementedError(f"Unhandled expression type {type(node)}")


@_to_ast.register
def _(node: expr.Col, self: Transformer) -> plc_expr.Expression:
    if self.state["for_parquet"]:
        return plc_expr.ColumnNameReference(node.name)
    raise TypeError("Should always be wrapped in a ColRef node before translation")


@_to_ast.register
def _(node: expr.ColRef, self: Transformer) -> plc_expr.Expression:
    if self.state["for_parquet"]:
        raise TypeError("Not expecting ColRef node in parquet filter")
    return plc_expr.ColumnReference(node.index, node.table_ref)


@_to_ast.register
def _(node: expr.Literal, self: Transformer) -> plc_expr.Expression:
    return plc_expr.Literal(
        plc.Scalar.from_py(node.value, node.dtype.plc_type, stream=self.state["stream"])
    )


@_to_ast.register
def _(node: expr.BinOp, self: Transformer) -> plc_expr.Expression:
    if node.op == plc.binaryop.BinaryOperator.NULL_NOT_EQUALS:
        return plc_expr.Operation(
            plc_expr.ASTOperator.NOT,
            self(
                # Reconstruct and apply, rather than directly
                # constructing the right expression so we get the
                # handling of parquet special cases for free.
                expr.BinOp(
                    node.dtype, plc.binaryop.BinaryOperator.NULL_EQUALS, *node.children
                )
            ),
        )
    c1, c2 = node.children
    if c1.dtype != c2.dtype:
        if isinstance(c1, expr.Literal):  # pragma: no cover
            c1 = c1.astype(c2.dtype)
        elif isinstance(c2, expr.Literal):
            c2 = c2.astype(c1.dtype)
        elif (
            isinstance(c1, (expr.Col, expr.ColRef)) and c1.dtype.id() in _DECIMAL_IDS
        ) or (
            isinstance(c2, (expr.Col, expr.ColRef)) and c2.dtype.id() in _DECIMAL_IDS
        ):
            # Allow mixed-precision decimal, or mixed decimal-float operations through
            # unchanged.
            pass
        else:
            raise NotImplementedError(
                "BinOp with mismatching dtypes"
            )  # pragma: no cover
    children = (c1, c2)
    return plc_expr.Operation(BINOP_TO_ASTOP[node.op], *map(self, children))


@_to_ast.register
def _(node: expr.BooleanFunction, self: Transformer) -> plc_expr.Expression:
    if node.name is expr.BooleanFunction.Name.IsIn:
        needles, haystack = node.children
        if isinstance(haystack, expr.LiteralColumn) and len(haystack.value) < 16:
            # 16 is an arbitrary limit
            needle_ref = self(needles)
            if haystack.dtype.id() == plc.TypeId.LIST:
                # Because we originally translated pl_expr.Literal with a list scalar
                # to a expr.LiteralColumn, so the actual type is in the inner type
                # .inner returns DataTypeClass | DataType, need to cast to DataType
                plc_dtype = DataType(
                    cast(
                        "pl.DataType", cast("pl.List", haystack.dtype.polars_type).inner
                    )
                ).plc_type
            else:
                plc_dtype = haystack.dtype.plc_type  # pragma: no cover
            values = (
                plc_expr.Literal(
                    plc.Scalar.from_py(val, plc_dtype, stream=self.state["stream"])
                )
                for val in haystack.value
            )
            return reduce(
                partial(plc_expr.Operation, plc_expr.ASTOperator.LOGICAL_OR),
                (
                    plc_expr.Operation(plc_expr.ASTOperator.EQUAL, needle_ref, value)
                    for value in values
                ),
            )
    if self.state["for_parquet"] and isinstance(node.children[0], expr.Col):
        if node.name not in (
            expr.BooleanFunction.Name.IsNull,
            expr.BooleanFunction.Name.IsNotNull,
        ):
            raise NotImplementedError(
                f"Parquet filters don't support {node.name} on columns"
            )
        if node.children[0].dtype.id() in (plc.TypeId.STRUCT, plc.TypeId.LIST):
            # TODO: Remove once https://github.com/rapidsai/cudf/issues/23397 is resolved.
            raise NotImplementedError(
                f"Parquet filters don't support {node.name} on nested types"
            )
    if node.name is expr.BooleanFunction.Name.IsNull:
        return plc_expr.Operation(plc_expr.ASTOperator.IS_NULL, self(node.children[0]))
    elif node.name is expr.BooleanFunction.Name.IsNotNull:
        return plc_expr.Operation(
            plc_expr.ASTOperator.NOT,
            plc_expr.Operation(plc_expr.ASTOperator.IS_NULL, self(node.children[0])),
        )
    elif node.name is expr.BooleanFunction.Name.Not:
        return plc_expr.Operation(plc_expr.ASTOperator.NOT, self(node.children[0]))
    raise NotImplementedError(f"AST conversion does not support {node.name}")


@_to_ast.register
def _(node: expr.UnaryFunction, self: Transformer) -> plc_expr.Expression:
    if isinstance(node.children[0], expr.Col) and self.state["for_parquet"]:
        raise NotImplementedError(
            "Parquet filters don't support {node.name} on columns"
        )
    return plc_expr.Operation(
        UOP_TO_ASTOP[node._OP_MAPPING[node.name]], self(node.children[0])
    )


def _extract_conjuncts(node: expr.Expr) -> list[expr.Expr]:
    if (
        isinstance(node, expr.BinOp)
        and node.op == plc.binaryop.BinaryOperator.NULL_LOGICAL_AND
    ):
        return [c for child in node.children for c in _extract_conjuncts(child)]
    return [node]


def _to_parquet_filter(
    node: expr.Expr, mapper: Transformer
) -> plc_expr.Expression | None:
    # Converts a boolean column reference (e.g., filter(pl.col("foo")))
    # to an explicit comparison for parquet filters (e.g., filter(pl.col("foo") == True)).
    # TODO: Have polars pass us the comparison instead
    if isinstance(node, expr.Col) and node.dtype.id() == plc.TypeId.BOOL8:
        node = expr.BinOp(
            node.dtype,
            plc.binaryop.BinaryOperator.EQUAL,
            node,
            expr.Literal(node.dtype, value=True),
        )
    try:
        return mapper(node)
    except (KeyError, NotImplementedError):
        return None


def to_parquet_filter(
    node: expr.Expr, stream: Stream
) -> tuple[plc_expr.Expression | None, expr.Expr | None]:
    """
    Convert an expression to libcudf AST nodes suitable for parquet filtering.

    Parameters
    ----------
    node
        Expression to convert.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Returns
    -------
    filter
        pylibcudf Expression suitable for parquet filtering, or None if no part
        of the predicate can be converted.
    residual
        Expression still to be applied as a post-read filter, or None when
        ``filter`` is exact (equivalent to ``node``).  When ``filter`` is None
        the caller must apply the full original predicate post-read.
    """
    mapper: Transformer = CachingVisitor(
        _to_ast, state={"for_parquet": True, "stream": stream}
    )
    whole = _to_parquet_filter(node, mapper)
    if whole is not None:
        return whole, None
    can_handle_filters = []
    cant_handle_exprs = []
    for conjunct in _extract_conjuncts(node):
        f = _to_parquet_filter(conjunct, mapper)
        if f is not None:
            can_handle_filters.append(f)
        else:
            cant_handle_exprs.append(conjunct)
    if not can_handle_filters:
        return None, None
    combined = reduce(
        partial(plc_expr.Operation, plc_expr.ASTOperator.LOGICAL_AND),
        can_handle_filters,
    )
    residual = reduce(
        lambda a, b: expr.BinOp(
            b.dtype, plc.binaryop.BinaryOperator.NULL_LOGICAL_AND, a, b
        ),
        cant_handle_exprs,
    )
    return combined, residual


def to_ast(node: expr.Expr, stream: Stream) -> plc_expr.Expression | None:
    """
    Convert an expression to libcudf AST nodes suitable for compute_column.

    Parameters
    ----------
    node
        Expression to convert.
    stream
        CUDA stream used for device memory operations and kernel launches.

    Notes
    -----
    `Col` nodes must always be wrapped in `TableRef` nodes when
    converting to an ast expression so that their table reference and
    index are provided.

    Returns
    -------
    pylibcudf Expression if conversion is possible, otherwise None.
    """
    mapper: Transformer = CachingVisitor(
        _to_ast, state={"for_parquet": False, "stream": stream}
    )
    try:
        return mapper(node)
    except (KeyError, NotImplementedError):
        return None


def _insert_colrefs(node: expr.Expr, rec: ExprTransformer) -> expr.Expr:
    if isinstance(node, expr.Col):
        return expr.ColRef(
            node.dtype,
            rec.state["name_to_index"][node.name],
            rec.state["table_ref"],
            node,
        )
    return reuse_if_unchanged(node, rec)


def insert_colrefs(
    node: expr.Expr,
    *,
    table_ref: plc.expressions.TableReference,
    name_to_index: Mapping[str, int],
) -> expr.Expr:
    """
    Insert column references into an expression before conversion to libcudf AST.

    Parameters
    ----------
    node
        Expression to insert references into.
    table_ref
        pylibcudf `TableReference` indicating whether column
        references are coming from the left or right table.
    name_to_index:
        Mapping from column names to column indices in the table
        eventually used for evaluation.

    Notes
    -----
    All column references are wrapped in the same, singular, table
    reference, so this function relies on the expression only
    containing column references from a single table.

    Returns
    -------
    New expression with column references inserted.
    """
    mapper: ExprTransformer = CachingVisitor(
        _insert_colrefs,
        state={"name_to_index": name_to_index, "table_ref": table_ref},
    )
    return mapper(node)

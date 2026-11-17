# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conversion of expression nodes to libcudf AST nodes."""

from __future__ import annotations

from decimal import Decimal, localcontext
from functools import partial, reduce, singledispatch
from typing import TYPE_CHECKING, NamedTuple, TypedDict, cast

import polars as pl

import pylibcudf as plc
from pylibcudf import expressions as plc_expr

from cudf_polars.containers import DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.traversal import CachingVisitor, reuse_if_unchanged, traversal
from cudf_polars.dsl.utils.naming import unique_names
from cudf_polars.dsl.utils.replace import replace
from cudf_polars.typing import GenericTransformer

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping, Sequence

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


class JoinPredicateState(TypedDict):
    """
    State used for splitting the extra condition of a join.

    Parameters
    ----------
    columns
        For each output column of the join, the table it comes from and its
        name there.
    indices
        For each table, the index of each of its columns.
    names
        For each table, a generator of names for columns to append to it.
    extras
        For each table, the expressions to append to it and their names.
    """

    columns: Mapping[str, tuple[plc_expr.TableReference, str]]
    indices: Mapping[plc_expr.TableReference, Mapping[str, int]]
    names: Mapping[plc_expr.TableReference, Generator[str, None, None]]
    extras: Mapping[plc_expr.TableReference, dict[expr.Expr, str]]


type Transformer = GenericTransformer[expr.Expr, plc_expr.Expression, ASTState]
type ExprTransformer = GenericTransformer[expr.Expr, expr.Expr, ExprTransformerState]
"""Protocol for transformation of Expr nodes."""
type JoinPredicateTransformer = GenericTransformer[
    expr.Expr, expr.Expr, JoinPredicateState
]


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


def _with_polars_nan_semantics(
    op: plc.binaryop.BinaryOperator,
    left: plc_expr.Expression,
    right: plc_expr.Expression,
    result: plc_expr.Expression,
) -> plc_expr.Expression:
    """
    Make an AST float comparison treat NaN the way Polars does.

    Polars orders NaN above every other value and considers it equal to
    itself, but an AST comparison follows IEEE 754.
    """
    # TODO: Use NaN-aware comparison operators once libcudf has them, and
    # remove this workaround. https://github.com/NVIDIA/cudf/issues/18930
    binop = plc.binaryop.BinaryOperator

    def is_nan(operand: plc_expr.Expression) -> plc_expr.Expression:
        # NaN is the only value that is not equal to itself.
        if op == binop.NULL_EQUALS:
            # A null-aware comparison is never null, so a null must not make
            # the NaN test null. Nulls are equal to themselves here.
            return plc_expr.Operation(
                plc_expr.ASTOperator.NOT,
                plc_expr.Operation(plc_expr.ASTOperator.NULL_EQUAL, operand, operand),
            )
        return plc_expr.Operation(plc_expr.ASTOperator.NOT_EQUAL, operand, operand)

    def logical(
        astop: plc_expr.ASTOperator, *operands: plc_expr.Expression
    ) -> plc_expr.Expression:
        return plc_expr.Operation(astop, *operands)

    AND, OR, NOT = (
        plc_expr.ASTOperator.LOGICAL_AND,
        plc_expr.ASTOperator.LOGICAL_OR,
        plc_expr.ASTOperator.NOT,
    )
    left_nan, right_nan = is_nan(left), is_nan(right)
    if op == binop.GREATER:
        return logical(OR, result, logical(AND, left_nan, logical(NOT, right_nan)))
    if op == binop.GREATER_EQUAL:
        return logical(OR, result, left_nan)
    if op == binop.LESS:
        return logical(OR, result, logical(AND, right_nan, logical(NOT, left_nan)))
    if op == binop.LESS_EQUAL:
        return logical(OR, result, right_nan)
    both_nan = logical(AND, left_nan, right_nan)
    if op in {binop.EQUAL, binop.NULL_EQUALS}:
        return logical(OR, result, both_nan)
    assert op == binop.NOT_EQUAL
    return logical(AND, result, logical(NOT, both_nan))


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
    if (
        node.op == plc.binaryop.BinaryOperator.BITWISE_XOR
        and c1.dtype.id() == plc.TypeId.BOOL8
    ):
        # libcudf's AST rejects boolean BITWISE_XOR. Boolean NOT_EQUAL has the
        # same truth table and null propagation.
        ast_op = plc_expr.ASTOperator.NOT_EQUAL
    else:
        ast_op = BINOP_TO_ASTOP[node.op]
    if c1.dtype != c2.dtype:
        if isinstance(c1, expr.Literal):  # pragma: no cover
            c1 = c1.astype(c2.dtype)
        elif isinstance(c2, expr.Literal):  # pragma: no cover
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
    left, right = map(self, (c1, c2))
    result = plc_expr.Operation(ast_op, left, right)
    if (
        node.op in expr.BinOp._NAN_AWARE_COMPARISONS
        and plc.traits.is_floating_point(c1.dtype.plc_type)
        and plc.traits.is_floating_point(c2.dtype.plc_type)
    ):
        return _with_polars_nan_semantics(node.op, left, right, result)
    return result


@_to_ast.register
def _(node: expr.BooleanFunction, self: Transformer) -> plc_expr.Expression:
    if node.name is expr.BooleanFunction.Name.IsIn:
        needles, haystack = node.children
        (nulls_equal,) = node.options
        if isinstance(haystack, expr.LiteralColumn):
            haystack_values = haystack.value
            if haystack_values.dtype == pl.List:
                # An imploded literal is a one-row list column, use the inner values
                haystack_values = haystack_values.explode()
            if 0 < len(haystack_values) < 16:
                # 16 is an arbitrary limit
                if haystack_values.dtype.is_float() and haystack_values.is_nan().any():
                    # NaN never compares equal in an AST, but is_in matches it
                    raise NotImplementedError("is_in with NaN in the haystack")
                if not nulls_equal:
                    # A null in the haystack never matches when nulls are not equal
                    haystack_values = haystack_values.drop_nulls()
                if not len(haystack_values):
                    raise NotImplementedError("is_in with only nulls in the haystack")
                needle_ref = self(needles)
                if haystack.dtype.id() == plc.TypeId.LIST:
                    # Because we originally translated pl_expr.Literal with a list scalar
                    # to a expr.LiteralColumn, so the actual type is in the inner type
                    # .inner returns DataTypeClass | DataType, need to cast to DataType
                    plc_dtype = DataType(
                        cast(
                            "pl.DataType",
                            cast("pl.List", haystack.dtype.polars_type).inner,
                        )
                    ).plc_type
                else:
                    plc_dtype = haystack.dtype.plc_type  # pragma: no cover
                values = (
                    plc_expr.Literal(
                        plc.Scalar.from_py(val, plc_dtype, stream=self.state["stream"])
                    )
                    for val in haystack_values
                )
                equal = (
                    plc_expr.ASTOperator.NULL_EQUAL
                    if nulls_equal
                    else plc_expr.ASTOperator.EQUAL
                )
                return reduce(
                    partial(plc_expr.Operation, plc_expr.ASTOperator.LOGICAL_OR),
                    (plc_expr.Operation(equal, needle_ref, value) for value in values),
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
            # TODO: Remove once https://github.com/NVIDIA/cudf/issues/23397 is resolved.
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
    node: expr.Expr, mapper: Transformer, unreadable_columns: frozenset[str]
) -> plc_expr.Expression | None:
    if unreadable_columns and any(
        isinstance(child, expr.Col) and child.name in unreadable_columns
        for child in traversal([node])
    ):
        return None
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
    node: expr.Expr,
    stream: Stream,
    unreadable_columns: frozenset[str] = frozenset(),
) -> tuple[plc_expr.Expression | None, expr.Expr | None]:
    """
    Convert an expression to libcudf AST nodes suitable for parquet filtering.

    Parameters
    ----------
    node
        Expression to convert.
    stream
        CUDA stream used for device memory operations and kernel launches.
    unreadable_columns
        Names of columns that are part of the scan's schema but are not stored
        in the files, such as hive partition keys. Conjuncts referencing them
        are left to the residual so they can be applied once those columns have
        been materialized.

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
    whole = _to_parquet_filter(node, mapper, unreadable_columns)
    if whole is not None:
        return whole, None
    can_handle_filters = []
    cant_handle_exprs = []
    for conjunct in _extract_conjuncts(node):
        f = _to_parquet_filter(conjunct, mapper, unreadable_columns)
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


class JoinPredicate(NamedTuple):
    """
    An extra condition of an equi join, split for libcudf's mixed join.

    Parts of the condition that use columns of one input only are evaluated
    on that input before the join. Only the remaining comparison between the
    inputs is left to the AST.
    """

    predicate: expr.Expr
    """Expression with column references into the left and right tables."""
    left_exprs: tuple[expr.NamedExpr, ...]
    """Columns appended to the left table, evaluated on the left input."""
    right_exprs: tuple[expr.NamedExpr, ...]
    """Columns appended to the right table, evaluated on the right input."""


def _sides(node: expr.Expr, state: JoinPredicateState) -> set[plc_expr.TableReference]:
    return {
        state["columns"][col.name][0]
        for col in traversal([node])
        if isinstance(col, expr.Col)
    }


def _is_float_to_decimal_cast(node: expr.Expr) -> bool:
    return (
        isinstance(node, expr.Cast)
        and node.dtype.id() in _DECIMAL_IDS
        and node.children[0].dtype.id() == plc.TypeId.FLOAT64
    )


# 10**22 is the largest power of ten that a Float64 holds exactly.
_MAX_EXACT_POWER_OF_TEN = 22
_MAX_DECIMAL_DIGITS = 38
# 10**15 is below 2**53, so a decimal with at most 15 digits is exact in Float64.
_MAX_EXACT_DECIMAL_DIGITS = 15


def _converts_exactly_to_float64(operand: expr.Expr, scale: int) -> bool:
    if _is_float_to_decimal_cast(operand):
        return True
    if isinstance(operand, expr.Literal):
        if operand.value is None:
            return True
        # Polars gives a Decimal for a decimal literal, but Literal.value is
        # untyped, so leave anything else to the fallback.
        if not isinstance(operand.value, Decimal):
            return False
        with localcontext(prec=2 * _MAX_DECIMAL_DIGITS):
            unscaled = operand.value.scaleb(scale)
            return unscaled == int(float(unscaled))
    polars_type = operand.dtype.polars_type
    precision = polars_type.precision if isinstance(polars_type, pl.Decimal) else None
    return (precision or _MAX_DECIMAL_DIGITS) <= _MAX_EXACT_DECIMAL_DIGITS


def _emulate_decimal_comparison(node: expr.BinOp) -> expr.Expr | None:
    """
    Rewrite a comparison with an operand cast from a float to a decimal.

    libcudf's AST cannot cast to a decimal, and comparing the floats before the
    cast gives a different answer when two floats round to the same decimal.
    Polars rounds a float to a decimal by rounding ``x * 10**scale`` to the
    nearest integer, ties to even, in Float64, which is the AST's ``RINT``.
    So compare the two operands as ``RINT(x * 10**scale)``, which holds the
    unscaled decimal, instead.

    The operand that is not a cast is converted to Float64 first. Float64 only
    holds integers exactly below 2**53, so two different decimals above that
    could compare equal. The rewrite is skipped unless that operand is a
    literal that converts exactly or a column with at most 15 digits.

    Parameters
    ----------
    node
        Comparison to rewrite.

    Returns
    -------
    The rewritten comparison, or None when it is not a comparison of two
    decimals with the same scale where one is cast from a float, or when the
    other operand does not convert exactly to Float64.
    """
    left, right = node.children
    if not (_is_float_to_decimal_cast(left) or _is_float_to_decimal_cast(right)):
        return None
    # libcudf stores the scale as a negative exponent.
    scales = {
        -child.dtype.plc_type.scale() if child.dtype.id() in _DECIMAL_IDS else None
        for child in node.children
    }
    if (
        len(scales) > 1
        or (scale := scales.pop()) is None
        or scale > _MAX_EXACT_POWER_OF_TEN
    ):
        return None
    if not all(_converts_exactly_to_float64(child, scale) for child in node.children):
        return None
    float64 = DataType(pl.Float64())

    def unscaled(operand: expr.Expr) -> expr.Expr:
        if isinstance(operand, expr.Literal):
            if operand.value is None:
                return expr.Literal(float64, None)
            # The default context rounds to 28 digits, fewer than a decimal has.
            with localcontext(prec=2 * _MAX_DECIMAL_DIGITS):
                return expr.Literal(float64, float(operand.value.scaleb(scale)))
        value = (
            operand.children[0]
            if _is_float_to_decimal_cast(operand)
            else expr.Cast(float64, True, operand)  # noqa: FBT003
        )
        scaled = expr.BinOp(
            float64,
            plc.binaryop.BinaryOperator.MUL,
            value,
            expr.Literal(float64, 10.0**scale),
        )
        return expr.UnaryFunction(float64, "rint", (), scaled)

    return expr.BinOp(node.dtype, node.op, unscaled(left), unscaled(right))


def _split_join_predicate(node: expr.Expr, rec: JoinPredicateTransformer) -> expr.Expr:
    state = rec.state
    if isinstance(node, expr.Col):
        table_ref, name = state["columns"][node.name]
        return expr.ColRef(
            node.dtype,
            state["indices"][table_ref][name],
            table_ref,
            expr.Col(node.dtype, name),
        )
    sides = _sides(node, state)
    if len(sides) == 1:
        (table_ref,) = sides
        if not node.all_pointwise():
            raise NotImplementedError("Join predicate with a non-pointwise expression")
        renames: dict[expr.Expr, expr.Expr] = {
            col: expr.Col(col.dtype, state["columns"][col.name][1])
            for col in traversal([node])
            if isinstance(col, expr.Col)
        }
        (local,) = replace([node], renames)
        extras = state["extras"][table_ref]
        if local not in extras:
            extras[local] = next(state["names"][table_ref])
        index = len(state["indices"][table_ref]) + list(extras).index(local)
        return expr.ColRef(
            node.dtype, index, table_ref, expr.Col(node.dtype, extras[local])
        )
    if isinstance(node, expr.BinOp) and node.op in expr.BinOp._COMPARISONS:
        # The AST has no conditional, so compare each branch of a conditional
        # whose condition only uses one input, and select with AND and OR.
        for position, child in enumerate(node.children):
            if (
                isinstance(child, expr.Ternary)
                and len(_sides(child.children[0], state)) <= 1
            ):
                return rec(_distribute_comparison(node, position))
        if (emulated := _emulate_decimal_comparison(node)) is not None:
            return rec(emulated)
    return reuse_if_unchanged(node, rec)


def _distribute_comparison(node: expr.BinOp, position: int) -> expr.Expr:
    """
    Rewrite a comparison with a conditional operand as AND and OR.

    ``(when c then x otherwise y) < z`` becomes
    ``(c and x < z) or (not c and y < z)``, where a null ``c`` counts as false,
    as it does in the conditional.
    """
    ternary = node.children[position]
    other = node.children[1 - position]
    when, then, otherwise = ternary.children
    boolean = when.dtype
    true = expr.Literal(boolean, True)  # noqa: FBT003
    false = expr.Literal(boolean, False)  # noqa: FBT003

    def compare(branch: expr.Expr) -> expr.Expr:
        operands = (branch, other) if position == 0 else (other, branch)
        return expr.BinOp(node.dtype, node.op, *operands)

    def select(condition: expr.Expr, branch: expr.Expr) -> expr.Expr:
        return expr.BinOp(
            node.dtype,
            plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
            condition,
            compare(branch),
        )

    return expr.BinOp(
        node.dtype,
        plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
        select(expr.Ternary(boolean, when, true, false), then),
        select(expr.Ternary(boolean, when, false, true), otherwise),
    )


def split_join_predicate(
    node: expr.Expr,
    *,
    columns: Mapping[str, tuple[plc_expr.TableReference, str]],
    left_names: Sequence[str],
    right_names: Sequence[str],
) -> JoinPredicate:
    """
    Split an equi join's extra condition for libcudf's mixed join.

    Parameters
    ----------
    node
        Condition over the columns of the join's output.
    columns
        For each output column, the table it comes from and its name there.
    left_names, right_names
        Column names of the left and right inputs.

    Returns
    -------
    The condition with column references inserted, and the columns to add
    to each input so that every comparison in the condition is between
    columns of the two inputs.

    Raises
    ------
    NotImplementedError
        If a part of the condition that uses one input cannot be evaluated
        on that input.
    """
    left = plc_expr.TableReference.LEFT
    right = plc_expr.TableReference.RIGHT
    extras: dict[plc_expr.TableReference, dict[expr.Expr, str]] = {left: {}, right: {}}
    mapper: JoinPredicateTransformer = CachingVisitor(
        _split_join_predicate,
        state={
            "columns": columns,
            "indices": {
                left: {name: i for i, name in enumerate(left_names)},
                right: {name: i for i, name in enumerate(right_names)},
            },
            "names": {left: unique_names(left_names), right: unique_names(right_names)},
            "extras": extras,
        },
    )
    predicate = mapper(node)
    return JoinPredicate(
        predicate,
        tuple(expr.NamedExpr(name, e) for e, name in extras[left].items()),
        tuple(expr.NamedExpr(name, e) for e, name in extras[right].items()),
    )

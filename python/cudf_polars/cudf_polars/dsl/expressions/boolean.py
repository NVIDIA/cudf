# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: Document BooleanFunction to remove noqa
# ruff: noqa: D101
"""Boolean DSL nodes."""

from __future__ import annotations

from enum import IntEnum, auto
from functools import partial, reduce
from typing import TYPE_CHECKING, Any, ClassVar, cast

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import (
    ExecutionContext,
    Expr,
)
from cudf_polars.dsl.expressions.literal import LiteralColumn

if TYPE_CHECKING:
    from typing import Self

    import polars.type_aliases as pl_types
    from polars import polars  # type: ignore[attr-defined]

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame

__all__ = ["BooleanFunction"]


def _nesting_level(dtype: pl.DataType) -> int:
    level = 0
    current = dtype
    while isinstance(current, pl.List):
        level += 1
        current = cast("pl.DataType", current.inner)
    return level


class BooleanFunction(Expr):
    class Name(IntEnum):
        """Internal and picklable representation of polars' `BooleanFunction`."""

        All = auto()
        AllHorizontal = auto()
        Any = auto()
        AnyHorizontal = auto()
        HasNulls = auto()
        IsBetween = auto()
        IsClose = auto()
        IsDuplicated = auto()
        IsEmpty = auto()
        IsFinite = auto()
        IsFirstDistinct = auto()
        IsIn = auto()
        IsInfinite = auto()
        IsLastDistinct = auto()
        IsNan = auto()
        IsNotNan = auto()
        IsNotNull = auto()
        IsNull = auto()
        IsSorted = auto()
        IsUnique = auto()
        Not = auto()

        @classmethod
        def from_polars(cls, obj: polars._expr_nodes.BooleanFunction) -> Self:
            """Convert from polars' `BooleanFunction`."""
            try:
                function, name = str(obj).split(".", maxsplit=1)
            except ValueError:
                # Failed to unpack string
                function = None
            if function != "BooleanFunction":
                raise ValueError("BooleanFunction required")
            return getattr(cls, name)

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")

    def __init__(
        self,
        dtype: DataType,
        name: BooleanFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = self.name not in (
            BooleanFunction.Name.All,
            BooleanFunction.Name.Any,
            BooleanFunction.Name.HasNulls,
            BooleanFunction.Name.IsDuplicated,
            BooleanFunction.Name.IsEmpty,
            BooleanFunction.Name.IsFirstDistinct,
            BooleanFunction.Name.IsLastDistinct,
            BooleanFunction.Name.IsSorted,
            BooleanFunction.Name.IsUnique,
        )
        if self.name is BooleanFunction.Name.IsIn and len(children) == 2:
            # TODO: Polars should raise an error ahead of time
            # for us for these kind of shape mismatches
            needles, haystack = children
            if (
                isinstance(needles, LiteralColumn)
                and isinstance(haystack, LiteralColumn)
                and len(needles.value) != len(haystack.value)
            ):
                needles_level = _nesting_level(needles.dtype.polars_type)
                haystack_level = _nesting_level(haystack.dtype.polars_type)

                if needles_level != haystack_level:
                    raise NotImplementedError(
                        f"arguments for `is_in` have different lengths ({len(needles.value)} != {len(haystack.value)})"
                    )
        if self.name is BooleanFunction.Name.IsClose:
            abs_tol, rel_tol, _ = self.options
            if abs_tol < 0.0:
                raise ComputeError(f"`abs_tol` must be non-negative but got {abs_tol}")
            if rel_tol < 0.0:
                raise ComputeError(f"`rel_tol` must be non-negative but got {rel_tol}")
            for child in self.children:
                typ = child.dtype.polars_type
                if not typ.is_numeric():
                    raise InvalidOperationError(
                        f"is_close operation not supported for dtype `{typ}`"
                    )

    @staticmethod
    def _distinct(
        column: Column,
        dtype: DataType,
        *,
        keep: plc.stream_compaction.DuplicateKeepOption,
        source_value: plc.Scalar,
        target_value: plc.Scalar,
        stream: Stream,
    ) -> Column:
        table = plc.Table([column.obj])
        indices = plc.stream_compaction.distinct_indices(
            table,
            keep,
            # TODO: polars doesn't expose options for these
            plc.types.NullEquality.EQUAL,
            plc.types.NanEquality.ALL_EQUAL,
            stream=stream,
        )
        return Column(
            plc.copying.scatter(
                [source_value],
                indices,
                plc.Table(
                    [
                        plc.Column.from_scalar(
                            target_value, table.num_rows(), stream=stream
                        )
                    ]
                ),
                stream=stream,
            ).columns()[0],
            dtype=dtype,
        )

    def _is_close(self, df: DataFrame, *, context: ExecutionContext) -> Column:
        # Matches polars' PEP 485 semantics (see
        # crates/polars-ops/src/series/ops/is_close.rs):
        #   |x - y| <= max(rel_tol * max(|x|, |y|), abs_tol)
        # with special handling for non-finite values. The whole predicate is
        # evaluated as a single fused libcudf AST expression via
        # ``compute_column`` to minimize kernel launches. Nulls propagate
        # through the (non-Kleene) AST logical ops, so the result is null iff
        # either input is null, matching polars.
        abs_tol, rel_tol, nans_equal = self.options
        left, right = (child.evaluate(df, context=context) for child in self.children)
        f64 = plc.DataType(plc.TypeId.FLOAT64)
        stream = df.stream

        table_columns: list[plc.Column] = []
        both_scalar = left.is_scalar and right.is_scalar

        def prep(col: Column) -> plc.expressions.Expression:
            obj = col.obj
            if obj.type().id() != plc.TypeId.FLOAT64:
                obj = plc.unary.cast(obj, f64, stream=stream)
            if col.is_scalar and not both_scalar:
                return plc.expressions.Literal(obj.to_scalar(stream=stream))
            table_columns.append(obj)
            return plc.expressions.ColumnReference(len(table_columns) - 1)

        x = prep(left)
        y = prep(right)
        table = plc.Table(table_columns)

        def maximum_nonnegative(
            lhs: plc.expressions.Expression, rhs: plc.expressions.Expression
        ) -> plc.expressions.Operation:
            half = plc.expressions.Literal(plc.Scalar.from_py(0.5, f64, stream=stream))
            return plc.expressions.Operation(
                plc.expressions.ASTOperator.ADD,
                plc.expressions.Operation(
                    plc.expressions.ASTOperator.ADD,
                    plc.expressions.Operation(
                        plc.expressions.ASTOperator.MUL, lhs, half
                    ),
                    plc.expressions.Operation(
                        plc.expressions.ASTOperator.MUL, rhs, half
                    ),
                ),
                plc.expressions.Operation(
                    plc.expressions.ASTOperator.MUL,
                    plc.expressions.Operation(
                        plc.expressions.ASTOperator.ABS,
                        plc.expressions.Operation(
                            plc.expressions.ASTOperator.SUB, lhs, rhs
                        ),
                    ),
                    half,
                ),
            )

        inf = plc.expressions.Literal(
            plc.Scalar.from_py(float("inf"), f64, stream=stream)
        )
        absx = plc.expressions.Operation(plc.expressions.ASTOperator.ABS, x)
        absy = plc.expressions.Operation(plc.expressions.ASTOperator.ABS, y)
        absdiff = plc.expressions.Operation(
            plc.expressions.ASTOperator.ABS,
            plc.expressions.Operation(plc.expressions.ASTOperator.SUB, x, y),
        )

        tol = maximum_nonnegative(
            plc.expressions.Operation(
                plc.expressions.ASTOperator.MUL,
                plc.expressions.Literal(
                    plc.Scalar.from_py(rel_tol, f64, stream=stream)
                ),
                maximum_nonnegative(absx, absy),
            ),
            plc.expressions.Literal(plc.Scalar.from_py(abs_tol, f64, stream=stream)),
        )
        cmp = plc.expressions.Operation(
            plc.expressions.ASTOperator.LESS_EQUAL, absdiff, tol
        )

        # NaN iff value != itself; Inf iff |value| == inf.
        nan_x = plc.expressions.Operation(plc.expressions.ASTOperator.NOT_EQUAL, x, x)
        nan_y = plc.expressions.Operation(plc.expressions.ASTOperator.NOT_EQUAL, y, y)
        inf_x = plc.expressions.Operation(plc.expressions.ASTOperator.EQUAL, absx, inf)
        inf_y = plc.expressions.Operation(plc.expressions.ASTOperator.EQUAL, absy, inf)
        finite_x = plc.expressions.Operation(
            plc.expressions.ASTOperator.NOT,
            plc.expressions.Operation(
                plc.expressions.ASTOperator.LOGICAL_OR, nan_x, inf_x
            ),
        )
        finite_y = plc.expressions.Operation(
            plc.expressions.ASTOperator.NOT,
            plc.expressions.Operation(
                plc.expressions.ASTOperator.LOGICAL_OR, nan_y, inf_y
            ),
        )
        both_finite = plc.expressions.Operation(
            plc.expressions.ASTOperator.LOGICAL_AND, finite_x, finite_y
        )
        part_finite = plc.expressions.Operation(
            plc.expressions.ASTOperator.LOGICAL_AND, both_finite, cmp
        )

        # Infinities are close iff both are infinite with the same sign.
        part_inf = plc.expressions.Operation(
            plc.expressions.ASTOperator.LOGICAL_AND,
            plc.expressions.Operation(
                plc.expressions.ASTOperator.LOGICAL_AND, inf_x, inf_y
            ),
            plc.expressions.Operation(plc.expressions.ASTOperator.EQUAL, x, y),
        )

        predicate = plc.expressions.Operation(
            plc.expressions.ASTOperator.LOGICAL_OR, part_finite, part_inf
        )
        if nans_equal:
            predicate = plc.expressions.Operation(
                plc.expressions.ASTOperator.LOGICAL_OR,
                predicate,
                plc.expressions.Operation(
                    plc.expressions.ASTOperator.LOGICAL_AND, nan_x, nan_y
                ),
            )

        return Column(
            plc.transform.compute_column(table, predicate, stream=stream),
            dtype=self.dtype,
        )

    _BETWEEN_OPS: ClassVar[
        dict[
            pl_types.ClosedInterval,
            tuple[plc.binaryop.BinaryOperator, plc.binaryop.BinaryOperator],
        ]
    ] = {
        "none": (
            plc.binaryop.BinaryOperator.GREATER,
            plc.binaryop.BinaryOperator.LESS,
        ),
        "left": (
            plc.binaryop.BinaryOperator.GREATER_EQUAL,
            plc.binaryop.BinaryOperator.LESS,
        ),
        "right": (
            plc.binaryop.BinaryOperator.GREATER,
            plc.binaryop.BinaryOperator.LESS_EQUAL,
        ),
        "both": (
            plc.binaryop.BinaryOperator.GREATER_EQUAL,
            plc.binaryop.BinaryOperator.LESS_EQUAL,
        ),
    }

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if self.name is BooleanFunction.Name.IsClose:
            return self._is_close(df, context=context)
        if self.name in (BooleanFunction.Name.IsEmpty, BooleanFunction.Name.HasNulls):
            (child,) = self.children
            column = child.evaluate(df, context=context)
            return Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(
                        column.size == 0
                        if self.name is BooleanFunction.Name.IsEmpty
                        else column.null_count > 0,
                        self.dtype.plc_type,
                        stream=df.stream,
                    ),
                    1,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        if self.name in (
            BooleanFunction.Name.IsFinite,
            BooleanFunction.Name.IsInfinite,
        ):
            # Avoid evaluating the child if the dtype tells us it's unnecessary.
            (child,) = self.children
            values = child.evaluate(df, context=context)
            is_float = values.obj.type().id() in (
                plc.TypeId.FLOAT32,
                plc.TypeId.FLOAT64,
            )
            is_finite = self.name is BooleanFunction.Name.IsFinite
            if not is_float:
                base = plc.Column.from_scalar(
                    plc.Scalar.from_py(py_val=is_finite, stream=df.stream),
                    values.size,
                    stream=df.stream,
                )
                out = base.with_mask(values.obj.null_mask(), values.null_count)
                return Column(out, dtype=self.dtype)
            to_search = [-float("inf"), float("inf")]
            if is_finite:
                # NaN is neither finite not infinite
                to_search.append(float("nan"))
            nonfinite_values = plc.Column.from_iterable_of_py(
                to_search,
                dtype=values.obj.type(),
                stream=df.stream,
            )
            result = plc.search.contains(nonfinite_values, values.obj, stream=df.stream)
            if is_finite:
                result = plc.unary.unary_operation(
                    result, plc.unary.UnaryOperator.NOT, stream=df.stream
                )
            return Column(
                result.with_mask(values.obj.null_mask(), values.null_count),
                dtype=self.dtype,
            )
        columns = [child.evaluate(df, context=context) for child in self.children]
        # Kleene logic for Any (OR) and All (AND) if ignore_nulls is
        # False
        if self.name in (BooleanFunction.Name.Any, BooleanFunction.Name.All):
            (ignore_nulls,) = self.options
            (column,) = columns
            is_any = self.name is BooleanFunction.Name.Any
            agg = plc.aggregation.any() if is_any else plc.aggregation.all()
            scalar_result = plc.reduce.reduce(
                column.obj, agg, self.dtype.plc_type, stream=df.stream
            )
            if not ignore_nulls and column.null_count > 0:
                #      Truth tables
                #     Any         All
                #   | F U T     | F U T
                # --+------   --+------
                # F | F U T   F | F F F
                # U | U U T   U | F U U
                # T | T T T   T | F U T
                #
                # If the input null count was non-zero, we must
                # post-process the result to insert the correct value.
                h_result = scalar_result.to_py(stream=df.stream)
                if (is_any and not h_result) or (not is_any and h_result):
                    # Any                     All
                    # False || Null => Null   True && Null => Null
                    return Column(
                        plc.Column.all_null_like(column.obj, 1, stream=df.stream),
                        dtype=self.dtype,
                    )
            return Column(
                plc.Column.from_scalar(scalar_result, 1, stream=df.stream),
                dtype=self.dtype,
            )
        if self.name is BooleanFunction.Name.IsNull:
            (column,) = columns
            return Column(
                plc.unary.is_null(column.obj, stream=df.stream), dtype=self.dtype
            )
        elif self.name is BooleanFunction.Name.IsNotNull:
            (column,) = columns
            return Column(
                plc.unary.is_valid(column.obj, stream=df.stream), dtype=self.dtype
            )
        elif self.name in (BooleanFunction.Name.IsNan, BooleanFunction.Name.IsNotNan):
            (column,) = columns
            is_float = column.obj.type().id() in (
                plc.TypeId.FLOAT32,
                plc.TypeId.FLOAT64,
            )
            if is_float:
                op = (
                    plc.unary.is_nan
                    if self.name is BooleanFunction.Name.IsNan
                    else plc.unary.is_not_nan
                )
                base = op(column.obj)
            else:
                base = plc.Column.from_scalar(
                    plc.Scalar.from_py(
                        py_val=self.name is not BooleanFunction.Name.IsNan,
                        stream=df.stream,
                    ),
                    column.size,
                    stream=df.stream,
                )
            out = base.with_mask(column.obj.null_mask(), column.null_count)
            return Column(out, dtype=self.dtype)
        elif self.name is BooleanFunction.Name.IsFirstDistinct:
            (column,) = columns
            return self._distinct(
                column,
                dtype=self.dtype,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_FIRST,
                source_value=plc.Scalar.from_py(
                    py_val=True, dtype=self.dtype.plc_type, stream=df.stream
                ),
                target_value=plc.Scalar.from_py(
                    py_val=False, dtype=self.dtype.plc_type, stream=df.stream
                ),
                stream=df.stream,
            )
        elif self.name is BooleanFunction.Name.IsLastDistinct:
            (column,) = columns
            return self._distinct(
                column,
                dtype=self.dtype,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_LAST,
                source_value=plc.Scalar.from_py(
                    py_val=True, dtype=self.dtype.plc_type, stream=df.stream
                ),
                target_value=plc.Scalar.from_py(
                    py_val=False,
                    dtype=self.dtype.plc_type,
                    stream=df.stream,
                ),
                stream=df.stream,
            )
        elif self.name is BooleanFunction.Name.IsUnique:
            (column,) = columns
            return self._distinct(
                column,
                dtype=self.dtype,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
                source_value=plc.Scalar.from_py(
                    py_val=True, dtype=self.dtype.plc_type, stream=df.stream
                ),
                target_value=plc.Scalar.from_py(
                    py_val=False, dtype=self.dtype.plc_type, stream=df.stream
                ),
                stream=df.stream,
            )
        elif self.name is BooleanFunction.Name.IsDuplicated:
            (column,) = columns
            return self._distinct(
                column,
                dtype=self.dtype,
                keep=plc.stream_compaction.DuplicateKeepOption.KEEP_NONE,
                source_value=plc.Scalar.from_py(
                    py_val=False, dtype=self.dtype.plc_type, stream=df.stream
                ),
                target_value=plc.Scalar.from_py(
                    py_val=True, dtype=self.dtype.plc_type, stream=df.stream
                ),
                stream=df.stream,
            )
        elif self.name is BooleanFunction.Name.AllHorizontal:
            return Column(
                reduce(
                    partial(
                        plc.binaryop.binary_operation,
                        op=plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
                        output_type=self.dtype.plc_type,
                    ),
                    (c.obj for c in columns),
                ),
                dtype=self.dtype,
            )
        elif self.name is BooleanFunction.Name.AnyHorizontal:
            return Column(
                reduce(
                    partial(
                        plc.binaryop.binary_operation,
                        op=plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
                        output_type=self.dtype.plc_type,
                    ),
                    (c.obj for c in columns),
                ),
                dtype=self.dtype,
            )
        elif self.name is BooleanFunction.Name.IsIn:
            needles, haystack = columns
            if haystack.obj.type().id() == plc.TypeId.LIST:
                # Unwrap values from the list column
                # .inner returns DataTypeClass | DataType, need to cast to DataType
                haystack = Column(
                    haystack.obj.children()[1],
                    dtype=DataType(
                        cast(
                            "pl.DataType",
                            cast("pl.List", haystack.dtype.polars_type).inner,
                        )
                    ),
                ).astype(needles.dtype, stream=df.stream)
            if haystack.size:
                return Column(
                    plc.search.contains(
                        haystack.obj,
                        needles.obj,
                        stream=df.stream,
                    ),
                    dtype=self.dtype,
                )
            return Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(py_val=False, stream=df.stream),
                    needles.size,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is BooleanFunction.Name.IsSorted:
            (column,) = columns
            (descending, nulls_last) = self.options
            order = (
                plc.types.Order.DESCENDING if descending else plc.types.Order.ASCENDING
            )
            null_order = (
                plc.types.NullOrder.AFTER if nulls_last else plc.types.NullOrder.BEFORE
            )
            bool_result: bool = column.check_sorted(
                order=order, null_order=null_order, stream=df.stream
            )
            return Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(py_val=bool_result, stream=df.stream),
                    1,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is BooleanFunction.Name.Not:
            (column,) = columns
            # Polars semantics:
            #   integer input: NOT => bitwise invert.
            #   boolean input: NOT => logical NOT.
            return Column(
                plc.unary.unary_operation(
                    column.obj,
                    plc.unary.UnaryOperator.NOT
                    if column.obj.type().id() == plc.TypeId.BOOL8
                    else plc.unary.UnaryOperator.BIT_INVERT,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        else:
            raise NotImplementedError(
                f"BooleanFunction {self.name}"
            )  # pragma: no cover; handled by init raising

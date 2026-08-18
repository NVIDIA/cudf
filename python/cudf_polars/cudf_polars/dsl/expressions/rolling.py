# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""Rolling and Window DSL nodes."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, ClassVar

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataFrame, DataType
from cudf_polars.dsl import expr
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.utils.reshape import broadcast
from cudf_polars.dsl.utils.windows import (
    duration_to_int,
    offsets_to_windows,
    range_window_bounds,
)
from cudf_polars.utils.versions import POLARS_VERSION_LT_136, POLARS_VERSION_LT_139

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.typing import ClosedInterval, Duration

__all__ = ["FixedSizeRollingWindow", "GroupedWindow", "RollingWindow", "to_request"]

_REPLACE_POLICY = {
    "forward": plc.replace.ReplacePolicy.PRECEDING,
    "backward": plc.replace.ReplacePolicy.FOLLOWING,
}


@dataclass(frozen=True)
class UnaryOp:
    named_exprs: list[expr.NamedExpr]
    order_index: plc.Column | None = None
    by_cols_for_scan: list[Column] | None = None
    local_grouper: plc.groupby.GroupBy | None = None


@dataclass(frozen=True)
class RankOp(UnaryOp):
    pass


@dataclass(frozen=True)
class FillNullWithStrategyOp(UnaryOp):
    policy: plc.replace.ReplacePolicy = plc.replace.ReplacePolicy.PRECEDING


@dataclass(frozen=True)
class CumSumOp(UnaryOp):
    pass


@dataclass(frozen=True)
class ShiftOp(UnaryOp):
    pass


@dataclass(frozen=True)
class FixedSizeRollingOp(UnaryOp):
    pass


@dataclass(frozen=True)
class RollingWindowOp(UnaryOp):
    pass


def to_request(
    value: expr.Expr,
    orderby: Column,
    df: DataFrame,
    order_index: plc.Column | None = None,
) -> plc.rolling.RollingRequest:
    """
    Produce a rolling request for evaluation with pylibcudf.

    Parameters
    ----------
    value
        The expression to perform the rolling aggregation on.
    orderby
        Orderby column, used as input to the request when the aggregation is Len.
    df
        DataFrame used to evaluate the inputs to the aggregation.
    order_index
        Optional row ordering to apply to the request column.
    """
    min_periods = 1
    if isinstance(value, expr.Len):
        # A count aggregation, we need a column so use the orderby column
        col_obj = orderby.obj
    elif isinstance(value, expr.Agg):
        child = value.children[0]
        col_obj = child.evaluate(df, context=ExecutionContext.ROLLING).obj
        if (POLARS_VERSION_LT_136 or not POLARS_VERSION_LT_139) and value.name == "var":
            # Polars variance produces null if nvalues <= ddof
            # libcudf produces NaN. However, we can get the polars
            # behaviour by setting the minimum window size to ddof +
            # 1.
            # In polars 1.36-1.38 this code path is not hit because
            # rolling goes through the IR path (not RollingWindow).
            # See https://github.com/pola-rs/polars/pull/25117
            min_periods = value.options + 1
    else:
        col_obj = value.evaluate(
            df, context=ExecutionContext.ROLLING
        ).obj  # pragma: no cover; raise before we get here because we
        # don't do correct handling of empty groups
    if order_index is not None:
        assert order_index.size() == df.num_rows
        assert col_obj.size() == df.num_rows
        col_obj = plc.copying.gather(
            plc.Table([col_obj]),
            order_index,
            plc.copying.OutOfBoundsPolicy.NULLIFY,
            stream=df.stream,
        ).columns()[0]
    return plc.rolling.RollingRequest(col_obj, min_periods, value.agg_request)


class RollingWindow(Expr):  # pragma: no cover; polars >1.36 uses AExpr::Rolling now
    __slots__ = (
        "closed_window",
        "following_ordinal",
        "offset",
        "orderby",
        "orderby_dtype",
        "period",
        "preceding_ordinal",
    )
    _non_child = (
        "dtype",
        "orderby_dtype",
        "offset",
        "period",
        "closed_window",
        "orderby",
    )

    def __init__(
        self,
        dtype: DataType,
        orderby_dtype: plc.DataType,
        offset: Duration,
        period: Duration,
        closed_window: ClosedInterval,
        orderby: str,
        agg: Expr,
    ) -> None:
        self.dtype = dtype
        self.orderby_dtype = orderby_dtype
        # NOTE: Save original `offset` and `period` args,
        # because the `preceding` and `following` attributes
        # cannot be serialized (and must be reconstructed
        # within `__init__`).
        self.offset = offset
        self.period = period
        self.orderby_dtype = orderby_dtype
        self.offset = offset
        self.period = period
        self.preceding_ordinal = duration_to_int(orderby_dtype, *offset)
        self.following_ordinal = duration_to_int(orderby_dtype, *period)
        self.closed_window = closed_window
        self.orderby = orderby
        self.children = (agg,)
        self.is_pointwise = False
        if agg.agg_request.kind() == plc.aggregation.Kind.COLLECT_LIST:
            raise NotImplementedError(
                "Incorrect handling of empty groups for list collection"
            )
        if (
            POLARS_VERSION_LT_136 or not POLARS_VERSION_LT_139
        ) and not plc.rolling.is_valid_rolling_aggregation(
            agg.dtype.plc_type, agg.agg_request
        ):
            raise NotImplementedError(f"Unsupported rolling aggregation {agg}")

    def do_evaluate(  # noqa: D102
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        if context != ExecutionContext.FRAME:
            raise RuntimeError(
                "Rolling aggregation inside groupby/over/rolling"
            )  # pragma: no cover; translation raises first
        (agg,) = self.children
        orderby = df.column_map[self.orderby]
        # Polars casts integral orderby to int64, but only for calculating window bounds
        if (
            plc.traits.is_integral(orderby.obj.type())
            and orderby.obj.type().id() != plc.TypeId.INT64
        ):
            orderby_obj = plc.unary.cast(
                orderby.obj, plc.DataType(plc.TypeId.INT64), stream=df.stream
            )
        else:
            orderby_obj = orderby.obj
        preceding_scalar, following_scalar = offsets_to_windows(
            self.orderby_dtype,
            self.preceding_ordinal,
            self.following_ordinal,
            stream=df.stream,
        )
        preceding, following = range_window_bounds(
            preceding_scalar, following_scalar, self.closed_window
        )
        if orderby.obj.null_count() != 0:
            raise RuntimeError(
                f"Index column '{self.orderby}' in rolling may not contain nulls"
            )
        if not orderby.check_sorted(
            order=plc.types.Order.ASCENDING,
            null_order=plc.types.NullOrder.BEFORE,
            stream=df.stream,
        ):
            raise RuntimeError(
                f"Index column '{self.orderby}' in rolling is not sorted, please sort first"
            )
        (result,) = plc.rolling.grouped_range_rolling_window(
            plc.Table([]),
            orderby_obj,
            plc.types.Order.ASCENDING,
            plc.types.NullOrder.BEFORE,
            preceding,
            following,
            [to_request(agg, orderby, df)],
            stream=df.stream,
        ).columns()
        return Column(result, dtype=self.dtype)


class FixedSizeRollingWindow(Expr):
    """
    Fixed-size integer-based rolling window aggregation.

    Handles expressions like ``pl.col("x").rolling_sum(window_size=3)``.
    Uses ``pylibcudf.rolling.rolling_window`` with integer preceding
    and following window sizes.
    """

    __slots__ = (
        "_agg_request",
        "agg_name",
        "fn_params",
        "following",
        "min_periods",
        "preceding",
    )
    _non_child = (
        "dtype",
        "agg_name",
        "preceding",
        "following",
        "min_periods",
        "fn_params",
    )

    _aggregations: ClassVar[dict[str, Callable[..., plc.aggregation.Aggregation]]] = {
        "sum": plc.aggregation.sum,
        "min": plc.aggregation.min,
        "max": plc.aggregation.max,
        "mean": plc.aggregation.mean,
        "var": plc.aggregation.variance,
        "std": plc.aggregation.std,
    }

    def __init__(
        self,
        dtype: DataType,
        agg_name: str,
        preceding: int,
        following: int,
        min_periods: int,
        fn_params: tuple[Any, ...],
        child: Expr,
    ) -> None:
        self.dtype = dtype
        self.agg_name = agg_name
        self.preceding = preceding
        self.following = following
        self.min_periods = min_periods
        self.fn_params = fn_params
        self.children = (child,)
        self.is_pointwise = False
        self._agg_request = self._make_agg_request()
        if not plc.rolling.is_valid_rolling_aggregation(
            child.dtype.plc_type, self._agg_request
        ):
            raise NotImplementedError(
                f"Unsupported fixed-size rolling aggregation {agg_name}"
            )

    def _make_agg_request(self) -> plc.aggregation.Aggregation:
        agg_fn = self._aggregations.get(self.agg_name)
        if agg_fn is None:
            raise NotImplementedError(
                f"Unsupported fixed-size rolling aggregation: {self.agg_name}"
            )  # pragma: no cover; translation validates aggregation names
        return agg_fn(*self.fn_params)

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if context != ExecutionContext.FRAME:
            raise RuntimeError(
                "Rolling aggregation inside groupby/over/rolling"
            )  # pragma: no cover; translation raises first
        (child,) = self.children
        col = child.evaluate(df, context=context)

        result = plc.rolling.rolling_window(
            col.obj,
            self.preceding,
            self.following,
            self.min_periods,
            self._agg_request,
            stream=df.stream,
        )
        if result.type() != self.dtype.plc_type:
            result = plc.unary.cast(result, self.dtype.plc_type, stream=df.stream)

        return Column(result, dtype=self.dtype)


class GroupedWindow(Expr):
    """
    Compute a window ``.over(...)`` aggregation and broadcast to rows.

    Notes
    -----
    - This expression node implements **grouped window mapping**
      (aggregate once per group, then broadcast back), plus scan-like
      window operations that require row ordering within each group.
    """

    __slots__ = (
        "_order_by_expr",
        "by_count",
        "named_aggs",
        "options",
        "post",
    )
    _non_child = (
        "dtype",
        "options",
        "named_aggs",
        "post",
        "by_count",
    )

    def __init__(
        self,
        dtype: DataType,
        options: Any,
        named_aggs: Sequence[expr.NamedExpr],
        post: expr.NamedExpr,
        by_count: int,
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.named_aggs = tuple(named_aggs)
        self.post = post
        self.by_count = by_count
        has_order_by = self.options[1]
        self.is_pointwise = False
        self.children = tuple(children)
        self._order_by_expr = children[by_count] if has_order_by else None

        unsupported = [
            type(named_expr.value).__name__
            for named_expr in self.named_aggs
            if not (
                isinstance(
                    named_expr.value,
                    (expr.Len, expr.Agg, FixedSizeRollingWindow, RollingWindow),
                )
                or (
                    isinstance(named_expr.value, expr.UnaryFunction)
                    and named_expr.value.name
                    in {
                        "rank",
                        "fill_null_with_strategy",
                        "cum_sum",
                        "diff",
                        "shift",
                        "shift_and_fill",
                    }
                )
            )
        ]
        if unsupported:
            kinds = ", ".join(sorted(set(unsupported)))
            raise NotImplementedError(
                f"Unsupported over(...) only expression: {kinds}="
            )
        if has_order_by:
            ob = self._order_by_expr
            is_multi_order_by = (
                isinstance(ob, expr.UnaryFunction)
                and ob.name == "as_struct"
                and len(ob.children) > 1
            )
            has_order_sensitive_agg = any(
                isinstance(ne.value, expr.Agg)
                and ne.value.agg_request.kind() == plc.aggregation.Kind.NTH_ELEMENT
                for ne in self.named_aggs
            )
            if is_multi_order_by and has_order_sensitive_agg:
                raise NotImplementedError(
                    "Multiple order_by keys with order-sensitive aggregations"
                )

    @staticmethod
    def _sorted_grouper(by_cols_for_scan: list[Column]) -> plc.groupby.GroupBy:
        return plc.groupby.GroupBy(
            plc.Table([c.obj for c in by_cols_for_scan]),
            null_handling=plc.types.NullPolicy.INCLUDE,
            keys_are_sorted=plc.types.Sorted.YES,
            column_order=[k.order for k in by_cols_for_scan],
            null_precedence=[k.null_order for k in by_cols_for_scan],
        )

    @singledispatchmethod
    def _apply_unary_op(
        self,
        op: UnaryOp,
        _: DataFrame,
        __: plc.groupby.GroupBy,
    ) -> tuple[list[str], list[DataType], list[plc.Table]]:
        raise NotImplementedError(
            f"Unsupported unary op: {type(op).__name__}"
        )  # pragma: no cover; translation raises first

    @_apply_unary_op.register
    def _(
        self,
        op: RankOp,
        df: DataFrame,
        grouper: plc.groupby.GroupBy,
    ) -> tuple[list[str], list[DataType], list[plc.Table]]:
        rank_named = op.named_exprs
        order_index = op.order_index
        by_cols_for_scan = op.by_cols_for_scan

        rank_requests: list[plc.groupby.GroupByRequest] = []
        rank_out_names: list[str] = []
        rank_out_dtypes: list[DataType] = []

        for ne in rank_named:
            rank_expr = ne.value
            (child_expr,) = rank_expr.children
            val_col = child_expr.evaluate(df, context=ExecutionContext.FRAME).obj
            if order_index is not None:
                val_col = plc.copying.gather(
                    plc.Table([val_col]),
                    order_index,
                    plc.copying.OutOfBoundsPolicy.NULLIFY,
                    stream=df.stream,
                ).columns()[0]
            assert isinstance(rank_expr, expr.UnaryFunction)
            method_str, descending, _ = rank_expr.options

            rank_method = {
                "average": plc.aggregation.RankMethod.AVERAGE,
                "min": plc.aggregation.RankMethod.MIN,
                "max": plc.aggregation.RankMethod.MAX,
                "dense": plc.aggregation.RankMethod.DENSE,
                "ordinal": plc.aggregation.RankMethod.FIRST,
            }[method_str]

            order = (
                plc.types.Order.DESCENDING if descending else plc.types.Order.ASCENDING
            )
            # Polars semantics: exclude nulls from domain; nulls get null ranks.
            null_precedence = (
                plc.types.NullOrder.BEFORE if descending else plc.types.NullOrder.AFTER
            )
            agg = plc.aggregation.rank(
                rank_method,
                column_order=order,
                null_handling=plc.types.NullPolicy.EXCLUDE,
                null_precedence=null_precedence,
                percentage=plc.aggregation.RankPercentage.NONE,
            )

            rank_requests.append(plc.groupby.GroupByRequest(val_col, [agg]))
            rank_out_names.append(ne.name)
            rank_out_dtypes.append(rank_expr.dtype)

        if order_index is not None and by_cols_for_scan is not None:
            # order_by expressions require us order each group
            lg = op.local_grouper
            assert isinstance(lg, plc.groupby.GroupBy)
            _, rank_tables = lg.scan(rank_requests)
        else:
            _, rank_tables = grouper.scan(rank_requests)
        return rank_out_names, rank_out_dtypes, rank_tables

    @_apply_unary_op.register
    def _(  # type: ignore[no-untyped-def]
        self,
        op: FillNullWithStrategyOp,
        df: DataFrame,
        _,
    ) -> tuple[list[str], list[DataType], list[plc.Table]]:
        named_exprs = op.named_exprs

        plc_cols = [
            ne.value.children[0].evaluate(df, context=ExecutionContext.FRAME).obj
            for ne in named_exprs
        ]
        if op.order_index is not None:
            vals_tbl = plc.copying.gather(
                plc.Table(plc_cols),
                op.order_index,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
                stream=df.stream,
            )
        else:
            vals_tbl = plc.Table(plc_cols)
        local_grouper = op.local_grouper
        assert isinstance(local_grouper, plc.groupby.GroupBy)
        _, filled_tbl = local_grouper.replace_nulls(
            vals_tbl,
            [op.policy] * len(plc_cols),
        )

        tables = [plc.Table([column]) for column in filled_tbl.columns()]
        names = [ne.name for ne in named_exprs]
        dtypes = [ne.value.dtype for ne in named_exprs]
        return names, dtypes, tables

    @_apply_unary_op.register
    def _(  # type: ignore[no-untyped-def]
        self,
        op: CumSumOp,
        df: DataFrame,
        _,
    ) -> tuple[list[str], list[DataType], list[plc.Table]]:
        cum_named = op.named_exprs
        order_index = op.order_index

        data_exprs: list[expr.Expr] = []
        fill_policies: list[plc.replace.ReplacePolicy | None] = []
        for ne in cum_named:
            v = ne.value
            if (
                isinstance(v, expr.UnaryFunction)
                and v.name == "fill_null_with_strategy"
            ):
                data_exprs.append(v.children[0].children[0])
                fill_policies.append(_REPLACE_POLICY[v.options[0]])
            else:
                data_exprs.append(v.children[0])
                fill_policies.append(None)

        # Instead of calling self._gather_columns, let's call plc.copying.gather directly
        # since we need plc.Column objects, not cudf_polars Column objects
        val_cols: Sequence[plc.Column]
        if order_index is not None:
            plc_cols = [
                e.evaluate(df, context=ExecutionContext.FRAME).obj for e in data_exprs
            ]
            val_cols = plc.copying.gather(
                plc.Table(plc_cols),
                order_index,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
                stream=df.stream,
            ).columns()
        else:
            val_cols = [
                e.evaluate(df, context=ExecutionContext.FRAME).obj for e in data_exprs
            ]
        agg = plc.aggregation.sum()

        requests: list[plc.groupby.GroupByRequest] = []
        out_names: list[str] = []
        out_dtypes: list[DataType] = []
        for ne, val_col in zip(cum_named, val_cols, strict=True):
            requests.append(plc.groupby.GroupByRequest(val_col, [agg]))
            out_names.append(ne.name)
            out_dtypes.append(ne.value.dtype)

        local_grouper = op.local_grouper
        assert isinstance(local_grouper, plc.groupby.GroupBy)
        _, tables = local_grouper.scan(requests)

        result_tables: list[plc.Table] = []
        for tbl, policy in zip(tables, fill_policies, strict=True):
            if policy is None:
                result_tables.append(tbl)
            else:
                _, filled = local_grouper.replace_nulls(tbl, [policy])
                result_tables.append(filled)
        return out_names, out_dtypes, result_tables

    @_apply_unary_op.register
    def _(
        self,
        op: ShiftOp,
        df: DataFrame,
        _: plc.groupby.GroupBy,
    ) -> tuple[list[str], list[DataType], list[plc.Table]]:
        plc_cols: list[plc.Column] = []
        offsets: list[int] = []
        fill_scalars: list[plc.Scalar] = []
        out_names: list[str] = []
        out_dtypes: list[DataType] = []

        for ne in op.named_exprs:
            shift_expr = ne.value
            assert isinstance(shift_expr, expr.UnaryFunction)
            data_expr, offset_expr = shift_expr.children[:2]
            assert isinstance(offset_expr, expr.Literal)
            offset = offset_expr.value
            assert isinstance(offset, int)

            plc_col = data_expr.evaluate(df, context=ExecutionContext.FRAME).obj
            plc_cols.append(plc_col)
            offsets.append(offset)
            out_names.append(ne.name)
            out_dtypes.append(shift_expr.dtype)
            if shift_expr.name in {"diff", "shift"}:
                fill_scalars.append(
                    plc.Scalar.from_py(None, plc_col.type(), stream=df.stream)
                )
            else:
                assert shift_expr.name == "shift_and_fill"
                fill_expr = shift_expr.children[2]
                assert isinstance(fill_expr, expr.Literal)
                fill_scalars.append(
                    plc.Scalar.from_py(
                        fill_expr.value, plc_col.type(), stream=df.stream
                    )
                )

        assert op.order_index is not None
        val_cols = plc.copying.gather(
            plc.Table(plc_cols),
            op.order_index,
            plc.copying.OutOfBoundsPolicy.NULLIFY,
            stream=df.stream,
        ).columns()

        assert isinstance(op.local_grouper, plc.groupby.GroupBy)
        shifted_tbl = op.local_grouper.shift(
            plc.Table(val_cols), offsets, fill_scalars, stream=df.stream
        )[1]
        result_tables: list[plc.Table] = []
        for val_col, shifted_col, ne in zip(
            val_cols, shifted_tbl.columns(), op.named_exprs, strict=True
        ):
            shift_expr = ne.value
            assert isinstance(shift_expr, expr.UnaryFunction)
            if shift_expr.name == "diff":
                result_tables.append(
                    plc.Table(
                        [
                            plc.binaryop.binary_operation(
                                val_col,
                                shifted_col,
                                plc.binaryop.BinaryOperator.SUB,
                                shift_expr.dtype.plc_type,
                                stream=df.stream,
                            )
                        ]
                    )
                )
            else:
                result_tables.append(plc.Table([shifted_col]))
        return out_names, out_dtypes, result_tables

    @_apply_unary_op.register
    def _(
        self,
        op: FixedSizeRollingOp,
        df: DataFrame,
        _: plc.groupby.GroupBy,
    ) -> tuple[list[str], list[DataType], list[plc.Table]]:
        assert op.order_index is not None
        assert op.by_cols_for_scan is not None

        names: list[str] = []
        dtypes: list[DataType] = []
        tables: list[plc.Table] = []
        dense_index = plc.filling.sequence(
            df.num_rows,
            plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=df.stream),
            plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=df.stream),
            stream=df.stream,
        )
        group_keys = plc.Table([c.obj for c in op.by_cols_for_scan])

        def get_window_key(ne: expr.NamedExpr) -> tuple[int, int]:
            rolling_expr = ne.value
            assert isinstance(rolling_expr, FixedSizeRollingWindow)
            return rolling_expr.preceding, rolling_expr.following

        window_counts = Counter(get_window_key(ne) for ne in op.named_exprs)
        window_bounds_cache: dict[tuple[int, int], tuple[plc.Column, plc.Column]] = {}

        for ne in op.named_exprs:
            rolling_expr = ne.value
            assert isinstance(rolling_expr, FixedSizeRollingWindow)
            (data_expr,) = rolling_expr.children
            val_col = data_expr.evaluate(df, context=ExecutionContext.FRAME).obj
            val_col = plc.copying.gather(
                plc.Table([val_col]),
                op.order_index,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
                stream=df.stream,
            ).columns()[0]

            window_key = get_window_key(ne)
            bounds: tuple[plc.Column, plc.Column] | None = window_bounds_cache.get(
                window_key
            )
            if bounds is None:
                bounds = plc.rolling.make_range_windows(
                    group_keys,
                    dense_index,
                    plc.types.Order.ASCENDING,
                    plc.types.NullOrder.BEFORE,
                    plc.rolling.BoundedOpen(
                        plc.Scalar.from_py(
                            rolling_expr.preceding,
                            plc.types.SIZE_TYPE,
                            stream=df.stream,
                        )
                    ),
                    plc.rolling.BoundedClosed(
                        plc.Scalar.from_py(
                            rolling_expr.following,
                            plc.types.SIZE_TYPE,
                            stream=df.stream,
                        )
                    ),
                    stream=df.stream,
                )
                if window_counts[window_key] > 1:
                    window_bounds_cache[window_key] = bounds

            result = plc.rolling.rolling_window(
                val_col,
                bounds[0],
                bounds[1],
                rolling_expr.min_periods,
                rolling_expr._agg_request,
                stream=df.stream,
            )
            if result.type() != rolling_expr.dtype.plc_type:
                result = plc.unary.cast(
                    result, rolling_expr.dtype.plc_type, stream=df.stream
                )

            names.append(ne.name)
            dtypes.append(rolling_expr.dtype)
            tables.append(plc.Table([result]))

        return names, dtypes, tables

    @_apply_unary_op.register
    def _(
        self,
        op: RollingWindowOp,
        df: DataFrame,
        _: plc.groupby.GroupBy,
    ) -> tuple[list[str], list[DataType], list[plc.Table]]:
        assert op.order_index is not None
        assert op.by_cols_for_scan is not None

        orderby_name = self._rolling_orderby_name(op.named_exprs)
        orderby = df.column_map[orderby_name]
        if orderby.obj.null_count() != 0:
            raise RuntimeError(
                f"Index column '{orderby_name}' in rolling may not contain nulls"
            )
        if plc.traits.is_integral(orderby.obj.type()):
            orderby = orderby.astype(DataType(pl.Int64()), stream=df.stream)
        sorted_orderby_obj = plc.copying.gather(
            plc.Table([orderby.obj]),
            op.order_index,
            plc.copying.OutOfBoundsPolicy.NULLIFY,
            stream=df.stream,
        ).columns()[0]

        request_groups: dict[tuple[int, int, Any], list[expr.NamedExpr]] = defaultdict(
            list
        )
        for ne in op.named_exprs:
            rolling_expr = ne.value
            assert isinstance(rolling_expr, RollingWindow)
            request_groups[
                (
                    rolling_expr.preceding_ordinal,
                    rolling_expr.following_ordinal,
                    rolling_expr.closed_window,
                )
            ].append(ne)

        group_keys = plc.Table([c.obj for c in op.by_cols_for_scan])
        results_by_name: dict[str, plc.Table] = {}
        for window_exprs in request_groups.values():
            sample = window_exprs[0].value
            assert isinstance(sample, RollingWindow)
            preceding_scalar, following_scalar = offsets_to_windows(
                sample.orderby_dtype,
                sample.preceding_ordinal,
                sample.following_ordinal,
                stream=df.stream,
            )
            preceding, following = range_window_bounds(
                preceding_scalar, following_scalar, sample.closed_window
            )
            requests: list[plc.rolling.RollingRequest] = []
            for ne in window_exprs:
                rolling_expr = ne.value
                assert isinstance(rolling_expr, RollingWindow)
                requests.append(
                    to_request(
                        rolling_expr.children[0],
                        orderby,
                        df,
                        order_index=op.order_index,
                    )
                )

            result_cols = plc.rolling.grouped_range_rolling_window(
                group_keys,
                sorted_orderby_obj,
                plc.types.Order.ASCENDING,
                plc.types.NullOrder.BEFORE,
                preceding,
                following,
                requests,
                stream=df.stream,
            ).columns()

            for ne, result_col in zip(window_exprs, result_cols, strict=True):
                rolling_expr = ne.value
                assert isinstance(rolling_expr, RollingWindow)
                output_col = result_col
                if result_col.type() != rolling_expr.dtype.plc_type:
                    output_col = plc.unary.cast(
                        result_col, rolling_expr.dtype.plc_type, stream=df.stream
                    )
                results_by_name[ne.name] = plc.Table([output_col])

        names = [ne.name for ne in op.named_exprs]
        dtypes = [ne.value.dtype for ne in op.named_exprs]
        tables = [results_by_name[name] for name in names]
        return names, dtypes, tables

    def _reorder_to_input(
        self,
        row_id: plc.Column,
        by_cols: list[Column],
        n_rows: int,
        rank_tables: list[plc.Table],
        rank_out_names: list[str],
        rank_out_dtypes: list[DataType],
        *,
        order_index: plc.Column | None = None,
        stream: Stream,
    ) -> list[Column]:
        # Reorder scan results from grouped-order back to input row order
        if order_index is None:
            key_orders = [k.order for k in by_cols]
            key_nulls = [k.null_order for k in by_cols]
            order_index = plc.sorting.stable_sorted_order(
                plc.Table([*(c.obj for c in by_cols), row_id]),
                [*key_orders, plc.types.Order.ASCENDING],
                [*key_nulls, plc.types.NullOrder.AFTER],
                stream=stream,
            )

        return [
            Column(
                plc.copying.scatter(
                    plc.Table([tbl.columns()[0]]),
                    order_index,
                    plc.Table(
                        [
                            plc.Column.from_scalar(
                                plc.Scalar.from_py(
                                    None, tbl.columns()[0].type(), stream=stream
                                ),
                                n_rows,
                                stream=stream,
                            )
                        ]
                    ),
                    stream=stream,
                ).columns()[0],
                name=name,
                dtype=dtype,
            )
            for name, dtype, tbl in zip(
                rank_out_names, rank_out_dtypes, rank_tables, strict=True
            )
        ]

    def _split_named_expr(
        self,
    ) -> tuple[list[expr.NamedExpr], dict[str, list[expr.NamedExpr]]]:
        """Split into reductions vs unary window operations."""
        reductions: list[expr.NamedExpr] = []
        unary_window_ops: dict[str, list[expr.NamedExpr]] = {
            "rank": [],
            "fill_null_with_strategy": [],
            "cum_sum": [],
            "shift": [],
            "fixed_size_rolling": [],
            "range_rolling": [],
        }

        for ne in self.named_aggs:
            v = ne.value
            if (
                isinstance(v, expr.UnaryFunction)
                and v.name == "fill_null_with_strategy"
                and isinstance(v.children[0], expr.UnaryFunction)
                and v.children[0].name == "cum_sum"
            ):
                unary_window_ops["cum_sum"].append(ne)
            elif isinstance(v, expr.UnaryFunction) and v.name in {
                "diff",
                "shift_and_fill",
            }:
                unary_window_ops["shift"].append(ne)
            elif isinstance(v, expr.UnaryFunction) and v.name in unary_window_ops:
                unary_window_ops[v.name].append(ne)
            elif isinstance(v, FixedSizeRollingWindow):
                unary_window_ops["fixed_size_rolling"].append(ne)
            elif isinstance(v, RollingWindow):
                unary_window_ops["range_rolling"].append(ne)
            else:
                reductions.append(ne)
        return reductions, unary_window_ops

    @staticmethod
    def _rolling_orderby_name(named_exprs: Sequence[expr.NamedExpr]) -> str:
        orderby_names: set[str] = set()
        for ne in named_exprs:
            rolling_expr = ne.value
            assert isinstance(rolling_expr, RollingWindow)
            orderby_names.add(rolling_expr.orderby)
        if len(orderby_names) != 1:
            raise NotImplementedError(
                "rolling(...).over(...) only supports one rolling index column"
            )
        return orderby_names.pop()

    def _build_window_order_index(
        self,
        by_cols: list[Column],
        *,
        row_id: plc.Column,
        order_by_col: Column | None,
        ob_desc: bool,
        ob_nulls_last: bool,
        value_col: plc.Column | None = None,
        value_desc: bool = False,
        reverse: bool = False,
        stream: Stream,
    ) -> plc.Column:
        """Compute a stable row ordering for unary operations in a grouped context."""
        cols: list[plc.Column] = [c.obj for c in by_cols]
        orders: list[plc.types.Order] = [k.order for k in by_cols]
        nulls: list[plc.types.NullOrder] = [k.null_order for k in by_cols]

        if value_col is not None:
            # for rank(...).over(...) the ranked ("sorted") order takes precedence over order_by
            cols.append(value_col)
            orders.append(
                plc.types.Order.DESCENDING if value_desc else plc.types.Order.ASCENDING
            )
            nulls.append(
                plc.types.NullOrder.BEFORE if value_desc else plc.types.NullOrder.AFTER
            )

        if order_by_col is not None:
            cols.append(order_by_col.obj)
            orders.append(
                plc.types.Order.DESCENDING
                if ob_desc ^ reverse
                else plc.types.Order.ASCENDING
            )
            nulls.append(
                plc.types.NullOrder.AFTER
                if ob_desc ^ ob_nulls_last
                else plc.types.NullOrder.BEFORE
            )

        # Use the row id to break ties
        cols.append(row_id)
        orders.append(
            plc.types.Order.DESCENDING if reverse else plc.types.Order.ASCENDING
        )
        nulls.append(plc.types.NullOrder.AFTER)

        return plc.sorting.stable_sorted_order(
            plc.Table(cols), orders, nulls, stream=stream
        )

    def _gather_columns(
        self, cols: Sequence[Column], order_index: plc.Column, stream: Stream
    ) -> list[Column]:
        gathered_tbl = plc.copying.gather(
            plc.Table([c.obj for c in cols]),
            order_index,
            plc.copying.OutOfBoundsPolicy.NULLIFY,
            stream=stream,
        )

        return [
            Column(
                gathered,
                name=c.name,
                dtype=c.dtype,
                order=c.order,
                null_order=c.null_order,
                is_sorted=c.is_sorted,
            )
            for gathered, c in zip(gathered_tbl.columns(), cols, strict=True)
        ]

    def _grouped_window_scan_setup(
        self,
        by_cols: list[Column],
        *,
        row_id: plc.Column,
        order_by_col: Column | None,
        ob_desc: bool,
        ob_nulls_last: bool,
        grouper: plc.groupby.GroupBy,
        reverse: bool = False,
        stream: Stream,
        require_sorted_groups: bool = False,
    ) -> tuple[plc.Column | None, list[Column] | None, plc.groupby.GroupBy]:
        if order_by_col is None and not (require_sorted_groups or reverse):
            # keep the original ordering
            return None, None, grouper
        order_index = self._build_window_order_index(
            by_cols,
            row_id=row_id,
            order_by_col=order_by_col,
            ob_desc=ob_desc,
            ob_nulls_last=ob_nulls_last,
            reverse=reverse,
            stream=stream,
        )
        by_cols_for_scan = self._gather_columns(by_cols, order_index, stream=stream)
        assert by_cols_for_scan is not None
        local = self._sorted_grouper(by_cols_for_scan)
        return order_index, by_cols_for_scan, local

    # TODO: this is an ordered left-join that drops the join keys.
    # Rename it and replace the manual scatter+gather with the Join IR's
    # _reorder_maps helper (lifted to a shared utility) so the streaming
    # and in-memory over paths share the same primitive.
    def _broadcast_agg_results(
        self,
        by_tbl: plc.Table,
        group_keys_tbl: plc.Table,
        value_tbl: plc.Table,
        names: Sequence[str],
        dtypes: Sequence[DataType],
        stream: Stream,
    ) -> list[Column]:
        # We do a left-join between the input keys to group-keys
        # so every input row appears exactly once. left_order is
        # returned un-ordered by libcudf.
        left_order, right_order = plc.join.left_join(
            by_tbl, group_keys_tbl, plc.types.NullEquality.EQUAL, stream
        )

        # Scatter the right order indices into an all-null table
        # and at the position of the index in left order. Now we
        # have the map between rows and groups with the correct ordering.
        left_rows = left_order.size()
        target = plc.Column.from_scalar(
            plc.Scalar.from_py(None, plc.types.SIZE_TYPE, stream), left_rows, stream
        )
        aligned_map = plc.copying.scatter(
            plc.Table([right_order]),
            left_order,
            plc.Table([target]),
            stream,
        ).columns()[0]

        # Broadcast each scalar aggregated result back to row-shape using
        # the aligned mapping between row indices and group indices.
        out_cols = value_tbl.columns()
        return [
            Column(
                plc.copying.gather(
                    plc.Table([col]),
                    aligned_map,
                    plc.copying.OutOfBoundsPolicy.NULLIFY,
                    stream,
                ).columns()[0],
                name=name,
                dtype=dtype,
            )
            for name, dtype, col in zip(names, dtypes, out_cols, strict=True)
        ]

    def _apply_ordered_unary_op(
        self,
        op: UnaryOp,
        df: DataFrame,
        grouper: plc.groupby.GroupBy,
        by_cols: list[Column],
        row_id: plc.Column,
        *,
        order_by_col: Column | None,
        ob_desc: bool = False,
        ob_nulls_last: bool = False,
        reverse: bool = False,
        require_sorted_groups: bool = False,
    ) -> list[Column]:
        order_index, by_cols_for_scan, local = self._grouped_window_scan_setup(
            by_cols,
            row_id=row_id,
            order_by_col=order_by_col,
            ob_desc=ob_desc,
            ob_nulls_last=ob_nulls_last,
            grouper=grouper,
            reverse=reverse,
            stream=df.stream,
            require_sorted_groups=require_sorted_groups,
        )
        names, dtypes, tables = self._apply_unary_op(
            replace(
                op,
                order_index=order_index,
                by_cols_for_scan=by_cols_for_scan,
                local_grouper=local,
            ),
            df,
            grouper,
        )
        return self._reorder_to_input(
            row_id,
            by_cols,
            df.num_rows,
            tables,
            names,
            dtypes,
            order_index=order_index,
            stream=df.stream,
        )

    def _build_groupby_requests(
        self,
        named_exprs: list[expr.NamedExpr],
        df: DataFrame,
        order_index: plc.Column | None = None,
        by_cols: list[Column] | None = None,
    ) -> tuple[list[plc.groupby.GroupByRequest], list[str], list[DataType]]:
        assert by_cols is not None
        gb_requests: list[plc.groupby.GroupByRequest] = []
        out_names: list[str] = []
        out_dtypes: list[DataType] = []

        eval_cols: list[plc.Column] = []
        val_nodes: list[tuple[expr.NamedExpr, expr.Agg]] = []

        for ne in named_exprs:
            val = ne.value
            out_names.append(ne.name)
            out_dtypes.append(val.dtype)
            if isinstance(val, expr.Agg):
                (child,) = (
                    val.children
                    if getattr(val, "name", None) != "quantile"
                    else (val.children[0],)
                )
                eval_cols.append(child.evaluate(df, context=ExecutionContext.FRAME).obj)
                val_nodes.append((ne, val))

        gathered_cols: Sequence[plc.Column] = eval_cols
        if order_index is not None and eval_cols:
            gathered_cols = plc.copying.gather(
                plc.Table(eval_cols),
                order_index,
                plc.copying.OutOfBoundsPolicy.NULLIFY,
                stream=df.stream,
            ).columns()

        gathered_iter = iter(gathered_cols)
        for ne in named_exprs:
            val = ne.value
            if isinstance(val, expr.Len):
                col = by_cols[0].obj
            else:
                col = next(gathered_iter)
            gb_requests.append(plc.groupby.GroupByRequest(col, [val.agg_request]))

        return gb_requests, out_names, out_dtypes

    def do_evaluate(  # noqa: D102
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        if context != ExecutionContext.FRAME:
            raise RuntimeError(
                "Window mapping (.over) can only be evaluated at the frame level"
            )  # pragma: no cover; translation raises first

        by_exprs = self.children[: self.by_count]
        order_by_expr = (
            self.children[self.by_count] if self._order_by_expr is not None else None
        )
        by_cols = broadcast(
            *(b.evaluate(df) for b in by_exprs),
            target_length=df.num_rows,
            stream=df.stream,
        )
        order_by_col = (
            broadcast(
                order_by_expr.evaluate(df), target_length=df.num_rows, stream=df.stream
            )[0]
            if order_by_expr is not None
            else None
        )

        by_tbl = plc.Table([c.obj for c in by_cols])

        sorted_flag = (
            plc.types.Sorted.YES
            if all(k.is_sorted for k in by_cols)
            else plc.types.Sorted.NO
        )
        grouper = plc.groupby.GroupBy(
            by_tbl,
            null_handling=plc.types.NullPolicy.INCLUDE,
            keys_are_sorted=sorted_flag,
            column_order=[k.order for k in by_cols],
            null_precedence=[k.null_order for k in by_cols],
        )

        scalar_named, unary_window_ops = self._split_named_expr()

        # Build GroupByRequests for scalar aggregations
        order_sensitive: list[expr.NamedExpr] = []
        other_scalars: list[expr.NamedExpr] = []
        for ne in scalar_named:
            val = ne.value
            if (
                self._order_by_expr is not None
                and isinstance(val, expr.Agg)
                and val.agg_request.kind() == plc.aggregation.Kind.NTH_ELEMENT
            ):
                order_sensitive.append(ne)
            else:
                other_scalars.append(ne)

        gb_requests, out_names, out_dtypes = self._build_groupby_requests(
            other_scalars, df, by_cols=by_cols
        )

        group_keys_tbl, value_tables = grouper.aggregate(gb_requests)
        broadcasted_cols = self._broadcast_agg_results(
            by_tbl,
            group_keys_tbl,
            plc.Table([col for t in value_tables for col in t.columns()]),
            out_names,
            out_dtypes,
            df.stream,
        )

        if order_sensitive:
            row_id = plc.filling.sequence(
                df.num_rows,
                plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=df.stream),
                plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=df.stream),
                stream=df.stream,
            )
            _, _, ob_desc, ob_nulls_last = self.options
            order_index, _, local = self._grouped_window_scan_setup(
                by_cols,
                row_id=row_id,
                order_by_col=order_by_col,
                ob_desc=ob_desc,
                ob_nulls_last=ob_nulls_last,
                grouper=grouper,
                stream=df.stream,
            )
            assert order_index is not None

            gb_requests, out_names, out_dtypes = self._build_groupby_requests(
                order_sensitive, df, order_index=order_index, by_cols=by_cols
            )

            group_keys_tbl_local, value_tables_local = local.aggregate(gb_requests)
            broadcasted_cols.extend(
                self._broadcast_agg_results(
                    by_tbl,
                    group_keys_tbl_local,
                    plc.Table([col for t in value_tables_local for col in t.columns()]),
                    out_names,
                    out_dtypes,
                    df.stream,
                )
            )

        row_id = plc.filling.sequence(
            df.num_rows,
            plc.Scalar.from_py(0, plc.types.SIZE_TYPE, stream=df.stream),
            plc.Scalar.from_py(1, plc.types.SIZE_TYPE, stream=df.stream),
            stream=df.stream,
        )
        if self._order_by_expr is not None:
            over_order_by_col, over_ob_desc, over_ob_nulls_last = (
                order_by_col,
                self.options[2],
                self.options[3],
            )
        else:
            over_order_by_col, over_ob_desc, over_ob_nulls_last = (None, False, False)

        if rank_named := unary_window_ops["rank"]:
            if self._order_by_expr is not None:
                _, _, ob_desc, ob_nulls_last = self.options
                for ne in rank_named:
                    rank_expr = ne.value
                    assert isinstance(rank_expr, expr.UnaryFunction)
                    (child,) = rank_expr.children
                    desc = rank_expr.options[1]

                    order_index = self._build_window_order_index(
                        by_cols,
                        row_id=row_id,
                        order_by_col=order_by_col,
                        ob_desc=ob_desc,
                        ob_nulls_last=ob_nulls_last,
                        value_col=child.evaluate(
                            df, context=ExecutionContext.FRAME
                        ).obj,
                        value_desc=desc,
                        stream=df.stream,
                    )
                    rank_by_cols_for_scan = self._gather_columns(
                        by_cols, order_index, stream=df.stream
                    )
                    local = GroupedWindow._sorted_grouper(rank_by_cols_for_scan)
                    names, dtypes, tables = self._apply_unary_op(
                        RankOp(
                            named_exprs=[ne],
                            order_index=order_index,
                            by_cols_for_scan=rank_by_cols_for_scan,
                            local_grouper=local,
                        ),
                        df,
                        grouper,
                    )
                    broadcasted_cols.extend(
                        self._reorder_to_input(
                            row_id,
                            by_cols,
                            df.num_rows,
                            tables,
                            names,
                            dtypes,
                            order_index=order_index,
                            stream=df.stream,
                        )
                    )
            else:
                names, dtypes, tables = self._apply_unary_op(
                    RankOp(
                        named_exprs=rank_named, order_index=None, by_cols_for_scan=None
                    ),
                    df,
                    grouper,
                )
                broadcasted_cols.extend(
                    self._reorder_to_input(
                        row_id,
                        by_cols,
                        df.num_rows,
                        tables,
                        names,
                        dtypes,
                        stream=df.stream,
                    )
                )

        if fill_named := unary_window_ops["fill_null_with_strategy"]:
            order_index, fill_null_by_cols_for_scan, local = (
                self._grouped_window_scan_setup(
                    by_cols,
                    row_id=row_id,
                    order_by_col=order_by_col
                    if self._order_by_expr is not None
                    else None,
                    ob_desc=self.options[2]
                    if self._order_by_expr is not None
                    else False,
                    ob_nulls_last=self.options[3]
                    if self._order_by_expr is not None
                    else False,
                    grouper=grouper,
                    stream=df.stream,
                )
            )

            strategy_exprs: dict[str, list[expr.NamedExpr]] = defaultdict(list)
            for ne in fill_named:
                fill_null_expr = ne.value
                assert isinstance(fill_null_expr, expr.UnaryFunction)
                strategy_exprs[fill_null_expr.options[0]].append(ne)

            for strategy, fill_exprs in strategy_exprs.items():
                names, dtypes, tables = self._apply_unary_op(
                    FillNullWithStrategyOp(
                        named_exprs=fill_exprs,
                        order_index=order_index,
                        by_cols_for_scan=fill_null_by_cols_for_scan,
                        local_grouper=local,
                        policy=_REPLACE_POLICY[strategy],
                    ),
                    df,
                    grouper,
                )
                broadcasted_cols.extend(
                    self._reorder_to_input(
                        row_id,
                        by_cols,
                        df.num_rows,
                        tables,
                        names,
                        dtypes,
                        order_index=order_index,
                        stream=df.stream,
                    )
                )

        if cum_named := unary_window_ops["cum_sum"]:
            cum_reverse = []
            for ne in cum_named:
                v = ne.value
                assert isinstance(v, expr.UnaryFunction)
                if v.name == "fill_null_with_strategy":
                    cum_sum_expr = v.children[0]
                    assert isinstance(cum_sum_expr, expr.UnaryFunction)
                    cum_reverse.append(bool(cum_sum_expr.options[0]))
                else:
                    cum_reverse.append(bool(v.options[0]))
            for is_reverse in (False, True):
                subset = [
                    ne
                    for ne, rev in zip(cum_named, cum_reverse, strict=True)
                    if rev is is_reverse
                ]
                if not subset:
                    continue
                # A fill_null_with_strategy fill runs on the scan output, which is
                # always in sorted-group order, so it needs a sorted grouper even when
                # there is no order_by.
                has_fill = any(
                    isinstance(ne.value, expr.UnaryFunction)
                    and ne.value.name == "fill_null_with_strategy"
                    for ne in subset
                )
                broadcasted_cols.extend(
                    self._apply_ordered_unary_op(
                        CumSumOp(named_exprs=subset),
                        df,
                        grouper,
                        by_cols,
                        row_id,
                        order_by_col=over_order_by_col,
                        ob_desc=over_ob_desc,
                        ob_nulls_last=over_ob_nulls_last,
                        reverse=is_reverse,
                        require_sorted_groups=has_fill,
                    )
                )

        if shift_named := unary_window_ops["shift"]:
            broadcasted_cols.extend(
                self._apply_ordered_unary_op(
                    ShiftOp(named_exprs=shift_named),
                    df,
                    grouper,
                    by_cols,
                    row_id,
                    order_by_col=over_order_by_col,
                    ob_desc=over_ob_desc,
                    ob_nulls_last=over_ob_nulls_last,
                    require_sorted_groups=True,
                )
            )

        if rolling_named := unary_window_ops["range_rolling"]:
            orderby_name = self._rolling_orderby_name(rolling_named)
            rolling_order_by_col = df.column_map[orderby_name]
            broadcasted_cols.extend(
                self._apply_ordered_unary_op(
                    RollingWindowOp(named_exprs=rolling_named),
                    df,
                    grouper,
                    by_cols,
                    row_id,
                    order_by_col=rolling_order_by_col,
                    require_sorted_groups=True,
                )
            )

        if fixed_rolling_named := unary_window_ops["fixed_size_rolling"]:
            broadcasted_cols.extend(
                self._apply_ordered_unary_op(
                    FixedSizeRollingOp(named_exprs=fixed_rolling_named),
                    df,
                    grouper,
                    by_cols,
                    row_id,
                    order_by_col=over_order_by_col,
                    ob_desc=over_ob_desc,
                    ob_nulls_last=over_ob_nulls_last,
                    require_sorted_groups=True,
                )
            )

        # Create a temporary DataFrame with the broadcasted columns named by their
        # placeholder names from agg decomposition, then evaluate the post-expression.
        df = DataFrame(broadcasted_cols, stream=df.stream)
        return self.post.value.evaluate(df, context=ExecutionContext.FRAME)

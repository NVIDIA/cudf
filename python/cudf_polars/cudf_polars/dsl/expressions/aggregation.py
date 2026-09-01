# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: remove need for this
# ruff: noqa: D101
"""DSL nodes for aggregations."""

from __future__ import annotations

import math
import sys
from decimal import Decimal
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

from polars.exceptions import ComputeError

import pylibcudf as plc

from cudf_polars.containers import Column
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr
from cudf_polars.dsl.expressions.literal import Literal
from cudf_polars.utils.versions import POLARS_VERSION_LT_136

if TYPE_CHECKING:
    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame, DataType

__all__ = ["Agg", "Item", "Kurtosis", "Skew", "SortedAgg"]


class Item(Expr):
    """Validate and return the result of an ``item`` aggregation."""

    __slots__ = ("allow_empty",)
    _non_child = ("dtype", "allow_empty")

    def __init__(
        self,
        dtype: DataType,
        allow_empty: bool,  # noqa: FBT001
        value: Expr,
        count: Expr,
    ) -> None:
        self.dtype = dtype
        self.allow_empty = allow_empty
        self.children = (value, count)
        self.is_pointwise = False

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate the validated ``item`` aggregation result."""
        value, count = (child.evaluate(df, context=context) for child in self.children)
        if count.size == 0:
            return value

        min_count_scalar, max_count_scalar = plc.reduce.minmax(
            count.obj, stream=df.stream
        )
        max_count = max_count_scalar.to_py(stream=df.stream)
        assert isinstance(max_count, int)
        if max_count > 1:
            qualifier = "no or a single value" if self.allow_empty else "a single value"
            raise ComputeError(
                f"aggregation 'item' expected {qualifier}, got {max_count} values"
            )
        if not self.allow_empty:
            min_count = min_count_scalar.to_py(stream=df.stream)
            assert isinstance(min_count, int)
            if min_count == 0:
                raise ComputeError(
                    "aggregation 'item' expected a single value, got none"
                )
        return value


class SortedAgg(Expr):
    """
    ``first``/``last`` aggregation ordered by one or more expressions.

    Notes
    -----
    This expression is used by GroupBy infrastructure for ordered
    first/last aggregations. The ordering may depend on columns other than
    the value being aggregated, so this cannot be evaluated independently
    nor can it be represented as a plain :class:`Agg` variant.
    """

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")

    def __init__(
        self,
        dtype: DataType,
        name: str,
        options: tuple[Any, ...],
        value: Expr,
        *by: Expr,
    ) -> None:
        self.dtype = dtype
        self.name = name
        stable, nulls_last, descending = options
        self.options = (stable, tuple(nulls_last), tuple(descending))
        self.children = (value, *by)
        self.is_pointwise = False
        if name not in {"first", "last"}:
            raise NotImplementedError(f"Sorted aggregation {name=}")
        if not by:
            raise NotImplementedError(
                "Sorted aggregation requires order-by expressions"
            )
        if len(self.options[1]) != len(by) or len(self.options[2]) != len(by):
            raise NotImplementedError(
                "Sorted aggregation requires one null/descending option per order key"
            )

    @property
    def agg_request(self) -> plc.aggregation.Aggregation:  # noqa: D102
        raise NotImplementedError(
            "Sorted aggregation cannot be represented as a pylibcudf "
            "aggregation request"
        )


class Agg(Expr):
    __slots__ = ("context", "name", "op", "options", "request")
    _non_child = ("dtype", "name", "options", "context")

    def __init__(
        self,
        dtype: DataType,
        name: str,
        options: Any,
        context: ExecutionContext,
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.name = name
        self.options = options
        self.is_pointwise = False
        self.children = children
        self.context = context
        if name not in Agg._SUPPORTED:
            raise NotImplementedError(
                f"Unsupported aggregation {name=}"
            )  # pragma: no cover; all valid aggs are supported
        # TODO: nan handling in groupby case
        if name == "min":
            req = plc.aggregation.min()
        elif name == "max":
            req = plc.aggregation.max()
        elif name == "median":
            (child,) = self.children
            if plc.traits.is_timestamp(child.dtype.plc_type):
                raise NotImplementedError("Median with temporal data types")
            req = plc.aggregation.median()
        elif name == "n_unique":
            # TODO: datatype of result
            req = plc.aggregation.nunique(
                null_handling=plc.types.NullPolicy.EXCLUDE
                if options
                else plc.types.NullPolicy.INCLUDE
            )
        elif name in {"first", "last", "item", "first_non_null"}:
            req = None
        elif name == "implode":
            req = plc.aggregation.collect_list(plc.types.NullPolicy.INCLUDE)
        elif name == "mean":
            req = plc.aggregation.mean()
        elif name == "sum":
            req = plc.aggregation.sum()
        elif name == "product":
            req = plc.aggregation.product()
        elif name == "std":
            # TODO: handle nans
            req = plc.aggregation.std(ddof=options)
        elif name == "var":
            # TODO: handle nans
            req = plc.aggregation.variance(ddof=options)
        elif name == "count":
            req = plc.aggregation.count(
                null_handling=plc.types.NullPolicy.EXCLUDE
                if not options
                else plc.types.NullPolicy.INCLUDE
            )
        elif name == "m2":  # pragma: no cover; doesn't have a direct polars equivalent
            req = plc.aggregation.m2()
        elif (
            name == "merge_m2"
        ):  # pragma: no cover; doesn't have a direct polars equivalent
            req = plc.aggregation.merge_m2()
        elif name == "quantile":
            child, quantile = self.children
            if not isinstance(quantile, Literal):
                raise NotImplementedError("Only support literal quantile values")
            if options == "equiprobable":
                raise NotImplementedError("Quantile with equiprobable interpolation")
            if plc.traits.is_duration(child.dtype.plc_type):
                raise NotImplementedError("Quantile with duration data type")
            if plc.traits.is_timestamp(child.dtype.plc_type):
                raise NotImplementedError("Quantile with temporal data types")
            req = plc.aggregation.quantile(
                quantiles=[quantile.value], interp=Agg.interp_mapping[options]
            )
        else:
            raise NotImplementedError(
                f"Unreachable, {name=} is incorrectly listed in _SUPPORTED"
            )  # pragma: no cover
        if (
            POLARS_VERSION_LT_136
            and context == ExecutionContext.FRAME
            and req is not None
            and not plc.aggregation.is_valid_aggregation(dtype.plc_type, req)
        ):  # pragma: no cover; polars may raise ahead of time
            # TODO: Check which cases polars raises vs returns all-NULL column.
            # For the all-NULL column cases, we could build it using Column.all_null_like
            # at evaluation time.
            raise NotImplementedError(f"Invalid aggregation {req} with dtype {dtype}")
        self.request = req
        op = getattr(self, f"_{name}", None)
        if op is None:
            assert req is not None  # Ensure req is not None for _reduce
            op = partial(self._reduce, request=req)
        elif name in {"min", "max"}:
            op = partial(op, propagate_nans=options)
        elif name == "count":
            op = partial(op, include_nulls=options)
        elif name in {
            "sum",
            "product",
            "first",
            "last",
            "item",
            "first_non_null",
            "implode",
        }:
            pass
        else:
            raise NotImplementedError(
                f"Unreachable, supported agg {name=} has no implementation"
            )  # pragma: no cover
        self.op = op

    _SUPPORTED: ClassVar[frozenset[str]] = frozenset(
        [
            "min",
            "max",
            "median",
            "n_unique",
            "first",
            "first_non_null",
            "item",
            "implode",
            "last",
            "mean",
            "m2",
            "merge_m2",
            "sum",
            "product",
            "count",
            "std",
            "var",
            "quantile",
        ]
    )

    interp_mapping: ClassVar[dict[str, plc.types.Interpolation]] = {
        "nearest": plc.types.Interpolation.NEAREST_HALF_UP,
        "higher": plc.types.Interpolation.HIGHER,
        "lower": plc.types.Interpolation.LOWER,
        "midpoint": plc.types.Interpolation.MIDPOINT,
        "linear": plc.types.Interpolation.LINEAR,
    }

    @property
    def agg_request(self) -> plc.aggregation.Aggregation:  # noqa: D102
        if self.name in {"first", "item"}:
            return plc.aggregation.nth_element(
                0, null_handling=plc.types.NullPolicy.INCLUDE
            )
        elif self.name == "first_non_null":
            return plc.aggregation.nth_element(
                0, null_handling=plc.types.NullPolicy.EXCLUDE
            )
        elif self.name == "last":
            return plc.aggregation.nth_element(
                -1, null_handling=plc.types.NullPolicy.INCLUDE
            )
        else:
            assert self.request is not None, "Init should have raised"
            return self.request

    def _reduce(
        self, column: Column, *, request: plc.aggregation.Aggregation, stream: Stream
    ) -> Column:
        if plc.traits.is_fixed_point(
            column.dtype.plc_type
        ) and self.dtype.plc_type.id() in {plc.TypeId.FLOAT32, plc.TypeId.FLOAT64}:
            column = column.astype(self.dtype, stream=stream)
        return Column(
            plc.Column.from_scalar(
                plc.reduce.reduce(
                    column.obj, request, self.dtype.plc_type, stream=stream
                ),
                1,
                stream=stream,
            ),
            name=column.name,
            dtype=self.dtype,
        )

    def _count(self, column: Column, *, include_nulls: bool, stream: Stream) -> Column:
        null_count = column.null_count if not include_nulls else 0
        return Column(
            plc.Column.from_scalar(
                plc.Scalar.from_py(
                    column.size - null_count, self.dtype.plc_type, stream=stream
                ),
                1,
                stream=stream,
            ),
            name=column.name,
            dtype=self.dtype,
        )

    def _sum(self, column: Column, stream: Stream) -> Column:
        if column.size == 0 or column.null_count == column.size:
            dtype = self.dtype.plc_type
            return Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(
                        Decimal(0).scaleb(dtype.scale())
                        if plc.traits.is_fixed_point(dtype)
                        else 0,
                        dtype,
                        stream=stream,
                    ),
                    1,
                    stream=stream,
                ),
                name=column.name,
                dtype=self.dtype,
            )
        return self._reduce(column, request=plc.aggregation.sum(), stream=stream)

    def _product(self, column: Column, stream: Stream) -> Column:
        if column.size == 0 or column.null_count == column.size:
            # The product of an empty or all-null column is 1 in polars.
            return Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(1, self.dtype.plc_type, stream=stream),
                    1,
                    stream=stream,
                ),
                name=column.name,
                dtype=self.dtype,
            )
        return self._reduce(column, request=plc.aggregation.product(), stream=stream)

    def _min(self, column: Column, *, propagate_nans: bool, stream: Stream) -> Column:
        nan_count = column.nan_count(stream=stream)
        if propagate_nans and nan_count > 0:
            return Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(
                        float("nan"), self.dtype.plc_type, stream=stream
                    ),
                    1,
                    stream=stream,
                ),
                name=column.name,
                dtype=self.dtype,
            )
        if nan_count > 0:
            column = column.mask_nans(stream=stream)
        return self._reduce(column, request=plc.aggregation.min(), stream=stream)

    def _max(self, column: Column, *, propagate_nans: bool, stream: Stream) -> Column:
        nan_count = column.nan_count(stream=stream)
        if propagate_nans and nan_count > 0:
            return Column(
                plc.Column.from_scalar(
                    plc.Scalar.from_py(
                        float("nan"), self.dtype.plc_type, stream=stream
                    ),
                    1,
                    stream=stream,
                ),
                name=column.name,
                dtype=self.dtype,
            )
        if nan_count > 0:
            column = column.mask_nans(stream=stream)
        return self._reduce(column, request=plc.aggregation.max(), stream=stream)

    def _first(self, column: Column, stream: Stream) -> Column:
        if column.size == 0:
            plc_result = plc.Column.all_null_like(column.obj, 1, stream=stream)
        else:
            plc_result = plc.copying.slice(column.obj, [0, 1], stream=stream)[0]
        return Column(
            plc_result,
            name=column.name,
            dtype=self.dtype,
        )

    def _last(self, column: Column, stream: Stream) -> Column:
        n = column.size
        if n == 0:
            plc_result = plc.Column.all_null_like(column.obj, 1, stream=stream)
        else:
            plc_result = plc.copying.slice(column.obj, [n - 1, n], stream=stream)[0]
        return Column(
            plc_result,
            name=column.name,
            dtype=self.dtype,
        )

    def _first_non_null(self, column: Column, stream: Stream) -> Column:
        if column.size == 0:
            plc_result = plc.Column.all_null_like(column.obj, 1, stream=stream)
        elif column.null_count == 0:
            plc_result = plc.copying.slice(column.obj, [0, 1], stream=stream)[0]
        else:
            return self._reduce(
                column,
                request=plc.aggregation.nth_element(
                    0, null_handling=plc.types.NullPolicy.EXCLUDE
                ),
                stream=stream,
            )
        return Column(
            plc_result,
            name=column.name,
            dtype=self.dtype,
        )

    def _implode(self, column: Column, stream: Stream) -> Column:
        size_type = plc.DataType(plc.TypeId.INT32)
        offsets = plc.filling.sequence(
            2,
            plc.Scalar.from_py(0, size_type, stream=stream),
            plc.Scalar.from_py(column.size, size_type, stream=stream),
            stream=stream,
        )
        return Column(
            plc.Column(
                self.dtype.plc_type,
                1,
                None,
                None,
                0,
                0,
                [offsets, column.obj],
            ),
            name=column.name,
            dtype=self.dtype,
        )

    def _item(self, column: Column, stream: Stream) -> Column:
        n = column.size
        if n == 0:
            if not self.options:
                raise ComputeError(
                    "aggregation 'item' expected a single value, got none"
                )
            plc_result = plc.Column.all_null_like(column.obj, 1, stream=stream)
        elif n == 1:
            plc_result = plc.copying.slice(column.obj, [0, 1], stream=stream)[0]
        else:
            qualifier = "no or a single value" if self.options else "a single value"
            raise ComputeError(
                f"aggregation 'item' expected {qualifier}, got {n} values"
            )
        return Column(
            plc_result,
            name=column.name,
            dtype=self.dtype,
        )

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        if context is not ExecutionContext.FRAME:
            raise NotImplementedError(
                f"Agg in context {context}"
            )  # pragma: no cover; unreachable

        # Aggregations like quantiles may have additional children that were
        # preprocessed into pylibcudf requests.
        child = self.children[0]
        return self.op(child.evaluate(df, context=context), stream=df.stream)


class Skew(Expr):
    """Sample skewness of a column."""

    __slots__ = ("bias",)
    _non_child = ("dtype", "bias")

    def __init__(self, dtype: DataType, bias: bool, child: Expr) -> None:  # noqa: FBT001
        self.dtype = dtype
        self.bias = bias
        self.children = (child,)
        self.is_pointwise = False

    @staticmethod
    def _central_moments(
        column: plc.Column, plc_type: plc.DataType, *, stream: Stream
    ) -> tuple[float, float, float]:
        """
        Compute the ``mean`` and central moments ``m2``, ``m3``.

        Nulls are excluded by the ``mean`` reductions.

        Notes
        -----
        This follows Polars' per-chunk moment accumulation
        (``SkewState::from_iter`` in ``polars-compute/src/moment.rs``), which
        also centers on the mean before summing. This numerically stable
        two-pass computation reduces the mean first, then reduces the centered
        powers ``mean((x - mean)**k)``.

        Could be done in a single pass with https://www.osti.gov/servlets/purl/1028931
        and in a single kernel with https://github.com/NVIDIA/cudf/pull/23621.
        """
        mean = plc.reduce.reduce(
            column, plc.aggregation.mean(), plc_type, stream=stream
        ).to_py(stream=stream)
        table = plc.Table([column])
        dev = plc.expressions.Operation(
            plc.expressions.ASTOperator.SUB,
            plc.expressions.ColumnReference(0),
            plc.expressions.Literal(plc.Scalar.from_py(mean, plc_type, stream=stream)),
        )
        mul = plc.expressions.ASTOperator.MUL
        dev2 = plc.expressions.Operation(mul, dev, dev)
        dev3 = plc.expressions.Operation(mul, dev2, dev)
        d2 = plc.transform.compute_column(table, dev2, stream=stream)
        d3 = plc.transform.compute_column(table, dev3, stream=stream)
        return (  # type: ignore[return-value]
            mean,
            plc.reduce.reduce(
                d2, plc.aggregation.mean(), plc_type, stream=stream
            ).to_py(stream=stream),
            plc.reduce.reduce(
                d3, plc.aggregation.mean(), plc_type, stream=stream
            ).to_py(stream=stream),
        )

    @staticmethod
    def _finalize(n: int, mean: float, m2: float, m3: float, *, bias: bool) -> float:
        """
        Compute the sample skewness from the mean and central moments.

        Notes
        -----
        This follows Polars' ``SkewState::finalize``
        (``polars-compute/src/moment.rs``): the biased Fisher-Pearson
        coefficient ``m3 / m2**1.5`` (returning NaN when the variance is
        effectively zero, matching Polars' ``m2 <= (eps * mean)**2`` check),
        with the sample bias correction ``sqrt(n * (n - 1)) / (n - 2)``
        applied when ``bias=False``.

        Overflow follows IEEE-754 (producing ``inf``/``nan``) matching Rust,
        rather than raising Python's ``OverflowError``.
        """
        eps = sys.float_info.epsilon
        is_zero = m2 <= (eps * mean) * (eps * mean)
        if is_zero:
            biased = math.nan
        else:
            try:
                denom = m2**1.5
            except OverflowError:
                denom = math.inf
            biased = m3 / denom
        if bias:
            return biased
        return math.sqrt(n * (n - 1)) / (n - 2) * biased

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context)
        n = column.size - column.null_count
        if n == 0 or (not self.bias and n <= 2):
            value: float | None = None
        else:
            casted = column.astype(self.dtype, stream=df.stream)
            mean, m2, m3 = self._central_moments(
                casted.obj, self.dtype.plc_type, stream=df.stream
            )
            value = self._finalize(n, mean, m2, m3, bias=self.bias)
        return Column(
            plc.Column.from_scalar(
                plc.Scalar.from_py(value, self.dtype.plc_type, stream=df.stream),
                1,
                stream=df.stream,
            ),
            dtype=self.dtype,
        )


class Kurtosis(Expr):
    """Kurtosis (Fisher or Pearson) of a column."""

    __slots__ = ("bias", "fisher")
    _non_child = ("dtype", "fisher", "bias")

    def __init__(
        self,
        dtype: DataType,
        fisher: bool,  # noqa: FBT001
        bias: bool,  # noqa: FBT001
        child: Expr,
    ) -> None:
        self.dtype = dtype
        self.fisher = fisher
        self.bias = bias
        self.children = (child,)
        self.is_pointwise = False

    @staticmethod
    def _central_moments(
        column: plc.Column, plc_type: plc.DataType, *, stream: Stream
    ) -> tuple[float, float, float]:
        """
        Compute the ``mean`` and central moments ``m2``, ``m4``.

        Nulls are excluded by the ``mean`` reductions.

        Notes
        -----
        This follows Polars' per-chunk moment accumulation
        (``KurtosisState::from_iter`` in ``polars-compute/src/moment.rs``),
        which also centers on the mean before summing. This numerically stable
        two-pass computation reduces the mean first, then reduces the centered
        powers ``mean((x - mean)**k)``.

        Could be done in a single pass with https://www.osti.gov/servlets/purl/1028931
        and in a single kernel with https://github.com/NVIDIA/cudf/pull/23621.
        """
        mean = plc.reduce.reduce(
            column, plc.aggregation.mean(), plc_type, stream=stream
        ).to_py(stream=stream)
        table = plc.Table([column])
        dev = plc.expressions.Operation(
            plc.expressions.ASTOperator.SUB,
            plc.expressions.ColumnReference(0),
            plc.expressions.Literal(plc.Scalar.from_py(mean, plc_type, stream=stream)),
        )
        mul = plc.expressions.ASTOperator.MUL
        dev2 = plc.expressions.Operation(mul, dev, dev)
        dev4 = plc.expressions.Operation(mul, dev2, dev2)
        d2 = plc.transform.compute_column(table, dev2, stream=stream)
        d4 = plc.transform.compute_column(table, dev4, stream=stream)
        return (  # type: ignore[return-value]
            mean,
            plc.reduce.reduce(
                d2, plc.aggregation.mean(), plc_type, stream=stream
            ).to_py(stream=stream),
            plc.reduce.reduce(
                d4, plc.aggregation.mean(), plc_type, stream=stream
            ).to_py(stream=stream),
        )

    @staticmethod
    def _finalize(
        n: int, mean: float, m2: float, m4: float, *, fisher: bool, bias: bool
    ) -> float:
        """
        Compute the kurtosis from the mean and central moments.

        Notes
        -----
        This follows Polars' ``KurtosisState::finalize``
        (``polars-compute/src/moment.rs``): the biased estimate
        ``m4 / m2**2`` (returning NaN when the variance is effectively zero,
        matching Polars' ``m2 <= (eps * mean)**2`` check), the k-statistic
        bias correction applied when ``bias=False``, and subtracting 3.0 for
        Fisher's definition.
        """
        eps = sys.float_info.epsilon
        is_zero = m2 <= (eps * mean) * (eps * mean)
        biased = math.nan if is_zero else m4 / (m2 * m2)
        if bias:
            out = biased
        else:
            nm1_nm2 = (n - 1) / (n - 2)
            np1_nm3 = (n + 1) / (n - 3)
            nm1_nm3 = (n - 1) / (n - 3)
            out = nm1_nm2 * (np1_nm3 * biased - 3.0 * nm1_nm3) + 3.0
        return out - 3.0 if fisher else out

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        (child,) = self.children
        column = child.evaluate(df, context=context)
        n = column.size - column.null_count
        if n == 0 or (not self.bias and n <= 3):
            value: float | None = None
        else:
            casted = column.astype(self.dtype, stream=df.stream)
            mean, m2, m4 = self._central_moments(
                casted.obj, self.dtype.plc_type, stream=df.stream
            )
            value = self._finalize(n, mean, m2, m4, fisher=self.fisher, bias=self.bias)
        return Column(
            plc.Column.from_scalar(
                plc.Scalar.from_py(value, self.dtype.plc_type, stream=df.stream),
                1,
                stream=df.stream,
            ),
            dtype=self.dtype,
        )

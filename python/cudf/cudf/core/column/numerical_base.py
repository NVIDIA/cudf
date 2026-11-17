# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Define an interface for columns that can perform numerical operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np

import pylibcudf as plc

import cudf
from cudf.core.column import column_empty
from cudf.core.column.column import (
    ColumnBase,
    PylibcudfFunction,
    pylibcudf_result_dtype_policy,
    pylibcudf_result_same_kind_dtype_policy,
    same_dtype_policy,
)
from cudf.core.missing import NA
from cudf.core.mixins import Scannable
from cudf.utils.dtypes import _get_nan_for_dtype

if TYPE_CHECKING:
    from cudf._typing import ScalarLike


_unaryop_map = {
    "ASIN": "ARCSIN",
    "ACOS": "ARCCOS",
    "ATAN": "ARCTAN",
    "INVERT": "BIT_INVERT",
}


class NumericalBaseColumn(ColumnBase, Scannable):
    """
    A column composed of numerical (bool, integer, float, decimal) data.

    This class encodes a standard interface for different types of columns
    containing numerical types of data. In particular, mathematical operations
    that make sense whether a column is integral or real, fixed or floating
    point, should be encoded here.
    """

    _VALID_REDUCTIONS = {
        "mean",
        "product",
        "std",
        "sum",
        "sum_of_squares",
        "var",
    }

    _VALID_SCANS = {
        "cumsum",
        "cumprod",
        "cummin",
        "cummax",
        "ewma",
    }

    def _can_return_nan(self, skipna: bool | None = None) -> bool:
        return not skipna and self.has_nulls()

    def _reduce(
        self,
        op: str,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> ScalarLike:
        """Override to handle var/std NA conversion."""
        result = super()._reduce(
            op, skipna=skipna, min_count=min_count, **kwargs
        )

        # Convert NA to NaN for var/std operations
        if op in {"var", "std"} and result is NA:
            return _get_nan_for_dtype(self.dtype)

        return result

    def kurtosis(self, skipna: bool = True) -> float:
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )

        if len(self) == 0 or self._can_return_nan(skipna=skipna):
            return _get_nan_for_dtype(self.dtype)

        original_dtype = self.dtype
        self = self.nans_to_nulls().dropna()

        if len(self) < 4:
            return _get_nan_for_dtype(original_dtype)

        if original_dtype.kind == "f" and original_dtype.itemsize < 8:
            # Compute in float64 to match pandas precision for narrower
            # floating point dtypes (e.g. float32).
            self = self.astype(np.dtype(np.float64))  # type: ignore[assignment]

        n = len(self)
        miu = self.mean()
        m4_numerator = ((self - miu) ** 4).sum()
        V = self.var()

        if V == 0:
            return original_dtype.type(0)

        term_one_section_one = (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        term_one_section_two = m4_numerator / (V**2)
        term_two = ((n - 1) ** 2) / ((n - 2) * (n - 3))
        kurt = term_one_section_one * term_one_section_two - 3 * term_two
        if original_dtype.kind == "f" and original_dtype.itemsize < 8:
            kurt = original_dtype.type(kurt)
        return kurt

    def skew(self, skipna: bool = True) -> ScalarLike:
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )

        if len(self) == 0 or self._can_return_nan(skipna=skipna):
            return _get_nan_for_dtype(self.dtype)

        self = self.nans_to_nulls().dropna()

        if len(self) < 3:
            return _get_nan_for_dtype(self.dtype)

        n = len(self)
        miu = self.mean()
        m3 = (((self - miu) ** 3).sum()) / n
        m2 = self.var(ddof=0)

        if m2 == 0:
            return np.float64(0)

        unbiased_coef = ((n * (n - 1)) ** 0.5) / (n - 2)
        skew = unbiased_coef * m3 / (m2 ** (3 / 2))
        return skew

    def quantile(
        self,
        q: np.ndarray,
        interpolation: str,
        exact: bool,
        return_scalar: bool,
    ) -> NumericalBaseColumn:
        if self.dtype.kind == "b":
            raise NotImplementedError(
                "quantile is not implemented for boolean dtype"
            )
        if np.logical_or(q < 0, q > 1).any():
            raise ValueError(
                "percentiles should all be in the interval [0, 1]"
            )
        # Beyond this point, q either being scalar or list-like
        # will only have values in range [0, 1]
        if len(self) == 0:
            result = cast(
                "cudf.core.column.numerical_base.NumericalBaseColumn",
                column_empty(row_count=len(q), dtype=self.dtype),
            )
        else:
            no_nans = self.nans_to_nulls()
            # get sorted indices and exclude nulls
            indices = (
                no_nans.argsort(ascending=True, na_position="first")
                .slice(no_nans.null_count, len(no_nans))
                .astype(np.dtype(np.int32))
            )
            interpolation_type = plc.types.Interpolation[interpolation.upper()]

            # For non-arithmetic interpolations the chosen value is an
            # actual element of the input - ask libcudf to preserve the
            # input dtype rather than casting through float64, which
            # would lose precision for int64 values exceeding 2**53
            # (e.g. nanosecond timestamps).
            plc_exact = exact and interpolation in {"linear", "midpoint"}

            result = cast(
                "cudf.core.column.numerical_base.NumericalBaseColumn",
                PylibcudfFunction(
                    plc.quantiles.quantile,
                    pylibcudf_result_same_kind_dtype_policy,
                ).execute_with_args(
                    no_nans, q, interpolation_type, indices, plc_exact
                ),
            )
        if return_scalar:
            scalar_result = result.element_indexing(0)
            if interpolation in {"lower", "higher", "nearest"}:
                try:
                    new_scalar = self.dtype.type(scalar_result)
                    scalar_result = (
                        new_scalar
                        if new_scalar == scalar_result
                        else scalar_result
                    )
                except (TypeError, ValueError):
                    pass
            return (
                _get_nan_for_dtype(self.dtype)
                if scalar_result is NA
                else scalar_result
            )
        return result

    def median(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> NumericalBaseColumn:
        if not isinstance(skipna, bool):
            raise ValueError(
                f"For argument 'skipna' expected type bool, got {type(skipna).__name__}."
            )

        if self._can_return_nan(skipna=skipna):
            return _get_nan_for_dtype(self.dtype)

        # enforce linear in case the default ever changes
        result = self.quantile(
            np.array([0.5]),
            interpolation="linear",
            exact=True,
            return_scalar=True,
        )
        if self.dtype.kind == "f":
            result = self.dtype.type(result)
        return result

    def cov(self, other: NumericalBaseColumn) -> float:
        if (
            len(self) == 0
            or len(other) == 0
            or (len(self) == 1 and len(other) == 1)
        ):
            return _get_nan_for_dtype(self.dtype)

        result = (self - self.mean()) * (other - other.mean())
        cov_sample = result.sum() / (len(self) - 1)
        return cov_sample

    def corr(self, other: NumericalBaseColumn) -> float:
        if len(self) == 0 or len(other) == 0:
            return _get_nan_for_dtype(self.dtype)

        cov = self.cov(other)
        lhs_std, rhs_std = self.std(), other.std()

        if not cov or lhs_std == 0 or rhs_std == 0:
            return _get_nan_for_dtype(self.dtype)
        return cov / lhs_std / rhs_std

    def round(
        self,
        decimals: int = 0,
        how: Literal["half_even", "half_up"] = "half_even",
    ) -> NumericalBaseColumn:
        if not cudf.api.types.is_integer(decimals):
            raise TypeError("Argument 'decimals' must an integer")
        if how not in {"half_even", "half_up"}:
            raise ValueError(f"{how=} must be either 'half_even' or 'half_up'")
        plc_how = plc.round.RoundingMethod[how.upper()]

        return cast(
            "cudf.core.column.numerical_base.NumericalBaseColumn",
            PylibcudfFunction(
                plc.round.round,
                pylibcudf_result_dtype_policy,
            ).execute_with_args(self, decimals, plc_how),
        )

    def unary_operator(self, unaryop: str) -> ColumnBase:
        unaryop_str = unaryop.upper()
        unaryop_str = _unaryop_map.get(unaryop_str, unaryop_str)
        unaryop_enum = plc.unary.UnaryOperator[unaryop_str]

        return PylibcudfFunction(
            plc.unary.unary_operation,
            same_dtype_policy,
        ).execute_with_args(self, unaryop_enum)

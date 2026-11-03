# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import datetime
import functools
from typing import TYPE_CHECKING, Any, ClassVar, cast

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.api.types import is_scalar
from cudf.core.column.column import ColumnBase, as_column, column_empty
from cudf.core.dtype.validators import (
    is_dtype_obj_datetime_tz,
    is_dtype_obj_string,
)
from cudf.core.mixins import Scannable
from cudf.errors import MixedTypeError
from cudf.utils.dtypes import (
    cudf_dtype_from_pa_type,
    cudf_dtype_to_pa_type,
    dtype_to_pylibcudf_type,
    find_common_type,
)
from cudf.utils.temporal import unit_to_nanoseconds_conversion
from cudf.utils.utils import is_na_like


def _raise_on_invalid_ordering_scalar(
    self_dtype: DtypeObj, other: Any, op: str
) -> None:
    """Raise TypeError for ordering ops against pandas-invalid scalars.

    pandas raises ``Invalid comparison between dtype=<dtype> and <type>``
    when a datetime64/timedelta64 array is compared (``<``, ``<=``,
    ``>``, ``>=``) against ``None``, a plain (non-typed) ``float`` NaN,
    or a ``datetime.date`` that is not a ``datetime.datetime``. Without
    this guard those scalars silently coerce to NaT in cudf and
    ``_binaryop`` would return an incorrect boolean column.
    """
    if op not in {"__lt__", "__le__", "__gt__", "__ge__"}:
        return
    if (
        other is None
        or (isinstance(other, float) and np.isnan(other))
        or (
            isinstance(other, datetime.date)
            and not isinstance(other, datetime.datetime)
        )
    ):
        raise TypeError(
            f"Invalid comparison between dtype={self_dtype} "
            f"and {type(other).__name__}"
        )


if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from cudf._typing import ColumnLike, DtypeObj, ScalarLike
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.string import StringColumn


class TemporalBaseColumn(ColumnBase, Scannable):
    """
    Base class for TimeDeltaColumn and DatetimeColumn.
    """

    _VALID_REDUCTIONS = {
        "mean",
        "std",
    }
    _VALID_SCANS = {
        "cummin",
        "cummax",
    }
    _PANDAS_NA_VALUE = pd.NaT
    _UNDERLYING_DTYPE: np.dtype[np.int64] = np.dtype(np.int64)
    _NP_SCALAR: ClassVar[type[np.datetime64] | type[np.timedelta64]]
    _PD_SCALAR: pd.Timestamp | pd.Timedelta

    def __contains__(self, item: np.datetime64 | np.timedelta64) -> bool:
        """
        Check if the column contains a given value.
        """
        return item.view(self._UNDERLYING_DTYPE) in self.astype(  # type:ignore[operator]
            self._UNDERLYING_DTYPE
        )

    def _validate_fillna_value(
        self, fill_value: ScalarLike | ColumnLike
    ) -> plc.Scalar | ColumnBase:
        """Align fill_value for .fillna based on column type."""
        if (
            isinstance(fill_value, self._NP_SCALAR)
            and self.time_unit != np.datetime_data(fill_value)[0]
        ):
            if isinstance(fill_value, self._NP_SCALAR):
                unit = np.datetime_data(fill_value)[0]
                if unit not in {"s", "ms", "us", "ns"}:
                    fill_value = fill_value.astype(
                        np.dtype(f"{fill_value.dtype.kind}8[ns]")
                    )
            fill_value = fill_value.astype(self.dtype)
        elif isinstance(fill_value, str) and fill_value.lower() == "nat":
            # call-overload must be ignored because numpy stubs only accept literal
            # time unit strings, but we're passing self.time_unit which is valid at runtime
            fill_value = self._NP_SCALAR(fill_value, self.time_unit)  # type: ignore[call-overload]
        elif is_scalar(fill_value) and not isinstance(
            fill_value, (self._NP_SCALAR, self._PD_SCALAR)
        ):
            raise MixedTypeError(
                f"Cannot use fill_value of type {type(fill_value)} with "
                f"TemporalBaseColumn of dtype {self.dtype}."
            )
        return super()._validate_fillna_value(fill_value)

    def _cast_setitem_value(self, value: Any) -> plc.Scalar | ColumnBase:
        if isinstance(value, (np.str_, self._NP_SCALAR)):
            value = self._PD_SCALAR(value.item())
        elif isinstance(value, str):
            value = self._PD_SCALAR(value)
        elif value is pd.NaT:
            value = None
        elif cudf.get_option("mode.pandas_compatible"):
            # pyarrow silently coerces int/float to epoch values; pandas
            # raises TypeError for these.
            value_is_scalar = is_scalar(value)
            if (
                value_is_scalar
                and not is_na_like(value)
                and not isinstance(value, (bool, np.bool_))
                and isinstance(value, (int, float, np.integer, np.floating))
            ):
                raise TypeError(
                    f"Invalid value '{value}' for dtype '{self.dtype}'"
                )
            if (
                not value_is_scalar
                and not isinstance(value, ColumnBase)
                and getattr(getattr(value, "dtype", None), "kind", None)
                in {"i", "u", "f", "b"}
            ):
                raise TypeError(
                    f"Invalid value '{value}' for dtype '{self.dtype}'"
                )
            if isinstance(value, ColumnBase) and value.dtype.kind in {
                "i",
                "u",
                "f",
                "b",
            }:
                raise TypeError(
                    f"Invalid value '{value}' for dtype '{self.dtype}'"
                )
        return super()._cast_setitem_value(value)

    def _process_values_for_isin(
        self, values: Sequence | ColumnBase
    ) -> tuple[ColumnBase, ColumnBase]:
        lhs, rhs = super()._process_values_for_isin(values)
        if isinstance(rhs, type(self)):
            rhs = rhs.astype(lhs.dtype)
        return lhs, rhs

    def _normalize_binop_operand(self, other: Any) -> pa.Scalar | ColumnBase:
        if isinstance(other, ColumnBase):
            return other
        elif isinstance(other, (cp.ndarray, np.ndarray)) and other.ndim == 0:
            other = other[()]

        if is_scalar(other):
            if is_na_like(other):
                if isinstance(
                    other, (np.datetime64, np.timedelta64)
                ) and np.isnat(other):
                    try:
                        arrow_type = cudf_dtype_to_pa_type(other.dtype)
                    except pa.ArrowNotImplementedError:
                        arrow_type = cudf_dtype_to_pa_type(
                            np.dtype(
                                f"{other.dtype.type.__name__}[{self.time_unit}]"
                            )
                        )
                else:
                    arrow_type = cudf_dtype_to_pa_type(self.dtype)
                return pa.scalar(None, type=arrow_type)
            elif self.dtype.kind == "M" and isinstance(other, pd.Timestamp):
                self_tz = getattr(self.dtype, "tz", None)
                if (other.tz is None) != (self_tz is None):
                    raise TypeError(
                        f"Invalid comparison between dtype={self.dtype} "
                        f"and {type(other).__name__}"
                    )
                if other.tz is None:
                    other = other.to_numpy()
                else:
                    return pa.scalar(other)
            elif self.dtype.kind == "M" and isinstance(other, str):
                try:
                    ts = pd.Timestamp(other)
                except ValueError:
                    return NotImplemented
                if is_dtype_obj_datetime_tz(self.dtype):
                    # pandas parses the string and interprets it in the
                    # column's tz for comparison, so localize here.
                    if isinstance(self.dtype, pd.DatetimeTZDtype):
                        tz = self.dtype.tz
                    else:
                        tz = cast("pd.ArrowDtype", self.dtype).pyarrow_dtype.tz
                    other = ts.tz_localize(tz)
                else:
                    other = ts
            elif self.dtype.kind == "m" and isinstance(other, pd.Timedelta):
                other = other.to_numpy()
            elif isinstance(other, (np.datetime64, np.timedelta64)):
                unit = np.datetime_data(other)[0]
                if unit not in {"s", "ms", "us", "ns"}:
                    if np.isnat(other):
                        # TODO: Use self.time_unit to not modify the result resolution?
                        to_unit = "ns"
                    else:
                        to_unit = self.time_unit
                    if np.isnat(other):
                        # Workaround for https://github.com/numpy/numpy/issues/28496
                        # Can always use the astype below once minimum numpy version is 2.4
                        # call-overload must be ignored because numpy stubs only accept literal strings
                        # for time units (e.g., "ns", "us") to allow compile-time validation,
                        # but we're passing a variable string (to_unit) with a time unit that
                        # we know is valid at runtime
                        other = type(other)("NaT", to_unit)  # type: ignore[call-overload]
                    else:
                        other = other.astype(
                            np.dtype(f"{other.dtype.kind}8[{to_unit}]")
                        )
            elif isinstance(other, datetime.date) and not isinstance(
                other, datetime.datetime
            ):
                other = datetime.datetime(other.year, other.month, other.day)
            scalar = pa.scalar(other)
            if pa.types.is_timestamp(scalar.type):
                self_tz = getattr(self.dtype, "tz", None)
                if (scalar.type.tz is None) != (self_tz is None):
                    raise TypeError(
                        f"Invalid comparison between dtype={self.dtype} "
                        f"and {type(other).__name__}"
                    )
                return scalar
            elif pa.types.is_duration(scalar.type):
                if self.dtype.kind == "m":
                    common_dtype = find_common_type(
                        (self.dtype, cudf_dtype_from_pa_type(scalar.type))
                    )
                    scalar = scalar.cast(cudf_dtype_to_pa_type(common_dtype))
                return scalar
            elif self.dtype.kind == "m":
                return scalar
            else:
                return NotImplemented
        return NotImplemented

    @functools.cached_property
    def time_unit(self) -> str:
        if isinstance(self.dtype, np.dtype):
            # np.dtype defaults to dtype[float64] for typing
            return np.datetime_data(self.dtype)[0]
        else:
            return cast("pd.ArrowDtype", self.dtype).pyarrow_dtype.unit

    @property
    def values(self) -> cp.ndarray:
        """
        Return a CuPy representation of the TemporalBaseColumn.
        """
        if isinstance(self.dtype, pd.ArrowDtype):
            dtype = self.dtype.numpy_dtype
        else:
            dtype = self.dtype

        if len(self) == 0:
            return cp.empty(0, dtype=self._UNDERLYING_DTYPE).view(dtype)

        if self.has_nulls():
            raise ValueError("cupy does not support NaT.")
        return cp.asarray(self).view(dtype)

    def element_indexing(self, index: int) -> ScalarLike:
        result = super().element_indexing(index)
        if isinstance(result, (datetime.datetime, datetime.timedelta)):
            return self._PD_SCALAR(result).as_unit(self.time_unit)
        return result

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if nullable and not arrow_type:
            raise NotImplementedError(
                f"pandas does not have a native nullable type for {self.dtype}."
            )
        else:
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)

    def as_numerical_column(self, dtype: np.dtype) -> NumericalColumn:
        if cudf.get_option("mode.pandas_compatible") and dtype.kind == "f":
            # pandas raises TypeError for datetime/timedelta → float casts
            # (only integer casts are allowed via Series constructor).
            kind_name = "Datetime" if self.dtype.kind == "M" else "Timedelta"
            raise TypeError(f"Cannot cast {kind_name}Array to dtype {dtype}")
        new_plc_column = plc.Column(
            data_type=dtype_to_pylibcudf_type(self._UNDERLYING_DTYPE),
            size=self.size,
            data=self.data,
            mask=self.mask,
            null_count=self.null_count,
            offset=self.offset,
            children=[],
        )
        return cast(
            "cudf.core.column.numerical.NumericalColumn",
            ColumnBase.create(new_plc_column, self._UNDERLYING_DTYPE).astype(
                dtype
            ),
        )

    def ceil(self, freq: str) -> ColumnBase:
        raise NotImplementedError("ceil is currently not implemented")

    def floor(self, freq: str) -> ColumnBase:
        raise NotImplementedError("floor is currently not implemented")

    def round(self, freq: str) -> ColumnBase:
        raise NotImplementedError("round is currently not implemented")

    def strftime(self, format: str, dtype: DtypeObj) -> StringColumn:
        if len(self) == 0:
            return column_empty(0, dtype=dtype)  # type:ignore[return-value]
        else:
            raise NotImplementedError("strftime is currently not implemented")

    def find_and_replace(
        self,
        to_replace: ColumnBase | list,
        replacement: ColumnBase | list,
        all_nan: bool = False,
    ) -> Self:
        if not isinstance(to_replace, type(self)):
            to_replace = as_column(to_replace)
            if to_replace.can_cast_safely(self.dtype):
                to_replace = to_replace.astype(self.dtype)
        if not isinstance(replacement, type(self)):
            replacement = as_column(replacement)
            if replacement.can_cast_safely(self.dtype):
                replacement = replacement.astype(self.dtype)
        if isinstance(to_replace, type(self)):
            to_replace = to_replace.astype(self._UNDERLYING_DTYPE)
        if isinstance(replacement, type(self)):
            replacement = replacement.astype(self._UNDERLYING_DTYPE)
        try:
            return cast(
                "Self",
                self.astype(self._UNDERLYING_DTYPE)
                .find_and_replace(to_replace, replacement, all_nan)
                .astype(self.dtype),
            )
        except TypeError:
            return self.copy(deep=True)

    def can_cast_safely(self, to_dtype: DtypeObj) -> bool:
        if to_dtype.kind == self.dtype.kind:
            to_res, _ = np.datetime_data(to_dtype)
            if (
                unit_to_nanoseconds_conversion[to_res]
                > unit_to_nanoseconds_conversion[self.time_unit]
            ):
                # Casting to a coarser resolution truncates any
                # sub-resolution components; callers (replace, join key
                # matching) rely on "safely" meaning lossless.
                return False
            max_val = self.max()
            if isinstance(max_val, (pd.Timedelta, pd.Timestamp)):
                max_val = max_val.to_numpy().astype(self.dtype)
            max_val = max_val.astype(self._UNDERLYING_DTYPE, copy=False)
            min_val = self.min()
            if isinstance(min_val, (pd.Timedelta, pd.Timestamp)):
                min_val = min_val.to_numpy().astype(self.dtype)
            min_val = min_val.astype(self._UNDERLYING_DTYPE, copy=False)
            # call-overload must be ignored because numpy stubs only accept literal strings
            # for time units (e.g., "ns", "us") to allow compile-time validation,
            # but we're passing variables (self.time_unit) with time units that
            # we know are valid at runtime
            max_dist = np.timedelta64(
                max_val,
                self.time_unit,  # type: ignore[call-overload]
            )
            min_dist = np.timedelta64(
                min_val,
                self.time_unit,  # type: ignore[call-overload]
            )
            max_to_res = np.timedelta64(
                np.iinfo(self._UNDERLYING_DTYPE).max,
                to_res,  # type: ignore[call-overload]
            ).astype(f"m8[{self.time_unit}]", copy=False)
            # The negative bound is symmetric: int64 min is the NaT
            # sentinel, so the valid range is +/-(2**63 - 1).
            return bool(max_dist <= max_to_res and min_dist >= -max_to_res)
        elif to_dtype == self._UNDERLYING_DTYPE or is_dtype_obj_string(
            to_dtype
        ):
            # can safely cast to representation, or string
            return True
        else:
            return False

    def mean(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> pd.Timestamp | pd.Timedelta:
        return self._PD_SCALAR(
            self.astype(self._UNDERLYING_DTYPE).mean(
                skipna=skipna, min_count=min_count, **kwargs
            ),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def std(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> pd.Timedelta:
        # Extract ddof from kwargs, defaulting to 1
        ddof = kwargs.pop("ddof", 1)
        return pd.Timedelta(
            self.astype(self._UNDERLYING_DTYPE).std(
                skipna=skipna, min_count=min_count, ddof=ddof, **kwargs
            ),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def median(
        self, skipna: bool = True, min_count: int = 0, **kwargs: Any
    ) -> pd.Timestamp | pd.Timedelta:
        return self._PD_SCALAR(
            self.astype(self._UNDERLYING_DTYPE).median(
                skipna=skipna, min_count=min_count, **kwargs
            ),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    def cov(self, other: Self) -> float:
        if not isinstance(other, type(self)):
            raise TypeError(
                f"cannot perform cov with types {self.dtype}, {other.dtype}"
            )
        return self.astype(self._UNDERLYING_DTYPE).cov(  # type:ignore[attr-defined]
            other.astype(self._UNDERLYING_DTYPE)
        )

    def corr(self, other: Self) -> float:
        if not isinstance(other, type(self)):
            raise TypeError(
                f"cannot perform corr with types {self.dtype}, {other.dtype}"
            )
        return self.astype(self._UNDERLYING_DTYPE).corr(  # type:ignore[attr-defined]
            other.astype(self._UNDERLYING_DTYPE)
        )

    def quantile(
        self,
        q: np.ndarray,
        interpolation: str,
        exact: bool,
        return_scalar: bool,
    ) -> ColumnBase | pd.Timestamp | pd.Timedelta:
        result = self.astype(self._UNDERLYING_DTYPE).quantile(
            q=q,
            interpolation=interpolation,
            exact=exact,
            return_scalar=return_scalar,
        )
        if return_scalar:
            ts = self._PD_SCALAR(result, unit=self.time_unit).as_unit(
                self.time_unit
            )
            # For tz-aware datetimes the underlying int64 stores UTC; the
            # scalar above is tz-naive, so re-attach the original tz.
            if (tz := getattr(self, "tz", None)) is not None:
                ts = ts.tz_localize("UTC").tz_convert(tz)
            return ts
        return result.astype(self.dtype)

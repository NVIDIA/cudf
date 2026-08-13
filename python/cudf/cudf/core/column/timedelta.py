# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.core._internals import binaryop
from cudf.core.column.column import (
    ColumnBase,
    PylibcudfFunction,
    as_column,
    fixed_dtype_policy,
)
from cudf.core.column.temporal_base import (
    TemporalBaseColumn,
    _raise_on_invalid_ordering_scalar,
)
from cudf.errors import MixedTypeError
from cudf.utils.dtypes import (
    cudf_dtype_from_pa_type,
    cudf_dtype_to_pa_type,
    dtype_to_pylibcudf_type,
    find_common_type,
    get_dtype_of_same_kind,
    is_pandas_nullable_extension_dtype,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.temporal import unit_to_nanoseconds_conversion
from cudf.utils.utils import _EQUALITY_OPS, is_na_like

if TYPE_CHECKING:
    from cudf._typing import (
        ColumnBinaryOperand,
        DatetimeLikeScalar,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.string import StringColumn


@functools.cache
def get_np_td_unit_conversion(
    reso: str, dtype: None | np.dtype
) -> np.timedelta64:
    td = np.timedelta64(unit_to_nanoseconds_conversion[reso], "ns")
    if dtype is not None:
        return td.astype(dtype)
    return td


class TimeDeltaColumn(TemporalBaseColumn):
    _NP_SCALAR = np.timedelta64
    _PD_SCALAR = pd.Timedelta
    _VALID_REDUCTIONS = {
        "median",
    }
    _VALID_BINARY_OPERATIONS = {
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__add__",
        "__sub__",
        "__mul__",
        "__mod__",
        "__truediv__",
        "__floordiv__",
        "__radd__",
        "__rsub__",
        "__rmul__",
        "__rmod__",
        "__rtruediv__",
        "__rfloordiv__",
    }

    def _reduce(
        self,
        op: str,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> ScalarLike:
        # Pandas raises TypeError for certain unsupported timedelta reductions
        if op == "product":
            raise TypeError(
                f"'{type(self).__name__}' with dtype {self.dtype} "
                f"does not support reduction '{op}'"
            )
        if op == "var":
            raise TypeError(
                f"'{type(self).__name__}' with dtype {self.dtype} "
                f"does not support reduction '{op}'"
            )
        return super()._reduce(
            op, skipna=skipna, min_count=min_count, **kwargs
        )

    def __contains__(self, item: DatetimeLikeScalar) -> bool:
        try:
            # call-overload must be ignored because numpy stubs only accept literal
            # time unit strings, but we're passing self.time_unit which is valid at runtime
            item = self._NP_SCALAR(item, self.time_unit)  # type: ignore[call-overload]
        except ValueError:
            # If item cannot be converted to duration type
            # np.timedelta64 raises ValueError, hence `item`
            # cannot exist in `self`.
            return False
        return super().__contains__(item.to_numpy())

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        reflect, op = self._check_reflected_op(op)
        _raise_on_invalid_ordering_scalar(self.dtype, other, op)
        other = self._normalize_binop_operand(other)
        if other is NotImplemented:
            return NotImplemented

        this: ColumnBinaryOperand = self
        out_dtype = None
        other_cudf_dtype = (
            cudf_dtype_from_pa_type(other.type)
            if isinstance(other, pa.Scalar)
            else other.dtype
        )
        other_is_null_scalar = is_na_like(other)
        if (
            isinstance(self.dtype, pd.ArrowDtype)
            or isinstance(other_cudf_dtype, pd.ArrowDtype)
        ) and op == "__mod__":
            raise NotImplementedError("ArrowDtype does not support modulo")

        if other_cudf_dtype.kind == "m":
            # TODO: pandas will allow these operators to work but return false
            # when comparing to non-timedelta dtypes. We should do the same.
            if op in {
                "__eq__",
                "__ne__",
                "__lt__",
                "__gt__",
                "__le__",
                "__ge__",
                "NULL_EQUALS",
                "NULL_NOT_EQUALS",
            }:
                out_dtype = get_dtype_of_same_kind(
                    self.dtype, np.dtype(np.bool_)
                )
            elif op == "__mod__":
                out_dtype = find_common_type((self.dtype, other_cudf_dtype))
            elif op in {"__truediv__", "__floordiv__"}:
                common_dtype = find_common_type((self.dtype, other_cudf_dtype))
                out_dtype = (
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.float64))
                    if op == "__truediv__"
                    else self._UNDERLYING_DTYPE
                )
                this = self.astype(common_dtype).astype(out_dtype)
                if isinstance(other, pa.Scalar):
                    if other.is_valid:
                        # pyarrow.cast doesn't support casting duration to float
                        # so go through numpy
                        other_np = pa.array([other]).to_numpy(
                            zero_copy_only=False
                        )
                        other_np = other_np.astype(common_dtype).astype(
                            out_dtype
                        )
                        other = pa.array(other_np)[0]
                    else:
                        other = pa.scalar(
                            None, type=cudf_dtype_to_pa_type(out_dtype)
                        )
                else:
                    other = other.astype(common_dtype).astype(out_dtype)
            elif op in {"__add__", "__sub__"}:
                out_dtype = find_common_type((self.dtype, other_cudf_dtype))
        elif other_cudf_dtype.kind in {"f", "i", "u"}:
            if op in {"__mul__", "__mod__", "__truediv__", "__floordiv__"}:
                out_dtype = self.dtype
            elif op in {"__eq__", "__ne__", "NULL_EQUALS", "NULL_NOT_EQUALS"}:
                if isinstance(other, ColumnBase) and not isinstance(
                    other, TimeDeltaColumn
                ):
                    fill_value = op in ("__ne__", "NULL_NOT_EQUALS")
                    result = self._all_bools_with_nulls(
                        other,
                        bool_fill_value=fill_value,
                    )
                    result = result.fillna(fill_value)
                    return result

        if out_dtype is None:
            return NotImplemented
        elif isinstance(other, pa.Scalar):
            other = pa_scalar_to_plc_scalar(other)

        lhs, rhs = (other, this) if reflect else (this, other)

        result = binaryop.binaryop(lhs, rhs, op, out_dtype)
        if (
            out_dtype.kind == "b"
            and (op in _EQUALITY_OPS or not other_is_null_scalar)
            and not is_pandas_nullable_extension_dtype(out_dtype)
        ):
            result = result.fillna(op == "__ne__")
        return result

    def total_seconds(self) -> ColumnBase:
        conversion = unit_to_nanoseconds_conversion[self.time_unit] / 1e9
        # Typecast to decimal128 to avoid floating point precision issues
        # https://github.com/rapidsai/cudf/issues/17664
        result = (
            (self.astype(self._UNDERLYING_DTYPE) * conversion)
            .astype(
                cudf.Decimal128Dtype(cudf.Decimal128Dtype.MAX_PRECISION, 9)
            )
            .round(decimals=abs(int(math.log10(conversion))))
            .astype(np.dtype(np.float64))
        )
        if isinstance(self.dtype, pd.ArrowDtype):
            result = result.astype(
                get_dtype_of_same_kind(self.dtype, np.dtype(np.float64))
            )
        return result

    def as_datetime_column(self, dtype: np.dtype) -> None:  # type: ignore[override]
        raise TypeError(
            f"cannot astype a timedelta from {self.dtype} to {dtype}"
        )

    def strftime(self, format: str, dtype: DtypeObj) -> StringColumn:
        if len(self) == 0:
            return super().strftime(format, dtype)
        return cast(
            "cudf.core.column.string.StringColumn",
            PylibcudfFunction(
                plc.strings.convert.convert_durations.from_durations,
                fixed_dtype_policy(dtype),
            ).execute_with_args(self, format),
        )

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        if (
            cudf.get_option("mode.pandas_compatible")
            and isinstance(dtype, np.dtype)
            and dtype.kind == "O"
        ):
            raise MixedTypeError(
                f"cannot astype a timedelta like from {self.dtype} to {dtype}"
            )
        if isinstance(dtype, np.dtype) and dtype.kind == "U":
            dtype = np.dtype("object")
        components = self.components
        has_hours = components["hours"].any()
        has_minutes = components["minutes"].any()
        has_seconds = components["seconds"].any()
        has_millis = (
            self.time_unit in {"ns", "us", "ms"}
            and components["milliseconds"].any()
        )
        has_micros = self.time_unit in {"ns", "us"} and self.microseconds.any()
        has_nanos = self.time_unit == "ns" and self.nanoseconds.any()

        has_subday = has_hours or has_minutes or has_seconds
        has_fraction = has_millis or has_micros or has_nanos
        return self._as_string_pandas_compat(
            dtype,
            has_subday=has_subday,
            has_fraction=has_fraction,
            has_nanos=has_nanos,
        )

    def _as_string_pandas_compat(
        self,
        dtype: DtypeObj,
        *,
        has_subday: bool,
        has_fraction: bool,
        has_nanos: bool,
    ) -> StringColumn:
        """Convert to string using pandas-compatible formatting."""
        if not (has_subday or has_fraction):
            fmt = "%D days"
        else:
            fmt = "%D days %H:%M:%S"

        with self.access(mode="read", scope="internal"):
            # Determine which pylibcudf column to format: for ms we
            # cast to us so %S natively outputs 6 fractional digits.
            if has_fraction and self.time_unit == "ms":
                cast_type = dtype_to_pylibcudf_type(
                    np.dtype("timedelta64[us]")
                )
                plc_col = plc.unary.cast(self.plc_column, cast_type)
            else:
                plc_col = self.plc_column

            # Convert durations to strings.
            plc_result = plc.strings.convert.convert_durations.from_durations(
                plc_col, fmt
            )

        if has_fraction:
            if has_nanos:
                # Rows that are exact microseconds: strip the
                # trailing "000" → 6 digits.
                plc_result = plc.strings.replace_re.replace_with_backrefs(
                    plc_result,
                    plc.strings.regex_program.RegexProgram.create(
                        r"(\.\d{6})000$",
                        plc.strings.regex_flags.RegexFlags.DEFAULT,
                    ),
                    r"\1",
                )
            elif self.time_unit == "ns":
                # No actual nanos but stored as ns: trim 9 → 6.
                plc_result = plc.strings.replace_re.replace_with_backrefs(
                    plc_result,
                    plc.strings.regex_program.RegexProgram.create(
                        r"(\.\d{6})\d{3}$",
                        plc.strings.regex_flags.RegexFlags.DEFAULT,
                    ),
                    r"\1",
                )

            # Drop fully-zero fractional seconds on rows that
            # have no sub-second component (keeps HH:MM:SS for
            # mixed columns).
            plc_result = plc.strings.replace_re.replace_with_backrefs(
                plc_result,
                plc.strings.regex_program.RegexProgram.create(
                    r"(\d{2}:\d{2}:\d{2})\.000000$",
                    plc.strings.regex_flags.RegexFlags.DEFAULT,
                ),
                r"\1",
            )

        return cast(
            "cudf.core.column.string.StringColumn",
            ColumnBase.create(plc_result, dtype),
        )

    def as_timedelta_column(self, dtype: np.dtype) -> TimeDeltaColumn:
        if dtype == self.dtype:
            return self
        return self.cast(dtype=dtype)  # type: ignore[return-value]

    def sum(
        self,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> pd.Timedelta:
        return self._PD_SCALAR(
            self.astype(self._UNDERLYING_DTYPE).sum(
                skipna=skipna, min_count=min_count, **kwargs
            ),
            unit=self.time_unit,
        ).as_unit(self.time_unit)

    @functools.cached_property
    def components(self) -> dict[str, NumericalColumn]:
        """
        Return a dict of the components of the Timedeltas.

        Returns
        -------
        dict[str, NumericalColumn]
        """
        date_meta = {
            "hours": ["D", "h"],
            "minutes": ["h", "m"],
            "seconds": ["m", "s"],
            "milliseconds": ["s", "ms"],
            "microseconds": ["ms", "us"],
            "nanoseconds": ["us", "ns"],
        }
        data = {"days": self.days}
        reached_self_unit = False
        for result_key, (mod_unit, div_unit) in date_meta.items():
            if not reached_self_unit:
                res_col = (
                    self % get_np_td_unit_conversion(mod_unit, self.dtype)
                ) // get_np_td_unit_conversion(div_unit, self.dtype)
                reached_self_unit = self.time_unit == div_unit
            else:
                res_col = as_column(
                    0, length=len(self), dtype=self._UNDERLYING_DTYPE
                )
                if self.nullable:
                    res_col = res_col.set_mask(self.mask, self.null_count)
            data[result_key] = res_col
        return data

    @functools.cached_property
    def days(self) -> NumericalColumn:
        """
        Number of days for each element.

        Returns
        -------
        NumericalColumn
        """
        return self // get_np_td_unit_conversion("D", self.dtype)

    @functools.cached_property
    def seconds(self) -> NumericalColumn:
        """
        Number of seconds (>= 0 and less than 1 day).

        Returns
        -------
        NumericalColumn
        """
        # This property must return the number of seconds (>= 0 and
        # less than 1 day) for each element, hence first performing
        # mod operation to remove the number of days and then performing
        # division operation to extract the number of seconds.

        return (
            self % get_np_td_unit_conversion("D", self.dtype)
        ) // get_np_td_unit_conversion("s", None)

    @functools.cached_property
    def microseconds(self) -> NumericalColumn:
        """
        Number of microseconds (>= 0 and less than 1 second).

        Returns
        -------
        NumericalColumn
        """
        # This property must return the number of microseconds (>= 0 and
        # less than 1 second) for each element, hence first performing
        # mod operation to remove the number of seconds and then performing
        # division operation to extract the number of microseconds.

        return (
            self % get_np_td_unit_conversion("s", self.dtype)
        ) // get_np_td_unit_conversion("us", None)

    @functools.cached_property
    def nanoseconds(self) -> NumericalColumn:
        """
        Return the number of nanoseconds (n), where 0 <= n < 1 microsecond.

        Returns
        -------
        NumericalColumn
        """
        # This property must return the number of nanoseconds (>= 0 and
        # less than 1 microsecond) for each element, hence first performing
        # mod operation to remove the number of microseconds and then
        # performing division operation to extract the number
        # of nanoseconds.

        if self.time_unit != "ns":
            res_col = as_column(
                0, length=len(self), dtype=self._UNDERLYING_DTYPE
            )
            if self.nullable:
                res_col = res_col.set_mask(self.mask, self.null_count)
            return cast("cudf.core.column.NumericalColumn", res_col)
        return (
            self % get_np_td_unit_conversion("us", None)
        ) // get_np_td_unit_conversion("ns", None)

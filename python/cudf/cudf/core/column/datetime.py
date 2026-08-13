# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import calendar
import datetime
import functools
import locale
import os
import re
import zoneinfo
from locale import nl_langinfo
from typing import TYPE_CHECKING, Any, Literal, cast

import numpy as np
import pandas as pd
import pyarrow as pa

import pylibcudf as plc

import cudf
from cudf.core._internals import binaryop
from cudf.core._internals.timezones import (
    get_compatible_timezone,
)
from cudf.core.column import as_column, column_empty
from cudf.core.column.column import (
    ColumnBase,
    PylibcudfFunction,
    bool_same_kind_policy,
    int16_same_kind_policy,
    pylibcudf_result_dtype_policy,
    same_dtype_policy,
)
from cudf.core.column.temporal_base import (
    TemporalBaseColumn,
    _raise_on_invalid_ordering_scalar,
)
from cudf.core.dtype.validators import is_dtype_obj_datetime_tz
from cudf.utils.dtypes import (
    _get_base_dtype,
    cudf_dtype_from_pa_type,
    dtype_from_pylibcudf_column,
    dtype_to_pylibcudf_type,
    get_dtype_of_same_kind,
)
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.temporal import (
    _unit_to_name,
    unit_to_nanoseconds_conversion,
)
from cudf.utils.utils import _EQUALITY_OPS, is_na_like

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Self

    from cudf._typing import (
        ColumnBinaryOperand,
        DtypeObj,
        ScalarLike,
    )
    from cudf.core.column.numerical import NumericalColumn
    from cudf.core.column.string import StringColumn

# nanoseconds per time_unit
_dtype_to_format_conversion = {
    "datetime64[ns]": "%Y-%m-%d %H:%M:%S.%9f",
    "datetime64[us]": "%Y-%m-%d %H:%M:%S.%6f",
    "datetime64[ms]": "%Y-%m-%d %H:%M:%S.%3f",
    "datetime64[s]": "%Y-%m-%d %H:%M:%S",
    "timestamp[ns][pyarrow]": "%Y-%m-%d %H:%M:%S.%9f",
    "timestamp[us][pyarrow]": "%Y-%m-%d %H:%M:%S.%6f",
    "timestamp[ms][pyarrow]": "%Y-%m-%d %H:%M:%S.%3f",
    "timestamp[s][pyarrow]": "%Y-%m-%d %H:%M:%S",
}


def _resolve_binop_resolution(
    left_unit: Literal["s", "ms", "us", "ns"],
    right_unit: Literal["s", "ms", "us", "ns"],
) -> Literal["s", "ms", "us", "ns"]:
    units: list[Literal["s", "ms", "us", "ns"]] = ["s", "ms", "us", "ns"]
    left_idx = units.index(left_unit)
    right_idx = units.index(right_unit)
    return units[max(left_idx, right_idx)]


@functools.lru_cache(maxsize=20)
def _get_tz_data(zone_name: str) -> tuple[ColumnBase, ColumnBase]:
    """
    Return timezone data (transition times and UTC offsets) for the
    given IANA time zone.

    Parameters
    ----------
    zone_name: str
        IANA time zone name

    Returns
    -------
    Tuple with two columns containing the transition times
    and corresponding UTC offsets.
    """
    try:
        # like zoneinfo, we first look in TZPATH
        tz_table = _find_and_read_tzfile_tzpath(zone_name)
    except zoneinfo.ZoneInfoNotFoundError:
        # if that fails, we fall back to using `tzdata`
        tz_table = _find_and_read_tzfile_tzdata(zone_name)
    return tz_table


def _find_and_read_tzfile_tzpath(
    zone_name: str,
) -> tuple[ColumnBase, ColumnBase]:
    for search_path in zoneinfo.TZPATH:
        if os.path.isfile(os.path.join(search_path, zone_name)):
            return _read_tzfile_as_columns(search_path, zone_name)
    raise zoneinfo.ZoneInfoNotFoundError(zone_name)


def _find_and_read_tzfile_tzdata(
    zone_name: str,
) -> tuple[ColumnBase, ColumnBase]:
    import importlib.resources

    package_base = "tzdata.zoneinfo"
    try:
        return _read_tzfile_as_columns(
            str(importlib.resources.files(package_base)), zone_name
        )
    # TODO: make it so that the call to libcudf raises a
    # FileNotFoundError instead of a RuntimeError
    except (ImportError, FileNotFoundError, UnicodeEncodeError, RuntimeError):
        # the "except" part of this try-except is basically vendored
        # from the zoneinfo library.
        #
        # There are three types of exception that can be raised that all amount
        # to "we cannot find this key":
        #
        # ImportError: If package_name doesn't exist (e.g. if tzdata is not
        #   installed, or if there's an error in the folder name like
        #   Amrica/New_York)
        # FileNotFoundError: If resource_name doesn't exist in the package
        #   (e.g. Europe/Krasnoy)
        # UnicodeEncodeError: If package_name or resource_name are not UTF-8,
        #   such as keys containing a surrogate character.
        raise zoneinfo.ZoneInfoNotFoundError(zone_name)


def _read_tzfile_as_columns(
    tzdir: str, zone_name: str
) -> tuple[ColumnBase, ColumnBase]:
    plc_table = plc.io.timezone.make_timezone_transition_table(
        tzdir, zone_name
    )
    transition_times_and_offsets = plc_table.columns()

    if not transition_times_and_offsets:
        # this happens for UTC-like zones
        min_date: np.datetime64 = np.int64(np.iinfo("int64").min + 1).astype(
            np.dtype("M8[s]")
        )
        return (as_column([min_date]), as_column([np.timedelta64(0, "s")]))

    result = tuple(
        ColumnBase.create(col, dtype=dtype_from_pylibcudf_column(col))
        for col in transition_times_and_offsets
    )
    assert len(result) == 2  # for mypy
    return result


def _check_ambiguous_and_nonexistent(
    ambiguous: Literal["NaT"], nonexistent: Literal["NaT"]
) -> tuple[Literal["NaT"], Literal["NaT"]]:
    if ambiguous != "NaT":
        raise NotImplementedError(
            "Only ambiguous='NaT' is currently supported"
        )
    if nonexistent != "NaT":
        raise NotImplementedError(
            "Only nonexistent='NaT' is currently supported"
        )
    return ambiguous, nonexistent


class DatetimeColumn(TemporalBaseColumn):
    """A Column implementation for Date-time types."""

    _NP_SCALAR = np.datetime64
    _PD_SCALAR = pd.Timestamp
    _VALID_BINARY_OPERATIONS = {
        "__eq__",
        "__ne__",
        "__lt__",
        "__le__",
        "__gt__",
        "__ge__",
        "__add__",
        "__sub__",
        "__radd__",
        "__rsub__",
    }

    def _reduce(
        self,
        op: str,
        skipna: bool = True,
        min_count: int = 0,
        **kwargs: Any,
    ) -> ScalarLike:
        # Pandas raises TypeError for certain unsupported datetime reductions
        if op in {"sum", "product"}:
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

    def __contains__(self, item: ScalarLike) -> bool:
        try:
            ts = self._PD_SCALAR(item).as_unit(self.time_unit)
        except Exception:
            # pandas can raise a variety of errors
            # item cannot exist in self.
            return False
        if ts.tzinfo is None and isinstance(self.dtype, pd.DatetimeTZDtype):
            return False
        elif ts.tzinfo is not None:
            ts = ts.tz_convert(None)
        return super().__contains__(ts.to_numpy())

    @functools.cached_property
    def quarter(self) -> ColumnBase:
        return PylibcudfFunction(
            plc.datetime.extract_quarter,
            int16_same_kind_policy,
        ).execute_with_args(self)

    @functools.cached_property
    def year(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.YEAR)

    @functools.cached_property
    def month(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.MONTH)

    @functools.cached_property
    def day(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.DAY)

    @functools.cached_property
    def hour(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.HOUR)

    @functools.cached_property
    def minute(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.MINUTE)

    @functools.cached_property
    def second(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.SECOND)

    @functools.cached_property
    def millisecond(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.MILLISECOND)

    @functools.cached_property
    def microsecond(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.MICROSECOND)

    @functools.cached_property
    def nanosecond(self) -> ColumnBase:
        return self._get_dt_field(plc.datetime.DatetimeComponent.NANOSECOND)

    @functools.cached_property
    def weekday(self) -> ColumnBase:
        # pandas counts Monday-Sunday as 0-6
        # while libcudf counts Monday-Sunday as 1-7
        result = self._get_dt_field(plc.datetime.DatetimeComponent.WEEKDAY)
        return result - result.dtype.type(1)

    @functools.cached_property
    def day_of_year(self) -> ColumnBase:
        result = PylibcudfFunction(
            plc.datetime.day_of_year,
            pylibcudf_result_dtype_policy,
        ).execute_with_args(self)
        if isinstance(self.dtype, pd.ArrowDtype):
            # pyarrow temporal kernels return int64; match pandas.
            result = result.astype(
                get_dtype_of_same_kind(self.dtype, np.dtype(np.int64))
            )
        return result

    def _maybe_fillna_false(self, col: ColumnBase) -> ColumnBase:
        # For nullable extension dtype sources (e.g. ArrowDtype), preserve
        # nulls in the result to match pandas. Otherwise fill with False
        # so the result is a plain bool column.
        if isinstance(self.dtype, pd.ArrowDtype):
            return col
        return col.fillna(False)

    @functools.cached_property
    def is_month_start(self) -> ColumnBase:
        return self._maybe_fillna_false(self.day == 1)

    @functools.cached_property
    def is_month_end(self) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            plc_result = plc.unary.cast(
                plc.datetime.last_day_of_month(self.plc_column),
                plc.DataType(plc.TypeId.TIMESTAMP_SECONDS),
            )
            last_day_col = ColumnBase.create(
                plc_result,
                dtype_from_pylibcudf_column(plc_result),
            )
        return self._maybe_fillna_false(
            self.day == cast("Self", last_day_col).day
        )

    @functools.cached_property
    def is_quarter_end(self) -> ColumnBase:
        last_month = self.month.isin([3, 6, 9, 12])
        return self._maybe_fillna_false(self.is_month_end & last_month)

    @functools.cached_property
    def is_quarter_start(self) -> ColumnBase:
        first_month = self.month.isin([1, 4, 7, 10])
        return self._maybe_fillna_false(self.is_month_start & first_month)

    @functools.cached_property
    def is_year_end(self) -> ColumnBase:
        day_of_year = self.day_of_year
        leap_dates = self.is_leap_year

        leap = day_of_year == 366
        non_leap = day_of_year == 365
        return self._maybe_fillna_false(
            leap.copy_if_else(non_leap, leap_dates)
        )

    @functools.cached_property
    def is_leap_year(self) -> ColumnBase:
        return PylibcudfFunction(
            plc.datetime.is_leap_year,
            bool_same_kind_policy,
        ).execute_with_args(self)

    @functools.cached_property
    def is_year_start(self) -> ColumnBase:
        return self._maybe_fillna_false(self.day_of_year == 1)

    @functools.cached_property
    def days_in_month(self) -> ColumnBase:
        result = PylibcudfFunction(
            plc.datetime.days_in_month,
            pylibcudf_result_dtype_policy,
        ).execute_with_args(self)
        if isinstance(self.dtype, pd.ArrowDtype):
            # pyarrow temporal kernels return int64; match pandas.
            result = result.astype(
                get_dtype_of_same_kind(self.dtype, np.dtype(np.int64))
            )
        return result

    @functools.cached_property
    def day_of_week(self) -> ColumnBase:
        raise NotImplementedError("day_of_week is currently not implemented.")

    @functools.cached_property
    def tz(self) -> datetime.tzinfo | None:
        """
        Return the timezone.

        Returns
        -------
        datetime.tzinfo or None
            Returns None when the array is tz-naive.
        """
        if isinstance(self.dtype, pd.DatetimeTZDtype):
            return self.dtype.tz
        elif (
            isinstance(self.dtype, pd.ArrowDtype)
            and (tz := self.dtype.pyarrow_dtype.tz) is not None
        ):
            return zoneinfo.ZoneInfo(tz)
        return None

    @functools.cached_property
    def freq(self) -> str | None:
        raise NotImplementedError("freq is not yet implemented.")

    @functools.cached_property
    def date(self) -> None:
        raise NotImplementedError("date is not yet implemented.")

    @functools.cached_property
    def time(self) -> None:
        raise NotImplementedError("time is not yet implemented.")

    @functools.cached_property
    def timetz(self) -> None:
        raise NotImplementedError("timetz is not yet implemented.")

    @functools.cached_property
    def is_normalized(self) -> bool:
        raise NotImplementedError(
            "is_normalized is currently not implemented."
        )

    def to_julian_date(self) -> ColumnBase:
        raise NotImplementedError(
            "to_julian_date is currently not implemented."
        )

    def normalize(self) -> ColumnBase:
        raise NotImplementedError("normalize is currently not implemented.")

    def _get_dt_field(
        self, field: plc.datetime.DatetimeComponent
    ) -> ColumnBase:
        with self.access(mode="read", scope="internal"):
            result = ColumnBase.create(
                plc.datetime.extract_datetime_component(
                    self.plc_column,
                    field,
                ),
                get_dtype_of_same_kind(self.dtype, np.dtype(np.int16)),
            )
            if isinstance(self.dtype, pd.ArrowDtype):
                # pyarrow's temporal compute kernels return int64; match that
                # when the source is an ArrowDtype.
                result = result.astype(
                    get_dtype_of_same_kind(self.dtype, np.dtype(np.int64))
                )
            elif result.dtype == np.dtype("int16"):
                result = result.astype(np.dtype("int32"))
            return result

    def _get_field_names(
        self,
        field: Literal["month", "weekday"],
        labels: list[str],
        locale: str | None = None,
    ) -> ColumnBase:
        if locale is not None:
            raise NotImplementedError(
                "Setting a locale is currently not supported. "
                "Results will be returned in your current locale."
            )
        col_labels = as_column(labels)
        indices = getattr(self, field)
        has_nulls = indices.has_nulls()
        if has_nulls:
            indices = indices.fillna(len(col_labels))
        return col_labels.take(indices, nullify=True, check_bounds=has_nulls)

    def get_day_names(self, locale: str | None = None) -> ColumnBase:
        return self._get_field_names(
            "weekday", list(calendar.day_name), locale=locale
        )

    def get_month_names(self, locale: str | None = None) -> ColumnBase:
        return self._get_field_names(
            "month", list(calendar.month_name), locale=locale
        )

    def _round_dt(
        self,
        round_func: Callable[
            [plc.Column, plc.datetime.RoundingFrequency], plc.Column
        ],
        freq: str,
    ) -> ColumnBase:
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Timedelta.resolution_string.html
        rounding_fequency_map = {
            "D": plc.datetime.RoundingFrequency.DAY,
            "h": plc.datetime.RoundingFrequency.HOUR,
            "min": plc.datetime.RoundingFrequency.MINUTE,
            "s": plc.datetime.RoundingFrequency.SECOND,
            "ms": plc.datetime.RoundingFrequency.MILLISECOND,
            "us": plc.datetime.RoundingFrequency.MICROSECOND,
            "ns": plc.datetime.RoundingFrequency.NANOSECOND,
        }
        if (plc_freq := rounding_fequency_map.get(freq)) is None:
            raise ValueError(f"Invalid resolution: '{freq}'")

        return PylibcudfFunction(
            round_func, same_dtype_policy
        ).execute_with_args(self, plc_freq)

    def ceil(self, freq: str) -> ColumnBase:
        return self._round_dt(plc.datetime.ceil_datetimes, freq)

    def floor(self, freq: str) -> ColumnBase:
        return self._round_dt(plc.datetime.floor_datetimes, freq)

    def round(self, freq: str) -> ColumnBase:
        return self._round_dt(plc.datetime.round_datetimes, freq)

    def isocalendar(self) -> dict[str, ColumnBase]:
        # pyarrow's isocalendar returns int64 fields and fills nulls with 0;
        # pandas' isocalendar for datetime64 returns UInt32 with NA preserved.
        is_arrow = isinstance(self.dtype, pd.ArrowDtype)
        target_dtype: DtypeObj = (
            np.dtype(np.int64) if is_arrow else pd.UInt32Dtype()
        )
        result = {
            field: self.strftime(
                format=directive, dtype=np.dtype("object")
            ).astype(get_dtype_of_same_kind(self.dtype, target_dtype))
            for field, directive in zip(
                ["year", "week", "day"], ["%G", "%V", "%u"], strict=True
            )
        }
        if is_arrow:
            result = {field: col.fillna(0) for field, col in result.items()}
        return result

    def as_datetime_column(self, dtype: np.dtype) -> DatetimeColumn:
        if dtype == self.dtype:
            return self
        elif isinstance(dtype, pd.DatetimeTZDtype):
            raise TypeError(
                "Cannot use .astype to convert from timezone-naive dtype to timezone-aware dtype. "
                "Use tz_localize instead."
            )
        target_unit = (
            np.datetime_data(dtype)[0]
            if isinstance(dtype, np.dtype)
            else dtype.pyarrow_dtype.unit
        )
        if (
            len(self) != self.null_count
            and unit_to_nanoseconds_conversion[target_unit]
            < unit_to_nanoseconds_conversion[self.time_unit]
        ):
            # Casting to a finer resolution multiplies the underlying
            # int64 values and can silently wrap around. pandas
            # bounds-checks every narrowing conversion
            # (astype_overflowsafe) and raises instead. Compare on the
            # integer view: a host round-trip through pd.Timestamp can
            # itself overflow for extreme values.
            lo, hi = self.astype(np.dtype(np.int64)).minmax()
            bound = np.iinfo(np.int64).max // (
                unit_to_nanoseconds_conversion[self.time_unit]
                // unit_to_nanoseconds_conversion[target_unit]
            )
            offender = None
            if hi > bound:
                offender = np.datetime64(int(hi), self.time_unit)  # type: ignore[call-overload]
            elif lo < -bound:
                offender = np.datetime64(int(lo), self.time_unit)  # type: ignore[call-overload]
            if offender is not None:
                raise pd.errors.OutOfBoundsDatetime(
                    f"Out of bounds {_unit_to_name[target_unit]} "
                    f"timestamp: {str(offender).replace('T', ' ')}"
                )
        return self.cast(dtype=dtype)  # type: ignore[return-value]

    def as_timedelta_column(self, dtype: np.dtype) -> None:  # type: ignore[override]
        raise TypeError(
            f"cannot astype a datetimelike from {self.dtype} to {dtype}"
        )

    @functools.cached_property
    def _strftime_names(self) -> plc.Column:
        """Strftime names for %A, %a, %B, %b"""
        return plc.Column.from_iterable_of_py(
            [
                nl_langinfo(loc)
                for loc in (
                    locale.AM_STR,
                    locale.PM_STR,
                    locale.DAY_1,
                    locale.DAY_2,
                    locale.DAY_3,
                    locale.DAY_4,
                    locale.DAY_5,
                    locale.DAY_6,
                    locale.DAY_7,
                    locale.ABDAY_1,
                    locale.ABDAY_2,
                    locale.ABDAY_3,
                    locale.ABDAY_4,
                    locale.ABDAY_5,
                    locale.ABDAY_6,
                    locale.ABDAY_7,
                    locale.MON_1,
                    locale.MON_2,
                    locale.MON_3,
                    locale.MON_4,
                    locale.MON_5,
                    locale.MON_6,
                    locale.MON_7,
                    locale.MON_8,
                    locale.MON_9,
                    locale.MON_10,
                    locale.MON_11,
                    locale.MON_12,
                    locale.ABMON_1,
                    locale.ABMON_2,
                    locale.ABMON_3,
                    locale.ABMON_4,
                    locale.ABMON_5,
                    locale.ABMON_6,
                    locale.ABMON_7,
                    locale.ABMON_8,
                    locale.ABMON_9,
                    locale.ABMON_10,
                    locale.ABMON_11,
                    locale.ABMON_12,
                )
            ]
        )

    def strftime(self, format: str, dtype: DtypeObj) -> StringColumn:
        if len(self) == 0:
            return super().strftime(format, dtype)
        if re.search("%[aAbB]", format):
            names = self._strftime_names
        else:
            names = plc.Column.from_scalar(
                plc.Scalar.from_py(None, plc.DataType(plc.TypeId.STRING)), 0
            )
        with self.access(mode="read", scope="internal"):
            return cast(
                "cudf.core.column.string.StringColumn",
                ColumnBase.create(
                    plc.strings.convert.convert_datetime.from_timestamps(
                        self.plc_column,
                        format,
                        names,
                    ),
                    dtype,
                ),
            )

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        format = _dtype_to_format_conversion.get(
            self.dtype.name, "%Y-%m-%d %H:%M:%S"
        )
        if cudf.get_option("mode.pandas_compatible"):
            if isinstance(dtype, np.dtype) and dtype.kind == "O":
                raise TypeError(
                    f"Cannot astype a datetimelike from {self.dtype} to {dtype}"
                )
            if format.endswith("f"):
                sub_second_res_len = 3
            else:
                sub_second_res_len = 0

            has_nanos = self.time_unit == "ns" and self.nanosecond.any()
            has_micros = (
                self.time_unit in {"ns", "us"} and self.microsecond.any()
            )
            has_millis = (
                self.time_unit in {"ns", "us", "ms"} and self.millisecond.any()
            )
            has_seconds = self.second.any()
            has_minutes = self.minute.any()
            has_hours = self.hour.any()
            if sub_second_res_len:
                if has_nanos:
                    # format should be intact and rest of the
                    # following conditions shouldn't execute.
                    pass
                elif has_micros:
                    format = format[:-sub_second_res_len] + "%6f"
                elif has_millis:
                    format = format[:-sub_second_res_len] + "%3f"
                elif has_seconds or has_minutes or has_hours:
                    format = format[:-4]
                else:
                    format = format.split(" ")[0]
            elif not (has_seconds or has_minutes or has_hours):
                format = format.split(" ")[0]
        if isinstance(dtype, np.dtype) and dtype.kind == "U":
            dtype = np.dtype("object")
        return self.strftime(format, dtype)

    def _binaryop(self, other: ColumnBinaryOperand, op: str) -> ColumnBase:
        reflect, op = self._check_reflected_op(op)

        if isinstance(
            other, (pd.tseries.offsets.Tick, pd.tseries.offsets.Week)
        ):
            other = cudf.DateOffset._from_pandas_ticks_or_weeks(other)
        if isinstance(other, cudf.DateOffset):
            return other._datetime_binop(self, op, reflect=reflect)

        pandas_compatible = cudf.get_option("mode.pandas_compatible")

        # pandas raises TypeError on inequality ops against None / NaN
        # (which are not datetime-like). Equality ops still allow these
        # by returning False/True.
        if (
            pandas_compatible
            and op in {"__lt__", "__le__", "__gt__", "__ge__"}
            and (
                other is None or (isinstance(other, float) and np.isnan(other))
            )
        ):
            raise TypeError(
                f"Invalid comparison between dtype={self.dtype} and "
                f"{type(other).__name__}"
            )

        _raise_on_invalid_ordering_scalar(self.dtype, other, op)
        # pandas treats a bare ``datetime.date`` as not comparable with a
        # datetime64 column — ``==`` yields all False, ``!=`` yields all
        # True. Short-circuit here so cudf doesn't implicitly promote the
        # date to a midnight datetime and produce a spurious match.
        if (
            op in {"__eq__", "__ne__"}
            and isinstance(other, datetime.date)
            and not isinstance(other, datetime.datetime)
        ):
            fill_value = op == "__ne__"
            result = self._all_bools_with_nulls(
                self, bool_fill_value=fill_value
            )
            return result.fillna(fill_value)
        other = self._normalize_binop_operand(other)
        if other is NotImplemented:
            return NotImplemented

        # tz-mismatch detection: pandas raises TypeError when comparing or
        # adding/subtracting tz-naive vs tz-aware datetime-like operands.
        if pandas_compatible:
            self_is_tz = isinstance(self, DatetimeTZColumn)
            # Ternary (True/False/None) is required here: True/False
            # differentiate tz-aware vs tz-naive datetime-like operands, while
            # None means the operand is not datetime-like at all (and so the
            # tz-mismatch check should be skipped).
            other_is_tz_dt: bool | None = None
            if isinstance(other, DatetimeColumn):
                other_is_tz_dt = isinstance(other, DatetimeTZColumn)
            elif isinstance(other, pa.Scalar) and pa.types.is_timestamp(
                other.type
            ):
                other_is_tz_dt = other.type.tz is not None
            if other_is_tz_dt is not None and self_is_tz != other_is_tz_dt:
                if op in {"__lt__", "__le__", "__gt__", "__ge__"}:
                    raise TypeError(
                        f"Invalid comparison between dtype={self.dtype} and "
                        f"{type(other).__name__}"
                    )
                if op in {"__sub__"}:
                    raise TypeError(
                        "Cannot subtract tz-naive and tz-aware "
                        "datetime-like objects"
                    )

        other_is_null_scalar = False
        if reflect:
            lhs = other
            rhs = self
            if isinstance(lhs, pa.Scalar):
                lhs_unit = lhs.type.unit
                other_dtype = cudf_dtype_from_pa_type(lhs.type)
                other_is_null_scalar = is_na_like(lhs)
            else:
                lhs_unit = getattr(lhs, "time_unit", None)
                other_dtype = lhs.dtype
            rhs_unit = getattr(rhs, "time_unit", None)
        else:
            lhs = self
            rhs = other  # type: ignore[assignment]
            if isinstance(rhs, pa.Scalar):
                rhs_unit = rhs.type.unit
                other_dtype = cudf_dtype_from_pa_type(rhs.type)
                other_is_null_scalar = is_na_like(rhs)
            else:
                rhs_unit = getattr(rhs, "time_unit", None)
                other_dtype = rhs.dtype
            lhs_unit = getattr(lhs, "time_unit", None)

        other_is_timedelta = other_dtype.kind == "m"
        other_is_datetime64 = other_dtype.kind == "M"

        # Determine tz awareness of each operand. pandas raises for any
        # datetime-datetime operation (subtraction, ordering comparison)
        # that mixes a tz-aware and a tz-naive operand, and preserves the
        # tz when adding/subtracting a timedelta.
        self_is_tz = is_dtype_obj_datetime_tz(self.dtype)
        other_is_tz = is_dtype_obj_datetime_tz(other_dtype)
        if (
            other_is_datetime64
            and self_is_tz != other_is_tz
            and not other_is_null_scalar
            and op in {"__sub__", "__lt__", "__le__", "__gt__", "__ge__"}
        ):
            raise TypeError(
                "Invalid comparison between tz-naive and tz-aware "
                "datetime-like objects."
            )

        def _preserve_tz(result_dtype: np.dtype) -> DtypeObj:
            # When the operation should yield another datetime and self
            # is tz-aware, carry the tz forward with the resolved unit.
            if self_is_tz and result_dtype.kind == "M":
                tz_dtype = cast("pd.DatetimeTZDtype", self.dtype)
                unit = np.datetime_data(
                    cast("np.dtype[np.datetime64]", result_dtype)
                )[0]
                return pd.DatetimeTZDtype(unit, tz_dtype.tz)
            return result_dtype

        out_dtype = None
        if (
            other_is_datetime64
            and self_is_tz != other_is_tz
            and not other_is_null_scalar
            and op in {"__eq__", "__ne__"}
        ):
            # pandas treats tz-aware and tz-naive datetimes as unequal;
            # short-circuit instead of byte-comparing the UTC values.
            fill_value = op == "__ne__"
            other_col = other if isinstance(other, ColumnBase) else self
            result = self._all_bools_with_nulls(
                other_col, bool_fill_value=fill_value
            )
            return result.fillna(fill_value)
        if (
            op
            in {
                "__ne__",
                "__lt__",
                "__gt__",
                "__le__",
                "__ge__",
            }
            and other_is_datetime64
        ):
            out_dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))
        elif op == "__add__" and (other_is_timedelta or other_is_null_scalar):
            # The only thing we can add to a datetime is a timedelta. This
            # operation is symmetric, i.e. we allow `datetime + timedelta` or
            # `timedelta + datetime`. Both result in DatetimeColumns.
            out_dtype = get_dtype_of_same_kind(
                self.dtype,
                _preserve_tz(
                    np.dtype(
                        f"datetime64[{_resolve_binop_resolution(lhs_unit, rhs_unit)}]"  # type: ignore[arg-type]
                    )
                ),
            )
            if other_is_null_scalar:
                # return a column with all nulls in the size of `self`
                return column_empty(len(self), out_dtype)
        elif op == "__sub__":
            # Subtracting a datetime from a datetime results in a timedelta.
            if other_is_datetime64:
                out_dtype = get_dtype_of_same_kind(
                    self.dtype,
                    np.dtype(
                        f"timedelta64[{_resolve_binop_resolution(lhs_unit, rhs_unit)}]"  # type: ignore[arg-type]
                    ),
                )
                if other_is_null_scalar:
                    # return a column with all nulls in the size of `self`
                    return column_empty(len(self), out_dtype)
            # We can subtract a timedelta from a datetime, but not vice versa.
            # Not only is subtraction antisymmetric (as is normal), it is only
            # well-defined if this operation was not invoked via reflection.
            elif other_is_timedelta and not reflect:
                out_dtype = get_dtype_of_same_kind(
                    self.dtype,
                    _preserve_tz(
                        np.dtype(
                            f"datetime64[{_resolve_binop_resolution(lhs_unit, rhs_unit)}]"  # type: ignore[arg-type]
                        )
                    ),
                )
                if other_is_null_scalar:
                    # return a column with all nulls in the size of `self`
                    return column_empty(len(self), out_dtype)
        elif op in {
            "__eq__",
            "__ne__",
            "NULL_EQUALS",
            "NULL_NOT_EQUALS",
        }:
            out_dtype = get_dtype_of_same_kind(self.dtype, np.dtype(np.bool_))
            if isinstance(other, ColumnBase) and not isinstance(
                other, DatetimeColumn
            ):
                fill_value = op in ("__ne__", "NULL_NOT_EQUALS")
                result = self._all_bools_with_nulls(
                    other, bool_fill_value=fill_value
                )
                result = result.fillna(fill_value)
                return result

        if out_dtype is None:
            return NotImplemented

        lhs_binop: plc.Scalar | ColumnBase = (
            pa_scalar_to_plc_scalar(lhs) if isinstance(lhs, pa.Scalar) else lhs
        )
        rhs_binop: plc.Scalar | ColumnBase = (
            pa_scalar_to_plc_scalar(rhs) if isinstance(rhs, pa.Scalar) else rhs
        )

        result_col = binaryop.binaryop(lhs_binop, rhs_binop, op, out_dtype)
        if out_dtype.kind != "b" and op == "__add__":
            return result_col
        elif out_dtype.kind == "b" and op in _EQUALITY_OPS:
            return result_col.fillna(op == "__ne__")
        else:
            return result_col

    def _find_ambiguous_and_nonexistent(
        self, zone_name: str
    ) -> tuple[NumericalColumn, NumericalColumn] | tuple[bool, bool]:
        """
        Recognize ambiguous and nonexistent timestamps for the given timezone.

        Returns a tuple of columns, both of "bool" dtype and of the same
        size as `self`, that respectively indicate ambiguous and
        nonexistent timestamps in `self` with the value `True`.

        Ambiguous and/or nonexistent timestamps are only possible if any
        transitions occur in the time zone database for the given timezone.
        If no transitions occur, the tuple `(False, False)` is returned.
        """
        transition_times, offsets = _get_tz_data(zone_name)
        offsets = offsets.astype(np.dtype(f"timedelta64[{self.time_unit}]"))

        if len(offsets) == 1:  # no transitions
            return False, False

        transition_times, offsets, old_offsets = (
            transition_times.slice(1, len(transition_times)),
            offsets.slice(1, len(offsets)),
            offsets.slice(0, len(offsets) - 1),
        )

        # Assume we have two clocks at the moment of transition:
        # - Clock 1 is turned forward or backwards correctly
        # - Clock 2 makes no changes
        clock_1 = transition_times + offsets
        clock_2 = transition_times + old_offsets

        # At the start of an ambiguous time period, Clock 1 (which has
        # been turned back) reads less than Clock 2:
        cond = clock_1 < clock_2
        ambiguous_begin = clock_1.apply_boolean_mask(cond)

        # The end of an ambiguous time period is what Clock 2 reads at
        # the moment of transition:
        ambiguous_end = clock_2.apply_boolean_mask(cond)
        ambiguous = self.label_bins(
            left_edge=ambiguous_begin,
            left_inclusive=True,
            right_edge=ambiguous_end,
            right_inclusive=False,
        ).notnull()

        # At the start of a non-existent time period, Clock 2 reads less
        # than Clock 1 (which has been turned forward):
        cond = clock_1 > clock_2
        nonexistent_begin = clock_2.apply_boolean_mask(cond)

        # The end of the non-existent time period is what Clock 1 reads
        # at the moment of transition:
        nonexistent_end = clock_1.apply_boolean_mask(cond)
        nonexistent = self.label_bins(
            left_edge=nonexistent_begin,
            left_inclusive=True,
            right_edge=nonexistent_end,
            right_inclusive=False,
        ).notnull()

        return ambiguous, nonexistent  # type: ignore[return-value]

    def tz_localize(
        self,
        tz: str | None,
        ambiguous: Literal["NaT"] = "NaT",
        nonexistent: Literal["NaT"] = "NaT",
    ) -> DatetimeColumn | DatetimeTZColumn:
        if tz is None:
            return self.copy()
        ambiguous, nonexistent = _check_ambiguous_and_nonexistent(
            ambiguous, nonexistent
        )
        is_arrow = isinstance(self.dtype, pd.ArrowDtype)
        dtype = get_compatible_timezone(pd.DatetimeTZDtype(self.time_unit, tz))
        tzname = dtype.tz.key
        ambiguous_col, nonexistent_col = self._find_ambiguous_and_nonexistent(
            tzname
        )
        localized = self._scatter_by_column(
            self.isnull() | (ambiguous_col | nonexistent_col),
            plc.Scalar.from_py(
                None, dtype=dtype_to_pylibcudf_type(self.dtype)
            ),
        )

        transition_times, offsets = _get_tz_data(tzname)
        # Use the raw cast: transition tables contain sentinel entries
        # beyond the finer units' bounds that intentionally wrap around.
        transition_times_local = (transition_times + offsets).cast(
            localized.dtype
        )
        indices = (
            transition_times_local.searchsorted(localized, side="right") - 1
        )
        offsets_to_utc = offsets.take(indices, nullify=True)
        gmt_data = localized - offsets_to_utc
        if is_arrow:
            dtype = pd.ArrowDtype(pa.timestamp(self.time_unit, tz))  # type: ignore[call-overload]
        result = cast(
            "DatetimeTZColumn", ColumnBase.create(gmt_data.plc_column, dtype)
        )
        # Avoid re-computing local times from UTC times
        result._local_time = localized
        return result

    def tz_convert(self, tz: str | None) -> DatetimeColumn:
        raise TypeError(
            "Cannot convert tz-naive timestamps, use tz_localize to localize"
        )


class DatetimeTZColumn(DatetimeColumn):
    def _round_dt(
        self,
        round_func: Callable[
            [plc.Column, plc.datetime.RoundingFrequency], plc.Column
        ],
        freq: str,
    ) -> ColumnBase:
        # Rounding must operate on the local wall-clock time and then be
        # re-localized to the original timezone to match pandas semantics.
        rounded_local = cast(
            "DatetimeColumn", self._local_time._round_dt(round_func, freq)
        )
        return rounded_local.tz_localize(str(self.tz))

    def to_pandas(
        self,
        *,
        nullable: bool = False,
        arrow_type: bool = False,
    ) -> pd.Index:
        if arrow_type or isinstance(self.dtype, pd.ArrowDtype):
            return super().to_pandas(nullable=nullable, arrow_type=arrow_type)
        elif nullable and not arrow_type:
            raise NotImplementedError(
                "There is no nullable pandas type for datetime with timezone."
            )
        else:
            # TODO: Using self._utc_time.to_pandas().tz_localize("UTC").tz_convert(self.tz)
            # would be the more definitive conversion
            if "_local_time" in self.__dict__:
                # _local_time was explicitly cached (e.g., by tz_localize) and holds
                # the original naive timestamps; re-localizing with pandas is correct.
                return self._local_time.to_pandas().tz_localize(
                    self.tz,
                    ambiguous="NaT",
                    nonexistent="NaT",
                )
            # When _local_time is not cached, computing it via searchsorted on the
            # timezone transition table can return wrong results due to the table
            # not being globally sorted.  Use UTC directly instead.
            return (
                self._utc_time.to_pandas()
                .tz_localize("UTC")
                .tz_convert(self.tz)
            )

    @functools.cached_property
    def time_unit(self) -> str:
        if isinstance(self.dtype, pd.DatetimeTZDtype):
            return self.dtype.unit
        return super().time_unit

    @property
    def _utc_time(self) -> DatetimeColumn:
        """Return UTC time as naive timestamps."""
        if isinstance(self.dtype, pd.ArrowDtype):
            target_dtype: DtypeObj = pd.ArrowDtype(
                pa.timestamp(self.time_unit)  # type: ignore[call-overload]
            )
        else:
            target_dtype = _get_base_dtype(self.dtype)
        return cast(
            "DatetimeColumn",
            DatetimeColumn.create(
                self.plc_column, target_dtype, validate=False
            ),
        )

    @functools.cached_property
    def _local_time(self) -> DatetimeColumn:
        """Return the local time as naive timestamps."""
        transition_times, offsets = _get_tz_data(str(self.tz))
        base_dtype = _get_base_dtype(self.dtype)
        indices = (
            # Use the raw cast: transition tables contain sentinel
            # entries beyond the finer units' bounds that intentionally
            # wrap around.
            transition_times.cast(base_dtype).searchsorted(
                self._utc_time.astype(base_dtype), side="right"
            )
            - 1
        )
        offsets_from_utc = offsets.take(indices, nullify=True)
        # Perform the add on the tz-naive UTC view so the result is naive
        # (``self + offsets`` now preserves tz by default). The local-time
        # column must be tz-naive to represent the wall-clock values.
        return cast("DatetimeColumn", self._utc_time + offsets_from_utc)

    def as_string_column(self, dtype: DtypeObj) -> StringColumn:
        return self._local_time.as_string_column(dtype)

    def strftime(self, format: str, dtype: DtypeObj) -> StringColumn:
        if len(self) == 0:
            return super().strftime(format, dtype)
        if re.search(r"(?<!%)%[zZ]", format):
            # The cudf strftime kernel operates on raw timestamps and has no
            # knowledge of the column's timezone, so it would format ``%z`` as
            # ``+0000`` and ``%Z`` as ``UTC``. Materialize through pandas to
            # get correct local times with timezone designators. This is the
            # slow path but it matches pandas semantics including DST.
            pd_result = self.to_pandas().strftime(format)
            return cast(
                "StringColumn",
                as_column(pd_result, dtype=dtype),
            )
        # No tz designator in the format; formatting on the local-time view
        # (wall-clock, tz-naive) produces the right answer.
        return self._local_time.strftime(format, dtype)

    def as_datetime_column(
        self, dtype: pd.ArrowDtype | pd.DatetimeTZDtype
    ) -> DatetimeColumn:
        if isinstance(dtype, pd.DatetimeTZDtype):
            dtype = get_compatible_timezone(dtype)
            if dtype.unit != self.time_unit:
                casted_plc = (
                    super()
                    .as_datetime_column(_get_base_dtype(dtype))
                    .plc_column
                )
                casted = cast(
                    "DatetimeTZColumn", ColumnBase.create(casted_plc, dtype)
                )
            else:
                casted = self
            return casted.tz_convert(str(dtype.tz))
        elif isinstance(dtype, pd.ArrowDtype):
            if dtype.pyarrow_dtype.unit != self.time_unit:
                casted_plc = (
                    super()
                    .as_datetime_column(_get_base_dtype(dtype))
                    .plc_column
                )
                casted = cast(
                    "DatetimeTZColumn", ColumnBase.create(casted_plc, dtype)
                )
            else:
                casted = self
            if dtype.pyarrow_dtype.tz != self.tz:
                raise NotImplementedError(
                    "Casting to a different timezone with ArrowDtype is not supported."
                )
            return casted
        raise TypeError(
            "Cannot use .astype to convert from timezone-aware dtype to timezone-naive dtype. "
            "Use tz_localize(None) instead."
        )

    def _get_dt_field(
        self, field: plc.datetime.DatetimeComponent
    ) -> ColumnBase:
        with self._local_time.access(
            mode="read", scope="internal"
        ) as local_time:
            plc_result = plc.datetime.extract_datetime_component(
                local_time.plc_column,
                field,
            )
            result = ColumnBase.create(
                plc_result, dtype_from_pylibcudf_column(plc_result)
            )
            # cast to int32 for pandas compatibility (no-op if already int32)
            result = result.astype(
                get_dtype_of_same_kind(self.dtype, np.dtype("int32"))
            )
            return result

    def __repr__(self) -> str:
        # Arrow prints the UTC timestamps, but we want to print the
        # local timestamps:
        arr = self._local_time.to_arrow().cast(
            pa.timestamp(self.time_unit, str(self.tz))
        )
        return (
            f"{object.__repr__(self)}\n{arr.to_string()}\ndtype: {self.dtype}"
        )

    def tz_localize(
        self,
        tz: str | None,
        ambiguous: Literal["NaT"] = "NaT",
        nonexistent: Literal["NaT"] = "NaT",
    ) -> DatetimeColumn:
        if tz is None:
            return self._local_time
        ambiguous, nonexistent = _check_ambiguous_and_nonexistent(
            ambiguous, nonexistent
        )
        raise ValueError(
            "Already localized. "
            "Use `tz_convert` to convert between time zones."
        )

    def tz_convert(self, tz: str | None) -> DatetimeColumn | DatetimeTZColumn:
        if tz is None:
            return self._utc_time
        elif tz == str(self.tz):
            return self.copy()

        if isinstance(self.dtype, pd.ArrowDtype):
            target_dtype: DtypeObj = pd.ArrowDtype(
                pa.timestamp(self.time_unit, tz)  # type: ignore[call-overload]
            )
        else:
            target_dtype = get_compatible_timezone(
                pd.DatetimeTZDtype(self.time_unit, tz)
            )
        return cast(
            "DatetimeTZColumn",
            ColumnBase.create(self.plc_column, target_dtype),
        )

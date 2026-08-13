# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import math
import re
import warnings
from decimal import Decimal
from functools import lru_cache
from typing import TYPE_CHECKING, Literal, Self, cast

import numpy as np
import pandas as pd
import pandas.tseries.offsets as pd_offset
import pyarrow as pa

import pylibcudf as plc

from cudf.api.types import is_integer, is_scalar
from cudf.core.column.column import ColumnBase, as_column
from cudf.core.dataframe import DataFrame
from cudf.core.index import DatetimeIndex, Index, ensure_index
from cudf.core.series import Series
from cudf.utils.dtypes import DEFAULT_STRING_DTYPE, dtype_from_pylibcudf_column
from cudf.utils.scalar import pa_scalar_to_plc_scalar
from cudf.utils.temporal import infer_format, unit_to_nanoseconds_conversion

if TYPE_CHECKING:
    from cudf.core.column.datetime import DatetimeColumn
    from cudf.core.column.string import StringColumn


# https://github.com/pandas-dev/pandas/blob/2.2.x/pandas/core/tools/datetimes.py#L1112
_unit_map = {
    "year": "year",
    "years": "year",
    "month": "month",
    "months": "month",
    "day": "day",
    "days": "day",
    "hour": "h",
    "hours": "h",
    "minute": "m",
    "minutes": "m",
    "second": "s",
    "seconds": "s",
    "ms": "ms",
    "millisecond": "ms",
    "milliseconds": "ms",
    "us": "us",
    "microsecond": "us",
    "microseconds": "us",
    "ns": "ns",
    "nanosecond": "ns",
    "nanoseconds": "ns",
}

_unit_dtype_map = {
    "ns": "datetime64[ns]",
    "us": "datetime64[us]",
    "ms": "datetime64[ms]",
    "m": "datetime64[s]",
    "h": "datetime64[s]",
    "s": "datetime64[s]",
    "D": "datetime64[s]",
}


_YEAR_DIRECTIVES = set("Yy")
_DAY_DIRECTIVES = set("dej")
_WEEKDAY_DIRECTIVES = set("Aawu")


def _validate_format_directives(format: str) -> None:
    """Raise ValueError for incompatible strftime directive combinations."""
    directives: set[str] = set()
    i = 0
    while i < len(format):
        if format[i] == "%" and i + 1 < len(format):
            directives.add(format[i + 1])
            i += 2
        else:
            i += 1
    has_G = "G" in directives
    has_V = "V" in directives
    has_Y = bool(directives & _YEAR_DIRECTIVES)
    has_weekday = bool(directives & _WEEKDAY_DIRECTIVES)
    has_day = bool(directives & _DAY_DIRECTIVES)
    has_j = "j" in directives

    if has_G and has_j:
        raise ValueError(
            "Day of the year directive '%j' is not compatible with ISO "
            "year directive '%G'. Use '%Y' instead."
        )
    if has_V and has_Y:
        raise ValueError(
            "ISO week directive '%V' is incompatible with the year "
            "directive '%Y'. Use the ISO year '%G' instead."
        )
    if has_V and not (has_G and has_weekday):
        raise ValueError(
            "ISO week directive '%V' must be used with the ISO year "
            "directive '%G' and a weekday directive '%A', '%a', "
            "'%w', or '%u'."
        )
    if has_G and not (has_V and has_weekday):
        raise ValueError(
            "ISO year directive '%G' must be used with the ISO week "
            "directive '%V' and a weekday directive '%A', '%a', "
            "'%w', or '%u'."
        )
    if ("W" in directives or "U" in directives) and not (
        has_Y and (has_day or has_weekday)
    ):
        # %W or %U requires both a year directive and either a day-of-month
        # or weekday directive to identify a specific date.
        raise ValueError("Cannot use '%W' or '%U' without day and year")


def to_datetime(
    arg,
    errors: Literal["raise", "coerce", "warn", "ignore"] = "raise",
    dayfirst: bool = False,
    yearfirst: bool = False,
    utc: bool = False,
    format: str | None = None,
    exact: bool = True,
    unit: str | None = None,
    origin="unix",
    cache: bool = True,
):
    """
    Convert argument to datetime.

    Parameters
    ----------
    arg : int, float, str, datetime, list, tuple, 1-d array,
        Series DataFrame/dict-like
        The object to convert to a datetime.
    errors : {'ignore', 'raise', 'coerce', 'warn'}, default 'raise'
        - If 'raise', then invalid parsing will raise an exception.
        - If 'coerce', then invalid parsing will be set as NaT.
        - If 'warn' : prints last exceptions as warnings and
            return the input.
        - If 'ignore', then invalid parsing will return the input.
    dayfirst : bool, default False
        Specify a date parse order if `arg` is str or its list-likes.
        If True, parses dates with the day first, eg 10/11/12 is parsed as
        2012-11-10.
        Warning: dayfirst=True is not strict, but will prefer to parse
        with day first (this is a known bug, based on dateutil behavior).
    utc : bool, default False
        Whether the result should be have a UTC timezone.
    format : str, default None
        The strftime to parse time, eg "%d/%m/%Y", note that "%f" will parse
        all the way up to nanoseconds.
        See strftime documentation for more information on choices:
        https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior.
    unit : str, default 'ns'
        The unit of the arg (D,s,ms,us,ns) denote the unit, which is an
        integer or float number. This will be based off the
        origin(unix epoch start).
        Example, with unit='ms' and origin='unix' (the default), this
        would calculate the number of milliseconds to the unix epoch start.

    Returns
    -------
    datetime
        If parsing succeeded.
        Return type depends on input:
        - list-like: DatetimeIndex
        - Series: Series of datetime64 dtype
        - scalar: Timestamp

    Examples
    --------
    Assembling a datetime from multiple columns of a DataFrame. The keys can be
    common abbreviations like ['year', 'month', 'day', 'minute', 'second',
    'ms', 'us', 'ns']) or plurals of the same

    >>> import cudf
    >>> df = cudf.DataFrame({'year': [2015, 2016],
    ...                    'month': [2, 3],
    ...                    'day': [4, 5]})
    >>> cudf.to_datetime(df)
    0   2015-02-04
    1   2016-03-05
    dtype: datetime64[us]
    >>> cudf.to_datetime(1490195805, unit='s')
    Timestamp('2017-03-22 15:16:45')
    >>> cudf.to_datetime(1490195805433502912, unit='ns')
    Timestamp('2017-03-22 15:16:45.433502912')
    """
    if errors not in {"ignore", "raise", "coerce", "warn"}:
        raise ValueError(
            f"errors parameter has to be either one of: "
            f"{['ignore', 'raise', 'coerce', 'warn']}, found: "
            f"{errors}"
        )
    elif errors in {"ignore", "coerce"} and not is_scalar(arg):
        raise NotImplementedError(
            f"{errors=} is not implemented when arg is not scalar-like"
        )

    if errors == "ignore":
        warnings.warn(
            "errors='ignore' is deprecated and will raise in a "
            "future version. Use to_datetime without passing `errors` "
            "and catch exceptions explicitly instead",
            FutureWarning,
        )

    if arg is None:
        return pd.NaT

    if exact is False:
        raise NotImplementedError("exact support is not yet implemented")

    if origin != "unix":
        raise NotImplementedError("origin support is not yet implemented")

    if yearfirst:
        raise NotImplementedError("yearfirst support is not yet implemented")

    if format is not None:
        if "%Z" in format or "%z" in format:
            raise NotImplementedError(
                "cuDF does not yet support timezone-aware datetimes"
            )
        _validate_format_directives(format)
        if "%f" in format:
            format = format.replace("%f", "%9f")

    if isinstance(arg, bool):
        if errors == "coerce":
            return pd.NaT
        elif errors == "raise":
            raise TypeError("dtype bool cannot be converted to datetime64[ns]")

    if not is_scalar(arg) and not isinstance(arg, DataFrame):
        try:
            iterator = iter(arg)
        except TypeError:
            iterator = None
        if iterator is not None:
            try:
                first = next(iterator)
            except StopIteration:
                first = None
            if first is not None and isinstance(first, Decimal):
                raise TypeError(
                    "<class 'decimal.Decimal'> is not convertible to datetime"
                )

    if unit in ("Y", "M"):
        # cuDF does not support calendrical addition for to_datetime
        # with unit='Y' or 'M'. Defer to pandas via the cudf.pandas
        # fast/slow proxy fallback.
        raise NotImplementedError(
            f"unit={unit!r} is not supported in cudf.to_datetime"
        )

    try:
        if isinstance(arg, DataFrame):
            # we require at least Ymd
            required = ["year", "month", "day"]
            req = list(set(required) - set(arg._column_names))
            if len(req):
                err_req = ",".join(req)
                raise ValueError(
                    f"to assemble mappings requires at least that "
                    f"[year, month, day] be specified: [{err_req}] "
                    f"is missing"
                )

            # replace passed column name with values in _unit_map
            got_units = {k: get_units(k) for k in arg._column_names}
            unit_rev = {v: k for k, v in got_units.items()}

            # keys we don't recognize
            excess = set(unit_rev.keys()) - set(_unit_map.values())
            if len(excess):
                err_excess = ",".join(excess)
                raise ValueError(
                    f"extra keys have been passed to the "
                    f"datetime assemblage: [{err_excess}]"
                )

            new_series = (
                arg[unit_rev["year"]].astype(DEFAULT_STRING_DTYPE)
                + "-"
                + arg[unit_rev["month"]]
                .astype(DEFAULT_STRING_DTYPE)
                .str.zfill(2)
                + "-"
                + arg[unit_rev["day"]]
                .astype(DEFAULT_STRING_DTYPE)
                .str.zfill(2)
            )
            format = "%Y-%m-%d"
            target_unit = "us"
            for u in ["h", "m", "s", "ms", "us", "ns"]:
                value = unit_rev.get(u)
                if value is not None and value in arg:
                    arg_col = arg._data[value]
                    if arg_col.dtype.kind == "f":
                        target_unit = "ns"
                        break
                    elif arg_col.dtype.kind == "O":
                        string_col = cast("StringColumn", arg_col)
                        if not string_col.is_all_integer():
                            target_unit = "ns"
                            break
                    elif u == "ns":
                        # An explicit nanosecond field forces ns precision
                        # (pandas widens to [ns] when ns is present).
                        target_unit = "ns"
            col = new_series._column.strptime(
                np.dtype(f"datetime64[{target_unit}]"), format=format
            )

            times_column = None
            col_unit = np.datetime_data(col.dtype)[0]
            factor_denominator = unit_to_nanoseconds_conversion.get(
                col_unit, 1
            )
            for u in ["h", "m", "s", "ms", "us", "ns"]:
                value = unit_rev.get(u)
                if value is not None and value in arg:
                    current_col = arg._data[value]
                    # If the arg[value] is of int or
                    # float dtype we don't want to type-cast
                    if current_col.dtype.kind in ("O"):
                        try:
                            current_col = current_col.astype(
                                np.dtype(np.int64)
                            )
                        except ValueError:
                            current_col = current_col.astype(
                                np.dtype(np.float64)
                            )

                    factor_numerator = unit_to_nanoseconds_conversion[u]
                    factor: int | float
                    if factor_numerator % factor_denominator == 0:
                        factor = factor_numerator // factor_denominator
                    else:
                        factor = factor_numerator / factor_denominator

                    if times_column is None:
                        times_column = current_col * factor
                    else:
                        times_column = times_column + (current_col * factor)
            if times_column is not None:
                col = (col.astype(np.dtype(np.int64)) + times_column).astype(
                    dtype=col.dtype
                )
            col = _process_col(
                col=col,
                unit=unit,
                dayfirst=dayfirst,
                format=format,
                utc=utc,
            )
            return Series._from_column(col, index=arg.index)
        else:
            col = _process_col(
                col=as_column(arg),
                unit=unit,
                dayfirst=dayfirst,
                format=format,
                utc=utc,
            )
            if isinstance(arg, (Index, pd.Index)):
                return DatetimeIndex._from_column(col, name=arg.name)
            elif isinstance(arg, (Series, pd.Series)):
                return Series._from_column(
                    col, name=arg.name, index=ensure_index(arg.index)
                )
            elif is_scalar(arg):
                return col.element_indexing(0)
            else:
                return Index._from_column(col)
    except Exception as e:
        if errors == "raise":
            raise e
        elif errors == "warn":
            import traceback

            tb = traceback.format_exc()
            warnings.warn(tb)
        elif errors == "ignore":
            pass
        elif errors == "coerce":
            if is_scalar(arg):
                return pd.NaT
            return np.datetime64("nat", "ns" if unit is None else unit)  # type: ignore[call-overload]
        return arg


def _process_col(
    col,
    unit: str | None,
    dayfirst: bool,
    format: str | None,
    utc: bool,
):
    if col.dtype.kind == "f":
        if unit not in (None, "ns"):
            col = col * unit_to_nanoseconds_conversion[unit]

        if format is not None:
            # Converting to int because,
            # pandas actually creates a datetime column
            # out of float values and then creates an
            # int column out of it to parse against `format`.
            # Instead we directly cast to int and perform
            # parsing against `format`.
            # Pandas 3 defaults parsed datetimes to `datetime64[us]`
            # regardless of format precision.
            col = (
                col.astype(np.dtype(np.int64))
                .astype(DEFAULT_STRING_DTYPE)
                .strptime(
                    dtype=np.dtype("datetime64[us]"),
                    format=format,
                )
            )
        else:
            col = col.astype(dtype=np.dtype("datetime64[ns]"))

    elif col.dtype.kind in "iu":
        if unit in ("D", "h", "m"):
            factor = (
                unit_to_nanoseconds_conversion[unit]
                / unit_to_nanoseconds_conversion["s"]
            )
            col = col * factor

        if format is not None:
            if unit is None:
                unit = "us"
            col = col.astype(DEFAULT_STRING_DTYPE).strptime(
                dtype=np.dtype(_unit_dtype_map[unit]), format=format
            )
        else:
            if len(col) == 0 and unit is None:
                unit = "s"
            elif unit is None:
                unit = "ns"

            col = col.astype(
                dtype=np.dtype(
                    _unit_dtype_map.get(unit, _unit_dtype_map["us"])
                )
            )

    elif col.dtype.kind == "O":
        if col.is_all_null:
            # Pandas converts all-null inputs to NaT at second precision
            # regardless of `unit`/`format`; mirror that here without
            # routing through the int/float path (which would land on
            # the [ns]/[us] defaults).
            return col.astype(np.dtype("datetime64[s]"))
        if unit not in (None, "ns"):
            try:
                col = col.astype(np.dtype(np.int64))
            except ValueError:
                col = col.astype(np.dtype(np.float64))
            return _process_col(
                col=col,
                unit=unit,
                dayfirst=dayfirst,
                format=format,
                utc=utc,
            )
        else:
            if format is not None and "f" in format and unit is None:
                col_us = col.strptime(
                    dtype=np.dtype(_unit_dtype_map["us"]), format=format
                )
                try:
                    col_ns = col.strptime(
                        dtype=np.dtype(_unit_dtype_map["ns"]), format=format
                    )
                except pd.errors.OutOfBoundsDatetime:
                    # Values beyond the nanosecond Timestamp range: keep
                    # microsecond precision, like pandas' unit inference.
                    col = col_us
                else:
                    res = col_ns != col_us
                    if res.any():
                        col = col_ns
                    else:
                        col = col_us
            else:
                if format is None:
                    format = infer_format(
                        element=col.element_indexing(0),
                        dayfirst=dayfirst,
                    )
                target_unit = unit
                if unit is None:
                    # pandas infers nanosecond precision from strings with
                    # more than 6 fractional-second digits
                    subsecond = re.search(r"%(\d)f", format)
                    if subsecond is not None and int(subsecond.group(1)) > 6:
                        target_unit = "ns"
                col = col.strptime(
                    dtype=np.dtype(
                        _unit_dtype_map.get(target_unit, _unit_dtype_map["us"])  # type: ignore[arg-type]
                    ),
                    format=format,
                )
    elif col.dtype.kind != "M":
        raise TypeError(
            f"dtype {col.dtype} cannot be converted to {_unit_dtype_map.get(unit, _unit_dtype_map['us'])}"  # type: ignore[arg-type]
        )
    if utc and not isinstance(col.dtype, pd.DatetimeTZDtype):
        return col.tz_localize("UTC")
    return col


def get_units(value):
    if value in _unit_map:
        return _unit_map[value]

    # m is case significant
    if value.lower() in _unit_map:
        return _unit_map[value.lower()]

    return value


# for pandas compatibility
class MonthEnd:
    def _maybe_as_fast_pandas_offset(self):
        return pd._libs.tslibs.offsets.MonthEnd()

    def __eq__(self, other):
        return self._maybe_as_fast_pandas_offset() == other


class YearEnd:
    def _maybe_as_fast_pandas_offset(self):
        return pd._libs.tslibs.offsets.YearEnd(month=12)

    def __eq__(self, other):
        return self._maybe_as_fast_pandas_offset() == other


class DateOffset:
    """
    An object used for binary ops where calendrical arithmetic
    is desired rather than absolute time arithmetic. Used to
    add or subtract a whole number of periods, such as several
    months or years, to a series or index of datetime dtype.
    Works similarly to pd.DateOffset, but stores the offset
    on the device (GPU).

    Parameters
    ----------
    n : int, default 1
        The number of time periods the offset represents.
    **kwds
        Temporal parameter that add to or replace the offset value.
        Parameters that **add** to the offset (like Timedelta):
        - months

    See Also
    --------
    pandas.tseries.offsets.DateOffset : The equivalent Pandas object that this
        object replicates.

    Examples
    --------
    >>> from cudf import DateOffset
    >>> ts = cudf.Series([
    ...     "2000-01-01 00:00:00.012345678",
    ...     "2000-01-31 00:00:00.012345678",
    ...     "2000-02-29 00:00:00.012345678",
    ... ], dtype='datetime64[ns]')
    >>> ts + DateOffset(months=3)
    0   2000-04-01 00:00:00.012345678
    1   2000-04-30 00:00:00.012345678
    2   2000-05-29 00:00:00.012345678
    dtype: datetime64[ns]
    >>> ts - DateOffset(months=12)
    0   1999-01-01 00:00:00.012345678
    1   1999-01-31 00:00:00.012345678
    2   1999-02-28 00:00:00.012345678
    dtype: datetime64[ns]

    Notes
    -----
    Note that cuDF does not yet support DateOffset arguments
    that 'replace' units in the datetime data being operated on
    such as:

    - year
    - month
    - week
    - day
    - hour
    - minute
    - second
    - microsecond
    - millisecond
    - nanosecond

    cuDF does not yet support rounding via a `normalize`
    keyword argument.
    """

    _UNITS_TO_CODES = {
        "nanoseconds": "ns",
        "microseconds": "us",
        "milliseconds": "ms",
        "seconds": "s",
        "minutes": "m",
        "hours": "h",
        "days": "D",
        "weeks": "W",
        "months": "M",
        "years": "Y",
    }

    _CODES_TO_UNITS = {
        "N": "nanoseconds",
        "ns": "nanoseconds",
        "U": "microseconds",
        "us": "microseconds",
        "ms": "milliseconds",
        "L": "milliseconds",
        "s": "seconds",
        "S": "seconds",
        "m": "minutes",
        "min": "minutes",
        "T": "minutes",
        "h": "hours",
        "H": "hours",
        "D": "days",
        "W": "weeks",
        "M": "months",
        "Y": "years",
    }

    _TICK_OR_WEEK_TO_UNITS = {
        pd_offset.Week: "weeks",
        pd_offset.Day: "days",
        pd_offset.Hour: "hours",
        pd_offset.Minute: "minutes",
        pd_offset.Second: "seconds",
        pd_offset.Milli: "milliseconds",
        pd_offset.Micro: "microseconds",
        pd_offset.Nano: "nanoseconds",
    }

    _FREQSTR_REGEX = re.compile("([-+]?[0-9]*)([a-zA-Z]+)")

    def __eq__(self, other):
        if isinstance(other, DateOffset):
            return self.kwds == other.kwds
        # Compare via the equivalent fast pandas offset so freq strings and
        # pandas offsets (including cudf.pandas-proxied ones) compare equal.
        return self._maybe_as_fast_pandas_offset() == other

    def __init__(self, n=1, normalize=False, **kwds):
        if normalize:
            raise NotImplementedError(
                "normalize not yet supported for DateOffset"
            )

        all_possible_units = {
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
            "year",
            "month",
            "week",
            "day",
            "hour",
            "minute",
            "second",
            "microsecond",
            "millisecond",
            "nanosecond",
        }

        supported_units = {
            "years",
            "months",
            "weeks",
            "days",
            "hours",
            "minutes",
            "seconds",
            "milliseconds",
            "microseconds",
            "nanoseconds",
        }

        unsupported_units = all_possible_units - supported_units

        invalid_kwds = set(kwds) - supported_units - unsupported_units
        if invalid_kwds:
            raise TypeError(
                f"Keyword arguments '{','.join(list(invalid_kwds))}'"
                " are not recognized"
            )

        unsupported_kwds = set(kwds) & unsupported_units
        if unsupported_kwds:
            raise NotImplementedError(
                f"Keyword arguments '{','.join(list(unsupported_kwds))}'"
                " are not yet supported."
            )

        if any(not is_integer(val) for val in kwds.values()):
            raise ValueError("Non-integer periods not supported")

        self._kwds = kwds
        kwds = self._combine_months_and_years(**kwds)
        kwds = self._combine_kwargs_to_seconds(**kwds)

        scalars = {}
        for k, v in kwds.items():
            if k in all_possible_units:
                # Months must be int16 or int32
                if k == "months":
                    # TODO: throw for out-of-bounds int32 values
                    scalars[k] = np.int32(v)
                else:
                    unit = self._UNITS_TO_CODES[k]
                    scalars[k] = np.timedelta64(v, unit)

        self._scalars = scalars

    @property
    def kwds(self):
        return self._kwds

    def _combine_months_and_years(self, **kwargs):
        # TODO: if months is zero, don't do a binop
        kwargs["months"] = kwargs.pop("years", 0) * 12 + kwargs.pop(
            "months", 0
        )
        return kwargs

    def _combine_kwargs_to_seconds(self, **kwargs):
        """
        Combine days, weeks, hours and minutes to a single
        scalar representing the total seconds
        """
        seconds = 0
        seconds += kwargs.pop("weeks", 0) * 604800
        seconds += kwargs.pop("days", 0) * 86400
        seconds += kwargs.pop("hours", 0) * 3600
        seconds += kwargs.pop("minutes", 0) * 60
        seconds += kwargs.pop("seconds", 0)

        if seconds > np.iinfo("int64").max:
            raise NotImplementedError(
                "Total days + weeks + hours + minutes + seconds can not exceed"
                f" {np.iinfo('int64').max} seconds"
            )

        if seconds != 0:
            kwargs["seconds"] = seconds
        return kwargs

    def _datetime_binop(
        self, datetime_col, op, reflect=False
    ) -> DatetimeColumn:
        if reflect and op == "__sub__":
            raise TypeError(
                f"Can not subtract a {type(datetime_col).__name__}"
                f" from a {type(self).__name__}"
            )
        if op not in {"__add__", "__sub__"}:
            raise TypeError(
                f"{op} not supported between {type(self).__name__}"
                f" and {type(datetime_col).__name__}"
            )
        if not self._is_no_op:
            tz = (
                datetime_col.dtype.tz
                if isinstance(datetime_col.dtype, pd.DatetimeTZDtype)
                else None
            )

            for unit, value in self._scalars.items():
                value = -value if op == "__sub__" else value
                if unit == "months":
                    with datetime_col.access(mode="read", scope="internal"):
                        plc_column = plc.datetime.add_calendrical_months(
                            datetime_col.plc_column,
                            pa_scalar_to_plc_scalar(pa.scalar(value)),
                        )
                        datetime_col = ColumnBase.create(
                            plc_column,
                            dtype=dtype_from_pylibcudf_column(plc_column),
                        )
                else:
                    datetime_col += as_column(value, length=len(datetime_col))

            if tz is not None and not isinstance(
                datetime_col.dtype, pd.DatetimeTZDtype
            ):
                # Older arithmetic stripped the tz; restore it. When the
                # arithmetic already preserves tz (e.g. datetime+timedelta),
                # the column is already tz-aware and another tz_localize
                # would raise "Already tz-aware".
                datetime_col = datetime_col.tz_localize("UTC").tz_convert(tz)

        return datetime_col

    @property
    def _is_no_op(self) -> bool:
        # some logic could be implemented here for more complex cases
        # such as +1 year, -12 months
        return all(i == 0 for i in self._kwds.values())

    def __neg__(self):
        new_scalars = {k: -v for k, v in self._kwds.items()}
        return DateOffset(**new_scalars)

    def __repr__(self):
        includes = []
        for unit in sorted(self._UNITS_TO_CODES):
            val = self._kwds.get(unit, None)
            if val is not None:
                includes.append(f"{unit}={val}")
        unit_data = ", ".join(includes)
        repr_str = f"<{self.__class__.__name__}: {unit_data}>"

        return repr_str

    @classmethod
    @lru_cache(maxsize=128)
    def _from_freqstr(cls, freqstr: str) -> Self | MonthEnd | YearEnd:
        """
        Parse a string and return a DateOffset object
        expects strings of the form 3D, 25W, 10ms, 42ns, etc.
        """
        match = cls._FREQSTR_REGEX.match(freqstr)

        if match is None:
            raise ValueError(f"Invalid frequency string: {freqstr}")

        numeric_part = match.group(1)
        if numeric_part == "":
            numeric_part = "1"
        freq_part = match.group(2)

        # Certain frequency strings are deprecated in pandas
        # and automatically swapped on construction
        if freq_part in ("M", "ME"):
            return MonthEnd()
        elif freq_part in ("Y", "YE"):
            return YearEnd()

        if freq_part not in cls._CODES_TO_UNITS:
            raise ValueError(f"Cannot interpret frequency str: {freqstr}")

        return cls(**{cls._CODES_TO_UNITS[freq_part]: int(numeric_part)})

    @classmethod
    def _from_pandas_ticks_or_weeks(
        cls,
        tick: pd.tseries.offsets.Tick | pd.tseries.offsets.Week,
    ) -> Self:
        return cls(**{cls._TICK_OR_WEEK_TO_UNITS[type(tick)]: tick.n})

    def _maybe_as_fast_pandas_offset(self):
        if (
            len(self.kwds) == 1
            and _has_fixed_frequency(self)
            and not _has_non_fixed_frequency(self)
        ):
            # Pandas computation between `n*offsets.Minute()` is faster than
            # `n*DateOffset`. If only single offset unit is in use, we return
            # the base offset for faster binary ops.
            # Use the pandas offset class directly to avoid type mismatches
            # (e.g. pd.Timedelta(days=1) -> <24 * Hours> instead of <Day>).
            unit, n = next(iter(self.kwds.items()))
            units_to_offset = {
                v: k for k, v in self._TICK_OR_WEEK_TO_UNITS.items()
            }
            offset_cls = units_to_offset.get(unit)
            if offset_cls is not None:
                return offset_cls(n)
        return pd.DateOffset(**self.kwds, n=1)


def date_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    tz=None,
    normalize: bool = False,
    name=None,
    inclusive: Literal["left", "right", "both", "neither"] = "both",
    *,
    unit: str | None = None,
):
    """Return a fixed frequency DatetimeIndex.

    Returns the range of equally spaced time points (where the difference
    between any two adjacent points is specified by the given frequency)
    such that they all satisfy `start` <[=] x <[=] `end`, where the first one
    and the last one are, resp., the first and last time points in that range
    that are valid for `freq`.

    Parameters
    ----------
    start : str or datetime-like, optional
        Left bound for generating dates.

    end : str or datetime-like, optional
        Right bound for generating dates.

    periods : int, optional
        Number of periods to generate.

    freq : str or DateOffset
        Frequencies to generate the datetime series. Mixed fixed-frequency and
        non-fixed frequency offset is unsupported. See notes for detail.
        Supported offset alias: ``D``, ``h``, ``H``, ``T``, ``min``, ``S``,
        ``U``, ``us``, ``N``, ``ns``.

    tz : str or tzinfo, optional
        Time zone name for returning localized DatetimeIndex.

    normalize : bool, default False
        Not Supported

    name : str, default None
        Name of the resulting DatetimeIndex

    inclusive : {"left", "right", "both", "neither"}, default "both"
        Whether to set each bound as closed or open.
        Currently only "both" is supported

    unit : str, default None
        Specify the desired resolution of the result. If not specified,
        the unit is inferred from `freq` (defaults to 'us').

    Returns
    -------
    DatetimeIndex

    Notes
    -----
    Of the four parameters `start`, `end`, `periods`, and `freq`, exactly three
    must be specified. If `freq` is omitted, the resulting DatetimeIndex will
    have periods linearly spaced elements between start and end (inclusive on both
    sides).

    cudf supports `freq` specified with either fixed-frequency offset
    (such as weeks, days, hours, minutes...) or non-fixed frequency offset
    (such as years and months). Specifying `freq` with a mixed fixed and
    non-fixed frequency is currently unsupported. For example:

    >>> cudf.date_range(
    ...     start='2021-08-23 08:00:00',
    ...     freq=cudf.DateOffset(months=2, days=5),
    ...     periods=5)
    ...
    Traceback (most recent call last):
    ...
    NotImplementedError: Mixing fixed and non-fixed frequency offset is
    unsupported.

    Examples
    --------
    >>> cudf.date_range(
    ...     start='2021-08-23 08:00:00',
    ...     freq=cudf.DateOffset(years=1, months=2),
    ...     periods=5)
    DatetimeIndex(['2021-08-23 08:00:00', '2022-10-23 08:00:00',
                   '2023-12-23 08:00:00', '2025-02-23 08:00:00',
                   '2026-04-23 08:00:00'],
                  dtype='datetime64[us]', freq='<DateOffset: months=2, years=1>')
    """
    if inclusive != "both":
        raise NotImplementedError(f"{inclusive=} is currently unsupported.")
    if normalize is not False:
        raise NotImplementedError(f"{normalize=} is currently unsupported.")

    if freq is None and any(arg is None for arg in (start, end, periods)):
        freq = "D"

    if (start, end, periods, freq).count(None) > 1:
        raise ValueError(
            "Of the four parameters: start, end, periods, and freq, exactly "
            "three must be specified"
        )

    if periods is not None and not is_integer(periods):
        raise TypeError(f"periods must be an integer, got {periods}")

    _unit = _infer_date_range_unit(freq, unit, start, end)
    dtype: np.dtype = np.dtype(f"datetime64[{_unit}]")

    if freq is None:
        # `start`, `end`, `periods` is specified, we treat the timestamps as
        # integers and divide the number range evenly with `periods` elements.
        start = (
            pd.Timestamp(start)
            .as_unit(_unit)
            .to_numpy()
            .astype(np.dtype(np.int64))
        )
        end = (
            pd.Timestamp(end)
            .as_unit(_unit)
            .to_numpy()
            .astype(np.dtype(np.int64))
        )
        arr = np.linspace(start=start, stop=end, num=periods).astype(dtype)
        result = as_column(arr)
        return DatetimeIndex._from_column(result, name=name).tz_localize(tz)

    # The code logic below assumes `freq` is defined. It is first normalized
    # into `DateOffset` for further computation with timestamps.

    if isinstance(freq, DateOffset):
        offset = freq
    elif isinstance(freq, str):
        offset = pd.tseries.frequencies.to_offset(freq)
        if not isinstance(
            offset,
            (pd.tseries.offsets.Tick, pd.offsets.Day, pd.tseries.offsets.Week),
        ):
            raise ValueError(
                f"Unrecognized frequency string {freq}. cuDF does "
                "not yet support month, quarter, year-anchored frequency."
            )
        offset = DateOffset._from_pandas_ticks_or_weeks(offset)
    else:
        raise TypeError("`freq` must be a `str` or cudf.DateOffset object.")

    if _has_fixed_frequency(offset) and _has_non_fixed_frequency(offset):
        raise NotImplementedError(
            "Mixing fixed and non-fixed frequency offset is unsupported."
        )

    # Depending on different combinations of `start`, `end`, `offset`,
    # `periods`, the following logic makes sure before computing the sequence,
    # `start`, `periods`, `offset` is defined

    _periods_not_specified = False

    if start is None:
        end = pd.Timestamp(end).as_unit(_unit).to_numpy()
        start = (
            pd.Timestamp(end)
            - (periods - 1) * offset._maybe_as_fast_pandas_offset()
        ).to_numpy()
    elif end is None:
        start = pd.Timestamp(start).as_unit(_unit).to_numpy()
    elif periods is None:
        # When `periods` is unspecified, its upper bound estimated by
        # dividing the number of timestamps between two timestamps with
        # the lower bound of `freq` in the target unit. While the final
        # result may contain extra elements that exceeds `end`, they are
        # trimmed as a post processing step. [1]
        _periods_not_specified = True
        start = pd.Timestamp(start).as_unit(_unit).to_numpy()
        end = pd.Timestamp(end).as_unit(_unit).to_numpy()
        _is_increment_sequence = end >= start

        periods = math.floor(
            (end - start).view("int64") / _offset_to_lower_bound(offset, _unit)
        )

        if periods < 0:
            # Mismatched sign between (end-start) and offset, return empty
            # column
            periods = 0
        else:
            # If end == start, periods == 0 and we return exactly 1 timestamp (start).
            # Otherwise, since inclusive="both", we ensure the end point is included.
            periods += 1

    # We compute `end_estim` (the estimated upper bound of the date
    # range) below, but don't always use it.  We do this to ensure
    # that the appropriate OverflowError is raised by Pandas in case
    # of overflow.
    # FIXME: when `end_estim` is out of bound, but the actual `end` is not,
    # we shouldn't raise but compute the sequence as is. The trailing overflow
    # part should get trimmed at the end.
    with warnings.catch_warnings():
        # Need to ignore userwarnings where nonzero nanoseconds
        # are dropped in conversion during the binops
        warnings.simplefilter("ignore", UserWarning)
        end_estim = (
            pd.Timestamp(start)
            + periods * offset._maybe_as_fast_pandas_offset()
        ).to_datetime64()

    if "months" in offset.kwds or "years" in offset.kwds:
        # If `offset` is non-fixed frequency, resort to libcudf.
        months = offset.kwds.get("years", 0) * 12 + offset.kwds.get(
            "months", 0
        )
        # No columns to access here - calendrical_month_sequence creates new data
        plc_column = plc.filling.calendrical_month_sequence(
            periods,
            pa_scalar_to_plc_scalar(pa.scalar(start)),
            months,
        )
        res = ColumnBase.create(
            plc_column, dtype=dtype_from_pylibcudf_column(plc_column)
        )
        if _periods_not_specified:
            # As mentioned in [1], this is a post processing step to trim extra
            # elements when `periods` is an estimated value. Only offset
            # specified with non fixed frequencies requires trimming.
            res = res.apply_boolean_mask(
                (res <= end) if _is_increment_sequence else (res <= start)
            )
    else:
        # If `offset` is fixed frequency, we generate a range of
        # treating `start`, `stop` and `step` as ints:
        stop = end_estim.astype(np.dtype(np.int64))
        start = start.astype(np.dtype(np.int64))
        step = _offset_to_lower_bound(offset, _unit)
        res = ColumnBase.from_range(range(int(start), int(stop), step)).astype(
            dtype
        )

    return DatetimeIndex._from_column(res, name=name, freq=freq).tz_localize(
        tz
    )


def _has_fixed_frequency(freq: DateOffset) -> bool:
    """Utility to determine if `freq` contains fixed frequency offset"""
    fixed_frequencies = {
        "weeks",
        "days",
        "hours",
        "minutes",
        "seconds",
        "milliseconds",
        "microseconds",
        "nanoseconds",
    }

    return len(freq.kwds.keys() & fixed_frequencies) > 0


def _has_non_fixed_frequency(freq: DateOffset) -> bool:
    """Utility to determine if `freq` contains non-fixed frequency offset"""
    non_fixed_frequencies = {"years", "months"}
    return len(freq.kwds.keys() & non_fixed_frequencies) > 0


def _infer_date_range_unit(
    freq, unit: str | None, start=None, end=None
) -> str:
    """Infer the datetime resolution for date_range.

    Matches pandas 3.0 behavior: default to microseconds ('us') for
    string inputs, falling back to nanoseconds ('ns') when the freq
    or input timestamps require nanosecond precision.
    """
    if unit is not None:
        return unit
    # Preserve resolution of np.datetime64 inputs (pandas 3.0 behavior)
    for val in (start, end):
        if isinstance(val, np.datetime64):
            val_unit, _ = np.datetime_data(val.dtype)
            if val_unit == "ns":
                return "ns"
    # Check if freq has nanosecond components
    if isinstance(freq, DateOffset):
        if freq.kwds.get("nanoseconds", 0) != 0:
            return "ns"
    elif isinstance(freq, str):
        offset = DateOffset._from_freqstr(freq)
        return _infer_date_range_unit(offset, unit, start, end)
    return "us"


def _offset_to_lower_bound(offset: DateOffset, unit: str = "ns") -> int:
    """Given a DateOffset, which can consist of either fixed frequency or
    non-fixed frequency offset, convert to the smallest possible fixed
    frequency offset in the specified unit.

    Specifically, the smallest fixed frequency conversion for {months=1}
    is 28 * units_per_day, because 1 month contains at least 28 days.
    Similarly, the smallest fixed frequency conversion for {year=1} is
    365 * units_per_day.

    This utility is used to compute the upper bound of the count of timestamps
    given a range of datetime and an offset.
    """
    nanoseconds_per_day = 24 * 60 * 60 * 10**9
    kwds = offset.kwds
    total_ns = (
        kwds.get("years", 0) * (365 * nanoseconds_per_day)
        + kwds.get("months", 0) * (28 * nanoseconds_per_day)
        + kwds.get("weeks", 0) * (7 * nanoseconds_per_day)
        + kwds.get("days", 0) * nanoseconds_per_day
        + kwds.get("hours", 0) * 3600 * 10**9
        + kwds.get("minutes", 0) * 60 * 10**9
        + kwds.get("seconds", 0) * 10**9
        + kwds.get("milliseconds", 0) * 10**6
        + kwds.get("microseconds", 0) * 10**3
        + kwds.get("nanoseconds", 0)
    )
    _unit_to_ns = {"ns": 1, "us": 10**3, "ms": 10**6, "s": 10**9}
    return total_ns // _unit_to_ns[unit]

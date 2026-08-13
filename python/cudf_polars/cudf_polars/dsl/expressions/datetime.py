# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: Document TemporalFunction to remove noqa
# ruff: noqa: D101
"""DSL nodes for datetime operations."""

from __future__ import annotations

import re
from enum import IntEnum, auto
from typing import TYPE_CHECKING, Any, ClassVar, cast

import polars as pl

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from typing import Self

    from polars import polars  # type: ignore[attr-defined]

    from cudf_polars.containers import DataFrame
    from cudf_polars.dsl.expressions.literal import Literal

__all__ = ["TemporalFunction"]


_unit_to_nanoseconds_conversion = {
    plc.TypeId.DURATION_NANOSECONDS: 1,
    plc.TypeId.DURATION_MICROSECONDS: 1_000,
    plc.TypeId.DURATION_MILLISECONDS: 1_000_000,
    plc.TypeId.DURATION_SECONDS: 1_000_000_000,
    plc.TypeId.DURATION_DAYS: 86_400_000_000_000,
}


class TemporalFunction(Expr):
    class Name(IntEnum):
        """Internal and picklable representation of polars' `TemporalFunction`."""

        BaseUtcOffset = auto()
        CastTimeUnit = auto()
        Century = auto()
        Combine = auto()
        ConvertTimeZone = auto()
        DSTOffset = auto()
        Date = auto()
        Datetime = auto()
        DatetimeFunction = auto()
        Day = auto()
        DaysInMonth = auto()
        Duration = auto()
        Hour = auto()
        IsLeapYear = auto()
        IsoYear = auto()
        Microsecond = auto()
        Millennium = auto()
        Millisecond = auto()
        Minute = auto()
        Month = auto()
        MonthEnd = auto()
        MonthStart = auto()
        Nanosecond = auto()
        OffsetBy = auto()
        OrdinalDay = auto()
        Quarter = auto()
        Replace = auto()
        ReplaceTimeZone = auto()
        Round = auto()
        Second = auto()
        Time = auto()
        TimeStamp = auto()
        ToString = auto()
        TotalDays = auto()
        TotalHours = auto()
        TotalMicroseconds = auto()
        TotalMilliseconds = auto()
        TotalMinutes = auto()
        TotalNanoseconds = auto()
        TotalSeconds = auto()
        Truncate = auto()
        Week = auto()
        WeekDay = auto()
        WithTimeUnit = auto()
        Year = auto()

        @classmethod
        def from_polars(cls, obj: polars._expr_nodes.TemporalFunction) -> Self:
            """Convert from polars' `TemporalFunction`."""
            try:
                function, name = str(obj).split(".", maxsplit=1)
            except ValueError:
                # Failed to unpack string
                function = None
            if function != "TemporalFunction":
                raise ValueError("TemporalFunction required")
            return getattr(cls, name)

    __slots__ = ("name", "options")
    _non_child = ("dtype", "name", "options")
    _COMPONENT_MAP: ClassVar[dict[Name, plc.datetime.DatetimeComponent]] = {
        Name.Year: plc.datetime.DatetimeComponent.YEAR,
        Name.Month: plc.datetime.DatetimeComponent.MONTH,
        Name.Day: plc.datetime.DatetimeComponent.DAY,
        Name.WeekDay: plc.datetime.DatetimeComponent.WEEKDAY,
        Name.Hour: plc.datetime.DatetimeComponent.HOUR,
        Name.Minute: plc.datetime.DatetimeComponent.MINUTE,
        Name.Second: plc.datetime.DatetimeComponent.SECOND,
        Name.Millisecond: plc.datetime.DatetimeComponent.MILLISECOND,
        Name.Microsecond: plc.datetime.DatetimeComponent.MICROSECOND,
        Name.Nanosecond: plc.datetime.DatetimeComponent.NANOSECOND,
    }
    _TRUNCATE_FREQ_MAP: ClassVar[dict[str, plc.datetime.RoundingFrequency]] = {
        "d": plc.datetime.RoundingFrequency.DAY,
        "h": plc.datetime.RoundingFrequency.HOUR,
        "m": plc.datetime.RoundingFrequency.MINUTE,
        "s": plc.datetime.RoundingFrequency.SECOND,
        "ms": plc.datetime.RoundingFrequency.MILLISECOND,
        "us": plc.datetime.RoundingFrequency.MICROSECOND,
        "ns": plc.datetime.RoundingFrequency.NANOSECOND,
    }

    # Number of nanoseconds represented by one unit of each ``total_*`` component.
    _TOTAL_COMPONENT_NANOSECONDS: ClassVar[dict[Name, int]] = {
        Name.TotalDays: 86_400_000_000_000,
        Name.TotalHours: 3_600_000_000_000,
        Name.TotalMinutes: 60_000_000_000,
        Name.TotalSeconds: 1_000_000_000,
        Name.TotalMilliseconds: 1_000_000,
        Name.TotalMicroseconds: 1_000,
        Name.TotalNanoseconds: 1,
    }
    # Divisor used to derive the century/millennium from the calendar year:
    # ``(year - 1) // divisor + 1`` (floor division, matching polars).
    _CENTURY_MILLENNIUM_DIVISOR: ClassVar[dict[Name, int]] = {
        Name.Millennium: 1_000,
        Name.Century: 100,
    }
    _valid_ops: ClassVar[set[Name]] = {
        *_COMPONENT_MAP.keys(),
        Name.Round,
        Name.IsLeapYear,
        Name.OrdinalDay,
        Name.ToString,
        Name.Week,
        Name.IsoYear,
        Name.MonthStart,
        Name.MonthEnd,
        Name.TimeStamp,
        Name.CastTimeUnit,
        Name.Truncate,
        Name.Date,
        Name.DaysInMonth,
        Name.Quarter,
        *_CENTURY_MILLENNIUM_DIVISOR.keys(),
        *_TOTAL_COMPONENT_NANOSECONDS.keys(),
    }

    def __init__(
        self,
        dtype: DataType,
        name: TemporalFunction.Name,
        options: tuple[Any, ...],
        *children: Expr,
    ) -> None:
        self.dtype = dtype
        self.options = options
        self.name = name
        self.children = children
        self.is_pointwise = True
        if self.name not in self._valid_ops:
            raise NotImplementedError(f"Temporal function {self.name}")
        if self.name is TemporalFunction.Name.ToString and plc.traits.is_duration(
            self.children[0].dtype.plc_type
        ):
            raise NotImplementedError("ToString is not supported on duration types")
        elif self.name in {
            TemporalFunction.Name.Truncate,
            TemporalFunction.Name.Round,
        }:
            every = cast("Literal", self.children[1]).value
            match = re.fullmatch(r"(\d+)(ns|us|ms|s|m|h|d)", every)
            if match is None or int(match.group(1)) != 1:
                # https://github.com/rapidsai/cudf/issues/18654 to support non-1 buckets
                raise NotImplementedError(f"Unsupported bucket: {every!r}")
            self.options = (self._TRUNCATE_FREQ_MAP[match.group(2)],)

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        columns = [child.evaluate(df, context=context) for child in self.children]
        if self.name in self._TOTAL_COMPONENT_NANOSECONDS:
            (column,) = columns
            source_ns = _unit_to_nanoseconds_conversion[column.obj.type().id()]
            target_ns = self._TOTAL_COMPONENT_NANOSECONDS[self.name]
            # Reinterpret the duration's integer tick count as int64.
            casted = column.astype(self.dtype, stream=df.stream)
            if source_ns >= target_ns:
                # Coarser (or equal) storage unit: exact integer multiply.
                op = plc.binaryop.BinaryOperator.MUL
                factor = source_ns // target_ns
            else:
                # Finer storage unit: integer divide. libcudf (like polars)
                # truncates toward zero for signed integer division.
                op = plc.binaryop.BinaryOperator.DIV
                factor = target_ns // source_ns
            if factor == 1:
                # Storage unit already matches the requested unit.
                return casted
            result = plc.binaryop.binary_operation(
                casted.obj,
                plc.Scalar.from_py(
                    factor, plc.DataType(plc.TypeId.INT64), stream=df.stream
                ),
                op,
                self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(result, dtype=self.dtype)
        if self.name is TemporalFunction.Name.TimeStamp:
            (column,) = columns
            (time_unit,) = self.options
            # Rescale the timestamp to the requested resolution
            df_stream = df.stream
            return column.astype(
                DataType(pl.Datetime(time_unit)), stream=df_stream
            ).astype(self.dtype, stream=df_stream)
        elif self.name is TemporalFunction.Name.Round:
            (column, _) = columns
            return Column(
                plc.datetime.round_datetimes(
                    column.obj,
                    self.options[0],
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is TemporalFunction.Name.Truncate:
            (column, _) = columns
            return Column(
                plc.datetime.floor_datetimes(
                    column.obj,
                    self.options[0],
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is TemporalFunction.Name.Date:
            (column,) = columns
            # Casting the timestamp to TIMESTAMP_DAYS (the storage of ``pl.Date``)
            # drops the sub-day component.
            return Column(
                plc.unary.cast(column.obj, self.dtype.plc_type, stream=df.stream),
                dtype=self.dtype,
            )
        elif self.name is TemporalFunction.Name.DaysInMonth:
            (column,) = columns
            return Column(
                plc.datetime.days_in_month(column.obj, stream=df.stream),
                dtype=DataType(pl.Int16()),
            ).astype(self.dtype, stream=df.stream)
        elif self.name is TemporalFunction.Name.Quarter:
            (column,) = columns
            return Column(
                plc.datetime.extract_quarter(column.obj, stream=df.stream),
                dtype=DataType(pl.Int16()),
            ).astype(self.dtype, stream=df.stream)
        elif self.name in self._CENTURY_MILLENNIUM_DIVISOR:
            (column,) = columns
            int32 = plc.DataType(plc.TypeId.INT32)
            # YEAR extraction yields INT16; cast up so the arithmetic (and the
            # INT32 output polars produces) does not overflow or need promotion.
            year = plc.unary.cast(
                plc.datetime.extract_datetime_component(
                    column.obj,
                    plc.datetime.DatetimeComponent.YEAR,
                    stream=df.stream,
                ),
                int32,
                stream=df.stream,
            )
            # polars computes ``(year - 1) // divisor + 1`` using floor division;
            one = plc.expressions.Literal(
                plc.Scalar.from_py(1, int32, stream=df.stream)
            )
            predicate = plc.expressions.Operation(
                plc.expressions.ASTOperator.ADD,
                plc.expressions.Operation(
                    plc.expressions.ASTOperator.FLOOR_DIV,
                    plc.expressions.Operation(
                        plc.expressions.ASTOperator.SUB,
                        plc.expressions.ColumnReference(0),
                        one,
                    ),
                    plc.expressions.Literal(
                        plc.Scalar.from_py(
                            self._CENTURY_MILLENNIUM_DIVISOR[self.name],
                            int32,
                            stream=df.stream,
                        )
                    ),
                ),
                one,
            )
            return Column(
                plc.transform.compute_column(
                    plc.Table([year]), predicate, stream=df.stream
                ),
                dtype=self.dtype,
            )
        elif self.name is TemporalFunction.Name.CastTimeUnit:
            (column,) = columns
            return Column(
                plc.unary.cast(column.obj, self.dtype.plc_type, stream=df.stream),
                dtype=self.dtype,
            )
        elif self.name == TemporalFunction.Name.ToString:
            (column,) = columns
            (format_string,) = self.options
            if format_string == "":
                # libcudf doesn't support empty format strings, but polars
                # returns empty strings for each row in this case
                return Column(
                    plc.Column.from_scalar(
                        plc.Scalar.from_py("", self.dtype.plc_type, stream=df.stream),
                        column.size,
                        stream=df.stream,
                    ),
                    dtype=self.dtype,
                )
            return Column(
                plc.strings.convert.convert_datetime.from_timestamps(
                    column.obj,
                    format_string,
                    plc.Column.from_iterable_of_py(
                        [], dtype=self.dtype.plc_type, stream=df.stream
                    ),
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is TemporalFunction.Name.Week:
            (column,) = columns
            result = plc.strings.convert.convert_integers.to_integers(
                plc.strings.convert.convert_datetime.from_timestamps(
                    column.obj,
                    format="%V",
                    input_strings_names=plc.Column.from_iterable_of_py(
                        [], dtype=plc.DataType(plc.TypeId.STRING), stream=df.stream
                    ),
                    stream=df.stream,
                ),
                self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(result, dtype=self.dtype)
        elif self.name is TemporalFunction.Name.IsoYear:
            (column,) = columns
            result = plc.strings.convert.convert_integers.to_integers(
                plc.strings.convert.convert_datetime.from_timestamps(
                    column.obj,
                    format="%G",
                    input_strings_names=plc.Column.from_iterable_of_py(
                        [], dtype=plc.DataType(plc.TypeId.STRING), stream=df.stream
                    ),
                    stream=df.stream,
                ),
                self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(result, dtype=self.dtype)
        elif self.name is TemporalFunction.Name.MonthStart:
            (column,) = columns
            ends = plc.datetime.last_day_of_month(column.obj, stream=df.stream)
            days_to_subtract = plc.datetime.days_in_month(column.obj, stream=df.stream)
            # must subtract 1 to avoid rolling over to the previous month
            days_to_subtract = plc.binaryop.binary_operation(
                days_to_subtract,
                plc.Scalar.from_py(1, plc.DataType(plc.TypeId.INT32), stream=df.stream),
                plc.binaryop.BinaryOperator.SUB,
                plc.DataType(plc.TypeId.DURATION_DAYS),
                stream=df.stream,
            )
            result = plc.binaryop.binary_operation(
                ends,
                days_to_subtract,
                plc.binaryop.BinaryOperator.SUB,
                self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(result, dtype=self.dtype)
        elif self.name is TemporalFunction.Name.MonthEnd:
            (column,) = columns
            return Column(
                plc.unary.cast(
                    plc.datetime.last_day_of_month(column.obj, stream=df.stream),
                    self.dtype.plc_type,
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )
        elif self.name is TemporalFunction.Name.IsLeapYear:
            (column,) = columns
            return Column(
                plc.datetime.is_leap_year(column.obj, stream=df.stream),
                dtype=self.dtype,
            )
        elif self.name is TemporalFunction.Name.OrdinalDay:
            (column,) = columns
            return Column(
                plc.datetime.day_of_year(column.obj, stream=df.stream), dtype=self.dtype
            )
        elif self.name is TemporalFunction.Name.Microsecond:
            (column,) = columns
            millis = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.MILLISECOND, stream=df.stream
            )
            micros = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.MICROSECOND, stream=df.stream
            )
            millis_as_micros = plc.binaryop.binary_operation(
                millis,
                plc.Scalar.from_py(
                    1_000, plc.DataType(plc.TypeId.INT32), stream=df.stream
                ),
                plc.binaryop.BinaryOperator.MUL,
                self.dtype.plc_type,
                stream=df.stream,
            )
            total_micros = plc.binaryop.binary_operation(
                micros,
                millis_as_micros,
                plc.binaryop.BinaryOperator.ADD,
                self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(total_micros, dtype=self.dtype)
        elif self.name is TemporalFunction.Name.Nanosecond:
            (column,) = columns
            millis = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.MILLISECOND, stream=df.stream
            )
            micros = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.MICROSECOND, stream=df.stream
            )
            nanos = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.NANOSECOND, stream=df.stream
            )
            millis_as_nanos = plc.binaryop.binary_operation(
                millis,
                plc.Scalar.from_py(
                    1_000_000, plc.DataType(plc.TypeId.INT32), stream=df.stream
                ),
                plc.binaryop.BinaryOperator.MUL,
                self.dtype.plc_type,
                stream=df.stream,
            )
            micros_as_nanos = plc.binaryop.binary_operation(
                micros,
                plc.Scalar.from_py(
                    1_000, plc.DataType(plc.TypeId.INT32), stream=df.stream
                ),
                plc.binaryop.BinaryOperator.MUL,
                self.dtype.plc_type,
                stream=df.stream,
            )
            total_nanos = plc.binaryop.binary_operation(
                nanos,
                millis_as_nanos,
                plc.binaryop.BinaryOperator.ADD,
                self.dtype.plc_type,
                stream=df.stream,
            )
            total_nanos = plc.binaryop.binary_operation(
                total_nanos,
                micros_as_nanos,
                plc.binaryop.BinaryOperator.ADD,
                self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(total_nanos, dtype=self.dtype)
        else:
            (column,) = columns
            return Column(
                plc.datetime.extract_datetime_component(
                    column.obj,
                    self._COMPONENT_MAP[self.name],
                    stream=df.stream,
                ),
                dtype=self.dtype,
            )

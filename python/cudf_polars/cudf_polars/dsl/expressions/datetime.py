# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# TODO: Document TemporalFunction to remove noqa
# ruff: noqa: D101
"""DSL nodes for datetime operations."""

from __future__ import annotations

import re
import zoneinfo
from enum import IntEnum, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError, SchemaError

import pylibcudf as plc

from cudf_polars.containers import Column, DataType
from cudf_polars.dsl.expressions.base import ExecutionContext, Expr

if TYPE_CHECKING:
    from typing import Self

    from polars import polars  # type: ignore[attr-defined]

    from rmm.pylibrmm.stream import Stream

    from cudf_polars.containers import DataFrame

__all__ = ["TemporalFunction"]

_SECONDS_TIMESTAMP = plc.DataType(plc.TypeId.TIMESTAMP_SECONDS)
_TIMESTAMP_TO_DURATION = {
    plc.TypeId.TIMESTAMP_MILLISECONDS: plc.TypeId.DURATION_MILLISECONDS,
    plc.TypeId.TIMESTAMP_MICROSECONDS: plc.TypeId.DURATION_MICROSECONDS,
    plc.TypeId.TIMESTAMP_NANOSECONDS: plc.TypeId.DURATION_NANOSECONDS,
}


def _tz_transition_columns(
    zone_name: str, tzif_dir: str, stream: Stream
) -> tuple[plc.Column, plc.Column] | None:
    """Return the (transition times, UTC offsets) columns for ``zone_name``."""
    table = plc.io.timezone.make_timezone_transition_table(
        tzif_dir, zone_name, stream=stream
    )
    columns = table.columns()
    if len(columns) == 0:  # pragma: no cover
        return None
    transition_times, offsets = columns
    return transition_times, offsets


def _local_wall_clock(
    column: plc.Column, from_zone_desc: tuple[str, str] | None, stream: Stream
) -> plc.Column:
    """Convert UTC timestamps to naive wall-clock timestamps in ``from_zone``."""
    if from_zone_desc is None:
        return column
    data = _tz_transition_columns(*from_zone_desc, stream)
    if data is None:  # pragma: no cover
        return column
    transition_times, offsets = data
    unit = column.type()
    duration_type = plc.DataType(_TIMESTAMP_TO_DURATION[unit.id()])
    seconds = plc.unary.cast(column, _SECONDS_TIMESTAMP, stream=stream)
    positions = plc.search.upper_bound(
        plc.Table([transition_times]),
        plc.Table([seconds]),
        [plc.types.Order.ASCENDING],
        [plc.types.NullOrder.BEFORE],
        stream=stream,
    )
    index_type = positions.type()
    shifted = plc.binaryop.binary_operation(
        positions,
        plc.Scalar.from_py(1, index_type, stream=stream),
        plc.binaryop.BinaryOperator.SUB,
        index_type,
        stream=stream,
    )
    index = plc.replace.clamp(
        shifted,
        plc.Scalar.from_py(0, index_type, stream=stream),
        plc.Scalar.from_py(offsets.size() - 1, index_type, stream=stream),
        stream=stream,
    )
    (gathered,) = plc.copying.gather(
        plc.Table([offsets]),
        index,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    ).columns()
    offset = plc.unary.cast(gathered, duration_type, stream=stream)
    return plc.binaryop.binary_operation(
        column, offset, plc.binaryop.BinaryOperator.ADD, unit, stream=stream
    )


def _ambiguous_nonexistent(
    transition_times: plc.Column,
    offsets: plc.Column,
    local_seconds: plc.Column,
    stream: Stream,
) -> tuple[plc.Column, plc.Column]:
    """Return boolean masks for ambiguous and non-existent wall-clock times."""
    size = offsets.size()
    (new_transitions,) = plc.copying.slice(transition_times, [1, size], stream=stream)
    (new_offsets,) = plc.copying.slice(offsets, [1, size], stream=stream)
    (old_offsets,) = plc.copying.slice(offsets, [0, size - 1], stream=stream)
    clock_new = plc.binaryop.binary_operation(
        new_transitions,
        new_offsets,
        plc.binaryop.BinaryOperator.ADD,
        _SECONDS_TIMESTAMP,
        stream=stream,
    )
    clock_old = plc.binaryop.binary_operation(
        new_transitions,
        old_offsets,
        plc.binaryop.BinaryOperator.ADD,
        _SECONDS_TIMESTAMP,
        stream=stream,
    )
    bool_type = plc.DataType(plc.TypeId.BOOL8)
    false = plc.Scalar.from_py(False, bool_type, stream=stream)  # noqa: FBT003
    n = local_seconds.size()

    ambiguous_cond = plc.binaryop.binary_operation(
        clock_new, clock_old, plc.binaryop.BinaryOperator.LESS, bool_type, stream=stream
    )
    (ambiguous_begin,) = plc.stream_compaction.apply_retention_mask(
        plc.Table([clock_new]), ambiguous_cond, stream=stream
    ).columns()
    (ambiguous_end,) = plc.stream_compaction.apply_retention_mask(
        plc.Table([clock_old]), ambiguous_cond, stream=stream
    ).columns()
    if ambiguous_begin.size() == 0:
        is_ambiguous = plc.Column.from_scalar(false, n, stream=stream)
    else:
        is_ambiguous = plc.unary.is_valid(
            plc.labeling.label_bins(
                local_seconds,
                ambiguous_begin,
                plc.labeling.Inclusive.YES,
                ambiguous_end,
                plc.labeling.Inclusive.NO,
                stream=stream,
            ),
            stream=stream,
        )

    nonexistent_cond = plc.binaryop.binary_operation(
        clock_new,
        clock_old,
        plc.binaryop.BinaryOperator.GREATER,
        bool_type,
        stream=stream,
    )
    (nonexistent_begin,) = plc.stream_compaction.apply_retention_mask(
        plc.Table([clock_old]), nonexistent_cond, stream=stream
    ).columns()
    (nonexistent_end,) = plc.stream_compaction.apply_retention_mask(
        plc.Table([clock_new]), nonexistent_cond, stream=stream
    ).columns()
    if nonexistent_begin.size() == 0:
        is_nonexistent = plc.Column.from_scalar(false, n, stream=stream)
    else:
        is_nonexistent = plc.unary.is_valid(
            plc.labeling.label_bins(
                local_seconds,
                nonexistent_begin,
                plc.labeling.Inclusive.YES,
                nonexistent_end,
                plc.labeling.Inclusive.NO,
                stream=stream,
            ),
            stream=stream,
        )
    return is_ambiguous, is_nonexistent


def _apply_ambiguous(
    utc_latest: plc.Column,
    utc_earliest: plc.Column,
    is_ambiguous: plc.Column,
    ambiguous_scalar: str | None,
    ambiguous_column: plc.Column,
    null_scalar: plc.Scalar,
    stream: Stream,
) -> plc.Column:
    bool_type = plc.DataType(plc.TypeId.BOOL8)
    if ambiguous_scalar is not None:
        if ambiguous_scalar == "raise":
            if bool(
                plc.reduce.reduce(
                    is_ambiguous, plc.aggregation.any(), bool_type, stream=stream
                ).to_py(stream=stream)
            ):
                raise ComputeError(
                    "datetime is ambiguous in the given time zone. Please use "
                    "`ambiguous` to tell how it should be localized."
                )
            return utc_latest
        if ambiguous_scalar == "latest":
            return utc_latest
        if ambiguous_scalar == "earliest":
            return plc.copying.copy_if_else(
                utc_earliest, utc_latest, is_ambiguous, stream=stream
            )
        return plc.copying.copy_if_else(
            null_scalar, utc_latest, is_ambiguous, stream=stream
        )
    string_type = plc.DataType(plc.TypeId.STRING)
    allowed = plc.Column.from_iterable_of_py(
        ["earliest", "latest", "null", "raise"],
        dtype=string_type,
        stream=stream,
    )
    is_invalid = plc.unary.unary_operation(
        plc.search.contains(allowed, ambiguous_column, stream=stream),
        plc.unary.UnaryOperator.NOT,
        stream=stream,
    )
    if bool(
        plc.reduce.reduce(
            is_invalid, plc.aggregation.any(), bool_type, stream=stream
        ).to_py(stream=stream)
    ):
        (invalid_values,) = plc.stream_compaction.apply_retention_mask(
            plc.Table([ambiguous_column]), is_invalid, stream=stream
        ).columns()
        invalid = invalid_values.to_scalar(stream=stream).to_py(stream=stream)
        raise InvalidOperationError(
            f"Invalid argument {invalid}, expected one of: "
            '"earliest", "latest", "null", "raise"'
        )
    is_raise = plc.binaryop.binary_operation(
        is_ambiguous,
        plc.binaryop.binary_operation(
            ambiguous_column,
            plc.Scalar.from_py("raise", string_type, stream=stream),
            plc.binaryop.BinaryOperator.EQUAL,
            bool_type,
            stream=stream,
        ),
        plc.binaryop.BinaryOperator.LOGICAL_AND,
        bool_type,
        stream=stream,
    )
    if bool(
        plc.reduce.reduce(
            is_raise, plc.aggregation.any(), bool_type, stream=stream
        ).to_py(stream=stream)
    ):
        raise ComputeError(
            "datetime is ambiguous in the given time zone. Please use `ambiguous` "
            "to tell how it should be localized."
        )
    is_earliest = plc.binaryop.binary_operation(
        is_ambiguous,
        plc.binaryop.binary_operation(
            ambiguous_column,
            plc.Scalar.from_py("earliest", string_type, stream=stream),
            plc.binaryop.BinaryOperator.EQUAL,
            bool_type,
            stream=stream,
        ),
        plc.binaryop.BinaryOperator.LOGICAL_AND,
        bool_type,
        stream=stream,
    )
    result = plc.copying.copy_if_else(
        utc_earliest, utc_latest, is_earliest, stream=stream
    )
    is_null = plc.binaryop.binary_operation(
        is_ambiguous,
        plc.binaryop.binary_operation(
            ambiguous_column,
            plc.Scalar.from_py("null", string_type, stream=stream),
            plc.binaryop.BinaryOperator.EQUAL,
            bool_type,
            stream=stream,
        ),
        plc.binaryop.BinaryOperator.LOGICAL_AND,
        bool_type,
        stream=stream,
    )
    result = plc.copying.copy_if_else(null_scalar, result, is_null, stream=stream)
    return plc.copying.copy_if_else(
        null_scalar,
        result,
        plc.unary.is_null(ambiguous_column, stream=stream),
        stream=stream,
    )


def _apply_ambiguous_without_transitions(
    local: plc.Column,
    ambiguous_scalar: str | None,
    ambiguous_column: plc.Column,
    stream: Stream,
) -> plc.Column:
    unit = local.type()
    return _apply_ambiguous(
        local,
        local,
        plc.Column.from_scalar(
            plc.Scalar.from_py(
                py_val=False,
                dtype=plc.DataType(plc.TypeId.BOOL8),
                stream=stream,
            ),
            local.size(),
            stream=stream,
        ),
        ambiguous_scalar,
        ambiguous_column,
        plc.Scalar.from_py(None, unit, stream=stream),
        stream,
    )


def _apply_nonexistent(
    utc: plc.Column,
    is_nonexistent: plc.Column,
    non_existent: str,
    null_scalar: plc.Scalar,
    stream: Stream,
) -> plc.Column:
    if non_existent == "raise":
        if bool(
            plc.reduce.reduce(
                is_nonexistent,
                plc.aggregation.any(),
                plc.DataType(plc.TypeId.BOOL8),
                stream=stream,
            ).to_py(stream=stream)
        ):
            raise ComputeError(
                "datetime is non-existent in the given time zone. You may be able "
                "to use `non_existent='null'` to return `null` in this case."
            )
        return utc
    return plc.copying.copy_if_else(null_scalar, utc, is_nonexistent, stream=stream)


def _localize(
    local: plc.Column,
    to_zone: str,
    tzif_dir: str,
    ambiguous_scalar: str | None,
    ambiguous_column: plc.Column,
    non_existent: str,
    stream: Stream,
) -> plc.Column:
    """Interpret naive wall-clock timestamps as local times in ``to_zone``."""
    data = _tz_transition_columns(to_zone, tzif_dir, stream)
    if data is None:  # pragma: no cover
        return _apply_ambiguous_without_transitions(
            local, ambiguous_scalar, ambiguous_column, stream
        )
    transition_times, offsets = data
    size = offsets.size()
    unit = local.type()
    duration_type = plc.DataType(_TIMESTAMP_TO_DURATION[unit.id()])
    local_seconds = plc.unary.cast(local, _SECONDS_TIMESTAMP, stream=stream)
    local_transitions = plc.binaryop.binary_operation(
        transition_times,
        offsets,
        plc.binaryop.BinaryOperator.ADD,
        _SECONDS_TIMESTAMP,
        stream=stream,
    )
    positions = plc.search.upper_bound(
        plc.Table([local_transitions]),
        plc.Table([local_seconds]),
        [plc.types.Order.ASCENDING],
        [plc.types.NullOrder.BEFORE],
        stream=stream,
    )
    index_type = positions.type()
    lower = plc.Scalar.from_py(0, index_type, stream=stream)
    upper = plc.Scalar.from_py(size - 1, index_type, stream=stream)
    index_latest = plc.replace.clamp(
        plc.binaryop.binary_operation(
            positions,
            plc.Scalar.from_py(1, index_type, stream=stream),
            plc.binaryop.BinaryOperator.SUB,
            index_type,
            stream=stream,
        ),
        lower,
        upper,
        stream=stream,
    )
    (gathered_latest,) = plc.copying.gather(
        plc.Table([offsets]),
        index_latest,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    ).columns()
    offset_latest = plc.unary.cast(gathered_latest, duration_type, stream=stream)
    utc = plc.binaryop.binary_operation(
        local, offset_latest, plc.binaryop.BinaryOperator.SUB, unit, stream=stream
    )
    is_ambiguous, is_nonexistent = _ambiguous_nonexistent(
        transition_times, offsets, local_seconds, stream
    )
    index_earliest = plc.replace.clamp(
        plc.binaryop.binary_operation(
            positions,
            plc.Scalar.from_py(2, index_type, stream=stream),
            plc.binaryop.BinaryOperator.SUB,
            index_type,
            stream=stream,
        ),
        lower,
        upper,
        stream=stream,
    )
    (gathered_earliest,) = plc.copying.gather(
        plc.Table([offsets]),
        index_earliest,
        plc.copying.OutOfBoundsPolicy.DONT_CHECK,
        stream=stream,
    ).columns()
    offset_earliest = plc.unary.cast(gathered_earliest, duration_type, stream=stream)
    utc_earliest = plc.binaryop.binary_operation(
        local, offset_earliest, plc.binaryop.BinaryOperator.SUB, unit, stream=stream
    )
    null_scalar = plc.Scalar.from_py(None, unit, stream=stream)
    utc = _apply_ambiguous(
        utc,
        utc_earliest,
        is_ambiguous,
        ambiguous_scalar,
        ambiguous_column,
        null_scalar,
        stream,
    )
    return _apply_nonexistent(utc, is_nonexistent, non_existent, null_scalar, stream)


_unit_to_nanoseconds_conversion = {
    plc.TypeId.DURATION_NANOSECONDS: 1,
    plc.TypeId.DURATION_MICROSECONDS: 1_000,
    plc.TypeId.DURATION_MILLISECONDS: 1_000_000,
    plc.TypeId.DURATION_SECONDS: 1_000_000_000,
    plc.TypeId.DURATION_DAYS: 86_400_000_000_000,
}

_NANOSECONDS_PER_DAY = 86_400_000_000_000
_MILLISECONDS_PER_DAY = 86_400_000
# Nanoseconds in one tick of each timestamp type that can carry a polars Datetime.
_TIMESTAMP_TICK_NANOSECONDS = {
    plc.TypeId.TIMESTAMP_NANOSECONDS: 1,
    plc.TypeId.TIMESTAMP_MICROSECONDS: 1_000,
    plc.TypeId.TIMESTAMP_MILLISECONDS: 1_000_000,
}
_DURATION_STRING = re.compile(r"(?:\d+(?:ns|us|ms|mo|s|m|h|d|w|q|y))+")
_DURATION_TOKEN = re.compile(r"(\d+)(ns|us|ms|mo|s|m|h|d|w|q|y)")
_DURATION_TOKEN_MONTHS = {"mo": 1, "q": 3, "y": 12}
_DURATION_TOKEN_NANOSECONDS = {
    "ns": 1,
    "us": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60_000_000_000,
    "h": 3_600_000_000_000,
}
# Largest bucket, in ticks, for which ``pymod(t, every)`` cannot overflow INT64.
_MAX_TRUNCATE_TICKS = 2**62
# Largest bucket, in months, for which the calendar arithmetic in
# ``_month_start_days`` is exact in INT64.
_MAX_TRUNCATE_MONTHS = 2**40
# The Gregorian calendar repeats every 400 years.
_DAYS_PER_400_YEARS = 146_097
_MONTHS_PER_400_YEARS = 4_800
# Durations, in nanoseconds, that libcudf's floor_datetimes supports directly,
# and whether libcudf counts them in 32 bits.
_FLOOR_DATETIMES_FREQUENCIES = (
    (_NANOSECONDS_PER_DAY, plc.datetime.RoundingFrequency.DAY, True),
    (3_600_000_000_000, plc.datetime.RoundingFrequency.HOUR, True),
    (60_000_000_000, plc.datetime.RoundingFrequency.MINUTE, True),
    (1_000_000_000, plc.datetime.RoundingFrequency.SECOND, False),
    (1_000_000, plc.datetime.RoundingFrequency.MILLISECOND, False),
    (1_000, plc.datetime.RoundingFrequency.MICROSECOND, False),
)
_INT64 = plc.DataType(plc.TypeId.INT64)
_UINT64 = plc.DataType(plc.TypeId.UINT64)
_AST = plc.expressions.ASTOperator


class _TruncateMethod(IntEnum):
    """How a ``dt.truncate`` is computed on device."""

    IDENTITY = auto()
    """Return the input unchanged."""
    FLOOR_DATETIMES = auto()
    """libcudf's ``floor_datetimes``."""
    FIXED = auto()
    """``t - pymod(t, every)``."""
    WEEKLY = auto()
    """``t - pymod(t - 4 days, every)``, anchoring weeks at Monday 1970-01-05."""
    MONTHLY = auto()
    """First day of the month, flooring the months since 1970-01 to ``every``."""


class _TruncatePlan(NamedTuple):
    """Device computation for a ``dt.truncate``, produced by :func:`_plan_truncate`."""

    method: _TruncateMethod
    every: int
    """Ticks (FIXED, WEEKLY), months (MONTHLY), or a ``RoundingFrequency``."""
    day: int
    """Ticks per day (WEEKLY, MONTHLY)."""
    on_date: bool
    """Whether a Date is truncated, on a millisecond timeline."""


def _parse_duration_string(every: str) -> tuple[int, int, int, int] | None:
    """
    Parse a non-negative polars duration string.

    Parameters
    ----------
    every
        Duration string such as ``"10m"``, ``"1h30m"`` or ``"1q"``.

    Returns
    -------
    ``(months, weeks, days, nanoseconds)``, or ``None`` if the string uses
    syntax this parser does not handle (for example negative durations or
    the ``i`` index unit).
    """
    if _DURATION_STRING.fullmatch(every) is None:
        return None
    months = weeks = days = nanoseconds = 0
    for count, unit in _DURATION_TOKEN.findall(every):
        n = int(count)
        if unit in _DURATION_TOKEN_MONTHS:
            months += n * _DURATION_TOKEN_MONTHS[unit]
        elif unit == "w":
            weeks += n
        elif unit == "d":
            days += n
        else:
            nanoseconds += n * _DURATION_TOKEN_NANOSECONDS[unit]
    return months, weeks, days, nanoseconds


def _floor_datetimes_frequency(
    tick: int, every_ns: int
) -> plc.datetime.RoundingFrequency | None:
    """
    Return the floor_datetimes frequency for ``every_ns``, if it is exact.

    libcudf counts days, hours and minutes in 32 bits, so those frequencies are
    only used if every value of the column fits; otherwise the counts overflow.
    """
    for frequency_ns, frequency, count_is_int32 in _FLOOR_DATETIMES_FREQUENCIES:
        if every_ns == frequency_ns:
            if count_is_int32 and 2**63 // (frequency_ns // tick) >= 2**31:
                return None
            return frequency
    return None


def _plan_truncate(dtype: DataType, every: str) -> _TruncatePlan:
    """
    Translate ``dt.truncate(every)`` into a device computation.

    Parameters
    ----------
    dtype
        Dtype of the column being truncated.
    every
        Polars duration string.

    Returns
    -------
    The device computation.

    Raises
    ------
    NotImplementedError
        For inputs evaluated on the CPU instead. These include every case in
        which polars raises, so the error comes from polars itself.

    Notes
    -----
    This mirrors polars' ``truncate`` (``polars-time``) exactly, including
    its wrapping 64-bit arithmetic near the limits of each time unit:

    - A naive or UTC Datetime truncated by a duration without months or weeks
      is floored relative to the epoch, by the whole duration converted to the
      column's time unit one component at a time; if that is zero the input is
      returned unchanged.
    - Otherwise each unit is handled on its own (polars raises for zero or
      mixed durations): fixed durations as above, weeks anchored at Monday
      1970-01-05, and months, quarters and years by flooring the months since
      1970-01 and returning midnight on the first day of the month.
    - A Date is truncated the same way on a millisecond timeline, then divided
      back to days rounding toward zero, which may differ from casting the
      floored instant to a Date before 1970.

    Polars computes calendar months with chrono, which panics beyond years
    +/-262143; here those values follow the proleptic Gregorian calendar.
    """
    parsed = _parse_duration_string(every)
    if parsed is None:
        raise NotImplementedError(f"dt.truncate with every={every!r}")
    months, weeks, days, nanoseconds = parsed
    type_id = dtype.id()
    if type_id == plc.TypeId.TIMESTAMP_DAYS:
        return _plan_unit_truncate(
            every, months, weeks, days, nanoseconds, 1_000_000, on_date=True
        )
    if type_id not in _TIMESTAMP_TICK_NANOSECONDS:
        raise NotImplementedError(  # pragma: no cover; polars raises first
            f"dt.truncate on {dtype.polars_type}"
        )
    time_zone = cast("pl.Datetime", dtype.polars_type).time_zone
    if time_zone not in {None, "UTC"}:
        # polars truncates in local time for time zone aware datetimes.
        raise NotImplementedError(
            f"dt.truncate on a datetime with time zone {time_zone!r}"
        )
    tick = _TIMESTAMP_TICK_NANOSECONDS[type_id]
    if months > 0 or weeks > 0:
        return _plan_unit_truncate(
            every, months, weeks, days, nanoseconds, tick, on_date=False
        )
    every_ticks = days * (_NANOSECONDS_PER_DAY // tick) + nanoseconds // tick
    if every_ticks <= 1:
        return _TruncatePlan(_TruncateMethod.IDENTITY, 0, 0, on_date=False)
    frequency = _floor_datetimes_frequency(
        tick, days * _NANOSECONDS_PER_DAY + nanoseconds
    )
    if frequency is not None:
        # libcudf divides by a compile-time constant, which is several times
        # faster than the general path's runtime 64-bit division.
        return _TruncatePlan(
            _TruncateMethod.FLOOR_DATETIMES, int(frequency), 0, on_date=False
        )
    if every_ticks > _MAX_TRUNCATE_TICKS:
        raise NotImplementedError(f"dt.truncate with every={every!r}")
    return _TruncatePlan(_TruncateMethod.FIXED, every_ticks, 0, on_date=False)


def _plan_unit_truncate(
    every: str,
    months: int,
    weeks: int,
    days: int,
    nanoseconds: int,
    tick: int,
    *,
    on_date: bool,
) -> _TruncatePlan:
    """Plan polars' per-unit truncation, for ticks of ``tick`` nanoseconds."""
    if (months > 0) + (weeks > 0) + (days > 0) + (nanoseconds > 0) != 1:
        # polars raises for a zero duration and for mixing units.
        raise NotImplementedError(f"dt.truncate with every={every!r}")
    day = _NANOSECONDS_PER_DAY // tick
    if months > 0:
        if months > _MAX_TRUNCATE_MONTHS:
            raise NotImplementedError(f"dt.truncate with every={every!r}")
        return _TruncatePlan(_TruncateMethod.MONTHLY, months, day, on_date)
    if weeks > 0:
        method, every_ticks = _TruncateMethod.WEEKLY, 7 * weeks * day
    else:
        method, every_ticks = _TruncateMethod.FIXED, days * day + nanoseconds // tick
        if every_ticks <= 1:
            return _TruncatePlan(_TruncateMethod.IDENTITY, 0, 0, on_date)
    if every_ticks > _MAX_TRUNCATE_TICKS:
        raise NotImplementedError(f"dt.truncate with every={every!r}")
    return _TruncatePlan(method, every_ticks, day, on_date)


def _int64_literal(value: int, stream: Stream) -> plc.expressions.Literal:
    return plc.expressions.Literal(plc.Scalar.from_py(value, _INT64, stream=stream))


def _evaluate_int64(
    columns: list[plc.Column], expression: plc.expressions.Expression, stream: Stream
) -> plc.Column:
    return plc.transform.compute_column(plc.Table(columns), expression, stream=stream)


def _wrapping(
    op: plc.binaryop.BinaryOperator,
    lhs: plc.Column,
    rhs: plc.Column | int,
    stream: Stream,
) -> plc.Column:
    """Apply ``op`` to INT64 values modulo 2**64, as polars' arithmetic does."""
    right = (
        plc.unary.bit_cast(rhs, _UINT64, stream=stream)
        if isinstance(rhs, plc.Column)
        else plc.Scalar.from_py(rhs % 2**64, _UINT64, stream=stream)
    )
    return plc.unary.bit_cast(
        plc.binaryop.binary_operation(
            plc.unary.bit_cast(lhs, _UINT64, stream=stream),
            right,
            op,
            _UINT64,
            stream=stream,
        ),
        _INT64,
        stream=stream,
    )


def _floor(ticks: plc.Column, every: int, stream: Stream) -> plc.Column:
    """``ticks - pymod(ticks, every)``, wrapping on overflow."""
    remainder = _evaluate_int64(
        [ticks],
        plc.expressions.Operation(
            _AST.PYMOD,
            plc.expressions.ColumnReference(0),
            _int64_literal(every, stream),
        ),
        stream,
    )
    return _wrapping(plc.binaryop.BinaryOperator.SUB, ticks, remainder, stream)


def _month_start_days(
    ticks: plc.Column,
    timestamp_type: plc.DataType,
    months: int,
    day: int,
    stream: Stream,
) -> plc.Column:
    """
    Days since the epoch of the first day of each value's month bucket.

    The number of months since 1970-01 is floored to a multiple of ``months``.

    libcudf's calendar functions use 16-bit years, so values are first moved
    by whole 400-year cycles into the years 1970 to 2369, where they are exact.
    The Gregorian calendar repeats every 400 years (146097 days, 4800 months),
    so the cycles are added back afterwards in INT64.
    """
    a, b, c = (plc.expressions.ColumnReference(i) for i in range(3))

    def lit(value: int) -> plc.expressions.Literal:
        return _int64_literal(value, stream)

    def op(
        operator: plc.expressions.ASTOperator, *operands: plc.expressions.Expression
    ) -> plc.expressions.Operation:
        return plc.expressions.Operation(operator, *operands)

    def evaluate(
        columns: list[plc.Column], expression: plc.expressions.Expression
    ) -> plc.Column:
        return _evaluate_int64(columns, expression, stream)

    def to_int64(column: plc.Column) -> plc.Column:
        return plc.unary.cast(column, _INT64, stream=stream)

    cycle = _DAYS_PER_400_YEARS * day
    if cycle < 2**63:
        cycles = evaluate([ticks], op(_AST.FLOOR_DIV, a, lit(cycle)))
        within_cycle = evaluate(
            [ticks, cycles], op(_AST.SUB, a, op(_AST.MUL, b, lit(cycle)))
        )
    else:
        # Nanosecond timestamps only cover the years 1677 to 2262.
        cycles = plc.Column.from_scalar(
            plc.Scalar.from_py(0, _INT64, stream=stream), ticks.size(), stream=stream
        )
        within_cycle = ticks
    timestamps = plc.unary.bit_cast(within_cycle, timestamp_type, stream=stream)
    year, month = (
        to_int64(
            plc.datetime.extract_datetime_component(
                timestamps, component, stream=stream
            )
        )
        for component in (
            plc.datetime.DatetimeComponent.YEAR,
            plc.datetime.DatetimeComponent.MONTH,
        )
    )
    # Months since 1970-01: (year - 1970 + 400 * cycles) * 12 + month - 1.
    months_since_epoch = op(
        _AST.ADD,
        op(
            _AST.MUL,
            op(_AST.ADD, op(_AST.SUB, a, lit(1970)), op(_AST.MUL, c, lit(400))),
            lit(12),
        ),
        op(_AST.SUB, b, lit(1)),
    )
    floored = evaluate(
        [year, month, cycles],
        op(
            _AST.SUB,
            months_since_epoch,
            op(_AST.PYMOD, months_since_epoch, lit(months)),
        ),
    )
    # Back to days, from the month within its 400-year cycle.
    cycles = evaluate([floored], op(_AST.FLOOR_DIV, a, lit(_MONTHS_PER_400_YEARS)))
    month_in_cycle = plc.unary.cast(
        evaluate(
            [floored, cycles],
            op(_AST.SUB, a, op(_AST.MUL, b, lit(_MONTHS_PER_400_YEARS))),
        ),
        plc.DataType(plc.TypeId.INT32),
        stream=stream,
    )
    epoch = plc.unary.bit_cast(
        plc.Column.from_scalar(
            plc.Scalar.from_py(0, plc.DataType(plc.TypeId.INT32), stream=stream),
            ticks.size(),
            stream=stream,
        ),
        plc.DataType(plc.TypeId.TIMESTAMP_DAYS),
        stream=stream,
    )
    days_in_cycle = to_int64(
        plc.unary.bit_cast(
            plc.datetime.add_calendrical_months(epoch, month_in_cycle, stream=stream),
            plc.DataType(plc.TypeId.INT32),
            stream=stream,
        )
    )
    return evaluate(
        [days_in_cycle, cycles],
        op(_AST.ADD, a, op(_AST.MUL, b, lit(_DAYS_PER_400_YEARS))),
    )


def _truncate(column: plc.Column, plan: _TruncatePlan, stream: Stream) -> plc.Column:
    """Evaluate a ``dt.truncate`` planned by :func:`_plan_truncate`."""
    method, every, day, on_date = plan
    if method is _TruncateMethod.IDENTITY:
        return column
    if method is _TruncateMethod.FLOOR_DATETIMES:
        return plc.datetime.floor_datetimes(
            column, plc.datetime.RoundingFrequency(every), stream=stream
        )
    if on_date:
        timestamp_type = plc.DataType(plc.TypeId.TIMESTAMP_MILLISECONDS)
        ticks = plc.unary.bit_cast(
            plc.unary.cast(column, timestamp_type, stream=stream), _INT64, stream=stream
        )
    else:
        timestamp_type = column.type()
        ticks = plc.unary.bit_cast(column, _INT64, stream=stream)
    if method is _TruncateMethod.FIXED:
        result = _floor(ticks, every, stream)
    elif method is _TruncateMethod.WEEKLY:
        # polars computes ``t - pymod(t - 4 days, every)``, wrapping on overflow.
        shifted = _wrapping(plc.binaryop.BinaryOperator.SUB, ticks, 4 * day, stream)
        remainder = _evaluate_int64(
            [shifted],
            plc.expressions.Operation(
                _AST.PYMOD,
                plc.expressions.ColumnReference(0),
                _int64_literal(every, stream),
            ),
            stream,
        )
        result = _wrapping(plc.binaryop.BinaryOperator.SUB, ticks, remainder, stream)
    else:
        result = _wrapping(
            plc.binaryop.BinaryOperator.MUL,
            _month_start_days(ticks, timestamp_type, every, day, stream),
            day,
            stream,
        )
    if not on_date:
        return plc.unary.bit_cast(result, column.type(), stream=stream)
    # polars divides by a day with integer division, which rounds toward zero,
    # and converts the result to 32 bits with wrapping.
    days = _evaluate_int64(
        [result],
        plc.expressions.Operation(
            _AST.DIV,
            plc.expressions.ColumnReference(0),
            _int64_literal(_MILLISECONDS_PER_DAY, stream),
        ),
        stream,
    )
    return plc.unary.bit_cast(
        plc.unary.cast(days, plc.DataType(plc.TypeId.INT32), stream=stream),
        column.type(),
        stream=stream,
    )


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

    __slots__ = ("ambiguous_scalar", "name", "options", "tzif_dirs")
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
    _ROUND_FREQ_MAP: ClassVar[dict[str, plc.datetime.RoundingFrequency]] = {
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
        Name.ConvertTimeZone,
        Name.ReplaceTimeZone,
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
        self.ambiguous_scalar = None
        self.tzif_dirs: tuple[str | None, str | None] = (None, None)
        if self.name not in self._valid_ops:
            raise NotImplementedError(f"Temporal function {self.name}")
        if self.name is TemporalFunction.Name.ToString and plc.traits.is_duration(
            self.children[0].dtype.plc_type
        ):
            raise NotImplementedError("ToString is not supported on duration types")
        elif self.name is TemporalFunction.Name.ReplaceTimeZone:
            from cudf_polars.dsl.expressions.literal import Literal

            ambiguous = self.children[1]
            if ambiguous.dtype.id() != plc.TypeId.STRING:
                raise SchemaError(
                    "invalid series dtype: expected `String`, got "
                    f"`{ambiguous.dtype.polars_type}`"
                )
            if isinstance(ambiguous, Literal):
                self.ambiguous_scalar = ambiguous.value
                if self.ambiguous_scalar is not None and self.ambiguous_scalar not in {
                    "earliest",
                    "latest",
                    "null",
                    "raise",
                }:
                    raise InvalidOperationError(
                        f"Invalid argument {self.ambiguous_scalar}, expected one of: "
                        '"earliest", "latest", "null", "raise"'
                    )
            from_zone = cast(
                "pl.Datetime", self.children[0].dtype.polars_type
            ).time_zone
            to_zone = self.options[0]
            tzif_dirs: list[str | None] = []
            for zone in (from_zone, to_zone):
                if zone is None or zone == "UTC":
                    # Normalize to not needing a tzif_dir lookup.
                    tzif_dirs.append(None)
                    continue
                tzif_dir = next(
                    (
                        search_path
                        for search_path in zoneinfo.TZPATH
                        if (Path(search_path) / zone).is_file()
                    ),
                    None,
                )
                if tzif_dir is None:
                    raise NotImplementedError(
                        f"Time zone {zone!r} not found in system time zone data "
                        "(zoneinfo.TZPATH)"
                    )
                tzif_dirs.append(tzif_dir)
            self.tzif_dirs = (tzif_dirs[0], tzif_dirs[1])
        elif self.name in {
            TemporalFunction.Name.Truncate,
            TemporalFunction.Name.Round,
        }:
            from cudf_polars.dsl.expressions.literal import Literal

            every_expr = self.children[1]
            if not isinstance(every_expr, Literal) or not isinstance(
                every_expr.value, str
            ):
                raise NotImplementedError(
                    f"dt.{self.name.name.lower()} with a non-literal every"
                )
            every = every_expr.value
            if self.name is TemporalFunction.Name.Truncate:
                self.options = _plan_truncate(self.children[0].dtype, every)
            else:
                match = re.fullmatch(r"(\d+)(ns|us|ms|s|m|h|d)", every)
                if match is None or int(match.group(1)) != 1:
                    # https://github.com/NVIDIA/cudf/issues/18654 to support non-1 buckets
                    raise NotImplementedError(f"Unsupported bucket: {every!r}")
                self.options = (self._ROUND_FREQ_MAP[match.group(2)],)

    def do_evaluate(
        self, df: DataFrame, *, context: ExecutionContext = ExecutionContext.FRAME
    ) -> Column:
        """Evaluate this expression given a dataframe for context."""
        columns = [child.evaluate(df, context=context) for child in self.children]
        if self.name is TemporalFunction.Name.ConvertTimeZone:
            (column,) = columns
            return Column(
                column.obj,
                dtype=self.dtype,
                is_sorted=column.is_sorted,
                order=column.order,
                null_order=column.null_order,
                name=column.name,
            )
        if self.name is TemporalFunction.Name.ReplaceTimeZone:
            column, ambiguous = columns
            from_zone = cast(
                "pl.Datetime", self.children[0].dtype.polars_type
            ).time_zone
            to_zone = self.options[0]
            non_existent = self.options[1]
            from_dir, to_dir = self.tzif_dirs
            from_zone_desc = (
                (from_zone, from_dir)
                if from_zone is not None and from_dir is not None
                else None
            )
            stream = df.stream
            same_zone = from_zone == to_zone or (from_dir is None and to_dir is None)
            if same_zone and (from_dir is None or self.ambiguous_scalar == "raise"):
                return Column(
                    column.obj,
                    dtype=self.dtype,
                    is_sorted=column.is_sorted,
                    order=column.order,
                    null_order=column.null_order,
                    name=column.name,
                )
            local = _local_wall_clock(column.obj, from_zone_desc, stream)
            if to_dir is None:
                return Column(
                    _apply_ambiguous_without_transitions(
                        local, self.ambiguous_scalar, ambiguous.obj, stream
                    ),
                    dtype=self.dtype,
                )
            return Column(
                _localize(
                    local,
                    to_zone,
                    to_dir,
                    self.ambiguous_scalar,
                    ambiguous.obj,
                    non_existent,
                    stream,
                ),
                dtype=self.dtype,
            )
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
                _truncate(column.obj, cast("_TruncatePlan", self.options), df.stream),
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
        elif self.name in {
            TemporalFunction.Name.MonthStart,
            TemporalFunction.Name.MonthEnd,
        }:
            (column,) = columns
            # Shift by a whole number of days so the time of day is kept
            day = plc.datetime.extract_datetime_component(
                column.obj, plc.datetime.DatetimeComponent.DAY, stream=df.stream
            )
            if self.name is TemporalFunction.Name.MonthStart:
                # day - 1 days back to the first of the month
                days_to_shift = plc.binaryop.binary_operation(
                    plc.Scalar.from_py(
                        1, plc.DataType(plc.TypeId.INT32), stream=df.stream
                    ),
                    day,
                    plc.binaryop.BinaryOperator.SUB,
                    plc.DataType(plc.TypeId.DURATION_DAYS),
                    stream=df.stream,
                )
            else:
                # days_in_month - day days forward to the last of the month
                days_to_shift = plc.binaryop.binary_operation(
                    plc.datetime.days_in_month(column.obj, stream=df.stream),
                    day,
                    plc.binaryop.BinaryOperator.SUB,
                    plc.DataType(plc.TypeId.DURATION_DAYS),
                    stream=df.stream,
                )
            result = plc.binaryop.binary_operation(
                column.obj,
                days_to_shift,
                plc.binaryop.BinaryOperator.ADD,
                self.dtype.plc_type,
                stream=df.stream,
            )
            return Column(result, dtype=self.dtype)
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

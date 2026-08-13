# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re

import numpy as np
import pandas as pd

import cudf

unit_to_nanoseconds_conversion = {
    "ns": 1,
    "us": 1_000,
    "ms": 1_000_000,
    "s": 1_000_000_000,
    "m": 60_000_000_000,
    "h": 3_600_000_000_000,
    "D": 86_400_000_000_000,
}

_unit_to_name = {
    "s": "second",
    "ms": "millisecond",
    "us": "microsecond",
    "ns": "nanosecond",
}


def raise_if_datetime_seconds_out_of_bounds(
    lo: pd.Timestamp, hi: pd.Timestamp, target_unit: str
) -> None:
    """Raise OutOfBoundsDatetime if [lo, hi] exceeds target_unit's range.

    The representable range in ``target_unit`` is +/-(2**63 - 1) of that
    unit since the epoch (the int64 minimum is the NaT sentinel).
    ``lo``/``hi`` are second-truncated values of the data being checked,
    so the range is widened to the enclosing whole seconds: values that
    only exceed the bounds within the boundary second are not detected,
    but in-bounds values are never rejected. ``lo``/``hi`` may be NaT
    (e.g. from an all-null column), whose comparisons are always False.
    """
    bound = np.iinfo(np.int64).max
    factor = unit_to_nanoseconds_conversion[target_unit]
    max_allowed = pd.Timestamp(np.datetime64((bound * factor) // 10**9, "s"))
    min_allowed = pd.Timestamp(
        np.datetime64(-((bound * factor + 10**9 - 1) // 10**9), "s")
    )
    offender = None
    if hi > max_allowed:
        offender = hi
    elif lo < min_allowed:
        offender = lo
    if offender is not None:
        raise pd.errors.OutOfBoundsDatetime(
            f"Out of bounds {_unit_to_name[target_unit]} timestamp: {offender}"
        )


def infer_format(element: str, **kwargs) -> str:
    """
    Infers datetime format from a string, also takes cares for `ms` and `ns`
    """
    if not cudf.get_option("mode.pandas_compatible"):
        # We allow "Z" but don't localize it to datetime64[ns, UTC] type (yet)
        element = element.replace("Z", "")
    fmt = pd.tseries.api.guess_datetime_format(element, **kwargs)

    if fmt is not None:
        if "%z" in fmt or "%Z" in fmt:
            raise NotImplementedError(
                "cuDF does not yet support timezone-aware datetimes"
            )
        if ".%f" not in fmt:
            # For context read:
            # https://github.com/pandas-dev/pandas/issues/52418
            # We cannot rely on format containing only %f
            # c++/libcudf expects .%3f, .%6f, .%9f
            # Logic below handles those cases well.
            return fmt

    element_parts = element.split(".")
    if len(element_parts) != 2:
        raise ValueError("Given date string not likely a datetime.")

    # There is possibility that the element is of following format
    # '00:00:03.333333 2016-01-01'
    second_parts = re.split(r"(\D+)", element_parts[1], maxsplit=1)
    subsecond_fmt = ".%" + str(len(second_parts[0])) + "f"

    first_part = pd.tseries.api.guess_datetime_format(
        element_parts[0], **kwargs
    )
    # For the case where first_part is '00:00:03'
    if first_part is None:
        tmp = "1970-01-01 " + element_parts[0]
        tmp_fmt = pd.tseries.api.guess_datetime_format(tmp, **kwargs)
        if tmp_fmt is None:
            raise ValueError(
                "Unable to infer the timestamp format from the data"
            )
        first_part = tmp_fmt.split(" ", 1)[1]
    if first_part is None:
        raise ValueError("Unable to infer the timestamp format from the data")

    if len(second_parts) > 1:
        # We may have a non-digit, timezone-like component
        # like Z, UTC-3, +01:00
        if any(re.search(r"\D", part) for part in second_parts):
            raise NotImplementedError(
                "cuDF does not yet support timezone-aware datetimes"
            )
        second_part = "".join(second_parts[1:])

        if len(second_part) > 1:
            # Only infer if second_parts is not an empty string.
            second_part = pd.tseries.api.guess_datetime_format(
                second_part, **kwargs
            )
    else:
        second_part = ""

    try:
        fmt = first_part + subsecond_fmt + second_part
    except Exception:
        raise ValueError("Unable to infer the timestamp format from the data")

    return fmt

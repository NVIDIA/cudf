# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import datetime
import zoneinfo

import pandas as pd

from cudf.options import get_option


def get_compatible_timezone(dtype: pd.DatetimeTZDtype) -> pd.DatetimeTZDtype:
    """Convert dtype.tz object to zoneinfo object if possible."""
    tz = dtype.tz
    if isinstance(tz, zoneinfo.ZoneInfo):
        return dtype
    if get_option("mode.pandas_compatible"):
        raise NotImplementedError(
            f"{tz} must be a zoneinfo.ZoneInfo object in pandas_compatible mode."
        )
    elif (tzname := getattr(tz, "zone", None)) is not None:
        # pytz-like
        key = tzname
    elif (tz_file := getattr(tz, "_filename", None)) is not None:
        # dateutil-like
        key = tz_file.split("zoneinfo/")[-1]
    elif isinstance(tz, datetime.tzinfo):
        # Try to get UTC-like tzinfos
        reference = datetime.datetime.now()
        key = tz.tzname(reference)
        if not (isinstance(key, str) and key.lower() == "utc"):
            raise NotImplementedError(f"cudf does not support {tz}")
    else:
        raise NotImplementedError(f"cudf does not support {tz}")
    new_tz = zoneinfo.ZoneInfo(key)
    return pd.DatetimeTZDtype(dtype.unit, new_tz)

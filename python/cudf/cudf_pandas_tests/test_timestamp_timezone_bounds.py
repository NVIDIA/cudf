# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

from cudf.core._compat import PANDAS_GE_31
from cudf.pandas.fast_slow_proxy import _slow_arg
from cudf.pandas.module_accelerator import disable_module_accelerator


@pytest.mark.skipif(
    not PANDAS_GE_31,
    reason="Timezone sentinel underflow is checked in pandas 3.1",
)
@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("constructor", [False, True])
def test_timestamp_timezone_sentinel_underflow(unit, constructor):
    # Tokyo's historical local offset makes this wall time land on NaT.
    offset = np.timedelta64(9 * 3600 + 18 * 60 + 59, "s")
    ticks = -(2**63) + int(offset // np.timedelta64(1, unit))
    value = np.datetime64(ticks, unit)

    def localize():
        if constructor:
            return pd.Timestamp(value, tz="Asia/Tokyo")
        return pd.Timestamp(value).tz_localize("Asia/Tokyo")

    with disable_module_accelerator():
        with pytest.raises(pd.errors.OutOfBoundsDatetime) as expected:
            localize()

    with pytest.raises(type(expected.value), match="underflows past"):
        localize()


@pytest.mark.parametrize("timezone", ["UTC", "Asia/Tokyo", "US/Pacific"])
def test_timestamp_timezone_roundtrip(timezone):
    with disable_module_accelerator():
        expected = pd.Timestamp("2000-01-01").tz_localize(timezone)
    result = pd.Timestamp("2000-01-01").tz_localize(timezone)

    with disable_module_accelerator():
        assert _slow_arg(result) == expected
        assert _slow_arg(result).tz == expected.tz

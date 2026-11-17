# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("box", [pd.Series, pd.IntervalIndex])
@pytest.mark.parametrize("tz", ["US/Eastern", None])
def test_interval_with_datetime(tz, box):
    dti = pd.date_range(
        start=pd.Timestamp("20180101", tz=tz),
        end=pd.Timestamp("20181231", tz=tz),
        freq="ME",
    )
    pobj = box(pd.IntervalIndex.from_breaks(dti))
    if tz is None:
        gobj = cudf.from_pandas(pobj)
        assert_eq(pobj, gobj)
    else:
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pobj)

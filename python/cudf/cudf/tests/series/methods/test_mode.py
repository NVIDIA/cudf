# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "gs",
    [
        lambda: cudf.Series([1, 2, 3]),
        lambda: cudf.Series([None]),
        lambda: cudf.Series([4]),
        lambda: cudf.Series([2, 3, -1, 0, 1], name="test name"),
        lambda: cudf.Series(
            [1, 2, 3, None, 2, 1], index=["a", "v", "d", "e", "f", "g"]
        ),
        lambda: cudf.Series([1, 2, 3, None, 2, 1, None], name="abc"),
        lambda: cudf.Series(["ab", "bc", "ab", None, "bc", None, None]),
        lambda: cudf.Series([None, None, None, None, None], dtype="str"),
        lambda: cudf.Series([None, None, None, None, None]),
        lambda: cudf.Series(
            [
                123213,
                23123,
                123123,
                12213123,
                12213123,
                12213123,
                23123,
                2312323123,
                None,
                None,
            ],
            dtype="timedelta64[ns]",
        ),
        lambda: cudf.Series(
            [
                None,
                1,
                2,
                3242434,
                3233243,
                1,
                2,
                1023,
                None,
                12213123,
                None,
                2312323123,
                None,
                None,
            ],
            dtype="datetime64[ns]",
        ),
        lambda: cudf.Series(name="empty series", dtype="float64"),
        lambda: cudf.Series(
            ["a", "b", "c", " ", "a", "b", "z"], dtype="category"
        ),
    ],
)
def test_series_mode(gs, dropna):
    gs = gs()
    ps = gs.to_pandas()

    expected = ps.mode(dropna=dropna)
    actual = gs.mode(dropna=dropna)

    if (
        not dropna
        and len(gs)
        and gs.null_count == len(gs)
        and isinstance(gs.dtype, np.dtype)
        and gs.dtype.kind == "O"
    ):
        # cudf coerces an all-null object column to float64 (groupby, used
        # internally by ``mode``, labels an all-null object key with a float64
        # NaN), so the all-null mode is float64 NaN rather than pandas' object
        # None.
        expected = expected.astype("float64")

    assert_eq(expected, actual, check_dtype=False)


@pytest.mark.parametrize(
    "pdata",
    [
        pd.Series([1, 1, 2, None, None], dtype="datetime64[ns]"),
        pd.Series([1, 1, 2, None, None], dtype="timedelta64[ns]"),
        pd.Series(
            pd.Categorical(
                [1, 1, 2, 3, 3, np.nan, np.nan],
                categories=[3, 2, 1],
                ordered=True,
            )
        ),
        pd.Series(pd.Categorical([1, 2, np.nan, np.nan])),
        pd.Series([1, 1, None, None], dtype="Int64"),
        pd.Series([1.0, 1.0, np.nan, np.nan]),
        pd.Series([1, 1, None, None], dtype=pd.ArrowDtype(pa.timestamp("ns"))),
        pd.Series([1, 1, None, None], dtype=pd.ArrowDtype(pa.duration("ns"))),
    ],
)
def test_mode_null_position_matches_pandas(pdata, dropna):
    # pandas sorts mode results on the underlying representation: NaT
    # (INT64_MIN as i8) and the categorical null code (-1) sort before
    # valid values, while float NaN and the <NA> of nullable and arrow
    # dtypes (including arrow timestamps/durations) sort last.
    gs = cudf.from_pandas(pdata)
    assert_eq(pdata.mode(dropna=dropna), gs.mode(dropna=dropna))

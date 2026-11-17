# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import datetime

import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "start_time, end_time, kwargs",
    [
        ["09:00", "15:00", {}],
        ["09:00", "15:00", {"include_start": False}],
        ["09:00", "15:00", {"include_end": False}],
        ["09:00", "15:00", {"include_start": False, "include_end": False}],
        # wrap-around midnight (start_time > end_time)
        ["23:00", "01:00", {}],
        ["23:00", "01:00", {"include_start": False, "include_end": False}],
        # datetime.time objects and 12-hour strings
        [datetime.time(9, 0), datetime.time(12, 0), {}],
        ["9:00am", "3:00pm", {}],
        # no matches
        ["03:00", "04:00", {}],
        # equal bounds with the inclusion flags
        ["12:00", "12:00", {}],
        ["12:00", "12:00", {"include_start": False}],
        ["12:00", "12:00", {"include_start": False, "include_end": False}],
    ],
)
def test_indexer_between_time(start_time, end_time, kwargs):
    pd_dti = pd.DatetimeIndex(
        [
            "2024-01-01 09:00:00",
            "2024-01-01 12:00:00",
            "2024-01-01 15:00:00",
            "2024-01-01 23:30:00",
            "2024-01-02 00:30:00",
        ],
        name="foo",
    )
    cudf_dti = cudf.from_pandas(pd_dti)

    expected = pd_dti.indexer_between_time(start_time, end_time, **kwargs)
    result = cudf_dti.indexer_between_time(start_time, end_time, **kwargs)

    assert_eq(result, expected)


@pytest.mark.parametrize(
    "start_time, end_time", [("09:00", "15:00"), ("23:00", "01:00")]
)
def test_indexer_between_time_arrow_dtype(start_time, end_time):
    # ArrowDtype regression: pandas has no arrow-backed DatetimeIndex, so the
    # NumPy-backed index is the oracle.
    pd_dti = pd.DatetimeIndex(
        [
            "2024-01-01 09:00:00",
            "2024-01-01 12:00:00",
            "2024-01-01 15:00:00",
            "2024-01-01 23:30:00",
            "2024-01-02 00:30:00",
        ],
        name="foo",
    )
    cudf_dti = cudf.from_pandas(pd_dti).astype(
        pd.ArrowDtype(pa.timestamp("us"))
    )

    assert_eq(
        cudf_dti.indexer_between_time(start_time, end_time),
        pd_dti.indexer_between_time(start_time, end_time),
    )


def test_indexer_between_time_nat():
    # NaT entries are excluded, matching pandas
    pd_dti = pd.DatetimeIndex(
        ["2024-01-01 09:00:00", "NaT", "2024-01-01 12:00:00"]
    )
    cudf_dti = cudf.from_pandas(pd_dti)

    assert_eq(
        cudf_dti.indexer_between_time("08:00", "13:00"),
        pd_dti.indexer_between_time("08:00", "13:00"),
    )


def test_indexer_between_time_nat_wraparound():
    # In the wrap-around branch NaT is *included* by pandas (its -1 sentinel
    # satisfies the `<= end` predicate); cudf must match.
    pd_dti = pd.DatetimeIndex(
        [
            "2024-01-01 09:00:00",
            "NaT",
            "2024-01-01 23:30:00",
            "2024-01-02 00:30:00",
        ]
    )
    cudf_dti = cudf.from_pandas(pd_dti)

    assert_eq(
        cudf_dti.indexer_between_time("23:00", "01:00"),
        pd_dti.indexer_between_time("23:00", "01:00"),
    )


def test_indexer_between_time_nanosecond_resolution():
    # pandas truncates to microsecond resolution; cudf must do the same even
    # for a nanosecond-resolution index with a sub-microsecond component.
    pd_dti = pd.DatetimeIndex(
        ["2024-01-01 09:00:00.000000999", "2024-01-01 12:00:00"]
    )
    cudf_dti = cudf.from_pandas(pd_dti)

    assert_eq(
        cudf_dti.indexer_between_time("09:00:00", "09:00:00"),
        pd_dti.indexer_between_time("09:00:00", "09:00:00"),
    )


@pytest.mark.parametrize("tz", ["US/Eastern", "UTC", "Asia/Kolkata"])
@pytest.mark.parametrize(
    "start_time, end_time", [("08:00", "10:00"), ("23:00", "01:00")]
)
def test_indexer_between_time_timezone(tz, start_time, end_time):
    # A tz-aware index must filter on local wall-clock time, matching pandas
    # (which uses the local timestamps), including the wrap-around branch.
    pd_dti = pd.DatetimeIndex(
        [
            "2024-01-01 09:00:00",
            "2024-01-01 23:30:00",
            "2024-01-02 00:30:00",
        ],
        tz=tz,
    )
    cudf_dti = cudf.from_pandas(pd_dti)

    assert_eq(
        cudf_dti.indexer_between_time(start_time, end_time),
        pd_dti.indexer_between_time(start_time, end_time),
    )


def test_indexer_between_time_empty():
    pd_dti = pd.DatetimeIndex([])
    cudf_dti = cudf.from_pandas(pd_dti)

    assert_eq(
        cudf_dti.indexer_between_time("09:00", "15:00"),
        pd_dti.indexer_between_time("09:00", "15:00"),
    )


@pytest.mark.parametrize(
    "start_time, end_time", [("09:00", "15:00"), ("23:00", "01:00")]
)
def test_indexer_between_time_all_nat(start_time, end_time):
    # In the wrap-around branch all NaT rows are included because their -1
    # sentinel satisfies the `<= end` predicate; a normal range excludes them.
    pd_dti = pd.DatetimeIndex(["NaT", "NaT", "NaT"])
    cudf_dti = cudf.from_pandas(pd_dti)

    assert_eq(
        cudf_dti.indexer_between_time(start_time, end_time),
        pd_dti.indexer_between_time(start_time, end_time),
    )


@pytest.mark.parametrize(
    "start_time, end_time",
    [("09:00", "15:00"), ("13:00", "14:00"), ("18:00", "13:00")],
)
def test_indexer_between_time_single_element(start_time, end_time):
    pd_dti = pd.DatetimeIndex(["2024-01-01 12:00:00"])
    cudf_dti = cudf.from_pandas(pd_dti)

    assert_eq(
        cudf_dti.indexer_between_time(start_time, end_time),
        pd_dti.indexer_between_time(start_time, end_time),
    )

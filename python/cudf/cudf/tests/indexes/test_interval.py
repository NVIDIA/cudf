# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core.index import IntervalIndex, interval_range
from cudf.testing import assert_eq


def test_interval_constructor_default_closed():
    idx = cudf.IntervalIndex([pd.Interval(0, 1)])
    assert idx.closed == "right"
    assert idx.dtype.closed == "right"


def test_interval_to_arrow():
    expect = pa.Array.from_pandas(pd.IntervalIndex([pd.Interval(0, 1)]))
    got = cudf.IntervalIndex([pd.Interval(0, 1)]).to_arrow()
    assert_eq(expect, got)


INTERVAL_BOUNDARY_TYPES = [
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float32,
    np.float64,
]

PERIODS_TYPES = [
    int,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
]


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("start", [0, 1, 2, 3])
@pytest.mark.parametrize("end", [4, 5, 6, 7])
def test_interval_range_basic(start, end, closed):
    pindex = pd.interval_range(start=start, end=end, closed=closed)
    gindex = cudf.interval_range(start=start, end=end, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("start_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("end_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_dtype_basic(start_t, end_t):
    start, end = start_t(24), end_t(42)
    pindex = pd.interval_range(start=start, end=end, closed="left")
    gindex = cudf.interval_range(start=start, end=end, closed="left")

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
def test_interval_range_empty(closed):
    pindex = pd.interval_range(start=0, end=0, closed=closed)
    gindex = cudf.interval_range(start=0, end=0, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("freq", [1, 2, 3])
@pytest.mark.parametrize("start", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("end", [6, 8, 10, 43, 70])
def test_interval_range_freq_basic(start, end, freq, closed):
    pindex = pd.interval_range(start=start, end=end, freq=freq, closed=closed)
    gindex = cudf.interval_range(
        start=start, end=end, freq=freq, closed=closed
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("start_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("end_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("freq_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_freq_basic_dtype(start_t, end_t, freq_t):
    start, end, freq = start_t(5), end_t(70), freq_t(3)
    pindex = pd.interval_range(start=start, end=end, freq=freq, closed="left")
    gindex = cudf.interval_range(
        start=start, end=end, freq=freq, closed="left"
    )
    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("periods", [1, 2, 3])
@pytest.mark.parametrize("start", [0, 0.0, 1.0, 1, 2, 2.0, 3.0, 3])
@pytest.mark.parametrize("end", [4, 4.0, 5.0, 5, 6, 6.0, 7.0, 7])
def test_interval_range_periods_basic(start, end, periods, closed):
    pindex = pd.interval_range(
        start=start, end=end, periods=periods, closed=closed
    )
    gindex = cudf.interval_range(
        start=start, end=end, periods=periods, closed=closed
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("start_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("end_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("periods_t", PERIODS_TYPES)
def test_interval_range_periods_basic_dtype(start_t, end_t, periods_t):
    start, end, periods = start_t(0), end_t(4), periods_t(1)
    pindex = pd.interval_range(
        start=start, end=end, periods=periods, closed="left"
    )
    gindex = cudf.interval_range(
        start=start, end=end, periods=periods, closed="left"
    )

    assert_eq(pindex, gindex)


def test_interval_range_periods_non_integer_raises():
    start_val, end_val, periods_val = 0, 4, 1.0

    with pytest.raises(TypeError):
        pd.interval_range(
            start=start_val, end=end_val, periods=periods_val, closed="left"
        )
    with pytest.raises(TypeError):
        cudf.interval_range(
            start=start_val, end=end_val, periods=periods_val, closed="left"
        )


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("periods", [1, 2, 3])
@pytest.mark.parametrize("freq", [1, 2, 3, 4])
@pytest.mark.parametrize("end", [4, 8, 9, 10])
def test_interval_range_periods_freq_end(end, freq, periods, closed):
    pindex = pd.interval_range(
        end=end, freq=freq, periods=periods, closed=closed
    )
    gindex = cudf.interval_range(
        end=end, freq=freq, periods=periods, closed=closed
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("periods_t", PERIODS_TYPES)
@pytest.mark.parametrize("freq_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("end_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_periods_freq_end_dtype(periods_t, freq_t, end_t):
    periods, freq, end = periods_t(2), freq_t(3), end_t(10)
    pindex = pd.interval_range(
        end=end, freq=freq, periods=periods, closed="left"
    )
    gindex = cudf.interval_range(
        end=end, freq=freq, periods=periods, closed="left"
    )
    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
@pytest.mark.parametrize("periods", [1, 2, 3])
@pytest.mark.parametrize("freq", [1, 2, 3, 4])
@pytest.mark.parametrize("start", [1, 4, 9, 12])
def test_interval_range_periods_freq_start(start, freq, periods, closed):
    pindex = pd.interval_range(
        start=start, freq=freq, periods=periods, closed=closed
    )
    gindex = cudf.interval_range(
        start=start, freq=freq, periods=periods, closed=closed
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("periods_t", PERIODS_TYPES)
@pytest.mark.parametrize("freq_t", INTERVAL_BOUNDARY_TYPES)
@pytest.mark.parametrize("start_t", INTERVAL_BOUNDARY_TYPES)
def test_interval_range_periods_freq_start_dtype(periods_t, freq_t, start_t):
    periods, freq, start = periods_t(2), freq_t(3), start_t(9)
    pindex = pd.interval_range(
        start=start, freq=freq, periods=periods, closed="left"
    )
    gindex = cudf.interval_range(
        start=start, freq=freq, periods=periods, closed="left"
    )
    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["right", "left", "both", "neither"])
@pytest.mark.parametrize(
    "data",
    [
        ([pd.Interval(30, 50)]),
        ([pd.Interval(0, 3), pd.Interval(1, 7)]),
        ([pd.Interval(0.2, 60.3), pd.Interval(1, 7), pd.Interval(0, 0)]),
        ([]),
    ],
)
def test_interval_index_basic(data, closed):
    pindex = pd.IntervalIndex(data, closed=closed)
    gindex = IntervalIndex(data, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["right", "left", "both", "neither"])
def test_interval_index_empty(closed):
    pindex = pd.IntervalIndex([], closed=closed)
    gindex = IntervalIndex([], closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["right", "left", "both", "neither"])
@pytest.mark.parametrize(
    "data",
    [
        ([pd.Interval(1, 6), pd.Interval(1, 10), pd.Interval(1, 3)]),
        (
            [
                pd.Interval(3.5, 6.0),
                pd.Interval(1.0, 7.0),
                pd.Interval(0.0, 10.0),
            ]
        ),
        (
            [
                pd.Interval(50, 100, closed="left"),
                pd.Interval(1.0, 7.0, closed="left"),
                pd.Interval(16, 322, closed="left"),
            ]
        ),
        (
            [
                pd.Interval(50, 100, closed="right"),
                pd.Interval(1.0, 7.0, closed="right"),
                pd.Interval(16, 322, closed="right"),
            ]
        ),
    ],
)
def test_interval_index_many_params(data, closed):
    pindex = pd.IntervalIndex(data, closed=closed)
    gindex = IntervalIndex(data, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
def test_interval_index_from_breaks(closed):
    breaks = [0, 3, 6, 10]
    pindex = pd.IntervalIndex.from_breaks(breaks, closed=closed)
    gindex = IntervalIndex.from_breaks(breaks, closed=closed)

    assert_eq(pindex, gindex)


@pytest.mark.parametrize(
    "start, stop, freq, periods",
    [
        (0.0, None, 0.2, 5),
        (0.0, 1.0, None, 5),
        (
            0.0,
            1.0,
            0.2,
            None,
        ),
        (None, 1.0, 0.2, 5),
        (
            0.0,
            1.0,
            0.1,
            None,
        ),
        (0.0, 1.0, None, 10),
        (0.0, None, 0.25, 4),
        (1.0, None, 2.5, 2),
    ],
)
def test_interval_range_floating(start, stop, freq, periods):
    expected = pd.interval_range(
        start=start, end=stop, freq=freq, periods=periods
    )
    got = interval_range(start=start, end=stop, freq=freq, periods=periods)
    assert_eq(expected, got)


def test_intervalindex_empty_typed_non_int():
    data = np.array([], dtype="datetime64[ns]")
    result = cudf.IntervalIndex(data)
    expected = pd.IntervalIndex(data)
    assert_eq(result, expected)


def test_intervalindex_invalid_dtype():
    with pytest.raises(TypeError):
        cudf.IntervalIndex([pd.Interval(1, 2)], dtype="int64")


def test_intervalindex_conflicting_closed():
    with pytest.raises(ValueError):
        cudf.IntervalIndex(
            [pd.Interval(1, 2)],
            dtype=cudf.IntervalDtype("int64", closed="left"),
            closed="right",
        )


def test_intervalindex_invalid_data():
    with pytest.raises(TypeError):
        cudf.IntervalIndex([1, 2])


@pytest.mark.parametrize(
    "attr",
    [
        "is_empty",
        "length",
        "left",
        "right",
        "mid",
    ],
)
def test_intervalindex_properties(attr):
    pd_ii = pd.IntervalIndex.from_arrays([0, 1], [0, 2])
    cudf_ii = cudf.from_pandas(pd_ii)

    result = getattr(cudf_ii, attr)
    expected = getattr(pd_ii, attr)
    assert_eq(result, expected)


def test_set_closed():
    data = [pd.Interval(0, 1)]
    result = cudf.IntervalIndex(data).set_closed("both")
    expected = pd.IntervalIndex(data).set_closed("both")
    assert_eq(result, expected)


def test_from_tuples():
    data = [(1, 2), (10, 20)]
    result = cudf.IntervalIndex.from_tuples(data, closed="left", name="a")
    expected = pd.IntervalIndex.from_tuples(data, closed="left", name="a")
    assert_eq(result, expected)


def test_interval_range_name():
    expected = pd.interval_range(start=0, periods=5, freq=2, name="foo")
    result = cudf.interval_range(start=0, periods=5, freq=2, name="foo")
    assert_eq(result, expected)


def test_from_interval_range_indexing():
    result = cudf.interval_range(start=0, end=1, name="a").repeat(2)
    expected = pd.interval_range(start=0, end=1, name="a").repeat(2)
    assert_eq(result, expected)


def test_interval_equality_series_eq():
    s = cudf.Series(pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3]))
    result = s == s
    expected = cudf.Series([True, True, True])
    assert_eq(result, expected)


def test_interval_equality_series_ne():
    s = cudf.Series(pd.arrays.IntervalArray.from_breaks([0, 1, 2, 3]))
    other = cudf.Series(pd.arrays.IntervalArray.from_breaks([0, 1, 2, 4]))
    result = s != other
    expected = cudf.Series([False, False, True])
    assert_eq(result, expected)


def test_interval_equality_series_eq_with_nulls():
    pi = pd.arrays.IntervalArray.from_breaks([0.0, 1.0, 2.0, 3.0])
    ps = pd.array(pi, dtype=pi.dtype)
    ps[1] = pd.NA
    s = cudf.Series(ps)
    result = s == s
    # cudf uses NULL_EQUALS semantics: null == null is True
    expected = cudf.Series([True, True, True])
    assert_eq(result, expected)


def test_interval_equality_index_eq():
    idx = cudf.IntervalIndex.from_breaks([0, 1, 2, 3])
    other = cudf.IntervalIndex.from_breaks([0, 1, 5, 3])
    result = idx == other
    expected = np.array([True, False, False])
    np.testing.assert_array_equal(result.get(), expected)


@pytest.mark.parametrize("closed", ["left", "right", "both", "neither"])
def test_interval_equality_eq_respects_closed(closed):
    s = cudf.Series(
        pd.arrays.IntervalArray.from_breaks([0, 1, 2], closed=closed)
    )
    result = s == s
    expected = cudf.Series([True, True])
    assert_eq(result, expected)

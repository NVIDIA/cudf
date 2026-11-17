# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def assert_resample_results_equal(lhs, rhs, **kwargs):
    assert_eq(
        lhs.sort_index(),
        rhs.sort_index(),
        check_dtype=False,
        check_freq=False,
        check_index_type=False,
        **kwargs,
    )


@pytest.mark.parametrize("ts_resolution", ["ns", "s", "ms"])
def test_series_downsample_simple(ts_resolution):
    # Series with and index of 5min intervals:

    index = pd.date_range(start="2001-01-01", periods=10, freq="1min")
    psr = pd.Series(range(10), index=index)
    gsr = cudf.from_pandas(psr)
    gsr.index = gsr.index.astype(f"datetime64[{ts_resolution}]")
    assert_resample_results_equal(
        psr.resample("3min").sum(),
        gsr.resample("3min").sum(),
        check_index=False,
    )


def test_series_upsample_simple():
    # Series with and index of 5min intervals:

    index = pd.date_range(start="2001-01-01", periods=10, freq="1min")
    psr = pd.Series(range(10), index=index)
    gsr = cudf.from_pandas(psr)
    assert_resample_results_equal(
        psr.resample("3min").sum(),
        gsr.resample("3min").sum(),
        check_index=False,
    )


def test_series_resample_preserves_frequency():
    index = pd.DatetimeIndex(
        pd.date_range("2020-01-01", freq="D", periods=3).to_numpy()
    )
    psr = pd.Series(range(3), index=index)
    gsr = cudf.from_pandas(psr)

    expected = psr.resample("D").max()
    result = gsr.resample("D").max()

    assert_eq(result, expected, check_dtype=False, check_index_type=False)


@pytest.mark.parametrize("rule", ["2s", "10s"])
def test_series_resample_ffill(rule):
    date_idx = pd.date_range("1/1/2012", periods=10, freq="5s")
    rng = np.random.default_rng(seed=0)
    ts = pd.Series(rng.integers(0, 500, len(date_idx)), index=date_idx)
    gts = cudf.from_pandas(ts)
    assert_resample_results_equal(
        ts.resample(rule).ffill(), gts.resample(rule).ffill()
    )


@pytest.mark.parametrize("rule", ["2s", "10s"])
def test_series_resample_bfill(rule):
    date_idx = pd.date_range("1/1/2012", periods=10, freq="5s")
    rng = np.random.default_rng(seed=0)
    ts = pd.Series(rng.integers(0, 500, len(date_idx)), index=date_idx)
    gts = cudf.from_pandas(ts)
    assert_resample_results_equal(
        ts.resample(rule).bfill(), gts.resample(rule).bfill()
    )


@pytest.mark.parametrize("rule", ["2s", "10s"])
def test_series_resample_asfreq(rule):
    date_range = pd.date_range("1/1/2012", periods=100, freq="5s")
    rng = np.random.default_rng(seed=0)
    ts = pd.Series(rng.integers(0, 500, len(date_range)), index=date_range)
    gts = cudf.from_pandas(ts)
    assert_resample_results_equal(
        ts.resample(rule).asfreq(), gts.resample(rule).asfreq()
    )


def test_dataframe_resample_aggregation_simple():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        rng.standard_normal(size=(1000, 3)),
        index=pd.date_range("1/1/2012", freq="s", periods=1000),
        columns=["A", "B", "C"],
    )
    gdf = cudf.from_pandas(pdf)
    assert_resample_results_equal(
        pdf.resample("3min").mean(), gdf.resample("3min").mean()
    )


def test_dataframe_resample_multiagg():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        rng.standard_normal(size=(1000, 3)),
        index=pd.date_range("1/1/2012", freq="s", periods=1000),
        columns=["A", "B", "C"],
    )
    gdf = cudf.from_pandas(pdf)
    assert_resample_results_equal(
        pdf.resample("3min").agg(["sum", "mean", "std"]),
        gdf.resample("3min").agg(["sum", "mean", "std"]),
    )


def test_dataframe_resample_on():
    rng = np.random.default_rng(seed=0)
    # test resampling on a specified column
    pdf = pd.DataFrame(
        {
            "x": rng.standard_normal(size=(1000)),
            "y": pd.date_range("1/1/2012", freq="s", periods=1000),
        }
    )
    gdf = cudf.from_pandas(pdf)
    assert_resample_results_equal(
        pdf.resample("3min", on="y").mean(),
        gdf.resample("3min", on="y").mean(),
    )


def test_dataframe_resample_level():
    rng = np.random.default_rng(seed=0)
    # test resampling on a specific level of a MultIndex
    pdf = pd.DataFrame(
        {
            "x": rng.standard_normal(size=1000),
            "y": pd.date_range("1/1/2012", freq="s", periods=1000),
        }
    )
    pdi = pd.MultiIndex.from_frame(pdf)
    pdf = pd.DataFrame({"a": rng.standard_normal(size=1000)}, index=pdi)
    gdf = cudf.from_pandas(pdf)
    assert_resample_results_equal(
        pdf.resample("3min", level="y").mean(),
        gdf.resample("3min", level="y").mean(),
    )


@pytest.mark.parametrize(
    "in_freq, sampling_freq",
    [
        ("1ns", "1us"),
        ("1us", "10us"),
        ("ms", "100us"),
        ("ms", "1s"),
        ("s", "1min"),
        ("1min", "30s"),
        ("1D", "10D"),
        ("10D", "1D"),
    ],
)
def test_resampling_frequency_conversion(in_freq, sampling_freq):
    rng = np.random.default_rng(seed=0)
    # Pandas resample preserves the input column's unit; verify cuDF
    # matches that behavior across sampling frequencies.
    pdf = pd.DataFrame(
        {
            "x": rng.standard_normal(size=100),
            "y": pd.date_range("1/1/2012", freq=in_freq, periods=100),
        }
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.resample(sampling_freq, on="y").mean()
    got = gdf.resample(sampling_freq, on="y").mean()
    assert_resample_results_equal(expect, got)

    assert got.index.dtype == pdf["y"].dtype


def test_resampling_downsampling_ms():
    pdf = pd.DataFrame(
        {
            "time": pd.date_range("2020-01-01", periods=5, freq="1ns"),
            "sign": range(5),
        }
    )
    gdf = cudf.from_pandas(pdf)
    expected = pdf.resample("10ms", on="time").mean()
    result = gdf.resample("10ms", on="time").mean()
    result.index = result.index.astype("datetime64[ns]")
    assert_eq(result, expected, check_freq=False)


@pytest.mark.parametrize("input_unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("freq", ["D", "h", "30min"])
@pytest.mark.parametrize(
    "agg", ["mean", "sum", "min", "max", "first", "last", "count", "var"]
)
def test_resample_empty_preserves_input_unit_and_freq(input_unit, freq, agg):
    # Resample on an empty datetime index must preserve the input column's
    # unit (pandas behavior; cuDF previously collapsed to [s] for D/h offsets)
    # and must keep the offset attached to the result index.
    idx = pd.DatetimeIndex([], dtype=f"datetime64[{input_unit}]", name="t")
    pser = pd.Series([], index=idx, dtype=float)
    gser = cudf.from_pandas(pser)

    expected = getattr(pser.resample(freq), agg)()
    actual = getattr(gser.resample(freq), agg)()

    assert actual.index.dtype == expected.index.dtype
    assert_eq(actual, expected, check_dtype=False, check_index_type=False)


def test_resample_size_matches_pandas_with_empty_buckets():
    # GroupBy.size bypasses _Resampler.agg, so the freq has to be re-attached
    # by the size override; empty buckets must come back as 0 (not NaN), and
    # the result must be sorted by bin label like pandas.
    idx = pd.date_range("2020-01-01", periods=4, freq="1h")
    pser = pd.Series(range(4), index=idx)
    gser = cudf.from_pandas(pser)

    expected = pser.resample("30min").size()
    actual = gser.resample("30min").size()

    assert_eq(actual, expected, check_dtype=False, check_index_type=False)

    # Empty case
    idx_empty = pd.DatetimeIndex([], dtype="datetime64[us]", name="t")
    pser_empty = pd.Series([], index=idx_empty, dtype=float)
    gser_empty = cudf.from_pandas(pser_empty)

    expected_empty = pser_empty.resample("h").size()
    actual_empty = gser_empty.resample("h").size()

    assert_eq(
        actual_empty,
        expected_empty,
        check_dtype=False,
        check_index_type=False,
    )

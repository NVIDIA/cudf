# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture
def gdf_pdf():
    dates = cudf.date_range("2018-04-09", periods=4, freq="1D20min")
    df = cudf.DataFrame({"A": [1, 2, 3, 4]}, index=dates)
    return df, df.to_pandas()


def test_between_time_basic(gdf_pdf):
    gdf, pdf = gdf_pdf
    expected = pdf.between_time("0:15", "0:45")
    actual = gdf.between_time("0:15", "0:45")
    assert_eq(actual, expected)


@pytest.mark.parametrize("inclusive", ["both", "neither", "left", "right"])
def test_between_time_inclusive_modes(gdf_pdf, inclusive):
    gdf, pdf = gdf_pdf
    expected = pdf.between_time("0:20", "0:40", inclusive=inclusive)
    actual = gdf.between_time("0:20", "0:40", inclusive=inclusive)
    assert_eq(actual, expected)


def test_between_time_wraparound(gdf_pdf):
    gdf, pdf = gdf_pdf
    expected = pdf.between_time("0:45", "0:15")
    actual = gdf.between_time("0:45", "0:15")
    assert_eq(actual, expected)


def test_between_time_fractional_seconds():
    dates = cudf.date_range("2021-01-01 00:00:00", periods=5, freq="500ms")
    gdf = cudf.DataFrame({"A": range(5)}, index=dates)
    pdf = gdf.to_pandas()

    expected = pdf.between_time("00:00:00.250", "00:00:01.750")
    actual = gdf.between_time("00:00:00.250", "00:00:01.750")
    assert_eq(actual, expected)


def test_between_time_invalid_index():
    gdf = cudf.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(TypeError):
        gdf.between_time("0:15", "0:45")


def test_between_time_invalid_inclusive(gdf_pdf):
    gdf, _ = gdf_pdf
    with pytest.raises(ValueError):
        gdf.between_time("0:15", "0:45", inclusive="oops")


def test_at_time_basic():
    dates = cudf.date_range("2018-04-09", periods=4, freq="12h")
    gdf = cudf.DataFrame({"A": [1, 2, 3, 4]}, index=dates)
    pdf = gdf.to_pandas()

    expected = pdf.at_time("12:00")
    actual = gdf.at_time("12:00")
    assert_eq(actual, expected)


def test_at_time_fractional_seconds():
    dates = cudf.date_range("2021-01-01 00:00:00", periods=5, freq="500ms")
    gdf = cudf.DataFrame({"A": range(5)}, index=dates)
    pdf = gdf.to_pandas()

    expected = pdf.at_time("00:00:01.500")
    actual = gdf.at_time("00:00:01.500")
    assert_eq(actual, expected)


def test_at_time_invalid_index():
    gdf = cudf.DataFrame({"A": [1, 2, 3]})
    with pytest.raises(TypeError):
        gdf.at_time("12:00")


def test_at_time_invalid_axis():
    dates = cudf.date_range("2018-04-09", periods=4, freq="12h")
    gdf = cudf.DataFrame({"A": [1, 2, 3, 4]}, index=dates)
    with pytest.raises(NotImplementedError):
        gdf.at_time("12:00", axis=1)

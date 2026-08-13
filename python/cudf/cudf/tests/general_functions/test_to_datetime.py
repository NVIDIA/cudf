# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)
from cudf.utils.temporal import infer_format


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        pd.Series([], dtype="float64"),
        pd.Index([]),
        pd.Series([1, 2, 3]),
        pd.Series([0, 1, -1]),
        pd.Series([0, 1, -1, 100.3, 200, 47637289]),
        pd.Series(["2012-10-11", "2010-01-01", "2016-07-07", "2014-02-02"]),
        [1, 2, 3, 100, -123, -1, 0, 1000000000000679367],
        pd.DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]}),
        pd.DataFrame(
            {"year": ["2015", "2016"], "month": ["2", "3"], "day": [4, 5]}
        ),
        pd.DataFrame(
            {
                "year": [2015, 2016],
                "month": [2, 3],
                "day": [4, 5],
                "minute": [1, 100],
                "second": [90, 10],
                "hour": [1, 0.5],
            },
            index=["a", "b"],
        ),
        pd.DataFrame(
            {
                "year": [],
                "month": [],
                "day": [],
                "minute": [],
                "second": [],
                "hour": [],
            },
        ),
        ["2012-10-11", "2010-01-01", "2016-07-07", "2014-02-02"],
        pd.Index([1, 2, 3, 4]),
        pd.DatetimeIndex(
            ["1970-01-01 00:00:00.000000001", "1970-01-01 00:00:00.000000002"],
            dtype="datetime64[ns]",
            freq=None,
        ),
        pd.DatetimeIndex(
            [],
            dtype="datetime64[ns]",
            freq=None,
        ),
        pd.Series([1, 2, 3]).astype("datetime64[ns]"),
        pd.Series([1, 2, 3]).astype("datetime64[us]"),
        pd.Series([1, 2, 3]).astype("datetime64[ms]"),
        pd.Series([1, 2, 3]).astype("datetime64[s]"),
        pd.Series([1, 2, 3]).astype("datetime64[D]"),
        1,
        100,
        17,
        53.638435454,
        np.array([1, 10, 15, 478925, 2327623467]),
        np.array([0.3474673, -10, 15, 478925.34345, 2327623467]),
    ],
)
@pytest.mark.parametrize("dayfirst", [True, False])
def test_cudf_to_datetime(data, dayfirst):
    pd_data = data
    if isinstance(pd_data, (pd.Series, pd.DataFrame, pd.Index)):
        gd_data = cudf.from_pandas(pd_data)
    else:
        if type(pd_data).__module__ == np.__name__:
            gd_data = cp.array(pd_data)
        else:
            gd_data = pd_data

    expected = pd.to_datetime(pd_data, dayfirst=dayfirst)
    actual = cudf.to_datetime(gd_data, dayfirst=dayfirst)
    if isinstance(expected, pd.Series):
        assert_eq(actual, expected, check_dtype=False)
    else:
        if expected is pd.NaT:
            assert actual is expected
        else:
            assert_eq(actual, expected, check_exact=False)


@pytest.mark.parametrize(
    "data",
    [
        "2",
        ["1", "2", "3"],
        ["1/1/1", "2/2/2", "1"],
        pd.Series([1, 2, 3], dtype="timedelta64[ns]"),
        pd.DataFrame(
            {
                "year": [2015, 2016],
                "month": [2, 3],
                "day": [4, 5],
                "minute": [1, 100],
                "second": [90, 10],
                "hour": [1, 0],
                "blablacol": [1, 1],
            }
        ),
        pd.DataFrame(
            {
                "month": [2, 3],
                "day": [4, 5],
                "minute": [1, 100],
                "second": [90, 10],
                "hour": [1, 0],
            }
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:Could not infer format:UserWarning")
def test_to_datetime_errors(data):
    pd_data = data
    if isinstance(pd_data, (pd.Series, pd.DataFrame, pd.Index)):
        gd_data = cudf.from_pandas(pd_data)
    else:
        gd_data = pd_data

    assert_exceptions_equal(
        pd.to_datetime,
        cudf.to_datetime,
        ([pd_data],),
        ([gd_data],),
    )


def test_to_datetime_not_implemented():
    with pytest.raises(NotImplementedError):
        cudf.to_datetime([], exact=False)

    with pytest.raises(NotImplementedError):
        cudf.to_datetime([], origin="julian")

    with pytest.raises(NotImplementedError):
        cudf.to_datetime([], yearfirst=True)


@pytest.mark.parametrize(
    "data",
    [
        1,
        [],
        pd.Series([], dtype="float64"),
        pd.Index([]),
        pd.Series([1, 2, 3]),
        pd.Series([1, 2.4, 3]),
        pd.Series([0, 1, -1]),
        pd.Series([0, 1, -1, 100, 200, 47637]),
        [10, 12, 1200, 15003],
        pd.DatetimeIndex(
            [],
            dtype="datetime64[ns]",
            freq=None,
        ),
        pd.Index([1, 2, 3, 4]),
    ],
)
@pytest.mark.parametrize("unit", ["D", "s", "ms", "us", "ns"])
def test_to_datetime_units(data, unit):
    pd_data = data
    if isinstance(pd_data, (pd.Series, pd.DataFrame, pd.Index)):
        gd_data = cudf.from_pandas(pd_data)
    else:
        gd_data = pd_data

    expected = pd.to_datetime(pd_data, unit=unit)
    actual = cudf.to_datetime(gd_data, unit=unit)

    if isinstance(expected, pd.Series):
        assert_eq(actual, expected, check_dtype=False)
    else:
        assert_eq(actual, expected, exact=False, check_exact=False)


@pytest.mark.parametrize(
    "data,format",
    [
        ("2012-10-11", None),
        ("2012-10-11", "%Y-%m-%d"),
        ("2012-10-11", "%Y-%d-%m"),
        (["2012-10-11", "2010-01-01", "2016-07-07", "2014-02-02"], None),
        (["2012-10-11", "2010-01-01", "2016-07-07", "2014-02-02"], "%Y-%m-%d"),
        (["2012-10-11", "2010-01-01", "2016-07-07", "2014-02-02"], "%Y-%d-%m"),
        (["10-11-2012", "01-01-2010", "07-07-2016", "02-02-2014"], "%m-%d-%Y"),
        (["10-11-2012", "01-01-2010", "07-07-2016", "02-02-2014"], "%d-%m-%Y"),
        (["10-11-2012", "01-01-2010", "07-07-2016", "02-02-2014"], None),
        (["2012/10/11", "2010/01/01", "2016/07/07", "2014/02/02"], None),
        (["2012/10/11", "2010/01/01", "2016/07/07", "2014/02/02"], "%Y/%m/%d"),
        (["2012/10/11", "2010/01/01", "2016/07/07", "2014/02/02"], "%Y/%d/%m"),
        (["10/11/2012", "01/01/2010", "07/07/2016", "02/02/2014"], "%m/%d/%Y"),
        (["10/11/2012", "01/01/2010", "07/07/2016", "02/02/2014"], "%d/%m/%Y"),
        (["10/11/2012", "01/01/2010", "07/07/2016", "02/02/2014"], None),
        (
            ["2021-04-13 12:30:04.123456789"],
            "%Y-%m-%d %H:%M:%S.%f",
        ),
        (pd.Series([2015, 2020, 2021]), "%Y"),
        pytest.param(
            pd.Series(["1", "2", "1"]),
            "%m",
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/6109"
                "https://github.com/pandas-dev/pandas/issues/35934"
            ),
        ),
        pytest.param(
            pd.Series(["14", "20", "10"]),
            "%d",
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/6109"
                "https://github.com/pandas-dev/pandas/issues/35934"
            ),
        ),
        (pd.Series([2015, 2020.0, 2021.2]), "%Y"),
    ],
)
def test_to_datetime_format(data, format):
    pd_data = data
    if isinstance(pd_data, (pd.Series, pd.DataFrame, pd.Index)):
        gd_data = cudf.from_pandas(pd_data)
    else:
        gd_data = pd_data

    expected = pd.to_datetime(pd_data, format=format)
    actual = cudf.to_datetime(gd_data, format=format)

    if isinstance(expected, pd.Series):
        assert_eq(actual, expected, check_dtype=False)
    else:
        assert_eq(actual, expected, check_exact=False)


def test_to_datetime_data_out_of_range_for_format():
    with pytest.raises(ValueError):
        cudf.to_datetime("2015-02-99", format="%Y-%m-%d")


def test_to_datetime_different_formats_notimplemented():
    with pytest.raises(NotImplementedError):
        cudf.to_datetime(["2015-02-01", "2015-02-01 10:10:10"])


def test_datetime_to_datetime_error():
    assert_exceptions_equal(
        lfunc=pd.to_datetime,
        rfunc=cudf.to_datetime,
        lfunc_args_and_kwargs=(["02-Oct-2017 09:30", "%d-%B-%Y %H:%M"],),
        rfunc_args_and_kwargs=(["02-Oct-2017 09:30", "%d-%B-%Y %H:%M"],),
        check_exception_type=False,
    )


@pytest.mark.parametrize("code", ["z", "Z"])
def test_format_timezone_not_implemented(code):
    with pytest.raises(NotImplementedError):
        cudf.to_datetime(
            ["2020-01-01 00:00:00 UTC"], format=f"%Y-%m-%d %H:%M:%S %{code}"
        )


@pytest.mark.parametrize("tz", ["UTC-3", "+01:00"])
def test_utc_offset_not_implemented(tz):
    with pytest.raises((NotImplementedError, ValueError)):
        cudf.to_datetime([f"2020-01-01 00:00:00{tz}"])


def test_Z_utc_offset():
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.to_datetime(["2020-01-01 00:00:00Z"])

    result = cudf.to_datetime(["2020-01-01 00:00:00Z"])
    expected = cudf.to_datetime(["2020-01-01 00:00:00"])
    assert_eq(result, expected)


@pytest.mark.parametrize("arg", [True, False])
def test_args_not_datetime_typerror(arg):
    with pytest.raises(TypeError):
        cudf.to_datetime([arg])


@pytest.mark.parametrize("errors", ["coerce", "ignore"])
def test_to_datetime_errors_non_scalar_not_implemented(errors):
    with pytest.raises(NotImplementedError):
        cudf.to_datetime([1, ""], unit="s", errors=errors)


def test_to_datetime_errors_ignore_deprecated():
    with pytest.warns(FutureWarning):
        cudf.to_datetime("2001-01-01 00:04:45", errors="ignore")


@pytest.mark.parametrize(
    "data",
    [
        # all-None object inputs land on [s] in pandas 3 — see
        # pandas-dev/pandas#55901 (NPY_FR_GENERIC fallback to "s").
        pd.Series([None, None, None]),
        pd.Series([None] * 5),
        pd.Series([], dtype="object"),
    ],
)
def test_to_datetime_all_null_object_returns_seconds(data):
    expected = pd.to_datetime(data)
    actual = cudf.to_datetime(cudf.from_pandas(data))
    assert actual.dtype == expected.dtype
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "scalar",
    ["1/1/2000", "2020-01-01", "2020-01-01 12:34:56"],
)
def test_to_datetime_scalar_string_returns_us(scalar):
    # Scalar string parsing should land on [us] (pandas 3 default).
    expected = pd.to_datetime(scalar)
    actual = cudf.to_datetime(scalar)
    assert actual.unit == expected.unit
    assert actual == expected


@pytest.mark.parametrize(
    "values",
    [
        [19801222, 20010112, None],
        [19801222, 20010112, np.nan],
        [19801222, 20010112],
    ],
)
def test_to_datetime_int_with_format_us(values):
    # Float-with-format path (triggered when a None/nan widens int -> float)
    # must land on [us] regardless of whether the format contains "%f".
    expected = pd.to_datetime(values, format="%Y%m%d")
    actual = cudf.to_datetime(values, format="%Y%m%d")
    assert actual.dtype == expected.dtype
    assert_eq(actual, expected, check_exact=False)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame({"year": [2015, 2016], "month": [2, 3], "day": [4, 5]}),
        pd.DataFrame(
            {
                "year": [2015, 2016],
                "month": [2, 3],
                "day": [4, 5],
            }
        ).astype("int16"),
        pd.DataFrame(
            {
                "year": [2015, 2016],
                "month": [2, 3],
                "day": [4, 5],
                "hour": [6, 7],
                "minute": [58, 59],
                "second": [10, 11],
            }
        ),
    ],
)
def test_to_datetime_dataframe_default_us(df):
    # DataFrame -> datetime defaults to [us] in pandas 3 (was [s] in cuDF).
    expected = pd.to_datetime(df)
    actual = cudf.to_datetime(cudf.from_pandas(df))
    assert actual.dtype == expected.dtype
    assert_eq(actual, expected)


def test_to_datetime_dataframe_with_ns_field_widens_to_ns():
    # When a ns field is explicitly present, the result must widen to [ns]
    # (and the integer factor arithmetic must not lose the trailing ns).
    df = pd.DataFrame(
        {
            "year": [2015, 2016],
            "month": [2, 3],
            "day": [4, 5],
            "hour": [6, 7],
            "minute": [58, 59],
            "second": [10, 11],
            "ms": [1, 1],
            "us": [2, 2],
            "ns": [3, 3],
        }
    )
    expected = pd.to_datetime(df)
    actual = cudf.to_datetime(cudf.from_pandas(df))
    assert actual.dtype == expected.dtype
    assert_eq(actual, expected)


def test_to_datetime_nanosecond_precision_strings():
    # pandas infers nanosecond precision from strings with more than 6
    # fractional-second digits.
    data = ["2015-01-03T00:00:00.000000000", "2015-01-01T00:00:00.000000000"]
    assert_eq(pd.to_datetime(data), cudf.to_datetime(data))
    data_us = ["2015-01-03T00:00:00.000000", "2015-01-01T00:00:00.000000"]
    assert_eq(pd.to_datetime(data_us), cudf.to_datetime(data_us))


@pytest.mark.parametrize(
    "data",
    [
        ["2016-01-01 00:00:00"],
        [1451606400, 1451610000],
    ],
)
def test_datetime_index_tz_dtype_wall_time(data):
    # Timezone-naive string data with a tz-aware dtype is interpreted as
    # wall time in the target timezone, while integers are interpreted as
    # epoch time (UTC), matching pandas.
    result = cudf.DatetimeIndex(
        data, dtype=pd.DatetimeTZDtype("s", "US/Eastern")
    )
    expected = pd.DatetimeIndex(data, dtype="datetime64[s, US/Eastern]")
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "bad_input",
    ["1.2", "9.9", "2078~1.15"],
)
def test_infer_format_rejects_non_datetime_dotted_strings(bad_input):
    with pytest.raises(
        ValueError, match="Unable to infer the timestamp format"
    ):
        infer_format(bad_input)


def test_to_datetime_rejects_non_datetime_dotted_string():
    with pytest.raises(
        ValueError, match="Unable to infer the timestamp format"
    ):
        cudf.to_datetime(cudf.Series(["1.2"]))


def test_to_datetime_out_of_bounds_nanosecond_precision_string():
    # More than 6 fractional-second digits imply nanosecond precision,
    # and year 2263 exceeds the nanosecond Timestamp range.
    data = np.array(["2263-01-01 00:00:00.123456789"], dtype=object)
    assert_exceptions_equal(
        lfunc=pd.to_datetime,
        rfunc=cudf.to_datetime,
        lfunc_args_and_kwargs=([data],),
        rfunc_args_and_kwargs=([data],),
    )


def test_to_datetime_format_f_out_of_ns_bounds_keeps_us():
    # With an explicit %f format, values beyond the nanosecond range
    # keep microsecond precision like pandas' unit inference.
    data = ["2263-01-01 00:00:00.123456"]
    fmt = "%Y-%m-%d %H:%M:%S.%f"
    expected = pd.to_datetime(data, format=fmt)
    result = cudf.to_datetime(data, format=fmt)
    assert_eq(result, expected)

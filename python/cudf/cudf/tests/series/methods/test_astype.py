# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import datetime
import zoneinfo
from decimal import Decimal

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core.column.decimal import DecimalColumn
from cudf.core.column.numerical import NumericalColumn
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        [
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]],
            [[1, 2, 3], [4, 5, 6]],
        ],
        [[[1, 2, 3], [4, None, 6]], [], None, [[7, 8], [], None, [9]]],
        [[1, 2, 3], [4, None, 6], [7, 8], [], None, [9]],
        [[1.0, 2.0, 3.0], [4.0, None, 6.0], [7.0, 8.0], [], None, [9.0]],
    ],
)
def test_listcol_as_string(data):
    got = cudf.Series(data).astype("str")
    expect = pd.Series(data).astype("str")
    assert_eq(expect, got)


def test_list_astype():
    s = cudf.Series([[1, 2], [3, 4]])
    s2 = s.list.astype("float64")
    assert s2.dtype == cudf.ListDtype("float64")
    assert_eq(s.list.leaves.astype("float64"), s2.list.leaves)

    s = cudf.Series([[[1, 2], [3]], [[5, 6], None]])
    s2 = s.list.astype("string")
    assert s2.dtype == cudf.ListDtype(cudf.ListDtype("string"))
    assert_eq(s.list.leaves.astype("string"), s2.list.leaves)


def test_series_typecast_to_object_error():
    actual = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(TypeError):
            actual.astype(object)
        with pytest.raises(TypeError):
            actual.astype(np.dtype("object"))
        new_series = actual.astype("str")
        assert new_series[0] == "1970-01-01 00:00:00.000000001"


def test_series_typecast_to_object():
    actual = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    with cudf.option_context("mode.pandas_compatible", False):
        new_series = actual.astype(object)
        assert new_series[0] == "1970-01-01 00:00:00.000000001"
        new_series = actual.astype(np.dtype("object"))
        assert new_series[0] == "1970-01-01 00:00:00.000000001"


@pytest.mark.parametrize(
    "dtype",
    [
        pd.StringDtype(storage="python", na_value=pd.NA),
        pd.StringDtype(storage="pyarrow", na_value=pd.NA),
        pd.ArrowDtype(pa.string()),
    ],
)
def test_string_astype_object_pd_na_pandas_compat(dtype):
    sr = cudf.Series(["a", None, "b"], dtype=dtype)

    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(
            NotImplementedError,
            match=r"Casting nullable string columns with pd.NA to object",
        ):
            sr.astype(object)

    with cudf.option_context("mode.pandas_compatible", False):
        result = sr.astype(object)
    assert result.dtype == np.dtype("object")


@pytest.mark.parametrize(
    "dtype",
    [
        pd.StringDtype(storage="python", na_value=pd.NA),
        pd.StringDtype(storage="pyarrow", na_value=pd.NA),
        pd.ArrowDtype(pa.string()),
    ],
)
def test_string_astype_object_pd_na_pandas_compat_no_nulls(dtype):
    sr = cudf.Series(["a", "b", "c"], dtype=dtype)

    with cudf.option_context("mode.pandas_compatible", True):
        result = sr.astype(object)
    assert result.dtype == np.dtype("object")
    assert result.to_arrow().to_pylist() == ["a", "b", "c"]


@pytest.mark.parametrize(
    "dtype",
    [
        pd.ArrowDtype(pa.int8()),
        pd.ArrowDtype(pa.int16()),
        pd.ArrowDtype(pa.int32()),
        pd.ArrowDtype(pa.int64()),
        pd.ArrowDtype(pa.uint8()),
        pd.ArrowDtype(pa.uint16()),
        pd.ArrowDtype(pa.uint32()),
        pd.ArrowDtype(pa.uint64()),
        pd.ArrowDtype(pa.float32()),
        pd.ArrowDtype(pa.float64()),
        pd.Int8Dtype(),
        pd.Int16Dtype(),
        pd.Int32Dtype(),
        pd.Int64Dtype(),
        pd.UInt8Dtype(),
        pd.UInt16Dtype(),
        pd.UInt32Dtype(),
        pd.UInt64Dtype(),
        pd.Float32Dtype(),
        pd.Float64Dtype(),
    ],
)
@pytest.mark.parametrize("klass", [cudf.Series, cudf.DataFrame, cudf.Index])
@pytest.mark.parametrize("kind", [lambda x: x, str], ids=["obj", "string"])
def test_astype_pandas_nullable_pandas_compat(dtype, klass, kind):
    ser = klass([1, 2, 3])
    with cudf.option_context("mode.pandas_compatible", True):
        actual = ser.astype(kind(dtype))
        expected = klass([1, 2, 3], dtype=kind(dtype))
        assert_eq(actual, expected)


def test_cast_float_nan_to_bool_pandas_compat():
    with cudf.option_context("mode.pandas_compatible", True):
        data = [1.0, 0.0, np.nan, None]
        gs = cudf.Series(data, dtype="float64")
        got = gs.astype("bool")
        expected = pd.Series([True, False, True, True], dtype="bool")
        assert got.null_count == 0
        assert_eq(expected, got)


@pytest.mark.parametrize(
    "type1",
    [
        "category",
        "interval[int64, right]",
        "int64",
        "float64",
        "str",
        "datetime64[ns]",
        "timedelta64[ns]",
    ],
)
@pytest.mark.parametrize(
    "type2",
    [
        "category",
        "interval[int64, right]",
        "int64",
        "float64",
        "str",
        "datetime64[ns]",
        "timedelta64[ns]",
    ],
)
@pytest.mark.parametrize(
    "as_dtype", [lambda x: x, cudf.dtype], ids=["string", "object"]
)
@pytest.mark.parametrize("copy", [True, False])
def test_empty_astype_always_castable(type1, type2, as_dtype, copy):
    ser = cudf.Series([], dtype=as_dtype(type1))
    result = ser.astype(as_dtype(type2), copy=copy)
    if type2 == "category":
        # Empty astype to category inherits the source dtype as the
        # categories dtype, matching pandas behavior.
        expected = cudf.Series([], dtype=result.dtype)
    else:
        expected = cudf.Series([], dtype=as_dtype(type2))
    assert_eq(result, expected)
    if not copy and type1 == type2 == "category":
        # Categorical shallow copy reuses the same column object.
        assert ser._column is result._column
    else:
        # Under CoW, copy=False returns a shallow copy (distinct column
        # object sharing the same buffer), so column identity never holds.
        assert ser._column is not result._column


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [12, 12, 22, 343, 4353534, 435342],
        [0.3534, 12, 22, 343, 43.53534, 4353.42],
        cp.asarray([10, 20, 30, 100]),
    ],
)
@pytest.mark.parametrize("cast_dtype", ["int64", "category"])
def test_timedelta_from_typecast(data, timedelta_types_as_str, cast_dtype):
    if timedelta_types_as_str != "timedelta64[ns]":
        pytest.skip(
            "Bug in pandas : https://github.com/pandas-dev/pandas/issues/35465"
        )
    psr = pd.Series(
        cp.asnumpy(data) if isinstance(data, cp.ndarray) else data,
        dtype=timedelta_types_as_str,
    )
    gsr = cudf.Series(data, dtype=timedelta_types_as_str)

    if cast_dtype == "int64":
        assert_eq(psr.values.view(cast_dtype), gsr.astype(cast_dtype).values)
    else:
        assert_eq(psr.astype(cast_dtype), gsr.astype(cast_dtype))


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [12, 12, 22, 343, 4353534, 435342],
        [0.3534, 12, 22, 343, 43.53534, 4353.42],
        cp.asarray([10, 20, 30, 100]),
    ],
)
def test_timedelta_to_typecast(data, timedelta_types_as_str):
    psr = pd.Series(cp.asnumpy(data) if isinstance(data, cp.ndarray) else data)
    gsr = cudf.Series(data)

    assert_eq(
        psr.astype(timedelta_types_as_str), gsr.astype(timedelta_types_as_str)
    )


@pytest.mark.parametrize("data", [[], [1, 2, 3, 4, 5]])
def test_numeric_to_timedelta(
    data, numeric_types_as_str, timedelta_types_as_str
):
    sr = cudf.Series(data, dtype=numeric_types_as_str)
    psr = sr.to_pandas()

    actual = sr.astype(timedelta_types_as_str)
    expected = psr.astype(timedelta_types_as_str)

    assert_eq(expected, actual)


def test_astype_with_aliases(all_supported_pandas_nullable_extension_dtypes):
    scalar, dtype = all_supported_pandas_nullable_extension_dtypes
    expected = pd.Series([scalar]).astype(str(dtype))
    result = cudf.Series([scalar]).astype(str(dtype))

    assert_eq(expected, result)


def test_timedelta_datetime_cast_invalid():
    sr = cudf.Series([1, 2, 3], dtype="timedelta64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        psr.astype,
        sr.astype,
        (["datetime64[ns]"],),
        (["datetime64[ns]"],),
    )

    sr = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        psr.astype,
        sr.astype,
        (["timedelta64[ns]"],),
        (["timedelta64[ns]"],),
    )


@pytest.mark.parametrize(
    "sr_data, sr_dtype",
    [
        ([1, 2, 3], "timedelta64[ns]"),
        ([1000000, 200000, 3000000], "timedelta64[ms]"),
        ([1000000, 200000, 3000000], "timedelta64[s]"),
        ([None, None, None, None, None], "timedelta64[us]"),
        (
            [136457654, None, 245345345, 223432411, None, 3634548734, 23234],
            "timedelta64[us]",
        ),
        (
            [136457654, None, 245345345, 223432411, None, 3634548734, 23234],
            "timedelta64[ms]",
        ),
        (
            [136457654, None, 245345345, 223432411, None, 3634548734, 23234],
            "timedelta64[s]",
        ),
        (
            [136457654, None, 245345345, 223432411, None, 3634548734, 23234],
            "timedelta64[ns]",
        ),
    ],
)
def test_timedelta_str_roundtrip(sr_data, sr_dtype):
    gsr = cudf.Series(sr_data, dtype=sr_dtype)
    psr = gsr.to_pandas()

    actual_series = gsr.astype("str")
    expected_series = psr.astype("str")

    assert_eq(expected_series, actual_series)

    assert_eq(gsr, actual_series.astype(gsr.dtype))


def test_timedelta_astype_object_raises_pandas_compat():
    sr = cudf.Series([1, 2, 3], dtype="timedelta64[ns]")
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(TypeError):
            sr.astype(object)
        with pytest.raises(TypeError):
            sr.astype(np.dtype("object"))


@pytest.mark.parametrize(
    "dtype", ["datetime64[ns]", "datetime64[us]", "timedelta64[ns]"]
)
@pytest.mark.parametrize(
    "target_dtype", ["float32", "float64", np.float64, np.float32]
)
def test_temporal_astype_float_raises_pandas_compat(dtype, target_dtype):
    # pandas raises TypeError when casting datetime/timedelta to float;
    # cuDF must match this under pandas-compatible mode.
    from cudf.testing._utils import assert_exceptions_equal

    psr = pd.Series([1, 2, 3], dtype=dtype)
    gsr = cudf.from_pandas(psr)
    with cudf.option_context("mode.pandas_compatible", True):
        assert_exceptions_equal(
            lfunc=psr.astype,
            rfunc=gsr.astype,
            lfunc_args_and_kwargs=([target_dtype], {}),
            rfunc_args_and_kwargs=([target_dtype], {}),
        )


def test_timedelta_astype_unicode_dtype_pandas_compat():
    sr = cudf.Series([1000000000], dtype="timedelta64[ns]")
    with cudf.option_context("mode.pandas_compatible", True):
        result = sr.astype(np.dtype("U"))
    expected = pd.Series(pd.to_timedelta([1000000000], unit="ns")).astype(
        np.dtype("U")
    )
    assert result.dtype == np.dtype("object")
    assert_eq(result, cudf.Series(expected))


@pytest.mark.parametrize(
    "data, min_unit",
    [
        # days-only
        (["1 days", "5 days", "0 days"], "s"),
        # whole seconds (sub-day, no fractions)
        (["1 days 02:03:04", "0 days 00:00:01"], "s"),
        # millisecond fractions
        (["0 days 00:00:01.500", "1 days 00:00:00.001"], "ms"),
        # microsecond fractions
        (["0 days 00:00:00.000001", "0 days 00:00:01.123456"], "us"),
        # nanosecond fractions
        (["0 days 00:00:00.000000001", "0 days 00:00:00.123456789"], "ns"),
        # nanoseconds that are exact microseconds (trailing zeros stripped)
        (["0 days 00:00:00.000001000", "0 days 00:00:00.100000000"], "us"),
        # mixed fractional and whole seconds
        (["0 days 00:00:01.500000", "0 days 00:00:02"], "us"),
        # mixed days-only and sub-day rows
        (["3 days", "0 days 01:00:00"], "s"),
    ],
    ids=[
        "days_only",
        "whole_seconds",
        "millis",
        "micros",
        "nanos",
        "nanos_exact_micros",
        "mixed_frac_and_whole",
        "mixed_days_and_subday",
    ],
)
@pytest.mark.parametrize(
    "unit",
    [
        "timedelta64[ns]",
        "timedelta64[us]",
        "timedelta64[ms]",
        "timedelta64[s]",
    ],
)
def test_timedelta_astype_str_pandas_compat(data, min_unit, unit):
    ptd = pd.to_timedelta(data).astype(unit)
    psr = pd.Series(ptd)
    gsr = cudf.Series(ptd)
    expected = psr.astype("str")
    with cudf.option_context("mode.pandas_compatible", True):
        result = gsr.astype("str")
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "unit",
    [
        "timedelta64[ns]",
        "timedelta64[us]",
        "timedelta64[ms]",
        "timedelta64[s]",
    ],
)
def test_timedelta_astype_str_with_nulls_pandas_compat(unit):
    ptd = pd.to_timedelta(
        ["1 days 02:03:04", pd.NaT, "0 days 00:00:01"]
    ).astype(unit)
    psr = pd.Series(ptd)
    gsr = cudf.Series(ptd)
    with cudf.option_context("mode.pandas_compatible", True):
        result = gsr.astype("str")
    expected = psr.astype("str")
    assert_eq(result, cudf.Series(expected))


def test_typecast_from_datetime(numeric_types_as_str):
    data = pd.date_range(
        "2019-07-16 00:00:00",
        "2019-07-16 00:00:01",
        freq="5555us",
        name="times",
    )
    pd_data = pd.Series(data)
    np_data = np.array(pd_data)
    gdf_data = cudf.Series(pd_data)

    np_casted = np_data.astype(numeric_types_as_str)
    gdf_casted = gdf_data.astype(numeric_types_as_str)

    np.testing.assert_equal(np_casted, gdf_casted.to_numpy())


def test_typecast_from_datetime_to_int64_to_datetime(datetime_types_as_str):
    data = pd.date_range(
        "2019-07-16 00:00:00",
        "2019-07-16 00:00:01",
        freq="5555us",
        name="times",
    )
    pd_data = pd.Series(data)
    np_data = np.array(pd_data)
    gdf_data = cudf.Series(pd_data)

    np_casted = np_data.astype(np.int64).astype(datetime_types_as_str)
    gdf_casted = gdf_data.astype(np.int64).astype(datetime_types_as_str)

    np.testing.assert_equal(np_casted, gdf_casted.to_numpy())


def test_typecast_to_different_datetime_resolutions(datetime_types_as_str):
    data = pd.date_range(
        "2019-07-16 00:00:00",
        "2019-07-16 00:00:01",
        freq="5555us",
        name="times",
    )
    pd_data = pd.Series(data)
    np_data = np.array(pd_data).astype(datetime_types_as_str)
    gdf_series = cudf.Series(pd_data).astype(datetime_types_as_str)
    np.testing.assert_equal(np_data, gdf_series.to_numpy())


@pytest.mark.parametrize(
    "data",
    [
        np.array(["2263-01-01"], dtype="datetime64[s]"),
        np.array(["1600-01-01"], dtype="datetime64[s]"),
        np.array(["2263-01-01", "NaT"], dtype="datetime64[s]"),
    ],
    ids=["above-range", "below-range", "with-null"],
)
def test_typecast_datetime_narrowing_out_of_bounds(data):
    # Casting to a finer resolution whose int64 range cannot hold the
    # values must raise like pandas instead of wrapping around.
    psr = pd.Series(data)
    gsr = cudf.Series(data)
    assert_exceptions_equal(
        lfunc=psr.astype,
        rfunc=gsr.astype,
        lfunc_args_and_kwargs=(["datetime64[ns]"],),
        rfunc_args_and_kwargs=(["datetime64[ns]"],),
    )


def test_typecast_datetime_narrowing_in_bounds():
    # The bounds check is relative to the target unit: year 9999 fits in
    # seconds and microseconds, just not nanoseconds.
    data = [datetime.date(9999, 12, 31)]
    psr = pd.Series(data, dtype="datetime64[s]")
    gsr = cudf.Series(data, dtype="datetime64[s]")
    assert_eq(psr.astype("datetime64[us]"), gsr.astype("datetime64[us]"))

    # All-null columns can always be narrowed.
    all_null = np.array(["NaT", "NaT"], dtype="datetime64[s]")
    assert_eq(
        pd.Series(all_null).astype("datetime64[ns]"),
        cudf.Series(all_null).astype("datetime64[ns]"),
    )


def test_typecast_datetime_tz_narrowing_out_of_bounds():
    data = np.array(["2263-01-01"], dtype="datetime64[s]")
    psr = pd.Series(data).dt.tz_localize("UTC")
    gsr = cudf.Series(data).dt.tz_localize("UTC")
    target = pd.DatetimeTZDtype("ns", "UTC")
    assert_exceptions_equal(
        lfunc=psr.astype,
        rfunc=gsr.astype,
        lfunc_args_and_kwargs=([target],),
        rfunc_args_and_kwargs=([target],),
    )


def test_string_astype_datetime_tz():
    target = pd.DatetimeTZDtype("ns", "UTC")

    data = ["2000-01-01"]
    assert_eq(pd.Series(data).astype(target), cudf.Series(data).astype(target))

    out_of_bounds = ["2263-01-01"]
    assert_exceptions_equal(
        lfunc=pd.Series(out_of_bounds).astype,
        rfunc=cudf.Series(out_of_bounds).astype,
        lfunc_args_and_kwargs=([target],),
        rfunc_args_and_kwargs=([target],),
    )


@pytest.mark.parametrize(
    "data",
    [
        [
            "2019-07-16 00:00:00.333",
            "2019-07-16 00:00:00.666",
            "2019-07-16 00:00:00.888",
        ],
        [
            "2019-07-16 00:00:00.333333",
            "2019-07-16 00:00:00.666666",
            "2019-07-16 00:00:00.888888",
        ],
        [
            "2019-07-16 00:00:00.333333333",
            "2019-07-16 00:00:00.666666666",
            "2019-07-16 00:00:00.888888888",
        ],
    ],
    ids=["ms_data", "us_data", "ns_data"],
)
def test_string_timstamp_typecast_to_different_datetime_resolutions(
    data, datetime_types_as_str
):
    pd_sr = pd.Series(data)
    gdf_sr = cudf.Series(pd_sr)

    expect = np.array(data).astype(datetime_types_as_str)
    got = gdf_sr.astype(datetime_types_as_str).to_numpy()

    np.testing.assert_equal(expect, got)


def test_typecast_to_datetime(numeric_types_as_str, datetime_types_as_str):
    data = np.arange(1, 10)
    np_data = data.astype(numeric_types_as_str)
    gdf_data = cudf.Series(np_data)

    np_casted = np_data.astype(datetime_types_as_str)
    gdf_casted = gdf_data.astype(datetime_types_as_str)

    np.testing.assert_equal(np_casted, gdf_casted.to_numpy())


def test_typecast_to_from_datetime(
    numeric_types_as_str, datetime_types_as_str
):
    data = np.arange(1, 10)
    np_data = data.astype(numeric_types_as_str)
    gdf_data = cudf.Series(np_data)

    np_casted = np_data.astype(datetime_types_as_str).astype(
        numeric_types_as_str
    )
    gdf_casted = gdf_data.astype(datetime_types_as_str).astype(
        numeric_types_as_str
    )

    np.testing.assert_equal(np_casted, gdf_casted.to_numpy())


@pytest.fixture
def datetime_types_as_str2(datetime_types_as_str):
    return datetime_types_as_str


def test_typecast_from_datetime_to_datetime(
    datetime_types_as_str, datetime_types_as_str2
):
    data = np.arange(1, 10)
    np_data = data.astype(datetime_types_as_str)
    ser = cudf.Series(np_data)

    np_casted = np_data.astype(datetime_types_as_str2)
    ser_casted = ser.astype(datetime_types_as_str2)

    np.testing.assert_equal(np_casted, ser_casted.to_numpy())


@pytest.mark.parametrize(
    "data",
    [
        ["2001-01-01", "2002-02-02", "2000-01-05", "NaT"],
        ["2001-01-01", "2002-02-02", "2000-01-05", None],
        [None, None, None, None, None],
    ],
)
def test_str_null_to_datetime(data, datetime_types_as_str):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    assert_eq(
        psr.astype(datetime_types_as_str), gsr.astype(datetime_types_as_str)
    )


def test_str_to_datetime_error():
    psr = pd.Series(["2001-01-01", "2002-02-02", "2000-01-05", "None"])
    gsr = cudf.Series(["2001-01-01", "2002-02-02", "2000-01-05", "None"])

    assert_exceptions_equal(
        lfunc=psr.astype,
        rfunc=gsr.astype,
        lfunc_args_and_kwargs=(["datetime64[s]"],),
        rfunc_args_and_kwargs=(["datetime64[s]"],),
        check_exception_type=False,
    )


@pytest.mark.parametrize("timezone", ["", "Z"])
@pytest.mark.parametrize(
    "data",
    [
        "2002-10-27T04:30",
        "2002-10-27T04:30:00",
        "2002-10-27T04:30:00.000",
        "2002-10-27T04:30:00.000000",
        "2002-10-27T04:30:00.000000000",
    ],
)
def test_datetime_infer_format(data, timezone, datetime_types_as_str):
    ts_data = [data + timezone]
    sr = cudf.Series(ts_data)
    psr = pd.Series(ts_data)
    if not timezone:
        expected = psr.astype(datetime_types_as_str)
        actual = sr.astype(datetime_types_as_str)

        assert_eq(expected, actual)
    else:
        with cudf.option_context("mode.pandas_compatible", True):
            with pytest.raises(NotImplementedError):
                # pandas doesn't allow parsing "Z" to naive type
                sr.astype(datetime_types_as_str)


def test_string_astype(all_supported_types_as_str):
    if all_supported_types_as_str.startswith(
        "int"
    ) or all_supported_types_as_str.startswith("uint"):
        data = ["1", "2", "3", "4", "5"]
    elif all_supported_types_as_str.startswith("float"):
        data = [
            "1.0",
            "2.0",
            "3.0",
            "4.0",
            None,
            "5.0",
            "nan",
            "-INF",
            "NaN",
            "inF",
            "NAn",
        ]
    elif all_supported_types_as_str.startswith("bool"):
        data = ["True", "False", "True", "False", "False"]
    elif all_supported_types_as_str.startswith("datetime64"):
        data = [
            "2019-06-04T00:00:00",
            "2019-06-04T12:12:12",
            "2019-06-03T00:00:00",
            "2019-05-04T00:00:00",
            "2018-06-04T00:00:00",
            "1922-07-21T01:02:03",
        ]
    elif all_supported_types_as_str.startswith("timedelta64"):
        data = [
            "1 days 00:00:00",
            "2 days 00:00:00",
            "3 days 00:00:00",
        ]
    elif all_supported_types_as_str in {"str", "category"}:
        data = ["ab", "cd", "ef", "gh", "ij"]
    ps = pd.Series(data)
    gs = cudf.Series(data)

    expect = ps.astype(all_supported_types_as_str)
    got = gs.astype(all_supported_types_as_str)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data, scale, precision",
    [
        (["1.11", "2.22", "3.33"], 2, 3),
        (["111", "222", "33"], 0, 3),
        (["111000", "22000", "3000"], -3, 3),
        ([None, None, None], 0, 5),
        ([None, "-2345", None], 0, 5),
        ([], 0, 5),
    ],
)
@pytest.mark.parametrize(
    "decimal_dtype",
    [cudf.Decimal128Dtype, cudf.Decimal64Dtype, cudf.Decimal32Dtype],
)
def test_string_to_decimal(data, scale, precision, decimal_dtype):
    gs = cudf.Series(data, dtype="str")
    fp = gs.astype(decimal_dtype(scale=scale, precision=precision))
    got = fp.astype("str")
    assert_eq(gs, got)


def test_string_empty_to_decimal():
    gs = cudf.Series(["", "-85", ""], dtype="str")
    got = gs.astype(cudf.Decimal64Dtype(scale=0, precision=5))
    expected = cudf.Series(
        [0, -85, 0],
        dtype=cudf.Decimal64Dtype(scale=0, precision=5),
    )
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data, scale, precision",
    [
        (["1.23", "-2.34", "3.45"], 2, 3),
        (["123", "-234", "345"], 0, 3),
        (["12300", "-400", "5000.0"], -2, 5),
        ([None, None, None], 0, 5),
        ([None, "-100", None], 0, 5),
        ([], 0, 5),
    ],
)
@pytest.mark.parametrize(
    "decimal_dtype",
    [cudf.Decimal128Dtype, cudf.Decimal32Dtype, cudf.Decimal64Dtype],
)
def test_string_from_decimal(data, scale, precision, decimal_dtype):
    decimal_data = []
    for d in data:
        if d is None:
            decimal_data.append(None)
        else:
            decimal_data.append(Decimal(d))
    fp = cudf.Series(
        decimal_data,
        dtype=decimal_dtype(scale=scale, precision=precision),
    )
    gs = fp.astype("str")
    got = gs.astype(decimal_dtype(scale=scale, precision=precision))
    assert_eq(fp, got)


def test_string_empty_astype(all_supported_types_as_str):
    data = []
    ps = pd.Series(data, dtype="str")
    gs = cudf.Series(data, dtype="str")

    expect = ps.astype(all_supported_types_as_str)
    got = gs.astype(all_supported_types_as_str)

    assert_eq(expect, got)


def test_string_numeric_astype(numeric_and_temporal_types_as_str):
    if numeric_and_temporal_types_as_str.startswith("timedelta64"):
        pytest.skip(
            f"Test not applicable for {numeric_and_temporal_types_as_str}"
        )
    if numeric_and_temporal_types_as_str.startswith("bool"):
        data = [1, 0, 1, 0, 1]
    elif numeric_and_temporal_types_as_str.startswith(
        "int"
    ) or numeric_and_temporal_types_as_str.startswith("uint"):
        data = [1, 2, 3, 4, 5]
    elif numeric_and_temporal_types_as_str.startswith("float"):
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
    elif numeric_and_temporal_types_as_str.startswith("datetime64"):
        # pandas rounds the output format based on the data
        # Use numpy instead
        # but fix '2011-01-01T00:00:00' -> '2011-01-01 00:00:00'
        data = [1000000001, 2000000001, 3000000001, 4000000001, 5000000001]
        ps = np.asarray(data, dtype=numeric_and_temporal_types_as_str).astype(
            str
        )
        ps = np.array([i.replace("T", " ") for i in ps])

    if not numeric_and_temporal_types_as_str.startswith("datetime64"):
        ps = pd.Series(data, dtype=numeric_and_temporal_types_as_str)

    gs = cudf.Series(data, dtype=numeric_and_temporal_types_as_str)

    expect = pd.Series(ps.astype("str"))
    got = gs.astype("str")

    assert_eq(expect, got)


def test_string_empty_numeric_astype(numeric_and_temporal_types_as_str):
    data = []

    ps = pd.Series(data, dtype=numeric_and_temporal_types_as_str)
    gs = cudf.Series(data, dtype=numeric_and_temporal_types_as_str)

    expect = ps.astype("str")
    got = gs.astype("str")

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data,dtype",
    [
        (["0.1", "10.2", "10.876"], "float"),
        (["-0.1", "10.2", "+10.876"], "float"),
        (["1", "10.2", "10.876"], "float32"),
        (["+123", "6344556789", "0"], "int"),
        (["+123", "6344556789", "0"], "uint64"),
        (["+123", "6344556789", "0"], "float"),
        (["0.1", "-10.2", "10.876", None], "float"),
    ],
)
@pytest.mark.parametrize("obj_type", [None, "str", "category"])
def test_string_typecast(data, obj_type, dtype):
    psr = pd.Series(data, dtype=obj_type)
    gsr = cudf.Series(data, dtype=obj_type)

    expect = psr.astype(dtype=dtype)
    actual = gsr.astype(dtype=dtype)
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "data,dtype",
    [
        (["0.1", "10.2", "10.876"], "int"),
        (["1", "10.2", "+10.876"], "int"),
        (["abc", "1", "2", " "], "int"),
        (["0.1", "10.2", "10.876"], "uint64"),
        (["1", "10.2", "+10.876"], "uint64"),
        (["abc", "1", "2", " "], "uint64"),
        ([" ", "0.1", "2"], "float"),
        ([""], "int"),
        ([""], "uint64"),
        ([" "], "float"),
        (["\n"], "int"),
        (["\n"], "uint64"),
        (["0.1", "-10.2", "10.876", None], "int"),
        (["0.1", "-10.2", "10.876", None], "uint64"),
        (["0.1", "-10.2", "10.876", None, "ab"], "float"),
        (["+", "-"], "float"),
        (["+", "-"], "int"),
        (["+", "-"], "uint64"),
        (["1++++", "--2"], "float"),
        (["1++++", "--2"], "int"),
        (["1++++", "--2"], "uint64"),
        (["++++1", "--2"], "float"),
        (["++++1", "--2"], "int"),
        (["++++1", "--2"], "uint64"),
    ],
)
@pytest.mark.parametrize("obj_type", [None, "str", "category"])
def test_string_typecast_error(data, obj_type, dtype):
    psr = pd.Series(data, dtype=obj_type)
    gsr = cudf.Series(data, dtype=obj_type)

    assert_exceptions_equal(
        lfunc=psr.astype,
        rfunc=gsr.astype,
        lfunc_args_and_kwargs=([dtype],),
        rfunc_args_and_kwargs=([dtype],),
    )


@pytest.mark.parametrize("unit", ["ns", "us"])
def test_astype_aware_to_aware(unit):
    ser = cudf.Series(
        [datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)]
    )
    result = ser.astype(f"datetime64[{unit}, US/Pacific]")
    expected = ser.to_pandas().astype(f"datetime64[{unit}, US/Pacific]")
    zoneinfo_type = pd.DatetimeTZDtype(
        expected.dtype.unit, zoneinfo.ZoneInfo(str(expected.dtype.tz))
    )
    expected = ser.astype(zoneinfo_type)
    assert_eq(result, expected)


def test_astype_naive_to_aware_raises():
    ser = cudf.Series([datetime.datetime(2020, 1, 1)])
    with pytest.raises(TypeError):
        ser.astype("datetime64[ns, UTC]")
    with pytest.raises(TypeError):
        ser.to_pandas().astype("datetime64[ns, UTC]")


def test_series_astype_pandas_nullable(
    all_supported_pandas_nullable_extension_dtypes,
):
    scalar, dtype = all_supported_pandas_nullable_extension_dtypes
    got = cudf.Series([scalar, None]).astype(dtype)
    expect = cudf.Series([scalar, None], dtype=dtype)

    assert_eq(expect, got)


def test_series_astype_numeric_to_numeric(
    numeric_types_as_str, numeric_types_as_str2
):
    psr = pd.Series([1, 2, 4, 3], dtype=numeric_types_as_str)
    gsr = cudf.from_pandas(psr)
    assert_eq(
        psr.astype(numeric_types_as_str2), gsr.astype(numeric_types_as_str2)
    )


def test_series_astype_numeric_to_numeric_nulls(
    numeric_types_as_str, numeric_types_as_str2
):
    data = [1, 2, None, 3]
    sr = cudf.Series(data, dtype=numeric_types_as_str)
    got = sr.astype(numeric_types_as_str2)
    expect = cudf.Series([1, 2, None, 3], dtype=numeric_types_as_str2)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "as_dtype",
    [
        "str",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_series_astype_numeric_to_other(
    request, all_supported_types_as_str, as_dtype
):
    if all_supported_types_as_str.startswith(
        "timedelta64"
    ) and as_dtype.startswith("datetime64"):
        pytest.skip(
            f"Casting {all_supported_types_as_str} to {as_dtype} is invalid"
        )
    if all_supported_types_as_str == "str" and as_dtype.startswith(
        "datetime64"
    ):
        pytest.skip(
            f"Casting {all_supported_types_as_str} to {as_dtype} for test data is invalid."
        )
    psr = pd.Series([1, 2, 3], dtype=all_supported_types_as_str)
    gsr = cudf.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize(
    "as_dtype",
    [
        "str",
        "int32",
        "uint32",
        "float32",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_series_astype_string_to_other(as_dtype):
    if "datetime64" in as_dtype:
        data = ["2001-01-01", "2002-02-02", "2000-01-05"]
    else:
        data = ["1", "2", "3"]
    psr = pd.Series(data)
    gsr = cudf.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize(
    "as_dtype",
    [
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
    ],
)
def test_series_astype_datetime_to_other(as_dtype):
    data = ["2001-01-01", "2002-02-02", "2001-01-05"]
    psr = pd.Series(data)
    gsr = cudf.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


@pytest.mark.parametrize(
    "inp",
    [
        ("datetime64[ns]", "2011-01-01 00:00:00.000000000"),
        ("datetime64[us]", "2011-01-01 00:00:00.000000"),
        ("datetime64[ms]", "2011-01-01 00:00:00.000"),
        ("datetime64[s]", "2011-01-01 00:00:00"),
    ],
)
def test_series_astype_datetime_to_string(inp):
    dtype, expect = inp
    base_date = "2011-01-01"
    sr = cudf.Series([base_date], dtype=dtype)
    got = sr.astype(str)[0]
    assert expect == got


@pytest.mark.parametrize(
    "as_dtype",
    [
        "int32",
        "uint32",
        "float32",
        "category",
        "datetime64[s]",
        "datetime64[ms]",
        "datetime64[us]",
        "datetime64[ns]",
        "str",
    ],
)
def test_series_astype_categorical_to_other(as_dtype):
    if "datetime64" in as_dtype:
        data = ["2001-01-01", "2002-02-02", "2000-01-05", "2001-01-01"]
    else:
        data = [1, 2, 3, 1]
    psr = pd.Series(data, dtype="category")
    gsr = cudf.from_pandas(psr)
    assert_eq(psr.astype(as_dtype), gsr.astype(as_dtype))


def test_series_astype_to_categorical_ordered(categorical_ordered):
    psr = pd.Series([1, 2, 3, 1], dtype="category")
    gsr = cudf.from_pandas(psr)

    ordered_dtype_pd = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=categorical_ordered
    )
    ordered_dtype_gd = cudf.CategoricalDtype(
        categories=ordered_dtype_pd.categories,
        ordered=ordered_dtype_pd.ordered,
    )
    assert_eq(
        psr.astype("int32").astype(ordered_dtype_pd).astype("int32"),
        gsr.astype("int32").astype(ordered_dtype_gd).astype("int32"),
    )


def test_series_astype_cat_ordered_to_unordered(categorical_ordered):
    pd_dtype = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=categorical_ordered
    )
    pd_to_dtype = pd.CategoricalDtype(
        categories=[1, 2, 3], ordered=not categorical_ordered
    )
    gd_dtype = cudf.CategoricalDtype(
        categories=pd_dtype.categories,
        ordered=pd_dtype.ordered,
    )
    gd_to_dtype = cudf.CategoricalDtype(
        categories=pd_to_dtype.categories,
        ordered=pd_to_dtype.ordered,
    )

    psr = pd.Series([1, 2, 3], dtype=pd_dtype)
    gsr = cudf.Series([1, 2, 3], dtype=gd_dtype)

    expect = psr.astype(pd_to_dtype)
    got = gsr.astype(gd_to_dtype)

    assert_eq(expect, got)


def test_series_astype_null_cases():
    data = [1, 2, None, 3]

    # numerical to other
    assert_eq(cudf.Series(data, dtype="str"), cudf.Series(data).astype("str"))

    assert_eq(
        cudf.Series(data, dtype="category"),
        cudf.Series(data).astype("category"),
    )

    assert_eq(
        cudf.Series(data, dtype="float32"),
        cudf.Series(data, dtype="int32").astype("float32"),
    )

    assert_eq(
        cudf.Series(data, dtype="float32"),
        cudf.Series(data, dtype="uint32").astype("float32"),
    )

    assert_eq(
        cudf.Series(data, dtype="datetime64[ms]"),
        cudf.Series(data).astype("datetime64[ms]"),
    )

    # categorical to other
    assert_eq(
        cudf.Series(data, dtype="str"),
        cudf.Series(data, dtype="category").astype("str"),
    )

    assert_eq(
        cudf.Series(data, dtype="float32"),
        cudf.Series(data, dtype="category").astype("float32"),
    )

    assert_eq(
        cudf.Series(data, dtype="datetime64[ms]"),
        cudf.Series(data, dtype="category").astype("datetime64[ms]"),
    )

    # string to other
    assert_eq(
        cudf.Series([1, 2, None, 3], dtype="int32"),
        cudf.Series(["1", "2", None, "3"]).astype("int32"),
    )

    assert_eq(
        cudf.Series(
            ["2001-01-01", "2001-02-01", None, "2001-03-01"],
            dtype="datetime64[ms]",
        ),
        cudf.Series(["2001-01-01", "2001-02-01", None, "2001-03-01"]).astype(
            "datetime64[ms]"
        ),
    )

    assert_eq(
        cudf.Series(["a", "b", "c", None], dtype="category").to_pandas(),
        cudf.Series(["a", "b", "c", None]).astype("category").to_pandas(),
    )

    # datetime to other
    data = [
        "2001-01-01 00:00:00.000000",
        "2001-02-01 00:00:00.000000",
        None,
        "2001-03-01 00:00:00.000000",
    ]
    assert_eq(
        cudf.Series(data),
        cudf.Series(data, dtype="datetime64[us]").astype("str"),
    )

    assert_eq(
        pd.Series(data, dtype="datetime64[ns]").astype("category"),
        cudf.from_pandas(pd.Series(data, dtype="datetime64[ns]")).astype(
            "category"
        ),
    )


def test_series_astype_null_categorical():
    # A null-containing categorical cannot be cast to a non-nullable integer
    # dtype, matching pandas (the categories are promoted to float).
    psr = pd.Series([None, None, None], dtype="category")
    gsr = cudf.Series([None, None, None], dtype="category")
    assert_exceptions_equal(
        psr.astype,
        gsr.astype,
        (["int32"],),
        (["int32"],),
    )


# Converting a categorical with nulls to a non-nullable numpy integer dtype
# raises, matching pandas.
@pytest.mark.parametrize(
    "data",
    [
        [1, 2, None],
        [1, 2, 3, None, None],
        [None, None, None],
    ],
)
def test_categorical_astype_nan_to_int_pandas_compat(
    data, integer_types_as_str
):
    psr = pd.Series(data, dtype="category")
    gsr = cudf.Series(data, dtype="category")
    assert_exceptions_equal(
        psr.astype,
        gsr.astype,
        ([integer_types_as_str],),
        ([integer_types_as_str],),
    )


def test_categorical_astype_nan_to_int_via_np_dtype_pandas_compat(
    integer_types_as_str,
):
    psr = pd.Series([1, 2, None], dtype="category")
    gsr = cudf.Series([1, 2, None], dtype="category")
    assert_exceptions_equal(
        psr.astype,
        gsr.astype,
        ([np.dtype(integer_types_as_str)],),
        ([np.dtype(integer_types_as_str)],),
    )


@pytest.mark.parametrize("dtype", ["bool", np.dtype("bool")])
def test_categorical_astype_nan_to_bool_pandas_compat_deviation(dtype):
    # cuDF intentionally deviates from pandas here. Casting a null-containing
    # categorical to ``bool``:
    #   * pandas promotes to ``object`` dtype, e.g. ``[True, True, nan]``
    #   * cuDF has no object-backed bool, so it keeps a nullable ``bool``
    #     column, e.g. ``[True, True, <NA>]``
    psr = pd.Series([1, 2, None], dtype="category")
    gsr = cudf.Series([1, 2, None], dtype="category")
    got = gsr.astype(dtype)
    # Confirm pandas' (divergent) object-dtype behavior so this test fails if
    # pandas ever changes and the deviation needs revisiting.
    assert psr.astype(dtype).dtype == object
    # cuDF's intentional nullable-bool result.
    assert_eq(got, cudf.Series([True, True, None], dtype="bool"))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_categorical_astype_nan_to_float_pandas_compat(dtype):
    psr = pd.Series([1, 2, None], dtype="category")
    gsr = cudf.Series([1, 2, None], dtype="category")
    assert_eq(psr.astype(dtype), gsr.astype(dtype))


def test_categorical_astype_no_nulls_int_pandas_compat(integer_types_as_str):
    psr = pd.Series([1, 2, 3], dtype="category")
    gsr = cudf.Series([1, 2, 3], dtype="category")
    assert_eq(
        psr.astype(integer_types_as_str),
        gsr.astype(integer_types_as_str),
    )


def test_categorical_astype_no_nulls_bool_pandas_compat():
    psr = pd.Series([1, 2, 3], dtype="category")
    gsr = cudf.Series([1, 2, 3], dtype="category")
    assert_eq(psr.astype("bool"), gsr.astype("bool"))


def test_categorical_astype_nan_to_int_default_mode(integer_types_as_str):
    # Casting a null-containing categorical to a non-nullable integer dtype
    # raises like pandas regardless of ``mode.pandas_compatible``.
    psr = pd.Series([1, 2, None], dtype="category")
    gsr = cudf.Series([1, 2, None], dtype="category")
    assert_exceptions_equal(
        psr.astype,
        gsr.astype,
        ([integer_types_as_str],),
        ([integer_types_as_str],),
    )


def test_categorical_astype_nan_to_bool_default_mode():
    gsr = cudf.Series([1, 2, None], dtype="category")
    assert_eq(
        gsr.astype("bool"), cudf.Series([True, True, None], dtype="bool")
    )


@pytest.mark.parametrize(
    "ext_dtype",
    [
        pd.Int8Dtype(),
        pd.Int16Dtype(),
        pd.Int32Dtype(),
        pd.Int64Dtype(),
        pd.UInt8Dtype(),
        pd.UInt16Dtype(),
        pd.UInt32Dtype(),
        pd.UInt64Dtype(),
    ],
)
def test_categorical_astype_nan_to_nullable_extension_pandas_compat(ext_dtype):
    psr = pd.Series([1, 2, None], dtype="category")
    gsr = cudf.Series([1, 2, None], dtype="category")
    assert_eq(psr.astype(ext_dtype), gsr.astype(ext_dtype))


@pytest.mark.parametrize(
    "pa_type",
    [pa.int8(), pa.int64(), pa.uint32(), pa.bool_()],
)
def test_categorical_astype_nan_to_arrow_pandas_compat(pa_type):
    arrow_dtype = pd.ArrowDtype(pa_type)
    psr = pd.Series([1, 2, None], dtype="category")
    gsr = cudf.Series([1, 2, None], dtype="category")
    assert_eq(psr.astype(arrow_dtype), gsr.astype(arrow_dtype))


@pytest.mark.parametrize("dtype", ["int8", "int64", "uint32"])
def test_categoricalindex_astype_nan_to_int_pandas_compat(dtype):
    pi = pd.CategoricalIndex([1, 2, None])
    gi = cudf.CategoricalIndex([1, 2, None])
    assert_exceptions_equal(
        pi.astype,
        gi.astype,
        ([dtype],),
        ([dtype],),
    )


def test_categoricalindex_astype_nan_to_bool_pandas_compat():
    gi = cudf.CategoricalIndex([1, 2, None])
    got = gi.astype("bool")
    assert_eq(got, cudf.Index([True, True, None], dtype="bool"))


def test_categoricalindex_astype_nan_to_float_pandas_compat():
    pi = pd.CategoricalIndex([1, 2, None])
    gi = cudf.CategoricalIndex([1, 2, None])
    assert_eq(pi.astype("float64"), gi.astype("float64"))


@pytest.mark.parametrize("precision, scale", [(7, 2), (11, 4), (18, 9)])
def test_typecast_from_float_to_decimal(
    request, float_types_as_str, precision, scale
):
    to_dtype = cudf.Decimal64Dtype(precision, scale)
    data = cudf.Series(
        [
            14.12302,
            97938.2,
            np.nan,
            0.0,
            -8.302014,
            np.nan,
            94.31304,
            -112.2314,
            0.3333333,
            np.nan,
        ]
    )
    request.applymarker(
        pytest.mark.xfail(
            float_types_as_str == "float32" and to_dtype.precision > 12,
            reason="https://github.com/rapidsai/cudf/issues/14169",
        )
    )
    got = data.astype(float_types_as_str)

    pa_arr = got.to_arrow().cast(
        pa.decimal64(to_dtype.precision, to_dtype.scale)
    )
    expected = cudf.Series._from_column(DecimalColumn.from_arrow(pa_arr))

    got = got.astype(to_dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize("precision, scale", [(9, 3), (11, 4), (18, 9)])
def test_typecast_from_int_to_decimal(integer_types_as_str, precision, scale):
    to_dtype = cudf.Decimal64Dtype(precision, scale)
    data = cudf.Series(
        [
            14.12302,
            38.2,
            np.nan,
            0.0,
            -8.302014,
            np.nan,
            94.31304,
            np.nan,
            -112.2314,
            0.3333333,
            np.nan,
        ]
    )
    got = data.astype(integer_types_as_str)

    pa_arr = (
        got.to_arrow()
        .cast("float64")
        .cast(pa.decimal64(to_dtype.precision, to_dtype.scale))
    )
    expected = cudf.Series._from_column(DecimalColumn.from_arrow(pa_arr))

    got = got.astype(to_dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize(
    "from_dtype",
    [
        cudf.Decimal64Dtype(7, 2),
        cudf.Decimal64Dtype(11, 4),
        cudf.Decimal64Dtype(18, 10),
        cudf.Decimal32Dtype(7, 2),
        cudf.Decimal32Dtype(5, 3),
        cudf.Decimal32Dtype(9, 5),
    ],
)
@pytest.mark.parametrize(
    "to_dtype",
    [
        cudf.Decimal64Dtype(7, 2),
        cudf.Decimal64Dtype(18, 10),
        cudf.Decimal64Dtype(11, 4),
        cudf.Decimal32Dtype(7, 2),
        cudf.Decimal32Dtype(9, 5),
        cudf.Decimal32Dtype(5, 3),
    ],
)
def test_typecast_to_from_decimal(from_dtype, to_dtype):
    data = cudf.Series(
        [
            14.12309,
            2.343942,
            np.nan,
            0.0,
            -8.302082,
            np.nan,
            94.31308,
            -112.2364,
            -8.029972,
            np.nan,
        ]
    )
    if from_dtype.scale > to_dtype.MAX_PRECISION:
        pytest.skip(
            "This is supposed to overflow because the representation value in "
            "the source exceeds the max representable in destination dtype."
        )
    s = data.astype(from_dtype)

    if isinstance(to_dtype, cudf.Decimal32Dtype):
        pa_arr = s.to_arrow().cast(
            pa.decimal32(to_dtype.precision, to_dtype.scale), safe=False
        )
        expected = cudf.Series._from_column(DecimalColumn.from_arrow(pa_arr))
    elif isinstance(to_dtype, cudf.Decimal64Dtype):
        pa_arr = s.to_arrow().cast(
            pa.decimal64(to_dtype.precision, to_dtype.scale), safe=False
        )
        expected = cudf.Series._from_column(DecimalColumn.from_arrow(pa_arr))

    with expect_warning_if(to_dtype.scale < s.dtype.scale, UserWarning):
        got = s.astype(to_dtype)

    assert_eq(got, expected)


@pytest.mark.parametrize("precision, scale", [(7, 2), (11, 4), (17, 10)])
def test_typecast_from_decimal(precision, scale, signed_integer_types_as_str):
    from_dtype = cudf.Decimal64Dtype(precision, scale)
    data = cudf.Series(
        [
            14.12309,
            2.343942,
            np.nan,
            0.0,
            -8.302082,
            np.nan,
            94.31308,
            -112.2364,
            -8.029972,
            np.nan,
        ]
    )
    got = data.astype(from_dtype)
    pa_arr = got.to_arrow().cast(signed_integer_types_as_str, safe=False)

    got = got.astype(signed_integer_types_as_str)
    expected = cudf.Series._from_column(NumericalColumn.from_arrow(pa_arr))

    assert_eq(got, expected)
    assert_eq(got.dtype, expected.dtype)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([1, 2, 3, 89]),
        pd.Series([1, 2, 3, 89, 3, 1, 89], dtype="category"),
        pd.Series(["1", "2", "3", "4", "5"], dtype="category"),
        pd.Series(["1.0", "2.5", "3.001", "9"], dtype="category"),
        pd.Series(["1", "2", "3", None, "4", "5"], dtype="category"),
        pd.Series(["1.0", "2.5", "3.001", None, "9"], dtype="category"),
        pd.Series(["a", "b", "c", "c", "b", "a", "b", "b"]),
        pd.Series(["aa", "b", "c", "c", "bb", "bb", "a", "b", "b"]),
        pd.Series([1, 2, 3, 89, None, np.nan, np.nan], dtype="float64"),
        pd.Series([1, 2, 3, 89], dtype="float64"),
        pd.Series([1, 2.5, 3.001, 89], dtype="float64"),
        pd.Series([None, None, None]),
        pd.Series([], dtype="float64"),
    ],
)
@pytest.mark.parametrize(
    "categories",
    [
        ["aa", "bb", "cc"],
        [2, 4, 10, 100],
        ["aa", "bb", "c"],
        ["a", "bb", "c"],
        ["a", "b", "c"],
        ["1", "2", "3", "4"],
        ["1.0", "2.5", "3.001", "9"],
        [],
    ],
)
def test_categorical_typecast(data, categories):
    pd_data = data
    gd_data = cudf.from_pandas(data)
    cat_type = pd.CategoricalDtype(categories)

    actual = gd_data.astype(cat_type)
    warns = actual.null_count != gd_data.null_count
    with expect_warning_if(warns, pd.errors.Pandas4Warning):
        expected = pd_data.astype(cat_type)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # cat.codes uses a signed dtype matching pandas (which widens
        # int8 -> int16 -> int32 once the category count reaches each type's
        # max), so e.g. 127 categories -> int16 and 32767 categories -> int32.
        ([1], np.int8),
        ([1, None], np.int8),
        (np.arange(np.iinfo(np.int8).max), np.int16),
        (np.append(np.arange(np.iinfo(np.int8).max), [None]), np.int16),
        (np.arange(np.iinfo(np.int16).max), np.int32),
        (np.append(np.arange(np.iinfo(np.int16).max), [None]), np.int32),
        (np.arange(np.iinfo(np.uint8).max), np.int16),
        (np.append(np.arange(np.iinfo(np.uint8).max), [None]), np.int16),
        (np.arange(np.iinfo(np.uint16).max), np.int32),
        (np.append(np.arange(np.iinfo(np.uint16).max), [None]), np.int32),
    ],
)
def test_astype_dtype(values, expected):
    data = cudf.Series(values)
    got = data.astype("category").cat.codes.dtype
    np.testing.assert_equal(got, expected)


@pytest.mark.parametrize("ordered", [True, False])
def test_empty_series_category_cast(ordered):
    dtype = cudf.CategoricalDtype(ordered=ordered)
    ps = pd.Series([], dtype="str")
    gs = cudf.from_pandas(ps)

    expected = ps.astype(dtype.to_pandas())
    actual = gs.astype(dtype)

    assert_eq(expected, actual)
    assert_eq(expected.dtype.ordered, actual.dtype.ordered)


@pytest.mark.parametrize("copy", [True, False])
def test_series_astype_no_copy(copy):
    gsr = cudf.Series([1, 2, 3])
    result = gsr.astype("int64", copy=copy)
    assert_eq(result, gsr)
    # Under CoW, copy=False returns a shallow copy (distinct object).
    assert result is not gsr


def test_astype_aware_to_naive_raises():
    data = [datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)]
    pd_ser = pd.Series(data)
    cudf_ser = cudf.Series(data)
    with pytest.raises(TypeError):
        cudf_ser.astype("datetime64[ns]")
    with pytest.raises(TypeError):
        pd_ser.astype("datetime64[ns]")


@pytest.mark.parametrize(
    "data",
    [
        ["123_1"],
        ["1_2", "1_000", "20_000"],
        ["1_2_3"],
        ["123", "456"],
    ],
)
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint64"])
def test_string_astype_int_pep515_underscores(data, dtype):
    # https://github.com/rapidsai/cudf/issues/12047
    # Python (PEP 515) allows underscores between digits, so pandas
    # parses "123_1" as 1231; cudf should match.
    got = cudf.Series(data).astype(dtype)
    expect = pd.Series(data).astype(dtype)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        ["_12"],
        ["12_"],
        ["1__2"],
        ["1_2", "_3"],
    ],
)
@pytest.mark.parametrize("dtype", ["int32", "int64", "uint64"])
def test_string_astype_int_invalid_underscores_raises(data, dtype):
    # https://github.com/rapidsai/cudf/issues/12047
    # Underscores not surrounded by digits are invalid (PEP 515);
    # both pandas and cudf must reject them.
    assert_exceptions_equal(
        lfunc=pd.Series(data).astype,
        rfunc=cudf.Series(data).astype,
        lfunc_args_and_kwargs=((), {"dtype": dtype}),
        rfunc_args_and_kwargs=((), {"dtype": dtype}),
    )


@pytest.mark.parametrize(
    "data, src_dtype, masked_dtype",
    [
        ([1.0, 2.0, float("nan")], "float64", pd.Float64Dtype()),
        ([1, 2, 3], "int64", pd.Int64Dtype()),
    ],
)
def test_astype_masked_equivalent_dtype_no_source_mutation(
    data, src_dtype, masked_dtype
):
    # casting to the equivalent masked dtype takes a short-circuit path;
    # it must not mutate the source column's dtype in place (the column
    # is shared with the source Series/frame)
    ser = cudf.Series(data, dtype=src_dtype)
    result = ser.astype(masked_dtype)

    assert ser.dtype == np.dtype(src_dtype)
    assert result.dtype == masked_dtype
    assert_eq(
        result.to_pandas(),
        pd.Series(data, dtype=src_dtype).astype(masked_dtype),
    )

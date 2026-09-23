# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.fixture(params=["integer", "signed", "unsigned", "float"])
def downcast(request):
    return request.param


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        (1.0, 2.0, 3.0),
        [float("nan"), None],
        np.array([1, 2.0, -3, float("nan")]),
        pd.Series(["123", "2.0"]),
        pd.Series(["1.0", "2.", "-.3", "1e6"]),
        pd.Series(
            ["1", "2", "3"],
            dtype=pd.CategoricalDtype(categories=["1", "2", "3"]),
        ),
        pd.Series(
            ["1.0", "2.0", "3.0"],
            dtype=pd.CategoricalDtype(categories=["1.0", "2.0", "3.0"]),
        ),
        # Categories with nulls
        pd.Series(pd.Categorical.from_codes([0, 1, -1], categories=[1, 2])),
        pd.Series(
            [5.0, 6.0], dtype=pd.CategoricalDtype(categories=[5.0, 6.0])
        ),
        pd.Series(
            ["2020-08-01 08:00:00", "1960-08-01 08:00:00"],
            dtype=np.dtype("<M8[ns]"),
        ),
        pd.Series(
            [pd.Timedelta(days=1, seconds=1), pd.Timedelta("-3 seconds 4ms")],
            dtype=np.dtype("<m8[ns]"),
        ),
        [
            "inf",
            "-inf",
            "+inf",
            "infinity",
            "-infinity",
            "+infinity",
            "inFInity",
        ],
    ],
)
def test_to_numeric_basic_1d(data):
    expected = pd.to_numeric(data)
    got = cudf.to_numeric(data)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2**11],
        [1, 2**33],
        [1, 2**63],
        [np.iinfo(np.int64).max, np.iinfo(np.int64).min],
    ],
)
def test_to_numeric_downcast_int(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0**11],
        [-1.0, -(2.0**11)],
        [1.0, 2.0**33],
        [-1.0, -(2.0**33)],
        [1.0, 2.0**65],
        [-1.0, -(2.0**65)],
        [1.0, float("inf")],
        [1.0, float("-inf")],
        [1.0, float("nan")],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 1.5, 2.6, 3.4],
    ],
)
def test_to_numeric_downcast_float(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast")
@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0**129],
        [1.0, 2.0**257],
        [1.0, 1.79e308],
        [-1.0, -(2.0**129)],
        [-1.0, -(2.0**257)],
        [-1.0, -1.79e308],
    ],
)
def test_to_numeric_downcast_large_float(data, downcast):
    if downcast == "float":
        pytest.skip(f"{downcast=} not applicable for test")

    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.filterwarnings("ignore:overflow encountered in cast")
@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0**129],
        [1.0, 2.0**257],
        [1.0, 1.79e308],
        [-1.0, -(2.0**129)],
        [-1.0, -(2.0**257)],
        [-1.0, -1.79e308],
    ],
)
def test_to_numeric_downcast_large_float_pd_bug(data):
    downcast = "float"
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        ["1", "2", "3"],
        [str(np.iinfo(np.int64).max), str(np.iinfo(np.int64).min)],
    ],
)
def test_to_numeric_downcast_string_int(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        [""],  # pure empty strings
        ["10.0", "11.0", "2e3"],
        ["1.0", "2e3"],
        ["1", "10", "1.0", "2e3"],  # int-float mixed
        ["1", "10", "1.0", "2e3", "2e+3", "2e-3"],
        ["1", "10", "1.0", "2e3", "", ""],  # mixed empty strings
    ],
)
def test_to_numeric_downcast_string_float(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expected = pd.to_numeric(ps, downcast=downcast)

    if downcast in {"signed", "integer", "unsigned"}:
        with pytest.warns(
            UserWarning,
            match="Downcasting from float to int "
            "will be limited by float32 precision.",
        ):
            got = cudf.to_numeric(gs, downcast=downcast)
    else:
        got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expected, got)


@pytest.mark.filterwarnings("ignore:overflow encountered in cast")
@pytest.mark.parametrize(
    "data",
    [
        ["2e128", "-2e128"],
        [
            "1.79769313486231e308",
            "-1.79769313486231e308",
        ],  # 2 digits relaxed from np.finfo(np.float64).min/max
    ],
)
def test_to_numeric_downcast_string_large_float(data, downcast):
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    if downcast == "float":
        expected = pd.to_numeric(ps, downcast=downcast)
        got = cudf.to_numeric(gs, downcast=downcast)

        assert_eq(expected, got)
    else:
        expected = pd.Series([np.inf, -np.inf])
        with pytest.warns(
            UserWarning,
            match="Downcasting from float to int "
            "will be limited by float32 precision.",
        ):
            got = cudf.to_numeric(gs, downcast=downcast)

        assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series(["1", "a", "3"]),
        pd.Series(["1", "a", "3", ""]),  # mix of unconvertible and empty str
    ],
)
@pytest.mark.parametrize("errors", ["raise", "coerce"])
def test_to_numeric_error(data, errors):
    if errors == "raise":
        with pytest.raises(
            ValueError, match=r"Unable to convert some strings to numerics."
        ):
            cudf.to_numeric(data, errors=errors)
    else:
        expect = pd.to_numeric(data, errors=errors)
        got = cudf.to_numeric(data, errors=errors)

        assert_eq(expect, got)


def test_series_to_numeric_bool(downcast):
    data = [True, False, True]
    ps = pd.Series(data)
    gs = cudf.from_pandas(ps)

    expect = pd.to_numeric(ps, downcast=downcast)
    got = cudf.to_numeric(gs, downcast=downcast)

    assert_eq(expect, got)


@pytest.mark.parametrize("klass", [cudf.Series, pd.Series])
def test_series_to_numeric_preserve_index_name(klass):
    ser = klass(["1"] * 8, index=range(2, 10), name="name")
    result = cudf.to_numeric(ser)
    expected = cudf.Series([1] * 8, index=range(2, 10), name="name")
    assert_eq(result, expected)


@pytest.mark.parametrize("klass", [cudf.Index, pd.Index])
@pytest.mark.parametrize(
    "data",
    [
        ["1", "2", "3"],
        [1, 2, 3],
        [1.0, np.nan, 3.0],
    ],
)
def test_to_numeric_index_returns_index(klass, data):
    # to_numeric on an Index returns an Index, preserving the name (matching
    # pandas) rather than an ndarray.
    expected = pd.to_numeric(pd.Index(data, name="idx"))
    got = cudf.to_numeric(klass(data, name="idx"))

    assert isinstance(got, cudf.Index)
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        ["12345678901234567890"],  # > int64 max, fits uint64
        [str(np.iinfo(np.uint64).max)],
    ],
)
def test_to_numeric_string_uint64(data):
    # An integer string that overflows int64 but fits uint64 is parsed as
    # uint64 (previously cudf silently wrapped it to a garbage int64).
    expected = pd.to_numeric(pd.Series(data))
    got = cudf.to_numeric(cudf.Series(data))

    assert got.dtype == np.dtype(np.uint64)
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        ["47393996303418497800"],  # > uint64 max
        ["100000000000000000000"],
        ["-9223372036854775809"],  # < int64 min, negative so cannot use uint64
    ],
)
def test_to_numeric_string_int_overflow_raises(data):
    # Values beyond the int64/uint64 range have no cudf numeric representation
    # (pandas returns an object array), so cudf raises instead of wrapping.
    with pytest.raises(OverflowError):
        cudf.to_numeric(cudf.Series(data))


@pytest.mark.parametrize(
    "data",
    [
        ["47393996303418497800"],
        ["100000000000000000000"],
    ],
)
def test_to_numeric_string_int_overflow_coerce(data):
    # With errors="coerce" an out-of-range integer string is represented as a
    # float, matching pandas.
    expected = pd.to_numeric(pd.Series(data), errors="coerce")
    got = cudf.to_numeric(cudf.Series(data), errors="coerce")

    assert got.dtype == np.dtype(np.float64)
    assert_eq(expected, got)


@pytest.mark.parametrize("kind", ["datetime64", "timedelta64"])
@pytest.mark.parametrize("unit", ["D", "s", "ms", "us", "ns"])
def test_to_numeric_temporal_native_resolution(kind, unit):
    # to_numeric returns the raw integer view at the array's native
    # resolution, e.g. datetime64[D] -> [1, 2, 3] (not the [s] upcast value).
    arr = np.array([1, 2, 3], dtype=f"{kind}[{unit}]")
    expected = pd.to_numeric(arr)
    got = cudf.to_numeric(arr)

    assert_eq(expected, got)


@pytest.mark.parametrize("dtype", ["string[python]", "string[pyarrow]"])
@pytest.mark.parametrize(
    "data",
    [
        ["1", "2", None],
        ["1", "2", "3"],
        ["1", "2", "3.5"],
        ["1", None, "3.5"],
    ],
)
def test_to_numeric_nullable_string(data, dtype):
    # A pandas masked (nullable) string dtype yields a masked Int64/Float64
    # result, preserving nulls as pd.NA.
    expected = pd.to_numeric(pd.Series(data, dtype=dtype))
    got = cudf.to_numeric(cudf.Series(pd.array(data, dtype=dtype)))

    assert_eq(expected, got)


@pytest.mark.parametrize("dtype", ["string[python]", "string[pyarrow]"])
def test_to_numeric_nullable_string_coerce(dtype):
    # Coerced (nulled) values do not force a float result: an all-integer
    # remainder still yields Int64.
    data = ["a", "1"]
    expected = pd.to_numeric(pd.Series(data, dtype=dtype), errors="coerce")
    got = cudf.to_numeric(
        cudf.Series(pd.array(data, dtype=dtype)), errors="coerce"
    )

    assert_eq(expected, got)


@pytest.mark.parametrize("dtype", ["string[python]", "string[pyarrow]"])
@pytest.mark.parametrize(
    "value",
    [
        "9223372036854775808",  # int64 max + 1
        "18446744073709551615",  # uint64 max
    ],
)
def test_to_numeric_nullable_string_uint64(value, dtype):
    # Values above the int64 range route through the uint64 path and yield a
    # masked UInt64 result, instead of silently wrapping to a negative int64.
    expected = pd.to_numeric(pd.Series([value], dtype=dtype))
    got = cudf.to_numeric(cudf.Series(pd.array([value], dtype=dtype)))

    assert_eq(expected, got)


@pytest.mark.parametrize("dtype", ["string[python]", "string[pyarrow]"])
def test_to_numeric_nullable_string_overflow_coerce(dtype):
    # A value above the uint64 range is represented as a (masked) float when
    # coercing, matching pandas, instead of wrapping to int64.
    value = "18446744073709551616"  # uint64 max + 1
    expected = pd.to_numeric(pd.Series([value], dtype=dtype), errors="coerce")
    got = cudf.to_numeric(
        cudf.Series(pd.array([value], dtype=dtype)), errors="coerce"
    )

    assert_eq(expected, got)


@pytest.mark.parametrize("dtype", ["string[python]", "string[pyarrow]"])
def test_to_numeric_nullable_string_overflow_raises(dtype):
    # cudf has no object numeric type, so an integer string beyond the uint64
    # range raises (under cudf.pandas this triggers a fallback to pandas, which
    # returns an object array of Python ints).
    value = "18446744073709551616"  # uint64 max + 1
    with pytest.raises(OverflowError):
        cudf.to_numeric(cudf.Series(pd.array([value], dtype=dtype)))


@pytest.mark.parametrize("val", [9876543210.0, 2.0**128])
def test_to_numeric_large_float_not_downcast_to_float32(val):
    # float64 is preserved when narrowing to float32 would lose information
    # (precision loss for 9876543210.0, overflow to inf for 2**128).
    expected = pd.to_numeric(pd.Series([val]), downcast="float")
    got = cudf.to_numeric(cudf.Series([val]), downcast="float")

    assert got.dtype == np.dtype(np.float64)
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "dtype, downcast, min_max",
    [
        ("int8", "integer", [np.iinfo(np.int8).min, np.iinfo(np.int8).max]),
        ("int16", "integer", [np.iinfo(np.int16).min, np.iinfo(np.int16).max]),
        ("int32", "integer", [np.iinfo(np.int32).min, np.iinfo(np.int32).max]),
        (
            "uint8",
            "unsigned",
            [np.iinfo(np.uint8).min, np.iinfo(np.uint8).max],
        ),
        (
            "uint16",
            "unsigned",
            [np.iinfo(np.uint16).min, np.iinfo(np.uint16).max],
        ),
        (
            "uint32",
            "unsigned",
            [np.iinfo(np.uint32).min, np.iinfo(np.uint32).max],
        ),
    ],
)
def test_to_numeric_downcast_limits(dtype, downcast, min_max):
    # Downcast selects the smallest dtype whose range *inclusively* covers the
    # values (regression: the exact dtype max was previously rejected).
    got = cudf.to_numeric(cudf.Series(min_max), downcast=downcast)
    assert got.dtype == np.dtype(dtype)

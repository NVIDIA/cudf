# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import textwrap

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st

import cudf


@pytest.mark.parametrize("nrows", [0, 5, 10])
def test_null_series(nrows, all_supported_types_as_str, request):
    request.applymarker(
        pytest.mark.xfail(
            all_supported_types_as_str in {"bool"},
            reason=f"cuDF repr doesn't match pandas repr for {all_supported_types_as_str}",
        )
    )
    rng = np.random.default_rng(seed=0)
    size = 5
    sr = cudf.Series(rng.integers(1, 9, size)).astype(
        all_supported_types_as_str
    )
    sr[rng.choice([False, True], size=size)] = None
    if all_supported_types_as_str != "category" and cudf.dtype(
        all_supported_types_as_str
    ).kind in {"u", "i"}:
        ps = sr.to_pandas(nullable=True)
    else:
        ps = sr.to_pandas()

    with pd.option_context("display.max_rows", int(nrows)):
        psrepr = repr(ps)
        if "category" in psrepr:
            psrepr = psrepr.replace("NaN", "<NA>")
        if "UInt" in psrepr:
            psrepr = psrepr.replace("UInt", "uint")
        elif "Int" in psrepr:
            psrepr = psrepr.replace("Int", "int")
        assert psrepr.split() == repr(sr).split()


@pytest.mark.parametrize("nrows", [None, 0, 2, 10, 20, 21])
def test_full_series(nrows, all_supported_types_as_str):
    size = 20
    rng = np.random.default_rng(seed=0)
    ps = pd.Series(rng.integers(0, 100, size)).astype(
        all_supported_types_as_str
    )
    sr = cudf.from_pandas(ps)
    with pd.option_context("display.max_rows", nrows):
        assert repr(ps) == repr(sr)


@given(
    st.lists(
        st.integers(-9223372036854775808, 9223372036854775807), max_size=1000
    )
)
@settings(deadline=None, max_examples=20)
def test_integer_series(x):
    sr = cudf.Series(x, dtype=int)
    ps = pd.Series(data=x, dtype=int)

    assert repr(sr) == repr(ps)


@given(st.lists(st.floats()))
@settings(deadline=None, max_examples=20)
def test_float_series(x):
    sr = cudf.Series(x, dtype=float, nan_as_null=False)
    ps = pd.Series(data=x, dtype=float)
    assert repr(sr) == repr(ps)


@pytest.mark.parametrize(
    "psr",
    [
        pd.Series([1, 2, 3], index=[10, 20, None]),
        pd.Series([1, None, 3], name="a", index=[None, "a", "b"]),
        pd.Series(None, index=[None, "a", "b"], dtype="float"),
        pd.Series([None, None], name="aa", index=[None, None]),
        pd.Series([1, 2, 3], index=[None, None, None]),
        pd.Series(
            [None, 2, 3],
            index=np.array([1, None, None], dtype="datetime64[ns]"),
        ),
        pd.Series(
            [None, None, None],
            index=np.array([None, None, None], dtype="datetime64[ns]"),
        ),
        pd.Series(
            [1, None, 3],
            index=np.array([10, 15, None], dtype="datetime64[ns]"),
        ),
        pd.DataFrame(
            {"a": [1, 2, None], "v": [10, None, 22], "p": [100, 200, 300]}
        ).set_index(["a", "v"])["p"],
        pd.DataFrame(
            {
                "a": [1, 2, None],
                "v": ["n", "c", "a"],
                "p": [None, None, None],
            }
        ).set_index(["a", "v"])["p"],
        pd.DataFrame(
            {
                "a": np.array([1, None, None], dtype="datetime64[ns]"),
                "v": ["n", "c", "a"],
                "p": [None, None, None],
            }
        ).set_index(["a", "v"])["p"],
    ],
)
def test_series_null_index_repr(psr):
    gsr = cudf.from_pandas(psr)

    expected_repr = repr(psr)
    actual_repr = repr(gsr)
    assert expected_repr.split() == actual_repr.split()


@pytest.mark.parametrize(
    "data",
    [
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [
            136457654,
            134736784,
            245345345,
            223432411,
            2343241,
            3634548734,
            23234,
        ],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize("dtype", ["timedelta64[s]", "timedelta64[us]"])
def test_timedelta_series_s_us_repr(data, dtype):
    sr = cudf.Series(data, dtype=dtype)
    psr = sr.to_pandas()

    expected = repr(psr).replace("timedelta64[ns]", dtype)
    actual = repr(sr)

    assert expected.split() == actual.split()


@pytest.mark.parametrize(
    "ser",
    [
        lambda: cudf.Series([], dtype="timedelta64[ns]"),
        lambda: cudf.Series([], dtype="timedelta64[ms]"),
        lambda: cudf.Series(
            [1000000, 200000, 3000000], dtype="timedelta64[ns]"
        ),
        lambda: cudf.Series(
            [1000000, 200000, 3000000], dtype="timedelta64[ms]"
        ),
        lambda: cudf.Series([1000000, 200000, None], dtype="timedelta64[ns]"),
        lambda: cudf.Series([1000000, 200000, None], dtype="timedelta64[ms]"),
        lambda: cudf.Series(
            [None, None, None, None, None], dtype="timedelta64[ns]"
        ),
        lambda: cudf.Series(
            [None, None, None, None, None], dtype="timedelta64[ms]"
        ),
        lambda: cudf.Series(
            [12, 12, 22, 343, 4353534, 435342], dtype="timedelta64[ns]"
        ),
        lambda: cudf.Series(
            [12, 12, 22, 343, 4353534, 435342], dtype="timedelta64[ms]"
        ),
        lambda: cudf.Series(
            [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
            dtype="timedelta64[ns]",
        ),
        lambda: cudf.Series(
            [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
            dtype="timedelta64[ms]",
        ),
        lambda: cudf.Series(
            [
                13645765432432,
                134736784,
                245345345,
                223432411,
                999992343241,
                3634548734,
                23234,
            ],
            dtype="timedelta64[ms]",
        ),
        lambda: cudf.Series(
            [
                13645765432432,
                134736784,
                245345345,
                223432411,
                999992343241,
                3634548734,
                23234,
            ],
            dtype="timedelta64[ns]",
        ),
        lambda: cudf.Series(
            [
                13645765432432,
                134736784,
                245345345,
                223432411,
                999992343241,
                3634548734,
                23234,
            ],
            dtype="timedelta64[ms]",
            name="abc",
        ),
        lambda: cudf.Series(
            [
                13645765432432,
                134736784,
                245345345,
                223432411,
                999992343241,
                3634548734,
                23234,
            ],
            dtype="timedelta64[ns]",
            index=["a", "b", "z", "x", "y", "l", "m"],
            name="hello",
        ),
    ],
)
def test_timedelta_series_ns_ms_repr(ser):
    gsr = ser()
    psr = gsr.to_pandas()
    assert repr(psr).split() == repr(gsr).split()


def test_categorical_series_with_nan_repr():
    series = cudf.Series([1, 2.0, np.nan, 10, np.nan, np.nan]).astype(
        "category"
    )

    expected_repr = textwrap.dedent(
        """
    0     1.0
    1     2.0
    2    <NA>
    3    10.0
    4    <NA>
    5    <NA>
    dtype: category
    Categories (3, float64): [1.0, 2.0, 10.0]
    """
    )
    assert repr(series).split() == expected_repr.split()

    sliced_expected_repr = textwrap.dedent(
        """
        2    <NA>
        3    10.0
        4    <NA>
        5    <NA>
        dtype: category
        Categories (3, float64): [1.0, 2.0, 10.0]
        """
    )

    assert repr(series[2:]).split() == sliced_expected_repr.split()


@pytest.mark.parametrize("tz", [None, "US/Eastern"])
@pytest.mark.parametrize("ordered", [False, True])
def test_categorical_series_repr_datetime_multiline(tz, ordered):
    idx = pd.date_range(
        "2011-01-01 09:00", freq="h", periods=5, tz=tz, unit="ns"
    )
    cat = pd.Categorical(idx, ordered=ordered)
    expected = repr(pd.Series(cat))
    got = repr(cudf.Series(cat))
    assert got == expected


def test_empty_series_name():
    ps = pd.Series([], name="abc", dtype="int")
    gs = cudf.from_pandas(ps)

    assert repr(ps) == repr(gs)


@pytest.mark.parametrize("item", [0, slice(0, 1)])
@pytest.mark.parametrize("data", [["a"], ["a", None], [None]])
def test_string_repr(data, item, request):
    ps = pd.Series(data, dtype="str", name="nice name")
    gs = cudf.Series(data, dtype="str", name="nice name")

    got_out = gs.iloc[item]
    expect_out = ps.iloc[item]

    expect = str(expect_out)
    got = str(got_out)
    assert expect == got

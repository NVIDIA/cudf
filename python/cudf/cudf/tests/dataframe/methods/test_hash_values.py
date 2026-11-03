# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import contextlib

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq, assert_neq


@pytest.mark.parametrize(
    "method",
    [
        "murmur3",
        "md5",
        "sha1",
        "sha224",
        "sha256",
        "sha384",
        "sha512",
        "xxhash32",
        "xxhash64",
    ],
)
@pytest.mark.parametrize("seed", [None, 42])
def test_dataframe_hash_values(method, seed):
    nrows = 10
    warning_expected = seed is not None and method not in {
        "murmur3",
        "xxhash32",
        "xxhash64",
    }
    potential_warning = (
        pytest.warns(UserWarning, match="Provided seed value has no effect*")
        if warning_expected
        else contextlib.nullcontext()
    )

    gdf = cudf.DataFrame()
    data = np.arange(nrows)
    data[0] = data[-1]  # make first and last the same
    gdf["a"] = data
    gdf["b"] = gdf.a + 100
    with potential_warning:
        out = gdf.hash_values(method=method, seed=seed)
    assert isinstance(out, cudf.Series)
    assert len(out) == nrows
    string_dtype = pd.StringDtype(na_value=np.nan)
    expected_dtypes = {
        "murmur3": np.uint32,
        "md5": string_dtype,
        "sha1": string_dtype,
        "sha224": string_dtype,
        "sha256": string_dtype,
        "sha384": string_dtype,
        "sha512": string_dtype,
        "xxhash32": np.uint32,
        "xxhash64": np.uint64,
    }
    assert out.dtype == expected_dtypes[method]

    # Check single column
    with potential_warning:
        out_one = gdf[["a"]].hash_values(method=method, seed=seed)
    # First matches last
    assert out_one.iloc[0] == out_one.iloc[-1]
    # Equivalent to the cudf.Series.hash_values()
    with potential_warning:
        assert_eq(gdf["a"].hash_values(method=method, seed=seed), out_one)


@pytest.mark.parametrize("method", ["murmur3", "xxhash32", "xxhash64"])
def test_dataframe_hash_values_seed(method):
    gdf = cudf.DataFrame()
    data = np.arange(10)
    data[0] = data[-1]  # make first and last the same
    gdf["a"] = data
    gdf["b"] = gdf.a + 100
    out_one = gdf.hash_values(method=method, seed=0)
    out_two = gdf.hash_values(method=method, seed=1)
    assert out_one.iloc[0] == out_one.iloc[-1]
    assert out_two.iloc[0] == out_two.iloc[-1]
    assert_neq(out_one, out_two)


def test_dataframe_hash_values_xxhash32():
    # xxhash32 has no built-in implementation in Python and we don't want to
    # add a testing dependency, so we use regression tests against known good
    # values.
    gdf = cudf.DataFrame({"a": [0.0, 1.0, 2.0, np.inf, np.nan]})
    gdf["b"] = -gdf["a"]
    out_a = gdf["a"].hash_values(method="xxhash32", seed=0)
    expected_a = cudf.Series(
        [3736311059, 2307980487, 2906647130, 746578903, 4294967295],
        dtype=np.uint32,
    )
    assert_eq(out_a, expected_a)

    out_b = gdf["b"].hash_values(method="xxhash32", seed=42)
    expected_b = cudf.Series(
        [1076387279, 2261349915, 531498073, 650869264, 4294967295],
        dtype=np.uint32,
    )
    assert_eq(out_b, expected_b)

    out_df = gdf.hash_values(method="xxhash32", seed=0)
    expected_df = cudf.Series(
        [3217993155, 2243235024, 3783294776, 940382429, 566789768],
        dtype=np.uint32,
    )
    assert_eq(out_df, expected_df)


def test_dataframe_hash_values_xxhash64():
    # xxhash64 has no built-in implementation in Python and we don't want to
    # add a testing dependency, so we use regression tests against known good
    # values.
    gdf = cudf.DataFrame({"a": [0.0, 1.0, 2.0, np.inf, np.nan]})
    gdf["b"] = -gdf["a"]
    out_a = gdf["a"].hash_values(method="xxhash64", seed=0)
    expected_a = cudf.Series(
        [
            3803688792395291579,
            10706502109028787093,
            9835943264235290955,
            18031741628920313605,
            18446744073709551615,
        ],
        dtype=np.uint64,
    )
    assert_eq(out_a, expected_a)

    out_b = gdf["b"].hash_values(method="xxhash64", seed=42)
    expected_b = cudf.Series(
        [
            9826995235083043316,
            10150515573749944095,
            5005707091092326006,
            5326262080505358431,
            18446744073709551615,
        ],
        dtype=np.uint64,
    )
    assert_eq(out_b, expected_b)

    out_df = gdf.hash_values(method="xxhash64", seed=0)
    expected_df = cudf.Series(
        [
            3012780887911066904,
            2108190815019463645,
            1057888907238192171,
            7243162636073134573,
            2434343235958965292,
        ],
        dtype=np.uint64,
    )
    assert_eq(out_df, expected_df)

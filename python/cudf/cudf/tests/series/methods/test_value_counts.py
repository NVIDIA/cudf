# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from string import ascii_letters, digits

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core.column.column import as_column
from cudf.testing import assert_eq


@pytest.fixture(params=[True, False])
def normalize(request):
    """Argument for value_counts"""
    return request.param


@pytest.mark.parametrize(
    "data",
    [
        [],
        pd.date_range("2010-01-01", "2010-02-01"),
        [None, None],
        [pd.Timestamp(2020, 1, 1), pd.NaT],
    ],
)
def test_series_datetime_value_counts(data, normalize, dropna):
    psr = pd.Series(data, dtype="datetime64[ns]")
    gsr = cudf.from_pandas(psr)
    expected = psr.value_counts(dropna=dropna, normalize=normalize)
    got = gsr.value_counts(dropna=dropna, normalize=normalize)

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=False)
    assert_eq(
        expected.reset_index(drop=True),
        got.reset_index(drop=True),
        check_dtype=False,
        check_index_type=True,
    )


def test_categorical_value_counts(dropna, normalize):
    num_elements = 20
    rng = np.random.default_rng(seed=12)
    pd_cat = pd.Categorical(
        pd.Series(
            rng.choice(list(ascii_letters + digits), num_elements),
            dtype="category",
        )
    )

    # gdf
    gdf_value_counts = cudf.Series(pd_cat).value_counts(
        dropna=dropna, normalize=normalize
    )

    # pandas
    pdf_value_counts = pd.Series(pd_cat).value_counts(
        dropna=dropna, normalize=normalize
    )

    # verify
    assert_eq(
        pdf_value_counts.sort_index(),
        gdf_value_counts.sort_index(),
        check_dtype=False,
        check_index_type=True,
    )
    assert_eq(
        pdf_value_counts.reset_index(drop=True),
        gdf_value_counts.reset_index(drop=True),
        check_dtype=False,
        check_index_type=True,
    )


def test_series_value_counts(dropna, normalize):
    rng = np.random.default_rng(seed=0)
    size = 10
    arr = rng.integers(low=-1, high=10, size=size)
    mask = arr != -1
    mask_buff, null_count = cudf.Series(mask)._column.as_mask()
    sr = cudf.Series._from_column(
        as_column(arr).set_mask(mask_buff, null_count)
    )
    sr.name = "col"

    expect = (
        sr.to_pandas()
        .value_counts(dropna=dropna, normalize=normalize)
        .sort_index()
    )
    got = sr.value_counts(dropna=dropna, normalize=normalize).sort_index()

    assert_eq(expect, got, check_dtype=True, check_index_type=False)


@pytest.mark.parametrize("bins", [1, 3])
def test_series_value_counts_bins(bins):
    psr = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0])
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(bins=bins)
    got = gsr.value_counts(bins=bins)

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=True)


@pytest.mark.parametrize("bins", [1, 3])
def test_series_value_counts_bins_dropna(bins, dropna):
    psr = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, np.nan])
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(bins=bins, dropna=dropna)
    got = gsr.value_counts(bins=bins, dropna=dropna)

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=True)


def test_series_value_counts_optional_arguments(ascending, dropna, normalize):
    psr = pd.Series([1.0, 2.0, 2.0, 3.0, 3.0, 3.0, None])
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(
        ascending=ascending, dropna=dropna, normalize=normalize
    )
    got = gsr.value_counts(
        ascending=ascending, dropna=dropna, normalize=normalize
    )

    assert_eq(expected.sort_index(), got.sort_index(), check_dtype=True)
    assert_eq(
        expected.reset_index(drop=True),
        got.reset_index(drop=True),
        check_dtype=True,
    )


@pytest.mark.parametrize(
    "categorical",
    [
        # No nulls
        pd.Categorical(list("abcccb"), categories=list("cabd")),
        # With nulls (used to drop the NaN group when ``dropna=False``)
        pd.Categorical(
            [*list("aaaaa"), None, *list("bbbcc")], categories=list("abc")
        ),
        # With nulls + ordered (used to lose the ``ordered`` flag too)
        pd.Categorical(
            [*list("aaaa"), None, *list("bbbcc")],
            categories=list("bac"),
            ordered=True,
        ),
    ],
    ids=["no-nulls", "with-nulls", "with-nulls-ordered"],
)
def test_series_categorical_missing_value_count(categorical, dropna):
    ps = pd.Series(categorical)
    gs = cudf.from_pandas(ps)

    expected = ps.value_counts(dropna=dropna)
    actual = gs.value_counts(dropna=dropna)

    assert_eq(expected, actual, check_dtype=False)


def test_numeric_alpha_value_counts():
    pdf = pd.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5, 6, 1, 2, 4] * 10,
            "alpha": ["u", "h", "d", "a", "m", "u", "h", "d", "a"] * 10,
        }
    )

    gdf = cudf.DataFrame(
        {
            "numeric": [1, 2, 3, 4, 5, 6, 1, 2, 4] * 10,
            "alpha": ["u", "h", "d", "a", "m", "u", "h", "d", "a"] * 10,
        }
    )

    assert_eq(
        pdf.numeric.value_counts().sort_index(),
        gdf.numeric.value_counts().sort_index(),
        check_dtype=False,
    )
    assert_eq(
        pdf.alpha.value_counts().sort_index(),
        gdf.alpha.value_counts().sort_index(),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "dtype",
    [
        "Float32",
        "Float64",
        "Int8",
        "Int16",
        "Int32",
        "Int64",
        "UInt8",
        "UInt16",
        "UInt32",
        "UInt64",
        "boolean",
    ],
)
def test_value_counts_empty_pandas_nullable(dtype, normalize, dropna):
    psr = pd.Series([], dtype=dtype)
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(dropna=dropna, normalize=normalize)
    got = gsr.value_counts(dropna=dropna, normalize=normalize)

    assert_eq(expected, got, check_dtype=True, check_index_type=True)


@pytest.mark.parametrize(
    "dtype",
    ["Float64", "Int64", "UInt32", "boolean"],
)
def test_value_counts_all_null_pandas_nullable(dtype, normalize, dropna):
    psr = pd.Series([pd.NA, pd.NA, pd.NA], dtype=dtype)
    gsr = cudf.from_pandas(psr)

    expected = psr.value_counts(dropna=dropna, normalize=normalize)
    got = gsr.value_counts(dropna=dropna, normalize=normalize)

    assert_eq(
        expected.sort_index(),
        got.sort_index(),
        check_dtype=True,
        check_index_type=True,
    )


def test_value_counts_first_appearance_order(sort):
    # pandas returns groups in order of first appearance with sort=False
    # and keeps that order for equal counts with sort=True (GH 63155).
    arr = np.random.default_rng(2).integers(0, 32, 64)
    psr = pd.Series(arr)
    gsr = cudf.Series(arr)
    assert_eq(
        psr.value_counts(sort=sort),
        gsr.value_counts(sort=sort),
    )

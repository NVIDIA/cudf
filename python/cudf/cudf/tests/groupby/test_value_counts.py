# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pytest

import cudf
from cudf.testing import assert_groupby_results_equal


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("sort", [True, False])
def test_group_by_value_counts(normalize, sort, ascending, dropna, as_index):
    # From Issue#12789
    df = cudf.DataFrame(
        {
            "gender": ["male", "male", "female", "male", "female", "male"],
            "education": ["low", "medium", np.nan, "low", "high", "low"],
            "country": ["US", "FR", "US", "FR", "FR", "FR"],
        }
    )
    pdf = df.to_pandas()

    actual = df.groupby("gender", as_index=as_index).value_counts(
        normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
    )
    expected = pdf.groupby("gender", as_index=as_index).value_counts(
        normalize=normalize, sort=sort, ascending=ascending, dropna=dropna
    )
    # Always sort for comparison since row order is
    # implementation-defined when value_counts(sort=False).
    assert_groupby_results_equal(
        actual,
        expected,
        as_index=as_index,
        by=["gender", "education", "country"],
    )


def test_group_by_value_counts_subset():
    # From Issue#12789
    df = cudf.DataFrame(
        {
            "gender": ["male", "male", "female", "male", "female", "male"],
            "education": ["low", "medium", "high", "low", "high", "low"],
            "country": ["US", "FR", "US", "FR", "FR", "FR"],
        }
    )
    pdf = df.to_pandas()

    actual = df.groupby("gender").value_counts(["education"])
    expected = pdf.groupby("gender").value_counts(["education"])

    # TODO: Remove `check_names=False` once testing against `pandas>=2.0.0`
    assert_groupby_results_equal(
        actual, expected, check_names=False, check_index_type=False
    )


def test_group_by_value_counts_clash_with_subset():
    df = cudf.DataFrame({"a": [1, 5, 3], "b": [2, 5, 2]})
    with pytest.raises(ValueError):
        df.groupby("a").value_counts(["a"])


def test_group_by_value_counts_subset_not_exists():
    df = cudf.DataFrame({"a": [1, 5, 3], "b": [2, 5, 2]})
    with pytest.raises(ValueError):
        df.groupby("a").value_counts(["c"])


def test_group_by_value_counts_with_count_column():
    df = cudf.DataFrame({"a": [1, 5, 3], "count": [2, 5, 2]})
    with pytest.raises(ValueError):
        df.groupby("a", as_index=False).value_counts()

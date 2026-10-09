# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize("axis", [None, 0, "index", 1, "columns"])
def test_dataframe_add_prefix(axis):
    cdf = cudf.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
    pdf = cdf.to_pandas()

    got = cdf.add_prefix("item_", axis=axis)
    expected = pdf.add_prefix("item_", axis=axis)

    assert_eq(got, expected, check_index_type=axis not in (0, "index"))


@pytest.mark.parametrize("axis", [None, 0, "index", 1, "columns"])
def test_dataframe_add_suffix(axis):
    cdf = cudf.DataFrame({"A": [1, 2, 3, 4], "B": [3, 4, 5, 6]})
    pdf = cdf.to_pandas()

    got = cdf.add_suffix("_item", axis=axis)
    expected = pdf.add_suffix("_item", axis=axis)

    assert_eq(got, expected, check_index_type=axis not in (0, "index"))


@pytest.mark.parametrize(
    ("method", "affix"),
    [("add_prefix", "item_"), ("add_suffix", "_item")],
)
def test_dataframe_add_prefix_suffix_multiindex(method, affix):
    cdf = cudf.DataFrame(
        {"A": [1, 2, 3]},
        index=cudf.MultiIndex.from_arrays(
            [[1, 1, 2], ["a", "b", "c"]],
            names=[1, "letter"],
        ),
    )
    pdf = cdf.to_pandas()

    got = getattr(cdf, method)(affix, axis="index")
    expected = getattr(pdf, method)(affix, axis="index")

    assert_eq(got, expected, check_index_type=False)


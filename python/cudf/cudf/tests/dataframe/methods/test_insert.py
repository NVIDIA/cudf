# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_insert_reset_label_dtype():
    result = cudf.DataFrame({1: [2]})
    expected = pd.DataFrame({1: [2]})
    result.insert(1, "a", [2])
    expected.insert(1, "a", [2])
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [5.0, 6.0, 7.0],
        "single value",
        np.array(1, dtype="int64"),
        np.array(0.6273643, dtype="float64"),
    ],
)
def test_insert(data):
    pdf = pd.DataFrame.from_dict({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    gdf = cudf.DataFrame(pdf)

    # insertion by index

    pdf.insert(0, "foo", data)
    gdf.insert(0, "foo", data)

    assert_eq(pdf, gdf)

    pdf.insert(3, "bar", data)
    gdf.insert(3, "bar", data)

    assert_eq(pdf, gdf)

    pdf.insert(1, "baz", data)
    gdf.insert(1, "baz", data)

    assert_eq(pdf, gdf)

    # pandas insert doesn't support negative indexing
    pdf.insert(len(pdf.columns), "qux", data)
    gdf.insert(-1, "qux", data)

    assert_eq(pdf, gdf)


def test_insert_NA():
    pdf = pd.DataFrame.from_dict({"A": [1, 2, 3], "B": ["a", "b", "c"]})
    gdf = cudf.DataFrame(pdf)

    pdf["C"] = pd.NA
    gdf["C"] = cudf.NA
    # As of pandas 3.0, "C" becomes an object column broadcasting NA,
    # cuDF doesn't support object type.
    pdf["C"] = pdf["C"].astype(gdf["C"].dtype)
    assert_eq(pdf, gdf)

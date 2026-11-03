# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_groupby_results_equal
from cudf.testing.dataset_generator import rand_dataframe


@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
def test_groupby_diff_row(shift_perc, direction):
    nelem = 20
    pdf = pd.DataFrame(np.ones((nelem, 4)), columns=["x", "y", "val", "val2"])
    gdf = cudf.from_pandas(pdf)
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["x", "y"]).diff(periods=n_shift)
    got = gdf.groupby(["x", "y"]).diff(periods=n_shift)

    assert_groupby_results_equal(expected, got)


@pytest.mark.parametrize("shift_perc", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("direction", [1, -1])
def test_groupby_diff_row_mixed_numerics(shift_perc, direction):
    nelem = 20
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "decimal64", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    gdf = cudf.DataFrame.from_arrow(t)
    pdf = gdf.to_pandas(arrow_type=True)
    n_shift = int(nelem * shift_perc) * direction

    expected = pdf.groupby(["0"]).diff(periods=n_shift)
    got = gdf.groupby(["0"]).diff(periods=n_shift)

    assert_groupby_results_equal(
        expected, got.to_pandas(arrow_type=True), check_dtype=False
    )


def test_groupby_diff_row_zero_shift():
    nelem = 20
    t = rand_dataframe(
        dtypes_meta=[
            {"dtype": "int64", "null_frequency": 0, "cardinality": 10},
            {"dtype": "int64", "null_frequency": 0.4, "cardinality": 10},
            {"dtype": "float32", "null_frequency": 0.4, "cardinality": 10},
            {
                "dtype": "datetime64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
            {
                "dtype": "timedelta64[ns]",
                "null_frequency": 0.4,
                "cardinality": 10,
            },
        ],
        rows=nelem,
        use_threads=False,
        seed=0,
    )
    gdf = cudf.from_pandas(t.to_pandas())

    expected = gdf
    got = gdf.groupby(["0"]).shift(periods=0)

    assert_groupby_results_equal(
        expected[["1", "2", "3", "4"]], got[["1", "2", "3", "4"]]
    )


def test_groupby_select_then_diff():
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 2], "b": [1, 2, 3, 4, 5], "c": [3, 4, 5, 6, 7]}
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("a")["c"].diff(1)
    actual = gdf.groupby("a")["c"].diff(1)

    assert_groupby_results_equal(expected, actual)

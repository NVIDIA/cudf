# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq, assert_groupby_results_equal


def test_rank_return_type_compatible_mode():
    # in compatible mode, rank() always returns floats
    df = cudf.DataFrame({"a": range(10), "b": [0] * 10}, index=[0] * 10)
    pdf = df.to_pandas()
    expect = pdf.groupby("b").get_group(0)
    result = df.groupby("b").get_group(0)
    assert_eq(expect, result)


@pytest.mark.parametrize(
    "group, name",
    [
        (
            "X",
            "A",
        ),
        (
            "X",
            "B",
        ),
        (
            "Y",
            1,
        ),
        (
            "Y",
            3,
        ),
    ],
)
def test_groupby_get_group(group, name):
    pdf = pd.DataFrame({"X": ["A", "B", "A", "B"], "Y": [1, 4, 3, 2]})
    gdf = cudf.from_pandas(pdf)
    pgb = pdf.groupby(group)
    ggb = gdf.groupby(group)
    expected = pgb.get_group(name=name)
    actual = ggb.get_group(name=name)

    assert_groupby_results_equal(expected, actual)

    expected = pdf.iloc[pgb.indices.get(name)]
    actual = gdf.iloc[ggb.indices.get(name)]

    assert_eq(expected, actual)


def test_get_group_list_like():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    result = df.groupby(["a"]).get_group((1,))
    expected = df.to_pandas().groupby(["a"]).get_group((1,))
    assert_eq(result, expected)

    with pytest.raises(KeyError):
        df.groupby(["a"]).get_group((1, 2))

    with pytest.raises(KeyError):
        df.groupby(["a"]).get_group([1])


def test_get_group_list_like_len_2():
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [3, 2, 1]})
    result = df.groupby(["a", "b"]).get_group((1, 4))
    expected = df.to_pandas().groupby(["a", "b"]).get_group((1, 4))
    assert_eq(result, expected)

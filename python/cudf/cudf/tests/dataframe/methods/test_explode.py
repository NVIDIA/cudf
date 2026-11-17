# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


def test_explode_preserve_categorical():
    gdf = cudf.DataFrame(
        {
            "A": [[1, 2], None, [2, 3]],
            "B": cudf.Series([0, 1, 2], dtype="category"),
        }
    )
    result = gdf.explode("A")
    expected = cudf.DataFrame(
        {
            "A": [1, 2, None, 2, 3],
            "B": cudf.Series([0, 0, 1, 2, 2], dtype="category"),
        }
    )
    expected.index = cudf.Index([0, 0, 1, 2, 2])
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        [
            [[1, 2, 3], 11, "a"],
            [None, 22, "e"],
            [[4], 33, "i"],
            [[], 44, "o"],
            [[5, 6], 55, "u"],
        ],  # nested
        [
            [1, 11, "a"],
            [2, 22, "e"],
            [3, 33, "i"],
            [4, 44, "o"],
            [5, 55, "u"],
        ],  # non-nested
    ],
)
@pytest.mark.parametrize(
    ("labels", "label_to_explode"),
    [
        (None, 0),
        (pd.Index(["a", "b", "c"]), "a"),
        (
            pd.MultiIndex.from_tuples(
                [(0, "a"), (0, "b"), (1, "a")], names=["l0", "l1"]
            ),
            (0, "a"),
        ),
    ],
)
@pytest.mark.parametrize(
    "p_index",
    [
        None,
        ["ia", "ib", "ic", "id", "ie"],
        pd.MultiIndex.from_tuples(
            [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b")]
        ),
    ],
)
def test_explode(data, labels, ignore_index, p_index, label_to_explode):
    gdf = cudf.DataFrame(data, index=p_index, columns=labels)
    pdf = gdf.to_pandas(arrow_type=True)
    gdf = cudf.from_pandas(pdf)

    expect = pdf.explode(label_to_explode, ignore_index)
    got = gdf.explode(label_to_explode, ignore_index)

    assert_eq(expect, got)

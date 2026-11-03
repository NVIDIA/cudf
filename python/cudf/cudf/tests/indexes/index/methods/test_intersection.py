# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "idx1, idx2",
    [
        (pd.RangeIndex(0, 10), pd.RangeIndex(3, 7)),
        (pd.RangeIndex(0, 10), pd.RangeIndex(-10, 20)),
        (pd.RangeIndex(0, 10, name="a"), pd.RangeIndex(90, 100, name="b")),
        (pd.Index([0, 1, 2, 30], name=pd.NA), pd.Index([30, 0, 90, 100])),
        (pd.Index([0, 1, 2, 30], name="a"), [90, 100]),
        (pd.Index([0, 1, 2, 30]), pd.Index([0, 10, 1.0, 11])),
        (
            pd.Index(["a", "b", "c", "d", "c"]),
            pd.Index(["a", "c", "z"], name="abc"),
        ),
        (
            pd.Index(["a", "b", "c", "d", "c"]),
            pd.Index(["a", "b", "c", "d", "c"]),
        ),
        (pd.Index([True, False, True, True]), pd.Index([10, 11, 12, 0, 1, 2])),
        (pd.Index([True, False, True, True]), pd.Index([True, True])),
        (pd.RangeIndex(0, 10, name="a"), pd.Index([5, 6, 7], name="b")),
        (pd.Index(["a", "b", "c"], dtype="category"), pd.Index(["a", "b"])),
        (pd.Index([0, 1, 2], dtype="category"), pd.RangeIndex(0, 10)),
        (pd.Index(["a", "b", "c"], name="abc"), []),
        (pd.Index([], name="abc"), pd.RangeIndex(0, 4)),
        (pd.Index([1, 2, 3]), pd.Index([1, 2], dtype="category")),
        (pd.Index([]), pd.Index([1, 2], dtype="category")),
    ],
)
@pytest.mark.parametrize("sort", [None, False, True])
@pytest.mark.parametrize("pandas_compatible", [True, False])
def test_intersection_index(idx1, idx2, sort, pandas_compatible):
    expected = idx1.intersection(idx2, sort=sort)

    with cudf.option_context("mode.pandas_compatible", pandas_compatible):
        idx1 = cudf.from_pandas(idx1) if isinstance(idx1, pd.Index) else idx1
        idx2 = cudf.from_pandas(idx2) if isinstance(idx2, pd.Index) else idx2

        actual = idx1.intersection(idx2, sort=sort)

        # TODO: Resolve the bool vs ints mixed issue
        # once pandas has a direction on this issue
        # https://github.com/pandas-dev/pandas/issues/44000
        assert_eq(
            expected,
            actual,
            exact=False
            if (idx1.dtype.kind == "b" and idx2.dtype.kind != "b")
            or (idx1.dtype.kind != "b" or idx2.dtype.kind == "b")
            else True,
        )


@pytest.mark.parametrize("sort", [None, False, True])
@pytest.mark.parametrize("pandas_compatible", [True, False])
def test_intersection_index_mixed_string_categorical_integer(
    sort, pandas_compatible
):
    idx1 = pd.Index(["a", "b", "c"], dtype="category")
    idx2 = pd.Index([1, 2, 3])
    expected = idx1.intersection(idx2, sort=sort)

    with cudf.option_context("mode.pandas_compatible", pandas_compatible):
        idx1 = cudf.from_pandas(idx1) if isinstance(idx1, pd.Index) else idx1
        idx2 = cudf.from_pandas(idx2) if isinstance(idx2, pd.Index) else idx2
        actual = idx1.intersection(idx2, sort=sort)
    # As of pandas 3.0, empty default type of object isn't
    # necessarily equivalent to cuDF's empty default type of
    # pandas.StringDtype
    assert_eq(expected.astype(actual.dtype), actual)

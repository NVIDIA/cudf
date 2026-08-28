# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test related to Cut
"""

import numpy as np
import pandas as pd
import pytest

from cudf.core.cut import cut
from cudf.testing import assert_eq


@pytest.fixture(params=[True, False])
def right(request):
    return request.param


@pytest.fixture(params=[True, False])
def include_lowest(request):
    return request.param


@pytest.fixture(params=[True, False])
def ordered(request):
    return request.param


@pytest.mark.parametrize("bins", [1, 2, 3])
@pytest.mark.parametrize("precision", [1, 3])
def test_cut_basic(
    one_dimensional_array_types, bins, right, include_lowest, precision
):
    # will test optional labels, retbins and duplicates separately
    # they need more specific parameters to work
    x = one_dimensional_array_types([1, 7, 5, 4, 6, 3])
    ordered = True
    pcat = pd.cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )
    pindex = pd.CategoricalIndex(pcat)
    gindex = cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("precision", [1, 3])
@pytest.mark.parametrize(
    "labels", [["bad", "medium", "good"], [1, 2, 3], False]
)
def test_cut_labels(right, include_lowest, ordered, precision, labels):
    x = [1, 7, 5, 4, 6, 3]
    bins = 3
    pcat = pd.cut(
        x=x,
        bins=bins,
        right=right,
        labels=labels,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )
    pindex = pd.CategoricalIndex(pcat) if labels else pcat
    gindex = cut(
        x=x,
        bins=bins,
        right=right,
        labels=labels,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("precision", [1, 3])
@pytest.mark.parametrize("labels", [["bad", "good", "good"], [1, 2, 2], False])
def test_cut_labels_non_unique(right, include_lowest, precision, labels):
    x = [1, 7, 5, 4, 6, 3]
    bins = 3
    ordered = False
    pcat = pd.cut(
        x=x,
        bins=bins,
        right=right,
        labels=labels,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )
    pindex = pd.CategoricalIndex(pcat) if labels else pcat
    gindex = cut(
        x=x,
        bins=bins,
        right=right,
        labels=labels,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize(
    "x",
    [
        [1, 7, 5],
        [2, 4, 6, 8, 10],
    ],
)
@pytest.mark.parametrize(
    "bins",
    [1, 2, 3, [1, 2, 3], [0, 2, 4, 6, 10]],
)
def test_cut_right(x, one_dimensional_array_types, bins, right):
    arg = one_dimensional_array_types(x)
    precision = 3
    pcat = pd.cut(
        x=arg,
        bins=bins,
        right=right,
        precision=precision,
    )
    pindex = pd.CategoricalIndex(pcat)
    gindex = cut(
        x=arg,
        bins=bins,
        right=right,
        precision=precision,
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize(
    "x",
    [
        [1, 7, 5],
        [2, 4, 6, 8, 10],
    ],
)
@pytest.mark.parametrize(
    "bins",
    [[0, 2, 4, 6, 10, 10], [1, 2, 2, 3, 3]],
)
@pytest.mark.parametrize("precision", [1, 3])
def test_cut_drop_duplicates(
    x, one_dimensional_array_types, bins, right, precision, include_lowest
):
    ordered = True
    duplicates = "drop"
    arg = one_dimensional_array_types(x)
    pcat = pd.cut(
        x=arg,
        bins=bins,
        right=right,
        precision=precision,
        duplicates=duplicates,
        include_lowest=include_lowest,
        ordered=ordered,
    )
    pindex = pd.CategoricalIndex(pcat)
    gindex = cut(
        x=arg,
        bins=bins,
        right=right,
        precision=precision,
        duplicates=duplicates,
        include_lowest=include_lowest,
        ordered=ordered,
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize(
    "x",
    [
        [1, 7, 5],
        [2, 4, 6, 8, 10],
    ],
)
@pytest.mark.parametrize(
    "bins",
    [[0, 2, 4, 6, 10, 10], [1, 2, 2, 3, 3]],
)
@pytest.mark.parametrize("precision", [1, 3])
def test_cut_drop_duplicates_raises(
    x, one_dimensional_array_types, bins, right, precision, include_lowest
):
    arg = one_dimensional_array_types(x)
    ordered = True
    duplicates = "raise"
    msg = "Bin edges must be unique"
    with pytest.raises(ValueError, match=msg):
        cut(
            x=arg,
            bins=bins,
            right=right,
            precision=precision,
            duplicates=duplicates,
            include_lowest=include_lowest,
            ordered=ordered,
        )
    with pytest.raises(ValueError, match=msg):
        pd.cut(
            x=arg,
            bins=bins,
            right=right,
            precision=precision,
            duplicates=duplicates,
            include_lowest=include_lowest,
            ordered=ordered,
        )


def test_cut_unique_bins_larger_than_integer_cache():
    x = [1, 2]
    bins = list(range(257))

    expected = pd.cut(x=x, bins=bins, labels=False)
    actual = cut(x=x, bins=bins, labels=False)

    assert_eq(expected, actual)


def test_cut_duplicate_bins_with_distinct_large_integers():
    x = [1, 2]
    # Construct equal bin edges independently so they are distinct objects.
    duplicate_edge = int("257")
    bins = [0, duplicate_edge, int("257"), 258]

    assert bins[1] == bins[2]
    assert bins[1] is not bins[2]

    with pytest.raises(ValueError, match="Bin edges must be unique"):
        cut(x=x, bins=bins, labels=False)


@pytest.mark.parametrize(
    "x",
    [
        [1, 7, 5],
        [2, 4, 6, 8, 10],
    ],
)
@pytest.mark.parametrize("precision", [1, 3])
@pytest.mark.parametrize("duplicates", ["drop", "raise"])
def test_cut_intervalindex_bin(
    x, one_dimensional_array_types, right, precision, duplicates
):
    arg = one_dimensional_array_types(x)
    bins = pd.IntervalIndex.from_tuples([(0, 1), (2, 3), (4, 5)])
    pcat = pd.cut(
        x=arg,
        bins=bins,
        right=right,
        precision=precision,
        duplicates=duplicates,
    )
    pindex = pd.CategoricalIndex(pcat)
    gindex = cut(
        x=arg,
        bins=bins,
        right=right,
        precision=precision,
        duplicates=duplicates,
    )

    assert_eq(pindex, gindex)


@pytest.mark.parametrize("bins", [1, 3])
def test_cut_series(bins, right, include_lowest):
    x = pd.Series(np.array([2, 4, 6, 8, 10]), index=["a", "b", "c", "d", "e"])
    precision = 3
    ordered = True
    pcat = pd.cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )

    gcat = cut(
        x=x,
        bins=bins,
        right=right,
        precision=precision,
        include_lowest=include_lowest,
        ordered=ordered,
    )

    assert_eq(pcat, gcat)

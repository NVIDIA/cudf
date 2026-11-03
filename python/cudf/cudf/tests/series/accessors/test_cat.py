# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from textwrap import dedent

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def test_categorical_basic():
    cat = pd.Categorical(["a", "a", "b", "c", "a"], categories=["a", "b", "c"])
    cudf_cat = cudf.Index(cat)
    assert_eq(cat.codes, cudf_cat.codes.to_numpy())

    pdsr = pd.Series(cat, index=["p", "q", "r", "s", "t"])
    sr = cudf.Series(cat, index=["p", "q", "r", "s", "t"])
    assert_eq(pdsr.cat.codes, sr.cat.codes, check_dtype=False)

    # Test attributes
    assert_eq(pdsr.cat.categories, sr.cat.categories)
    assert pdsr.cat.ordered == sr.cat.ordered

    np.testing.assert_array_equal(
        pdsr.cat.codes.values, sr.cat.codes.to_numpy()
    )

    assert str(sr) == str(pdsr)


@pytest.mark.parametrize("num_categories", [1, 3, 200, 40000])
def test_cat_codes_signed_dtype(num_categories):
    # cat.codes must use a signed integer dtype matching pandas (which widens
    # int8 -> int16 -> int32 as the category count grows), not an unsigned one.
    categories = [f"c{i}" for i in range(num_categories)]
    ps = pd.Series(pd.Categorical(categories, categories=categories))
    gs = cudf.from_pandas(ps)

    expect = ps.cat.codes
    got = gs.cat.codes

    assert got.dtype.kind == "i"
    assert got.dtype == expect.dtype
    assert_eq(expect, got)


def test_categorical_integer():
    cat = pd.Categorical.from_codes(
        [0, -1, -1, 2, 0], categories=["a", "b", "c"]
    )
    pdsr = pd.Series(cat)
    sr = cudf.Series(cat)
    np.testing.assert_array_equal(
        cat.codes, sr.cat.codes.astype(cat.codes.dtype).fillna(-1).to_numpy()
    )
    assert sr.null_count == 2

    np.testing.assert_array_equal(
        pdsr.cat.codes.values,
        sr.cat.codes.astype(pdsr.cat.codes.dtype).fillna(-1).to_numpy(),
    )

    expect_str = dedent(
        """\
        0       a
        1    <NA>
        2    <NA>
        3       c
        4       a
        dtype: category
        Categories (3, str): ['a', 'b', 'c']"""
    )
    assert str(sr) == expect_str


def test_categorical_element_indexing():
    """
    Element indexing to a cat column must give the underlying object
    not the numerical index.
    """
    cat = pd.Categorical(["a", "a", "b", "c", "a"], categories=["a", "b", "c"])
    pdsr = pd.Series(cat)
    sr = cudf.Series(cat)
    assert_eq(pdsr, sr)
    assert_eq(pdsr.cat.codes, sr.cat.codes, check_dtype=False)


def test_categorical_empty():
    cat = pd.Categorical([])
    pdsr = pd.Series(cat)
    sr = cudf.Series(cat)
    np.testing.assert_array_equal(cat.codes, sr.cat.codes.to_numpy())

    # Test attributes
    assert sr.cat.categories.dtype == np.dtype(object)
    assert pdsr.cat.categories.dtype == np.dtype(object)
    assert_eq(
        pdsr.cat.categories.astype(sr.cat.categories.dtype), sr.cat.categories
    )
    assert pdsr.cat.ordered == sr.cat.ordered

    np.testing.assert_array_equal(
        pdsr.cat.codes.values, sr.cat.codes.to_numpy()
    )


def test_categorical_set_categories():
    cat = pd.Categorical(["a", "a", "b", "c", "a"], categories=["a", "b", "c"])
    psr = pd.Series(cat)
    sr = cudf.Series(cat)

    # adding category
    expect = psr.cat.set_categories(["a", "b", "c", "d"])
    got = sr.cat.set_categories(["a", "b", "c", "d"])
    assert_eq(expect, got)

    # removing category
    expect = psr.cat.set_categories(["a", "b"])
    got = sr.cat.set_categories(["a", "b"])
    assert_eq(expect, got)


def test_categorical_set_categories_preserves_order():
    series = pd.Series([1, 0, 0, 0, 2]).astype("category")
    # reassigning categories should preserve element ordering
    assert_eq(
        series.cat.set_categories([1, 2]),
        cudf.Series(series).cat.set_categories([1, 2]),
    )


def test_categorical_as_ordered():
    categories = list("abc")
    codes = [0, 0, 1, 0, 1, 2, 0, 1, 1, 2]
    pd_str_cat = pd.Categorical.from_codes(codes, categories=categories)
    pd_sr = pd.Series(pd_str_cat.set_ordered(False))
    cd_sr = cudf.Series(pd_str_cat.set_ordered(False))

    assert cd_sr.cat.ordered is False
    assert cd_sr.cat.ordered == pd_sr.cat.ordered

    pd_sr_1 = pd_sr.cat.as_ordered()
    cd_sr_1 = cd_sr.cat.as_ordered()

    assert cd_sr_1.cat.ordered is True
    assert cd_sr_1.cat.ordered == pd_sr_1.cat.ordered
    assert str(cd_sr_1) == str(pd_sr_1)


def test_categorical_as_unordered():
    categories = list("abc")
    codes = [0, 0, 1, 0, 1, 2, 0, 1, 1, 2]
    pd_str_cat = pd.Categorical.from_codes(codes, categories=categories)
    pd_sr = pd.Series(pd_str_cat.set_ordered(True))
    cd_sr = cudf.Series(pd_str_cat.set_ordered(True))

    assert cd_sr.cat.ordered is True
    assert cd_sr.cat.ordered == pd_sr.cat.ordered

    pd_sr_1 = pd_sr.cat.as_unordered()
    cd_sr_1 = cd_sr.cat.as_unordered()

    assert cd_sr_1.cat.ordered is False
    assert cd_sr_1.cat.ordered == pd_sr_1.cat.ordered
    assert str(cd_sr_1) == str(pd_sr_1)


@pytest.mark.parametrize("from_ordered", [True, False])
@pytest.mark.parametrize("to_ordered", [True, False])
def test_categorical_reorder_categories(from_ordered, to_ordered):
    categories = list("abc")
    codes = [0, 0, 1, 0, 1, 2, 0, 1, 1, 2]
    pd_str_cat = pd.Categorical.from_codes(codes, categories=categories)
    pd_sr = pd.Series(pd_str_cat.set_ordered(from_ordered))
    cd_sr = cudf.Series(pd_str_cat.set_ordered(from_ordered))

    assert_eq(pd_sr, cd_sr)

    assert str(pd_sr) == str(cd_sr)

    pd_sr_1 = pd_sr.cat.reorder_categories(list("cba"), ordered=to_ordered)
    cd_sr_1 = cd_sr.cat.reorder_categories(list("cba"), ordered=to_ordered)

    assert_eq(pd_sr_1, cd_sr_1)

    assert str(cd_sr_1) == str(pd_sr_1)


def test_categorical_add_categories():
    categories = list("abc")
    codes = [0, 0, 1, 0, 1, 2, 0, 1, 1, 2]
    pd_str_cat = pd.Categorical.from_codes(codes, categories=categories)
    pd_sr = pd.Series(pd_str_cat)
    cd_sr = cudf.Series(pd_str_cat)

    assert_eq(pd_sr, cd_sr)

    assert str(pd_sr) == str(cd_sr)

    pd_sr_1 = pd_sr.cat.add_categories(["d"])
    cd_sr_1 = cd_sr.cat.add_categories(["d"])

    assert "d" in pd_sr_1.cat.categories.to_list()
    assert "d" in cd_sr_1.cat.categories.to_pandas().to_list()

    assert_eq(pd_sr_1, cd_sr_1)


def test_categorical_remove_categories():
    categories = list("abc")
    codes = [0, 0, 1, 0, 1, 2, 0, 1, 1, 2]
    pd_str_cat = pd.Categorical.from_codes(codes, categories=categories)
    pd_sr = pd.Series(pd_str_cat)
    cd_sr = cudf.Series(pd_str_cat)

    assert_eq(pd_sr, cd_sr)

    assert str(pd_sr) == str(cd_sr)

    pd_sr_1 = pd_sr.cat.remove_categories(["a"])
    cd_sr_1 = cd_sr.cat.remove_categories(["a"])

    assert "a" not in pd_sr_1.cat.categories.to_list()
    assert "a" not in cd_sr_1.cat.categories.to_pandas().to_list()

    assert_eq(pd_sr_1, cd_sr_1)

    # test using ordered operators
    assert_exceptions_equal(
        lfunc=cd_sr.to_pandas().cat.remove_categories,
        rfunc=cd_sr.cat.remove_categories,
        lfunc_args_and_kwargs=([["a", "d"]], {}),
        rfunc_args_and_kwargs=([["a", "d"]], {}),
    )


@pytest.mark.filterwarnings("ignore:Can't safely cast column:UserWarning")
@pytest.mark.parametrize(
    "data",
    [
        pd.Series([1, 2, 3, 89]),
        pd.Series(["a", "b", "c", "c", "b", "a", "b", "b"]),
        pd.Series(["aa", "b", "c", "c", "bb", "bb", "a", "b", "b"]),
        pd.Series([1, 2, 3, 89, None, np.nan, np.nan], dtype="float64"),
        pd.Series([1, 2, 3, 89], dtype="float64"),
        pd.Series([1, 2.5, 3.001, 89], dtype="float64"),
        pd.Series([None, None, None]),
        pd.Series([], dtype="float64"),
    ],
)
@pytest.mark.parametrize(
    "new_categories",
    [
        ["aa", "bb", "cc"],
        [2, 4, 10, 100],
        ["a", "b", "c"],
        [],
        pd.Series(["a", "b", "c"]),
        pd.Series(["a", "b", "c"], dtype="category"),
        pd.Series([-100, 10, 11, 0, 1, 2], dtype="category"),
    ],
)
def test_categorical_set_categories_categoricals(data, new_categories):
    pd_data = data.astype("category")
    gd_data = cudf.from_pandas(pd_data)

    expected = pd_data.cat.set_categories(new_categories=new_categories)
    actual = gd_data.cat.set_categories(new_categories=new_categories)

    if isinstance(new_categories, list) and len(new_categories) == 0:
        # As of pandas 3.0, empty default type of object isn't
        # necessarily equivalent to cuDF's empty default type of
        # pandas.StringDtype
        expected = expected.cat.set_categories(
            pd.Index([], dtype=actual.cat.categories.dtype)
        )
    assert_eq(expected, actual)

    expected = pd_data.cat.set_categories(
        new_categories=pd.Series(new_categories, dtype="category")
    )
    actual = gd_data.cat.set_categories(
        new_categories=cudf.Series(new_categories, dtype="category")
    )
    if isinstance(new_categories, list) and len(new_categories) == 0:
        # As of pandas 3.0, empty default type of object isn't
        # necessarily equivalent to cuDF's empty default type of
        # pandas.StringDtype
        expected = expected.cat.set_categories(
            pd.Index([], dtype=actual.cat.categories.dtype)
        )

    assert_eq(expected, actual)


@pytest.mark.filterwarnings("ignore:Can't safely cast column:UserWarning")
@pytest.mark.parametrize(
    "data,add",
    [
        ([1, 2, 3], [100, 11, 12]),
        ([1, 2, 3], [0.01, 9.7, 15.0]),
        ([0.0, 6.7, 10.0], [100, 11, 12]),
        ([0.0, 6.7, 10.0], [0.01, 9.7, 15.0]),
        (["a", "bd", "ef"], ["asdfsdf", "bddf", "eff"]),
        ([1, 2, 3], []),
        ([0.0, 6.7, 10.0], []),
        (["a", "bd", "ef"], []),
    ],
)
def test_add_categories(data, add):
    pds = pd.Series(data, dtype="category")
    gds = cudf.Series(data, dtype="category")

    expected = pds.cat.add_categories(add)
    actual = gds.cat.add_categories(add)

    assert_eq(
        expected.cat.codes, actual.cat.codes.astype(expected.cat.codes.dtype)
    )

    # Need to type-cast pandas object to str due to mixed-type
    # support in "object"
    assert_eq(
        expected.cat.categories.astype("str")
        if (expected.cat.categories.dtype == "object")
        else expected.cat.categories,
        actual.cat.categories,
    )


@pytest.mark.parametrize(
    "data,add",
    [
        ([1, 2, 3], [1, 3, 11]),
        ([0.0, 6.7, 10.0], [1, 2, 0.0]),
        (["a", "bd", "ef"], ["a", "bd", "a"]),
    ],
)
def test_add_categories_error(data, add):
    pds = pd.Series(data, dtype="category")
    gds = cudf.Series(data, dtype="category")

    assert_exceptions_equal(
        pds.cat.add_categories,
        gds.cat.add_categories,
        ([add],),
        ([add],),
    )


def test_add_categories_mixed_error():
    gds = cudf.Series(["a", "bd", "ef"], dtype="category")

    with pytest.raises(TypeError):
        gds.cat.add_categories([1, 2, 3])

    gds = cudf.Series([1, 2, 3], dtype="category")

    with pytest.raises(TypeError):
        gds.cat.add_categories(["a", "bd", "ef"])


def test_categorical_allow_nan():
    gs = cudf.Series([1, 2, np.nan, 10, np.nan, None], nan_as_null=False)
    gs = gs.astype("category")
    # cat.codes uses a signed dtype to match pandas (int8 here, not uint8).
    expected_codes = cudf.Series([0, 1, None, 2, None, None], dtype="int8")
    assert_eq(expected_codes, gs.cat.codes)
    assert gs.cat.codes.dtype == np.dtype("int8")

    expected_categories = cudf.Index([1.0, 2.0, 10.0], dtype="float64")
    assert_eq(expected_categories, gs.cat.categories)

    actual_ps = gs.to_pandas()
    expected_ps = pd.Series(
        [1.0, 2.0, np.nan, 10.0, np.nan, np.nan], dtype="category"
    )
    assert_eq(actual_ps, expected_ps)


def test_cat_iterate_error():
    s = cudf.Series([1, 2, 3], dtype="category")
    with pytest.raises(TypeError):
        iter(s.cat)

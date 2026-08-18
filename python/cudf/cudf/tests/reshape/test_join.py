# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


@pytest.fixture(
    params=(
        "left",
        "inner",
        "outer",
        "right",
        "leftanti",
        "leftsemi",
        "cross",
    )
)
def how(request):
    return request.param


def assert_join_results_equal(expect, got, how, **kwargs):
    if how == "right":
        got = got[expect.columns]

    if isinstance(expect, (pd.Series, cudf.Series)):
        return assert_eq(
            expect.sort_values().reset_index(drop=True),
            got.sort_values().reset_index(drop=True),
            **kwargs,
        )
    elif isinstance(expect, (pd.DataFrame, cudf.DataFrame)):
        if not len(
            expect.columns
        ):  # can't sort_values() on a df without columns
            return assert_eq(expect, got, **kwargs)

        assert_eq(
            expect.sort_values(expect.columns.to_list()).reset_index(
                drop=True
            ),
            got.sort_values(got.columns.to_list()).reset_index(drop=True),
            **kwargs,
        )
    elif isinstance(expect, (pd.Index, cudf.Index)):
        return assert_eq(expect.sort_values(), got.sort_values(), **kwargs)
    else:
        raise ValueError(f"Not a join result: {type(expect).__name__}")


@pytest.mark.parametrize(
    "aa, bb",
    [
        [[0, 0, 4, 5, 5], [0, 0, 2, 3, 5]],
        [[0, 0, 1, 2, 3], [0, 1, 2, 2, 3]],
        [range(5), range(5, 10)],
        [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
    ],
)
def test_dataframe_join_how(aa, bb, how):
    df = cudf.DataFrame(
        {
            "a": aa,
            "b": bb,
        }
    )

    def work_pandas(df, how):
        df1 = df.set_index("a")
        df2 = df.set_index("b")
        if how == "leftanti":
            joined = df1[~df1.index.isin(df2.index)][df1.columns]
        elif how == "leftsemi":
            joined = df1[df1.index.isin(df2.index)][df1.columns]
        else:
            joined = df1.join(df2, how=how, sort=True)
        return joined

    def work_gdf(df):
        df1 = df.set_index("a")
        df2 = df.set_index("b")
        joined = df1.join(df2, how=how, sort=True)
        return joined

    expect = work_pandas(df.to_pandas(), how)
    got = work_gdf(df)
    expecto = expect.copy()
    goto = got.copy()

    expect = expect.astype(np.float64).fillna(np.nan)[expect.columns]
    got = got.astype(np.float64).fillna(np.nan)[expect.columns]

    assert got.index.name is None

    assert list(expect.columns) == list(got.columns)
    if how in {"left", "inner", "right", "leftanti", "leftsemi"}:
        assert_eq(sorted(expect.index.values), sorted(got.index.values))
        if how != "outer":
            # Newly introduced ambiguous ValueError thrown when
            # an index and column have the same name. Rename the
            # index so sorts work.
            # TODO: What is the less hacky way?
            expect.index.name = "bob"
            got.index.name = "mary"
            assert_join_results_equal(expect, got, how=how)
        # if(how=='right'):
        #     _sorted_check_series(expect['a'], expect['b'],
        #                          got['a'], got['b'])
        # else:
        #     _sorted_check_series(expect['b'], expect['a'], got['b'],
        #                          got['a'])
        else:
            magic = 0xDEADBEAF
            for c in expecto.columns:
                expect = expecto[c].fillna(-1)
                got = goto[c].fillna(-1)

                direct_equal = np.all(expect.values == got.to_numpy())
                nanfilled_equal = np.all(
                    expect.fillna(magic).values == got.fillna(magic).to_numpy()
                )
                msg = "direct_equal={}, nanfilled_equal={}".format(
                    direct_equal, nanfilled_equal
                )
                assert direct_equal or nanfilled_equal, msg


def test_dataframe_join_suffix():
    rng = np.random.default_rng(seed=0)

    df = cudf.DataFrame(rng.integers(0, 5, (5, 3)), columns=list("abc"))

    left = df.set_index("a")
    right = df.set_index("c")
    msg = (
        "there are overlapping columns but lsuffix and rsuffix are not defined"
    )
    with pytest.raises(ValueError, match=msg):
        left.join(right)

    got = left.join(right, lsuffix="_left", rsuffix="_right", sort=True)
    expect = left.to_pandas().join(
        right.to_pandas(),
        lsuffix="_left",
        rsuffix="_right",
        sort=True,
    )
    # TODO: Retain result index name
    expect.index.name = None
    assert_join_results_equal(expect, got, how="inner")


def test_dataframe_join_cats():
    lhs = cudf.DataFrame()
    lhs["a"] = pd.Categorical(list("aababcabbc"), categories=list("abc"))
    lhs["b"] = bb = np.arange(len(lhs))
    lhs = lhs.set_index("a")

    rhs = cudf.DataFrame()
    rhs["a"] = pd.Categorical(list("abcac"), categories=list("abc"))
    rhs["c"] = cc = np.arange(len(rhs))
    rhs = rhs.set_index("a")

    got = lhs.join(rhs)
    expect = lhs.to_pandas().join(rhs.to_pandas())

    # Note: pandas make an object Index after joining
    assert_join_results_equal(expect, got, how="inner")

    # Just do some rough checking here.
    assert list(got.columns) == ["b", "c"]
    assert len(got) > 0
    assert set(got.index.to_pandas()) & set("abc")
    assert set(got["b"].to_numpy()) & set(bb)
    assert set(got["c"].to_numpy()) & set(cc)


def test_dataframe_join_combine_cats():
    lhs = cudf.DataFrame({"join_index": ["a", "b", "c"], "data_x": [1, 2, 3]})
    rhs = cudf.DataFrame({"join_index": ["b", "c", "d"], "data_y": [2, 3, 4]})

    lhs["join_index"] = lhs["join_index"].astype("category")
    rhs["join_index"] = rhs["join_index"].astype("category")

    lhs = lhs.set_index("join_index")
    rhs = rhs.set_index("join_index")

    lhs_pd = lhs.to_pandas()
    rhs_pd = rhs.to_pandas()

    lhs_pd.index = lhs_pd.index.astype(pd.StringDtype(na_value=np.nan))
    rhs_pd.index = rhs_pd.index.astype(pd.StringDtype(na_value=np.nan))

    # Categorical keys with different category sets decategorize to their
    # common (string) dtype, matching pandas.
    expect = lhs_pd.join(rhs_pd, how="outer")
    got = lhs.join(rhs, how="outer")

    assert_eq(expect.index.sort_values(), got.index.sort_values())


def test_dataframe_join_mismatch_cats(how):
    if how in {"leftanti", "leftsemi"}:
        pytest.skip(f"{how} not implemented in pandas")

    pdf1 = pd.DataFrame(
        {
            "join_col": ["a", "b", "c", "d", "e"],
            "data_col_left": [10, 20, 30, 40, 50],
        }
    )
    pdf2 = pd.DataFrame(
        {"join_col": ["c", "e", "f"], "data_col_right": [6, 7, 8]}
    )

    pdf1["join_col"] = pdf1["join_col"].astype("category")
    pdf2["join_col"] = pdf2["join_col"].astype("category")

    gdf1 = cudf.from_pandas(pdf1)
    gdf2 = cudf.from_pandas(pdf2)

    gdf1 = gdf1.set_index("join_col")
    gdf2 = gdf2.set_index("join_col")

    pdf1 = pdf1.set_index("join_col")
    pdf2 = pdf2.set_index("join_col")
    join_gdf = gdf1.join(gdf2, how=how, sort=True)
    join_pdf = pdf1.join(pdf2, how=how)

    got = join_gdf.fillna(-1).to_pandas()
    expect = join_pdf.fillna(-1)  # note: cudf join doesn't mask NA

    # cudf creates the columns in different order than pandas for right join
    if how == "right":
        got = got[["data_col_left", "data_col_right"]]

    # Unmatched rows introduce NaN, upcasting the integer data columns to
    # float64 (matching pandas).
    assert_join_results_equal(expect, got, how=how, check_categorical=False)


def test_join_datetimes_index(datetime_types_as_str):
    datetimes = pd.Series(pd.date_range("20010101", "20010102", freq="12h"))
    pdf_lhs = pd.DataFrame(index=[1, 0, 1, 2, 0, 0, 1])
    pdf_rhs = pd.DataFrame({"d": datetimes})
    gdf_lhs = cudf.from_pandas(pdf_lhs)
    gdf_rhs = cudf.from_pandas(pdf_rhs)

    gdf_rhs["d"] = gdf_rhs["d"].astype(datetime_types_as_str)

    pdf = pdf_lhs.join(pdf_rhs, sort=True)
    gdf = gdf_lhs.join(gdf_rhs, sort=True)

    assert gdf["d"].dtype == cudf.dtype(datetime_types_as_str)

    assert_join_results_equal(pdf, gdf, how="inner", check_dtype=False)


@pytest.mark.parametrize(
    "column_a",
    [
        (
            pd.Series([None, 1, 2, 3, 4, 5, 6, 7], dtype=np.float64),
            pd.Series([8, 9, 10, 11, 12, None, 14, 15], dtype=np.float64),
        )
    ],
)
@pytest.mark.parametrize(
    "column_b",
    [
        (
            pd.Series([0, 1, 0, None, 1, 0, 0, 0], dtype=np.float64),
            pd.Series([None, 1, 2, 1, 2, 2, 0, 0], dtype=np.float64),
        )
    ],
)
@pytest.mark.parametrize(
    "column_c",
    [
        (
            pd.Series(["dog", "cat", "fish", "bug"] * 2),
            pd.Series(["bird", "cat", "mouse", "snake"] * 2),
        ),
        (
            pd.Series(["dog", "cat", "fish", "bug"] * 2).astype("category"),
            pd.Series(["bird", "cat", "mouse", "snake"] * 2).astype(
                "category"
            ),
        ),
    ],
)
def test_join_multi(how, column_a, column_b, column_c):
    if how in {"leftanti", "leftsemi"}:
        pytest.skip(f"{how} not implemented in pandas")

    index = ["b", "c"]
    df1 = pd.DataFrame()
    df1["a1"] = column_a[0]
    df1["b"] = column_b[0]
    df1["c"] = column_c[0]
    df1 = df1.set_index(index)
    gdf1 = cudf.from_pandas(df1)

    df2 = pd.DataFrame()
    df2["a2"] = column_a[1]
    df2["b"] = column_b[1]
    df2["c"] = column_c[1]
    df2 = df2.set_index(index)
    gdf2 = cudf.from_pandas(df2)

    gdf_result = gdf1.join(gdf2, how=how, sort=True)
    pdf_result = df1.join(df2, how=how, sort=True)

    assert_join_results_equal(pdf_result, gdf_result, how="inner")


@pytest.mark.parametrize(
    ("lhs", "rhs"),
    [
        (["a", "b"], ["a"]),
        (["a"], ["a", "b"]),
        (["a", "b"], ["b"]),
        (["b"], ["a", "b"]),
        (["a"], ["a"]),
    ],
)
@pytest.mark.parametrize("level", ["a", "b", 0, 1])
def test_index_join(lhs, rhs, how, level):
    if how in {"leftanti", "leftsemi", "cross"}:
        pytest.skip(f"{how} not implemented in pandas")

    l_pdf = pd.DataFrame({"a": [2, 3, 1, 4], "b": [3, 7, 8, 1]})
    r_pdf = pd.DataFrame({"a": [1, 5, 4, 0], "b": [3, 9, 8, 4]})
    l_df = cudf.from_pandas(l_pdf)
    r_df = cudf.from_pandas(r_pdf)
    p_lhs = l_pdf.set_index(lhs).index
    p_rhs = r_pdf.set_index(rhs).index
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index

    expected = p_lhs.join(p_rhs, level=level, how=how).to_frame(index=False)
    got = g_lhs.join(g_rhs, level=level, how=how).to_frame(index=False)

    assert_join_results_equal(expected, got, how=how)


def test_index_join_corner_cases():
    l_pdf = pd.DataFrame({"a": [2, 3, 1, 4], "b": [3, 7, 8, 1]})
    r_pdf = pd.DataFrame(
        {"a": [1, 5, 4, 0], "b": [3, 9, 8, 4], "c": [2, 3, 6, 0]}
    )
    l_df = cudf.from_pandas(l_pdf)
    r_df = cudf.from_pandas(r_pdf)

    # Join when column name doesn't match with level
    lhs = ["a", "b"]
    # level and rhs don't match
    rhs = ["c"]
    level = "b"
    how = "outer"
    p_lhs = l_pdf.set_index(lhs).index
    p_rhs = r_pdf.set_index(rhs).index
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index
    expected = p_lhs.join(p_rhs, level=level, how=how).to_frame(index=False)
    got = g_lhs.join(g_rhs, level=level, how=how).to_frame(index=False)

    assert_join_results_equal(expected, got, how=how)

    # sort is supported only in case of two non-MultiIndex join
    # Join when column name doesn't match with level
    lhs = ["a"]
    # level and rhs don't match
    rhs = ["a"]
    level = "b"
    how = "left"
    p_lhs = l_pdf.set_index(lhs).index
    p_rhs = r_pdf.set_index(rhs).index
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index
    expected = p_lhs.join(p_rhs, how=how, sort=True)
    got = g_lhs.join(g_rhs, how=how, sort=True)

    assert_join_results_equal(expected, got, how=how)

    # Pandas Index.join on categorical column returns generic column
    # but cudf will be returning a categorical column itself.
    lhs = ["a", "b"]
    rhs = ["a"]
    level = "a"
    how = "inner"
    l_df["a"] = l_df["a"].astype("category")
    r_df["a"] = r_df["a"].astype("category")
    p_lhs = l_pdf.set_index(lhs).index
    p_rhs = r_pdf.set_index(rhs).index
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index
    expected = p_lhs.join(p_rhs, level=level, how=how).to_frame(index=False)
    got = g_lhs.join(g_rhs, level=level, how=how).to_frame(index=False)

    got["a"] = got["a"].astype(expected["a"].dtype)

    assert_join_results_equal(expected, got, how=how)


def test_index_join_exception_cases():
    l_df = cudf.DataFrame({"a": [2, 3, 1, 4], "b": [3, 7, 8, 1]})
    r_df = cudf.DataFrame(
        {"a": [1, 5, 4, 0], "b": [3, 9, 8, 4], "c": [2, 3, 6, 0]}
    )

    # Join between two MultiIndex
    lhs = ["a", "b"]
    rhs = ["a", "c"]
    level = "a"
    how = "outer"
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index

    with pytest.raises(TypeError):
        g_lhs.join(g_rhs, level=level, how=how)

    # Improper level value, level should be an int or scalar value
    level = ["a"]
    rhs = ["a"]
    g_lhs = l_df.set_index(lhs).index
    g_rhs = r_df.set_index(rhs).index
    with pytest.raises(ValueError):
        g_lhs.join(g_rhs, level=level, how=how)


def test_typecast_on_join_indexes():
    join_data_l = cudf.Series([1, 2, 3, 4, 5], dtype="int8")
    join_data_r = cudf.Series([1, 2, 3, 4, 6], dtype="int32")
    other_data = ["a", "b", "c", "d", "e"]

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    gdf_l = gdf_l.set_index("join_col")
    gdf_r = gdf_r.set_index("join_col")

    exp_join_data = [1, 2, 3, 4]
    exp_other_data = ["a", "b", "c", "d"]

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_data,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )
    expect = expect.set_index("join_col")

    got = gdf_l.join(gdf_r, how="inner", lsuffix="_x", rsuffix="_y")

    assert_join_results_equal(expect, got, how="inner")


def test_typecast_on_join_multiindices():
    join_data_l_0 = cudf.Series([1, 2, 3, 4, 5], dtype="int8")
    join_data_l_1 = cudf.Series([2, 3, 4.1, 5.9, 6], dtype="float32")
    join_data_l_2 = cudf.Series([7, 8, 9, 0, 1], dtype="float32")

    join_data_r_0 = cudf.Series([1, 2, 3, 4, 5], dtype="int32")
    join_data_r_1 = cudf.Series([2, 3, 4, 5, 6], dtype="int32")
    join_data_r_2 = cudf.Series([7, 8, 9, 0, 0], dtype="float64")

    other_data = ["a", "b", "c", "d", "e"]

    gdf_l = cudf.DataFrame(
        {
            "join_col_0": join_data_l_0,
            "join_col_1": join_data_l_1,
            "join_col_2": join_data_l_2,
            "B": other_data,
        }
    )
    gdf_r = cudf.DataFrame(
        {
            "join_col_0": join_data_r_0,
            "join_col_1": join_data_r_1,
            "join_col_2": join_data_r_2,
            "B": other_data,
        }
    )

    gdf_l = gdf_l.set_index(["join_col_0", "join_col_1", "join_col_2"])
    gdf_r = gdf_r.set_index(["join_col_0", "join_col_1", "join_col_2"])

    exp_join_data_0 = cudf.Series([1, 2], dtype="int32")
    exp_join_data_1 = cudf.Series([2, 3], dtype="float64")
    exp_join_data_2 = cudf.Series([7, 8], dtype="float64")
    exp_other_data = cudf.Series(["a", "b"])

    expect = cudf.DataFrame(
        {
            "join_col_0": exp_join_data_0,
            "join_col_1": exp_join_data_1,
            "join_col_2": exp_join_data_2,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )
    expect = expect.set_index(["join_col_0", "join_col_1", "join_col_2"])
    got = gdf_l.join(gdf_r, how="inner", lsuffix="_x", rsuffix="_y")

    assert_join_results_equal(expect, got, how="inner")


def test_typecast_on_join_indexes_matching_categorical():
    join_data_l = cudf.Series(["a", "b", "c", "d", "e"], dtype="category")
    join_data_r = cudf.Series(["a", "b", "c", "d", "e"], dtype="str")
    other_data = [1, 2, 3, 4, 5]

    gdf_l = cudf.DataFrame({"join_col": join_data_l, "B": other_data})
    gdf_r = cudf.DataFrame({"join_col": join_data_r, "B": other_data})

    gdf_l = gdf_l.set_index("join_col")
    gdf_r = gdf_r.set_index("join_col")

    exp_join_data = ["a", "b", "c", "d", "e"]
    exp_other_data = [1, 2, 3, 4, 5]

    expect = cudf.DataFrame(
        {
            "join_col": exp_join_data,
            "B_x": exp_other_data,
            "B_y": exp_other_data,
        }
    )
    expect = expect.set_index("join_col")
    got = gdf_l.join(gdf_r, how="inner", lsuffix="_x", rsuffix="_y")

    assert_join_results_equal(expect, got, how="inner")


@pytest.mark.parametrize("empty_side", ["left", "right"])
def test_outer_merge_dtype_preserved_with_empty_side(empty_side):
    # https://github.com/rapidsai/cudf/issues/9981
    # In an outer merge, every output row's key value comes from whichever
    # side is non-empty, so if one side of the merge key has no rows, no
    # value could actually be lost. The non-empty side's dtype should
    # survive the merge just like it does in pandas, instead of being
    # promoted to float64.
    pdf_data = pd.DataFrame({"a": [1, 2, 3, 4]})
    pdf_empty = pd.DataFrame(columns=["a"])
    lhs, rhs = (
        (pdf_empty, pdf_data)
        if empty_side == "left"
        else (pdf_data, pdf_empty)
    )
    glhs, grhs = cudf.from_pandas(lhs), cudf.from_pandas(rhs)

    expect = lhs.merge(rhs, how="outer")
    got = glhs.merge(grhs, how="outer")

    assert expect["a"].dtype == got["a"].dtype
    assert_join_results_equal(expect, got, how="outer")


@pytest.mark.parametrize("empty_side", ["left", "right"])
def test_outer_merge_int_width_preserved_with_empty_side(empty_side):
    # https://github.com/rapidsai/cudf/issues/9981
    # In an outer merge, every output row's key value comes from whichever
    # side is non-empty, so if one side of the merge key has no rows, no
    # value could actually be lost. The non-empty side's width should
    # survive the merge just like it does in pandas, instead of being
    # promoted to max(ltype, rtype).
    pdf_data = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="int8")})
    pdf_empty = pd.DataFrame({"a": pd.Series([], dtype="int64")})
    lhs, rhs = (
        (pdf_empty, pdf_data)
        if empty_side == "left"
        else (pdf_data, pdf_empty)
    )
    glhs, grhs = cudf.from_pandas(lhs), cudf.from_pandas(rhs)

    expect = lhs.merge(rhs, how="outer")
    got = glhs.merge(grhs, how="outer")

    assert expect["a"].dtype == got["a"].dtype
    assert_join_results_equal(expect, got, how="outer")


def test_outer_merge_both_sides_empty_dtype():
    # https://github.com/rapidsai/cudf/issues/9981
    # With BOTH sides empty the one-empty-side shortcut does not apply;
    # pandas keeps the left (object) dtype for the combined key, and so
    # should cudf via the empty-key dtype restore.
    lhs = pd.DataFrame(columns=["a"])
    rhs = pd.DataFrame({"a": pd.Series([], dtype="int64")})
    glhs, grhs = cudf.from_pandas(lhs), cudf.from_pandas(rhs)

    expect = lhs.merge(rhs, how="outer")
    got = glhs.merge(grhs, how="outer")

    assert expect["a"].dtype == got["a"].dtype
    assert_join_results_equal(expect, got, how="outer")


@pytest.mark.parametrize("empty_side", ["left", "right"])
def test_outer_merge_datetime_resolution_preserved_with_empty_side(
    empty_side,
):
    # https://github.com/rapidsai/cudf/issues/9981
    # Same reasoning as test_outer_merge_dtype_preserved_with_empty_side,
    # but for a case that used to go through the datetime/timedelta
    # promotion branch (max(ltype, rtype)) rather than the float64
    # fallback: an empty side should not force promotion to the finer
    # resolution, it should just adopt the non-empty side's resolution.
    pdf_data = pd.DataFrame(
        {"a": pd.to_datetime(["2020-01-01", "2020-01-02"])}
    ).astype({"a": "datetime64[s]"})
    pdf_empty = pd.DataFrame({"a": pd.Series([], dtype="datetime64[ns]")})
    lhs, rhs = (
        (pdf_empty, pdf_data)
        if empty_side == "left"
        else (pdf_data, pdf_empty)
    )
    glhs, grhs = cudf.from_pandas(lhs), cudf.from_pandas(rhs)

    expect = lhs.merge(rhs, how="outer")
    got = glhs.merge(grhs, how="outer")

    assert expect["a"].dtype == got["a"].dtype
    assert_join_results_equal(expect, got, how="outer")


@pytest.mark.parametrize("empty_side", ["left", "right"])
def test_outer_merge_categorical_preserved_with_empty_side(empty_side):
    # https://github.com/rapidsai/cudf/issues/9981
    # Same reasoning as test_outer_merge_dtype_preserved_with_empty_side,
    # but for a categorical key: an empty side should not force the two
    # sides' categories to be unioned or decategorized, it should just
    # adopt the non-empty side's (categorical) dtype.
    pdf_data = pd.DataFrame(
        {"a": pd.Categorical(["x", "y", "z"], categories=["x", "y", "z"])}
    )
    pdf_empty = pd.DataFrame(
        {"a": pd.Categorical([], categories=["x", "y", "z"])}
    )
    lhs, rhs = (
        (pdf_empty, pdf_data)
        if empty_side == "left"
        else (pdf_data, pdf_empty)
    )
    glhs, grhs = cudf.from_pandas(lhs), cudf.from_pandas(rhs)

    expect = lhs.merge(rhs, how="outer")
    got = glhs.merge(grhs, how="outer")

    assert isinstance(got["a"].dtype, cudf.CategoricalDtype)
    assert_join_results_equal(expect, got, how="outer")


@pytest.mark.parametrize("empty_side", ["left", "right"])
def test_outer_merge_categorical_category_mismatch_with_empty_side(
    empty_side,
):
    # https://github.com/rapidsai/cudf/issues/9981
    # When the empty side's categories differ from the non-empty side's,
    # pandas keeps the non-empty side's categorical dtype (categories and
    # all); the shortcut adopts the non-empty side's dtype, matching that.
    pdf_data = pd.DataFrame(
        {"a": pd.Categorical(["x", "y"], categories=["x", "y"])}
    )
    pdf_empty = pd.DataFrame({"a": pd.Categorical([], categories=["y", "z"])})
    lhs, rhs = (
        (pdf_empty, pdf_data)
        if empty_side == "left"
        else (pdf_data, pdf_empty)
    )
    glhs, grhs = cudf.from_pandas(lhs), cudf.from_pandas(rhs)

    expect = lhs.merge(rhs, how="outer")
    got = glhs.merge(grhs, how="outer")

    assert isinstance(got["a"].dtype, cudf.CategoricalDtype)
    assert list(got["a"].dtype.categories.to_pandas()) == list(
        expect["a"].dtype.categories
    )
    assert_join_results_equal(expect, got, how="outer")


@pytest.mark.parametrize("empty_side", ["left", "right"])
def test_outer_merge_int_vs_empty_categorical(empty_side):
    # https://github.com/rapidsai/cudf/issues/9981
    # A non-empty int64 key against an empty categorical key adopts the
    # int64 dtype, matching pandas.
    pdf_data = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="int64")})
    pdf_empty = pd.DataFrame({"a": pd.Categorical([])})
    lhs, rhs = (
        (pdf_empty, pdf_data)
        if empty_side == "left"
        else (pdf_data, pdf_empty)
    )
    glhs, grhs = cudf.from_pandas(lhs), cudf.from_pandas(rhs)

    expect = lhs.merge(rhs, how="outer")
    got = glhs.merge(grhs, how="outer")

    assert expect["a"].dtype == got["a"].dtype
    assert_join_results_equal(expect, got, how="outer")


@pytest.mark.parametrize("empty_side", ["left", "right"])
def test_outer_merge_categorical_ordering_mismatch_with_empty_side(
    empty_side,
):
    # https://github.com/rapidsai/cudf/issues/9981
    # Merging on categorical variables with mismatched ordering is normally
    # ambiguous and rejected, but that ambiguity is a value-comparison
    # concern that is vacuous when one side has no values to compare. An
    # empty side should adopt the non-empty side's (unordered) dtype
    # instead of raising, matching pandas.
    pdf_data = pd.DataFrame(
        {"a": pd.Categorical(["a", "b", "a"], categories=["a", "b"])}
    )
    pdf_empty = pd.DataFrame(
        {"a": pd.Categorical([], categories=["a", "b"], ordered=True)}
    )
    lhs, rhs = (
        (pdf_empty, pdf_data)
        if empty_side == "left"
        else (pdf_data, pdf_empty)
    )
    glhs, grhs = cudf.from_pandas(lhs), cudf.from_pandas(rhs)

    expect = lhs.merge(rhs, how="outer")
    got = glhs.merge(grhs, how="outer")

    assert isinstance(got["a"].dtype, cudf.CategoricalDtype)
    assert got["a"].dtype.ordered == expect["a"].dtype.ordered
    assert_join_results_equal(expect, got, how="outer")


@pytest.mark.parametrize("empty_side", ["left", "right"])
def test_outer_merge_datetime_vs_int_with_empty_side(empty_side):
    # https://github.com/rapidsai/cudf/issues/9981
    # Merging a datetime64 key against an int64 key normally raises, since
    # neither can safely cast to the other. But an empty side has no data
    # that could actually be lost, so it should adopt the non-empty side's
    # dtype instead of raising, matching pandas.
    pdf_data = pd.DataFrame({"a": pd.Series([1, 2, 3], dtype="int64")})
    pdf_empty = pd.DataFrame({"a": pd.Series([], dtype="datetime64[ns]")})
    lhs, rhs = (
        (pdf_empty, pdf_data)
        if empty_side == "left"
        else (pdf_data, pdf_empty)
    )
    glhs, grhs = cudf.from_pandas(lhs), cudf.from_pandas(rhs)

    expect = lhs.merge(rhs, how="outer")
    got = glhs.merge(grhs, how="outer")

    assert expect["a"].dtype == got["a"].dtype
    assert_join_results_equal(expect, got, how="outer")


def test_join_multiindex_empty():
    lhs = pd.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4]}, index=["a", "b", "c"])
    lhs.columns = pd.MultiIndex.from_tuples([("a", "x"), ("a", "y")])
    rhs = pd.DataFrame(index=["a", "c", "d"])
    g_lhs = cudf.from_pandas(lhs)
    g_rhs = cudf.from_pandas(rhs)
    assert_exceptions_equal(
        lfunc=lhs.join,
        rfunc=g_lhs.join,
        lfunc_args_and_kwargs=([rhs], {"how": "inner"}),
        rfunc_args_and_kwargs=([g_rhs], {"how": "inner"}),
        check_exception_type=False,
    )


def test_join_on_index_with_duplicate_names():
    # although index levels with duplicate names are poorly supported
    # overall, we *should* be able to join on them:
    lhs = pd.DataFrame({"a": [1, 2, 3]})
    rhs = pd.DataFrame({"b": [1, 2, 3]})
    lhs.index = pd.MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1)], names=["x", "x"]
    )
    rhs.index = pd.MultiIndex.from_tuples(
        [(1, 1), (1, 3), (2, 1)], names=["x", "x"]
    )
    expect = lhs.join(rhs, how="inner")

    lhs = cudf.from_pandas(lhs)
    rhs = cudf.from_pandas(rhs)
    got = lhs.join(rhs, how="inner")

    assert_join_results_equal(expect, got, how="inner")


def test_join_multiindex_index():
    # test joining a MultiIndex with an Index with overlapping name
    lhs = (
        cudf.DataFrame({"a": [2, 3, 1], "b": [3, 4, 2]})
        .set_index(["a", "b"])
        .index
    )
    rhs = cudf.DataFrame({"a": [1, 4, 3]}).set_index("a").index
    expect = lhs.to_pandas().join(rhs.to_pandas(), how="inner")
    got = lhs.join(rhs, how="inner")
    assert_join_results_equal(expect, got, how="inner")


def test_dataframe_join_on():
    """Verify that specifying the on parameter gives a NotImplementedError."""
    df = cudf.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(NotImplementedError):
        df.join(df, on="a")


def test_index_join_return_indexers_notimplemented():
    index = cudf.RangeIndex(start=0, stop=20, step=2)
    other = cudf.Index([4, 4, 3, 3])
    with pytest.raises(NotImplementedError):
        index.join(other, how="left", return_indexers=True)


@pytest.mark.xfail(
    reason="https://github.com/pandas-dev/pandas/issues/57065",
)
@pytest.mark.parametrize("how", ["inner", "outer"])
def test_index_join_names(how):
    idx1 = cudf.Index([10, 1, 2, 4, 2, 1], name="a")
    idx2 = cudf.Index([-10, 2, 3, 1, 2], name="b")
    pidx1 = idx1.to_pandas()
    pidx2 = idx2.to_pandas()

    expected = pidx1.join(pidx2, how=how)
    actual = idx1.join(idx2, how=how)
    assert_join_results_equal(actual, expected, how=how)


@pytest.mark.parametrize(
    "left_data",
    [
        {"lkey": ["foo", "bar", "baz", "foo"], "value": [1, 2, 3, 5]},
        {"lkey": ["foo", "bar", "baz", "foo"], "value": [5, 3, 2, 1]},
        {
            "lkey": ["foo", "bar", "baz", "foo"],
            "value": [5, 3, 2, 1],
            "extra_left": [1, 2, 3, 4],
        },
    ],
)
@pytest.mark.parametrize(
    "right_data",
    [
        {"rkey": ["foo", "bar", "baz", "foo"], "value": [5, 6, 7, 8]},
        {"rkey": ["foo", "bar", "baz", "foo"], "value": [8, 7, 6, 5]},
        {
            "rkey": ["foo", "bar", "baz", "foo"],
            "value": [8, 7, 6, 5],
            "extra_right": [10, 2, 30, 4],
        },
    ],
)
@pytest.mark.parametrize("sort", [True, False])
def test_cross_join_overlapping(left_data, right_data, sort):
    df1 = cudf.DataFrame(left_data)
    df2 = cudf.DataFrame(right_data)

    pdf1 = df1.to_pandas()
    pdf2 = df2.to_pandas()
    expected = pdf1.join(
        pdf2, how="cross", lsuffix="_x", rsuffix="_y", sort=sort
    )
    result = df1.join(df2, how="cross", lsuffix="_x", rsuffix="_y", sort=sort)
    assert_eq(result, expected)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import concat
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize("subset", ["a", ["a"], ["a", "B"]])
def test_duplicated_with_misspelled_column_name(subset):
    df = pd.DataFrame({"A": [0, 0, 1], "B": [0, 0, 1], "C": [0, 0, 1]})
    gdf = cudf.DataFrame(df)

    assert_exceptions_equal(
        lfunc=df.drop_duplicates,
        rfunc=gdf.drop_duplicates,
        lfunc_args_and_kwargs=([subset],),
        rfunc_args_and_kwargs=([subset],),
    )


@pytest.mark.xfail(reason="cudf does not support duplicate column names yet")
def test_drop_duplicates_with_duplicate_column_names():
    df = pd.DataFrame(
        [[1, 2, 5], [3, 4, 6], [3, 4, 7]], columns=["a", "a", "b"]
    )
    df = cudf.DataFrame(df)

    result0 = df.drop_duplicates()
    assert_eq(result0, df)

    result1 = df.drop_duplicates("a")
    expected1 = df[:2]
    assert_eq(result1, expected1)


def test_drop_duplicates():
    pdf = pd.DataFrame(
        {
            "AAA": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    gdf = cudf.DataFrame(pdf)
    # single column
    result = gdf.copy()
    result.drop_duplicates("AAA", inplace=True)
    expected = pdf.drop_duplicates("AAA")
    assert_eq(result, expected)

    result = gdf.drop_duplicates("AAA", keep="last")
    expected = pdf.drop_duplicates("AAA", keep="last")
    assert_eq(result, expected)

    result = gdf.drop_duplicates("AAA", keep=False)
    expected = pdf.drop_duplicates("AAA", keep=False)
    assert_eq(result, expected)
    assert len(result) == 0

    # multi column
    expected = pdf.loc[[0, 1, 2, 3]]
    result = gdf.drop_duplicates(np.array(["AAA", "B"]))
    assert_eq(result, expected)
    result = pdf.drop_duplicates(np.array(["AAA", "B"]))
    assert_eq(result, expected)

    result = gdf.drop_duplicates(("AAA", "B"), keep="last")
    expected = pdf.drop_duplicates(("AAA", "B"), keep="last")
    assert_eq(result, expected)

    result = gdf.drop_duplicates(("AAA", "B"), keep=False)
    expected = pdf.drop_duplicates(("AAA", "B"), keep=False)
    assert_eq(result, expected)

    # consider everything
    df2 = gdf.loc[:, ["AAA", "B", "C"]]

    result = df2.drop_duplicates()
    # in this case only
    expected = df2.drop_duplicates(["AAA", "B"])
    assert_eq(result, expected)

    result = df2.drop_duplicates(keep="last")
    expected = df2.drop_duplicates(["AAA", "B"], keep="last")
    assert_eq(result, expected)

    result = df2.drop_duplicates(keep=False)
    expected = df2.drop_duplicates(["AAA", "B"], keep=False)
    assert_eq(result, expected)

    # integers
    result = gdf.drop_duplicates("C")
    expected = pdf.drop_duplicates("C")
    assert_eq(result, expected)
    result = gdf.drop_duplicates("C", keep="last")
    expected = pdf.drop_duplicates("C", keep="last")
    assert_eq(result, expected)

    gdf["E"] = gdf["C"].astype("int8")
    result = gdf.drop_duplicates("E")
    pdf["E"] = pdf["C"].astype("int8")
    expected = pdf.drop_duplicates("E")
    assert_eq(result, expected)
    result = gdf.drop_duplicates("E", keep="last")
    expected = pdf.drop_duplicates("E", keep="last")
    assert_eq(result, expected)


def test_drop_duplicates_integers():
    pdf = pd.DataFrame(
        {"x": [7, 6, 3, 3, 4, 8, 0], "y": [0, 6, 5, 5, 9, 1, 2]}
    )
    gdf = cudf.DataFrame(pdf)
    assert_eq(gdf.drop_duplicates(), pdf.drop_duplicates())


def test_drop_duplicates_integers_positive():
    pdf = pd.DataFrame([[1, 0], [0, 2]])
    gdf = cudf.DataFrame(pdf)
    assert_eq(gdf.drop_duplicates(), pdf.drop_duplicates())


def test_drop_duplicates_integers_negative():
    pdf = pd.DataFrame([[-2, 0], [0, -4]])
    gdf = cudf.DataFrame(pdf)
    assert_eq(gdf.drop_duplicates(), pdf.drop_duplicates())


def test_drop_duplicates_integers_max():
    x = np.iinfo(np.int64).max / 3 * 2
    pdf = pd.DataFrame([[-x, x], [0, x + 4]])
    gdf = cudf.DataFrame(pdf)
    assert_eq(gdf.drop_duplicates(), pdf.drop_duplicates())

    pdf = pd.DataFrame([[-x, x], [x, x + 4]])
    gdf = cudf.DataFrame(pdf)
    assert_eq(gdf.drop_duplicates(), pdf.drop_duplicates())


def test_drop_duplicates_integers_unique():
    pdf = pd.DataFrame([i] * 9 for i in range(16))
    pdf = pd.concat([pdf, pd.DataFrame([[1] + [0] * 8])], ignore_index=True)
    gdf = cudf.DataFrame(pdf)
    assert_eq(gdf.drop_duplicates(), pdf.drop_duplicates())


@pytest.mark.parametrize("subset", ["AAA", ["AAA", "B"]])
@pytest.mark.parametrize("keep", ["first", "last", False])
def test_drop_duplicates_for_take_all(subset, keep):
    pdf = pd.DataFrame(
        {
            "AAA": ["foo", "bar", "baz", "bar", "foo", "bar", "qux", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    gdf = cudf.DataFrame(pdf)
    result = gdf.drop_duplicates(subset, keep=keep)
    expected = pdf.drop_duplicates(subset, keep=keep)
    assert_eq(result, expected)


def test_drop_duplicates_tuple():
    pdf = pd.DataFrame(
        {
            ("AA", "AB"): [
                "foo",
                "bar",
                "foo",
                "bar",
                "foo",
                "bar",
                "bar",
                "foo",
            ],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    gdf = cudf.DataFrame(pdf)
    # single column
    result = gdf.drop_duplicates(("AA", "AB"))
    expected = pdf.drop_duplicates(("AA", "AB"))
    assert_eq(result, expected)

    result = gdf.drop_duplicates(("AA", "AB"), keep="last")
    expected = pdf.drop_duplicates(("AA", "AB"), keep="last")
    assert_eq(result, expected)

    result = gdf.drop_duplicates(("AA", "AB"), keep=False)
    expected = pdf.drop_duplicates(("AA", "AB"), keep=False)  # empty df
    assert len(result) == 0
    assert_eq(result, expected)

    # multi column
    expected = pdf.drop_duplicates((("AA", "AB"), "B"))
    result = gdf.drop_duplicates((("AA", "AB"), "B"))
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(columns=[]),
        pd.DataFrame(columns=["A", "B", "C"]),
        pd.DataFrame(index=[]),
        pd.DataFrame(index=["A", "B", "C"]),
    ],
)
def test_drop_duplicates_empty(df):
    df = cudf.DataFrame(df)
    result = df.drop_duplicates()
    assert_eq(result, df)

    result = df.copy()
    result.drop_duplicates(inplace=True)
    assert_eq(result, df)


def test_dataframe_drop_duplicates_numeric_method():
    pdf = pd.DataFrame(
        {
            "A": [1, 1, 1, 4, 5],
            "B": [1, 1, 1, 4, 5],
            "C": [1, 1, 1, 4, 5],
        }
    )
    gdf = cudf.DataFrame(pdf)
    assert_eq(gdf.drop_duplicates(), pdf.drop_duplicates())

    # subset columns, single columns
    assert_eq(
        gdf.drop_duplicates(pdf.columns[:-1]),
        pdf.drop_duplicates(pdf.columns[:-1]),
    )
    assert_eq(
        gdf.drop_duplicates(pdf.columns[-1]),
        pdf.drop_duplicates(pdf.columns[-1]),
    )
    assert_eq(
        gdf.drop_duplicates(pdf.columns[0]),
        pdf.drop_duplicates(pdf.columns[0]),
    )

    # subset columns shuffled
    cols = ["B", "C", "A"]
    assert_eq(gdf.drop_duplicates(cols), pdf.drop_duplicates(cols))
    assert_eq(gdf.drop_duplicates(cols[:-1]), pdf.drop_duplicates(cols[:-1]))
    assert_eq(gdf.drop_duplicates(cols[-1]), pdf.drop_duplicates(cols[-1]))
    assert_eq(
        gdf.drop_duplicates(pdf.columns, keep="last"),
        pdf.drop_duplicates(pdf.columns, keep="last"),
    )


def test_dataframe_drop_duplicates_method():
    pdf = pd.DataFrame(
        [(1, 2, "a"), (2, 3, "b"), (3, 4, "c"), (2, 3, "d"), (3, 5, "c")],
        columns=["n1", "n2", "s1"],
    )
    gdf = cudf.DataFrame(pdf)
    assert_eq(gdf.drop_duplicates(), pdf.drop_duplicates())

    assert_eq(
        gdf.drop_duplicates("n1")["n1"].reset_index(drop=True),
        pdf.drop_duplicates("n1")["n1"].reset_index(drop=True),
    )
    assert_eq(
        gdf.drop_duplicates("n2")["n2"].reset_index(drop=True),
        pdf.drop_duplicates("n2")["n2"].reset_index(drop=True),
    )
    assert_eq(
        gdf.drop_duplicates("s1")["s1"].reset_index(drop=True),
        pdf.drop_duplicates("s1")["s1"].reset_index(drop=True),
    )
    assert_eq(
        gdf.drop_duplicates("s1", keep="last")["s1"]
        .sort_index()
        .reset_index(drop=True),
        pdf.drop_duplicates("s1", keep="last")["s1"].reset_index(drop=True),
    )
    assert gdf.drop_duplicates("s1", inplace=True) is None

    gdf = cudf.DataFrame(pdf)
    assert_eq(gdf.drop_duplicates("n1"), pdf.drop_duplicates("n1"))
    assert_eq(gdf.drop_duplicates("n2"), pdf.drop_duplicates("n2"))
    assert_eq(gdf.drop_duplicates("s1"), pdf.drop_duplicates("s1"))
    assert_eq(
        gdf.drop_duplicates(["n1", "n2"]), pdf.drop_duplicates(["n1", "n2"])
    )
    assert_eq(
        gdf.drop_duplicates(["n1", "s1"]), pdf.drop_duplicates(["n1", "s1"])
    )

    # Test drop error
    assert_exceptions_equal(
        lfunc=pdf.drop_duplicates,
        rfunc=gdf.drop_duplicates,
        lfunc_args_and_kwargs=(["n3"],),
        rfunc_args_and_kwargs=(["n3"],),
    )

    assert_exceptions_equal(
        lfunc=pdf.drop_duplicates,
        rfunc=gdf.drop_duplicates,
        lfunc_args_and_kwargs=([["n1", "n4", "n3"]],),
        rfunc_args_and_kwargs=([["n1", "n4", "n3"]],),
    )


def test_datetime_drop_duplicates():
    rng = np.random.default_rng(seed=0)
    date_df = cudf.DataFrame(
        {
            "date": pd.date_range("11/20/2018", periods=6, freq="D"),
            "value": rng.random(6),
        }
    )

    df = concat([date_df, date_df[:4]])
    assert_eq(df[:-4], df.drop_duplicates())

    df2 = df.reset_index()
    assert_eq(df2[:-4], df2.drop_duplicates())

    df3 = df.set_index("date")
    assert_eq(df3[:-4], df3.drop_duplicates())


def test_drop_duplicates_NA():
    # none
    df = pd.DataFrame(
        {
            "A": [None, None, "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1, 1.0],
            "D": range(8),
        }
    )
    df = cudf.DataFrame(df)
    # single column
    result = df.drop_duplicates("A")
    expected = df.to_pandas().loc[[0, 2, 3]]
    assert_eq(result, expected)

    result = df.drop_duplicates("A", keep="last")
    expected = df.to_pandas().loc[[1, 6, 7]]
    assert_eq(result, expected)

    result = df.drop_duplicates("A", keep=False)
    expected = df.to_pandas().loc[[]]  # empty df
    assert_eq(result, expected)
    assert len(result) == 0

    # multi column
    result = df.drop_duplicates(["A", "B"])
    expected = df.to_pandas().loc[[0, 2, 3, 6]]
    assert_eq(result, expected)

    result = df.drop_duplicates(["A", "B"], keep="last")
    expected = df.to_pandas().loc[[1, 5, 6, 7]]
    assert_eq(result, expected)

    result = df.drop_duplicates(["A", "B"], keep=False)
    expected = df.to_pandas().loc[[6]]
    assert_eq(result, expected)

    # nan
    df = pd.DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1, 1.0],
            "D": range(8),
        }
    )
    df = cudf.DataFrame(df)
    # single column
    result = df.drop_duplicates("C")
    expected = df[:2]
    assert_eq(result, expected)

    result = df.drop_duplicates("C", keep="last")
    expected = df.to_pandas().loc[[3, 7]]
    assert_eq(result, expected)

    result = df.drop_duplicates("C", keep=False)
    expected = df.to_pandas().loc[[]]  # empty df
    assert_eq(result, expected)
    assert len(result) == 0

    # multi column
    result = df.drop_duplicates(["C", "B"])
    expected = df.to_pandas().loc[[0, 1, 2, 4]]
    assert_eq(result, expected)

    result = df.drop_duplicates(["C", "B"], keep="last")
    expected = df.to_pandas().loc[[1, 3, 6, 7]]
    assert_eq(result, expected)

    result = df.drop_duplicates(["C", "B"], keep=False)
    expected = df.to_pandas().loc[[1]]
    assert_eq(result, expected)


def test_drop_duplicates_NA_for_take_all():
    pdf = pd.DataFrame(
        {
            "A": [None, None, "foo", "bar", "foo", "baz", "bar", "qux"],
            "C": [1.0, np.nan, np.nan, np.nan, 1.0, 2.0, 3, 1.0],
        }
    )
    df = cudf.DataFrame(pdf)

    # single column with None
    result = df.drop_duplicates("A")
    expected = pdf.drop_duplicates("A")
    assert_eq(result, expected)
    assert_eq(
        df.drop_duplicates("A", ignore_index=True),
        pdf.drop_duplicates("A", ignore_index=True),
    )

    result = df.drop_duplicates("A", keep="last")
    expected = pdf.drop_duplicates("A", keep="last")
    assert_eq(result, expected)
    assert_eq(
        df.drop_duplicates("A", ignore_index=True, keep="last"),
        pdf.drop_duplicates("A", ignore_index=True, keep="last"),
    )

    result = df.drop_duplicates("A", keep=False)
    expected = pdf.drop_duplicates("A", keep=False)
    assert_eq(result, expected)
    assert_eq(
        df.drop_duplicates("A", ignore_index=True, keep=False),
        pdf.drop_duplicates("A", ignore_index=True, keep=False),
    )

    # single column with nan
    result = df.drop_duplicates("C")
    expected = pdf.drop_duplicates("C")
    assert_eq(result, expected)

    result = df.drop_duplicates("C", keep="last")
    expected = pdf.drop_duplicates("C", keep="last")
    assert_eq(result, expected)

    result = df.drop_duplicates("C", keep=False)
    expected = pdf.drop_duplicates("C", keep=False)
    assert_eq(result, expected)


def test_drop_duplicates_inplace():
    orig = pd.DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "bar", "foo"],
            "B": ["one", "one", "two", "two", "two", "two", "one", "two"],
            "C": [1, 1, 2, 2, 2, 2, 1, 2],
            "D": range(8),
        }
    )
    orig = cudf.DataFrame(orig)
    # single column
    df = orig.copy()
    df.drop_duplicates("A", inplace=True)
    expected = orig[:2]
    result = df
    assert_eq(result, expected)

    df = orig.copy()
    df.drop_duplicates("A", keep="last", inplace=True)
    expected = orig.loc[[6, 7]]
    result = df
    assert_eq(result, expected)

    df = orig.copy()
    df.drop_duplicates("A", keep=False, inplace=True)
    expected = orig.loc[[]]
    result = df
    assert_eq(result, expected)
    assert len(df) == 0

    # multi column
    df = orig.copy()
    df.drop_duplicates(["A", "B"], inplace=True)
    expected = orig.loc[[0, 1, 2, 3]]
    result = df
    assert_eq(result, expected)

    df = orig.copy()
    df.drop_duplicates(["A", "B"], keep="last", inplace=True)
    expected = orig.loc[[0, 5, 6, 7]]
    result = df
    assert_eq(result, expected)

    df = orig.copy()
    df.drop_duplicates(["A", "B"], keep=False, inplace=True)
    expected = orig.loc[[0]]
    result = df
    assert_eq(result, expected)

    # consider everything
    orig2 = orig.loc[:, ["A", "B", "C"]].copy()

    df2 = orig2.copy()
    df2.drop_duplicates(inplace=True)
    # in this case only
    expected = orig2.drop_duplicates(["A", "B"])
    result = df2
    assert_eq(result, expected)

    df2 = orig2.copy()
    df2.drop_duplicates(keep="last", inplace=True)
    expected = orig2.drop_duplicates(["A", "B"], keep="last")
    result = df2
    assert_eq(result, expected)

    df2 = orig2.copy()
    df2.drop_duplicates(keep=False, inplace=True)
    expected = orig2.drop_duplicates(["A", "B"], keep=False)
    result = df2
    assert_eq(result, expected)


def test_drop_duplicates_multi_index():
    arrays = [
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        ["one", "two", "one", "two", "one", "two", "one", "two"],
    ]

    idx = pd.MultiIndex.from_tuples(
        list(zip(*arrays, strict=True)), names=["a", "b"]
    )
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.integers(0, 2, (8, 4)), index=idx)
    gdf = cudf.DataFrame(pdf)

    expected = pdf.drop_duplicates()
    result = gdf.drop_duplicates()
    assert_eq(result, expected)
    # FIXME: to_pandas needed until sort_index support for MultiIndex

    for col in gdf.columns:
        assert_eq(
            gdf[col].drop_duplicates(),
            pdf[col].drop_duplicates(),
        )


def test_drop_duplicates_ignore_index_wrong_type():
    gdf = cudf.DataFrame([1, 1, 2])
    with pytest.raises(ValueError):
        gdf.drop_duplicates(ignore_index="True")

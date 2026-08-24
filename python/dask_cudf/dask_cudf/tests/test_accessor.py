# SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext as does_not_raise

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_series_equal

import dask
from dask import dataframe as dd

from cudf import DataFrame, Series, date_range
from cudf.testing import assert_eq

import dask_cudf

#############################################################################
#                        Datetime Accessor                                  #
#############################################################################


def data_dt_1():
    return pd.date_range("20010101", "20020215", freq="400h")


def data_dt_2():
    return np.random.default_rng(seed=0).standard_normal(size=100)


dt_fields = ["year", "month", "day", "hour", "minute", "second"]


@pytest.mark.parametrize("data", [data_dt_2()])
def test_datetime_accessor_initialization(data):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dask_cudf.from_cudf(sr, npartitions=5)
    with pytest.raises(AttributeError):
        dsr.dt


@pytest.mark.parametrize("data", [data_dt_1()])
def test_series(data):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dask_cudf.from_cudf(sr, npartitions=5)

    np.testing.assert_equal(np.array(pdsr), dsr.compute().to_numpy())


@pytest.mark.parametrize("data", [data_dt_1()])
@pytest.mark.parametrize("field", dt_fields)
def test_dt_series(data, field):
    pdsr = pd.Series(data.copy())
    sr = Series(pdsr)
    dsr = dask_cudf.from_cudf(sr, npartitions=5)
    base = getattr(pdsr.dt, field)
    test = getattr(dsr.dt, field).compute()
    assert_eq(base, test, check_dtype=False)


@pytest.mark.parametrize("data", [data_dt_1()])
def test_dt_accessor(data):
    df = DataFrame({"dt_col": data.copy()})
    ddf = dask_cudf.from_cudf(df, npartitions=5)

    for i in ["year", "month", "day", "hour", "minute", "second", "weekday"]:
        assert i in dir(ddf.dt_col.dt)
        assert_series_equal(
            getattr(ddf.dt_col.dt, i).compute().to_pandas(),
            getattr(df.dt_col.dt, i).to_pandas(),
        )


#############################################################################
#                        Categorical Accessor                               #
#############################################################################


def data_cat_1():
    cat = pd.Categorical(["a", "a", "b", "c", "a"], categories=["a", "b", "c"])
    return cat


def data_cat_2():
    return pd.Series([1, 2, 3])


def data_cat_3():
    cat1 = pd.Categorical(
        ["a", "a", "b", "c", "a"], categories=["a", "b", "c"], ordered=True
    )
    cat2 = pd.Categorical(
        ["a", "b", "a", "c", "b"], categories=["a", "b", "c"], ordered=True
    )
    return cat1, cat2


@pytest.mark.parametrize("data", [data_cat_1()])
def test_categorical_accessor_initialization1(data):
    sr = Series(data.copy())
    dsr = dask_cudf.from_cudf(sr, npartitions=5)
    dsr.cat


@pytest.mark.parametrize("data", [data_cat_2()])
def test_categorical_accessor_initialization2(data):
    sr = Series(data.copy())
    dsr = dask_cudf.from_cudf(sr, npartitions=5)
    with pytest.raises(AttributeError):
        dsr.cat


@pytest.mark.parametrize("data", [data_cat_1()])
def test_categorical_basic(data):
    cat = data.copy()
    pdsr = pd.Series(cat)
    sr = Series(cat)
    dsr = dask_cudf.from_cudf(sr, npartitions=2)
    result = dsr.compute()
    np.testing.assert_array_equal(cat.codes, result.cat.codes.to_numpy())

    assert dsr.dtype.to_pandas() == pdsr.dtype
    # Test attributes
    assert pdsr.cat.ordered == dsr.cat.ordered

    assert_eq(pdsr.cat.categories, dsr.cat.categories)

    np.testing.assert_array_equal(
        pdsr.cat.codes.values, result.cat.codes.to_numpy()
    )

    assert str(result) == str(pdsr)
    with dask.config.set({"dataframe.convert-string": False}):
        df = DataFrame()
        df["a"] = ["xyz", "abc", "def"] * 10

        pdf = df.to_pandas()
        cddf = dask_cudf.from_cudf(df, 1)
        cddf["b"] = cddf["a"].astype("category")

        ddf = dd.from_pandas(pdf, 1)
        ddf["b"] = ddf["a"].astype("category")

        assert_eq(ddf._meta_nonempty["b"], cddf._meta_nonempty["b"])

        with pytest.raises(NotImplementedError):
            cddf["b"].cat.categories

        with pytest.raises(NotImplementedError):
            ddf["b"].cat.categories

        cddf = cddf.categorize()
        ddf = ddf.categorize()

        assert_eq(ddf["b"].cat.categories, cddf["b"].cat.categories)
        assert_eq(ddf["b"].cat.ordered, cddf["b"].cat.ordered)


@pytest.mark.parametrize("data", [data_cat_1()])
def test_categorical_compare_unordered(data):
    cat = data.copy()
    pdsr = pd.Series(cat)
    sr = Series(cat)
    dsr = dask_cudf.from_cudf(sr, npartitions=2)

    # Test equality
    out = dsr == dsr
    assert out.dtype == np.bool_
    assert np.all(out.compute())
    assert np.all(pdsr == pdsr)

    # Test inequality
    out = dsr != dsr
    assert not np.any(out.compute())
    assert not np.any(pdsr != pdsr)

    assert not dsr.cat.ordered
    assert not pdsr.cat.ordered

    with pytest.raises(
        (TypeError, ValueError),
        match="Unordered Categoricals can only compare equality or not",
    ):
        pdsr < pdsr

    with pytest.raises(
        (TypeError, ValueError),
        match=(
            r"The only binary operations supported by unordered categorical "
            "columns are equality and inequality."
        ),
    ):
        dsr < dsr


@pytest.mark.parametrize("data", [data_cat_3()])
def test_categorical_compare_ordered(data):
    cat1 = data[0].copy()
    cat2 = data[1].copy()
    pdsr1 = pd.Series(cat1)
    pdsr2 = pd.Series(cat2)
    sr1 = Series(cat1)
    sr2 = Series(cat2)
    dsr1 = dask_cudf.from_cudf(sr1, npartitions=2)
    dsr2 = dask_cudf.from_cudf(sr2, npartitions=2)

    # Test equality
    out = dsr1 == dsr1
    assert out.dtype == np.bool_
    assert np.all(out.compute().to_numpy())
    assert np.all(pdsr1 == pdsr1)

    # Test inequality
    out = dsr1 != dsr1
    assert not np.any(out.compute().to_numpy())
    assert not np.any(pdsr1 != pdsr1)

    assert dsr1.cat.ordered
    assert pdsr1.cat.ordered

    # Test ordered operators
    np.testing.assert_array_equal(
        pdsr1 < pdsr2, (dsr1 < dsr2).compute().to_numpy()
    )
    np.testing.assert_array_equal(
        pdsr1 > pdsr2, (dsr1 > dsr2).compute().to_numpy()
    )


#############################################################################
#                        String Accessor                                    #
#############################################################################


def data_str_1():
    return pd.Series(["20190101", "20190102", "20190103"])


@pytest.mark.parametrize("data", [data_str_1()])
def test_string_slicing(data):
    with dask.config.set({"dataframe.convert-string": False}):
        pdsr = pd.Series(data.copy())
        sr = Series(pdsr)
        dsr = dask_cudf.from_cudf(sr, npartitions=2)
        base = pdsr.str.slice(0, 4)
        test = dsr.str.slice(0, 4).compute()
        assert_eq(base, test)


def test_categorical_categories():
    df = DataFrame(
        {"a": ["a", "b", "c", "d", "e", "e", "a", "d"], "b": range(8)}
    )
    df["a"] = df["a"].astype("category")
    pdf = df.to_pandas(nullable=False)

    ddf = dask_cudf.from_cudf(df, 2)
    dpdf = dd.from_pandas(pdf, 2)

    dd.assert_eq(
        ddf.a.cat.categories.to_series().to_pandas(nullable=False),
        dpdf.a.cat.categories.to_series(),
        check_index=False,
    )


def test_categorical_as_known():
    df = dask_cudf.from_cudf(DataFrame({"col_1": [0, 1, 2, 3]}), npartitions=2)
    df["col_1"] = df["col_1"].astype("category")
    actual = df["col_1"].cat.as_known()

    pdf = dd.from_pandas(pd.DataFrame({"col_1": [0, 1, 2, 3]}), npartitions=2)
    pdf["col_1"] = pdf["col_1"].astype("category")
    expected = pdf["col_1"].cat.as_known()

    # Note: Categories may be ordered differently in
    # cudf and pandas. Therefore, we need to compare
    # the global set of categories (before and after
    # calling `compute`), then we need to check that
    # the initial order of rows was preserved.
    assert set(expected.cat.categories) == set(
        actual.cat.categories.to_numpy()
    )
    assert set(expected.compute().cat.categories) == set(
        actual.compute().cat.categories.to_numpy()
    )
    dd.assert_eq(expected, actual, check_categorical=False)


def test_str_slice():
    df = DataFrame({"a": ["abc,def,123", "xyz,hi,bye"]})

    ddf = dask_cudf.from_cudf(df, 1)
    pdf = df.to_pandas()

    dd.assert_eq(
        pdf.a.str.split(",", expand=True, n=1),
        ddf.a.str.split(",", expand=True, n=1),
    )
    dd.assert_eq(
        pdf.a.str.split(",", expand=True, n=2),
        ddf.a.str.split(",", expand=True, n=2),
    )


#############################################################################
#                              List Accessor                                #
#############################################################################


def data_test_1():
    return [list(range(100)) for _ in range(100)]


def data_test_2():
    return [list(i for _ in range(i)) for i in range(500)]


def data_test_non_numeric():
    return [list(chr(97 + i % 20) for _ in range(i)) for i in range(500)]


def data_test_nested():
    return [
        list(list(y for y in range(x % 5)) for x in range(i))
        for i in range(40)
    ]


def data_test_sort():
    return [[1, 2, 3, 1, 2, 5] for _ in range(20)]


@pytest.mark.parametrize(
    "data",
    [
        [[]],
        [[[]]],
        [[0]],
        [[0, 1]],
        [[0, 1], [2, 3]],
        [[[0, 1], [2]], [[3, 4]]],
        [[None]],
        [[[None]]],
        [[None], None],
        [[1, None], [1]],
        [[1, None], None],
        [[[1, None], None], None],
    ],
)
def test_create_list_series(data):
    expect = pd.Series(data)
    ds_got = dask_cudf.from_cudf(Series(data), 4)
    assert_eq(expect, ds_got.compute())


@pytest.mark.parametrize(
    "data",
    [data_test_1(), data_test_2(), data_test_non_numeric()],
)
def test_unique(data):
    expect = Series(data).list.unique()
    ds = dask_cudf.from_cudf(Series(data), 5)
    assert_eq(expect, ds.list.unique().compute())


@pytest.mark.parametrize(
    "data",
    [data_test_2(), data_test_non_numeric()],
)
def test_len(data):
    expect = Series(data).list.len()
    ds = dask_cudf.from_cudf(Series(data), 5)
    assert_eq(expect, ds.list.len().compute())


@pytest.mark.parametrize(
    "data, search_key",
    [(data_test_2(), 1)],
)
def test_contains(data, search_key):
    expect = Series(data).list.contains(search_key)
    ds = dask_cudf.from_cudf(Series(data), 5)
    assert_eq(expect, ds.list.contains(search_key).compute())


@pytest.mark.parametrize(
    "data, index",
    [
        (data_test_1(), 1),
        (data_test_2(), 2),
    ],
)
def test_get(data, index):
    expect = Series(data).list.get(index)
    ds = dask_cudf.from_cudf(Series(data), 5)
    assert_eq(expect, ds.list.get(index).compute())


@pytest.mark.parametrize(
    "data",
    [data_test_1(), data_test_2(), data_test_nested()],
)
def test_leaves(data):
    expect = Series(data).list.leaves
    ds = dask_cudf.from_cudf(Series(data), 5)
    got = ds.list.leaves.compute().reset_index(drop=True)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data, list_indices, expectation",
    [
        (
            data_test_1(),
            [[0, 1] for _ in range(len(data_test_1()))],
            does_not_raise(),
        ),
        (data_test_2(), [[0]], pytest.raises(ValueError)),
    ],
)
def test_take(data, list_indices, expectation):
    with expectation:
        expect = Series(data).list.take(list_indices)

    if expectation == does_not_raise():
        ds = dask_cudf.from_cudf(Series(data), 5)
        assert_eq(expect, ds.list.take(list_indices).compute())


@pytest.mark.parametrize(
    "data, ascending, na_position, ignore_index",
    [
        (data_test_sort(), True, "first", False),
        (data_test_sort(), False, "last", True),
    ],
)
def test_sorting(data, ascending, na_position, ignore_index):
    expect = Series(data).list.sort_values(
        ascending=ascending, na_position=na_position, ignore_index=ignore_index
    )
    got = (
        dask_cudf.from_cudf(Series(data), 5)
        .list.sort_values(
            ascending=ascending,
            na_position=na_position,
            ignore_index=ignore_index,
        )
        .compute()
        .reset_index(drop=True)
    )
    assert_eq(expect, got)


#############################################################################
#                            Struct Accessor                                #
#############################################################################
struct_accessor_data_params = [
    [{"a": 5, "b": 10}, {"a": 3, "b": 7}, {"a": -3, "b": 11}],
    [{"a": None, "b": 1}, {"a": None, "b": 0}, {"a": -3, "b": None}],
    [{"a": 1, "b": 2}],
    [{"a": 1, "b": 3, "c": 4}],
]


@pytest.mark.parametrize(
    "data",
    struct_accessor_data_params,
)
def test_create_struct_series(data):
    expect = pd.Series(data)
    ds_got = dask_cudf.from_cudf(Series(data), 2)
    assert_eq(expect, ds_got.compute())


@pytest.mark.parametrize(
    "data",
    struct_accessor_data_params,
)
def test_struct_field_str(data):
    for test_key in ["a", "b"]:
        expect = Series(data).struct.field(test_key)
        ds_got = dask_cudf.from_cudf(Series(data), 2).struct.field(test_key)
        assert_eq(expect, ds_got.compute())


@pytest.mark.parametrize(
    "data",
    struct_accessor_data_params,
)
def test_struct_field_integer(data):
    for test_key in [0, 1]:
        expect = Series(data).struct.field(test_key)
        ds_got = dask_cudf.from_cudf(Series(data), 2).struct.field(test_key)
        assert_eq(expect, ds_got.compute())


@pytest.mark.parametrize(
    "data",
    struct_accessor_data_params,
)
def test_dask_struct_field_Key_Error(data):
    got = dask_cudf.from_cudf(Series(data), 2)

    with pytest.raises(KeyError):
        got.struct.field("notakey").compute()


@pytest.mark.parametrize(
    "data",
    struct_accessor_data_params,
)
def test_dask_struct_field_Int_Error(data):
    got = dask_cudf.from_cudf(Series(data), 2)

    with pytest.raises(IndexError):
        got.struct.field(1000).compute()


@pytest.mark.parametrize(
    "data",
    [
        [{}, {}, {}],
        [{"a": 100, "b": "abc"}, {"a": 42, "b": "def"}, {"a": -87, "b": ""}],
        [{"a": [1, 2, 3], "b": {"c": 101}}, {"a": [4, 5], "b": {"c": 102}}],
    ],
)
def test_struct_explode(data):
    expect = Series(data).struct.explode()
    got = dask_cudf.from_cudf(Series(data), 2).struct.explode().compute()
    # Output index will not agree for >1 partitions
    if len(data[0]) == 0:
        # expect.columns is an empty RangeIndex, got is an empty Index[object]
        got.columns = expect.columns
    assert_eq(expect, got)


def test_tz_localize():
    data = Series(date_range("2000-04-01", "2000-04-03", freq="h"))
    expect = data.dt.tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="NaT"
    )
    got = dask_cudf.from_cudf(data, 2).dt.tz_localize(
        "US/Eastern", ambiguous="NaT", nonexistent="NaT"
    )
    dd.assert_eq(expect, got)

    expect = expect.dt.tz_localize(None)
    got = got.dt.tz_localize(None)
    dd.assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        date_range("2000-04-01", "2000-04-03", freq="h").tz_localize("UTC"),
        date_range("2000-04-01", "2000-04-03", freq="h").tz_localize(
            "US/Eastern"
        ),
    ],
)
def test_tz_convert(data):
    expect = Series(data).dt.tz_convert("US/Pacific")
    got = dask_cudf.from_cudf(Series(data), 2).dt.tz_convert("US/Pacific")
    dd.assert_eq(expect, got)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import operator
from contextlib import contextmanager

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal, expect_warning_if


@pytest.mark.parametrize("arg", [[True, False, True], [True, True, True]])
@pytest.mark.parametrize("value", [0, -1])
def test_dataframe_setitem_bool_mask_scalar(arg, value):
    df = pd.DataFrame({"a": [1, 2, 3]})
    gdf = cudf.from_pandas(df)

    df[arg] = value
    gdf[arg] = value
    assert_eq(df, gdf)


def test_dataframe_setitem_scalar_bool():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df[[True, False, True]] = pd.DataFrame({"a": [-1, -2]})

    gdf = cudf.DataFrame({"a": [1, 2, 3]})
    gdf[[True, False, True]] = cudf.DataFrame({"a": [-1, -2]})
    assert_eq(df, gdf)


@pytest.mark.parametrize(
    "df",
    [pd.DataFrame({"a": [1, 2, 3]}), pd.DataFrame({"a": ["x", "y", "z"]})],
)
@pytest.mark.parametrize("arg", [["a"], "a", "b"])
@pytest.mark.parametrize(
    "value", [-10, pd.DataFrame({"a": [-1, -2, -3]}), "abc"]
)
def test_dataframe_setitem_columns(df, arg, value):
    gdf = cudf.from_pandas(df)
    cudf_replace_value = value

    if isinstance(cudf_replace_value, pd.DataFrame):
        cudf_replace_value = cudf.from_pandas(value)

    df[arg] = value
    gdf[arg] = cudf_replace_value
    assert_eq(df, gdf, check_dtype=False)


@pytest.mark.parametrize(
    "value",
    [
        pd.DataFrame({"0": [-1, -2, -3], "1": [-0, -10, -1]}),
        10,
        "rapids",
        0.32234,
        np.datetime64(1324232423423342, "ns"),
        np.timedelta64(34234324234324234, "ns"),
    ],
)
def test_dataframe_setitem_new_columns(value):
    df = pd.DataFrame({"a": [1, 2, 3]})
    arg = ["b", "c"]
    gdf = cudf.from_pandas(df)
    cudf_replace_value = value

    if isinstance(cudf_replace_value, pd.DataFrame):
        cudf_replace_value = cudf.from_pandas(value)

    df[arg] = value
    gdf[arg] = cudf_replace_value
    assert_eq(df, gdf, check_dtype=True)


def test_series_setitem_index():
    df = pd.DataFrame(
        data={"b": [-1, -2, -3], "c": [1, 2, 3]}, index=[1, 2, 3]
    )

    df["b"] = pd.Series(data=[12, 11, 10], index=[3, 2, 1])
    gdf = cudf.DataFrame(
        data={"b": [-1, -2, -3], "c": [1, 2, 3]}, index=[1, 2, 3]
    )
    gdf["b"] = cudf.Series(data=[12, 11, 10], index=[3, 2, 1])
    assert_eq(df, gdf, check_dtype=False)


@pytest.mark.xfail(reason="Copy-on-Write should make a copy")
@pytest.mark.parametrize(
    "index",
    [
        pd.MultiIndex.from_frame(
            pd.DataFrame({"b": [3, 2, 1], "c": ["a", "b", "c"]})
        ),
        ["a", "b", "c"],
    ],
)
def test_setitem_dataframe_series_inplace(index):
    gdf = cudf.DataFrame({"a": [1, 2, 3]}, index=index)
    expected = gdf.copy()
    gdf["a"].replace(1, 500, inplace=True)

    assert_eq(expected, gdf)


def test_listcol_setitem_retain_dtype():
    df = cudf.DataFrame(
        {"a": cudf.Series([["a", "b"], []]), "b": [1, 2], "c": [123, 321]}
    )
    df1 = df.head(0)
    # Performing a setitem on `b` triggers a `column.column_empty` call
    # which tries to create an empty ListColumn.
    df1["b"] = df1["c"]
    # Performing a copy to trigger a copy dtype which is obtained by accessing
    # `ListColumn.children` that would have been corrupted in previous call
    # prior to this fix: https://github.com/rapidsai/cudf/pull/10151/
    df2 = df1.copy()
    assert df2["a"].dtype == df["a"].dtype


def test_setitem_reset_label_dtype():
    result = cudf.DataFrame({1: [2]})
    expected = pd.DataFrame({1: [2]})
    result["a"] = [2]
    expected["a"] = [2]
    assert_eq(result, expected)


def test_dataframe_assign_scalar_to_empty_series():
    expected = pd.DataFrame({"a": []})
    actual = cudf.DataFrame({"a": []})
    expected.a = 0
    actual.a = 0
    assert_eq(expected, actual)


def test_dataframe_assign_cp_np_array():
    m, n = 5, 3
    rng = cp.random.default_rng(0)
    cp_ndarray = rng.standard_normal(size=(m, n))
    pdf = pd.DataFrame({f"f_{i}": range(m) for i in range(n)})
    gdf = cudf.DataFrame({f"f_{i}": range(m) for i in range(n)})
    pdf[[f"f_{i}" for i in range(n)]] = cp.asnumpy(cp_ndarray)
    gdf[[f"f_{i}" for i in range(n)]] = cp_ndarray

    assert_eq(pdf, gdf)


def test_dataframe_setitem_cupy_array():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.standard_normal(size=(10, 2)))
    gdf = cudf.from_pandas(pdf)

    gpu_array = cp.array([True, False] * 5)
    pdf[gpu_array.get()] = 1.5
    gdf[gpu_array] = 1.5

    assert_eq(pdf, gdf)


def test_setitem_datetime():
    df = cudf.DataFrame({"date": pd.date_range("20010101", "20010105").values})
    assert df.date.dtype.kind == "M"


@pytest.mark.parametrize("scalar", ["a", None])
def test_string_set_scalar(scalar):
    pdf = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
        }
    )
    gdf = cudf.DataFrame(pdf)

    pdf["b"] = "a"
    gdf["b"] = "a"

    assert_eq(pdf["b"], gdf["b"])
    assert_eq(pdf, gdf)


def test_dataframe_cow_slice_setitem():
    df = cudf.DataFrame({"a": [10, 11, 12, 13, 14], "b": [20, 30, 40, 50, 60]})
    slice_df = df[1:4]

    assert_eq(
        slice_df,
        cudf.DataFrame(
            {"a": [11, 12, 13], "b": [30, 40, 50]}, index=[1, 2, 3]
        ),
    )

    slice_df["a"][2] = 1111

    assert_eq(
        slice_df,
        cudf.DataFrame(
            {"a": [11, 1111, 13], "b": [30, 40, 50]}, index=[1, 2, 3]
        ),
    )
    assert_eq(
        df,
        cudf.DataFrame({"a": [10, 11, 12, 13, 14], "b": [20, 30, 40, 50, 60]}),
    )


def test_multiindex_row_shape():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(0, 5)))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex([["a", "b", "c"]], [[0]])
    pdfIndex.names = ["alpha"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)

    assert_exceptions_equal(
        lfunc=operator.setitem,
        rfunc=operator.setitem,
        lfunc_args_and_kwargs=([], {"a": pdf, "b": "index", "c": pdfIndex}),
        rfunc_args_and_kwargs=([], {"a": gdf, "b": "index", "c": gdfIndex}),
    )


def test_multiindex_column_shape():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(5, 0)))
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex([["a", "b", "c"]], [[0]])
    pdfIndex.names = ["alpha"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)

    assert_exceptions_equal(
        lfunc=operator.setitem,
        rfunc=operator.setitem,
        lfunc_args_and_kwargs=([], {"a": pdf, "b": "columns", "c": pdfIndex}),
        rfunc_args_and_kwargs=([], {"a": gdf, "b": "columns", "c": gdfIndex}),
    )


@contextmanager
def expect_pandas_performance_warning(idx):
    with expect_warning_if(
        (not isinstance(idx[0], tuple) and len(idx) > 2)
        or (isinstance(idx[0], tuple) and len(idx[0]) > 2),
        pd.errors.PerformanceWarning,
    ):
        yield


@pytest.mark.parametrize(
    "query",
    [
        ("a", "store", "clouds", "fire"),
        ("a", "store", "storm", "smoke"),
        ("a", "store"),
        ("b", "house"),
        ("a", "store", "storm"),
        ("a",),
        ("c", "forest", "clear"),
    ],
)
def test_multiindex_columns(query):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdf = pdf.T
    gdf = cudf.from_pandas(pdf)
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    gdfIndex = cudf.from_pandas(pdfIndex)
    assert_eq(pdfIndex, gdfIndex)
    pdf.columns = pdfIndex
    gdf.columns = gdfIndex
    # The index is unsorted, which makes things slow but is fine for testing.
    with expect_pandas_performance_warning(query):
        expected = pdf[query]
    got = gdf[query]
    assert_eq(expected, got)


@pytest.mark.xfail(
    reason="https://github.com/pandas-dev/pandas/issues/43351",
)
def test_multicolumn_set_item():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdfIndex = pd.MultiIndex(
        [
            ["a", "b", "c"],
            ["house", "store", "forest"],
            ["clouds", "clear", "storm"],
            ["fire", "smoke", "clear"],
            [
                np.datetime64("2001-01-01", "ns"),
                np.datetime64("2002-01-01", "ns"),
                np.datetime64("2003-01-01", "ns"),
            ],
        ],
        [
            [0, 0, 0, 0, 1, 1, 2],
            [1, 1, 1, 1, 0, 0, 2],
            [0, 0, 2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2, 0, 1],
            [1, 0, 1, 2, 0, 0, 1],
        ],
    )
    pdfIndex.names = ["alpha", "location", "weather", "sign", "timestamp"]
    pdf = pdf.T
    pdf.columns = pdfIndex
    gdf = cudf.from_pandas(pdf)
    pdf["d"] = [1, 2, 3, 4, 5]
    gdf["d"] = [1, 2, 3, 4, 5]
    assert_eq(pdf, gdf)


def test_dataframe_basic():
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame()

    # Populate with cuda memory
    df["keys"] = np.arange(10, dtype=np.float64)
    np.testing.assert_equal(df["keys"].to_numpy(), np.arange(10))
    assert len(df) == 10

    # Populate with numpy array
    rnd_vals = rng.random(10)
    df["vals"] = rnd_vals
    np.testing.assert_equal(df["vals"].to_numpy(), rnd_vals)
    assert len(df) == 10
    assert tuple(df.columns) == ("keys", "vals")

    # Make another dataframe
    df2 = cudf.DataFrame()
    df2["keys"] = np.array([123], dtype=np.float64)
    df2["vals"] = np.array([321], dtype=np.float64)

    # Concat
    df = cudf.concat([df, df2])
    assert len(df) == 11

    hkeys = np.asarray([*np.arange(10, dtype=np.float64).tolist(), 123])
    hvals = np.asarray([*rnd_vals.tolist(), 321])

    np.testing.assert_equal(df["keys"].to_numpy(), hkeys)
    np.testing.assert_equal(df["vals"].to_numpy(), hvals)

    # As matrix
    mat = df.to_numpy()

    expect = np.vstack([hkeys, hvals]).T

    np.testing.assert_equal(mat, expect)

    # test dataframe with tuple name
    df_tup = cudf.DataFrame()
    data = np.arange(10)
    df_tup[(1, "foobar")] = data
    np.testing.assert_equal(data, df_tup[(1, "foobar")].to_numpy())

    df = cudf.DataFrame(pd.DataFrame({"a": [1, 2, 3], "c": ["a", "b", "c"]}))
    pdf = pd.DataFrame(pd.DataFrame({"a": [1, 2, 3], "c": ["a", "b", "c"]}))
    assert_eq(df, pdf)

    gdf = cudf.DataFrame({"id": [0, 1], "val": [None, None]})
    gdf["val"] = gdf["val"].astype("int")

    assert gdf["val"].isnull().all()


def test_dataframe_column_add_drop_via_setitem():
    df = cudf.DataFrame()
    data = np.asarray(range(10))
    df["a"] = data
    df["b"] = data
    assert tuple(df.columns) == ("a", "b")
    del df["a"]
    assert tuple(df.columns) == ("b",)
    df["c"] = data
    assert tuple(df.columns) == ("b", "c")
    df["a"] = data
    assert tuple(df.columns) == ("b", "c", "a")


def test_dataframe_column_set_via_attr():
    data_0 = np.asarray([0, 2, 4, 5])
    data_1 = np.asarray([1, 4, 2, 3])
    data_2 = np.asarray([2, 0, 3, 0])
    df = cudf.DataFrame({"a": data_0, "b": data_1, "c": data_2})

    for i in range(10):
        df.c = df.a
        assert assert_eq(df.c, df.a, check_names=False)
        assert tuple(df.columns) == ("a", "b", "c")

        df.c = df.b
        assert assert_eq(df.c, df.b, check_names=False)
        assert tuple(df.columns) == ("a", "b", "c")


@pytest.mark.parametrize("nelem", [0, 10])
def test_dataframe_astype(nelem):
    df = cudf.DataFrame()
    data = np.asarray(range(nelem), dtype=np.int32)
    df["a"] = data
    assert df["a"].dtype is np.dtype(np.int32)
    df["b"] = df["a"].astype(np.float32)
    assert df["b"].dtype is np.dtype(np.float32)
    np.testing.assert_equal(df["a"].to_numpy(), df["b"].to_numpy())


def test_dataframe_add_col_to_object_dataframe():
    # Test for adding column to an empty object dataframe
    cols = ["a", "b", "c"]
    df = pd.DataFrame(columns=cols, dtype="str")

    data = {k: ["a"] for k in cols}

    gdf = cudf.DataFrame(data)
    gdf = gdf[:0]

    assert gdf.dtypes.equals(df.dtypes)
    gdf["a"] = [1]
    df["a"] = [10]
    assert gdf.dtypes.equals(df.dtypes)
    gdf["b"] = [1.0]
    df["b"] = [10.0]
    assert gdf.dtypes.equals(df.dtypes)


def test_dataframe_append_empty():
    pdf = pd.DataFrame(
        {
            "key": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
            "value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        }
    )
    gdf = cudf.DataFrame(pdf)

    gdf["newcol"] = 100
    pdf["newcol"] = 100

    assert len(gdf["newcol"]) == len(pdf)
    assert len(pdf["newcol"]) == len(pdf)
    assert_eq(gdf, pdf)


def test_dataframe_setitem_from_masked_object():
    rng = np.random.default_rng(seed=0)
    ary = rng.standard_normal(100)
    mask = np.zeros(100, dtype=bool)
    mask[:20] = True
    rng.shuffle(mask)
    ary[mask] = np.nan

    test1_null = cudf.Series(ary, nan_as_null=True)
    assert test1_null.null_count == 20
    test1_nan = cudf.Series(ary, nan_as_null=False)
    assert test1_nan.null_count == 0

    test2_null = cudf.DataFrame(pd.DataFrame({"a": ary}), nan_as_null=True)
    assert test2_null["a"].null_count == 20
    test2_nan = cudf.DataFrame(pd.DataFrame({"a": ary}), nan_as_null=False)
    assert test2_nan["a"].null_count == 0

    gpu_ary = cp.asarray(ary)
    test3_null = cudf.Series(gpu_ary, nan_as_null=True)
    assert test3_null.null_count == 20
    test3_nan = cudf.Series(gpu_ary, nan_as_null=False)
    assert test3_nan.null_count == 0

    test4 = cudf.DataFrame()
    lst = [1, 2, None, 4, 5, 6, None, 8, 9]
    test4["lst"] = lst
    assert test4["lst"].null_count == 2


def test_dataframe_append_to_empty():
    pdf = pd.DataFrame()
    pdf["a"] = []
    pdf["a"] = pdf["a"].astype("str")
    pdf["b"] = [1, 2, 3]

    gdf = cudf.DataFrame()
    gdf["a"] = []
    gdf["a"] = gdf["a"].astype("str")
    gdf["b"] = [1, 2, 3]

    assert_eq(gdf, pdf)


def test_dataframe_setitem_index_len1():
    gdf = cudf.DataFrame()
    gdf["a"] = [1]
    gdf["b"] = gdf.index._column

    np.testing.assert_equal(gdf.b.to_numpy(), [0])


def test_empty_dataframe_setitem_df():
    gdf1 = cudf.DataFrame()
    gdf2 = cudf.DataFrame({"a": [1, 2, 3, 4, 5]})
    gdf1["a"] = gdf2["a"]
    assert_eq(gdf1, gdf2)


@pytest.mark.parametrize("nrows", [0, 3])
def test_nonmatching_index_setitem(nrows):
    rng = np.random.default_rng(seed=0)

    gdf = cudf.DataFrame()
    gdf["a"] = rng.integers(2147483647, size=nrows)
    gdf["b"] = rng.integers(2147483647, size=nrows)
    gdf = gdf.set_index("b")

    test_values = rng.integers(2147483647, size=nrows)
    gdf["c"] = test_values
    assert len(test_values) == len(gdf["c"])
    gdf_series = cudf.Series(test_values, index=gdf.index, name="c")
    assert_eq(gdf["c"].to_pandas(), gdf_series.to_pandas())


def test_dataframe_assignment():
    pdf = pd.DataFrame()
    for col in "abc":
        pdf[col] = np.array([0, 1, 1, -2, 10])
    gdf = cudf.DataFrame(pdf)
    gdf[gdf < 0] = 999
    pdf[pdf < 0] = 999
    assert_eq(gdf, pdf)


@pytest.mark.parametrize("a", [[], ["123"]])
@pytest.mark.parametrize("b", ["123", ["123"]])
@pytest.mark.parametrize(
    "misc_data",
    ["123", ["123"] * 20, 123, [1, 2, 0.8, 0.9] * 50, 0.9, 0.00001],
)
@pytest.mark.parametrize("non_list_data", [123, "abc", "zyx", "rapids", 0.8])
def test_create_dataframe_cols_empty_data(a, b, misc_data, non_list_data):
    expected = pd.DataFrame({"a": a})
    actual = cudf.DataFrame(expected)
    expected["b"] = b
    actual["b"] = b
    assert_eq(actual, expected)

    expected = pd.DataFrame({"a": []})
    actual = cudf.DataFrame(expected)
    expected["b"] = misc_data
    actual["b"] = misc_data
    assert_eq(actual, expected)

    expected = pd.DataFrame({"a": a})
    actual = cudf.DataFrame(expected)
    expected["b"] = non_list_data
    actual["b"] = non_list_data
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "list_input",
    [
        pytest.param([1, 2, 3, 4], id="smaller"),
        pytest.param([1, 2, 3, 4, 5, 6], id="larger"),
    ],
)
@pytest.mark.parametrize(
    "key",
    [
        pytest.param("list_test", id="new_column"),
        pytest.param("id", id="existing_column"),
    ],
)
def test_setitem_diff_size_list(list_input, key):
    gdf = cudf.DataFrame(range(5))
    with pytest.raises(
        ValueError, match="All columns must be of equal length"
    ):
        gdf[key] = list_input


@pytest.mark.parametrize(
    "data, index",
    [
        [[1, 2, 3, 4], None],
        [[1, 2, 3, 4, 5, 6], None],
        [[1, 2, 3], [4, 5, 6]],
    ],
)
@pytest.mark.parametrize("klass", [pd.Series, cudf.Series])
@pytest.mark.parametrize(
    "key",
    [
        pytest.param("list_test", id="new_column"),
        pytest.param("id", id="existing_column"),
    ],
)
def test_setitem_diff_size_series(klass, data, index, key):
    gdf = cudf.DataFrame(range(5))
    pdf = gdf.to_pandas()

    series_input = klass(data, index=index)
    pandas_input = series_input
    if isinstance(pandas_input, cudf.Series):
        pandas_input = pandas_input.to_pandas()

    expect = pdf
    expect[key] = pandas_input

    got = gdf
    got[key] = series_input

    # Pandas uses NaN and typecasts to float64 if there's missing values on
    # alignment, so need to typecast to float64 for equality comparison
    expect = expect.astype("float64")
    got = got.astype("float64")

    assert_eq(expect, got)


def test_tupleize_cols_False_set():
    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()
    pdf[("a", "b")] = [1]
    gdf[("a", "b")] = [1]
    assert_eq(pdf, gdf)
    assert_eq(pdf.columns, gdf.columns)


@pytest.mark.parametrize(
    "col_data",
    [
        range(5),
        ["a", "b", "x", "y", "z"],
        [1.0, 0.213, 0.34332],
        ["a"],
        [1],
        [0.2323],
        [],
    ],
)
@pytest.mark.parametrize(
    "assign_val",
    [
        1,
        2,
        np.array(2),
        cp.array(2),
        0.32324,
        np.array(0.34248),
        cp.array(0.34248),
        "abc",
        np.array("abc", dtype="object"),
        np.array("abc", dtype="str"),
        np.array("abc"),
        None,
    ],
)
def test_dataframe_assign_scalar(col_data, assign_val):
    pdf = pd.DataFrame({"a": col_data})
    gdf = cudf.DataFrame({"a": col_data})

    pdf["b"] = (
        cp.asnumpy(assign_val)
        if isinstance(assign_val, cp.ndarray)
        else assign_val
    )
    gdf["b"] = assign_val
    # Assigning None in pandas creates an object column
    # which cuDF doesn't support
    if assign_val is None:
        pdf["b"] = pdf["b"].astype(gdf["b"].dtype)
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "col_data",
    [
        1,
        2,
        np.array(2),
        cp.array(2),
        0.32324,
        np.array(0.34248),
        cp.array(0.34248),
        "abc",
        np.array("abc", dtype="object"),
        np.array("abc", dtype="str"),
        np.array("abc"),
        None,
    ],
)
@pytest.mark.parametrize(
    "assign_val",
    [
        1,
        2,
        np.array(2),
        cp.array(2),
        0.32324,
        np.array(0.34248),
        cp.array(0.34248),
        "abc",
        np.array("abc", dtype="object"),
        np.array("abc", dtype="str"),
        np.array("abc"),
        None,
    ],
)
def test_dataframe_assign_scalar_with_scalar_cols(col_data, assign_val):
    pdf = pd.DataFrame(
        {
            "a": cp.asnumpy(col_data)
            if isinstance(col_data, cp.ndarray)
            else col_data
        },
        index=["dummy_mandatory_index"],
    )
    gdf = cudf.DataFrame({"a": col_data}, index=["dummy_mandatory_index"])

    pdf["b"] = (
        cp.asnumpy(assign_val)
        if isinstance(assign_val, cp.ndarray)
        else assign_val
    )
    gdf["b"] = assign_val
    # Assigning None in pandas creates an object column
    # which cuDF doesn't support
    if col_data is None:
        pdf["a"] = pdf["a"].astype(gdf["a"].dtype)
    if assign_val is None:
        pdf["b"] = pdf["b"].astype(gdf["b"].dtype)
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "invalid",
    [
        np.datetime64("NaT", "ns"),
        np.timedelta64("NaT", "ns"),
        np.datetime64("2020-01-01", "ns"),
        pd.Timestamp("2020-01-01"),
        pd.Timedelta("1 days"),
    ],
)
@pytest.mark.parametrize("key", [(0, 0), ([0], 0), (slice(0, 2), 0)])
def test_iloc_setitem_temporal_scalar_into_numeric_raises(invalid, key):
    # A datetime64/timedelta64 scalar (including NaT) cannot be assigned
    # into a numeric column; cudf must raise like pandas instead of
    # silently reinterpreting the value's integer representation.
    pdf = pd.DataFrame({"a": [1, 2, 3]}, dtype="int64")
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        lfunc=operator.setitem,
        rfunc=operator.setitem,
        lfunc_args_and_kwargs=([pdf.iloc, key, invalid],),
        rfunc_args_and_kwargs=([gdf.iloc, key, invalid],),
    )


@pytest.mark.parametrize(
    "invalid", [np.datetime64("NaT", "ns"), np.timedelta64("NaT", "ns")]
)
def test_loc_setitem_temporal_scalar_into_numeric_raises(invalid):
    pdf = pd.DataFrame({"a": [1, 2, 3]}, dtype="int64")
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        lfunc=operator.setitem,
        rfunc=operator.setitem,
        lfunc_args_and_kwargs=([pdf.loc, (0, "a"), invalid],),
        rfunc_args_and_kwargs=([gdf.loc, (0, "a"), invalid],),
    )


@pytest.mark.parametrize("key", [{1}, {1: 1}, ({1}, "a"), (1, {"a"})])
def test_loc_setitem_dict_or_set_indexer_disallowed(key):
    # A set/dict (or a tuple nesting one) is not a valid indexer.
    pdf = pd.DataFrame([[1, 2], [3, 4]], columns=["a", "b"])
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        lfunc=operator.setitem,
        rfunc=operator.setitem,
        lfunc_args_and_kwargs=([pdf.loc, key, 1],),
        rfunc_args_and_kwargs=([gdf.loc, key, 1],),
    )


def test_setitem_boolean_series_mask_label_aligned():
    # A boolean Series mask is aligned to the frame's index by label, not
    # applied positionally.
    pdf = pd.DataFrame({"a": [1, 4, 2, 3], "b": [5, 6, 7, 8]})
    gdf = cudf.from_pandas(pdf)
    pmask = (pdf["a"] == 4).reindex(pdf.index[::-1])
    gmask = (gdf["a"] == 4).reindex(gdf.index[::-1])
    pdf[pmask] = 0
    gdf[gmask] = 0
    assert_eq(pdf, gdf)


def test_setitem_boolean_dataframe_mask_label_aligned():
    # A reordered boolean DataFrame mask is aligned by label before
    # scattering.
    pdf = pd.DataFrame({"a": [-1, 2, -3, 4], "b": [1, -2, 3, -4]})
    gdf = cudf.from_pandas(pdf)
    pdf[pdf[::-1] < 0] = 0
    gdf[gdf[::-1] < 0] = 0
    assert_eq(pdf, gdf)


def test_loc_setitem_series_rhs_label_aligned():
    # A Series RHS in a list-column loc setitem is aligned to the target
    # rows by label rather than picked positionally per column.
    pdf = pd.DataFrame({"a": [10, 20]})
    gdf = cudf.from_pandas(pdf)
    pdf.loc[[0, 1], ["a"]] = pd.Series([200, 100], index=[1, 0])
    gdf.loc[[0, 1], ["a"]] = cudf.Series([200, 100], index=[1, 0])
    assert_eq(pdf, gdf)


def test_loc_setitem_bool_mask_dataframe_rhs_label_aligned():
    # A DataFrame RHS assigned through a boolean row mask is aligned by
    # label, so a reordered RHS leaves the frame unchanged.
    pdf = pd.DataFrame({"x": list(range(5)), "y": list(range(5))})
    gdf = cudf.from_pandas(pdf)
    pval = pdf[["x", "y"]].sort_values("x", ascending=False)
    gval = gdf[["x", "y"]].sort_values("x", ascending=False)
    pdf.loc[[True] * 5, ["x", "y"]] = pval
    gdf.loc[[True] * 5, ["x", "y"]] = gval
    assert_eq(pdf, gdf)

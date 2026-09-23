# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import decimal
import functools

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.filterwarnings(
    "ignore:obj.round has no effect with datetime:UserWarning"
)
@pytest.mark.parametrize("digits", [0, 1, 4])
def test_dataframe_round_builtin(digits):
    pdf = pd.DataFrame(
        {
            "a": [1.2234242333234, 323432.3243423, np.nan],
            "b": ["a", "b", "c"],
            "c": pd.Series([34224, 324324, 324342], dtype="datetime64[ns]"),
            "d": pd.Series([224.242, None, 2424.234324], dtype="category"),
            "e": [
                decimal.Decimal("342.3243234234242"),
                decimal.Decimal("89.32432497687622"),
                None,
            ],
        }
    )
    gdf = cudf.from_pandas(pdf, nan_as_null=False)

    expected = round(pdf, digits)
    actual = round(gdf, digits)

    assert_eq(expected, actual)


def test_bool_raises():
    assert_exceptions_equal(
        lfunc=bool,
        rfunc=bool,
        lfunc_args_and_kwargs=[[cudf.DataFrame()]],
        rfunc_args_and_kwargs=[[pd.DataFrame()]],
    )


@pytest.mark.parametrize("name", [None, "foo", 1, 1.0])
def test_dataframe_column_name(name):
    df = cudf.DataFrame({"a": [1, 2, 3]})
    pdf = df.to_pandas()

    df.columns.name = name
    pdf.columns.name = name

    assert_eq(df, pdf)
    assert_eq(df.columns.name, pdf.columns.name)


def test_dataframe_columns_set_none_raises():
    df = cudf.DataFrame({"a": [0]})
    with pytest.raises(TypeError):
        df.columns = None


@pytest.mark.parametrize(
    "columns",
    [cudf.RangeIndex(1, name="foo"), pd.RangeIndex(1, name="foo"), range(1)],
)
def test_dataframe_columns_set_rangeindex(columns):
    df = cudf.DataFrame([1], columns=["a"])
    df.columns = columns
    result = df.columns
    expected = pd.RangeIndex(1, name=getattr(columns, "name", None))
    pd.testing.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize("klass", [cudf.MultiIndex, pd.MultiIndex])
def test_dataframe_columns_set_multiindex(klass):
    columns = klass.from_arrays([[10]], names=["foo"])
    df = cudf.DataFrame([1], columns=["a"])
    df.columns = columns
    result = df.columns
    expected = pd.MultiIndex.from_arrays([[10]], names=["foo"])
    pd.testing.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize(
    "klass",
    [
        functools.partial(cudf.Index, name="foo"),
        functools.partial(cudf.Series, name="foo"),
        functools.partial(pd.Index, name="foo"),
        functools.partial(pd.Series, name="foo"),
        np.array,
    ],
)
def test_dataframe_columns_set_preserve_type(klass):
    df = cudf.DataFrame([1], columns=["a"])
    columns = klass([10], dtype="int8")
    df.columns = columns
    result = df.columns
    expected = pd.Index(
        [10], dtype="int8", name=getattr(columns, "name", None)
    )
    pd.testing.assert_index_equal(result, expected)


@pytest.mark.parametrize("column", [range(1, 2), np.array([1], dtype=np.int8)])
@pytest.mark.parametrize(
    "operation",
    [
        lambda df: df.where(df < 2, 2),
        lambda df: df.nans_to_nulls(),
        lambda df: df.isna(),
        lambda df: df.notna(),
        lambda df: abs(df),
        lambda df: -df,
        lambda df: ~df,
        lambda df: df.cumsum(),
        lambda df: df.replace(1, 2),
        lambda df: df.replace(10, 20),
        lambda df: df.clip(0, 10),
        lambda df: df.rolling(1).mean(),
        lambda df: df.interpolate(),
        lambda df: df.shift(),
        lambda df: df.sort_values(1),
        lambda df: df.round(),
        lambda df: df.rank(),
    ],
)
def test_op_preserves_column_metadata(column, operation):
    df = cudf.DataFrame([1], columns=cudf.Index(column))
    result = operation(df).columns
    expected = pd.Index(column)
    pd.testing.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2]},
        {"a": [1, 2, 3], "b": [3, 4, 5]},
        {"a": [1, 2, 3, 4], "b": [3, 4, 5, 6], "c": [1, 3, 5, 7]},
        {"a": [np.nan, 2, 3, 4], "b": [3, 4, np.nan, 6], "c": [1, 3, 5, 7]},
        {1: [1, 2, 3], 2: [3, 4, 5]},
        {"a": [1, None, None], "b": [3, np.nan, np.nan]},
        {1: ["a", "b", "c"], 2: ["q", "w", "u"]},
        {1: ["a", np.nan, "c"], 2: ["q", None, "u"]},
        {},
        {1: [], 2: [], 3: []},
        [1, 2, 3],
    ],
)
def test_axes(data):
    csr = cudf.DataFrame(data)
    psr = pd.DataFrame(data)

    expected = psr.axes
    actual = csr.axes

    for e, a in zip(expected, actual, strict=True):
        assert_eq(e, a, exact=False)


def test_iter():
    data = {"a": [1, 2, 3], "b": [4, 5, 6]}
    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)
    assert list(pdf) == list(gdf)


def test_column_assignment():
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    new_cols = ["q", "r", "s"]
    gdf.columns = new_cols
    assert list(gdf.columns) == new_cols


def test_ndim():
    pdf = pd.DataFrame({"x": range(5), "y": range(5, 10)})
    gdf = cudf.DataFrame(pdf)
    assert pdf.ndim == gdf.ndim


@pytest.mark.parametrize("names", [["abc", "def"], [1, 2], ["abc", 10]])
def test_dataframe_multiindex_column_names(names):
    arrays = [["A", "A", "B", "B"], ["one", "two", "one", "two"]]
    tuples = list(zip(*arrays, strict=True))
    index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])

    pdf = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]], columns=index)
    df = cudf.from_pandas(pdf)

    assert_eq(df, pdf)
    assert_eq(df.columns.names, pdf.columns.names)
    pdf.columns.names = names
    df.columns.names = names
    assert_eq(df, pdf)
    assert_eq(df.columns.names, pdf.columns.names)


@pytest.mark.parametrize("name", ["a", 0, None, np.nan, cudf.NA])
@pytest.mark.parametrize("contains", ["a", 0, None, np.nan, cudf.NA])
@pytest.mark.parametrize("other_names", [[], ["b", "c"], [1, 2]])
def test_dataframe_contains(name, contains, other_names):
    column_names = [name, *other_names]
    gdf = cudf.DataFrame({c: [0] for c in column_names})
    pdf = pd.DataFrame({c: [0] for c in column_names})

    assert_eq(gdf, pdf)

    assert (contains in gdf) == (contains in pdf)


@pytest.mark.parametrize(
    "data",
    [
        {"col": [{"a": 1.1}, {"a": 2.1}, {"a": 10.0}, {"a": 11.2323}, None]},
        {"a": [[{"b": 567}], None] * 10},
        {"a": [decimal.Decimal(10), decimal.Decimal(20), None]},
    ],
)
def test_dataframe_values_complex_types(data):
    gdf = cudf.DataFrame(data)
    with pytest.raises(NotImplementedError):
        gdf.values


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[100, 10, 1, 0]),
        pd.DataFrame(columns=["a", "b", "c", "d"]),
        pd.DataFrame(columns=["a", "b", "c", "d"], index=[100]),
        pd.DataFrame(
            columns=["a", "b", "c", "d"], index=[100, 10000, 2131, 133]
        ),
        pd.DataFrame({"a": [1, 2, 3], "b": ["abc", "xyz", "klm"]}),
    ],
)
def test_dataframe_size(df):
    pdf = df
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.size, gdf.size)


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(),
        pd.DataFrame(index=[100, 10, 1, 0]),
        pd.DataFrame(columns=["a", "b", "c", "d"]),
        pd.DataFrame(columns=["a", "b", "c", "d"], index=[100]),
        pd.DataFrame(
            columns=["a", "b", "c", "d"], index=[100, 10000, 2131, 133]
        ),
        pd.DataFrame({"a": [1, 2, 3], "b": ["abc", "xyz", "klm"]}),
    ],
)
def test_dataframe_empty(df):
    pdf = df
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf.empty, gdf.empty)


def test_cudf_arrow_array_error():
    df = cudf.DataFrame({"a": [1, 2, 3]})

    with pytest.raises(
        TypeError,
        match=r"Implicit conversion to a host PyArrow object via "
        "__arrow_array__ is not allowed. Consider using .to_arrow()",
    ):
        df.__arrow_array__()

    sr = cudf.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match=r"Implicit conversion to a host PyArrow object via "
        "__arrow_array__ is not allowed. Consider using .to_arrow()",
    ):
        sr.__arrow_array__()

    sr = cudf.Series(["a", "b", "c"])
    with pytest.raises(
        TypeError,
        match=r"Implicit conversion to a host PyArrow object via "
        "__arrow_array__ is not allowed. Consider using .to_arrow()",
    ):
        sr.__arrow_array__()


@pytest.mark.parametrize(
    "index",
    [
        ["a", "b", "c", "d", "e"],
        np.array(["a", "b", "c", "d", "e"]),
        pd.Index(["a", "b", "c", "d", "e"], name="name"),
    ],
)
def test_string_index(index):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(5, 5)))
    gdf = cudf.DataFrame(pdf)
    pdf.index = index
    gdf.index = index
    assert_eq(pdf, gdf)


def test_set_index_as_property():
    cdf = cudf.DataFrame()
    col1 = np.arange(10)
    col2 = np.arange(0, 20, 2)
    cdf["a"] = col1
    cdf["b"] = col2

    # Check set_index(Series)
    cdf.index = cdf["b"]

    assert_eq(cdf.index.to_numpy(), col2)

    with pytest.raises(ValueError):
        cdf.index = [list(range(10))]

    idx = pd.Index(np.arange(0, 1000, 100))
    cdf.index = idx
    assert_eq(cdf.index.to_pandas(), idx)

    df = cdf.to_pandas()
    assert_eq(df.index, idx)

    head = cdf.head().to_pandas()
    assert_eq(head.index, idx[:5])


@pytest.mark.parametrize(
    "index",
    [
        lambda: cudf.Index([1]),
        lambda: cudf.RangeIndex(1),
        lambda: cudf.MultiIndex(levels=[[0]], codes=[[0]]),
    ],
)
def test_index_assignment_no_shallow_copy(index):
    index = index()
    df = cudf.DataFrame(range(1))
    df.index = index
    assert df.index is index


def test_multiindex_df_assignment():
    pdf = pd.DataFrame({"x": [1, 2, 3]})
    gdf = cudf.from_pandas(pdf)
    pdf.index = pd.MultiIndex([["a", "b"], ["c", "d"]], [[0, 1, 0], [1, 0, 1]])
    gdf.index = cudf.MultiIndex(
        levels=[["a", "b"], ["c", "d"]], codes=[[0, 1, 0], [1, 0, 1]]
    )
    assert_eq(pdf, gdf)


def test_multiindex_index_and_columns():
    rng = np.random.default_rng(seed=0)
    gdf = cudf.DataFrame(
        {
            "x": rng.integers(0, 5, 5),
            "y": rng.integers(0, 5, 5),
        }
    )
    pdf = gdf.to_pandas()
    mi = cudf.MultiIndex(
        levels=[[0, 1, 2], [3, 4]],
        codes=[[0, 0, 1, 1, 2], [0, 1, 0, 1, 1]],
        names=["x", "y"],
    )
    gdf.index = mi
    mc = cudf.MultiIndex(
        levels=[["val"], ["mean", "min"]], codes=[[0, 0], [0, 1]]
    )
    gdf.columns = mc
    pdf.index = mi.to_pandas()
    pdf.columns = mc.to_pandas()
    assert_eq(pdf, gdf)


def test_dataframe_column_drop_via_attr():
    df = cudf.DataFrame({"a": []})

    with pytest.raises(AttributeError):
        del df.a

    assert tuple(df.columns) == tuple("a")


@pytest.mark.parametrize("nelem", [0, 100])
def test_index_astype(nelem):
    df = cudf.DataFrame()
    data = np.asarray(range(nelem), dtype=np.int32)
    df["a"] = data
    assert df.index.dtype is np.dtype(np.int64)
    df.index = df.index.astype(np.float32)
    assert df.index.dtype is np.dtype(np.float32)
    df["a"] = df["a"].astype(np.float32)
    np.testing.assert_equal(df.index.to_numpy(), df["a"].to_numpy())
    df["b"] = df["a"]
    df = df.set_index("b")
    df["a"] = df["a"].astype(np.int16)
    df.index = df.index.astype(np.int16)
    np.testing.assert_equal(df.index.to_numpy(), df["a"].to_numpy())


def test_dataframe_dtypes():
    dtypes = pd.Series(
        [np.int32, np.float32, np.float64], index=["c", "a", "b"]
    )
    df = cudf.DataFrame({k: np.ones(10, dtype=v) for k, v in dtypes.items()})
    assert df.dtypes.equals(dtypes)


def test_dataframe_dir_and_getattr():
    df = cudf.DataFrame(
        {
            "a": np.ones(10),
            "b": np.ones(10),
            "not an id": np.ones(10),
            "oop$": np.ones(10),
        }
    )
    o = dir(df)
    assert {"a", "b"}.issubset(o)
    assert "not an id" not in o
    assert "oop$" not in o

    # Getattr works
    assert df.a.equals(df["a"])
    assert df.b.equals(df["b"])
    with pytest.raises(AttributeError):
        df.not_a_column


def test_dataframe_shape():
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.3]})
    gdf = cudf.DataFrame(pdf)

    assert pdf.shape == gdf.shape


def test_dataframe_shape_empty():
    pdf = pd.DataFrame()
    gdf = cudf.DataFrame()

    assert pdf.shape == gdf.shape


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.empty,
        lambda df: df.x.empty,
        lambda df: df.x.fillna(123, limit=None, axis=None),
        lambda df: df.drop("x", axis=1, errors="raise"),
    ],
)
def test_unary_operators(func):
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    gdf = cudf.DataFrame({"x": range(10), "y": range(10)})
    p = func(pdf)
    g = func(gdf)
    assert_eq(p, g)


@pytest.mark.parametrize(
    "data",
    [
        {"A": [1, 2, 3], "B": [4, 5, 6]},
        {"A": [1.0, 2.0, 3.0], "B": [4.0, 5.0, 6.0]},
        {"A": [1, 2, 3], "B": [1.0, 2.0, 3.0]},
        {"A": np.float32(np.arange(3)), "B": np.float64(np.arange(3))},
        {"A": [1, None, 3], "B": [1, 2, None]},
        pytest.param(
            {"A": [None, None, None], "B": [None, None, None]},
            marks=pytest.mark.xfail(
                reason="Nulls not supported by values accessor"
            ),
        ),
        {"A": [], "B": []},
        pytest.param(
            {"A": [1, 2, 3], "B": ["a", "b", "c"]},
            marks=pytest.mark.xfail(
                reason="str or categorical not supported by values accessor"
            ),
        ),
        pytest.param(
            {"A": pd.Categorical(["a", "b", "c"]), "B": ["d", "e", "f"]},
            marks=pytest.mark.xfail(
                reason="str or categorical not supported by values accessor"
            ),
        ),
    ],
)
def test_df_values_property(data):
    pdf = pd.DataFrame.from_dict(data)
    gdf = cudf.DataFrame(pdf)

    pmtr = pdf.values
    gmtr = gdf.values.get()

    np.testing.assert_array_equal(pmtr, gmtr)


def test_constructor_properties():
    df = cudf.DataFrame()
    key1 = "a"
    key2 = "b"
    val1 = np.array([123], dtype=np.float64)
    val2 = np.array([321], dtype=np.float64)
    df[key1] = val1
    df[key2] = val2

    # Correct use of _constructor_sliced (for DataFrame)
    assert_eq(df[key1], df._constructor_sliced(val1, name=key1))

    # Correct use of _constructor_expanddim (for cudf.Series)
    assert_eq(df, df[key2]._constructor_expanddim({key1: val1, key2: val2}))

    # Incorrect use of _constructor_sliced (Raises for cudf.Series)
    with pytest.raises(NotImplementedError):
        df[key1]._constructor_sliced

    # Incorrect use of _constructor_expanddim (Raises for DataFrame)
    with pytest.raises(NotImplementedError):
        df._constructor_expanddim

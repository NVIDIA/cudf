# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
from contextlib import nullcontext as does_not_raise

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core.column.column import as_column
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def test_init_via_list_of_tuples():
    data = [
        (5, "cats", "jump", np.nan),
        (2, "dogs", "dig", 7.5),
        (3, "cows", "moo", -2.1, "occasionally"),
    ]

    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize("columns", [["a", "b"], pd.Series(["a", "b"])])
def test_init_via_list_of_series(columns):
    data = [pd.Series([1, 2]), pd.Series([3, 4])]

    pdf = cudf.DataFrame(data, columns=columns)
    gdf = cudf.DataFrame(data, columns=columns)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize("index", [None, [0, 1, 2]])
def test_init_with_missing_columns(index):
    """Test initialization when columns and data keys are disjoint."""
    data = {"a": [1, 2, 3], "b": [2, 3, 4]}
    columns = ["c", "d"]

    pdf = cudf.DataFrame(data, columns=columns, index=index)
    gdf = cudf.DataFrame(data, columns=columns, index=index)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize("rows", [0, 1, 2, 100])
def test_init_via_list_of_empty_tuples(rows):
    data = [()] * rows

    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "dict_of_series",
    [
        {"a": pd.Series([1.0, 2.0, 3.0])},
        {"a": pd.Series([1.0, 2.0, 3.0], index=[4, 5, 6])},
        {
            "a": pd.Series([1.0, 2.0, 3.0], index=[4, 5, 6]),
            "b": pd.Series([1.0, 2.0, 4.0], index=[1, 2, 3]),
        },
        {"a": [1, 2, 3], "b": pd.Series([1.0, 2.0, 3.0], index=[4, 5, 6])},
        {
            "a": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
            "b": pd.Series([1.0, 2.0, 4.0], index=["c", "d", "e"]),
        },
        {
            "a": pd.Series(
                ["a", "b", "c"],
                index=pd.MultiIndex.from_tuples([(1, 2), (1, 3), (2, 3)]),
            ),
            "b": pd.Series(
                ["a", " b", "d"],
                index=pd.MultiIndex.from_tuples([(1, 2), (1, 3), (2, 3)]),
            ),
        },
    ],
)
def test_init_from_series_align(dict_of_series):
    pdf = pd.DataFrame(dict_of_series)
    gdf = cudf.DataFrame(dict_of_series)

    assert_eq(pdf, gdf)

    for key in dict_of_series:
        if isinstance(dict_of_series[key], pd.Series):
            dict_of_series[key] = cudf.Series(dict_of_series[key])

    gdf = cudf.DataFrame(dict_of_series)

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    ("dict_of_series", "expectation"),
    [
        (
            {
                "a": pd.Series(["a", "b", "c"], index=[4, 4, 5]),
                "b": pd.Series(["a", "b", "c"], index=[4, 5, 6]),
            },
            pytest.raises(
                ValueError, match="Cannot align indices with non-unique values"
            ),
        ),
        (
            {
                "a": pd.Series(["a", "b", "c"], index=[4, 4, 5]),
                "b": pd.Series(["a", "b", "c"], index=[4, 4, 5]),
            },
            does_not_raise(),
        ),
    ],
)
def test_init_from_series_align_nonunique(dict_of_series, expectation):
    with expectation:
        gdf = cudf.DataFrame(dict_of_series)

    if expectation == does_not_raise():
        pdf = pd.DataFrame(dict_of_series)
        assert_eq(pdf, gdf)


def test_init_unaligned_with_index():
    pdf = pd.DataFrame(
        {
            "a": pd.Series([1.0, 2.0, 3.0], index=[4, 5, 6]),
            "b": pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3]),
        },
        index=[7, 8, 9],
    )
    gdf = cudf.DataFrame(
        {
            "a": cudf.Series([1.0, 2.0, 3.0], index=[4, 5, 6]),
            "b": cudf.Series([1.0, 2.0, 3.0], index=[1, 2, 3]),
        },
        index=[7, 8, 9],
    )

    assert_eq(pdf, gdf, check_dtype=False)


def test_init_series_list_columns_unsort():
    pseries = [
        pd.Series(i, index=["b", "a", "c"], name=str(i)) for i in range(3)
    ]
    gseries = [
        cudf.Series(i, index=["b", "a", "c"], name=str(i)) for i in range(3)
    ]
    pdf = pd.DataFrame(pseries)
    gdf = cudf.DataFrame(gseries)
    assert_eq(pdf, gdf)


@pytest.mark.parametrize("nelem", [0, 10])
@pytest.mark.parametrize("nchunks", [1, 5])
def test_from_arrow_chunked_arrays(nelem, nchunks, numeric_types_as_str):
    rng = np.random.default_rng(seed=0)
    np_list_data = [
        rng.integers(0, 100, nelem).astype(numeric_types_as_str)
        for i in range(nchunks)
    ]
    pa_chunk_array = pa.chunked_array(np_list_data)

    expect = pa_chunk_array.to_pandas()
    got = cudf.Series(pa_chunk_array)

    assert_eq(expect, got)

    np_list_data2 = [
        rng.integers(0, 100, nelem).astype(numeric_types_as_str)
        for i in range(nchunks)
    ]
    pa_chunk_array2 = pa.chunked_array(np_list_data2)
    pa_table = pa.Table.from_arrays(
        [pa_chunk_array, pa_chunk_array2], names=["a", "b"]
    )

    expect = pa_table.to_pandas()
    got = cudf.DataFrame.from_arrow(pa_table)

    assert_eq(expect, got)


def test_1row_arrow_table():
    data = [pa.array([0]), pa.array([1])]
    batch = pa.RecordBatch.from_arrays(data, ["f0", "f1"])
    table = pa.Table.from_batches([batch])

    expect = table.to_pandas()
    got = cudf.DataFrame.from_arrow(table)
    assert_eq(expect, got)


def test_arrow_handle_no_index_name():
    pdf = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    gdf = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

    gdf_arrow = gdf.to_arrow()
    pdf_arrow = pa.Table.from_pandas(pdf)
    assert pa.Table.equals(pdf_arrow, gdf_arrow)

    got = cudf.DataFrame.from_arrow(gdf_arrow)
    expect = pdf_arrow.to_pandas()
    assert_eq(expect, got)


@pytest.mark.parametrize("order", ["C", "F"])
def test_pandas_non_contiguious(order):
    arr1 = np.ones((5, 5), order=order)
    df = pd.DataFrame(arr1)
    gdf = cudf.DataFrame(df)
    assert_eq(gdf.to_pandas(), df)


def test_from_records(numeric_types_as_str):
    h_ary = np.ndarray(shape=(10, 4), dtype=numeric_types_as_str)
    rec_ary = h_ary.view(np.recarray)

    gdf = cudf.DataFrame.from_records(rec_ary, columns=["a", "b", "c", "d"])
    df = pd.DataFrame.from_records(rec_ary, columns=["a", "b", "c", "d"])
    assert isinstance(gdf, cudf.DataFrame)
    assert_eq(df, gdf)

    gdf = cudf.DataFrame.from_records(rec_ary)
    df = pd.DataFrame.from_records(rec_ary)
    assert isinstance(gdf, cudf.DataFrame)
    assert_eq(df, gdf)


@pytest.mark.parametrize(
    "index",
    [
        None,
        ["first", "second"],
        "name",
        "age",
        "weight",
        [10, 11],
        ["abc", "xyz"],
    ],
)
def test_from_records_index(index):
    rec_ary = np.array(
        [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
        dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
    )
    gdf = cudf.DataFrame.from_records(rec_ary, index=index)
    df = pd.DataFrame.from_records(rec_ary, index=index)
    assert isinstance(gdf, cudf.DataFrame)
    assert_eq(df, gdf)


def test_from_records_index_invalid_columns_raises():
    rec_ary = np.array(
        [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
        dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
    )
    index = None
    columns = ["first", "second", "third"]
    with pytest.raises(ValueError):
        cudf.DataFrame.from_records(rec_ary, columns=columns, index=index)
    with pytest.raises(ValueError):
        pd.DataFrame.from_records(rec_ary, columns=columns, index=index)


def test_dataframe_construction_from_cp_arrays():
    h_ary = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    d_ary = cp.asarray(h_ary)

    gdf = cudf.DataFrame(d_ary, columns=["a", "b", "c"])
    df = pd.DataFrame(h_ary, columns=["a", "b", "c"])
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    gdf = cudf.DataFrame(d_ary)
    df = pd.DataFrame(h_ary)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    gdf = cudf.DataFrame(d_ary, index=["a", "b"])
    df = pd.DataFrame(h_ary, index=["a", "b"])
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    gdf = cudf.DataFrame(d_ary)
    gdf = gdf.set_index(keys=0, drop=False)
    df = pd.DataFrame(h_ary)
    df = df.set_index(keys=0, drop=False)
    assert isinstance(gdf, cudf.DataFrame)
    assert gdf.index.dtype == np.dtype("int32")
    # pandas retuns a RangeIndex
    assert not isinstance(gdf.index, cudf.RangeIndex)
    gdf.index = gdf.index.astype(np.int64)
    assert_eq(df, gdf)

    gdf = cudf.DataFrame(d_ary)
    gdf = gdf.set_index(keys=1, drop=False)
    df = pd.DataFrame(h_ary)
    df = df.set_index(keys=1, drop=False)
    assert gdf.index.dtype == np.dtype("int32")
    # pandas retuns a RangeIndex
    assert not isinstance(gdf.index, cudf.RangeIndex)
    gdf.index = gdf.index.astype(np.int64)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)


def test_dataframe_cp_wrong_dimensions():
    d_ary = cp.empty((2, 3, 4), dtype=np.int32)
    with pytest.raises(
        ValueError, match="records dimension expected 1 or 2 but found: 3"
    ):
        cudf.DataFrame(d_ary)


def test_dataframe_cp_array_wrong_index():
    d_ary = cp.empty((2, 3), dtype=np.int32)

    with pytest.raises(ValueError):
        cudf.DataFrame(d_ary, index=["a"])

    with pytest.raises(TypeError):
        cudf.DataFrame(d_ary, index="a")


def test_index_in_dataframe_constructor():
    a = pd.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])
    b = cudf.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0])

    assert_eq(a, b)
    assert_eq(a.loc[4:], b.loc[4:])


@pytest.mark.parametrize("nelem", [0, 2])
def test_from_arrow(nelem, all_supported_types_as_str):
    if all_supported_types_as_str in {"category", "str"}:
        pytest.skip(f"Test not applicable with {all_supported_types_as_str}")
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {
            "a": rng.integers(0, 1000, nelem).astype(
                all_supported_types_as_str
            ),
            "b": rng.integers(0, 1000, nelem).astype(
                all_supported_types_as_str
            ),
        }
    )
    padf = pa.Table.from_pandas(
        df, preserve_index=False
    ).replace_schema_metadata(None)
    gdf = cudf.DataFrame.from_arrow(padf)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf)

    s = pa.Array.from_pandas(df.a)
    gs = cudf.Series.from_arrow(s)
    assert isinstance(gs, cudf.Series)

    # For some reason PyArrow to_pandas() converts to numpy array and has
    # better type compatibility
    np.testing.assert_array_equal(s.to_pandas(), gs.to_numpy())


def test_from_arrow_chunked_categories():
    # Verify that categories are properly deduplicated across chunked arrays.
    indices = pa.array([0, 1, 0, 1, 2, 0, None, 2])
    dictionary = pa.array(["foo", "bar", "baz"])
    dict_array = pa.DictionaryArray.from_arrays(indices, dictionary)
    chunked_array = pa.chunked_array([dict_array, dict_array])
    table = pa.table({"a": chunked_array})
    df = cudf.DataFrame.from_arrow(table)
    final_dictionary = df["a"].dtype.categories.to_arrow().to_pylist()
    assert sorted(final_dictionary) == sorted(dictionary.to_pylist())


def test_from_scalar_typing(request, all_supported_types_as_str):
    if all_supported_types_as_str in {"category", "str"}:
        pytest.skip(f"Test not applicable with {all_supported_types_as_str}")
    request.applymarker(
        pytest.mark.xfail(
            all_supported_types_as_str
            in {"timedelta64[ms]", "timedelta64[us]", "timedelta64[ns]"},
            reason=f"{all_supported_types_as_str} incorrectly results in timedelta64[s]",
        )
    )
    rng = np.random.default_rng(seed=0)
    if all_supported_types_as_str == "datetime64[ms]":
        scalar = (
            np.dtype("int64").type(rng.integers(0, 5)).astype("datetime64[ms]")
        )
    elif all_supported_types_as_str.startswith("datetime64"):
        scalar = np.datetime64("2020-01-01").astype("datetime64[ms]")
        all_supported_types_as_str = "datetime64[ms]"
    else:
        scalar = np.dtype(all_supported_types_as_str).type(rng.integers(0, 5))

    gdf = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": scalar,
        }
    )
    assert gdf["b"].dtype == np.dtype(all_supported_types_as_str)
    assert len(gdf["b"]) == len(gdf["a"])


@pytest.mark.parametrize(
    "data",
    [
        {"a": [np.nan, 1, 2], "b": [None, None, None]},
        {"a": [1, 2, np.nan, 2], "b": [np.nan, np.nan, np.nan, np.nan]},
        {
            "a": [1, 2, np.nan, 2, None],
            "b": [np.nan, np.nan, None, np.nan, np.nan],
        },
        {"a": [1, 2, 2, None, 1.1], "b": [1, 2.2, 3, None, 5]},
    ],
)
def test_dataframe_constructor_nan_as_null(data, nan_as_null):
    actual = cudf.DataFrame(data, nan_as_null=nan_as_null)

    if nan_as_null:
        assert (
            not (
                actual.astype("float").replace(
                    cudf.Series([np.nan], nan_as_null=False), cudf.Series([-1])
                )
                == -1
            )
            .any()
            .any()
        )
    else:
        actual = actual.select_dtypes(exclude=["object"])
        assert (actual.replace(np.nan, -1) == -1).any().any()


@pytest.mark.parametrize(
    "data,columns,index",
    [
        (pd.Series([1, 2, 3]), None, None),
        (pd.Series(["a", "b", None, "c"], name="abc"), None, None),
        (
            pd.Series(["a", "b", None, "c"], name="abc"),
            ["abc", "b"],
            [1, 2, 3],
        ),
    ],
)
def test_dataframe_init_from_series(data, columns, index):
    expected = pd.DataFrame(data, columns=columns, index=index)
    actual = cudf.DataFrame(data, columns=columns, index=index)
    if columns == ["abc", "b"]:
        # In pandas, new columns are object types with NaN
        # which cuDF doesn't support
        expected["b"] = expected["b"].astype(actual["b"].dtype)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "dtype,expected_upcast_type,error",
    [
        (
            "float32",
            np.dtype("float32"),
            None,
        ),
        (
            "float16",
            None,
            TypeError,
        ),
        (
            "float64",
            np.dtype("float64"),
            None,
        ),
        (
            "float128",
            None,
            ValueError,
        ),
    ],
)
def test_from_pandas_unsupported_types(dtype, expected_upcast_type, error):
    data = pd.Series([1.1, 0.55, -1.23], dtype=dtype)
    pdf = pd.DataFrame({"one_col": data})
    if error is not None:
        with pytest.raises(error):
            cudf.from_pandas(data)

        with pytest.raises(error):
            cudf.Series(data)

        with pytest.raises(error):
            cudf.from_pandas(pdf)

        with pytest.raises(error):
            cudf.DataFrame(pdf)
    else:
        df = cudf.from_pandas(data)

        assert_eq(data, df, check_dtype=False)
        assert df.dtype == expected_upcast_type

        df = cudf.Series(data)
        assert_eq(data, df, check_dtype=False)
        assert df.dtype == expected_upcast_type

        df = cudf.from_pandas(pdf)
        assert_eq(pdf, df, check_dtype=False)
        assert df["one_col"].dtype == expected_upcast_type

        df = cudf.DataFrame(pdf)
        assert_eq(pdf, df, check_dtype=False)
        assert df["one_col"].dtype == expected_upcast_type


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": 4},
        {"c": 4, "a": [1, 2, 3], "b": ["x", "y", "z"]},
        {"a": [1, 2, 3], "c": 4},
    ],
)
def test_dataframe_init_from_scalar_and_lists(data):
    actual = cudf.DataFrame(data)
    expected = pd.DataFrame(data)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "columns",
    (
        [],
        ["c", "a"],
        ["a", "d", "b", "e", "c"],
        ["a", "b", "c"],
        pd.Index(["b", "a", "c"], name="custom_name"),
    ),
)
@pytest.mark.parametrize("index", (None, [4, 5, 6]))
def test_dataframe_dict_like_with_columns(columns, index):
    data = {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    expect = pd.DataFrame(data, columns=columns, index=index)
    actual = cudf.DataFrame(data, columns=columns, index=index)
    if isinstance(columns, list):
        if columns == []:
            # As of pandas 3.0, empty columns are returned as Index[object]
            # which cuDF doesn't support
            expect.columns = expect.columns.astype(actual.columns.dtype)
        elif columns == ["a", "d", "b", "e", "c"]:
            # In pandas, new columns are object types with NaN
            # which cuDF doesn't support
            expect["d"] = expect["d"].astype(actual["d"].dtype)
            expect["e"] = expect["e"].astype(actual["e"].dtype)
    assert_eq(expect, actual)


def test_dataframe_init_columns_named_multiindex():
    rng = np.random.default_rng(seed=0)
    data = rng.standard_normal(size=(2, 2))
    columns = cudf.MultiIndex.from_tuples(
        [("A", "one"), ("A", "two")], names=["y", "z"]
    )
    gdf = cudf.DataFrame(data, columns=columns)
    pdf = pd.DataFrame(data, columns=columns.to_pandas())

    assert_eq(gdf, pdf)


def test_dataframe_init_columns_named_index():
    rng = np.random.default_rng(seed=0)
    data = rng.standard_normal(size=(2, 2))
    columns = pd.Index(["a", "b"], name="custom_name")
    gdf = cudf.DataFrame(data, columns=columns)
    pdf = pd.DataFrame(data, columns=columns)

    assert_eq(gdf, pdf)


def test_dataframe_from_pandas_sparse():
    pdf = pd.DataFrame(range(2), dtype=pd.SparseDtype(np.int64, 0))
    with pytest.raises(NotImplementedError):
        cudf.DataFrame(pdf)


def test_dataframe_constructor_unbounded_sequence():
    class A:
        def __getitem__(self, key):
            return 1

    with pytest.raises(TypeError):
        cudf.DataFrame([A()])

    with pytest.raises(TypeError):
        cudf.DataFrame({"a": A()})


def test_dataframe_constructor_dataframe_list():
    df = cudf.DataFrame(range(2))
    with pytest.raises(TypeError):
        cudf.DataFrame([df])


def test_dataframe_constructor_from_namedtuple():
    Point1 = collections.namedtuple("Point1", ["a", "b", "c"])
    Point2 = collections.namedtuple("Point1", ["x", "y"])

    data = [Point1(1, 2, 3), Point2(4, 5)]
    idx = ["a", "b"]
    gdf = cudf.DataFrame(data, index=idx)
    pdf = pd.DataFrame(data, index=idx)

    assert_eq(gdf, pdf)

    data = [Point2(4, 5), Point1(1, 2, 3)]
    with pytest.raises(ValueError):
        cudf.DataFrame(data, index=idx)
    with pytest.raises(ValueError):
        pd.DataFrame(data, index=idx)


def test_series_data_no_name_with_columns():
    gdf = cudf.DataFrame(cudf.Series([1]), columns=[1])
    pdf = pd.DataFrame(pd.Series([1]), columns=[1])
    assert_eq(gdf, pdf)


def test_series_data_no_name_with_columns_more_than_one_raises():
    with pytest.raises(ValueError):
        cudf.DataFrame(cudf.Series([1]), columns=[1, 2])
    with pytest.raises(ValueError):
        pd.DataFrame(pd.Series([1]), columns=[1, 2])


def test_series_data_with_name_with_columns_matching():
    gdf = cudf.DataFrame(cudf.Series([1], name=1), columns=[1])
    pdf = pd.DataFrame(pd.Series([1], name=1), columns=[1])
    assert_eq(gdf, pdf)


def test_series_data_with_name_with_columns_not_matching():
    gdf = cudf.DataFrame(cudf.Series([1], name=2), columns=[1])
    pdf = pd.DataFrame(pd.Series([1], name=2), columns=[1])
    # As of pandas 3.0, pandas returns object for empty columns
    # which cuDF doesn't support
    pdf = gdf.astype(gdf.iloc[:, 0].dtype)
    assert_eq(gdf, pdf)


def test_series_data_with_name_with_columns_matching_align():
    gdf = cudf.DataFrame(cudf.Series([1], name=2), columns=[1, 2])
    pdf = pd.DataFrame(pd.Series([1], name=2), columns=[1, 2])
    # As of pandas 3.0, pandas returns object for a new broadcasted column
    # which cuDF doesn't support
    pdf[1] = pdf[1].astype(gdf[1].dtype)
    assert_eq(gdf, pdf)


def test_generated_column():
    gdf = cudf.DataFrame({"a": (i for i in range(5))})
    assert len(gdf) == 5


@pytest.mark.parametrize(
    "data",
    [
        (
            pd.Series([3, 3.0]),
            pd.Series([2.3, 3.9]),
            pd.Series([1.5, 3.9]),
            pd.Series([1.0, 2]),
        ),
        [
            pd.Series([3, 3.0]),
            pd.Series([2.3, 3.9]),
            pd.Series([1.5, 3.9]),
            pd.Series([1.0, 2]),
        ],
    ],
)
def test_create_dataframe_from_list_like(data):
    pdf = pd.DataFrame(data, index=["count", "mean", "std", "min"])
    gdf = cudf.DataFrame(data, index=["count", "mean", "std", "min"])

    assert_eq(pdf, gdf)

    pdf = pd.DataFrame(data)
    gdf = cudf.DataFrame(data)

    assert_eq(pdf, gdf)


def test_create_dataframe_column():
    pdf = pd.DataFrame(columns=["a", "b", "c"], index=["A", "Z", "X"])
    gdf = cudf.DataFrame(columns=["a", "b", "c"], index=["A", "Z", "X"])
    # As of pandas 3.0, pandas returns object for broadcasted columns
    # which cuDF doesn't support
    pdf = gdf.astype(gdf.iloc[:, 0].dtype)
    assert_eq(pdf, gdf)

    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": [2, 3, 5]},
        columns=["a", "b", "c"],
        index=["A", "Z", "X"],
    )
    gdf = cudf.DataFrame(
        {"a": [1, 2, 3], "b": [2, 3, 5]},
        columns=["a", "b", "c"],
        index=["A", "Z", "X"],
    )
    # As of pandas 3.0, pandas returns object for broadcasted columns
    # which cuDF doesn't support
    pdf["c"] = pdf["c"].astype(gdf["c"].dtype)
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data",
    [
        pd.DataFrame(np.eye(2)),
        cudf.DataFrame(np.eye(2)),
        np.eye(2),
        cp.eye(2),
        None,
        [[1, 0], [0, 1]],
        [cudf.Series([0, 1]), cudf.Series([1, 0])],
    ],
)
@pytest.mark.parametrize(
    "columns",
    [None, range(2), pd.RangeIndex(2), cudf.RangeIndex(2)],
)
def test_dataframe_columns_returns_rangeindex(data, columns):
    if data is None and columns is None:
        pytest.skip(f"{data=} and {columns=} not relevant.")
    result = cudf.DataFrame(data=data, columns=columns).columns
    expected = pd.RangeIndex(range(2))
    assert_eq(result, expected)


def test_dataframe_columns_returns_rangeindex_single_col():
    result = cudf.DataFrame([1, 2, 3]).columns
    expected = pd.RangeIndex(range(1))
    assert_eq(result, expected)


@pytest.mark.parametrize("dtype", ["int64", "datetime64[ns]", "int8"])
@pytest.mark.parametrize("idx_data", [[], [1, 2]])
@pytest.mark.parametrize("data", [None, [], {}])
def test_dataframe_columns_empty_data_preserves_dtype(dtype, idx_data, data):
    result = cudf.DataFrame(
        data, columns=cudf.Index(idx_data, dtype=dtype)
    ).columns
    expected = pd.Index(idx_data, dtype=dtype)
    assert_eq(result, expected)


def test_dataframe_init_from_nested_dict():
    ordered_dict = collections.OrderedDict(
        [
            (
                "one",
                collections.OrderedDict(
                    [("col_a", "foo1"), ("col_b", "bar1")]
                ),
            ),
            (
                "two",
                collections.OrderedDict(
                    [("col_a", "foo2"), ("col_b", "bar2")]
                ),
            ),
            (
                "three",
                collections.OrderedDict(
                    [("col_a", "foo3"), ("col_b", "bar3")]
                ),
            ),
        ]
    )
    pdf = pd.DataFrame(ordered_dict)
    gdf = cudf.DataFrame(ordered_dict)

    assert_eq(pdf, gdf)
    regular_dict = {key: dict(value) for key, value in ordered_dict.items()}

    pdf = pd.DataFrame(regular_dict)
    gdf = cudf.DataFrame(regular_dict)
    assert_eq(pdf, gdf)


def test_init_from_2_categoricalindex_series_diff_categories():
    s1 = cudf.Series(
        [39, 6, 4], index=cudf.CategoricalIndex(["female", "male", "unknown"])
    )
    s2 = cudf.Series(
        [2, 152, 2, 242, 150],
        index=cudf.CategoricalIndex(["f", "female", "m", "male", "unknown"]),
    )
    result = cudf.DataFrame([s1, s2])
    expected = pd.DataFrame([s1.to_pandas(), s2.to_pandas()])
    # TODO: Remove once https://github.com/pandas-dev/pandas/issues/57592
    # is adressed
    expected.columns = result.columns
    assert_eq(result, expected, check_dtype=False)


def test_data_frame_values_no_cols_but_index():
    result = cudf.DataFrame(index=range(5)).values
    expected = pd.DataFrame(index=range(5)).values
    assert_eq(result, expected)


def test_dataframe_from_ndarray_dup_columns():
    with pytest.raises(ValueError):
        cudf.DataFrame(np.eye(2), columns=["A", "A"])


def test_dataframe_init_with_nans():
    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.DataFrame({"a": [1, 2, 3, np.nan]})
    assert gdf["a"].dtype == np.dtype("float64")
    pdf = pd.DataFrame({"a": [1, 2, 3, np.nan]})
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data",
    [
        cudf.DataFrame(range(2)),
        None,
        [cudf.Series(range(2))],
        [[0], [1]],
        {1: range(2)},
        cp.arange(2),
    ],
)
def test_init_with_index_no_shallow_copy(data):
    idx = cudf.RangeIndex(2)
    df = cudf.DataFrame(data, index=idx)
    assert df.index is idx


def test_from_records_with_index_no_shallow_copy():
    idx = cudf.RangeIndex(2)
    data = np.array([(1.0, 2), (3.0, 4)], dtype=[("x", "<f8"), ("y", "<i8")])
    df = cudf.DataFrame(data.view(np.recarray), index=idx)
    assert df.index is idx


def test_from_pandas_preserve_column_dtype():
    df = pd.DataFrame([[1, 2]], columns=pd.Index([1, 2], dtype="int8"))
    result = cudf.DataFrame(df)
    pd.testing.assert_index_equal(result.columns, df.columns, exact=True)


def test_dataframe_init_column():
    s = cudf.Series([1, 2, 3])
    with pytest.raises(TypeError):
        cudf.DataFrame(s._column)
    expect = cudf.DataFrame({"a": s})
    actual = cudf.DataFrame([1, 2, 3], columns=["a"])
    assert_eq(expect, actual)


@pytest.mark.parametrize("data", [None, {}])
def test_empty_construction_rangeindex_columns(data):
    result = cudf.DataFrame(data=data).columns
    expected = pd.RangeIndex(0)
    pd.testing.assert_index_equal(result, expected, exact=True)


@pytest.mark.parametrize(
    "shape",
    [
        (0, 3),
        (3, 0),
        (0, 0),
    ],
)
def test_construct_zero_axis_ndarray(shape):
    arr = np.empty(shape, dtype=np.float64)
    result = cudf.DataFrame(arr)
    expected = pd.DataFrame(arr)
    assert_eq(result, expected)


def test_construct_dict_scalar_values_raises():
    data = {"a": 1, "b": "2"}
    with pytest.raises(ValueError):
        pd.DataFrame(data)
    with pytest.raises(ValueError):
        cudf.DataFrame(data)


@pytest.mark.parametrize("columns", [None, [3, 4]])
@pytest.mark.parametrize("index", [None, [1, 2]])
def test_construct_empty_listlike_index_and_columns(columns, index):
    result = cudf.DataFrame([], columns=columns, index=index)
    expected = pd.DataFrame([], columns=columns, index=index)
    if columns is not None:
        # As of pandas 3.0, pandas returns object for empty columns
        # which cuDF doesn't support
        expected = expected.astype(result.iloc[:, 0].dtype)
    assert_eq(result, expected)


@pytest.mark.parametrize("index", [None, "a", ["a", "b"]])
def test_from_pandas_nan_as_null(nan_as_null, index):
    data = [np.nan, 2.0, 3.0]

    if index is None:
        pdf = pd.DataFrame({"a": data, "b": data})
        expected = cudf.DataFrame(
            {
                "a": as_column(data, nan_as_null=nan_as_null),
                "b": as_column(data, nan_as_null=nan_as_null),
            }
        )
    else:
        pdf = pd.DataFrame({"a": data, "b": data}).set_index(index)
        expected = cudf.DataFrame(
            {
                "a": as_column(data, nan_as_null=nan_as_null),
                "b": as_column(data, nan_as_null=nan_as_null),
            }
        )
        expected = cudf.DataFrame(
            {
                "a": as_column(data, nan_as_null=nan_as_null),
                "b": as_column(data, nan_as_null=nan_as_null),
            }
        )
        expected = expected.set_index(index)

    got = cudf.from_pandas(pdf, nan_as_null=nan_as_null)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data,columns",
    [
        ([1, 2, 3, 100, 112, 35464], ["a"]),
        (range(100), None),
        (
            [],
            None,
        ),
        ((-10, 21, 32, 32, 1, 2, 3), ["p"]),
        (
            (),
            None,
        ),
        ([[1, 2, 3], [1, 2, 3]], ["col1", "col2", "col3"]),
        ([range(100), range(100)], ["range" + str(i) for i in range(100)]),
        (((1, 2, 3), (1, 2, 3)), ["tuple0", "tuple1", "tuple2"]),
        ([[1, 2, 3]], ["list col1", "list col2", "list col3"]),
        ([[1, 2, 3]], pd.Index(["col1", "col2", "col3"], name="rapids")),
        ([range(100)], ["range" + str(i) for i in range(100)]),
        (((1, 2, 3),), ["k1", "k2", "k3"]),
    ],
)
def test_dataframe_init_1d_list(data, columns):
    expect = pd.DataFrame(data, columns=columns)
    actual = cudf.DataFrame(data, columns=columns)

    assert_eq(
        expect,
        actual,
        check_index_type=len(data) != 0,
    )

    expect = pd.DataFrame(data, columns=None)
    actual = cudf.DataFrame(data, columns=None)

    assert_eq(
        expect,
        actual,
        check_index_type=len(data) != 0,
    )


@pytest.mark.parametrize("dtype", ["int64", "str"])
def test_dataframe_from_dictionary_series_same_name_index(dtype):
    pd_idx1 = pd.Index([1, 2, 0], name="test_index").astype(dtype)
    pd_idx2 = pd.Index([2, 0, 1], name="test_index").astype(dtype)
    pd_series1 = pd.Series([1, 2, 3], index=pd_idx1)
    pd_series2 = pd.Series([1, 2, 3], index=pd_idx2)

    gd_idx1 = cudf.from_pandas(pd_idx1)
    gd_idx2 = cudf.from_pandas(pd_idx2)
    gd_series1 = cudf.Series([1, 2, 3], index=gd_idx1)
    gd_series2 = cudf.Series([1, 2, 3], index=gd_idx2)

    expect = pd.DataFrame({"a": pd_series1, "b": pd_series2})
    got = cudf.DataFrame({"a": gd_series1, "b": gd_series2})

    if dtype == "str":
        # Pandas actually loses its index name erroneously here...
        expect.index.name = "test_index"

    assert_eq(expect, got)
    assert expect.index.names == got.index.names


def test_init_multiindex_from_dict():
    pdf = pd.DataFrame({("a", "b"): [1]})
    gdf = cudf.DataFrame({("a", "b"): [1]})
    assert_eq(pdf, gdf)
    assert_eq(pdf.columns, gdf.columns)


def test_change_column_dtype_in_empty():
    pdf = pd.DataFrame({"a": [], "b": []})
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf, gdf)
    pdf["b"] = pdf["b"].astype("int64")
    gdf["b"] = gdf["b"].astype("int64")
    assert_eq(pdf, gdf)


def test_init_from_dict_of_empty_lists():
    # Untyped empty sequences become float64 in the DataFrame
    # constructor to match pandas (while Series([]) remains object).
    pdf = pd.DataFrame({"a": [], "b": []})
    gdf = cudf.DataFrame({"a": [], "b": []})
    assert_eq(pdf, gdf)
    assert gdf["a"].dtype == np.dtype("float64")


def test_init_from_dict_of_empty_iterator():
    # Iterators drain into an untyped empty sequence, so they follow the
    # same float64 default as empty lists.
    pdf = pd.DataFrame({"a": iter([])})
    gdf = cudf.DataFrame({"a": iter([])})
    assert_eq(pdf, gdf)
    assert gdf["a"].dtype == np.dtype("float64")


def test_init_from_dict_of_empty_range():
    # An empty range stays int64 like pandas (which converts range via
    # np.arange), unlike untyped empty lists/tuples/iterators.
    pdf = pd.DataFrame({"a": range(0)})
    gdf = cudf.DataFrame({"a": range(0)})
    assert_eq(pdf, gdf)
    assert gdf["a"].dtype == np.dtype("int64")


@pytest.mark.parametrize(
    "data,cols,index",
    [
        (
            np.ndarray(shape=(4, 2), dtype=float, order="F"),
            ["a", "b"],
            ["a", "b", "c", "d"],
        ),
        (
            np.ndarray(shape=(4, 2), dtype=float, order="F"),
            ["a", "b"],
            [0, 20, 30, 10],
        ),
        (
            np.ndarray(shape=(4, 2), dtype=float, order="F"),
            ["a", "b"],
            [0, 1, 2, 3],
        ),
        (np.array([11, 123, -2342, 232]), ["a"], [1, 2, 11, 12]),
        (np.array([11, 123, -2342, 232]), ["a"], ["khsdjk", "a", "z", "kk"]),
        (
            cp.ndarray(shape=(4, 2), dtype=float, order="F"),
            ["a", "z"],
            ["a", "z", "a", "z"],
        ),
        (cp.array([11, 123, -2342, 232]), ["z"], [0, 1, 1, 0]),
        (cp.array([11, 123, -2342, 232]), ["z"], [1, 2, 3, 4]),
        (cp.array([11, 123, -2342, 232]), ["z"], ["a", "z", "d", "e"]),
        (
            np.random.default_rng(seed=0).standard_normal(size=(2, 4)),
            ["a", "b", "c", "d"],
            ["a", "b"],
        ),
        (
            np.random.default_rng(seed=0).standard_normal(size=(2, 4)),
            ["a", "b", "c", "d"],
            [1, 0],
        ),
        (
            cp.random.default_rng(0).standard_normal(size=(2, 4)),
            ["a", "b", "c", "d"],
            ["a", "b"],
        ),
        (
            cp.random.default_rng(0).standard_normal(size=(2, 4)),
            ["a", "b", "c", "d"],
            [1, 0],
        ),
    ],
)
def test_dataframe_init_from_arrays_cols(data, cols, index):
    gd_data = data
    if isinstance(data, cp.ndarray):
        # pandas can't handle cupy arrays
        pd_data = data.get()
    else:
        pd_data = data

    # verify with columns & index
    pdf = pd.DataFrame(pd_data, columns=cols, index=index)
    gdf = cudf.DataFrame(gd_data, columns=cols, index=index)

    assert_eq(pdf, gdf, check_dtype=False)

    # verify with columns
    pdf = pd.DataFrame(pd_data, columns=cols)
    gdf = cudf.DataFrame(gd_data, columns=cols)

    assert_eq(pdf, gdf, check_dtype=False)

    pdf = pd.DataFrame(pd_data)
    gdf = cudf.DataFrame(gd_data)

    assert_eq(pdf, gdf, check_dtype=False)


@pytest.mark.parametrize(
    "data, orient, dtype, columns",
    [
        (
            {"col_1": [3, 2, 1, 0], "col_2": [3, 2, 1, 0]},
            "columns",
            None,
            None,
        ),
        ({"col_1": [3, 2, 1, 0], "col_2": [3, 2, 1, 0]}, "index", None, None),
        (
            {"col_1": [None, 2, 1, 0], "col_2": [3, None, 1, 0]},
            "index",
            None,
            ["A", "B", "C", "D"],
        ),
        (
            {
                "col_1": ["ab", "cd", "ef", "gh"],
                "col_2": ["zx", "one", "two", "three"],
            },
            "index",
            None,
            ["A", "B", "C", "D"],
        ),
        (
            {
                "index": [("a", "b"), ("a", "c")],
                "columns": [("x", 1), ("y", 2)],
                "data": [[1, 3], [2, 4]],
                "index_names": ["n1", "n2"],
                "column_names": ["z1", "z2"],
            },
            "tight",
            "float64",
            None,
        ),
    ],
)
def test_dataframe_from_dict(data, orient, dtype, columns):
    expected = pd.DataFrame.from_dict(
        data=data, orient=orient, dtype=dtype, columns=columns
    )

    actual = cudf.DataFrame.from_dict(
        data=data, orient=orient, dtype=dtype, columns=columns
    )

    assert_eq(expected, actual)


@pytest.mark.parametrize("dtype", ["int64", "str", None])
def test_dataframe_from_dict_transposed(dtype):
    pd_data = {"a": [3, 2, 1, 0], "col_2": [3, 2, 1, 0]}
    gd_data = {key: cudf.Series(val) for key, val in pd_data.items()}

    expected = pd.DataFrame.from_dict(pd_data, orient="index", dtype=dtype)
    actual = cudf.DataFrame.from_dict(gd_data, orient="index", dtype=dtype)

    gd_data = {key: cp.asarray(val) for key, val in pd_data.items()}
    actual = cudf.DataFrame.from_dict(gd_data, orient="index", dtype=dtype)
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "pd_data, gd_data, orient, dtype, columns",
    [
        (
            {"col_1": np.array([3, 2, 1, 0]), "col_2": np.array([3, 2, 1, 0])},
            {
                "col_1": cp.array([3, 2, 1, 0]),
                "col_2": cp.array([3, 2, 1, 0]),
            },
            "columns",
            None,
            None,
        ),
        (
            {"col_1": np.array([3, 2, 1, 0]), "col_2": np.array([3, 2, 1, 0])},
            {
                "col_1": cp.array([3, 2, 1, 0]),
                "col_2": cp.array([3, 2, 1, 0]),
            },
            "index",
            None,
            None,
        ),
        (
            {
                "col_1": np.array([None, 2, 1, 0]),
                "col_2": np.array([3, None, 1, 0]),
            },
            {
                "col_1": cp.array([np.nan, 2, 1, 0]),
                "col_2": cp.array([3, np.nan, 1, 0]),
            },
            "index",
            None,
            ["A", "B", "C", "D"],
        ),
        (
            {
                "col_1": np.array(["ab", "cd", "ef", "gh"]),
                "col_2": np.array(["zx", "one", "two", "three"]),
            },
            {
                "col_1": np.array(["ab", "cd", "ef", "gh"]),
                "col_2": np.array(["zx", "one", "two", "three"]),
            },
            "index",
            None,
            ["A", "B", "C", "D"],
        ),
        (
            {
                "index": [("a", "b"), ("a", "c")],
                "columns": [("x", 1), ("y", 2)],
                "data": [np.array([1, 3]), np.array([2, 4])],
                "index_names": ["n1", "n2"],
                "column_names": ["z1", "z2"],
            },
            {
                "index": [("a", "b"), ("a", "c")],
                "columns": [("x", 1), ("y", 2)],
                "data": [cp.array([1, 3]), cp.array([2, 4])],
                "index_names": ["n1", "n2"],
                "column_names": ["z1", "z2"],
            },
            "tight",
            "float64",
            None,
        ),
    ],
)
def test_dataframe_from_dict_cp_np_arrays(
    pd_data, gd_data, orient, dtype, columns
):
    expected = pd.DataFrame.from_dict(
        data=pd_data, orient=orient, dtype=dtype, columns=columns
    )

    actual = cudf.DataFrame.from_dict(
        data=gd_data, orient=orient, dtype=dtype, columns=columns
    )

    assert_eq(expected, actual, check_dtype=dtype is not None)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [[]]},
        {"a": [[None]]},
        {"a": [[1, 2, 3]]},
        {"a": [[1, 2, 3]], "b": [[2, 3, 4]]},
        {"a": [[1, 2, 3, None], [None]], "b": [[2, 3, 4], [5]], "c": None},
        {"a": [[1]], "b": [[1, 2, 3]]},
        pd.DataFrame({"a": [[1, 2, 3]]}),
    ],
)
def test_df_list_dtypes(data):
    expect = pd.DataFrame(data)
    got = cudf.DataFrame(data)
    if isinstance(data, dict) and "c" in data:
        # As of pandas 3.0, pandas returns object for broadcasted columns
        # which cuDF doesn't support
        expect["c"] = expect["c"].astype(got["c"].dtype)
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [[]]},
        {"a": [[1, 2, None, 4]]},
        {"a": [["cat", None, "dog"]]},
        {
            "a": [[1, 2, 3, None], [4, None, 5]],
            "b": [None, ["fish", "bird"]],
            "c": [[], []],
        },
        {"a": [[1, 2, 3, None], [4, None, 5], None, [6, 7]]},
    ],
)
def test_serialize_list_columns(data):
    df = cudf.DataFrame(data)
    reconstructed = df.__class__.deserialize(*df.serialize())
    assert_eq(reconstructed, df)


@pytest.mark.parametrize(
    "data1, data2",
    [(1, 2), (1.0, 2.0), (3, 4.0)],
)
@pytest.mark.parametrize("data3, data4", [(6, 10), (5.0, 9.0), (2, 6.0)])
def test_create_interval_df(data1, data2, data3, data4, interval_closed):
    # df for both pandas and cudf only works when interval is in a list
    expect = pd.DataFrame(
        [pd.Interval(data1, data2, interval_closed)], dtype="interval"
    )
    got = cudf.DataFrame(
        [pd.Interval(data1, data2, interval_closed)], dtype="interval"
    )
    assert_eq(expect, got)

    expect_two = pd.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
            "b": [
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
            ],
        },
        dtype="interval",
    )
    got_two = cudf.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
            "b": [
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
            ],
        },
        dtype="interval",
    )
    assert_eq(expect_two, got_two)

    expect_three = pd.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
            ],
            "b": [
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
            "c": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
        },
        dtype="interval",
    )

    got_three = cudf.DataFrame(
        {
            "a": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
            ],
            "b": [
                pd.Interval(data3, data4, interval_closed),
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
            "c": [
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data1, data2, interval_closed),
                pd.Interval(data3, data4, interval_closed),
            ],
        },
        dtype="interval",
    )
    assert_eq(expect_three, got_three)


def test_roundtrip_dataframe_plc_table():
    pdf = pd.DataFrame(
        {
            "a": [None, None, np.nan, None],
            "b": [np.nan, None, np.nan, None],
        }
    )
    expect = cudf.DataFrame(pdf)
    actual = cudf.DataFrame.from_pylibcudf(*expect.to_pylibcudf())
    assert_eq(expect, actual)


def test_dataframe_from_generator():
    pdf = pd.DataFrame((i for i in range(5)))
    gdf = cudf.DataFrame((i for i in range(5)))
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "dtype", ["datetime64[ns]", "timedelta64[ns]", "int64", "float32"]
)
def test_dataframe_mixed_dtype_error(dtype):
    pdf = pd.Series([1, 2, 3], dtype=dtype).to_frame().astype(object)
    with pytest.raises(TypeError):
        cudf.from_pandas(pdf)


def test_dataframe_from_arrow_slice():
    table = pa.Table.from_pandas(
        pd.DataFrame.from_dict(
            {"a": ["aa", "bb", "cc"] * 3, "b": [1, 2, 3] * 3}
        )
    )
    table_slice = table.slice(3, 7)

    expected = table_slice.to_pandas()
    actual = cudf.DataFrame.from_arrow(table_slice)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data,index",
    [
        ({"a": [1, 2, 3], "b": ["x", "y", "z", "z"], "c": 4}, None),
        (
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            },
            [10, 11],
        ),
        (
            {
                "a": [1, 2, 3],
                "b": ["x", "y", "z"],
            },
            [10, 11],
        ),
        ([[10, 11], [12, 13]], ["a", "b", "c"]),
    ],
)
def test_dataframe_init_length_error(data, index):
    assert_exceptions_equal(
        lfunc=pd.DataFrame,
        rfunc=cudf.DataFrame,
        lfunc_args_and_kwargs=(
            [],
            {"data": data, "index": index},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"data": data, "index": index},
        ),
    )


def test_complex_types_from_arrow():
    expected = pa.Table.from_arrays(
        [
            pa.array([1, 2, 3]),
            pa.array([10, 20, 30]),
            pa.array([{"a": 9}, {"b": 10}, {"c": 11}]),
            pa.array([[{"a": 1}], [{"b": 2}], [{"c": 3}]]),
            pa.array([10, 11, 12]).cast(pa.decimal128(21, 2)),
            pa.array([{"a": 9}, {"b": 10, "c": {"g": 43}}, {"c": {"a": 10}}]),
        ],
        names=["a", "b", "c", "d", "e", "f"],
    )

    df = cudf.DataFrame.from_arrow(expected)
    actual = df.to_arrow()

    assert expected.equals(actual)


def test_dataframe_constructor_column_index_only():
    columns = ["a", "b", "c"]
    index = ["r1", "r2", "r3"]

    gdf = cudf.DataFrame(index=index, columns=columns)
    assert gdf["a"]._column is not gdf["b"]._column
    assert gdf["b"]._column is not gdf["c"]._column


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {"a": [1, 2, 3], "b": [10, 11, 20], "c": ["a", "bcd", "xyz"]}
        ),
        pd.DataFrame(),
    ],
)
@pytest.mark.parametrize(
    "columns",
    [
        None,
        ["a"],
        ["c", "a"],
        ["b", "a", "c"],
        [],
        pd.Index(["c", "a"]),
        cudf.Index(["c", "a"]),
        ["abc", "a"],
        ["column_not_exists1", "column_not_exists2"],
    ],
)
@pytest.mark.parametrize("index", [["abc", "def", "ghi"]])
def test_dataframe_constructor_columns(df, columns, index, request):
    def assert_local_eq(actual, df, expected, host_columns):
        check_index_type = not expected.empty
        if host_columns is not None and any(
            col not in df.columns for col in host_columns
        ):
            assert_eq(
                expected,
                actual,
                check_dtype=False,
                check_index_type=check_index_type,
            )
        else:
            assert_eq(
                expected,
                actual,
                check_index_type=check_index_type,
                check_column_type=False,
            )

    gdf = cudf.from_pandas(df)
    host_columns = (
        columns.to_pandas() if isinstance(columns, cudf.Index) else columns
    )

    expected = pd.DataFrame(df, columns=host_columns, index=index)
    actual = cudf.DataFrame(gdf, columns=columns, index=index)

    assert_local_eq(actual, df, expected, host_columns)


def test_dataframe_from_pandas_duplicate_columns():
    pdf = pd.DataFrame(columns=["a", "b", "c", "a"])
    pdf["a"] = [1, 2, 3]

    with pytest.raises(
        ValueError, match="Duplicate column names are not allowed"
    ):
        cudf.from_pandas(pdf)


@pytest.mark.parametrize(
    "data",
    [
        [{"a": 1, "b": 2, "c": 3}, {"a": 4, "b": 5, "c": 6}],
        [{"a": 1, "b": 2, "c": None}, {"a": None, "b": 5, "c": 6}],
        [{"a": 1, "b": 2}, {"a": 1, "b": 5, "c": 6}],
        [{"a": 1, "b": 2}, {"b": 5, "c": 6}],
        [{}, {"a": 1, "b": 5, "c": 6}],
        [{"a": 1, "b": 2, "c": 3}, {"a": 4.5, "b": 5.5, "c": 6.5}],
    ],
)
def test_dataframe_init_from_list_of_dicts(data):
    expect = pd.DataFrame(data)
    got = cudf.DataFrame(data)

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        None,
        [],
        [1],
        {"a": [10, 11, 12]},
        {
            "a": [10, 11, 12],
            "another column name": [12, 22, 34],
            "xyz": [0, 10, 11],
        },
    ],
)
@pytest.mark.parametrize(
    "columns",
    [["a"], ["another column name"], None, pd.Index(["a"], name="index name")],
)
def test_dataframe_init_with_columns(data, columns):
    pdf = pd.DataFrame(data, columns=columns)
    gdf = cudf.DataFrame(data, columns=columns)

    assert_eq(
        pdf,
        gdf,
        check_index_type=len(pdf.index) != 0,
        check_dtype=not (pdf.empty and len(pdf.columns)),
        check_column_type=False,
    )


@pytest.mark.parametrize(
    "data, ignore_dtype",
    [
        ([pd.Series([1, 2, 3])], False),
        ([pd.Series(index=[1, 2, 3], dtype="float64")], False),
        ([pd.Series(name="empty series name", dtype="float64")], False),
        (
            [pd.Series([1]), pd.Series([], dtype="float64"), pd.Series([3])],
            False,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], dtype="float64"),
                pd.Series([3], name="series that is named"),
            ],
            False,
        ),
        ([pd.Series([1, 2, 3], name="hi")] * 10, False),
        ([pd.Series([1, 2, 3], name=None, index=[10, 11, 12])] * 10, False),
        (
            [
                pd.Series([1, 2, 3], name=None, index=[10, 11, 12]),
                pd.Series([1, 2, 30], name=None, index=[13, 144, 15]),
            ],
            True,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], dtype="float64"),
                pd.Series(index=[10, 11, 12], dtype="float64"),
            ],
            False,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], name="abc", dtype="float64"),
                pd.Series(index=[10, 11, 12], dtype="float64"),
            ],
            False,
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([1, -100, 200, -399, 400], name="abc"),
                pd.Series([111, 222, 333], index=[10, 11, 12]),
            ],
            False,
        ),
    ],
)
@pytest.mark.parametrize(
    "columns",
    [
        None,
        ["0"],
        [0],
        ["abc"],
        [144, 13],
        [2, 1, 0],
        pd.Index(["abc"], name="custom_name"),
    ],
)
def test_dataframe_init_from_series_list(data, ignore_dtype, columns):
    gd_data = [cudf.from_pandas(obj) for obj in data]

    expected = pd.DataFrame(data, columns=columns)
    actual = cudf.DataFrame(gd_data, columns=columns)

    if ignore_dtype:
        # When a union is performed to generate columns,
        # the order is never guaranteed. Hence sort by
        # columns before comparison.
        if not expected.columns.equals(actual.columns):
            expected = expected.sort_index(axis=1)
            actual = actual.sort_index(axis=1)
        assert_eq(
            expected.fillna(-1),
            actual.fillna(-1),
            check_dtype=False,
            check_index_type=True,
        )
    else:
        assert_eq(
            expected,
            actual,
            check_index_type=True,
            check_column_type=False,
        )


@pytest.mark.parametrize(
    "data, ignore_dtype, index",
    [
        ([pd.Series([1, 2, 3])], False, ["a", "b", "c"]),
        ([pd.Series(index=[1, 2, 3], dtype="float64")], False, ["a", "b"]),
        (
            [pd.Series(name="empty series name", dtype="float64")],
            False,
            ["index1"],
        ),
        (
            [pd.Series([1]), pd.Series([], dtype="float64"), pd.Series([3])],
            False,
            ["0", "2", "1"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], dtype="float64"),
                pd.Series([3], name="series that is named"),
            ],
            False,
            ["_", "+", "*"],
        ),
        ([pd.Series([1, 2, 3], name="hi")] * 10, False, ["mean"] * 10),
        (
            [pd.Series([1, 2, 3], name=None, index=[10, 11, 12])] * 10,
            False,
            ["abc"] * 10,
        ),
        (
            [
                pd.Series([1, 2, 3], name=None, index=[10, 11, 12]),
                pd.Series([1, 2, 30], name=None, index=[13, 144, 15]),
            ],
            True,
            ["set_index_a", "set_index_b"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], dtype="float64"),
                pd.Series(index=[10, 11, 12], dtype="float64"),
            ],
            False,
            ["a", "b", "c"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], name="abc", dtype="float64"),
                pd.Series(index=[10, 11, 12], dtype="float64"),
            ],
            False,
            ["a", "v", "z"],
        ),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([1, -100, 200, -399, 400], name="abc"),
                pd.Series([111, 222, 333], index=[10, 11, 12]),
            ],
            False,
            ["a", "v", "z"],
        ),
    ],
)
@pytest.mark.parametrize(
    "columns", [None, ["0"], [0], ["abc"], [144, 13], [2, 1, 0]]
)
def test_dataframe_init_from_series_list_with_index(
    data,
    ignore_dtype,
    index,
    columns,
):
    gd_data = [cudf.from_pandas(obj) for obj in data]

    expected = pd.DataFrame(data, columns=columns, index=index)
    actual = cudf.DataFrame(gd_data, columns=columns, index=index)

    if ignore_dtype:
        # When a union is performed to generate columns,
        # the order is never guaranteed. Hence sort by
        # columns before comparison.
        if not expected.columns.equals(actual.columns):
            expected = expected.sort_index(axis=1)
            actual = actual.sort_index(axis=1)
        assert_eq(expected.fillna(-1), actual.fillna(-1), check_dtype=False)
    else:
        assert_eq(expected, actual, check_column_type=False)


@pytest.mark.parametrize(
    "data, index",
    [
        ([pd.Series([1, 2]), pd.Series([1, 2])], ["a", "b", "c"]),
        (
            [
                pd.Series([1, 0.324234, 32424.323, -1233, 34242]),
                pd.Series([], dtype="float64"),
                pd.Series([3], name="series that is named"),
            ],
            ["_", "+"],
        ),
        ([pd.Series([1, 2, 3], name="hi")] * 10, ["mean"] * 9),
    ],
)
def test_dataframe_init_from_series_list_with_index_error(data, index):
    gd_data = [cudf.from_pandas(obj) for obj in data]

    assert_exceptions_equal(
        pd.DataFrame,
        cudf.DataFrame,
        ([data], {"index": index}),
        ([gd_data], {"index": index}),
    )


@pytest.mark.parametrize(
    "data",
    [
        [pd.Series([1, 2, 3], index=["a", "a", "a"])],
        [pd.Series([1, 2, 3], index=["a", "a", "a"])] * 4,
        [
            pd.Series([1, 2, 3], index=["a", "b", "a"]),
            pd.Series([1, 2, 3], index=["b", "b", "a"]),
        ],
        [
            pd.Series([1, 2, 3], index=["a", "b", "z"]),
            pd.Series([1, 2, 3], index=["u", "b", "a"]),
            pd.Series([1, 2, 3], index=["u", "b", "u"]),
        ],
    ],
)
def test_dataframe_init_from_series_list_duplicate_index_error(data):
    gd_data = [cudf.from_pandas(obj) for obj in data]

    assert_exceptions_equal(
        lfunc=pd.DataFrame,
        rfunc=cudf.DataFrame,
        lfunc_args_and_kwargs=([], {"data": data}),
        rfunc_args_and_kwargs=([], {"data": gd_data}),
        check_exception_type=False,
    )


def test_from_pandas():
    pdf = pd.DataFrame(
        {
            "a": np.arange(10, dtype=np.int32),
            "b": np.arange(10, 20, dtype=np.float64),
        }
    )

    df = cudf.DataFrame(pdf)

    assert tuple(df.columns) == tuple(pdf.columns)

    assert df["a"].dtype == pdf["a"].dtype
    assert df["b"].dtype == pdf["b"].dtype

    assert len(df["a"]) == len(pdf["a"])
    assert len(df["b"]) == len(pdf["b"])


def test_from_pandas_ex1():
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.3]})
    df = cudf.DataFrame(pdf)

    assert tuple(df.columns) == tuple(pdf.columns)
    assert np.all(df["a"].to_numpy() == pdf["a"])
    matches = df["b"].to_numpy(na_value=np.nan) == pdf["b"]
    # the 3d element is False due to (nan == nan) == False
    assert np.all(matches == [True, True, False, True])
    assert np.isnan(df["b"].to_numpy(na_value=np.nan)[2])
    assert np.isnan(pdf["b"][2])


def test_from_pandas_with_index():
    pdf = pd.DataFrame({"a": [0, 1, 2, 3], "b": [0.1, 0.2, None, 0.3]})
    pdf = pdf.set_index(np.asarray([4, 3, 2, 1]))
    df = cudf.DataFrame(pdf)

    # Check columns
    assert_eq(df.a, pdf.a)
    assert_eq(df.b, pdf.b)
    # Check index
    assert_eq(df.index.values, pdf.index.values)
    # Check again using pandas testing tool on frames
    assert_eq(df, pdf)


@pytest.mark.parametrize("columns", [None, ("a", "b"), ("a",), ("b",)])
def test_from_records_noindex(columns):
    recdtype = np.dtype([("a", np.int32), ("b", np.float64)])
    rec = np.recarray(10, dtype=recdtype)
    rec.a = aa = np.arange(10, dtype=np.int32)
    rec.b = bb = np.arange(10, 20, dtype=np.float64)
    df = cudf.DataFrame.from_records(rec, columns=columns)

    if columns and "a" in columns:
        assert_eq(aa, df["a"].values)
    if columns and "b" in columns:
        assert_eq(bb, df["b"].values)
    assert_eq(np.arange(10), df.index.values)


@pytest.mark.parametrize("columns", [None, ("a", "b"), ("a",), ("b",)])
def test_from_records_withindex(columns):
    recdtype = np.dtype(
        [("index", np.int64), ("a", np.int32), ("b", np.float64)]
    )
    rec = np.recarray(10, dtype=recdtype)
    rec.index = ii = np.arange(30, 40)
    rec.a = aa = np.arange(10, dtype=np.int32)
    rec.b = bb = np.arange(10, 20, dtype=np.float64)
    df = cudf.DataFrame.from_records(rec, index="index")

    if columns and "a" in columns:
        assert_eq(aa, df["a"].values)
    if columns and "b" in columns:
        assert_eq(bb, df["b"].values)
    assert_eq(ii, df.index.values)


def test_numpy_non_contiguous():
    recdtype = np.dtype([("index", np.int64), ("a", np.int32)])
    rec = np.recarray(10, dtype=recdtype)
    rec.index = np.arange(30, 40)
    rec.a = aa = np.arange(20, dtype=np.int32)[::2]
    assert rec.a.flags["C_CONTIGUOUS"] is False

    gdf = cudf.DataFrame.from_records(rec, index="index")
    assert_eq(aa, gdf["a"].values)


def test_from_pandas_series():
    pdf = pd.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]}
    ).set_index(["a", "b"])

    result = cudf.from_pandas(pdf)
    assert_eq(pdf, result)

    test_pdf = pdf["c"]
    result = cudf.from_pandas(test_pdf)
    assert_eq(test_pdf, result)


def test_from_pandas_with_multiindex():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(rng.random(size=(7, 5)))
    pdf.index = pd.MultiIndex.from_arrays([range(7)])
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf, gdf)


@pytest.mark.parametrize("dtype", ["int", "int64[pyarrow]"])
def test_from_pandas_with_nullable_pandas_type(dtype):
    df = pd.DataFrame({"x": [1, 2, 3]}, index=[4.0, 5.0, 6.0], dtype=dtype)
    df.columns.name = "custom_column_name"
    gdf = cudf.DataFrame(df)
    assert isinstance(gdf, cudf.DataFrame)

    assert_eq(df, gdf, check_dtype="pyarrow" not in dtype)

    s = df.x
    gs = cudf.Series(s)
    assert isinstance(gs, cudf.Series)

    assert_eq(s, gs, check_dtype="pyarrow" not in dtype)


def test_df_constructor_dtype(all_supported_types_as_str):
    if "datetime" in all_supported_types_as_str:
        data = ["1991-11-20", "2004-12-04", "2016-09-13", None]
    elif all_supported_types_as_str == "str":
        data = ["a", "b", "c", None]
    elif "float" in all_supported_types_as_str:
        data = [1.0, 0.5, -1.1, np.nan, None]
    elif "bool" in all_supported_types_as_str:
        data = [True, False, None]
    else:
        data = [1, 2, 3, None]

    sr = cudf.Series(data, dtype=all_supported_types_as_str)

    expect = cudf.DataFrame({"foo": sr, "bar": sr})
    got = cudf.DataFrame(
        {"foo": data, "bar": data}, dtype=all_supported_types_as_str
    )

    assert_eq(expect, got)


def test_build_df_from_nullable_pandas_dtype(
    all_supported_pandas_nullable_extension_dtypes,
):
    scalar, dtype = all_supported_pandas_nullable_extension_dtypes

    expected = pd.DataFrame({"a": [scalar, pd.NA]}, dtype=dtype)
    result = cudf.DataFrame(expected)

    assert result["a"].dtype == expected["a"].dtype

    expect_mask = expected["a"].isna()
    got_mask = result["a"].isna().to_numpy()

    np.testing.assert_array_equal(expect_mask, got_mask)


@pytest.mark.parametrize(
    "index", [None, [], pd.Index([], name="a"), pd.RangeIndex(0)]
)
@pytest.mark.parametrize("columns", [["a", "b"], pd.Index(["a", "b"])])
def test_empty_dataframe_columns_default_object_dtype(index, columns):
    # An empty (zero-row) DataFrame built from only column labels has no
    # data to infer from and defaults to object dtype, matching pandas
    # (cudf otherwise uses the default string dtype).
    expected = pd.DataFrame(columns=columns, index=index)
    got = cudf.DataFrame(columns=columns, index=index)
    assert_eq(got, expected)


def test_empty_dataframe_data_none_with_columns():
    # data=None with explicit columns and an empty index -> object columns.
    expected = pd.DataFrame(data=None, columns=["x", "y"])
    got = cudf.DataFrame(data=None, columns=["x", "y"])
    assert_eq(got, expected)

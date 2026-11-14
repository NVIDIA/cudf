# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "selector",
    [
        np.dtype("int64"),
        pd.Int64Dtype(),
        pd.ArrowDtype(pa.int64()),
        pd.DatetimeTZDtype("ns", "UTC"),
        pd.CategoricalDtype(["a", "b"]),
    ],
)
def test_select_dtypes_instances(selector):
    pdf = pd.DataFrame(
        {
            "numpy": pd.Series([1, 2], dtype="int64"),
            "nullable": pd.Series([1, 2], dtype="Int64"),
            "arrow": pd.Series([1, 2], dtype=pd.ArrowDtype(pa.int64())),
            "utc": pd.date_range("2020-01-01", periods=2, tz="UTC", unit="ns"),
            "tokyo": pd.date_range(
                "2020-01-01", periods=2, tz="Asia/Tokyo", unit="ns"
            ),
            "category_ab": pd.Categorical(["a", "b"]),
            "category_ac": pd.Categorical(["a", "c"]),
        }
    )
    gdf = cudf.from_pandas(pdf)
    for keyword in ("include", "exclude"):
        assert_eq(
            pdf.select_dtypes(**{keyword: selector}),
            gdf.select_dtypes(**{keyword: selector}),
        )


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="New pandas 3.1 selector validation"
)
@pytest.mark.parametrize(
    "selector",
    ["datetime64[10s]", "timedelta64[Y]", pd.CategoricalDtype(ordered=True)],
)
def test_select_dtypes_invalid_parameterized_dtype(selector):
    pdf = pd.DataFrame({"a": [1, 2]})
    gdf = cudf.from_pandas(pdf)
    assert_exceptions_equal(
        pdf.select_dtypes,
        gdf.select_dtypes,
        ([selector],),
        ([selector],),
    )


def test_select_dtypes_cudf_metadata():
    gdf = cudf.DataFrame(
        {
            "integer": [1, 2],
            "list": [[1], [2]],
            "struct": [{"x": 1}, {"x": 2}],
            "decimal": cudf.Series([1, 2], dtype=cudf.Decimal64Dtype(5, 0)),
        }
    )
    assert list(gdf.select_dtypes("int64").columns) == ["integer"]
    assert list(gdf.select_dtypes(cudf.Decimal64Dtype).columns) == ["decimal"]
    if PANDAS_GE_31:
        assert list(gdf.select_dtypes(pd.ArrowDtype).columns) == [
            "list",
            "struct",
            "decimal",
        ]


@pytest.mark.parametrize(
    "columns",
    [
        [(1, "a"), (2, "b")],
        [1, "b"],
        pd.MultiIndex.from_tuples([("a", 1), ("b", 2)]),
    ],
)
def test_select_dtypes_preserves_column_labels(columns):
    pdf = pd.DataFrame({"first": [1, 2], "second": [1.5, 2.5]})
    pdf.columns = columns
    gdf = cudf.from_pandas(pdf)
    assert_eq(pdf.select_dtypes("int64"), gdf.select_dtypes("int64"))


def test_select_dtype():
    gdf = cudf.datasets.randomdata(
        nrows=20, dtypes={"a": "category", "b": int, "c": float, "d": str}
    )
    pdf = gdf.to_pandas()

    assert_eq(pdf.select_dtypes("float64"), gdf.select_dtypes("float64"))
    assert_eq(pdf.select_dtypes(np.float64), gdf.select_dtypes(np.float64))
    assert_eq(
        pdf.select_dtypes(include=["float64"]),
        gdf.select_dtypes(include=["float64"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["str", "int", "category"]),
        gdf.select_dtypes(include=["str", "int", "category"]),
    )

    assert_eq(
        pdf.select_dtypes(include=["int64", "float64"]),
        gdf.select_dtypes(include=["int64", "float64"]),
    )
    assert_eq(
        pdf.select_dtypes(include=np.number),
        gdf.select_dtypes(include=np.number),
    )
    assert_eq(
        pdf.select_dtypes(include=[np.int64, np.float64]),
        gdf.select_dtypes(include=[np.int64, np.float64]),
    )

    assert_eq(
        pdf.select_dtypes(include=["category"]),
        gdf.select_dtypes(include=["category"]),
    )
    assert_eq(
        pdf.select_dtypes(exclude=np.number),
        gdf.select_dtypes(exclude=np.number),
    )

    assert_exceptions_equal(
        lfunc=pdf.select_dtypes,
        rfunc=gdf.select_dtypes,
        lfunc_args_and_kwargs=([], {"includes": ["Foo"]}),
        rfunc_args_and_kwargs=([], {"includes": ["Foo"]}),
    )

    assert_exceptions_equal(
        lfunc=pdf.select_dtypes,
        rfunc=gdf.select_dtypes,
        lfunc_args_and_kwargs=(
            [],
            {"exclude": np.number, "include": np.number},
        ),
        rfunc_args_and_kwargs=(
            [],
            {"exclude": np.number, "include": np.number},
        ),
    )

    gdf = cudf.DataFrame(
        {"A": [3, 4, 5], "C": [1, 2, 3], "D": ["a", "b", "c"]}
    )
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes(include=["str", "int", "category"]),
        gdf.select_dtypes(include=["str", "int", "category"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["str"], exclude=["category"]),
        gdf.select_dtypes(include=["str"], exclude=["category"]),
    )

    gdf = cudf.DataFrame({"a": range(10), "b": range(10, 20)})
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes(include=["category"]),
        gdf.select_dtypes(include=["category"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["float"]),
        gdf.select_dtypes(include=["float"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["str"]),
        gdf.select_dtypes(include=["str"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["int"]), gdf.select_dtypes(include=["int"])
    )
    assert_eq(
        pdf.select_dtypes(exclude=["float"]),
        gdf.select_dtypes(exclude=["float"]),
    )
    assert_eq(
        pdf.select_dtypes(exclude=["str"]),
        gdf.select_dtypes(exclude=["str"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["int"], exclude=["str"]),
        gdf.select_dtypes(include=["int"], exclude=["str"]),
    )

    assert_exceptions_equal(
        lfunc=pdf.select_dtypes,
        rfunc=gdf.select_dtypes,
    )

    gdf = cudf.DataFrame(
        {"a": cudf.Series([], dtype="int"), "b": cudf.Series([], dtype="str")}
    )
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes(exclude=["str"]),
        gdf.select_dtypes(exclude=["str"]),
    )
    assert_eq(
        pdf.select_dtypes(include=["int"], exclude=["str"]),
        gdf.select_dtypes(include=["int"], exclude=["str"]),
    )

    gdf = cudf.DataFrame(
        {"int_col": [0, 1, 2], "list_col": [[1, 2], [3, 4], [5, 6]]}
    )
    pdf = gdf.to_pandas()
    assert_eq(
        pdf.select_dtypes("int64"),
        gdf.select_dtypes("int64"),
    )


def test_select_dtype_datetime():
    gdf = cudf.datasets.timeseries(
        start="2000-01-01", end="2000-01-02", freq="3600s", dtypes={"x": int}
    )
    gdf = gdf.reset_index()
    pdf = gdf.to_pandas()

    assert_eq(pdf.select_dtypes("datetime64"), gdf.select_dtypes("datetime64"))
    assert_eq(
        pdf.select_dtypes(np.dtype("datetime64")),
        gdf.select_dtypes(np.dtype("datetime64")),
    )
    assert_eq(
        pdf.select_dtypes(include="datetime64"),
        gdf.select_dtypes(include="datetime64"),
    )


def test_select_dtype_datetime_with_frequency():
    gdf = cudf.datasets.timeseries(
        start="2000-01-01", end="2000-01-02", freq="3600s", dtypes={"x": int}
    )
    gdf = gdf.reset_index()
    pdf = gdf.to_pandas()

    if PANDAS_GE_31:
        assert_eq(
            pdf.select_dtypes("datetime64[ms]"),
            gdf.select_dtypes("datetime64[ms]"),
        )
    else:
        assert_exceptions_equal(
            pdf.select_dtypes,
            gdf.select_dtypes,
            (["datetime64[ms]"],),
            (["datetime64[ms]"],),
        )


@pytest.mark.parametrize("selector", [int, "int", float, "float"])
def test_select_dtypes_int_float_families(selector):
    # int/"int" selects both int32 and int64; float/"float" selects both
    # float32 and float64, matching pandas' check_int_infer_dtype.
    pdf = pd.DataFrame(
        {
            "a": np.array([1, 2], dtype="int32"),
            "b": np.array([1, 2], dtype="int64"),
            "c": np.array([1, 2], dtype="float32"),
            "d": np.array([1, 2], dtype="float64"),
        }
    )
    gdf = cudf.DataFrame(pdf)
    assert_eq(
        pdf.select_dtypes(include=[selector]),
        gdf.select_dtypes(include=[selector]),
    )


@pytest.mark.parametrize("dtype", [np.str_, np.bytes_, "S1", "U1"])
@pytest.mark.parametrize("arg", ["include", "exclude"])
def test_select_dtypes_numpy_string_dtypes_raise(dtype, arg):
    gdf = cudf.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    with pytest.raises(TypeError, match="string dtypes are not allowed"):
        gdf.select_dtypes(**{arg: [dtype]})


def test_select_dtypes_object_includes_str_with_warning():
    pdf = pd.DataFrame({"a": ["x", "y"], "b": [1, 2]})
    gdf = cudf.DataFrame(pdf)
    with pytest.warns(match="For backward compatibility, 'str' dtypes"):
        expected = pdf.select_dtypes(include=["object"])
    with pytest.warns(match="For backward compatibility, 'str' dtypes"):
        result = gdf.select_dtypes(include=["object"])
    assert_eq(expected, result)


def test_reindex_empty_columns_preserves_label_dtype():
    # pandas' Index.reindex treats an empty non-Index target as
    # ``columns[:0]``, preserving the columns' metadata.
    pdf = pd.DataFrame({"a": [1], "b": [2]})
    gdf = cudf.DataFrame(pdf)
    assert_eq(pdf.reindex(columns=[]), gdf.reindex(columns=[]))
    assert (
        gdf.reindex(columns=[]).to_pandas().columns.dtype
        == pdf.reindex(columns=[]).columns.dtype
    )


@pytest.mark.parametrize("rows", [0, 2])
@pytest.mark.parametrize(
    "selection", [{"include": "number"}, {"exclude": "str"}]
)
def test_select_dtypes_empty_selection_preserves_rangeindex_columns(
    rows, selection
):
    pdf = pd.DataFrame(
        {
            0: pd.Series(["x"] * rows, dtype="str"),
            1: pd.Series(["y"] * rows, dtype="str"),
        }
    )
    pdf.columns = pd.RangeIndex(2, name="columns")
    expected = pdf.select_dtypes(**selection)
    actual = cudf.from_pandas(pdf).select_dtypes(**selection).to_pandas()
    pd.testing.assert_index_equal(expected.columns, actual.columns, exact=True)
    pd.testing.assert_frame_equal(expected, actual)

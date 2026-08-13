# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import expect_warning_if


@pytest.mark.parametrize("null_flag", [False, True])
def test_kurtosis_df(null_flag, numeric_only):
    data = cudf.DataFrame(
        {
            "a": np.arange(10, dtype="float64"),
            "b": np.arange(10, dtype="int64"),
            "c": np.arange(10, dtype="float64"),
            "d": ["a"] * 10,
        }
    )
    if not numeric_only:
        data = data.select_dtypes(include="number")
    pdata = data.to_pandas()

    if null_flag and len(data) > 2:
        data.iloc[[0, 2]] = None
        pdata.iloc[[0, 2]] = None

    got = data.kurtosis(numeric_only=numeric_only)
    got = got if np.isscalar(got) else got.to_numpy()

    expected = pdata.kurtosis(numeric_only=numeric_only)
    np.testing.assert_array_almost_equal(got, expected)

    got = data.kurt(numeric_only=numeric_only)
    got = got if np.isscalar(got) else got.to_numpy()

    expected = pdata.kurt(numeric_only=numeric_only)
    np.testing.assert_array_almost_equal(got, expected)


@pytest.mark.parametrize("null_flag", [False, True])
def test_skew_df(null_flag, numeric_only):
    data = cudf.DataFrame(
        {
            "a": np.arange(10, dtype="float64"),
            "b": np.arange(10, dtype="int64"),
            "c": np.arange(10, dtype="float64"),
            "d": ["a"] * 10,
        }
    )
    if not numeric_only:
        data = data.select_dtypes(include="number")
    pdata = data.to_pandas()

    if null_flag and len(data) > 2:
        data.iloc[[0, 2]] = None
        pdata.iloc[[0, 2]] = None

    got = data.skew(numeric_only=numeric_only)
    expected = pdata.skew(numeric_only=numeric_only)
    got = got if np.isscalar(got) else got.to_numpy()
    np.testing.assert_array_almost_equal(got, expected)


def test_single_q():
    q = 0.5

    pdf = pd.DataFrame({"a": [4, 24, 13, 8, 7]})
    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantile(q, interpolation="nearest", method="table")

    assert_eq(pdf_q, gdf_q, check_index_type=False)


def test_with_index():
    q = [0, 0.5, 1]

    pdf = pd.DataFrame({"a": [7, 4, 4, 9, 13]}, index=[0, 4, 3, 2, 7])
    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantile(q, interpolation="nearest", method="table")

    assert_eq(pdf_q, gdf_q, check_index_type=False)


@pytest.mark.parametrize("method", ["single", "table"])
def test_quantile_columns_subset(method):
    # The cudf-specific ``columns`` argument restricts the computation in
    # both the per-column and the row-selecting table methods.
    q = [0, 0.5, 1]

    pdf = pd.DataFrame({"a": [4, 24, 13, 8, 7], "b": [1, 2, 3, 4, 5]})
    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf[["a"]].quantile(q, interpolation="nearest")
    gdf_q = gdf.quantile(
        q, interpolation="nearest", method=method, columns=["a"]
    )

    assert list(gdf_q._column_names) == ["a"]
    assert_eq(pdf_q, gdf_q, check_index_type=False)


def test_with_multiindex():
    q = [0, 0.5, 1]

    pdf = pd.DataFrame(
        {
            "index_1": [3, 1, 9, 7, 5],
            "index_2": [2, 4, 3, 5, 1],
            "a": [8, 4, 2, 3, 8],
        }
    )
    pdf.set_index(["index_1", "index_2"], inplace=True)

    gdf = cudf.from_pandas(pdf)

    pdf_q = pdf.quantile(q, interpolation="nearest")
    gdf_q = gdf.quantile(q, interpolation="nearest", method="table")

    assert_eq(pdf_q, gdf_q, check_index_type=False)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [10, 11, 12]},
        {"a": [1, 0, 3], "b": [10, 11, 12]},
        {"a": [1, 2, 3], "b": [10, 11, None]},
        {
            "a": [],
        },
        {},
    ],
)
@pytest.mark.parametrize("op", ["all", "any"])
def test_any_all_axis_none(data, op):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    expected = getattr(pdf, op)(axis=None)
    actual = getattr(gdf, op)(axis=None)

    assert expected == actual


def test_reductions_axis_none(request, reduction_methods):
    if reduction_methods == "quantile":
        pytest.skip(f"pandas {reduction_methods} doesn't support axis=None")
    if reduction_methods in {"std", "var"}:
        request.applymarker(
            pytest.mark.xfail(
                reason=f"cuDF result incorrect for {reduction_methods}"
            )
        )
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [10, 2, 3]})
    pdf = df.to_pandas()
    actual = getattr(df, reduction_methods)(axis=None)
    expected = getattr(pdf, reduction_methods)(axis=None)
    assert_eq(expected, actual, check_dtype=False)


def test_dataframe_reduction_no_args(reduction_methods):
    df = cudf.DataFrame({"a": range(10), "b": range(10)})
    pdf = df.to_pandas()
    result = getattr(df, reduction_methods)()
    expected = getattr(pdf, reduction_methods)()
    assert_eq(result, expected)


def test_reduction_column_multiindex():
    idx = cudf.MultiIndex.from_tuples(
        [("a", 1), ("a", 2)], names=["foo", "bar"]
    )
    df = cudf.DataFrame(np.array([[1, 3], [2, 4]]), columns=idx)
    result = df.mean()
    expected = df.to_pandas().mean()
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "columns", [pd.RangeIndex(2), pd.Index([0, 1], dtype="int8")]
)
def test_dataframe_axis_0_preserve_column_type_in_index(columns):
    pd_df = pd.DataFrame([[1, 2]], columns=columns)
    cudf_df = cudf.DataFrame(pd_df)
    result = cudf_df.sum(axis=0)
    expected = pd_df.sum(axis=0)
    assert_eq(result, expected, check_index_type=True)


def test_dataframe_reduction_error():
    gdf = cudf.DataFrame(
        {
            "a": cudf.Series([1, 2, 3], dtype="float"),
            "d": cudf.Series([10, 20, 30], dtype="timedelta64[ns]"),
        }
    )

    with pytest.raises(TypeError):
        gdf.sum()


def test_mean_timeseries(numeric_only):
    gdf = cudf.DataFrame(
        {"a": ["a", "b", "c"], "b": range(3), "c": [-1.0, 12.2, 0.0]},
        index=pd.date_range("2020-01-01", periods=3, name="timestamp"),
    )
    if not numeric_only:
        gdf = gdf.select_dtypes(include="number")
    pdf = gdf.to_pandas()

    expected = pdf.mean(numeric_only=numeric_only)
    actual = gdf.mean(numeric_only=numeric_only)

    assert_eq(expected, actual)


def test_std_different_dtypes(numeric_only):
    gdf = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "b", "c", "d", "e"],
            "c": [1.0, 2.0, 3.0, 4.0, 5.0],
        }
    )
    if not numeric_only:
        gdf = gdf.select_dtypes(include="number")
    pdf = gdf.to_pandas()

    expected = pdf.std(numeric_only=numeric_only)
    actual = gdf.std(numeric_only=numeric_only)

    assert_eq(expected, actual)


def test_empty_numeric_only():
    gdf = cudf.DataFrame(
        {
            "id": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "val1": ["v", "n", "k", "l", "m", "i", "y", "r", "w"],
            "val2": ["d", "d", "d", "e", "e", "e", "f", "f", "f"],
        }
    )
    pdf = gdf.to_pandas()
    expected = pdf.prod(numeric_only=True)
    actual = gdf.prod(numeric_only=True)
    assert_eq(expected, actual, check_dtype=True)


@pytest.mark.parametrize(
    "op",
    ["kurt", "kurtosis", "skew"],
)
def test_dataframe_axis1_unsupported_ops(op):
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [8, 9, 10]})

    with pytest.raises(
        NotImplementedError, match=r"Only axis=0 is currently supported."
    ):
        getattr(df, op)(axis=1)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [1, 2, 3], "b": [8, 9, 10]},
        {"a": [1.0, np.nan, 3.0], "b": ["x", None, "z"], "c": [1, 2, None]},
        {"a": [None, None], "b": [None, "y"]},
        {"a": [1, 2, 3]},
    ],
)
@pytest.mark.parametrize("numeric_only", [False, True])
def test_dataframe_count_axis1(data, numeric_only):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()
    assert_eq(
        gdf.count(axis=1, numeric_only=numeric_only),
        pdf.count(axis=1, numeric_only=numeric_only),
    )


@pytest.mark.parametrize(
    "data",
    [
        {
            "x": [np.nan, 2, 3, 4, 100, np.nan],
            "y": [4, 5, 6, 88, 99, np.nan],
            "z": [7, 8, 9, 66, np.nan, 77],
        },
        {"x": [1, 2, 3], "y": [4, 5, 6], "z": [7, 8, 9]},
        {
            "x": [np.nan, np.nan, np.nan],
            "y": [np.nan, np.nan, np.nan],
            "z": [np.nan, np.nan, np.nan],
        },
        {"x": [], "y": [], "z": []},
        {"x": []},
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize(
    "func",
    [
        "min",
        "max",
        "sum",
        "prod",
        "product",
        "cummin",
        "cummax",
        "cumsum",
        "cumprod",
        "mean",
        "median",
        "sum",
        "std",
        "var",
        "kurt",
        "skew",
        "all",
        "any",
    ],
)
def test_dataframe_reductions(request, data, axis, func, skipna):
    pdf = pd.DataFrame(data=data)
    gdf = cudf.DataFrame(pdf, nan_as_null=False)

    if request.node.callspec.id in {
        "True-cumsum-1-data0",
        "True-cumprod-1-data0",
        "True-any-1-data2",
    }:
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/20628"
            )
        )

    # Reductions can fail in numerous possible ways when attempting row-wise
    # reductions, which are only partially supported. Catching the appropriate
    # exception here allows us to detect API breakage in the form of changing
    # exceptions.
    expected_exception = None
    if axis == 1:
        if func in ("kurt", "skew"):
            expected_exception = NotImplementedError
        elif func not in cudf.core.dataframe._cupy_nan_methods_map:
            if skipna is False:
                expected_exception = NotImplementedError
            elif any(col._column.nullable for name, col in gdf.items()):
                expected_exception = ValueError
            elif func in ("cummin", "cummax"):
                expected_exception = AttributeError

    # Test different degrees of freedom for var and std.
    all_kwargs = [{"ddof": 1}, {"ddof": 2}] if func in ("var", "std") else [{}]
    for kwargs in all_kwargs:
        if expected_exception is not None:
            with pytest.raises(expected_exception):
                (getattr(gdf, func)(axis=axis, skipna=skipna, **kwargs),)
        else:
            expect = getattr(pdf, func)(axis=axis, skipna=skipna, **kwargs)
            with expect_warning_if(
                skipna
                and func in {"min", "max"}
                and axis == 1
                and any(gdf.T[col].isna().all() for col in gdf.T),
                RuntimeWarning,
            ):
                got = getattr(gdf, func)(axis=axis, skipna=skipna, **kwargs)
            assert_eq(got, expect)


@pytest.mark.parametrize(
    "data",
    [
        {"x": [np.nan, 2, 3, 4, 100, np.nan], "y": [4, 5, 6, 88, 99, np.nan]},
        {"x": [1, 2, 3], "y": [4, 5, 6]},
        {"x": [np.nan, np.nan, np.nan], "y": [np.nan, np.nan, np.nan]},
        {"x": [], "y": []},
        {"x": []},
    ],
)
def test_dataframe_count_reduction(data):
    pdf = pd.DataFrame(data=data)
    gdf = cudf.DataFrame(pdf)

    assert_eq(pdf.count(), gdf.count())


@pytest.mark.parametrize(
    "data",
    [
        {"x": [np.nan, 2, 3, 4, 100, np.nan], "y": [4, 5, 6, 88, 99, np.nan]},
        {"x": [1, 2, 3], "y": [4, 5, 6]},
        {"x": [np.nan, np.nan, np.nan], "y": [np.nan, np.nan, np.nan]},
        {"x": pd.Series([], dtype="float"), "y": pd.Series([], dtype="float")},
        {"x": pd.Series([], dtype="int")},
    ],
)
@pytest.mark.parametrize("ops", ["sum", "product", "prod"])
@pytest.mark.parametrize("min_count", [-10, -1, 0, 3])
def test_dataframe_min_count_ops(data, ops, skipna, min_count):
    psr = pd.DataFrame(data)
    gsr = cudf.from_pandas(psr)

    assert_eq(
        getattr(psr, ops)(skipna=skipna, min_count=min_count),
        getattr(gsr, ops)(skipna=skipna, min_count=min_count),
        check_dtype=False,
    )


@pytest.mark.parametrize("q", [0.5, 1, 0.001, [0.5], [], [0.005, 0.5, 1]])
def test_quantile(q, numeric_only):
    ts = pd.date_range("2018-08-24", periods=5, freq="D")
    td = pd.to_timedelta(np.arange(5), unit="h")
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {"date": ts, "delta": td, "val": rng.standard_normal(len(ts))}
    )
    gdf = cudf.DataFrame(pdf)

    assert_eq(pdf["date"].quantile(q), gdf["date"].quantile(q))
    assert_eq(pdf["delta"].quantile(q), gdf["delta"].quantile(q))
    assert_eq(pdf["val"].quantile(q), gdf["val"].quantile(q))

    q = q if isinstance(q, list) else [q]
    assert_eq(
        pdf.quantile(q, numeric_only=numeric_only),
        gdf.quantile(q, numeric_only=numeric_only),
    )


@pytest.mark.parametrize("q", [0.2, 1, 0.001, [0.5], [], [0.005, 0.8, 0.03]])
@pytest.mark.parametrize("interpolation", ["higher", "lower", "nearest"])
@pytest.mark.parametrize(
    "decimal_type",
    [cudf.Decimal32Dtype, cudf.Decimal64Dtype, cudf.Decimal128Dtype],
)
def test_decimal_quantile(q, interpolation, decimal_type):
    rng = np.random.default_rng(seed=0)
    data = ["244.8", "32.24", "2.22", "98.14", "453.23", "5.45"]
    gdf = cudf.DataFrame(
        {"id": rng.integers(0, 10, size=len(data)), "val": data}
    )
    gdf["id"] = gdf["id"].astype("float64")
    gdf["val"] = gdf["val"].astype(decimal_type(7, 2))
    pdf = gdf.to_pandas()

    # Scalar q would require a Series result, which is not possible for
    # this mix of float64 and decimal columns.
    q = q if isinstance(q, list) else [q]
    got = gdf.quantile(q, numeric_only=False, interpolation=interpolation)
    expected = pdf.quantile(q, numeric_only=False, interpolation=interpolation)

    assert_eq(got, expected)


def test_empty_quantile():
    pdf = pd.DataFrame({"x": []}, dtype="float64")
    df = cudf.DataFrame({"x": []}, dtype="float64")

    actual = df.quantile()
    expected = pdf.quantile()

    assert_eq(actual, expected)


def test_quantile_q_scalar_datetime():
    # For a scalar q, a Series is returned when the columns share a
    # common dtype, matching pandas.
    ts = pd.date_range("2018-08-24", periods=5, freq="D")
    pdf = pd.DataFrame({"a": ts, "b": ts + pd.Timedelta(hours=1)})
    gdf = cudf.DataFrame(pdf)

    assert_eq(
        pdf.quantile(0.5, numeric_only=False),
        gdf.quantile(0.5, numeric_only=False),
    )


def test_quantile_q_scalar_mixed_dtypes_raises():
    gdf = cudf.DataFrame(
        {"a": [1.0, 2.0], "b": pd.to_datetime(["2010", "2011"])}
    )
    with pytest.raises(cudf.errors.MixedTypeError):
        gdf.quantile(0.5, numeric_only=False)


def test_quantile_table_method_tz_metadata():
    dti = pd.date_range("2016-01-01", periods=3, tz="US/Pacific")
    pdf = pd.DataFrame({"a": dti})
    gdf = cudf.DataFrame(pdf)

    assert_eq(
        pdf.quantile(0.5, numeric_only=False, interpolation="nearest"),
        gdf.quantile(0.5, interpolation="nearest", method="table"),
    )


def test_quantile_table_method_numeric_only():
    pdf = pd.DataFrame(
        {
            "date": pd.date_range("2018-08-24", periods=3, freq="D"),
            "val": [1, 2, 3],
        }
    )
    gdf = cudf.DataFrame(pdf)

    assert_eq(
        pdf.quantile(0.5, numeric_only=True, interpolation="nearest"),
        gdf.quantile(
            0.5,
            numeric_only=True,
            interpolation="nearest",
            method="table",
        ),
    )


def test_quantile_table_method_empty():
    pdf = pd.DataFrame({"x": [], "y": []})
    gdf = cudf.DataFrame({"x": [], "y": []})

    assert_eq(
        pdf.quantile(0.5, interpolation="nearest"),
        gdf.quantile(0.5, interpolation="nearest", method="table"),
    )
    assert_eq(
        pdf.quantile([0.5], interpolation="nearest"),
        gdf.quantile([0.5], interpolation="nearest", method="table"),
    )


@pytest.mark.parametrize(
    "data",
    [
        [0, 1, 2, 3],
        [-2, -1, 2, 3, 5],
        [-2, -1, 0, 3, 5],
        [True, False, False],
        [True],
        [False],
        [],
        [True, None, False],
        [True, True, None],
        [None, None],
        [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]],
        [[1, True], [2, False], [3, False]],
        [["a", True], ["b", False], ["c", False]],
    ],
)
def test_all(data):
    # Provide a dtype when data is empty to avoid future pandas changes.
    dtype = None if data else float
    if np.array(data).ndim <= 1:
        pdata = pd.Series(data=data, dtype=dtype)
        gdata = cudf.Series(pdata)
        got = gdata.all()
        expected = pdata.all()
        assert_eq(got, expected)
    else:
        pdata = pd.DataFrame(data, columns=["a", "b"], dtype=dtype).replace(
            [None], False
        )
        gdata = cudf.DataFrame(pdata)

        # test bool_only
        if pdata["b"].dtype == "bool":
            got = gdata.all(bool_only=True)
            expected = pdata.all(bool_only=True)
            assert_eq(got, expected)
        else:
            got = gdata.all()
            expected = pdata.all()
            assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        [0, 1, 2, 3],
        [-2, -1, 2, 3, 5],
        [-2, -1, 0, 3, 5],
        [0, 0, 0, 0, 0],
        [0, 0, None, 0],
        [True, False, False],
        [True],
        [False],
        [],
        [True, None, False],
        [True, True, None],
        [None, None],
        [[0, 5], [1, 6], [2, 7], [3, 8], [4, 9]],
        [[1, True], [2, False], [3, False]],
        [["a", True], ["b", False], ["c", False]],
    ],
)
@pytest.mark.parametrize("axis", [0, 1])
def test_any(data, axis):
    # Provide a dtype when data is empty to avoid future pandas changes.
    dtype = float if all(x is None for x in data) or len(data) < 1 else None
    if np.array(data).ndim <= 1:
        pdata = pd.Series(data=data, dtype=dtype)
        gdata = cudf.Series(data=data, dtype=dtype)

        if axis == 1:
            with pytest.raises(NotImplementedError):
                gdata.any(axis=axis)
        else:
            got = gdata.any(axis=axis)
            expected = pdata.any(axis=axis)
            assert_eq(got, expected)
    else:
        pdata = pd.DataFrame(data, columns=["a", "b"])
        gdata = cudf.DataFrame(pdata)

        # test bool_only
        if pdata["b"].dtype == "bool":
            got = gdata.any(bool_only=True)
            expected = pdata.any(bool_only=True)
            assert_eq(got, expected)
        else:
            got = gdata.any(axis=axis)
            expected = pdata.any(axis=axis)
            assert_eq(got, expected)


def test_empty_dataframe_any(axis):
    pdf = pd.DataFrame({}, columns=["a", "b"], dtype=float)
    gdf = cudf.DataFrame(pdf)
    got = gdf.any(axis=axis)
    expected = pdf.any(axis=axis)
    assert_eq(got, expected, check_index_type=False)


@pytest.mark.parametrize(
    "data",
    [
        lambda: cudf.datasets.randomdata(
            nrows=10, dtypes={"a": "category", "b": int, "c": float, "d": int}
        ),
        lambda: cudf.datasets.randomdata(
            nrows=10, dtypes={"a": "category", "b": int, "c": float, "d": str}
        ),
        lambda: cudf.datasets.randomdata(
            nrows=10, dtypes={"a": bool, "b": int, "c": float, "d": str}
        ),
        lambda: cudf.DataFrame(),
        lambda: cudf.DataFrame({"a": [0, 1, 2], "b": [1, None, 3]}),
        lambda: cudf.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [7, np.nan, 9, 10],
                "c": cudf.Series(
                    [np.nan, np.nan, np.nan, np.nan], nan_as_null=False
                ),
                "d": cudf.Series([None, None, None, None], dtype="int64"),
                "e": [100, None, 200, None],
                "f": cudf.Series([10, None, np.nan, 11], nan_as_null=False),
            }
        ),
        lambda: cudf.DataFrame(
            {
                "a": [10, 11, 12, 13, 14, 15],
                "b": cudf.Series(
                    [10, None, np.nan, 2234, None, np.nan], nan_as_null=False
                ),
            }
        ),
    ],
)
def test_rowwise_ops(data, reduction_methods, skipna, numeric_only):
    if reduction_methods in (
        "median",
        "quantile",
        "skew",
        "kurtosis",
        "any",
        "all",
    ):
        pytest.skip(f"Test not meant to test {reduction_methods}")
    gdf = data()
    pdf = gdf.to_pandas()

    kwargs = {"axis": 1, "skipna": skipna, "numeric_only": numeric_only}
    if reduction_methods in ("var", "std"):
        kwargs["ddof"] = 0

    if not numeric_only and not all(
        (
            (pdf[column].count() == 0)
            if skipna
            else (pdf[column].notna().count() == 0)
        )
        or cudf.api.types.is_numeric_dtype(pdf[column].dtype)
        or pdf[column].dtype.kind == "b"
        for column in pdf
    ):
        with pytest.raises(TypeError):
            expected = getattr(pdf, reduction_methods)(**kwargs)
        with pytest.raises(TypeError):
            got = getattr(gdf, reduction_methods)(**kwargs)
    else:
        expected = getattr(pdf, reduction_methods)(**kwargs)
        got = getattr(gdf, reduction_methods)(**kwargs)

        assert_eq(
            expected,
            got,
            check_dtype=False,
            check_index_type=False if len(got.index) == 0 else True,
        )


def test_rowwise_ops_nullable_dtypes_all_null(reduction_methods):
    if reduction_methods in (
        "median",
        "quantile",
        "skew",
        "kurtosis",
        "any",
        "all",
    ):
        pytest.skip(f"Test not meant to test {reduction_methods}")
    gdf = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [7, np.nan, 9, 10],
            "c": cudf.Series([np.nan, np.nan, np.nan, np.nan], dtype=float),
            "d": cudf.Series([None, None, None, None], dtype="int64"),
            "e": [100, None, 200, None],
            "f": cudf.Series([10, None, np.nan, 11], nan_as_null=False),
        }
    )

    expected = cudf.Series([None, None, None, None], dtype="float64")

    if reduction_methods in ("var", "std"):
        got = getattr(gdf, reduction_methods)(axis=1, ddof=0, skipna=False)
    else:
        got = getattr(gdf, reduction_methods)(axis=1, skipna=False)

    assert_eq(got.null_count, expected.null_count)
    assert_eq(got, expected)


def test_rowwise_ops_nullable_dtypes_partial_null(reduction_methods):
    if reduction_methods in (
        "median",
        "quantile",
        "skew",
        "kurtosis",
        "any",
        "all",
    ):
        pytest.skip(f"Test not meant to test {reduction_methods}")
    gdf = cudf.DataFrame(
        {
            "a": [10, 11, 12, 13, 14, 15],
            "b": cudf.Series(
                [10, None, np.nan, 2234, None, np.nan],
                nan_as_null=False,
            ),
        }
    )

    if reduction_methods in ("var", "std"):
        got = getattr(gdf, reduction_methods)(axis=1, ddof=0, skipna=False)
        expected = getattr(gdf.to_pandas(), reduction_methods)(
            axis=1, ddof=0, skipna=False
        )
    else:
        got = getattr(gdf, reduction_methods)(axis=1, skipna=False)
        expected = getattr(gdf.to_pandas(), reduction_methods)(
            axis=1, skipna=False
        )

    assert_eq(got.null_count, 2)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "op,expected",
    [
        (
            "max",
            lambda: cudf.Series(
                [10, None, None, 2234, None, 453],
                dtype="int64",
            ),
        ),
        (
            "min",
            lambda: cudf.Series(
                [10, None, None, 13, None, 15],
                dtype="int64",
            ),
        ),
        (
            "sum",
            lambda: cudf.Series(
                [20, None, None, 2247, None, 468],
                dtype="int64",
            ),
        ),
        (
            "product",
            lambda: cudf.Series(
                [100, None, None, 29042, None, 6795],
                dtype="int64",
            ),
        ),
        (
            "mean",
            lambda: cudf.Series(
                [10.0, None, None, 1123.5, None, 234.0],
                dtype="float32",
            ),
        ),
        (
            "var",
            lambda: cudf.Series(
                [0.0, None, None, 1233210.25, None, 47961.0],
                dtype="float32",
            ),
        ),
        (
            "std",
            lambda: cudf.Series(
                [0.0, None, None, 1110.5, None, 219.0],
                dtype="float32",
            ),
        ),
    ],
)
def test_rowwise_ops_nullable_int_dtypes(op, expected):
    gdf = cudf.DataFrame(
        {
            "a": [10, 11, None, 13, None, 15],
            "b": cudf.Series(
                [10, None, 323, 2234, None, 453],
                nan_as_null=False,
            ),
        }
    )

    if op in ("var", "std"):
        got = getattr(gdf, op)(axis=1, ddof=0, skipna=False)
    else:
        got = getattr(gdf, op)(axis=1, skipna=False)

    expected = expected()
    assert_eq(got.null_count, expected.null_count)
    assert_eq(got, expected)


@pytest.mark.parametrize(
    "data",
    [
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": pd.Series(
                ["1940-08-31 06:00:00", "2020-08-02 10:00:00"], dtype="<M8[ms]"
            ),
        },
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": pd.Series(
                ["1940-08-31 06:00:00", "2020-08-02 10:00:00"], dtype="<M8[ns]"
            ),
            "t3": pd.Series(
                ["1960-08-31 06:00:00", "2030-08-02 10:00:00"], dtype="<M8[s]"
            ),
        },
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": pd.Series(
                ["1940-08-31 06:00:00", "2020-08-02 10:00:00"], dtype="<M8[us]"
            ),
        },
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": pd.Series(
                ["1940-08-31 06:00:00", "2020-08-02 10:00:00"], dtype="<M8[ms]"
            ),
            "i1": pd.Series([1001, 2002], dtype="int64"),
        },
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "t2": pd.Series(["1940-08-31 06:00:00", None], dtype="<M8[ms]"),
            "i1": pd.Series([1001, 2002], dtype="int64"),
        },
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "i1": pd.Series([1001, 2002], dtype="int64"),
            "f1": pd.Series([-100.001, 123.456], dtype="float64"),
        },
        {
            "t1": pd.Series(
                ["2020-08-01 09:00:00", "1920-05-01 10:30:00"], dtype="<M8[ms]"
            ),
            "i1": pd.Series([1001, 2002], dtype="int64"),
            "f1": pd.Series([-100.001, 123.456], dtype="float64"),
            "b1": pd.Series([True, False], dtype="bool"),
        },
    ],
)
@pytest.mark.parametrize("op", ["max", "min"])
def test_rowwise_ops_datetime_dtypes(data, op, skipna, numeric_only):
    gdf = cudf.DataFrame(data)
    pdf = gdf.to_pandas()

    if not numeric_only and not all(dt.kind == "M" for dt in gdf.dtypes):
        with pytest.raises(TypeError):
            got = getattr(gdf, op)(
                axis=1, skipna=skipna, numeric_only=numeric_only
            )
        with pytest.raises(TypeError):
            expected = getattr(pdf, op)(
                axis=1, skipna=skipna, numeric_only=numeric_only
            )
    else:
        got = getattr(gdf, op)(
            axis=1, skipna=skipna, numeric_only=numeric_only
        )
        expected = getattr(pdf, op)(
            axis=1, skipna=skipna, numeric_only=numeric_only
        )
        if got.dtype == cudf.dtype(
            "datetime64[us]"
        ) and expected.dtype == np.dtype("datetime64[ns]"):
            # Workaround for a PANDAS-BUG:
            # https://github.com/pandas-dev/pandas/issues/52524
            assert_eq(got.astype("datetime64[ns]"), expected)
        else:
            assert_eq(got, expected, check_dtype=False)


@pytest.mark.parametrize(
    "data,op,skipna",
    [
        (
            {
                "t1": pd.Series(
                    ["2020-08-01 09:00:00", "1920-05-01 10:30:00"],
                    dtype="<M8[ms]",
                ),
                "t2": pd.Series(
                    ["1940-08-31 06:00:00", None], dtype="<M8[ms]"
                ),
            },
            "max",
            True,
        ),
        (
            {
                "t1": pd.Series(
                    ["2020-08-01 09:00:00", "1920-05-01 10:30:00"],
                    dtype="<M8[ms]",
                ),
                "t2": pd.Series(
                    ["1940-08-31 06:00:00", None], dtype="<M8[ms]"
                ),
            },
            "min",
            False,
        ),
        (
            {
                "t1": pd.Series(
                    ["2020-08-01 09:00:00", "1920-05-01 10:30:00"],
                    dtype="<M8[ms]",
                ),
                "t2": pd.Series(
                    ["1940-08-31 06:00:00", None], dtype="<M8[ms]"
                ),
            },
            "min",
            True,
        ),
    ],
)
def test_rowwise_ops_datetime_dtypes_2(data, op, skipna):
    gdf = cudf.DataFrame(data)

    pdf = gdf.to_pandas()

    got = getattr(gdf, op)(axis=1, skipna=skipna)
    expected = getattr(pdf, op)(axis=1, skipna=skipna)

    assert_eq(got, expected)


def test_rowwise_ops_datetime_dtypes_pdbug():
    data = {
        "t1": pd.Series(
            ["2020-08-01 09:00:00", "1920-05-01 10:30:00"],
            dtype="<M8[ns]",
        ),
        "t2": pd.Series(["1940-08-31 06:00:00", pd.NaT], dtype="<M8[ns]"),
    }
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    expected = pdf.max(axis=1, skipna=False)
    got = gdf.max(axis=1, skipna=False)

    assert_eq(got, expected)

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import decimal
import itertools

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core.dtypes import ListDtype, StructDtype
from cudf.testing import assert_eq, assert_groupby_results_equal


@pytest.mark.parametrize("empty", [True, False])
def test_agg_count_dtype(empty):
    df = cudf.DataFrame({"a": [1, 2, 1], "c": ["a", "b", "c"]})
    if empty:
        df = df.iloc[:0]
    result = df.groupby("a").agg({"c": "count"})
    assert result["c"].dtype == np.dtype("int64")


@pytest.mark.parametrize("attr", ["agg", "aggregate"])
def test_series_agg(attr):
    df = cudf.DataFrame({"a": [1, 2, 1, 2], "b": [0, 0, 0, 0]})
    pdf = df.to_pandas()
    agg = getattr(df.groupby("a")["a"], attr)("count")
    pd_agg = getattr(pdf.groupby(["a"])["a"], attr)("count")

    assert agg.ndim == pd_agg.ndim


@pytest.mark.parametrize("func", ["sum", "prod", "mean", "count"])
@pytest.mark.parametrize("attr", ["agg", "aggregate"])
def test_dataframe_agg(attr, func):
    df = cudf.DataFrame({"a": [1, 2, 1, 2], "b": [0, 0, 0, 0]})
    pdf = df.to_pandas()

    agg = getattr(df.groupby("a"), attr)(func)
    pd_agg = getattr(pdf.groupby(["a"]), attr)(func)

    assert_eq(agg, pd_agg)

    agg = getattr(df.groupby("a"), attr)({"b": func})
    pd_agg = getattr(pdf.groupby(["a"]), attr)({"b": func})

    assert_eq(agg, pd_agg)

    agg = getattr(df.groupby("a"), attr)([func])
    pd_agg = getattr(pdf.groupby(["a"]), attr)([func])

    assert_eq(agg, pd_agg)

    agg = getattr(df.groupby("a"), attr)(foo=("b", func), bar=("a", func))
    pd_agg = getattr(pdf.groupby(["a"]), attr)(
        foo=("b", func), bar=("a", func)
    )

    assert_eq(agg, pd_agg)

    agg = getattr(df.groupby("a"), attr)(
        foo=cudf.NamedAgg(column="b", aggfunc=func),
        bar=cudf.NamedAgg(column="a", aggfunc=func),
    )
    pd_agg = getattr(pdf.groupby(["a"]), attr)(
        foo=("b", func), bar=("a", func)
    )

    assert_eq(agg, pd_agg)


def test_dataframe_agg_with_invalid_kwarg():
    with pytest.raises(TypeError, match="Invalid keyword argument"):
        df = cudf.DataFrame({"a": [1, 2, 1, 2], "b": [0, 0, 0, 0]})
        df.groupby("a").agg(foo=set())


@pytest.mark.parametrize("with_nulls", [False, True])
def test_groupby_agg_maintain_order_random(with_nulls):
    nrows = 20
    nkeys = 3
    rng = np.random.default_rng(seed=0)
    key_names = [f"key{key}" for key in range(nkeys)]
    key_values = [rng.integers(100, size=nrows) for _ in key_names]
    value = rng.integers(-100, 100, size=nrows)
    df = cudf.DataFrame(
        dict(zip(key_names, key_values, strict=True), value=value)
    )
    if with_nulls:
        for key in key_names:
            df.loc[df[key] == 1, key] = None
    with cudf.option_context("mode.pandas_compatible", True):
        got = df.groupby(key_names, sort=False).agg({"value": "sum"})
    expect = (
        df.to_pandas().groupby(key_names, sort=False).agg({"value": "sum"})
    )
    assert_eq(expect, got, check_index_type=not with_nulls)


def test_groupby_agg_mean_min():
    pdf = pd.DataFrame(np.ones((20, 3)), columns=["x", "y", "val"])
    gdf = cudf.DataFrame(pdf)
    got_df = gdf.groupby(["x", "y"]).agg(["mean", "min"])
    expect_df = pdf.groupby(["x", "y"]).agg(["mean", "min"])
    assert_groupby_results_equal(got_df, expect_df)


def test_groupby_agg_min_max_dictargs():
    pdf = pd.DataFrame(np.ones((20, 5)), columns=["x", "y", "val", "a", "b"])
    gdf = cudf.DataFrame(pdf)
    expect_df = pdf.groupby(["x", "y"]).agg({"a": "min", "b": "max"})
    got_df = gdf.groupby(["x", "y"]).agg({"a": "min", "b": "max"})
    assert_groupby_results_equal(expect_df, got_df)


def test_groupby_agg_min_max_dictlist():
    pdf = pd.DataFrame(np.ones((20, 5)), columns=["x", "y", "val", "a", "b"])
    gdf = cudf.DataFrame(pdf)
    expect_df = pdf.groupby(["x", "y"]).agg(
        {"a": ["min", "max"], "b": ["min", "max"]}
    )
    got_df = gdf.groupby(["x", "y"]).agg(
        {"a": ["min", "max"], "b": ["min", "max"]}
    )
    assert_groupby_results_equal(got_df, expect_df)


def test_groupby_as_index_single_agg(as_index):
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = gdf.groupby("y", as_index=as_index).agg({"x": "mean"})
    pdf = pdf.groupby("y", as_index=as_index).agg({"x": "mean"})
    assert_groupby_results_equal(pdf, gdf, as_index=as_index, by="y")


def test_groupby_default():
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = gdf.groupby("y").agg({"x": "mean"})
    pdf = pdf.groupby("y").agg({"x": "mean"})
    assert_groupby_results_equal(pdf, gdf)


def test_groupby_as_index_multiindex(as_index):
    pdf = pd.DataFrame(
        {"a": [1, 2, 1], "b": [3, 3, 3], "c": [2, 2, 3], "d": [3, 1, 2]}
    )
    gdf = cudf.from_pandas(pdf)

    gdf = gdf.groupby(["a", "b"], as_index=as_index, sort=True).agg(
        {"c": "mean"}
    )
    pdf = pdf.groupby(["a", "b"], as_index=as_index, sort=True).agg(
        {"c": "mean"}
    )

    if as_index:
        assert_eq(pdf, gdf)
    else:
        # column names don't match - check just the values
        for gcol, pcol in zip(gdf, pdf, strict=True):
            np.testing.assert_array_equal(
                gdf[gcol].to_numpy(), pdf[pcol].values
            )


@pytest.mark.parametrize(
    "func",
    [
        "mean",
        "std",
        "var",
        "min",
        "max",
        "idxmin",
        "idxmax",
        "count",
        "sum",
        "prod",
    ],
)
def test_groupby_2keys_agg(func):
    nelem = 20
    pdf = pd.DataFrame(np.ones((nelem, 2)), columns=["x", "y"])
    gdf = cudf.DataFrame(pdf)
    expect_df = pdf.groupby(["x", "y"]).agg(func)
    got_df = gdf.groupby(["x", "y"]).agg(func)
    # As of pandas 3.0, empty default type of object isn't
    # necessarily equivalent to cuDF's empty default type of
    # pandas.StringDtype
    expect_df.columns = expect_df.columns.astype(got_df.columns.dtype)
    assert_groupby_results_equal(got_df, expect_df)


def test_series_groupby_agg(groupby_reduction_methods):
    s = pd.Series([1, 2, 3])
    g = cudf.Series([1, 2, 3])
    sg = s.groupby(s // 2).agg(groupby_reduction_methods)
    gg = g.groupby(g // 2).agg(groupby_reduction_methods)
    assert_groupby_results_equal(sg, gg)


def test_groupby_agg_decimal(groupby_reduction_methods, request):
    request.applymarker(
        pytest.mark.xfail(
            groupby_reduction_methods in ["prod", "mean"],
            raises=pd.errors.DataError,
            reason=f"{groupby_reduction_methods} not supported with Decimals in pandas",
        )
    )
    rng = np.random.default_rng(seed=0)
    num_groups = 4
    nelem_per_group = 10
    # The number of digits after the decimal to use.
    decimal_digits = 2
    # The number of digits before the decimal to use.
    whole_digits = 2

    scale = 10**whole_digits
    nelem = num_groups * nelem_per_group

    # The unique is necessary because otherwise if there are duplicates idxmin
    # and idxmax may return different results than pandas (see
    # https://github.com/rapidsai/cudf/issues/7756). This is not relevant to
    # the current version of the test, because idxmin and idxmax simply don't
    # work with pandas Series composed of Decimal objects (see
    # https://github.com/pandas-dev/pandas/issues/40685). However, if that is
    # ever enabled, then this issue will crop up again so we may as well have
    # it fixed now.
    x = np.unique((rng.random(nelem) * scale).round(decimal_digits))
    y = np.unique((rng.random(nelem) * scale).round(decimal_digits))

    if x.size < y.size:
        total_elements = x.size
        y = y[: x.size]
    else:
        total_elements = y.size
        x = x[: y.size]

    # Note that this filtering can lead to one group with fewer elements, but
    # that shouldn't be a problem and is probably useful to test.
    idx_col = np.tile(np.arange(num_groups), nelem_per_group)[:total_elements]

    decimal_x = pd.Series([decimal.Decimal(str(d)) for d in x])
    decimal_y = pd.Series([decimal.Decimal(str(d)) for d in y])

    pdf = pd.DataFrame({"idx": idx_col, "x": decimal_x, "y": decimal_y})
    gdf = cudf.DataFrame(
        {
            "idx": idx_col,
            "x": cudf.Series(decimal_x),
            "y": cudf.Series(decimal_y),
        }
    )

    expect_df = pdf.groupby("idx", sort=True).agg(groupby_reduction_methods)
    got_df = gdf.groupby("idx", sort=True).agg(groupby_reduction_methods)
    assert_eq(expect_df["x"], got_df["x"], check_dtype=False)
    assert_eq(expect_df["y"], got_df["y"], check_dtype=False)


def test_groupby_use_agg_column_as_index():
    pdf = pd.DataFrame({"a": [1, 1, 1, 3, 5]})
    gdf = cudf.DataFrame({"a": [1, 1, 1, 3, 5]})
    gdf["a"] = [1, 1, 1, 3, 5]
    pdg = pdf.groupby("a").agg({"a": "count"})
    gdg = gdf.groupby("a").agg({"a": "count"})
    assert_groupby_results_equal(pdg, gdg, check_dtype=False)


def test_groupby_list_then_string():
    gdf = cudf.DataFrame(
        {"a": [0, 1, 0, 1, 2], "b": [11, 2, 15, 12, 2], "c": [6, 7, 6, 7, 6]}
    )
    pdf = gdf.to_pandas()
    gdg = gdf.groupby("a", as_index=True).agg(
        {"b": ["min", "max"], "c": "max"}
    )
    pdg = pdf.groupby("a", as_index=True).agg(
        {"b": ["min", "max"], "c": "max"}
    )
    assert_groupby_results_equal(gdg, pdg)


def test_groupby_different_unequal_length_column_aggregations():
    gdf = cudf.DataFrame(
        {"a": [0, 1, 0, 1, 2], "b": [11, 2, 15, 12, 2], "c": [6, 7, 6, 7, 6]}
    )
    pdf = gdf.to_pandas()
    gdg = gdf.groupby("a", as_index=True).agg(
        {"b": "min", "c": ["max", "min"]}
    )
    pdg = pdf.groupby("a", as_index=True).agg(
        {"b": "min", "c": ["max", "min"]}
    )
    assert_groupby_results_equal(pdg, gdg)


def test_groupby_single_var_two_aggs():
    gdf = cudf.DataFrame(
        {"a": [0, 1, 0, 1, 2], "b": [11, 2, 15, 12, 2], "c": [6, 7, 6, 7, 6]}
    )
    pdf = gdf.to_pandas()
    gdg = gdf.groupby("a", as_index=True).agg({"b": ["min", "max"]})
    pdg = pdf.groupby("a", as_index=True).agg({"b": ["min", "max"]})
    assert_groupby_results_equal(pdg, gdg)


def test_groupby_double_var_two_aggs():
    gdf = cudf.DataFrame(
        {"a": [0, 1, 0, 1, 2], "b": [11, 2, 15, 12, 2], "c": [6, 7, 6, 7, 6]}
    )
    pdf = gdf.to_pandas()
    gdg = gdf.groupby(["a", "b"], as_index=True).agg({"c": ["min", "max"]})
    pdg = pdf.groupby(["a", "b"], as_index=True).agg({"c": ["min", "max"]})
    assert_groupby_results_equal(pdg, gdg)


def test_groupby_multi_agg_single_groupby_series():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "x": rng.integers(0, 5, size=100),
            "y": rng.normal(size=100),
        }
    )
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby("x").y.agg(["sum", "max"])
    gdg = gdf.groupby("x").y.agg(["sum", "max"])

    assert_groupby_results_equal(pdg, gdg)


def test_groupby_multi_agg_multi_groupby():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": rng.integers(0, 5, 10),
            "b": rng.integers(0, 5, 10),
            "c": rng.integers(0, 5, 10),
            "d": rng.integers(0, 5, 10),
        }
    )
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby(["a", "b"]).agg(["sum", "max"])
    gdg = gdf.groupby(["a", "b"]).agg(["sum", "max"])
    assert_groupby_results_equal(pdg, gdg)


def test_groupby_datetime_multi_agg_multi_groupby():
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": pd.date_range(
                "2020-01-01",
                freq="D",
                periods=10,
            ),
            "b": rng.integers(0, 5, 10),
            "c": rng.integers(0, 5, 10),
            "d": rng.integers(0, 5, 10),
        }
    )
    gdf = cudf.from_pandas(pdf)
    pdg = pdf.groupby(["a", "b"]).agg(["sum", "max"])
    gdg = gdf.groupby(["a", "b"]).agg(["sum", "max"])

    assert_groupby_results_equal(pdg, gdg)


@pytest.mark.parametrize(
    "agg",
    [
        ["min", "max", "count", "mean"],
        ["mean", "var", "std"],
        ["count", "mean", "var", "std"],
    ],
)
def test_groupby_multi_agg_hash_groupby(agg):
    gdf = cudf.DataFrame(
        {"id": [0, 0, 1, 1, 2, 2, 0], "a": [0, 1, 2, 3, 4, 5, 6]}
    )
    pdf = gdf.to_pandas()
    check_dtype = "count" not in agg
    pdg = pdf.groupby("id").agg(agg)
    gdg = gdf.groupby("id").agg(agg)
    assert_groupby_results_equal(pdg, gdg, check_dtype=check_dtype)


def test_groupby_nulls_basic(groupby_reduction_methods):
    pdf = pd.DataFrame({"a": [0, 0, 1, 1, 2, 2], "b": [1, 2, 1, 2, 1, None]})
    gdf = cudf.from_pandas(pdf)
    assert_groupby_results_equal(
        getattr(pdf.groupby("a"), groupby_reduction_methods)(),
        getattr(gdf.groupby("a"), groupby_reduction_methods)(),
    )

    pdf = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 2, 2],
            "b": [1, 2, 1, 2, 1, None],
            "c": [1, 2, 1, None, 1, 2],
        }
    )
    gdf = cudf.from_pandas(pdf)
    assert_groupby_results_equal(
        getattr(pdf.groupby("a"), groupby_reduction_methods)(),
        getattr(gdf.groupby("a"), groupby_reduction_methods)(),
    )


def test_groupby_nulls_all_na_group(groupby_reduction_methods, request):
    pdf = pd.DataFrame(
        {
            "a": [0, 0, 1, 1, 2, 2],
            "b": [1, 2, 1, 2, 1, None],
            "c": [1, 2, None, None, 1, 2],
        }
    )
    gdf = cudf.from_pandas(pdf)

    if groupby_reduction_methods in ["idxmin", "idxmax"]:
        with pytest.raises(ValueError):
            getattr(gdf.groupby("a"), groupby_reduction_methods)()
        with pytest.raises(ValueError):
            getattr(pdf.groupby("a"), groupby_reduction_methods)()
    else:
        request.applymarker(
            pytest.mark.xfail(
                groupby_reduction_methods in ["prod", "sum"],
                reason="cuDF returns NaN instead of an actual value",
            )
        )
        assert_groupby_results_equal(
            getattr(pdf.groupby("a"), groupby_reduction_methods)(),
            getattr(gdf.groupby("a"), groupby_reduction_methods)(),
        )


@pytest.mark.parametrize("agg", [lambda x: x.count(), "count"])
@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_count(agg, by):
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 3], "b": [1, 2, 2, 2, 1], "c": [1, 2, None, 4, 5]}
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by).agg(agg)
    got = gdf.groupby(by).agg(agg)

    assert_groupby_results_equal(expect, got, check_dtype=True)


@pytest.mark.parametrize("agg", [lambda x: x.median(), "median"])
@pytest.mark.parametrize("by", ["a", ["a", "b"], ["a", "c"]])
def test_groupby_median(agg, by):
    pdf = pd.DataFrame(
        {"a": [1, 1, 1, 2, 3], "b": [1, 2, 2, 2, 1], "c": [1, 2, None, 4, 5]}
    )
    gdf = cudf.from_pandas(pdf)

    expect = pdf.groupby(by).agg(agg)
    got = gdf.groupby(by).agg(agg)

    assert_groupby_results_equal(expect, got, check_dtype=False)


def test_multi_agg():
    gdf = cudf.DataFrame(
        {"a": [1, 1, 2, 2], "b": [1, 2, 3, 4], "c": ["a", "b", "c", "d"]}
    )
    pdf = gdf.to_pandas()
    assert_groupby_results_equal(
        pdf.groupby("a").agg({"b": ["count", "mean"], "c": ["count"]}),
        gdf.groupby("a").agg({"b": ["count", "mean"], "c": ["count"]}),
    )


@pytest.mark.parametrize(
    "agg",
    (
        [
            *itertools.combinations(["count", "max", "min", "nunique"], 2),
            {"b": "min", "c": "mean"},
            {"b": "max", "c": "mean"},
            {"b": "count", "c": "mean"},
            {"b": "nunique", "c": "mean"},
        ]
    ),
)
def test_groupby_agg_combinations(agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 2, 2, 3],
            "b": ["a", "a", "b", "c", "d"],
            "c": [1, 2, 3, 4, 5],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").agg(agg),
        gdf.groupby("a").agg(agg),
        check_dtype=False,
    )


@pytest.mark.parametrize("list_agg", [list, "collect"])
def test_groupby_list_simple(list_agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": pd.array([1, 2, None, 4, 5, 6], dtype="double[pyarrow]"),
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").agg({"b": list}),
        gdf.groupby("a").agg({"b": list_agg}),
        check_dtype=False,
    )


@pytest.mark.parametrize("list_agg", [list, "collect"])
def test_groupby_list_of_lists(list_agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [[1, 2], [3, None, 5], None, [], [7, 8], [9]],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").agg({"b": list}),
        gdf.groupby("a").agg({"b": list_agg}),
        check_dtype=False,
    )


@pytest.mark.parametrize("list_agg", [list, "collect"])
def test_groupby_list_of_structs(list_agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 2],
            "b": [
                {"c": "1", "d": 1},
                {"c": "2", "d": 2},
                {"c": "3", "d": 3},
                {"c": "4", "d": 4},
                {"c": "5", "d": 5},
                {"c": "6", "d": 6},
            ],
        }
    )
    gdf = cudf.from_pandas(pdf)
    grouped = gdf.groupby("a").agg({"b": list_agg})
    assert_groupby_results_equal(
        pdf.groupby("a").agg({"b": list}),
        grouped,
        check_dtype=True,
    )
    assert grouped["b"].dtype.element_type == gdf["b"].dtype


@pytest.mark.parametrize("list_agg", [list, "collect"])
def test_groupby_list_single_element(list_agg):
    pdf = pd.DataFrame(
        {"a": [1, 2], "b": pd.array([3, None], dtype="double[pyarrow]")}
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").agg({"b": list}),
        gdf.groupby("a").agg({"b": list_agg}),
        check_dtype=False,
    )


@pytest.mark.parametrize(
    "agg", [list, [list, "count"], {"b": list, "c": "sum"}]
)
def test_groupby_list_strings(agg):
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": pd.array(["b", "a", None, "e", "d"], dtype="string[pyarrow]"),
            "c": [1, 2, 3, 4, 5],
        }
    )
    gdf = cudf.from_pandas(pdf)

    assert_groupby_results_equal(
        pdf.groupby("a").agg(agg),
        gdf.groupby("a").agg(agg),
        check_dtype=False,
    )


def test_groupby_list_columns_excluded():
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [1, 2, 3, 4],
            "c": [[1, 2], [3, 4], [5, 6], [7, 8]],
        }
    )
    gdf = cudf.from_pandas(pdf)

    pandas_result = pdf.groupby("a").mean(numeric_only=True)
    pandas_agg_result = pdf.groupby("a").agg("mean", numeric_only=True)

    assert_groupby_results_equal(
        pandas_result,
        gdf.groupby("a").mean(numeric_only=True),
        check_dtype=False,
    )

    assert_groupby_results_equal(
        pandas_agg_result,
        gdf.groupby("a").agg("mean"),
        check_dtype=False,
    )


def test_groupby_mix_agg_scan():
    err_msg = "Cannot perform both aggregation and scan in one operation"
    func = ["cumsum", "sum"]
    gb = cudf.DataFrame(np.ones((10, 3)), columns=["x", "y", "z"]).groupby(
        ["x", "y"], sort=True
    )

    gb.agg(func[0])
    gb.agg(func[1])
    gb.agg(func[1:])
    with pytest.raises(NotImplementedError, match=err_msg):
        gb.agg(func)


@pytest.mark.parametrize(
    "op", ["cummax", "cummin", "cumprod", "cumsum", "mean", "median"]
)
def test_group_by_raises_string_error(op):
    df = cudf.DataFrame({"a": [1, 2, 3, 4, 5], "b": ["a", "b", "c", "d", "e"]})

    with pytest.raises(TypeError):
        df.groupby(df.a).agg(op)


@pytest.mark.parametrize(
    "op",
    [
        "cummax",
        "cummin",
        "cumprod",
        "cumsum",
        "mean",
        "median",
        "prod",
        "sum",
        list,
    ],
)
def test_group_by_raises_category_error(op):
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": cudf.Series(["a", "b", "c", "d", "e"], dtype="category"),
        }
    )

    with pytest.raises(TypeError):
        df.groupby(df.a).agg(op)


def test_agg_duplicate_aggs_pandas_compat_raises():
    agg = {"b": ["mean", "mean"]}
    dfgb = cudf.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]}).groupby(["a"])
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            dfgb.agg(agg)

    with pytest.warns(UserWarning):
        result = dfgb.agg(agg)
    expected = cudf.DataFrame(
        [4.5, 6.0],
        index=cudf.Index([1, 2], name="a"),
        columns=pd.MultiIndex.from_tuples([("b", "mean")]),
    )
    assert_groupby_results_equal(result, expected)


def test_groupby_collect_nested_lists():
    """Test groupby collect on list columns creates properly nested dtypes."""
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [[1, 2], [3, 4], [5], [6, 7]],
        }
    )
    gdf = cudf.from_pandas(pdf)

    result = gdf.groupby("a").agg({"b": "collect"})

    assert result["b"].dtype == ListDtype(ListDtype("int64"))
    assert result["b"].dtype.element_type == ListDtype("int64")
    assert result["b"].dtype.element_type.element_type.name == "int64"


def test_groupby_collect_triple_nested():
    """Test groupby collect with multi-level nesting."""
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [[[1, 2], [3]], [[4], [5, 6]], [[7]], [[8, 9], [10]]],
        }
    )
    gdf = cudf.from_pandas(pdf)

    result = gdf.groupby("a").agg({"b": "collect"})

    assert result["b"].dtype == ListDtype(ListDtype(ListDtype("int64")))


def test_groupby_collect_struct_lists():
    """Test groupby collect with struct-containing lists."""
    pdf = pd.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [[{"x": 1}], [{"x": 2}], [{"x": 3}], [{"x": 4}]],
        }
    )
    gdf = cudf.from_pandas(pdf)

    result = gdf.groupby("a").agg({"b": "collect"})

    assert isinstance(result["b"].dtype, ListDtype)
    assert isinstance(result["b"].dtype.element_type, ListDtype)
    assert isinstance(result["b"].dtype.element_type.element_type, StructDtype)


def test_copy_with_nested_lists():
    """Test that copy preserves dtype for nested list columns."""
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [[1, 2], [3, 4], [5], [6, 7]],
        }
    )
    result = gdf.groupby("a").agg({"b": "collect"})

    copied = result["b"]._column.copy()
    assert copied.dtype == result["b"].dtype
    assert copied.dtype == ListDtype(ListDtype("int64"))


def test_sliced_child_dtype_accuracy():
    """Test that sliced child dtype matches stored element_type."""
    gdf = cudf.DataFrame(
        {
            "a": [1, 1, 2, 2],
            "b": [[1, 2], [3, 4], [5], [6, 7]],
        }
    )
    result = gdf.groupby("a").agg({"b": "collect"})

    col = result["b"]._column
    child = col._get_sliced_child()
    assert child.dtype == col.dtype.element_type


@pytest.mark.parametrize("op", ["idxmin", "idxmax"])
@pytest.mark.parametrize(
    "index", [[10, 20, 30, 40], ["w", "x", "y", "z"], None]
)
def test_groupby_idxminmax_returns_index_labels(op, index):
    # pandas returns the row's index *label* (with the index dtype), not
    # the positional row number, from agg/transform/the direct method
    pdf = pd.DataFrame(
        {"key": [1, 1, 2, 2], "val": [4.0, 3.0, 5.0, 6.0]}, index=index
    )
    gdf = cudf.DataFrame(pdf)

    assert_groupby_results_equal(
        getattr(pdf.groupby("key"), op)(), getattr(gdf.groupby("key"), op)()
    )
    assert_groupby_results_equal(
        pdf.groupby("key").agg(op), gdf.groupby("key").agg(op)
    )
    assert_eq(
        pdf.groupby("key")["val"].transform(op),
        gdf.groupby("key")["val"].transform(op),
    )


@pytest.mark.parametrize("op", ["idxmin", "idxmax"])
def test_groupby_idxminmax_all_na_group_raises(op):
    pdf = pd.DataFrame({"key": [1, 1, 2, 2], "val": [4.0, 3.0, None, None]})
    gdf = cudf.DataFrame(pdf)

    with pytest.raises(ValueError):
        getattr(pdf.groupby("key"), op)()
    with pytest.raises(ValueError):
        getattr(gdf.groupby("key"), op)()


def test_agg_multiindex_columns_preserved():
    # aggregating a MultiIndex-column frame keeps hierarchical columns
    pdf = pd.DataFrame(
        [[1, 2, 3], [1, 5, 6], [2, 8, 9]],
        columns=pd.MultiIndex.from_tuples(
            [("k", ""), ("x", "a"), ("x", "b")], names=["l0", "l1"]
        ),
    )
    gdf = cudf.DataFrame(pdf)

    expect = pdf.groupby(("k", "")).agg("sum")
    got = gdf.groupby(("k", "")).agg("sum")
    assert_eq(expect, got)


def test_agg_relabel_flat_columns_from_multiindex():
    # relabeling aggregations emit new flat labels; the source's
    # multi-level column metadata must not be attached to them
    pdf = pd.DataFrame(
        [[1, 2], [1, 5], [2, 8]],
        columns=pd.MultiIndex.from_tuples([("k", ""), ("x", "a")]),
    )
    gdf = cudf.DataFrame(pdf)

    expect = pdf.groupby(("k", "")).agg(total=(("x", "a"), "sum"))
    got = gdf.groupby(("k", "")).agg(total=(("x", "a"), "sum"))
    assert_eq(expect, got)

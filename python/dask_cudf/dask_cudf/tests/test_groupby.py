# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import pytest

import dask
from dask import dataframe as dd
from dask.utils_test import hlg_layer

import cudf
from cudf.core._compat import PANDAS_CURRENT_SUPPORTED_VERSION, PANDAS_VERSION

import dask_cudf
from dask_cudf._expr.groupby import OPTIMIZED_AGGS, _aggs_optimized


def assert_cudf_groupby_layers(ddf):
    for prefix in ("cudf-aggregate-chunk", "cudf-aggregate-agg"):
        try:
            hlg_layer(ddf.dask, prefix)
        except KeyError:
            raise AssertionError(
                "Expected Dask dataframe to contain groupby layer with "
                f"prefix {prefix}"
            )


@pytest.fixture(params=["non_null", "null"])
def pdf(request):
    rng = np.random.default_rng(seed=0)

    # note that column name "x" is a substring of the groupby key;
    # this gives us coverage for cudf#10829
    pdf = pd.DataFrame(
        {
            "xx": rng.integers(0, 5, size=10000),
            "x": rng.normal(size=10000),
            "y": rng.normal(size=10000),
        }
    )

    # insert nulls into dataframe at random
    if request.param == "null":
        pdf = pdf.mask(rng.choice([True, False], size=pdf.shape))

    return pdf


# NOTE: We only want to test aggregation "methods" here,
# so we need to leave out `list`. We also include a
# deprecation check for "collect".
@pytest.mark.parametrize(
    "aggregation",
    sorted(set(OPTIMIZED_AGGS) - {list}),
)
@pytest.mark.parametrize("series", [False, True])
def test_groupby_basic(series, aggregation, pdf):
    gdf = cudf.DataFrame(pdf)
    gdf_grouped = gdf.groupby("xx", dropna=True)
    ddf_grouped = dask_cudf.from_cudf(gdf, npartitions=5).groupby(
        "xx", dropna=True
    )

    if series:
        gdf_grouped = gdf_grouped.x
        ddf_grouped = ddf_grouped.x

    check_dtype = aggregation != "count"

    expect = getattr(gdf_grouped, aggregation)()
    actual = getattr(ddf_grouped, aggregation)()

    dd.assert_eq(expect, actual, check_dtype=check_dtype)

    if not series:
        expect = gdf_grouped.agg({"x": aggregation})
        actual = ddf_grouped.agg({"x": aggregation})

        dd.assert_eq(expect, actual, check_dtype=check_dtype)


# TODO: explore adding support with `.agg()`
@pytest.mark.parametrize("series", [True, False])
@pytest.mark.parametrize("aggregation", ["cumsum", "cumcount"])
def test_groupby_cumulative(aggregation, pdf, series):
    gdf = cudf.DataFrame(pdf)
    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    gdf_grouped = gdf.groupby("xx")
    ddf_grouped = ddf.groupby("xx")

    if series:
        gdf_grouped = gdf_grouped.xx
        ddf_grouped = ddf_grouped.xx

    a = getattr(gdf_grouped, aggregation)()
    b = getattr(ddf_grouped, aggregation)()

    dd.assert_eq(a, b)


@pytest.mark.parametrize("aggregation", OPTIMIZED_AGGS)
@pytest.mark.parametrize(
    "func",
    [
        lambda df, agg: df.groupby("xx").agg({"y": agg}),
        lambda df, agg: df.groupby("xx").y.agg({"y": agg}),
        lambda df, agg: df.groupby("xx").agg([agg]),
        lambda df, agg: df.groupby("xx").y.agg([agg]),
        lambda df, agg: df.groupby("xx").agg(agg),
        lambda df, agg: df.groupby("xx").y.agg(agg),
    ],
)
def test_groupby_agg(func, aggregation, pdf):
    gdf = cudf.DataFrame(pdf)
    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    actual = func(ddf, aggregation)
    expect = func(gdf, aggregation)

    check_dtype = aggregation != "count"

    dd.assert_eq(expect, actual, check_names=False, check_dtype=check_dtype)


@pytest.mark.parametrize("split_out", [1, 3])
def test_groupby_agg_empty_partition(tmpdir, split_out):
    # Write random and empty cudf DataFrames
    # to two distinct files.
    df = cudf.datasets.randomdata()
    df.to_parquet(str(tmpdir.join("f0.parquet")))
    cudf.DataFrame(
        columns=["id", "x", "y"],
        dtype={"id": "int64", "x": "float64", "y": "float64"},
    ).to_parquet(str(tmpdir.join("f1.parquet")))

    # Read back our two partitions as a single
    # dask_cudf DataFrame (one partition is now empty)
    ddf = dask_cudf.read_parquet(str(tmpdir))
    gb = ddf.groupby(["id"]).agg({"x": ["sum"]}, split_out=split_out)

    expect = df.groupby(["id"]).agg({"x": ["sum"]}).sort_index()
    dd.assert_eq(gb.compute().sort_index(), expect)


# reason gotattr in cudf
@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.groupby(["a", "b"]).x.sum(),
        lambda df: df.groupby(["a", "b"]).sum(),
        pytest.param(
            lambda df: df.groupby(["a", "b"]).agg({"x", "sum"}),
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_groupby_multi_column(func):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "a": rng.integers(0, 20, size=1000),
            "b": rng.integers(0, 5, size=1000),
            "x": rng.normal(size=1000),
        }
    )

    gdf = cudf.DataFrame(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = func(gdf).to_pandas()
    b = func(ddf).compute().to_pandas()

    dd.assert_eq(a, b)


def test_reset_index_multiindex():
    with dask.config.set({"dataframe.convert-string": False}):
        df = cudf.DataFrame()
        df["id_1"] = ["a", "a", "b"]
        df["id_2"] = [0, 0, 1]
        df["val"] = [1, 2, 3]

        df_lookup = cudf.DataFrame()
        df_lookup["id_1"] = ["a", "b"]
        df_lookup["metadata"] = [0, 1]

        gddf = dask_cudf.from_cudf(df, npartitions=2)
        gddf_lookup = dask_cudf.from_cudf(df_lookup, npartitions=2)

        ddf = dd.from_pandas(df.to_pandas(), npartitions=2)
        ddf_lookup = dd.from_pandas(df_lookup.to_pandas(), npartitions=2)

        # Note: 'id_2' has wrong type (object) until after compute
        dd.assert_eq(
            gddf.groupby(by=["id_1", "id_2"])
            .val.sum()
            .reset_index()
            .merge(gddf_lookup, on="id_1")
            .compute(),
            ddf.groupby(by=["id_1", "id_2"])
            .val.sum()
            .reset_index()
            .merge(ddf_lookup, on="id_1"),
        )


@pytest.mark.parametrize("split_out", [1, 2, 3])
@pytest.mark.parametrize(
    "column", ["c", "d", "e", ["b", "c"], ["b", "d"], ["b", "e"]]
)
def test_groupby_split_out(split_out, column):
    df = pd.DataFrame(
        {
            "a": np.arange(8),
            "b": [1, 0, 0, 2, 1, 1, 2, 0],
            "c": [0, 1] * 4,
            "d": ["dog", "cat", "cat", "dog", "dog", "dog", "cat", "bird"],
        }
    ).fillna(0)
    df["e"] = df["d"].astype("category")

    gdf = cudf.from_pandas(df)

    ddf = dd.from_pandas(df, npartitions=3)
    gddf = dask_cudf.from_cudf(gdf, npartitions=3)

    ddf_result = (
        ddf.groupby(column, observed=True)
        .a.mean(split_out=split_out)
        .compute()
        .sort_values()
        .dropna()
    )
    gddf_result = (
        gddf.groupby(column)
        .a.mean(split_out=split_out)
        .compute()
        .sort_values()
    )

    dd.assert_eq(gddf_result, ddf_result, check_index=False)


@pytest.mark.parametrize("dropna", [False, True, None])
@pytest.mark.parametrize(
    "by", ["a", "b", "c", "d", ["a", "b"], ["a", "c"], ["a", "d"]]
)
# invalid value encountered in cast
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_groupby_dropna_cudf(dropna, by):
    # NOTE: This test is borrowed from upstream dask
    #       (dask/dask/dataframe/tests/test_groupby.py)
    df = cudf.DataFrame(
        {
            "a": [1, 2, 3, 4, None, None, 7, 8],
            "b": [1, None, 1, 3, None, 3, 1, 3],
            "c": ["a", "b", None, None, "e", "f", "g", "h"],
            "e": [4, 5, 6, 3, 2, 1, 0, 0],
        }
    )
    df["b"] = df["b"].astype("datetime64[ns]")
    df["d"] = df["c"].astype("category")
    ddf = dask_cudf.from_cudf(df, npartitions=3)

    if dropna is None:
        dask_result = ddf.groupby(by).e.sum()
        cudf_result = df.groupby(by).e.sum()
    else:
        dask_result = ddf.groupby(by, dropna=dropna).e.sum()
        cudf_result = df.groupby(by, dropna=dropna).e.sum()
    if by in ["c", "d"]:
        # Loose string/category index name in cudf...
        dask_result = dask_result.compute()
        dask_result.index.name = cudf_result.index.name

    dd.assert_eq(dask_result, cudf_result)


@pytest.mark.parametrize(
    "dropna,by",
    [
        (False, "a"),
        (False, "b"),
        (False, "c"),
        (False, "d"),
        (False, ["a", "b"]),
        (False, ["a", "c"]),
        (False, ["a", "d"]),
        (True, "a"),
        (True, "b"),
        (True, "c"),
        (True, "d"),
        (True, ["a", "b"]),
        (True, ["a", "c"]),
        (True, ["a", "d"]),
        (None, "a"),
        (None, "b"),
        (None, "c"),
        (None, "d"),
        (None, ["a", "b"]),
        (None, ["a", "c"]),
        (None, ["a", "d"]),
    ],
)
@pytest.mark.skipif(
    PANDAS_VERSION < PANDAS_CURRENT_SUPPORTED_VERSION,
    reason="Fails in older versions of pandas",
)
def test_groupby_dropna_dask(dropna, by):
    # NOTE: This test is borrowed from upstream dask
    #       (dask/dask/dataframe/tests/test_groupby.py)
    df = pd.DataFrame(
        {
            "a": [1, 2, 3, 4, None, None, 7, 8],
            "b": [1, None, 1, 3, None, 3, 1, 3],
            "c": ["a", "b", None, None, "e", "f", "g", "h"],
            "e": [4, 5, 6, 3, 2, 1, 0, 0],
        }
    )
    df["b"] = df["b"].astype("datetime64[ns]")
    df["d"] = df["c"].astype("category")

    gdf = cudf.from_pandas(df)
    ddf = dd.from_pandas(df, npartitions=3)
    gddf = dask_cudf.from_cudf(gdf, npartitions=3)

    if dropna is None:
        dask_cudf_result = gddf.groupby(by).e.sum()
        dask_result = ddf.groupby(by, observed=True).e.sum()
    else:
        dask_cudf_result = gddf.groupby(by, dropna=dropna).e.sum()
        dask_result = ddf.groupby(by, dropna=dropna, observed=True).e.sum()

    dd.assert_eq(dask_cudf_result, dask_result)


@pytest.mark.parametrize("myindex", [[1, 2] * 4, ["s1", "s2"] * 4])
def test_groupby_string_index_name(myindex):
    # GH-Issue #3420
    data = {"index": myindex, "data": [0, 1] * 4}
    df = cudf.DataFrame(data=data)
    ddf = dask_cudf.from_cudf(df, npartitions=2)
    gdf = ddf.groupby("index").agg({"data": "count"})

    assert gdf.compute().index.name == gdf.index.name


@pytest.mark.parametrize(
    "agg_func",
    [
        lambda gb: gb.agg({"c": ["count"]}, split_out=2),
        lambda gb: gb.agg({"c": "count"}, split_out=2),
        lambda gb: gb.agg({"c": ["count", "sum"]}, split_out=2),
        lambda gb: gb.count(split_out=2),
        lambda gb: gb.c.count(split_out=2),
    ],
)
def test_groupby_split_out_multiindex(agg_func):
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(
        {
            "a": rng.integers(0, 10, 100),
            "b": rng.integers(0, 5, 100),
            "c": rng.random(100),
        }
    )
    ddf = dask_cudf.from_cudf(df, 5)
    pddf = dd.from_pandas(df.to_pandas(), 5)
    gr = agg_func(ddf.groupby(["a", "b"]))
    pr = agg_func(pddf.groupby(["a", "b"]))
    dd.assert_eq(gr.compute(), pr.compute())


@pytest.mark.parametrize("npartitions", [1, 2])
def test_groupby_multiindex_reset_index(npartitions):
    df = cudf.DataFrame(
        {"a": [1, 1, 2, 3, 4], "b": [5, 2, 1, 2, 5], "c": [1, 2, 2, 3, 5]}
    )
    ddf = dask_cudf.from_cudf(df, npartitions=npartitions)
    pddf = dd.from_pandas(df.to_pandas(), npartitions=npartitions)
    gr = ddf.groupby(["a", "c"]).agg({"b": ["count"]}).reset_index()
    pr = pddf.groupby(["a", "c"]).agg({"b": ["count"]}).reset_index()

    # CuDF uses "int32" for count. Pandas uses "int64"
    gr_out = gr.compute().sort_values(by=["a", "c"]).reset_index(drop=True)
    gr_out[("b", "count")] = gr_out[("b", "count")].astype("int64")

    dd.assert_eq(
        gr_out,
        pr.compute().sort_values(by=["a", "c"]).reset_index(drop=True),
    )


@pytest.mark.parametrize(
    "groupby_keys", [["a"], ["a", "b"], ["a", "b", "dd"], ["a", "dd", "b"]]
)
@pytest.mark.parametrize(
    "agg_func",
    [
        lambda gb: gb.agg({"c": ["count"]}),
        lambda gb: gb.agg({"c": "count"}),
        lambda gb: gb.agg({"c": ["count", "sum"]}),
        lambda gb: gb.count(),
        lambda gb: gb.c.count(),
    ],
)
def test_groupby_reset_index_multiindex(groupby_keys, agg_func):
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(
        {
            "a": rng.integers(0, 10, 10),
            "b": rng.integers(0, 5, 10),
            "c": rng.integers(0, 5, 10),
            "dd": rng.integers(0, 5, 10),
        }
    )
    ddf = dask_cudf.from_cudf(df, 5)
    pddf = dd.from_pandas(df.to_pandas(), 5)
    gr = agg_func(ddf.groupby(groupby_keys)).reset_index()
    pr = agg_func(pddf.groupby(groupby_keys)).reset_index()
    gf = gr.compute().sort_values(groupby_keys).reset_index(drop=True)
    pf = pr.compute().sort_values(groupby_keys).reset_index(drop=True)
    dd.assert_eq(gf, pf)


def test_groupby_reset_index_drop_True():
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(
        {"a": rng.integers(0, 10, 10), "b": rng.integers(0, 5, 10)}
    )
    ddf = dask_cudf.from_cudf(df, 5)
    pddf = dd.from_pandas(df.to_pandas(), 5)
    gr = ddf.groupby(["a"]).agg({"b": ["count"]}).reset_index(drop=True)
    pr = pddf.groupby(["a"]).agg({"b": ["count"]}).reset_index(drop=True)
    gf = gr.compute().sort_values(by=["b"]).reset_index(drop=True)
    pf = pr.compute().sort_values(by=[("b", "count")]).reset_index(drop=True)
    dd.assert_eq(gf, pf)


def test_groupby_mean_sort_false():
    df = cudf.datasets.randomdata(nrows=150, dtypes={"a": int, "b": int})
    ddf = dask_cudf.from_cudf(df, 1)
    pddf = dd.from_pandas(df.to_pandas(), 1)

    gr = ddf.groupby(["a"]).agg({"b": "mean"})
    pr = pddf.groupby(["a"]).agg({"b": "mean"})
    assert pr.index.name == gr.index.name
    assert pr.head(0).index.name == gr.head(0).index.name

    gf = gr.compute().sort_values(by=["b"]).reset_index(drop=True)
    pf = pr.compute().sort_values(by=["b"]).reset_index(drop=True)
    dd.assert_eq(gf, pf)


def test_groupby_reset_index_dtype():
    # Make sure int8 dtype is properly preserved
    # Through various cudf/dask_cudf ops
    #
    # Note: GitHub Issue#4090 reproducer

    df = cudf.DataFrame()
    df["a"] = np.arange(10, dtype="int8")
    df["b"] = np.arange(10, dtype="int8")
    df = dask_cudf.from_cudf(df, 1)

    a = df.groupby("a").agg({"b": ["count"]})

    assert a.index.dtype == "int8"
    assert a.reset_index().dtypes.iloc[0] == "int8"


def test_groupby_reset_index_names():
    df = cudf.datasets.randomdata(
        nrows=10, dtypes={"a": str, "b": int, "c": int}
    )
    pdf = df.to_pandas()

    gddf = dask_cudf.from_cudf(df, 2)
    pddf = dd.from_pandas(pdf, 2)

    g_res = gddf.groupby("a", sort=True).sum()
    p_res = pddf.groupby("a", sort=True).sum()

    got = g_res.reset_index().compute().sort_values(["a", "b", "c"])
    expect = p_res.reset_index().compute().sort_values(["a", "b", "c"])

    dd.assert_eq(got, expect)


def test_groupby_reset_index_string_name():
    df = cudf.DataFrame({"value": range(5), "key": ["a", "a", "b", "a", "c"]})
    pdf = df.to_pandas()

    gddf = dask_cudf.from_cudf(df, npartitions=1)
    pddf = dd.from_pandas(pdf, npartitions=1)

    g_res = (
        gddf.groupby(["key"]).agg({"value": "mean"}).reset_index(drop=False)
    )
    p_res = (
        pddf.groupby(["key"]).agg({"value": "mean"}).reset_index(drop=False)
    )

    got = g_res.compute().sort_values(["key", "value"]).reset_index(drop=True)
    expect = (
        p_res.compute().sort_values(["key", "value"]).reset_index(drop=True)
    )

    dd.assert_eq(got, expect)
    assert len(g_res) == len(p_res)


def test_groupby_categorical_key():
    # See https://github.com/rapidsai/cudf/issues/4608
    df = dask.datasets.timeseries(seed=1)
    gddf = df.to_backend("cudf")
    gddf["name"] = gddf["name"].astype("category")
    ddf = gddf.to_backend("pandas")

    got = gddf.groupby("name", sort=True).agg(
        {"x": ["mean", "max"], "y": ["mean", "count"]}
    )
    # Use `compute` to avoid upstream issue for now
    # (See: https://github.com/dask/dask/issues/9515)
    expect = (
        ddf.compute()
        .groupby("name", sort=True, observed=True)
        .agg({"x": ["mean", "max"], "y": ["mean", "count"]})
    )
    dd.assert_eq(expect, got, atol=1e-3)


@pytest.mark.parametrize(
    "as_index",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="as_index not supported in dask-expr"
            ),
        ),
    ],
)
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("split_out", ["use_dask_default", 1, 2])
@pytest.mark.parametrize("split_every", [False, 4])
@pytest.mark.parametrize("npartitions", [1, 10])
def test_groupby_agg_params(
    npartitions, split_every, split_out, fused, as_index
):
    df = cudf.datasets.randomdata(
        nrows=150,
        dtypes={"name": str, "a": int, "b": int, "c": float},
    )
    df["a"] = [0, 1, 2] * 50
    ddf = dask_cudf.from_cudf(df, npartitions)
    pddf = dd.from_pandas(df.to_pandas(), npartitions)

    agg_dict = {
        "a": "sum",
        "b": ["min", "max", "mean"],
        "c": ["mean", "std", "var"],
    }

    fused_kwarg = {"fused": fused}
    split_kwargs = {"split_every": split_every, "split_out": split_out}
    if split_out == "use_dask_default":
        split_kwargs.pop("split_out")

    # Avoid using as_index when query-planning is enabled
    with pytest.warns(FutureWarning, match="argument is now deprecated"):
        # Should warn when `as_index` is used
        ddf.groupby(["name", "a"], sort=False, as_index=as_index)
    maybe_as_index = {"as_index": as_index} if as_index is False else {}

    # Check `sort=True` behavior
    if split_out == 1:
        gf = (
            ddf.groupby(["name", "a"], sort=True, **maybe_as_index)
            .aggregate(
                agg_dict,
                **fused_kwarg,
                **split_kwargs,
            )
            .compute()
        )
        if as_index:
            # Groupby columns became the index.
            # Sorting the index should not change anything.
            dd.assert_eq(gf.index.to_frame(), gf.sort_index().index.to_frame())
        else:
            # Groupby columns are did NOT become the index.
            # Sorting by these columns should not change anything.
            sort_cols = [("name", ""), ("a", "")]
            dd.assert_eq(
                gf[sort_cols],
                gf[sort_cols].sort_values(sort_cols),
                check_index=False,
            )

    # Full check (`sort=False`)
    gr = ddf.groupby(["name", "a"], sort=False, **maybe_as_index).aggregate(
        agg_dict,
        **fused_kwarg,
        **split_kwargs,
    )
    pr = pddf.groupby(["name", "a"], sort=False).agg(
        agg_dict,
        **split_kwargs,
    )

    # Test `as_index` argument
    if as_index:
        # Groupby columns should NOT be in columns
        assert ("name", "") not in gr.columns and ("a", "") not in gr.columns
    else:
        # Groupby columns SHOULD be in columns
        assert ("name", "") in gr.columns and ("a", "") in gr.columns

    # Check `split_out` argument
    assert gr.npartitions == (
        1 if split_out == "use_dask_default" else split_out
    )

    # Compute for easier multiindex handling
    gf = gr.compute()
    pf = pr.compute()

    # Reset index and sort by groupby columns
    if as_index:
        gf = gf.reset_index(drop=False)
    sort_cols = [("name", ""), ("a", ""), ("c", "mean")]
    gf = gf.sort_values(sort_cols).reset_index(drop=True)
    pf = (
        pf.reset_index(drop=False)
        .sort_values(sort_cols)
        .reset_index(drop=True)
    )

    dd.assert_eq(gf, pf)


@pytest.mark.parametrize(
    "aggregations", [(sum, "sum"), (max, "max"), (min, "min")]
)
def test_groupby_agg_redirect(aggregations):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "x": rng.integers(0, 5, size=10000),
            "y": rng.normal(size=10000),
        }
    )

    gdf = cudf.DataFrame(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    a = ddf.groupby("x").agg({"x": aggregations[0]}).compute()
    b = ddf.groupby("x").agg({"x": aggregations[1]}).compute()

    dd.assert_eq(a, b)


@pytest.mark.parametrize(
    "arg,supported",
    [
        ("sum", True),
        (["sum"], True),
        ({"a": "sum"}, True),
        ({"a": ["sum"]}, True),
        ("not_supported", False),
        (["not_supported"], False),
        ({"a": "not_supported"}, False),
        ({"a": ["not_supported"]}, False),
    ],
)
def test_is_supported(arg, supported):
    assert _aggs_optimized(arg, OPTIMIZED_AGGS) is supported


def test_groupby_unique_lists():
    df = pd.DataFrame({"a": [0, 0, 0, 1, 1, 1], "b": [10, 10, 10, 7, 8, 9]})
    gdf = cudf.from_pandas(df)
    gddf = dask_cudf.from_cudf(gdf, 2)

    dd.assert_eq(
        gdf.groupby("a").b.unique(),
        gddf.groupby("a").b.unique().compute(),
    )


@pytest.mark.parametrize(
    "data",
    [
        {"a": [], "b": []},
        {"a": [2, 1, 2, 1, 1, 3], "b": [None, 1, 2, None, 2, None]},
        {"a": [None], "b": [None]},
        {"a": [2, 1, 1], "b": [None, 1, 0], "c": [None, 0, 1]},
    ],
)
@pytest.mark.parametrize("agg", ["first", "last"])
def test_groupby_first_last(data, agg):
    with dask.config.set({"dataframe.convert-string": False}):
        pdf = pd.DataFrame(data)
        gdf = cudf.DataFrame(pdf)

        ddf = dd.from_pandas(pdf, npartitions=2)
        gddf = dask_cudf.from_cudf(gdf, npartitions=2)

        dd.assert_eq(
            ddf.groupby("a").agg(agg),
            gddf.groupby("a").agg(agg),
        )

        dd.assert_eq(
            getattr(ddf.groupby("a"), agg)(),
            getattr(gddf.groupby("a"), agg)(),
        )

        dd.assert_eq(gdf.groupby("a").agg(agg), gddf.groupby("a").agg(agg))

        dd.assert_eq(
            getattr(gdf.groupby("a"), agg)(),
            getattr(gddf.groupby("a"), agg)(),
        )


@pytest.mark.xfail(reason="Co-alignment check fails in dask-expr")
def test_groupby_with_list_of_series():
    df = cudf.DataFrame({"a": [1, 2, 3, 4, 5]})
    gdf = dask_cudf.from_cudf(df, npartitions=2)
    gs = cudf.Series([1, 1, 1, 2, 2], name="id")
    ggs = dask_cudf.from_cudf(gs, npartitions=2)

    ddf = dd.from_pandas(df.to_pandas(), npartitions=2)
    pgs = dd.from_pandas(gs.to_pandas(), npartitions=2)

    dd.assert_eq(
        gdf.groupby([ggs]).agg(["sum"]), ddf.groupby([pgs]).agg(["sum"])
    )


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.groupby("x").agg({"y": {"foo": "sum"}}),
        lambda df: df.groupby("x").agg({"y": {"foo": "sum", "bar": "count"}}),
    ],
)
def test_groupby_nested_dict(func):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "x": rng.integers(0, 5, size=10000),
            "y": rng.normal(size=10000),
        }
    )

    ddf = dd.from_pandas(pdf, npartitions=5)
    c_ddf = ddf.map_partitions(cudf.from_pandas)

    a = func(ddf).compute()
    b = func(c_ddf).compute().to_pandas()

    a.index.name = None
    a.name = None
    b.index.name = None
    b.name = None

    dd.assert_eq(a, b)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df.groupby(["x", "y"]).min(),
        pytest.param(
            lambda df: df.groupby(["x", "y"]).agg("min"),
            marks=pytest.mark.skip(
                reason="https://github.com/dask/dask/issues/9093"
            ),
        ),
        lambda df: df.groupby(["x", "y"]).y.min(),
        lambda df: df.groupby(["x", "y"]).y.agg("min"),
    ],
)
def test_groupby_all_columns(func):
    rng = np.random.default_rng(seed=0)
    pdf = pd.DataFrame(
        {
            "x": rng.integers(0, 5, size=10000),
            "y": rng.normal(size=10000),
        }
    )

    ddf = dd.from_pandas(pdf, npartitions=5)
    gddf = ddf.to_backend("cudf")

    expect = func(ddf)
    actual = func(gddf)

    dd.assert_eq(expect, actual, check_names=False)


def test_groupby_shuffle():
    df = cudf.datasets.randomdata(
        nrows=640, dtypes={"a": str, "b": int, "c": int}
    )
    gddf = dask_cudf.from_cudf(df, 8)
    spec = {"b": "mean", "c": "max"}
    expect = df.groupby("a", sort=True).agg(spec)

    # Sorted aggregation, single-partition output
    # (sort=True, split_out=1)
    got = gddf.groupby("a", sort=True).agg(
        spec, shuffle_method=True, split_out=1
    )
    dd.assert_eq(expect, got)

    # Sorted aggregation, multi-partition output
    # (sort=True, split_out=2)
    got = gddf.groupby("a", sort=True).agg(
        spec, shuffle_method=True, split_out=2
    )
    dd.assert_eq(expect, got)

    # Un-sorted aggregation, single-partition output
    # (sort=False, split_out=1)
    got = gddf.groupby("a", sort=False).agg(
        spec, shuffle_method=True, split_out=1
    )
    dd.assert_eq(expect.sort_index(), got.compute().sort_index())

    # Un-sorted aggregation, multi-partition output
    # (sort=False, split_out=2)
    # NOTE: `shuffle_method=True` should be default
    got = gddf.groupby("a", sort=False).agg(spec, split_out=2)
    dd.assert_eq(expect, got.compute().sort_index())

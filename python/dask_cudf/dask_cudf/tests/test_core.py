# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import dask
from dask import dataframe as dd
from dask.dataframe.dispatch import make_meta as dask_make_meta, meta_nonempty
from dask.utils import M

import cudf

import dask_cudf

rng = np.random.default_rng(seed=0)


def test_from_dict_backend_dispatch():
    # Test ddf.from_dict cudf-backend dispatch
    rng = np.random.default_rng(seed=0)
    data = {
        "x": rng.integers(0, 5, size=10000),
        "y": rng.normal(size=10000),
    }
    expect = cudf.DataFrame(data)
    with dask.config.set({"dataframe.backend": "cudf"}):
        ddf = dd.from_dict(data, npartitions=2)
    assert isinstance(ddf, dask_cudf.DataFrame)
    dd.assert_eq(expect, ddf)


def test_to_backend():
    rng = np.random.default_rng(seed=0)
    data = {
        "x": rng.integers(0, 5, size=10000),
        "y": rng.normal(size=10000),
    }
    with dask.config.set({"dataframe.backend": "pandas"}):
        ddf = dd.from_dict(data, npartitions=2)
        assert isinstance(ddf._meta, pd.DataFrame)

        gdf = ddf.to_backend("cudf")
        assert isinstance(gdf, dask_cudf.DataFrame)
        dd.assert_eq(cudf.DataFrame(data), ddf)

        assert isinstance(gdf.to_backend()._meta, pd.DataFrame)


def test_to_backend_kwargs():
    data = {"x": [0, 2, np.nan, 3, 4, 5]}
    with dask.config.set({"dataframe.backend": "pandas"}):
        dser = dd.from_dict(data, npartitions=2)["x"]
        assert isinstance(dser._meta, pd.Series)

        # Using `nan_as_null=False` will result in a cudf-backed
        # Series with a NaN element (ranther than <NA>)
        gser_nan = dser.to_backend("cudf", nan_as_null=False)
        assert isinstance(gser_nan, dask_cudf.Series)
        assert np.isnan(gser_nan.compute()).sum() == 1

        # Using `nan_as_null=True` will result in a cudf-backed
        # Series with a <NA> element (ranther than NaN)
        gser_null = dser.to_backend("cudf", nan_as_null=True)
        assert isinstance(gser_null, dask_cudf.Series)
        assert np.isnan(gser_null.compute()).sum() == 0

        # Check `nullable` argument for `cudf.Series.to_pandas`
        dser_null = gser_null.to_backend("pandas", nullable=False)
        assert dser_null.compute().dtype == "float"
        dser_null = gser_null.to_backend("pandas", nullable=True)
        assert isinstance(dser_null.compute().dtype, pd.Float64Dtype)

        # Check unsupported arguments
        with pytest.raises(ValueError, match="pandas-to-cudf"):
            dser.to_backend("cudf", bad_arg=True)

        with pytest.raises(ValueError, match="cudf-to-cudf"):
            gser_null.to_backend("cudf", bad_arg=True)

        with pytest.raises(ValueError, match="cudf-to-pandas"):
            gser_null.to_backend("pandas", bad_arg=True)


def test_from_pandas():
    rng = np.random.default_rng(seed=0)

    df = pd.DataFrame(
        {
            "x": rng.integers(0, 5, size=10000),
            "y": rng.normal(size=10000),
        }
    )

    gdf = cudf.DataFrame(df)

    # Test simple around to/from cudf
    ingested = dd.from_pandas(gdf, npartitions=2)
    dd.assert_eq(ingested, df)

    # Test conversion back to pandas
    ddf = ingested.to_backend("pandas")
    dd.assert_eq(ddf, df)


def test_from_pandas_multiindex_raises():
    df = cudf.DataFrame({"x": list("abc"), "y": [1, 2, 3], "z": [1, 2, 3]})

    with pytest.raises(NotImplementedError):
        # dask_cudf does not support MultiIndex yet
        dask_cudf.from_cudf(df.set_index(["x", "y"]))


def test_from_pandas_with_generic_idx():
    cdf = cudf.DataFrame(
        {
            "a": list(range(20)),
            "b": list(reversed(range(20))),
            "c": list(range(20)),
        }
    )

    ddf = dask_cudf.from_cudf(cdf, npartitions=2)

    assert isinstance(ddf.index.compute(), cudf.RangeIndex)
    dd.assert_eq(ddf.loc[1:2, ["a"]], cdf.loc[1:2, ["a"]])


def _fragmented_gdf(df, nsplit):
    n = len(df)

    # Split dataframe in *nsplit*
    subdivsize = n // nsplit
    starts = [i * subdivsize for i in range(nsplit)]
    ends = [*starts[1:], None]
    frags = [df[s:e] for s, e in zip(starts, ends, strict=True)]
    return frags


def test_query():
    rng = np.random.default_rng(seed=0)

    df = pd.DataFrame(
        {"x": rng.integers(0, 5, size=10), "y": rng.normal(size=10)}
    )
    gdf = cudf.DataFrame(df)
    expr = "x > 2"

    dd.assert_eq(gdf.query(expr), df.query(expr))

    queried = dd.from_pandas(gdf, npartitions=2).query(expr)

    got = queried
    expect = gdf.query(expr)

    dd.assert_eq(got, expect)


def test_query_local_dict():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {"x": rng.integers(0, 5, size=10), "y": rng.normal(size=10)}
    )
    gdf = cudf.DataFrame(df)
    ddf = dask_cudf.from_cudf(gdf, npartitions=2)

    val = 2

    gdf_queried = gdf.query("x > @val")
    ddf_queried = ddf.query("x > @val", local_dict={"val": val})

    dd.assert_eq(gdf_queried, ddf_queried)


def test_head():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {
            "x": rng.integers(0, 5, size=100),
            "y": rng.normal(size=100),
        }
    )
    gdf = cudf.DataFrame(df)
    dgf = dd.from_pandas(gdf, npartitions=2)

    dd.assert_eq(dgf.head(), df.head())


@pytest.mark.parametrize("nelem", [10, 200, 1333])
def test_set_index(nelem):
    with dask.config.set(scheduler="single-threaded"):
        rng = np.random.default_rng(seed=0)
        # Use unique index range as the sort may not be stable-ordering
        x = np.arange(nelem)
        rng.shuffle(x)
        df = pd.DataFrame({"x": x, "y": rng.integers(0, nelem, size=nelem)})
        ddf = dd.from_pandas(df, npartitions=2)
        ddf2 = ddf.to_backend("cudf")

        expect = ddf.set_index("x")
        got = ddf2.set_index("x")

        dd.assert_eq(expect, got, check_index=False, check_divisions=False)


@pytest.mark.parametrize("by", ["a", "b"])
@pytest.mark.parametrize("nelem", [10, 500])
@pytest.mark.parametrize("nparts", [1, 10])
def test_set_index_quantile(nelem, nparts, by):
    df = cudf.DataFrame()
    df["a"] = np.ascontiguousarray(np.arange(nelem)[::-1])
    df["b"] = rng.choice(cudf.datasets.names, size=nelem)
    ddf = dd.from_pandas(df, npartitions=nparts)

    with pytest.warns(FutureWarning, match="deprecated"):
        got = ddf.set_index(by, divisions="quantile")
    expect = df.sort_values(by=by).set_index(by)
    dd.assert_eq(got, expect)


def assert_frame_equal_by_index_group(expect, got):
    assert sorted(expect.columns) == sorted(got.columns)
    assert sorted(set(got.index)) == sorted(set(expect.index))
    # Note the set_index sort is not stable,
    unique_values = sorted(set(got.index))
    for iv in unique_values:
        sr_expect = expect.loc[[iv]]
        sr_got = got.loc[[iv]]

        for k in expect.columns:
            # Sort each column before we compare them
            sorted_expect = sr_expect.sort_values(k)[k]
            sorted_got = sr_got.sort_values(k)[k]
            np.testing.assert_array_equal(sorted_expect, sorted_got)


@pytest.mark.parametrize("nelem", [10, 200, 1333])
def test_set_index_2(nelem):
    with dask.config.set(scheduler="single-threaded"):
        rng = np.random.default_rng(seed=0)
        df = pd.DataFrame(
            {
                "x": 100 + rng.integers(0, nelem // 2, size=nelem),
                "y": rng.normal(size=nelem),
            }
        )
        expect = df.set_index("x").sort_index()

        dgf = dd.from_pandas(cudf.DataFrame(df), npartitions=4)
        res = dgf.set_index("x")  # sort by default
        got = res.compute().to_pandas()

        assert_frame_equal_by_index_group(expect, got)


def test_set_index_w_series():
    with dask.config.set(scheduler="single-threaded"):
        nelem = 20
        rng = np.random.default_rng(seed=0)
        df = pd.DataFrame(
            {
                "x": 100 + rng.integers(0, nelem // 2, size=nelem),
                "y": rng.normal(size=nelem),
            }
        )
        expect = df.set_index(df.x).sort_index()

        dgf = dd.from_pandas(cudf.DataFrame(df), npartitions=4)
        res = dgf.set_index(dgf.x)  # sort by default
        got = res.compute().to_pandas()

        dd.assert_eq(expect, got)


def test_set_index_sorted():
    with dask.config.set(scheduler="single-threaded"):
        df1 = pd.DataFrame({"val": [4, 3, 2, 1, 0], "id": [0, 1, 3, 5, 7]})
        ddf1 = dd.from_pandas(df1, npartitions=2)

        gdf1 = cudf.from_pandas(df1)
        gddf1 = dask_cudf.from_cudf(gdf1, npartitions=2)

        expect = ddf1.set_index("id", sorted=True)
        got = gddf1.set_index("id", sorted=True)

        dd.assert_eq(expect, got)

        with pytest.raises(ValueError):
            # Cannot set `sorted=True` for non-sorted column
            gddf1.set_index("val", sorted=True)


def test_assign():
    rng = np.random.default_rng(seed=0)
    df = pd.DataFrame(
        {"x": rng.integers(0, 5, size=20), "y": rng.normal(size=20)}
    )

    dgf = dd.from_pandas(cudf.DataFrame(df), npartitions=2)
    pdcol = pd.Series(np.arange(20) + 1000)
    newcol = dd.from_pandas(cudf.Series(pdcol), npartitions=dgf.npartitions)
    got = dgf.assign(z=newcol)

    # Using `loc[:, ["x", "y"]]` was broken for dask-expr 0.4.0
    dd.assert_eq(got[["x", "y"]], df)
    np.testing.assert_array_equal(got["z"].compute().to_numpy(), pdcol.values)


@pytest.mark.parametrize("data_type", ["int8", "int16", "int32", "int64"])
def test_setitem_scalar_integer(data_type):
    rng = np.random.default_rng(seed=0)
    scalar = rng.integers(0, 100, dtype=data_type)
    df = pd.DataFrame(
        {"x": rng.integers(0, 5, size=20), "y": rng.normal(size=20)}
    )
    dgf = dd.from_pandas(cudf.DataFrame(df), npartitions=2)

    df["z"] = scalar
    dgf["z"] = scalar

    got = dgf.compute().to_pandas()
    np.testing.assert_array_equal(got["z"], df["z"])


@pytest.mark.parametrize("data_type", ["float32", "float64"])
def test_setitem_scalar_float(data_type):
    rng = np.random.default_rng(seed=0)
    scalar = rng.standard_normal(size=1).astype(data_type)[0]
    df = pd.DataFrame(
        {"x": rng.integers(0, 5, size=20), "y": rng.normal(size=20)}
    )
    dgf = dd.from_pandas(cudf.DataFrame(df), npartitions=2)

    df["z"] = scalar
    dgf["z"] = scalar

    got = dgf.compute().to_pandas()
    np.testing.assert_array_equal(got["z"], df["z"])


def test_setitem_scalar_datetime():
    rng = np.random.default_rng(seed=0)
    scalar = np.int64(rng.integers(0, 100)).astype("datetime64[ms]")
    df = pd.DataFrame(
        {"x": rng.integers(0, 5, size=20), "y": rng.normal(size=20)}
    )
    dgf = dd.from_pandas(cudf.DataFrame(df), npartitions=2)

    df["z"] = scalar
    dgf["z"] = scalar

    got = dgf.compute().to_pandas()
    np.testing.assert_array_equal(got["z"], df["z"])


@pytest.mark.skip(reason="datetime indexes not fully supported in cudf")
@pytest.mark.parametrize("start", ["1d", "5d", "1w", "12h"])
@pytest.mark.parametrize("stop", ["1d", "3d", "8h"])
def test_repartition_timeseries(start, stop):
    # This test is currently absurdly slow.  It should not be unskipped without
    # slimming it down.
    pdf = dask.datasets.timeseries(
        "2000-01-01",
        "2000-01-31",
        freq="1s",
        partition_freq=start,
        dtypes={"x": int, "y": float},
        seed=1,
    )
    gdf = pdf.map_partitions(cudf.DataFrame)

    a = pdf.repartition(freq=stop)
    b = gdf.repartition(freq=stop)
    assert a.divisions == b.divisions

    dd.utils.assert_eq(a, b)


@pytest.mark.parametrize("start", [1, 2, 5])
@pytest.mark.parametrize("stop", [1, 3, 7])
def test_repartition_simple_divisions(start, stop):
    pdf = pd.DataFrame({"x": range(100)})

    pdf = dd.from_pandas(pdf, npartitions=start)
    gdf = pdf.map_partitions(cudf.DataFrame)

    a = pdf.repartition(npartitions=stop)
    b = gdf.repartition(npartitions=stop)
    assert a.divisions == b.divisions

    dd.assert_eq(a, b)


@pytest.mark.parametrize("npartitions", [2, 17, 20])
def test_repartition_hash_staged(npartitions):
    by = ["b"]
    datarange = 35
    size = 100
    rng = np.random.default_rng(seed=0)
    gdf = cudf.DataFrame(
        {
            "a": np.arange(size, dtype="int64"),
            "b": rng.integers(datarange, size=size),
        }
    )
    # WARNING: Specific npartitions-max_branch combination
    # was specifically chosen to cover changes in #4676
    npartitions_initial = 17
    ddf = dask_cudf.from_cudf(gdf, npartitions=npartitions_initial)
    ddf_new = ddf.shuffle(
        on=by, ignore_index=True, npartitions=npartitions, max_branch=4
    )

    # Make sure we are getting a dask_cudf dataframe
    assert type(ddf_new) is type(ddf)

    # Check that the length was preserved
    assert len(ddf_new) == len(ddf)

    # Check that the partitions have unique keys,
    # and that the key values are preserved
    expect_unique = gdf[by].drop_duplicates().sort_values(by)
    got_unique = cudf.concat(
        [
            part[by].compute().drop_duplicates()
            for part in ddf_new[by].partitions
        ],
        ignore_index=True,
    ).sort_values(by)
    dd.assert_eq(got_unique, expect_unique, check_index=False)


@pytest.mark.parametrize("by", [["b"], ["c"], ["d"], ["b", "c"]])
@pytest.mark.parametrize("npartitions", [3, 4, 5])
@pytest.mark.parametrize("max_branch", [3, 32])
def test_repartition_hash(by, npartitions, max_branch):
    npartitions_i = 4
    datarange = 26
    size = 100
    rng = np.random.default_rng(seed=0)
    gdf = cudf.DataFrame(
        {
            "a": np.arange(0, stop=size, dtype="int64"),
            "b": rng.integers(datarange, size=size),
            "c": rng.choice(list("abcdefgh"), size=size),
            "d": rng.choice(np.arange(26), size=size),
        }
    )
    gdf.d = gdf.d.astype("datetime64[ms]")
    ddf = dask_cudf.from_cudf(gdf, npartitions=npartitions_i)
    ddf_new = ddf.shuffle(
        on=by,
        ignore_index=True,
        npartitions=npartitions,
        max_branch=max_branch,
    )

    # Check that the length was preserved
    assert len(ddf_new) == len(ddf)

    # Check that the partitions have unique keys,
    # and that the key values are preserved
    expect_unique = gdf[by].drop_duplicates().sort_values(by)
    got_unique = cudf.concat(
        [
            part[by].compute().drop_duplicates()
            for part in ddf_new[by].partitions
        ],
        ignore_index=True,
    ).sort_values(by)
    dd.assert_eq(got_unique, expect_unique, check_index=False)


def test_repartition_no_extra_row():
    # see https://github.com/rapidsai/cudf/issues/11930
    gdf = cudf.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]}).set_index("a")
    ddf = dask_cudf.from_cudf(gdf, npartitions=1)
    ddf_new = ddf.repartition([0, 5, 10, 30], force=True)
    dd.assert_eq(ddf, ddf_new)
    dd.assert_eq(gdf, ddf_new)


@pytest.fixture
def pdf():
    return pd.DataFrame(
        {"x": [1, 2, 3, 4, 5, 6], "y": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]}
    )


@pytest.fixture
def gdf(pdf):
    return cudf.from_pandas(pdf)


@pytest.fixture
def ddf(pdf):
    return dd.from_pandas(pdf, npartitions=3)


@pytest.fixture
def gddf(gdf):
    return dd.from_pandas(gdf, npartitions=3)


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df + 1,
        lambda df: df.index,
        lambda df: df.x.sum(),
        lambda df: df.x.astype(float),
        lambda df: df.assign(z=df.x.astype("int")),
    ],
)
def test_unary_ops(func, gdf, gddf):
    p = func(gdf)
    g = func(gddf)

    dd.assert_eq(p, g, check_names=False)


@pytest.mark.parametrize("series", [True, False])
def test_concat(gdf, gddf, series):
    if series:
        gdf = gdf.x
        gddf = gddf.x
        a = (
            cudf.concat([gdf, gdf + 1, gdf + 2])
            .sort_values()
            .reset_index(drop=True)
        )
        b = (
            dd.concat([gddf, gddf + 1, gddf + 2], interleave_partitions=True)
            .compute()
            .sort_values()
            .reset_index(drop=True)
        )
    else:
        a = (
            cudf.concat([gdf, gdf + 1, gdf + 2])
            .sort_values("x")
            .reset_index(drop=True)
        )
        b = (
            dd.concat([gddf, gddf + 1, gddf + 2], interleave_partitions=True)
            .compute()
            .sort_values("x")
            .reset_index(drop=True)
        )
    dd.assert_eq(a, b)


def test_boolean_index(gdf, gddf):
    gdf2 = gdf[gdf.x > 2]
    gddf2 = gddf[gddf.x > 2]

    dd.assert_eq(gdf2, gddf2)


def test_drop(gdf, gddf):
    gdf2 = gdf.drop(columns="x")
    gddf2 = gddf.drop(columns="x").compute()

    dd.assert_eq(gdf2, gddf2)


@pytest.mark.parametrize("index", [True, False])
def test_hash_object_dispatch(index):
    obj = cudf.DataFrame(
        {"x": ["a", "b", "c"], "y": [1, 2, 3], "z": [1, 1, 0]}, index=[2, 4, 6]
    )

    # DataFrame
    result = dd.dispatch.hash_object_dispatch(obj, index=index)
    expected = dask_cudf.backends.hash_object_cudf(obj, index=index)
    assert isinstance(result, cudf.Series)
    dd.assert_eq(result, expected)

    # Series
    result = dd.dispatch.hash_object_dispatch(obj["x"], index=index)
    expected = dask_cudf.backends.hash_object_cudf(obj["x"], index=index)
    assert isinstance(result, cudf.Series)
    dd.assert_eq(result, expected)

    # DataFrame with MultiIndex
    obj_multi = obj.set_index(["x", "z"], drop=True)
    result = dd.dispatch.hash_object_dispatch(obj_multi, index=index)
    expected = dask_cudf.backends.hash_object_cudf(obj_multi, index=index)
    assert isinstance(result, cudf.Series)
    dd.assert_eq(result, expected)


@pytest.mark.parametrize(
    "index",
    [
        "int8",
        "int32",
        "int64",
        "float64",
        "strings",
        "cats",
        "time_s",
        "time_ms",
        "time_ns",
        ["int32", "int64"],
        ["int8", "float64", "strings"],
        ["cats", "int8", "float64"],
        ["time_ms", "cats"],
    ],
)
def test_make_meta_backends(index):
    with dask.config.set({"dataframe.convert-string": False}):
        dtypes = ["int8", "int32", "int64", "float64"]
        df = cudf.DataFrame({dt: np.arange(0, 3, dtype=dt) for dt in dtypes})
        df["strings"] = ["cat", "dog", "fish"]
        df["cats"] = df["strings"].astype("category")
        df["time_s"] = np.array(
            ["2018-10-07", "2018-10-08", "2018-10-09"], dtype="datetime64[s]"
        )
        df["time_ms"] = df["time_s"].astype("datetime64[ms]")
        df["time_ns"] = df["time_s"].astype("datetime64[ns]")
        df = df.set_index(index)

        # Check "empty" metadata types
        chk_meta = dask_make_meta(df)
        dd.assert_eq(chk_meta.dtypes, df.dtypes)

        # Check "non-empty" metadata types
        chk_meta_nonempty = meta_nonempty(df)
        dd.assert_eq(chk_meta.dtypes, chk_meta_nonempty.dtypes)

        # Check dask code path if not MultiIndex
        if not isinstance(df.index, cudf.MultiIndex):
            ddf = dask_cudf.from_cudf(df, npartitions=1)

            # Check "empty" metadata types
            dd.assert_eq(ddf._meta.dtypes, df.dtypes)

            # Check "non-empty" metadata types
            dd.assert_eq(ddf._meta.dtypes, ddf._meta_nonempty.dtypes)


@pytest.mark.parametrize(
    "data",
    [
        pd.Series([], dtype="float64"),
        pd.DataFrame({"abc": [], "xyz": []}),
        pd.Series([1, 2, 10, 11]),
        pd.DataFrame({"abc": [1, 2, 10, 11], "xyz": [100, 12, 120, 1]}),
    ],
)
def test_dataframe_series_replace(data):
    pdf = data.copy()
    gdf = cudf.from_pandas(pdf)

    ddf = dask_cudf.from_cudf(gdf, npartitions=5)

    dd.assert_eq(ddf.replace(1, 2), pdf.replace(1, 2))


def test_dataframe_assign_col():
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame(list(range(100)))
    pdf = pd.DataFrame(list(range(100)))

    ddf = dask_cudf.from_cudf(df, npartitions=4)
    ddf["fold"] = 0
    ddf["fold"] = ddf["fold"].map_partitions(
        lambda cudf_df: cudf.Series(cp.random.randint(0, 4, len(cudf_df)))
    )

    pddf = dd.from_pandas(pdf, npartitions=4)
    pddf["fold"] = 0
    pddf["fold"] = pddf["fold"].map_partitions(
        lambda p_df: pd.Series(rng.integers(0, 4, len(p_df)))
    )

    dd.assert_eq(ddf[0], pddf[0])
    dd.assert_eq(len(ddf["fold"]), len(pddf["fold"]))


def test_dataframe_set_index():
    random.seed(0)
    df = cudf.datasets.randomdata(26, dtypes={"a": float, "b": int})
    df["str"] = list("abcdefghijklmnopqrstuvwxyz")
    pdf = df.to_pandas()

    with dask.config.set({"dataframe.convert-string": False}):
        ddf = dask_cudf.from_cudf(df, npartitions=4)
        ddf = ddf.set_index("str")

        pddf = dd.from_pandas(pdf, npartitions=4)
        pddf = pddf.set_index("str")

        from cudf.testing import assert_eq

        assert_eq(ddf.compute(), pddf.compute())


def test_series_describe():
    random.seed(0)
    sr = cudf.datasets.randomdata(20)["x"]
    psr = sr.to_pandas()

    dsr = dask_cudf.from_cudf(sr, npartitions=4)
    pdsr = dd.from_pandas(psr, npartitions=4)

    dd.assert_eq(
        dsr.describe(),
        pdsr.describe(),
        rtol=1e-3,
    )


def test_dataframe_describe():
    random.seed(0)
    df = cudf.datasets.randomdata(20)
    pdf = df.to_pandas()

    ddf = dask_cudf.from_cudf(df, npartitions=4)
    pddf = dd.from_pandas(pdf, npartitions=4)

    dd.assert_eq(
        ddf.describe(), pddf.describe(), check_exact=False, atol=0.0001
    )


def test_zero_std_describe():
    num = 84886781
    df = cudf.DataFrame(
        {
            "x": np.full((20,), num, dtype=np.float64),
            "y": np.full((20,), num, dtype=np.float64),
        }
    )
    pdf = df.to_pandas()
    ddf = dask_cudf.from_cudf(df, npartitions=4)
    pddf = dd.from_pandas(pdf, npartitions=4)

    dd.assert_eq(ddf.describe(), pddf.describe(), rtol=1e-3)


def test_large_numbers_var():
    num = 8488678001
    df = cudf.DataFrame(
        {
            "x": np.arange(num, num + 1000, dtype=np.float64),
            "y": np.arange(num, num + 1000, dtype=np.float64),
        }
    )
    pdf = df.to_pandas()
    ddf = dask_cudf.from_cudf(df, npartitions=4)
    pddf = dd.from_pandas(pdf, npartitions=4)

    dd.assert_eq(ddf.var(), pddf.var(), rtol=1e-3)


def test_index_map_partitions():
    # https://github.com/rapidsai/cudf/issues/6738

    ddf = dd.from_pandas(pd.DataFrame({"a": range(10)}), npartitions=2)
    mins_pd = ddf.index.map_partitions(M.min, meta=ddf.index).compute()

    gddf = dask_cudf.from_cudf(cudf.DataFrame({"a": range(10)}), npartitions=2)
    mins_gd = gddf.index.map_partitions(M.min, meta=gddf.index).compute()

    dd.assert_eq(mins_pd, mins_gd)


def test_merging_categorical_columns():
    with dask.config.set({"dataframe.convert-string": False}):
        df_1 = cudf.DataFrame(
            {"id_1": [0, 1, 2, 3], "cat_col": ["a", "b", "f", "f"]}
        )

        ddf_1 = dask_cudf.from_cudf(df_1, npartitions=2)

        ddf_1 = ddf_1.categorize(columns=["cat_col"])

        df_2 = cudf.DataFrame(
            {"id_2": [111, 112, 113], "cat_col": ["g", "h", "f"]}
        )

        ddf_2 = dask_cudf.from_cudf(df_2, npartitions=2)

        ddf_2 = ddf_2.categorize(columns=["cat_col"])

        # Merging on categorical keys with different category sets
        # decategorizes the key, matching pandas.
        expected = cudf.DataFrame(
            {
                "id_1": [2, 3],
                "cat_col": ["f", "f"],
                "id_2": [113, 113],
            }
        )
        with pytest.warns(UserWarning, match="mismatch"):
            dd.assert_eq(ddf_1.merge(ddf_2), expected)


def test_correct_meta():
    # Need these local imports in this specific order.
    # For context: https://github.com/rapidsai/cudf/issues/7946
    import pandas as pd

    from dask import dataframe as dd

    import dask_cudf  # noqa: F401

    df = pd.DataFrame({"a": [3, 4], "b": [1, 2]})
    ddf = dd.from_pandas(df, npartitions=1)
    emb = ddf["a"].apply(pd.Series, meta={"c0": "int64", "c1": "int64"})

    assert isinstance(emb, dd.DataFrame)
    assert isinstance(emb._meta, pd.DataFrame)


def test_categorical_dtype_round_trip():
    s = cudf.Series(4 * ["foo"], dtype="category")
    assert s.dtype.ordered is False

    ds = dask_cudf.from_cudf(s, npartitions=2)
    pds = dd.from_pandas(s.to_pandas(), npartitions=2)
    dd.assert_eq(ds, pds)
    assert ds.dtype.ordered is False

    # Below validations are required, see:
    # https://github.com/rapidsai/cudf/issues/11487#issuecomment-1208912383
    actual = ds.compute()
    expected = pds.compute()
    assert actual.dtype.ordered == expected.dtype.ordered


def test_implicit_array_conversion_cupy():
    s = cudf.Series(range(10))
    ds = dask_cudf.from_cudf(s, npartitions=2)

    def func(x):
        return x.values

    result = ds.map_partitions(func, meta=s.values)

    # Check Array and round-tripped DataFrame
    dask.array.assert_eq(result, func(s))
    dd.assert_eq(result.to_dask_dataframe(), s, check_index=False)


def test_array_conversion_cupy_chunks():
    pdf = pd.DataFrame({"x": range(10), "y": range(10)})
    df = dd.from_pandas(pdf, npartitions=2)
    arr = df.to_dask_array()
    garr = df.to_backend("cudf").to_dask_array()
    arr.compute_chunk_sizes()
    garr.compute_chunk_sizes()
    assert arr.chunks == garr.chunks


def test_implicit_array_conversion_cupy_sparse():
    cupyx = pytest.importorskip("cupyx")

    s = cudf.Series(range(10), dtype="float32")
    ds = dask_cudf.from_cudf(s, npartitions=2)

    def func(x):
        return cupyx.scipy.sparse.csr_matrix(x.values)

    result = ds.map_partitions(func, meta=s.values).compute()
    expect = func(s)

    # NOTE: The calculation here doesn't need to make sense.
    # We just need to make sure we get the right type back.
    assert type(result) is type(expect)


@pytest.mark.parametrize("data", [[1, 2, 3], [1.1, 2.3, 4.5]])
@pytest.mark.parametrize("values", [[1, 5], [1.1, 2.4, 2.3]])
def test_series_isin(data, values):
    ser = cudf.Series(data)
    pddf = dd.from_pandas(ser.to_pandas(), 1)
    ddf = dask_cudf.from_cudf(ser, 1)

    actual = ddf.isin(values)
    expected = pddf.isin(values)

    dd.assert_eq(actual, expected)


def test_series_isin_error():
    ser = cudf.Series([1, 2, 3])
    ddf = dask_cudf.from_cudf(ser, 1)
    with pytest.raises(TypeError):
        ser.isin([1, 5, "a"])
    with pytest.raises(TypeError):
        ddf.isin([1, 5, "a"]).compute()


def test_to_backend_simplify():
    # Check that column projection is not blocked by to_backend
    with dask.config.set({"dataframe.backend": "pandas"}):
        df = dd.from_dict({"x": [1, 2, 3], "y": [4, 5, 6]}, npartitions=2)
        df2 = df.to_backend("cudf")[["y"]].simplify()
        df3 = df[["y"]].to_backend("cudf").to_backend("cudf").simplify()
        assert df2._name == df3._name


@pytest.mark.parametrize("numeric_only", [True, False])
@pytest.mark.parametrize("op", ["corr", "cov"])
# the implementation might warn about ddof<=0 or divide by zero
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_cov_corr(op, numeric_only):
    rng = np.random.default_rng(seed=0)
    df = cudf.DataFrame.from_dict(
        {
            "x": rng.integers(0, 5, size=10),
            "y": rng.normal(size=10),
        }
    )
    ddf = dd.from_pandas(df, npartitions=2)
    res = getattr(ddf, op)(numeric_only=numeric_only)
    # Use to_pandas until cudf supports numeric_only
    # (See: https://github.com/rapidsai/cudf/issues/12626)
    expect = getattr(df.to_pandas(), op)(numeric_only=numeric_only)
    dd.assert_eq(res, expect)


def test_rename_axis_after_join():
    df1 = cudf.DataFrame(index=["a", "b", "c"], data=dict(a=[1, 2, 3]))
    df1.index.name = "test"
    ddf1 = dd.from_pandas(df1, 2)

    df2 = cudf.DataFrame(index=["a", "b", "d"], data=dict(b=[1, 2, 3]))
    ddf2 = dd.from_pandas(df2, 2)
    result = ddf1.join(ddf2, how="outer")
    expected = df1.join(df2, how="outer")
    dd.assert_eq(result, expected, check_index=False)


def test_clip_dataframe():
    df = cudf.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "score": [-1, 1, 4, 6],
        }
    )
    expect = df.clip(lower=["b", 1], upper=["d", 5], axis=1)
    got = dd.from_pandas(df, npartitions=2).clip(
        lower=["b", 1], upper=["d", 5], axis=1
    )
    dd.assert_eq(expect, got)


def test_clip_series():
    ser = cudf.Series([-0.5, 0.5, 4.5, 5.5])
    expect = ser.clip(lower=0, upper=5).round().astype(int)
    got = (
        dd.from_pandas(ser, npartitions=2)
        .clip(lower=0, upper=5)
        .round()
        .astype(int)
    )
    dd.assert_eq(expect, got)

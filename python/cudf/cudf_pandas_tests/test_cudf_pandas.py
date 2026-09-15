# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import collections
import copy
import cProfile
import datetime
import operator
import os
import pathlib
import pickle
import pstats
import subprocess
import sys
import tempfile
import time
import types
from collections.abc import Callable
from io import BytesIO, StringIO

import cupy as cp
import jupyter_client
import nbformat
import numpy as np
import pyarrow as pa
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
from numba import (
    vectorize,
)

from rmm import RMMError

from cudf.pandas import LOADED, Profiler
from cudf.pandas.fast_slow_proxy import (
    AttributeFallbackError,
    FallbackError,
    NotImplementedFallbackError,
    OOMFallbackError,
    TypeFallbackError,
    _State,
    _Unusable,
    as_proxy_object,
    is_proxy_object,
)
from cudf.testing import assert_eq

if not LOADED:
    raise ImportError("These tests must be run with cudf.pandas loaded")

import pandas as xpd
import pandas._testing as tm
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    EasterMonday,
    GoodFriday,
    Holiday,
    USColumbusDay,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    get_calendar,
)

from cudf.pandas import (
    is_proxy_instance,
)

# Accelerated pandas has the real pandas and cudf modules as attributes
pd = xpd._fsproxy_slow
cudf = xpd._fsproxy_fast


@pytest.fixture
def dataframe():
    pdf = pd.DataFrame({"a": [1, 1, 1, 2, 3], "b": [1, 2, 3, 4, 5]})
    df = xpd.DataFrame(pdf)
    return (pdf, df)


@pytest.fixture
def series(dataframe):
    pdf, df = dataframe
    return (pdf["a"], df["a"])


@pytest.fixture
def index():
    return (
        pd.Index(["a", "b", "c", "d", "e"]),
        xpd.Index(["a", "b", "c", "d", "e"]),
    )


@pytest.fixture
def multiindex(dataframe):
    pdf, df = dataframe
    pmi = pd.MultiIndex.from_frame(pdf)
    mi = xpd.MultiIndex.from_frame(df)
    return (pmi, mi)


@pytest.fixture
def array(series):
    arr, xarr = series
    return (arr.values, xarr.values)


@pytest.fixture(
    params=[
        lambda group: group["b"].sum(),
        lambda group: group.sum().apply(lambda val: [val]),
    ]
)
def groupby_udf(request):
    return request.param


def test_assert_equal():
    tm.assert_frame_equal(
        pd.DataFrame({"a": [1, 2, 3]}), xpd.DataFrame({"a": [1, 2, 3]})
    )

    with pytest.raises(AssertionError):
        tm.assert_frame_equal(
            pd.DataFrame({"a": [1, 2, 3]}), xpd.DataFrame({"a": [1, 2, 4]})
        )


def test_construction():
    # test that constructing a DataFrame returns an DataFrame
    data = {"a": [1, 2, 3], "b": ["x", "y", "z"]}
    pdf = pd.DataFrame(data)
    df = xpd.DataFrame(data)
    tm.assert_frame_equal(pdf, df)


def test_construction_object():
    # test that we can construct a Series with `object` dtype
    psr = pd.Series([1, "a", [1, 2, 3]])
    sr = xpd.Series([1, "a", [1, 2, 3]])
    tm.assert_series_equal(psr, sr)


def test_construction_from_frame(dataframe):
    pdf, _ = dataframe
    df = xpd.DataFrame(pdf)
    tm.assert_frame_equal(pdf, df)


def test_groupby(dataframe):
    pdf, df = dataframe
    expected = pdf.groupby("a", sort=True).max()
    gb = df.groupby("a", sort=True)
    got = gb.max()
    tm.assert_frame_equal(expected, got)


def test_groupby_reflects_parent_frame_mutation():
    # A groupby keyed by a string-column-derived array (df["a"].values) falls
    # back to the slow path, so the proxy groupby is created in the "slow"
    # state. A column added to the parent frame *after* the groupby is created
    # must still be visible through it, matching pandas' live-reference
    # semantics (the groupby holds a reference to the frame, not a snapshot).
    data = {"a": ["x", "y", "x", "y"], "b": [1, 2, 3, 4]}

    pdf = pd.DataFrame(data)
    pgb = pdf.groupby(pdf["a"].values)
    pdf["c"] = pdf["b"] * 10
    expected = pgb["c"].sum()

    df = xpd.DataFrame(data)
    gb = df.groupby(df["a"].values)
    df["c"] = df["b"] * 10
    got = gb["c"].sum()

    tm.assert_series_equal(expected, got)


def test_intermediate_proxy_rederivation_refreshes_cache():
    # When a parent proxy changes, an intermediate re-derives its wrapped object
    # on the next conversion. The re-derivation must refresh the cache
    # (``_fsproxy_wrapped``), not only the parent-id snapshot: otherwise the
    # snapshot matches the live parents again while the cache still points at the
    # stale object, so a later conversion returns stale data.
    data = {"a": ["x", "y", "x", "y"], "b": [1, 2, 3, 4]}

    df = xpd.DataFrame(data)
    # Grouping by a string-column-derived array falls back to the slow path, so
    # the groupby proxy is created (and stays) in the "slow" state.
    gb = df.groupby(df["a"].values)
    assert gb._fsproxy_state is _State.SLOW
    df["c"] = df["b"] * 10  # mutate the parent frame after the groupby exists

    # First conversion re-derives from the live parent and must see column "c".
    assert "c" in gb._fsproxy_fast_to_slow().sum().columns
    # The cache is now refreshed, so the parent-id snapshot matches again and a
    # subsequent conversion returns the cached object -- which must also see "c".
    assert "c" in gb._fsproxy_wrapped.sum().columns
    assert "c" in gb._fsproxy_fast_to_slow().sum().columns


def test_repr(dataframe):
    pdf, df = dataframe
    assert df.__repr__() == pdf.__repr__()


def test_binops_series(series):
    psr, sr = series
    expected = psr + psr
    got = sr + sr
    tm.assert_series_equal(expected, got)


def test_binops_df(dataframe):
    pdf, df = dataframe
    expected = pdf + pdf
    got = df + df
    tm.assert_frame_equal(expected, got)


def test_attribute(dataframe):
    pdf, df = dataframe
    assert pdf.shape == df.shape


def test_tz_localize():
    psr = pd.Series(["2001-01-01", "2002-02-02"], dtype="datetime64[ms]")
    sr = xpd.Series(psr)
    tm.assert_series_equal(
        psr.dt.tz_localize("America/New_York"),
        sr.dt.tz_localize("America/New_York"),
        check_dtype=False,
    )


def test_index_tz_localize():
    pti = pd.Index(pd.date_range("2020-01-01", periods=3, freq="D"))
    xti = xpd.Index(xpd.date_range("2020-01-01", periods=3, freq="D"))
    pti = pti.tz_localize("UTC")
    xti = xti.tz_localize("UTC")
    tm.assert_equal(pti, xti)


def test_index_generator():
    pi = pd.Index(iter(range(10)))
    xi = xpd.Index(iter(range(10)))
    tm.assert_equal(pi, xi)


@pytest.mark.filterwarnings(
    "ignore:DataFrameGroupBy.apply operated on the grouping columns"
)
def test_groupby_apply_fallback(dataframe, groupby_udf):
    pdf, df = dataframe
    tm.assert_equal(
        pdf.groupby("a", sort=True, group_keys=True).apply(groupby_udf),
        df.groupby("a", sort=True, group_keys=True).apply(groupby_udf),
    )


def test_groupby_external_series_apply_fallback(dataframe, groupby_udf):
    pdf, df = dataframe
    tm.assert_equal(
        pdf.groupby(
            pd.Series([1, 2, 1, 2, 1]), sort=True, group_keys=True
        ).apply(groupby_udf),
        df.groupby(
            xpd.Series([1, 2, 1, 2, 1]), sort=True, group_keys=True
        ).apply(groupby_udf),
    )


def test_read_csv():
    data = "1,2,3\n4,5,6"
    expected = pd.read_csv(StringIO(data))
    got = xpd.read_csv(StringIO(data))
    tm.assert_frame_equal(expected, got)


def test_iloc(dataframe):
    pdf, df = dataframe
    tm.assert_frame_equal(pdf.iloc[:, :], df.iloc[:, :])


def test_neg(dataframe):
    pdf, df = dataframe
    tm.assert_frame_equal(-pdf, -df)


def test_groupby_filter(dataframe):
    pdf, df = dataframe
    expected = pdf.groupby("a").filter(lambda df: len(df) > 2)
    got = df.groupby("a").filter(lambda df: len(df) > 2)
    tm.assert_frame_equal(expected, got)


def test_groupby_rolling(dataframe):
    pdf, df = dataframe
    expected = pdf.groupby("a").rolling(2).sum()
    got = df.groupby("a").rolling(2).sum()
    tm.assert_frame_equal(expected, got)


def test_groupby_rolling_window(dataframe):
    pdf, df = dataframe
    expected = pdf.groupby("a").rolling(2, win_type="triang").mean()
    got = df.groupby("a").rolling(2, win_type="triang").mean()
    tm.assert_frame_equal(expected, got)


def test_ewm():
    pdf = pd.DataFrame(range(5))
    df = xpd.DataFrame(range(5))
    result = df.ewm(0.5).mean()
    expected = pdf.ewm(0.5).mean()
    tm.assert_equal(result, expected)


def test_setitem_frame(dataframe):
    pdf, df = dataframe
    pdf[pdf > 1] = -pdf
    df[df > 1] = -df
    tm.assert_frame_equal(pdf, df)


def test_concat(dataframe):
    pdf, df = dataframe
    expected = pd.concat([pdf, pdf])
    got = xpd.concat([df, df])
    tm.assert_frame_equal(expected, got)


def test_attribute_error():
    df = xpd.DataFrame()
    with pytest.raises(AttributeError):
        df.blah


def test_df_from_series(series):
    psr, sr = series
    tm.assert_frame_equal(pd.DataFrame(psr), xpd.DataFrame(sr))


def test_iloc_change_type(series):
    # In pandas 3.0+, setting an item of incompatible dtype raises TypeError.
    psr, sr = series
    with pytest.raises(TypeError):
        psr.iloc[0] = "a"
    with pytest.raises(TypeError):
        sr.iloc[0] = "a"


def test_rename_categories():
    psr = pd.Series([1, 2, 3], dtype="category")
    sr = xpd.Series([1, 2, 3], dtype="category")
    psr = psr.cat.rename_categories({1: 5})
    sr = sr.cat.rename_categories({1: 5})
    tm.assert_series_equal(psr, sr)


def test_column_rename(dataframe):
    pdf, df = dataframe
    pdf.columns = ["x", "y"]
    df.columns = ["x", "y"]
    tm.assert_frame_equal(pdf, df)


def test_shape(dataframe):
    pdf, df = dataframe
    assert pdf.shape == df.shape
    pdf["c"] = range(5)
    df["c"] = range(5)
    assert pdf.shape == df.shape


def test_isnull():
    psr = pd.Series([1, 2, 3])
    sr = xpd.Series(psr)
    # Test that invoking `Pandas` functions works.
    tm.assert_series_equal(pd.isnull(psr), xpd.isnull(sr))


def test_copy_deepcopy_recursion(dataframe):
    # test that we don't recurse when calling the copy/deepcopy
    # methods, which can happen due to
    # https://nedbatchelder.com/blog/201010/surprising_getattr_recursion.html
    import copy

    _pdf, df = dataframe
    copy.copy(df)
    copy.deepcopy(df)


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy])
def test_copy_deepcopy(copier):
    s = xpd.Series([1, 2, 3])
    s2 = copier(s)
    assert isinstance(s2, s.__class__)
    tm.assert_equal(s, s2)

    df = xpd.DataFrame({"a": [1, 2, 3]})
    df2 = copier(df)
    assert isinstance(df2, df.__class__)
    tm.assert_equal(df, df2)

    idx = xpd.Index([1, 2, 3])
    idx2 = copier(idx)
    assert isinstance(idx2, idx.__class__)
    tm.assert_equal(idx, idx2)


def test_classmethod():
    pdf = pd.DataFrame.from_dict({"a": [1, 2, 3]})
    df = xpd.DataFrame.from_dict({"a": [1, 2, 3]})
    tm.assert_frame_equal(pdf, df)


def test_rolling(dataframe):
    pdf, df = dataframe
    tm.assert_frame_equal(pdf.rolling(2).agg("sum"), df.rolling(2).agg("sum"))


def test_array_function_series(series):
    psr, sr = series
    np.testing.assert_allclose(np.average(psr), np.average(sr))


def test_array_function_ndarray(array):
    arr, xarr = array
    np.isclose(np.average(arr), np.average(xarr))


def test_histogram_ndarray(array):
    arr, xarr = array
    expected_hist, expected_edges = np.histogram(arr, bins="auto")
    got_hist, got_edges = np.histogram(xarr, bins="auto")
    tm.assert_almost_equal(expected_hist, got_hist)
    tm.assert_almost_equal(expected_edges, got_edges)


def test_pickle_round_trip(dataframe):
    pdf, df = dataframe
    pickled_pdf = BytesIO()
    pickled_cudf_pandas = BytesIO()
    pdf.to_pickle(pickled_pdf)
    df.to_pickle(pickled_cudf_pandas)

    pickled_pdf.seek(0)
    pickled_cudf_pandas.seek(0)

    tm.assert_frame_equal(
        pd.read_pickle(pickled_pdf), xpd.read_pickle(pickled_cudf_pandas)
    )


def test_excel_round_trip(dataframe):
    pytest.importorskip("openpyxl")

    pdf, df = dataframe
    excel_pdf = BytesIO()
    excel_cudf_pandas = BytesIO()
    pdf.to_excel(excel_pdf)
    df.to_excel(excel_cudf_pandas)

    excel_pdf.seek(0)
    excel_cudf_pandas.seek(0)

    tm.assert_frame_equal(
        pd.read_excel(excel_pdf), xpd.read_excel(excel_cudf_pandas)
    )


def test_hash_array(series):
    ps, xs = series
    expected = pd.util.hash_array(ps.values)
    actual = xpd.util.hash_array(xs.values)

    tm.assert_almost_equal(expected, actual)


def test_is_sparse():
    psa = pd.arrays.SparseArray([0, 0, 1, 0])
    xsa = xpd.arrays.SparseArray([0, 0, 1, 0])

    assert isinstance(psa.dtype, pd.SparseDtype) == isinstance(
        xsa.dtype, xpd.SparseDtype
    )


def test_is_file_like():
    assert pd.api.types.is_file_like("a") == xpd.api.types.is_file_like("a")  # noqa: TID251
    assert pd.api.types.is_file_like(BytesIO()) == xpd.api.types.is_file_like(  # noqa: TID251
        BytesIO()
    )
    assert pd.api.types.is_file_like(
        StringIO("abc")
    ) == xpd.api.types.is_file_like(StringIO("abc"))  # noqa: TID251


def test_is_re_compilable():
    assert pd.api.types.is_re_compilable(
        ".^"
    ) == xpd.api.types.is_re_compilable(".^")  # noqa: TID251
    assert pd.api.types.is_re_compilable(
        ".*"
    ) == xpd.api.types.is_re_compilable(".*")  # noqa: TID251


def test_module_attribute_types():
    assert isinstance(xpd.read_csv, types.FunctionType)
    assert isinstance(xpd.tseries.frequencies.Day, type)
    assert isinstance(xpd.api, types.ModuleType)


def test_infer_freq():
    expected = pd.infer_freq(
        pd.date_range(start="2020/12/01", end="2020/12/30", periods=30)
    )
    got = xpd.infer_freq(
        xpd.date_range(start="2020/12/01", end="2020/12/30", periods=30)
    )
    assert expected == got


@pytest.mark.filterwarnings(
    "ignore:DataFrameGroupBy.apply operated on the grouping columns"
)
def test_groupby_grouper_fallback(dataframe, groupby_udf):
    pdf, df = dataframe
    tm.assert_equal(
        pdf.groupby(pd.Grouper("a"), sort=True, group_keys=True).apply(
            groupby_udf
        ),
        df.groupby(xpd.Grouper("a"), sort=True, group_keys=True).apply(
            groupby_udf
        ),
    )


def test_options_mode():
    assert xpd.options.mode.string_storage == pd.options.mode.string_storage


# Codecov and Profiler interfere with each-other,
# hence we don't want to run code-cov on this test.
@pytest.mark.no_cover
def test_cudf_pandas_profiler():
    pytest.importorskip("cudf")

    # test that the profiler correctly reports
    # when we use the GPU v/s CPU
    with Profiler() as p:
        df = xpd.DataFrame({"a": [1, 2, 3], "b": "b"})
        df.groupby("a").max()

    assert len(p.per_line_stats) == 2
    for line_no, line, gpu_time, cpu_time in p.per_line_stats:
        assert gpu_time
        assert not cpu_time

    with Profiler() as p:
        s = xpd.Series([1, "a"])
        s = s + s

    assert len(p.per_line_stats) == 2
    for line_no, line, gpu_time, cpu_time in p.per_line_stats:
        assert cpu_time


def test_column_access_as_attribute():
    pdf = pd.DataFrame({"fast": [1, 2, 3], "slow": [2, 3, 4]})
    df = xpd.DataFrame({"fast": [1, 2, 3], "slow": [2, 3, 4]})

    tm.assert_series_equal(pdf.fast, df.fast)
    tm.assert_series_equal(pdf.slow, df.slow)


def test_binop_dataframe_list(dataframe):
    pdf, df = dataframe
    expect = pdf[["a"]] == [[1, 2, 3, 4, 5]]
    got = df[["a"]] == [[1, 2, 3, 4, 5]]
    tm.assert_frame_equal(expect, got)


def test_binop_array_series(series):
    psr, sr = series
    arr = psr.array
    expect = arr + psr
    got = arr + sr
    tm.assert_series_equal(expect, got)


def test_array_ufunc_reduction(series):
    psr, sr = series
    expect = np.ufunc.reduce(np.subtract, psr)
    got = np.ufunc.reduce(np.subtract, sr)
    tm.assert_equal(expect, got)


def test_array_ufunc(series):
    psr, sr = series
    expect = np.subtract(psr, psr)
    got = np.subtract(sr, sr)
    assert isinstance(got, sr.__class__)
    tm.assert_equal(expect, got)


@pytest.mark.xfail(strict=False, reason="Fails in CI, passes locally.")
def test_groupby_apply_func_returns_series(dataframe):
    pdf, df = dataframe
    expect = pdf.groupby("a").apply(
        lambda group: pd.Series({"x": 1}), include_groups=False
    )
    got = df.groupby("a").apply(
        lambda group: xpd.Series({"x": 1}), include_groups=False
    )
    tm.assert_equal(expect, got)


@pytest.mark.parametrize("data", [[1, 2, 3], ["a", None, "b"]])
def test_pyarrow_array_construction(data):
    cudf_pandas_series = xpd.Series(data)
    pandas_series = pd.Series(data)
    actual_pa_array = pa.array(cudf_pandas_series)
    expected_pa_array = pa.array(pandas_series)
    assert actual_pa_array.equals(expected_pa_array)


@pytest.mark.parametrize(
    "op", [">", "<", "==", "<=", ">=", "+", "%", "-", "*", "/"]
)
def test_cudf_pandas_eval_series(op):
    lhs = xpd.Series([10, 11, 12])  # noqa: F841
    rhs = xpd.Series([100, 1, 12])  # noqa: F841

    actual = xpd.eval(f"lhs {op} rhs")

    pd_lhs = pd.Series([10, 11, 12])  # noqa: F841
    pd_rhs = pd.Series([100, 1, 12])  # noqa: F841

    expected = pd.eval(f"pd_lhs {op} pd_rhs")

    tm.assert_series_equal(expected, actual)


@pytest.mark.parametrize(
    "op", [">", "<", "==", "<=", ">=", "+", "%", "-", "*", "/"]
)
def test_cudf_pandas_eval_dataframe(op):
    lhs = xpd.DataFrame({"a": [10, 11, 12], "b": [1, 2, 3]})  # noqa: F841
    rhs = xpd.DataFrame({"a": [100, 1, 12], "b": [15, -10, 3]})  # noqa: F841

    actual = xpd.eval(f"lhs {op} rhs")

    pd_lhs = pd.DataFrame({"a": [10, 11, 12], "b": [1, 2, 3]})  # noqa: F841
    pd_rhs = pd.DataFrame({"a": [100, 1, 12], "b": [15, -10, 3]})  # noqa: F841

    expected = pd.eval(f"pd_lhs {op} pd_rhs")

    tm.assert_frame_equal(expected, actual)


@pytest.mark.parametrize(
    "expr", ["((a + b) * c % d) > e", "((a + b) * c % d)"]
)
def test_cudf_pandas_eval_complex(expr):
    data = {
        "a": [10, 11, 12],
        "b": [1, 2, 3],
        "c": [100, 1, 12],
        "d": [15, -10, 3],
        "e": [100, 200, 300],
    }
    cudf_pandas_frame = xpd.DataFrame(data)
    pd_frame = pd.DataFrame(data)
    actual = cudf_pandas_frame.eval(expr)
    expected = pd_frame.eval(expr)
    tm.assert_series_equal(expected, actual)


def test_array_function_series_fallback(series):
    psr, sr = series
    expect = np.unique(psr, return_counts=True)
    got = np.unique(sr, return_counts=True)
    tm.assert_equal(expect, got)


def test_timedeltaproperties(series):
    psr, sr = series
    psr, sr = psr.astype("timedelta64[ns]"), sr.astype("timedelta64[ns]")
    tm.assert_equal(psr.dt.days, sr.dt.days)
    tm.assert_equal(psr.dt.components, sr.dt.components)
    tm.assert_equal(psr.dt.total_seconds(), sr.dt.total_seconds())


@pytest.mark.parametrize("scalar_type", [int, float, complex, bool])
@pytest.mark.parametrize("scalar", [1, 1.0, True, 0])
def test_coerce_zero_d_array_to_scalar(scalar_type, scalar):
    expected = scalar_type(pd.Series([scalar]).values[0])
    got = scalar_type(xpd.Series([scalar]).values[0])
    tm.assert_equal(expected, got)


def test_cupy_asarray_zero_copy():
    cp = pytest.importorskip("cupy")

    sr = xpd.Series([1, 2, 3])
    cpary = cp.asarray(sr.values)

    assert (
        sr.__cuda_array_interface__["data"][0]
        == cpary.__cuda_array_interface__["data"][0]
    )


def test_pipe(dataframe):
    pdf, df = dataframe

    def func(df, x):
        return df + x

    expect = pdf.pipe(func, 1)
    got = df.pipe(func, 1)
    tm.assert_frame_equal(expect, got)


def test_pipe_tuple(dataframe):
    pdf, df = dataframe

    def func(x, df):
        return df + x

    expect = pdf.pipe((func, "df"), 1)
    got = df.pipe((func, "df"), 1)
    tm.assert_frame_equal(expect, got)


def test_maintain_container_subclasses(multiindex):
    # pandas Frozenlist is a list subclass
    pmi, mi = multiindex
    got = mi.names.difference(["b"])
    expect = pmi.names.difference(["b"])
    assert got == expect
    assert isinstance(got, xpd.core.indexes.frozen.FrozenList)


def test_rolling_win_type():
    pdf = pd.DataFrame(range(5))
    df = xpd.DataFrame(range(5))
    result = df.rolling(2, win_type="boxcar").mean()
    expected = pdf.rolling(2, win_type="boxcar").mean()
    tm.assert_equal(result, expected)


def test_rolling_apply_numba_engine():
    def weighted_mean(x):
        arr = np.ones((1, x.shape[1]))
        arr[:, :2] = (x[:, :2] * x[:, 2]).sum(axis=0) / x[:, 2].sum()
        return arr

    pdf = pd.DataFrame([[1, 2, 0.6], [2, 3, 0.4], [3, 4, 0.2], [4, 5, 0.7]])
    df = xpd.DataFrame([[1, 2, 0.6], [2, 3, 0.4], [3, 4, 0.2], [4, 5, 0.7]])

    expect = pdf.rolling(2, method="table", min_periods=0).apply(
        weighted_mean, raw=True, engine="numba"
    )
    got = df.rolling(2, method="table", min_periods=0).apply(
        weighted_mean, raw=True, engine="numba"
    )
    tm.assert_equal(expect, got)


def test_expanding():
    pdf = pd.DataFrame(range(5))
    df = xpd.DataFrame(range(5))
    result = df.expanding().mean()
    expected = pdf.expanding().mean()
    tm.assert_equal(result, expected)


def test_pipe_with_data_creating_func():
    def pandas_func(df):
        df2 = pd.DataFrame({"b": np.arange(len(df))})
        return df.join(df2)

    def cudf_pandas_func(df):
        df2 = xpd.DataFrame({"b": np.arange(len(df))})
        return df.join(df2)

    pdf = pd.DataFrame({"a": [1, 2, 3]})
    df = xpd.DataFrame({"a": [1, 2, 3]})

    tm.assert_frame_equal(pdf.pipe(pandas_func), df.pipe(cudf_pandas_func))


@pytest.mark.parametrize(
    "data",
    [
        '{"a": 1, "b": 2, "c": 3}',
        '{"a": 1, "b": 2, "c": 3}\n{"a": 4, "b": 5, "c": 6}',
    ],
)
def test_chunked_json_reader(tmpdir, data):
    file_path = tmpdir / "test.json"
    with open(file_path, "w") as f:
        f.write(data)

    with (
        pd.read_json(file_path, lines=True, chunksize=1) as pd_reader,
        xpd.read_json(file_path, lines=True, chunksize=1) as xpd_reader,
    ):
        for pd_chunk, xpd_chunk in zip(pd_reader, xpd_reader, strict=True):
            tm.assert_equal(pd_chunk, xpd_chunk)

    with (
        pd.read_json(StringIO(data), lines=True, chunksize=1) as pd_reader,
        xpd.read_json(StringIO(data), lines=True, chunksize=1) as xpd_reader,
    ):
        for pd_chunk, xpd_chunk in zip(pd_reader, xpd_reader, strict=True):
            tm.assert_equal(pd_chunk, xpd_chunk)


@pytest.mark.parametrize(
    "data",
    [
        "1,2,3",
        "1,2,3\n4,5,6",
    ],
)
def test_chunked_csv_reader(tmpdir, data):
    file_path = tmpdir / "test.json"
    with open(file_path, "w") as f:
        f.write(data)

    with (
        pd.read_csv(file_path, chunksize=1) as pd_reader,
        xpd.read_csv(file_path, chunksize=1) as xpd_reader,
    ):
        for pd_chunk, xpd_chunk in zip(pd_reader, xpd_reader, strict=True):
            tm.assert_equal(pd_chunk, xpd_chunk, check_index_type=False)

    with (
        pd.read_json(StringIO(data), lines=True, chunksize=1) as pd_reader,
        xpd.read_json(StringIO(data), lines=True, chunksize=1) as xpd_reader,
    ):
        for pd_chunk, xpd_chunk in zip(pd_reader, xpd_reader, strict=True):
            tm.assert_equal(pd_chunk, xpd_chunk, check_index_type=False)


@pytest.mark.parametrize(
    "data", [(), (1,), (1, 2, 3), ("a", "b", "c"), (1, 2, "test")]
)
def test_construct_from_generator(data):
    expect = pd.Series((x for x in data))
    got = xpd.Series((x for x in data))
    tm.assert_series_equal(expect, got)


def test_read_csv_stringio_usecols():
    data = "col1,col2,col3\na,b,1\na,b,2\nc,d,3"
    expect = pd.read_csv(StringIO(data), usecols=lambda x: x.upper() != "COL3")
    got = xpd.read_csv(StringIO(data), usecols=lambda x: x.upper() != "COL3")
    tm.assert_frame_equal(expect, got)


def test_construct_datetime_index():
    expect = pd.DatetimeIndex([10, 20, 30], dtype="datetime64[ns]")
    got = xpd.DatetimeIndex([10, 20, 30], dtype="datetime64[ns]")
    tm.assert_index_equal(expect, got)


def test_construct_timedelta_index():
    expect = pd.TimedeltaIndex([10, 20, 30], dtype="timedelta64[ns]")
    got = xpd.TimedeltaIndex([10, 20, 30], dtype="timedelta64[ns]")
    tm.assert_index_equal(expect, got)


@pytest.mark.parametrize(
    "op",
    [
        operator.eq,
        operator.sub,
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
    ],
)
def test_datetime_ops(op):
    pd_dt_idx1 = pd.DatetimeIndex([10, 20, 30], dtype="datetime64[ns]")
    cudf_pandas_dt_idx = xpd.DatetimeIndex(
        [10, 20, 30], dtype="datetime64[ns]"
    )

    tm.assert_equal(
        op(pd_dt_idx1, pd_dt_idx1), op(cudf_pandas_dt_idx, cudf_pandas_dt_idx)
    )


@pytest.mark.parametrize(
    "op",
    [
        operator.eq,
        operator.add,
        operator.sub,
        operator.lt,
        operator.gt,
        operator.le,
        operator.ge,
    ],
)
def test_timedelta_ops(op):
    pd_td_idx1 = pd.TimedeltaIndex([10, 20, 30], dtype="timedelta64[ns]")
    cudf_pandas_td_idx = xpd.TimedeltaIndex(
        [10, 20, 30], dtype="timedelta64[ns]"
    )

    tm.assert_equal(
        op(pd_td_idx1, pd_td_idx1), op(cudf_pandas_td_idx, cudf_pandas_td_idx)
    )


@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_datetime_timedelta_ops(op):
    pd_dt_idx1 = pd.DatetimeIndex([10, 20, 30], dtype="datetime64[ns]")
    cudf_pandas_dt_idx = xpd.DatetimeIndex(
        [10, 20, 30], dtype="datetime64[ns]"
    )

    pd_td_idx1 = pd.TimedeltaIndex([10, 20, 30], dtype="timedelta64[ns]")
    cudf_pandas_td_idx = xpd.TimedeltaIndex(
        [10, 20, 30], dtype="timedelta64[ns]"
    )

    tm.assert_equal(
        op(pd_dt_idx1, pd_td_idx1), op(cudf_pandas_dt_idx, cudf_pandas_td_idx)
    )


def test_itertuples():
    df = xpd.DataFrame(range(1))
    result = next(iter(df.itertuples()))
    tup = collections.namedtuple("Pandas", ["Index", "1"], rename=True)
    expected = tup(0, 0)
    assert result == expected
    assert result._fields == expected._fields


def test_namedagg_namedtuple():
    df = xpd.DataFrame(
        {
            "kind": ["cat", "dog", "cat", "dog"],
            "height": [9.1, 6.0, 9.5, 34.0],
            "weight": [7.9, 7.5, 9.9, 198.0],
        }
    )
    result = df.groupby("kind").agg(
        min_height=pd.NamedAgg(column="height", aggfunc="min"),
        max_height=pd.NamedAgg(column="height", aggfunc="max"),
        average_weight=pd.NamedAgg(column="weight", aggfunc="mean"),
    )
    expected = xpd.DataFrame(
        {
            "min_height": [9.1, 6.0],
            "max_height": [9.5, 34.0],
            "average_weight": [8.90, 102.75],
        },
        index=xpd.Index(["cat", "dog"], name="kind"),
    )
    tm.assert_frame_equal(result, expected)


def test_dataframe_dir(dataframe):
    """Test that column names are present in the dataframe dir

    We do not test direct dir equality because pandas does some runtime
    modifications of dir that we cannot replicate without forcing D2H
    conversions (e.g. modifying what elements are visible based on the contents
    of a DataFrame instance).
    """
    _, df = dataframe
    assert "a" in dir(df)
    assert "b" in dir(df)

    # Only string types are added to dir
    df[1] = [1] * len(df)
    assert 1 not in dir(df)


def test_array_copy(array):
    arr, xarr = array
    tm.assert_equal(copy.copy(arr), copy.copy(xarr))


def test_datetime_values_dtype_roundtrip():
    s = pd.Series([1, 2, 3], dtype="datetime64[ns]")
    xs = xpd.Series([1, 2, 3], dtype="datetime64[ns]")
    expected = np.asarray(s.values)
    actual = np.asarray(xs.values)
    assert expected.dtype == actual.dtype
    tm.assert_equal(expected, actual)


def test_resample():
    ser = pd.Series(
        range(3), index=pd.date_range("2020-01-01", freq="D", periods=3)
    )
    xser = xpd.Series(
        range(3), index=xpd.date_range("2020-01-01", freq="D", periods=3)
    )
    expected = ser.resample("D").max()
    result = xser.resample("D").max()
    # TODO: See if as_unit can be avoided
    expected.index = expected.index.as_unit("s")
    result.index = result.index.as_unit("s")
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize("accessor", ["str", "dt", "cat"])
def test_accessor_types(accessor):
    assert isinstance(getattr(xpd.Series, accessor), type)


@pytest.mark.parametrize(
    "op",
    [operator.itemgetter(1), operator.methodcaller("sum")],
    ids=["getitem[1]", ".sum()"],
)
def test_values_zero_dim_result_is_scalar(op):
    s = pd.Series([1, 2, 3])
    x = xpd.Series([1, 2, 3])
    expect = op(s.values)
    got = op(x.values)
    assert expect == got
    assert type(expect) is type(got)


@pytest.mark.parametrize("box", ["DataFrame", "Series"])
def test_round_builtin(box):
    xobj = getattr(xpd, box)([1.23])
    pobj = getattr(pd, box)([1.23])
    result = round(xobj, 1)
    expected = round(pobj, 1)
    tm.assert_equal(result, expected)


def test_hash():
    xobj = xpd.Timedelta(1)
    pobj = pd.Timedelta(1)
    assert hash(xobj) == hash(pobj)


def test_non_hashable_object():
    with pytest.raises(TypeError):
        hash(pd.DataFrame(range(1)))

    with pytest.raises(TypeError):
        hash(xpd.DataFrame(range(1)))


@pytest.mark.parametrize("offset", ["DateOffset", "Day", "BDay"])
def test_timestamp_offset_binop(offset):
    ts = xpd.Timestamp(2020, 1, 1)
    result = ts + getattr(xpd.offsets, offset)()
    expected = pd.Timestamp(2020, 1, 2)
    tm.assert_equal(result, expected)


def test_string_dtype():
    xobj = xpd.StringDtype()
    pobj = pd.StringDtype()
    tm.assert_equal(xobj, pobj)


def test_string_array():
    data = np.array(["1"], dtype=object)
    xobj = xpd.arrays.StringArray(data)
    pobj = pd.arrays.StringArray(data)
    tm.assert_extension_array_equal(xobj, pobj)


def test_string_array_is_numpy_extension_subclass():
    assert issubclass(xpd.arrays.StringArray, xpd.arrays.NumpyExtensionArray)


@pytest.mark.parametrize(
    "comparison_op",
    [
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ],
)
def test_string_array_object_array_comparison(comparison_op):
    xa = xpd.array(["a", None, "c"], dtype=object)
    xb = xpd.array([None, None, "c"], dtype="string[python]")
    pa_ = pd.array(["a", None, "c"], dtype=object)
    pb_ = pd.array([None, None, "c"], dtype="string[python]")

    tm.assert_extension_array_equal(
        comparison_op(xa, xb), comparison_op(pa_, pb_)
    )
    tm.assert_extension_array_equal(
        comparison_op(xb, xa), comparison_op(pb_, pa_)
    )
    tm.assert_extension_array_equal(
        comparison_op(xa, xb), comparison_op(xb, xa)
    )


def test_subclass_series():
    class foo(pd.Series):
        def __init__(self, myinput):
            super().__init__(myinput)

    s1 = pd.Series([1, 2, 3])
    s2 = foo(myinput=[1, 2, 3])

    tm.assert_equal(s1, s2, check_series_type=False)


@pytest.mark.parametrize(
    "index_type",
    [
        xpd.RangeIndex,
        xpd.CategoricalIndex,
        xpd.DatetimeIndex,
        xpd.TimedeltaIndex,
        xpd.PeriodIndex,
        xpd.MultiIndex,
        xpd.IntervalIndex,
    ],
)
def test_index_subclass(index_type):
    # test that proxy index types are derived
    # from Index
    assert issubclass(index_type, xpd.Index)
    assert not issubclass(xpd.Index, index_type)


def test_np_array_of_timestamps():
    expected = np.array([pd.Timestamp(1)]) + pd.tseries.offsets.MonthEnd()
    got = np.array([xpd.Timestamp(1)]) + xpd.tseries.offsets.MonthEnd()
    tm.assert_equal(expected, got)


@pytest.mark.parametrize(
    "obj",
    [
        # Basic types
        xpd.Series(dtype="float64"),
        xpd.Series([1, 2, 3]),
        xpd.DataFrame(dtype="float64"),
        xpd.DataFrame({"a": [1, 2, 3]}),
        xpd.Series([1, 2, 3]),
        # Index (doesn't support nullary construction)
        xpd.Index([1, 2, 3]),
        xpd.RangeIndex(0, 10),
        xpd.Index(["a", "b", "c"]),
        # Complex index
        xpd.to_datetime(
            [
                "1/1/2018",
                np.datetime64("2018-01-01"),
                datetime.datetime(2018, 1, 1),
            ]
        ),
        xpd.TimedeltaIndex([100, 200, 300], dtype="timedelta64[ns]"),
        xpd.MultiIndex.from_tuples([(1, 2), (3, 4)]),
        # Objects where the underlying store is the slow type.
        xpd.Series(["a", 2, 3]),
        xpd.Index(["a", 2, 3]),
        # Other types
        xpd.tseries.offsets.BDay(5),
        xpd.Timestamp("2001-01-01"),
        xpd.Timestamp("2001-01-01", tz="UTC"),
        xpd.Timedelta("1 days"),
        xpd.Timedelta(1, "D"),
    ],
)
@pytest.mark.parametrize("pickle_func", [pickle.dump, xpd.to_pickle])
@pytest.mark.parametrize("read_pickle_func", [pickle.load, xpd.read_pickle])
def test_pickle(obj, pickle_func, read_pickle_func):
    with tempfile.TemporaryFile() as f:
        pickle_func(obj, f)
        f.seek(0)
        copy = read_pickle_func(f)

    tm.assert_equal(obj, copy)


def test_dataframe_query():
    cudf_pandas_df = xpd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})
    pd_df = pd.DataFrame({"foo": [1, 2, 3], "bar": [4, 5, 6]})

    actual = cudf_pandas_df.query("foo > 2")
    expected = pd_df.query("foo > 2")

    tm.assert_equal(actual, expected)

    bizz = 2  # noqa: F841
    actual = cudf_pandas_df.query("foo > @bizz")
    expected = pd_df.query("foo > @bizz")

    tm.assert_equal(actual, expected)


def test_private_method_result_wrapped():
    xoffset = xpd.offsets.Day()
    dt = datetime.datetime(2020, 1, 1)
    result = xoffset._apply(dt)
    assert isinstance(result, xpd.Timestamp)


def test_numpy_var():
    rng = np.random.default_rng(seed=42)
    data = rng.random(1000)
    psr = pd.Series(data)
    sr = xpd.Series(data)

    tm.assert_almost_equal(np.var(psr), np.var(sr))


def test_index_new():
    expected = pd.Index.__new__(pd.Index, [1, 2, 3])
    got = xpd.Index.__new__(xpd.Index, [1, 2, 3])
    tm.assert_equal(expected, got)

    expected = pd.Index.__new__(pd.Index, [1, 2, 3], dtype="int8")
    got = xpd.Index.__new__(xpd.Index, [1, 2, 3], dtype="int8")
    tm.assert_equal(expected, got)

    expected = pd.RangeIndex.__new__(pd.RangeIndex, 0, 10, 2)
    got = xpd.RangeIndex.__new__(xpd.RangeIndex, 0, 10, 2)
    tm.assert_equal(expected, got)


@pytest.mark.filterwarnings(
    "ignore:DataFrameGroupBy.apply operated on the grouping columns"
)
@pytest.mark.xfail(not LOADED, reason="Should not fail in accelerated mode")
def test_groupby_apply_callable_referencing_pandas(dataframe):
    pdf, df = dataframe

    class Callable1:
        def __call__(self, df):
            if not isinstance(df, pd.DataFrame):
                raise TypeError
            return 1

    class Callable2:
        def __call__(self, df):
            if not isinstance(df, xpd.DataFrame):
                raise TypeError
            return 1

    expect = pdf.groupby("a").apply(Callable1())
    got = df.groupby("a").apply(Callable2())

    tm.assert_equal(expect, got)


def test_constructor_properties(dataframe, series, index):
    _, df = dataframe
    _, sr = series
    _, idx = index

    assert df._constructor is xpd.DataFrame
    assert sr._constructor is xpd.Series
    assert idx._constructor is xpd.Index
    assert sr._constructor_expanddim is xpd.DataFrame
    assert df._constructor_sliced is xpd.Series


def test_pos():
    xser = +xpd.Series([-1])
    ser = +pd.Series([-1])
    tm.assert_equal(xser, ser)


def test_intermediates_are_proxied():
    df = xpd.DataFrame({"a": [1, 2, 3]})
    grouper = df.groupby("a")
    assert isinstance(grouper, xpd.core.groupby.generic.DataFrameGroupBy)


def test_multiindex_values_returns_1d_tuples():
    mi = xpd.MultiIndex.from_tuples([(1, 2), (3, 4)])
    result = mi.values
    expected = np.empty(2, dtype=object)
    expected[...] = [(1, 2), (3, 4)]
    tm.assert_equal(result, expected)


def test_read_sas_context():
    cudf_path = pathlib.Path(__file__).parent.parent
    path = cudf_path / "cudf" / "tests" / "data" / "sas" / "cars.sas7bdat"
    with xpd.read_sas(path, format="sas7bdat", iterator=True) as reader:
        df = reader.read()
    assert isinstance(df, xpd.DataFrame)


def test_concat_fast():
    pytest.importorskip("cudf")

    assert type(xpd.concat._fsproxy_fast) is not _Unusable


def test_func_namespace():
    # note: this test is sensitive to Pandas' internal module layout
    assert xpd.concat is xpd.core.reshape.concat.concat


def test_register_accessor():
    @xpd.api.extensions.register_dataframe_accessor("xyz")
    class XYZ:
        def __init__(self, obj):
            self._obj = obj

        @property
        def foo(self):
            return "spam"

    # the accessor must be registered with the proxy type,
    # not the underlying fast or slow type
    assert "xyz" in xpd.DataFrame.__dict__

    df = xpd.DataFrame()
    assert df.xyz.foo == "spam"


def test_pickle_groupby(dataframe):
    pdf, df = dataframe
    pgb = pdf.groupby("a")
    gb = df.groupby("a")
    gb = pickle.loads(pickle.dumps(gb))
    tm.assert_equal(pgb.sum(), gb.sum())


def test_numpy_extension_array():
    np_array = np.array([0, 1, 2, 3])
    try:
        xarray = xpd.arrays.NumpyExtensionArray(np_array)
        array = pd.arrays.NumpyExtensionArray(np_array)
    except AttributeError:
        xarray = xpd.arrays.PandasArray(np_array)
        array = pd.arrays.PandasArray(np_array)

    tm.assert_equal(xarray, array)


def test_isinstance_base_offset():
    offset = xpd.tseries.frequencies.to_offset("1s")
    assert isinstance(offset, xpd.tseries.offsets.BaseOffset)


def test_super_attribute_lookup():
    # test that we can use super() to access attributes
    # of the base class when subclassing proxy types

    class Foo(xpd.Series):
        def max_times_two(self):
            return super().max() * 2

    s = Foo([1, 2, 3])
    assert s.max_times_two() == 6


def test_floordiv_array_vs_df():
    xarray = xpd.Series([1, 2, 3], dtype="datetime64[ns]").array
    parray = pd.Series([1, 2, 3], dtype="datetime64[ns]").array

    xdf = xpd.DataFrame(xarray)
    pdf = pd.DataFrame(parray)

    actual = xarray.__floordiv__(xdf)
    expected = parray.__floordiv__(pdf)

    tm.assert_equal(actual, expected)


def test_apply_slow_path_udf_references_global_module():
    def my_apply(df, unused):
        # `datetime` Raised `KeyError: __import__`
        datetime.datetime.strptime(df["Minute"], "%H:%M:%S")
        return pd.to_numeric(1)

    df = xpd.DataFrame({"Minute": ["09:00:00"]})
    result = df.apply(my_apply, axis=1, unused=True)
    expected = xpd.Series([1])
    tm.assert_series_equal(result, expected)


@pytest.mark.parametrize(
    "op",
    [
        "__iadd__",
        "__iand__",
        "__ifloordiv__",
        "__imod__",
        "__imul__",
        "__ior__",
        "__ipow__",
        "__isub__",
        "__itruediv__",
        "__ixor__",
    ],
)
def test_inplace_ops(op):
    xdf1 = xpd.DataFrame({"a": [10, 11, 12]})
    xdf2 = xpd.DataFrame({"a": [1, 2, 3]})

    df1 = pd.DataFrame({"a": [10, 11, 12]})
    df2 = pd.DataFrame({"a": [1, 2, 3]})

    actual = getattr(xdf1, op)(xdf2)
    expected = getattr(df1, op)(df2)

    tm.assert_equal(actual, expected)


@pytest.mark.parametrize(
    "op",
    [
        "__iadd__",
        "__iand__",
        "__ifloordiv__",
        "__imod__",
        "__imul__",
        "__ior__",
        "__ipow__",
        "__isub__",
        "__itruediv__",
        "__ixor__",
    ],
)
def test_inplace_ops_series(op):
    xser1 = xpd.Series([10, 11, 12])
    xser2 = xpd.Series([1, 2, 3])

    ser1 = pd.Series([10, 11, 12])
    ser2 = pd.Series([1, 2, 3])

    actual = getattr(xser1, op)(xser2)
    expected = getattr(ser1, op)(ser2)

    tm.assert_equal(actual, expected)


@pytest.mark.parametrize("data", [pd.NaT, 1234, "nat"])
def test_timestamp(data):
    xtimestamp = xpd.Timestamp(data)
    assert isinstance(xtimestamp, datetime.datetime)
    timestamp = pd.Timestamp(data)
    tm.assert_equal(xtimestamp, timestamp)


@pytest.mark.parametrize("data", [pd.NaT, 1234, "nat"])
def test_timedelta(data):
    xtimedelta = xpd.Timedelta(data)
    if not (data is pd.NaT or data == "nat"):
        # pandas.NaT subclasses datetime.datetime
        assert isinstance(xtimedelta, datetime.timedelta)
    timedelta = pd.Timedelta(data)
    tm.assert_equal(xtimedelta, timedelta)


def test_abstract_holiday_calendar():
    class TestCalendar(AbstractHolidayCalendar):
        def __init__(self, name=None, rules=None) -> None:
            super().__init__(name=name, rules=rules)

    jan1 = TestCalendar(rules=[Holiday("jan1", year=2015, month=1, day=1)])
    jan2 = TestCalendar(rules=[Holiday("jan2", year=2015, month=1, day=2)])

    # Getting holidays for Jan 1 should not alter results for Jan 2.
    expected = xpd.DatetimeIndex(["01-Jan-2015"]).as_unit("ns")
    tm.assert_index_equal(jan1.holidays(), expected)

    expected2 = xpd.DatetimeIndex(["02-Jan-2015"]).as_unit("ns")
    tm.assert_index_equal(jan2.holidays(), expected2)


@pytest.mark.parametrize(
    "holiday,start,expected",
    [
        (USMemorialDay, datetime.datetime(2015, 7, 1), []),
        (USLaborDay, "2015-09-07", [xpd.Timestamp("2015-09-07")]),
        (USColumbusDay, "2015-10-12", [xpd.Timestamp("2015-10-12")]),
        (USThanksgivingDay, "2015-11-26", [xpd.Timestamp("2015-11-26")]),
        (USMartinLutherKingJr, "2015-01-19", [xpd.Timestamp("2015-01-19")]),
        (USPresidentsDay, datetime.datetime(2015, 7, 1), []),
        (GoodFriday, datetime.datetime(2015, 7, 1), []),
        (EasterMonday, "2015-04-06", [xpd.Timestamp("2015-04-06")]),
        ("New Year's Day", "2010-12-31", [xpd.Timestamp("2010-12-31")]),
        ("Independence Day", "2015-07-03", [xpd.Timestamp("2015-07-03")]),
        ("Veterans Day", "2012-11-11", []),
        ("Christmas Day", "2011-12-26", [xpd.Timestamp("2011-12-26")]),
        (
            "Juneteenth National Independence Day",
            "2021-06-18",
            [xpd.Timestamp("2021-06-18")],
        ),
        ("Juneteenth National Independence Day", "2022-06-19", []),
        (
            "Juneteenth National Independence Day",
            "2022-06-20",
            [xpd.Timestamp("2022-06-20")],
        ),
    ],
)
def test_holidays_within_dates(holiday, start, expected):
    if isinstance(holiday, str):
        calendar = get_calendar("USFederalHolidayCalendar")
        holiday = calendar.rule_from_name(holiday)

    assert list(holiday.dates(start, start)) == expected

    # Verify that timezone info is preserved.
    assert list(
        holiday.dates(
            xpd.Timestamp(start, tz="UTC"),
            xpd.Timestamp(start, tz="UTC"),
        )
    ) == [dt.tz_localize("UTC") for dt in expected]


@pytest.mark.serial
@pytest.mark.parametrize(
    "env_value",
    ["", "cuda", "pool", "async", "managed", "managed_pool", "abc"],
)
def test_rmm_option_on_import(env_value):
    data_directory = os.path.dirname(os.path.abspath(__file__))
    # Create a copy of the current environment variables
    env = os.environ.copy()
    env["CUDF_PANDAS_RMM_MODE"] = env_value

    sp_completed = subprocess.run(
        [
            "python",
            "-m",
            "cudf.pandas",
            data_directory + "/data/profile_basic.py",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    if env_value in {"cuda", "pool", "async", "managed", "managed_pool"}:
        assert sp_completed.returncode == 0
    else:
        assert sp_completed.returncode == 1


def test_cudf_pandas_debugging_different_results(monkeypatch):
    cudf_mean = cudf.Series.mean

    def mock_mean_one(self, *args, **kwargs):
        return np.float64(1.0)

    with monkeypatch.context() as monkeycontext:
        monkeypatch.setattr(xpd.Series.mean, "_fsproxy_fast", mock_mean_one)
        monkeycontext.setenv("CUDF_PANDAS_DEBUGGING", "True")
        s = xpd.Series([1, 2])
        with pytest.warns(
            UserWarning,
            match="The results from cudf and pandas were different.",
        ):
            assert s.mean() == 1.0
    # Must explicitly undo the patch. Proxy dispatch doesn't work with monkeypatch contexts.
    monkeypatch.setattr(xpd.Series.mean, "_fsproxy_fast", cudf_mean)


def test_cudf_pandas_debugging_pandas_error(monkeypatch):
    pd_mean = pd.Series.mean

    def mock_mean_exception(self, *args, **kwargs):
        raise Exception()

    with monkeypatch.context() as monkeycontext:
        monkeycontext.setattr(
            xpd.Series.mean, "_fsproxy_slow", mock_mean_exception
        )
        monkeycontext.setenv("CUDF_PANDAS_DEBUGGING", "True")
        s = xpd.Series([1, 2])
        with pytest.warns(
            UserWarning,
            match="The result from pandas could not be computed.",
        ):
            s = xpd.Series([1, 2])
            assert s.mean() == 1.5
    # Must explicitly undo the patch. Proxy dispatch doesn't work with monkeypatch contexts.
    monkeypatch.setattr(xpd.Series.mean, "_fsproxy_slow", pd_mean)


def test_cudf_pandas_debugging_failed(monkeypatch):
    pd_mean = pd.Series.mean

    def mock_mean_none(self, *args, **kwargs):
        return None

    with monkeypatch.context() as monkeycontext:
        monkeycontext.setattr(xpd.Series.mean, "_fsproxy_slow", mock_mean_none)
        monkeycontext.setenv("CUDF_PANDAS_DEBUGGING", "True")
        s = xpd.Series([1, 2])
        with pytest.warns(
            UserWarning,
            match="Pandas debugging mode failed.",
        ):
            s = xpd.Series([1, 2])
            assert s.mean() == 1.5
    # Must explicitly undo the patch. Proxy dispatch doesn't work with monkeypatch contexts.
    monkeypatch.setattr(xpd.Series.mean, "_fsproxy_slow", pd_mean)


def test_excelwriter_pathlike(tmpdir):
    assert isinstance(pd.ExcelWriter(tmpdir.join("foo.xlsx")), os.PathLike)


def test_is_proxy_object():
    np_arr = np.array([1])

    s1 = xpd.Series([1])
    s2 = pd.Series([1])

    np_arr_proxy = s1.to_numpy()

    assert not is_proxy_object(np_arr)
    assert is_proxy_object(np_arr_proxy)
    assert is_proxy_object(s1)
    assert not is_proxy_object(s2)


def test_numpy_cupy_flatiter(series):
    cp = pytest.importorskip("cupy")

    _, s = series
    arr = s.values

    assert type(arr.flat._fsproxy_fast) is cp.flatiter
    assert type(arr.flat._fsproxy_slow) is np.flatiter


def test_arrow_string_arrays():
    cu_s = xpd.Series(["a", "b", "c"])
    pd_s = pd.Series(["a", "b", "c"])

    cu_arr = xpd.arrays.ArrowStringArray._from_sequence(
        cu_s, dtype=xpd.StringDtype(storage="pyarrow")
    )
    pd_arr = pd.arrays.ArrowStringArray._from_sequence(
        pd_s, dtype=pd.StringDtype(storage="pyarrow")
    )

    tm.assert_equal(cu_arr, pd_arr)

    # TODO: Should this use xpd.StringDtype?
    xpd_pa_np_storage_type = pd.StringDtype(storage="pyarrow", na_value=np.nan)

    pd_pa_np_storage_type = pd.StringDtype(storage="pyarrow", na_value=np.nan)

    cu_arr = xpd.core.arrays.string_arrow.ArrowStringArray._from_sequence(
        cu_s, dtype=xpd_pa_np_storage_type
    )
    pd_arr = pd.core.arrays.string_arrow.ArrowStringArray._from_sequence(
        pd_s, dtype=pd_pa_np_storage_type
    )

    tm.assert_equal(cu_arr, pd_arr)


@pytest.mark.parametrize("indexer", ["at", "iat"])
def test_at_iat(indexer):
    df = xpd.DataFrame(range(3))
    result = getattr(df, indexer)[0, 0]
    assert result == 0

    getattr(df, indexer)[0, 0] = 1
    expected = pd.DataFrame([1, 1, 2])
    tm.assert_frame_equal(df, expected)


def test_at_setitem_empty():
    df = xpd.DataFrame({"name": []}, dtype="float64")
    df.at[0, "name"] = 1.0
    df.at[0, "new"] = 2.0
    expected = pd.DataFrame({"name": [1.0], "new": [2.0]})
    tm.assert_frame_equal(df, expected)


@pytest.mark.parametrize(
    "index",
    [
        xpd.Index([1, 2, 3], name="foo"),
        xpd.Index(["a", "b", "c"], name="foo"),
        xpd.RangeIndex(start=0, stop=3, step=1, name="foo"),
        xpd.CategoricalIndex(["a", "b", "a"], name="foo"),
        xpd.DatetimeIndex(
            ["2024-04-24", "2025-04-24", "2026-04-24"], name="foo"
        ),
        xpd.TimedeltaIndex(["1 days", "2 days", "3 days"], name="foo"),
        xpd.PeriodIndex(
            ["2024-06", "2023-06", "2022-06"], freq="M", name="foo"
        ),
        xpd.IntervalIndex.from_breaks([0, 1, 2, 3], name="foo"),
        xpd.MultiIndex.from_tuples(
            [(1, "a"), (2, "b"), (3, "c")], names=["foo1", "bar1"]
        ),
    ],
)
def test_change_index_name(index):
    s = xpd.Series([1, 2, object()], index=index)
    df = xpd.DataFrame({"values": [1, 2, object()]}, index=index)

    if isinstance(index, xpd.MultiIndex):
        names = ["foo2", "bar2"]
        s.index.names = names
        df.index.names = names

        assert s.index.names == names
        assert df.index.names == names
    else:
        name = "bar"
        s.index.name = name
        df.index.name = name

        assert s.index.name == name
        assert df.index.name == name


@pytest.mark.flaky(reruns=5, delay=4)
def test_notebook_slow_repr():
    notebook_filename = (
        os.path.dirname(os.path.abspath(__file__))
        + "/data/repr_slow_down_test.ipynb"
    )
    with open(notebook_filename, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(
        timeout=30, kernel_name=jupyter_client.KernelManager().kernel_name
    )

    try:
        ep.preprocess(nb, {"metadata": {"path": "./"}})
    except Exception as e:
        assert False, f"Error executing the notebook: {e}"

    # Collect the outputs
    html_result = nb.cells[2]["outputs"][0]["data"]["text/html"]
    for string in {
        "div",
        "Column_1",
        "Column_2",
        "Column_3",
        "Column_4",
        "tbody",
        "</table>",
    }:
        assert string in html_result, (
            f"Expected string {string} not found in the output"
        )


def test_numpy_ndarray_isinstancecheck(array):
    arr1, arr2 = array
    assert isinstance(arr1, np.ndarray)
    assert isinstance(arr2, np.ndarray)


def test_numpy_ndarray_np_ufunc(array):
    arr1, arr2 = array

    @np.vectorize
    def add_one_ufunc(arr):
        return arr + 1

    assert_eq(add_one_ufunc(arr1), add_one_ufunc(arr2))


def test_numpy_ndarray_cp_ufunc(array):
    arr1, arr2 = array

    @cp.vectorize
    def add_one_ufunc(arr):
        return arr + 1

    assert_eq(add_one_ufunc(cp.asarray(arr1)), add_one_ufunc(arr2))


def test_numpy_ndarray_numba_ufunc(array):
    arr1, arr2 = array

    @vectorize
    def add_one_ufunc(arr):
        return arr + 1

    assert_eq(add_one_ufunc(arr1), add_one_ufunc(arr2))


@pytest.mark.filterwarnings(
    "ignore:Grid size:numba.core.errors.NumbaPerformanceWarning"
)
@pytest.mark.filterwarnings(
    "ignore:Grid size:numba.cuda.core.errors.NumbaPerformanceWarning"
)
def test_numpy_ndarray_numba_cuda_ufunc(array):
    arr1, arr2 = array

    @vectorize(["int64(int64)"], target="cuda")
    def add_one_ufunc(a):
        return a + 1

    assert_eq(cp.asarray(add_one_ufunc(arr1)), cp.asarray(add_one_ufunc(arr2)))


@pytest.mark.xfail(
    reason="Fallback expected because casting to object is not supported",
)
def test_fallback_raises_error(monkeypatch):
    with monkeypatch.context() as monkeycontext:
        monkeycontext.setenv("CUDF_PANDAS_FAIL_ON_FALLBACK", "True")
        with pytest.raises(FallbackError):
            pd.Series(range(2)).astype(object)


def mock_mean_memory_error(self, *args, **kwargs):
    raise MemoryError()


def mock_mean_rmm_error(self, *args, **kwargs):
    raise RMMError(1, "error")


def mock_mean_not_impl_error(self, *args, **kwargs):
    raise NotImplementedError()


def mock_mean_attr_error(self, *args, **kwargs):
    raise AttributeError()


def mock_mean_type_error(self, *args, **kwargs):
    raise TypeError()


@pytest.mark.parametrize(
    "mock_mean, err",
    [
        (
            mock_mean_memory_error,
            OOMFallbackError,
        ),
        (
            mock_mean_rmm_error,
            OOMFallbackError,
        ),
        (
            mock_mean_not_impl_error,
            NotImplementedFallbackError,
        ),
        (
            mock_mean_attr_error,
            AttributeFallbackError,
        ),
        (
            mock_mean_type_error,
            TypeFallbackError,
        ),
    ],
)
def test_fallback_raises_specific_error(
    monkeypatch,
    mock_mean,
    err,
):
    with monkeypatch.context() as monkeycontext:
        monkeypatch.setattr(xpd.Series.mean, "_fsproxy_fast", mock_mean)
        monkeycontext.setenv("CUDF_PANDAS_FAIL_ON_FALLBACK", "True")
        s = xpd.Series([1, 2])
        with pytest.raises(err, match="Falling back to the slow path"):
            assert s.mean() == 1.5

    # Must explicitly undo the patch. Proxy dispatch doesn't work with monkeypatch contexts.
    monkeypatch.setattr(xpd.Series.mean, "_fsproxy_fast", cudf.Series.mean)


@pytest.mark.parametrize(
    "attrs",
    [
        "_exceptions",
        "version",
        "_print_versions",
        "_validators",
        "_decorators",
    ],
)
def test_cudf_pandas_util_version(attrs):
    assert hasattr(pd.util, attrs)


def test_iteration_over_dataframe_dtypes_produces_proxy_objects(dataframe):
    _, xdf = dataframe
    xdf["b"] = xpd.IntervalIndex.from_arrays(xdf["a"], xdf["b"])
    xdf["a"] = xpd.Series([1, 1, 1, 2, 3], dtype="category")
    dtype_series = xdf.dtypes
    assert all(is_proxy_object(x) for x in dtype_series)
    assert isinstance(dtype_series.iloc[0], xpd.CategoricalDtype)
    assert isinstance(dtype_series.iloc[1], xpd.IntervalDtype)


def test_iter_doesnot_raise(monkeypatch):
    s = xpd.Series([1, 2, 3])
    with monkeypatch.context() as monkeycontext:
        monkeycontext.setenv("CUDF_PANDAS_FAIL_ON_FALLBACK", "True")
        for _ in s:
            pass


def test_dataframe_setitem_slowdown():
    # We are explicitly testing the slowdown of the setitem operation
    df = xpd.DataFrame(
        {"a": [1, 2, 3] * 100000, "b": [1, 2, 3] * 100000}
    ).astype("float64")
    df = xpd.DataFrame({"a": df["a"].repeat(1000), "b": df["b"].repeat(1000)})
    new_df = df + 1
    start_time = time.time()
    df[df.columns] = new_df
    end_time = time.time()
    delta = int(end_time - start_time)
    if delta > 5:
        pytest.fail(f"Test took too long to run, runtime: {delta}")


def test_dataframe_setitem():
    df = xpd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]}).astype("float64")
    new_df = df + 1
    df[df.columns] = new_df
    tm.assert_equal(df, new_df)


def test_dataframe_get_fast_slow_methods():
    df = xpd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    assert isinstance(df.as_gpu_object(), cudf.DataFrame)
    assert isinstance(df.as_cpu_object(), pd.DataFrame)


def test_is_cudf_pandas():
    s = xpd.Series([1, 2, 3])
    df = xpd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    index = xpd.Index([1, 2, 3])

    assert is_proxy_instance(s, pd.Series)
    assert is_proxy_instance(df, pd.DataFrame)
    assert is_proxy_instance(index, pd.Index)
    assert is_proxy_instance(index.values, np.ndarray)

    for obj in [s, df, index, index.values]:
        assert not is_proxy_instance(obj._fsproxy_slow, pd.Series)
        assert not is_proxy_instance(obj._fsproxy_fast, pd.Series)

        assert not is_proxy_instance(obj._fsproxy_slow, pd.DataFrame)
        assert not is_proxy_instance(obj._fsproxy_fast, pd.DataFrame)

        assert not is_proxy_instance(obj._fsproxy_slow, pd.Index)
        assert not is_proxy_instance(obj._fsproxy_fast, pd.Index)

        assert not is_proxy_instance(obj._fsproxy_slow, np.ndarray)
        assert not is_proxy_instance(obj._fsproxy_fast, np.ndarray)


def test_series_dtype_property():
    s = pd.Series([1, 2, 3])
    xs = xpd.Series([1, 2, 3])
    expected = np.dtype(s)
    actual = np.dtype(xs)
    assert expected == actual


def assert_functions_called(profiler, functions):
    # Process profiling data
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)

    # Get all called functions as (filename, lineno, func_name)
    called_functions = {func[2] for func in stats.stats.keys()}
    for func_str in functions:
        assert func_str in called_functions


def test_cudf_series_from_cudf_pandas():
    s = xpd.Series([1, 2, 3])

    with cProfile.Profile() as profiler:
        gs = cudf.Series(s)

    assert_functions_called(
        profiler, ["as_gpu_object", "<method 'update' of 'dict' objects>"]
    )

    tm.assert_equal(s.as_gpu_object(), gs)


def test_cudf_dataframe_from_cudf_pandas():
    df = xpd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})

    with cProfile.Profile() as profiler:
        gdf = cudf.DataFrame(df)

    assert_functions_called(
        profiler, ["as_gpu_object", "<method 'update' of 'dict' objects>"]
    )
    tm.assert_frame_equal(df.as_gpu_object(), gdf)

    df = xpd.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    gdf = cudf.DataFrame(
        {"a": xpd.Series([1, 2, 3]), "b": xpd.Series([1, 2, 3])}
    )

    tm.assert_frame_equal(df.as_gpu_object(), gdf)

    df = xpd.DataFrame({0: [1, 2, 3], 1: [1, 2, 3]})
    gdf = cudf.DataFrame(
        [xpd.Series([1, 1]), xpd.Series([2, 2]), xpd.Series([3, 3])]
    )

    tm.assert_frame_equal(df.as_gpu_object(), gdf)


def test_cudf_index_from_cudf_pandas():
    idx = xpd.Index([1, 2, 3])
    with cProfile.Profile() as profiler:
        gidx = cudf.Index(idx)
    assert_functions_called(profiler, ["as_gpu_object"])

    tm.assert_equal(idx.as_gpu_object(), gidx)


def test_numpy_data_access():
    s = pd.Series([1, 2, 3])
    xs = xpd.Series([1, 2, 3])
    expected = s.values.data
    actual = xs.values.data

    assert type(expected) is type(actual)


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame({"a": [1, 2, 3]}),
        pd.Series([1, 2, 3]),
        pd.Index([1, 2, 3]),
        pd.Categorical([1, 2, 3]),
        pd.to_datetime(["2021-01-01", "2021-01-02"]),
        pd.to_timedelta(["1 days", "2 days"]),
        xpd.DataFrame({"a": [1, 2, 3]}),
        xpd.Series([1, 2, 3]),
        xpd.Index([1, 2, 3]),
        xpd.Categorical([1, 2, 3]),
        xpd.to_datetime(["2021-01-01", "2021-01-02"]),
        xpd.to_timedelta(["1 days", "2 days"]),
        cudf.DataFrame({"a": [1, 2, 3]}),
        cudf.Series([1, 2, 3]),
        cudf.Index([1, 2, 3]),
        cudf.Index([1, 2, 3], dtype="category"),
        cudf.to_datetime(["2021-01-01", "2021-01-02"]),
        cudf.Index([1, 2, 3], dtype="timedelta64[ns]"),
        [1, 2, 3],
        {"a": 1, "b": 2},
        (1, 2, 3),
    ],
)
def test_as_proxy_object(obj):
    proxy_obj = as_proxy_object(obj)
    if isinstance(
        obj,
        (
            pd.DataFrame,
            pd.Series,
            pd.Index,
            pd.Categorical,
            xpd.DataFrame,
            xpd.Series,
            xpd.Index,
            xpd.Categorical,
            cudf.DataFrame,
            cudf.Series,
            cudf.Index,
        ),
    ):
        assert is_proxy_object(proxy_obj)
        if isinstance(proxy_obj, xpd.DataFrame):
            tm.assert_frame_equal(proxy_obj, xpd.DataFrame(obj))
        elif isinstance(proxy_obj, xpd.Series):
            tm.assert_series_equal(proxy_obj, xpd.Series(obj))
        elif isinstance(proxy_obj, xpd.Index):
            tm.assert_index_equal(proxy_obj, xpd.Index(obj))
        else:
            tm.assert_equal(proxy_obj, obj)
    else:
        assert not is_proxy_object(proxy_obj)
        assert proxy_obj == obj


def test_as_proxy_object_doesnot_copy_series():
    s = pd.Series([1, 2, 3])
    proxy_obj = as_proxy_object(s)
    s[0] = 10
    assert proxy_obj[0] == 10
    tm.assert_series_equal(s, proxy_obj)


def test_as_proxy_object_doesnot_copy_dataframe():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    proxy_obj = as_proxy_object(df)
    df.iloc[0, 0] = 10
    assert proxy_obj.iloc[0, 0] == 10
    tm.assert_frame_equal(df, proxy_obj)


def test_as_proxy_object_doesnot_copy_index():
    idx = pd.Index([1, 2, 3])
    proxy_obj = as_proxy_object(idx)
    assert proxy_obj._fsproxy_wrapped is idx


def test_as_proxy_object_no_op_for_intermediates():
    s = pd.Series(["abc", "def", "ghi"])
    str_attr = s.str
    proxy_obj = as_proxy_object(str_attr)
    assert proxy_obj is str_attr


def test_pickle_round_trip_proxy_numpy_array(array):
    arr, proxy_arr = array
    pickled_arr = BytesIO()
    pickled_proxy_arr = BytesIO()
    pickle.dump(arr, pickled_arr)
    pickle.dump(proxy_arr, pickled_proxy_arr)

    pickled_arr.seek(0)
    pickled_proxy_arr.seek(0)

    np.testing.assert_equal(
        pickle.load(pickled_proxy_arr), pickle.load(pickled_arr)
    )


def test_pandas_objects_not_callable():
    series = xpd.Series([1, 2, 3])
    dataframe = xpd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    index = xpd.Index([1, 2, 3])
    range_index = xpd.RangeIndex(start=0, stop=10, step=1)
    assert not isinstance(series, Callable)
    assert not isinstance(dataframe, Callable)
    assert not isinstance(index, Callable)
    assert not isinstance(range_index, Callable)

    assert isinstance(xpd.Series, Callable)
    assert isinstance(xpd.DataFrame, Callable)
    assert isinstance(xpd.Index, Callable)
    assert isinstance(xpd.RangeIndex, Callable)


def test_memory_usage():
    s = xpd.Series(range(10), index=[f"i-{i}" for i in range(10)], name="a")

    res_deep = s.memory_usage(deep=True)
    res = sys.getsizeof(s)

    assert abs(res_deep - res) < 100


def test_module_proxy_write_through_config(monkeypatch):
    cf = xpd._config.config
    cf.register_option("foo", 1)
    monkeypatch.setattr(cf, "_registered_options", {})
    cf.register_option("foo", 1)


def test_class_monkeypatch_roundtrip_restores_real_pandas(monkeypatch):
    # Class-level patches on proxy types are mirrored onto the real pandas
    # type (so they stay visible to fallback code running under
    # ``disable_module_accelerator``); undoing them must restore the real
    # type's own attributes — including for attributes that cudf.pandas
    # replaces on the proxy, like ``columns``/``eval``/``str``.
    real_df = xpd.DataFrame._fsproxy_slow
    real_series = xpd.Series._fsproxy_slow
    orig_columns = real_df.__dict__["columns"]
    orig_eval = real_df.__dict__["eval"]
    orig_str = real_series.__dict__["str"]

    def fake_eval(self, *args, **kwargs):
        return "patched"

    monkeypatch.setattr(xpd.DataFrame, "eval", fake_eval)
    assert real_df.__dict__["eval"] is fake_eval
    monkeypatch.setattr(
        xpd.DataFrame, "columns", property(lambda self: "patched")
    )
    monkeypatch.setattr(xpd.Series, "str", property(lambda self: "patched"))
    monkeypatch.undo()

    assert real_df.__dict__["columns"] is orig_columns
    assert real_df.__dict__["eval"] is orig_eval
    assert real_series.__dict__["str"] is orig_str
    # The real type must remain fully functional on the fallback path.
    df = real_df({"a": [1, 2]})
    assert list(df.columns) == ["a"]
    assert list(df.eval("b = a + 1").columns) == ["a", "b"]


@pytest.mark.parametrize("box", ["Series", "array"])
@pytest.mark.parametrize("na_value", [pd.NA, np.nan], ids=["NA", "NaN"])
@pytest.mark.parametrize("storage", ["python", "pyarrow"])
def test_numpy_random_permutation_string(storage, na_value, box):
    # https://github.com/pandas-dev/pandas/issues/63935
    # Under copy-on-write, ``np.asarray`` on a python-storage string
    # Series returns a read-only view; ``Generator.permutation`` relies
    # on ``np.may_share_memory(np.asarray(x), x)`` to decide whether to
    # copy before shuffling in place, which requires ``_transform_arg``
    # to preserve the identity of object-dtype ndarrays that need no
    # transformation (otherwise this raises "ValueError: array is
    # read-only").
    data = ["a", "bb", "ccc"]
    obj = getattr(xpd, box)(
        data, dtype=xpd.StringDtype(storage=storage, na_value=na_value)
    )
    pandas_obj = getattr(pd, box)(
        data, dtype=pd.StringDtype(storage=storage, na_value=na_value)
    )

    result = np.random.default_rng(seed=2).permutation(obj)
    expected = np.random.default_rng(seed=2).permutation(pandas_obj)
    assert isinstance(result, np.ndarray)
    assert result.tolist() == expected.tolist()

    # The original object is left untouched by the shuffle.
    tm.assert_equal(
        obj,
        getattr(xpd, box)(
            data, dtype=xpd.StringDtype(storage=storage, na_value=na_value)
        ),
    )

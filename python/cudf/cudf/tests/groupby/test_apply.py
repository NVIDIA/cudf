# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import textwrap
from functools import partial

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf import DataFrame
from cudf.core.udf._ops import arith_ops, comparison_ops, unary_ops
from cudf.core.udf.groupby_typing import SUPPORTED_GROUPBY_NUMPY_TYPES
from cudf.core.udf.utils import UDFError, precompiled
from cudf.testing import assert_eq, assert_groupby_results_equal
from cudf.testing._utils import expect_warning_if


@pytest.fixture(params=["cudf", "jit"])
def engine(request):
    return request.param


def test_groupby_as_index_apply(as_index, engine):
    pdf = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = cudf.DataFrame({"x": [1, 2, 3], "y": [0, 1, 1]})
    gdf = gdf.groupby("y", as_index=as_index).apply(
        lambda df: df["x"].mean(), engine=engine
    )
    pdf = pdf.groupby("y", as_index=as_index).apply(
        func=lambda df: df["x"].mean(), include_groups=False
    )
    if not as_index:
        # pandas uses object type for the resulting ["y", None] column labels,
        # cuDF uses StringDtype instead
        pdf.columns = pdf.columns.astype(gdf.columns.dtype)
    assert_groupby_results_equal(pdf, gdf, as_index=as_index, by="y")


def test_groupby_apply():
    rng = np.random.default_rng(seed=0)
    nelem = 20
    df = cudf.DataFrame(
        {
            "key1": rng.integers(0, 3, nelem),
            "key2": rng.integers(0, 2, nelem),
            "val1": rng.random(nelem),
            "val2": rng.random(nelem),
        }
    )

    expect_grpby = df.to_pandas().groupby(
        ["key1", "key2"], as_index=False, group_keys=False
    )
    got_grpby = df.groupby(["key1", "key2"])

    def foo(df):
        df["out"] = df["val1"] + df["val2"]
        return df

    expect = expect_grpby.apply(foo, include_groups=False)
    got = got_grpby.apply(foo, include_groups=False)
    assert_groupby_results_equal(expect, got)


def f1(df, k):
    df["out"] = df["val1"] + df["val2"] + k
    return df


def f2(df, k, L):
    df["out"] = df["val1"] - df["val2"] + (k / L)
    return df


def f3(df, k, L, m):
    df["out"] = ((k * df["val1"]) + (L * df["val2"])) / m
    return df


@pytest.mark.parametrize(
    "func,args", [(f1, (42,)), (f2, (42, 119)), (f3, (42, 119, 212.1))]
)
def test_groupby_apply_args(func, args):
    rng = np.random.default_rng(seed=0)
    nelem = 20
    df = cudf.DataFrame(
        {
            "key1": rng.integers(0, 3, nelem),
            "key2": rng.integers(0, 2, nelem),
            "val1": rng.random(nelem),
            "val2": rng.random(nelem),
        }
    )

    expect_grpby = df.to_pandas().groupby(
        ["key1", "key2"], as_index=False, group_keys=False
    )
    got_grpby = df.groupby(["key1", "key2"])
    expect = expect_grpby.apply(func, *args, include_groups=False)
    got = got_grpby.apply(func, *args, include_groups=False)
    assert_groupby_results_equal(expect, got)


@pytest.fixture
def groupby_jit_data_small():
    """
    Return a small dataset for testing JIT Groupby Apply. The dataframe
    contains 4 groups of size 1, 2, 3, 4 as well as an additional key
    column that can be used to test subgroups within groups. This data
    is useful for smoke testing basic numeric results
    """
    rng = np.random.default_rng(42)
    df = DataFrame(
        {
            "key1": [1] + [2] * 2 + [3] * 3 + [4] * 4,
            "key2": [1, 2] * 5,
            "val1": rng.integers(0, 10, 10),
            "val2": rng.integers(0, 10, 10),
        }
    )
    # randomly permute data
    df = df.sample(frac=1, random_state=1, ignore_index=True)
    return df


@pytest.fixture
def groupby_jit_data_large(groupby_jit_data_small):
    """
    Larger version of groupby_jit_data_small which contains enough data
    to require more than one block per group. This data is useful for
    testing if JIT GroupBy algorithms scale to larger dastasets without
    manifesting numerical issues such as overflow.
    """
    max_tpb = 1024
    factor = (
        max_tpb + 1
    )  # bigger than a block but not always an exact multiple
    df = cudf.concat([groupby_jit_data_small] * factor)

    return df


@pytest.fixture
def groupby_jit_data_nans(groupby_jit_data_small):
    """
    Returns a modified version of groupby_jit_data_small which contains
    nan values.
    """

    df = groupby_jit_data_small.sort_values(["key1", "key2"])
    df["val1"] = df["val1"].astype("float64")
    df.loc[df.index[::2], "val1"] = np.nan
    df = df.sample(frac=1, random_state=1, ignore_index=True)
    return df


@pytest.fixture
def groupby_jit_datasets(
    groupby_jit_data_small, groupby_jit_data_large, groupby_jit_data_nans
):
    return {
        "small": groupby_jit_data_small,
        "large": groupby_jit_data_large,
        "nans": groupby_jit_data_nans,
    }


def run_groupby_apply_jit_test(data, func, keys, *args):
    expect_groupby_obj = data.to_pandas().groupby(keys)
    got_groupby_obj = data.groupby(keys)

    # compare cuDF jit to pandas
    cudf_jit_result = got_groupby_obj.apply(
        func, *args, engine="jit", include_groups=False
    )
    pandas_result = expect_groupby_obj.apply(func, *args, include_groups=False)
    assert_groupby_results_equal(cudf_jit_result, pandas_result)


def groupby_apply_jit_reductions_test_inner(func, data, dtype):
    # ideally we'd just have:
    # lambda group: getattr(group, func)()
    # but the current kernel caching mechanism relies on pickle which
    # does not play nice with local functions. What's below uses
    # exec as a workaround to write the test functions dynamically

    funcstr = textwrap.dedent(
        f"""
        def func(df):
            return df['val1'].{func}()
        """
    )
    lcl = {}
    exec(funcstr, lcl)
    func = lcl["func"]

    data["val1"] = data["val1"].astype(dtype)
    data["val2"] = data["val2"].astype(dtype)

    run_groupby_apply_jit_test(data, func, ["key1"])


# test unary reductions
@pytest.mark.parametrize(
    "dtype",
    SUPPORTED_GROUPBY_NUMPY_TYPES,
    ids=[str(t) for t in SUPPORTED_GROUPBY_NUMPY_TYPES],
)
@pytest.mark.parametrize(
    "func", ["min", "max", "sum", "mean", "var", "std", "idxmin", "idxmax"]
)
@pytest.mark.parametrize("dataset", ["small", "large", "nans"])
def test_groupby_apply_jit_unary_reductions(
    request, func, dtype, dataset, groupby_jit_datasets
):
    request.applymarker(
        pytest.mark.xfail(
            condition=(
                (
                    dataset == "nans"
                    and func in {"var", "std", "mean"}
                    and str(dtype) in {"int64", "float32", "float64"}
                )
                or (
                    dataset == "nans"
                    and func in {"idxmax", "idxmin", "sum"}
                    and dtype.kind == "f"
                )
            ),
            reason=("https://github.com/rapidsai/cudf/issues/14860"),
        )
    )
    dataset = groupby_jit_datasets[dataset].copy(deep=True)
    groupby_apply_jit_reductions_test_inner(func, dataset, dtype)


# test unary reductions for special values
def groupby_apply_jit_reductions_special_vals_inner(
    func, data, dtype, special_val
):
    funcstr = textwrap.dedent(
        f"""
        def func(df):
            return df['val1'].{func}()
        """
    )
    lcl = {}
    exec(funcstr, lcl)
    func = lcl["func"]

    data["val1"] = data["val1"].astype(dtype)
    data["val2"] = data["val2"].astype(dtype)
    data["val1"] = special_val
    data["val2"] = special_val

    run_groupby_apply_jit_test(data, func, ["key1"])


# test unary index reductions for special values
def groupby_apply_jit_idx_reductions_special_vals_inner(
    func, data, dtype, special_val
):
    funcstr = textwrap.dedent(
        f"""
        def func(df):
            return df['val1'].{func}()
        """
    )
    lcl = {}
    exec(funcstr, lcl)
    func = lcl["func"]

    data["val1"] = data["val1"].astype(dtype)
    data["val2"] = data["val2"].astype(dtype)
    data["val1"] = special_val
    data["val2"] = special_val

    run_groupby_apply_jit_test(data, func, ["key1"])


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("func", ["min", "max", "sum", "mean", "var", "std"])
@pytest.mark.parametrize("special_val", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("dataset", ["small", "large", "nans"])
def test_groupby_apply_jit_reductions_special_vals(
    func, dtype, dataset, groupby_jit_datasets, special_val
):
    dataset = groupby_jit_datasets[dataset].copy(deep=True)
    with expect_warning_if(
        func in {"var", "std"} and not np.isnan(special_val), RuntimeWarning
    ):
        groupby_apply_jit_reductions_special_vals_inner(
            func, dataset, dtype, special_val
        )


@pytest.mark.parametrize("func", ["idxmax", "idxmin"])
@pytest.mark.parametrize(
    "special_val",
    [
        pytest.param(
            np.nan,
            marks=pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/13832"
            ),
        ),
        np.inf,
        -np.inf,
    ],
)
@pytest.mark.parametrize("dataset", ["small", "large", "nans"])
def test_groupby_apply_jit_idx_reductions_special_vals(
    func, dataset, groupby_jit_datasets, special_val
):
    dataset = groupby_jit_datasets[dataset].copy(deep=True)
    groupby_apply_jit_idx_reductions_special_vals_inner(
        func, dataset, "float64", special_val
    )


def test_groupby_apply_jit_sum_integer_overflow():
    max = np.iinfo("int32").max

    data = DataFrame(
        {
            "a": [0, 0, 0],
            "b": [max, max, max],
        }
    )

    def func(group):
        return group["b"].sum()

    run_groupby_apply_jit_test(data, func, ["a"])


@pytest.mark.parametrize("dtype", ["int32", "int64", "float32", "float64"])
@pytest.mark.parametrize(
    "dataset",
    [
        pytest.param(
            "small",
            marks=[
                pytest.mark.filterwarnings(
                    "ignore:Degrees of Freedom <= 0 for slice"
                ),
                pytest.mark.filterwarnings(
                    "ignore:divide by zero encountered in divide"
                ),
            ],
        ),
        "large",
    ],
)
def test_groupby_apply_jit_correlation(dataset, groupby_jit_datasets, dtype):
    dataset = groupby_jit_datasets[dataset].copy(deep=True)

    dataset["val1"] = dataset["val1"].astype(dtype)
    dataset["val2"] = dataset["val2"].astype(dtype)

    keys = ["key1"]

    def func(group):
        return group["val1"].corr(group["val2"])

    if np.dtype(dtype).kind == "f":
        # Correlation of floating types is not yet supported:
        # https://github.com/rapidsai/cudf/issues/13839
        m = (
            f"Series.corr\\(Series\\) is not "
            f"supported for \\({dtype}, {dtype}\\)"
        )
        with pytest.raises(UDFError, match=m):
            run_groupby_apply_jit_test(dataset, func, keys)
        return
    with expect_warning_if(dtype in {"int32", "int64"}, RuntimeWarning):
        run_groupby_apply_jit_test(dataset, func, keys)


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_groupby_apply_jit_correlation_zero_variance(dtype):
    # pearson correlation is undefined when the variance of either
    # variable is zero. This test ensures that the jit implementation
    # returns the same result as pandas in this case.
    data = DataFrame(
        {"a": [0, 0, 0, 0, 0], "b": [1, 1, 1, 1, 1], "c": [2, 2, 2, 2, 2]}
    )

    def func(group):
        return group["b"].corr(group["c"])

    with expect_warning_if(dtype in {"int32", "int64"}, RuntimeWarning):
        run_groupby_apply_jit_test(data, func, ["a"])


@pytest.mark.parametrize("op", unary_ops)
def test_groupby_apply_jit_invalid_unary_ops_error(groupby_jit_data_small, op):
    keys = ["key1"]

    def func(group):
        return op(group["val1"])

    with pytest.raises(
        UDFError,
        match=f"{op.__name__}\\(Series\\) is not supported by JIT GroupBy",
    ):
        run_groupby_apply_jit_test(groupby_jit_data_small, func, keys)


@pytest.mark.parametrize("op", arith_ops + comparison_ops)
def test_groupby_apply_jit_invalid_binary_ops_error(
    groupby_jit_data_small, op
):
    keys = ["key1"]

    def func(group):
        return op(group["val1"], group["val2"])

    with pytest.raises(
        UDFError,
        match=f"{op.__name__}\\(Series, Series\\) is not supported",
    ):
        run_groupby_apply_jit_test(groupby_jit_data_small, func, keys)


def test_groupby_apply_jit_no_df_ops(groupby_jit_data_small):
    # DataFrame level operations are not yet supported.
    def func(group):
        return group.sum()

    with pytest.raises(
        UDFError,
        match=r"JIT GroupBy.apply\(\) does not support DataFrame.sum\(\)",
    ):
        run_groupby_apply_jit_test(groupby_jit_data_small, func, ["key1"])


@pytest.mark.parametrize("dtype", ["uint8", "str"])
def test_groupby_apply_unsupported_dtype(dtype):
    df = cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
    df["b"] = df["b"].astype(dtype)

    # a UDAF that doesn't actually use the input column
    # with the unsupported dtype should still succeed
    def func(group):
        return group["c"].sum()

    run_groupby_apply_jit_test(df, func, ["a"])

    # however a UDAF that does use the unsupported dtype
    # should fail
    def func(group):
        return group["b"].sum()

    with pytest.raises(UDFError, match="Only columns of the following dtypes"):
        run_groupby_apply_jit_test(df, func, ["a"])


@pytest.mark.parametrize(
    "func",
    [
        lambda df: df["val1"].max() + df["val2"].min(),
        lambda df: df["val1"].sum() + df["val2"].var(),
        lambda df: df["val1"].mean() + df["val2"].std(),
    ],
)
def test_groupby_apply_jit_basic(func, groupby_jit_data_small):
    run_groupby_apply_jit_test(groupby_jit_data_small, func, ["key1", "key2"])


def f1(df, k):
    return df["val1"].max() + df["val2"].min() + k


def f2(df, k, L):
    return df["val1"].sum() - df["val2"].var() + (k / L)


def f3(df, k, L, m):
    return ((k * df["val1"].mean()) + (L * df["val2"].std())) / m


@pytest.mark.parametrize(
    "func,args", [(f1, (42,)), (f2, (42, 119)), (f3, (42, 119, 212.1))]
)
def test_groupby_apply_jit_args(func, args, groupby_jit_data_small):
    run_groupby_apply_jit_test(
        groupby_jit_data_small, func, ["key1", "key2"], *args
    )


def test_groupby_apply_jit_block_divergence():
    # https://github.com/rapidsai/cudf/issues/12686
    df = cudf.DataFrame(
        {
            "a": [0, 0, 0, 1, 1, 1],
            "b": [1, 1, 1, 2, 3, 4],
        }
    )

    def diverging_block(grp_df):
        if grp_df["b"].mean() > 1:
            return grp_df["b"].mean()
        return 0

    run_groupby_apply_jit_test(df, diverging_block, ["a"])


def test_groupby_apply_caching():
    # Make sure similar functions that differ
    # by simple things like constants actually
    # recompile

    # begin with a clear cache
    precompiled.clear()
    assert precompiled.currsize == 0

    data = cudf.DataFrame({"a": [1, 1, 1, 2, 2, 2], "b": [1, 2, 3, 4, 5, 6]})

    def f(group):
        return group["b"].mean() * 2

    # a single run should result in a cache size of 1
    run_groupby_apply_jit_test(data, f, ["a"])
    assert precompiled.currsize == 1

    # a second run with f should not increase the count
    run_groupby_apply_jit_test(data, f, ["a"])
    assert precompiled.currsize == 1

    # changing a constant value inside the UDF should miss
    def f(group):
        return group["b"].mean() * 3

    run_groupby_apply_jit_test(data, f, ["a"])
    assert precompiled.currsize == 2

    # changing the dtypes of the columns should miss
    data["b"] = data["b"].astype("float64")
    run_groupby_apply_jit_test(data, f, ["a"])

    assert precompiled.currsize == 3


def test_groupby_apply_no_bytecode_fallback():
    # tests that a function which contains no bytecode
    # attribute, but would still be executable using
    # the iterative groupby apply approach, still works.

    gdf = cudf.DataFrame({"a": [0, 1, 1], "b": [1, 2, 3]})
    pdf = gdf.to_pandas()

    def f(group):
        return group.sum()

    part = partial(f)

    expect = pdf.groupby("a").apply(part, include_groups=False)
    got = gdf.groupby("a").apply(part, engine="auto", include_groups=False)
    assert_groupby_results_equal(expect, got)


def test_groupby_apply_return_col_from_df():
    # tests a UDF that consists of purely colwise
    # ops, such as `lambda group: group.x + group.y`
    # which returns a column
    df = cudf.DataFrame(
        {
            "id": range(10),
            "x": range(10),
            "y": range(10),
        }
    )
    pdf = df.to_pandas()

    def func(df):
        return df.x + df.y

    got = df.groupby("id").apply(func, include_groups=False)
    expect = pdf.groupby("id").apply(func, include_groups=False)
    assert_groupby_results_equal(expect, got)


@pytest.mark.parametrize("func", [lambda group: group.sum()])
def test_groupby_apply_return_df(func):
    # tests a UDF that reduces over a dataframe
    # and produces a series with the original column names
    # as its index, such as lambda group: group.sum() + group.min()
    df = cudf.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    pdf = df.to_pandas()

    expect = pdf.groupby("a").apply(func, include_groups=False)
    got = df.groupby("a").apply(func, include_groups=False)
    assert_groupby_results_equal(expect, got)


@pytest.mark.parametrize("as_index", [True, False])
def test_groupby_apply_return_reindexed_series(as_index):
    def gdf_func(df):
        return cudf.Series([df["a"].sum(), df["b"].min(), df["c"].max()])

    def pdf_func(df):
        return pd.Series([df["a"].sum(), df["b"].min(), df["c"].max()])

    df = cudf.DataFrame(
        {
            "key": [0, 0, 1, 1, 2, 2],
            "a": [1, 2, 3, 4, 5, 6],
            "b": [7, 8, 9, 10, 11, 12],
            "c": [13, 14, 15, 16, 17, 18],
        }
    )
    pdf = df.to_pandas()

    expect = pdf.groupby("key", as_index=as_index).apply(
        pdf_func, include_groups=False
    )
    got = df.groupby("key", as_index=as_index).apply(
        gdf_func, include_groups=False
    )
    assert_groupby_results_equal(expect, got)


def test_groupby_apply_noempty_group():
    pdf = pd.DataFrame(
        {"a": [1, 1, 2, 2], "b": [1, 2, 1, 2], "c": [1, 2, 3, 4]}
    )
    gdf = cudf.from_pandas(pdf)

    expect = (
        pdf.groupby("a", group_keys=False)
        .apply(lambda x: x.iloc[[0, 1]], include_groups=False)
        .reset_index(drop=True)
    )
    got = (
        gdf.groupby("a")
        .apply(lambda x: x.iloc[[0, 1]], include_groups=False)
        .reset_index(drop=True)
    )
    assert_groupby_results_equal(expect, got)


def create_test_groupby_apply_return_scalars_params():
    def f0(x):
        x = x[~x["B"].isna()]
        ticker = x.shape[0]
        full = ticker / 10
        return full

    def f1(x, k):
        x = x[~x["B"].isna()]
        ticker = x.shape[0]
        full = ticker / k
        return full

    def f2(x, k, L):
        x = x[~x["B"].isna()]
        ticker = x.shape[0]
        full = L * (ticker / k)
        return full

    def f3(x, k, L, m):
        x = x[~x["B"].isna()]
        ticker = x.shape[0]
        full = L * (ticker / k) % m
        return full

    return [(f0, ()), (f1, (42,)), (f2, (42, 119)), (f3, (42, 119, 212.1))]


@pytest.mark.parametrize(
    "func,args", create_test_groupby_apply_return_scalars_params()
)
def test_groupby_apply_return_scalars(func, args):
    pdf = pd.DataFrame(
        {
            "A": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "B": [
                0.01,
                np.nan,
                0.03,
                0.04,
                np.nan,
                0.06,
                0.07,
                0.08,
                0.09,
                1.0,
            ],
        }
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby("A").apply(func, *args, include_groups=False)
    actual = gdf.groupby("A").apply(func, *args, include_groups=False)

    assert_groupby_results_equal(expected, actual)


def create_test_groupby_apply_return_series_dataframe_params():
    def f0(x):
        return x - x.max()

    def f1(x):
        return x.min() - x.max()

    def f2(x):
        return x.min()

    def f3(x, k):
        return x - x.max() + k

    def f4(x, k, L):
        return x.min() - x.max() + (k / L)

    def f5(x, k, L, m):
        return m * x.min() + (k / L)

    return [
        (f0, ()),
        (f1, ()),
        (f2, ()),
        (f3, (42,)),
        (f4, (42, 119)),
        (f5, (41, 119, 212.1)),
    ]


@pytest.mark.parametrize(
    "func,args", create_test_groupby_apply_return_series_dataframe_params()
)
def test_groupby_apply_return_series_dataframe(func, args):
    pdf = pd.DataFrame(
        {"key": [0, 0, 1, 1, 2, 2, 2], "val": [0, 1, 2, 3, 4, 5, 6]}
    )
    gdf = cudf.from_pandas(pdf)

    expected = pdf.groupby(["key"], group_keys=False).apply(
        func, *args, include_groups=False
    )
    actual = gdf.groupby(["key"]).apply(func, *args, include_groups=False)

    assert_groupby_results_equal(expected, actual)


def test_groupby_apply_series_results_misaligned_lengths():
    # Series results whose per-group lengths differ from their input are
    # concatenated with the group keys as the outer index level, each key
    # repeated by its chunk's actual length, keeping the UDF-returned
    # index as the inner level (pandas GH8467) -- even when the total
    # output length coincides with len(df) (2 + 3 == 3 + 2 here).
    pdf = pd.DataFrame({"k": [1, 1, 2, 2, 2], "v": [10, 20, 30, 40, 50]})
    gdf = cudf.from_pandas(pdf)

    def make_swap_sizes(series_type):
        # group 1 has 2 rows -> 3 outputs; group 2 has 3 rows -> 2 outputs
        def swap_sizes(g):
            if len(g) == 2:
                return series_type([1, 2, 3])
            return series_type([4, 5])

        return swap_sizes

    expected = pdf.groupby("k").apply(
        make_swap_sizes(pd.Series), include_groups=False
    )
    actual = gdf.groupby("k").apply(
        make_swap_sizes(cudf.Series), include_groups=False
    )
    assert_eq(expected, actual)


def test_groupby_apply_series_results_fresh_index():
    # Per-group result lengths match the input, but the UDF rewrote the
    # index: the UDF-returned index is kept as the inner level rather
    # than the input rows' original labels.
    pdf = pd.DataFrame({"k": [1, 1, 2, 2, 2], "v": [10, 20, 30, 40, 50]})
    gdf = cudf.from_pandas(pdf)

    def make_fresh_index(series_type):
        def fresh_index(g):
            return series_type(range(len(g)))

        return fresh_index

    expected = pdf.groupby("k").apply(
        make_fresh_index(pd.Series), include_groups=False
    )
    actual = gdf.groupby("k").apply(
        make_fresh_index(cudf.Series), include_groups=False
    )
    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "pdf",
    [pd.DataFrame(), pd.DataFrame({"a": []}), pd.Series([], dtype="float64")],
)
def test_groupby_apply_no_keys(pdf):
    gdf = cudf.from_pandas(pdf)
    if isinstance(pdf, pd.DataFrame):
        kwargs = {"check_column_type": False}
    else:
        kwargs = {}
    assert_groupby_results_equal(
        pdf.groupby([], group_keys=False).apply(lambda x: x.max()),
        gdf.groupby([]).apply(lambda x: x.max()),
        check_index_type=False,  # Int64 v/s Float64
        **kwargs,
    )


@pytest.mark.parametrize(
    "data",
    [
        {"Speed": [380.0, 370.0, 24.0, 26.0], "Score": [50, 30, 90, 80]},
        {
            "Speed": [380.0, 370.0, 24.0, 26.0],
            "Score": [50, 30, 90, 80],
            "Other": [10, 20, 30, 40],
        },
    ],
)
@pytest.mark.parametrize("group", ["Score", "Speed"])
def test_groupby_describe(data, group):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)

    got = gdf.groupby(group).describe()
    expect = pdf.groupby(group).describe()

    assert_groupby_results_equal(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        {"a": [], "b": []},
        {"a": [2, 1, 2, 1, 1, 3], "b": [None, 1, 2, None, 2, None]},
        {"a": [None], "b": [None]},
        {"a": [2, 1, 1], "b": [None, 1, 0], "c": [None, 0, 1]},
    ],
)
@pytest.mark.parametrize("agg", ["first", "last", ["first", "last"]])
def test_groupby_first(data, agg):
    pdf = pd.DataFrame(data)
    gdf = cudf.from_pandas(pdf)
    expect = pdf.groupby("a").agg(agg)
    got = gdf.groupby("a").agg(agg)
    if data["a"] == [None]:
        # As of pandas 3.0, empty default type of object isn't
        # necessarily equivalent to cuDF's empty default type of
        # pandas.StringDtype
        expect.index = expect.index.astype(got.index.dtype)
    assert_groupby_results_equal(expect, got, check_dtype=False)


def test_groupby_apply_series():
    def foo(x):
        return x.sum()

    pdf = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "val": np.arange(10, dtype="float64"),
        }
    )
    gdf = cudf.from_pandas(pdf)

    got = gdf.groupby("x").y.apply(foo)
    expect = pdf.groupby("x").y.apply(foo)

    assert_groupby_results_equal(expect, got)


@pytest.mark.parametrize(
    "func,args",
    [
        (lambda x, k: x + k, (42,)),
        (lambda x, k, L: x + k - L, (42, 191)),
        (lambda x, k, L, m: (x + k) / (L * m), (42, 191, 99.9)),
    ],
)
def test_groupby_apply_series_args(func, args):
    pdf = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "y": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "val": np.arange(10, dtype="float64"),
        }
    )
    gdf = cudf.from_pandas(pdf)

    got = gdf.groupby("x").y.apply(func, *args)
    expect = pdf.groupby("x", group_keys=False).y.apply(func, *args)

    assert_groupby_results_equal(expect, got)


def test_groupby_apply_series_preserves_multiindex_names():
    pdf = pd.DataFrame(
        {"value": [1.0, 2.0, 3.0, 4.0]},
        index=pd.MultiIndex.from_product([range(2), range(2)]),
    )
    gdf = cudf.from_pandas(pdf)
    by = np.array([0, 0, 1, 1])

    expect = pdf.groupby(by=by, group_keys=False)["value"].apply(
        lambda x: x.cumprod()
    )
    got = gdf.groupby(by=by, group_keys=False)["value"].apply(
        lambda x: x.cumprod()
    )

    assert_eq(expect, got)


@pytest.mark.parametrize("group_keys", [None, True, False])
@pytest.mark.parametrize("by", ["A", ["A", "B"]])
def test_groupby_group_keys(group_keys, by):
    gdf = cudf.DataFrame(
        {
            "A": "a a a a b b".split(),
            "B": [1, 1, 2, 2, 3, 3],
            "C": [4, 6, 5, 9, 8, 7],
        }
    )
    pdf = gdf.to_pandas()

    g_group = gdf.groupby(by, group_keys=group_keys)
    p_group = pdf.groupby(by, group_keys=group_keys)

    actual = g_group[["B", "C"]].apply(lambda x: x / x.sum())
    expected = p_group[["B", "C"]].apply(lambda x: x / x.sum())
    assert_eq(actual, expected)


def test_groupby_apply_default_include_groups_excludes_keys():
    # The cudf default for `include_groups` matches pandas 3.x (False):
    # the grouping column should not be present in the per-group frame.
    gdf = cudf.DataFrame(
        {"A": [1, 1, 2, 2], "B": [1, 2, 3, 4], "C": [10, 20, 30, 40]}
    )
    seen_columns: list[list[str]] = []

    def record(x):
        seen_columns.append(list(x.columns))
        return x["B"].sum()

    gdf.groupby("A").apply(record)
    assert all("A" not in cols for cols in seen_columns), seen_columns


def test_seriesgroupby_apply_default_include_groups():
    # Regression: SeriesGroupBy.apply used to crash with `include_groups=False`
    # because `_grouped` tried to ``del`` from a Series.
    gdf = cudf.DataFrame({"A": [1, 1, 2, 2], "B": [1, 2, 3, 4]})
    pdf = gdf.to_pandas()
    got = gdf.groupby("A")["B"].apply(lambda x: x.sum())
    expect = pdf.groupby("A")["B"].apply(lambda x: x.sum())
    assert_eq(got, expect)


def test_groupby_default_group_keys_under_pandas_compat():
    # In pandas-compatible mode, `group_keys` defaults to True so that
    # ``apply`` prepends group labels to the result index, matching pandas.
    gdf = cudf.DataFrame({"A": [1, 1, 2, 2], "B": [1, 2, 3, 4]})
    pdf = gdf.to_pandas()
    with cudf.option_context("mode.pandas_compatible", True):
        got = gdf.groupby("A").apply(lambda x: x["B"] * 2)
    expect = pdf.groupby("A").apply(lambda x: x["B"] * 2)
    assert_eq(got, expect)


@pytest.mark.parametrize(
    "dtype",
    ["int32", "int64", "float64", "datetime64[ns]", "timedelta64[ns]", "bool"],
)
@pytest.mark.parametrize(
    "apply_op",
    ["sum", "min", "max", "idxmax"],
)
def test_group_by_empty_apply(request, dtype, apply_op):
    request.applymarker(
        pytest.mark.xfail(
            condition=(dtype == "datetime64[ns]" and apply_op == "sum"),
            reason=("sum isn't supported for datetime64[ns]"),
        )
    )

    gdf = cudf.DataFrame({"a": [], "b": [], "c": []}, dtype=dtype)
    pdf = gdf.to_pandas()

    gg = gdf.groupby("a")["c"]
    pg = pdf.groupby("a")["c"]

    assert_eq(
        gg.apply(apply_op),
        pg.apply(apply_op),
        check_dtype=True,
        check_index_type=True,
    )


def test_groupby_apply_preserves_inner_index_name():
    # The concatenated apply result keeps the UDF result's index name as
    # the inner MultiIndex level name, matching pandas.
    pdf = pd.DataFrame(
        {
            "name": ["a", "a", "b"],
            "amount": [100.0, 200.0, 300.0],
        },
        index=pd.Index([1, 2, 3], name="stamp"),
    )
    gdf = cudf.from_pandas(pdf)
    expect = pdf.groupby("name").apply(
        lambda x: x["amount"].cumsum(), include_groups=False
    )
    got = gdf.groupby("name").apply(
        lambda x: x["amount"].cumsum(), include_groups=False
    )
    assert expect.index.names == got.index.names
    assert_eq(expect, got)

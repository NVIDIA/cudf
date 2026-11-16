# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ibis
import numpy as np
import pandas as pd
import pytest

ibis.options.interactive = False


def ibis_assert_equal(expect, got, rtol: float = 1e-7, atol: float = 0.0):
    pd._testing.assert_almost_equal(expect, got, rtol=rtol, atol=atol)


pytestmark = [pytest.mark.assert_eq(fn=ibis_assert_equal)]
if hasattr(pd.errors, "Pandas4Warning"):
    # Ibis 12 still reads this deprecated option during dtype conversion.
    # https://github.com/ibis-project/ibis/blob/12.0.0/ibis/formats/pandas.py
    pytestmark.append(
        pytest.mark.filterwarnings(
            r"ignore:The 'future\.infer_string' option is deprecated and will be "
            r"removed in pandas 4\.0:pandas.errors.Pandas4Warning:ibis\.formats\.pandas$"
        )
    )


COLUMN_REDUCTIONS = ["sum", "min", "max", "mean", "var", "std"]
ELEMENTWISE_UFUNCS = [
    "sin",
    "cos",
    "atan",
    "exp",
    "log",
    "abs",
]
STRING_UNARY_FUNCS = [
    "lower",
    "upper",
    "capitalize",
    "reverse",
]


@pytest.fixture
def ibis_table_num_str():
    N = 1000
    K = 5
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        rng.integers(0, 100, (N, K)), columns=[f"col{x}" for x in np.arange(K)]
    )
    df["key"] = rng.choice(np.arange(10), N)
    df["str_col"] = rng.choice(["Hello", "World", "It's", "Me", "Again"], N)
    table = ibis.memtable(df)
    return table


@pytest.fixture
def ibis_table_num():
    N = 100
    K = 2
    rng = np.random.default_rng(42)

    df = pd.DataFrame(
        rng.integers(0, 100, (N, K)), columns=[f"val{x}" for x in np.arange(K)]
    )
    df["key"] = rng.choice(np.arange(10), N)
    table = ibis.memtable(df)
    return table


@pytest.mark.parametrize("op", COLUMN_REDUCTIONS)
def test_column_reductions(ibis_table_num_str, op):
    t = ibis_table_num_str
    return getattr(t.col1, op)().to_pandas()


@pytest.mark.parametrize("op", ["mean", "sum", "min", "max"])
def test_groupby_reductions(ibis_table_num_str, op):
    t = ibis_table_num_str
    return getattr(t.group_by("key").col1, "min")().order_by("key").to_pandas()


@pytest.mark.parametrize("op", ELEMENTWISE_UFUNCS)
def test_mutate_ufunc(ibis_table_num_str, op):
    t = ibis_table_num_str
    if op == "log":
        # avoid duckdb log of 0 error
        t = t.mutate(col1=t.col1 + 1)
    expr = getattr(t.col1, op)()
    return t.mutate(col1_sin=expr).to_pandas()


@pytest.mark.parametrize("op", STRING_UNARY_FUNCS)
def test_string_unary(ibis_table_num_str, op):
    t = ibis_table_num_str
    return getattr(t.str_col, op)().to_pandas()


def test_nunique(ibis_table_num_str):
    t = ibis_table_num_str
    return t.col1.nunique().to_pandas()


def test_count(ibis_table_num_str):
    t = ibis_table_num_str
    return t.col1.count().to_pandas()


def test_select(ibis_table_num_str):
    t = ibis_table_num_str
    return t.select("col0", "col1").to_pandas()


def test_between(ibis_table_num_str):
    t = ibis_table_num_str
    return t.key.between(4, 8).to_pandas()


def test_notin(ibis_table_num_str):
    t = ibis_table_num_str
    return t.key.notin([0, 1, 8, 3]).to_pandas()


@pytest.mark.skip(reason="Failing after Ibis 11 and DuckDB 1.4.0 upgrade")
def test_window(ibis_table_num_str):
    t = ibis_table_num_str
    return (
        t.group_by("key")
        .mutate(demeaned=t.col1 - t.col1.mean())
        .order_by("key")
        .to_pandas()
    )


def test_limit(ibis_table_num_str):
    t = ibis_table_num_str
    return t.limit(5).to_pandas()


def test_filter(ibis_table_num_str):
    t = ibis_table_num_str
    return t.filter([t.key == 4, t.col0 > 15]).to_pandas()


@pytest.mark.skip(reason="Join ordering not currently guaranteed, i.e., flaky")
@pytest.mark.parametrize("join_type", ["inner", "left", "right"])
def test_join_exact_ordering(ibis_table_num_str, ibis_table_num, join_type):
    t1 = ibis_table_num_str
    t2 = ibis_table_num
    res = t1.join(t2, "key", how=join_type).to_pandas()
    return res


@pytest.mark.parametrize("join_type", ["inner", "left", "right"])
def test_join_sort_correctness(ibis_table_num_str, ibis_table_num, join_type):
    """
    While we don't currently guarantee exact row ordering
    we can still test join correctness with ex-post sorting.
    """
    t1 = ibis_table_num_str
    t2 = ibis_table_num
    res = t1.join(t2, "key", how=join_type).to_pandas()

    res_sorted = res.sort_values(by=res.columns.tolist()).reset_index(
        drop=True
    )
    return res_sorted


def test_order_by(ibis_table_num_str):
    t = ibis_table_num_str
    return t.order_by(ibis.desc("col1")).to_pandas()


def test_aggregate_having(ibis_table_num_str):
    t = ibis_table_num_str
    return (
        t.aggregate(
            by=["key"],
            sum_c0=t.col0.sum(),
            avg_c0=t.col0.mean(),
            having=t.col1.mean() > 50,
        )
        .order_by("key")
        .to_pandas()
    )

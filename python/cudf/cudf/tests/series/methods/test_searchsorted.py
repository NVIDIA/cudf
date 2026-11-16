# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cupy
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.core._compat import PANDAS_GE_31
from cudf.testing import assert_eq
from cudf.testing._utils import gen_rand


@pytest.mark.skipif(
    not PANDAS_GE_31, reason="Pandas-compatible validation in 3.1"
)
@pytest.mark.parametrize("target", [1, np.int64(1), np.arange(3, dtype="i8")])
@pytest.mark.parametrize("box", [cudf.Series, cudf.TimedeltaIndex])
def test_searchsorted_timedelta_rejects_numeric(target, box):
    source = box(np.arange(3).view("timedelta64[ns]"))
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(TypeError, match="compatible dtype"):
            source.searchsorted(target)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("obj_class", ["series", "index", "column"])
@pytest.mark.parametrize("vals_class", ["series", "index"])
def test_searchsorted(side, obj_class, vals_class):
    nelem = 1000
    column_data = gen_rand("float64", nelem)
    rng = np.random.default_rng(0)
    column_mask = rng.choice([True, False], size=nelem)

    values_data = gen_rand("float64", nelem)
    values_mask = rng.choice([True, False], size=nelem)

    sr = cudf.Series(column_data)
    sr.loc[column_mask] = None
    vals = cudf.Series(values_data)
    vals.loc[values_mask] = None

    sr = sr.sort_values()

    # Reference object can be Series, Index, or Column
    if obj_class == "index":
        sr.reset_index(drop=True)
    elif obj_class == "column":
        sr = sr._column

    # Values can be Series or Index
    if vals_class == "index":
        vals.reset_index(drop=True)

    psr = sr.to_pandas()
    pvals = vals.to_pandas()

    expect = psr.searchsorted(pvals, side)
    if obj_class == "column":
        got = sr.searchsorted(vals._column, side)
    else:
        got = sr.searchsorted(vals, side)

    assert_eq(expect, cupy.asnumpy(got))


@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted_categorical(side):
    cat1 = pd.Categorical(
        ["a", "a", "b", "c", "a"], categories=["a", "b", "c"], ordered=True
    )
    psr1 = pd.Series(cat1).sort_values()
    sr1 = cudf.Series(cat1).sort_values()
    cat2 = pd.Categorical(
        ["a", "b", "a", "c", "b"], categories=["a", "b", "c"], ordered=True
    )
    psr2 = pd.Series(cat2)
    sr2 = cudf.Series(cat2)

    expect = psr1.searchsorted(psr2, side)
    got = sr1.searchsorted(sr2, side)

    assert_eq(expect, cupy.asnumpy(got))


@pytest.mark.parametrize("side", ["left", "right"])
def test_searchsorted_datetime(side):
    psr1 = pd.Series(
        pd.date_range("20190101", "20200101", freq="400h", name="times")
    )
    sr1 = cudf.from_pandas(psr1)

    psr2 = pd.Series(
        np.array(
            [
                np.datetime64("2019-11-20"),
                np.datetime64("2019-04-15"),
                np.datetime64("2019-02-20"),
                np.datetime64("2019-05-31"),
                np.datetime64("2020-01-02"),
            ]
        )
    )

    sr2 = cudf.from_pandas(psr2)

    expect = psr1.searchsorted(psr2, side)
    got = sr1.searchsorted(sr2, side)

    assert_eq(expect, cupy.asnumpy(got))


def test_searchsorted_misc():
    psr = pd.Series([1, 2, 3.4, 6])
    sr = cudf.from_pandas(psr)

    assert_eq(psr.searchsorted(1), sr.searchsorted(1))
    assert_eq(psr.searchsorted(0), sr.searchsorted(0))
    assert_eq(psr.searchsorted(4), sr.searchsorted(4))
    assert_eq(psr.searchsorted(5), sr.searchsorted(5))
    assert_eq(
        psr.searchsorted([-100, 3.4, 2.2, 2.0, 2.000000001]),
        sr.searchsorted([-100, 3.4, 2.2, 2.0, 2.000000001]),
    )

    psr = pd.Series([1, 2, 3])
    sr = cudf.from_pandas(psr)
    assert_eq(psr.searchsorted(1), sr.searchsorted(1))
    assert_eq(
        psr.searchsorted([0, 1, 2, 3, 4, -4, -3, -2, -1, 0, -120]),
        sr.searchsorted([0, 1, 2, 3, 4, -4, -3, -2, -1, 0, -120]),
    )
    assert_eq(psr.searchsorted(1.5), sr.searchsorted(1.5))
    assert_eq(psr.searchsorted(1.99), sr.searchsorted(1.99))
    assert_eq(psr.searchsorted(3.00001), sr.searchsorted(3.00001))
    assert_eq(
        psr.searchsorted([-100, 3.00001, 2.2, 2.0, 2.000000001]),
        sr.searchsorted([-100, 3.00001, 2.2, 2.0, 2.000000001]),
    )


@pytest.mark.xfail(reason="https://github.com/pandas-dev/pandas/issues/54668")
def test_searchsorted_mixed_str_int():
    psr = pd.Series([1, 2, 3], dtype="int")
    sr = cudf.from_pandas(psr)

    with pytest.raises(ValueError):
        actual = sr.searchsorted("a")
    with pytest.raises(ValueError):
        expect = psr.searchsorted("a")
    assert_eq(expect, actual)

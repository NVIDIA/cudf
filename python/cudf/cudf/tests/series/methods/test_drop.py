# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    assert_exceptions_equal,
)


@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in Series\\.drop is deprecated:pandas.errors.Pandas4Warning"
)
@pytest.mark.parametrize(
    "ps",
    [
        pd.Series(["a"] * 20, index=range(0, 20)),
        pd.Series(["b", None] * 10, index=range(0, 20), name="ASeries"),
        pd.Series(
            ["b", None] * 5,
            index=pd.Index(list(range(10)), dtype="uint64"),
            name="BSeries",
        ),
    ],
)
@pytest.mark.parametrize(
    "labels",
    [[0], 1, pd.Index([1, 3, 5]), np.array([1, 3, 5], dtype="float32")],
)
def test_series_drop_labels(ps, labels, inplace):
    ps = ps.copy()
    gs = cudf.from_pandas(ps)

    expected = ps.drop(labels=labels, axis=0)
    actual = gs.drop(labels=labels, axis=0, inplace=inplace)

    if inplace:
        assert actual is None
        actual = gs

    assert_eq(expected, actual)


@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in Series\\.drop is deprecated:pandas.errors.Pandas4Warning"
)
@pytest.mark.parametrize("data", [["a"] * 20, ["b", None] * 10])
@pytest.mark.parametrize("index", [[0], 1, pd.Index([1, 3, 5])])
def test_series_drop_index(data, index, inplace):
    ps = pd.Series(data, index=range(0, 20), name="a")
    gs = cudf.from_pandas(ps)

    expected = ps.drop(index=index)
    actual = gs.drop(index=index, inplace=inplace)

    if inplace:
        assert actual is None
        actual = gs

    assert_eq(expected, actual)


@pytest.mark.filterwarnings(
    "ignore:The inplace keyword in Series\\.drop is deprecated:pandas.errors.Pandas4Warning"
)
@pytest.mark.parametrize(
    "index,level",
    [
        ("cow", 0),
        ("lama", 0),
        ("falcon", 0),
        ("speed", 1),
        ("weight", 1),
        ("length", 1),
        (
            "cow",
            None,
        ),
        (
            "lama",
            None,
        ),
        (
            "falcon",
            None,
        ),
    ],
)
def test_series_drop_multiindex(index, level, inplace):
    ps = pd.Series(
        ["a" if i % 2 == 0 else "b" for i in range(0, 10)],
        index=pd.MultiIndex(
            levels=[
                ["lama", "cow", "falcon"],
                ["speed", "weight", "length"],
            ],
            codes=[
                [0, 0, 0, 1, 1, 1, 2, 2, 2, 1],
                [0, 1, 2, 0, 1, 2, 0, 1, 2, 1],
            ],
        ),
        name="abc",
    )
    gs = cudf.from_pandas(ps)

    expected = ps.drop(index=index, level=level)
    actual = gs.drop(index=index, inplace=inplace, level=level)

    if inplace:
        assert actual is None
        actual = gs

    assert_eq(expected, actual)


def test_series_drop_edge_inputs():
    gs = cudf.Series([42], name="a")
    ps = gs.to_pandas()

    assert_eq(ps.drop(columns=["b"]), gs.drop(columns=["b"]))

    assert_eq(ps.drop(columns="b"), gs.drop(columns="b"))

    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=(["a"], {"columns": "a", "axis": 1}),
        rfunc_args_and_kwargs=(["a"], {"columns": "a", "axis": 1}),
    )

    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=([], {}),
        rfunc_args_and_kwargs=([], {}),
    )

    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=(["b"], {"axis": 1}),
        rfunc_args_and_kwargs=(["b"], {"axis": 1}),
    )


def test_series_drop_raises():
    gs = cudf.Series([10, 20, 30], index=["x", "y", "z"], name="c")
    ps = gs.to_pandas()

    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=(["p"],),
        rfunc_args_and_kwargs=(["p"],),
    )

    # dtype specified mismatch
    assert_exceptions_equal(
        lfunc=ps.drop,
        rfunc=gs.drop,
        lfunc_args_and_kwargs=([3],),
        rfunc_args_and_kwargs=([3],),
    )

    expect = ps.drop("p", errors="ignore")
    actual = gs.drop("p", errors="ignore")

    assert_eq(actual, expect)

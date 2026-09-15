# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


def test_series_setitem_singleton_range():
    sr = cudf.Series([1, 2, 3], dtype=np.int64)
    psr = sr.to_pandas()
    value = np.asarray([7], dtype=np.int64)
    sr.iloc[:1] = value
    psr.iloc[:1] = value
    assert_eq(sr, cudf.Series([7, 2, 3], dtype=np.int64))
    assert_eq(sr, psr, check_dtype=True)


@pytest.mark.parametrize(
    "indices",
    [slice(0, 3), slice(1, 4), slice(None, None, 2), slice(1, None, 2)],
    ids=[":3", "1:4", "0::2", "1::2"],
)
@pytest.mark.parametrize(
    "values",
    [[None, {}, {}, None], [{}, {}, {}, {}]],
    ids=["nulls", "no_nulls"],
)
def test_struct_empty_children_slice(indices, values):
    s = cudf.Series(values)
    actual = s.iloc[indices]
    expect = cudf.Series(values[indices], index=range(len(values))[indices])
    assert_eq(actual, expect)


@pytest.mark.parametrize(
    "item",
    [
        0,
        2,
        4,
        slice(1, 3),
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 4, 3, 2, 1, 0],
        np.array([0, 1, 2, 3, 4]),
        cp.asarray(np.array([0, 1, 2, 3, 4])),
    ],
)
@pytest.mark.parametrize("data", [["a"] * 5, ["a", None] * 3, [None] * 5])
def test_string_get_item(data, item):
    ps = pd.Series(data, dtype="str", name="nice name")
    gs = cudf.Series(data, dtype="str", name="nice name")

    if isinstance(item, cp.ndarray):
        item_np = cp.asnumpy(item)
    else:
        item_np = item

    got = gs.iloc[item]
    expect = ps.iloc[item_np]

    if isinstance(expect, pd.Series):
        assert_eq(expect, got)
    else:
        if got is cudf.NA and expect is None:
            return
        if isinstance(expect, float) and np.isnan(expect):
            assert isinstance(got, float) and np.isnan(got)
        else:
            assert expect == got


@pytest.mark.parametrize("bool_", [True, False])
@pytest.mark.parametrize("data", [["a"], ["a", None], [None]])
@pytest.mark.parametrize("box", [list, np.array, cp.array])
def test_string_bool_mask(data, bool_, box):
    ps = pd.Series(data, dtype="str", name="nice name")
    gs = cudf.Series(data, dtype="str", name="nice name")
    item = box([bool_] * len(data))

    if isinstance(item, cp.ndarray):
        item_np = cp.asnumpy(item)
    else:
        item_np = item

    got = gs.iloc[item]
    expect = ps[item_np]

    if isinstance(expect, pd.Series):
        assert_eq(expect, got)
    else:
        got_scalar = got
        if got_scalar is cudf.NA and expect is None:
            return
        assert expect == got_scalar


def test_series_iloc():
    # create random cudf.Series
    nelem = 20
    rng = np.random.default_rng(seed=0)
    ps = pd.Series(rng.random(nelem))

    # gpu cudf.Series
    gs = cudf.Series(ps)

    # positive tests for indexing
    np.testing.assert_allclose(gs.iloc[-1 * nelem], ps.iloc[-1 * nelem])
    np.testing.assert_allclose(gs.iloc[-1], ps.iloc[-1])
    np.testing.assert_allclose(gs.iloc[0], ps.iloc[0])
    np.testing.assert_allclose(gs.iloc[1], ps.iloc[1])
    np.testing.assert_allclose(gs.iloc[nelem - 1], ps.iloc[nelem - 1])

    # positive tests for slice
    np.testing.assert_allclose(gs.iloc[-1:1].to_numpy(), ps.iloc[-1:1])
    np.testing.assert_allclose(
        gs.iloc[nelem - 1 : -1].to_numpy(), ps.iloc[nelem - 1 : -1]
    )
    np.testing.assert_allclose(
        gs.iloc[0 : nelem - 1].to_pandas(), ps.iloc[0 : nelem - 1]
    )
    np.testing.assert_allclose(gs.iloc[0:nelem].to_pandas(), ps.iloc[0:nelem])
    np.testing.assert_allclose(gs.iloc[1:1].to_pandas(), ps.iloc[1:1])
    np.testing.assert_allclose(gs.iloc[1:2].to_pandas(), ps.iloc[1:2].values)
    np.testing.assert_allclose(
        gs.iloc[nelem - 1 : nelem + 1].to_pandas(),
        ps.iloc[nelem - 1 : nelem + 1],
    )
    np.testing.assert_allclose(
        gs.iloc[nelem : nelem * 2].to_pandas(), ps.iloc[nelem : nelem * 2]
    )


@pytest.mark.parametrize(
    "key, value",
    [
        (0, 4),
        (1, 4),
        ([0, 1], 4),
        ([0, 1], [4, 5]),
        (slice(0, 2), [4, 5]),
        (slice(1, None), [4, 5, 6, 7]),
        ([], 1),
        (slice(None, None), 1),
        (slice(-1, -3), 7),
    ],
)
@pytest.mark.parametrize("nulls", ["none", "some", "all"])
def test_series_setitem_iloc(key, value, nulls):
    psr = pd.Series([1, 2, 3, 4, 5])
    if nulls == "some":
        psr[[0, 4]] = None
    elif nulls == "all":
        psr[:] = None
    gsr = cudf.from_pandas(psr)
    gsr.iloc[key] = value
    psr.iloc[key] = value

    assert_eq(psr, gsr, check_dtype=False)


def test_iloc_negative_indices():
    psr = pd.Series([1, 2, 3, 4, 5])
    gsr = cudf.from_pandas(psr)
    assert_eq(psr.iloc[[-1, -2, -4]], gsr.iloc[[-1, -2, -4]])


def test_out_of_bounds_indexing_empty():
    psr = pd.Series(dtype="int64")
    gsr = cudf.from_pandas(psr)
    assert_exceptions_equal(
        lambda: psr.iloc.__setitem__(-1, 2),
        lambda: gsr.iloc.__setitem__(-1, 2),
    )
    assert_exceptions_equal(
        lambda: psr.iloc.__setitem__(1, 2),
        lambda: gsr.iloc.__setitem__(1, 2),
    )


@pytest.mark.parametrize(
    "gdf",
    [
        lambda: cudf.DataFrame({"a": range(10000)}),
        lambda: cudf.DataFrame(
            {
                "a": range(10000),
                "b": range(10000),
                "c": range(10000),
                "d": range(10000),
                "e": range(10000),
                "f": range(10000),
            }
        ),
        lambda: cudf.DataFrame({"a": range(20), "b": range(20)}),
        lambda: cudf.DataFrame(
            {
                "a": range(20),
                "b": range(20),
                "c": ["abc", "def", "xyz", "def", "pqr"] * 4,
            }
        ),
        lambda: cudf.DataFrame(index=[1, 2, 3]),
        lambda: cudf.DataFrame(index=range(10000)),
        lambda: cudf.DataFrame(columns=["a", "b", "c", "d"]),
        lambda: cudf.DataFrame(columns=["a"], index=range(10000)),
        lambda: cudf.DataFrame(
            columns=["a", "col2", "...col n"], index=range(10000)
        ),
        lambda: cudf.DataFrame(index=cudf.Series(range(10000)).astype("str")),
        lambda: cudf.DataFrame(
            columns=["a", "b", "c", "d"],
            index=cudf.Series(range(10000)).astype("str"),
        ),
    ],
)
@pytest.mark.parametrize(
    "slice",
    [slice(6), slice(1), slice(7), slice(1, 3)],
)
def test_dataframe_iloc_index(gdf, slice):
    gdf = gdf()
    pdf = gdf.to_pandas()

    actual = gdf.iloc[:, slice]
    expected = pdf.iloc[:, slice]

    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        [[0], [1], [2]],
        [[0, 1], [2, 3], [4, 5]],
        [[[0, 1], [2]], [[3, 4]], [[5, 6]]],
        [None, [[0, 1], [2]], [[3, 4], [5, 6]]],
        [[], [[0, 1], [2]], [[3, 4], [5, 6]]],
        [[], [["a", "b"], None], [["c", "d"], []]],
    ],
)
@pytest.mark.parametrize(
    "key", [[], [0], [0, 1], [0, 1, 0], slice(None), slice(0, 2), slice(1, 3)]
)
def test_iloc_with_lists(data, key):
    psr = pd.Series(data)
    gsr = cudf.Series(data)
    assert_eq(psr.iloc[key], gsr.iloc[key])

    pdf = pd.DataFrame({"a": data, "b": data})
    gdf = cudf.DataFrame({"a": data, "b": data})
    assert_eq(pdf.iloc[key], gdf.iloc[key])


@pytest.mark.parametrize(
    "arg",
    [
        slice(None, None, -1),
        slice(None, -1, -1),
        slice(4, -1, -1),
        slice(None, None, -3),
        slice(None, -1, -3),
        slice(4, -1, -3),
    ],
)
@pytest.mark.parametrize(
    "pobj", [pd.DataFrame({"a": [1, 2, 3, 4, 5]}), pd.Series([1, 2, 3, 4, 5])]
)
def test_iloc_before_zero_terminate(arg, pobj):
    gobj = cudf.from_pandas(pobj)

    assert_eq(pobj.iloc[arg], gobj.iloc[arg])


def test_iloc_decimal():
    sr = cudf.Series(["1.00", "2.00", "3.00", "4.00"]).astype(
        cudf.Decimal64Dtype(scale=2, precision=3)
    )
    got = sr.iloc[[3, 2, 1, 0]]
    expect = cudf.Series(
        ["4.00", "3.00", "2.00", "1.00"],
    ).astype(cudf.Decimal64Dtype(scale=2, precision=3))
    assert_eq(expect.reset_index(drop=True), got.reset_index(drop=True))


@pytest.mark.parametrize("indexer", [[1], [0, 2]])
def test_iloc_integer_categorical_issue_13013(indexer):
    # https://github.com/NVIDIA/cudf/issues/13013
    s = pd.Series([0, 1, 2])
    index = pd.Categorical(indexer)
    expect = s.iloc[index]
    c = cudf.from_pandas(s)
    actual = c.iloc[index]
    assert_eq(expect, actual)


def test_iloc_incorrect_boolean_mask_length_issue_13015():
    # https://github.com/NVIDIA/cudf/issues/13015
    s = pd.Series([0, 1, 2])
    with pytest.raises(IndexError):
        s.iloc[[True, False]]
    c = cudf.from_pandas(s)
    with pytest.raises(IndexError):
        c.iloc[[True, False]]


@pytest.mark.parametrize("typ", ["datetime64[ns]", "timedelta64[ns]"])
@pytest.mark.parametrize("idx_method, key", [["iloc", 0], ["loc", "a"]])
def test_series_iloc_scalar_datetimelike_return_pd_scalar(
    typ, idx_method, key
):
    obj = cudf.Series([1, 2, 3], index=list("abc"), dtype=typ)
    with cudf.option_context("mode.pandas_compatible", True):
        result = getattr(obj, idx_method)[key]
    expected = getattr(obj.to_pandas(), idx_method)[key]
    assert result == expected


@pytest.mark.parametrize("idx_method, key", [["iloc", 0], ["loc", "a"]])
def test_series_iloc_scalar_interval_return_pd_scalar(idx_method, key):
    iidx = cudf.IntervalIndex.from_breaks([1, 2, 3])
    obj = cudf.Series(iidx, index=list("ab"))
    with cudf.option_context("mode.pandas_compatible", True):
        result = getattr(obj, idx_method)[key]
    expected = getattr(obj.to_pandas(), idx_method)[key]
    assert result == expected

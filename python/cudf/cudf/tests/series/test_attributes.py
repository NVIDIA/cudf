# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import re

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "testlist",
    [
        [1, 2, 3, 4],
        [1, 2, 3, 3, 4],
        [10, 9, 8, 7],
        [10, 9, 8, 8, 7],
        ["c", "d", "e", "f"],
        ["c", "d", "e", "e", "f"],
        ["z", "y", "x", "r"],
        ["z", "y", "x", "x", "r"],
    ],
)
def test_series_is_unique_monotonic(testlist):
    series = cudf.Series(testlist)
    series_pd = pd.Series(testlist)

    assert series.is_unique == series_pd.is_unique
    assert series.is_monotonic_increasing == series_pd.is_monotonic_increasing
    assert series.is_monotonic_decreasing == series_pd.is_monotonic_decreasing


@pytest.mark.parametrize(
    "data",
    [
        [pd.Timestamp("2018-01-01"), pd.Timestamp("2019-01-31"), None],
        [1, 2, 3, None],
        [None, 1, 2, 3],
        ["a", "b", "c", None],
        [None, "a", "b", "c"],
    ],
)
def test_is_monotonic_always_false_for_null(data):
    ser = cudf.Series(data)
    assert ser.is_monotonic_increasing is False
    assert ser.is_monotonic_decreasing is False


@pytest.mark.parametrize("box", [cudf.Series, cudf.Index])
@pytest.mark.parametrize(
    "value,na_like",
    [
        [1, None],
        [np.datetime64("2020-01-01", "ns"), np.datetime64("nat", "ns")],
        ["s", None],
        [1.0, np.nan],
    ],
    ids=repr,
)
def test_is_unique(box, value, na_like):
    obj = box([value], nan_as_null=False)
    assert obj.is_unique

    obj = box([value, value], nan_as_null=False)
    assert not obj.is_unique

    obj = box([None, value], nan_as_null=False)
    assert obj.is_unique

    obj = box([None, None, value], nan_as_null=False)
    assert not obj.is_unique

    if na_like is not None:
        obj = box([na_like, value], nan_as_null=False)
        assert obj.is_unique

        obj = box([na_like, na_like], nan_as_null=False)
        assert not obj.is_unique

        try:
            if not np.isnat(na_like):
                # pyarrow coerces nat to null
                obj = box([None, na_like, value], nan_as_null=False)
                assert obj.is_unique
        except TypeError:
            pass


@pytest.fixture(
    params=[
        pd.Series([0, 1, 2, np.nan, 4, None, 6]),
        pd.Series(
            [0, 1, 2, np.nan, 4, None, 6],
            index=["q", "w", "e", "r", "t", "y", "u"],
            name="a",
        ),
        pd.Series([0, 1, 2, 3, 4]),
        pd.Series(["a", "b", "u", "h", "d"]),
        pd.Series([None, None, np.nan, None, np.inf, -np.inf]),
        pd.Series([], dtype="float64"),
        pd.Series(
            [pd.NaT, pd.Timestamp("1939-05-27"), pd.Timestamp("1940-04-25")]
        ),
        pd.Series([np.nan]),
        pd.Series([None]),
        pd.Series(["a", "b", "", "c", None, "e"]),
    ]
)
def ps(request):
    return request.param


def test_series_iter_error():
    gs = cudf.Series([1, 2, 3])

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.to_numpy()` "
            f"if you wish to iterate over the values."
        ),
    ):
        iter(gs)

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.to_numpy()` "
            f"if you wish to iterate over the values."
        ),
    ):
        gs.items()

    with pytest.raises(
        TypeError,
        match=re.escape(
            f"{gs.__class__.__name__} object is not iterable. "
            f"Consider using `.to_arrow()`, `.to_pandas()` or `.to_numpy()` "
            f"if you wish to iterate over the values."
        ),
    ):
        gs.iteritems()

    with pytest.raises(TypeError):
        iter(gs._column)


@pytest.mark.parametrize(
    "data",
    [
        lambda: cudf.Series([1, 2, 3, -12, 12, 44]),
        lambda: cudf.Series([1, 2, 3, -12, 12, 44], dtype="str"),
        lambda: cudf.DataFrame(
            {"a": [1, 2, 3, -1234], "b": [0.1, 0.2222, 0.4, -3.14]}
        ),
    ],
)
@pytest.mark.parametrize("dtype", [None, "float", "int", "str"])
def test_series_dataframe__array__(data, dtype):
    gs = data()

    with pytest.raises(TypeError):
        gs.__array__(dtype=dtype)

    with pytest.raises(TypeError):
        gs.index.__array__(dtype=dtype)


@pytest.mark.parametrize("data", [[], [None, None], ["a", None]])
def test_series_size(data):
    psr = pd.Series(data)
    gsr = cudf.Series(data)

    assert_eq(psr.size, gsr.size)


def test_set_index_unequal_length():
    s = cudf.Series(dtype="float64")
    with pytest.raises(ValueError):
        s.index = [1, 2, 3]


@pytest.mark.parametrize(
    "data",
    [
        [],
        [1, 2, 3, 4],
        ["a", "b", "c"],
        [1.2, 2.2, 4.5],
        [np.nan, np.nan],
        [None, None, None],
    ],
)
def test_axes(data):
    csr = cudf.Series(data)
    psr = csr.to_pandas()

    expected = psr.axes
    actual = csr.axes

    for e, a in zip(expected, actual, strict=True):
        assert_eq(e, a)


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series(dtype="float64"),
        pd.Series(index=[100, 10, 1, 0], dtype="float64"),
        pd.Series([], dtype="float64"),
        pd.Series(["a", "b", "c", "d"]),
        pd.Series(["a", "b", "c", "d"], index=[0, 1, 10, 11]),
    ],
)
def test_series_empty(ps):
    ps = ps
    gs = cudf.from_pandas(ps)

    assert_eq(ps.empty, gs.empty)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        [np.nan, 10, 15, 16],
        [np.nan, None, 10, 20],
        ["ab", "zx", "pq"],
        ["ab", "zx", None, "pq"],
        [],
    ],
)
def test_series_hasnans(data):
    gs = cudf.Series(data, nan_as_null=False)
    ps = gs.to_pandas(nullable=True)

    # Check type to avoid mixing Python bool and NumPy bool
    assert isinstance(gs.hasnans, bool)
    assert gs.hasnans == ps.hasnans


def test_category_dtype_attribute():
    psr = pd.Series(["a", "b", "a", "c"], dtype="category")
    sr = cudf.Series(["a", "b", "a", "c"], dtype="category")
    assert isinstance(sr.dtype, cudf.CategoricalDtype)
    assert_eq(sr.dtype.categories, psr.dtype.categories)


def test_dtype_dtypes_equal():
    ser = cudf.Series([0])
    assert ser.dtype is ser.dtypes
    assert ser.dtypes is ser.to_pandas().dtypes


@pytest.mark.parametrize("data", [[], [1, 2, 3, 4, 5]])
@pytest.mark.parametrize(
    "scalar",
    [
        1,
        2,
        3,
        "a",
        np.timedelta64(1, "s"),
        np.timedelta64(2, "s"),
        np.timedelta64(2, "D"),
        np.timedelta64(3, "ms"),
        np.timedelta64(4, "us"),
        np.timedelta64(5, "ns"),
        np.timedelta64(6, "ns"),
        np.datetime64(6, "s"),
    ],
)
def test_timedelta_contains(data, timedelta_types_as_str, scalar):
    sr = cudf.Series(data, dtype=timedelta_types_as_str)
    psr = sr.to_pandas()

    expected = scalar in sr
    actual = scalar in psr

    assert_eq(expected, actual)


def test_cai_after_indexing():
    df = cudf.DataFrame({"a": [1, 2, 3]})
    cai1 = df["a"].__cuda_array_interface__
    df[["a"]]
    cai2 = df["a"].__cuda_array_interface__
    assert cai1 == cai2


@pytest.mark.parametrize(
    "data, expected",
    [
        [["2018-01-01", None, "2019-01-31", None, "2018-01-01"], True],
        [
            [
                "2018-01-01",
                "2018-01-02",
                "2019-01-31",
                "2018-03-01",
                "2018-01-01",
            ],
            False,
        ],
        [
            np.array(
                ["2018-01-01", None, "2019-12-30"], dtype="datetime64[ms]"
            ),
            True,
        ],
    ],
)
def test_datetime_has_null_test(data, expected):
    data = cudf.Series(data, dtype="datetime64[ms]")
    pd_data = data.to_pandas()
    count = pd_data.notna().value_counts()
    expected_count = 0
    if False in count.keys():
        expected_count = count[False]

    assert expected is data.has_nulls
    assert expected_count == data.null_count


def test_datetime_has_null_test_pyarrow():
    data = cudf.Series(
        pa.array(
            [0, np.iinfo("int64").min, np.iinfo("int64").max, None],
            type=pa.timestamp("ns"),
        )
    )
    assert data.has_nulls is True
    assert data.null_count == 1


def test_error_values_datetime():
    s = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    result = s.values
    expected = cp.array([1, 2, 3]).view("datetime64[ns]")
    assert_eq(result, expected)


def test_ndim():
    s = pd.Series(dtype="float64")
    gs = cudf.Series()
    assert s.ndim == gs.ndim


def test_multiindex_series_assignment():
    ps = pd.Series([1, 2, 3])
    gs = cudf.from_pandas(ps)
    ps.index = pd.MultiIndex([["a", "b"], ["c", "d"]], [[0, 1, 0], [1, 0, 1]])
    gs.index = cudf.MultiIndex(
        levels=[["a", "b"], ["c", "d"]], codes=[[0, 1, 0], [1, 0, 1]]
    )
    assert_eq(ps, gs)


def test_series_multiindex():
    pdfIndex = pd.MultiIndex.from_arrays([range(7)])
    rng = np.random.default_rng(seed=0)
    ps = pd.Series(rng.random(7))
    gs = cudf.from_pandas(ps)
    ps.index = pdfIndex
    gs.index = cudf.from_pandas(pdfIndex)
    assert_eq(ps, gs)


def test_series_shape():
    ps = pd.Series([1, 2, 3, 4])
    cs = cudf.Series([1, 2, 3, 4])

    assert ps.shape == cs.shape


def test_series_shape_empty():
    ps = pd.Series([], dtype="float64")
    cs = cudf.Series([], dtype="float64")

    assert ps.shape == cs.shape


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 4],
        [],
        [5.0, 7.0, 8.0],
        pytest.param(
            pd.Categorical(["a", "b", "c"]),
            marks=pytest.mark.xfail(raises=NotImplementedError),
        ),
        pytest.param(
            ["m", "a", "d", "v"],
            marks=pytest.mark.xfail(raises=TypeError),
        ),
    ],
)
def test_series_values_property(data):
    pds = pd.Series(data=data, dtype=None if data else float)
    gds = cudf.from_pandas(pds)
    gds_vals = gds.values
    assert isinstance(gds_vals, cp.ndarray)
    np.testing.assert_array_equal(gds_vals.get(), pds.values)


@pytest.mark.parametrize("dtype", ["Int64", "Float64", "boolean"])
def test_series_values_masked_dtype_with_nulls_raises(dtype):
    # A pandas nullable/masked extension dtype with nulls cannot be
    # losslessly represented as a numpy array, so cudf raises instead of
    # silently downcasting the NA to NaN. pandas instead returns the
    # ExtensionArray (deliberate cudf divergence), so the exceptions
    # cannot be compared with assert_exceptions_equal here.
    gs = cudf.Series([0, None], dtype=dtype)
    with pytest.raises(ValueError, match="cannot convert"):
        gs.values
    # pandas returns the masked ExtensionArray without raising.
    ps = pd.Series([0, pd.NA], dtype=dtype)
    assert isinstance(ps.values, pd.api.extensions.ExtensionArray)

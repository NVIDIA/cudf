# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "data",
    [
        [1],
        [1, 2, 3],
        [[1, 2, 3], [4, 5, 6]],
        [pd.NA],
        [1, pd.NA, 3],
        [[1, pd.NA, 3], [pd.NA, 5, 6]],
        [[1.1, pd.NA, 3.3], [4.4, 5.5, pd.NA]],
        [["a", pd.NA, "c"], ["d", "e", pd.NA]],
        [["a", "b", "c"], ["d", "e", "f"]],
    ],
)
def test_list_getitem(data):
    list_sr = cudf.Series([data])
    assert list_sr[0] == data


@pytest.mark.parametrize("nesting_level", [1, 3])
def test_list_scalar_device_construction_null(nesting_level):
    data = [[]]
    for i in range(nesting_level - 1):
        data = [data]

    arrow_type = pa.infer_type(data)
    arrow_arr = pa.array([None], type=arrow_type)

    res = cudf.Series(arrow_arr)[0]
    assert res is cudf.NA


@pytest.mark.parametrize(
    "data, idx",
    [
        (
            [[{"f2": {"a": 100}, "f1": "a"}, {"f1": "sf12", "f2": pd.NA}]],
            0,
        ),
        (
            [
                [
                    {"f2": {"a": 100, "c": 90, "f2": 10}, "f1": "a"},
                    {"f1": "sf12", "f2": pd.NA},
                ]
            ],
            0,
        ),
        (
            [[[[1, 2]], [[2], [3]]], [[[2]]], [[[3]]]],
            0,
        ),
        ([[[[1, 2]], [[2], [3]]], [[[2]]], [[[3]]]], 2),
        ([[[{"a": 1, "b": 2, "c": 10}]]], 0),
    ],
)
def test_nested_list_extract_host_scalars(data, idx):
    series = cudf.Series(data)

    assert series[idx] == data[idx]


@pytest.mark.parametrize(
    "data, idx, expected",
    [
        (
            [{"f2": {"a": "sf21"}, "f1": "a"}, {"f1": "sf12", "f2": None}],
            0,
            {"f1": "a", "f2": {"a": "sf21"}},
        ),
        (
            [
                {"f2": {"a": "sf21"}},
                {"f1": "sf12", "f2": None},
            ],
            0,
            {"f1": cudf.NA, "f2": {"a": "sf21"}},
        ),
        (
            [{"a": "123"}, {"a": "sf12", "b": {"a": {"b": "c"}}}],
            1,
            {"a": "sf12", "b": {"a": {"b": "c"}}},
        ),
    ],
)
def test_nested_struct_extract_host_scalars(data, idx, expected):
    series = cudf.Series(data)

    def _nested_na_replace(struct_scalar):
        """
        Replace `cudf.NA` with `None` in the dict
        """
        for key, value in struct_scalar.items():
            if value is cudf.NA:
                struct_scalar[key] = None
        return struct_scalar

    assert _nested_na_replace(series[idx]) == _nested_na_replace(expected)


def test_nested_struct_from_pandas_empty():
    # tests constructing nested structs columns that would result in
    # libcudf EMPTY type child columns inheriting their parent's null
    # mask. See GH PR: #10761
    pdf = pd.Series([[{"c": {"x": None}}], [{"c": None}]])
    gdf = cudf.from_pandas(pdf)

    assert_eq(pdf, gdf)


def test_struct_int_values():
    series = cudf.Series(
        [{"a": 1, "b": 2}, {"a": 10, "b": None}, {"a": 5, "b": 6}]
    )
    actual_series = series.to_pandas()

    assert isinstance(actual_series[0]["b"], int)
    assert isinstance(actual_series[1]["b"], type(None))
    assert isinstance(actual_series[2]["b"], int)


def test_struct_slice_nested_struct():
    data = [
        {"a": {"b": 42, "c": "abc"}},
        {"a": {"b": 42, "c": "hello world"}},
    ]

    got = cudf.Series(data)[0:1]
    expect = cudf.Series(data[0:1])
    assert got.to_arrow() == expect.to_arrow()


@pytest.mark.parametrize(
    "series, slce",
    [
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": 1},
                {},
                None,
            ],
            slice(1, None),
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": 1},
                {},
                None,
                {"d": ["Hello", "rapids"]},
                None,
                cudf.NA,
            ],
            slice(1, 5),
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": 1},
                {},
                None,
                {"c": 5},
                None,
                cudf.NA,
            ],
            slice(None, 4),
        ),
        ([{"a": {"b": 42, "c": -1}}, {"a": {"b": 0, "c": None}}], slice(0, 1)),
    ],
)
def test_struct_slice(series, slce):
    got = cudf.Series(series)[slce]
    expected = cudf.Series(series[slce])
    assert got.to_arrow() == expected.to_arrow()


@pytest.mark.parametrize(
    "series, expected",
    [
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": 1},
                {},
            ],
            {"a": "Hello world", "b": [], "c": cudf.NA},
        ),
        ([{}], {}),
        (
            [{"b": True}, {"a": 1, "c": [1, 2, 3], "d": "1", "b": False}],
            {"a": cudf.NA, "c": cudf.NA, "d": cudf.NA, "b": True},
        ),
    ],
)
def test_struct_getitem(series, expected):
    sr = cudf.Series(series)
    assert sr[0] == expected


def test_datetime_getitem_na():
    s = cudf.Series([1, 2, None, 3], dtype="datetime64[ns]")
    assert s[2] is cudf.NaT


def test_timedelta_getitem_na():
    s = cudf.Series([1, 2, None, 3], dtype="timedelta64[ns]")
    assert s[2] is cudf.NaT


def test_string_table_view_creation():
    data = ["hi"] * 25 + [None] * 2027
    psr = pd.Series(data)
    gsr = cudf.Series(psr)

    expect = psr[:1]
    got = gsr[:1]

    assert_eq(expect, got)


def test_string_slice_with_mask():
    actual = cudf.Series(["hi", "hello", None])
    expected = actual[0:3]

    assert actual._column.size == 3
    assert_eq(actual._column.size, expected._column.size)
    assert_eq(actual._column.null_count, expected._column.null_count)

    assert_eq(actual, expected)


def test_categorical_masking():
    """
    Test common operation for getting a all rows that matches a certain
    category.
    """
    cat = pd.Categorical(["a", "a", "b", "c", "a"], categories=["a", "b", "c"])
    pdsr = pd.Series(cat)
    sr = cudf.Series(cat)

    # check scalar comparison
    expect_matches = pdsr == "a"
    got_matches = sr == "a"

    np.testing.assert_array_equal(
        expect_matches.values, got_matches.to_numpy()
    )

    # mask series
    expect_masked = pdsr[expect_matches]
    got_masked = sr[got_matches]

    assert len(expect_masked) == len(got_masked)
    assert got_masked.null_count == 0
    assert_eq(got_masked, expect_masked)


@pytest.mark.parametrize(
    "i1, i2, i3",
    (
        [
            (slice(None, 12), slice(3, None), slice(None, None, 2)),
            (range(12), range(3, 12), range(0, 9, 2)),
            (np.arange(12), np.arange(3, 12), np.arange(0, 9, 2)),
            (list(range(12)), list(range(3, 12)), list(range(0, 9, 2))),
            (
                pd.Series(range(12)),
                pd.Series(range(3, 12)),
                pd.Series(range(0, 9, 2)),
            ),
            (
                cudf.Series(range(12)),
                cudf.Series(range(3, 12)),
                cudf.Series(range(0, 9, 2)),
            ),
            (
                [i in range(12) for i in range(20)],
                [i in range(3, 12) for i in range(12)],
                [i in range(0, 9, 2) for i in range(9)],
            ),
            (
                np.array([i in range(12) for i in range(20)], dtype=bool),
                np.array([i in range(3, 12) for i in range(12)], dtype=bool),
                np.array([i in range(0, 9, 2) for i in range(9)], dtype=bool),
            ),
        ]
    ),
    ids=(
        [
            "slice",
            "range",
            "numpy.array",
            "list",
            "pandas.Series",
            "Series",
            "list[bool]",
            "numpy.array[bool]",
        ]
    ),
)
def test_series_indexing(i1, i2, i3):
    a1 = np.arange(20)
    series = cudf.Series(a1)

    # Indexing
    sr1 = series.iloc[i1]
    assert sr1.null_count == 0
    np.testing.assert_equal(sr1.to_numpy(), a1[:12])

    sr2 = sr1.iloc[i2]
    assert sr2.null_count == 0
    np.testing.assert_equal(sr2.to_numpy(), a1[3:12])

    # Index with stride
    sr3 = sr2.iloc[i3]
    assert sr3.null_count == 0
    np.testing.assert_equal(sr3.to_numpy(), a1[3:12:2])

    # Integer indexing
    if isinstance(i1, range):
        for i in i1:  # Python int-s
            assert series[i] == a1[i]
    if isinstance(i1, np.ndarray) and i1.dtype == "i":
        for i in i1:  # numpy integers
            assert series[i] == a1[i]


@pytest.mark.parametrize(
    "arg",
    [
        1,
        -1,
        np.int32(1),
        np.uint32(1),
        np.int8(1),
        np.uint8(1),
        np.int16(1),
        np.uint16(1),
        np.int64(1),
        np.uint64(1),
    ],
)
def test_series_get_item_integer_label_non_integer_index_raises(arg):
    # Indexing for non-numeric dtype Index
    ps = pd.Series([1, 2, 3], index=pd.Index(["a", "b", "c"]))
    gs = cudf.from_pandas(ps)

    with pytest.raises(KeyError):
        ps[arg]
    with pytest.raises(KeyError):
        gs[arg]


def test_series_indexing_large_size():
    n_elem = 100_000
    gsr = cudf.Series(cp.ones(n_elem))
    gsr[0] = None
    got = gsr[gsr.isna()]
    expect = cudf.Series([None], dtype="float64")

    assert_eq(expect, got)


@pytest.mark.parametrize("psr", [pd.Series([1, 2, 3], index=["a", "b", "c"])])
@pytest.mark.parametrize(
    "arg", ["b", ["a", "c"], slice(1, 2, 1), [True, False, True]]
)
def test_series_get_item(psr, arg):
    gsr = cudf.from_pandas(psr)

    expect = psr[arg]
    got = gsr[arg]

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4],
        [1.0, 2.0, 3.0, 4.0],
        ["one", "two", "three", "four"],
        pd.Series(["a", "b", "c", "d"], dtype="category"),
        pd.Series(pd.date_range("2010-01-01", "2010-01-04")),
    ],
)
@pytest.mark.parametrize(
    "mask_vals",
    [
        [True, True, True, True],
        [False, False, False, False],
        [True, False, True, False],
        [True, False, False, True],
    ],
)
@pytest.mark.parametrize(
    "mask_class", [list, np.array, pd.Series, cudf.Series]
)
@pytest.mark.parametrize("nulls", ["one", "some", "all", "none"])
def test_series_apply_boolean_mask(data, mask_vals, mask_class, nulls):
    rng = np.random.default_rng(seed=0)
    psr = pd.Series(data)

    if len(data) > 0:
        if nulls == "one":
            p = rng.integers(0, 4)
            psr[p] = None
        elif nulls == "some":
            p1, p2 = rng.integers(0, 4, (2,))
            psr[p1] = None
            psr[p2] = None
        elif nulls == "all":
            psr[:] = None

    gsr = cudf.from_pandas(psr)

    # TODO: from_pandas(psr) has dtype "float64"
    # when psr has dtype "object" and is all None
    if psr.dtype == "object" and nulls == "all":
        gsr = cudf.Series([None, None, None, None], dtype="object")

    mask = mask_class(mask_vals)
    if isinstance(mask, cudf.Series):
        expect = psr[mask.to_pandas()]
    else:
        expect = psr[mask]
    got = gsr[mask]

    assert_eq(expect, got)


@pytest.mark.parametrize("key", [5, -10, "0", "a", np.array(5), np.array("a")])
def test_loc_bad_key_type(key):
    psr = pd.Series([1, 2, 3])
    gsr = cudf.from_pandas(psr)
    assert_exceptions_equal(lambda: psr[key], lambda: gsr[key])
    assert_exceptions_equal(lambda: psr.loc[key], lambda: gsr.loc[key])


@pytest.mark.parametrize("key", ["b", 1.0, np.array("b")])
def test_loc_bad_key_type_string_index(key):
    psr = pd.Series([1, 2, 3], index=["a", "1", "c"])
    gsr = cudf.from_pandas(psr)
    assert_exceptions_equal(lambda: psr[key], lambda: gsr[key])
    assert_exceptions_equal(lambda: psr.loc[key], lambda: gsr.loc[key])


def test_loc_zero_dim_array():
    psr = pd.Series([1, 2, 3])
    gsr = cudf.from_pandas(psr)

    assert_eq(psr[np.array(0)], gsr[np.array(0)])
    assert_eq(psr[np.array([0])[0]], gsr[np.array([0])[0]])

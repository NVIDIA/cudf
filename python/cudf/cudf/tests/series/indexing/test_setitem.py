# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import decimal
import operator

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal


@pytest.mark.parametrize(
    "arg", ["b", ["a", "c"], slice(1, 2, 1), [True, False, True]]
)
def test_series_set_item(arg):
    psr = pd.Series([1, 2, 3], index=["a", "b", "c"])
    gsr = cudf.from_pandas(psr)

    psr[arg] = 11
    gsr[arg] = 11

    assert_eq(psr, gsr)


def test_column_set_unequal_length_object_by_mask():
    data = [1, 2, 3, 4, 5]
    replace_data_1 = [8, 9]
    replace_data_2 = [8, 9, 10, 11]
    mask = [True, True, False, True, False]

    psr = pd.Series(data)
    gsr = cudf.Series(data)
    assert_exceptions_equal(
        psr.__setitem__,
        gsr.__setitem__,
        ([mask, replace_data_1], {}),
        ([mask, replace_data_1], {}),
    )

    psr = pd.Series(data)
    gsr = cudf.Series(data)
    assert_exceptions_equal(
        psr.__setitem__,
        gsr.__setitem__,
        ([mask, replace_data_2], {}),
        ([mask, replace_data_2], {}),
    )


def test_categorical_setitem_invalid():
    ps = pd.Series([1, 2, 3], dtype="category")
    gs = cudf.Series([1, 2, 3], dtype="category")

    assert_exceptions_equal(
        lfunc=ps.__setitem__,
        rfunc=gs.__setitem__,
        lfunc_args_and_kwargs=([0, 5], {}),
        rfunc_args_and_kwargs=([0, 5], {}),
    )


def test_series_slice_setitem_list():
    actual = cudf.Series([[[1, 2], [2, 3]], [[3, 4]], [[4, 5]], [[6, 7]]])
    actual[slice(0, 3, 1)] = [[10, 11], [12, 23]]
    expected = cudf.Series(
        [
            [[10, 11], [12, 23]],
            [[10, 11], [12, 23]],
            [[10, 11], [12, 23]],
            [[6, 7]],
        ]
    )
    assert_eq(actual, expected)


def test_series_slice_setitem_struct():
    actual = cudf.Series(
        [
            {"a": {"b": 10}, "b": 11},
            {"a": {"b": 100}, "b": 5},
            {"a": {"b": 50}, "b": 2},
            {"a": {"b": 1000}, "b": 67},
            {"a": {"b": 4000}, "b": 1090},
        ]
    )
    actual[slice(0, 3, 1)] = {"a": {"b": 5050}, "b": 101}
    expected = cudf.Series(
        [
            {"a": {"b": 5050}, "b": 101},
            {"a": {"b": 5050}, "b": 101},
            {"a": {"b": 5050}, "b": 101},
            {"a": {"b": 1000}, "b": 67},
            {"a": {"b": 4000}, "b": 1090},
        ]
    )
    assert_eq(actual, expected)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32])
@pytest.mark.parametrize("indices", [0, [1, 2]])
def test_series_setitem_upcasting_raises(dtype, indices):
    sr = pd.Series([0, 0, 0], dtype=dtype)
    cr = cudf.from_pandas(sr)
    new_value = np.float64(np.pi)
    with pytest.raises(TypeError):
        sr[indices] = new_value
    with pytest.raises(TypeError):
        cr[indices] = new_value


@pytest.mark.parametrize(
    "klass",
    [
        list,
        cudf.Series,
        lambda x: cudf.Series(x, index=[2, 3, 4, 5, 6]),
    ],
)
def test_series_set_equal_length_object_by_mask(klass):
    replace_data = klass([100, 200, 300, 400, 500])
    psr = pd.Series([1, 2, 3, 4, 5], dtype="Int64")
    gsr = cudf.from_pandas(psr)

    # Lengths match in trivial case
    pd_bool_col = pd.Series([True] * len(psr), dtype="boolean")
    gd_bool_col = cudf.from_pandas(pd_bool_col)
    psr[pd_bool_col] = (
        replace_data.to_pandas(nullable=True)
        if hasattr(replace_data, "to_pandas")
        else pd.Series(replace_data)
    )
    gsr[gd_bool_col] = replace_data

    assert_eq(psr.astype("float"), gsr.astype("float"))

    # Test partial masking
    psr[psr > 1] = (
        replace_data.to_pandas()
        if hasattr(replace_data, "to_pandas")
        else pd.Series(replace_data)
    )
    gsr[gsr > 1] = replace_data

    assert_eq(psr.astype("float"), gsr.astype("float"))


# TODO: these two tests could perhaps be changed once specifics of
# pandas compat wrt upcasting are decided on; this is just baking in
# status-quo.
def test_series_setitem_upcasting_string_column():
    sr = pd.Series([0, 0, 0], dtype=str)
    cr = cudf.from_pandas(sr)
    new_value = np.float64(10.5)
    sr[0] = str(new_value)
    cr[0] = str(new_value)
    assert_eq(sr, cr)


def test_series_setitem_upcasting_string_value():
    sr = cudf.Series([0, 0, 0], dtype=int)
    # Matches pandas: assigning a string scalar to a numeric column raises.
    with pytest.raises(TypeError, match="Invalid value"):
        sr[0] = "10"
    with pytest.raises(TypeError, match="Invalid value"):
        sr[0] = "non-integer"


def test_scatter_by_slice_with_start_and_step():
    source = pd.Series([1, 2, 3, 4, 5])
    csource = cudf.from_pandas(source)
    target = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    ctarget = cudf.from_pandas(target)
    target[1::2] = source
    ctarget[1::2] = csource
    assert_eq(target, ctarget)


@pytest.mark.parametrize("n", [1, 3])
def test_setitem_str_trailing_null(n):
    trailing_nulls = "\x00" * n
    s = cudf.Series(["a", "b", "c" + trailing_nulls])
    assert s[2] == "c" + trailing_nulls
    s[0] = "a" + trailing_nulls
    assert s[0] == "a" + trailing_nulls
    s[1] = trailing_nulls
    assert s[1] == trailing_nulls
    s[0] = ""
    assert s[0] == ""
    s[0] = "\x00"
    assert s[0] == "\x00"


@pytest.mark.parametrize(
    "ps",
    [
        pd.Series([1, 2, 3], index=pd.RangeIndex(0, 3)),
        pd.Series([1, 2, 3], index=pd.RangeIndex(start=2, stop=-1, step=-1)),
        pd.Series([1, 2, 3], index=pd.RangeIndex(start=1, stop=6, step=2)),
        pd.Series(
            [1, 2, 3, 4, 5], index=pd.RangeIndex(start=1, stop=-9, step=-2)
        ),
        pd.Series(
            [1, 2, 3, 4, 5], index=pd.RangeIndex(start=1, stop=-12, step=-3)
        ),
        pd.Series([1, 2, 3, 4], index=pd.RangeIndex(start=1, stop=14, step=4)),
        pd.Series(
            [1, 2, 3, 4], index=pd.RangeIndex(start=1, stop=-14, step=-4)
        ),
    ],
)
@pytest.mark.parametrize("arg", [[1], 5.6, 3.1])
def test_series_set_item_range_index(ps, arg):
    gsr = cudf.from_pandas(ps)
    psr = ps.copy(deep=True)
    psr[arg] = 11
    gsr[arg] = 11

    assert_eq(psr, gsr, check_index_type=True)


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/7448")
def test_iloc_setitem_7448():
    index = pd.MultiIndex.from_product([(1, 2), (3, 4)])
    expect = cudf.Series([1, 2, 3, 4], index=index)
    actual = cudf.from_pandas(expect)
    expect[(1, 3)] = 101
    actual[(1, 3)] = 101
    assert_eq(expect, actual)


@pytest.mark.parametrize(
    "data,item",
    [
        (
            # basic list into a list column
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [0, 0, 0],
        ),
        (
            # nested list into nested list column
            [
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]],
            ],
            [[0, 0, 0], [0, 0, 0]],
        ),
        (
            # NA into a list column
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            pd.NA,
        ),
        (
            # NA into nested list column
            [
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]],
                [[1, 2, 3], [4, 5, 6]],
            ],
            pd.NA,
        ),
    ],
)
def test_listcol_setitem(data, item):
    sr = cudf.Series(data)

    sr[1] = item
    data[1] = item
    expect = cudf.Series(data)

    assert_eq(expect, sr)


@pytest.mark.parametrize(
    "data,item,error_msg,error_type",
    [
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[1, 2, 3], [4, 5, 6]],
            "Could not convert .* with type list: tried to convert to int64",
            pa.ArrowInvalid,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            0,
            "Can not set 0 into ListColumn",
            ValueError,
        ),
    ],
)
def test_listcol_setitem_error_cases(data, item, error_msg, error_type):
    sr = cudf.Series(data)
    with pytest.raises(error_type, match=error_msg):
        sr[1] = item


def test_fill_new_category():
    gs = cudf.Series(pd.Categorical(["a", "b", "c"]))
    with pytest.raises(TypeError):
        gs[0:1] = "d"


@pytest.mark.parametrize("dtype", ["int64", "float64"])
@pytest.mark.parametrize("bool_scalar", [True, False])
def test_set_bool_error(dtype, bool_scalar):
    sr = cudf.Series([1, 2, 3], dtype=dtype)
    psr = sr.to_pandas(nullable=True)

    assert_exceptions_equal(
        lfunc=sr.__setitem__,
        rfunc=psr.__setitem__,
        lfunc_args_and_kwargs=([bool_scalar],),
        rfunc_args_and_kwargs=([bool_scalar],),
    )


@pytest.mark.parametrize(
    "data", [[0, 1, 2], ["a", "b", "c"], [0.324, 32.32, 3243.23]]
)
def test_series_setitem_nat_with_non_datetimes(data):
    s = cudf.Series(data)
    with pytest.raises(TypeError):
        s[0] = cudf.NaT


def test_series_string_setitem():
    gs = cudf.Series(["abc", "def", "ghi", "xyz", "pqr"])
    ps = gs.to_pandas()

    gs[0] = "NaT"
    gs[1] = "NA"
    gs[2] = "<NA>"
    gs[3] = "NaN"

    ps[0] = "NaT"
    ps[1] = "NA"
    ps[2] = "<NA>"
    ps[3] = "NaN"

    assert_eq(gs, ps)


def test_series_error_nan_non_float_dtypes():
    s = cudf.Series(["a", "b", "c"])
    with pytest.raises(TypeError):
        s[0] = np.nan

    s = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    with pytest.raises(TypeError):
        s[0] = np.nan


def test_series_setitem_mixed_bool_dtype():
    s = cudf.Series([True, False, True])
    with pytest.raises(TypeError):
        s[0] = 10


@pytest.mark.parametrize(
    "data, item",
    [
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": cudf.NA},
                {"a": "abcde", "b": [4, 5, 6], "c": 9},
            ],
            {"a": "Hello world", "b": [], "c": cudf.NA},
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": cudf.NA},
                {"a": "abcde", "b": [4, 5, 6], "c": 9},
            ],
            {},
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": cudf.NA},
                {"a": "abcde", "b": [4, 5, 6], "c": 9},
            ],
            cudf.NA,
        ),
        (
            [
                {"a": "Hello world", "b": []},
                {"a": "CUDF", "b": [1, 2, 3], "c": cudf.NA},
                {"a": "abcde", "b": [4, 5, 6], "c": 9},
            ],
            {"a": "Second element", "b": [1, 2], "c": 1000},
        ),
    ],
)
def test_struct_setitem(data, item):
    sr = cudf.Series(data)
    sr[1] = item
    data[1] = item
    expected = cudf.Series(data)
    assert sr.to_arrow() == expected.to_arrow()


def test_null_copy():
    col = cudf.Series(range(2049))
    col[:] = None
    assert len(col) == 2049


def test_series_setitem_cow():
    actual = cudf.Series([1, 2, 3, 4, 5])
    new_copy = actual.copy(deep=False)

    actual[1] = 100
    assert_eq(actual, cudf.Series([1, 100, 3, 4, 5]))
    assert_eq(new_copy, cudf.Series([1, 2, 3, 4, 5]))


def test_series_setitem_both_slice():
    actual = cudf.Series([1, 2, 3, 4, 5])
    new_copy = actual.copy(deep=False)

    actual[slice(0, 2, 1)] = 100
    assert_eq(actual, cudf.Series([100, 100, 3, 4, 5]))
    assert_eq(new_copy, cudf.Series([1, 2, 3, 4, 5]))

    new_copy[slice(2, 4, 1)] = 300
    assert_eq(actual, cudf.Series([100, 100, 3, 4, 5]))
    assert_eq(new_copy, cudf.Series([1, 2, 300, 300, 5]))


def test_series_setitem_partial_slice():
    actual = cudf.Series([1, 2, 3, 4, 5])
    new_copy = actual.copy(deep=False)

    new_copy[slice(2, 4, 1)] = 300
    assert_eq(actual, cudf.Series([1, 2, 3, 4, 5]))
    assert_eq(new_copy, cudf.Series([1, 2, 300, 300, 5]))

    new_slice = actual[2:]
    assert new_slice._column.data.owner == actual._column.data.owner
    new_slice[0:2] = 10
    assert_eq(new_slice, cudf.Series([10, 10, 5], index=[2, 3, 4]))
    assert_eq(actual, cudf.Series([1, 2, 3, 4, 5]))


def test_multiple_series_cow():
    # Verify constructing, modifying, deleting
    # multiple copies of a series preserves
    # the data appropriately when COW is enabled.
    s = cudf.Series([10, 20, 30, 40, 50])
    s1 = s.copy(deep=False)
    s2 = s.copy(deep=False)
    s3 = s.copy(deep=False)
    s4 = s2.copy(deep=False)
    s5 = s4.copy(deep=False)
    s6 = s3.copy(deep=False)

    s1[0:3] = 10000
    # s1 will be unlinked from actual data in s,
    # and then modified. Rest all should
    # contain the original data.
    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    for ser in [s, s2, s3, s4, s5, s6]:
        assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

    s6[0:3] = 3000
    # s6 will be unlinked from actual data in s,
    # and then modified. Rest all should
    # contain the original data.
    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    for ser in [s2, s3, s4, s5]:
        assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

    s2[1:4] = 4000
    # s2 will be unlinked from actual data in s,
    # and then modified. Rest all should
    # contain the original data.
    assert_eq(s2, cudf.Series([10, 4000, 4000, 4000, 50]))
    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    for ser in [s3, s4, s5]:
        assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

    s4[2:4] = 5000
    # s4 will be unlinked from actual data in s,
    # and then modified. Rest all should
    # contain the original data.
    assert_eq(s4, cudf.Series([10, 20, 5000, 5000, 50]))
    assert_eq(s2, cudf.Series([10, 4000, 4000, 4000, 50]))
    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    for ser in [s3, s5]:
        assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

    s5[2:4] = 6000
    # s5 will be unlinked from actual data in s,
    # and then modified. Rest all should
    # contain the original data.
    assert_eq(s5, cudf.Series([10, 20, 6000, 6000, 50]))
    assert_eq(s4, cudf.Series([10, 20, 5000, 5000, 50]))
    assert_eq(s2, cudf.Series([10, 4000, 4000, 4000, 50]))
    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    for ser in [s3]:
        assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

    s7 = s5.copy(deep=False)
    assert_eq(s7, cudf.Series([10, 20, 6000, 6000, 50]))
    s7[1:3] = 55
    # Making a copy of s5, i.e., s7 and modifying shouldn't
    # be touching/modifying data in other series.
    assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))

    assert_eq(s4, cudf.Series([10, 20, 5000, 5000, 50]))
    assert_eq(s2, cudf.Series([10, 4000, 4000, 4000, 50]))
    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    for ser in [s3]:
        assert_eq(ser, cudf.Series([10, 20, 30, 40, 50]))

    # Deleting any of the following series objects
    # shouldn't delete rest of the weekly referenced data
    # elsewhere.

    del s2

    assert_eq(s1, cudf.Series([10000, 10000, 10000, 40, 50]))
    assert_eq(s3, cudf.Series([10, 20, 30, 40, 50]))
    assert_eq(s4, cudf.Series([10, 20, 5000, 5000, 50]))
    assert_eq(s5, cudf.Series([10, 20, 6000, 6000, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))

    del s4
    del s1

    assert_eq(s3, cudf.Series([10, 20, 30, 40, 50]))
    assert_eq(s5, cudf.Series([10, 20, 6000, 6000, 50]))
    assert_eq(s6, cudf.Series([3000, 3000, 3000, 40, 50]))
    assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))

    del s
    del s6

    assert_eq(s3, cudf.Series([10, 20, 30, 40, 50]))
    assert_eq(s5, cudf.Series([10, 20, 6000, 6000, 50]))
    assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))

    del s5

    assert_eq(s3, cudf.Series([10, 20, 30, 40, 50]))
    assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))

    del s3
    assert_eq(s7, cudf.Series([10, 55, 55, 6000, 50]))


def test_series_zero_copy_cow():
    s = cudf.Series([1, 2, 3, 4, 5])
    s1 = s.copy(deep=False)
    cp_array = cp.asarray(s)

    # Ensure all original data & zero-copied
    # data is same.
    assert_eq(s, cudf.Series([1, 2, 3, 4, 5]))
    assert_eq(s1, cudf.Series([1, 2, 3, 4, 5]))
    assert_eq(cp_array, cp.array([1, 2, 3, 4, 5]))

    cp_array[0:3] = 10

    assert_eq(s.to_numpy(), np.array([10, 10, 10, 4, 5]))
    assert_eq(s, cudf.Series([10, 10, 10, 4, 5]))
    assert_eq(s1, cudf.Series([10, 10, 10, 4, 5]))
    assert_eq(cp_array, cp.array([10, 10, 10, 4, 5]))

    s2 = cudf.Series(cp_array)
    assert_eq(s2, cudf.Series([10, 10, 10, 4, 5]))

    s3 = s2.copy(deep=False)
    cp_array[0] = 20

    assert_eq(s, cudf.Series([20, 10, 10, 4, 5]))
    assert_eq(s1, cudf.Series([20, 10, 10, 4, 5]))
    assert_eq(cp_array, cp.array([20, 10, 10, 4, 5]))
    assert_eq(s2, cudf.Series([20, 10, 10, 4, 5]))
    assert_eq(s3, cudf.Series([20, 10, 10, 4, 5]))

    s4 = cudf.Series([10, 20, 30, 40, 50])
    s5 = cudf.Series(s4)
    assert_eq(s5, cudf.Series([10, 20, 30, 40, 50]))
    s5[0:2] = 1
    # With CoW, modifying `s5` should NOT modify `s4`.
    assert_eq(s5, cudf.Series([1, 1, 30, 40, 50]))
    assert_eq(s4, cudf.Series([10, 20, 30, 40, 50]))


def test_series_str_copy():
    s = cudf.Series(["a", "b", "c", "d", "e"])
    s1 = s.copy(deep=True)
    s2 = s.copy(deep=True)

    assert_eq(s, cudf.Series(["a", "b", "c", "d", "e"]))
    assert_eq(s1, cudf.Series(["a", "b", "c", "d", "e"]))
    assert_eq(s2, cudf.Series(["a", "b", "c", "d", "e"]))

    s[0:3] = "abc"

    assert_eq(s, cudf.Series(["abc", "abc", "abc", "d", "e"]))
    assert_eq(s1, cudf.Series(["a", "b", "c", "d", "e"]))
    assert_eq(s2, cudf.Series(["a", "b", "c", "d", "e"]))

    s2[1:4] = "xyz"

    assert_eq(s, cudf.Series(["abc", "abc", "abc", "d", "e"]))
    assert_eq(s1, cudf.Series(["a", "b", "c", "d", "e"]))
    assert_eq(s2, cudf.Series(["a", "xyz", "xyz", "xyz", "e"]))


def test_series_cat_copy():
    s = cudf.Series([10, 20, 30, 40, 50], dtype="category")
    s1 = s.copy(deep=True)
    s2 = s1.copy(deep=True)
    s3 = s1.copy(deep=True)

    s[0] = 50
    assert_eq(s, cudf.Series([50, 20, 30, 40, 50], dtype=s.dtype))
    assert_eq(s1, cudf.Series([10, 20, 30, 40, 50], dtype="category"))
    assert_eq(s2, cudf.Series([10, 20, 30, 40, 50], dtype="category"))
    assert_eq(s3, cudf.Series([10, 20, 30, 40, 50], dtype="category"))

    s2[3] = 10
    s3[2:5] = 20
    assert_eq(s, cudf.Series([50, 20, 30, 40, 50], dtype=s.dtype))
    assert_eq(s1, cudf.Series([10, 20, 30, 40, 50], dtype=s.dtype))
    assert_eq(s2, cudf.Series([10, 20, 30, 10, 50], dtype=s.dtype))
    assert_eq(s3, cudf.Series([10, 20, 20, 20, 20], dtype=s.dtype))


@pytest.mark.parametrize(
    "data, dtype, item, to, expect",
    [
        # scatter to a single index
        (
            ["1", "2", "3"],
            cudf.Decimal64Dtype(1, 0),
            decimal.Decimal(5),
            1,
            ["1", "5", "3"],
        ),
        (
            ["1.5", "2.5", "3.5"],
            cudf.Decimal64Dtype(2, 1),
            decimal.Decimal("5.5"),
            1,
            ["1.5", "5.5", "3.5"],
        ),
        (
            ["1.0042", "2.0042", "3.0042"],
            cudf.Decimal64Dtype(5, 4),
            decimal.Decimal("5.0042"),
            1,
            ["1.0042", "5.0042", "3.0042"],
        ),
        # scatter via boolmask
        (
            ["1", "2", "3"],
            cudf.Decimal64Dtype(1, 0),
            decimal.Decimal(5),
            [True, False, True],
            ["5", "2", "5"],
        ),
        (
            ["1.5", "2.5", "3.5"],
            cudf.Decimal64Dtype(2, 1),
            decimal.Decimal("5.5"),
            [True, True, True],
            ["5.5", "5.5", "5.5"],
        ),
        (
            ["1.0042", "2.0042", "3.0042"],
            cudf.Decimal64Dtype(5, 4),
            decimal.Decimal("5.0042"),
            [False, False, True],
            ["1.0042", "2.0042", "5.0042"],
        ),
        # We will allow assigning a decimal with less precision
        (
            ["1.00", "2.00", "3.00"],
            cudf.Decimal64Dtype(3, 2),
            decimal.Decimal(5),
            1,
            ["1.00", "5.00", "3.00"],
        ),
        # But not truncation
        (
            ["1", "2", "3"],
            cudf.Decimal64Dtype(1, 0),
            decimal.Decimal("5.5"),
            1,
            pa.ArrowInvalid,
        ),
        # We will allow for setting scalars into decimal columns
        (["1", "2", "3"], cudf.Decimal64Dtype(1, 0), 5, 1, ["1", "5", "3"]),
        # But not if it has too many digits to fit the precision
        (["1", "2", "3"], cudf.Decimal64Dtype(1, 0), 50, 1, pa.ArrowInvalid),
    ],
)
def test_series_setitem_decimal(data, dtype, item, to, expect):
    data = cudf.Series([decimal.Decimal(x) for x in data], dtype=dtype)

    if expect is pa.ArrowInvalid:
        with pytest.raises(expect):
            data[to] = item
        return
    else:
        expect = cudf.Series([decimal.Decimal(x) for x in expect], dtype=dtype)
        data[to] = item
        assert_eq(data, expect)


def test_categorical_setitem_with_na():
    gs = cudf.Series(
        [1, 2, np.nan, 10, np.nan, None], nan_as_null=False
    ).astype("category")
    gs[[1, 3]] = None

    expected_series = cudf.Series(
        [1, np.nan, np.nan, np.nan, np.nan, None], nan_as_null=False
    ).astype(gs.dtype)
    assert_eq(gs, expected_series)


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
def test_series_setitem_basics(key, value, nulls):
    psr = pd.Series([1, 2, 3, 4, 5])
    if nulls == "some":
        psr[[0, 4]] = None
    elif nulls == "all":
        psr[:] = None
    gsr = cudf.from_pandas(psr)
    psr[key] = value
    gsr[key] = value
    assert_eq(psr, gsr, check_dtype=False)


def test_series_setitem_empty_list_raises():
    psr = pd.Series([1, 2, 3, 4, 5])
    gsr = cudf.Series([1, 2, 3, 4, 5])
    with pytest.raises(TypeError):
        psr[[]] = []
    with pytest.raises(TypeError):
        gsr[[]] = []


def test_series_setitem_null():
    gsr = cudf.Series([1, 2, 3, 4])
    gsr[0] = None

    expect = cudf.Series([None, 2, 3, 4])
    got = gsr
    assert_eq(expect, got)

    gsr = cudf.Series([None, 2, 3, 4])
    gsr[0] = 1

    expect = cudf.Series([1, 2, 3, 4])
    got = gsr
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "key, value",
    [
        ([0, 1], [0.5, 2.5]),
        (slice(0, 2), [0.5, 0.25]),
    ],
)
def test_series_setitem_upcasting_list_like_raises(key, value):
    psr = pd.Series([1, 2, 3], dtype="int32")
    gsr = cudf.from_pandas(psr)

    with pytest.raises(TypeError):
        psr[key] = value
    with pytest.raises(TypeError):
        gsr[key] = value


def test_series_setitem_datetime():
    psr = pd.Series(["2001", "2002", "2003"], dtype="datetime64[ns]")
    gsr = cudf.from_pandas(psr)

    psr[0] = np.datetime64("2005")
    gsr[0] = np.datetime64("2005")

    assert_eq(psr, gsr)


def test_series_setitem_datetime_coerced():
    psr = pd.Series(["2001", "2002", "2003"], dtype="datetime64[ns]")
    gsr = cudf.from_pandas(psr)

    psr[0] = "2005"
    gsr[0] = "2005"

    assert_eq(psr, gsr)


def test_series_setitem_categorical():
    psr = pd.Series(["a", "b", "a", "c", "d"], dtype="category")
    gsr = cudf.from_pandas(psr)

    psr[0] = "d"
    gsr[0] = "d"
    assert_eq(psr, gsr)

    psr = psr.cat.add_categories(["e"])
    gsr = gsr.cat.add_categories(["e"])
    psr[0] = "e"
    gsr[0] = "e"
    assert_eq(psr, gsr)

    psr[[0, 1]] = "b"
    gsr[[0, 1]] = "b"
    assert_eq(psr, gsr)

    psr[0:3] = "e"
    gsr[0:3] = "e"
    assert_eq(psr, gsr)


@pytest.mark.parametrize(
    "key, value",
    [
        (0, "d"),
        (0, "g"),
        ([0, 1], "g"),
        ([0, 1], None),
        (slice(None, 2), "g"),
        (slice(None, 2), ["g", None]),
    ],
)
def test_series_setitem_string(key, value):
    psr = pd.Series(["a", "b", "c", "d", "e"])
    gsr = cudf.from_pandas(psr)
    psr[key] = value
    gsr[key] = value
    assert_eq(psr, gsr)

    psr = pd.Series(["a", None, "c", "d", "e"])
    gsr = cudf.from_pandas(psr)
    psr[key] = value
    gsr[key] = value
    assert_eq(psr, gsr)


def test_out_of_bounds_indexing():
    psr = pd.Series([1, 2, 3])
    gsr = cudf.from_pandas(psr)

    assert_exceptions_equal(
        lambda: psr[[0, 1, 9]],
        lambda: gsr[[0, 1, 9]],
    )
    assert_exceptions_equal(
        lambda: psr[[0, 1, -4]],
        lambda: gsr[[0, 1, -4]],
    )
    assert_exceptions_equal(
        lambda: psr.__setitem__([0, 1, 9], 2),
        lambda: gsr.__setitem__([0, 1, 9], 2),
    )
    assert_exceptions_equal(
        lambda: psr.__setitem__([0, 1, -4], 2),
        lambda: gsr.__setitem__([0, 1, -4], 2),
    )


@pytest.mark.parametrize(
    "dtype",
    ["datetime64[ns]", "datetime64[us]", "timedelta64[ns]", "timedelta64[us]"],
)
@pytest.mark.parametrize("value", [0, 1, 0.0, 1.5])
def test_temporal_setitem_numeric_scalar_pandas_compat(dtype, value):
    # cuDF must reject scalar int/float assignment into datetime/timedelta
    # columns under pandas-compatible mode (pyarrow would silently coerce
    # them to epoch values).
    psr = pd.Series([1, 2, 3], dtype=dtype)
    gsr = cudf.from_pandas(psr)
    with cudf.option_context("mode.pandas_compatible", True):
        assert_exceptions_equal(
            lfunc=psr.__setitem__,
            rfunc=gsr.__setitem__,
            lfunc_args_and_kwargs=([0, value], {}),
            rfunc_args_and_kwargs=([0, value], {}),
        )


@pytest.mark.parametrize(
    "dtype",
    ["datetime64[ns]", "datetime64[us]", "timedelta64[ns]", "timedelta64[us]"],
)
def test_temporal_setitem_numeric_array_pandas_compat(dtype):
    # cuDF must reject numeric array assignment into temporal columns
    # under pandas-compatible mode.
    psr = pd.Series([1, 2, 3], dtype=dtype)
    gsr = cudf.from_pandas(psr)
    with cudf.option_context("mode.pandas_compatible", True):
        assert_exceptions_equal(
            lfunc=psr.__setitem__,
            rfunc=gsr.__setitem__,
            lfunc_args_and_kwargs=([[0, 1], pd.Series([0, 1])], {}),
            rfunc_args_and_kwargs=([[0, 1], cudf.Series([0, 1])], {}),
        )


@pytest.mark.parametrize("value", [None, np.nan, pd.NA])
def test_bool_setitem_null_pandas_compat(value):
    # bool dtype cannot hold NA without dtype change; pandas raises
    # TypeError when assigning a null-like into a bool column.
    psr = pd.Series([True, False, True])
    gsr = cudf.from_pandas(psr)
    with cudf.option_context("mode.pandas_compatible", True):
        assert_exceptions_equal(
            lfunc=psr.__setitem__,
            rfunc=gsr.__setitem__,
            lfunc_args_and_kwargs=([0, value], {}),
            rfunc_args_and_kwargs=([0, value], {}),
        )


@pytest.mark.parametrize("value", [1, 1.5, True, [1, 2]])
def test_string_setitem_non_string_pandas_compat(value):
    # pandas string dtype rejects non-string values (int/float/bool/etc.)
    # cuDF must raise TypeError to match.
    psr = pd.Series(["a", "b", "c"], dtype="str")
    gsr = cudf.from_pandas(psr)
    with cudf.option_context("mode.pandas_compatible", True):
        if isinstance(value, list):
            assert_exceptions_equal(
                lfunc=psr.__setitem__,
                rfunc=gsr.__setitem__,
                lfunc_args_and_kwargs=([[0, 1], value], {}),
                rfunc_args_and_kwargs=([[0, 1], value], {}),
            )
        else:
            assert_exceptions_equal(
                lfunc=psr.__setitem__,
                rfunc=gsr.__setitem__,
                lfunc_args_and_kwargs=([0, value], {}),
                rfunc_args_and_kwargs=([0, value], {}),
            )


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64"])
def test_int_setitem_lossy_float_array_pandas_compat(dtype):
    # pandas rejects float→int array assignment when values can't be
    # losslessly cast (have non-zero fractional parts).
    psr = pd.Series([10, 11, 12], dtype=dtype)
    gsr = cudf.from_pandas(psr)
    with cudf.option_context("mode.pandas_compatible", True):
        assert_exceptions_equal(
            lfunc=psr.__setitem__,
            rfunc=gsr.__setitem__,
            lfunc_args_and_kwargs=(
                [[0, 1], pd.Series([0.5, 1.5], index=[0, 1])],
                {},
            ),
            rfunc_args_and_kwargs=(
                [[0, 1], cudf.Series([0.5, 1.5], index=[0, 1])],
                {},
            ),
        )


@pytest.mark.parametrize(
    "value",
    [
        np.datetime64("NaT", "ns"),
        np.timedelta64("NaT", "ns"),
        [np.timedelta64("NaT", "ns"), np.timedelta64("NaT", "ns")],
        [np.datetime64("2020-01-01", "ns")] * 2,
    ],
)
def test_series_setitem_temporal_into_numeric_raises(value):
    # Assigning a datetime64/timedelta64 scalar or array into a numeric
    # Series is invalid (exercises both the scalar and array branches of
    # NumericalColumn._cast_setitem_value).
    key = slice(0, 2) if isinstance(value, list) else 0
    ps = pd.Series([1, 2, 3])
    gs = cudf.from_pandas(ps)
    assert_exceptions_equal(
        lfunc=operator.setitem,
        rfunc=operator.setitem,
        lfunc_args_and_kwargs=([ps, key, value],),
        rfunc_args_and_kwargs=([gs, key, value],),
    )

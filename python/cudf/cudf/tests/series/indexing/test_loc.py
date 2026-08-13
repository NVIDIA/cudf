# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import cupy as cp
import numpy as np
import pandas as pd
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/13031")
@pytest.mark.parametrize("other_index", [["1", "3", "2"], [1, 2, 3]])
def test_loc_setitem_series_index_alignment_13031(other_index):
    s = pd.Series([1, 2, 3], index=["1", "2", "3"])
    other = pd.Series([5, 6, 7], index=other_index)

    cs = cudf.from_pandas(s)
    cother = cudf.from_pandas(other)

    s.loc[["1", "3"]] = other

    cs.loc[["1", "3"]] = cother

    assert_eq(s, cs)


def test_series_set_item_index_reference():
    gs1 = cudf.Series([1], index=[7])
    gs2 = cudf.Series([2], index=gs1.index)

    gs1.loc[11] = 2
    ps1 = pd.Series([1], index=[7])
    ps2 = pd.Series([2], index=ps1.index)
    ps1.loc[11] = 2

    assert_eq(ps1, gs1)
    assert_eq(ps2, gs2)


def test_series_loc_numerical():
    ps = pd.Series([1, 2, 3, 4, 5], index=[5, 6, 7, 8, 9])
    gs = cudf.Series(ps)

    assert_eq(ps.loc[5], gs.loc[5])
    assert_eq(ps.loc[6], gs.loc[6])
    assert_eq(ps.loc[6:8], gs.loc[6:8])
    assert_eq(ps.loc[:8], gs.loc[:8])
    assert_eq(ps.loc[6:], gs.loc[6:])
    assert_eq(ps.loc[::2], gs.loc[::2])
    assert_eq(ps.loc[[5, 8, 9]], gs.loc[[5, 8, 9]])
    assert_eq(
        ps.loc[[True, False, True, False, True]],
        gs.loc[[True, False, True, False, True]],
    )
    assert_eq(ps.loc[[5, 8, 9]], gs.loc[cp.array([5, 8, 9])])


def test_series_loc_float_index():
    ps = pd.Series([1, 2, 3, 4, 5], index=[5.43, 6.34, 7.34, 8.0, 9.1])
    gs = cudf.Series(ps)

    assert_eq(ps.loc[5.43], gs.loc[5.43])
    assert_eq(ps.loc[8], gs.loc[8])
    assert_eq(ps.loc[6.1:8], gs.loc[6.1:8])
    assert_eq(ps.loc[:7.1], gs.loc[:7.1])
    assert_eq(ps.loc[6.345:], gs.loc[6.345:])
    assert_eq(ps.loc[::2], gs.loc[::2])
    assert_eq(
        ps.loc[[True, False, True, False, True]],
        gs.loc[[True, False, True, False, True]],
    )


def test_series_loc_string():
    ps = pd.Series(
        [1, 2, 3, 4, 5], index=["one", "two", "three", "four", "five"]
    )
    gs = cudf.Series(ps)

    assert_eq(ps.loc["one"], gs.loc["one"])
    assert_eq(ps.loc["five"], gs.loc["five"])
    assert_eq(ps.loc["two":"four"], gs.loc["two":"four"])
    assert_eq(ps.loc[:"four"], gs.loc[:"four"])
    assert_eq(ps.loc["two":], gs.loc["two":])
    assert_eq(ps.loc[::2], gs.loc[::2])
    assert_eq(ps.loc[["one", "four", "five"]], gs.loc[["one", "four", "five"]])
    assert_eq(
        ps.loc[[True, False, True, False, True]],
        gs.loc[[True, False, True, False, True]],
    )


def test_series_loc_datetime():
    ps = pd.Series(
        [1, 2, 3, 4, 5], index=pd.date_range("20010101", "20010105")
    )
    gs = cudf.Series(ps)

    # a few different ways of specifying a datetime label:
    assert_eq(ps.loc["20010101"], gs.loc["20010101"])
    assert_eq(ps.loc["2001-01-01"], gs.loc["2001-01-01"])
    assert_eq(
        ps.loc[pd.to_datetime("2001-01-01")],
        gs.loc[pd.to_datetime("2001-01-01")],
    )
    assert_eq(
        ps.loc[np.datetime64("2001-01-01")],
        gs.loc[np.datetime64("2001-01-01")],
    )

    assert_eq(
        ps.loc["2001-01-02":"2001-01-05"],
        gs.loc["2001-01-02":"2001-01-05"],
        check_freq=False,
    )
    assert_eq(ps.loc["2001-01-02":], gs.loc["2001-01-02":], check_freq=False)
    assert_eq(ps.loc[:"2001-01-04"], gs.loc[:"2001-01-04"], check_freq=False)
    assert_eq(ps.loc[::2], gs.loc[::2], check_freq=False)

    assert_eq(
        ps.loc[["2001-01-01", "2001-01-04", "2001-01-05"]],
        gs.loc[["2001-01-01", "2001-01-04", "2001-01-05"]],
    )

    assert_eq(
        ps.loc[
            [
                pd.to_datetime("2001-01-01"),
                pd.to_datetime("2001-01-04"),
                pd.to_datetime("2001-01-05"),
            ]
        ],
        gs.loc[
            [
                pd.to_datetime("2001-01-01"),
                pd.to_datetime("2001-01-04"),
                pd.to_datetime("2001-01-05"),
            ]
        ],
    )
    assert_eq(
        ps.loc[[True, False, True, False, True]],
        gs.loc[[True, False, True, False, True]],
        check_freq=False,
    )

    just_less_than_max = ps.index.max() - pd.Timedelta("5m")

    assert_eq(
        ps.loc[:just_less_than_max],
        gs.loc[:just_less_than_max],
        check_freq=False,
    )


def test_series_loc_categorical():
    ps = pd.Series(
        [1, 2, 3, 4, 5], index=pd.Categorical(["a", "b", "c", "d", "e"])
    )
    gs = cudf.Series(ps)

    assert_eq(ps.loc["a"], gs.loc["a"])
    assert_eq(ps.loc["e"], gs.loc["e"])
    assert_eq(ps.loc["b":"d"], gs.loc["b":"d"])
    assert_eq(ps.loc[:"d"], gs.loc[:"d"])
    assert_eq(ps.loc["b":], gs.loc["b":])
    assert_eq(ps.loc[::2], gs.loc[::2])

    # order of categories changes, so we can only
    # compare values:
    assert_eq(
        ps.loc[["a", "d", "e"]].values, gs.loc[["a", "d", "e"]].to_numpy()
    )

    assert_eq(
        ps.loc[[True, False, True, False, True]],
        gs.loc[[True, False, True, False, True]],
    )


@pytest.mark.parametrize(
    "obj",
    [
        pd.DataFrame(
            {"a": [1, 2, 3, 4]},
            index=pd.MultiIndex.from_frame(
                pd.DataFrame(
                    {"A": [2, 3, 1, 4], "B": ["low", "high", "high", "low"]}
                )
            ),
        ),
        pd.Series(
            [1, 2, 3, 4],
            index=pd.MultiIndex.from_frame(
                pd.DataFrame(
                    {"A": [2, 3, 1, 4], "B": ["low", "high", "high", "low"]}
                )
            ),
        ),
    ],
)
def test_dataframe_series_loc_multiindex(obj):
    pindex = pd.MultiIndex.from_frame(
        pd.DataFrame({"A": [3, 2], "B": ["high", "low"]})
    )

    gobj = cudf.from_pandas(obj)
    gindex = cudf.MultiIndex(
        levels=pindex.levels, codes=pindex.codes, names=pindex.names
    )

    # cudf MultiIndex as arg
    expected = obj.loc[pindex]
    got = gobj.loc[gindex]
    assert_eq(expected, got)

    # pandas MultiIndex as arg
    expected = obj.loc[pindex]
    got = gobj.loc[pindex]
    assert_eq(expected, got)


@pytest.mark.parametrize(
    "key, value",
    [
        ("a", 4),
        ("b", 4),
        ("b", np.int8(8)),
        ("d", 4),
        ("d", np.int8(16)),
        ("d", np.float32(16)),
        (["a", "b"], 4),
        (["a", "b"], [4, 5]),
        ([True, False, True], 4),
        ([False, False, False], 4),
        ([True, False, True], [4, 5]),
    ],
)
def test_series_setitem_loc(key, value):
    psr = pd.Series([1, 2, 3], ["a", "b", "c"])
    gsr = cudf.from_pandas(psr)
    psr.loc[key] = value
    gsr.loc[key] = value
    assert_eq(psr, gsr)


@pytest.mark.parametrize(
    "key, value",
    [
        (1, "d"),
        (2, "e"),
        (4, "f"),
        ([1, 3], "g"),
        ([1, 3], ["g", "h"]),
        ([True, False, True], "i"),
        ([False, False, False], "j"),
        ([True, False, True], ["k", "l"]),
    ],
)
def test_series_setitem_loc_numeric_index(key, value):
    psr = pd.Series(["a", "b", "c"], [1, 2, 3])
    gsr = cudf.from_pandas(psr)
    psr.loc[key] = value
    gsr.loc[key] = value
    assert_eq(psr, gsr)


@pytest.mark.parametrize(
    "sli",
    [
        slice("2001", "2020"),
        slice(None, "2020"),
    ],
)
def test_loc_datetime_index_slice_not_in(sli):
    pd_data = pd.Series(
        [1, 2, 3],
        pd.Series(["2001", "2009", "2002"], dtype="datetime64[ns]"),
    )
    gd_data = cudf.from_pandas(pd_data)
    with pytest.raises(KeyError):
        assert_eq(pd_data.loc[sli], gd_data.loc[sli])

    with pytest.raises(KeyError):
        sli = slice(pd.to_datetime(sli.start), pd.to_datetime(sli.stop))
        assert_eq(pd_data.loc[sli], gd_data.loc[sli])


@pytest.mark.parametrize(
    "sli",
    [
        slice("2001", "2009"),
        slice("2001", "2006"),
        slice(None, "2009"),
    ],
)
def test_loc_datetime_index_string_slice_non_monotonic(sli):
    pd_data = pd.Series(
        [1, 2, 3],
        pd.Series(["2001", "2009", "2002"], dtype="datetime64[ns]"),
    )
    gd_data = cudf.from_pandas(pd_data)

    with pytest.raises(KeyError, match="non-monotonic DatetimeIndexes"):
        pd_data.loc[sli]

    with pytest.raises(KeyError, match="non-monotonic DatetimeIndexes"):
        gd_data.loc[sli]


@pytest.mark.parametrize(
    "arg",
    [
        slice(None),
        slice((1, 2), None),
        slice(None, (1, 2)),
        (1, 1),
        # ``.loc[(1, slice(None))]`` now drops the scalar-selected level to
        # match pandas (pandas-dev/pandas#46704 is fixed in pandas 3.0).
        (1, slice(None)),
        1,
        2,
    ],
)
def test_loc_series_multiindex(arg):
    gsr = cudf.DataFrame(
        {"a": [1, 1, 2], "b": [1, 2, 3], "c": ["a", "b", "c"]}
    ).set_index(["a", "b"])["c"]
    psr = gsr.to_pandas()
    assert_eq(psr.loc[arg], gsr.loc[arg])


@pytest.mark.parametrize("indexer", ["loc", "iloc"])
@pytest.mark.parametrize(
    "mask",
    [[False, True], [False, False, True, True, True]],
    ids=["too-short", "too-long"],
)
def test_boolean_mask_wrong_length(indexer, mask):
    s = pd.Series([1, 2, 3, 4])

    indexee = getattr(s, indexer)
    with pytest.raises(IndexError):
        indexee[mask]

    c = cudf.from_pandas(s)
    indexee = getattr(c, indexer)
    with pytest.raises(IndexError):
        indexee[mask]


def test_loc_repeated_index_label_issue_8693():
    # https://github.com/rapidsai/cudf/issues/8693
    s = pd.Series([1, 2, 3, 4], index=[0, 1, 1, 2])
    cs = cudf.from_pandas(s)
    expect = s.loc[1]
    actual = cs.loc[1]
    assert_eq(expect, actual)


@pytest.mark.parametrize("index", [None, [2, 1, 3, 5, 4]])
def test_loc_bool_key_numeric_index_raises(index):
    ser = cudf.Series(range(5), index=index)
    with pytest.raises(KeyError):
        ser.loc[True]


@pytest.mark.parametrize(
    "arg", [slice(2, 4), slice(2, 5), slice(2.3, 5), slice(4.6, 6)]
)
def test_series_iloc_float_int(arg):
    gs = cudf.Series(range(4), index=[2.0, 3.0, 4.5, 5.5])
    ps = gs.to_pandas()

    actual = gs.loc[arg]
    expected = ps.loc[arg]

    assert_eq(actual, expected)


@pytest.mark.parametrize("indexer", [[1], [0, 2]])
def test_loc_integer_categorical_issue_13014(indexer):
    # https://github.com/rapidsai/cudf/issues/13014
    s = pd.Series([0, 1, 2])
    index = pd.Categorical(indexer)
    expect = s.loc[index]
    c = cudf.from_pandas(s)
    actual = c.loc[index]
    assert_eq(expect, actual)


@pytest.mark.parametrize("index_is_ordered", [False, True])
@pytest.mark.parametrize("label_is_ordered", [False, True])
def test_loc_categorical_ordering_mismatch_issue_13652(
    index_is_ordered, label_is_ordered
):
    # https://github.com/rapidsai/cudf/issues/13652
    s = cudf.Series(
        [0, 2, 8, 4, 2],
        index=cudf.CategoricalIndex(
            [1, 2, 3, 4, 5],
            categories=[1, 2, 3, 4, 5],
            ordered=index_is_ordered,
        ),
    )
    labels = cudf.CategoricalIndex(
        [1, 4], categories=[1, 4], ordered=label_is_ordered
    )
    actual = s.loc[labels]
    expect = s.to_pandas().loc[labels.to_pandas()]
    assert_eq(actual, expect)


def test_loc_categorical_no_integer_fallback_issue_13653():
    # https://github.com/rapidsai/cudf/issues/13653
    s = cudf.Series(
        [1, 2], index=cudf.CategoricalIndex([3, 4], categories=[3, 4])
    )
    actual = s.loc[3]
    expect = s.to_pandas().loc[3]
    assert_eq(actual, expect)


def test_loc_wrong_type_slice_datetimeindex():
    ser_cudf = cudf.Series(
        range(3), index=cudf.date_range("2020-01-01", periods=3, freq="D")
    )
    with pytest.raises(TypeError):
        ser_cudf.loc[2:]

    ser_pd = pd.Series(
        range(3), index=pd.date_range("2020-01-01", periods=3, freq="D")
    )
    with pytest.raises(TypeError):
        ser_pd.loc[2:]


def test_series_loc_multiindex_scalar_selection():
    mi = pd.MultiIndex.from_tuples([(0, 0), (1, 1), (2, 1)], names=["A", "B"])
    psr = pd.Series([1, 2, 3], index=mi)
    gsr = cudf.from_pandas(psr)

    # Selecting the second level by a scalar drops it, keeping level "A".
    assert_eq(gsr.loc[:, 1], psr.loc[:, 1])
    assert isinstance(gsr.loc[:, 1].index, cudf.Index)
    assert not isinstance(gsr.loc[:, 1].index, cudf.MultiIndex)

    # Every level selected by a scalar -> scalar.
    assert gsr.loc[(1, 1)] == psr.loc[(1, 1)]


def test_series_loc_drops_multiple_scalar_levels():
    mi = pd.MultiIndex.from_arrays(
        [["x"], ["y"], ["z"]], names=["a", "b", "c"]
    )
    psr = pd.Series([0], index=mi)
    gsr = cudf.from_pandas(psr)
    # "x" and "z" are scalar-selected -> dropped; the slice on "b" is kept.
    expected = psr.loc["x", :, "z"]
    result = gsr.loc["x", :, "z"]
    assert_eq(result, expected)


def test_series_loc_tuple_of_lists_stays_series():
    index = pd.MultiIndex.from_product([[10, 20, 30], [1, 2, 3]])
    psr = pd.Series(np.arange(9), index=index).sort_index()
    gsr = cudf.from_pandas(psr)
    with cudf.option_context("mode.pandas_compatible", True):
        result = gsr.loc[([10, 20], [2, 3])]
    expected = psr.loc[([10, 20], [2, 3])]
    assert_eq(result, expected)
    assert isinstance(result, cudf.Series)


def test_loc_datetime_key_out_of_bounds():
    # A label beyond the index unit's range cannot be present: pandas
    # raises KeyError for scalar lookups and OutOfBoundsDatetime for
    # list keys.
    index = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]")
    ser_pd = pd.Series([1, 2], index=pd.Index(index))
    ser_cudf = cudf.Series([1, 2], index=cudf.Index(index))
    key = np.datetime64("9999-01-01", "s")

    with pytest.raises(KeyError):
        ser_pd.loc[key]
    with pytest.raises(KeyError):
        ser_cudf.loc[key]

    with pytest.raises(pd.errors.OutOfBoundsDatetime):
        ser_pd.loc[[key]]
    with pytest.raises(pd.errors.OutOfBoundsDatetime):
        ser_cudf.loc[[key]]

# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import array
import datetime
import decimal
import types
import zoneinfo

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from packaging.version import parse

import cudf
from cudf.core.buffer.spill_manager import get_global_manager
from cudf.core.column.column import as_column
from cudf.errors import MixedTypeError
from cudf.testing import assert_eq
from cudf.testing._utils import assert_exceptions_equal
from cudf.utils.dtypes import np_dtypes_to_pandas_dtypes


@pytest.mark.parametrize(
    "data1, data2",
    [(1, 2), (1.0, 2.0), (3, 4.0)],
)
@pytest.mark.parametrize("data3, data4", [(6, 10), (5.0, 9.0), (2, 6.0)])
def test_create_interval_series(data1, data2, data3, data4, interval_closed):
    expect = pd.Series(
        pd.Interval(data1, data2, interval_closed), dtype="interval"
    )
    got = cudf.Series(
        pd.Interval(data1, data2, interval_closed), dtype="interval"
    )
    assert_eq(expect, got)

    expect_two = pd.Series(
        [
            pd.Interval(data1, data2, interval_closed),
            pd.Interval(data3, data4, interval_closed),
        ],
        dtype="interval",
    )
    got_two = cudf.Series(
        [
            pd.Interval(data1, data2, interval_closed),
            pd.Interval(data3, data4, interval_closed),
        ],
        dtype="interval",
    )
    assert_eq(expect_two, got_two)

    expect_three = pd.Series(
        [
            pd.Interval(data1, data2, interval_closed),
            pd.Interval(data3, data4, interval_closed),
            pd.Interval(data1, data2, interval_closed),
        ],
        dtype="interval",
    )
    got_three = cudf.Series(
        [
            pd.Interval(data1, data2, interval_closed),
            pd.Interval(data3, data4, interval_closed),
            pd.Interval(data1, data2, interval_closed),
        ],
        dtype="interval",
    )
    assert_eq(expect_three, got_three)


def test_from_pandas_for_series_nan_as_null(nan_as_null):
    data = [np.nan, 2.0, 3.0]
    psr = pd.Series(data)

    expected = cudf.Series._from_column(
        as_column(data, nan_as_null=nan_as_null)
    )
    got = cudf.from_pandas(psr, nan_as_null=nan_as_null)

    assert_eq(expected, got)


@pytest.mark.parametrize(
    "data",
    [
        [[{"name": 123}]],
        [
            [
                {
                    "IsLeapYear": False,
                    "data": {"Year": 1999, "Month": 7},
                    "names": ["Mike", None],
                },
                {
                    "IsLeapYear": True,
                    "data": {"Year": 2004, "Month": 12},
                    "names": None,
                },
                {
                    "IsLeapYear": False,
                    "data": {"Year": 1996, "Month": 2},
                    "names": ["Rose", "Richard"],
                },
            ]
        ],
        [
            [None, {"human?": True, "deets": {"weight": 2.4, "age": 27}}],
            [
                {"human?": None, "deets": {"weight": 5.3, "age": 25}},
                {"human?": False, "deets": {"weight": 8.0, "age": 31}},
                {"human?": False, "deets": None},
            ],
            [],
            None,
            [{"human?": None, "deets": {"weight": 6.9, "age": None}}],
        ],
        [
            {
                "name": "var0",
                "val": [
                    {"name": "var1", "val": None, "type": "optional<struct>"}
                ],
                "type": "list",
            },
            {},
            {
                "name": "var2",
                "val": [
                    {
                        "name": "var3",
                        "val": {"field": 42},
                        "type": "optional<struct>",
                    },
                    {
                        "name": "var4",
                        "val": {"field": 3.14},
                        "type": "optional<struct>",
                    },
                ],
                "type": "list",
            },
            None,
        ],
    ],
)
def test_lists_of_structs_data(data):
    got = cudf.Series(data)
    expected = cudf.Series(pa.array(data))
    assert_eq(got, expected)


@pytest.fixture(
    params=[
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [],
        [None],
        [None, None, None, None, None],
        [12, 12, 22, 343, 4353534, 435342],
        np.array([10, 20, 30, None, 100]),
        cp.asarray([10, 20, 30, 100]),
        [1000000, 200000, 3000000],
        [1000000, 200000, None],
        [1],
        [12, 11, 232, 223432411, 2343241, 234324, 23234],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
        [
            136457654736252,
            134736784364431,
            245345345545332,
            223432411,
            2343241,
            3634548734,
            23234,
        ],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ]
)
def timedelta_data(request):
    return request.param


def test_timedelta_series_create(timedelta_data, timedelta_types_as_str):
    if timedelta_types_as_str != "timedelta64[ns]":
        pytest.skip(
            "Bug in pandas : https://github.com/pandas-dev/pandas/issues/35465"
        )
    psr = pd.Series(
        cp.asnumpy(timedelta_data)
        if isinstance(timedelta_data, cp.ndarray)
        else timedelta_data,
        dtype=timedelta_types_as_str,
    )
    gsr = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)

    assert_eq(psr, gsr)


def test_timedelta_from_pandas(timedelta_data, timedelta_types_as_str):
    psr = pd.Series(
        cp.asnumpy(timedelta_data)
        if isinstance(timedelta_data, cp.ndarray)
        else timedelta_data,
        dtype=timedelta_types_as_str,
    )
    gsr = cudf.from_pandas(psr)

    assert_eq(psr, gsr)


def test_construct_int_series_with_nulls_compat_mode():
    # in compatibility mode, constructing a Series
    # with nulls should result in a floating Series:
    with cudf.option_context("mode.pandas_compatible", True):
        s = cudf.Series([1, 2, None])
    assert s.dtype == np.dtype("float64")


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 24, "d": 1010},
        {"a": 1},
        {1: "a", 2: "b", 24: "c", 1010: "d"},
        {1: "a"},
        {"a": [1]},
    ],
)
def test_series_init_dict(data):
    pandas_series = pd.Series(data)
    cudf_series = cudf.Series(data)

    assert_eq(pandas_series, cudf_series)


@pytest.mark.parametrize(
    "data",
    [
        [[]],
        [[[]]],
        [[0]],
        [[0, 1]],
        [[0, 1], [2, 3]],
        [[[0, 1], [2]], [[3, 4]]],
        [[None]],
        [[[None]]],
        [[None], None],
        [[1, None], [1]],
        [[1, None], None],
        [[[1, None], None], None],
    ],
)
def test_create_list_series(data):
    expect = pd.Series(data)
    got = cudf.Series(data)
    assert_eq(expect, got)
    assert isinstance(got[0], type(expect[0]))
    assert isinstance(got.to_pandas()[0], type(expect[0]))


@pytest.mark.parametrize(
    "input_obj", [[[1, pd.NA, 3]], [[1, pd.NA, 3], [4, 5, pd.NA]]]
)
def test_construction_series_with_nulls(input_obj):
    expect = pa.array(input_obj, from_pandas=True)
    got = cudf.Series(input_obj).to_arrow()

    assert expect == got


def test_series_unitness_np_datetimelike_units():
    data = np.array([np.timedelta64(1)])
    with pytest.raises(TypeError):
        cudf.Series(data)
    with pytest.raises(TypeError):
        pd.Series(data)


def test_from_numpyextensionarray_string_object_pandas_compat_mode():
    data = pd.arrays.NumpyExtensionArray(np.array(["a", None], dtype=object))
    with cudf.option_context("mode.pandas_compatible", True):
        result = cudf.Series(data)
    expected = pd.Series(data)
    assert_eq(result, expected)


def test_list_category_like_maintains_dtype():
    dtype = cudf.CategoricalDtype(categories=[1, 2, 3, 4], ordered=True)
    data = [1, 2, 3]
    result = cudf.Series._from_column(as_column(data, dtype=dtype))
    expected = pd.Series(data, dtype=dtype.to_pandas())
    assert_eq(result, expected)


def test_list_interval_like_maintains_dtype():
    dtype = cudf.IntervalDtype(subtype=np.int8)
    data = [pd.Interval(1, 2)]
    result = cudf.Series._from_column(as_column(data, dtype=dtype))
    expected = pd.Series(data, dtype=dtype.to_pandas())
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "klass", [cudf.Series, cudf.Index, pd.Series, pd.Index]
)
def test_series_from_named_object_name_priority(klass):
    result = cudf.Series(klass([1], name="a"), name="b")
    assert result.name == "b"


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 3},
        cudf.Series([1, 2, 3], index=list("abc")),
        pd.Series([1, 2, 3], index=list("abc")),
    ],
)
def test_series_from_object_with_index_index_arg_reindex(data):
    result = cudf.Series(data, index=list("bca"))
    expected = cudf.Series([2, 3, 1], index=list("bca"))
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "data",
    [
        {0: 1, 1: 2, 2: 3},
        cudf.Series([1, 2, 3]),
        cudf.Index([1, 2, 3]),
        pd.Series([1, 2, 3]),
        pd.Index([1, 2, 3]),
        [1, 2, 3],
    ],
)
def test_series_dtype_astypes(data):
    result = cudf.Series(data, dtype="float64")
    expected = cudf.Series([1.0, 2.0, 3.0])
    assert_eq(result, expected)


@pytest.mark.parametrize("pa_type", [pa.string, pa.large_string])
def test_series_from_large_string(pa_type):
    pa_string_array = pa.array(["a", "b", "c"]).cast(pa_type())
    got = cudf.Series(pa_string_array)
    expected = pd.Series(pa_string_array)

    assert_eq(expected, got)


def test_series_init_with_nans():
    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.Series([1, 2, 3, np.nan])
    assert gs.dtype == np.dtype("float64")
    ps = pd.Series([1, 2, 3, np.nan])
    assert_eq(ps, gs)


@pytest.mark.parametrize(
    "data",
    [
        [[1, 2, 3], [10, 20]],
        [[1.0, 2.0, 3.0], None, [10.0, 20.0, np.nan]],
        [[5, 6], None, [1]],
        [None, None, None, None, None, [10, 20]],
    ],
)
@pytest.mark.parametrize("klass", [cudf.Series, list, cp.array])
def test_nested_series_from_sequence_data(data, klass):
    actual = cudf.Series(
        [klass(val) if val is not None else val for val in data]
    )
    expected = cudf.Series(data)
    assert_eq(actual, expected)


@pytest.mark.parametrize(
    "data",
    [
        lambda: cp.ones(5, dtype=cp.float16),
        lambda: np.ones(5, dtype="float16"),
        lambda: pd.Series([0.1, 1.2, 3.3], dtype="float16"),
        pytest.param(
            lambda: pa.array(np.ones(5, dtype="float16")),
            marks=pytest.mark.xfail(
                reason="https://issues.apache.org/jira/browse/ARROW-13762"
            ),
        ),
    ],
)
def test_series_raises_float16(data):
    data = data()
    with pytest.raises(TypeError):
        cudf.Series(data)


@pytest.mark.parametrize(
    "data", [[True, False, None, True, False], [None, None], []]
)
@pytest.mark.parametrize("bool_dtype", ["bool", "boolean", pd.BooleanDtype()])
def test_nullable_bool_dtype_series(data, bool_dtype):
    psr = pd.Series(data, dtype=pd.BooleanDtype())
    gsr = cudf.Series(data, dtype=bool_dtype)

    assert_eq(psr, gsr.to_pandas(nullable=True))


@pytest.mark.parametrize("data", [None, 123, 33243243232423, 0])
@pytest.mark.parametrize("klass", [pd.Timestamp, pd.Timedelta])
def test_temporal_scalar_series_init(data, klass):
    scalar = klass(data)
    expected = pd.Series([scalar])
    actual = cudf.Series([scalar])

    assert_eq(expected, actual)

    expected = pd.Series(scalar)
    actual = cudf.Series(scalar)

    assert_eq(expected, actual)


def test_series_from_series_index_no_shallow_copy():
    ser1 = cudf.Series(range(3), index=list("abc"))
    ser2 = cudf.Series(ser1)
    assert ser1.index is ser2.index


def test_int8_int16_construction():
    s = cudf.Series([np.int8(8), np.int16(128)])
    assert s.dtype == np.dtype("i2")


@pytest.mark.parametrize(
    "data", [[0, 1, 2, 3, 4], range(5), [np.int8(8), np.int16(128)]]
)
def test_default_integer_bitwidth_construction(default_integer_bitwidth, data):
    s = cudf.Series(data)
    assert s.dtype == np.dtype(f"i{default_integer_bitwidth // 8}")


@pytest.mark.parametrize("data", [[1.5, 2.5, 4.5], [1000, 2000, 4000, 3.14]])
def test_default_float_bitwidth_construction(default_float_bitwidth, data):
    s = cudf.Series(data)
    assert s.dtype == np.dtype(f"f{default_float_bitwidth // 8}")


def test_series_ordered_dedup():
    # part of https://github.com/rapidsai/cudf/issues/11486
    rng = np.random.default_rng(seed=0)
    sr = cudf.Series(rng.integers(0, 100, 1000))
    # pandas unique() preserves order
    expect = pd.Series(sr.to_pandas().unique())
    got = cudf.Series._from_column(sr._column.unique())
    assert_eq(expect.values, got.values)


def test_int64_equality():
    s = cudf.Series(np.asarray([2**63 - 10, 2**63 - 100], dtype=np.int64))
    assert (s != np.int64(2**63 - 1)).all()


@pytest.mark.parametrize(
    "data",
    [
        {"a": 1, "b": 2, "c": 24, "d": 1010},
        {"a": 1},
    ],
)
@pytest.mark.parametrize(
    "index", [None, ["b", "c"], ["d", "a", "c", "b"], ["a"]]
)
def test_series_init_dict_with_index(data, index):
    pandas_series = pd.Series(data, index=index)
    cudf_series = cudf.Series(data, index=index)

    assert_eq(pandas_series, cudf_series)


def test_series_data_and_index_length_mismatch():
    assert_exceptions_equal(
        lfunc=pd.Series,
        rfunc=cudf.Series,
        lfunc_args_and_kwargs=([], {"data": [11], "index": [10, 11]}),
        rfunc_args_and_kwargs=([], {"data": [11], "index": [10, 11]}),
    )


@pytest.mark.parametrize(
    "dtype", ["datetime64[ns]", "timedelta64[ns]", "object", "str"]
)
def test_series_mixed_dtype_error(dtype):
    ps = pd.concat([pd.Series([1, 2, 3], dtype=dtype), pd.Series([10, 11])])
    with pytest.raises(TypeError):
        cudf.Series(ps)
    with pytest.raises(TypeError):
        cudf.Series(ps.array)


@pytest.mark.parametrize("data", ["abc", None, 1, 3.7])
@pytest.mark.parametrize(
    "index", [None, ["b", "c"], ["d", "a", "c", "b"], ["a"]]
)
def test_series_init_scalar_with_index(data, index):
    pandas_series = pd.Series(data, index=index)
    cudf_series = cudf.Series(data, index=index)

    assert_eq(pandas_series, cudf_series)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4],
        [10, 20, None, None],
    ],
)
@pytest.mark.parametrize("copy", [True, False])
def test_series_copy(data, copy):
    psr = pd.Series(data)
    gsr = cudf.from_pandas(psr)

    new_psr = pd.Series(psr, copy=copy)
    new_gsr = cudf.Series(gsr, copy=copy)

    new_psr.iloc[0] = 999
    new_gsr.iloc[0] = 999

    assert_eq(psr, gsr)
    assert_eq(new_psr, new_gsr)


def test_series_init_from_series_and_index():
    ser = cudf.Series([4, 7, -5, 3], index=["d", "b", "a", "c"])
    result = cudf.Series(ser, index=list("abcd"))
    expected = cudf.Series([-5, 7, 3, 4], index=list("abcd"))
    assert_eq(result, expected)


def test_series_constructor_unbounded_sequence():
    class A:
        def __getitem__(self, key):
            return 1

    with pytest.raises(TypeError):
        cudf.Series(A())


def test_series_from_pandas_sparse():
    pser = pd.Series(range(2), dtype=pd.SparseDtype(np.int64, 0))
    with pytest.raises(NotImplementedError):
        cudf.Series(pser)


def test_multi_dim_series_error():
    arr = cp.array([(1, 2), (3, 4)])
    with pytest.raises(ValueError):
        cudf.Series(arr)


def test_bool_series_mixed_dtype_error():
    ps = pd.Series([True, False, None])
    all_bool_ps = pd.Series([True, False, True], dtype="object")
    # ps now has `object` dtype, which
    # isn't supported by `cudf`.
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(TypeError):
            cudf.Series(ps)
        with pytest.raises(TypeError):
            cudf.from_pandas(ps)
        with pytest.raises(TypeError):
            cudf.Series(ps, dtype=bool)
        expected = cudf.Series(all_bool_ps, dtype=bool)
        assert_eq(expected, all_bool_ps.astype(bool))
    nan_bools_mix = pd.Series([True, False, True, np.nan], dtype="object")
    gs = cudf.Series(nan_bools_mix, nan_as_null=True)
    assert_eq(gs.to_pandas(nullable=True), nan_bools_mix.astype("boolean"))
    with pytest.raises(TypeError):
        cudf.Series(nan_bools_mix, nan_as_null=False)


@pytest.mark.parametrize("klass", [cudf.Index, cudf.Series])
@pytest.mark.parametrize(
    "data", [pa.array([float("nan")]), pa.chunked_array([[float("nan")]])]
)
def test_nan_as_null_from_arrow_objects(klass, data):
    result = klass(data, nan_as_null=True)
    expected = klass(pa.array([None], type=pa.float64()))
    assert_eq(result, expected)


@pytest.mark.parametrize("reso", ["M", "ps"])
@pytest.mark.parametrize("typ", ["M", "m"])
def test_series_invalid_reso_dtype(reso, typ):
    with pytest.raises(TypeError):
        cudf.Series([], dtype=f"{typ}8[{reso}]")


@pytest.mark.parametrize("base_name", [None, "a"])
def test_series_to_frame_none_name(base_name):
    result = cudf.Series(range(1), name=base_name).to_frame(name=None)
    expected = pd.Series(range(1), name=base_name).to_frame(name=None)
    assert_eq(result, expected)


@pytest.mark.parametrize("klass", [cudf.Series, cudf.Index])
@pytest.mark.parametrize(
    "data",
    [
        pa.array([1, None], type=pa.int64()),
        pa.chunked_array([[1, None]], type=pa.int64()),
    ],
)
def test_from_arrow_array_dtype(klass, data):
    obj = klass(data, dtype="int8")
    assert obj.dtype == np.dtype("int8")


@pytest.mark.parametrize(
    "nat, value",
    [
        [np.datetime64("nat", "ns"), np.datetime64("2020-01-01", "ns")],
        [np.timedelta64("nat", "ns"), np.timedelta64(1, "ns")],
    ],
)
@pytest.mark.parametrize("nan_as_null", [True, False])
def test_series_np_array_nat_nan_as_nulls(nat, value, nan_as_null):
    expected = np.array([nat, value])
    ser = cudf.Series(expected, nan_as_null=nan_as_null)
    assert ser[0] is pd.NaT
    assert ser[1] == value


def test_null_like_to_nan_pandas_compat():
    with cudf.option_context("mode.pandas_compatible", True):
        ser = cudf.Series([1, 2, np.nan, 10, None])
        pser = pd.Series([1, 2, np.nan, 10, None])

        assert pser.dtype == ser.dtype
        assert_eq(ser, pser)


def test_non_strings_dtype_object_pandas_compat_raises():
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(TypeError):
            cudf.Series([1], dtype=object)


@pytest.mark.parametrize("arr", [np.array, cp.array, pd.Series])
def test_construct_nonnative_array(arr):
    data = [1, 2, 3.5, 4]
    dtype = np.dtype("f4")
    native = arr(data, dtype=dtype)
    nonnative = arr(data, dtype=dtype.newbyteorder())
    result = cudf.Series(nonnative)
    expected = cudf.Series(native)
    assert_eq(result, expected)


@pytest.mark.parametrize("input_obj", [[1, cudf.NA, 3]])
def test_series_construction_with_nulls(numeric_types_as_str, input_obj):
    dtype = np.dtype(numeric_types_as_str)
    # numpy case

    expect = pd.Series(input_obj, dtype=np_dtypes_to_pandas_dtypes[dtype])
    got = cudf.Series(input_obj, dtype=dtype).to_pandas(nullable=True)

    assert_eq(expect, got)

    # Test numpy array of objects case
    np_data = [
        dtype.type(v) if v is not cudf.NA else cudf.NA for v in input_obj
    ]

    expect = pd.Series(np_data, dtype=np_dtypes_to_pandas_dtypes[dtype])
    got = cudf.Series(np_data, dtype=dtype).to_pandas(nullable=True)
    assert_eq(expect, got)


@pytest.mark.parametrize("nan_as_null", [True, False])
def test_construct_all_pd_NA_with_dtype(nan_as_null):
    result = cudf.Series(
        [pd.NA, pd.NA], dtype=np.dtype(np.float64), nan_as_null=nan_as_null
    )
    expected = cudf.Series(pa.array([None, None], type=pa.float64()))
    assert_eq(result, expected)


def test_to_from_arrow_nulls(all_supported_types_as_str):
    if all_supported_types_as_str in {"category", "str"}:
        pytest.skip(f"Test not applicable with {all_supported_types_as_str}")
    data_type = all_supported_types_as_str
    if data_type == "bool":
        s1 = pa.array([True, None, False, None, True], type=data_type)
    else:
        dtype = np.dtype(data_type)
        if dtype.type == np.datetime64:
            time_unit, _ = np.datetime_data(dtype)
            data_type = pa.timestamp(unit=time_unit)
        elif dtype.type == np.timedelta64:
            time_unit, _ = np.datetime_data(dtype)
            data_type = pa.duration(unit=time_unit)
        s1 = pa.array([1, None, 3, None, 5], type=data_type)
    gs1 = cudf.Series.from_arrow(s1)
    assert isinstance(gs1, cudf.Series)
    # We have 64B padded buffers for nulls whereas Arrow returns a minimal
    # number of bytes, so only check the first byte in this case
    np.testing.assert_array_equal(
        np.asarray(s1.buffers()[0]).view("u1")[0],
        cp.asarray(gs1._column.plc_column.null_mask()).get().view("u1")[0],
    )
    assert pa.Array.equals(s1, gs1.to_arrow())

    s2 = pa.array([None, None, None, None, None], type=data_type)
    gs2 = cudf.Series.from_arrow(s2)
    assert isinstance(gs2, cudf.Series)
    # We have 64B padded buffers for nulls whereas Arrow returns a minimal
    # number of bytes, so only check the first byte in this case
    np.testing.assert_array_equal(
        np.asarray(s2.buffers()[0]).view("u1")[0],
        cp.asarray(gs2._column.plc_column.null_mask()).get().view("u1")[0],
    )
    assert pa.Array.equals(s2, gs2.to_arrow())


def test_cuda_array_interface(numeric_and_bool_types_as_str):
    np_data = np.arange(10).astype(numeric_and_bool_types_as_str)
    cupy_data = cp.array(np_data)
    pd_data = pd.Series(np_data)

    cudf_data = cudf.Series(cupy_data)
    assert_eq(pd_data, cudf_data)

    gdf = cudf.DataFrame()
    gdf["test"] = cupy_data
    pd_data.name = "test"
    assert_eq(pd_data, gdf["test"])


@pytest.mark.parametrize("nan_as_null", [True, False])
def test_series_list_nanasnull(nan_as_null):
    data = [1.0, 2.0, 3.0, np.nan, None]

    expect = pa.array(data, from_pandas=nan_as_null)
    got = cudf.Series(data, nan_as_null=nan_as_null).to_arrow()

    # Bug in Arrow 0.14.1 where NaNs aren't handled
    expect = expect.cast("int64", safe=False)
    got = got.cast("int64", safe=False)

    assert pa.Array.equals(expect, got)


@pytest.mark.parametrize("num_elements", [0, 10])
@pytest.mark.parametrize("null_type", [np.nan, None, "mixed"])
def test_series_all_null(num_elements, null_type):
    if null_type == "mixed":
        data = []
        data1 = [np.nan] * int(num_elements / 2)
        data2 = [None] * int(num_elements / 2)
        for idx in range(len(data1)):
            data.append(data1[idx])
            data.append(data2[idx])
    else:
        data = [null_type] * num_elements

    # Typecast Pandas because None will return `object` dtype
    expect = pd.Series(data, dtype="float64")
    got = cudf.Series(data, dtype="float64")

    assert_eq(expect, got)


@pytest.mark.parametrize("num_elements", [0, 10])
def test_series_all_valid_nan(num_elements):
    data = [np.nan] * num_elements
    sr = cudf.Series(data, nan_as_null=False)
    np.testing.assert_equal(sr.null_count, 0)


def test_series_empty_dtype():
    expected = pd.Series([])
    actual = cudf.Series([])
    assert expected.dtype == actual.dtype
    assert_eq(expected, actual)


@pytest.mark.parametrize("data", [None, {}, []])
def test_series_empty_index_rangeindex(data):
    expected = cudf.RangeIndex(0)
    result = cudf.Series(data).index
    assert_eq(result, expected)


@pytest.mark.parametrize("index", [None, []])
def test_series_no_data_empty_index_defaults_to_object(index):
    # When no data is supplied and the index is empty (None or []), the
    # resulting dtype should be object to match pandas. Only a non-empty
    # index without explicit dtype should default to float64.
    expected = pd.Series(index=index)
    result = cudf.Series(index=index)
    assert result.dtype == np.dtype("object")
    assert_eq(result, expected)


def test_series_no_data_nonempty_index_defaults_to_float64():
    # Sanity check: a non-empty index without explicit dtype keeps the
    # float64 default that pandas uses.
    expected = pd.Series(index=[1, 2, 3])
    result = cudf.Series(index=[1, 2, 3])
    assert result.dtype == np.dtype("float64")
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "pandas_type",
    [
        pd.ArrowDtype(pa.int8()),
        pd.ArrowDtype(pa.int16()),
        pd.ArrowDtype(pa.int32()),
        pd.ArrowDtype(pa.int64()),
        pd.ArrowDtype(pa.uint8()),
        pd.ArrowDtype(pa.uint16()),
        pd.ArrowDtype(pa.uint32()),
        pd.ArrowDtype(pa.uint64()),
        pd.ArrowDtype(pa.float32()),
        pd.ArrowDtype(pa.float64()),
        pd.Int8Dtype(),
        pd.Int16Dtype(),
        pd.Int32Dtype(),
        pd.Int64Dtype(),
        pd.UInt8Dtype(),
        pd.UInt16Dtype(),
        pd.UInt32Dtype(),
        pd.UInt64Dtype(),
        pd.Float32Dtype(),
        pd.Float64Dtype(),
    ],
)
def test_series_arrow_numeric_types_roundtrip(pandas_type):
    ps = pd.Series([1, 2, 3], dtype=pandas_type)
    pi = pd.Index(ps)
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.from_pandas(ps)
        assert_eq(ps, gs)

    with cudf.option_context("mode.pandas_compatible", True):
        gi = cudf.from_pandas(pi)
        assert_eq(pi, gi)

    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.from_pandas(pdf)
        assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "pandas_type", [pd.ArrowDtype(pa.bool_()), pd.BooleanDtype()]
)
def test_series_arrow_bool_types_roundtrip(pandas_type):
    ps = pd.Series([True, False, None], dtype=pandas_type)
    pi = pd.Index(ps)
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.from_pandas(ps)
        assert_eq(ps, gs)

    with cudf.option_context("mode.pandas_compatible", True):
        gi = cudf.from_pandas(pi)
        assert_eq(pi, gi)

    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.from_pandas(pdf)
        assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "pandas_type", [pd.ArrowDtype(pa.string()), pd.StringDtype()]
)
def test_series_arrow_string_types_roundtrip(pandas_type):
    ps = pd.Series(["abc", None, "xyz"], dtype=pandas_type)
    pi = pd.Index(ps)
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.from_pandas(ps)
        assert_eq(ps, gs)

    with cudf.option_context("mode.pandas_compatible", True):
        gi = cudf.from_pandas(pi)
        assert_eq(pi, gi)

    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.from_pandas(pdf)
        assert_eq(pdf, gdf)


def test_series_arrow_category_types_roundtrip():
    pa_array = pa.array(pd.Series([1, 2, 3], dtype="category"))
    ps = pd.Series([1, 2, 3], dtype=pd.ArrowDtype(pa_array.type))
    pi = pd.Index(ps)
    pdf = pi.to_frame()

    with pytest.raises(TypeError):
        cudf.from_pandas(ps)

    with pytest.raises(TypeError):
        cudf.from_pandas(pi)

    with pytest.raises(TypeError):
        cudf.from_pandas(pdf)


@pytest.mark.parametrize(
    "pa_type",
    [pa.decimal128(10, 2), pa.decimal128(5, 2), pa.decimal128(20, 2)],
)
def test_series_arrow_decimal_types_roundtrip(pa_type):
    ps = pd.Series(
        [
            decimal.Decimal("1.2"),
            decimal.Decimal("20.56"),
            decimal.Decimal("3"),
        ],
        dtype=pd.ArrowDtype(pa_type),
    )
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.from_pandas(ps)
        assert_eq(ps, gs)

    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.from_pandas(pdf)
        assert_eq(pdf, gdf)


def test_cuda_array_interface_interop_in(numeric_and_temporal_types_as_str):
    if numeric_and_temporal_types_as_str.startswith(
        "datetime"
    ) or numeric_and_temporal_types_as_str.startswith("timedelta"):
        pytest.skip(
            f"cupy doesn't support {numeric_and_temporal_types_as_str}"
        )

    np_data = np.arange(10).astype(numeric_and_temporal_types_as_str)
    module_data = cp.array(np_data)

    pd_data = pd.Series(np_data)
    # Test using a specific function for __cuda_array_interface__ here
    cudf_data = cudf.Series(module_data)

    assert_eq(pd_data, cudf_data)

    gdf = cudf.DataFrame()
    gdf["test"] = module_data
    pd_data.name = "test"
    assert_eq(pd_data, gdf["test"])


def test_cuda_array_interface_interop_out(numeric_and_temporal_types_as_str):
    np_data = np.arange(10).astype(numeric_and_temporal_types_as_str)
    cudf_data = cudf.Series(np_data)
    assert isinstance(cudf_data.__cuda_array_interface__, dict)

    module_data = cp.asarray(cudf_data)
    got = cp.asnumpy(module_data)

    expect = np_data

    assert_eq(expect, got)


def test_cuda_array_interface_interop_out_masked(
    numeric_and_temporal_types_as_str,
):
    np_data = np.arange(10).astype("float64")
    np_data[[0, 2, 4, 6, 8]] = np.nan

    cudf_data = cudf.Series(np_data).astype(numeric_and_temporal_types_as_str)
    cai = cudf_data.__cuda_array_interface__
    assert isinstance(cai, dict)
    assert "mask" in cai


@pytest.mark.parametrize("nulls", ["all", "some", "bools", "none"])
@pytest.mark.parametrize("mask_type", ["bits", "bools"])
def test_cuda_array_interface_as_column(
    numeric_and_temporal_types_as_str, nulls, mask_type
):
    sr = cudf.Series(np.arange(10))

    if nulls == "some":
        mask = [
            True,
            False,
            True,
            False,
            False,
            True,
            True,
            False,
            True,
            True,
        ]
        sr[sr[~np.asarray(mask)]] = None
    elif nulls == "all":
        sr[:] = None

    sr = sr.astype(numeric_and_temporal_types_as_str)

    obj = types.SimpleNamespace(
        __cuda_array_interface__=sr.__cuda_array_interface__
    )

    if mask_type == "bools":
        if nulls == "some":
            obj.__cuda_array_interface__["mask"] = cp.asarray(mask)
        elif nulls == "all":
            obj.__cuda_array_interface__["mask"] = cp.array([False] * 10)

    expect = sr
    got = cudf.Series(obj)

    assert_eq(expect, got)


def test_series_from_ephemeral_cupy():
    # Test that we keep a reference to the ephemeral
    # CuPy array. If we didn't, then `a` would end
    # up referring to the same memory as `b` due to
    # CuPy's caching allocator
    a = cudf.Series(cp.asarray([1, 2, 3]))
    b = cudf.Series(cp.asarray([1, 1, 1]))
    assert_eq(pd.Series([1, 2, 3]), a)
    assert_eq(pd.Series([1, 1, 1]), b)


def test_column_from_ephemeral_cupy_try_lose_reference():
    # Try to lose the reference we keep to the ephemeral
    # CuPy array
    a = cudf.Series(cp.asarray([1, 2, 3]))._column
    a = cudf.core.column.as_column(a)
    b = cp.asarray([1, 1, 1])
    assert_eq(pd.Index([1, 2, 3]), a.to_pandas())

    a = cudf.Series(cp.asarray([1, 2, 3]))._column
    a.name = "b"
    b = cp.asarray([1, 1, 1])  # noqa: F841
    assert_eq(pd.Index([1, 2, 3]), a.to_pandas())


@pytest.mark.xfail(
    get_global_manager() is not None,
    reason=(
        "spilling doesn't support PyTorch, see "
        "`cudf.core.buffer.spillable_buffer.DelayedPointerTuple`"
    ),
)
def test_cuda_array_interface_pytorch():
    torch = pytest.importorskip("torch", minversion="2.4.0")
    if not torch.cuda.is_available():
        pytest.skip("need gpu version of pytorch to be installed")

    series = cudf.Series([1, -1, 10, -56])
    tensor = torch.tensor(series)
    got = cudf.Series(tensor)

    assert_eq(got, series)
    buffer = cudf.core.buffer.as_buffer(cp.ones(10, dtype=np.bool_))
    tensor = torch.tensor(buffer)
    got = cudf.Series(tensor, dtype=np.bool_)

    assert_eq(got, cudf.Series(buffer, dtype=np.bool_))

    index = cudf.Index([], dtype="float64")
    tensor = torch.tensor(index)
    got = cudf.Index(tensor)
    assert_eq(got, index)

    index = cudf.RangeIndex(start=0, stop=3)
    tensor = torch.tensor(index)
    got = cudf.Series(tensor)

    assert_eq(got, cudf.Series(index))

    index = cudf.Index([1, 2, 8, 6])
    tensor = torch.tensor(index)
    got = cudf.Index(tensor)

    assert_eq(got, index)

    str_series = cudf.Series(["a", "g"])

    with pytest.raises(AttributeError):
        str_series.__cuda_array_interface__

    cat_series = str_series.astype("category")

    with pytest.raises(TypeError):
        cat_series.__cuda_array_interface__


def test_series_arrow_struct_types_roundtrip():
    ps = pd.Series(
        [{"a": 1}, {"b": "abc"}],
        dtype=pd.ArrowDtype(pa.struct({"a": pa.int64(), "b": pa.string()})),
    )
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.from_pandas(ps)
        assert_eq(ps, gs)

    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.from_pandas(pdf)
        assert_eq(pdf, gdf)


def test_series_arrow_list_types_roundtrip():
    ps = pd.Series([[1], [2], [4]], dtype=pd.ArrowDtype(pa.list_(pa.int64())))
    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.from_pandas(ps)
        assert_eq(ps, gs)
    pdf = ps.to_frame()

    with cudf.option_context("mode.pandas_compatible", True):
        gdf = cudf.from_pandas(pdf)
        assert_eq(pdf, gdf)


def test_series_error_nan_mixed_types():
    ps = pd.Series([np.nan, "ab", "cd"], dtype=object)
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(MixedTypeError):
            cudf.from_pandas(ps)


@pytest.mark.parametrize("klass", [cudf.Series, cudf.Index])
def test_from_pandas_object_dtype_passed_dtype(klass):
    result = klass(pd.Series([True, False], dtype=object), dtype="int8")
    expected = klass(pa.array([1, 0], type=pa.int8()))
    assert_eq(result, expected)


def test_series_basic():
    # Make series from buffer
    a1 = np.arange(10, dtype=np.float64)
    series = cudf.Series(a1)
    assert len(series) == 10
    np.testing.assert_equal(series.to_numpy(), np.hstack([a1]))


def test_series_from_cupy_scalars():
    data = [0.1, 0.2, 0.3]
    data_np = np.array(data)
    data_cp = cp.array(data)
    s_np = cudf.Series([data_np[0], data_np[2]])
    s_cp = cudf.Series([data_cp[0], data_cp[2]])
    assert_eq(s_np, s_cp)


def test_to_dense_array():
    rng = np.random.default_rng(seed=0)
    data = rng.random(8)
    mask = rng.choice([True, False], size=len(data))
    sr = cudf.Series(data)
    sr.loc[mask] = None
    assert sr.has_nulls
    assert sr.null_count != len(sr)
    filled = sr.to_numpy(na_value=np.nan)
    dense = sr.dropna().to_numpy()
    assert dense.size < filled.size
    assert filled.size == len(sr)


def test_series_np_array_all_nan_object_raises():
    with pytest.raises(MixedTypeError):
        cudf.Series(np.array([np.nan, np.nan], dtype=object))


@pytest.mark.parametrize(
    "ps",
    [
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
    ],
)
def test_roundtrip_series_plc_column(ps):
    expect = cudf.Series(ps)
    actual = cudf.Series.from_pylibcudf(*expect.to_pylibcudf())
    if expect.dtype == "object":
        actual = actual.astype("object")
    assert_eq(expect, actual)


def test_series_structarray_construction_with_nulls():
    fields = [
        pa.array([1], type=pa.int64()),
        pa.array([None], type=pa.int64()),
        pa.array([3], type=pa.int64()),
    ]
    expect = pa.StructArray.from_arrays(fields, ["a", "b", "c"])
    got = cudf.Series(expect).to_arrow()

    assert expect == got


@pytest.mark.parametrize(
    "data",
    [
        [{}],
        [{"a": None}],
        [{"a": 1}],
        [{"a": "one"}],
        [{"a": 1}, {"a": 2}],
        [{"a": 1, "b": "one"}, {"a": 2, "b": "two"}],
        [{"b": "two", "a": None}, None, {"a": "one", "b": "two"}],
    ],
)
def test_create_struct_series(data):
    expect = pd.Series(data)
    got = cudf.Series(data)
    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize(
    "data",
    [
        pd.date_range("20010101", "20020215", freq="400h", name="times"),
        pd.date_range(
            "20010101", freq="243434324423423234ns", name="times", periods=10
        ),
    ],
)
def test_series_from_pandas_datetime_index(data):
    pd_data = pd.Series(data)
    gdf_data = cudf.Series(pd_data)
    assert_eq(pd_data, gdf_data)


@pytest.mark.parametrize(
    "dtype",
    ["datetime64[D]", "datetime64[W]", "datetime64[M]", "datetime64[Y]"],
)
def test_datetime_array_timeunit_cast(dtype):
    testdata = np.array(
        [
            np.datetime64("2016-11-20"),
            np.datetime64("2020-11-20"),
            np.datetime64("2019-11-20"),
            np.datetime64("1918-11-20"),
            np.datetime64("2118-11-20"),
        ],
        dtype=dtype,
    )

    gs = cudf.Series(testdata)
    ps = pd.Series(testdata)

    assert_eq(ps, gs)

    gdf = cudf.DataFrame()
    gdf["a"] = np.arange(5)
    gdf["b"] = testdata

    pdf = pd.DataFrame()
    pdf["a"] = np.arange(5)
    pdf["b"] = testdata
    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data",
    [
        np.array(["2263-01-01"], dtype="datetime64[D]"),
        np.array([datetime.datetime(2263, 1, 1)], dtype=object),
        np.array([np.datetime64("2263-01-01", "D")], dtype=object),
        np.array(["2263-01-01"], dtype=object),
        np.array(["1600-01-01"], dtype="datetime64[D]"),
    ],
    ids=[
        "datetime64[D]",
        "object-datetime.datetime",
        "object-numpy-scalar",
        "object-string",
        "pre-epoch",
    ],
)
@pytest.mark.parametrize("box", ["Series", "Index", "DataFrame"])
def test_constructor_datetime_out_of_bounds(box, data):
    # Forcing values beyond the target unit's representable range must
    # raise like pandas instead of silently wrapping around int64.
    assert_exceptions_equal(
        lfunc=getattr(pd, box),
        rfunc=getattr(cudf, box),
        lfunc_args_and_kwargs=([data], {"dtype": "datetime64[ns]"}),
        rfunc_args_and_kwargs=([data], {"dtype": "datetime64[ns]"}),
    )


@pytest.mark.parametrize("timeunit", ["D", "W", "M", "Y"])
def test_datetime_scalar_timeunit_cast(timeunit):
    testscalar = np.datetime64("2016-11-20", timeunit)

    gs = cudf.Series(testscalar)
    ps = pd.Series(testscalar)

    assert_eq(ps, gs, check_dtype=False)

    gdf = cudf.DataFrame()
    gdf["a"] = np.arange(5)
    gdf["b"] = testscalar

    pdf = pd.DataFrame()
    pdf["a"] = np.arange(5)
    pdf["b"] = testscalar

    assert gdf["b"].dtype == np.dtype("datetime64[s]")
    assert_eq(pdf, gdf, check_dtype=True)


def test_datetime_string_to_datetime_resolution_loss_raises():
    data = ["2020-01-01 00:00:00.00001"]
    dtype = "datetime64[s]"
    with pytest.raises(ValueError):
        cudf.Series(data, dtype=dtype)
    with pytest.raises(ValueError):
        pd.Series(data, dtype=dtype)


def test_timezone_pyarrow_array():
    pa_array = pa.array(
        [datetime.datetime(2020, 1, 1, tzinfo=datetime.timezone.utc)],
        type=pa.timestamp("ns", "UTC"),
    )
    result = cudf.Series(pa_array)
    expected = pa_array.to_pandas()
    assert_eq(result, expected)


def test_string_ingest(one_dimensional_array_types):
    expect = ["a", "a", "b", "c", "a"]
    data = one_dimensional_array_types(expect)
    got = cudf.Series(data)
    assert got.dtype == pd.StringDtype(na_value=np.nan)
    assert len(got) == 5
    for idx, val in enumerate(expect):
        assert expect[idx] == got[idx]


def test_decimal_invalid_precision():
    with pytest.raises(pa.ArrowInvalid):
        cudf.Series([10, 20, 30], dtype=cudf.Decimal64Dtype(2, 2))

    with pytest.raises(pa.ArrowInvalid):
        cudf.Series([decimal.Decimal("300")], dtype=cudf.Decimal64Dtype(2, 1))


@pytest.mark.parametrize(
    "input_obj", [[decimal.Decimal(1), cudf.NA, decimal.Decimal(3)]]
)
def test_series_construction_decimals_with_nulls(input_obj):
    expect = pa.array(input_obj, from_pandas=True)
    got = cudf.Series(input_obj).to_arrow()

    assert expect.equals(got)


@pytest.mark.parametrize(
    "klass", ["Series", "DatetimeIndex", "Index", "CategoricalIndex"]
)
def test_pandas_compatible_non_zoneinfo_raises(klass):
    pytz = pytest.importorskip("pytz")
    tz = pytz.timezone("US/Pacific")
    tz_aware_data = [pd.Timestamp("2020-01-01", tz="UTC").tz_convert(tz)]
    pandas_obj = getattr(pd, klass)(tz_aware_data)
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(NotImplementedError):
            cudf.from_pandas(pandas_obj)


@pytest.mark.parametrize(
    "klass", ["Series", "DatetimeIndex", "Index", "CategoricalIndex"]
)
def test_from_pandas_obj_tz_aware(klass):
    tz = zoneinfo.ZoneInfo("US/Pacific")
    tz_aware_data = [pd.Timestamp("2020-01-01", tz="UTC").tz_convert(tz)]
    pandas_obj = getattr(pd, klass)(tz_aware_data)
    result = cudf.from_pandas(pandas_obj)
    expected = getattr(cudf, klass)(tz_aware_data)
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "klass", ["Series", "DatetimeIndex", "Index", "CategoricalIndex"]
)
def test_from_pandas_obj_tz_aware_unsupported(klass):
    tz = datetime.timezone(datetime.timedelta(hours=1))
    tz_aware_data = [pd.Timestamp("2020-01-01", tz="UTC").tz_convert(tz)]
    pandas_obj = getattr(pd, klass)(tz_aware_data)
    with pytest.raises(NotImplementedError):
        cudf.from_pandas(pandas_obj)


@pytest.mark.parametrize(
    "codes",
    [
        [],
        [0],
        [0, 1, 2],
        [0, 1, -1],
        [0, 1, 2, -1],
    ],
)
@pytest.mark.parametrize(
    "categories",
    [
        ["aa", "bb", "cc"],
        [2, 4, 10],
    ],
)
def test_categorical_creation(codes, categories):
    data = pd.Categorical.from_codes(codes, categories)
    expected = pd.Series(data)
    got = cudf.Series(data)
    assert_eq(expected, got)

    got = cudf.Series(data, dtype=cudf.from_pandas(data.dtype))
    assert_eq(expected, got)

    expected = pd.Series(data, dtype="category")
    got = cudf.Series(data, dtype="category")
    assert_eq(expected, got)


@pytest.mark.parametrize("tz", ["US/Eastern", "Asia/Tokyo", "UTC"])
@pytest.mark.parametrize(
    "data",
    [
        ["2024-01-01", "2024-02-01", "2024-01-01"],
        ["2024-01-01", None, "2024-02-01"],
    ],
)
def test_categorical_creation_tz_aware(data, tz):
    cats = pd.to_datetime(data).tz_localize(tz)
    pd_cat = pd.Categorical(cats)
    expected = pd.Series(pd_cat)
    got = cudf.Series(pd_cat)
    assert_eq(expected, got)
    assert got.cat.categories.dtype == expected.cat.categories.dtype


@pytest.mark.parametrize("input_obj", [[1, cudf.NA, 3]])
def test_series_construction_with_nulls_as_category(
    input_obj, all_supported_types_as_str
):
    if all_supported_types_as_str == "category":
        pytest.skip(f"No {all_supported_types_as_str} scalar.")
    if all_supported_types_as_str.startswith(
        "datetime"
    ) or all_supported_types_as_str.startswith("timedelta"):
        pytest.skip("Test intended for numeric and string scalars.")
    dtype = cudf.dtype(all_supported_types_as_str)
    input_obj = [
        dtype.type(v) if v is not cudf.NA else cudf.NA for v in input_obj
    ]

    expect = pd.Series(input_obj, dtype="category")
    got = cudf.Series(input_obj, dtype="category")

    assert_eq(expect, got)


@pytest.mark.parametrize("scalar", [1, "a", None, 10.2])
def test_cat_from_scalar(scalar):
    ps = pd.Series(scalar, dtype="category")
    gs = cudf.Series(scalar, dtype="category")

    assert_eq(ps, gs)


def test_categorical_interval_pandas_roundtrip():
    expected = cudf.Series(cudf.interval_range(0, 5)).astype("category")
    result = cudf.Series(expected.to_pandas())
    assert_eq(result, expected)

    expected = pd.Series(pd.interval_range(0, 5)).astype("category")
    result = cudf.Series(expected).to_pandas()
    assert_eq(result, expected)


def test_from_arrow_missing_categorical():
    pd_cat = pd.Categorical.from_codes([0, 1, -1], categories=["a", "b"])
    pa_cat = pa.array(pd_cat, from_pandas=True)
    gd_cat = cudf.Series(pa_cat)

    assert isinstance(gd_cat, cudf.Series)
    assert_eq(
        pd.Series(pa_cat.to_pandas()),  # PyArrow returns a pd.Categorical
        gd_cat.to_pandas(),
    )


def test_from_python_array(numeric_types_as_str):
    rng = np.random.default_rng(seed=0)
    np_arr = rng.integers(0, 100, 10).astype(numeric_types_as_str)
    data = memoryview(np_arr)
    data = array.array(data.format, data)

    gs = cudf.Series(data)

    np.testing.assert_equal(gs.to_numpy(), np_arr)


def test_as_column_types():
    col = as_column(cudf.Series([], dtype="float64"))
    assert_eq(col.dtype, np.dtype("float64"))
    gds = cudf.Series._from_column(col)
    pds = pd.Series(pd.Series([], dtype="float64"))

    assert_eq(pds, gds)

    col = as_column(
        cudf.Series([], dtype="float64"), dtype=np.dtype(np.float32)
    )
    assert_eq(col.dtype, np.dtype("float32"))
    gds = cudf.Series._from_column(col)
    pds = pd.Series(pd.Series([], dtype="float32"))

    assert_eq(pds, gds)

    col = as_column(cudf.Series([], dtype="float64"), dtype=cudf.dtype("str"))
    assert_eq(col.dtype, pd.StringDtype(na_value=np.nan))
    gds = cudf.Series._from_column(col)
    pds = pd.Series(pd.Series([], dtype="str"))
    assert_eq(pds, gds)

    col = as_column(cudf.Series([], dtype="float64"), dtype=cudf.dtype("str"))
    assert_eq(col.dtype, pd.StringDtype(na_value=np.nan))
    gds = cudf.Series._from_column(col)
    pds = pd.Series(pd.Series([], dtype="str"))
    assert_eq(pds, gds)

    pds = pd.Series(np.array([1, 2, 3]), dtype="float32")
    gds = cudf.Series._from_column(
        as_column(np.array([1, 2, 3]), dtype=np.dtype(np.float32))
    )

    assert_eq(pds, gds)

    pds = pd.Series([1, 2, 3], dtype="float32")
    gds = cudf.Series([1, 2, 3], dtype="float32")

    assert_eq(pds, gds)

    pds = pd.Series([], dtype="float64")
    gds = cudf.Series._from_column(as_column(pds))
    assert_eq(pds, gds)

    pds = pd.Series([1, 2, 4], dtype="int64")
    gds = cudf.Series._from_column(
        as_column(cudf.Series([1, 2, 4]), dtype=np.dtype("int64"))
    )

    assert_eq(pds, gds)

    pds = pd.Series([1.2, 18.0, 9.0], dtype="float32")
    gds = cudf.Series._from_column(
        as_column(cudf.Series([1.2, 18.0, 9.0]), dtype=np.dtype(np.float32))
    )

    assert_eq(pds, gds)

    pds = pd.Series([1.2, 18.0, 9.0], dtype="str")
    gds = cudf.Series._from_column(
        as_column(cudf.Series([1.2, 18.0, 9.0]), dtype=cudf.dtype("str"))
    )

    assert_eq(pds, gds)

    pds = pd.Series(pd.Index(["1", "18", "9"]), dtype="int")
    gds = cudf.Series(cudf.Index(["1", "18", "9"]), dtype="int")

    assert_eq(pds, gds)


def test_series_type_invalid_error():
    with cudf.option_context("mode.pandas_compatible", True):
        with pytest.raises(ValueError):
            cudf.Series(["a", "b", "c"], dtype="Int64")


def test_series_constructor_numpy_dtype_str(pandas_compatible):
    data = ["a"]
    dtype = np.dtype(str)
    expected = pd.Series(data, dtype=dtype)
    result = cudf.Series(data, dtype=dtype)
    assert result.dtype == np.dtype(object)
    assert_eq(result, expected)


def test_series_constructor_dtype_is_pandas_nullable_extension_type(
    all_supported_pandas_nullable_extension_dtypes,
):
    scalar, dtype = all_supported_pandas_nullable_extension_dtypes
    result = cudf.Series([scalar], dtype=dtype)
    expected = pd.Series([scalar], dtype=dtype)
    assert result.dtype == expected.dtype
    assert_eq(result, expected)


def test_series_constructor_dtype_is_pandas_arrowdtype(
    all_supported_pandas_arrowdtypes,
):
    scalar, dtype = all_supported_pandas_arrowdtypes
    if (
        pa.types.is_decimal32(dtype.pyarrow_dtype)
        or pa.types.is_decimal64(dtype.pyarrow_dtype)
    ) and parse(pa.__version__) < parse("20"):
        pytest.skip("pyarrow < 19 converts decimal32/64 to object")
    result = cudf.Series([scalar], dtype=dtype)
    expected = pd.Series([scalar], dtype=dtype)
    assert result.dtype == expected.dtype
    assert_eq(result, expected)


def test_build_series_from_nullable_pandas_dtype(
    all_supported_pandas_nullable_extension_dtypes,
):
    scalar, dtype = all_supported_pandas_nullable_extension_dtypes
    expected = pd.Series([scalar, pd.NA], dtype=dtype)
    result = cudf.Series(expected)

    assert result.dtype == expected.dtype

    expect_mask = expected.isna()
    got_mask = result.isna().to_numpy()

    np.testing.assert_array_equal(expect_mask, got_mask)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
import datetime
import decimal
import operator
import re
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import cudf
from cudf.testing import assert_eq
from cudf.testing._utils import (
    _decimal_series,
    assert_exceptions_equal,
    gen_rand_series,
)


@pytest.mark.parametrize(
    "sr1", [pd.Series([10, 11, 12], index=["a", "b", "z"]), pd.Series(["a"])]
)
@pytest.mark.parametrize(
    "sr2",
    [pd.Series([], dtype="float64"), pd.Series(["a", "a", "c", "z", "A"])],
)
def test_series_error_equality(sr1, sr2, comparison_op):
    gsr1 = cudf.from_pandas(sr1)
    gsr2 = cudf.from_pandas(sr2)

    assert_exceptions_equal(
        comparison_op, comparison_op, ([sr1, sr2],), ([gsr1, gsr2],)
    )


@pytest.mark.parametrize(
    "data,other",
    [
        ([1000000, 200000, 3000000], [1000000, 200000, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, None]),
        ([], []),
        ([None], [None]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [12, 12, 22, 343, 4353534, 435342],
        ),
        ([1000000, 200000, 3000000], [200000, 34543, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, 3000000]),
        ([None], [1]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [None, 1, 220, 3, 34, 4353423287],
        ),
        (np.array([10, 20, 30, None, 100]), np.array([10, 20, 30, None, 100])),
        (cp.asarray([10, 20, 30, 100]), cp.asarray([10, 20, 30, 100])),
    ],
)
def test_timedelta_ops_misc_inputs(
    data, other, timedelta_types_as_str, binary_op_method
):
    if binary_op_method in {"mul", "rmul", "pow", "rpow"}:
        pytest.skip(f"Test not applicable for {binary_op_method}")
    gsr = cudf.Series(data, dtype=timedelta_types_as_str)
    other_gsr = cudf.Series(other, dtype=timedelta_types_as_str)

    psr = gsr.to_pandas()
    other_psr = other_gsr.to_pandas()

    expected = getattr(psr, binary_op_method)(other_psr)
    actual = getattr(gsr, binary_op_method)(other_gsr)
    if binary_op_method in ("eq", "lt", "gt", "le", "ge"):
        actual = actual.fillna(False)
    elif binary_op_method == "ne":
        actual = actual.fillna(True)

    if binary_op_method == "floordiv":
        expected[actual.isna().to_pandas()] = np.nan

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "datetime_data,timedelta_data",
    [
        ([1000000, 200000, 3000000], [1000000, 200000, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, None]),
        ([], []),
        ([None], [None]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [12, 12, 22, 343, 4353534, 435342],
        ),
        ([1000000, 200000, 3000000], [200000, 34543, 3000000]),
        ([1000000, 200000, None], [1000000, 200000, 3000000]),
        ([None], [1]),
        (
            [12, 12, 22, 343, 4353534, 435342],
            [None, 1, 220, 3, 34, 4353423287],
        ),
        (np.array([10, 20, 30, None, 100]), np.array([10, 20, 30, None, 100])),
        (cp.asarray([10, 20, 30, 100]), cp.asarray([10, 20, 30, 100])),
        (
            [12, 11, 232, 223432411, 2343241, 234324, 23234],
            [11, 1132324, 2322323111, 23341, 2434, 332, 323],
        ),
        (
            [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
            [11, 1132324, 2322323111, 23341, 2434, 332, 323],
        ),
        (
            [11, 1132324, 2322323111, 23341, 2434, 332, 323],
            [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        ),
        (
            [1.321, 1132.324, 23223231.11, 233.41, 0.2434, 332, 323],
            [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
        ),
    ],
)
@pytest.mark.parametrize("ops", ["add", "sub"])
def test_timedelta_ops_datetime_inputs(
    datetime_types_as_str,
    timedelta_types_as_str,
    datetime_data,
    timedelta_data,
    ops,
):
    gsr_datetime = cudf.Series(datetime_data, dtype=datetime_types_as_str)
    gsr_timedelta = cudf.Series(timedelta_data, dtype=timedelta_types_as_str)

    psr_datetime = gsr_datetime.to_pandas()
    psr_timedelta = gsr_timedelta.to_pandas()

    expected = getattr(psr_datetime, ops)(psr_timedelta)
    actual = getattr(gsr_datetime, ops)(gsr_timedelta)

    assert_eq(expected, actual)

    if ops == "add":
        expected = getattr(psr_timedelta, ops)(psr_datetime)
        actual = getattr(gsr_timedelta, ops)(gsr_datetime)

        assert_eq(expected, actual)
    elif ops == "sub":
        assert_exceptions_equal(
            lfunc=operator.sub,
            rfunc=operator.sub,
            lfunc_args_and_kwargs=([psr_timedelta, psr_datetime],),
            rfunc_args_and_kwargs=([gsr_timedelta, gsr_datetime],),
        )


@pytest.mark.parametrize(
    "df",
    [
        pd.DataFrame(
            {
                "A": pd.Series(pd.date_range("2012-1-1", periods=3, freq="D")),
                "B": pd.Series(
                    pd.timedelta_range(start="1 day", periods=3, freq="D")
                ),
            }
        ),
        pd.DataFrame(
            {
                "A": pd.Series(
                    pd.date_range("1994-1-1", periods=10, freq="D")
                ),
                "B": pd.Series(
                    pd.timedelta_range(start="1 day", periods=10, freq="D")
                ),
            }
        ),
    ],
)
@pytest.mark.parametrize("op", ["add", "sub"])
def test_timedelta_dataframe_ops(df, op):
    pdf = df
    gdf = cudf.from_pandas(pdf)

    if op == "add":
        pdf["C"] = pdf["A"] + pdf["B"]
        gdf["C"] = gdf["A"] + gdf["B"]
    elif op == "sub":
        pdf["C"] = pdf["A"] - pdf["B"]
        gdf["C"] = gdf["A"] - gdf["B"]

    assert_eq(pdf, gdf)


@pytest.mark.parametrize(
    "data",
    [
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
        [1.321, 1132.324, 23223231.11, 233.41, 332, 323],
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize(
    "other_scalars",
    [
        datetime.timedelta(days=768),
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        datetime.timedelta(minutes=447),
        datetime.timedelta(hours=447),
        datetime.timedelta(weeks=734),
        np.timedelta64(4, "s"),
        np.timedelta64(456, "D"),
        np.timedelta64(46, "h"),
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
def test_timedelta_series_ops_with_scalars(
    data, other_scalars, timedelta_types_as_str, arithmetic_op_method, request
):
    if arithmetic_op_method in {
        "mul",
        "rmul",
        "rtruediv",
        "pow",
        "rpow",
        "radd",
        "rsub",
        "rfloordiv",
        "rmod",
    }:
        pytest.skip(f"Test not applicable for {arithmetic_op_method}")
    gsr = cudf.Series(data=data, dtype=timedelta_types_as_str)
    psr = gsr.to_pandas()

    if arithmetic_op_method == "add":
        expected = psr + other_scalars
        actual = gsr + other_scalars
    elif arithmetic_op_method == "sub":
        expected = psr - other_scalars
        actual = gsr - other_scalars
    elif arithmetic_op_method == "truediv":
        expected = psr / other_scalars
        actual = gsr / other_scalars
    elif arithmetic_op_method == "floordiv":
        expected = psr // other_scalars
        actual = gsr // other_scalars
    elif arithmetic_op_method == "mod":
        expected = psr % other_scalars
        actual = gsr % other_scalars

    assert_eq(expected, actual)

    if arithmetic_op_method == "add":
        expected = other_scalars + psr
        actual = other_scalars + gsr
    elif arithmetic_op_method == "sub":
        expected = other_scalars - psr
        actual = other_scalars - gsr
    elif arithmetic_op_method == "truediv":
        expected = other_scalars / psr
        actual = other_scalars / gsr
    elif arithmetic_op_method == "floordiv":
        expected = other_scalars // psr
        actual = other_scalars // gsr
    elif arithmetic_op_method == "mod":
        expected = other_scalars % psr
        actual = other_scalars % gsr

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "reverse",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "timedelta modulo by zero is dubiously defined in "
                    "both pandas and cuDF "
                    "(see https://github.com/rapidsai/cudf/issues/5938)"
                ),
            ),
        ),
    ],
)
def test_timedelta_series_mod_with_scalar_zero(reverse):
    gsr = cudf.Series(data=[0.2434], dtype=np.timedelta64(1, "ns"))
    psr = gsr.to_pandas()
    scalar = datetime.timedelta(days=768)
    if reverse:
        expected = scalar % psr
        actual = scalar % gsr
    else:
        expected = psr % scalar
        actual = gsr % scalar
    assert_eq(expected, actual)


def test_timedelta_invalid_ops():
    sr = cudf.Series([1, 2, 3], dtype="timedelta64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, 1],),
        rfunc_args_and_kwargs=([sr, 1],),
    )

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, "a"],),
        rfunc_args_and_kwargs=([sr, "a"],),
    )

    dt_sr = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    dt_psr = dt_sr.to_pandas()

    assert_exceptions_equal(
        lfunc=operator.mod,
        rfunc=operator.mod,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.mod,
        rfunc=operator.mod,
        lfunc_args_and_kwargs=([psr, "a"],),
        rfunc_args_and_kwargs=([sr, "a"],),
        check_exception_type=False,
    )

    assert_exceptions_equal(
        lfunc=operator.gt,
        rfunc=operator.gt,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.lt,
        rfunc=operator.lt,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.ge,
        rfunc=operator.ge,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.le,
        rfunc=operator.le,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.truediv,
        rfunc=operator.truediv,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.floordiv,
        rfunc=operator.floordiv,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.mul,
        rfunc=operator.mul,
        lfunc_args_and_kwargs=([psr, dt_psr],),
        rfunc_args_and_kwargs=([sr, dt_sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.mul,
        rfunc=operator.mul,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
        check_exception_type=False,
    )

    assert_exceptions_equal(
        lfunc=operator.xor,
        rfunc=operator.xor,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
    )


@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_timdelta_binop_tz_timestamp(op):
    s = cudf.Series([1, 2, 3], dtype="timedelta64[ns]")
    pd_tz_timestamp = pd.Timestamp("1970-01-01 00:00:00.000000001", tz="utc")
    with pytest.raises(TypeError):
        op(s, pd_tz_timestamp)
    date_tz_scalar = datetime.datetime.now(datetime.timezone.utc)
    with pytest.raises(TypeError):
        op(s, date_tz_scalar)


def test_timedelta_series_cmpops_pandas_compatibility(comparison_op):
    gsr1 = cudf.Series(
        data=[123, 456, None, 321, None], dtype="timedelta64[ns]"
    )
    psr1 = gsr1.to_pandas()

    gsr2 = cudf.Series(
        data=[123, 456, 789, None, None], dtype="timedelta64[ns]"
    )
    psr2 = gsr2.to_pandas()

    expect = comparison_op(psr1, psr2)
    with cudf.option_context("mode.pandas_compatible", True):
        got = comparison_op(gsr1, gsr2)

    assert_eq(expect, got)


def test_compare_ops_numeric_to_null_pandas_compatible(comparison_op):
    data = [None, 1, 3]
    pser = pd.Series(data)
    gser = cudf.Series(data)
    expected = comparison_op(pser, 2)
    with cudf.option_context("mode.pandas_compatible", True):
        result = comparison_op(gser, 2)
    assert_eq(expected, result)


@pytest.mark.parametrize("op", [operator.add, operator.sub])
@pytest.mark.parametrize("reflect", [False, True])
def test_numeric_series_pd_nat_raises(op, reflect):
    data = [0, 1, 2, 3, 4]
    pser = pd.Series(data, dtype="int64")
    gser = cudf.Series(data, dtype="int64")
    pleft, pright = (pd.NaT, pser) if reflect else (pser, pd.NaT)
    gleft, gright = (pd.NaT, gser) if reflect else (gser, pd.NaT)
    assert_exceptions_equal(
        lfunc=op,
        rfunc=op,
        lfunc_args_and_kwargs=([pleft, pright],),
        rfunc_args_and_kwargs=([gleft, gright],),
    )


def test_compare_ops_decimal_to_null_pandas_compatible(comparison_op):
    data = pa.array([None, 1, 3], pa.decimal128(3, 2))
    gser = cudf.Series(data)
    expected = comparison_op(gser.to_pandas(arrow_type=True), 2)
    result = comparison_op(gser, 2).to_pandas(arrow_type=True)
    assert_eq(expected, result)


def test_string_equality():
    data1 = ["b", "c", "d", "a", "c"]
    data2 = ["a", None, "c", "a", "c"]

    ps1 = pd.Series(data1)
    ps2 = pd.Series(data2)
    gs1 = cudf.Series(data1)
    gs2 = cudf.Series(data2)

    expect = ps1 == ps2
    got = gs1 == gs2

    assert_eq(expect, got.fillna(False))

    expect = ps1 == "m"
    got = gs1 == "m"

    assert_eq(expect, got.fillna(False))

    ps1 = pd.Series(["a"])
    gs1 = cudf.Series(["a"])

    expect = ps1 == "m"
    got = gs1 == "m"

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "lhs",
    [
        ["Cbe", "cbe", "CbeD", "Cb", "ghi", "Cb"],
        ["abc", "xyz", "a", "ab", "123", "097"],
    ],
)
@pytest.mark.parametrize(
    "rhs",
    [
        ["Cbe", "cbe", "CbeD", "Cb", "ghi", "Cb"],
        ["a", "a", "a", "a", "A", "z"],
    ],
)
def test_string_binary_op_add(lhs, rhs):
    pds = pd.Series(lhs) + pd.Series(rhs)
    gds = cudf.Series(lhs) + cudf.Series(rhs)

    assert_eq(pds, gds)


def test_concatenate_rows_of_lists():
    pser = pd.Series([["a", "a"], ["b"], ["c"]])
    gser = cudf.Series([["a", "a"], ["b"], ["c"]])

    expect = pser + pser
    got = gser + gser

    assert_eq(expect, got)


def test_concatenate_list_with_nonlist():
    gser1 = cudf.Series([["a", "c"], ["b", "d"], ["c", "d"]])
    gser2 = cudf.Series(["a", "b", "c"])
    with pytest.raises(TypeError):
        gser1 + gser2


def test_datetime_series_binops_pandas(
    datetime_types_as_str, datetime_types_as_str2
):
    dti = pd.date_range("20010101", "20020215", freq="400h", name="times")
    pd_data_1 = pd.Series(dti).astype(datetime_types_as_str)
    pd_data_2 = pd_data_1.copy().astype(datetime_types_as_str2)
    gdf_data_1 = cudf.Series(pd_data_1).astype(datetime_types_as_str)
    gdf_data_2 = cudf.Series(pd_data_2).astype(datetime_types_as_str2)
    assert_eq(pd_data_1, gdf_data_1)
    assert_eq(pd_data_2, gdf_data_2)
    assert_eq(pd_data_1 < pd_data_2, gdf_data_1 < gdf_data_2)
    assert_eq(pd_data_1 > pd_data_2, gdf_data_1 > gdf_data_2)
    assert_eq(pd_data_1 == pd_data_2, gdf_data_1 == gdf_data_2)
    assert_eq(pd_data_1 <= pd_data_2, gdf_data_1 <= gdf_data_2)
    assert_eq(pd_data_1 >= pd_data_2, gdf_data_1 >= gdf_data_2)


def test_datetime_series_binops_numpy(
    datetime_types_as_str, datetime_types_as_str2
):
    dti = pd.date_range("20010101", "20020215", freq="400h", name="times")
    pd_data_1 = pd.Series(dti)
    pd_data_2 = pd_data_1
    gdf_data_1 = cudf.Series(pd_data_1).astype(datetime_types_as_str)
    gdf_data_2 = cudf.Series(pd_data_2).astype(datetime_types_as_str2)
    np_data_1 = np.array(pd_data_1).astype(datetime_types_as_str)
    np_data_2 = np.array(pd_data_2).astype(datetime_types_as_str2)
    np.testing.assert_equal(np_data_1, gdf_data_1.to_numpy())
    np.testing.assert_equal(np_data_2, gdf_data_2.to_numpy())
    np.testing.assert_equal(
        np.less(np_data_1, np_data_2), (gdf_data_1 < gdf_data_2).to_numpy()
    )
    np.testing.assert_equal(
        np.greater(np_data_1, np_data_2), (gdf_data_1 > gdf_data_2).to_numpy()
    )
    np.testing.assert_equal(
        np.equal(np_data_1, np_data_2), (gdf_data_1 == gdf_data_2).to_numpy()
    )
    np.testing.assert_equal(
        np.less_equal(np_data_1, np_data_2),
        (gdf_data_1 <= gdf_data_2).to_numpy(),
    )
    np.testing.assert_equal(
        np.greater_equal(np_data_1, np_data_2),
        (gdf_data_1 >= gdf_data_2).to_numpy(),
    )


@pytest.mark.parametrize(
    "data",
    [
        pd.date_range("20010101", "20020215", freq="400h", name="times"),
        pd.date_range(
            "20010101", freq="243434324423423234ns", name="times", periods=10
        ),
    ],
)
def test_dt_ops(data):
    pd_data = pd.Series(data)
    gdf_data = cudf.Series(data)

    assert_eq(pd_data == pd_data, gdf_data == gdf_data)
    assert_eq(pd_data < pd_data, gdf_data < gdf_data)
    assert_eq(pd_data > pd_data, gdf_data > gdf_data)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3, 4, 10, 100, 20000],
        [None] * 7,
        [10, 20, 30, None, 100, 200, None],
        [3223.234, 342.2332, 23423.23, 3343.23324, 23432.2323, 242.23, 233],
    ],
)
@pytest.mark.parametrize(
    "other",
    [
        [1, 2, 3, 4, 10, 100, 20000],
        [None] * 7,
        [10, 20, 30, None, 100, 200, None],
        [3223.234, 342.2332, 23423.23, 3343.23324, 23432.2323, 242.23, 233],
        datetime.datetime(1993, 6, 22, 13, 30),
        datetime.datetime(2005, 1, 22, 10, 00),
        np.datetime64("2005-02"),
        np.datetime64("2005-02-25"),
        np.datetime64("2005-02-25T03:30"),
        np.datetime64("nat"),
        # TODO: https://github.com/pandas-dev/pandas/issues/52295
    ],
)
def test_datetime_subtract(
    data, other, datetime_types_as_str, datetime_types_as_str2
):
    gsr = cudf.Series(data, dtype=datetime_types_as_str)
    psr = gsr.to_pandas()

    if isinstance(other, np.datetime64):
        gsr_other = other
        psr_other = other
    else:
        gsr_other = cudf.Series(other, dtype=datetime_types_as_str2)
        psr_other = gsr_other.to_pandas()

    expected = psr - psr_other
    actual = gsr - gsr_other

    assert_eq(expected, actual)

    expected = psr_other - psr
    actual = gsr_other - gsr

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "data",
    [
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
        [12, 11, 2.32, 2234.32411, 2343.241, 23432.4, 23234],
    ],
)
@pytest.mark.parametrize(
    "other_scalars",
    [
        datetime.timedelta(days=768),
        datetime.timedelta(seconds=768),
        datetime.timedelta(microseconds=7),
        datetime.timedelta(minutes=447),
        datetime.timedelta(hours=447),
        datetime.timedelta(weeks=734),
        np.timedelta64(4, "s"),
        np.timedelta64(456, "D"),
        np.timedelta64(46, "h"),
        np.timedelta64("nat"),
        np.timedelta64(1, "s"),
        np.timedelta64(1, "ms"),
        np.timedelta64(1, "us"),
        np.timedelta64(1, "ns"),
    ],
)
@pytest.mark.parametrize("op", ["add", "sub"])
def test_datetime_series_ops_with_scalars(
    data, other_scalars, datetime_types_as_str, op
):
    gsr = cudf.Series(data=data, dtype=datetime_types_as_str)
    psr = gsr.to_pandas()

    if op == "add":
        expected = psr + other_scalars
        actual = gsr + other_scalars
    elif op == "sub":
        expected = psr - other_scalars
        actual = gsr - other_scalars

    assert_eq(expected, actual)

    if op == "add":
        expected = other_scalars + psr
        actual = other_scalars + gsr

        assert_eq(expected, actual)

    elif op == "sub":
        assert_exceptions_equal(
            lfunc=operator.sub,
            rfunc=operator.sub,
            lfunc_args_and_kwargs=([other_scalars, psr],),
            rfunc_args_and_kwargs=([other_scalars, gsr],),
        )


@pytest.mark.parametrize("data", ["20110101", "20120101", "20130101"])
@pytest.mark.parametrize("other_scalars", ["20110101", "20120101", "20130101"])
def test_datetime_series_cmpops_with_scalars(
    data, other_scalars, datetime_types_as_str, comparison_op
):
    gsr = cudf.Series(data=data, dtype=datetime_types_as_str)
    psr = gsr.to_pandas()

    expect = comparison_op(psr, other_scalars)
    got = comparison_op(gsr, other_scalars)

    assert_eq(expect, got)


def test_datetime_invalid_ops():
    sr = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    psr = sr.to_pandas()

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, pd.Timestamp(1513393355.5, unit="s")],),
        rfunc_args_and_kwargs=([sr, pd.Timestamp(1513393355.5, unit="s")],),
    )

    assert_exceptions_equal(
        lfunc=operator.truediv,
        rfunc=operator.truediv,
        lfunc_args_and_kwargs=([psr, pd.Timestamp(1513393355.5, unit="s")],),
        rfunc_args_and_kwargs=([sr, pd.Timestamp(1513393355.5, unit="s")],),
    )

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.floordiv,
        rfunc=operator.floordiv,
        lfunc_args_and_kwargs=([psr, psr],),
        rfunc_args_and_kwargs=([sr, sr],),
    )

    assert_exceptions_equal(
        lfunc=operator.floordiv,
        rfunc=operator.floordiv,
        lfunc_args_and_kwargs=([psr, pd.Timestamp(1513393355.5, unit="s")],),
        rfunc_args_and_kwargs=([sr, pd.Timestamp(1513393355.5, unit="s")],),
    )

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([psr, 1],),
        rfunc_args_and_kwargs=([sr, 1],),
    )

    assert_exceptions_equal(
        lfunc=operator.truediv,
        rfunc=operator.truediv,
        lfunc_args_and_kwargs=([psr, "a"],),
        rfunc_args_and_kwargs=([sr, "a"],),
    )

    assert_exceptions_equal(
        lfunc=operator.mul,
        rfunc=operator.mul,
        lfunc_args_and_kwargs=([psr, 1],),
        rfunc_args_and_kwargs=([sr, 1],),
    )


def test_datetime_binop_tz_timestamp(comparison_op):
    s = cudf.Series([1, 2, 3], dtype="datetime64[ns]")
    pd_tz_timestamp = pd.Timestamp("1970-01-01 00:00:00.000000001", tz="utc")
    with pytest.raises(TypeError):
        comparison_op(s, pd_tz_timestamp)

    date_scalar = datetime.datetime.now(datetime.timezone.utc)
    with pytest.raises(TypeError):
        comparison_op(s, date_scalar)


def test_datetime_binop_tz_timestamp_pandas_compat(comparison_op):
    # Under pandas-compatible mode, comparing a tz-naive datetime Series
    # with a tz-aware Timestamp scalar must raise TypeError (matching
    # pandas), not NotImplementedError. eq/ne return False/True without
    # raising, matching pandas semantics.
    if comparison_op in (operator.eq, operator.ne):
        pytest.skip("eq/ne do not raise on tz mismatch")
    psr = pd.Series([1, 2, 3], dtype="datetime64[ns]")
    gsr = cudf.from_pandas(psr)
    pd_tz_timestamp = pd.Timestamp("1970-01-01 00:00:00.000000001", tz="utc")
    with cudf.option_context("mode.pandas_compatible", True):
        assert_exceptions_equal(
            lfunc=comparison_op,
            rfunc=comparison_op,
            lfunc_args_and_kwargs=([psr, pd_tz_timestamp],),
            rfunc_args_and_kwargs=([gsr, pd_tz_timestamp],),
        )


@pytest.mark.parametrize(
    "lhs_tz,rhs_tz",
    [(None, "US/Pacific"), ("US/Pacific", None)],
)
def test_datetime_column_tz_mismatch_pandas_compat(
    comparison_op, lhs_tz, rhs_tz
):
    # Column-vs-column comparison between tz-naive and tz-aware datetime
    # Series must raise TypeError under pandas-compatible mode for the
    # inequality ops; equality ops still return False/True.
    if comparison_op in (operator.eq, operator.ne):
        pytest.skip("eq/ne do not raise on tz mismatch")
    psr_lhs = pd.Series(pd.date_range("2020-01-01", periods=3, tz=lhs_tz))
    psr_rhs = pd.Series(pd.date_range("2020-01-01", periods=3, tz=rhs_tz))
    gsr_lhs = cudf.from_pandas(psr_lhs)
    gsr_rhs = cudf.from_pandas(psr_rhs)
    with cudf.option_context("mode.pandas_compatible", True):
        assert_exceptions_equal(
            lfunc=comparison_op,
            rfunc=comparison_op,
            lfunc_args_and_kwargs=([psr_lhs, psr_rhs],),
            rfunc_args_and_kwargs=([gsr_lhs, gsr_rhs],),
        )


def test_datetime_sub_tz_mismatch_pandas_compat():
    # Subtracting a tz-aware datetime Series from a tz-naive one must
    # raise TypeError under pandas-compatible mode.
    psr_naive = pd.Series(pd.date_range("2020-01-01", periods=3))
    psr_aware = pd.Series(
        pd.date_range("2020-01-01", periods=3, tz="US/Pacific")
    )
    gsr_naive = cudf.from_pandas(psr_naive)
    gsr_aware = cudf.from_pandas(psr_aware)
    with cudf.option_context("mode.pandas_compatible", True):
        assert_exceptions_equal(
            lfunc=operator.sub,
            rfunc=operator.sub,
            lfunc_args_and_kwargs=([psr_naive, psr_aware],),
            rfunc_args_and_kwargs=([gsr_naive, gsr_aware],),
        )


@pytest.mark.parametrize(
    "op", [operator.lt, operator.le, operator.gt, operator.ge]
)
@pytest.mark.parametrize("scalar", [None, np.nan])
def test_datetime_inequality_vs_none_pandas_compat(op, scalar):
    # pandas raises TypeError on inequality ops between datetime Series
    # and None / NaN scalars; cuDF must too under pandas-compatible mode.
    psr = pd.Series(pd.date_range("2020-01-01", periods=3))
    gsr = cudf.from_pandas(psr)
    with cudf.option_context("mode.pandas_compatible", True):
        assert_exceptions_equal(
            lfunc=op,
            rfunc=op,
            lfunc_args_and_kwargs=([psr, scalar],),
            rfunc_args_and_kwargs=([gsr, scalar],),
        )


def test_datetime_series_cmpops_pandas_compatibility(comparison_op):
    data1 = ["20110101", "20120101", None, "20140101", None]
    data2 = ["20110101", "20120101", "20130101", None, None]
    gsr1 = cudf.Series(data=data1, dtype="datetime64[ns]")
    psr1 = gsr1.to_pandas()

    gsr2 = cudf.Series(data=data2, dtype="datetime64[ns]")
    psr2 = gsr2.to_pandas()

    expect = comparison_op(psr1, psr2)
    with cudf.option_context("mode.pandas_compatible", True):
        got = comparison_op(gsr1, gsr2)
    assert_eq(expect, got)


def test_decimal_overflow():
    s = cudf.Series(
        [decimal.Decimal("0.0009384233522166997927180531650178250")]
    )
    result = s * s
    assert_eq(cudf.Decimal128Dtype(precision=38, scale=37), result.dtype)

    s = cudf.Series([1, 2], dtype=cudf.Decimal128Dtype(precision=38, scale=0))
    result = s * decimal.Decimal("1.0")
    assert_eq(cudf.Decimal128Dtype(precision=38, scale=1), result.dtype)


def test_decimal_binop_upcast_operands():
    ser1 = cudf.Series([0.51, 1.51, 2.51]).astype(cudf.Decimal64Dtype(18, 2))
    ser2 = cudf.Series([0.90, 0.96, 0.99]).astype(cudf.Decimal128Dtype(19, 2))
    result = ser1 + ser2
    expected = cudf.Series([1.41, 2.47, 3.50]).astype(
        cudf.Decimal128Dtype(20, 2)
    )
    assert_eq(result, expected)


def test_categorical_compare_ordered():
    cat1 = pd.Categorical(
        ["a", "a", "b", "c", "a"], categories=["a", "b", "c"], ordered=True
    )
    pdsr1 = pd.Series(cat1)
    sr1 = cudf.Series(cat1)
    cat2 = pd.Categorical(
        ["a", "b", "a", "c", "b"], categories=["a", "b", "c"], ordered=True
    )
    pdsr2 = pd.Series(cat2)
    sr2 = cudf.Series(cat2)

    # test equal
    out = sr1 == sr1
    assert out.dtype == np.bool_
    assert type(out[0]) is np.bool_
    assert np.all(out.to_numpy())
    assert np.all(pdsr1 == pdsr1)

    # test inequality
    out = sr1 != sr1
    assert not np.any(out.to_numpy())
    assert not np.any(pdsr1 != pdsr1)

    assert pdsr1.cat.ordered
    assert sr1.cat.ordered

    # test using ordered operators
    np.testing.assert_array_equal(pdsr1 < pdsr2, (sr1 < sr2).to_numpy())
    np.testing.assert_array_equal(pdsr1 > pdsr2, (sr1 > sr2).to_numpy())


def test_categorical_binary_add():
    cat = pd.Categorical(["a", "a", "b", "c", "a"], categories=["a", "b", "c"])
    pdsr = pd.Series(cat)
    sr = cudf.Series(cat)

    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([pdsr, pdsr],),
        rfunc_args_and_kwargs=([sr, sr],),
    )


def test_cat_series_binop_error():
    data_a = pd.Categorical(list("aababcabbc"), categories=list("abc"))
    data_b = np.arange(len(data_a))

    pd_ser_a = pd.Series(data_a)
    pd_ser_b = pd.Series(data_b)
    gdf_ser_a = cudf.Series(data_a)
    gdf_ser_b = cudf.Series(data_b)

    # lhs is categorical
    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([pd_ser_a, pd_ser_b],),
        rfunc_args_and_kwargs=([gdf_ser_a, gdf_ser_b],),
    )

    # lhs is numerical
    assert_exceptions_equal(
        lfunc=operator.add,
        rfunc=operator.add,
        lfunc_args_and_kwargs=([pd_ser_b, pd_ser_a],),
        rfunc_args_and_kwargs=([gdf_ser_b, gdf_ser_a],),
    )


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
def test_series_binop(arithmetic_op, obj_class):
    nelem = 1000
    rng = np.random.default_rng(seed=0)
    arr1 = rng.random(nelem) * 10000
    # Keeping a low value because CUDA 'pow' has 2 full range error
    arr2 = rng.random(nelem) * 10

    sr1 = cudf.Series(arr1)
    sr2 = cudf.Series(arr2)
    psr1 = sr1.to_pandas()
    psr2 = sr2.to_pandas()

    if obj_class == "Index":
        sr1 = cudf.Index(sr1)
        sr2 = cudf.Index(sr2)

    expect = arithmetic_op(psr1, psr2)
    result = arithmetic_op(sr1, sr2)

    if obj_class == "Index":
        result = cudf.Series(result)

    assert_eq(result, expect)


def test_series_binop_concurrent(arithmetic_op):
    def func(index):
        rng = np.random.default_rng(seed=0)
        arr = rng.random(100) * 10
        sr = cudf.Series(arr)

        result = arithmetic_op(sr.astype("int32"), sr)
        expect = arithmetic_op(arr.astype("int32"), arr)

        np.testing.assert_almost_equal(result.to_numpy(), expect, decimal=5)

    indices = range(10)
    with ThreadPoolExecutor(4) as e:  # four processes
        list(e.map(func, indices))


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
def test_series_binop_scalar(arithmetic_op, obj_class):
    nelem = 10
    rng = np.random.default_rng(seed=0)
    arr = rng.random(nelem)
    rhs = rng.choice(arr).item()

    sr = cudf.Series(arr)
    if obj_class == "Index":
        sr = cudf.Index(sr)

    result = arithmetic_op(sr, rhs)

    if obj_class == "Index":
        result = cudf.Series(result)

    np.testing.assert_almost_equal(result.to_numpy(), arithmetic_op(arr, rhs))


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize("binop", [operator.and_, operator.or_, operator.xor])
def test_series_bitwise_binop(
    binop, obj_class, integer_types_as_str, integer_types_as_str2
):
    rng = np.random.default_rng(seed=0)
    arr1 = (rng.random(100) * 100).astype(integer_types_as_str)
    sr1 = cudf.Series(arr1)

    arr2 = (rng.random(100) * 100).astype(integer_types_as_str2)
    sr2 = cudf.Series(arr2)

    if obj_class == "Index":
        sr1 = cudf.Index(sr1)
        sr2 = cudf.Index(sr2)

    result = binop(sr1, sr2)

    if obj_class == "Index":
        result = cudf.Series(result)

    np.testing.assert_almost_equal(result.to_numpy(), binop(arr1, arr2))


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
def test_series_compare(
    comparison_op, obj_class, numeric_and_temporal_types_as_str
):
    rng = np.random.default_rng(seed=0)
    arr1 = rng.integers(0, 100, 100).astype(numeric_and_temporal_types_as_str)
    arr2 = rng.integers(0, 100, 100).astype(numeric_and_temporal_types_as_str)
    sr1 = cudf.Series(arr1)
    sr2 = cudf.Series(arr2)

    if obj_class == "Index":
        sr1 = cudf.Index(sr1)
        sr2 = cudf.Index(sr2)

    result1 = comparison_op(sr1, sr1)
    result2 = comparison_op(sr2, sr2)
    result3 = comparison_op(sr1, sr2)

    if obj_class == "Index":
        result1 = cudf.Series(result1)
        result2 = cudf.Series(result2)
        result3 = cudf.Series(result3)

    np.testing.assert_equal(result1.to_numpy(), comparison_op(arr1, arr1))
    np.testing.assert_equal(result2.to_numpy(), comparison_op(arr2, arr2))
    np.testing.assert_equal(result3.to_numpy(), comparison_op(arr1, arr2))


@pytest.mark.parametrize(
    "dtype,val",
    [("int8", 200), ("int32", 2**32), ("uint8", -128), ("uint64", -1)],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_series_compare_integer(dtype, val, comparison_op, reverse):
    # Tests that these actually work, even though they are out of bound.
    force_cast_val = np.array(val).astype(dtype)
    sr = cudf.Series(
        [np.iinfo(dtype).min, np.iinfo(dtype).max, force_cast_val, None],
        dtype=dtype,
    )
    # We expect the same result as comparing to a value within range (e.g. 0)
    # except that a NULL value evaluates to False
    exp = False
    if reverse:
        if comparison_op(val, 0):
            exp = True
        res = comparison_op(val, sr)
    else:
        if comparison_op(0, val):
            exp = True
        res = comparison_op(sr, val)

    expected = cudf.Series([exp, exp, exp, None])
    assert_eq(res, expected)


@pytest.mark.parametrize(
    "ltype, rtype",
    [
        ("datetime64[ms]", "datetime64[ms]"),
        ("datetime64[ns]", "datetime64[ms]"),
        ("timedelta64[s]", "timedelta64[s]"),
        ("timedelta64[s]", "timedelta64[ms]"),
        ("float32", "float64"),
        ("int32", "float64"),
        ("str", "str"),
    ],
)
def test_series_compare_nulls(comparison_op, ltype, rtype):
    ldata = [1, 2, None, None, 5]
    rdata = [2, 1, None, 4, None]

    lser = cudf.Series(ldata, dtype=ltype)
    rser = cudf.Series(rdata, dtype=rtype)

    lmask = ~lser.isnull()
    rmask = ~rser.isnull()

    got = comparison_op(lser, rser)
    if ltype in {"datetime64[ms]", "datetime64[ns]", "timedelta64[s]"}:
        expect = comparison_op(lser.to_pandas(), rser.to_pandas())
    else:
        expect_mask = np.logical_and(lmask, rmask)
        expect = cudf.Series([None] * 5, dtype="bool")
        expect[expect_mask] = comparison_op(
            lser[expect_mask], rser[expect_mask]
        )
    assert_eq(expect, got)


def test_str_series_compare_str(comparison_op):
    str_series_cmp_data = pd.Series(
        ["a", "b", None, "d", "e", None], dtype="string"
    )
    with cudf.option_context("mode.pandas_compatible", True):
        expect = comparison_op(str_series_cmp_data, "a")
        got = comparison_op(cudf.Series(str_series_cmp_data), "a")
        assert_eq(expect, got)


def test_str_series_compare_str_reflected(comparison_op):
    str_series_cmp_data = pd.Series(
        ["a", "b", None, "d", "e", None], dtype="string"
    )
    with cudf.option_context("mode.pandas_compatible", True):
        expect = comparison_op("a", str_series_cmp_data)
        got = comparison_op("a", cudf.Series(str_series_cmp_data))

        assert_eq(expect, got)


@pytest.mark.parametrize("cmp_scalar", [1, 1.5, True])
def test_str_series_compare_num(comparison_op, cmp_scalar):
    if comparison_op not in {operator.eq, operator.ne}:
        pytest.skip("Only eq and ne are relevant for this test")
    str_series_cmp_data = pd.Series(
        ["a", "b", None, "d", "e", None], dtype="string"
    )
    with cudf.option_context("mode.pandas_compatible", True):
        expect = comparison_op(str_series_cmp_data, cmp_scalar)
        got = comparison_op(cudf.Series(str_series_cmp_data), cmp_scalar)

        assert_eq(expect, got)


@pytest.mark.parametrize("cmp_scalar", [1, 1.5, True])
def test_str_series_compare_num_reflected(comparison_op, cmp_scalar):
    if comparison_op not in {operator.eq, operator.ne}:
        pytest.skip("Only eq and ne are relevant for this test")
    str_series_cmp_data = pd.Series(
        ["a", "b", None, "d", "e", None], dtype="string"
    )
    with cudf.option_context("mode.pandas_compatible", True):
        expect = comparison_op(cmp_scalar, str_series_cmp_data)
        got = comparison_op(cmp_scalar, cudf.Series(str_series_cmp_data))

        assert_eq(expect, got)


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
def test_series_compare_scalar(
    request, comparison_op, obj_class, numeric_and_temporal_types_as_str
):
    request.applymarker(
        pytest.mark.xfail(
            numeric_and_temporal_types_as_str == "timedelta64[ns]"
            or (
                numeric_and_temporal_types_as_str == "datetime64[ns]"
                and comparison_op not in {operator.eq, operator.ne}
            ),
            reason=f"Fails with {numeric_and_temporal_types_as_str} with {comparison_op.__name__}",
        )
    )

    rng = np.random.default_rng(seed=0)
    arr1 = rng.integers(0, 100, 100).astype(numeric_and_temporal_types_as_str)
    sr1 = cudf.Series(arr1)
    rhs = rng.choice(arr1).item()

    if obj_class == "Index":
        sr1 = cudf.Index(sr1)

    result1 = comparison_op(sr1, rhs)
    result2 = comparison_op(rhs, sr1)

    if obj_class == "Index":
        result1 = cudf.Series(result1)
        result2 = cudf.Series(result2)

    np.testing.assert_equal(result1.to_numpy(), comparison_op(arr1, rhs))
    np.testing.assert_equal(result2.to_numpy(), comparison_op(rhs, arr1))


@pytest.mark.parametrize("lhs_nulls", ["none", "some"])
@pytest.mark.parametrize("rhs_nulls", ["none", "some"])
def test_validity_add(lhs_nulls, rhs_nulls):
    nelem = 10
    rng = np.random.default_rng(seed=0)
    # LHS
    lhs_data = rng.random(nelem)
    if lhs_nulls == "some":
        lhs_mask = rng.choice([True, False], size=nelem)
        lhs = cudf.Series(lhs_data)
        lhs.loc[lhs_mask] = None
    else:
        lhs = cudf.Series(lhs_data)
    # RHS
    rhs_data = rng.random(nelem)
    if rhs_nulls == "some":
        rhs_mask = rng.choice([True, False], size=nelem)
        rhs = cudf.Series(rhs_data)
        rhs.loc[rhs_mask] = None
    else:
        rhs = cudf.Series(rhs_data)
    # Result
    res = lhs + rhs
    if lhs_nulls == "some" and rhs_nulls == "some":
        res_mask = lhs_mask | rhs_mask
    if lhs_nulls == "some" and rhs_nulls == "none":
        res_mask = lhs_mask
    if lhs_nulls == "none" and rhs_nulls == "some":
        res_mask = rhs_mask
    # Fill NA values
    na_value = -10000
    got = res.fillna(na_value).to_numpy()
    expect = lhs_data + rhs_data
    if lhs_nulls == "some" or rhs_nulls == "some":
        expect[res_mask] = na_value

    np.testing.assert_array_equal(expect, got)


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
@pytest.mark.parametrize("binop", [operator.add, operator.mul])
def test_series_binop_mixed_dtype(
    binop, numeric_types_as_str, numeric_types_as_str2, obj_class
):
    nelem = 10
    rng = np.random.default_rng(seed=0)
    lhs = (rng.random(nelem) * nelem).astype(numeric_types_as_str)
    rhs = (rng.random(nelem) * nelem).astype(numeric_types_as_str2)

    sr1 = cudf.Series(lhs)
    sr2 = cudf.Series(rhs)

    if obj_class == "Index":
        sr1 = cudf.Index(sr1)
        sr2 = cudf.Index(sr2)

    result = binop(cudf.Series(sr1), cudf.Series(sr2))

    if obj_class == "Index":
        result = cudf.Series(result)

    np.testing.assert_almost_equal(result.to_numpy(), binop(lhs, rhs))


@pytest.mark.parametrize("obj_class", ["Series", "Index"])
def test_series_cmpop_mixed_dtype(
    comparison_op, numeric_types_as_str, numeric_types_as_str2, obj_class
):
    nelem = 5
    rng = np.random.default_rng(seed=0)
    lhs = (rng.random(nelem) * nelem).astype(numeric_types_as_str)
    rhs = (rng.random(nelem) * nelem).astype(numeric_types_as_str2)

    sr1 = cudf.Series(lhs)
    sr2 = cudf.Series(rhs)

    if obj_class == "Index":
        sr1 = cudf.Index(sr1)
        sr2 = cudf.Index(sr2)

    result = comparison_op(cudf.Series(sr1), cudf.Series(sr2))

    if obj_class == "Index":
        result = cudf.Series(result)

    np.testing.assert_array_equal(result.to_numpy(), comparison_op(lhs, rhs))


@pytest.mark.filterwarnings(
    "ignore:invalid value encountered in power:RuntimeWarning"
)
@pytest.mark.filterwarnings(
    "ignore:divide by zero encountered in power:RuntimeWarning"
)
@pytest.mark.parametrize("obj_class", [cudf.Series, cudf.Index])
@pytest.mark.parametrize("scalar", [-1, 0, 1])
def test_series_reflected_ops_scalar(
    arithmetic_op, scalar, numeric_types_as_str, obj_class
):
    # create random series
    func = lambda x: arithmetic_op(scalar, x)  # noqa: E731
    rng = np.random.default_rng(0)
    random_series = pd.Series(
        rng.integers(10, 100, 100).astype(numeric_types_as_str)
    )

    gs = obj_class(random_series)

    try:
        gs_result = func(gs)
    except OverflowError:
        # An error is fine, if pandas raises the same error:
        with pytest.raises(OverflowError):
            func(random_series)

        return

    # class typing
    if obj_class == "Index":
        gs = cudf.Series(gs)

    # pandas
    ps_result = func(random_series)

    # verify
    np.testing.assert_allclose(ps_result, gs_result.to_numpy())


def test_boolean_scalar_binop(comparison_op):
    rng = np.random.default_rng(seed=0)
    psr = pd.Series(rng.choice([True, False], 10))
    gsr = cudf.from_pandas(psr)
    assert_eq(comparison_op(psr, True), comparison_op(gsr, True))
    assert_eq(comparison_op(psr, False), comparison_op(gsr, False))


@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("fill_value", [None, 27])
def test_operator_func_between_series(
    float_types_as_str, arithmetic_op_method, has_nulls, fill_value
):
    count = 1000
    gdf_series_a = gen_rand_series(
        float_types_as_str, count, has_nulls=has_nulls, stride=10000
    )
    gdf_series_b = gen_rand_series(
        float_types_as_str, count, has_nulls=has_nulls, stride=100
    )
    pdf_series_a = gdf_series_a.to_pandas()
    pdf_series_b = gdf_series_b.to_pandas()

    gdf_result = getattr(gdf_series_a, arithmetic_op_method)(
        gdf_series_b, fill_value=fill_value
    )
    pdf_result = getattr(pdf_series_a, arithmetic_op_method)(
        pdf_series_b, fill_value=fill_value
    )

    assert_eq(pdf_result, gdf_result)


@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("fill_value", [None, 27])
def test_operator_func_series_and_scalar(
    float_types_as_str, arithmetic_op_method, has_nulls, fill_value
):
    count = 1000
    scalar = 59
    gdf_series = gen_rand_series(
        float_types_as_str, count, has_nulls=has_nulls, stride=10000
    )
    pdf_series = gdf_series.to_pandas()

    gdf_series_result = getattr(gdf_series, arithmetic_op_method)(
        scalar,
        fill_value=fill_value,
    )
    pdf_series_result = getattr(pdf_series, arithmetic_op_method)(
        scalar,
        fill_value=fill_value,
    )

    assert_eq(pdf_series_result, gdf_series_result)


@pytest.mark.parametrize("fill_value", [0, 1, None, np.nan])
@pytest.mark.parametrize("scalar_a", [0, 1, None, np.nan])
@pytest.mark.parametrize("scalar_b", [0, 1, None, np.nan])
def test_operator_func_between_series_logical(
    float_types_as_str, comparison_op_method, scalar_a, scalar_b, fill_value
):
    with cudf.option_context("mode.pandas_compatible", True):
        gdf_series_a = cudf.Series([scalar_a], nan_as_null=False).astype(
            float_types_as_str.capitalize()
        )
        gdf_series_b = cudf.Series([scalar_b], nan_as_null=False).astype(
            float_types_as_str.capitalize()
        )

        pdf_series_a = gdf_series_a.to_pandas()
        pdf_series_b = gdf_series_b.to_pandas()

        expect = getattr(pdf_series_a, comparison_op_method)(
            pdf_series_b, fill_value=fill_value
        )
        got = getattr(gdf_series_a, comparison_op_method)(
            gdf_series_b, fill_value=fill_value
        ).to_pandas()
    assert_eq(expect, got)


@pytest.mark.parametrize("has_nulls", [True, False])
@pytest.mark.parametrize("scalar", [-59.0, np.nan, 0, 59.0])
@pytest.mark.parametrize("fill_value", [None, 1.0])
def test_operator_func_series_and_scalar_logical(
    request,
    float_types_as_str,
    comparison_op_method,
    has_nulls,
    scalar,
    fill_value,
):
    request.applymarker(
        pytest.mark.xfail(
            fill_value == 1.0
            and scalar is np.nan
            and (
                has_nulls
                or (not has_nulls and comparison_op_method not in {"eq", "ne"})
            ),
            reason="https://github.com/pandas-dev/pandas/issues/57447",
        )
    )
    if has_nulls:
        gdf_series = cudf.Series(
            [-1.0, 0, cudf.NA, 1.1], dtype=float_types_as_str
        )
    else:
        gdf_series = cudf.Series(
            [-1.0, 0, 10.5, 1.1], dtype=float_types_as_str
        )
    pdf_series = gdf_series.to_pandas(nullable=True)
    gdf_series_result = getattr(gdf_series, comparison_op_method)(
        scalar,
        fill_value=fill_value,
    )
    pdf_series_result = getattr(pdf_series, comparison_op_method)(
        scalar, fill_value=fill_value
    )

    expect = pdf_series_result
    got = gdf_series_result.to_pandas(nullable=True)

    assert_eq(expect, got)


@pytest.mark.parametrize("rhs", [0, 10])
def test_binop_bool_uint(request, binary_op_method, rhs):
    if binary_op_method in {"rmod", "rfloordiv"}:
        request.applymarker(
            pytest.mark.xfail(
                reason="https://github.com/rapidsai/cudf/issues/12162"
            ),
        )
    psr = pd.Series([True, False, False])
    gsr = cudf.from_pandas(psr)
    assert_eq(
        getattr(psr, binary_op_method)(rhs),
        getattr(gsr, binary_op_method)(rhs),
        check_dtype=False,
    )


@pytest.mark.parametrize("scalar_divisor", [False, True])
def test_floordiv_zero_float64(
    integer_types_as_str, integer_types_as_str2, scalar_divisor
):
    sr = pd.Series([1, 2, 3], dtype=integer_types_as_str)
    cr = cudf.from_pandas(sr)

    if scalar_divisor:
        pd_div = getattr(np, integer_types_as_str2)(0)
        cudf_div = pd_div
    else:
        pd_div = pd.Series([0], dtype=integer_types_as_str2)
        cudf_div = cudf.from_pandas(pd_div)
    assert_eq(sr // pd_div, cr // cudf_div)


@pytest.mark.parametrize("scalar_divisor", [False, True])
@pytest.mark.xfail(reason="https://github.com/rapidsai/cudf/issues/12162")
def test_floordiv_zero_bool(scalar_divisor):
    sr = pd.Series([True, True, False], dtype=np.bool_)
    cr = cudf.from_pandas(sr)

    if scalar_divisor:
        pd_div = np.bool_(0)
        cudf_div = pd_div
    else:
        pd_div = pd.Series([0], dtype=np.bool_)
        cudf_div = cudf.from_pandas(pd_div)

    with pytest.raises((NotImplementedError, ZeroDivisionError)):
        sr // pd_div
    with pytest.raises((NotImplementedError, ZeroDivisionError)):
        cr // cudf_div


def test_rmod_zero_nan(numeric_and_bool_types_as_str, request):
    request.applymarker(
        pytest.mark.xfail(
            numeric_and_bool_types_as_str == "bool",
            reason="pandas returns int8, cuDF returns int64",
        )
    )
    sr = pd.Series([1, 1, 0], dtype=numeric_and_bool_types_as_str)
    cr = cudf.from_pandas(sr)
    assert_eq(1 % sr, 1 % cr)
    expected_dtype = (
        np.float64 if cr.dtype.kind != "f" else numeric_and_bool_types_as_str
    )
    assert_eq(1 % cr, cudf.Series([0, 0, None], dtype=expected_dtype))


def test_series_misc_binop():
    pds = pd.Series([1, 2, 4], name="abc xyz")
    gds = cudf.Series([1, 2, 4], name="abc xyz")

    assert_eq(pds + 1, gds + 1)
    assert_eq(1 + pds, 1 + gds)

    assert_eq(pds + pds, gds + gds)

    pds1 = pd.Series([1, 2, 4], name="hello world")
    gds1 = cudf.Series([1, 2, 4], name="hello world")

    assert_eq(pds + pds1, gds + gds1)
    assert_eq(pds1 + pds, gds1 + gds)

    assert_eq(pds1 + pds + 5, gds1 + gds + 5)


def test_int8_float16_binop():
    a = cudf.Series([1], dtype="int8")
    b = np.float16(2)
    expect = cudf.Series([0.5])
    got = a / b
    assert_eq(expect, got, check_dtype=False)


@pytest.mark.parametrize("dtype", ["int64", "float64", "str"])
def test_vector_to_none_binops(dtype):
    data = cudf.Series([1, 2, 3, None], dtype=dtype)

    expect = cudf.Series([None] * 4).astype(dtype)
    got = data + None

    assert_eq(expect, got)


def is_timezone_aware_dtype(dtype: str) -> bool:
    return bool(re.match(r"^datetime64\[ns, .+\]$", dtype))


@pytest.mark.parametrize("n_periods", [0, 1, -12])
@pytest.mark.parametrize(
    "frequency",
    [
        "months",
        "years",
        "days",
        "hours",
        "minutes",
        "seconds",
        "microseconds",
        "nanoseconds",
    ],
)
@pytest.mark.parametrize(
    "dtype, components",
    [
        ["datetime64[ns]", "00.012345678"],
        ["datetime64[us]", "00.012345"],
        ["datetime64[ms]", "00.012"],
        ["datetime64[s]", "00"],
        ["datetime64[ns, Asia/Kathmandu]", "00.012345678"],
    ],
)
@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_datetime_dateoffset_binaryop(
    request, n_periods, frequency, dtype, components, op
):
    request.applymarker(
        pytest.mark.xfail(
            dtype in {"datetime64[ms]", "datetime64[s]"}
            and frequency == "microseconds"
            and n_periods == 0,
            reason="https://github.com/pandas-dev/pandas/issues/57448",
        )
    )

    date_col = [
        f"2000-01-01 00:00:{components}",
        f"2000-01-31 00:00:{components}",
        f"2000-02-29 00:00:{components}",
    ]
    if is_timezone_aware_dtype(dtype):
        # Construct naive datetime64[ns] Series
        gsr = cudf.Series(date_col, dtype="datetime64[ns]")
        psr = gsr.to_pandas()

        # Convert to timezone-aware (both cudf and pandas)
        gsr = gsr.dt.tz_localize("UTC").dt.tz_convert("Asia/Kathmandu")
        psr = psr.dt.tz_localize("UTC").dt.tz_convert("Asia/Kathmandu")
    else:
        gsr = cudf.Series(date_col, dtype=dtype)
        psr = gsr.to_pandas()

    kwargs = {frequency: n_periods}

    goffset = cudf.DateOffset(**kwargs)
    poffset = pd.DateOffset(**kwargs)

    expect = op(psr, poffset)
    got = op(gsr, goffset)

    if is_timezone_aware_dtype(dtype):
        assert isinstance(expect.dtype, pd.DatetimeTZDtype)
        assert str(expect.dtype.tz) == str(got.dtype.tz)
        expect = expect.dt.tz_convert("UTC")
        got = got.dt.tz_convert("UTC")

    assert_eq(expect, got)

    expect = op(psr, -poffset)
    got = op(gsr, -goffset)

    if is_timezone_aware_dtype(dtype):
        assert isinstance(expect.dtype, pd.DatetimeTZDtype)
        assert str(expect.dtype.tz) == str(got.dtype.tz)
        expect = expect.dt.tz_convert("UTC")
        got = got.dt.tz_convert("UTC")

    assert_eq(expect, got)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"months": 2, "years": 5},
        {"microseconds": 1, "seconds": 1},
        {"months": 2, "years": 5, "seconds": 923, "microseconds": 481},
        {"milliseconds": 4},
        {"milliseconds": 4, "years": 2},
        {"nanoseconds": 12},
    ],
)
@pytest.mark.filterwarnings(
    "ignore:Non-vectorized DateOffset:pandas.errors.PerformanceWarning"
)
@pytest.mark.filterwarnings(
    "ignore:Discarding nonzero nanoseconds:UserWarning"
)
@pytest.mark.parametrize("op", [operator.add, operator.sub])
def test_datetime_dateoffset_binaryop_multiple(kwargs, op):
    gsr = cudf.Series(
        [
            "2000-01-01 00:00:00.012345678",
            "2000-01-31 00:00:00.012345678",
            "2000-02-29 00:00:00.012345678",
        ],
        dtype="datetime64[ns]",
    )
    psr = gsr.to_pandas()

    poffset = pd.DateOffset(**kwargs)
    goffset = cudf.DateOffset(**kwargs)

    expect = op(psr, poffset)
    got = op(gsr, goffset)

    assert_eq(expect, got)


@pytest.mark.parametrize("n_periods", [0, 1, -12])
@pytest.mark.parametrize(
    "frequency",
    [
        "months",
        "years",
        "days",
        "hours",
        "minutes",
        "seconds",
        "microseconds",
        "nanoseconds",
    ],
)
@pytest.mark.parametrize(
    "dtype, components",
    [
        ["datetime64[ns]", "00.012345678"],
        ["datetime64[us]", "00.012345"],
        ["datetime64[ms]", "00.012"],
        ["datetime64[s]", "00"],
    ],
)
def test_datetime_dateoffset_binaryop_reflected(
    n_periods, frequency, dtype, components
):
    date_col = [
        f"2000-01-01 00:00:{components}",
        f"2000-01-31 00:00:{components}",
        f"2000-02-29 00:00:{components}",
    ]
    gsr = cudf.Series(date_col, dtype=dtype)
    psr = gsr.to_pandas()  # converts to nanos

    kwargs = {frequency: n_periods}

    goffset = cudf.DateOffset(**kwargs)
    poffset = pd.DateOffset(**kwargs)

    expect = poffset + psr
    got = goffset + gsr

    # TODO: Remove check_dtype once we get some clarity on:
    # https://github.com/pandas-dev/pandas/issues/57448
    assert_eq(expect, got, check_dtype=False)

    with pytest.raises(TypeError):
        poffset - psr

    with pytest.raises(TypeError):
        goffset - gsr


@pytest.mark.parametrize("frame", [cudf.Series, cudf.Index, cudf.DataFrame])
@pytest.mark.parametrize(
    "dtype", ["int", "str", "datetime64[s]", "timedelta64[s]", "category"]
)
def test_binops_with_lhs_numpy_scalar(frame, dtype):
    data = [1, 2, 3, 4, 5]

    data = (
        frame({"a": data}, dtype=dtype)
        if isinstance(frame, cudf.DataFrame)
        else frame(data, dtype=dtype)
    )

    if dtype == "datetime64[s]":
        val = cudf.dtype(dtype).type(4, "s")
    elif dtype == "timedelta64[s]":
        val = cudf.dtype(dtype).type(4, "s")
    elif dtype == "category":
        val = np.int64(4)
    elif dtype == "str":
        val = str(4)
    else:
        val = cudf.dtype(dtype).type(4)

    # Compare equality with series on left side to dispatch to the pandas/cudf
    # __eq__ operator and avoid a DeprecationWarning from numpy.
    expected = data.to_pandas() == val
    got = data == val
    assert_eq(expected, got)


def test_binops_with_NA_consistent(
    numeric_and_temporal_types_as_str, comparison_op_method
):
    data = [1, 2, 3]
    sr = cudf.Series(data, dtype=numeric_and_temporal_types_as_str)

    result = getattr(sr, comparison_op_method)(cudf.NA)
    if comparison_op_method == "ne":
        expect_all = True
    else:
        expect_all = False
    assert (result == expect_all).all()


@pytest.mark.parametrize(
    "op, lhs, l_dtype, rhs, r_dtype, expect, expect_dtype",
    [
        (
            operator.add,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["3.0", "4.0"],
            cudf.Decimal64Dtype(scale=2, precision=4),
        ),
        (
            operator.add,
            2,
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["3.5", "4.0"],
            cudf.Decimal64Dtype(scale=2, precision=4),
        ),
        (
            operator.add,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["3.75", "3.005"],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=17),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["100.1", "200.2"],
            cudf.Decimal128Dtype(scale=3, precision=23),
        ),
        (
            operator.sub,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=1, precision=2),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", "0.995"],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=1, precision=2),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", "0.995"],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=10),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=6, precision=10),
            ["99.9", "199.8"],
            cudf.Decimal128Dtype(scale=6, precision=19),
        ),
        (
            operator.sub,
            2,
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.25", "0.995"],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.mul,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["1.5", "3.0"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["2.25", "6.0"],
            cudf.Decimal64Dtype(scale=5, precision=8),
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["10.0", "40.0"],
            cudf.Decimal64Dtype(scale=1, precision=8),
        ),
        (
            operator.mul,
            ["1000", "2000"],
            cudf.Decimal64Dtype(scale=-3, precision=4),
            ["0.343", "0.500"],
            cudf.Decimal64Dtype(scale=3, precision=3),
            ["343.0", "1000.0"],
            cudf.Decimal64Dtype(scale=0, precision=8),
        ),
        (
            operator.mul,
            200,
            cudf.Decimal64Dtype(scale=3, precision=6),
            ["0.343", "0.500"],
            cudf.Decimal64Dtype(scale=3, precision=6),
            ["68.60", "100.0"],
            cudf.Decimal64Dtype(scale=3, precision=10),
        ),
        (
            operator.truediv,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=4),
            ["1.5", "3.0"],
            cudf.Decimal64Dtype(scale=1, precision=4),
            ["1.0", "0.6"],
            cudf.Decimal64Dtype(scale=7, precision=10),
        ),
        (
            operator.truediv,
            ["110", "200"],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=2, precision=4),
            ["1000.0", "1000.0"],
            cudf.Decimal64Dtype(scale=6, precision=12),
        ),
        (
            operator.truediv,
            ["132.86", "15.25"],
            cudf.Decimal64Dtype(scale=4, precision=14),
            ["2.34", "8.50"],
            cudf.Decimal64Dtype(scale=2, precision=8),
            ["56.77", "1.79"],
            cudf.Decimal128Dtype(scale=13, precision=25),
        ),
        (
            operator.truediv,
            ["20", "20"],
            cudf.Decimal128Dtype(scale=2, precision=6),
            ["20", "20"],
            cudf.Decimal128Dtype(scale=2, precision=6),
            ["1.0", "1.0"],
            cudf.Decimal128Dtype(scale=9, precision=15),
        ),
        (
            operator.add,
            ["1.5", None, "2.0"],
            cudf.Decimal64Dtype(scale=1, precision=2),
            ["1.5", None, "2.0"],
            cudf.Decimal64Dtype(scale=1, precision=2),
            ["3.0", None, "4.0"],
            cudf.Decimal64Dtype(scale=1, precision=3),
        ),
        (
            operator.add,
            ["1.5", None],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["2.25", "1.005"],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["3.75", None],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["1.5", None],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["2.25", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", None],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.sub,
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["2.25", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["-0.75", None],
            cudf.Decimal64Dtype(scale=3, precision=5),
        ),
        (
            operator.mul,
            ["1.5", None],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["1.5", None],
            cudf.Decimal64Dtype(scale=3, precision=4),
            ["2.25", None],
            cudf.Decimal64Dtype(scale=5, precision=8),
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=10),
            ["0.1", None],
            cudf.Decimal64Dtype(scale=3, precision=12),
            ["10.0", None],
            cudf.Decimal128Dtype(scale=1, precision=23),
        ),
        (
            operator.eq,
            ["0.18", "0.42"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.18", "0.21"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [True, False],
            bool,
        ),
        (
            operator.eq,
            ["0.18", "0.42"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1800", "0.2100"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [True, False],
            bool,
        ),
        (
            operator.eq,
            ["100", None],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [True, None],
            bool,
        ),
        (
            operator.ne,
            ["0.06", "0.42"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.18", "0.42"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [True, False],
            bool,
        ),
        (
            operator.ne,
            ["1.33", "1.21"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1899", "1.21"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [True, False],
            bool,
        ),
        (
            operator.ne,
            ["300", None],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["110", "5500"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [True, None],
            bool,
        ),
        (
            operator.lt,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.10", "0.87", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [False, True, False],
            bool,
        ),
        (
            operator.lt,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1000", "0.8700", "1.0000"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [False, True, False],
            bool,
        ),
        (
            operator.lt,
            ["200", None, "100"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200", "100"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [False, None, False],
            bool,
        ),
        (
            operator.gt,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.10", "0.87", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [True, False, False],
            bool,
        ),
        (
            operator.gt,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1000", "0.8700", "1.0000"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [True, False, False],
            bool,
        ),
        (
            operator.gt,
            ["300", None, "100"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200", "100"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [True, None, False],
            bool,
        ),
        (
            operator.le,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.10", "0.87", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [False, True, True],
            bool,
        ),
        (
            operator.le,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1000", "0.8700", "1.0000"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [False, True, True],
            bool,
        ),
        (
            operator.le,
            ["300", None, "100"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200", "100"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [False, None, True],
            bool,
        ),
        (
            operator.ge,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.10", "0.87", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            [True, False, True],
            bool,
        ),
        (
            operator.ge,
            ["0.18", "0.42", "1.00"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["0.1000", "0.8700", "1.0000"],
            cudf.Decimal64Dtype(scale=4, precision=5),
            [True, False, True],
            bool,
        ),
        (
            operator.ge,
            ["300", None, "100"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            ["100", "200", "100"],
            cudf.Decimal64Dtype(scale=-1, precision=4),
            [True, None, True],
            bool,
        ),
    ],
)
def test_binops_decimal(op, lhs, l_dtype, rhs, r_dtype, expect, expect_dtype):
    if isinstance(lhs, (int, float)):
        a = lhs
    else:
        a = _decimal_series(lhs, l_dtype)
    b = _decimal_series(rhs, r_dtype)
    expect = (
        _decimal_series(expect, expect_dtype)
        if isinstance(
            expect_dtype,
            (cudf.Decimal64Dtype, cudf.Decimal32Dtype, cudf.Decimal128Dtype),
        )
        else cudf.Series(expect, dtype=expect_dtype)
    )

    got = op(a, b)
    assert expect.dtype == got.dtype
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "op,lhs,l_dtype,rhs,r_dtype,expect,expect_dtype",
    [
        (
            "radd",
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            ["3.0", "4.0"],
            cudf.Decimal64Dtype(scale=2, precision=4),
        ),
        (
            "rsub",
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=10),
            ["0.1", "0.2"],
            cudf.Decimal64Dtype(scale=6, precision=10),
            ["-99.9", "-199.8"],
            cudf.Decimal128Dtype(scale=6, precision=19),
        ),
        (
            "rmul",
            ["1000", "2000"],
            cudf.Decimal64Dtype(scale=-3, precision=4),
            ["0.343", "0.500"],
            cudf.Decimal64Dtype(scale=3, precision=3),
            ["343.0", "1000.0"],
            cudf.Decimal64Dtype(scale=0, precision=8),
        ),
        (
            "rtruediv",
            ["1.5", "0.5"],
            cudf.Decimal64Dtype(scale=3, precision=6),
            ["1.5", "2.0"],
            cudf.Decimal64Dtype(scale=3, precision=6),
            ["1.0", "4.0"],
            cudf.Decimal64Dtype(scale=10, precision=16),
        ),
    ],
)
def test_binops_reflect_decimal(
    op, lhs, l_dtype, rhs, r_dtype, expect, expect_dtype
):
    a = _decimal_series(lhs, l_dtype)
    b = _decimal_series(rhs, r_dtype)
    expect = _decimal_series(expect, expect_dtype)

    got = getattr(a, op)(b)
    assert expect.dtype == got.dtype
    assert_eq(expect, got)


@pytest.mark.parametrize("powers", [0, 1, 2])
def test_binops_decimal_pow(powers):
    s = cudf.Series(
        [
            decimal.Decimal("1.324324"),
            None,
            decimal.Decimal("2"),
            decimal.Decimal("3"),
            decimal.Decimal("5"),
        ]
    )
    ps = s.to_pandas(arrow_type=True)
    got = s**powers
    expect = ps**powers
    expect = expect.astype(pd.ArrowDtype(got.dtype.to_arrow()))
    assert_eq(got, expect, check_dtype=False)


def test_binops_raise_error():
    s = cudf.Series([decimal.Decimal("1.324324")])

    with pytest.raises(TypeError):
        s // 1


@pytest.mark.parametrize(
    "op, ldata, ldtype, rdata, expected1, expected2",
    [
        (
            operator.eq,
            ["100", "41", None],
            cudf.Decimal64Dtype(scale=0, precision=5),
            [100, 42, 12],
            [True, False, None],
            [True, False, None],
        ),
        (
            operator.eq,
            ["100.000", "42.001", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 12],
            [True, False, None],
            [True, False, None],
        ),
        (
            operator.eq,
            ["100", "40", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 12],
            [True, False, None],
            [True, False, None],
        ),
        (
            operator.ne,
            ["100", "42", "24", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 40, 24, 12],
            [False, True, False, None],
            [False, True, False, None],
        ),
        (
            operator.ne,
            ["10.1", "88", "11", None],
            cudf.Decimal64Dtype(scale=1, precision=3),
            [10, 42, 11, 12],
            [True, True, False, None],
            [True, True, False, None],
        ),
        (
            operator.ne,
            ["100.000", "42", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            [False, False, True, None],
            [False, False, True, None],
        ),
        (
            operator.lt,
            ["100", "40", "28", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 42, 24, 12],
            [False, True, False, None],
            [False, False, True, None],
        ),
        (
            operator.lt,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            [False, False, True, None],
            [False, True, False, None],
        ),
        (
            operator.lt,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            [False, True, False, None],
            [False, False, True, None],
        ),
        (
            operator.gt,
            ["100", "42", "20", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 40, 24, 12],
            [False, True, False, None],
            [False, False, True, None],
        ),
        (
            operator.gt,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            [False, True, False, None],
            [False, False, True, None],
        ),
        (
            operator.gt,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            [False, False, True, None],
            [False, True, False, None],
        ),
        (
            operator.le,
            ["100", "40", "28", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 42, 24, 12],
            [True, True, False, None],
            [True, False, True, None],
        ),
        (
            operator.le,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            [True, False, True, None],
            [True, True, False, None],
        ),
        (
            operator.le,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            [True, True, False, None],
            [True, False, True, None],
        ),
        (
            operator.ge,
            ["100", "42", "20", None],
            cudf.Decimal64Dtype(scale=0, precision=3),
            [100, 40, 24, 12],
            [True, True, False, None],
            [True, False, True, None],
        ),
        (
            operator.ge,
            ["100.000", "42.002", "23.999", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            [100, 42, 24, 12],
            [True, True, False, None],
            [True, False, True, None],
        ),
        (
            operator.ge,
            ["100", "40", "10", None],
            cudf.Decimal64Dtype(scale=-1, precision=3),
            [100, 42, 8, 12],
            [True, False, True, None],
            [True, True, False, None],
        ),
    ],
)
@pytest.mark.parametrize("reflected", [True, False])
def test_binops_decimal_comp_mixed_integer(
    op,
    ldata,
    ldtype,
    rdata,
    expected1,
    expected2,
    integer_types_as_str,
    reflected,
):
    """
    Tested compare operations:
        eq, lt, gt, le, ge
    Each operation has 3 decimal data setups, with scale from {==0, >0, <0}.
    Decimal precisions are sufficient to hold the digits.
    For each decimal data setup, there is at least one row that lead to one
    of the following compare results: {True, False, None}.
    """
    if not reflected:
        expected = cudf.Series(expected1, dtype=bool)
    else:
        expected = cudf.Series(expected2, dtype=bool)

    lhs = _decimal_series(ldata, ldtype)
    rhs = cudf.Series(rdata, dtype=integer_types_as_str)

    if reflected:
        rhs, lhs = lhs, rhs

    actual = op(lhs, rhs)

    assert_eq(expected, actual)


@pytest.mark.parametrize(
    "op, lhs, l_dtype, rhs, expect, expect_dtype, reflect",
    [
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(1),
            ["101", "201"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            1,
            ["101", "201"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("1.5"),
            ["101.5", "201.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            False,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(1),
            ["101", "201"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            True,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            1,
            ["101", "201"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            True,
        ),
        (
            operator.add,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("1.5"),
            ["101.5", "201.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            True,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            1,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            False,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(2),
            ["200", "400"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            False,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("1.5"),
            ["150", "300"],
            cudf.Decimal64Dtype(scale=-1, precision=6),
            False,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            1,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            True,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(2),
            ["200", "400"],
            cudf.Decimal64Dtype(scale=-2, precision=5),
            True,
        ),
        (
            operator.mul,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("1.5"),
            ["150", "300"],
            cudf.Decimal64Dtype(scale=-1, precision=6),
            True,
        ),
        (
            operator.truediv,
            ["1000", "2000"],
            cudf.Decimal64Dtype(scale=-2, precision=4),
            1,
            ["1000", "2000"],
            cudf.Decimal64Dtype(scale=6, precision=12),
            False,
        ),
        (
            operator.truediv,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=2, precision=5),
            decimal.Decimal(2),
            ["50", "100"],
            cudf.Decimal64Dtype(scale=6, precision=9),
            False,
        ),
        (
            operator.truediv,
            ["35.23", "54.91"],
            cudf.Decimal64Dtype(scale=2, precision=4),
            decimal.Decimal("1.5"),
            ["23.4", "36.6"],
            cudf.Decimal64Dtype(scale=6, precision=9),
            False,
        ),
        (
            operator.truediv,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=2, precision=5),
            1,
            ["0", "0"],
            cudf.Decimal64Dtype(scale=6, precision=9),
            True,
        ),
        (
            operator.truediv,
            ["1.2", "0.5"],
            cudf.Decimal64Dtype(scale=1, precision=6),
            decimal.Decimal(20),
            ["10", "40"],
            cudf.Decimal64Dtype(scale=7, precision=10),
            True,
        ),
        (
            operator.truediv,
            ["1.22", "5.24"],
            cudf.Decimal64Dtype(scale=2, precision=3),
            decimal.Decimal("8.55"),
            ["7", "1"],
            cudf.Decimal64Dtype(scale=6, precision=9),
            True,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(2),
            ["98", "198"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("2.5"),
            ["97.5", "197.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            False,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            4,
            ["96", "196"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            False,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal(2),
            ["-98", "-198"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            True,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            4,
            ["-96", "-196"],
            cudf.Decimal64Dtype(scale=0, precision=6),
            True,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("2.5"),
            ["-97.5", "-197.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            True,
        ),
        (
            operator.sub,
            ["100", "200"],
            cudf.Decimal64Dtype(scale=-2, precision=3),
            decimal.Decimal("2.5"),
            ["-97.5", "-197.5"],
            cudf.Decimal64Dtype(scale=1, precision=7),
            True,
        ),
    ],
)
def test_binops_decimal_scalar(
    op, lhs, l_dtype, rhs, expect, expect_dtype, reflect
):
    lhs = cudf.Series(
        [x if x is None else decimal.Decimal(x) for x in lhs],
        dtype=l_dtype,
    )
    expect = cudf.Series(
        [x if x is None else decimal.Decimal(x) for x in expect],
        dtype=expect_dtype,
    )

    if reflect:
        lhs, rhs = rhs, lhs

    got = op(lhs, rhs)
    assert expect.dtype == got.dtype
    assert_eq(expect, got)


@pytest.mark.parametrize(
    "op, ldata, ldtype, rdata, expected1, expected2",
    [
        (
            operator.eq,
            ["100.00", "41", None],
            cudf.Decimal64Dtype(scale=0, precision=5),
            100,
            [True, False, None],
            [True, False, None],
        ),
        (
            operator.eq,
            ["100.123", "41", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            [True, False, None],
            [True, False, None],
        ),
        (
            operator.ne,
            ["100.00", "41", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            [False, True, None],
            [False, True, None],
        ),
        (
            operator.ne,
            ["100.123", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            [False, True, None],
            [False, True, None],
        ),
        (
            operator.gt,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            [False, False, True, None],
            [False, True, False, None],
        ),
        (
            operator.gt,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            [False, False, True, None],
            [False, True, False, None],
        ),
        (
            operator.ge,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            [True, False, True, None],
            [True, True, False, None],
        ),
        (
            operator.ge,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            [True, False, True, None],
            [True, True, False, None],
        ),
        (
            operator.lt,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            [False, True, False, None],
            [False, False, True, None],
        ),
        (
            operator.lt,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            [False, True, False, None],
            [False, False, True, None],
        ),
        (
            operator.le,
            ["100.00", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=2, precision=5),
            100,
            [True, True, False, None],
            [True, False, True, None],
        ),
        (
            operator.le,
            ["100.123", "41", "120.21", None],
            cudf.Decimal64Dtype(scale=3, precision=6),
            decimal.Decimal("100.123"),
            [True, True, False, None],
            [True, False, True, None],
        ),
    ],
)
@pytest.mark.parametrize("reflected", [True, False])
def test_binops_decimal_scalar_compare(
    op, ldata, ldtype, rdata, expected1, expected2, reflected
):
    """
    Tested compare operations:
        eq, lt, gt, le, ge
    Each operation has 3 data setups: pyints, Decimal, and
    For each data setup, there is at least one row that lead to one of the
    following compare results: {True, False, None}.
    """
    if not reflected:
        expected = cudf.Series(expected1, dtype=bool)
    else:
        expected = cudf.Series(expected2, dtype=bool)

    lhs = _decimal_series(ldata, ldtype)
    rhs = rdata

    if reflected:
        rhs, lhs = lhs, rhs

    actual = op(lhs, rhs)

    assert_eq(expected, actual)


@pytest.mark.parametrize("null_scalar", [None, cudf.NA, np.datetime64("NaT")])
def test_column_null_scalar_comparison(
    request, all_supported_types_as_str, null_scalar, comparison_op
):
    # This test is meant to validate that comparing
    # a series of any dtype with a null scalar produces
    # a new series where all the elements are <NA>.
    request.applymarker(
        pytest.mark.xfail(
            all_supported_types_as_str == "category",
            raises=ValueError,
            reason="Value ... not found in column",
        )
    )
    dtype = cudf.dtype(all_supported_types_as_str)

    if isinstance(null_scalar, np.datetime64):
        if dtype.kind not in "mM":
            pytest.skip(f"{null_scalar} not applicable for {dtype}")
        null_scalar = null_scalar.astype(dtype)

    data = [1, 2, 3, 4, 5]
    sr = cudf.Series(data, dtype=dtype)
    # Ordering comparisons between a datetime64/timedelta64 series and a
    # non-typed null (``None``) now raise ``TypeError`` to match pandas.
    if (
        null_scalar is None
        and comparison_op.__name__ in {"lt", "le", "gt", "ge"}
        and (
            all_supported_types_as_str.startswith("datetime64")
            or all_supported_types_as_str.startswith("timedelta64")
        )
    ):
        with pytest.raises(TypeError, match="Invalid comparison"):
            comparison_op(sr, null_scalar)
        return
    result = comparison_op(sr, null_scalar)
    if all_supported_types_as_str.startswith(
        "datetime64"
    ) or all_supported_types_as_str.startswith("timedelta64"):
        assert not result.isnull().all()
    else:
        assert result.isnull().all()


def test_equality_ops_index_mismatch(comparison_op_method):
    a = cudf.Series(
        [1, 2, 3, None, None, 4], index=["a", "b", "c", "d", "e", "f"]
    )
    b = cudf.Series(
        [-5, 4, 3, 2, 1, 0, 19, 11],
        index=["aa", "b", "c", "d", "e", "f", "y", "z"],
    )

    pa = a.to_pandas(nullable=True)
    pb = b.to_pandas(nullable=True)
    expected = getattr(pa, comparison_op_method)(pb)
    actual = getattr(a, comparison_op_method)(b).to_pandas(nullable=True)

    assert_eq(expected, actual)


@pytest.mark.parametrize("null_case", ["neither", "left", "right", "both"])
def test_null_equals_columnops(all_supported_types_as_str, null_case):
    # Generate tuples of:
    # (left_data, right_data, compare_bool
    # where compare_bool is the correct answer to
    # if the columns should compare as null equals

    def set_null_cases(column_l, column_r, case):
        if case == "neither":
            return column_l, column_r
        elif case == "left":
            column_l[1] = None
        elif case == "right":
            column_r[1] = None
        elif case == "both":
            column_l[1] = None
            column_r[1] = None
        else:
            raise ValueError("Unknown null case")
        return column_l, column_r

    data = [1, 2, 3]

    left = cudf.Series(data, dtype=all_supported_types_as_str)
    right = cudf.Series(data, dtype=all_supported_types_as_str)
    if null_case in {"left", "right"}:
        answer = False
    else:
        answer = True
    left, right = set_null_cases(left, right, null_case)
    assert left._column.equals(right._column) is answer


def test_add_series_to_dataframe():
    """Verify that missing columns result in NaNs, not NULLs."""
    assert cp.all(
        cp.isnan(
            (
                cudf.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
                + cudf.Series([1, 2, 3], index=["a", "b", "c"])
            )["c"]
        )
    )


@pytest.mark.parametrize("obj_class", [cudf.Series, cudf.Index])
def test_binops_cupy_array(obj_class, arithmetic_op):
    # Skip 0 to not deal with NaNs from division.
    data = range(1, 100)
    lhs = obj_class(data)
    rhs = cp.array(data)
    assert (arithmetic_op(lhs, rhs) == arithmetic_op(lhs, lhs)).all()


def test_binop_series_with_repeated_index():
    # GH: #11094
    psr1 = pd.Series([1, 1], index=["a", "a"])
    psr2 = pd.Series([1], index=["a"])
    gsr1 = cudf.from_pandas(psr1)
    gsr2 = cudf.from_pandas(psr2)
    expected = psr1 - psr2
    got = gsr1 - gsr2
    assert_eq(expected, got)


def test_binop_integer_power_series_series():
    # GH: #10178
    gs_base = cudf.Series([3, -3, 8, -8])
    gs_exponent = cudf.Series([1, 1, 7, 7])
    ps_base = gs_base.to_pandas()
    ps_exponent = gs_exponent.to_pandas()
    expected = ps_base**ps_exponent
    got = gs_base**gs_exponent
    assert_eq(expected, got)


def test_binop_integer_power_series_int():
    # GH: #10178
    gs_base = cudf.Series([3, -3, 8, -8])
    exponent = 1
    ps_base = gs_base.to_pandas()
    expected = ps_base**exponent
    got = gs_base**exponent
    assert_eq(expected, got)


def test_binop_integer_power_int_series():
    # GH: #10178
    base = 3
    gs_exponent = cudf.Series([1, 1, 7, 7])
    ps_exponent = gs_exponent.to_pandas()
    expected = base**ps_exponent
    got = base**gs_exponent
    assert_eq(expected, got)


def test_binop_index_series(arithmetic_op):
    gi = cudf.Index([10, 11, 12])
    gs = cudf.Series([1, 2, 3])

    actual = arithmetic_op(gi, gs)
    expected = arithmetic_op(gi.to_pandas(), gs.to_pandas())

    assert_eq(expected, actual)


@pytest.mark.parametrize("name1", [None, "name1"])
@pytest.mark.parametrize("name2", [None, "name2"])
def test_binop_index_dt_td_series_with_names(name1, name2):
    gi = cudf.Index([1, 2, 3], dtype="datetime64[ns]", name=name1)
    gs = cudf.Series([10, 11, 12], dtype="timedelta64[ns]", name=name2)
    expected = gi.to_pandas() + gs.to_pandas()
    actual = gi + gs

    assert_eq(expected, actual)


@pytest.mark.parametrize("data1", [[1, 2, 3], [10, 11, None]])
@pytest.mark.parametrize("data2", [[1, 2, 3], [10, 11, None]])
def test_binop_eq_ne_index_series(data1, data2):
    gi = cudf.Index(data1, dtype="datetime64[ns]", name=np.nan)
    gs = cudf.Series(data2, dtype="timedelta64[ns]", name="abc")

    actual = gi == gs
    expected = gi.to_pandas() == gs.to_pandas()

    assert_eq(expected, actual)

    actual = gi != gs
    expected = gi.to_pandas() != gs.to_pandas()

    assert_eq(expected, actual)


@pytest.mark.parametrize("scalar", [np.datetime64, np.timedelta64])
def test_binop_lhs_numpy_datetimelike_scalar(scalar):
    slr1 = scalar(1, "ms")
    slr2 = scalar(1, "ns")
    result = slr1 < cudf.Series([slr2])
    expected = slr1 < pd.Series([slr2])
    assert_eq(result, expected)

    result = slr2 < cudf.Series([slr1])
    expected = slr2 < pd.Series([slr1])
    assert_eq(result, expected)


@pytest.mark.parametrize("ordered", [True, False])
@pytest.mark.parametrize(
    "data_left, data_right",
    [
        [[1, 2], [1, 2]],
        [[1, 2], [1, 3]],
    ],
)
def test_cat_non_cat_compare_ops(
    comparison_op, data_left, data_right, ordered
):
    pd_non_cat = pd.Series(data_left)
    pd_cat = pd.Series(
        data_right,
        dtype=pd.CategoricalDtype(categories=data_right, ordered=ordered),
    )

    cudf_non_cat = cudf.Series(pd_non_cat)
    cudf_cat = cudf.Series(pd_cat)

    if (
        not ordered and comparison_op not in {operator.eq, operator.ne}
    ) or comparison_op in {
        operator.gt,
        operator.lt,
        operator.le,
        operator.ge,
    }:
        with pytest.raises(TypeError):
            comparison_op(pd_non_cat, pd_cat)
        with pytest.raises(TypeError):
            comparison_op(cudf_non_cat, cudf_cat)
    else:
        expected = comparison_op(pd_non_cat, pd_cat)
        result = comparison_op(cudf_non_cat, cudf_cat)
        assert_eq(result, expected)


@pytest.mark.parametrize(
    "left_data, right_data",
    [[["a", "b"], [1, 2]], [[[1, 2, 3], [4, 5]], [{"a": 1}, {"a": 2}]]],
)
@pytest.mark.parametrize(
    "op, expected_data",
    [[operator.eq, [False, False]], [operator.ne, [True, True]]],
)
@pytest.mark.parametrize("with_na", [True, False])
def test_eq_ne_non_comparable_types(
    left_data, right_data, op, expected_data, with_na
):
    left_data = left_data.copy()
    expected_data = expected_data.copy()
    left = cudf.Series(left_data)
    right = cudf.Series(right_data)
    result = op(left, right)
    expected = cudf.Series(expected_data)
    assert_eq(result, expected)


def test_binops_compare_stdlib_date_scalar(comparison_op):
    dt = datetime.date(2020, 1, 1)
    data = [dt]
    sr = cudf.Series(data)
    # cudf promotes ``date`` to ``datetime64``. pandas treats a bare
    # ``date`` as not comparable with a datetime64 column: ordering ops
    # raise, ``==``/``!=`` return all-False/True.
    ps = pd.Series(data, dtype=sr.dtype)
    if comparison_op.__name__ in {"lt", "le", "gt", "ge"}:
        with pytest.raises(TypeError, match="Invalid comparison"):
            comparison_op(sr, dt)
        with pytest.raises(TypeError, match="Invalid comparison"):
            comparison_op(ps, dt)
        return
    result = comparison_op(sr, dt)
    expected = comparison_op(ps, dt)
    assert_eq(result, expected)


@pytest.mark.parametrize("xp", [cp, np])
def test_singleton_array(binary_op, xp):
    # Validate that we handle singleton numpy/cupy arrays appropriately
    lhs = cudf.Series([1, 2, 3])
    rhs_device = xp.array(1)
    rhs_host = np.array(1)
    expect = binary_op(lhs.to_pandas(), rhs_host)
    got = binary_op(lhs, rhs_device)
    assert_eq(expect, got)


def test_binops_float_scalar_decimal():
    result = 1.0 - cudf.Series(
        [decimal.Decimal("1"), decimal.Decimal("-2.5"), None],
        dtype=cudf.Decimal32Dtype(3, 2),
    )
    expected = cudf.Series([0.0, -3.5, None], dtype="float64")
    assert_eq(result, expected)


@pytest.mark.parametrize(
    "scalars",
    [
        pd.NaT,
        np.datetime64("NaT"),
        None,
        pd.NA,
        np.nan,
        np.datetime64("2020-01-01"),
    ],
)
@pytest.mark.parametrize(
    "comparison_op",
    [
        operator.eq,
        operator.ne,
    ],
)
def test_binops_comparisons_datatime_with_scalars(scalars, comparison_op):
    with cudf.option_context("mode.pandas_compatible", True):
        ser = cudf.Series(
            [
                np.datetime64("2020-01-01"),
                np.datetime64("2021-06-15"),
                np.datetime64("2022-12-31"),
                None,
            ]
        )
        expect = comparison_op(ser.to_pandas(), scalars)
        got = comparison_op(ser, scalars)
        assert_eq(expect, got)

        expect = comparison_op(scalars, ser.to_pandas())
        got = comparison_op(scalars, ser)
        assert_eq(expect, got)


def test_timedelta_arrow_backed_comparisions_pandas_compat():
    s = pd.Series(
        pd.arrays.ArrowExtensionArray(
            pa.array([1, None, 3], type=pa.duration("ns"))
        )
    )

    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.from_pandas(s)
        assert_eq(s == s, gs == gs)
        assert_eq(s != s, gs != gs)


def test_decimal_arrow_backed_comparisons_pandas_compat(comparison_op):
    s = pd.Series(
        pd.arrays.ArrowExtensionArray(
            pa.array(
                [Decimal("1.234"), Decimal("0.000"), None],
                type=pa.decimal128(7, 3),
            )
        )
    )

    with cudf.option_context("mode.pandas_compatible", True):
        gs = cudf.from_pandas(s)
        expect = comparison_op(s, s)
        got = comparison_op(gs, gs)
        assert_eq(expect, got)


def test_arrow_backed_unsigned_subtract_python_int():
    # An unsigned ArrowDtype minus a Python int is promoted to signed
    # int64[pyarrow] by pandas, so it must not raise a spurious unsigned
    # overflow even when elements are smaller than the subtrahend.
    s = pd.Series(
        pd.arrays.ArrowExtensionArray(pa.array([0, 1, 2, 5], type=pa.uint8()))
    )
    gs = cudf.from_pandas(s)
    assert_eq(s - 3, gs - 3)
    assert_eq(3 - s, 3 - gs)


def test_arrow_backed_unsigned_subtract_overflow():
    # Subtraction of two unsigned ArrowDtype operands that underflows raises,
    # mirroring pyarrow.compute.subtract_checked.
    s = pd.Series(
        pd.arrays.ArrowExtensionArray(pa.array([0, 1], type=pa.uint8()))
    )
    other = pd.Series(
        pd.arrays.ArrowExtensionArray(pa.array([1, 0], type=pa.uint8()))
    )
    gs = cudf.from_pandas(s)
    gother = cudf.from_pandas(other)
    with pytest.raises(pa.ArrowInvalid):
        gs - gother

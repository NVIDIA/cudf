# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
import pyarrow as pa
import pytest
from utils import assert_column_eq

import pylibcudf as plc


def idfn(param):
    ltype, rtype, outtype, plc_op, _ = param
    params = (plc_op.name, ltype, rtype, outtype)
    return "-".join(map(str, params))


@pytest.fixture(params=[True, False], ids=["nulls", "no_nulls"])
def nulls(request):
    return request.param


def make_col(dtype, nulls):
    if dtype == "int64":
        data = [1, 2, 3, 4, 5]
        pa_type = pa.int64()
    elif dtype == "uint64":
        data = [1, 2, 3, 4, 5]
        pa_type = pa.uint64()
    elif dtype == "float64":
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        pa_type = pa.float64()
    elif dtype == "bool":
        data = [True, False, True, False, True]
        pa_type = pa.bool_()
    elif dtype == "timestamp64[ns]":
        data = [
            np.datetime64("2022-01-01"),
            np.datetime64("2022-01-02"),
            np.datetime64("2022-01-03"),
            np.datetime64("2022-01-04"),
            np.datetime64("2022-01-05"),
        ]
        pa_type = pa.timestamp("ns")
    elif dtype == "timedelta64[ns]":
        data = [
            np.timedelta64(1, "ns"),
            np.timedelta64(2, "ns"),
            np.timedelta64(3, "ns"),
            np.timedelta64(4, "ns"),
            np.timedelta64(5, "ns"),
        ]
        pa_type = pa.duration("ns")
    else:
        raise ValueError("Unsupported dtype")

    if nulls:
        data[3] = None

    return pa.array(data, type=pa_type)


@pytest.fixture
def tests(request, nulls):
    ltype, rtype, py_outtype, plc_op, py_op = request.param
    pa_lhs, pa_rhs = make_col(ltype, nulls), make_col(rtype, nulls)
    plc_lhs, plc_rhs = (
        plc.Column.from_arrow(pa_lhs),
        plc.Column.from_arrow(pa_rhs),
    )
    plc_dtype = plc.DataType.from_arrow(
        pa.from_numpy_dtype(np.dtype(py_outtype))
    )
    return (
        pa_lhs,
        pa_rhs,
        py_outtype,
        plc_lhs,
        plc_rhs,
        plc_dtype,
        py_op,
        plc_op,
    )


def custom_pyop(func):
    def wrapper(x, y):
        x = x.to_pylist()
        y = y.to_pylist()

        def inner(x, y):
            if x is None or y is None:
                return None
            return func(x, y)

        return pa.array([inner(x, y) for x, y in zip(x, y, strict=True)])

    return wrapper


@custom_pyop
def py_floordiv(x, y):
    return x // y


@custom_pyop
def py_pmod(x, y):
    return (x % y + y) % y


@custom_pyop
def py_mod(x, y):
    return x % y


@custom_pyop
def py_atan2(x, y):
    return math.atan2(x, y)


@custom_pyop
def py_shift_right_unsigned(x, y):
    unsigned_x = np.uint32(x)
    result = unsigned_x >> y
    return result


@pytest.mark.parametrize(
    "tests",
    [
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.ADD,
            pa.compute.add,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.ADD,
            pa.compute.add,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.ADD,
            pa.compute.add,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.SUB,
            pa.compute.subtract,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.SUB,
            pa.compute.subtract,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.SUB,
            pa.compute.subtract,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.MUL,
            pa.compute.multiply,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.MUL,
            pa.compute.multiply,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.MUL,
            pa.compute.multiply,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.DIV,
            pa.compute.divide,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.DIV,
            pa.compute.divide,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.DIV,
            pa.compute.divide,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.TRUE_DIV,
            pa.compute.divide,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.TRUE_DIV,
            pa.compute.divide,
        ),
        (
            "int64",
            "int64",
            "timedelta64[ns]",
            plc.binaryop.BinaryOperator.TRUE_DIV,
            pa.compute.divide,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.FLOOR_DIV,
            py_floordiv,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.FLOOR_DIV,
            py_floordiv,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.FLOOR_DIV,
            py_floordiv,
        ),
        ("int64", "int64", "int64", plc.binaryop.BinaryOperator.MOD, py_mod),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.MOD,
            py_mod,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.MOD,
            py_mod,
        ),
        ("int64", "int64", "int64", plc.binaryop.BinaryOperator.PMOD, py_pmod),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.PMOD,
            py_pmod,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.PMOD,
            py_pmod,
        ),
        ("int64", "int64", "int64", plc.binaryop.BinaryOperator.PYMOD, py_mod),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.PYMOD,
            py_mod,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.PYMOD,
            py_mod,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.POW,
            pa.compute.power,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.POW,
            pa.compute.power,
        ),
        (
            "int64",
            "int64",
            "timedelta64[ns]",
            plc.binaryop.BinaryOperator.POW,
            pa.compute.power,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.INT_POW,
            pa.compute.power,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.INT_POW,
            pa.compute.power,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.INT_POW,
            pa.compute.power,
        ),
        (
            "float64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.LOG_BASE,
            pa.compute.logb,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.LOG_BASE,
            pa.compute.logb,
        ),
        (
            "int64",
            "int64",
            "timedelta64[ns]",
            plc.binaryop.BinaryOperator.LOG_BASE,
            pa.compute.logb,
        ),
        (
            "float64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.ATAN2,
            py_atan2,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.ATAN2,
            py_atan2,
        ),
        (
            "int64",
            "int64",
            "timedelta64[ns]",
            plc.binaryop.BinaryOperator.ATAN2,
            py_atan2,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.SHIFT_LEFT,
            pa.compute.shift_left,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.SHIFT_LEFT,
            pa.compute.shift_left,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.SHIFT_LEFT,
            pa.compute.shift_left,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.SHIFT_RIGHT,
            pa.compute.shift_right,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.SHIFT_RIGHT,
            pa.compute.shift_right,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.SHIFT_RIGHT,
            pa.compute.shift_right,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.SHIFT_RIGHT_UNSIGNED,
            py_shift_right_unsigned,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.SHIFT_RIGHT_UNSIGNED,
            py_shift_right_unsigned,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.SHIFT_RIGHT_UNSIGNED,
            py_shift_right_unsigned,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.BITWISE_AND,
            pa.compute.bit_wise_and,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.BITWISE_AND,
            pa.compute.bit_wise_and,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.BITWISE_AND,
            pa.compute.bit_wise_and,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.BITWISE_OR,
            pa.compute.bit_wise_or,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.BITWISE_OR,
            pa.compute.bit_wise_or,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.BITWISE_OR,
            pa.compute.bit_wise_or,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.BITWISE_XOR,
            pa.compute.bit_wise_xor,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.BITWISE_XOR,
            pa.compute.bit_wise_xor,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.BITWISE_XOR,
            pa.compute.bit_wise_xor,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.LOGICAL_AND,
            pa.compute.and_,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.LOGICAL_AND,
            pa.compute.and_,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.LOGICAL_OR,
            pa.compute.or_,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.LOGICAL_OR,
            pa.compute.or_,
        ),
        (
            "int64",
            "int64",
            "bool",
            plc.binaryop.BinaryOperator.EQUAL,
            pa.compute.equal,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.EQUAL,
            pa.compute.equal,
        ),
        (
            "int64",
            "int64",
            "bool",
            plc.binaryop.BinaryOperator.NOT_EQUAL,
            pa.compute.not_equal,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.NOT_EQUAL,
            pa.compute.not_equal,
        ),
        (
            "int64",
            "int64",
            "bool",
            plc.binaryop.BinaryOperator.LESS,
            pa.compute.less,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.LESS,
            pa.compute.less,
        ),
        (
            "int64",
            "int64",
            "bool",
            plc.binaryop.BinaryOperator.GREATER,
            pa.compute.greater,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.GREATER,
            pa.compute.greater,
        ),
        (
            "int64",
            "int64",
            "bool",
            plc.binaryop.BinaryOperator.LESS_EQUAL,
            pa.compute.less_equal,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.LESS_EQUAL,
            pa.compute.less_equal,
        ),
        (
            "int64",
            "int64",
            "bool",
            plc.binaryop.BinaryOperator.GREATER_EQUAL,
            pa.compute.greater_equal,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.GREATER_EQUAL,
            pa.compute.greater_equal,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.NULL_EQUALS,
            pa.compute.equal,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.NULL_EQUALS,
            pa.compute.equal,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.NULL_MAX,
            pa.compute.max_element_wise,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.NULL_MAX,
            pa.compute.max_element_wise,
        ),
        (
            "int64",
            "int64",
            "datetime64[ns]",
            plc.binaryop.BinaryOperator.NULL_MIN,
            pa.compute.min_element_wise,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.NULL_MIN,
            pa.compute.min_element_wise,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.NULL_NOT_EQUALS,
            pa.compute.not_equal,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.NULL_NOT_EQUALS,
            pa.compute.not_equal,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
            pa.compute.and_,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.NULL_LOGICAL_AND,
            pa.compute.and_,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
            pa.compute.or_,
        ),
        (
            "int64",
            "float64",
            "float64",
            plc.binaryop.BinaryOperator.NULL_LOGICAL_OR,
            pa.compute.or_,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.GENERIC_BINARY,
            None,
        ),
        (
            "int64",
            "int64",
            "int64",
            plc.binaryop.BinaryOperator.INVALID_BINARY,
            None,
        ),
    ],
    indirect=True,
    ids=idfn,
)
def test_binaryops(tests):
    (
        pa_lhs,
        pa_rhs,
        py_outtype,
        plc_lhs,
        plc_rhs,
        plc_outtype,
        py_op,
        plc_op,
    ) = tests

    def get_result():
        return plc.binaryop.binary_operation(
            plc_lhs,
            plc_rhs,
            plc_op,
            plc_outtype,
        )

    if not plc.binaryop.is_supported_operation(
        plc_outtype, plc_lhs.type(), plc_rhs.type(), plc_op
    ):
        with pytest.raises(TypeError):
            get_result()
    else:
        expect = py_op(pa_lhs, pa_rhs).cast(py_outtype)
        got = get_result()
        assert_column_eq(expect, got)


_NAN = float("nan")


@pytest.mark.parametrize(
    "op, ieee754, etl",
    [
        (
            plc.binaryop.BinaryOperator.EQUAL,
            [False, False, False, True],
            [False, True, False, True],
        ),
        (
            plc.binaryop.BinaryOperator.NOT_EQUAL,
            [True, True, True, False],
            [True, False, True, False],
        ),
        (
            plc.binaryop.BinaryOperator.LESS,
            [False, False, False, False],
            [True, False, False, False],
        ),
        (
            plc.binaryop.BinaryOperator.GREATER_EQUAL,
            [False, False, False, True],
            [False, True, True, True],
        ),
    ],
)
def test_binaryop_nan_semantics(op, ieee754, etl):
    lhs = plc.Column.from_arrow(pa.array([1.0, _NAN, _NAN, 3.0]))
    rhs = plc.Column.from_arrow(pa.array([_NAN, _NAN, 1.0, 3.0]))
    bool_type = plc.DataType(plc.TypeId.BOOL8)

    # The default is IEEE 754
    got = plc.binaryop.binary_operation(lhs, rhs, op, bool_type)
    assert got.to_arrow().to_pylist() == ieee754
    for semantics in (
        plc.types.NanSemantics.DEFAULT,
        plc.types.NanSemantics.IEEE754,
    ):
        got = plc.binaryop.binary_operation(lhs, rhs, op, bool_type, semantics)
        assert got.to_arrow().to_pylist() == ieee754
    got = plc.binaryop.binary_operation(
        lhs, rhs, op, bool_type, semantics=plc.types.NanSemantics.ETL
    )
    assert got.to_arrow().to_pylist() == etl

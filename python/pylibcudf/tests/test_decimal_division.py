# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from decimal import Decimal

import pyarrow as pa
import pytest

import pylibcudf as plc

DECIMAL_TYPES = [
    plc.TypeId.DECIMAL32,
    plc.TypeId.DECIMAL64,
    plc.TypeId.DECIMAL128,
]


def _column(values, type_id, scale):
    """Build a decimal column. ``scale`` is the number of digits after the
    decimal point, as in Python and Arrow.
    """
    arr = pa.array(
        [None if v is None else Decimal(v) for v in values],
        type=pa.decimal128(38, scale),
    )
    return plc.unary.cast(
        plc.Column.from_arrow(arr), plc.DataType(type_id, -scale)
    )


def _scalar(value, type_id, scale):
    return plc.copying.get_element(_column([value], type_id, scale), 0)


def _invalid_scalar(type_id, scale):
    return _scalar(None, type_id, scale)


def _values(col):
    as_128 = plc.unary.cast(
        col, plc.DataType(plc.TypeId.DECIMAL128, col.type().scale())
    )
    return as_128.to_arrow().to_pylist()


def _expect(values):
    return [None if v is None else Decimal(v) for v in values]


@pytest.fixture(params=DECIMAL_TYPES, ids=lambda t: t.name)
def type_id(request):
    return request.param


@pytest.mark.parametrize(
    "round_method, expected",
    [
        (
            plc.round.RoundingMethod.HALF_UP,
            ["0.3", "0.4", "-0.3", "-0.4", None, None, None],
        ),
        (
            plc.round.RoundingMethod.HALF_EVEN,
            ["0.2", "0.4", "-0.2", "-0.4", None, None, None],
        ),
    ],
)
def test_divide_decimal_columns(type_id, round_method, expected):
    # 0.25, 0.35, -0.25, -0.35, division by zero, null dividend, null divisor
    lhs = _column(
        ["0.5", "0.7", "-0.5", "-0.7", "1.0", None, "1.0"], type_id, 1
    )
    rhs = _column(["2", "2", "2", "2", "0", "1", None], type_id, 0)

    got = plc.decimal_division.divide_decimal(lhs, rhs, round_method)

    assert got.type() == lhs.type()
    assert _values(got) == _expect(expected)


def test_divide_decimal_default_rounding_is_half_up(type_id):
    lhs = _column(["0.5"], type_id, 1)
    rhs = _column(["2"], type_id, 0)

    got = plc.decimal_division.divide_decimal(lhs, rhs)

    assert _values(got) == _expect(["0.3"])


@pytest.mark.parametrize(
    "type_id, minimum",
    [
        (plc.TypeId.DECIMAL32, -(2**31)),
        (plc.TypeId.DECIMAL64, -(2**63)),
    ],
    ids=["DECIMAL32", "DECIMAL64"],
)
def test_divide_decimal_overflow_is_null(type_id, minimum):
    # MIN / -1 does not fit, while MAX / -1 does
    lhs = _column([str(minimum), str(-minimum - 1)], type_id, 0)
    rhs = _column(["-1", "-1"], type_id, 0)

    got = plc.decimal_division.divide_decimal(lhs, rhs)

    assert _values(got) == _expect([None, str(minimum + 1)])


def test_divide_decimal_column_scalar(type_id):
    # 1.23 / 2.0 = 0.615 -> 0.62, 4.56 / 2.0 = 2.28, the divisor has a
    # different scale from the column
    lhs = _column(["1.23", "4.56", None], type_id, 2)
    rhs = _scalar("2.0", type_id, 1)

    got = plc.decimal_division.divide_decimal_column_scalar(lhs, rhs)

    assert got.type() == lhs.type()
    assert _values(got) == _expect(["0.62", "2.28", None])


def test_divide_decimal_column_scalar_zero(type_id):
    lhs = _column(["1.23", "4.56"], type_id, 2)
    rhs = _scalar("0", type_id, 0)

    got = plc.decimal_division.divide_decimal_column_scalar(lhs, rhs)

    assert _values(got) == [None, None]


def test_divide_decimal_scalar_column(type_id):
    # 1.5 / 3 = 0.5, 1.5 / 4 = 0.375 -> 0.4, 1.5 / 0 is null
    lhs = _scalar("1.5", type_id, 1)
    rhs = _column(["3", "4", "0", None], type_id, 0)

    got = plc.decimal_division.divide_decimal_scalar_column(
        lhs, rhs, plc.round.RoundingMethod.HALF_EVEN
    )

    assert got.type() == plc.DataType(type_id, -1)
    assert _values(got) == _expect(["0.5", "0.4", None, None])


def test_divide_decimal_invalid_scalar(type_id):
    col = _column(["1.00", "2.00"], type_id, 2)
    scalar = _invalid_scalar(type_id, 2)

    got_column_scalar = plc.decimal_division.divide_decimal_column_scalar(
        col, scalar
    )
    got_scalar_column = plc.decimal_division.divide_decimal_scalar_column(
        scalar, col
    )

    assert _values(got_column_scalar) == [None, None]
    assert _values(got_scalar_column) == [None, None]


def test_divide_decimal_empty(type_id):
    lhs = _column([], type_id, 2)
    rhs = _column([], type_id, 1)

    got = plc.decimal_division.divide_decimal(lhs, rhs)

    assert got.size() == 0
    assert got.type() == lhs.type()


@pytest.mark.parametrize(
    "func, make_lhs, make_rhs",
    [
        (
            plc.decimal_division.divide_decimal,
            lambda t: _column(["1"], t, 0),
            lambda t: _column(["1"], t, 0),
        ),
        (
            plc.decimal_division.divide_decimal_column_scalar,
            lambda t: _column(["1"], t, 0),
            lambda t: _scalar("1", t, 0),
        ),
        (
            plc.decimal_division.divide_decimal_scalar_column,
            lambda t: _scalar("1", t, 0),
            lambda t: _column(["1"], t, 0),
        ),
    ],
    ids=["column_column", "column_scalar", "scalar_column"],
)
class TestInvalidArguments:
    def test_none_lhs(self, func, make_lhs, make_rhs):
        with pytest.raises(TypeError, match="Left operand must not be None"):
            func(None, make_rhs(plc.TypeId.DECIMAL64))

    def test_none_rhs(self, func, make_lhs, make_rhs):
        with pytest.raises(TypeError, match="Right operand must not be None"):
            func(make_lhs(plc.TypeId.DECIMAL64), None)

    def test_mismatched_decimal_types(self, func, make_lhs, make_rhs):
        with pytest.raises(RuntimeError, match="must match"):
            func(
                make_lhs(plc.TypeId.DECIMAL32),
                make_rhs(plc.TypeId.DECIMAL64),
            )


def test_divide_decimal_non_decimal_type_error():
    ints = plc.Column.from_arrow(pa.array([1, 2], type=pa.int32()))
    decimals = _column(["1", "2"], plc.TypeId.DECIMAL32, 0)

    with pytest.raises(TypeError, match="Left operand must be a decimal"):
        plc.decimal_division.divide_decimal(ints, decimals)
    with pytest.raises(TypeError, match="Right operand must be a decimal"):
        plc.decimal_division.divide_decimal(decimals, ints)

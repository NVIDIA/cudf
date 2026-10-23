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

    got = plc.decimal_division.divide_decimal(
        lhs, rhs, lhs.type(), round_method
    )

    assert got.type() == lhs.type()
    assert _values(got) == _expect(expected)


def test_divide_decimal_default_rounding_is_half_up(type_id):
    lhs = _column(["0.5"], type_id, 1)
    rhs = _column(["2"], type_id, 0)

    got = plc.decimal_division.divide_decimal(lhs, rhs, lhs.type())

    assert _values(got) == _expect(["0.3"])


@pytest.mark.parametrize(
    "type_id, digits",
    [
        (plc.TypeId.DECIMAL32, 9),
        (plc.TypeId.DECIMAL64, 18),
    ],
    ids=["DECIMAL32", "DECIMAL64"],
)
def test_divide_decimal_precision_limit_is_null(type_id, digits):
    # Quotients with more digits than the output type can hold are null
    largest = 10**digits - 1
    lhs = _column([str(largest), str(largest + 1)], type_id, 0)
    rhs = _column(["-1", "-1"], type_id, 0)

    got = plc.decimal_division.divide_decimal(lhs, rhs, lhs.type())

    assert _values(got) == _expect([str(-largest), None])


def test_divide_decimal_output_scale(type_id):
    # 1.23 / 2.0 = 0.615
    lhs = _column(["1.23", "-1.23"], type_id, 2)
    rhs = _column(["2.0", "2.0"], type_id, 1)

    finer = plc.decimal_division.divide_decimal(
        lhs, rhs, plc.DataType(type_id, -4)
    )
    coarser = plc.decimal_division.divide_decimal(
        lhs, rhs, plc.DataType(type_id, 0)
    )

    assert finer.type() == plc.DataType(type_id, -4)
    assert _values(finer) == _expect(["0.6150", "-0.6150"])
    assert coarser.type() == plc.DataType(type_id, 0)
    assert _values(coarser) == _expect(["1", "-1"])


def test_divide_decimal_mixed_widths():
    # 1.23 / 2.0 = 0.615 -> 0.62, with three different widths
    lhs = _column(["1.23", "-1.23"], plc.TypeId.DECIMAL32, 2)
    rhs = _column(["2.0", "2.0"], plc.TypeId.DECIMAL128, 1)
    rhs_scalar = _scalar("2.0", plc.TypeId.DECIMAL128, 1)
    output_type = plc.DataType(plc.TypeId.DECIMAL64, -2)

    got = plc.decimal_division.divide_decimal(lhs, rhs, output_type)
    got_scalar = plc.decimal_division.divide_decimal_column_scalar(
        lhs, rhs_scalar, output_type
    )

    assert got.type() == output_type
    assert _values(got) == _expect(["0.62", "-0.62"])
    assert _values(got_scalar) == _expect(["0.62", "-0.62"])


def test_divide_decimal_column_scalar(type_id):
    # 1.23 / 2.0 = 0.615 -> 0.62, 4.56 / 2.0 = 2.28, the divisor has a
    # different scale from the column
    lhs = _column(["1.23", "4.56", None], type_id, 2)
    rhs = _scalar("2.0", type_id, 1)

    got = plc.decimal_division.divide_decimal_column_scalar(
        lhs, rhs, lhs.type()
    )

    assert got.type() == lhs.type()
    assert _values(got) == _expect(["0.62", "2.28", None])


def test_divide_decimal_column_scalar_zero(type_id):
    lhs = _column(["1.23", "4.56"], type_id, 2)
    rhs = _scalar("0", type_id, 0)

    got = plc.decimal_division.divide_decimal_column_scalar(
        lhs, rhs, lhs.type()
    )

    assert _values(got) == [None, None]


def test_divide_decimal_scalar_column(type_id):
    # 1.5 / 3 = 0.5, 1.5 / 4 = 0.375 -> 0.4, 1.5 / 0 is null
    lhs = _scalar("1.5", type_id, 1)
    rhs = _column(["3", "4", "0", None], type_id, 0)

    got = plc.decimal_division.divide_decimal_scalar_column(
        lhs,
        rhs,
        plc.DataType(type_id, -1),
        plc.round.RoundingMethod.HALF_EVEN,
    )

    assert got.type() == plc.DataType(type_id, -1)
    assert _values(got) == _expect(["0.5", "0.4", None, None])


def test_divide_decimal_invalid_scalar(type_id):
    col = _column(["1.00", "2.00"], type_id, 2)
    scalar = _invalid_scalar(type_id, 2)

    got_column_scalar = plc.decimal_division.divide_decimal_column_scalar(
        col, scalar, col.type()
    )
    got_scalar_column = plc.decimal_division.divide_decimal_scalar_column(
        scalar, col, col.type()
    )

    assert _values(got_column_scalar) == [None, None]
    assert _values(got_scalar_column) == [None, None]


def test_divide_decimal_empty(type_id):
    lhs = _column([], type_id, 2)
    rhs = _column([], type_id, 1)

    output_type = plc.DataType(plc.TypeId.DECIMAL64, -4)

    got = plc.decimal_division.divide_decimal(lhs, rhs, output_type)

    assert got.size() == 0
    assert got.type() == output_type


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
    output_type = plc.DataType(plc.TypeId.DECIMAL64, 0)

    def test_none_lhs(self, func, make_lhs, make_rhs):
        with pytest.raises(TypeError, match="Left operand must not be None"):
            func(None, make_rhs(plc.TypeId.DECIMAL64), self.output_type)

    def test_none_rhs(self, func, make_lhs, make_rhs):
        with pytest.raises(TypeError, match="Right operand must not be None"):
            func(make_lhs(plc.TypeId.DECIMAL64), None, self.output_type)

    def test_none_output_type(self, func, make_lhs, make_rhs):
        with pytest.raises(TypeError, match="output_type must not be None"):
            func(
                make_lhs(plc.TypeId.DECIMAL64),
                make_rhs(plc.TypeId.DECIMAL64),
                None,
            )

    def test_non_decimal_output_type(self, func, make_lhs, make_rhs):
        with pytest.raises(
            TypeError, match="output_type must be a decimal type"
        ):
            func(
                make_lhs(plc.TypeId.DECIMAL64),
                make_rhs(plc.TypeId.DECIMAL64),
                plc.DataType(plc.TypeId.INT64),
            )


def test_decimal_intermediate_values():
    intermediate = plc.decimal_division.DecimalIntermediate
    assert [intermediate.STANDARD, intermediate.WIDE] == [0, 1]


@pytest.mark.parametrize(
    "intermediate, expected",
    [
        (None, [None, None, None]),
        (
            plc.decimal_division.DecimalIntermediate.STANDARD,
            [None, None, None],
        ),
        (plc.decimal_division.DecimalIntermediate.WIDE, ["1", "-1", "-3"]),
    ],
)
def test_divide_decimal_intermediate(intermediate, expected):
    # Rescaling the dividends to give quotients with 20 fractional digits
    # needs more than 128 bits, so only the wide intermediate gives results.
    lhs = _column(["1", "-1", "3"], plc.TypeId.DECIMAL128, 10)
    rhs = _column(["1", "1", "-1"], plc.TypeId.DECIMAL128, 20)
    output_type = plc.DataType(plc.TypeId.DECIMAL128, -20)
    kwargs = {} if intermediate is None else {"intermediate": intermediate}

    got = plc.decimal_division.divide_decimal(
        lhs, rhs, output_type, plc.round.RoundingMethod.HALF_UP, **kwargs
    )

    assert _values(got) == _expect(expected)


def test_divide_decimal_wide_intermediate_with_scalars():
    lhs = _column(["1", "-1", "3"], plc.TypeId.DECIMAL128, 10)
    rhs = _column(["1", "1", "-1"], plc.TypeId.DECIMAL128, 20)
    output_type = plc.DataType(plc.TypeId.DECIMAL128, -20)
    wide = plc.decimal_division.DecimalIntermediate.WIDE
    half_up = plc.round.RoundingMethod.HALF_UP

    got = plc.decimal_division.divide_decimal_column_scalar(
        lhs,
        _scalar("1", plc.TypeId.DECIMAL128, 20),
        output_type,
        half_up,
        wide,
    )
    assert _values(got) == _expect(["1", "-1", "3"])

    got = plc.decimal_division.divide_decimal_scalar_column(
        _scalar("1", plc.TypeId.DECIMAL128, 10),
        rhs,
        output_type,
        half_up,
        intermediate=wide,
    )
    assert _values(got) == _expect(["1", "1", "-1"])


def test_divide_decimal_non_decimal_type_error():
    ints = plc.Column.from_arrow(pa.array([1, 2], type=pa.int32()))
    decimals = _column(["1", "2"], plc.TypeId.DECIMAL32, 0)

    with pytest.raises(TypeError, match="Left operand must be a decimal"):
        plc.decimal_division.divide_decimal(ints, decimals, decimals.type())
    with pytest.raises(TypeError, match="Right operand must be a decimal"):
        plc.decimal_division.divide_decimal(decimals, ints, decimals.type())

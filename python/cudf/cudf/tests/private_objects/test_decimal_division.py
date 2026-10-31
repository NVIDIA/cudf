# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for decimal division with scale preservation (divide_decimal function)
"""

from decimal import Decimal

import pytest

import cudf
from cudf.testing import assert_eq


class TestDecimalDivision:
    """Test suite for divide_decimal functionality"""

    def test_divide_decimal_basic(self):
        """Test basic divide_decimal functionality with scale preservation"""
        # Create decimal series
        s1 = cudf.Series([Decimal("1.23"), Decimal("4.56"), Decimal("7.89")])
        s2 = cudf.Series([Decimal("2.0"), Decimal("3.0"), Decimal("4.0")])

        # divide_decimal (scale preserved)
        decimal_result = s1._column.divide_decimal(s2._column)

        # Check that scales are different
        # Standard division: scale = lhs.scale - rhs.scale
        # divide_decimal: scale = lhs.scale
        assert decimal_result.dtype.scale == s1._column.dtype.scale

        # Check values are approximately correct
        # 1.23 / 2.0 = 0.615 -> 0.62 (with HALF_UP)
        # 4.56 / 3.0 = 1.52
        # 7.89 / 4.0 = 1.9725 -> 1.97 (with HALF_UP)
        expected = cudf.Series(
            [Decimal("0.62"), Decimal("1.52"), Decimal("1.97")]
        )

        # Convert result to series for comparison
        result_series = cudf.Series._from_column(decimal_result)

        # Compare decimal values using cudf's testing utility
        # Note: check_dtype=False because precision might differ
        assert_eq(result_series, expected, check_dtype=False)

    def test_rounding_mode_half_up(self):
        """Test HALF_UP rounding mode"""
        s1 = cudf.Series([Decimal("1.25"), Decimal("2.35"), Decimal("3.45")])
        s2 = cudf.Series([Decimal("2.0"), Decimal("2.0"), Decimal("2.0")])

        # HALF_UP: 0.5 rounds away from zero
        result = s1._column.divide_decimal(s2._column, rounding_mode="HALF_UP")

        # 1.25 / 2.0 = 0.625 -> 0.63
        # 2.35 / 2.0 = 1.175 -> 1.18
        # 3.45 / 2.0 = 1.725 -> 1.73
        expected = cudf.Series(
            [Decimal("0.63"), Decimal("1.18"), Decimal("1.73")]
        )

        # Convert result to series for comparison
        result_series = cudf.Series._from_column(result)

        # Verify HALF_UP rounding behavior
        assert_eq(result_series, expected, check_dtype=False)

    def test_rounding_mode_half_even(self):
        """Test HALF_EVEN rounding mode (banker's rounding)"""
        s1 = cudf.Series(
            [
                Decimal("1.25"),
                Decimal("1.35"),
                Decimal("2.25"),
                Decimal("2.35"),
            ]
        )
        s2 = cudf.Series(
            [Decimal("2.0"), Decimal("2.0"), Decimal("2.0"), Decimal("2.0")]
        )

        # HALF_EVEN: 0.5 rounds to nearest even
        result = s1._column.divide_decimal(
            s2._column, rounding_mode="HALF_EVEN"
        )

        # 1.25 / 2.0 = 0.625 -> 0.62 (even)
        # 1.35 / 2.0 = 0.675 -> 0.68 (even)
        # 2.25 / 2.0 = 1.125 -> 1.12 (even)
        # 2.35 / 2.0 = 1.175 -> 1.18 (even)
        expected = cudf.Series(
            [
                Decimal("0.62"),
                Decimal("0.68"),
                Decimal("1.12"),
                Decimal("1.18"),
            ]
        )

        # Convert result to series for comparison
        result_series = cudf.Series._from_column(result)

        # Verify HALF_EVEN rounding behavior
        assert_eq(result_series, expected, check_dtype=False)

    def test_negative_numbers(self):
        """Test divide_decimal with negative numbers"""
        s1 = cudf.Series([Decimal("-1.23"), Decimal("1.23"), Decimal("-1.23")])
        s2 = cudf.Series([Decimal("2.0"), Decimal("-2.0"), Decimal("-2.0")])

        result = s1._column.divide_decimal(s2._column)

        # -1.23 / 2.0 = -0.615 -> -0.62
        # 1.23 / -2.0 = -0.615 -> -0.62
        # -1.23 / -2.0 = 0.615 -> 0.62
        expected = cudf.Series(
            [Decimal("-0.62"), Decimal("-0.62"), Decimal("0.62")]
        )

        # Convert result to series for comparison
        result_series = cudf.Series._from_column(result)

        # Verify negative number handling
        assert_eq(result_series, expected, check_dtype=False)

    def test_divide_by_scalar(self):
        """Test dividing by a scalar decimal value"""
        s = cudf.Series([Decimal("10.50"), Decimal("20.75"), Decimal("30.25")])
        scalar = Decimal("2.00")

        result = s._column.divide_decimal(scalar)

        # Scale should be preserved
        assert result.dtype.scale == s._column.dtype.scale

        # Check actual values: 10.50/2.00=5.25, 20.75/2.00=10.38, 30.25/2.00=15.13
        expected_values = cudf.Series(
            [Decimal("5.25"), Decimal("10.38"), Decimal("15.13")]
        )
        result_series = cudf.Series._from_column(result)

        assert_eq(result_series, expected_values, check_dtype=False)

    @pytest.mark.parametrize(
        "dtype_class",
        [cudf.Decimal32Dtype, cudf.Decimal64Dtype, cudf.Decimal128Dtype],
    )
    @pytest.mark.parametrize(
        "rounding_mode, expected",
        [
            ("HALF_UP", ["0.63", "-0.63", "0.68", "-0.68", "2.28"]),
            ("HALF_EVEN", ["0.62", "-0.62", "0.68", "-0.68", "2.28"]),
        ],
    )
    def test_different_decimal_types(
        self, dtype_class, rounding_mode, expected
    ):
        """Test with Decimal32, Decimal64, and Decimal128"""
        s1 = cudf.Series(
            [
                Decimal("1.25"),
                Decimal("-1.25"),
                Decimal("1.35"),
                Decimal("-1.35"),
                Decimal("4.56"),
            ],
            dtype=dtype_class(5, 2),
        )
        s2 = cudf.Series([Decimal("2.0")] * 5, dtype=dtype_class(3, 1))

        result = s1._column.divide_decimal(
            s2._column, rounding_mode=rounding_mode
        )

        assert isinstance(result.dtype, dtype_class)
        assert result.dtype.scale == 2
        assert result.dtype.precision == 6
        assert_eq(
            cudf.Series._from_column(result),
            cudf.Series([Decimal(v) for v in expected], dtype=result.dtype),
        )

    def test_divide_by_zero_is_null(self):
        """Division by zero gives null rows"""
        s1 = cudf.Series(
            [Decimal("1.23"), Decimal("4.56"), Decimal("0.00")],
            dtype=cudf.Decimal64Dtype(5, 2),
        )
        s2 = cudf.Series(
            [Decimal("0.0"), Decimal("2.0"), Decimal("0.0")],
            dtype=cudf.Decimal64Dtype(3, 1),
        )

        result = cudf.Series._from_column(
            s1._column.divide_decimal(s2._column)
        )

        assert result.isnull().to_pandas().tolist() == [True, False, True]
        assert result[1] == Decimal("2.28")

        by_zero_scalar = cudf.Series._from_column(s1._column.divide_decimal(0))
        assert by_zero_scalar.isnull().all()

    @pytest.mark.parametrize(
        "divisor, expected",
        [
            (2, ["0.62", "2.28"]),
            (Decimal("2.0"), ["0.62", "2.28"]),
            (Decimal("0.004"), ["307.50", "1140.00"]),
            (Decimal("-3"), ["-0.41", "-1.52"]),
        ],
    )
    def test_scalar_divisors(self, divisor, expected):
        """Scalar divisors may have any scale"""
        s = cudf.Series(
            [Decimal("1.23"), Decimal("4.56")], dtype=cudf.Decimal64Dtype(5, 2)
        )

        result = s._column.divide_decimal(divisor)

        assert result.dtype.scale == 2
        assert_eq(
            cudf.Series._from_column(result),
            cudf.Series([Decimal(v) for v in expected], dtype=result.dtype),
        )

    def test_none_divisor(self):
        """A None divisor gives an all-null result"""
        s = cudf.Series(
            [Decimal("1.23"), Decimal("4.56")], dtype=cudf.Decimal64Dtype(5, 2)
        )

        result = cudf.Series._from_column(s._column.divide_decimal(None))

        assert result.isnull().all()
        assert result.dtype == cudf.Decimal64Dtype(5, 2)

    def test_mixed_width_columns(self):
        """A divisor of a different decimal width is cast to the dividend's"""
        s1 = cudf.Series(
            [Decimal("1.23"), Decimal("4.56")],
            dtype=cudf.Decimal64Dtype(10, 2),
        )
        s2 = cudf.Series(
            [Decimal("2.0"), Decimal("3.0")], dtype=cudf.Decimal32Dtype(3, 1)
        )

        result = s1._column.divide_decimal(s2._column)

        assert isinstance(result.dtype, cudf.Decimal64Dtype)
        assert_eq(
            cudf.Series._from_column(result),
            cudf.Series(
                [Decimal("0.62"), Decimal("1.52")], dtype=result.dtype
            ),
        )

    def test_divisor_too_wide(self):
        """A divisor whose precision does not fit the dividend's width"""
        s1 = cudf.Series([Decimal("1.23")], dtype=cudf.Decimal32Dtype(5, 2))
        s2 = cudf.Series([Decimal("2.0")], dtype=cudf.Decimal128Dtype(20, 1))

        with pytest.raises(TypeError, match="precision exceeds"):
            s1._column.divide_decimal(s2._column)

    @pytest.mark.parametrize(
        "divisor",
        [
            1.5,
            True,
            "2",
            cudf.Series([1, 2])._column,
            cudf.Series([1.0, 2.0])._column,
        ],
        ids=["float", "bool", "str", "int_column", "float_column"],
    )
    def test_unsupported_divisor(self, divisor):
        """Unsupported divisors raise TypeError"""
        s = cudf.Series(
            [Decimal("1.23"), Decimal("4.56")], dtype=cudf.Decimal64Dtype(5, 2)
        )

        with pytest.raises(TypeError, match="divide_decimal only supports"):
            s._column.divide_decimal(divisor)

    def test_null_values(self):
        """Test handling of null values in decimal division"""
        s1 = cudf.Series([Decimal("1.23"), None, Decimal("4.56")])
        s2 = cudf.Series([Decimal("2.0"), Decimal("3.0"), Decimal("4.0")])

        result = s1._column.divide_decimal(s2._column)

        # Result should have null at index 1
        assert result.isnull().sum() == 1

    def test_invalid_rounding_mode(self):
        """Test that invalid rounding mode raises error"""
        s1 = cudf.Series([Decimal("1.23")])
        s2 = cudf.Series([Decimal("2.0")])

        with pytest.raises(ValueError):
            s1._column.divide_decimal(s2._column, rounding_mode="INVALID")

    def test_non_decimal_type_error(self):
        """Test that non-decimal types raise TypeError"""
        s1 = cudf.Series([1.23, 4.56])  # Float series, not decimal
        s2 = cudf.Series([2.0, 3.0])

        # This should fail as the columns are not decimal type
        with pytest.raises(AttributeError):
            s1._column.divide_decimal(s2._column)


class TestDecimalDivisionIntegration:
    """Integration tests with other cuDF operations"""

    def test_chained_operations(self):
        """Test divide_decimal in a chain of operations"""
        s = cudf.Series([Decimal("10.50"), Decimal("20.75"), Decimal("30.25")])

        # Chain operations
        result = s._column.divide_decimal(Decimal("2.0"))

        # Verify the result
        expected_values = cudf.Series(
            [Decimal("5.25"), Decimal("10.38"), Decimal("15.13")]
        )
        result_series = cudf.Series._from_column(result)

        assert_eq(result_series, expected_values, check_dtype=False)

    def test_mixed_operations(self):
        """Test mixing standard division with divide_decimal"""
        s1 = cudf.Series([Decimal("10.0"), Decimal("20.0")])
        s2 = cudf.Series([Decimal("3.0"), Decimal("7.0")])

        # Standard division
        std_result = s1 / s2

        # Decimal division
        dec_result = s1._column.divide_decimal(s2._column)

        # Scales should be different
        assert std_result._column.dtype.scale != dec_result.dtype.scale


# Parameterized tests
@pytest.mark.parametrize("precision", [1, 2, 3, 4])
def test_various_scales(precision):
    """Test divide_decimal with various scale values"""
    # Create series with specific scale based on precision
    # Using string formatting to control decimal places
    format_str = f"{{:.{precision}f}}"
    values = [
        Decimal(format_str.format(1.5)),
        Decimal(format_str.format(2.5)),
        Decimal(format_str.format(3.5)),
    ]
    s1 = cudf.Series(values)
    s2 = cudf.Series([Decimal("2.0")] * 3)

    result = s1._column.divide_decimal(s2._column)

    # Result should preserve s1's scale and have correct values
    assert result.dtype.scale == s1._column.dtype.scale

    # Basic verification: 1.5/2.0=0.75, 2.5/2.0=1.25, 3.5/2.0=1.75
    # With HALF_UP rounding and different precisions:
    if precision == 1:
        # 0.75 -> 0.8, 1.25 -> 1.3, 1.75 -> 1.8 (HALF_UP)
        expected_values = cudf.Series(
            [Decimal("0.8"), Decimal("1.3"), Decimal("1.8")]
        )
    elif precision == 2:
        # 0.75 -> 0.75, 1.25 -> 1.25, 1.75 -> 1.75
        expected_values = cudf.Series(
            [Decimal("0.75"), Decimal("1.25"), Decimal("1.75")]
        )
    elif precision == 3:
        # 0.750, 1.250, 1.750
        expected_values = cudf.Series(
            [Decimal("0.750"), Decimal("1.250"), Decimal("1.750")]
        )
    else:  # precision == 4
        # 0.7500, 1.2500, 1.7500
        expected_values = cudf.Series(
            [Decimal("0.7500"), Decimal("1.2500"), Decimal("1.7500")]
        )

    result_series = cudf.Series._from_column(result)

    assert_eq(result_series, expected_values, check_dtype=False)


if __name__ == "__main__":
    pytest.main([__file__])

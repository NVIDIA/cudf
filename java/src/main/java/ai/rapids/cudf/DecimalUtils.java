/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.math.BigDecimal;
import java.util.AbstractMap;
import java.util.Map;

/**
 * Utility methods for creating decimal types and comparing decimal columns.
 */
public class DecimalUtils {

  /**
   * Creates a cuDF decimal type with precision and scale
   */
  public static DType createDecimalType(int precision, int scale) {
    if (precision <= DType.DECIMAL32_MAX_PRECISION) {
      return DType.create(DType.DTypeEnum.DECIMAL32, -scale);
    } else if (precision <= DType.DECIMAL64_MAX_PRECISION) {
      return DType.create(DType.DTypeEnum.DECIMAL64, -scale);
    } else if (precision <= DType.DECIMAL128_MAX_PRECISION) {
      return DType.create(DType.DTypeEnum.DECIMAL128, -scale);
    }
    throw new IllegalArgumentException("precision overflow: " + precision);
  }

  /**
   * Given decimal precision and scale, returns the lower and upper bound of current decimal type.
   *
   * Be very careful when comparing these CUDF decimal comparisons really only work
   * when both types are already the same precision and scale, and when you change the scale
   * you end up losing information.
   * @param precision the max precision of decimal type
   * @param scale the scale of decimal type
   * @return a Map Entry of BigDecimal, lower bound as the key, upper bound as the value
   */
  public static Map.Entry<BigDecimal, BigDecimal> bounds(int precision, int scale) {
    StringBuilder sb = new StringBuilder();
    for (int i = 0; i < precision; i++) sb.append("9");
    sb.append("e");
    sb.append(-scale);
    String boundStr = sb.toString();
    BigDecimal upperBound = new BigDecimal(boundStr);
    BigDecimal lowerBound = new BigDecimal("-" + boundStr);
    return new AbstractMap.SimpleImmutableEntry<>(lowerBound, upperBound);
  }

  /**
   * With precision and scale, checks each value of input decimal column for out of bound.
   * @return the boolean column represents whether specific values are out of bound or not
   */
  public static ColumnVector outOfBounds(ColumnView input, int precision, int scale) {
    Map.Entry<BigDecimal, BigDecimal> boundPair = bounds(precision, scale);
    BigDecimal lowerBound = boundPair.getKey();
    BigDecimal upperBound = boundPair.getValue();
    try (ColumnVector over = greaterThan(input, upperBound);
         ColumnVector under = lessThan(input, lowerBound)) {
      return over.or(under);
    }
  }

  /**
   * Because the native lessThan operator has issues with comparing decimal values that have different
   * precision and scale accurately. This method takes some special steps to get rid of these issues.
   */
  public static ColumnVector lessThan(ColumnView lhs, BigDecimal rhs) {
    assert (lhs.getType().isDecimalType());
    int leftScale = lhs.getType().getScale();
    int leftPrecision = lhs.getType().getDecimalMaxPrecision();

    // First we have to round the scalar (rhs) to the same scale as lhs.
    // For comparing the two values they should be the same scale, we round the value to positive infinity to maintain
    // the relation. Ex:
    // 10.2 < 10.29 = true, after rounding rhs to ceiling ===> 10.2 < 10.3 = true, relation is maintained
    // 10.3 < 10.29 = false, after rounding rhs to ceiling ===> 10.3 < 10.3 = false, relation is maintained
    // 10.1 < 10.10 = false, after rounding rhs to ceiling ===> 10.1 < 10.1 = false, relation is maintained
    BigDecimal roundedRhs = rhs.setScale(-leftScale, BigDecimal.ROUND_CEILING);

    if (roundedRhs.precision() > leftPrecision) {
      // converting rhs to the same precision as lhs would result in an overflow/error, but
      // the scale is the same so we can still figure this out. For example if LHS precision is
      // 4 and RHS precision is 5 we get the following...
      //  9999 <  99999 => true
      // -9999 <  99999 => true
      //  9999 < -99999 => false
      // -9999 < -99999 => false
      // so the result should be the same as RHS > 0
      try (Scalar isPositive = Scalar.fromBool(roundedRhs.compareTo(BigDecimal.ZERO) > 0)) {
        return ColumnVector.fromScalar(isPositive, (int) lhs.getRowCount());
      }
    }
    try (Scalar scalarRhs = Scalar.fromDecimal(roundedRhs.unscaledValue(), lhs.getType())) {
      return lhs.lessThan(scalarRhs);
    }
  }

  /**
   * Because the native lessThan operator has issues with comparing decimal values that have different
   * precision and scale accurately. This method takes some special steps to get rid of these issues.
   */
  public static ColumnVector lessThan(BinaryOperable lhs, BigDecimal rhs, int numRows) {
    if (lhs instanceof ColumnView) {
      return lessThan((ColumnView) lhs, rhs);
    }
    Scalar scalarLhs = (Scalar) lhs;
    if (scalarLhs.isValid()) {
      try (Scalar isLess = Scalar.fromBool(scalarLhs.getBigDecimal().compareTo(rhs) < 0)) {
        return ColumnVector.fromScalar(isLess, numRows);
      }
    }
    try (Scalar nullScalar = Scalar.fromNull(DType.BOOL8)) {
      return ColumnVector.fromScalar(nullScalar, numRows);
    }
  }

  /**
   * Because the native greaterThan operator has issues with comparing decimal values that have different
   * precision and scale accurately. This method takes some special steps to get rid of these issues.
   */
  public static ColumnVector greaterThan(ColumnView lhs, BigDecimal rhs) {
    assert (lhs.getType().isDecimalType());
    int cvScale = lhs.getType().getScale();
    int maxPrecision = lhs.getType().getDecimalMaxPrecision();

    // First we have to round the scalar (rhs) to the same scale as lhs.
    // For comparing the two values they should be the same scale, we round the value to negative infinity to maintain
    // the relation. Ex:
    // 10.3 > 10.29 = true, after rounding rhs to floor ===> 10.3 > 10.2 = true, relation is maintained
    // 10.2 > 10.29 = false, after rounding rhs to floor ===> 10.2 > 10.2 = false, relation is maintained
    // 10.1 > 10.10 = false, after rounding rhs to floor ===> 10.1 > 10.1 = false, relation is maintained
    BigDecimal roundedRhs = rhs.setScale(-cvScale, BigDecimal.ROUND_FLOOR);

    if (roundedRhs.precision() > maxPrecision) {
      // converting rhs to the same precision as lhs would result in an overflow/error, but
      // the scale is the same so we can still figure this out. For example if LHS precision is
      // 4 and RHS precision is 5 we get the following...
      //  9999 >  99999 => false
      // -9999 >  99999 => false
      //  9999 > -99999 => true
      // -9999 > -99999 => true
      // so the result should be the same as RHS < 0
      try (Scalar isNegative = Scalar.fromBool(roundedRhs.compareTo(BigDecimal.ZERO) < 0)) {
        return ColumnVector.fromScalar(isNegative, (int) lhs.getRowCount());
      }
    }
    try (Scalar scalarRhs = Scalar.fromDecimal(roundedRhs.unscaledValue(), lhs.getType())) {
      return lhs.greaterThan(scalarRhs);
    }
  }
}

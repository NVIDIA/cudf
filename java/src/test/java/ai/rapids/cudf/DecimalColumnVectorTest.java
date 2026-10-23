/*
 *  SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.HostColumnVector.Builder;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;
import java.util.function.Consumer;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.*;

public class DecimalColumnVectorTest extends CudfTestBase {
  private static final Random rdSeed = new Random(1234);
  private static final int dec32Scale = 4;
  private static final int dec64Scale = 10;
  private static final int dec128Scale = 30;

  private static final BigDecimal[] decimal32Zoo = new BigDecimal[20];
  private static final BigDecimal[] decimal64Zoo = new BigDecimal[20];
  private static final BigDecimal[] decimal128Zoo = new BigDecimal[20];
  private static final int[] unscaledDec32Zoo = new int[decimal32Zoo.length];
  private static final long[] unscaledDec64Zoo = new long[decimal64Zoo.length];
  private static final BigInteger[] unscaledDec128Zoo = new BigInteger[decimal128Zoo.length];

  private final BigDecimal[] boundaryDecimal32 = new BigDecimal[]{
      new BigDecimal("999999999"), new BigDecimal("-999999999")};

  private final BigDecimal[] boundaryDecimal64 = new BigDecimal[]{
      new BigDecimal("999999999999999999"), new BigDecimal("-999999999999999999")};

  private final BigDecimal[] boundaryDecimal128 = new BigDecimal[]{
      new BigDecimal("99999999999999999999999999999999999999"), new BigDecimal("-99999999999999999999999999999999999999")};

  private final BigDecimal[] overflowDecimal32 = new BigDecimal[]{
      BigDecimal.valueOf(Integer.MAX_VALUE), BigDecimal.valueOf(Integer.MIN_VALUE)};

  private final BigDecimal[] overflowDecimal64 = new BigDecimal[]{
      BigDecimal.valueOf(Long.MAX_VALUE), BigDecimal.valueOf(Long.MIN_VALUE)};

  private final BigDecimal[] overflowDecimal128 = new BigDecimal[]{
      new BigDecimal("340282367000000000000000000000000000001"),
      new BigDecimal("-340282367000000000000000000000000000001")};

  @BeforeAll
  public static void setup() {
    for (int i = 0; i < decimal32Zoo.length; i++) {
      unscaledDec32Zoo[i] = rdSeed.nextInt() / 100;
      unscaledDec64Zoo[i] = rdSeed.nextLong() / 100;
      unscaledDec128Zoo[i] = BigInteger.valueOf(rdSeed.nextLong()).multiply(BigInteger.valueOf(rdSeed.nextLong()));
      if (rdSeed.nextBoolean()) {
        // Create BigDecimal with slight variance on scale, in order to test building cv from inputs with different scales.
        decimal32Zoo[i] = BigDecimal.valueOf(rdSeed.nextInt() / 100, dec32Scale - rdSeed.nextInt(2));
      } else {
        decimal32Zoo[i] = null;
      }
      if (rdSeed.nextBoolean()) {
        // Create BigDecimal with slight variance on scale, in order to test building cv from inputs with different scales.
        decimal64Zoo[i] = BigDecimal.valueOf(rdSeed.nextLong() / 100, dec64Scale - rdSeed.nextInt(2));
      } else {
        decimal64Zoo[i] = null;
      }
      if (rdSeed.nextBoolean()) {
        BigInteger unscaledVal = BigInteger.valueOf(rdSeed.nextLong()).multiply(BigInteger.valueOf(rdSeed.nextLong()));
        decimal128Zoo[i] = new BigDecimal(unscaledVal, dec128Scale);
      } else {
        decimal128Zoo[i] = null;
      }
    }
  }

  @Test
  public void testCreateColumnVectorBuilder() {
    try (ColumnVector cv = ColumnVector.build(DType.create(DType.DTypeEnum.DECIMAL32, -5), 3,
        (b) -> b.append(BigDecimal.valueOf(123456789, 5)))) {
      assertFalse(cv.hasNulls());
    }
    try (ColumnVector cv = ColumnVector.build(DType.create(DType.DTypeEnum.DECIMAL64, -10), 3,
        (b) -> b.append(BigDecimal.valueOf(1023040506070809L, 10)))) {
      assertFalse(cv.hasNulls());
    }
    // test building ColumnVector from BigDecimal values with varying scales
    try (ColumnVector cv = ColumnVector.build(DType.create(DType.DTypeEnum.DECIMAL64, -5), 7,
        (b) -> b.append(BigDecimal.valueOf(123456, 0), RoundingMode.UNNECESSARY)
            .append(BigDecimal.valueOf(123456, 2), RoundingMode.UNNECESSARY)
            .append(BigDecimal.valueOf(123456, 5))
            .append(BigDecimal.valueOf(123456, 7), RoundingMode.HALF_UP)
            .append(BigDecimal.valueOf(123456, 7), RoundingMode.FLOOR)
            .append(BigDecimal.valueOf(123456, 9), RoundingMode.HALF_DOWN)
            .append(BigDecimal.valueOf(123456, 9), RoundingMode.CEILING))) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        assertEquals(12345600000L, hcv.getLong(0));
        assertEquals(123456000L, hcv.getLong(1));
        assertEquals(123456L, hcv.getLong(2));
        assertEquals(1235L, hcv.getLong(3));
        assertEquals(1234L, hcv.getLong(4));
        assertEquals(12L, hcv.getLong(5));
        assertEquals(13L, hcv.getLong(6));
      }
    }
  }

  @Test
  public void testUpperIndexOutOfBoundsException() {
    try (HostColumnVector decColumnVector = HostColumnVector.fromDecimals(decimal32Zoo)) {
      assertThrows(AssertionError.class, () -> decColumnVector.getBigDecimal(decimal32Zoo.length));
    }
  }

  @Test
  public void testLowerIndexOutOfBoundsException() {
    try (HostColumnVector doubleColumnVector = HostColumnVector.fromDecimals(decimal32Zoo)) {
      assertThrows(AssertionError.class, () -> doubleColumnVector.getBigDecimal(-1));
    }
  }

  @Test
  public void testAddingNullValues() {
    try (HostColumnVector cv = HostColumnVector.fromDecimals(decimal64Zoo)) {
      for (int i = 0; i < decimal64Zoo.length; ++i) {
        assertEquals(decimal64Zoo[i] == null, cv.isNull(i));
      }
      assertEquals(Arrays.stream(decimal64Zoo).filter(Objects::isNull).count(), cv.getNullCount());
    }
  }

  @Test
  public void testOverrunningTheBuffer() {
    try (Builder builder = HostColumnVector.builder(DType.create(DType.DTypeEnum.DECIMAL32, -dec32Scale), 3)) {
      assertThrows(AssertionError.class, () -> builder.appendBoxed(decimal32Zoo).build());
    }
    try (Builder builder = HostColumnVector.builder(DType.create(DType.DTypeEnum.DECIMAL64, -dec64Scale), 3)) {
      assertThrows(AssertionError.class, () -> builder.appendUnscaledDecimalArray(unscaledDec64Zoo).build());
    }
  }

  @Test
  public void testDecimalValidation() {
    // precision overflow
    assertThrows(IllegalArgumentException.class, () -> HostColumnVector.fromDecimals(overflowDecimal128));

    assertThrows(IllegalArgumentException.class, () -> {
      try (ColumnVector ignored = ColumnVector.decimalFromInts(
          -(DType.DECIMAL32_MAX_PRECISION + 1), unscaledDec32Zoo)) {
      }
    });
    assertThrows(IllegalArgumentException.class, () -> {
      try (ColumnVector ignored = ColumnVector.decimalFromLongs(
          -(DType.DECIMAL64_MAX_PRECISION + 1), unscaledDec64Zoo)) {
      }
    });
    // precision overflow due to rescaling by min scale
    assertThrows(IllegalArgumentException.class, () -> {
      try (ColumnVector ignored = ColumnVector.fromDecimals(
          BigDecimal.valueOf(1.23e30), BigDecimal.valueOf(1.2e-7))) {
      }
    });
    // exactly hit the MAX_PRECISION_DECIMAL128 after rescaling
    assertDoesNotThrow(() -> {
      try (ColumnVector ignored = ColumnVector.fromDecimals(
          BigDecimal.valueOf(1.23e30), BigDecimal.valueOf(1.2e-6))) {
      }
    });
  }

  @Test
  public void testDecimalGeneral() {
    // Safe max precision of Decimal32 is 9, so integers have 10 digits will be backed by DECIMAL64.
    try (ColumnVector cv = ColumnVector.fromDecimals(overflowDecimal32)) {
      assertEquals(DType.create(DType.DTypeEnum.DECIMAL64, 0), cv.getType());
    }

    try (ColumnVector cv = ColumnVector.fromDecimals(overflowDecimal64)) {
      assertEquals(DType.create(DType.DTypeEnum.DECIMAL128, 0), cv.getType());
    }
    // Create DECIMAL64 vector with small values
    try (ColumnVector cv =  ColumnVector.decimalFromLongs(0, 0L)) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        assertTrue(hcv.getType().isBackedByLong());
        assertEquals(0L, hcv.getBigDecimal(0).longValue());
      }
    }
  }

  @Test
  public void testDecimalFromDecimals() {
    DecimalColumnVectorTest.testDecimalImpl(DType.DTypeEnum.DECIMAL32, dec32Scale, decimal32Zoo);
    DecimalColumnVectorTest.testDecimalImpl(DType.DTypeEnum.DECIMAL64, dec64Scale, decimal64Zoo);
    DecimalColumnVectorTest.testDecimalImpl(DType.DTypeEnum.DECIMAL128, dec128Scale, decimal128Zoo);
    DecimalColumnVectorTest.testDecimalImpl(DType.DTypeEnum.DECIMAL32, 0, boundaryDecimal32);
    DecimalColumnVectorTest.testDecimalImpl(DType.DTypeEnum.DECIMAL64, 0, boundaryDecimal64);
    DecimalColumnVectorTest.testDecimalImpl(DType.DTypeEnum.DECIMAL128, 0, boundaryDecimal128);
  }

  private static void testDecimalImpl(DType.DTypeEnum decimalType, int scale, BigDecimal[] decimalZoo) {
    Consumer<HostColumnVector> assertions = (hcv) -> {
      assertEquals(-scale, hcv.getType().getScale());
      assertEquals(hcv.getType().typeId, decimalType);
      assertEquals(decimalZoo.length, hcv.rows);
      for (int i = 0; i < decimalZoo.length; i++) {
        assertEquals(decimalZoo[i] == null, hcv.isNull(i));
        if (decimalZoo[i] != null) {
          BigDecimal actual;
          switch (decimalType) {
          case DECIMAL32:
            actual = BigDecimal.valueOf(hcv.getInt(i), scale);
            break;
          case DECIMAL64:
            actual = BigDecimal.valueOf(hcv.getLong(i), scale);
            break;
          default:
            actual = hcv.getBigDecimal(i);
          }
          assertEquals(decimalZoo[i].subtract(actual).longValueExact(), 0L);
        }
      }
    };
    try (ColumnVector cv = ColumnVector.fromDecimals(decimalZoo)) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        assertions.accept(hcv);
      }
    }
    try (HostColumnVector hcv = ColumnBuilderHelper.fromDecimals(decimalZoo)) {
      assertions.accept(hcv);
    }
  }

  @Test
  public void testDecimalFromInts() {
    try (ColumnVector cv = ColumnVector.decimalFromInts(-DecimalColumnVectorTest.dec32Scale, DecimalColumnVectorTest.unscaledDec32Zoo)) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        for (int i = 0; i < DecimalColumnVectorTest.unscaledDec32Zoo.length; i++) {
          assertEquals(DecimalColumnVectorTest.unscaledDec32Zoo[i], hcv.getInt(i));
          assertEquals(BigDecimal.valueOf(DecimalColumnVectorTest.unscaledDec32Zoo[i], DecimalColumnVectorTest.dec32Scale), hcv.getBigDecimal(i));
        }
      }
    }
  }

  @Test
  public void testDecimalFromLongs() {
    try (ColumnVector cv = ColumnVector.decimalFromLongs(-DecimalColumnVectorTest.dec64Scale, DecimalColumnVectorTest.unscaledDec64Zoo)) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        for (int i = 0; i < DecimalColumnVectorTest.unscaledDec64Zoo.length; i++) {
          assertEquals(DecimalColumnVectorTest.unscaledDec64Zoo[i], hcv.getLong(i));
          assertEquals(BigDecimal.valueOf(DecimalColumnVectorTest.unscaledDec64Zoo[i], DecimalColumnVectorTest.dec64Scale), hcv.getBigDecimal(i));
        }
      }
    }
  }

  @Test
  public void testDecimalFromBigInts() {
    try (ColumnVector cv = ColumnVector.decimalFromBigInt(-DecimalColumnVectorTest.dec128Scale, DecimalColumnVectorTest.unscaledDec128Zoo)) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        for (int i = 0; i < DecimalColumnVectorTest.unscaledDec128Zoo.length; i++) {
          assertEquals(DecimalColumnVectorTest.unscaledDec128Zoo[i], hcv.getBigDecimal(i).unscaledValue());
        }
      }
    }
    try (HostColumnVector hcv = ColumnBuilderHelper.decimalFromBigInts(-DecimalColumnVectorTest.dec128Scale, DecimalColumnVectorTest.unscaledDec128Zoo)) {
      for (int i = 0; i < DecimalColumnVectorTest.unscaledDec128Zoo.length; i++) {
        assertEquals(DecimalColumnVectorTest.unscaledDec128Zoo[i], hcv.getBigDecimal(i).unscaledValue());
      }
    }
  }

  @Test
  public void testDecimalFromDoubles() {
    DType dt = DType.create(DType.DTypeEnum.DECIMAL32, -3);
    try (ColumnVector cv = ColumnVector.decimalFromDoubles(dt, RoundingMode.DOWN,123456, -2.4567, 3.00001, -1111e-5)) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        assertEquals(123456, hcv.getBigDecimal(0).doubleValue());
        assertEquals(-2.456, hcv.getBigDecimal(1).doubleValue());
        assertEquals(3, hcv.getBigDecimal(2).doubleValue());
        assertEquals(-0.011, hcv.getBigDecimal(3).doubleValue());
      }
    }
    dt = DType.create(DType.DTypeEnum.DECIMAL64, -10);
    try (ColumnVector cv = ColumnVector.decimalFromDoubles(dt, RoundingMode.HALF_UP, 1.2345678, -2.45e-9, 3.000012, -51111e-15)) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        assertEquals(1.2345678, hcv.getBigDecimal(0).doubleValue());
        assertEquals(-2.5e-9, hcv.getBigDecimal(1).doubleValue());
        assertEquals(3.000012, hcv.getBigDecimal(2).doubleValue());
        assertEquals(-1e-10, hcv.getBigDecimal(3).doubleValue());
      }
    }
    dt = DType.create(DType.DTypeEnum.DECIMAL64, 10);
    try (ColumnVector cv = ColumnVector.decimalFromDoubles(dt, RoundingMode.UP, 1.234e20, -12.34e8, 1.1e10)) {
      try (HostColumnVector hcv = cv.copyToHost()) {
        assertEquals(1.234e20, hcv.getBigDecimal(0).doubleValue());
        assertEquals(-1e10, hcv.getBigDecimal(1).doubleValue());
        assertEquals(2e10, hcv.getBigDecimal(2).doubleValue());
      }
    }
    assertThrows(ArithmeticException.class,
        () -> {
          final DType dt1 = DType.create(DType.DTypeEnum.DECIMAL32, -5);
          try (ColumnVector cv = ColumnVector.decimalFromDoubles(dt1, RoundingMode.UNNECESSARY, 30000)) {
          }
        });
    assertThrows(ArithmeticException.class,
        () -> {
          final DType dt1 = DType.create(DType.DTypeEnum.DECIMAL64, 10);
          try (ColumnVector cv = ColumnVector.decimalFromDoubles(dt1, RoundingMode.FLOOR, 1e100)) {
          }
        });
  }

  @Test
  public void testAppendVector() {
    for (DType decType : new DType[]{
        DType.create(DType.DTypeEnum.DECIMAL32, -6),
        DType.create(DType.DTypeEnum.DECIMAL64, -10)}) {
      for (int dstSize = 1; dstSize <= 100; dstSize++) {
        for (int dstPrefilledSize = 0; dstPrefilledSize < dstSize; dstPrefilledSize++) {
          final int srcSize = dstSize - dstPrefilledSize;
          for (int sizeOfDataNotToAdd = 0; sizeOfDataNotToAdd <= dstPrefilledSize; sizeOfDataNotToAdd++) {
            try (Builder dst = HostColumnVector.builder(decType, dstSize);
                 HostColumnVector src = HostColumnVector.build(decType, srcSize, (b) -> {
                   for (int i = 0; i < srcSize; i++) {
                     if (rdSeed.nextBoolean()) {
                       b.appendNull();
                     } else {
                       b.append(BigDecimal.valueOf(rdSeed.nextInt() / 100, -decType.getScale()));
                     }
                   }
                 });
                 Builder gtBuilder = HostColumnVector.builder(decType, dstPrefilledSize)) {
              assertEquals(dstSize, srcSize + dstPrefilledSize);
              //add the first half of the prefilled list
              for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd; i++) {
                if (rdSeed.nextBoolean()) {
                  dst.appendNull();
                  gtBuilder.appendNull();
                } else {
                  BigDecimal a = BigDecimal.valueOf(rdSeed.nextInt() / 100, -decType.getScale());
                  if (decType.typeId == DType.DTypeEnum.DECIMAL32) {
                    dst.appendUnscaledDecimal(a.unscaledValue().intValueExact());
                  } else {
                    dst.appendUnscaledDecimal(a.unscaledValue().longValueExact());
                  }
                  gtBuilder.append(a);
                }
              }
              // append the src vector
              dst.append(src);
              try (HostColumnVector dstVector = dst.build();
                   HostColumnVector gt = gtBuilder.build()) {
                for (int i = 0; i < dstPrefilledSize - sizeOfDataNotToAdd; i++) {
                  assertEquals(gt.isNull(i), dstVector.isNull(i));
                  if (!gt.isNull(i)) {
                    assertEquals(gt.getBigDecimal(i), dstVector.getBigDecimal(i));
                  }
                }
                for (int i = dstPrefilledSize - sizeOfDataNotToAdd, j = 0; i < dstSize - sizeOfDataNotToAdd && j < srcSize; i++, j++) {
                  assertEquals(src.isNull(j), dstVector.isNull(i));
                  if (!src.isNull(j)) {
                    assertEquals(src.getBigDecimal(j), dstVector.getBigDecimal(i));
                  }
                }
                if (dstVector.hasValidityVector()) {
                  long maxIndex =
                      BitVectorHelper.getValidityAllocationSizeInBytes(dstVector.getRowCount()) * 8;
                  for (long i = dstSize - sizeOfDataNotToAdd; i < maxIndex; i++) {
                    assertFalse(dstVector.isNullExtendedRange(i));
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  @Test
  public void testColumnVectorFromScalar() {
    try (Scalar s = Scalar.fromDecimal(-3, 1233456)) {
      try (ColumnVector cv = ColumnVector.fromScalar(s, 10)) {
        assertEquals(s.getType(), cv.getType());
        assertEquals(10L, cv.getRowCount());
        try (HostColumnVector hcv = cv.copyToHost()) {
          for (int i = 0; i < cv.getRowCount(); i++) {
            assertEquals(s.getInt(), hcv.getInt(i));
            assertEquals(s.getBigDecimal(), hcv.getBigDecimal(i));
          }
        }
      }
    }
    try (Scalar s = Scalar.fromDecimal(-6, 123456789098L)) {
      try (ColumnVector cv = ColumnVector.fromScalar(s, 10)) {
        assertEquals(s.getType(), cv.getType());
        assertEquals(10L, cv.getRowCount());
        try (HostColumnVector hcv = cv.copyToHost()) {
          for (int i = 0; i < cv.getRowCount(); i++) {
            assertEquals(s.getLong(), hcv.getLong(i));
            assertEquals(s.getBigDecimal(), hcv.getBigDecimal(i));
          }
        }
      }
    }
  }

  @Test
  public void testDivideDecimalColumnColumn() {
    DType outType = DType.create(DType.DTypeEnum.DECIMAL32, -1);
    // 2.5/2, 3.5/2, -2.5/2, 1.0/3, null/1, 7.0/0
    try (ColumnVector lhs = ColumnVector.decimalFromBoxedInts(-1, 25, 35, -25, 10, null, 70);
         ColumnVector rhs = ColumnVector.decimalFromBoxedInts(0, 2, 2, 2, 3, 1, 0);
         ColumnVector halfUp = lhs.divideDecimal(rhs, outType, RoundMode.HALF_UP);
         ColumnVector halfEven = lhs.divideDecimal(rhs, outType, RoundMode.HALF_EVEN);
         ColumnVector byDefault = lhs.divideDecimal(rhs, outType);
         ColumnVector expectedHalfUp =
             ColumnVector.decimalFromBoxedInts(-1, 13, 18, -13, 3, null, null);
         ColumnVector expectedHalfEven =
             ColumnVector.decimalFromBoxedInts(-1, 12, 18, -12, 3, null, null)) {
      assertColumnsAreEqual(expectedHalfUp, halfUp);
      assertColumnsAreEqual(expectedHalfEven, halfEven);
      assertColumnsAreEqual(expectedHalfUp, byDefault);
    }
  }

  @Test
  public void testDivideDecimalWiderOutputType() {
    DType outType = DType.create(DType.DTypeEnum.DECIMAL128, -6);
    // 1.00/3.00, 2.00/3.00, -1.00/7.00
    try (ColumnVector lhs = ColumnVector.decimalFromLongs(-2, 100, 200, -100);
         ColumnVector rhs = ColumnVector.decimalFromLongs(-2, 300, 300, 700);
         ColumnVector result = lhs.divideDecimal(rhs, outType);
         ColumnVector expected = ColumnVector.decimalFromBigInt(-6,
             BigInteger.valueOf(333333), BigInteger.valueOf(666667),
             BigInteger.valueOf(-142857))) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  public void testDivideDecimalColumnScalar() {
    DType outType = DType.create(DType.DTypeEnum.DECIMAL64, -2);
    // 10.00/4.0, 10.50/4.0, null/4.0, -3.33/4.0
    try (ColumnVector lhs = ColumnVector.decimalFromBoxedLongs(-2, 1000L, 1050L, null, -333L);
         Scalar rhs = Scalar.fromDecimal(-1, 40L);
         Scalar nullRhs = Scalar.fromNull(DType.create(DType.DTypeEnum.DECIMAL64, -1));
         ColumnVector halfUp = lhs.divideDecimal(rhs, outType);
         ColumnVector halfEven = lhs.divideDecimal(rhs, outType, RoundMode.HALF_EVEN);
         ColumnVector allNull = lhs.divideDecimal(nullRhs, outType);
         ColumnVector expectedHalfUp =
             ColumnVector.decimalFromBoxedLongs(-2, 250L, 263L, null, -83L);
         ColumnVector expectedHalfEven =
             ColumnVector.decimalFromBoxedLongs(-2, 250L, 262L, null, -83L);
         ColumnVector expectedAllNull =
             ColumnVector.decimalFromBoxedLongs(-2, null, null, null, null)) {
      assertColumnsAreEqual(expectedHalfUp, halfUp);
      assertColumnsAreEqual(expectedHalfEven, halfEven);
      assertColumnsAreEqual(expectedAllNull, allNull);
    }
  }

  @Test
  public void testDivideDecimalScalarColumn() {
    DType outType = DType.create(DType.DTypeEnum.DECIMAL64, -3);
    // 1/3, 1/6, 1/0, 1/null, 1/-8
    try (Scalar lhs = Scalar.fromDecimal(0, 1L);
         Scalar nullLhs = Scalar.fromNull(DType.create(DType.DTypeEnum.DECIMAL64, 0));
         ColumnVector rhs = ColumnVector.decimalFromBoxedInts(0, 3, 6, 0, null, -8);
         ColumnVector result = lhs.divideDecimal(rhs, outType, RoundMode.HALF_UP);
         ColumnVector allNull = nullLhs.divideDecimal(rhs, outType);
         ColumnVector expected =
             ColumnVector.decimalFromBoxedLongs(-3, 333L, 167L, null, null, -125L);
         ColumnVector expectedAllNull =
             ColumnVector.decimalFromBoxedLongs(-3, null, null, null, null, null)) {
      assertColumnsAreEqual(expected, result);
      assertColumnsAreEqual(expectedAllNull, allNull);
    }
  }

  @Test
  public void testDivideDecimalWideIntermediate() {
    DType outType = DType.create(DType.DTypeEnum.DECIMAL128, -20);
    BigInteger ten10 = BigInteger.TEN.pow(10);
    BigInteger ten20 = BigInteger.TEN.pow(20);
    // 1.0000000000/1.00000000000000000000, -1.0000000000/1.00000000000000000000,
    // 3.0000000000/-1.00000000000000000000, at scale -20, rescale the dividends to 10^40
    try (ColumnVector lhs = ColumnVector.decimalFromBigInt(-10, ten10, ten10.negate(),
             ten10.multiply(BigInteger.valueOf(3)));
         ColumnVector rhs = ColumnVector.decimalFromBigInt(-20, ten20, ten20, ten20.negate());
         Scalar lhsScalar = Scalar.fromDecimal(-10, ten10);
         Scalar rhsScalar = Scalar.fromDecimal(-20, ten20);
         ColumnVector byDefault = lhs.divideDecimal(rhs, outType);
         ColumnVector standard = lhs.divideDecimal(rhs, outType, RoundMode.HALF_UP,
             DecimalIntermediate.STANDARD);
         ColumnVector standardVS = lhs.divideDecimal(rhsScalar, outType, RoundMode.HALF_UP,
             DecimalIntermediate.STANDARD);
         ColumnVector standardSV = lhsScalar.divideDecimal(rhs, outType, RoundMode.HALF_UP,
             DecimalIntermediate.STANDARD);
         ColumnVector wide = lhs.divideDecimal(rhs, outType, RoundMode.HALF_UP,
             DecimalIntermediate.WIDE);
         ColumnVector wideVS = lhs.divideDecimal(rhsScalar, outType, RoundMode.HALF_EVEN,
             DecimalIntermediate.WIDE);
         ColumnVector wideSV = lhsScalar.divideDecimal(rhs, outType, RoundMode.HALF_EVEN,
             DecimalIntermediate.WIDE);
         ColumnVector expectedNull = ColumnVector.decimalFromBigInt(-20, null, null, null);
         ColumnVector expectedWide = ColumnVector.decimalFromBigInt(-20, ten20, ten20.negate(),
             ten20.multiply(BigInteger.valueOf(-3)));
         ColumnVector expectedWideVS = ColumnVector.decimalFromBigInt(-20, ten20, ten20.negate(),
             ten20.multiply(BigInteger.valueOf(3)));
         ColumnVector expectedWideSV = ColumnVector.decimalFromBigInt(-20, ten20, ten20,
             ten20.negate())) {
      assertColumnsAreEqual(expectedNull, byDefault);
      assertColumnsAreEqual(expectedNull, standard);
      assertColumnsAreEqual(expectedNull, standardVS);
      assertColumnsAreEqual(expectedNull, standardSV);
      assertColumnsAreEqual(expectedWide, wide);
      assertColumnsAreEqual(expectedWideVS, wideVS);
      assertColumnsAreEqual(expectedWideSV, wideSV);
    }
  }

  @Test
  public void testDivideDecimalInvalidTypes() {
    DType decimalType = DType.create(DType.DTypeEnum.DECIMAL32, 0);
    try (ColumnVector ints = ColumnVector.fromInts(1, 2);
         ColumnVector decimals = ColumnVector.decimalFromInts(0, 1, 2)) {
      assertThrows(CudfException.class, () -> ints.divideDecimal(decimals, decimalType));
      assertThrows(CudfException.class, () -> decimals.divideDecimal(ints, decimalType));
      assertThrows(CudfException.class, () -> decimals.divideDecimal(decimals, DType.INT32));
    }
  }
}

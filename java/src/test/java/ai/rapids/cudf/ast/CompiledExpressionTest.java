/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.CudfTestBase;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.params.provider.NullSource;

import java.math.BigInteger;
import java.math.RoundingMode;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class CompiledExpressionTest extends CudfTestBase {
  @Test
  public void testColumnReferenceTransform() {
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build()) {
      // use an implicit table reference
      ColumnReference expr = new ColumnReference(1);
      try (CompiledExpression compiledExpr = expr.compile();
           ColumnVector actual = compiledExpr.computeColumn(t)) {
        assertColumnsAreEqual(t.getColumn(1), actual);
      }

      // use an explicit table reference
      expr = new ColumnReference(1, TableReference.LEFT);
      try (CompiledExpression compiledExpr = expr.compile();
           ColumnVector actual = compiledExpr.computeColumn(t)) {
        assertColumnsAreEqual(t.getColumn(1), actual);
      }
    }
  }

  @Test
  public void testInvalidColumnReferenceTransform() {
    // Verify that computeColumn throws when passed an expression operating on TableReference.RIGHT.
    ColumnReference expr = new ColumnReference(1, TableReference.RIGHT);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile()) {
      Assertions.assertThrows(CudfException.class, () -> compiledExpr.computeColumn(t).close());
      Assertions.assertThrows(CudfException.class, () -> compiledExpr.computeColumnJit(t).close());
    }
  }

  @Test
  public void testBooleanLiteralTransform() {
    try (Table t = new Table.TestBuilder().column(true, false, null).build()) {
      Literal expr = Literal.ofBoolean(true);
      try (CompiledExpression trueCompiledExpr = expr.compile();
           ColumnVector trueExprActual = trueCompiledExpr.computeColumn(t);
           ColumnVector trueExprExpected = ColumnVector.fromBoxedBooleans(true, true, true)) {
        assertColumnsAreEqual(trueExprExpected, trueExprActual);
      }

      Literal nullLiteral = Literal.ofBoolean(null);
      UnaryOperation nullExpr = new UnaryOperation(UnaryOperator.IDENTITY, nullLiteral);
      try (CompiledExpression nullCompiledExpr = nullExpr.compile();
           ColumnVector nullExprActual = nullCompiledExpr.computeColumn(t);
           ColumnVector nullExprExpected = ColumnVector.fromBoxedBooleans(null, null, null)) {
        assertColumnsAreEqual(nullExprExpected, nullExprActual);
      }
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(bytes = 0x12)
  public void testByteLiteralTransform(Byte value) {
    Literal expr = Literal.ofByte(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBytes(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(shorts = 0x1234)
  public void testShortLiteralTransform(Short value) {
    Literal expr = Literal.ofShort(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedShorts(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(ints = 0x12345678)
  public void testIntLiteralTransform(Integer value) {
    Literal expr = Literal.ofInt(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedInts(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testLongLiteralTransform(Long value) {
    Literal expr = Literal.ofLong(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(floats = { 123456.789f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY} )
  public void testFloatLiteralTransform(Float value) {
    Literal expr = Literal.ofFloat(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedFloats(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(doubles = { 123456.789f, Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY} )
  public void testDoubleLiteralTransform(Double value) {
    Literal expr = Literal.ofDouble(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(ints = 0x12345678)
  public void testTimestampDaysLiteralTransform(Integer value) {
    Literal expr = Literal.ofTimestampDaysFromInt(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.timestampDaysFromBoxedInts(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testTimestampSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofTimestampFromLong(DType.TIMESTAMP_SECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.timestampSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testTimestampMilliSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofTimestampFromLong(DType.TIMESTAMP_MILLISECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.timestampMilliSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testTimestampMicroSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofTimestampFromLong(DType.TIMESTAMP_MICROSECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.timestampMicroSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testTimestampNanoSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofTimestampFromLong(DType.TIMESTAMP_NANOSECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.timestampNanoSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(ints = 0x12345678)
  public void testDurationDaysLiteralTransform(Integer value) {
    Literal expr = Literal.ofDurationDaysFromInt(value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.durationDaysFromBoxedInts(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testDurationSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofDurationFromLong(DType.DURATION_SECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.durationSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testDurationMilliSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofDurationFromLong(DType.DURATION_MILLISECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.durationMilliSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testDurationMicroSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofDurationFromLong(DType.DURATION_MICROSECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.durationMicroSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @NullSource
  @ValueSource(longs = 0x1234567890abcdefL)
  public void testDurationNanoSecondsLiteralTransform(Long value) {
    Literal expr = Literal.ofDurationFromLong(DType.DURATION_NANOSECONDS, value);
    try (Table t = new Table.TestBuilder().column(5, 4, 3, 2, 1).column(6, 7, 8, null, 10).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected =
             ColumnVector.durationNanoSecondsFromBoxedLongs(value, value, value, value, value)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createLegacyDecimalLiteralParams() {
    return Stream.of(
        Arguments.of(DType.create(DType.DTypeEnum.DECIMAL32, -2),
            new BigInteger("1234567")),
        Arguments.of(DType.create(DType.DTypeEnum.DECIMAL32, 128), BigInteger.ONE),
        Arguments.of(DType.create(DType.DTypeEnum.DECIMAL32, 0), (BigInteger) null),
        Arguments.of(DType.create(DType.DTypeEnum.DECIMAL64, -4),
            new BigInteger("-123456789012345678")),
        Arguments.of(DType.create(DType.DTypeEnum.DECIMAL64, -18), (BigInteger) null));
  }

  private static Stream<Arguments> createDecimal128LiteralParams() {
    return Stream.of(
        Arguments.of(DType.create(DType.DTypeEnum.DECIMAL128, -4),
            new BigInteger("-123456789012345678901234567890")),
        Arguments.of(DType.create(DType.DTypeEnum.DECIMAL128, -38), BigInteger.ONE),
        Arguments.of(DType.create(DType.DTypeEnum.DECIMAL128, 0),
            BigInteger.ONE.shiftLeft(127).subtract(BigInteger.ONE)),
        Arguments.of(DType.create(DType.DTypeEnum.DECIMAL128, 0),
            BigInteger.ONE.shiftLeft(127).negate()),
        Arguments.of(DType.create(DType.DTypeEnum.DECIMAL128, -10), (BigInteger) null));
  }

  private static Stream<Arguments> createDecimalLiteralParams() {
    return Stream.concat(createLegacyDecimalLiteralParams(), createDecimal128LiteralParams());
  }

  @ParameterizedTest
  @MethodSource("createLegacyDecimalLiteralParams")
  public void testDecimalLiteralTransform(DType type, BigInteger value) {
    Literal expr = Literal.ofDecimal(type, value);
    try (Table t = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         Scalar expectedScalar = value == null ?
             Scalar.fromNull(type) : Scalar.fromDecimal(value, type);
         ColumnVector expected = ColumnVector.fromScalar(expectedScalar, 3)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @MethodSource("createDecimal128LiteralParams")
  public void testDecimal128LiteralLegacyTransformFails(DType type, BigInteger value) {
    Literal expr = Literal.ofDecimal(type, value);
    try (Table t = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiledExpr = expr.compile()) {
      Assertions.assertThrows(CudfException.class, () -> compiledExpr.computeColumn(t).close());
    }
  }

  @ParameterizedTest
  @MethodSource("createDecimal128LiteralParams")
  public void testDecimal128IdentityLegacyTransformFails(DType type, BigInteger value) {
    UnaryOperation expr = new UnaryOperation(
        UnaryOperator.IDENTITY, Literal.ofDecimal(type, value));
    try (Table t = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiledExpr = expr.compile()) {
      Assertions.assertThrows(CudfException.class, () -> compiledExpr.computeColumn(t).close());
    }
  }

  @Test
  public void testDecimal128LiteralComparisonLegacyTransform() {
    DType type = DType.create(DType.DTypeEnum.DECIMAL128, 0);
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0), Literal.ofDecimal(type, BigInteger.ONE));
    try (Table t = new Table.TestBuilder()
             .decimal128Column(0, RoundingMode.UNNECESSARY,
                 BigInteger.ZERO, BigInteger.ONE, BigInteger.valueOf(2), null)
             .build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(false, false, true, null)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @ParameterizedTest
  @MethodSource("createDecimalLiteralParams")
  public void testJitDecimalLiteralTransform(DType type, BigInteger value) {
    Literal expr = Literal.ofDecimal(type, value);
    try (Table t = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumnJit(t);
         Scalar expectedScalar = value == null ?
             Scalar.fromNull(type) : Scalar.fromDecimal(value, type);
         ColumnVector expected = ColumnVector.fromScalar(expectedScalar, 3)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  public void testDecimalLiteralValidation() {
    Assertions.assertThrows(IllegalArgumentException.class,
        () -> Literal.ofDecimal(DType.INT32, BigInteger.ONE));
    Assertions.assertThrows(ArithmeticException.class,
        () -> Literal.ofDecimal(DType.create(DType.DTypeEnum.DECIMAL32, 0),
            BigInteger.ONE.shiftLeft(31)));
    Assertions.assertThrows(ArithmeticException.class,
        () -> Literal.ofDecimal(DType.create(DType.DTypeEnum.DECIMAL64, 0),
            BigInteger.ONE.shiftLeft(63)));

    DType decimal128 = DType.create(DType.DTypeEnum.DECIMAL128, 0);
    Assertions.assertThrows(ArithmeticException.class,
        () -> Literal.ofDecimal(decimal128, BigInteger.ONE.shiftLeft(127)));
    Assertions.assertThrows(ArithmeticException.class,
        () -> Literal.ofDecimal(decimal128,
            BigInteger.ONE.shiftLeft(127).negate().subtract(BigInteger.ONE)));
  }

  @Test
  public void testDecimal128LiteralByteOrderConversion() {
    DType type = DType.create(DType.DTypeEnum.DECIMAL128, 0);
    byte[] positiveBigEndian = new byte[type.getSizeInBytes()];
    positiveBigEndian[positiveBigEndian.length - 1] = 1;
    byte[] positiveLittleEndian = new byte[type.getSizeInBytes()];
    positiveLittleEndian[0] = 1;
    Assertions.assertArrayEquals(positiveBigEndian,
        Literal.convertDecimal128FromJavaToCudf(
            BigInteger.ONE.toByteArray(), type, ByteOrder.BIG_ENDIAN));
    Assertions.assertArrayEquals(positiveLittleEndian,
        Literal.convertDecimal128FromJavaToCudf(
            BigInteger.ONE.toByteArray(), type, ByteOrder.LITTLE_ENDIAN));

    byte[] negativeBigEndian = new byte[type.getSizeInBytes()];
    Arrays.fill(negativeBigEndian, (byte) 0xff);
    negativeBigEndian[negativeBigEndian.length - 1] = (byte) 0xfe;
    byte[] negativeLittleEndian = new byte[type.getSizeInBytes()];
    Arrays.fill(negativeLittleEndian, (byte) 0xff);
    negativeLittleEndian[0] = (byte) 0xfe;
    byte[] negativeValue = BigInteger.valueOf(-2).toByteArray();
    Assertions.assertArrayEquals(negativeBigEndian,
        Literal.convertDecimal128FromJavaToCudf(
            negativeValue, type, ByteOrder.BIG_ENDIAN));
    Assertions.assertArrayEquals(negativeLittleEndian,
        Literal.convertDecimal128FromJavaToCudf(
            negativeValue, type, ByteOrder.LITTLE_ENDIAN));
  }

  @Test
  void testJitOperationValidation() {
    assertJitCompileThrows(new JitOperation(JitOperator.ADD, new ColumnReference(0)));
    assertJitCompileThrows(new JitOperation(JitOperator.ADD,
        new ColumnReference(0), new ColumnReference(1), new ColumnReference(2)));
    assertJitCompileThrows(new JitOperation(JitOperator.ADD, JitErrorPolicy.NULLIFY,
        new ColumnReference(0), new ColumnReference(1)));
    assertJitCompileThrows(new JitOperation(JitOperator.ADD, -2,
        new ColumnReference(0), new ColumnReference(1)));
    assertJitCompileThrows(new JitOperation(JitOperator.RESCALE, new ColumnReference(0)));

    Assertions.assertThrows(
        NullPointerException.class,
        () -> new JitOperation(null, new ColumnReference(0)));
    Assertions.assertThrows(
        NullPointerException.class,
        () -> new JitOperation(JitOperator.ADD, (JitErrorPolicy) null,
            new ColumnReference(0), new ColumnReference(1)));
    NullPointerException nullInputError = Assertions.assertThrows(
        NullPointerException.class,
        () -> new JitOperation(JitOperator.ADD,
            new ColumnReference(0), (AstExpression) null));
    Assertions.assertEquals("input 1 is null", nullInputError.getMessage());
    Assertions.assertThrows(
        NullPointerException.class,
        () -> new JitOperation(JitOperator.ADD, (AstExpression[]) null));
  }

  private static void assertJitCompileThrows(JitOperation expr) {
    Assertions.assertThrows(CudfException.class, () -> {
      try (CompiledExpression ignored = expr.compile()) {
      }
    });
  }

  @Test
  void testJitMismatchedOperandTypes() {
    JitOperation expr = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(1).column(2L).build();
         CompiledExpression compiledExpr = expr.compile()) {
      Assertions.assertThrows(CudfException.class,
          () -> compiledExpr.computeColumnJit(t).close());
    }
  }

  @Test
  void testJitEmptyInputTransform() {
    JitOperation expr = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), Literal.ofInt(1));
    try (Table t = new Table.TestBuilder().column(new Integer[0]).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumnJit(t);
         ColumnVector expected = ColumnVector.fromInts()) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testJitNestedArithmeticTransform() {
    AstExpression expr = new JitOperation(JitOperator.ADD, new ColumnReference(0), Literal.ofInt(2));
    expr = new JitOperation(JitOperator.SUB, expr, Literal.ofInt(1));
    expr = new JitOperation(JitOperator.MUL, expr, Literal.ofInt(3));
    expr = new JitOperation(JitOperator.DIV, expr, Literal.ofInt(2));
    expr = new JitOperation(JitOperator.MOD, expr, Literal.ofInt(5));
    expr = new JitOperation(JitOperator.NEG, expr);
    expr = new JitOperation(JitOperator.ABS, expr);
    expr = new JitOperation(JitOperator.BITWISE_SHIFT_LEFT, expr, Literal.ofInt(2));
    expr = new JitOperation(JitOperator.BITWISE_SHIFT_RIGHT, expr, Literal.ofInt(1));

    try (Table t = new Table.TestBuilder().column(1, 2, 3, 4).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumnJit(t);
         ColumnVector expected = ColumnVector.fromInts(6, 8, 2, 4)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testJitNegTransform() {
    JitOperation expr = new JitOperation(JitOperator.NEG, new ColumnReference(0));
    try (Table t = new Table.TestBuilder().column(-5, 0, 7).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumnJit(t);
         ColumnVector expected = ColumnVector.fromInts(5, 0, -7)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testJitOverflowPolicies() {
    try (Table t = new Table.TestBuilder()
        .column(1, 3)
        .column(10, 7)
        .column(10, Integer.MAX_VALUE)
        .build()) {
      JitOperation successExpr = new JitOperation(JitOperator.ADD_OVERFLOW,
          new ColumnReference(0), new ColumnReference(1));
      try (CompiledExpression compiledExpr = successExpr.compile();
           ColumnVector actual = compiledExpr.computeColumnJit(t);
           ColumnVector expected = ColumnVector.fromInts(11, 10)) {
        assertColumnsAreEqual(expected, actual);
      }

      JitOperation propagateExpr = new JitOperation(JitOperator.ADD_OVERFLOW,
          new ColumnReference(0), new ColumnReference(2));
      try (CompiledExpression compiledExpr = propagateExpr.compile()) {
        Assertions.assertThrows(CudfException.class,
            () -> compiledExpr.computeColumnJit(t).close());
      }

      JitOperation nullifyExpr = new JitOperation(JitOperator.ADD_OVERFLOW,
          JitErrorPolicy.NULLIFY,
          new ColumnReference(0), new ColumnReference(2));
      try (CompiledExpression compiledExpr = nullifyExpr.compile();
           ColumnVector actual = compiledExpr.computeColumnJit(t);
           ColumnVector expected = ColumnVector.fromBoxedInts(11, null)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void testJitFusedNullifyingOverflowTransform() {
    try (Table t = new Table.TestBuilder()
        .column(1, 3, 20, 1, 50, 10)
        .column(1, 10, 7, 20, Integer.MAX_VALUE, 2)
        .column(1, 5, 4, Integer.MAX_VALUE, 2, 5)
        .column(0, 1, 0, 0, 1, 5)
        .build()) {
      AstExpression expr = new JitOperation(JitOperator.ADD_OVERFLOW, JitErrorPolicy.NULLIFY,
          new ColumnReference(0), new ColumnReference(1));
      expr = new JitOperation(JitOperator.MUL_OVERFLOW, JitErrorPolicy.NULLIFY,
          expr, new ColumnReference(2));
      expr = new JitOperation(JitOperator.DIV_OVERFLOW, JitErrorPolicy.NULLIFY,
          expr, new ColumnReference(3));
      try (CompiledExpression compiledExpr = expr.compile();
           ColumnVector actual = compiledExpr.computeColumnJit(t);
           ColumnVector expected = ColumnVector.fromBoxedInts(null, 65, null, null, null, 12)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void testJitUnaryAndSubtractOverflowTransform() {
    try (Table t = new Table.TestBuilder()
        .column(10, Integer.MIN_VALUE, 1)
        .column(3, 1, 0)
        .build()) {
      JitOperation subExpr = new JitOperation(JitOperator.SUB_OVERFLOW,
          JitErrorPolicy.NULLIFY, new ColumnReference(0), new ColumnReference(1));
      try (CompiledExpression compiledExpr = subExpr.compile();
           ColumnVector actual = compiledExpr.computeColumnJit(t);
           ColumnVector expected = ColumnVector.fromBoxedInts(7, null, 1)) {
        assertColumnsAreEqual(expected, actual);
      }

      JitOperation negExpr = new JitOperation(JitOperator.NEG_OVERFLOW,
          JitErrorPolicy.NULLIFY, new ColumnReference(0));
      try (CompiledExpression compiledExpr = negExpr.compile();
           ColumnVector actual = compiledExpr.computeColumnJit(t);
           ColumnVector expected = ColumnVector.fromBoxedInts(-10, null, -1)) {
        assertColumnsAreEqual(expected, actual);
      }

      JitOperation absExpr = new JitOperation(JitOperator.ABS_OVERFLOW,
          JitErrorPolicy.NULLIFY, new ColumnReference(0));
      try (CompiledExpression compiledExpr = absExpr.compile();
           ColumnVector actual = compiledExpr.computeColumnJit(t);
           ColumnVector expected = ColumnVector.fromBoxedInts(10, null, 1)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void testJitTryDivModTransform() {
    try (Table t = new Table.TestBuilder()
        .column(10, 7, null, 6, Integer.MIN_VALUE)
        .column(2, 0, 3, null, -1)
        .build()) {
      JitOperation divExpr = new JitOperation(JitOperator.DIV_OVERFLOW,
          JitErrorPolicy.NULLIFY, new ColumnReference(0), new ColumnReference(1));
      try (CompiledExpression compiledExpr = divExpr.compile();
           ColumnVector actual = compiledExpr.computeColumnJit(t);
           ColumnVector expected = ColumnVector.fromBoxedInts(5, null, null, null, null)) {
        assertColumnsAreEqual(expected, actual);
      }

      JitOperation modExpr = new JitOperation(JitOperator.MOD_OVERFLOW,
          JitErrorPolicy.NULLIFY, new ColumnReference(0), new ColumnReference(1));
      try (CompiledExpression compiledExpr = modExpr.compile();
           ColumnVector actual = compiledExpr.computeColumnJit(t);
           ColumnVector expected = ColumnVector.fromBoxedInts(0, null, null, null, 0)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  @Test
  void testJitMixedConditionalTransform() {
    try (Table t = new Table.TestBuilder()
        .column(1, null, 3, null)
        .column(10, 20, 30, 40)
        .build()) {
      AstExpression condition = new BinaryOperation(BinaryOperator.GREATER,
          new ColumnReference(0), Literal.ofInt(2));
      AstExpression predicate = new JitOperation(JitOperator.PREDICATE, condition);
      AstExpression coalesced = new JitOperation(JitOperator.COALESCE,
          new ColumnReference(0), Literal.ofInt(99));
      JitOperation expr = new JitOperation(JitOperator.IF_ELSE,
          coalesced, new ColumnReference(1), predicate);
      try (CompiledExpression compiledExpr = expr.compile();
           ColumnVector actual = compiledExpr.computeColumnJit(t);
           ColumnVector expected = ColumnVector.fromBoxedInts(10, 20, 3, 40)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  private static Arguments jitCastCase(
      JitOperator op, Supplier<ColumnVector> expectedFactory) {
    return Arguments.of(op, expectedFactory);
  }

  private static Stream<Arguments> createJitNumericCastParams() {
    return Stream.of(
        jitCastCase(JitOperator.CAST_TO_BOOL8,
            () -> ColumnVector.fromBooleans(false, true, true, true)),
        jitCastCase(JitOperator.CAST_TO_INT8,
            () -> ColumnVector.fromBytes((byte) 0, (byte) 1, (byte) 2, (byte) 3)),
        jitCastCase(JitOperator.CAST_TO_INT16,
            () -> ColumnVector.fromShorts((short) 0, (short) 1, (short) 2, (short) 3)),
        jitCastCase(JitOperator.CAST_TO_INT32,
            () -> ColumnVector.fromInts(0, 1, 2, 3)),
        jitCastCase(JitOperator.CAST_TO_INT64,
            () -> ColumnVector.fromLongs(0L, 1L, 2L, 3L)),
        jitCastCase(JitOperator.CAST_TO_UINT8,
            () -> ColumnVector.fromUnsignedBytes((byte) 0, (byte) 1, (byte) 2, (byte) 3)),
        jitCastCase(JitOperator.CAST_TO_UINT16,
            () -> ColumnVector.fromUnsignedShorts((short) 0, (short) 1, (short) 2, (short) 3)),
        jitCastCase(JitOperator.CAST_TO_UINT32,
            () -> ColumnVector.fromUnsignedInts(0, 1, 2, 3)),
        jitCastCase(JitOperator.CAST_TO_UINT64,
            () -> ColumnVector.fromUnsignedLongs(0L, 1L, 2L, 3L)),
        jitCastCase(JitOperator.CAST_TO_FLOAT32,
            () -> ColumnVector.fromFloats(0.0f, 1.0f, 2.0f, 3.0f)),
        jitCastCase(JitOperator.CAST_TO_FLOAT64,
            () -> ColumnVector.fromDoubles(0.0, 1.0, 2.0, 3.0)));
  }

  @ParameterizedTest
  @MethodSource("createJitNumericCastParams")
  void testJitNumericCastTransform(
      JitOperator op, Supplier<ColumnVector> expectedFactory) {
    JitOperation expr = new JitOperation(op, new ColumnReference(0));
    try (Table t = new Table.TestBuilder().column(0, 1, 2, 3).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumnJit(t);
         ColumnVector expected = expectedFactory.get()) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createJitDecimalCastParams() {
    return Stream.of(
        jitCastCase(JitOperator.CAST_TO_DECIMAL32,
            () -> ColumnVector.decimalFromInts(0, 0, 1, -2, 3)),
        jitCastCase(JitOperator.CAST_TO_DECIMAL64,
            () -> ColumnVector.decimalFromLongs(0, 0L, 1L, -2L, 3L)),
        jitCastCase(JitOperator.CAST_TO_DECIMAL128,
            () -> ColumnVector.decimalFromBigInt(0,
                BigInteger.ZERO, BigInteger.ONE, BigInteger.valueOf(-2), BigInteger.valueOf(3))));
  }

  @ParameterizedTest
  @MethodSource("createJitDecimalCastParams")
  void testJitDecimalCastTransform(
      JitOperator op, Supplier<ColumnVector> expectedFactory) {
    JitOperation expr = new JitOperation(op, new ColumnReference(0));
    try (Table t = new Table.TestBuilder().decimal64Column(0, 0L, 1L, -2L, 3L).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumnJit(t);
         ColumnVector expected = expectedFactory.get()) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testJitDecimalRescaleTransform() {
    JitOperation expr = new JitOperation(JitOperator.RESCALE, -2, new ColumnReference(0));
    try (Table t = new Table.TestBuilder()
        .decimal32Column(0, 123, 1234, 12345, 123456, 1234567)
        .build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumnJit(t);
         ColumnVector expected = ColumnVector.decimalFromInts(
             -2, 12300, 123400, 1234500, 12345600, 123456700)) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testJitDecimalPrecisionPolicies() {
    try (Table t = new Table.TestBuilder().decimal32Column(0, 3, 200, 250, 20000).build()) {
      JitOperation propagateExpr = new JitOperation(JitOperator.CHECK_PRECISION,
          new ColumnReference(0), Literal.ofInt(3));
      try (CompiledExpression compiledExpr = propagateExpr.compile()) {
        Assertions.assertThrows(CudfException.class,
            () -> compiledExpr.computeColumnJit(t).close());
      }

      JitOperation nullifyExpr = new JitOperation(JitOperator.CHECK_PRECISION,
          JitErrorPolicy.NULLIFY, new ColumnReference(0), Literal.ofInt(3));
      try (CompiledExpression compiledExpr = nullifyExpr.compile();
           ColumnVector actual = compiledExpr.computeColumnJit(t);
           ColumnVector expected = ColumnVector.decimalFromBoxedInts(0, 3, 200, 250, null)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  private static <T, R> ArrayList<R> mapArray(T[] input, Function<T, R> func) {
    ArrayList<R> result = new ArrayList<>(input.length);
    for (T t : input) {
      result.add(t == null ? null : func.apply(t));
    }
    return result;
  }

  private static <T, U, R> ArrayList<R> mapArray(T[] in1, U[] in2, BiFunction<T, U, R> func) {
    assert in1.length == in2.length;
    ArrayList<R> result = new ArrayList<>(in1.length);
    for (int i = 0; i < in1.length; i++) {
      result.add(in1[i] == null || in2[i] == null ? null : func.apply(in1[i], in2[i]));
    }
    return result;
  }

  private static Stream<Arguments> createUnaryDoubleOperationParams() {
    Double[] input = new Double[] { -5., 4.5, null, 2.7, 1.5 };
    return Stream.of(
        Arguments.of(UnaryOperator.IDENTITY, input, Arrays.asList(input)),
        Arguments.of(UnaryOperator.SIN, input, mapArray(input, Math::sin)),
        Arguments.of(UnaryOperator.COS, input, mapArray(input, Math::cos)),
        Arguments.of(UnaryOperator.TAN, input, mapArray(input, Math::tan)),
        Arguments.of(UnaryOperator.ARCSIN, input, mapArray(input, Math::asin)),
        Arguments.of(UnaryOperator.ARCCOS, input, mapArray(input, Math::acos)),
        Arguments.of(UnaryOperator.ARCTAN, input, mapArray(input, Math::atan)),
        Arguments.of(UnaryOperator.SINH, input, mapArray(input, Math::sinh)),
        Arguments.of(UnaryOperator.COSH, input, mapArray(input, Math::cosh)),
        Arguments.of(UnaryOperator.TANH, input, mapArray(input, Math::tanh)),
        Arguments.of(UnaryOperator.EXP, input, mapArray(input, Math::exp)),
        Arguments.of(UnaryOperator.LOG, input, mapArray(input, Math::log)),
        Arguments.of(UnaryOperator.SQRT, input, mapArray(input, Math::sqrt)),
        Arguments.of(UnaryOperator.CBRT, input, mapArray(input, Math::cbrt)),
        Arguments.of(UnaryOperator.CEIL, input, mapArray(input, Math::ceil)),
        Arguments.of(UnaryOperator.FLOOR, input, mapArray(input, Math::floor)),
        Arguments.of(UnaryOperator.ABS, input, mapArray(input, Math::abs)),
        Arguments.of(UnaryOperator.RINT, input, mapArray(input, Math::rint)));
  }

  @ParameterizedTest
  @MethodSource("createUnaryDoubleOperationParams")
  void testUnaryDoubleOperationTransform(UnaryOperator op, Double[] input,
                                          List<Double> expectedValues) {
    UnaryOperation expr = new UnaryOperation(op, new ColumnReference(0));
    try (Table t = new Table.TestBuilder().column(input).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(
             expectedValues.toArray(new Double[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testUnaryShortOperationTransform() {
    Short[] input = new Short[] { -5, 4, null, 2, 1 };
    try (Table t = new Table.TestBuilder().column(input).build()) {
      ColumnReference expr = new ColumnReference(0);
      try (CompiledExpression compiledExpr = expr.compile();
           ColumnVector actual = compiledExpr.computeColumn(t)) {
        assertColumnsAreEqual(t.getColumn(0), actual);
      }

      UnaryOperation expr2 = new UnaryOperation(UnaryOperator.BIT_INVERT, new ColumnReference(0));
      try (CompiledExpression compiledExpr = expr2.compile();
           ColumnVector actual = compiledExpr.computeColumn(t);
           ColumnVector expected = ColumnVector.fromBoxedInts(4, -5, null, -3, -2)) {
        assertColumnsAreEqual(expected, actual);
      }
    }
  }

  private static Stream<Arguments> createUnaryLogicalOperationParams() {
    Long[] input = new Long[] { -5L, 0L, null, 2L, 1L };
    return Stream.of(
        Arguments.of(UnaryOperator.NOT, input, Arrays.asList(false, true, null, false, false)),
        Arguments.of(UnaryOperator.IS_NULL, input, Arrays.asList(false, false, true, false, false)));
  }

  @ParameterizedTest
  @MethodSource("createUnaryLogicalOperationParams")
  void testUnaryLogicalOperationTransform(UnaryOperator op, Long[] input,
                                          List<Boolean> expectedValues) {
    UnaryOperation expr = new UnaryOperation(op, new ColumnReference(0));
    try (Table t = new Table.TestBuilder().column(input).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(
             expectedValues.toArray(new Boolean[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createBinaryFloatOperationParams() {
    Float[] in1 = new Float[] { -5f, 4.5f, null, 2.7f };
    Float[] in2 = new Float[] { 123f, -456f, null, 0f };
    return Stream.of(
        Arguments.of(BinaryOperator.ADD, in1, in2, mapArray(in1, in2, Float::sum)),
        Arguments.of(BinaryOperator.SUB, in1, in2, mapArray(in1, in2, (a, b) -> a - b)),
        Arguments.of(BinaryOperator.MUL, in1, in2, mapArray(in1, in2, (a, b) -> a * b)),
        Arguments.of(BinaryOperator.DIV, in1, in2, mapArray(in1, in2, (a, b) -> a / b)),
        Arguments.of(BinaryOperator.MOD, in1, in2, mapArray(in1, in2, (a, b) -> a % b)),
        Arguments.of(BinaryOperator.PYMOD, in1, in2, mapArray(in1, in2,
            (a, b) -> ((a % b) + b) % b)),
        Arguments.of(BinaryOperator.POW, in1, in2, mapArray(in1, in2,
            (a, b) -> (float) Math.pow(a, b))),
        Arguments.of(BinaryOperator.FLOOR_DIV, in1, in2, mapArray(in1, in2,
            (a, b) -> (float) Math.floor(a / b))));
  }

  @ParameterizedTest
  @MethodSource("createBinaryFloatOperationParams")
  void testBinaryFloatOperationTransform(BinaryOperator op, Float[] in1, Float[] in2,
                                          List<Float> expectedValues) {
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(in1).column(in2).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedFloats(
             expectedValues.toArray(new Float[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createBinaryDoublePromotedOperationParams() {
    Float[] in1 = new Float[] { -5f, 4.5f, null, 2.7f };
    Float[] in2 = new Float[] { 123f, -456f, null, 0f };
    return Stream.of(
        Arguments.of(BinaryOperator.TRUE_DIV, in1, in2, mapArray(in1, in2,
            (a, b) -> (double) a / b)));
  }

  @ParameterizedTest
  @MethodSource("createBinaryDoublePromotedOperationParams")
  void testBinaryDoublePromotedOperationTransform(BinaryOperator op, Float[] in1, Float[] in2,
                                                   List<Double> expectedValues) {
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(in1).column(in2).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(
             expectedValues.toArray(new Double[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createBinaryComparisonOperationParams() {
    Integer[] in1 = new Integer[] { -5, 4, null, 2, -3 };
    Integer[] in2 = new Integer[] { 123, -456, null, 0, -3 };
    return Stream.of(
        // nulls compare as equal by default
        Arguments.of(BinaryOperator.NULL_EQUAL, in1, in2, Arrays.asList(false, false, true, false, true)),
        Arguments.of(BinaryOperator.NOT_EQUAL, in1, in2, mapArray(in1, in2, (a, b) -> !a.equals(b))),
        Arguments.of(BinaryOperator.LESS, in1, in2, mapArray(in1, in2, (a, b) -> a < b)),
        Arguments.of(BinaryOperator.GREATER, in1, in2, mapArray(in1, in2, (a, b) -> a > b)),
        Arguments.of(BinaryOperator.LESS_EQUAL, in1, in2, mapArray(in1, in2, (a, b) -> a <= b)),
        Arguments.of(BinaryOperator.GREATER_EQUAL, in1, in2, mapArray(in1, in2, (a, b) -> a >= b)));
  }

  @ParameterizedTest
  @MethodSource("createBinaryComparisonOperationParams")
  void testBinaryComparisonOperationTransform(BinaryOperator op, Integer[] in1, Integer[] in2,
                                               List<Boolean> expectedValues) {
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(in1).column(in2).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(
             expectedValues.toArray(new Boolean[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createStringLiteralComparisonParams() {
    String[] in1 = new String[] {"a", "bb", null, "ccc", "dddd"};
    String in2 = "ccc";
    return Stream.of(
        // nulls compare as equal by default
        Arguments.of(BinaryOperator.NULL_EQUAL, in1, in2, Arrays.asList(false, false, false, true, false)),
        Arguments.of(BinaryOperator.NOT_EQUAL, in1, in2, mapArray(in1, (a) -> !a.equals(in2))),
        Arguments.of(BinaryOperator.LESS, in1, in2, mapArray(in1, (a) -> a.compareTo(in2) < 0)),
        Arguments.of(BinaryOperator.GREATER, in1, in2, mapArray(in1, (a) -> a.compareTo(in2) > 0)),
        Arguments.of(BinaryOperator.LESS_EQUAL, in1, in2, mapArray(in1, (a) -> a.compareTo(in2) <= 0)),
        Arguments.of(BinaryOperator.GREATER_EQUAL, in1, in2, mapArray(in1, (a) -> a.compareTo(in2) >= 0)),
        // null literal
        Arguments.of(BinaryOperator.NULL_EQUAL, in1, null, Arrays.asList(false, false, true, false, false)),
        Arguments.of(BinaryOperator.NOT_EQUAL, in1, null, Arrays.asList(null, null, null, null, null)),
        Arguments.of(BinaryOperator.LESS, in1, null, Arrays.asList(null, null, null, null, null)));
  }

  @ParameterizedTest
  @MethodSource("createStringLiteralComparisonParams")
  void testStringLiteralComparison(BinaryOperator op, String[] in1, String in2,
                                               List<Boolean> expectedValues) {
    Literal lit = Literal.ofString(in2);
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        lit);
    try (Table t = new Table.TestBuilder().column(in1).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(
             expectedValues.toArray(new Boolean[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createBinaryComparisonOperationStringParams() {
    String[] in1 = new String[] {"a", "bb", null, "ccc", "dddd"};
    String[] in2 = new String[] {"aa", "b", null, "ccc", "ddd"};
    return Stream.of(
        // nulls compare as equal by default
        Arguments.of(BinaryOperator.NULL_EQUAL, in1, in2, Arrays.asList(false, false, true, true, false)),
        Arguments.of(BinaryOperator.NOT_EQUAL, in1, in2, mapArray(in1, in2, (a, b) -> !a.equals(b))),
        Arguments.of(BinaryOperator.LESS, in1, in2, mapArray(in1, in2, (a, b) -> a.compareTo(b) < 0)),
        Arguments.of(BinaryOperator.GREATER, in1, in2, mapArray(in1, in2, (a, b) -> a.compareTo(b) > 0)),
        Arguments.of(BinaryOperator.LESS_EQUAL, in1, in2, mapArray(in1, in2, (a, b) -> a.compareTo(b) <= 0)),
        Arguments.of(BinaryOperator.GREATER_EQUAL, in1, in2, mapArray(in1, in2, (a, b) -> a.compareTo(b) >= 0)));
  }

  @ParameterizedTest
  @MethodSource("createBinaryComparisonOperationStringParams")
  void testBinaryComparisonOperationStringTransform(BinaryOperator op, String[] in1, String[] in2,
                                               List<Boolean> expectedValues) {
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(in1).column(in2).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(
             expectedValues.toArray(new Boolean[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createBinaryBitwiseOperationParams() {
    Integer[] in1 = new Integer[] { -5, 4, null, 2, -3 };
    Integer[] in2 = new Integer[] { 123, -456, null, 0, -3 };
    return Stream.of(
        Arguments.of(BinaryOperator.BITWISE_AND, in1, in2, mapArray(in1, in2, (a, b) -> a & b)),
        Arguments.of(BinaryOperator.BITWISE_OR, in1, in2, mapArray(in1, in2, (a, b) -> a | b)),
        Arguments.of(BinaryOperator.BITWISE_XOR, in1, in2, mapArray(in1, in2, (a, b) -> a ^ b)));
  }

  @ParameterizedTest
  @MethodSource("createBinaryBitwiseOperationParams")
  void testBinaryBitwiseOperationTransform(BinaryOperator op, Integer[] in1, Integer[] in2,
                                            List<Integer> expectedValues) {
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(in1).column(in2).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedInts(
             expectedValues.toArray(new Integer[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  private static Stream<Arguments> createBinaryBooleanOperationParams() {
    Boolean[] in1 = new Boolean[] { false, true, false, null, true, false };
    Boolean[] in2 = new Boolean[] { true, null, null, null, true, false };
    return Stream.of(
        Arguments.of(BinaryOperator.LOGICAL_AND, in1, in2, mapArray(in1, in2, (a, b) -> a && b)),
        Arguments.of(BinaryOperator.LOGICAL_OR, in1, in2, mapArray(in1, in2, (a, b) -> a || b)),
        Arguments.of(BinaryOperator.NULL_LOGICAL_AND, in1, in2, Arrays.asList(false, null, false, null, true, false)),
        Arguments.of(BinaryOperator.NULL_LOGICAL_OR, in1, in2, Arrays.asList(true, true, null, null, true, false)));
  }

  @ParameterizedTest
  @MethodSource("createBinaryBooleanOperationParams")
  void testBinaryBooleanOperationTransform(BinaryOperator op, Boolean[] in1, Boolean[] in2,
                                            List<Boolean> expectedValues) {
    BinaryOperation expr = new BinaryOperation(op,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(in1).column(in2).build();
         CompiledExpression compiledExpr = expr.compile();
         ColumnVector actual = compiledExpr.computeColumn(t);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(
             expectedValues.toArray(new Boolean[0]))) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testMismatchedBinaryOperationTypes() {
    // verify expression fails to transform if operands are not the same type
    BinaryOperation expr = new BinaryOperation(BinaryOperator.ADD,
        new ColumnReference(0),
        new ColumnReference(1));
    try (Table t = new Table.TestBuilder().column(1, 2, 3).column(1L, 2L, 3L).build();
         CompiledExpression compiledExpr = expr.compile()) {
      Assertions.assertThrows(CudfException.class, () -> compiledExpr.computeColumn(t).close());
    }
  }

  /**
   * Verifies that computeColumn throws CudfException when the expression contains a
   * ColumnNameReference, since plain table views carry no column-name metadata —
   * name resolution is only supported inside HybridScanReader.
   */
  @Test
  void testColumnNameReferenceThrowsOnComputeColumn() {
    BinaryOperation op = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnNameReference("col"), Literal.ofInt(42));
    try (Table t = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = op.compile()) {
      Assertions.assertThrows(CudfException.class, () -> compiled.computeColumn(t).close());
    }
  }

  /** Verifies that ExpressionType.COLUMN_NAME_REFERENCE has ordinal 5, matching the native plan's expected node-type ID. */
  @Test
  void testColumnNameReferenceTypeOrdinalMatchesPlan() {
    Assertions.assertEquals(5, AstExpression.ExpressionType.COLUMN_NAME_REFERENCE.ordinal());
  }
}

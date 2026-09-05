/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class IfElseTest extends CudfTestBase {
  private static Stream<Arguments> createBooleanVVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Boolean[]{false, false, false, true, true},
            new Boolean[]{true, true, true, false, false},
            new Boolean[]{true, true, false, false, true}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Boolean[]{null, false, false, true, null},
            new Boolean[]{true, null, null, false, null},
            new Boolean[]{true, null, false, false, null})
    );
  }

  private static Stream<Arguments> createBooleanVSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Boolean[]{false, false, false, true, true},
            Boolean.FALSE,
            new Boolean[]{false, false, false, false, true}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Boolean[]{null, false, false, true, null},
            null,
            new Boolean[]{null, null, false, null, null})
    );
  }

  private static Stream<Arguments> createBooleanSVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            Boolean.FALSE,
            new Boolean[]{false, false, false, true, true},
            new Boolean[]{false, false, false, true, false}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            null,
            new Boolean[]{null, false, false, true, null},
            new Boolean[]{null, false, null, true, null})
    );
  }

  private static Stream<Arguments> createBooleanSSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            Boolean.FALSE,
            Boolean.TRUE,
            new Boolean[]{true, true, false, true, false}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            null,
            Boolean.FALSE,
            new Boolean[]{false, false, null, false, null}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            Boolean.FALSE,
            null,
            new Boolean[]{null, null, false, null, false})
    );
  }

  private static Stream<Arguments> createByteVVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Byte[]{(byte) 10, (byte) -128, (byte) 127, (byte) -1, (byte) 0},
            new Byte[]{(byte) -2, (byte) 1, (byte) 16, (byte) -63, (byte) 42},
            new Byte[]{(byte) -2, (byte) 1, (byte) 127, (byte) -63, (byte) 0}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Byte[]{null, (byte) -128, (byte) 127, (byte) -1, null},
            new Byte[]{(byte) -2, null, null, (byte) -63, null},
            new Byte[]{(byte) -2, null, (byte) 127, (byte) -63, null})
        );
  }

  private static Stream<Arguments> createByteVSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Byte[]{(byte) 10, (byte) -128, (byte) 127, (byte) -1, (byte) 0},
            (byte) -2,
            new Byte[]{(byte) -2, (byte) -2, (byte) 127, (byte) -2, (byte) 0}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Byte[]{null, (byte) -128, (byte) 127, (byte) -1, null},
            null,
            new Byte[]{null, null, (byte) 127, null, null})
    );
  }

  private static Stream<Arguments> createByteSVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            (byte) -128,
            new Byte[]{(byte) -2, (byte) 1, (byte) 16, (byte) -63, (byte) 42},
            new Byte[]{(byte) -2, (byte) 1, (byte) -128, (byte) -63, (byte) -128}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            null,
            new Byte[]{null, (byte) 1, (byte) 16, null, (byte) 42},
            new Byte[]{null, (byte) 1, null, null, null})
    );
  }

  private static Stream<Arguments> createByteSSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            (byte) -128,
            (byte) 42,
            new Byte[]{(byte) 42, (byte) 42, (byte) -128, (byte) 42, (byte) -128}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            (byte) -128,
            null,
            new Byte[]{null, null, (byte) -128, null, (byte) -128})
    );
  }

  private static Stream<Arguments> createShortVVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Short[]{(short) 1024, (short) -128, (short) 127, (short) -1, (short) 0},
            new Short[]{(short) -2048, (short) 1, (short) 16, (short) -63, (short) 42},
            new Short[]{(short) -2048, (short) 1, (short) 127, (short) -63, (short) 0}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Short[]{null, (short) -128, (short) 127, (short) -1, null},
            new Short[]{(short) -2048, null, null, (short) -63, null},
            new Short[]{(short) -2048, null, (short) 127, (short) -63, null})
    );
  }

  private static Stream<Arguments> createShortVSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Short[]{(short) 1024, (short) -128, (short) 127, (short) -1, (short) 0},
            (short) -2048,
            new Short[]{(short) -2048, (short) -2048, (short) 127, (short) -2048, (short) 0}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Short[]{null, (short) -128, (short) 127, (short) -1, null},
            null,
            new Short[]{null, null, (short) 127, null, null})
    );
  }

  private static Stream<Arguments> createShortSVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            (short) -1287,
            new Short[]{(short) -2048, (short) 1, (short) 16, (short) -63, (short) 42},
            new Short[]{(short) -2048, (short) 1, (short) -1287, (short) -63, (short) -1287}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            null,
            new Short[]{null, (short) 1, (short) 16, null, (short) 42},
            new Short[]{null, (short) 1, null, null, null})
    );
  }

  private static Stream<Arguments> createShortSSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            (short) -1287,
            (short) 421,
            new Short[]{(short) 421, (short) 421, (short) -1287, (short) 421, (short) -1287}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            (short) -1287,
            null,
            new Short[]{null, null, (short) -1287, null, (short) -1287})
    );
  }

  private static Stream<Arguments> createIntVVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Integer[]{10240, -128, 127, -1, 0},
            new Integer[]{-20480, 1, 16, -63, 42},
            new Integer[]{-20480, 1, 127, -63, 0}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Integer[]{null, -128, 127, -1, null},
            new Integer[]{-20480, null, null, -63, null},
            new Integer[]{-20480, null, 127, -63, null})
    );
  }

  private static Stream<Arguments> createIntVSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Integer[]{10240, -128, 127, -1, 0},
            -20480,
            new Integer[]{-20480, -20480, 127, -20480, 0}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Integer[]{null, -128, 127, -1, null},
            null,
            new Integer[]{null, null, 127, null, null})
    );
  }

  private static Stream<Arguments> createIntSVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            -12875,
            new Integer[]{-2, 1, 16, -63, 42},
            new Integer[]{-2, 1, -12875, -63, -12875}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            null,
            new Integer[]{null, 1, 16, null, 42},
            new Integer[]{null, 1, null, null, null})
    );
  }

  private static Stream<Arguments> createIntSSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            -12875,
            42321,
            new Integer[]{42321, 42321, -12875, 42321, -12875}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            -12875,
            null,
            new Integer[]{null, null, -12875, null, -12875})
    );
  }

  private static Stream<Arguments> createLongVVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Long[]{1024056789L, -128L, 127L, -1L, 0L},
            new Long[]{-2048012345L, 1L, 16L, -63L, 42L},
            new Long[]{-2048012345L, 1L, 127L, -63L, 0L}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Long[]{null, -128L, 127L, -1L, null},
            new Long[]{-2048012345L, null, null, -63L, null},
            new Long[]{-2048012345L, null, 127L, -63L, null})
    );
  }

  private static Stream<Arguments> createLongVSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Long[]{10240L, -128L, 127L, -1L, 0L},
            -2048012345L,
            new Long[]{-2048012345L, -2048012345L, 127L, -2048012345L, 0L}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Long[]{null, -128L, 127L, -1L, null},
            null,
            new Long[]{null, null, 127L, null, null})
    );
  }

  private static Stream<Arguments> createLongSVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            -12875L,
            new Long[]{-2L, 1L, 16L, -63L, 42L},
            new Long[]{-2L, 1L, -12875L, -63L, -12875L}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            null,
            new Long[]{null, 1L, 16L, null, 42L},
            new Long[]{null, 1L, null, null, null})
    );
  }

  private static Stream<Arguments> createLongSSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            -12875L,
            42321L,
            new Long[]{42321L, 42321L, -12875L, 42321L, -12875L}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            -12875L,
            null,
            new Long[]{null, null, -12875L, null, -12875L})
    );
  }

  private static Stream<Arguments> createFloatVVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Float[]{10240.56789f, -128f, 127f, -1f, 0f},
            new Float[]{-20480.12345f, 1f, 16f, -6.3f, 42f},
            new Float[]{-20480.12345f, 1f, 127f, -6.3f, 0f}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Float[]{null, -128f, 127f, -1f, null},
            new Float[]{-20480.12345f, null, null, -6.3f, null},
            new Float[]{-20480.12345f, null, 127f, -6.3f, null})
    );
  }

  private static Stream<Arguments> createFloatVSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Float[]{10240f, -128f, 127f, -1f, 0f},
            -20480.12345f,
            new Float[]{-20480.12345f, -20480.12345f, 127f, -20480.12345f, 0f}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Float[]{null, -128f, 127f, -1f, null},
            null,
            new Float[]{null, null, 127f, null, null})
    );
  }

  private static Stream<Arguments> createFloatSVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            -128.75f,
            new Float[]{-2f, 1f, 16f, -6.3f, 42f},
            new Float[]{-2f, 1f, -128.75f, -6.3f, -128.75f}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            null,
            new Float[]{null, 1f, 16f, null, 42f},
            new Float[]{null, 1f, null, null, null})
    );
  }

  private static Stream<Arguments> createFloatSSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            -128.75f,
            4232.1f,
            new Float[]{4232.1f, 4232.1f, -128.75f, 4232.1f, -128.75f}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            -128.75f,
            null,
            new Float[]{null, null, -128.75f, null, -128.75f})
    );
  }

  private static Stream<Arguments> createDoubleVVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Double[]{10240.56789, -128., 127., -1., 0.},
            new Double[]{-20480.12345, 1., 16., -6.3, 42.},
            new Double[]{-20480.12345, 1., 127., -6.3, 0.}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Double[]{null, -128., 127., -1., null},
            new Double[]{-20480.12345, null, null, -6.3, null},
            new Double[]{-20480.12345, null, 127., -6.3, null})
    );
  }

  private static Stream<Arguments> createDoubleVSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Double[]{10240., -128., 127., -1., 0.},
            -20480.12345,
            new Double[]{-20480.12345, -20480.12345, 127., -20480.12345, 0.}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new Double[]{null, -128., 127., -1., null},
            null,
            new Double[]{null, null, 127., null, null})
    );
  }

  private static Stream<Arguments> createDoubleSVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            -128.75,
            new Double[]{-2., 1., 16., -6.3, 42.},
            new Double[]{-2., 1., -128.75, -6.3, -128.75}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            null,
            new Double[]{null, 1., 16., null, 42.},
            new Double[]{null, 1., null, null, null})
    );
  }

  private static Stream<Arguments> createDoubleSSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            -128.75,
            4232.1,
            new Double[]{4232.1, 4232.1, -128.75, 4232.1, -128.75}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            -128.75,
            null,
            new Double[]{null, null, -128.75, null, -128.75})
    );
  }

  private static Stream<Arguments> createStringVVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new String[]{"hello", "world", "how", "are", "you"},
            new String[]{"why", "fine", "thanks", "for", "asking"},
            new String[]{"why", "fine", "how", "for", "you"}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new String[]{null, "world", "how", "are", null},
            new String[]{"why", null, null, "for", null},
            new String[]{"why", null, "how", "for", null})
    );
  }

  private static Stream<Arguments> createStringVSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new String[]{"hello", "world", "how", "are", "you"},
            "foo",
            new String[]{"foo", "foo", "how", "foo", "you"}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            new String[]{null, "world", "how", "are", null},
            null,
            new String[]{null, null, "how", null, null})
    );
  }

  private static Stream<Arguments> createStringSVParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            "bar",
            new String[]{"why", "fine", "thanks", "for", "asking"},
            new String[]{"why", "fine", "bar", "for", "bar"}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            null,
            new String[]{null, "world", "how", "are", null},
            new String[]{null, "world", null, "are", null})
    );
  }

  private static Stream<Arguments> createStringSSParams() {
    return Stream.of(
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            "hello",
            "world",
            new String[]{"world", "world", "hello", "world", "hello"}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            null,
            "world",
            new String[]{"world", "world", null, "world", null}),
        Arguments.of(
            new Boolean[]{false, false, true, false, true},
            "hello",
            null,
            new String[]{null, null, "hello", null, "hello"})
    );
  }

  @ParameterizedTest
  @MethodSource("createBooleanVVParams")
  void testBooleanVV(Boolean[] predVals, Boolean[] trueVals, Boolean[] falseVals, Boolean[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedBooleans(trueVals);
         ColumnVector falseVec = ColumnVector.fromBoxedBooleans(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createBooleanVSParams")
  void testBooleanVS(Boolean[] predVals, Boolean[] trueVals, Boolean falseVal, Boolean[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedBooleans(trueVals);
         Scalar falseScalar = Scalar.fromBool(falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createBooleanSVParams")
  void testBooleanSV(Boolean[] predVals, Boolean trueVal, Boolean[] falseVals, Boolean[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromBool(trueVal);
         ColumnVector falseVec = ColumnVector.fromBoxedBooleans(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createBooleanSSParams")
  void testBooleanSS(Boolean[] predVals, Boolean trueVal, Boolean falseVal, Boolean[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromBool(trueVal);
         Scalar falseScalar = Scalar.fromBool(falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createByteVVParams")
  void testByteVV(Boolean[] predVals, Byte[] trueVals, Byte[] falseVals, Byte[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedBytes(trueVals);
         ColumnVector falseVec = ColumnVector.fromBoxedBytes(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedBytes(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createByteVSParams")
  void testByteVS(Boolean[] predVals, Byte[] trueVals, Byte falseVal, Byte[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedBytes(trueVals);
         Scalar falseScalar = Scalar.fromByte(falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedBytes(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createByteSVParams")
  void testBytesSV(Boolean[] predVals, Byte trueVal, Byte[] falseVals, Byte[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromByte(trueVal);
         ColumnVector falseVec = ColumnVector.fromBoxedBytes(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedBytes(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createByteSSParams")
  void testBytesSS(Boolean[] predVals, Byte trueVal, Byte falseVal, Byte[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromByte(trueVal);
         Scalar falseScalar = Scalar.fromByte(falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedBytes(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createShortVVParams")
  void testShortVV(Boolean[] predVals, Short[] trueVals, Short[] falseVals, Short[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedShorts(trueVals);
         ColumnVector falseVec = ColumnVector.fromBoxedShorts(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedShorts(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createShortVSParams")
  void testShortVS(Boolean[] predVals, Short[] trueVals, Short falseVal, Short[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedShorts(trueVals);
         Scalar falseScalar = Scalar.fromShort(falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedShorts(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createShortSVParams")
  void testShortsSV(Boolean[] predVals, Short trueVal, Short[] falseVals, Short[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromShort(trueVal);
         ColumnVector falseVec = ColumnVector.fromBoxedShorts(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedShorts(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createShortSSParams")
  void testShortsSS(Boolean[] predVals, Short trueVal, Short falseVal, Short[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromShort(trueVal);
         Scalar falseScalar = Scalar.fromShort(falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedShorts(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntVVParams")
  void testIntVV(Boolean[] predVals, Integer[] trueVals, Integer[] falseVals, Integer[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedInts(trueVals);
         ColumnVector falseVec = ColumnVector.fromBoxedInts(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedInts(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntVSParams")
  void testIntVS(Boolean[] predVals, Integer[] trueVals, Integer falseVal, Integer[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedInts(trueVals);
         Scalar falseScalar = Scalar.fromInt(falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedInts(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntSVParams")
  void testIntsSV(Boolean[] predVals, Integer trueVal, Integer[] falseVals, Integer[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromInt(trueVal);
         ColumnVector falseVec = ColumnVector.fromBoxedInts(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedInts(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntSSParams")
  void testIntsSS(Boolean[] predVals, Integer trueVal, Integer falseVal, Integer[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromInt(trueVal);
         Scalar falseScalar = Scalar.fromInt(falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedInts(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongVVParams")
  void testLongVV(Boolean[] predVals, Long[] trueVals, Long[] falseVals, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedLongs(trueVals);
         ColumnVector falseVec = ColumnVector.fromBoxedLongs(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongVSParams")
  void testLongVS(Boolean[] predVals, Long[] trueVals, Long falseVal, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedLongs(trueVals);
         Scalar falseScalar = Scalar.fromLong(falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongSVParams")
  void testLongsSV(Boolean[] predVals, Long trueVal, Long[] falseVals, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromLong(trueVal);
         ColumnVector falseVec = ColumnVector.fromBoxedLongs(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongSSParams")
  void testLongsSS(Boolean[] predVals, Long trueVal, Long falseVal, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromLong(trueVal);
         Scalar falseScalar = Scalar.fromLong(falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createFloatVVParams")
  void testFloatVV(Boolean[] predVals, Float[] trueVals, Float[] falseVals, Float[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedFloats(trueVals);
         ColumnVector falseVec = ColumnVector.fromBoxedFloats(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedFloats(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createFloatVSParams")
  void testFloatVS(Boolean[] predVals, Float[] trueVals, Float falseVal, Float[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedFloats(trueVals);
         Scalar falseScalar = Scalar.fromFloat(falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedFloats(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createFloatSVParams")
  void testFloatsSV(Boolean[] predVals, Float trueVal, Float[] falseVals, Float[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromFloat(trueVal);
         ColumnVector falseVec = ColumnVector.fromBoxedFloats(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedFloats(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createFloatSSParams")
  void testFloatsSS(Boolean[] predVals, Float trueVal, Float falseVal, Float[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromFloat(trueVal);
         Scalar falseScalar = Scalar.fromFloat(falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedFloats(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createDoubleVVParams")
  void testDoubleVV(Boolean[] predVals, Double[] trueVals, Double[] falseVals, Double[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedDoubles(trueVals);
         ColumnVector falseVec = ColumnVector.fromBoxedDoubles(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createDoubleVSParams")
  void testDoubleVS(Boolean[] predVals, Double[] trueVals, Double falseVal, Double[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromBoxedDoubles(trueVals);
         Scalar falseScalar = Scalar.fromDouble(falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createDoubleSVParams")
  void testDoublesSV(Boolean[] predVals, Double trueVal, Double[] falseVals, Double[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromDouble(trueVal);
         ColumnVector falseVec = ColumnVector.fromBoxedDoubles(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createDoubleSSParams")
  void testDoublesSS(Boolean[] predVals, Double trueVal, Double falseVal, Double[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromDouble(trueVal);
         Scalar falseScalar = Scalar.fromDouble(falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntVVParams")
  void testTimestampDaysVV(Boolean[] predVals, Integer[] trueVals, Integer[] falseVals, Integer[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.timestampDaysFromBoxedInts(trueVals);
         ColumnVector falseVec = ColumnVector.timestampDaysFromBoxedInts(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.timestampDaysFromBoxedInts(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntVSParams")
  void testTimestampDaysVS(Boolean[] predVals, Integer[] trueVals, Integer falseVal, Integer[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.timestampDaysFromBoxedInts(trueVals);
         Scalar falseScalar = Scalar.timestampDaysFromInt(falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.timestampDaysFromBoxedInts(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntSVParams")
  void testTimestampDaysSV(Boolean[] predVals, Integer trueVal, Integer[] falseVals, Integer[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.timestampDaysFromInt(trueVal);
         ColumnVector falseVec = ColumnVector.timestampDaysFromBoxedInts(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.timestampDaysFromBoxedInts(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createIntSSParams")
  void testTimestampDaysSS(Boolean[] predVals, Integer trueVal, Integer falseVal, Integer[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.timestampDaysFromInt(trueVal);
         Scalar falseScalar = Scalar.timestampDaysFromInt(falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.timestampDaysFromBoxedInts(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongVVParams")
  void testTimestampSecondsVV(Boolean[] predVals, Long[] trueVals, Long[] falseVals, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.timestampSecondsFromBoxedLongs(trueVals);
         ColumnVector falseVec = ColumnVector.timestampSecondsFromBoxedLongs(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.timestampSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongVSParams")
  void testTimestampSecondsVS(Boolean[] predVals, Long[] trueVals, Long falseVal, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.timestampSecondsFromBoxedLongs(trueVals);
         Scalar falseScalar = Scalar.timestampFromLong(DType.TIMESTAMP_SECONDS, falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.timestampSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongSVParams")
  void testTimestampSecondsSV(Boolean[] predVals, Long trueVal, Long[] falseVals, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.timestampFromLong(DType.TIMESTAMP_SECONDS, trueVal);
         ColumnVector falseVec = ColumnVector.timestampSecondsFromBoxedLongs(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.timestampSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongSSParams")
  void testTimestampSecondsSS(Boolean[] predVals, Long trueVal, Long falseVal, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.timestampFromLong(DType.TIMESTAMP_SECONDS, trueVal);
         Scalar falseScalar = Scalar.timestampFromLong(DType.TIMESTAMP_SECONDS, falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.timestampSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongVVParams")
  void testTimestampMilliSecondsVV(Boolean[] predVals, Long[] trueVals, Long[] falseVals, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.timestampMilliSecondsFromBoxedLongs(trueVals);
         ColumnVector falseVec = ColumnVector.timestampMilliSecondsFromBoxedLongs(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.timestampMilliSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongVSParams")
  void testTimestampMilliSecondsVS(Boolean[] predVals, Long[] trueVals, Long falseVal, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.timestampMilliSecondsFromBoxedLongs(trueVals);
         Scalar falseScalar = Scalar.timestampFromLong(DType.TIMESTAMP_MILLISECONDS, falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.timestampMilliSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongSVParams")
  void testTimestampMilliSecondsSV(Boolean[] predVals, Long trueVal, Long[] falseVals, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.timestampFromLong(DType.TIMESTAMP_MILLISECONDS, trueVal);
         ColumnVector falseVec = ColumnVector.timestampMilliSecondsFromBoxedLongs(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.timestampMilliSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongSSParams")
  void testTimestampMilliSecondsSS(Boolean[] predVals, Long trueVal, Long falseVal, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.timestampFromLong(DType.TIMESTAMP_MILLISECONDS, trueVal);
         Scalar falseScalar = Scalar.timestampFromLong(DType.TIMESTAMP_MILLISECONDS, falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.timestampMilliSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongVVParams")
  void testTimestampMicroSecondsVV(Boolean[] predVals, Long[] trueVals, Long[] falseVals, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.timestampMicroSecondsFromBoxedLongs(trueVals);
         ColumnVector falseVec = ColumnVector.timestampMicroSecondsFromBoxedLongs(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongVSParams")
  void testTimestampMicroSecondsVS(Boolean[] predVals, Long[] trueVals, Long falseVal, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.timestampMicroSecondsFromBoxedLongs(trueVals);
         Scalar falseScalar = Scalar.timestampFromLong(DType.TIMESTAMP_MICROSECONDS, falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongSVParams")
  void testTimestampMicroSecondsSV(Boolean[] predVals, Long trueVal, Long[] falseVals, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.timestampFromLong(DType.TIMESTAMP_MICROSECONDS, trueVal);
         ColumnVector falseVec = ColumnVector.timestampMicroSecondsFromBoxedLongs(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongSSParams")
  void testTimestampMicroSecondsSS(Boolean[] predVals, Long trueVal, Long falseVal, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.timestampFromLong(DType.TIMESTAMP_MICROSECONDS, trueVal);
         Scalar falseScalar = Scalar.timestampFromLong(DType.TIMESTAMP_MICROSECONDS, falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.timestampMicroSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongVVParams")
  void testTimestampNanoSecondsVV(Boolean[] predVals, Long[] trueVals, Long[] falseVals, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.timestampNanoSecondsFromBoxedLongs(trueVals);
         ColumnVector falseVec = ColumnVector.timestampNanoSecondsFromBoxedLongs(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.timestampNanoSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongVSParams")
  void testTimestampNanoSecondsVS(Boolean[] predVals, Long[] trueVals, Long falseVal, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.timestampNanoSecondsFromBoxedLongs(trueVals);
         Scalar falseScalar = Scalar.timestampFromLong(DType.TIMESTAMP_NANOSECONDS, falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.timestampNanoSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongSVParams")
  void testTimestampNanoSecondsSV(Boolean[] predVals, Long trueVal, Long[] falseVals, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.timestampFromLong(DType.TIMESTAMP_NANOSECONDS, trueVal);
         ColumnVector falseVec = ColumnVector.timestampNanoSecondsFromBoxedLongs(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.timestampNanoSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createLongSSParams")
  void testTimestampNanoSecondsSS(Boolean[] predVals, Long trueVal, Long falseVal, Long[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.timestampFromLong(DType.TIMESTAMP_NANOSECONDS, trueVal);
         Scalar falseScalar = Scalar.timestampFromLong(DType.TIMESTAMP_NANOSECONDS, falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.timestampNanoSecondsFromBoxedLongs(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createStringVVParams")
  void testStringVV(Boolean[] predVals, String[] trueVals, String[] falseVals, String[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromStrings(trueVals);
         ColumnVector falseVec = ColumnVector.fromStrings(falseVals);
         ColumnVector result = pred.ifElse(trueVec, falseVec);
         ColumnVector expected = ColumnVector.fromStrings(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createStringVSParams")
  void testStringVS(Boolean[] predVals, String[] trueVals, String falseVal, String[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         ColumnVector trueVec = ColumnVector.fromStrings(trueVals);
         Scalar falseScalar = Scalar.fromString(falseVal);
         ColumnVector result = pred.ifElse(trueVec, falseScalar);
         ColumnVector expected = ColumnVector.fromStrings(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createStringSVParams")
  void testStringSV(Boolean[] predVals, String trueVal, String[] falseVals, String[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromString(trueVal);
         ColumnVector falseVec = ColumnVector.fromStrings(falseVals);
         ColumnVector result = pred.ifElse(trueScalar, falseVec);
         ColumnVector expected = ColumnVector.fromStrings(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @ParameterizedTest
  @MethodSource("createStringSSParams")
  void testStringSS(Boolean[] predVals, String trueVal, String falseVal, String[] expectVals) {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(predVals);
         Scalar trueScalar = Scalar.fromString(trueVal);
         Scalar falseScalar = Scalar.fromString(falseVal);
         ColumnVector result = pred.ifElse(trueScalar, falseScalar);
         ColumnVector expected = ColumnVector.fromStrings(expectVals)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testMismatchedTypesVV() {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(true, false, false, true);
         ColumnVector trueVec = ColumnVector.fromBoxedInts(1, 2, 3, 4);
         ColumnVector falseVec = ColumnVector.fromBoxedLongs(5L, 6L, 7L, 8L)) {
      assertThrows(CudfException.class, () -> pred.ifElse(trueVec, falseVec));
    }
  }

  @Test
  void testMismatchedTypesVS() {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(true, false, false, true);
         ColumnVector trueVec = ColumnVector.fromBoxedLongs(1L, 2L, 3L, 4L);
         Scalar falseScalar = Scalar.fromString("hey")) {
      assertThrows(CudfException.class, () -> pred.ifElse(trueVec, falseScalar));
    }
  }

  @Test
  void testMismatchedTypesSV() {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(true, false, false, true);
         Scalar trueScalar = Scalar.fromByte((byte) 1);
         ColumnVector falseVec = ColumnVector.fromBoxedInts(0, 2, 4, 6)) {
      assertThrows(CudfException.class, () -> pred.ifElse(trueScalar, falseVec));
    }
  }

  @Test
  void testMismatchedTypesSS() {
    try (ColumnVector pred = ColumnVector.fromBoxedBooleans(true, false, false, true);
         Scalar trueScalar = Scalar.fromByte((byte) 1);
         Scalar falseScalar = Scalar.fromString("hey")) {
      assertThrows(CudfException.class, () -> pred.ifElse(trueScalar, falseScalar));
    }
  }

  @Test
  void testApplyNullMask() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(0, 100, 1, 2, Integer.MIN_VALUE, null);
         ColumnVector mask = ColumnVector.fromBoxedBooleans(true, false, true, null, false, true);
         ColumnVector result = input.applyNullMask(mask);
         ColumnVector expected = ColumnVector.fromBoxedInts(0, null, 1, null, null, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void testApplyNullMaskAllTrueIsValuePreserving() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(0, 100, null, 2);
         ColumnVector mask = ColumnVector.fromBoxedBooleans(true, true, true, true);
         ColumnVector result = input.applyNullMask(mask)) {
      assertColumnsAreEqual(input, result);
    }
  }

  @Test
  void testApplyNullMaskRejectsNonBooleanMask() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(0, 100, 1, 2);
         ColumnVector mask = ColumnVector.fromBoxedInts(1, 0, 1, 0)) {
      assertThrows(IllegalArgumentException.class, () -> input.applyNullMask(mask));
    }
  }

  @Test
  void testApplyNullMaskRejectsRowCountMismatch() {
    try (ColumnVector input = ColumnVector.fromBoxedInts(0, 100, 1, 2);
         ColumnVector mask = ColumnVector.fromBoxedBooleans(true, false, true)) {
      assertThrows(IllegalArgumentException.class, () -> input.applyNullMask(mask));
    }
  }
}

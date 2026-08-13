/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import ai.rapids.cudf.DType;

import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/** A literal value in an AST expression. */
public final class Literal extends AstExpression {
  private final DType type;
  private final byte[] serializedValue;

  @Override
  public String toString() {
    return "Literal " + type + " " + java.util.Arrays.toString(serializedValue);
  }

  /** Construct a null literal of the specified type. */
  public static Literal ofNull(DType type) {
    return new Literal(type, null);
  }

  /** Construct a boolean literal with the specified value. */
  public static Literal ofBoolean(boolean value) {
    return new Literal(DType.BOOL8, new byte[] { value ? (byte) 1 : (byte) 0 });
  }

  /** Construct a boolean literal with the specified value or null. */
  public static Literal ofBoolean(Boolean value) {
    if (value == null) {
      return ofNull(DType.BOOL8);
    }
    return ofBoolean(value.booleanValue());
  }

  /** Construct a byte literal with the specified value. */
  public static Literal ofByte(byte value) {
    return new Literal(DType.INT8, new byte[] { value });
  }

  /** Construct a byte literal with the specified value or null. */
  public static Literal ofByte(Byte value) {
    if (value == null) {
      return ofNull(DType.INT8);
    }
    return ofByte(value.byteValue());
  }

  /** Construct a short literal with the specified value. */
  public static Literal ofShort(short value) {
    byte[] serializedValue = new byte[Short.BYTES];
    ByteBuffer.wrap(serializedValue).order(ByteOrder.nativeOrder()).putShort(value);
    return new Literal(DType.INT16, serializedValue);
  }

  /** Construct a short literal with the specified value or null. */
  public static Literal ofShort(Short value) {
    if (value == null) {
      return ofNull(DType.INT16);
    }
    return ofShort(value.shortValue());
  }

  /** Construct an integer literal with the specified value. */
  public static Literal ofInt(int value) {
    return ofIntBasedType(DType.INT32, value);
  }

  /** Construct an integer literal with the specified value or null. */
  public static Literal ofInt(Integer value) {
    if (value == null) {
      return ofNull(DType.INT32);
    }
    return ofInt(value.intValue());
  }

  /** Construct a long literal with the specified value. */
  public static Literal ofLong(long value) {
    return ofLongBasedType(DType.INT64, value);
  }

  /** Construct a long literal with the specified value or null. */
  public static Literal ofLong(Long value) {
    if (value == null) {
      return ofNull(DType.INT64);
    }
    return ofLong(value.longValue());
  }

  /** Construct a float literal with the specified value. */
  public static Literal ofFloat(float value) {
    byte[] serializedValue = new byte[Float.BYTES];
    ByteBuffer.wrap(serializedValue).order(ByteOrder.nativeOrder()).putFloat(value);
    return new Literal(DType.FLOAT32, serializedValue);
  }

  /** Construct a float literal with the specified value or null. */
  public static Literal ofFloat(Float value) {
    if (value == null) {
      return ofNull(DType.FLOAT32);
    }
    return ofFloat(value.floatValue());
  }

  /** Construct a double literal with the specified value. */
  public static Literal ofDouble(double value) {
    byte[] serializedValue = new byte[Double.BYTES];
    ByteBuffer.wrap(serializedValue).order(ByteOrder.nativeOrder()).putDouble(value);
    return new Literal(DType.FLOAT64, serializedValue);
  }

  /** Construct a double literal with the specified value or null. */
  public static Literal ofDouble(Double value) {
    if (value == null) {
      return ofNull(DType.FLOAT64);
    }
    return ofDouble(value.doubleValue());
  }

  /**
   * Construct a decimal literal with the specified type and unscaled value.
   * A null {@code unscaledValue} produces a null literal of the requested type.
   * Root literals of type {@code DECIMAL32} or {@code DECIMAL64} can be evaluated with either
   * {@link CompiledExpression#computeColumn} or {@link CompiledExpression#computeColumnJit}.
   * A {@code DECIMAL128} root literal must use {@code computeColumnJit}; the legacy executor
   * cannot materialize it correctly.
   *
   * @param type decimal storage type and scale
   * @param unscaledValue unscaled decimal value, or null
   * @return decimal literal
   * @throws IllegalArgumentException if {@code type} is not a decimal type
   * @throws ArithmeticException if {@code unscaledValue} does not fit in {@code type}
   */
  public static Literal ofDecimal(DType type, BigInteger unscaledValue) {
    if (!type.isDecimalType()) {
      throw new IllegalArgumentException("type is not a decimal: " + type);
    }
    if (unscaledValue == null) {
      return ofNull(type);
    }
    if (type.getTypeId() == DType.DTypeEnum.DECIMAL32) {
      return ofIntBasedType(type, unscaledValue.intValueExact());
    } else if (type.getTypeId() == DType.DTypeEnum.DECIMAL64) {
      return ofLongBasedType(type, unscaledValue.longValueExact());
    } else {
      if (unscaledValue.bitLength() > type.getSizeInBytes() * Byte.SIZE - 1) {
        throw new ArithmeticException("BigInteger out of DECIMAL128 range");
      }
      return new Literal(type, convertDecimal128FromJavaToCudf(unscaledValue.toByteArray(), type));
    }
  }

  /** Construct a timestamp days literal with the specified value. */
  public static Literal ofTimestampDaysFromInt(int value) {
    return ofIntBasedType(DType.TIMESTAMP_DAYS, value);
  }

  /** Construct a timestamp days literal with the specified value or null. */
  public static Literal ofTimestampDaysFromInt(Integer value) {
    if (value == null) {
      return ofNull(DType.TIMESTAMP_DAYS);
    }
    return ofTimestampDaysFromInt(value.intValue());
  }

  /** Construct a long-based timestamp literal with the specified value. */
  public static Literal ofTimestampFromLong(DType type, long value) {
    if (!type.isTimestampType()) {
      throw new IllegalArgumentException("type is not a timestamp: " + type);
    }
    if (type.equals(DType.TIMESTAMP_DAYS)) {
      int intValue = (int)value;
      if (value != intValue) {
        throw new IllegalArgumentException("value too large for type " + type + ": " + value);
      }
      return ofTimestampDaysFromInt(intValue);
    }
    return ofLongBasedType(type, value);
  }

  /** Construct a long-based timestamp literal with the specified value or null. */
  public static Literal ofTimestampFromLong(DType type, Long value) {
    if (value == null) {
      return ofNull(type);
    }
    return ofTimestampFromLong(type, value.longValue());
  }

  /** Construct a duration days literal with the specified value. */
  public static Literal ofDurationDaysFromInt(int value) {
    return ofIntBasedType(DType.DURATION_DAYS, value);
  }

  /** Construct a duration days literal with the specified value or null. */
  public static Literal ofDurationDaysFromInt(Integer value) {
    if (value == null) {
      return ofNull(DType.DURATION_DAYS);
    }
    return ofDurationDaysFromInt(value.intValue());
  }

  /** Construct a long-based duration literal with the specified value. */
  public static Literal ofDurationFromLong(DType type, long value) {
    if (!type.isDurationType()) {
      throw new IllegalArgumentException("type is not a timestamp: " + type);
    }
    if (type.equals(DType.DURATION_DAYS)) {
      int intValue = (int)value;
      if (value != intValue) {
        throw new IllegalArgumentException("value too large for type " + type + ": " + value);
      }
      return ofDurationDaysFromInt(intValue);
    }
    return ofLongBasedType(type, value);
  }

  /** Construct a long-based duration literal with the specified value or null. */
  public static Literal ofDurationFromLong(DType type, Long value) {
    if (value == null) {
      return ofNull(type);
    }
    return ofDurationFromLong(type, value.longValue());
  }

  /** Construct a string literal with the specified value or null. */
  public static Literal ofString(String value) {
    if (value == null) {
      return ofNull(DType.STRING);
    }
    return ofUTF8String(value.getBytes(StandardCharsets.UTF_8));
  }

  /** Construct a string literal directly with byte array to skip transcoding. */
  public static Literal ofUTF8String(byte[] stringBytes) {
    if (stringBytes == null) {
      return ofNull(DType.STRING);
    }
    byte[] serializedValue = new byte[stringBytes.length + Integer.BYTES];
    ByteBuffer.wrap(serializedValue).order(ByteOrder.nativeOrder()).putInt(stringBytes.length);
    System.arraycopy(stringBytes, 0, serializedValue, Integer.BYTES, stringBytes.length);
    return new Literal(DType.STRING, serializedValue);
  }

  Literal(DType type, byte[] serializedValue) {
    this.type = type;
    this.serializedValue = serializedValue;
  }

  @Override
  int getSerializedSize() {
    ExpressionType nodeType = serializedValue != null
        ? ExpressionType.VALID_LITERAL : ExpressionType.NULL_LITERAL;
    int size = nodeType.getSerializedSize() + getDataTypeSerializedSize();
    if (serializedValue != null) {
      size += serializedValue.length;
    }
    return size;
  }

  @Override
  void serialize(ByteBuffer bb) {
    ExpressionType nodeType = serializedValue != null
        ? ExpressionType.VALID_LITERAL : ExpressionType.NULL_LITERAL;
    nodeType.serialize(bb);
    serializeDataType(bb);
    if (serializedValue != null) {
      bb.put(serializedValue);
    }
  }

  private int getDataTypeSerializedSize() {
    AstUtils.checkByte(type.getTypeId().getNativeId());
    if (type.isDecimalType()) {
      return Byte.BYTES + Integer.BYTES;
    }
    return Byte.BYTES;
  }

  private void serializeDataType(ByteBuffer bb) {
    byte nativeTypeId = AstUtils.checkByte(type.getTypeId().getNativeId());
    bb.put(nativeTypeId);
    if (type.isDecimalType()) {
      bb.putInt(type.getScale());
    }
  }

  private static Literal ofIntBasedType(DType type, int value) {
    assert type.getSizeInBytes() == Integer.BYTES;
    byte[] serializedValue = new byte[Integer.BYTES];
    ByteBuffer.wrap(serializedValue).order(ByteOrder.nativeOrder()).putInt(value);
    return new Literal(type, serializedValue);
  }

  private static Literal ofLongBasedType(DType type, long value) {
    assert type.getSizeInBytes() == Long.BYTES;
    byte[] serializedValue = new byte[Long.BYTES];
    ByteBuffer.wrap(serializedValue).order(ByteOrder.nativeOrder()).putLong(value);
    return new Literal(type, serializedValue);
  }

  private static byte[] convertDecimal128FromJavaToCudf(byte[] bytes, DType type) {
    return convertDecimal128FromJavaToCudf(bytes, type, ByteOrder.nativeOrder());
  }

  // Visible for testing so both possible native byte orders can be exercised.
  static byte[] convertDecimal128FromJavaToCudf(
      byte[] bytes, DType type, ByteOrder byteOrder) {
    // BigInteger uses big-endian bytes, while JNI reads the decimal128 value in native order.
    byte[] finalBytes = new byte[type.getSizeInBytes()];
    byte signByte = (bytes[0] & 0x80) > 0 ? (byte) 0xff : (byte) 0x00;
    if (byteOrder == ByteOrder.BIG_ENDIAN) {
      int offset = finalBytes.length - bytes.length;
      for (int i = 0; i < offset; i++) {
        finalBytes[i] = signByte;
      }
      System.arraycopy(bytes, 0, finalBytes, offset, bytes.length);
    } else {
      for (int i = bytes.length; i < finalBytes.length; i++) {
        finalBytes[i] = signByte;
      }
      for (int i = 0; i < bytes.length; i++) {
        finalBytes[i] = bytes[bytes.length - i - 1];
      }
    }
    return finalBytes;
  }
}

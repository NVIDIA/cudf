/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

import java.math.BigDecimal;
import java.util.EnumSet;
import java.util.Objects;

/**
 * The data type of a column or scalar, with a scale for decimal types.
 * Non-decimal types are available as constants such as {@link #INT32}; use
 * {@link #create(DTypeEnum, int)} to create decimal types.
 */
public final class DType {

  public static final int DECIMAL32_MAX_PRECISION = 9;
  public static final int DECIMAL64_MAX_PRECISION = 18;
  public static final int DECIMAL128_MAX_PRECISION = 38;

  /* enum representing various types. Whenever a new non-decimal type is added please make sure
  below sections are updated as well:
  1. Create a singleton object of the new type.
  2. Update SINGLETON_DTYPE_LOOKUP to reflect new type. The order should be maintained between
  DTypeEnum and SINGLETON_DTYPE_LOOKUP */
  public enum DTypeEnum {
    EMPTY(0, 0),
    INT8(1, 1),
    INT16(2, 2),
    INT32(4, 3),
    INT64(8, 4),
    UINT8(1, 5),
    UINT16(2, 6),
    UINT32(4, 7),
    UINT64(8, 8),
    FLOAT32(4, 9),
    FLOAT64(8, 10),
    /**
     * Byte wise true non-0/false 0.  In general true will be 1.
     */
    BOOL8(1, 11),
    /**
     * Days since the UNIX epoch
     */
    TIMESTAMP_DAYS(4, 12),
    /**
     * s since the UNIX epoch
     */
    TIMESTAMP_SECONDS(8, 13),
    /**
     * ms since the UNIX epoch
     */
    TIMESTAMP_MILLISECONDS(8, 14),
    /**
     * microseconds since the UNIX epoch
     */
    TIMESTAMP_MICROSECONDS(8, 15),
    /**
     * ns since the UNIX epoch
     */
    TIMESTAMP_NANOSECONDS(8, 16),

    DURATION_DAYS(4, 17),
    DURATION_SECONDS(8, 18),
    DURATION_MILLISECONDS(8, 19),
    DURATION_MICROSECONDS(8, 20),
    DURATION_NANOSECONDS(8, 21),
    //DICTIONARY32(4, 22),

    STRING(0, 23),
    LIST(0, 24),
    DECIMAL32(4, 25),
    DECIMAL64(8, 26),
    DECIMAL128(16, 27),
    STRUCT(0, 28);

    final int sizeInBytes;
    final int nativeId;

    DTypeEnum(int sizeInBytes, int nativeId) {
      this.sizeInBytes = sizeInBytes;
      this.nativeId = nativeId;
    }

    public int getNativeId() { return nativeId; }

    public boolean isDecimalType() { return DType.DECIMALS.contains(this); }
  }

  final DTypeEnum typeId;
  private final int scale;

  private DType(DTypeEnum id) {
    typeId = id;
    scale = 0;
  }

  /**
   * Constructor for Decimal Type
   * @param id Enum representing data type.
   * @param decimalScale Scale of fixed point decimal type
   */
  private DType(DTypeEnum id, int decimalScale) {
    typeId = id;
    scale = decimalScale;
  }

  public static final DType EMPTY = new DType(DTypeEnum.EMPTY);
  public static final DType INT8 = new DType(DTypeEnum.INT8);
  public static final DType INT16 = new DType(DTypeEnum.INT16);
  public static final DType INT32 = new DType(DTypeEnum.INT32);
  public static final DType INT64 = new DType(DTypeEnum.INT64);
  public static final DType UINT8 = new DType(DTypeEnum.UINT8);
  public static final DType UINT16 = new DType(DTypeEnum.UINT16);
  public static final DType UINT32 = new DType(DTypeEnum.UINT32);
  public static final DType UINT64 = new DType(DTypeEnum.UINT64);
  public static final DType FLOAT32 = new DType(DTypeEnum.FLOAT32);
  public static final DType FLOAT64 = new DType(DTypeEnum.FLOAT64);
  public static final DType BOOL8 = new DType(DTypeEnum.BOOL8);
  public static final DType TIMESTAMP_DAYS = new DType(DTypeEnum.TIMESTAMP_DAYS);
  public static final DType TIMESTAMP_SECONDS = new DType(DTypeEnum.TIMESTAMP_SECONDS);
  public static final DType TIMESTAMP_MILLISECONDS = new DType(DTypeEnum.TIMESTAMP_MILLISECONDS);
  public static final DType TIMESTAMP_MICROSECONDS = new DType(DTypeEnum.TIMESTAMP_MICROSECONDS);
  public static final DType TIMESTAMP_NANOSECONDS = new DType(DTypeEnum.TIMESTAMP_NANOSECONDS);
  public static final DType DURATION_DAYS = new DType(DTypeEnum.DURATION_DAYS);
  public static final DType DURATION_SECONDS = new DType(DTypeEnum.DURATION_SECONDS);
  public static final DType DURATION_MILLISECONDS = new DType(DTypeEnum.DURATION_MILLISECONDS);
  public static final DType DURATION_MICROSECONDS = new DType(DTypeEnum.DURATION_MICROSECONDS);
  public static final DType DURATION_NANOSECONDS = new DType(DTypeEnum.DURATION_NANOSECONDS);
  public static final DType STRING = new DType(DTypeEnum.STRING);
  public static final DType LIST = new DType(DTypeEnum.LIST);
  public static final DType STRUCT = new DType(DTypeEnum.STRUCT);

  /* This is used in fromNative method to return singleton object for non-decimal types.
  Please make sure the order here is same as that of DTypeEnum. Whenever a new non-decimal
  type is added in DTypeEnum, this array needs to be updated as well.*/
  private static final DType[] SINGLETON_DTYPE_LOOKUP = new DType[]{
      EMPTY,
      INT8,
      INT16,
      INT32,
      INT64,
      UINT8,
      UINT16,
      UINT32,
      UINT64,
      FLOAT32,
      FLOAT64,
      BOOL8,
      TIMESTAMP_DAYS,
      TIMESTAMP_SECONDS,
      TIMESTAMP_MILLISECONDS,
      TIMESTAMP_MICROSECONDS,
      TIMESTAMP_NANOSECONDS,
      DURATION_DAYS,
      DURATION_SECONDS,
      DURATION_MILLISECONDS,
      DURATION_MICROSECONDS,
      DURATION_NANOSECONDS,
      null, // DICTIONARY32
      STRING,
      LIST,
      null, // DECIMAL32
      null, // DECIMAL64
      null, // DECIMAL128
      STRUCT
  };

  /**
   * Returns max precision for Decimal Type.
   * @return max precision this Decimal Type can hold
   */
  public int getDecimalMaxPrecision() {
    if (!isDecimalType()) {
      throw new IllegalArgumentException("not a decimal type: " + this);
    }
    if (typeId == DTypeEnum.DECIMAL32) return DECIMAL32_MAX_PRECISION;
    if (typeId == DTypeEnum.DECIMAL64) return DECIMAL64_MAX_PRECISION;
    return DType.DECIMAL128_MAX_PRECISION;
  }

  /**
   * Get the number of decimal places needed to hold the Integral Type.
   * NOTE: this method is NOT for Decimal Type but for Integral Type.
   * @return the minimum decimal precision (places) for Integral Type
   */
  public int getPrecisionForInt() {
    // -128 to 127
    if (typeId == DTypeEnum.INT8) return 3;
    // -32768 to 32767
    if (typeId == DTypeEnum.INT16) return 5;
    // -2147483648 to 2147483647
    if (typeId == DTypeEnum.INT32) return 10;
    // -9223372036854775808 to 9223372036854775807
    if (typeId == DTypeEnum.INT64) return 19;

    throw new IllegalArgumentException("not an integral type: " + this);
  }

  /**
   * This only works for fixed width types. Variable width types like strings the value is
   * undefined and should be ignored.
   *
   * @return size of type in bytes.
   */
  public int getSizeInBytes() { return typeId.sizeInBytes; }

  /**
   * Returns scale for Decimal Type.
   * @return scale base-10 exponent to multiply the unscaled value to produce the decimal value.
   * Example: Consider unscaled value = 123456
   *         if scale = -2, decimal value = 123456 * 10^-2 = 1234.56
   *         if scale = 2, decimal value = 123456 * 10^2 = 12345600
   */
  public int getScale() { return scale; }

  /**
   * Return enum for this DType
   * @return DTypeEnum
   */
  public DTypeEnum getTypeId() {
    return typeId;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    DType type = (DType) o;
    return scale == type.scale && typeId == type.typeId;
  }

  @Override
  public int hashCode() {
    return Objects.hash(typeId, scale);
  }

  @Override
  public String toString() {
    if (isDecimalType()) {
      return typeId + " scale:" + scale;
    } else {
      return String.valueOf(typeId);
    }
  }

  /**
   * Factory method for non-decimal DType instances.
   * @param dt  enum corresponding to datatype.
   * @return DType
   */
  public static DType create(DTypeEnum dt) {
    if (DType.DECIMALS.contains(dt)) {
      throw new IllegalArgumentException("Could not create a Decimal DType without scale");
    }
    return DType.fromNative(dt.nativeId, 0);
  }

  /**
   * Factory method specialized for decimal DType instances.
   * @param dt  enum corresponding to datatype.
   * @param scale base-10 exponent to multiply the unscaled value to produce the decimal value.
   * Example: Consider unscaled value = 123456
   *         if scale = -2, decimal value = 123456 * 10^-2 = 1234.56
   *         if scale = 2, decimal value = 123456 * 10^2 = 12345600
   * @return DType
   */
  public static DType create(DTypeEnum dt, int scale) {
    if (!DType.DECIMALS.contains(dt)) {
      throw new IllegalArgumentException("Could not create a non-Decimal DType with scale");
    }
    return DType.fromNative(dt.nativeId, scale);
  }

  /**
   * Factory method for DType instances
   * @param nativeId nativeId of DataTypeEnun
   * @param scale base-10 exponent to multiply the unscaled value to produce the decimal value
   * Example: Consider unscaled value = 123456
   *         if scale = -2, decimal value = 123456 * 10^-2 = 1234.56
   *         if scale = 2, decimal value = 123456 * 10^2 = 12345600
   * @return DType
   */
  public static DType fromNative(int nativeId, int scale) {
    if (nativeId >=0 && nativeId < SINGLETON_DTYPE_LOOKUP.length) {
      DType ret = SINGLETON_DTYPE_LOOKUP[nativeId];
      if (ret != null) {
        assert ret.typeId.nativeId == nativeId : "Something went wrong and it looks like " +
          "SINGLETON_DTYPE_LOOKUP is out of sync";
        return ret;
      }
      if (nativeId == DTypeEnum.DECIMAL32.nativeId) {
        if (-scale > DECIMAL32_MAX_PRECISION) {
          throw new IllegalArgumentException(
              "Scale " + (-scale) + " exceeds DECIMAL32_MAX_PRECISION " + DECIMAL32_MAX_PRECISION);
        }
        return new DType(DTypeEnum.DECIMAL32, scale);
      }
      if (nativeId == DTypeEnum.DECIMAL64.nativeId) {
        if (-scale > DECIMAL64_MAX_PRECISION) {
          throw new IllegalArgumentException(
              "Scale " + (-scale) + " exceeds DECIMAL64_MAX_PRECISION " + DECIMAL64_MAX_PRECISION);
        }
        return new DType(DTypeEnum.DECIMAL64, scale);
      }
      if (nativeId == DTypeEnum.DECIMAL128.nativeId) {
        if (-scale > DECIMAL128_MAX_PRECISION) {
          throw new IllegalArgumentException(
              "Scale " + (-scale) + " exceeds DECIMAL128_MAX_PRECISION " + DECIMAL128_MAX_PRECISION);
        }
        return new DType(DTypeEnum.DECIMAL128, scale);
      }
    }
    throw new IllegalArgumentException("Could not translate " + nativeId + " into a DType");
  }

  /**
   * Create decimal-like DType using precision and scale of Java BigDecimal.
   *
   * @param dec BigDecimal
   * @return DType
   */
  public static DType fromJavaBigDecimal(BigDecimal dec) {
    // Notice: Compared to scale of Java BigDecimal, scale of libcudf works in opposite.
    // So, we negate the scale value before passing it into constructor.
    if (dec.precision() <= DECIMAL32_MAX_PRECISION) {
      return new DType(DTypeEnum.DECIMAL32, -dec.scale());
    } else if (dec.precision() <= DECIMAL64_MAX_PRECISION) {
      return new DType(DTypeEnum.DECIMAL64, -dec.scale());
    } else if (dec.precision() <= DECIMAL128_MAX_PRECISION) {
      return new DType(DTypeEnum.DECIMAL128, -dec.scale());
    }
    throw new IllegalArgumentException("Precision " + dec.precision() +
        " exceeds max precision cuDF can support " + DECIMAL128_MAX_PRECISION);
  }

  /**
   * Returns true for timestamps with time level resolution, as opposed to day level resolution
   */
  public boolean hasTimeResolution() {
    return TIME_RESOLUTION.contains(this.typeId);
  }

  /**
   * Returns true if this type is backed by int type
   * Namely this method will return true for the following types
   *       DType.INT32,
   *       DType.UINT32,
   *       DType.DURATION_DAYS,
   *       DType.TIMESTAMP_DAYS,
   *       DType.DECIMAL32
   */
  public boolean isBackedByInt() {
    return INTS.contains(this.typeId);
  }

  /**
   * Returns true if this type is backed by long type
   * Namely this method will return true for the following types
   *       DType.INT64,
   *       DType.UINT64,
   *       DType.DURATION_SECONDS,
   *       DType.DURATION_MILLISECONDS,
   *       DType.DURATION_MICROSECONDS,
   *       DType.DURATION_NANOSECONDS,
   *       DType.TIMESTAMP_SECONDS,
   *       DType.TIMESTAMP_MILLISECONDS,
   *       DType.TIMESTAMP_MICROSECONDS,
   *       DType.TIMESTAMP_NANOSECONDS,
   *       DType.DECIMAL64
   */
  public boolean isBackedByLong() {
    return LONGS.contains(this.typeId);
  }

  /**
   * Returns true if this type is backed by short type
   * Namely this method will return true for the following types
   *       DType.INT16,
   *       DType.UINT16
   */
  public boolean isBackedByShort() { return SHORTS.contains(this.typeId); }

  /**
   * Returns true if this type is backed by byte type
   * Namely this method will return true for the following types
   *       DType.INT8,
   *       DType.UINT8,
   *       DType.BOOL8
   */
  public boolean isBackedByByte() { return BYTES.contains(this.typeId); }

  /**
   * Returns true if this type is of decimal type
   * Namely this method will return true for the following types
   *       DType.DECIMAL32,
   *       DType.DECIMAL64
   */
  public boolean isDecimalType() { return this.typeId.isDecimalType(); }

  /**
   * Returns true for duration types
   */
  public boolean isDurationType() {
    return DURATION_TYPE.contains(this.typeId);
  }

  /**
   * Returns true for strictly Integer types not a type backed by
   * ints
   */
  public boolean isIntegral() {
    return INTEGRALS.contains(this.typeId);
  }

 /**
   * Returns true for nested types
   */
  public boolean isNestedType() {
    return NESTED_TYPE.contains(this.typeId);
  }

  @Deprecated
  public boolean isTimestamp() {
    return TIMESTAMPS.contains(this.typeId);
  }

  public boolean isTimestampType() {
    return TIMESTAMPS.contains(this.typeId);
  }

  /**
   * Returns true if the type uses a vector of offsets
   */
  public boolean hasOffsets() {
    return OFFSETS_TYPE.contains(this.typeId);
  }

  private static final EnumSet<DTypeEnum> TIMESTAMPS = EnumSet.of(
      DTypeEnum.TIMESTAMP_DAYS,
      DTypeEnum.TIMESTAMP_SECONDS,
      DTypeEnum.TIMESTAMP_MILLISECONDS,
      DTypeEnum.TIMESTAMP_MICROSECONDS,
      DTypeEnum.TIMESTAMP_NANOSECONDS);

  private static final EnumSet<DTypeEnum> TIME_RESOLUTION = EnumSet.of(
      DTypeEnum.TIMESTAMP_SECONDS,
      DTypeEnum.TIMESTAMP_MILLISECONDS,
      DTypeEnum.TIMESTAMP_MICROSECONDS,
      DTypeEnum.TIMESTAMP_NANOSECONDS);

  private static final EnumSet<DTypeEnum> DURATION_TYPE = EnumSet.of(
      DTypeEnum.DURATION_DAYS,
      DTypeEnum.DURATION_MICROSECONDS,
      DTypeEnum.DURATION_MILLISECONDS,
      DTypeEnum.DURATION_NANOSECONDS,
      DTypeEnum.DURATION_SECONDS
  );

  private static final EnumSet<DTypeEnum> LONGS = EnumSet.of(
      DTypeEnum.INT64,
      DTypeEnum.UINT64,
      DTypeEnum.DURATION_SECONDS,
      DTypeEnum.DURATION_MILLISECONDS,
      DTypeEnum.DURATION_MICROSECONDS,
      DTypeEnum.DURATION_NANOSECONDS,
      DTypeEnum.TIMESTAMP_SECONDS,
      DTypeEnum.TIMESTAMP_MILLISECONDS,
      DTypeEnum.TIMESTAMP_MICROSECONDS,
      DTypeEnum.TIMESTAMP_NANOSECONDS,
      // The unscaledValue of DECIMAL64 is of type INT64, which means it can be fetched by getLong.
      DTypeEnum.DECIMAL64
  );

  private static final EnumSet<DTypeEnum> INTS = EnumSet.of(
      DTypeEnum.INT32,
      DTypeEnum.UINT32,
      DTypeEnum.DURATION_DAYS,
      DTypeEnum.TIMESTAMP_DAYS,
      // The unscaledValue of DECIMAL32 is of type INT32, which means it can be fetched by getInt.
      DTypeEnum.DECIMAL32
  );

  private static final EnumSet<DTypeEnum> SHORTS = EnumSet.of(
      DTypeEnum.INT16,
      DTypeEnum.UINT16
  );

  private static final EnumSet<DTypeEnum> BYTES = EnumSet.of(
      DTypeEnum.INT8,
      DTypeEnum.UINT8,
      DTypeEnum.BOOL8
  );

  private static final EnumSet<DTypeEnum> DECIMALS = EnumSet.of(
      DTypeEnum.DECIMAL32,
      DTypeEnum.DECIMAL64,
      DTypeEnum.DECIMAL128
  );

  private static final EnumSet<DTypeEnum> NESTED_TYPE = EnumSet.of(
      DTypeEnum.LIST,
      DTypeEnum.STRUCT
  );

  private static final EnumSet<DTypeEnum> OFFSETS_TYPE = EnumSet.of(
      DTypeEnum.STRING,
      DTypeEnum.LIST
  );

  private static final EnumSet<DTypeEnum> INTEGRALS = EnumSet.of(
    DTypeEnum.INT8,
    DTypeEnum.INT16,
    DTypeEnum.INT32,
    DTypeEnum.INT64,
    DTypeEnum.UINT8,
    DTypeEnum.UINT16,
    DTypeEnum.UINT32,
    DTypeEnum.UINT64
  );
}

/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * A single scalar value.
 */
public final class Scalar implements AutoCloseable, BinaryOperable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger LOG = LoggerFactory.getLogger(Scalar.class);

  private final DType type;
  private int refCount;
  private final OffHeapState offHeap;

  public static Scalar fromNull(DType type) {
    switch (type.typeId) {
    case EMPTY:
    case BOOL8:
      return new Scalar(type, makeBool8Scalar(false, false));
    case INT8:
      return new Scalar(type, makeInt8Scalar((byte)0, false));
    case UINT8:
      return new Scalar(type, makeUint8Scalar((byte)0, false));
    case INT16:
      return new Scalar(type, makeInt16Scalar((short)0, false));
    case UINT16:
      return new Scalar(type, makeUint16Scalar((short)0, false));
    case INT32:
      return new Scalar(type, makeInt32Scalar(0, false));
    case UINT32:
      return new Scalar(type, makeUint32Scalar(0, false));
    case TIMESTAMP_DAYS:
      return new Scalar(type, makeTimestampDaysScalar(0, false));
    case FLOAT32:
      return new Scalar(type, makeFloat32Scalar(0, false));
    case FLOAT64:
      return new Scalar(type, makeFloat64Scalar(0, false));
    case INT64:
      return new Scalar(type, makeInt64Scalar(0, false));
    case UINT64:
      return new Scalar(type, makeUint64Scalar(0, false));
    case TIMESTAMP_SECONDS:
    case TIMESTAMP_MILLISECONDS:
    case TIMESTAMP_MICROSECONDS:
    case TIMESTAMP_NANOSECONDS:
      return new Scalar(type, makeTimestampTimeScalar(type.typeId.getNativeId(), 0, false));
    case STRING:
      return new Scalar(type, makeStringScalar(null, false));
    case DURATION_DAYS:
      return new Scalar(type, makeDurationDaysScalar(0, false));
    case DURATION_MICROSECONDS:
    case DURATION_MILLISECONDS:
    case DURATION_NANOSECONDS:
    case DURATION_SECONDS:
      return new Scalar(type, makeDurationTimeScalar(type.typeId.getNativeId(), 0, false));
    case DECIMAL32:
      return new Scalar(type, makeDecimal32Scalar(0, type.getScale(), false));
    case DECIMAL64:
      return new Scalar(type, makeDecimal64Scalar(0L, type.getScale(), false));
    case DECIMAL128:
      return new Scalar(type, makeDecimal128Scalar(BigInteger.ZERO.toByteArray(), type.getScale(), false));
    case LIST:
      throw new IllegalArgumentException("Please call 'listFromNull' to create a null list scalar.");
    default:
      throw new IllegalArgumentException("Unexpected type: " + type);
    }
  }

  public static Scalar fromBool(boolean value) {
    return new Scalar(DType.BOOL8, makeBool8Scalar(value, true));
  }

  public static Scalar fromBool(Boolean value) {
    if (value == null) {
      return Scalar.fromNull(DType.BOOL8);
    }
    return Scalar.fromBool(value.booleanValue());
  }

  public static Scalar fromByte(byte value) {
    return new Scalar(DType.INT8, makeInt8Scalar(value, true));
  }

  public static Scalar fromByte(Byte value) {
    if (value == null) {
      return Scalar.fromNull(DType.INT8);
    }
    return Scalar.fromByte(value.byteValue());
  }

  public static Scalar fromUnsignedByte(byte value) {
    return new Scalar(DType.UINT8, makeUint8Scalar(value, true));
  }

  public static Scalar fromUnsignedByte(Byte value) {
    if (value == null) {
      return Scalar.fromNull(DType.UINT8);
    }
    return Scalar.fromUnsignedByte(value.byteValue());
  }

  public static Scalar fromShort(short value) {
    return new Scalar(DType.INT16, makeInt16Scalar(value, true));
  }

  public static Scalar fromShort(Short value) {
    if (value == null) {
      return Scalar.fromNull(DType.INT16);
    }
    return Scalar.fromShort(value.shortValue());
  }

  public static Scalar fromUnsignedShort(short value) {
    return new Scalar(DType.UINT16, makeUint16Scalar(value, true));
  }

  public static Scalar fromUnsignedShort(Short value) {
    if (value == null) {
      return Scalar.fromNull(DType.UINT16);
    }
    return Scalar.fromUnsignedShort(value.shortValue());
  }

  /**
   * Returns a DURATION_DAYS scalar
   * @param value - days
   * @return - Scalar value
   */
  public static Scalar durationDaysFromInt(int value) {
    return new Scalar(DType.DURATION_DAYS, makeDurationDaysScalar(value, true));
  }

  /**
   * Returns a DURATION_DAYS scalar
   * @param value - days
   * @return - Scalar value
   */
  public static Scalar durationDaysFromInt(Integer value) {
    if (value == null) {
      return Scalar.fromNull(DType.DURATION_DAYS);
    }
    return Scalar.durationDaysFromInt(value.intValue());
  }

  public static Scalar fromInt(int value) {
    return new Scalar(DType.INT32, makeInt32Scalar(value, true));
  }

  public static Scalar fromInt(Integer value) {
    if (value == null) {
      return Scalar.fromNull(DType.INT32);
    }
    return Scalar.fromInt(value.intValue());
  }

  public static Scalar fromUnsignedInt(int value) {
    return new Scalar(DType.UINT32, makeUint32Scalar(value, true));
  }

  public static Scalar fromUnsignedInt(Integer value) {
    if (value == null) {
      return Scalar.fromNull(DType.UINT32);
    }
    return Scalar.fromUnsignedInt(value.intValue());
  }

  public static Scalar fromLong(long value) {
    return new Scalar(DType.INT64, makeInt64Scalar(value, true));
  }

  public static Scalar fromLong(Long value) {
    if (value == null) {
      return Scalar.fromNull(DType.INT64);
    }
    return Scalar.fromLong(value.longValue());
  }

  public static Scalar fromUnsignedLong(long value) {
    return new Scalar(DType.UINT64, makeUint64Scalar(value, true));
  }

  public static Scalar fromUnsignedLong(Long value) {
    if (value == null) {
      return Scalar.fromNull(DType.UINT64);
    }
    return Scalar.fromUnsignedLong(value.longValue());
  }

  public static Scalar fromFloat(float value) {
    return new Scalar(DType.FLOAT32, makeFloat32Scalar(value, true));
  }

  public static Scalar fromDecimal(int scale, int unscaledValue) {
    long handle = makeDecimal32Scalar(unscaledValue, scale, true);
    return new Scalar(DType.create(DType.DTypeEnum.DECIMAL32, scale), handle);
  }

  public static Scalar fromDecimal(int scale, long unscaledValue) {
    long handle = makeDecimal64Scalar(unscaledValue, scale, true);
    return new Scalar(DType.create(DType.DTypeEnum.DECIMAL64, scale), handle);
  }

  public static Scalar fromDecimal(int scale, BigInteger unscaledValue) {
    byte[] unscaledValueBytes = unscaledValue.toByteArray();
    byte[] finalBytes =  convertDecimal128FromJavaToCudf(unscaledValueBytes);
    long handle = makeDecimal128Scalar(finalBytes, scale, true);
    return new Scalar(DType.create(DType.DTypeEnum.DECIMAL128, scale), handle);
  }

  public static Scalar fromFloat(Float value) {
    if (value == null) {
      return Scalar.fromNull(DType.FLOAT32);
    }
    return Scalar.fromFloat(value.floatValue());
  }

  public static Scalar fromDouble(double value) {
    return new Scalar(DType.FLOAT64, makeFloat64Scalar(value, true));
  }

  public static Scalar fromDouble(Double value) {
    if (value == null) {
      return Scalar.fromNull(DType.FLOAT64);
    }
    return Scalar.fromDouble(value.doubleValue());
  }

  public static Scalar fromDecimal(BigDecimal value) {
    if (value == null) {
      return Scalar.fromNull(DType.create(DType.DTypeEnum.DECIMAL64, 0));
    }
    DType dt = DType.fromJavaBigDecimal(value);
    return fromDecimal(value.unscaledValue(), dt);
  }

  public static Scalar fromDecimal(BigInteger unscaledValue, DType dt) {
    if (unscaledValue == null) {
      return Scalar.fromNull(dt);
    }
    long handle;
    if (dt.typeId == DType.DTypeEnum.DECIMAL32) {
      handle = makeDecimal32Scalar(unscaledValue.intValueExact(), dt.getScale(), true);
    } else if (dt.typeId == DType.DTypeEnum.DECIMAL64) {
      handle = makeDecimal64Scalar(unscaledValue.longValueExact(), dt.getScale(), true);
    } else {
      byte[] unscaledValueBytes = unscaledValue.toByteArray();
      byte[] finalBytes =  convertDecimal128FromJavaToCudf(unscaledValueBytes);
      handle = makeDecimal128Scalar(finalBytes, dt.getScale(), true);
    }
    return new Scalar(dt, handle);
  }

  public static Scalar timestampDaysFromInt(int value) {
    return new Scalar(DType.TIMESTAMP_DAYS, makeTimestampDaysScalar(value, true));
  }

  public static Scalar timestampDaysFromInt(Integer value) {
    if (value == null) {
      return Scalar.fromNull(DType.TIMESTAMP_DAYS);
    }
    return Scalar.timestampDaysFromInt(value.intValue());
  }

  /**
   * Returns a duration scalar based on the type parameter.
   * @param type - dtype of scalar to be returned
   * @param value - corresponding value for the scalar
   * @return - Scalar of the respective type
   */
  public static Scalar durationFromLong(DType type, long value) {
    if (type.isDurationType()) {
      if (type.equals(DType.DURATION_DAYS)) {
        int intValue = (int)value;
        if (value != intValue) {
          throw new IllegalArgumentException("value too large for type " + type + ": " + value);
        }
        return durationDaysFromInt(intValue);
      } else {
        return new Scalar(type, makeDurationTimeScalar(type.typeId.getNativeId(), value, true));
      }
    } else {
      throw new IllegalArgumentException("type is not a timestamp: " + type);
    }
  }

  /**
   * Returns a duration scalar based on the type parameter.
   * @param type - dtype of scalar to be returned
   * @param value - corresponding value for the scalar
   * @return - Scalar of the respective type
   */
  public static Scalar durationFromLong(DType type, Long value) {
    if (value == null) {
      return Scalar.fromNull(type);
    }
    return Scalar.durationFromLong(type, value.longValue());
  }

  public static Scalar timestampFromLong(DType type, long value) {
    if (type.isTimestampType()) {
      if (type.equals(DType.TIMESTAMP_DAYS)) {
        int intValue = (int)value;
        if (value != intValue) {
          throw new IllegalArgumentException("value too large for type " + type + ": " + value);
        }
        return timestampDaysFromInt(intValue);
      } else {
        return new Scalar(type, makeTimestampTimeScalar(type.typeId.getNativeId(), value, true));
      }
    } else {
      throw new IllegalArgumentException("type is not a timestamp: " + type);
    }
  }

  public static Scalar timestampFromLong(DType type, Long value) {
    if (value == null) {
      return Scalar.fromNull(type);
    }
    return Scalar.timestampFromLong(type, value.longValue());
  }

  public static Scalar fromString(String value) {
    return fromUTF8String(value == null ? null : value.getBytes(StandardCharsets.UTF_8));
  }

  /**
   * Creates a String scalar from an array of UTF8 bytes.
   * @param value the array of UTF8 bytes
   * @return a String scalar
   */
  public static Scalar fromUTF8String(byte[] value) {
    if (value == null) {
      return fromNull(DType.STRING);
    }
    return new Scalar(DType.STRING, makeStringScalar(value, true));
  }

  /**
   * Creates a null scalar of list type.
   *
   * Having this special API because the element type is required to build an empty
   * nested column as the underlying column of the list scalar.
   *
   * @param elementType the data type of the element in the list.
   * @return a null scalar of list type
   */
  public static Scalar listFromNull(HostColumnVector.DataType elementType) {
    try (ColumnVector col = ColumnVector.empty(elementType)) {
      return new Scalar(DType.LIST, makeListScalar(col.getNativeView(), false));
    }
  }

  /**
   * Creates a scalar of list from a ColumnView.
   *
   * All the rows in the ColumnView will be copied into the Scalar. So the ColumnView
   * can be closed after this call completes.
   */
  public static Scalar listFromColumnView(ColumnView list) {
    if (list == null) {
      throw new IllegalArgumentException("'list' should NOT be null." +
          " Please call 'listFromNull' to create a null list scalar.");
    }
    return new Scalar(DType.LIST, makeListScalar(list.getNativeView(), true));
  }

  /**
   * Creates a null scalar of struct type.
   *
   * @param elementTypes data types of children in the struct
   * @return a null scalar of struct type
   */
  public static Scalar structFromNull(HostColumnVector.DataType... elementTypes) {
    ColumnVector[] children = new ColumnVector[elementTypes.length];
    long[] childHandles = new long[elementTypes.length];
    RuntimeException error = null;
    try {
      for (int i = 0; i < elementTypes.length; i++) {
        // Build column vector having single null value rather than empty column vector,
        // because struct scalar requires row count of children columns == 1.
        children[i] = buildNullColumnVector(elementTypes[i]);
        childHandles[i] = children[i].getNativeView();
      }
      return new Scalar(DType.STRUCT, makeStructScalar(childHandles, false));
    } catch (RuntimeException ex) {
      error = ex;
      throw ex;
    } catch (Exception ex) {
      error = new RuntimeException(ex);
      throw ex;
    } finally {
      // close all empty children
      for (ColumnVector child : children) {
        // We closed all created ColumnViews when we hit null. Therefore we exit the loop.
        if (child == null) break;
        // suppress exception during the close process to ensure that all elements are closed
        try {
          child.close();
        } catch (Exception ex) {
          if (error == null) {
            error = new RuntimeException(ex);
            continue;
          }
          error.addSuppressed(ex);
        }
      }
      if (error != null) throw error;
    }
  }

  /**
   * Creates a scalar of struct from a ColumnView.
   *
   * @param columns children columns of struct
   * @return a Struct scalar
   */
  public static Scalar structFromColumnViews(ColumnView... columns) {
    if (columns == null) {
      throw new IllegalArgumentException("input columns should NOT be null");
    }
    long[] columnHandles = new long[columns.length];
    for (int i = 0; i < columns.length; i++) {
      columnHandles[i] = columns[i].getNativeView();
    }
    return new Scalar(DType.STRUCT, makeStructScalar(columnHandles, true));
  }

  /**
   * Build column vector of single row who holds a null value
   *
   * @param hostType host data type of null column vector
   * @return the null vector
   */
  private static ColumnVector buildNullColumnVector(HostColumnVector.DataType hostType) {
    DType dt = hostType.getType();
    if (!dt.isNestedType()) {
      try (HostColumnVector.Builder builder = HostColumnVector.builder(dt, 1)) {
        builder.appendNull();
        try (HostColumnVector hcv = builder.build()) {
          return hcv.copyToDevice();
        }
      }
    } else if (dt.typeId == DType.DTypeEnum.LIST) {
      // type of List doesn't matter here because of type erasure in Java
      try (HostColumnVector hcv = HostColumnVector.fromLists(hostType, (List<Integer>) null)) {
        return hcv.copyToDevice();
      }
    } else if (dt.typeId == DType.DTypeEnum.STRUCT) {
      try (HostColumnVector hcv = HostColumnVector.fromStructs(
          hostType, (HostColumnVector.StructData) null)) {
        return hcv.copyToDevice();
      }
    } else {
      throw new IllegalArgumentException("Unsupported data type: " + hostType);
    }
  }

  private static native void closeScalar(long scalarHandle);
  private static native boolean isScalarValid(long scalarHandle);
  private static native byte getByte(long scalarHandle);
  private static native short getShort(long scalarHandle);
  private static native int getInt(long scalarHandle);
  private static native long getLong(long scalarHandle);
  private static native byte[] getBigIntegerBytes(long scalarHandle);
  private static native float getFloat(long scalarHandle);
  private static native double getDouble(long scalarHandle);
  private static native byte[] getUTF8(long scalarHandle);
  private static native long getListAsColumnView(long scalarHandle);
  private static native long[] getChildrenFromStructScalar(long scalarHandle);
  private static native long makeBool8Scalar(boolean isValid, boolean value);
  private static native long makeInt8Scalar(byte value, boolean isValid);
  private static native long makeUint8Scalar(byte value, boolean isValid);
  private static native long makeInt16Scalar(short value, boolean isValid);
  private static native long makeUint16Scalar(short value, boolean isValid);
  private static native long makeInt32Scalar(int value, boolean isValid);
  private static native long makeUint32Scalar(int value, boolean isValid);
  private static native long makeInt64Scalar(long value, boolean isValid);
  private static native long makeUint64Scalar(long value, boolean isValid);
  private static native long makeFloat32Scalar(float value, boolean isValid);
  private static native long makeFloat64Scalar(double value, boolean isValid);
  private static native long makeStringScalar(byte[] value, boolean isValid);
  private static native long makeDurationDaysScalar(int value, boolean isValid);
  private static native long makeDurationTimeScalar(int dtype, long value, boolean isValid);
  private static native long makeTimestampDaysScalar(int value, boolean isValid);
  private static native long makeTimestampTimeScalar(int dtypeNativeId, long value, boolean isValid);
  private static native long makeDecimal32Scalar(int value, int scale, boolean isValid);
  private static native long makeDecimal64Scalar(long value, int scale, boolean isValid);
  private static native long makeDecimal128Scalar(byte[] value, int scale, boolean isValid);
  private static native long makeListScalar(long viewHandle, boolean isValid);
  private static native long makeStructScalar(long[] viewHandles, boolean isValid);
  private static native long repeatString(long scalarHandle, int repeatTimes);

  /**
   * Constructor to create a scalar from a native handle and a type.
   *
   * @param type The type of the scalar
   * @param scalarHandle The native handle (pointer address) to the scalar data
   */
  public Scalar(DType type, long scalarHandle) {
    this.type = type;
    this.offHeap = new OffHeapState(scalarHandle);
    MemoryCleaner.register(this, offHeap);
    incRefCount();
  }

  /**
   * Get the native handle (native pointer address) for the scalar.
   *
   * @return The native handle
   */
  public long getScalarHandle() {
    return offHeap.scalarHandle;
  }

  /**
   * Increment the reference count for this scalar.  You need to call close on this
   * to decrement the reference count again.
   */
  public synchronized Scalar incRefCount() {
    if (offHeap.scalarHandle == 0) {
      offHeap.logRefCountDebug("INC AFTER CLOSE " + this);
      throw new IllegalStateException("Scalar is already closed");
    }
    offHeap.addRef();
    ++refCount;
    return this;
  }

  /**
   * Free the memory associated with a scalar.
   */
  @Override
  public synchronized void close() {
    refCount--;
    offHeap.delRef();
    if (refCount == 0) {
      offHeap.clean(false);
    } else if (refCount < 0) {
      offHeap.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times " + this);
    }
  }

  @Override
  public DType getType() {
    return type;
  }

  public boolean isValid() {
    return isScalarValid(getScalarHandle());
  }

  /**
   * Returns the scalar value as a boolean.
   */
  public boolean getBoolean() {
    return getByte(getScalarHandle()) != 0;
  }

  /**
   * Returns the scalar value as a byte.
   */
  public byte getByte() {
    return getByte(getScalarHandle());
  }

  /**
   * Returns the scalar value as a short.
   */
  public short getShort() {
    return getShort(getScalarHandle());
  }

  /**
   * Returns the scalar value as an int.
   */
  public int getInt() {
    return getInt(getScalarHandle());
  }

  /**
   * Returns the scalar value as a long.
   */
  public long getLong() {
    return getLong(getScalarHandle());
  }

  /**
   * Returns the BigDecimal unscaled scalar value as a byte array.
   */
  public byte[] getBigInteger() {
    byte[] res = getBigIntegerBytes(getScalarHandle());
    convertInPlaceToBigEndian(res);
    return res;
  }

  /**
   * Returns the scalar value as a float.
   */
  public float getFloat() {
    return getFloat(getScalarHandle());
  }

  /**
   * Returns the scalar value as a double.
   */
  public double getDouble() {
    return getDouble(getScalarHandle());
  }

  /**
   * Returns the scalar value as a BigDecimal.
   */
  public BigDecimal getBigDecimal() {
    if (this.type.typeId == DType.DTypeEnum.DECIMAL32) {
      return BigDecimal.valueOf(getInt(), -type.getScale());
    } else if (this.type.typeId == DType.DTypeEnum.DECIMAL64) {
      return BigDecimal.valueOf(getLong(), -type.getScale());
    } else if (this.type.typeId == DType.DTypeEnum.DECIMAL128) {
      return new BigDecimal(new BigInteger(getBigInteger()), -type.getScale());
    }
    throw new IllegalArgumentException("Couldn't getBigDecimal from nonDecimal scalar");
  }

  /**
   * Returns the scalar value as a Java string.
   */
  public String getJavaString() {
    return new String(getUTF8(getScalarHandle()), StandardCharsets.UTF_8);
  }

  /**
   * Returns the scalar value as UTF-8 data.
   */
  public byte[] getUTF8() {
    return getUTF8(getScalarHandle());
  }

  /**
   * Returns the scalar value as a ColumnView. Callers should close the returned ColumnView to
   * avoid memory leak.
   *
   * The returned ColumnView is only valid as long as the Scalar remains valid. If the Scalar
   * is closed before this ColumnView is closed, using this ColumnView will result in undefined
   * behavior.
   */
  public ColumnView getListAsColumnView() {
    assert DType.LIST.equals(type) : "Cannot get list for the vector of type " + type;
    return new ColumnView(getListAsColumnView(getScalarHandle()));
  }

  /**
   * Fetches views of children columns from struct scalar.
   * The returned ColumnViews should be closed appropriately. Otherwise, a native memory leak will occur.
   *
   * @return array of column views refer to children of struct scalar
   */
  public ColumnView[] getChildrenFromStructScalar() {
    assert DType.STRUCT.equals(type) : "Cannot get table for the vector of type " + type;

    long[] childHandles = getChildrenFromStructScalar(getScalarHandle());
    return ColumnView.getColumnViewsFromPointers(childHandles);
  }

  @Override
  public ColumnVector binaryOp(BinaryOp op, BinaryOperable rhs, DType outType) {
    if (rhs instanceof ColumnView) {
      ColumnView cvRhs = (ColumnView) rhs;
      return new ColumnVector(binaryOp(this, cvRhs, op, outType));
    } else {
      throw new IllegalArgumentException(rhs.getClass() + " is not supported as a binary op with " +
          "Scalar");
    }
  }

  static long binaryOp(Scalar lhs, ColumnView rhs, BinaryOp op, DType outputType) {
    return binaryOpSV(lhs.getScalarHandle(), rhs.getNativeView(),
        op.nativeId, outputType.typeId.getNativeId(), outputType.getScale());
  }

  private static native long binaryOpSV(long lhs, long rhs, int op, int dtype, int scale);

  /**
   * Divides this decimal scalar by each element of a decimal column. Each quotient is rounded to
   * the scale of {@code outType} using {@code mode}, unlike {@code div}, which truncates. This
   * scalar, {@code rhs} and {@code outType} may each be any decimal type, with any scale.
   *
   * An output element is null if this scalar is null, if the divisor element is null or zero, or
   * if the rounded quotient does not fit in {@code outType}.
   *
   * @param rhs     the divisor column
   * @param outType the decimal type and scale of the output
   * @param mode    the rounding mode to use
   * @return a new column of type {@code outType} containing the quotients
   * @throws CudfException if either input or {@code outType} is not a decimal type
   */
  public ColumnVector divideDecimal(ColumnView rhs, DType outType, RoundMode mode) {
    return divideDecimal(rhs, outType, mode, DecimalIntermediate.STANDARD);
  }

  /**
   * Divides this decimal scalar by each element of a decimal column, rescaling the dividend into
   * an intermediate value of the given width.
   * See {@link #divideDecimal(ColumnView, DType, RoundMode)} and
   * {@link ColumnView#divideDecimal(ColumnView, DType, RoundMode, DecimalIntermediate)}.
   *
   * @param rhs          the divisor column
   * @param outType      the decimal type and scale of the output
   * @param mode         the rounding mode to use
   * @param intermediate the width of the intermediate value
   * @return a new column of type {@code outType} containing the quotients
   * @throws CudfException if either input or {@code outType} is not a decimal type
   */
  public ColumnVector divideDecimal(ColumnView rhs, DType outType, RoundMode mode,
                                    DecimalIntermediate intermediate) {
    return new ColumnVector(divideDecimalSV(getScalarHandle(), rhs.getNativeView(),
        outType.typeId.getNativeId(), outType.getScale(), mode.nativeId, intermediate.nativeId));
  }

  /**
   * Divides this decimal scalar by each element of a decimal column, rounding HALF_UP.
   * See {@link #divideDecimal(ColumnView, DType, RoundMode)}.
   */
  public ColumnVector divideDecimal(ColumnView rhs, DType outType) {
    return divideDecimal(rhs, outType, RoundMode.HALF_UP);
  }

  private static native long divideDecimalSV(long lhs, long rhs, int dtype, int scale,
                                             int roundMode, int intermediate) throws CudfException;

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Scalar other = (Scalar) o;
    if (!type.equals(other.type)) return false;
    boolean valid = isValid();
    if (valid != other.isValid()) return false;
    if (!valid) return true;
    switch (type.typeId) {
    case EMPTY:
      return true;
    case BOOL8:
      return getBoolean() == other.getBoolean();
    case INT8:
    case UINT8:
      return getByte() == other.getByte();
    case INT16:
    case UINT16:
      return getShort() == other.getShort();
    case INT32:
    case UINT32:
    case TIMESTAMP_DAYS:
    case DECIMAL32:
      return getInt() == other.getInt();
    case FLOAT32:
      return getFloat() == other.getFloat();
    case FLOAT64:
      return getDouble() == other.getDouble();
    case INT64:
    case UINT64:
    case TIMESTAMP_SECONDS:
    case TIMESTAMP_MILLISECONDS:
    case TIMESTAMP_MICROSECONDS:
    case TIMESTAMP_NANOSECONDS:
    case DECIMAL64:
      return getLong() == other.getLong();
    case DECIMAL128:
      return getBigDecimal().equals(other.getBigDecimal());
    case STRING:
      return Arrays.equals(getUTF8(), other.getUTF8());
    case LIST:
      try (ColumnView viewMe = getListAsColumnView();
           ColumnView viewO = other.getListAsColumnView()) {
        return viewMe.equals(viewO);
      }
    default:
      throw new IllegalStateException("Unexpected type: " + type);
    }
  }

  @Override
  public int hashCode() {
    int valueHash = 0;
    if (isValid()) {
      switch (type.typeId) {
      case EMPTY:
        valueHash = 0;
        break;
      case BOOL8:
        valueHash = getBoolean() ? 1 : 0;
        break;
      case INT8:
      case UINT8:
        valueHash = getByte();
        break;
      case INT16:
      case UINT16:
        valueHash = getShort();
        break;
      case INT32:
      case UINT32:
      case TIMESTAMP_DAYS:
      case DECIMAL32:
      case DURATION_DAYS:
        valueHash = getInt();
        break;
      case INT64:
      case UINT64:
      case TIMESTAMP_SECONDS:
      case TIMESTAMP_MILLISECONDS:
      case TIMESTAMP_MICROSECONDS:
      case TIMESTAMP_NANOSECONDS:
      case DECIMAL64:
      case DURATION_MICROSECONDS:
      case DURATION_SECONDS:
      case DURATION_MILLISECONDS:
      case DURATION_NANOSECONDS:
        valueHash = Long.hashCode(getLong());
        break;
      case FLOAT32:
        valueHash = Float.hashCode(getFloat());
        break;
      case FLOAT64:
        valueHash = Double.hashCode(getDouble());
        break;
      case STRING:
        valueHash = Arrays.hashCode(getUTF8());
        break;
      case LIST:
        try (ColumnView v = getListAsColumnView()) {
          valueHash = v.hashCode();
        }
        break;
      case DECIMAL128:
        valueHash = getBigDecimal().hashCode();
        break;
      default:
        throw new IllegalStateException("Unknown scalar type: " + type);
      }
    }
    return Objects.hash(type, valueHash);
  }

  @Override
  public String toString() {
    StringBuilder sb = new StringBuilder("Scalar{type=");
    sb.append(type);
    if (getScalarHandle() != 0) {
      sb.append(" value=");
      switch (type.typeId) {
      case BOOL8:
        sb.append(getBoolean());
        break;
      case INT8:
        sb.append(getByte());
        break;
      case UINT8:
        sb.append(Byte.toUnsignedInt(getByte()));
        break;
      case INT16:
        sb.append(getShort());
        break;
      case UINT16:
        sb.append(Short.toUnsignedInt(getShort()));
        break;
      case INT32:
      case TIMESTAMP_DAYS:
        sb.append(getInt());
        break;
      case UINT32:
        sb.append(Integer.toUnsignedLong(getInt()));
        break;
      case INT64:
      case TIMESTAMP_SECONDS:
      case TIMESTAMP_MILLISECONDS:
      case TIMESTAMP_MICROSECONDS:
      case TIMESTAMP_NANOSECONDS:
        sb.append(getLong());
        break;
      case UINT64:
        sb.append(Long.toUnsignedString(getLong()));
        break;
      case FLOAT32:
        sb.append(getFloat());
        break;
      case FLOAT64:
        sb.append(getDouble());
        break;
      case STRING:
        sb.append('"');
        sb.append(getJavaString());
        sb.append('"');
        break;
      case DECIMAL32:
        // FALL THROUGH
      case DECIMAL64:
        // FALL THROUGH
      case DECIMAL128:
        sb.append(getBigDecimal());
        break;
      case LIST:
        try (ColumnView v = getListAsColumnView()) {
          // It's not easy to pull out the elements so just a simple string of some metadata.
          sb.append(v.toString());
        }
        break;
      default:
        throw new IllegalArgumentException("Unknown scalar type: " + type);
      }
    }

    sb.append("} (ID: ");
    sb.append(offHeap.id);
    sb.append(" ");
    sb.append(Long.toHexString(offHeap.scalarHandle));
    sb.append(")");
    return sb.toString();
  }


  /**
   * Repeat the given string scalar a number of times specified by the <code>repeatTimes</code>
   * parameter. If that parameter has a non-positive value, an empty (valid) string scalar will be
   * returned. An invalid input scalar will always result in an invalid output scalar regardless
   * of the value of <code>repeatTimes</code>.
   *
   * @param repeatTimes The number of times the input string is copied to the output.
   * @return The resulting scalar containing repeated result of the current string.
   */
  public Scalar repeatString(int repeatTimes) {
    return new Scalar(DType.STRING, repeatString(getScalarHandle(), repeatTimes));
  }

  private static byte[] convertDecimal128FromJavaToCudf(byte[] bytes) {
    byte[] finalBytes = new byte[DType.DTypeEnum.DECIMAL128.sizeInBytes];
    byte lastByte = bytes[0];
    //Convert to 2's complement representation and make sure the sign bit is extended correctly
    byte setByte = (lastByte & 0x80) > 0 ? (byte)0xff : (byte)0x00;
    for(int i = bytes.length; i < finalBytes.length; i++) {
      finalBytes[i] = setByte;
    }
    // After setting the sign bits, reverse the rest of the bytes for endianness
    for(int k = 0; k < bytes.length; k++) {
      finalBytes[k] = bytes[bytes.length - k - 1];
    }
    return finalBytes;
  }

  private void convertInPlaceToBigEndian(byte[] res) {
    assert ByteOrder.nativeOrder().equals(ByteOrder.LITTLE_ENDIAN);
    int i =0;
    int j = res.length -1;
    while (j > i) {
      byte tmp;
      tmp = res[j];
      res[j] = res[i];
      res[i] = tmp;
      j--;
      i++;
    }
  }

  /**
   * Holds the off-heap state of the scalar so it can be cleaned up, even if it is leaked.
   */
  private static class OffHeapState extends MemoryCleaner.Cleaner {
    private long scalarHandle;

    OffHeapState(long scalarHandle) {
      this.scalarHandle = scalarHandle;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      if (scalarHandle != 0) {
        if (logErrorIfNotClean) {
          LOG.error("A SCALAR WAS LEAKED(ID: " + id + " " + Long.toHexString(scalarHandle) + ")");
          logRefCountDebug("Leaked scalar");
        }
        try {
          closeScalar(scalarHandle);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          scalarHandle = 0;
        }
        return true;
      }
      return false;
    }

    @Override
    public boolean isClean() {
      return scalarHandle == 0;
    }
  }
}

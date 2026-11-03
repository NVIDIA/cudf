/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.StringJoiner;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/**
 * Similar to a ColumnVector, but the data is stored in host memory and accessible directly from
 * the JVM. This class holds references to off heap memory and is reference counted to know when
 * to release it.  Call close to decrement the reference count when you are done with the column,
 * and call incRefCount to increment the reference count.
 */
public final class HostColumnVector extends HostColumnVectorCore {
  /**
   * Interface to handle events for this HostColumnVector. Only invoked during
   * close, hence `onClosed` is the only event.
   */
  public interface EventHandler {
    /**
     * `onClosed` is invoked with the updated `refCount` during `close`.
     * The last invocation of `onClosed` will be with `refCount=0`.
     *
     * @note the callback is invoked with this `HostColumnVector`'s lock held.
     *
     * @param cv reference to the HostColumnVector we are closing
     * @param refCount the updated ref count for this HostColumnVector at
     *                 the time of invocation
     */
    void onClosed(HostColumnVector cv, int refCount);
  }

  /**
   * The size in bytes of an offset entry
   */
  static final int OFFSET_SIZE = DType.INT32.getSizeInBytes();

  private int refCount;
  private EventHandler eventHandler;

  /**
   * Create a new non-nested, fixed-width column vector with data populated on the host.
   * Convenience form of
   * {@link #HostColumnVector(DType, long, Optional, HostMemoryBuffer, HostMemoryBuffer, HostMemoryBuffer)}
   * for types that do not use an offsets buffer (i.e. all types except STRING).
   * <p>
   * NOTE: Buffers may still be receiving asynchronous device writes. This constructor
   * only stores references; callers must sync the relevant CUDA stream before reading.
   *
   * @param type               the type of the vector; must not be LIST, and must not be
   *                           STRING unless rows count is 0
   * @param rows               the number of rows in the vector
   * @param nullCount          the number of nulls in the vector
   * @param hostDataBuffer     the host side data for the vector
   * @param hostValidityBuffer Arrow-like validity buffer, 1 bit per row with padding for
   *                           64-bit alignment; may be null if nullCount is 0
   * @throws IllegalArgumentException if type is LIST, or if type is STRING with rows > 0
   * @throws IllegalStateException    if nullCount is greater than 0 but hostValidityBuffer is null
   */
  public HostColumnVector(DType type, long rows, Optional<Long> nullCount,
                          HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer) {
    this(type, rows, nullCount, hostDataBuffer, hostValidityBuffer, null);
  }

  /**
   * Create a new column vector with data populated on the host.
   * @param type               the type of the vector
   * @param rows               the number of rows in the vector.
   * @param nullCount          the number of nulls in the vector.
   * @param hostDataBuffer     The host side data for the vector. In the case of STRING
   *                           this is the string data stored as bytes.
   * @param hostValidityBuffer Arrow-like validity buffer 1 bit per row, with padding for
   *                           64-bit alignment.
   * @param offsetBuffer       only valid for STRING this is the offsets into
   *                           the hostDataBuffer indicating the start and end of a string
   *                           entry. It should be (rows + 1) ints.
   * @param nestedHcv          list of child HostColumnVectorCore(s) for complex types
   */

  //Constructor for lists and struct
  public HostColumnVector(DType type, long rows, Optional<Long> nullCount,
                          HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer,
                          HostMemoryBuffer offsetBuffer, List<HostColumnVectorCore> nestedHcv) {
    // NOTE: This constructor MUST NOT examine the contents of any host buffers, as they may be
    //       asynchronously written by the device.
    super(type, rows, nullCount, hostDataBuffer, hostValidityBuffer, offsetBuffer, nestedHcv);
    refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * Create a new non-nested column vector with data populated on the host.
   * Supports fixed-width types and STRING. For LIST or STRUCT types use
   * {@link #HostColumnVector(DType, long, Optional, HostMemoryBuffer, HostMemoryBuffer, HostMemoryBuffer, List)}.
   * <p>
   * NOTE: Buffers may still be receiving asynchronous device writes. This constructor
   * only stores references; callers must sync the relevant CUDA stream before reading.
   *
   * @param type               the type of the vector; must not be LIST
   * @param rows               the number of rows in the vector
   * @param nullCount          the number of nulls in the vector
   * @param hostDataBuffer     the host side data for the vector; for STRING this is the
   *                           string data stored as bytes
   * @param hostValidityBuffer Arrow-like validity buffer, 1 bit per row with padding for
   *                           64-bit alignment; may be null if nullCount is 0
   * @param offsetBuffer       for STRING, the offsets into hostDataBuffer indicating the
   *                           start and end of each entry, must be (rows + 1) ints; may
   *                           be null when rows count is 0; must be null for all other non-LIST
   *                           types
   * @throws IllegalArgumentException if type is LIST; if offsetBuffer is null for a STRING
   *                                  column with rows &gt; 0; or if offsetBuffer is non-null
   *                                  for a non-STRING type
   * @throws IllegalStateException    if nullCount is greater than 0 but hostValidityBuffer is null
   */
  public HostColumnVector(DType type, long rows, Optional<Long> nullCount,
                          HostMemoryBuffer hostDataBuffer, HostMemoryBuffer hostValidityBuffer,
                          HostMemoryBuffer offsetBuffer) {
    // NOTE: This constructor MUST NOT examine the contents of any host buffers, as they may be
    //       asynchronously written by the device.
    super(type, rows, nullCount, hostDataBuffer, hostValidityBuffer, offsetBuffer, new ArrayList<>());
    if (type.equals(DType.LIST)) {
      throw new IllegalArgumentException(
          "This constructor should not be used for LIST type. Use the nested constructor instead");
    }
    if (type.equals(DType.STRING) && offsetBuffer == null && rows > 0) {
      throw new IllegalArgumentException("offsetBuffer is required for non-empty STRING columns");
    }
    if (nullCount.isPresent() && nullCount.get() > 0 && hostValidityBuffer == null) {
      throw new IllegalStateException("Buffer cannot have a nullCount without a validity buffer");
    }
    if (!type.equals(DType.STRING) && offsetBuffer != null) {
      throw new IllegalArgumentException("offsetBuffer is only supported for STRING");
    }
    refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * Set an event handler for this host vector. This method can be invoked with
   * null to unset the handler.
   *
   * @param newHandler - the EventHandler to use from this point forward
   * @return the prior event handler, or null if not set.
   */
  public synchronized EventHandler setEventHandler(EventHandler newHandler) {
    EventHandler prev = this.eventHandler;
    this.eventHandler = newHandler;
    return prev;
  }

  /**
   * Returns the current event handler for this HostColumnVector or null if no
   * handler is associated.
   */
  public synchronized EventHandler getEventHandler() {
    return this.eventHandler;
  }

  /**
   * This is a really ugly API, but it is possible that the lifecycle of a column of
   * data may not have a clear lifecycle thanks to java and GC. This API informs the leak
   * tracking code that this is expected for this column, and big scary warnings should
   * not be printed when this happens.
   */
  public void noWarnLeakExpected() {
    offHeap.noWarnLeakExpected();
  }

  /**
   * Close this Vector and free memory allocated for HostMemoryBuffer and DeviceMemoryBuffer
   */
  @Override
  public synchronized void close() {
    refCount--;
    offHeap.delRef();
    try {
      if (refCount == 0) {
        offHeap.clean(false);
        for (HostColumnVectorCore child : children) {
          child.close();
        }
      } else if (refCount < 0) {
        offHeap.logRefCountDebug("double free " + this);
        throw new IllegalStateException("Close called too many times " + this);
      }
    } finally {
      if (eventHandler != null) {
        eventHandler.onClosed(this, refCount);
      }
    }
  }

  @Override
  public String toString() {
    return "HostColumnVector{" +
        "rows=" + rows +
        ", type=" + type +
        ", nullCount=" + nullCount +
        ", offHeap=" + offHeap +
        '}';
  }

  /////////////////////////////////////////////////////////////////////////////
  // METADATA ACCESS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Increment the reference count for this column.  You need to call close on this
   * to decrement the reference count again.
   */
  public HostColumnVector incRefCount() {
    return incRefCountInternal(false);
  }

  private synchronized HostColumnVector incRefCountInternal(boolean isFirstTime) {
    offHeap.addRef();
    if (refCount <= 0 && !isFirstTime) {
      offHeap.logRefCountDebug("INC AFTER CLOSE " + this);
      throw new IllegalStateException("Column is already closed");
    }
    refCount++;
    return this;
  }

  /**
   * Returns this column's current refcount
   */
  public synchronized int getRefCount() {
    return refCount;
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATA MOVEMENT
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Copy the data to the device.
   */
  public ColumnVector copyToDevice() {
    if (rows == 0) {
      if (type.isNestedType()) {
        return ColumnView.NestedColumnVector.createColumnVector(type, 0,
                null, null, null, Optional.of(0L), children);
      } else {
        return new ColumnVector(type, 0, Optional.of(0L), null, null, null);
      }
    }
    // The simplest way is just to copy the buffers and pass them down.
    DeviceMemoryBuffer data = null;
    DeviceMemoryBuffer valid = null;
    DeviceMemoryBuffer offsets = null;
    try {
      if (!type.isNestedType()) {
        HostMemoryBuffer hdata = this.offHeap.data;
        if (hdata != null) {
          long dataLen = rows * type.getSizeInBytes();
          if (type.equals(DType.STRING)) {
            // This needs a different type
            dataLen = getEndStringOffset(rows - 1);
            if (dataLen == 0 && getNullCount() == 0) {
              // This is a work around to an issue where a column of all empty strings must have at
              // least one byte or it will not be interpreted correctly.
              dataLen = 1;
            }
          }
          data = DeviceMemoryBuffer.allocate(dataLen);
          data.copyFromHostBuffer(hdata, 0, dataLen);
        }
        HostMemoryBuffer hvalid = this.offHeap.valid;
        if (hvalid != null) {
          long validLen = ColumnView.getValidityBufferSize((int) rows);
          valid = DeviceMemoryBuffer.allocate(validLen);
          valid.copyFromHostBuffer(hvalid, 0, validLen);
        }

        HostMemoryBuffer hoff = this.offHeap.offsets;
        if (hoff != null) {
          long offsetsLen = OFFSET_SIZE * (rows + 1);
          offsets = DeviceMemoryBuffer.allocate(offsetsLen);
          offsets.copyFromHostBuffer(hoff, 0, offsetsLen);
        }

        ColumnVector ret = new ColumnVector(type, rows, nullCount, data, valid, offsets);
        data = null;
        valid = null;
        offsets = null;
        return ret;
      } else {
        return ColumnView.NestedColumnVector.createColumnVector(
            type, (int) rows, offHeap.data, offHeap.valid, offHeap.offsets, nullCount, children);
      }
    } finally {
      if (data != null) {
        data.close();
      }
      if (valid != null) {
        valid.close();
      }
      if (offsets != null) {
        offsets.close();
      }
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // BUILDER
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Create a new Builder to hold the specified number of rows.  Be sure to close the builder when
   * done with it. Please try to use {@see #build(int, Consumer)} instead to avoid needing to
   * close the builder.
   * @param type the type of vector to build.
   * @param rows the number of rows this builder can hold
   * @return the builder to use.
   */
  public static Builder builder(DType type, int rows) {
    return new Builder(type, rows, 0);
  }

  /**
   * Create a new Builder to hold the specified number of rows and with enough space to hold the
   * given amount of string data. Be sure to close the builder when done with it. Please try to
   * use {@see #build(int, int, Consumer)} instead to avoid needing to close the builder.
   * @param rows the number of rows this builder can hold
   * @param stringBufferSize the size of the string buffer to allocate.
   * @return the builder to use.
   */
  public static Builder builder(int rows, long stringBufferSize) {
    return new HostColumnVector.Builder(DType.STRING, rows, stringBufferSize);
  }

  /**
   * Create a new vector.
   * @param type       the type of vector to build.
   * @param rows       maximum number of rows that the vector can hold.
   * @param init       what will initialize the vector.
   * @return the created vector.
   */
  public static HostColumnVector build(DType type, int rows, Consumer<Builder> init) {
    try (HostColumnVector.Builder builder = builder(type, rows)) {
      init.accept(builder);
      return builder.build();
    }
  }

  public static HostColumnVector build(int rows, long stringBufferSize, Consumer<Builder> init) {
    try (HostColumnVector.Builder builder = builder(rows, stringBufferSize)) {
      init.accept(builder);
      return builder.build();
    }
  }

  public static<T> HostColumnVector fromLists(DataType dataType, List<T>... values) {
    try (ColumnBuilder cb = new ColumnBuilder(dataType, values.length)) {
      cb.appendLists(values);
      return cb.build();
    }
  }

  public static HostColumnVector fromStructs(DataType dataType,
                                             List<StructData> values) {
    try (ColumnBuilder cb = new ColumnBuilder(dataType, values.size())) {
      cb.appendStructValues(values);
      return cb.build();
    }
  }

  public static HostColumnVector fromStructs(DataType dataType, StructData... values) {
    try (ColumnBuilder cb = new ColumnBuilder(dataType, values.length)) {
      cb.appendStructValues(values);
      return cb.build();
    }
  }

  public static HostColumnVector emptyStructs(DataType dataType, long rows) {
    StructData sd = new StructData();
    try (ColumnBuilder cb = new ColumnBuilder(dataType, rows)) {
      for (long i = 0; i < rows; i++) {
        cb.append(sd);
      }
      return cb.build();
    }
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector boolFromBytes(byte... values) {
    return build(DType.BOOL8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromBytes(byte... values) {
    return build(DType.INT8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned byte type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromUnsignedBytes(byte... values) {
    return build(DType.UINT8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromShorts(short... values) {
    return build(DType.INT16, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned short type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromUnsignedShorts(short... values) {
    return build(DType.UINT16, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector durationNanosecondsFromLongs(long... values) {
    return build(DType.DURATION_NANOSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector durationMicrosecondsFromLongs(long... values) {
    return build(DType.DURATION_MICROSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector durationMillisecondsFromLongs(long... values) {
    return build(DType.DURATION_MILLISECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector durationSecondsFromLongs(long... values) {
    return build(DType.DURATION_SECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector durationDaysFromInts(int... values) {
    return build(DType.DURATION_DAYS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromInts(int... values) {
    return build(DType.INT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned int type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromUnsignedInts(int... values) {
    return build(DType.UINT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromLongs(long... values) {
    return build(DType.INT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned long type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromUnsignedLongs(long... values) {
    return build(DType.UINT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromFloats(float... values) {
    return build(DType.FLOAT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector fromDoubles(double... values) {
    return build(DType.FLOAT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector daysFromInts(int... values) {
    return build(DType.TIMESTAMP_DAYS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector timestampSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_SECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector timestampMilliSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_MILLISECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector timestampMicroSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_MICROSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static HostColumnVector timestampNanoSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new decimal vector from unscaled values (int array) and scale.
   * The created vector is of type DType.DECIMAL32, whose max precision is 9.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static HostColumnVector decimalFromInts(int scale, int... values) {
    return build(DType.create(DType.DTypeEnum.DECIMAL32, scale), values.length, (b) -> b.appendUnscaledDecimalArray(values));
  }

  /**
   * Create a new decimal vector from boxed unscaled values (Integer array) and scale.
   * The created vector is of type DType.DECIMAL32, whose max precision is 9.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static HostColumnVector decimalFromBoxedInts(int scale, Integer... values) {
    return build(DType.create(DType.DTypeEnum.DECIMAL32, scale), values.length, (b) -> {
      for (Integer v : values) {
        if (v == null) {
          b.appendNull();
        } else {
          b.appendUnscaledDecimal(v);
        }
      }
    });
  }

  /**
   * Create a new decimal vector from unscaled values (long array) and scale.
   * The created vector is of type DType.DECIMAL64, whose max precision is 18.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static HostColumnVector decimalFromLongs(int scale, long... values) {
    return build(DType.create(DType.DTypeEnum.DECIMAL64, scale), values.length, (b) -> b.appendUnscaledDecimalArray(values));
  }

  /**
   * Create a new decimal vector from boxed unscaled values (Long array) and scale.
   * The created vector is of type DType.DECIMAL64, whose max precision is 18.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static HostColumnVector decimalFromBoxedLongs(int scale, Long... values) {
    return build(DType.create(DType.DTypeEnum.DECIMAL64, scale), values.length, (b) -> {
      for (Long v : values) {
        if (v == null) {
          b.appendNull();
        } else {
          b.appendUnscaledDecimal(v);
        }
      }
    });
  }

  /**
   * Create a new decimal vector from unscaled values (BigInteger array) and scale.
   * The created vector is of type DType.DECIMAL128.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static HostColumnVector decimalFromBigIntegers(int scale, BigInteger... values) {
    return build(DType.create(DType.DTypeEnum.DECIMAL128, scale), values.length, (b) -> {
      for (BigInteger v : values) {
        if (v == null) {
          b.appendNull();
        } else {
          b.appendUnscaledDecimal(v);
        }
      }
    });
  }

  /**
   * Create a new decimal vector from double floats with specific DecimalType and RoundingMode.
   * All doubles will be rescaled if necessary, according to scale of input DecimalType and RoundingMode.
   * If any overflow occurs in extracting integral part, an IllegalArgumentException will be thrown.
   * This API is inefficient because of slow double -> decimal conversion, so it is mainly for testing.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static HostColumnVector decimalFromDoubles(DType type, RoundingMode mode, double... values) {
    assert type.isDecimalType();
    if (type.typeId == DType.DTypeEnum.DECIMAL64) {
      long[] data = new long[values.length];
      for (int i = 0; i < values.length; i++) {
        BigDecimal dec = BigDecimal.valueOf(values[i]).setScale(-type.getScale(), mode);
        data[i] = dec.unscaledValue().longValueExact();
      }
      return build(type, values.length, (b) -> b.appendUnscaledDecimalArray(data));
    } else {
      int[] data = new int[values.length];
      for (int i = 0; i < values.length; i++) {
        BigDecimal dec = BigDecimal.valueOf(values[i]).setScale(-type.getScale(), mode);
        data[i] = dec.unscaledValue().intValueExact();
      }
      return build(type, values.length, (b) -> b.appendUnscaledDecimalArray(data));
    }
  }

  /**
   * Create a new string vector from the given values.  This API
   * supports inline nulls. This is really intended to be used only for testing as
   * it is slow and memory intensive to translate between java strings and UTF8 strings.
   */
  public static HostColumnVector fromStrings(String... values) {
    int rows = values.length;
    long nullCount = 0;
    // How many bytes do we need to hold the data.  Sorry this is really expensive
    long bufferSize = 0;
    for (String s: values) {
      if (s == null) {
        nullCount++;
      } else {
        bufferSize += s.getBytes(StandardCharsets.UTF_8).length;
      }
    }
    if (nullCount > 0) {
      return build(rows, bufferSize, (b) -> b.appendBoxed(values));
    }
    return build(rows, bufferSize, (b) -> {
      for (String s: values) {
        b.append(s);
      }
    });
  }

  /**
   * Create a new string vector from the given values.  This API
   * supports inline nulls.
   */
  public static HostColumnVector fromUTF8Strings(byte[]... values) {
    int rows = values.length;
    long nullCount = 0;
    long bufferSize = 0;
    // How many bytes do we need to hold the data.
    for (byte[] s: values) {
      if (s == null) {
        nullCount++;
      } else {
        bufferSize += s.length;
      }
    }

    BiConsumer<Builder, byte[]> appendUTF8 = nullCount == 0 ?
      (b, s) -> b.appendUTF8String(s) :
      (b, s) -> {
        if (s == null) {
          b.appendNull();
        } else {
          b.appendUTF8String(s);
        }
      };

    return build(rows, bufferSize, (b) -> {
      for (byte[] s: values) {
        appendUTF8.accept(b, s);
      }
    });
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than building from primitive array of unscaledValues.
   * Notice:
   *  1. Input values will be rescaled with min scale (max scale in terms of java.math.BigDecimal),
   *  which avoids potential precision loss due to rounding. But there exists risk of precision overflow.
   *  2. The scale will be zero if all input values are null.
   */
  public static HostColumnVector fromDecimals(BigDecimal... values) {
    // 1. Fetch the element with max precision (maxDec). Fill with ZERO if inputs is empty.
    // 2. Fetch the max scale. Fill with ZERO if inputs is empty.
    // 3. Rescale the maxDec with the max scale, so to come out the max precision capacity we need.
    BigDecimal maxDec = Arrays.stream(values).filter(Objects::nonNull)
        .max(Comparator.comparingInt(BigDecimal::precision))
        .orElse(BigDecimal.ZERO);
    int maxScale = Arrays.stream(values).filter(Objects::nonNull)
        .map(decimal -> decimal.scale())
        .max(Comparator.naturalOrder())
        .orElse(0);
    maxDec = maxDec.setScale(maxScale, RoundingMode.UNNECESSARY);

    return build(DType.fromJavaBigDecimal(maxDec), values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedBooleans(Boolean... values) {
    return build(DType.BOOL8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedBytes(Byte... values) {
    return build(DType.INT8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   * <p>
   * Java does not have an unsigned byte type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromBoxedUnsignedBytes(Byte... values) {
    return build(DType.UINT8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedShorts(Short... values) {
    return build(DType.INT16, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   * <p>
   * Java does not have an unsigned short type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromBoxedUnsignedShorts(Short... values) {
    return build(DType.UINT16, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector durationNanosecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_NANOSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector durationMicrosecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_MICROSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector durationMillisecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_MILLISECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector durationSecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_SECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector durationDaysFromBoxedInts(Integer... values) {
    return build(DType.DURATION_DAYS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedInts(Integer... values) {
    return build(DType.INT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   * <p>
   * Java does not have an unsigned int type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromBoxedUnsignedInts(Integer... values) {
    return build(DType.UINT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedLongs(Long... values) {
    return build(DType.INT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   * <p>
   * Java does not have an unsigned long type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static HostColumnVector fromBoxedUnsignedLongs(Long... values) {
    return build(DType.UINT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedFloats(Float... values) {
    return build(DType.FLOAT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector fromBoxedDoubles(Double... values) {
    return build(DType.FLOAT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampDaysFromBoxedInts(Integer... values) {
    return build(DType.TIMESTAMP_DAYS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_SECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampMilliSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_MILLISECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampMicroSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_MICROSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static HostColumnVector timestampNanoSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Build
   */

  public static final class ColumnBuilder implements AutoCloseable {

    private DType type;
    private HostMemoryBuffer data;
    private HostMemoryBuffer valid;
    private HostMemoryBuffer offsets;
    private long nullCount = 0l;
    //TODO nullable currently not used
    private boolean nullable;
    private long rows;
    private long estimatedRows;
    private long rowCapacity = 0L;
    private long validCapacity = 0L;
    private boolean built = false;
    private List<ColumnBuilder> childBuilders = new ArrayList<>();
    private Runnable nullHandler;

    // The value of currentIndex can't exceed Int32.Max. Storing currentIndex as a long is to
    // adapt HostMemoryBuffer.setXXX, which requires a long offset.
    private long currentIndex = 0;
    // Only for Strings: pointer of the byte (data) buffer
    private int currentStringByteIndex = 0;
    // Use bit shift instead of multiply to transform row offset to byte offset
    private int bitShiftBySize = 0;
    private static final int bitShiftByOffset = (int)(Math.log(OFFSET_SIZE) / Math.log(2));

    public ColumnBuilder(HostColumnVector.DataType type, long estimatedRows) {
      this.type = type.getType();
      this.nullable = type.isNullable();
      this.rows = 0;
      this.estimatedRows = Math.max(estimatedRows, 1L);
      this.bitShiftBySize = (int)(Math.log(this.type.getSizeInBytes()) / Math.log(2));

      // initialize the null handler according to the data type
      this.setupNullHandler();

      for (int i = 0; i < type.getNumChildren(); i++) {
        childBuilders.add(new ColumnBuilder(type.getChild(i), estimatedRows));
      }
    }

    private void setupNullHandler() {
      if (this.type == DType.LIST) {
        this.nullHandler = () -> {
          this.growListBuffersAndRows();
          this.growValidBuffer();
          setNullAt(currentIndex++);
          offsets.setInt(currentIndex << bitShiftByOffset, childBuilders.get(0).getCurrentIndex());
        };
      } else if (this.type == DType.STRING) {
        this.nullHandler = () -> {
          this.growStringBuffersAndRows(0);
          this.growValidBuffer();
          setNullAt(currentIndex++);
          offsets.setInt(currentIndex << bitShiftByOffset, currentStringByteIndex);
        };
      } else if (this.type == DType.STRUCT) {
        this.nullHandler = () -> {
          this.growStructBuffersAndRows();
          this.growValidBuffer();
          setNullAt(currentIndex++);
          for (ColumnBuilder childBuilder : childBuilders) {
            childBuilder.appendNull();
          }
        };
      } else {
        this.nullHandler = () -> {
          this.growFixedWidthBuffersAndRows();
          this.growValidBuffer();
          setNullAt(currentIndex++);
        };
      }
    }

    public HostColumnVector build() {
      List<HostColumnVectorCore> hostColumnVectorCoreList = new ArrayList<>();
      for (ColumnBuilder childBuilder : childBuilders) {
        hostColumnVectorCoreList.add(childBuilder.buildNestedInternal());
      }
      // Aligns the valid buffer size with other buffers in terms of row size, because it grows lazily.
      if (valid != null) {
        growValidBuffer();
      }
      HostColumnVector hostColumnVector = new HostColumnVector(type, rows, Optional.of(nullCount), data, valid, offsets,
          hostColumnVectorCoreList);
      built = true;
      return hostColumnVector;
    }

    private HostColumnVectorCore buildNestedInternal() {
      List<HostColumnVectorCore> hostColumnVectorCoreList = new ArrayList<>();
      for (ColumnBuilder childBuilder : childBuilders) {
        hostColumnVectorCoreList.add(childBuilder.buildNestedInternal());
      }
      // Aligns the valid buffer size with other buffers in terms of row size, because it grows lazily.
      if (valid != null) {
        growValidBuffer();
      }
      return new HostColumnVectorCore(type, rows, Optional.of(nullCount), data, valid, offsets, hostColumnVectorCoreList);
    }

    public ColumnBuilder appendLists(List... inputLists) {
      for (List inputList : inputLists) {
        // one row
        append(inputList);
      }
      return this;
    }

    public ColumnBuilder appendStructValues(List<StructData> inputList) {
      for (StructData structInput : inputList) {
        // one row
        append(structInput);
      }
      return this;
    }

    public ColumnBuilder appendStructValues(StructData... inputList) {
      for (StructData structInput : inputList) {
        append(structInput);
      }
      return this;
    }

    /**
     * Grows valid buffer lazily. The valid buffer won't be materialized until the first null
     * value appended. This method reuses the rowCapacity to track the sizes of column.
     * Therefore, please call specific growBuffer method to update rowCapacity before calling
     * this method.
     */
    private void growValidBuffer() {
      if (valid == null) {
        long maskBytes = ColumnView.getValidityBufferSize((int) rowCapacity);
        valid = HostMemoryBuffer.allocate(maskBytes);
        valid.setMemory(0, valid.length, (byte) 0xFF);
        validCapacity = rowCapacity;
        return;
      }
      if (validCapacity < rowCapacity) {
        long maskBytes = ColumnView.getValidityBufferSize((int) rowCapacity);
        HostMemoryBuffer newValid = HostMemoryBuffer.allocate(maskBytes);
        newValid.setMemory(0, newValid.length, (byte) 0xFF);
        valid = copyBuffer(newValid, valid);
        validCapacity = rowCapacity;
      }
    }

    /**
     * A method automatically grows data buffer for fixed-width columns as needed along with
     * incrementing the row counts. Please call this method before appending any value or null.
     */
    private void growFixedWidthBuffersAndRows() {
      growFixedWidthBuffersAndRows(1);
    }

    /**
     * A method automatically grows data buffer for fixed-width columns for a given size as needed
     * along with incrementing the row counts. Please call this method before appending
     * multiple values or nulls.
     */
    private void growFixedWidthBuffersAndRows(int numRows) {
      assert rows + numRows <= Integer.MAX_VALUE : "Row count cannot go over Integer.MAX_VALUE";
      rows += numRows;

      if (data == null) {
        long neededSize = Math.max(rows, estimatedRows);
        data = HostMemoryBuffer.allocate(neededSize << bitShiftBySize);
        rowCapacity = neededSize;
      } else if (rows > rowCapacity) {
        long neededSize = Math.max(rows, rowCapacity * 2);
        long newCap = Math.min(neededSize, Integer.MAX_VALUE - 1);
        data = copyBuffer(HostMemoryBuffer.allocate(newCap << bitShiftBySize), data);
        rowCapacity = newCap;
      }
    }

    /**
     * A method automatically grows offsets buffer for list columns as needed along with
     * incrementing the row counts. Please call this method before appending any value or null.
     */
    private void growListBuffersAndRows() {
      assert rows + 2 <= Integer.MAX_VALUE : "Row count cannot go over Integer.MAX_VALUE";
      rows++;

      if (offsets == null) {
        offsets = HostMemoryBuffer.allocate((estimatedRows + 1) << bitShiftByOffset);
        offsets.setInt(0, 0);
        rowCapacity = estimatedRows;
      } else if (rows > rowCapacity) {
        long newCap = Math.min(rowCapacity * 2, Integer.MAX_VALUE - 2);
        offsets = copyBuffer(HostMemoryBuffer.allocate((newCap + 1) << bitShiftByOffset), offsets);
        rowCapacity = newCap;
      }
    }

    /**
     * A method automatically grows offsets and data buffer for string columns as needed along with
     * incrementing the row counts. Please call this method before appending any value or null.
     *
     * @param stringLength number of bytes required by the next row
     */
    private void growStringBuffersAndRows(int stringLength) {
      assert rows + 2 <= Integer.MAX_VALUE : "Row count cannot go over Integer.MAX_VALUE";
      rows++;

      if (offsets == null) {
        // Initialize data buffer with at least 1 byte in case the first appended value is null.
        data = HostMemoryBuffer.allocate(Math.max(1, stringLength));
        offsets = HostMemoryBuffer.allocate((estimatedRows + 1) << bitShiftByOffset);
        offsets.setInt(0, 0);
        rowCapacity = estimatedRows;
        return;
      }

      if (rows > rowCapacity) {
        long newCap = Math.min(rowCapacity * 2, Integer.MAX_VALUE - 2);
        offsets = copyBuffer(HostMemoryBuffer.allocate((newCap + 1) << bitShiftByOffset), offsets);
        rowCapacity = newCap;
      }

      long currentLength = currentStringByteIndex + stringLength;
      if (currentLength > data.length) {
        long requiredLength = data.length;
        do {
          requiredLength = requiredLength * 2;
        } while (currentLength > requiredLength);
        data = copyBuffer(HostMemoryBuffer.allocate(requiredLength), data);
      }
    }

    /**
     * For struct columns, we only need to update rows and rowCapacity (for the growth of
     * valid buffer), because struct columns hold no buffer itself.
     * Please call this method before appending any value or null.
     */
    private void growStructBuffersAndRows() {
      assert rows + 1 <= Integer.MAX_VALUE : "Row count cannot go over Integer.MAX_VALUE";
      rows++;

      if (rowCapacity == 0) {
        rowCapacity = estimatedRows;
      } else if (rows > rowCapacity) {
        rowCapacity = Math.min(rowCapacity * 2, Integer.MAX_VALUE - 1);
      }
    }

    private HostMemoryBuffer copyBuffer(HostMemoryBuffer targetBuffer, HostMemoryBuffer buffer) {
      try {
        targetBuffer.copyFromHostBuffer(0, buffer, 0, buffer.length);
        buffer.close();
        buffer = targetBuffer;
        targetBuffer = null;
      } finally {
        if (targetBuffer != null) {
          targetBuffer.close();
        }
      }
      return buffer;
    }

    /**
     * Method that sets the null bit in the validity vector
     * @param index the row index at which the null is marked
     */
    private void setNullAt(long index) {
      assert index < rows : "Index for null value should fit the column with " + rows + " rows";
      nullCount += BitVectorHelper.setNullAt(valid, index);
    }

    public final ColumnBuilder appendNull() {
      nullHandler.run();
      return this;
    }

    //For structs
    private ColumnBuilder append(StructData structData) {
      assert type.isNestedType();
      if (type.equals(DType.STRUCT)) {
        if (structData == null || structData.isNull()) {
          return appendNull();
        } else {
          for (int i = 0; i < structData.getNumFields(); i++) {
            ColumnBuilder childBuilder = childBuilders.get(i);
            appendChildOrNull(childBuilder, structData.getField(i));
          }
          endStruct();
        }
      }
      return this;
    }

    private boolean allChildrenHaveSameIndex() {
      if (childBuilders.size() > 0) {
        int expected = childBuilders.get(0).getCurrentIndex();
        for (ColumnBuilder child: childBuilders) {
          if (child.getCurrentIndex() != expected) {
            return false;
          }
        }
      }
      return true;
    }

    /**
     * If you want to build up a struct column you can get each child `builder.getChild(N)` and
     * append to all of them, then when you are done call `endStruct` to update this builder.
     * Do not start to append to the child and then append a null to this without ending the struct
     * first or you might not get the results that you expected.
     * @return this for chaining.
     */
    public ColumnBuilder endStruct() {
      assert type.equals(DType.STRUCT) : "This only works for structs";
      assert allChildrenHaveSameIndex() : "Appending structs data appears to be off " +
          childBuilders + " should all have the same currentIndex " + type;
      growStructBuffersAndRows();
      currentIndex++;
      return this;
    }

    /**
     * If you want to build up a list column you can get `builder.getChild(0)` and append to than,
     * then when you are done call `endList` and everything that was appended to that builder
     * will now be in the next list. Do not start to append to the child and then append a null
     * to this without ending the list first or you might not get the results that you expected.
     * @return this for chaining.
     */
    public ColumnBuilder endList() {
      assert type.equals(DType.LIST);
      growListBuffersAndRows();
      offsets.setInt(++currentIndex << bitShiftByOffset, childBuilders.get(0).getCurrentIndex());
      return this;
    }

    // For lists
    private <T> ColumnBuilder append(List<T> inputList) {
      if (inputList == null) {
        appendNull();
      } else {
        ColumnBuilder childBuilder = childBuilders.get(0);
        for (Object listElement : inputList) {
          appendChildOrNull(childBuilder, listElement);
        }
        endList();
      }
      return this;
    }

    private void appendChildOrNull(ColumnBuilder childBuilder, Object listElement) {
      if (listElement == null) {
        childBuilder.appendNull();
      } else if (listElement instanceof Integer) {
        childBuilder.append((Integer) listElement);
      } else if (listElement instanceof String) {
        childBuilder.append((String) listElement);
      }  else if (listElement instanceof Double) {
        childBuilder.append((Double) listElement);
      } else if (listElement instanceof Float) {
        childBuilder.append((Float) listElement);
      } else if (listElement instanceof Boolean) {
        childBuilder.append((Boolean) listElement);
      } else if (listElement instanceof Long) {
        childBuilder.append((Long) listElement);
      } else if (listElement instanceof Byte) {
        childBuilder.append((Byte) listElement);
      } else if (listElement instanceof Short) {
        childBuilder.append((Short) listElement);
      } else if (listElement instanceof BigDecimal) {
        childBuilder.append((BigDecimal) listElement);
      } else if (listElement instanceof BigInteger) {
        childBuilder.append((BigInteger) listElement);
      } else if (listElement instanceof List) {
        childBuilder.append((List<?>) listElement);
      } else if (listElement instanceof StructData) {
        childBuilder.append((StructData) listElement);
      } else if (listElement instanceof byte[]) {
        childBuilder.appendUTF8String((byte[]) listElement);
      } else {
        throw new IllegalStateException("Unexpected element type: " + listElement.getClass());
      }
    }

    @Deprecated
    public void incrCurrentIndex() {
      currentIndex =  currentIndex + 1;
    }

    public int getCurrentIndex() {
      return (int) currentIndex;
    }

    @Deprecated
    public int getCurrentByteIndex() {
      return currentStringByteIndex;
    }

    public final ColumnBuilder append(byte value) {
      growFixedWidthBuffersAndRows();
      assert type.isBackedByByte();
      assert currentIndex < rows;
      data.setByte(currentIndex++ << bitShiftBySize, value);
      return this;
    }

    public final ColumnBuilder append(short value) {
      growFixedWidthBuffersAndRows();
      assert type.isBackedByShort();
      assert currentIndex < rows;
      data.setShort(currentIndex++ << bitShiftBySize, value);
      return this;
    }

    public final ColumnBuilder append(int value) {
      growFixedWidthBuffersAndRows();
      assert type.isBackedByInt();
      assert currentIndex < rows;
      data.setInt(currentIndex++ << bitShiftBySize, value);
      return this;
    }

    public final ColumnBuilder append(long value) {
      growFixedWidthBuffersAndRows();
      assert type.isBackedByLong();
      assert currentIndex < rows;
      data.setLong(currentIndex++ << bitShiftBySize, value);
      return this;
    }

    public final ColumnBuilder append(float value) {
      growFixedWidthBuffersAndRows();
      assert type.equals(DType.FLOAT32);
      assert currentIndex < rows;
      data.setFloat(currentIndex++ << bitShiftBySize, value);
      return this;
    }

    public final ColumnBuilder append(double value) {
      growFixedWidthBuffersAndRows();
      assert type.equals(DType.FLOAT64);
      assert currentIndex < rows;
      data.setDouble(currentIndex++ << bitShiftBySize, value);
      return this;
    }

    public final ColumnBuilder append(boolean value) {
      growFixedWidthBuffersAndRows();
      assert type.equals(DType.BOOL8);
      assert currentIndex < rows;
      data.setBoolean(currentIndex++ << bitShiftBySize, value);
      return this;
    }

    public ColumnBuilder append(BigDecimal value) {
      return append(value.setScale(-type.getScale(), RoundingMode.UNNECESSARY).unscaledValue());
    }

    public ColumnBuilder append(BigInteger unscaledVal) {
      growFixedWidthBuffersAndRows();
      assert currentIndex < rows;
      if (type.typeId == DType.DTypeEnum.DECIMAL32) {
        data.setInt(currentIndex++ << bitShiftBySize, unscaledVal.intValueExact());
      } else if (type.typeId == DType.DTypeEnum.DECIMAL64) {
        data.setLong(currentIndex++ << bitShiftBySize, unscaledVal.longValueExact());
      } else if (type.typeId == DType.DTypeEnum.DECIMAL128) {
        byte[] unscaledValueBytes = unscaledVal.toByteArray();
        byte[] result = convertDecimal128FromJavaToCudf(unscaledValueBytes);
        data.setBytes(currentIndex++ << bitShiftBySize, result, 0, result.length);
      } else {
        throw new IllegalStateException(type + " is not a supported decimal type.");
      }
      return this;
    }

    public ColumnBuilder append(String value) {
      assert value != null : "appendNull must be used to append null strings";
      return appendUTF8String(value.getBytes(StandardCharsets.UTF_8));
    }

    public ColumnBuilder appendUTF8String(byte[] value) {
      return appendUTF8String(value, 0, value.length);
    }

    public ColumnBuilder appendUTF8String(byte[] value, int srcOffset, int length) {
      assert value != null : "appendNull must be used to append null strings";
      assert srcOffset >= 0;
      assert length >= 0;
      assert value.length + srcOffset <= length;
      assert type.equals(DType.STRING) : " type " + type + " is not String";
      growStringBuffersAndRows(length);
      assert currentIndex < rows;
      if (length > 0) {
        data.setBytes(currentStringByteIndex, value, srcOffset, length);
      }
      currentStringByteIndex += length;
      offsets.setInt(++currentIndex << bitShiftByOffset, currentStringByteIndex);
      return this;
    }

    /**
     * Append multiple non-null byte values.
     */
    public ColumnBuilder append(byte[] value, int srcOffset, int length) {
      assert type.isBackedByByte();
      assert srcOffset >= 0;
      assert length >= 0;
      assert length + srcOffset <= value.length;

      if (length > 0) {
        growFixedWidthBuffersAndRows(length);
        assert currentIndex < rows;
        data.setBytes(currentIndex, value, srcOffset, length);
      }
      currentIndex += length;
      return this;
    }

    /**
     * Appends byte to a LIST of INT8/UINT8
     */
    public ColumnBuilder appendByteList(byte[] value) {
      return appendByteList(value, 0, value.length);
    }

    /**
     * Appends bytes to a LIST of INT8/UINT8
     */
    public ColumnBuilder appendByteList(byte[] value, int srcOffset, int length) {
      assert value != null : "appendNull must be used to append null bytes";
      assert type.equals(DType.LIST) : " type " + type + " is not LIST";
      getChild(0).append(value, srcOffset, length);
      return endList();
    }

    /**
     * Accepts a byte array containing the two's-complement representation of the unscaled value, which
     * is in big-endian byte-order. Then, transforms it into the representation of cuDF Decimal128 for
     * appending.
     * This method is more efficient than `append(BigInteger unscaledVal)` if we can directly access the
     * two's-complement representation of a BigDecimal without encoding via the method `toByteArray`.
     */
    public ColumnBuilder appendDecimal128(byte[] binary) {
      growFixedWidthBuffersAndRows();
      assert type.getTypeId().equals(DType.DTypeEnum.DECIMAL128);
      assert currentIndex < rows;
      assert binary.length <= type.getSizeInBytes();
      byte[] cuBinary = convertDecimal128FromJavaToCudf(binary);
      data.setBytes(currentIndex++ << bitShiftBySize, cuBinary, 0, cuBinary.length);
      return this;
    }

    public ColumnBuilder getChild(int index) {
      return childBuilders.get(index);
    }

    /**
     * Finish and create the immutable ColumnVector, copied to the device.
     */
    public final ColumnVector buildAndPutOnDevice() {
      try (HostColumnVector tmp = build()) {
        return tmp.copyToDevice();
      }
    }

    @Override
    public void close() {
      if (!built) {
        if (data != null) {
          data.close();
          data = null;
        }
        if (valid != null) {
          valid.close();
          valid = null;
        }
        if (offsets != null) {
          offsets.close();
          offsets = null;
        }
        for (ColumnBuilder childBuilder : childBuilders) {
          childBuilder.close();
        }
        built = true;
      }
    }

    @Override
    public String toString() {
      StringJoiner sj = new StringJoiner(",");
      for (ColumnBuilder cb : childBuilders) {
        sj.add(cb.toString());
      }
      return "ColumnBuilder{" +
          "type=" + type +
          ", children=" + sj +
          ", data=" + data +
          ", valid=" + valid +
          ", currentIndex=" + currentIndex +
          ", nullCount=" + nullCount +
          ", estimatedRows=" + estimatedRows +
          ", populatedRows=" + rows +
          ", built=" + built +
          '}';
    }
  }

  public static final class Builder implements AutoCloseable {
    private final long rows;
    private final DType type;
    private HostMemoryBuffer data;
    private HostMemoryBuffer valid;
    private HostMemoryBuffer offsets;
    private long currentIndex = 0;
    private long nullCount;
    private int currentStringByteIndex = 0;
    private boolean built;

    /**
     * Create a builder with a buffer of size rows
     * @param type       datatype
     * @param rows       number of rows to allocate.
     * @param stringBufferSize the size of the string data buffer if we are
     *                         working with Strings.  It is ignored otherwise.
     */
    Builder(DType type, long rows, long stringBufferSize) {
      this.type = type;
      this.rows = rows;
      if (type.equals(DType.STRING)) {
        if (stringBufferSize <= 0) {
          // We need at least one byte or we will get NULL back for data
          stringBufferSize = 1;
        }
        this.data = HostMemoryBuffer.allocate(stringBufferSize);
        // The offsets are ints and there is 1 more than the number of rows.
        this.offsets = HostMemoryBuffer.allocate((rows + 1) * OFFSET_SIZE);
        // The first offset is always 0
        this.offsets.setInt(0, 0);
      } else {
        this.data = HostMemoryBuffer.allocate(rows * type.getSizeInBytes());
      }
    }

    /**
     * Create a builder with a buffer of size rows (for testing ONLY).
     * @param type       datatype
     * @param rows       number of rows to allocate.
     * @param testData   a buffer to hold the data (should be large enough to hold rows entries).
     * @param testValid  a buffer to hold the validity vector (should be large enough to hold
     *                   rows entries or is null).
     * @param testOffsets a buffer to hold the offsets for strings and string categories.
     */
    Builder(DType type, long rows, HostMemoryBuffer testData,
            HostMemoryBuffer testValid, HostMemoryBuffer testOffsets) {
      this.type = type;
      this.rows = rows;
      this.data = testData;
      this.valid = testValid;
    }

    public final Builder append(boolean value) {
      assert type.equals(DType.BOOL8);
      assert currentIndex < rows;
      data.setByte(currentIndex * type.getSizeInBytes(), value ? (byte)1 : (byte)0);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value) {
      assert type.isBackedByByte();
      assert currentIndex < rows;
      data.setByte(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder append(byte value, long count) {
      assert (count + currentIndex) <= rows;
      assert type.isBackedByByte();
      data.setMemory(currentIndex * type.getSizeInBytes(), count, value);
      currentIndex += count;
      return this;
    }

    public final Builder append(short value) {
      assert type.isBackedByShort();
      assert currentIndex < rows;
      data.setShort(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder append(int value) {
      assert type.isBackedByInt();
      assert currentIndex < rows;
      data.setInt(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder append(long value) {
      assert type.isBackedByLong();
      assert currentIndex < rows;
      data.setLong(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder append(float value) {
      assert type.equals(DType.FLOAT32);
      assert currentIndex < rows;
      data.setFloat(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder append(double value) {
      assert type.equals(DType.FLOAT64);
      assert currentIndex < rows;
      data.setDouble(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    /**
     * Append java.math.BigDecimal into HostColumnVector with UNNECESSARY RoundingMode.
     * Input decimal should have a larger scale than column vector.Otherwise, an ArithmeticException will be thrown while rescaling.
     * If unscaledValue after rescaling exceeds the max precision of rapids type,
     * an ArithmeticException will be thrown while extracting integral.
     *
     * @param value BigDecimal value to be appended
     */
    public final Builder append(BigDecimal value) {
      return append(value, RoundingMode.UNNECESSARY);
    }

    /**
     * Append java.math.BigDecimal into HostColumnVector with user-defined RoundingMode.
     * Input decimal will be rescaled according to scale of column type and RoundingMode before appended.
     * If unscaledValue after rescaling exceeds the max precision of rapids type, an ArithmeticException will be thrown.
     *
     * @param value        BigDecimal value to be appended
     * @param roundingMode rounding mode determines rescaling behavior
     */
    public final Builder append(BigDecimal value, RoundingMode roundingMode) {
      assert type.isDecimalType();
      assert currentIndex < rows: "appended too many values " + currentIndex + " out of total rows " + rows;
      BigInteger unscaledValue = value.setScale(-type.getScale(), roundingMode).unscaledValue();
      if (type.typeId == DType.DTypeEnum.DECIMAL32) {
        assert value.precision() <= DType.DECIMAL32_MAX_PRECISION : "value exceeds maximum precision for DECIMAL32";
        data.setInt(currentIndex * type.getSizeInBytes(), unscaledValue.intValueExact());
      } else if (type.typeId == DType.DTypeEnum.DECIMAL64) {
        assert value.precision() <= DType.DECIMAL64_MAX_PRECISION : "value exceeds maximum precision for DECIMAL64 ";
        data.setLong(currentIndex * type.getSizeInBytes(), unscaledValue.longValueExact());
      } else if (type.typeId == DType.DTypeEnum.DECIMAL128) {
        assert value.precision() <= DType.DECIMAL128_MAX_PRECISION : "value exceeds maximum precision for DECIMAL128 ";
        appendUnscaledDecimal(value.unscaledValue());
        return this;
      } else {
        throw new IllegalStateException(type + " is not a supported decimal type.");
      }
      currentIndex++;
      return this;
    }

    public final Builder appendUnscaledDecimal(int value) {
      assert type.typeId == DType.DTypeEnum.DECIMAL32;
      assert currentIndex < rows;
      data.setInt(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder appendUnscaledDecimal(long value) {
      assert type.typeId == DType.DTypeEnum.DECIMAL64;
      assert currentIndex < rows;
      data.setLong(currentIndex * type.getSizeInBytes(), value);
      currentIndex++;
      return this;
    }

    public final Builder appendUnscaledDecimal(BigInteger value) {
      assert type.typeId == DType.DTypeEnum.DECIMAL128;
      assert currentIndex < rows;
      byte[] unscaledValueBytes = value.toByteArray();
      byte[] result = convertDecimal128FromJavaToCudf(unscaledValueBytes);
      data.setBytes(currentIndex*DType.DTypeEnum.DECIMAL128.sizeInBytes, result, 0, result.length);
      currentIndex++;
      return this;
    }

    public Builder append(String value) {
      assert value != null : "appendNull must be used to append null strings";
      return appendUTF8String(value.getBytes(StandardCharsets.UTF_8));
    }

    public Builder appendUTF8String(byte[] value) {
      return appendUTF8String(value, 0, value.length);
    }

    public Builder appendUTF8String(byte[] value, int offset, int length) {
      assert value != null : "appendNull must be used to append null strings";
      assert offset >= 0;
      assert length >= 0;
      assert length + offset <= value.length;
      assert type.equals(DType.STRING);
      assert currentIndex < rows;
      // just for strings we want to throw a real exception if we would overrun the buffer
      long oldLen = data.getLength();
      long newLen = oldLen;
      while (currentStringByteIndex + length > newLen) {
        newLen *= 2;
      }
      if (newLen > Integer.MAX_VALUE) {
        throw new IllegalStateException("A string buffer is not supported over 2GB in size");
      }
      if (newLen != oldLen) {
        // need to grow the size of the buffer.
        HostMemoryBuffer newData = HostMemoryBuffer.allocate(newLen);
        try {
          newData.copyFromHostBuffer(0, data, 0, currentStringByteIndex);
          data.close();
          data = newData;
          newData = null;
        } finally {
          if (newData != null) {
            newData.close();
          }
        }
      }
      if (length > 0) {
        data.setBytes(currentStringByteIndex, value, offset, length);
      }
      currentStringByteIndex += length;
      currentIndex++;
      offsets.setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
      return this;
    }

    public Builder appendArray(byte... values) {
      assert (values.length + currentIndex) <= rows;
      assert type.isBackedByByte();
      data.setBytes(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(short... values) {
      assert type.isBackedByShort();
      assert (values.length + currentIndex) <= rows;
      data.setShorts(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(int... values) {
      assert type.isBackedByInt();
      assert (values.length + currentIndex) <= rows;
      data.setInts(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(long... values) {
      assert type.isBackedByLong();
      assert (values.length + currentIndex) <= rows;
      data.setLongs(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(float... values) {
      assert type.equals(DType.FLOAT32);
      assert (values.length + currentIndex) <= rows;
      data.setFloats(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendArray(double... values) {
      assert type.equals(DType.FLOAT64);
      assert (values.length + currentIndex) <= rows;
      data.setDoubles(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendUnscaledDecimalArray(int... values) {
      assert type.typeId == DType.DTypeEnum.DECIMAL32;
      assert (values.length + currentIndex) <= rows;
      data.setInts(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    public Builder appendUnscaledDecimalArray(long... values) {
      assert type.typeId == DType.DTypeEnum.DECIMAL64;
      assert (values.length + currentIndex) <= rows;
      data.setLongs(currentIndex * type.getSizeInBytes(), values, 0, values.length);
      currentIndex += values.length;
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public Builder appendBoxed(BigDecimal... values) throws IndexOutOfBoundsException {
      assert type.isDecimalType();
      for (BigDecimal v : values) {
        if (v == null) {
          appendNull();
        } else {
          append(v);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Byte... values) throws IndexOutOfBoundsException {
      for (Byte b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Boolean... values) throws IndexOutOfBoundsException {
      for (Boolean b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b ? (byte) 1 : (byte) 0);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Short... values) throws IndexOutOfBoundsException {
      for (Short b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Integer... values) throws IndexOutOfBoundsException {
      for (Integer b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Long... values) throws IndexOutOfBoundsException {
      for (Long b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Float... values) throws IndexOutOfBoundsException {
      for (Float b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(Double... values) throws IndexOutOfBoundsException {
      for (Double b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    /**
     * Append multiple values.  This is very slow and should really only be used for tests.
     * @param values the values to append, including nulls.
     * @return this for chaining.
     * @throws {@link IndexOutOfBoundsException}
     */
    public final Builder appendBoxed(String... values) throws IndexOutOfBoundsException {
      for (String b : values) {
        if (b == null) {
          appendNull();
        } else {
          append(b);
        }
      }
      return this;
    }

    // TODO see if we can remove this...
    /**
     * Append this vector to the end of this vector
     * @param columnVector - Vector to be added
     * @return - The CudfColumn based on this builder values
     */
    public final Builder append(HostColumnVector columnVector) {
      assert columnVector.rows <= (rows - currentIndex);
      assert columnVector.type.equals(type);

      if (type.equals(DType.STRING)) {
        throw new UnsupportedOperationException(
            "Appending a string column vector client side is not currently supported");
      } else {
        data.copyFromHostBuffer(currentIndex * type.getSizeInBytes(), columnVector.offHeap.data,
            0L,
            columnVector.getRowCount() * type.getSizeInBytes());
      }

      //As this is doing the append on the host assume that a null count is available
      long otherNc = columnVector.getNullCount();
      if (otherNc != 0) {
        if (valid == null) {
          allocateBitmaskAndSetDefaultValues();
        }
        //copy values from intCudfColumn to this
        BitVectorHelper.append(columnVector.offHeap.valid, valid, currentIndex,
            columnVector.rows);
        nullCount += otherNc;
      }
      currentIndex += columnVector.rows;
      return this;
    }

    private void allocateBitmaskAndSetDefaultValues() {
      long bitmaskSize = ColumnView.getValidityBufferSize((int) rows);
      valid = HostMemoryBuffer.allocate(bitmaskSize);
      valid.setMemory(0, bitmaskSize, (byte) 0xFF);
    }

    /**
     * Append null value.
     */
    public final Builder appendNull() {
      setNullAt(currentIndex);
      currentIndex++;
      if (type.equals(DType.STRING)) {
        offsets.setInt(currentIndex * OFFSET_SIZE, currentStringByteIndex);
      }
      return this;
    }

    /**
     * Set a specific index to null.
     * @param index
     */
    public final Builder setNullAt(long index) {
      assert index < rows;

      // add null
      if (this.valid == null) {
        allocateBitmaskAndSetDefaultValues();
      }
      nullCount += BitVectorHelper.setNullAt(valid, index);
      return this;
    }

    /**
     * Finish and create the immutable CudfColumn.
     */
    public final HostColumnVector build() {
      HostColumnVector cv = new HostColumnVector(type,
          currentIndex, Optional.of(nullCount), data, valid, offsets);
      built = true;
      return cv;
    }

    /**
     * Finish and create the immutable ColumnVector, copied to the device.
     */
    public final ColumnVector buildAndPutOnDevice() {
      try (HostColumnVector tmp = build()) {
        return tmp.copyToDevice();
      }
    }

    /**
     * Close this builder and free memory if the CudfColumn wasn't generated. Verifies that
     * the data was released even in the case of an error.
     */
    @Override
    public final void close() {
      if (!built) {
        data.close();
        data = null;
        if (valid != null) {
          valid.close();
          valid = null;
        }
        if (offsets != null) {
          offsets.close();
          offsets = null;
        }
        built = true;
      }
    }

    @Override
    public String toString() {
      return "Builder{" +
          "data=" + data +
          "type=" + type +
          ", valid=" + valid +
          ", currentIndex=" + currentIndex +
          ", nullCount=" + nullCount +
          ", rows=" + rows +
          ", built=" + built +
          '}';
    }
  }

  public static abstract class DataType {
    public abstract DType getType();
    public abstract boolean isNullable();
    public abstract DataType getChild(int index);
    public abstract int getNumChildren();
  }

  public static class ListType extends HostColumnVector.DataType {
    private boolean isNullable;
    private HostColumnVector.DataType child;

    public ListType(boolean isNullable, DataType child) {
      this.isNullable = isNullable;
      this.child = child;
    }

    @Override
    public DType getType() {
      return DType.LIST;
    }

    @Override
    public boolean isNullable() {
      return isNullable;
    }

    @Override
    public HostColumnVector.DataType getChild(int index) {
      if (index > 0) {
        return null;
      }
      return child;
    }

    @Override
    public int getNumChildren() {
      return 1;
    }
  }

  public static class StructData {
    List<Object> dataRecord;

    public StructData(List<Object> dataRecord) {
      this.dataRecord = dataRecord;
    }

    public StructData(Object... data) {
      this(Arrays.asList(data));
    }

    public int getNumFields() {
      if (dataRecord != null) {
        return dataRecord.size();
      } else {
        return 0;
      }
    }

    public boolean isNull() {
      return (this.dataRecord == null);
    }

    public Object getField(int index) {
      return this.dataRecord.get(index);
    }
  }

  public static class StructType extends HostColumnVector.DataType {
    private boolean isNullable;
    private List<HostColumnVector.DataType> children;

    public StructType(boolean isNullable, List<HostColumnVector.DataType> children) {
      this.isNullable = isNullable;
      this.children = children;
    }

    public StructType(boolean isNullable, DataType... children) {
      this(isNullable, Arrays.asList(children));
    }

    @Override
    public DType getType() {
      return DType.STRUCT;
    }

    @Override
    public boolean isNullable() {
      return isNullable;
    }

    @Override
    public HostColumnVector.DataType getChild(int index) {
      return children.get(index);
    }

    @Override
    public int getNumChildren() {
      return children.size();
    }
  }

  public static class BasicType extends HostColumnVector.DataType {
    private DType type;
    private boolean isNullable;

    public BasicType(boolean isNullable, DType type) {
      this.isNullable = isNullable;
      this.type = type;
    }

    @Override
    public DType getType() {
      return type;
    }

    @Override
    public boolean isNullable() {
      return isNullable;
    }

    @Override
    public HostColumnVector.DataType getChild(int index) {
      return null;
    }

    @Override
    public int getNumChildren() {
      return 0;
    }
  }
}

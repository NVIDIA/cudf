/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.HostColumnVector.Builder;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;

/**
 * This class represents the immutable vector of data.  This class holds
 * references to device(GPU) memory and is reference counted to know when to release it.  Call
 * close to decrement the reference count when you are done with the column, and call incRefCount
 * to increment the reference count.
 */
public final class ColumnVector extends ColumnView {
  /**
   * Interface to handle events for this ColumnVector. Only invoked during
   * close, hence `onClosed` is the only event.
   */
  public interface EventHandler {
    /**
     * `onClosed` is invoked with the updated `refCount` during `close`.
     * The last invocation of `onClosed` will be with `refCount=0`.
     *
     * @note the callback is invoked with this `ColumnVector`'s lock held.
     *
     * @param cv reference to the ColumnVector we are closing
     * @param refCount the updated ref count for this ColumnVector at the time
     *                 of invocation
     */
    void onClosed(ColumnVector cv, int refCount);
  }

  private static final Logger log = LoggerFactory.getLogger(ColumnVector.class);

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private Optional<Long> nullCount = Optional.empty();
  private int refCount;
  private EventHandler eventHandler;

  /**
   * Wrap an existing on device cudf::column with the corresponding ColumnVector. The new
   * ColumnVector takes ownership of the pointer and will free it when the ref count reaches zero.
   * @param nativePointer host address of the cudf::column object which will be
   *                      owned by this instance.
   */
  public ColumnVector(long nativePointer) {
    super(new OffHeapState(nativePointer));
    assert nativePointer != 0;
    MemoryCleaner.register(this, offHeap);
    this.refCount = 0;
    incRefCountInternal(true);
  }

  private static OffHeapState makeOffHeap(DType type, long rows, Optional<Long> nullCount,
      DeviceMemoryBuffer dataBuffer, DeviceMemoryBuffer validityBuffer,
      DeviceMemoryBuffer offsetBuffer) {
    long viewHandle = initViewHandle(
        type, (int)rows, nullCount.orElse(UNKNOWN_NULL_COUNT).intValue(),
        dataBuffer, validityBuffer, offsetBuffer, null);
    return new OffHeapState(dataBuffer, validityBuffer, offsetBuffer, null, viewHandle);
  }

  /**
   * Create a new column vector based off of data already on the device.
   * @param type the type of the vector
   * @param rows the number of rows in this vector.
   * @param nullCount the number of nulls in the dataset.
   * @param dataBuffer the data stored on the device.  The column vector takes ownership of the
   *                   buffer.  Do not use the buffer after calling this.
   * @param validityBuffer an optional validity buffer. Must be provided if nullCount != 0. The
   *                      column vector takes ownership of the buffer. Do not use the buffer
   *                      after calling this.
   * @param offsetBuffer a host buffer required for strings and string categories. The column
   *                    vector takes ownership of the buffer. Do not use the buffer after calling
   *                    this.
   */
  public ColumnVector(DType type, long rows, Optional<Long> nullCount,
      DeviceMemoryBuffer dataBuffer, DeviceMemoryBuffer validityBuffer,
      DeviceMemoryBuffer offsetBuffer) {
    super(makeOffHeap(type, rows, nullCount, dataBuffer, validityBuffer, offsetBuffer));
    assert !type.equals(DType.LIST) : "This constructor should not be used for list type";
    if (!type.equals(DType.STRING)) {
      assert offsetBuffer == null : "offsets are only supported for STRING";
    }
    assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
        || !nullCount.isPresent();
    MemoryCleaner.register(this, offHeap);
    this.nullCount = nullCount;
    this.refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * This method is internal and exposed purely for testing purposes
   */
  static OffHeapState makeOffHeap(DType type, long rows, Optional<Long> nullCount,
      DeviceMemoryBuffer dataBuffer, DeviceMemoryBuffer validityBuffer,
      DeviceMemoryBuffer offsetBuffer, List<DeviceMemoryBuffer> toClose, long[] childHandles) {
    long viewHandle = initViewHandle(type, (int)rows, nullCount.orElse(UNKNOWN_NULL_COUNT).intValue(),
        dataBuffer, validityBuffer,
        offsetBuffer, childHandles);
    return new OffHeapState(dataBuffer, validityBuffer, offsetBuffer, toClose, viewHandle);
  }

  /**
   * Create a new column vector based off of data already on the device with child columns.
   * @param type the type of the vector, typically a nested type
   * @param rows the number of rows in this vector.
   * @param nullCount the number of nulls in the dataset.
   * @param dataBuffer the data stored on the device.  The column vector takes ownership of the
   *                   buffer.  Do not use the buffer after calling this.
   * @param validityBuffer an optional validity buffer. Must be provided if nullCount != 0. The
   *                      column vector takes ownership of the buffer. Do not use the buffer
   *                      after calling this.
   * @param offsetBuffer a host buffer required for strings and string categories. The column
   *                    vector takes ownership of the buffer. Do not use the buffer after calling
   *                    this.
   * @param toClose  List of buffers to track and close once done, usually in case of children
   * @param childHandles array of longs for child column view handles.
   */
  public ColumnVector(DType type, long rows, Optional<Long> nullCount,
                      DeviceMemoryBuffer dataBuffer, DeviceMemoryBuffer validityBuffer,
                      DeviceMemoryBuffer offsetBuffer, List<DeviceMemoryBuffer> toClose, long[] childHandles) {
    super(makeOffHeap(type, rows, nullCount, dataBuffer, validityBuffer, offsetBuffer, toClose, childHandles));
    if (!type.equals(DType.STRING) && !type.equals(DType.LIST)) {
      assert offsetBuffer == null : "offsets are only supported for STRING, LISTS";
    }
    assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
        || !nullCount.isPresent();
    MemoryCleaner.register(this, offHeap);

    this.refCount = 0;
    incRefCountInternal(true);
  }

  /**
   * This is a very special constructor that should only ever be called by
   * fromViewWithContiguousAllocation.  It takes a cudf::column_view * instead of a cudf::column *.
   * But to maintain memory ownership properly we need to slice the memory in the view off from
   * a separate buffer that actually owns the memory allocation.
   * @param viewAddress the address of the cudf::column_view
   * @param contiguousBuffer the buffer that this is based off of.
   */
  private ColumnVector(long viewAddress, DeviceMemoryBuffer contiguousBuffer) {
    super(new OffHeapState(viewAddress, contiguousBuffer));
    MemoryCleaner.register(this, offHeap);
    // TODO we may want to ask for the null count anyways...
    this.nullCount = Optional.empty();

    this.refCount = 0;
    incRefCountInternal(true);
  }


  /**
   * For a ColumnVector this is really just incrementing the reference count.
   * @return this
   */
  @Override
  public ColumnVector copyToColumnVector() {
    return incRefCount();
  }

  /**
   * Retrieves the column_view for a cudf::column and if it fails to do so, the column is deleted
   * and the exception is thrown to the caller.
   * @param nativePointer the cudf::column handle
   * @return the column_view handle
   */
  private static long getColumnViewFromColumn(long nativePointer) {
    try {
      return ColumnVector.getNativeColumnView(nativePointer);
    } catch (CudfException ce) {
      deleteCudfColumn(nativePointer);
      throw ce;
    }
  }

  static long initViewHandle(DType type, int numRows, int nullCount,
                             BaseDeviceMemoryBuffer dataBuffer,
                             BaseDeviceMemoryBuffer validityBuffer,
                             BaseDeviceMemoryBuffer offsetBuffer, long[] childHandles) {
    long cd = dataBuffer == null ? 0 : dataBuffer.address;
    long cdSize = dataBuffer == null ? 0 : dataBuffer.length;
    long od = offsetBuffer == null ? 0 : offsetBuffer.address;
    long vd = validityBuffer == null ? 0 : validityBuffer.address;
    return makeCudfColumnView(type.typeId.getNativeId(), type.getScale(), cd, cdSize,
        od, vd, nullCount, numRows, childHandles);
  }

  /**
   * Creates a ColumnVector from a native column_view using a contiguous device allocation.
   *
   * @param columnViewAddress address of the native column_view
   * @param buffer device buffer containing the data referenced by the column view
   */
  public static ColumnVector fromViewWithContiguousAllocation(long columnViewAddress, DeviceMemoryBuffer buffer) {
    return new ColumnVector(columnViewAddress, buffer);
  }

  /**
   * Set an event handler for this vector. This method can be invoked with null
   * to unset the handler.
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
   * Returns the current event handler for this ColumnVector or null if no handler
   * is associated.
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
        super.close();
        offHeap.clean(false);
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
    return "ColumnVector{" +
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
  public ColumnVector incRefCount() {
    return incRefCountInternal(false);
  }

  private synchronized ColumnVector incRefCountInternal(boolean isFirstTime) {
    offHeap.addRef();
    if (refCount <= 0 && !isFirstTime) {
      offHeap.logRefCountDebug("INC AFTER CLOSE " + this);
      throw new IllegalStateException("Column is already closed");
    }
    refCount++;
    return this;
  }

  /**
   * Returns the number of nulls in the data. Note that this might end up
   * being a very expensive operation because if the null count is not
   * known it will be calculated.
   */
  public long getNullCount() {
    if (!nullCount.isPresent()) {
      nullCount = Optional.of(offHeap.getNativeNullCount());
    }
    return nullCount.get();
  }

  /**
   * Returns this column's current refcount
   */
  public synchronized int getRefCount() {
    return refCount;
  }

  /**
   * Returns if the vector has a validity vector allocated or not.
   */
  public boolean hasValidityVector() {
    return (offHeap.getValid() != null);
  }

  /**
   * Returns if the vector has nulls.  Note that this might end up
   * being a very expensive operation because if the null count is not
   * known it will be calculated.
   */
  public boolean hasNulls() {
    return getNullCount() > 0;
  }

  /////////////////////////////////////////////////////////////////////////////
  // RAW DATA ACCESS
  /////////////////////////////////////////////////////////////////////////////


  /**
   * Get access to the raw device buffer for this column.  This is intended to be used with a lot
   * of caution.  The lifetime of the buffer is tied to the lifetime of the column (Do not close
   * the buffer, as the column will take care of it).  Do not modify the contents of the buffer or
   * it might negatively impact what happens on the column.  The data must be on the device for
   * this to work. Strings and string categories do not currently work because their underlying
   * device layout is currently hidden.
   * @param type the type of buffer to get access to.
   * @return the underlying buffer or null if no buffer is associated with it for this column.
   * Please note that if the column is empty there may be no buffers at all associated with the
   * column.
   */
  public BaseDeviceMemoryBuffer getDeviceBufferFor(BufferType type) {
    BaseDeviceMemoryBuffer srcBuffer;
    switch(type) {
      case VALIDITY:
        srcBuffer = offHeap.getValid();
        break;
      case DATA:
        srcBuffer = offHeap.getData();
        break;
      case OFFSET:
        srcBuffer = offHeap.getOffsets();
        break;
      default:
        throw new IllegalArgumentException(type + " is not a supported buffer type.");

    }
    return srcBuffer;
  }

  /**
   * Ensures the ByteBuffer passed in is a direct byte buffer.
   * If it is not then it creates one and copies the data in
   * the byte buffer passed in to the direct byte buffer
   * it created and returns it.
   */
  private static ByteBuffer bufferAsDirect(ByteBuffer buf) {
    ByteBuffer bufferOut = buf;
    if (bufferOut != null && !bufferOut.isDirect()) {
      bufferOut = ByteBuffer.allocateDirect(buf.remaining());
      bufferOut.put(buf);
      bufferOut.flip();
    }
    return bufferOut;
  }

  /**
   * Create a ColumnVector from the Apache Arrow byte buffers passed in.
   * Any of the buffers not used for that datatype should be set to null.
   * The buffers are expected to be off heap buffers, but if they are not,
   * it will handle copying them to direct byte buffers.
   * This only supports primitive types. Strings, Decimals and nested types
   * such as list and struct are not supported.
   * @param type - type of the column
   * @param numRows - Number of rows in the arrow column
   * @param nullCount - Null count
   * @param data - ByteBuffer of the Arrow data buffer
   * @param validity - ByteBuffer of the Arrow validity buffer
   * @param offsets - ByteBuffer of the Arrow offsets buffer
   * @return - new ColumnVector
   */
  public static ColumnVector fromArrow(
      DType type,
      long numRows,
      long nullCount,
      ByteBuffer data,
      ByteBuffer validity,
      ByteBuffer offsets) {
    long columnHandle = fromArrow(type.typeId.getNativeId(), numRows, nullCount,
      bufferAsDirect(data), bufferAsDirect(validity), bufferAsDirect(offsets));
    ColumnVector vec = new ColumnVector(columnHandle);
    return vec;
  }

  /**
   * Create a new vector of length rows, where each row is filled with the Scalar's
   * value
   * @param scalar - Scalar to use to fill rows
   * @param rows - Number of rows in the new ColumnVector
   * @return - new ColumnVector
   */
  public static ColumnVector fromScalar(Scalar scalar, int rows) {
    long columnHandle = fromScalar(scalar.getScalarHandle(), rows);
    return new ColumnVector(columnHandle);
  }

  /**
   * Create a new struct vector made up of existing columns. Note that this will copy
   * the contents of the input columns to make a new vector. If you only want to
   * do a quick temporary computation you can use ColumnView.makeStructView.
   * @param columns the columns to make the struct from.
   * @return the new ColumnVector
   */
  public static ColumnVector makeStruct(ColumnView... columns) {
    try (ColumnView cv = ColumnView.makeStructView(columns)) {
      return cv.copyToColumnVector();
    }
  }

  /**
   * Create a new struct vector made up of existing columns. Note that this will copy
   * the contents of the input columns to make a new vector. If you only want to
   * do a quick temporary computation you can use ColumnView.makeStructView.
   * @param rows the number of rows in the struct. Used for structs with no children.
   * @param columns the columns to make the struct from.
   * @return the new ColumnVector
   */
  public static ColumnVector makeStruct(long rows, ColumnView... columns) {
    try (ColumnView cv = ColumnView.makeStructView(rows, columns)) {
      return cv.copyToColumnVector();
    }
  }

  /**
   * Create a LIST column from the given columns. Each list in the returned column will have the
   * same number of entries in it as columns passed into this method. Be careful about the
   * number of rows passed in as there are limits on the maximum output size supported for
   * column lists.
   * @param columns the columns to make up the list column, in the order they will appear in the
   *                resulting lists.
   * @return the new LIST ColumnVector
   */
  public static ColumnVector makeList(ColumnView... columns) {
    if (columns.length <= 0) {
      throw new IllegalArgumentException("At least one column is needed to get the row count");
    }
    return makeList(columns[0].getRowCount(), columns[0].getType(), columns);
  }

  /**
   * Create a LIST column from the given columns. Each list in the returned column will have the
   * same number of entries in it as columns passed into this method. Be careful about the
   * number of rows passed in as there are limits on the maximum output size supported for
   * column lists.
   * @param rows the number of rows to create, for the special case of an empty list.
   * @param type the type of the child column, for the special case of an empty list.
   * @param columns the columns to make up the list column, in the order they will appear in the
   *                resulting lists.
   * @return the new LIST ColumnVector
   */
  public static ColumnVector makeList(long rows, DType type, ColumnView... columns) {
    long[] handles = new long[columns.length];
    for (int i = 0; i < columns.length; i++) {
      ColumnView cv = columns[i];
      if (rows != cv.getRowCount()) {
        throw new IllegalArgumentException("All columns must have the same number of rows");
      }
      if (!type.equals(cv.getType())) {
        throw new IllegalArgumentException("All columns must have the same type");
      }

      handles[i] = cv.getNativeView();
    }
    if (columns.length == 0 && type.isNestedType()) {
      throw new IllegalArgumentException(
          "Creating an empty list column of nested types is not currently supported");
    }
    return new ColumnVector(makeList(handles, type.typeId.nativeId, type.getScale(), rows));
  }

  /**
   * Create a LIST column from the current column and a given offsets column. The output column will
   * contain lists having elements that are copied from the current column and their sizes are
   * determined by the given offsets.
   *
   * Note that the caller is responsible to make sure the given offsets column is of type INT32 and
   * it contains valid indices to create a LIST column. There will not be any validity check for
   * these offsets during calling to this function. If the given offsets are invalid, we may have
   * bad memory accesses and/or data corruption.
   *
   * @param rows the number of rows to create.
   * @param offsets the offsets pointing to row indices of the current column to create an output
   *                LIST column.
   */
  public ColumnVector makeListFromOffsets(long rows, ColumnView offsets) {
    return new ColumnVector(makeListFromOffsets(getNativeView(), offsets.getNativeView(), rows));
  }

  /**
   * Create a new vector of length rows, starting at the initialValue and going by step each time.
   * Only numeric types are supported.
   * @param initialValue the initial value to start at.
   * @param step the step to add to each subsequent row.
   * @param rows the total number of rows
   * @return the new ColumnVector.
   */
  public static ColumnVector sequence(Scalar initialValue, Scalar step, int rows) {
    if (!initialValue.isValid() || !step.isValid()) {
      throw new IllegalArgumentException("nulls are not supported in sequence");
    }
    return new ColumnVector(sequence(initialValue.getScalarHandle(), step.getScalarHandle(), rows));
  }

  /**
   * Create a new vector of length rows, starting at the initialValue and going by 1 each time.
   * Only numeric types are supported.
   * @param initialValue the initial value to start at.
   * @param rows the total number of rows
   * @return the new ColumnVector.
   */
  public static ColumnVector sequence(Scalar initialValue, int rows) {
    if (!initialValue.isValid()) {
      throw new IllegalArgumentException("nulls are not supported in sequence");
    }
    return new ColumnVector(sequence(initialValue.getScalarHandle(), 0, rows));
  }

  /**
   * Create a list column in which each row is a sequence of values starting from a `start` value,
   * incrementing by one, and its cardinality is specified by a `size` value. The `start` and `size`
   * values used to generate each list is taken from the corresponding row of the input start and
   * size columns.
   * @param start first values in the result sequences
   * @param size numbers of values in the result sequences
   * @return the new ColumnVector.
   */
  public static ColumnVector sequence(ColumnView start, ColumnView size) {
    assert start.getNullCount() == 0 || size.getNullCount() == 0 : "starts and sizes input " +
        "columns must not have nulls.";
    return new ColumnVector(sequences(start.getNativeView(), size.getNativeView(), 0));
  }

  /**
   * Create a list column in which each row is a sequence of values starting from a `start` value,
   * incrementing by a `step` value, and its cardinality is specified by a `size` value.
   * The values `start`, `step`, and `size` used to generate each list is taken from the
   * corresponding row of the input starts, steps, and sizes columns.
   * @param start first values in the result sequences
   * @param size numbers of values in the result sequences
   * @param step increment values for the result sequences.
   * @return the new ColumnVector.
   */
  public static ColumnVector sequence(ColumnView start, ColumnView size, ColumnView step) {
    assert start.getNullCount() == 0 || size.getNullCount() == 0 || step.getNullCount() == 0:
        "start, size and step must not have nulls.";
    assert step.getType() == start.getType() : "start and step input columns must" +
        " have the same type.";

    return new ColumnVector(sequences(start.getNativeView(), size.getNativeView(),
        step.getNativeView()));
  }

  /**
   * Create a new vector by concatenating multiple columns together.
   * Note that all columns must have the same type.
   */
  public static ColumnVector concatenate(ColumnView... columns) {
    if (columns.length < 2) {
      throw new IllegalArgumentException("Concatenate requires 2 or more columns");
    }
    long[] columnHandles = new long[columns.length];
    for (int i = 0; i < columns.length; ++i) {
      columnHandles[i] = columns[i].getNativeView();
    }
    return new ColumnVector(concatenate(columnHandles));
  }

  /**
   * Concatenate columns of strings together, combining a corresponding row from each column
   * into a single string row of a new column with no separator string inserted between each
   * combined string and maintaining null values in combined rows.
   * @param columns array of columns containing strings, must be non-empty
   * @return A new java column vector containing the concatenated strings.
   */
  public static ColumnVector stringConcatenate(ColumnView[] columns) {
    try (Scalar emptyString = Scalar.fromString("");
         Scalar nullString = Scalar.fromString(null)) {
      return stringConcatenate(emptyString, nullString, columns);
    }
  }

  /**
   * Concatenate columns of strings together, combining a corresponding row from each column into
   * a single string row of a new column. This version includes the separator for null rows
   * if 'narep' is valid.
   * @param separator string scalar inserted between each string being merged.
   * @param narep string scalar indicating null behavior. If set to null and any string in the row
   *              is null the resulting string will be null. If not null, null values in any column
   *              will be replaced by the specified string.
   * @param columns array of columns containing strings, must be non-empty
   * @return A new java column vector containing the concatenated strings.
   */
  public static ColumnVector stringConcatenate(Scalar separator, Scalar narep, ColumnView[] columns) {
    return stringConcatenate(separator, narep, columns, true);
  }

  /**
   * Concatenate columns of strings together, combining a corresponding row from each column into
   * a single string row of a new column.
   * @param separator string scalar inserted between each string being merged.
   * @param narep string scalar indicating null behavior. If set to null and any string in the row
   *              is null the resulting string will be null. If not null, null values in any column
   *              will be replaced by the specified string.
   * @param columns array of columns containing strings, must be non-empty
   * @param separateNulls if true, then the separator is included for null rows if
   *                       `narep` is valid.
   * @return A new java column vector containing the concatenated strings.
   */
  public static ColumnVector stringConcatenate(Scalar separator, Scalar narep, ColumnView[] columns,
      boolean separateNulls) {
    assert columns != null : "input columns should not be null";
    assert columns.length > 0 : "input columns should not be empty";
    assert separator != null : "separator scalar provided may not be null";
    assert separator.getType().equals(DType.STRING) : "separator scalar must be a string scalar";
    assert narep != null : "narep scalar provided may not be null";
    assert narep.getType().equals(DType.STRING) : "narep scalar must be a string scalar";

    long[] columnViews = new long[columns.length];
    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      columnViews[i] = columns[i].getNativeView();
    }

    return new ColumnVector(stringConcatenation(columnViews, separator.getScalarHandle(),
        narep.getScalarHandle(), separateNulls));
  }

  /**
   * Concatenate columns of strings together using a separator specified for each row
   * and returns the result as a string column. If the row separator for a given row is null,
   * output column for that row is null. Null column values for a given row are skipped.
   * @param columns array of columns containing strings
   * @param sepCol strings column that provides the separator for a given row
   * @return A new java column vector containing the concatenated strings with separator between.
   */
  public static ColumnVector stringConcatenate(ColumnView[] columns, ColumnView sepCol) {
    try (Scalar nullString = Scalar.fromString(null);
         Scalar emptyString = Scalar.fromString("")) {
      return stringConcatenate(columns, sepCol, nullString, emptyString, false);
    }
  }

  /**
   * Concatenate columns of strings together using a separator specified for each row
   * and returns the result as a string column. If the row separator for a given row is null,
   * output column for that row is null unless separatorNarep is provided.
   * The separator is applied between two output row values if the separateNulls
   * is `YES` or only between valid rows if separateNulls is `NO`.
   * @param columns array of columns containing strings
   * @param sepCol strings column that provides the separator for a given row
   * @param separatorNarep string scalar indicating null behavior when a separator is null.
   *                        If set to null and the separator is null the resulting string will
   *                        be null. If not null, this string will be used in place of a null
   *                        separator.
   * @param colNarep string that should be used in place of any null strings
   *                  found in any column.
   * @param separateNulls if true, then the separator is included for null rows if
   *                       `colNarep` is valid.
   * @return A new java column vector containing the concatenated strings with separator between.
   */
  public static ColumnVector stringConcatenate(ColumnView[] columns,
      ColumnView sepCol, Scalar separatorNarep, Scalar colNarep, boolean separateNulls) {
    assert columns.length >= 1 : ".stringConcatenate() operation requires at least 1 column";
    assert separatorNarep != null : "separator narep scalar provided may not be null";
    assert colNarep != null : "column narep scalar provided may not be null";
    assert separatorNarep.getType().equals(DType.STRING) : "separator naprep scalar must be a string scalar";
    assert colNarep.getType().equals(DType.STRING) : "column narep scalar must be a string scalar";

    long[] columnViews = new long[columns.length];
    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      columnViews[i] = columns[i].getNativeView();
    }

    return new ColumnVector(stringConcatenationSepCol(columnViews, sepCol.getNativeView(),
      separatorNarep.getScalarHandle(), colNarep.getScalarHandle(), separateNulls));
  }

  /**
   * Concatenate columns of lists horizontally (row by row), combining a corresponding row
   * from each column into a single list row of a new column.
   * NOTICE: Any concatenation involving a null list element will result in a null list.
   *
   * @param columns array of columns containing lists, must be non-empty
   * @return A new java column vector containing the concatenated lists.
   */
  public static ColumnVector listConcatenateByRow(ColumnView... columns) {
    return listConcatenateByRow(false, columns);
  }

  /**
   * Concatenate columns of lists horizontally (row by row), combining a corresponding row
   * from each column into a single list row of a new column.
   *
   * @param ignoreNull whether to ignore null list element of input columns: If true, null list
   *                   will be ignored from concatenation; Otherwise, any concatenation involving
   *                   a null list element will result in a null list
   * @param columns    array of columns containing lists, must be non-empty
   * @return A new java column vector containing the concatenated lists.
   */
  public static ColumnVector listConcatenateByRow(boolean ignoreNull, ColumnView... columns) {
    assert columns != null : "input columns should not be null";
    assert columns.length > 0 : "input columns should not be empty";

    long[] columnViews = new long[columns.length];
    for(int i = 0; i < columns.length; i++) {
      columnViews[i] = columns[i].getNativeView();
    }

    return new ColumnVector(concatListByRow(columnViews, ignoreNull));
  }

  /**
   * Create a new vector containing the MD5 hash of each row in the table.
   *
   * @param columns array of columns to hash, must have identical number of rows.
   * @return the new ColumnVector of 32 character hex strings representing each row's hash value.
   */
  public static ColumnVector md5Hash(ColumnView... columns) {
    if (columns.length < 1) {
      throw new IllegalArgumentException("MD5 hashing requires at least 1 column of input");
    }
    long[] columnViews = new long[columns.length];
    long size = columns[0].getRowCount();

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must be the same size";
      assert !columns[i].getType().isDurationType() : "Unsupported column type Duration";
      assert !columns[i].getType().isTimestampType() : "Unsupported column type Timestamp";
      assert !columns[i].getType().isNestedType() || columns[i].getType().equals(DType.LIST) :
          "Unsupported nested type column";
      columnViews[i] = columns[i].getNativeView();
    }
    return new ColumnVector(md5(columnViews));
  }

  /**
   * Create a new column containing the Sha1 hash of each row in the table.
   *
   * @param columns columns to hash
   * @return the new ColumnVector of 40 character hex strings representing each row's hash value.
   */
  public static ColumnVector sha1Hash(ColumnView... columns) {
    if (columns.length < 1) {
      throw new IllegalArgumentException("Sha1 hashing requires at least 1 column of input");
    }
    long[] columnViews = new long[columns.length];

    for(int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      columnViews[i] = columns[i].getNativeView();
    }
    return new ColumnVector(sha1(columnViews));
  }

  /**
   * Generic method to cast ColumnVector
   * When casting from a Date, Timestamp, or Boolean to a numerical type the underlying numerical
   * representation of the data will be used for the cast.
   *
   * For Strings:
   * Casting strings from/to timestamp isn't supported atm.
   * Please look at {@link ColumnVector#asTimestamp(DType, String)}
   * and {@link ColumnVector#asStrings(String)} for casting string to timestamp when the format
   * is known
   *
   * Float values when converted to String could be different from the expected default behavior in
   * Java
   * e.g.
   * 12.3 => "12.30000019" instead of "12.3"
   * Double.POSITIVE_INFINITY => "Inf" instead of "INFINITY"
   * Double.NEGATIVE_INFINITY => "-Inf" instead of "-INFINITY"
   *
   * @param type type of the resulting ColumnVector
   * @return A new vector allocated on the GPU
   */
  @Override
  public ColumnVector castTo(DType type) {
    if (this.type.equals(type)) {
      // Optimization
      return incRefCount();
    }
    return super.castTo(type);
  }

  /**
   * Replace the null mask of a column. The resultant null mask is the bitwise {@code mergeOp} of
   * null masks in the columns given as arguments, AND-ed with this column's existing null mask.
   *
   * If applying the null mask would be a no-op, the original column is returned with incremented
   * refcount. Otherwise, a deep copy of the column is made.
   *
   * For STRUCT columns the new mask is also pushed down into every descendant column, to
   * stay consistent with the parent. For LIST/STRING columns the resultant offsets are
   * sanitized to not contain any non-empty nulls.
   *
   * If {@code columns} is empty, the column is returned unchanged (no-op).
   *
   * @param mergeOp binary operator (either BITWISE_AND or BITWISE_OR)
   * @param columns array of columns whose null masks are merged, must have identical number of rows.
   * @return the new ColumnVector with merged null mask.
   */
  @Override
  public ColumnVector mergeAndSetValidity(BinaryOp mergeOp, ColumnView... columns) {
    assert mergeOp == BinaryOp.BITWISE_AND || mergeOp == BinaryOp.BITWISE_OR : "Only BITWISE_AND and BITWISE_OR supported right now";
    long[] columnViews = new long[columns.length];
    long size = getRowCount();

    for (int i = 0; i < columns.length; i++) {
      assert columns[i] != null : "Column vectors passed may not be null";
      assert columns[i].getRowCount() == size : "Row count mismatch, all columns must be the same size";
      columnViews[i] = columns[i].getNativeView();
    }

    long mergeOutput = bitwiseMergeAndSetValidity(getNativeView(), columnViews, mergeOp.nativeId);
    if (mergeOutput == 0) {  // no-op, the current column is unchanged
      return incRefCount();
    }
    return new ColumnVector(mergeOutput);
  }

  /////////////////////////////////////////////////////////////////////////////
  // NATIVE METHODS
  /////////////////////////////////////////////////////////////////////////////

  private static native long sequence(long initialValue, long step, int rows);

  private static native long sequences(long startHandle, long sizeHandle, long stepHandle)
      throws CudfException;

  private static native long fromArrow(int type, long col_length,
      long null_count, ByteBuffer data, ByteBuffer validity,
      ByteBuffer offsets) throws CudfException;

  private static native long fromScalar(long scalarHandle, int rowCount) throws CudfException;

  private static native long makeList(long[] handles, long typeHandle, int scale, long rows)
      throws CudfException;

  private static native long makeListFromOffsets(long childHandle, long offsetsHandle, long rows)
      throws CudfException;

  private static native long concatenate(long[] viewHandles) throws CudfException;

  /**
   * Native method to replace a column's null mask. The null mask is the
   * bitwise merge of the null masks in the columns given as arguments.
   *
   * @param baseHandle column view of the column whose null mask is being replaced.
   * @param viewHandles array of views whose null masks are merged, must have identical row counts.
   * @param mergeOp native id of the binary op (BITWISE_AND or BITWISE_OR) used to merge the null masks.
   * @return native handle of the resulting column, or 0 when the original is unchanged
   *         (a no-op) and no copied column was produced.
   */
  private static native long bitwiseMergeAndSetValidity(long baseHandle, long[] viewHandles,
                                                        int mergeOp) throws CudfException;

  /**
   * Native method to concatenate columns of lists horizontally (row by row), combining a row
   * from each column into a single list.
   *
   * @param columnViews array of longs holding the native handles of the column_views to combine.
   * @return native handle of the resulting cudf column, used to construct the Java column
   * by the listConcatenateByRow method.
   */
  private static native long concatListByRow(long[] columnViews, boolean ignoreNull);

  /**
   * Native method to concatenate columns of strings together, combining a row from
   * each column into a single string.
   *
   * @param columnViews array of longs holding the native handles of the column_views to combine.
   * @param separator   string scalar inserted between each string being merged, may not be null.
   * @param narep       string scalar indicating null behavior. If set to null and any string in
   *                    the row is null the resulting string will be null. If not null, null
   *                    values in any column will be replaced by the specified string. The
   *                    underlying value in the string scalar may be null, but the object passed
   *                    in may not.
   * @param separate_nulls boolean if true, then the separator is included for null rows if
   *                       `narep` is valid.
   * @return native handle of the resulting cudf column, used to construct the Java column
   * by the stringConcatenate method.
   */
  private static native long stringConcatenation(long[] columnViews, long separator, long narep,
                                                 boolean separate_nulls);

  /**
   * Native method to concatenate columns of strings together using a separator specified for each row
   * and returns the result as a string column.
   * @param columnViews array of longs holding the native handles of the column_views to combine.
   * @param sep_column long holding the native handle of the strings_column_view used as separators.
   * @param separator_narep string scalar indicating null behavior when a separator is null.
   *                        If set to null and the separator is null the resulting string will
   *                        be null. If not null, this string will be used in place of a null
   *                        separator.
   * @param col_narep string String scalar that should be used in place of any null strings
   *                         found in any column.
   * @param separate_nulls boolean if true, then the separator is included for null rows if
   *                       `col_narep` is valid.
   * @return native handle of the resulting cudf column, used to construct the Java column.
   */
  private static native long stringConcatenationSepCol(long[] columnViews,
                                                       long sep_column,
                                                       long separator_narep,
                                                       long col_narep,
                                                       boolean separate_nulls);

  /**
   * Native method to MD5 hash each row of the given table
   *
   * @param viewHandles array of native handles to the cudf::column_view columns being operated on.
   * @return native handle of the resulting cudf column containing the hex-string hashing results.
   */
  private static native long md5(long[] viewHandles) throws CudfException;

  /**
   * Native method to sha1 hash each row of the given table
   *
   * @param viewHandles array of native handles to the columns being operated on.
   * @return native handle of the resulting cudf column containing the hex-string hashing results.
   */
  private static native long sha1(long[] viewHandles) throws CudfException;

  /////////////////////////////////////////////////////////////////////////////
  // INTERNAL/NATIVE ACCESS
  /////////////////////////////////////////////////////////////////////////////

  ////////
  // Native methods specific to cudf::column. These either take or create a cudf::column
  // instead of a cudf::column_view so they need to be used with caution. These should
  // only be called from the OffHeap inner class.
  ////////

  /**
   * Delete the column. This is not private because there are a few cases where Table
   * may get back an array of columns pointers and we want to have best effort in cleaning them up
   * on any failure.
   */
  static native void deleteCudfColumn(long cudfColumnHandle) throws CudfException;

  private static native int getNativeNullCountColumn(long cudfColumnHandle) throws CudfException;

  private static native void setNativeNullCountColumn(long cudfColumnHandle, int nullCount) throws CudfException;

  /**
   * Create a cudf::column_view from a cudf::column.
   * @param cudfColumnHandle the pointer to the cudf::column
   * @return a pointer to a cudf::column_view
   * @throws CudfException on any error
   */
  static native long getNativeColumnView(long cudfColumnHandle) throws CudfException;

  static native long makeEmptyCudfColumn(int type, int scale);

  /////////////////////////////////////////////////////////////////////////////
  // HELPER CLASSES
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Holds the off heap state of the column vector so we can clean it up, even if it is leaked.
   */
  protected static final class OffHeapState extends MemoryCleaner.Cleaner {
    // This must be kept in sync with the native code
    public static final long UNKNOWN_NULL_COUNT = -1;
    private long columnHandle;
    private long viewHandle = 0;
    private List<MemoryBuffer> toClose = new ArrayList<>();


    /**
     * Make a column form an existing cudf::column *.
     */
    public OffHeapState(long columnHandle) {
      this.columnHandle = columnHandle;
      this.toClose.add(getData());
      this.toClose.add(getValid());
      this.toClose.add(getOffsets());
    }

    /**
     * Create from existing cudf::column_view and buffers.
     */
    public OffHeapState(DeviceMemoryBuffer data, DeviceMemoryBuffer valid, DeviceMemoryBuffer offsets,
                        List<DeviceMemoryBuffer> buffers,
                        long viewHandle) {
      assert(viewHandle != 0);
      if (data != null) {
        this.toClose.add(data);
      }
      if (valid != null) {
        this.toClose.add(valid);
      }
      if (offsets != null) {
        this.toClose.add(offsets);
      }
      if (buffers != null) {
        toClose.addAll(buffers);
      }
      this.viewHandle = viewHandle;
    }

    /**
     * Create from existing cudf::column_view and contiguous buffer.
     */
    public OffHeapState(long viewHandle, DeviceMemoryBuffer contiguousBuffer) {
      assert viewHandle != 0;
      this.viewHandle = viewHandle;
      BaseDeviceMemoryBuffer valid = getValid();
      BaseDeviceMemoryBuffer data = getData();
      BaseDeviceMemoryBuffer offsets = getOffsets();
      toClose.add(data);
      toClose.add(valid);
      toClose.add(offsets);
      contiguousBuffer.incRefCount();
      toClose.add(contiguousBuffer);
    }

    public long getViewHandle() {
      if (viewHandle == 0) {
        viewHandle = ColumnVector.getNativeColumnView(columnHandle);
      }
      return viewHandle;
    }

    public long getNativeNullCount() {
      if (viewHandle != 0) {
        return ColumnView.getNativeNullCount(getViewHandle());
      }
      return getNativeNullCountColumn(columnHandle);
    }

    private void setNativeNullCount(int nullCount) throws CudfException {
      assert viewHandle == 0 : "Cannot set the null count if a view has already been created";
      assert columnHandle != 0;
      setNativeNullCountColumn(columnHandle, nullCount);
    }

    public BaseDeviceMemoryBuffer getData() {
      return getDataBuffer(getViewHandle());
    }

    public BaseDeviceMemoryBuffer getValid() {
      return getValidityBuffer(getViewHandle());
    }

    public BaseDeviceMemoryBuffer getOffsets() {
      return getOffsetsBuffer(getViewHandle());
    }

    @Override
    public void noWarnLeakExpected() {
      super.noWarnLeakExpected();

      BaseDeviceMemoryBuffer valid = getValid();
      BaseDeviceMemoryBuffer data = getData();
      BaseDeviceMemoryBuffer offsets = getOffsets();
      if (valid != null) {
        valid.noWarnLeakExpected();
      }
      if (data != null) {
        data.noWarnLeakExpected();
      }
      if(offsets != null) {
        offsets.noWarnLeakExpected();
      }
    }

    @Override
    public String toString() {
      return "(ID: " + id + " " + Long.toHexString(columnHandle == 0 ? viewHandle : columnHandle) + ")";
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long address = 0;

      // Always mark the resource as freed even if an exception is thrown.
      // We cannot know how far it progressed before the exception, and
      // therefore it is unsafe to retry.
      Throwable toThrow = null;
      if (viewHandle != 0) {
        address = viewHandle;
        try {
          ColumnView.deleteColumnView(viewHandle);
        } catch (Throwable t) {
          toThrow = t;
        } finally {
          viewHandle = 0;
        }
        neededCleanup = true;
      }
      if (columnHandle != 0) {
        if (address != 0) {
          address = columnHandle;
        }
        try {
          ColumnVector.deleteCudfColumn(columnHandle);
        } catch (Throwable t) {
          if (toThrow != null) {
            toThrow.addSuppressed(t);
          } else {
            toThrow = t;
          }
        } finally {
          columnHandle = 0;
        }
        neededCleanup = true;
      }
      if (!toClose.isEmpty()) {
        try {
          for (MemoryBuffer toCloseBuff : toClose) {
            if (toCloseBuff != null) {
              try {
                toCloseBuff.close();
              } catch (Throwable t) {
                if (toThrow != null) {
                  toThrow.addSuppressed(t);
                } else {
                  toThrow = t;
                }
              }
            }
          }
        } finally {
          toClose.clear();
        }
        neededCleanup = true;
      }
      if (toThrow != null) {
        throw new RuntimeException(toThrow);
      }
      if (neededCleanup) {
        if (logErrorIfNotClean) {
          log.error("A DEVICE COLUMN VECTOR WAS LEAKED (ID: " + id + " " + Long.toHexString(address)+ ")");
          logRefCountDebug("Leaked vector");
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return viewHandle == 0 && columnHandle == 0 && toClose.isEmpty();
    }
  }

  /////////////////////////////////////////////////////////////////////////////
  // BUILDER
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Create a new vector.
   * @param type       the type of vector to build.
   * @param rows       maximum number of rows that the vector can hold.
   * @param init       what will initialize the vector.
   * @return the created vector.
   */
  public static ColumnVector build(DType type, int rows, Consumer<Builder> init) {
    try (Builder builder = HostColumnVector.builder(type, rows)) {
      init.accept(builder);
      return builder.buildAndPutOnDevice();
    }
  }

  public static ColumnVector build(int rows, long stringBufferSize, Consumer<Builder> init) {
    try (Builder builder = HostColumnVector.builder(rows, stringBufferSize)) {
      init.accept(builder);
      return builder.buildAndPutOnDevice();
    }
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector boolFromBytes(byte... values) {
    return build(DType.BOOL8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * This method is evolving, unstable and currently test only.
   * Please use with caution and expect it to change in the future.
   */
  public static<T> ColumnVector fromLists(HostColumnVector.DataType dataType, List<T>... lists) {
    try (HostColumnVector host = HostColumnVector.fromLists(dataType, lists)) {
      return host.copyToDevice();
    }
  }

  /**
   * This method is evolving, unstable and currently test only.
   * Please use with caution and expect it to change in the future.
   */
  public static ColumnVector fromStructs(HostColumnVector.DataType dataType,
                                         List<HostColumnVector.StructData> lists) {
    try (HostColumnVector host = HostColumnVector.fromStructs(dataType, lists)) {
      return host.copyToDevice();
    }
  }

  /**
   * This method is evolving, unstable and currently test only.
   * Please use with caution and expect it to change in the future.
   */
  public static ColumnVector fromStructs(HostColumnVector.DataType dataType,
                                         HostColumnVector.StructData... lists) {
    try (HostColumnVector host = HostColumnVector.fromStructs(dataType, lists)) {
      return host.copyToDevice();
    }
  }
  /**
   * This method is evolving, unstable and currently test only.
   * Please use with caution and expect it to change in the future.
   */
  public static ColumnVector emptyStructs(HostColumnVector.DataType dataType, long numRows) {
    try (HostColumnVector host = HostColumnVector.emptyStructs(dataType, numRows)) {
      return host.copyToDevice();
    }
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromBooleans(boolean... values) {
    byte[] bytes = new byte[values.length];
    for (int i = 0; i < values.length; i++) {
      bytes[i] = values[i] ? (byte) 1 : (byte) 0;
    }
    return build(DType.BOOL8, values.length, (b) -> b.appendArray(bytes));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromBytes(byte... values) {
    return build(DType.INT8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned byte type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static ColumnVector fromUnsignedBytes(byte... values) {
    return build(DType.UINT8, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromShorts(short... values) {
    return build(DType.INT16, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned short type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static ColumnVector fromUnsignedShorts(short... values) {
    return build(DType.UINT16, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromInts(int... values) {
    return build(DType.INT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned int type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static ColumnVector fromUnsignedInts(int... values) {
    return build(DType.UINT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromLongs(long... values) {
    return build(DType.INT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   * <p>
   * Java does not have an unsigned long type, so the values will be
   * treated as if the bits represent an unsigned value.
   */
  public static ColumnVector fromUnsignedLongs(long... values) {
    return build(DType.UINT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromFloats(float... values) {
    return build(DType.FLOAT32, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector fromDoubles(double... values) {
    return build(DType.FLOAT64, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector daysFromInts(int... values) {
    return build(DType.TIMESTAMP_DAYS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector durationSecondsFromLongs(long... values) {
    return build(DType.DURATION_SECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_SECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector durationDaysFromInts(int... values) {
    return build(DType.DURATION_DAYS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector durationMilliSecondsFromLongs(long... values) {
    return build(DType.DURATION_MILLISECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampMilliSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_MILLISECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector durationMicroSecondsFromLongs(long... values) {
    return build(DType.DURATION_MICROSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampMicroSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_MICROSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector durationNanoSecondsFromLongs(long... values) {
    return build(DType.DURATION_NANOSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new vector from the given values.
   */
  public static ColumnVector timestampNanoSecondsFromLongs(long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendArray(values));
  }

  /**
   * Create a new decimal vector from unscaled values (int array) and scale.
   * The created vector is of type DType.DECIMAL32, whose max precision is 9.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static ColumnVector decimalFromInts(int scale, int... values) {
    try (HostColumnVector host = HostColumnVector.decimalFromInts(scale, values)) {
      return host.copyToDevice();
    }
  }

  /**
   * Create a new decimal vector from boxed unscaled values (Integer array) and scale.
   * The created vector is of type DType.DECIMAL32, whose max precision is 9.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static ColumnVector decimalFromBoxedInts(int scale, Integer... values) {
    try (HostColumnVector host = HostColumnVector.decimalFromBoxedInts(scale, values)) {
      return host.copyToDevice();
    }
  }

  /**
   * Create a new decimal vector from unscaled values (long array) and scale.
   * The created vector is of type DType.DECIMAL64, whose max precision is 18.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static ColumnVector decimalFromLongs(int scale, long... values) {
    try (HostColumnVector host = HostColumnVector.decimalFromLongs(scale, values)) {
      return host.copyToDevice();
    }
  }

  /**
   * Create a new decimal vector from boxed unscaled values (Long array) and scale.
   * The created vector is of type DType.DECIMAL64, whose max precision is 18.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static ColumnVector decimalFromBoxedLongs(int scale, Long... values) {
    try (HostColumnVector host = HostColumnVector.decimalFromBoxedLongs(scale, values)) {
      return host.copyToDevice();
    }
  }

  /**
   * Create a new decimal vector from double floats with specific DecimalType and RoundingMode.
   * All doubles will be rescaled if necessary, according to scale of input DecimalType and RoundingMode.
   * If any overflow occurs in extracting integral part, an IllegalArgumentException will be thrown.
   * This API is inefficient because of slow double -> decimal conversion, so it is mainly for testing.
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static ColumnVector decimalFromDoubles(DType type, RoundingMode mode, double... values) {
    try (HostColumnVector host = HostColumnVector.decimalFromDoubles(type, mode, values)) {
      return host.copyToDevice();
    }
  }


  /**
   * Create a new decimal vector from BigIntegers
   * Compared with scale of [[java.math.BigDecimal]], the scale here represents the opposite meaning.
   */
  public static ColumnVector decimalFromBigInt(int scale, BigInteger... values) {
    try (HostColumnVector host = HostColumnVector.decimalFromBigIntegers(scale, values)) {
      ColumnVector columnVector = host.copyToDevice();
      return columnVector;
    }
  }

  /**
   * Create a new string vector from the given values.  This API
   * supports inline nulls. This is really intended to be used only for testing as
   * it is slow and memory intensive to translate between java strings and UTF8 strings.
   */
  public static ColumnVector fromStrings(String... values) {
    try (HostColumnVector host = HostColumnVector.fromStrings(values)) {
      return host.copyToDevice();
    }
  }

  /**
   * Create a new string vector from the given values.  This API
   * supports inline nulls.
   */
  public static ColumnVector fromUTF8Strings(byte[]... values) {
    try (HostColumnVector host = HostColumnVector.fromUTF8Strings(values)) {
      return host.copyToDevice();
    }
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than building from primitive array of unscaledValues.
   * Notice:
   *  1. All input BigDecimals should share same scale.
   *  2. The scale will be zero if all input values are null.
   */
  public static ColumnVector fromDecimals(BigDecimal... values) {
    try (HostColumnVector hcv = HostColumnVector.fromDecimals(values)) {
      return hcv.copyToDevice();
    }
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedBooleans(Boolean... values) {
    return build(DType.BOOL8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedBytes(Byte... values) {
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
  public static ColumnVector fromBoxedUnsignedBytes(Byte... values) {
    return build(DType.UINT8, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedShorts(Short... values) {
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
  public static ColumnVector fromBoxedUnsignedShorts(Short... values) {
    return build(DType.UINT16, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedInts(Integer... values) {
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
  public static ColumnVector fromBoxedUnsignedInts(Integer... values) {
    return build(DType.UINT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedLongs(Long... values) {
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
  public static ColumnVector fromBoxedUnsignedLongs(Long... values) {
    return build(DType.UINT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedFloats(Float... values) {
    return build(DType.FLOAT32, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector fromBoxedDoubles(Double... values) {
    return build(DType.FLOAT64, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampDaysFromBoxedInts(Integer... values) {
    return build(DType.TIMESTAMP_DAYS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector durationDaysFromBoxedInts(Integer... values) {
    return build(DType.DURATION_DAYS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector durationSecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_SECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_SECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector durationMilliSecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_MILLISECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampMilliSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_MILLISECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector durationMicroSecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_MICROSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampMicroSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_MICROSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector durationNanoSecondsFromBoxedLongs(Long... values) {
    return build(DType.DURATION_NANOSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Create a new vector from the given values.  This API supports inline nulls,
   * but is much slower than using a regular array and should really only be used
   * for tests.
   */
  public static ColumnVector timestampNanoSecondsFromBoxedLongs(Long... values) {
    return build(DType.TIMESTAMP_NANOSECONDS, values.length, (b) -> b.appendBoxed(values));
  }

  /**
   * Creates an empty column according to the data type.
   *
   * It will create all the nested columns by iterating all the children in the input
   * type object 'colType'.
   *
   * The performance is not good, so use it carefully. We may want to move this implementation
   * to the native once figuring out a way to pass the nested data type to the native.
   *
   * @param colType the data type of the empty column
   * @return an empty ColumnVector with its children. Each children contains zero elements.
   * Users should close the ColumnVector to avoid memory leak.
   */
  public static ColumnVector empty(HostColumnVector.DataType colType) {
    if (colType == null || colType.getType() == null) {
      throw new IllegalArgumentException("The data type and its 'DType' should NOT be null.");
    }
    if (colType instanceof HostColumnVector.BasicType) {
      // Non nested type
      DType dt = colType.getType();
      return new ColumnVector(makeEmptyCudfColumn(dt.typeId.getNativeId(), dt.getScale()));
    } else if (colType instanceof HostColumnVector.ListType) {
      // List type
      assert colType.getNumChildren() == 1 : "List type requires one child type";
      try (ColumnVector child = empty(colType.getChild(0))) {
        return makeList(child);
      }
    } else if (colType instanceof HostColumnVector.StructType) {
      // Struct type
      ColumnVector[] children = new ColumnVector[colType.getNumChildren()];
      try {
        for (int i = 0; i < children.length; i++) {
          children[i] = empty(colType.getChild(i));
        }
        return makeStruct(children);
      } finally {
        for (ColumnVector cv : children) {
          if (cv != null) cv.close();
        }
      }
    } else {
      throw new IllegalArgumentException("Unsupported data type: " + colType);
    }
  }

  static ColumnVector[] getColumnVectorsFromPointers(long[] nativeHandles) {
    ColumnVector[] columns = new ColumnVector[nativeHandles.length];
    try {
      for (int i = 0; i < nativeHandles.length; i++) {
        long nativeHandle = nativeHandles[i];
        // setting address to zero, so we don't clean it in case of an exception as it
        // will be cleaned up by the constructor
        nativeHandles[i] = 0;
        columns[i] = new ColumnVector(nativeHandle);
      }
      return columns;
    } catch (Throwable t) {
      for (ColumnVector columnVector : columns) {
        if (columnVector != null) {
          try {
            columnVector.close();
          } catch (Throwable s) {
            t.addSuppressed(s);
          }
        }
      }
      for (long nativeHandle : nativeHandles) {
        if (nativeHandle != 0) {
          try {
            deleteCudfColumn(nativeHandle);
          } catch (Throwable s) {
            t.addSuppressed(s);
          }
        }
      }

      throw t;
    }
  }
}

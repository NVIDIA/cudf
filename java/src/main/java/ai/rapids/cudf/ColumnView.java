/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.*;
import java.util.stream.IntStream;

import static ai.rapids.cudf.HostColumnVector.OFFSET_SIZE;

/**
 * This class represents the column_view of a column analogous to its cudf cpp counterpart.
 * It holds view information like the native handle and other metadata for a column_view. It also
 * exposes APIs that would allow operations on a view.
 */
public class ColumnView implements AutoCloseable, BinaryOperable {

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  public static final long UNKNOWN_NULL_COUNT = -1;

  protected long viewHandle;
  protected final DType type;
  protected final long rows;
  protected final long nullCount;
  protected final ColumnVector.OffHeapState offHeap;

  /**
   * Constructs a Column View given a native view address. This asserts that if the ColumnView is
   * of nested-type it doesn't contain non-empty nulls
   * @param address the view handle
   * @throws AssertionError if the address points to a nested-type view with non-empty nulls
   */
  ColumnView(long address) {
    this.viewHandle = address;
    try {
      this.type = DType.fromNative(ColumnView.getNativeTypeId(viewHandle), ColumnView.getNativeTypeScale(viewHandle));
      this.rows = ColumnView.getNativeRowCount(viewHandle);
      this.nullCount = ColumnView.getNativeNullCount(viewHandle);
      this.offHeap = null;
      AssertEmptyNulls.assertNullsAreEmpty(this);
    } catch (Throwable t) {
      // offHeap state is null, so there is nothing to clean in offHeap
      // delete ColumnView to avoid memory leak
      deleteColumnView(viewHandle);
      viewHandle = 0;
      throw t;
    }
  }


  /**
   * Intended to be called from ColumnVector when it is being constructed. Because state creates a
   * cudf::column_view instance and will close it in all cases, we don't want to have to double
   * close it. This asserts that if the offHeapState is of nested-type it doesn't contain non-empty nulls
   * @param state the state this view is based off of.
   * @throws AssertionError if offHeapState points to a nested-type view with non-empty nulls
   */
  protected ColumnView(ColumnVector.OffHeapState state) {
    offHeap = state;
    try {
      viewHandle = state.getViewHandle();
      type = DType.fromNative(ColumnView.getNativeTypeId(viewHandle), ColumnView.getNativeTypeScale(viewHandle));
      rows = ColumnView.getNativeRowCount(viewHandle);
      nullCount = ColumnView.getNativeNullCount(viewHandle);
      AssertEmptyNulls.assertNullsAreEmpty(this);
    } catch (Throwable t) {
      // cleanup offHeap
      offHeap.clean(false);
      viewHandle = 0;
      throw t;
    }
  }

  /**
   * Create a new column view based off of data already on the device. Ref count on the buffers
   * is not incremented and none of the underlying buffers are owned by this view. The returned
   * ColumnView is only valid as long as the underlying buffers remain valid. If the buffers are
   * closed before this ColumnView is closed, it will result in undefined behavior.
   *
   * If ownership is needed, call {@link ColumnView#copyToColumnVector}
   *
   * @param type           the type of the vector
   * @param rows           the number of rows in this vector.
   * @param nullCount      the number of nulls in the dataset.
   * @param validityBuffer an optional validity buffer. Must be provided if nullCount != 0.
   *                       The ownership doesn't change on this buffer
   * @param offsetBuffer   a host buffer required for nested types including strings and string
   *                       categories. The ownership doesn't change on this buffer
   * @param children       an array of ColumnView children
   */
  public ColumnView(DType type, long rows, Optional<Long> nullCount,
                     BaseDeviceMemoryBuffer validityBuffer,
                     BaseDeviceMemoryBuffer offsetBuffer, ColumnView[] children) {
    this(type, (int) rows, nullCount.orElse(UNKNOWN_NULL_COUNT).intValue(),
        null, validityBuffer, offsetBuffer, children);
    assert(type.isNestedType());
    assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
        || !nullCount.isPresent();
  }

  /**
   * Create a new column view based off of data already on the device. Ref count on the buffers
   * is not incremented and none of the underlying buffers are owned by this view. The returned
   * ColumnView is only valid as long as the underlying buffers remain valid. If the buffers are
   * closed before this ColumnView is closed, it will result in undefined behavior.
   *
   * If ownership is needed, call {@link ColumnView#copyToColumnVector}
   *
   * @param type           the type of the vector
   * @param rows           the number of rows in this vector.
   * @param nullCount      the number of nulls in the dataset.
   * @param dataBuffer     a host buffer required for nested types including strings and string
   *                       categories. The ownership doesn't change on this buffer
   * @param validityBuffer an optional validity buffer. Must be provided if nullCount != 0.
   *                       The ownership doesn't change on this buffer
   */
  public ColumnView(DType type, long rows, Optional<Long> nullCount,
                    BaseDeviceMemoryBuffer dataBuffer,
                    BaseDeviceMemoryBuffer validityBuffer) {
    this(type, (int) rows, nullCount.orElse(UNKNOWN_NULL_COUNT).intValue(),
        dataBuffer, validityBuffer, null, null);
    assert (!type.isNestedType());
    assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
        || !nullCount.isPresent();
  }

  /**
   * Create a new column view based off of data already on the device. Ref count on the buffers
   * is not incremented and none of the underlying buffers are owned by this view. The returned
   * ColumnView is only valid as long as the underlying buffers remain valid. If the buffers are
   * closed before this ColumnView is closed, it will result in undefined behavior.
   *
   * If ownership is needed, call {@link ColumnView#copyToColumnVector}
   *
   * @param type           the type of the vector
   * @param rows           the number of rows in this vector.
   * @param nullCount      the number of nulls in the dataset.
   * @param dataBuffer     a host buffer required for nested types including strings and string
   *                       categories. The ownership doesn't change on this buffer
   * @param validityBuffer an optional validity buffer. Must be provided if nullCount != 0.
   *                       The ownership doesn't change on this buffer
   * @param offsetBuffer   The offsetbuffer for columns that need an offset buffer
   */
  public ColumnView(DType type, long rows, Optional<Long> nullCount,
                    BaseDeviceMemoryBuffer dataBuffer,
                    BaseDeviceMemoryBuffer validityBuffer, BaseDeviceMemoryBuffer offsetBuffer) {
    this(type, (int) rows, nullCount.orElse(UNKNOWN_NULL_COUNT).intValue(),
        dataBuffer, validityBuffer, offsetBuffer, null);
    assert (!type.isNestedType());
    assert (nullCount.isPresent() && nullCount.get() <= Integer.MAX_VALUE)
        || !nullCount.isPresent();
  }

  private ColumnView(DType type, long rows, int nullCount,
                     BaseDeviceMemoryBuffer dataBuffer, BaseDeviceMemoryBuffer validityBuffer,
                     BaseDeviceMemoryBuffer offsetBuffer, ColumnView[] children) {
    this(ColumnVector.initViewHandle(type, (int) rows, nullCount, dataBuffer, validityBuffer,
        offsetBuffer, children == null ? new long[]{} :
            Arrays.stream(children).mapToLong(c -> c.getNativeView()).toArray()));
  }

  /** Creates a ColumnVector from a column view handle
   * @return a new ColumnVector
   */
  public ColumnVector copyToColumnVector() {
    return new ColumnVector(ColumnView.copyColumnViewToCV(getNativeView()));
  }

  /**
   * USE WITH CAUTION: This method exposes the address of the native cudf::column_view.  This allows
   * writing custom kernels or other cuda operations on the data.  DO NOT close this column
   * vector until you are completely done using the native column_view.  DO NOT modify the column in
   * any way.  This should be treated as a read only data structure. This API is unstable as
   * the underlying C/C++ API is still not stabilized.  If the underlying data structure
   * is renamed this API may be replaced.  The underlying data structure can change from release
   * to release (it is not stable yet) so be sure that your native code is complied against the
   * exact same version of libcudf as this is released for.
   */
  public final long getNativeView() {
    return viewHandle;
  }

  static int getFixedPointOutputScale(BinaryOp op, DType lhsType, DType rhsType) {
    assert (lhsType.isDecimalType() && rhsType.isDecimalType());
    return fixedPointOutputScale(op.nativeId, lhsType.getScale(), rhsType.getScale());
  }

  private static native int fixedPointOutputScale(int op, int lhsScale, int rhsScale);

  public final DType getType() {
    return type;
  }

  /**
   * Returns the child column views for this view
   * Please note that it is the responsibility of the caller to close these views.
   * @return an array of child column views
   */
  public final ColumnView[] getChildColumnViews() {
    int numChildren = getNumChildren();
    if (!getType().isNestedType()) {
      return null;
    }
    ColumnView[] views = new ColumnView[numChildren];
    try {
      for (int i = 0; i < numChildren; i++) {
        views[i] = getChildColumnView(i);
      }
      return views;
    } catch(Throwable t) {
      for (ColumnView v: views) {
        if (v != null) {
          v.close();
        }
      }
      throw t;
    }
  }

  /**
   * Returns the child column view at a given index.
   * Please note that it is the responsibility of the caller to close this view.
   * @param childIndex the index of the child
   * @return a column view
   */
  public final ColumnView getChildColumnView(int childIndex) {
    int numChildren = getNumChildren();
    assert childIndex < numChildren : "children index should be less than " + numChildren;
    if (!getType().isNestedType()) {
      return null;
    }
    long childColumnView = ColumnView.getChildCvPointer(viewHandle, childIndex);
    return new ColumnView(childColumnView);
  }

  /**
   * Get a ColumnView that is the offsets for this list.
   * Please note that it is the responsibility of the caller to close this view, and the parent
   * column must out live this view.
   */
  public ColumnView getListOffsetsView() {
    assert(getType().equals(DType.LIST));
    return new ColumnView(getListOffsetCvPointer(viewHandle));
  }

  /**
   * Gets the data buffer for the current column view (viewHandle).
   * If the type is LIST, STRUCT it returns null.
   * @return    If the type is LIST, STRUCT or data buffer is empty it returns null,
   *            else return the data device buffer
   */
  public final BaseDeviceMemoryBuffer getData() {
    return getDataBuffer(viewHandle);
  }

  public final BaseDeviceMemoryBuffer getOffsets() {
    return getOffsetsBuffer(viewHandle);
  }

  public final BaseDeviceMemoryBuffer getValid() {
    return getValidityBuffer(viewHandle);
  }

  /**
   * Returns the number of nulls in the data. Note that this might end up
   * being a very expensive operation because if the null count is not
   * known it will be calculated.
   */
  public long getNullCount() {
    return nullCount;
  }

  /**
   * Returns the number of rows in this vector.
   */
  public final long getRowCount() {
    return rows;
  }

  public final int getNumChildren() {
    if (!getType().isNestedType()) {
      return 0;
    }
    return ColumnView.getNativeNumChildren(viewHandle);
  }


  /**
   * Returns the amount of device memory used.
   */
  public long getDeviceMemorySize() {
    return getDeviceMemorySize(getNativeView(), false);
  }

  @Override
  public void close() {
    // close the view handle so long as offHeap is not going to do it for us.
    if (offHeap == null) {
      ColumnView.deleteColumnView(viewHandle);
    }
    viewHandle = 0;
  }

  @Override
  public String toString() {
    return "ColumnView{" +
           "rows=" + rows +
           ", type=" + type +
           ", nullCount=" + nullCount +
           '}';
  }

  /**
   * Used for string strip function.
   * Indicates characters to be stripped from the beginning, end, or both of each string.
   */
  private enum StripType {
    LEFT(0),   // strip characters from the beginning of the string
    RIGHT(1),  // strip characters from the end of the string
    BOTH(2);   // strip characters from the beginning and end of the string
    final int nativeId;

    StripType(int nativeId) { this.nativeId = nativeId; }
  }

  /**
   * Returns a new ColumnVector with NaNs converted to nulls, preserving the existing null values.
   */
  public final ColumnVector nansToNulls() {
    assert type.equals(DType.FLOAT32) || type.equals(DType.FLOAT64);
    return new ColumnVector(nansToNulls(this.getNativeView()));
  }

  /////////////////////////////////////////////////////////////////////////////
  // DEVICE METADATA
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Retrieve the number of characters in each string. Null strings will have value of null.
   *
   * @return ColumnVector holding length of string at index 'i' in the original vector
   */
  public final ColumnVector getCharLengths() {
    assert DType.STRING.equals(type) : "char length only available for String type";
    return new ColumnVector(charLengths(getNativeView()));
  }

  /**
   * Retrieve the number of bytes for each string. Null strings will have value of null.
   *
   * @return ColumnVector, where each element at i = byte count of string at index 'i' in the original vector
   */
  public final ColumnVector getByteCount() {
    assert type.equals(DType.STRING) : "type has to be a String";
    return new ColumnVector(byteCount(getNativeView()));
  }

  /**
   * Get the code point values (integers) for each character of each string.
   *
   * @return ColumnVector, with code point integer values for each character as INT32
   */
  public final ColumnVector codePoints() {
    assert type.equals(DType.STRING) : "type has to be a String";
    return new ColumnVector(codePoints(getNativeView()));
  }

  /**
   * Get the number of elements for each list. Null lists will have a value of null.
   * @return the number of elements in each list as an INT32 value.
   */
  public final ColumnVector countElements() {
    assert DType.LIST.equals(type) : "Only lists are supported";
    return new ColumnVector(countElements(getNativeView()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is not null, and FALSE for any null entry (as per the validity mask)
   *
   * @return - Boolean vector
   */
  public final ColumnVector isNotNull() {
    return new ColumnVector(isNotNullNative(getNativeView()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * FALSE for any entry that is not null, and TRUE for any null entry (as per the validity mask)
   *
   * @return - Boolean vector
   */
  public final ColumnVector isNull() {
    return new ColumnVector(isNullNative(getNativeView()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is a fixed-point, and FALSE if its not a fixed-point.
   * A null will be returned for null entries.
   *
   * The sign and the exponent is optional. The decimal point may only appear once.
   * The integer component must fit within the size limits of the underlying fixed-point
   * storage type. The value of the integer component is based on the scale of the target
   * decimalType.
   *
   * Example:
   * vec = ["A", "nan", "Inf", "-Inf", "Infinity", "infinity", "2.1474", "112.383", "-2.14748",
   *        "NULL", "null", null, "1.2", "1.2e-4", "0.00012"]
   * vec.isFixedPoint() = [false, false, false, false, false, false, true, true, true, false, false,
   *                       null, true, true, true]
   *
   * @param decimalType the data type that should be used for bounds checking. Note that only
   *                Decimal types (fixed-point) are allowed.
   * @return Boolean vector
   */
  public final ColumnVector isFixedPoint(DType decimalType) {
    assert type.equals(DType.STRING);
    assert decimalType.isDecimalType();
    return new ColumnVector(isFixedPoint(getNativeView(),
        decimalType.getTypeId().getNativeId(), decimalType.getScale()));
  }


  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is an integer, and FALSE if its not an integer. A null will be returned
   * for null entries.
   *
   * NOTE: Integer doesn't mean a 32-bit integer. It means a number that is not a fraction.
   * i.e. If this method returns true for a value it could still result in an overflow or underflow
   * if you convert it to a Java integral type
   *
   * @return Boolean vector
   */
  public final ColumnVector isInteger() {
    assert type.equals(DType.STRING);
    return new ColumnVector(isInteger(getNativeView()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is an integer, and FALSE if its not an integer. A null will be returned
   * for null entries.
   *
   * @param intType the data type that should be used for bounds checking. Note that only
   *                cudf integer types are allowed including signed/unsigned int8 through int64
   * @return Boolean vector
   */
  public final ColumnVector isInteger(DType intType) {
    assert type.equals(DType.STRING);
    assert intType.isBackedByInt() || intType.isBackedByLong() || intType.isBackedByByte()
        || intType.isBackedByShort();
    return new ColumnVector(isIntegerWithType(getNativeView(),
        intType.getTypeId().getNativeId(), intType.getScale()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is a float, and FALSE if its not a float. A null will be returned
   * for null entries
   *
   * NOTE: Float doesn't mean a 32-bit float. It means a number that is a fraction or can be written
   * as a fraction. i.e. This method will return true for integers as well as floats. Also note if
   * this method returns true for a value it could still result in an overflow or underflow if you
   * convert it to a Java float or double
   *
   * @return - Boolean vector
   */
  public final ColumnVector isFloat() {
    assert type.equals(DType.STRING);
    return new ColumnVector(isFloat(getNativeView()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is NaN, and FALSE if null or a valid floating point value
   * @return - Boolean vector
   */
  public final ColumnVector isNan() {
    return new ColumnVector(isNanNative(getNativeView()));
  }

  /**
   * Returns a Boolean vector with the same number of rows as this instance, that has
   * TRUE for any entry that is null or a valid floating point value, FALSE otherwise
   * @return - Boolean vector
   */
  public final ColumnVector isNotNan() {
    return new ColumnVector(isNotNanNative(getNativeView()));
  }

  /////////////////////////////////////////////////////////////////////////////
  // Replacement
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Returns a vector with all values "oldValues[i]" replaced with "newValues[i]".
   * Warning:
   *    Currently this function doesn't work for Strings or StringCategories.
   *    NaNs can't be replaced in the original vector but regular values can be replaced with NaNs
   *    Nulls can't be replaced in the original vector but regular values can be replaced with Nulls
   *    Mixing of types isn't allowed, the resulting vector will be the same type as the original.
   *      e.g. You can't replace an integer vector with values from a long vector
   *
   * Usage:
   *    this = {1, 4, 5, 1, 5}
   *    oldValues = {1, 5, 7}
   *    newValues = {2, 6, 9}
   *
   *    result = this.findAndReplaceAll(oldValues, newValues);
   *    result = {2, 4, 6, 2, 6}  (1 and 5 replaced with 2 and 6 but 7 wasn't found so no change)
   *
   * @param oldValues - A vector containing values that should be replaced
   * @param newValues - A vector containing new values
   * @return - A new vector containing the old values replaced with new values
   */
  public final ColumnVector findAndReplaceAll(ColumnView oldValues, ColumnView newValues) {
    return new ColumnVector(findAndReplaceAll(oldValues.getNativeView(), newValues.getNativeView(), this.getNativeView()));
  }

  /**
   * Returns a ColumnVector with any null values replaced with a scalar.
   * The types of the input ColumnVector and Scalar must match, else an error is thrown.
   *
   * @param scalar - Scalar value to use as replacement
   * @return - ColumnVector with nulls replaced by scalar
   */
  public final ColumnVector replaceNulls(Scalar scalar) {
    return new ColumnVector(replaceNullsScalar(getNativeView(), scalar.getScalarHandle()));
  }

  /**
   * Returns a ColumnVector with any null values replaced with the corresponding row in the
   * specified replacement column.
   * This column and the replacement column must have the same type and number of rows.
   *
   * @param replacements column of replacement values
   * @return column with nulls replaced by corresponding row of replacements column
   */
  public final ColumnVector replaceNulls(ColumnView replacements) {
    return new ColumnVector(replaceNullsColumn(getNativeView(), replacements.getNativeView()));
  }

  public final ColumnVector replaceNulls(ReplacePolicy policy) {
    return new ColumnVector(replaceNullsPolicy(getNativeView(), policy.isPreceding));
  }

  /**
   * For a BOOL8 vector, computes a vector whose rows are selected from two other vectors
   * based on the boolean value of this vector in the corresponding row.
   * If the boolean value in a row is true, the corresponding row is selected from trueValues
   * otherwise the corresponding row from falseValues is selected.
   * Note that trueValues and falseValues vectors must be the same length as this vector,
   * and trueValues and falseValues must have the same data type.
   * @param trueValues the values to select if a row in this column is true
   * @param falseValues the values to select if a row in this column is not true
   * @return the computed vector
   */
  public final ColumnVector ifElse(ColumnView trueValues, ColumnView falseValues) {
    if (!type.equals(DType.BOOL8)) {
      throw new IllegalArgumentException("Cannot select with a predicate vector of type " + type);
    }
    long result = ifElseVV(getNativeView(), trueValues.getNativeView(), falseValues.getNativeView());
    return new ColumnVector(result);
  }

  /**
   * For a BOOL8 vector, computes a vector whose rows are selected from two other inputs
   * based on the boolean value of this vector in the corresponding row.
   * If the boolean value in a row is true, the corresponding row is selected from trueValues
   * otherwise the value from falseValue is selected.
   * Note that trueValues must be the same length as this vector,
   * and trueValues and falseValue must have the same data type.
   * Note that the trueValues vector and falseValue scalar must have the same data type.
   * @param trueValues the values to select if a row in this column is true
   * @param falseValue the value to select if a row in this column is not true
   * @return the computed vector
   */
  public final ColumnVector ifElse(ColumnView trueValues, Scalar falseValue) {
    if (!type.equals(DType.BOOL8)) {
      throw new IllegalArgumentException("Cannot select with a predicate vector of type " + type);
    }
    long result = ifElseVS(getNativeView(), trueValues.getNativeView(), falseValue.getScalarHandle());
    return new ColumnVector(result);
  }

  /**
   * For a BOOL8 vector, computes a vector whose rows are selected from two other inputs
   * based on the boolean value of this vector in the corresponding row.
   * If the boolean value in a row is true, the value from trueValue is selected
   * otherwise the corresponding row from falseValues is selected.
   * Note that falseValues must be the same length as this vector,
   * and trueValue and falseValues must have the same data type.
   * Note that the trueValue scalar and falseValues vector must have the same data type.
   * @param trueValue the value to select if a row in this column is true
   * @param falseValues the values to select if a row in this column is not true
   * @return the computed vector
   */
  public final ColumnVector ifElse(Scalar trueValue, ColumnView falseValues) {
    if (!type.equals(DType.BOOL8)) {
      throw new IllegalArgumentException("Cannot select with a predicate vector of type " + type);
    }
    long result = ifElseSV(getNativeView(), trueValue.getScalarHandle(), falseValues.getNativeView());
    return new ColumnVector(result);
  }

  /**
   * For a BOOL8 vector, computes a vector whose rows are selected from two other inputs
   * based on the boolean value of this vector in the corresponding row.
   * If the boolean value in a row is true, the value from trueValue is selected
   * otherwise the value from falseValue is selected.
   * Note that the trueValue and falseValue scalars must have the same data type.
   * @param trueValue the value to select if a row in this column is true
   * @param falseValue the value to select if a row in this column is not true
   * @return the computed vector
   */
  public final ColumnVector ifElse(Scalar trueValue, Scalar falseValue) {
    if (!type.equals(DType.BOOL8)) {
      throw new IllegalArgumentException("Cannot select with a predicate vector of type " + type);
    }
    long result = ifElseSS(getNativeView(), trueValue.getScalarHandle(), falseValue.getScalarHandle());
    return new ColumnVector(result);
  }

  /////////////////////////////////////////////////////////////////////////////
  // Slice/Split and Concatenate
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Slices a column (including null values) into a set of columns
   * according to a set of indices. The caller owns the ColumnVectors and is responsible
   * closing them
   *
   * The "slice" function divides part of the input column into multiple intervals
   * of rows using the indices values and it stores the intervals into the output
   * columns. Regarding the interval of indices, a pair of values are taken from
   * the indices array in a consecutive manner. The pair of indices are left-closed
   * and right-open.
   *
   * The pairs of indices in the array are required to comply with the following
   * conditions:
   * a, b belongs to Range[0, input column size]
   * a <= b, where the position of a is less or equal to the position of b.
   *
   * Exceptional cases for the indices array are:
   * When the values in the pair are equal, the function returns an empty column.
   * When the values in the pair are 'strictly decreasing', the outcome is
   * undefined.
   * When any of the values in the pair don't belong to the range[0, input column
   * size), the outcome is undefined.
   * When the indices array is empty, an empty vector of columns is returned.
   *
   * The caller owns the output ColumnVectors and is responsible for closing them.
   *
   * @param indices
   * @return A new ColumnVector array with slices from the original ColumnVector
   */
  public final ColumnVector[] slice(int... indices) {
    long[] nativeHandles = slice(this.getNativeView(), indices);
    ColumnVector[] columnVectors = new ColumnVector[nativeHandles.length];
    try {
      for (int i = 0; i < nativeHandles.length; i++) {
        long nativeHandle = nativeHandles[i];
        // setting address to zero, so we don't clean it in case of an exception as it
        // will be cleaned up by the constructor
        nativeHandles[i] = 0;
        columnVectors[i] = new ColumnVector(nativeHandle);
      }
    } catch (Throwable t) {
      try {
        cleanupColumnViews(nativeHandles, columnVectors, t);
      } catch (Throwable s) {
        t.addSuppressed(s);
      } finally {
        throw t;
      }
    }
    return columnVectors;
  }

  /**
   * Return a subVector from start inclusive to the end of the vector.
   * @param start the index to start at.
   */
  public final ColumnVector subVector(int start) {
    return subVector(start, (int)rows);
  }

  /**
   * Return a subVector.
   * @param start the index to start at (inclusive).
   * @param end the index to end at (exclusive).
   */
  public final ColumnVector subVector(int start, int end) {
    ColumnVector [] tmp = slice(start, end);
    assert tmp.length == 1;
    return tmp[0];
  }

  /**
   * Splits a column (including null values) into a set of columns
   * according to a set of indices. The caller owns the ColumnVectors and is responsible
   * closing them.
   *
   * The "split" function divides the input column into multiple intervals
   * of rows using the splits indices values and it stores the intervals into the
   * output columns. Regarding the interval of indices, a pair of values are taken
   * from the indices array in a consecutive manner. The pair of indices are
   * left-closed and right-open.
   *
   * The indices array ('splits') is require to be a monotonic non-decreasing set.
   * The indices in the array are required to comply with the following conditions:
   * a, b belongs to Range[0, input column size]
   * a <= b, where the position of a is less or equal to the position of b.
   *
   * The split function will take a pair of indices from the indices array
   * ('splits') in a consecutive manner. For the first pair, the function will
   * take the value 0 and the first element of the indices array. For the last pair,
   * the function will take the last element of the indices array and the size of
   * the input column.
   *
   * Exceptional cases for the indices array are:
   * When the values in the pair are equal, the function return an empty column.
   * When the values in the pair are 'strictly decreasing', the outcome is
   * undefined.
   * When any of the values in the pair don't belong to the range[0, input column
   * size), the outcome is undefined.
   * When the indices array is empty, an empty vector of columns is returned.
   *
   * The input columns may have different sizes. The number of
   * columns must be equal to the number of indices in the array plus one.
   *
   * Example:
   * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
   * splits: {2, 5, 9}
   * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
   *
   * Note that this is very similar to the output from a PartitionedTable.
   *
   * @param indices the indexes to split with
   * @return A new ColumnVector array with slices from the original ColumnVector
   */
  public final ColumnVector[] split(int... indices) {
    ColumnView[] views = splitAsViews(indices);
    ColumnVector[] columnVectors = new ColumnVector[views.length];
    try {
      for (int i = 0; i < views.length; i++) {
        columnVectors[i] = views[i].copyToColumnVector();
      }
      return columnVectors;
    } catch (Throwable t) {
      for (ColumnVector cv : columnVectors) {
        if (cv != null) {
          cv.close();
        }
      }
      throw t;
    } finally {
      for (ColumnView view : views) {
        view.close();
      }
    }
  }

  /**
   * Splits a ColumnView (including null values) into a set of ColumnViews
   * according to a set of indices. No data is moved or copied.
   *
   * IMPORTANT NOTE: Nothing is copied out from the vector and the slices will only be relevant for
   * the lifecycle of the underlying ColumnVector.
   *
   * The "split" function divides the input column into multiple intervals
   * of rows using the splits indices values and it stores the intervals into the
   * output columns. Regarding the interval of indices, a pair of values are taken
   * from the indices array in a consecutive manner. The pair of indices are
   * left-closed and right-open.
   *
   * The indices array ('splits') is required to be a monotonic non-decreasing set.
   * The indices in the array are required to comply with the following conditions:
   * a, b belongs to Range[0, input column size]
   * a <= b, where the position of 'a' is less or equal to the position of 'b'.
   *
   * The split function will take a pair of indices from the indices array
   * ('splits') in a consecutive manner. For the first pair, the function will
   * take the value 0 and the first element of the indices array. For the last pair,
   * the function will take the last element of the indices array and the size of
   * the input column.
   *
   * Exceptional cases for the indices array are:
   * When the values in the pair are equal, the function return an empty column.
   * When the values in the pair are 'strictly decreasing', the outcome is
   * undefined.
   * When any of the values in the pair don't belong to the range[0, input column
   * size), the outcome is undefined.
   * When the indices array is empty, an empty array of ColumnViews is returned.
   *
   * The output columns may have different sizes. The number of
   * columns must be equal to the number of indices in the array plus one.
   *
   * Example:
   * input:   {10, 12, 14, 16, 18, 20, 22, 24, 26, 28}
   * splits: {2, 5, 9}
   * output:  {{10, 12}, {14, 16, 18}, {20, 22, 24, 26}, {28}}
   *
   * Note that this is very similar to the output from a PartitionedTable.
   *
   *
   * @param indices the indices to split with
   * @return A new ColumnView array with slices from the original ColumnView
   */
  public ColumnView[] splitAsViews(int... indices) {
    long[] nativeHandles = split(this.getNativeView(), indices);
    ColumnView[] columnViews = new ColumnView[nativeHandles.length];
    try {
      for (int i = 0; i < nativeHandles.length; i++) {
        long nativeHandle = nativeHandles[i];
        // setting address to zero, so we don't clean it in case of an exception as it
        // will be cleaned up by the constructor
        nativeHandles[i] = 0;
        columnViews[i] = new ColumnView(nativeHandle);
      }
    } catch (Throwable t) {
      try {
        cleanupColumnViews(nativeHandles, columnViews, t);
      } catch (Throwable s) {
        t.addSuppressed(s);
      } finally {
        throw t;
      }
    }
    return columnViews;
  }

  static void cleanupColumnViews(long[] nativeHandles, ColumnView[] columnViews, Throwable throwable) {
    for (ColumnView columnView : columnViews) {
      if (columnView != null) {
        try {
          columnView.close();
        } catch (Throwable s) {
          throwable.addSuppressed(s);
        }
      }
    }
    for (long nativeHandle : nativeHandles) {
      if (nativeHandle != 0) {
        try {
          deleteColumnView(nativeHandle);
        } catch (Throwable s) {
          throwable.addSuppressed(s);
        }
      }
    }
  }

  /**
   * Create a new vector of "normalized" values, where:
   *  1. All representations of NaN (and -NaN) are replaced with the normalized NaN value
   *  2. All elements equivalent to 0.0 (including +0.0 and -0.0) are replaced with +0.0.
   *  3. All elements that are not equivalent to NaN or 0.0 remain unchanged.
   *
   * The documentation for {@link Double#longBitsToDouble(long)}
   * describes how equivalent values of NaN/-NaN might have different bitwise representations.
   *
   * This method may be used to compare different bitwise values of 0.0 or NaN as logically
   * equivalent. For instance, if these values appear in a groupby key column, without normalization
   * 0.0 and -0.0 would be erroneously treated as distinct groups, as will each representation of NaN.
   *
   * @return A new ColumnVector with all elements equivalent to NaN/0.0 replaced with a normalized equivalent.
   */
  public final ColumnVector normalizeNANsAndZeros() {
    return new ColumnVector(normalizeNANsAndZeros(getNativeView()));
  }

  /**
   * Replace the null mask of a column. The resultant null mask is the bitwise {@code mergeOp} of
   * null masks in the columns given as arguments, AND-ed with this column's existing null mask.
   *
   * If applying the null mask would be a no-op, the original column is returned with incremented
   * refcount. Otherwise, a deep copy of the column is made (for a non-owning ColumnView, a deep copy
   * must be made in either case).
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
   * @deprecated Use {@link ColumnVector#mergeAndSetValidity(BinaryOp, ColumnView...)} instead.
   */
  @Deprecated
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
      return copyToColumnVector();
    }
    return new ColumnVector(mergeOutput);
  }

  /////////////////////////////////////////////////////////////////////////////
  // DATE/TIME
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Extract a particular date time component from a timestamp.
   * @param component what should be extracted
   * @return a column with the extracted information in it.
   */
  public final ColumnVector extractDateTimeComponent(DateTimeComponent component) {
    assert type.isTimestampType();
    return new ColumnVector(extractDateTimeComponent(getNativeView(), component.getNativeId()));
  }

  /**
   * Get year from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public final ColumnVector year() {
    return extractDateTimeComponent(DateTimeComponent.YEAR);
  }

  /**
   * Get month from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public final ColumnVector month() {
    return extractDateTimeComponent(DateTimeComponent.MONTH);
  }

  /**
   * Get day from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public final ColumnVector day() {
    return extractDateTimeComponent(DateTimeComponent.DAY);
  }

  /**
   * Get hour from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public final ColumnVector hour() {
    return extractDateTimeComponent(DateTimeComponent.HOUR);
  }

  /**
   * Get minute from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return - A new INT16 vector allocated on the GPU.
   */
  public final ColumnVector minute() {
    return extractDateTimeComponent(DateTimeComponent.MINUTE);
  }

  /**
   * Get second from a timestamp with time resolution.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new INT16 vector allocated on the GPU.
   */
  public final ColumnVector second() {
    return extractDateTimeComponent(DateTimeComponent.SECOND);
  }

  /**
   * Get the day of the week from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new INT16 vector allocated on the GPU. Monday=1, ..., Sunday=7
   */
  public final ColumnVector weekDay() {
    return extractDateTimeComponent(DateTimeComponent.WEEKDAY);
  }

  /**
   * Get the date that is the last day of the month for this timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new TIMESTAMP_DAYS vector allocated on the GPU.
   */
  public final ColumnVector lastDayOfMonth() {
    assert type.isTimestampType();
    return new ColumnVector(lastDayOfMonth(getNativeView()));
  }

  /**
   * Get the day of the year from a timestamp.
   * <p>
   * Postconditions - A new vector is allocated with the result. The caller owns the vector and
   * is responsible for its lifecycle.
   * @return A new INT16 vector allocated on the GPU. The value is between [1, {365-366}]
   */
  public final ColumnVector dayOfYear() {
    assert type.isTimestampType();
    return new ColumnVector(dayOfYear(getNativeView()));
  }

  /**
   * Get the quarter of the year from a timestamp.
   * @return A new INT16 vector allocated on the GPU. It will be a value from {1, 2, 3, 4}
   * corresponding to the quarter of the year.
   */
  public final ColumnVector quarterOfYear() {
    assert type.isTimestampType();
    return new ColumnVector(quarterOfYear(getNativeView()));
  }

  /**
   * Add the specified number of months to the timestamp.
   * @param months must be a INT16 column indicating the number of months to add. A negative number
   *               of months works too.
   * @return the updated timestamp
   */
  public final ColumnVector addCalendricalMonths(ColumnView months) {
    return new ColumnVector(addCalendricalMonths(getNativeView(), months.getNativeView()));
  }

  /**
   * Add the specified number of months to the timestamp.
   * @param months must be a INT16 scalar indicating the number of months to add. A negative number
   *               of months works too.
   * @return the updated timestamp
   */
  public final ColumnVector addCalendricalMonths(Scalar months) {
    return new ColumnVector(addScalarCalendricalMonths(getNativeView(), months.getScalarHandle()));
  }

  /**
   * Check to see if the year for this timestamp is a leap year or not.
   * @return BOOL8 vector of results
   */
  public final ColumnVector isLeapYear() {
    return new ColumnVector(isLeapYear(getNativeView()));
  }

  /**
   * Extract the number of days in the month
   * @return INT16 column of the number of days in the corresponding month
   */
  public final ColumnVector daysInMonth() {
    assert type.isTimestampType();
    return new ColumnVector(daysInMonth(getNativeView()));
  }

  /**
   * Round the timestamp up to the given frequency keeping the type the same.
   * @param freq what part of the timestamp to round.
   * @return a timestamp with the same type, but rounded up.
   */
  public final ColumnVector dateTimeCeil(DateTimeRoundingFrequency freq) {
    assert type.isTimestampType();
    return new ColumnVector(dateTimeCeil(getNativeView(), freq.getNativeId()));
  }

  /**
   * Round the timestamp down to the given frequency keeping the type the same.
   * @param freq what part of the timestamp to round.
   * @return a timestamp with the same type, but rounded down.
   */
  public final ColumnVector dateTimeFloor(DateTimeRoundingFrequency freq) {
    assert type.isTimestampType();
    return new ColumnVector(dateTimeFloor(getNativeView(), freq.getNativeId()));
  }

  /**
   * Round the timestamp (half up) to the given frequency keeping the type the same.
   * @param freq what part of the timestamp to round.
   * @return a timestamp with the same type, but rounded (half up).
   */
  public final ColumnVector dateTimeRound(DateTimeRoundingFrequency freq) {
    assert type.isTimestampType();
    return new ColumnVector(dateTimeRound(getNativeView(), freq.getNativeId()));
  }


  /////////////////////////////////////////////////////////////////////////////
  // ARITHMETIC
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Transform a vector using a custom function.  Be careful this is not
   * simple to do.  You need to be positive you know what type of data you are
   * processing and how the data is laid out.  This also only works on fixed
   * length types.
   * @param udf This function will be applied to every element in the vector
   * @param isPtx is the code of the function ptx? true or C/C++ false.
   */
  public final ColumnVector transform(String udf, boolean isPtx) {
    return new ColumnVector(transform(getNativeView(), udf, isPtx));
  }

  /**
   * Multiple different unary operations. The output is the same type as input.
   * @param op      the operation to perform
   * @return the result
   */
  public final ColumnVector unaryOp(UnaryOp op) {
    return new ColumnVector(unaryOperation(getNativeView(), op.nativeId));
  }

  /**
   * Calculate the sin, output is the same type as input.
   */
  public final ColumnVector sin() {
    return unaryOp(UnaryOp.SIN);
  }

  /**
   * Calculate the cos, output is the same type as input.
   */
  public final ColumnVector cos() {
    return unaryOp(UnaryOp.COS);
  }

  /**
   * Calculate the tan, output is the same type as input.
   */
  public final ColumnVector tan() {
    return unaryOp(UnaryOp.TAN);
  }

  /**
   * Calculate the arcsin, output is the same type as input.
   */
  public final ColumnVector arcsin() {
    return unaryOp(UnaryOp.ARCSIN);
  }

  /**
   * Calculate the arccos, output is the same type as input.
   */
  public final ColumnVector arccos() {
    return unaryOp(UnaryOp.ARCCOS);
  }

  /**
   * Calculate the arctan, output is the same type as input.
   */
  public final ColumnVector arctan() {
    return unaryOp(UnaryOp.ARCTAN);
  }

  /**
   * Calculate the hyperbolic sin, output is the same type as input.
   */
  public final ColumnVector sinh() {
    return unaryOp(UnaryOp.SINH);
  }

  /**
   * Calculate the hyperbolic cos, output is the same type as input.
   */
  public final ColumnVector cosh() {
    return unaryOp(UnaryOp.COSH);
  }

  /**
   * Calculate the hyperbolic tan, output is the same type as input.
   */
  public final ColumnVector tanh() {
    return unaryOp(UnaryOp.TANH);
  }

  /**
   * Calculate the hyperbolic arcsin, output is the same type as input.
   */
  public final ColumnVector arcsinh() {
    return unaryOp(UnaryOp.ARCSINH);
  }

  /**
   * Calculate the hyperbolic arccos, output is the same type as input.
   */
  public final ColumnVector arccosh() {
    return unaryOp(UnaryOp.ARCCOSH);
  }

  /**
   * Calculate the hyperbolic arctan, output is the same type as input.
   */
  public final ColumnVector arctanh() {
    return unaryOp(UnaryOp.ARCTANH);
  }

  /**
   * Calculate the exp, output is the same type as input.
   */
  public final ColumnVector exp() {
    return unaryOp(UnaryOp.EXP);
  }

  /**
   * Calculate the log, output is the same type as input.
   */
  public final ColumnVector log() {
    return unaryOp(UnaryOp.LOG);
  }

  /**
   * Calculate the log with base 2, output is the same type as input.
   */
  public final ColumnVector log2() {
    try (Scalar base = Scalar.fromInt(2)) {
      return binaryOp(BinaryOp.LOG_BASE, base, getType());
    }
  }

  /**
   * Calculate the log with base 10, output is the same type as input.
   */
  public final ColumnVector log10() {
    try (Scalar base = Scalar.fromInt(10)) {
      return binaryOp(BinaryOp.LOG_BASE, base, getType());
    }
  }

  /**
   * Calculate the sqrt, output is the same type as input.
   */
  public final ColumnVector sqrt() {
    return unaryOp(UnaryOp.SQRT);
  }

  /**
   * Calculate the cube root, output is the same type as input.
   */
  public final ColumnVector cbrt() {
    return unaryOp(UnaryOp.CBRT);
  }

  /**
   * Calculate the ceil, output is the same type as input.
   */
  public final ColumnVector ceil() {
    return unaryOp(UnaryOp.CEIL);
  }

  /**
   * Calculate the floor, output is the same type as input.
   */
  public final ColumnVector floor() {
    return unaryOp(UnaryOp.FLOOR);
  }

  /**
   * Calculate the abs, output is the same type as input.
   */
  public final ColumnVector abs() {
    return unaryOp(UnaryOp.ABS);
  }

  /**
   * Rounds a floating-point argument to the closest integer value, but returns it as a float.
   */
  public final ColumnVector rint() {
    return unaryOp(UnaryOp.RINT);
  }

  /**
   * Count the number of set bit for each integer value.
   */
  public final ColumnVector bitCount() {
    return unaryOp(UnaryOp.BIT_COUNT);
  }

  /**
   * Invert the bits, output is the same type as input.
   * For BOOL8 type, this is equivalent to logical not (UnaryOp.NOT), but this does not
   * matter since Spark does not support bitwise inverting on boolean type.
   */
  public final ColumnVector bitInvert() {
    return unaryOp(UnaryOp.BIT_INVERT);
  }

  /**
   * Multiple different binary operations.
   * @param op      the operation to perform
   * @param rhs     the rhs of the operation
   * @param outType the type of output you want.
   * @return the result
   */
  @Override
  public final ColumnVector binaryOp(BinaryOp op, BinaryOperable rhs, DType outType) {
    if (rhs instanceof ColumnView) {
      assert rows == ((ColumnView) rhs).getRowCount();
      return new ColumnVector(binaryOp(this, (ColumnView) rhs, op, outType));
    } else {
      return new ColumnVector(binaryOp(this, (Scalar) rhs, op, outType));
    }
  }

  static long binaryOp(ColumnView lhs, ColumnView rhs, BinaryOp op, DType outputType) {
    return binaryOpVV(lhs.getNativeView(), rhs.getNativeView(),
        op.nativeId, outputType.typeId.getNativeId(), outputType.getScale());
  }

  static long binaryOp(ColumnView lhs, Scalar rhs, BinaryOp op, DType outputType) {
    return binaryOpVS(lhs.getNativeView(), rhs.getScalarHandle(),
        op.nativeId, outputType.typeId.getNativeId(), outputType.getScale());
  }

  /////////////////////////////////////////////////////////////////////////////
  // AGGREGATION
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Computes the sum of all values in the column, returning a scalar
   * of the same type as this column.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 4, 5], DType = INT32
   * Scalar result = col.sum();
   * // result = 15,              DType = INT32
   * }</pre>
   */
  public Scalar sum() {
    return sum(type);
  }

  /**
   * Computes the sum of all values in the column, returning a scalar
   * of the specified type.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 4, 5], DType = INT32
   * Scalar result = col.sum(DType.INT64);
   * // result = 15,              DType = INT64
   * }</pre>
   *
   * @param outType the output DType
   */
  public Scalar sum(DType outType) {
    return reduce(ReductionAggregation.sum(), outType);
  }

  /**
   * Find and return a scalar of type INT32 containing index of the minimum value in the column.
   * If there are multiple minimum values, index of the first encountered minimum value is returned.
   * If the input column is empty or all nulls, an empty scalar is returned.
   *
   * If the input is floating point type and contains NaNs, the result is undefined.
   *
   * Example:
   * <pre>{@code
   * // col    = [5, 2, 8, 1, 9], DType = INT32
   * Scalar result = col.argMin();
   * // result = 3,               DType = INT32
   * }</pre>
   */
  public Scalar argMin() {
    return reduce(ReductionAggregation.argMin(), DType.INT32);
  }

  /**
   * Find and return a scalar of type INT32 containing the index of the maximum value in the column.
   * If there are multiple maximum values, index of the first encountered maximum value is returned.
   * If the input column is empty or all nulls, an empty scalar is returned.
   *
   * If the input is floating point type and contains NaNs, the result is undefined.
   *
   * Example:
   * <pre>{@code
   * // col    = [5, 2, 8, 1, 9], DType = INT32
   * Scalar result = col.argMax();
   * // result = 4,               DType = INT32
   * }</pre>
   */
  public Scalar argMax() {
    return reduce(ReductionAggregation.argMax(), DType.INT32);
  }

  /**
   * Returns the minimum of all values in the column, returning a scalar
   * of the same type as this column.
   *
   * Example:
   * <pre>{@code
   * // col    = [5, 2, 8, 1, 9], DType = INT32
   * Scalar result = col.min();
   * // result = 1,               DType = INT32
   * }</pre>
   */
  public Scalar min() {
    return reduce(ReductionAggregation.min(), type);
  }

  /**
   * Returns the minimum of all values in the column, returning a scalar
   * of the specified type.
   *
   * Example:
   * <pre>{@code
   * // col    = [5, 2, 8, 1, 9], DType = INT32
   * Scalar result = col.min(DType.INT64);
   * // result = 1,               DType = INT64
   * }</pre>
   *
   * @param outType the output DType
   * @deprecated the min reduction no longer internally allows for setting the output type, as a
   * work around this API will cast the input type to the output type for you, but this may not
   * work in all cases.
   */
  @Deprecated
  public Scalar min(DType outType) {
    if (!outType.equals(type)) {
      try (ColumnVector tmp = this.castTo(outType)) {
        return tmp.min(outType);
      }
    }
    return reduce(ReductionAggregation.min(), outType);
  }

  /**
   * Returns the maximum of all values in the column, returning a scalar
   * of the same type as this column.
   *
   * Example:
   * <pre>{@code
   * // col    = [5, 2, 8, 1, 9], DType = INT32
   * Scalar result = col.max();
   * // result = 9,               DType = INT32
   * }</pre>
   */
  public Scalar max() {
    return reduce(ReductionAggregation.max(), type);
  }

  /**
   * Returns the maximum of all values in the column, returning a scalar
   * of the specified type.
   *
   * Example:
   * <pre>{@code
   * // col    = [5, 2, 8, 1, 9], DType = INT32
   * Scalar result = col.max(DType.INT64);
   * // result = 9,               DType = INT64
   * }</pre>
   *
   * @param outType the output DType
   * @deprecated the max reduction no longer internally allows for setting the output type, as a
   * work around this API will cast the input type to the output type for you, but this may not
   * work in all cases.
   */
  @Deprecated
  public Scalar max(DType outType) {
    if (!outType.equals(type)) {
      try (ColumnVector tmp = this.castTo(outType)) {
        return tmp.max(outType);
      }
    }
    return reduce(ReductionAggregation.max(), outType);
  }

  /**
   * Returns the product of all values in the column, returning a scalar
   * of the same type as this column.
   *
   * Example:
   * <pre>{@code
   * // col    = [2, 3, 4],  DType = INT32
   * Scalar result = col.product();
   * // result = 24,         DType = INT32
   * }</pre>
   */
  public Scalar product() {
    return product(type);
  }

  /**
   * Returns the product of all values in the column, returning a scalar
   * of the specified type.
   *
   * Example:
   * <pre>{@code
   * // col    = [2, 3, 4],  DType = INT32
   * Scalar result = col.product(DType.INT64);
   * // result = 24,         DType = INT64
   * }</pre>
   *
   * @param outType the output DType
   */
  public Scalar product(DType outType) {
    return reduce(ReductionAggregation.product(), outType);
  }

  /**
   * Returns the sum of squares of all values in the column, returning a
   * scalar of the same type as this column.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3],  DType = INT32
   * Scalar result = col.sumOfSquares();
   * // result = 14,         DType = INT32
   * }</pre>
   */
  public Scalar sumOfSquares() {
    return sumOfSquares(type);
  }

  /**
   * Returns the sum of squares of all values in the column, returning a
   * scalar of the specified type.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3],  DType = INT32
   * Scalar result = col.sumOfSquares(DType.INT64);
   * // result = 14,         DType = INT64
   * }</pre>
   *
   * @param outType the output DType
   */
  public Scalar sumOfSquares(DType outType) {
    return reduce(ReductionAggregation.sumOfSquares(), outType);
  }

  /**
   * Returns the arithmetic mean of all values in the column, returning a
   * FLOAT64 scalar unless the column type is FLOAT32 then a FLOAT32 scalar is returned.
   * Null values are skipped.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 4, 5], DType = INT32
   * Scalar result = col.mean();
   * // result = 3.0,             DType = FLOAT64
   * }</pre>
   */
  public Scalar mean() {
    DType outType = DType.FLOAT64;
    if (type.equals(DType.FLOAT32)) {
      outType = type;
    }
    return mean(outType);
  }

  /**
   * Returns the arithmetic mean of all values in the column, returning a
   * scalar of the specified type.
   * Null values are skipped.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 4, 5], DType = INT32
   * Scalar result = col.mean(DType.FLOAT32);
   * // result = 3.0,             DType = FLOAT32
   * }</pre>
   *
   * @param outType the output type to return.  Note that only floating point
   *                types are currently supported.
   */
  public Scalar mean(DType outType) {
    return reduce(ReductionAggregation.mean(), outType);
  }

  /**
   * Returns the sample variance of all values in the column, returning a
   * FLOAT64 scalar unless the column type is FLOAT32 then a FLOAT32 scalar is returned.
   * Null values are skipped.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 4, 5], DType = INT32
   * Scalar result = col.variance();
   * // result = 2.5,             DType = FLOAT64
   * }</pre>
   */
  public Scalar variance() {
    DType outType = DType.FLOAT64;
    if (type.equals(DType.FLOAT32)) {
      outType = type;
    }
    return variance(outType);
  }

  /**
   * Returns the sample variance of all values in the column, returning a
   * scalar of the specified type.
   * Null values are skipped.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 4, 5], DType = INT32
   * Scalar result = col.variance(DType.FLOAT32);
   * // result = 2.5,             DType = FLOAT32
   * }</pre>
   *
   * @param outType the output type to return.  Note that only floating point
   *                types are currently supported.
   */
  public Scalar variance(DType outType) {
    return reduce(ReductionAggregation.variance(), outType);
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a FLOAT64 scalar unless the column type is FLOAT32 then
   * a FLOAT32 scalar is returned. Nulls are not counted as an element
   * of the column when calculating the standard deviation.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 4, 5],    DType = INT32
   * Scalar result = col.standardDeviation();
   * // result = 1.5811388300841898, DType = FLOAT64
   * }</pre>
   */
  public Scalar standardDeviation() {
    DType outType = DType.FLOAT64;
    if (type.equals(DType.FLOAT32)) {
      outType = type;
    }
    return standardDeviation(outType);
  }

  /**
   * Returns the sample standard deviation of all values in the column,
   * returning a scalar of the specified type. Null's are not counted as
   * an element of the column when calculating the standard deviation.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 4, 5],    DType = INT32
   * Scalar result = col.standardDeviation(DType.FLOAT32);
   * // result = 1.5811388,          DType = FLOAT32
   * }</pre>
   *
   * @param outType the output type to return.  Note that only floating point
   *                types are currently supported.
   */
  public Scalar standardDeviation(DType outType) {
    return reduce(ReductionAggregation.standardDeviation(), outType);
  }

  /**
   * Returns a boolean scalar that is true if any of the elements in
   * the column are true or non-zero otherwise false.
   * Null values are skipped.
   *
   * Example:
   * <pre>{@code
   * // col    = [0, 0, 1, 0], DType = INT32
   * Scalar result = col.any();
   * // result = true,         DType = BOOL8
   * }</pre>
   */
  public Scalar any() {
    return any(DType.BOOL8);
  }

  /**
   * Returns a scalar is true or 1, depending on the specified type,
   * if any of the elements in the column are true or non-zero
   * otherwise false or 0.
   * Null values are skipped.
   *
   * Example:
   * <pre>{@code
   * // col    = [0, 0, 1, 0], DType = INT32
   * Scalar result = col.any(DType.INT32);
   * // result = 1,            DType = INT32
   * }</pre>
   *
   * @param outType the output DType
   */
  public Scalar any(DType outType) {
    return reduce(ReductionAggregation.any(), outType);
  }

  /**
   * Returns a boolean scalar that is true if all of the elements in
   * the column are true or non-zero otherwise false.
   * Null values are skipped.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 4], DType = INT32
   * Scalar result = col.all();
   * // result = true,         DType = BOOL8
   * }</pre>
   */
  public Scalar all() {
    return all(DType.BOOL8);
  }

  /**
   * Returns a scalar is true or 1, depending on the specified type,
   * if all of the elements in the column are true or non-zero
   * otherwise false or 0.
   * Null values are skipped.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 4], DType = INT32
   * Scalar result = col.all(DType.BOOL8);
   * // result = true,         DType = BOOL8
   * }</pre>
   *
   * @param outType the output DType
   * @deprecated the only output type supported is BOOL8.
   */
  @Deprecated
  public Scalar all(DType outType) {
    return reduce(ReductionAggregation.all(), outType);
  }

  /**
   * Computes the reduction of the values in all rows of a column.
   * Overflows in reductions are not detected. Specifying a higher precision
   * output type may prevent overflow. Only the MIN and MAX ops are
   * supported for reduction of non-arithmetic types (TIMESTAMP...)
   * The null values are skipped for the operation.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 4, 5], DType = INT32
   * Scalar result = col.reduce(ReductionAggregation.sum());
   * // result = 15,              DType = INT32
   * }</pre>
   *
   * @param aggregation The reduction aggregation to perform
   * @return The scalar result of the reduction operation. If the column is
   * empty or the reduction operation fails then the
   * {@link Scalar#isValid()} method of the result will return false.
   */
  public Scalar reduce(ReductionAggregation aggregation) {
    return reduce(aggregation, type);
  }

  /**
   * Computes the reduction of the values in all rows of a column.
   * Overflows in reductions are not detected. Specifying a higher precision
   * output type may prevent overflow. Only the MIN and MAX ops are
   * supported for reduction of non-arithmetic types (TIMESTAMP...)
   * The null values are skipped for the operation.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 4, 5], DType = INT32
   * Scalar result = col.reduce(ReductionAggregation.sum(), DType.INT64);
   * // result = 15,              DType = INT64
   * }</pre>
   *
   * @param aggregation The reduction aggregation to perform
   * @param outType The type of scalar value to return. Not all output types are supported
   *                by all aggregation operations.
   * @return The scalar result of the reduction operation. If the column is
   * empty or the reduction operation fails then the
   * {@link Scalar#isValid()} method of the result will return false.
   */
  public Scalar reduce(ReductionAggregation aggregation, DType outType) {
    long nativeId = aggregation.createNativeInstance();
    try {
      return new Scalar(outType, reduce(getNativeView(), nativeId, outType.typeId.getNativeId(), outType.getScale()));
    } finally {
      Aggregation.close(nativeId);
    }
  }

  /**
   * Do a segmented reduce where the offsets column indicates which groups in this to combine. The
   * output type is the same as the input type.
   *
   * Example:
   * <pre>{@code
   * // col     = [1, 2, 3, 4, 5], DType = INT32
   * // offsets = [0, 2, 5],       DType = INT32
   * ColumnVector result = col.segmentedReduce(offsets, SegmentedReductionAggregation.sum());
   * // result  = [3, 12],         DType = INT32
   * }</pre>
   *
   * @param offsets an INT32 column with no nulls.
   * @param aggregation the aggregation to do
   * @return the result.
   */
  public ColumnVector segmentedReduce(ColumnView offsets, SegmentedReductionAggregation aggregation) {
    return segmentedReduce(offsets, aggregation, NullPolicy.EXCLUDE, type);
  }

  /**
   * Do a segmented reduce where the offsets column indicates which groups in this to combine.
   *
   * Example:
   * <pre>{@code
   * // col     = [1, 2, 3, 4, 5], DType = INT32
   * // offsets = [0, 2, 5],       DType = INT32
   * ColumnVector result = col.segmentedReduce(offsets, SegmentedReductionAggregation.sum(), DType.INT64);
   * // result  = [3, 12],         DType = INT64
   * }</pre>
   *
   * @param offsets an INT32 column with no nulls.
   * @param aggregation the aggregation to do
   * @param outType the output data type.
   * @return the result.
   */
  public ColumnVector segmentedReduce(ColumnView offsets, SegmentedReductionAggregation aggregation,
      DType outType) {
    return segmentedReduce(offsets, aggregation, NullPolicy.EXCLUDE, outType);
  }

  /**
   * Do a segmented reduce where the offsets column indicates which groups in this to combine.
   *
   * Example:
   * <pre>{@code
   * // col     = [1, null, 3, 4, 5], DType = INT32
   * // offsets = [0, 2, 5],          DType = INT32
   * ColumnVector result = col.segmentedReduce(offsets, SegmentedReductionAggregation.sum(),
   *     NullPolicy.INCLUDE, DType.INT64);
   * // result  = [null, 12],         DType = INT64
   * }</pre>
   *
   * @param offsets an INT32 column with no nulls.
   * @param aggregation the aggregation to do
   * @param nullPolicy the null policy.
   * @param outType the output data type.
   * @return the result.
   */
  public ColumnVector segmentedReduce(ColumnView offsets, SegmentedReductionAggregation aggregation,
      NullPolicy nullPolicy, DType outType) {
    long nativeId = aggregation.createNativeInstance();
    try {
      return new ColumnVector(segmentedReduce(getNativeView(), offsets.getNativeView(), nativeId,
          nullPolicy.includeNulls, outType.typeId.getNativeId(), outType.getScale()));
    } finally {
      Aggregation.close(nativeId);
    }
  }

  /**
   * Segmented gather of the elements within a list element in each row of a list column.
   * For each list, assuming the size is N, valid indices of gather map ranges in [-N, N).
   * Out of bound indices refer to null.
   *
   * Example:
   * <pre>{@code
   * // col       = [[1, 2, 3], [4, 5], [6, 7, 8, 9]], DType = LIST of INT32
   * // gatherMap = [[0, 2],    [1],    [3, 0]],       DType = LIST of INT32
   * ColumnVector result = col.segmentedGather(gatherMap);
   * // result    = [[1, 3],    [5],    [9, 6]],       DType = LIST of INT32
   * }</pre>
   *
   * @param gatherMap ListColumnView carrying lists of integral indices which maps the
   * element in list of each row in the source columns to rows of lists in the result columns.
   * @return the result.
   */
  public ColumnVector segmentedGather(ColumnView gatherMap) {
    return segmentedGather(gatherMap, OutOfBoundsPolicy.NULLIFY);
  }

  /**
   * Segmented gather of the elements within a list element in each row of a list column.
   *
   * Example:
   * <pre>{@code
   * // col       = [[1, 2, 3], [4, 5], [6, 7, 8, 9]], DType = LIST of INT32
   * // gatherMap = [[0, 10],   [1],    [3, 0]],       DType = LIST of INT32
   * ColumnVector result = col.segmentedGather(gatherMap, OutOfBoundsPolicy.NULLIFY);
   * // result    = [[1, null], [5],    [9, 6]],       DType = LIST of INT32
   * }</pre>
   *
   * @param gatherMap ListColumnView carrying lists of integral indices which maps the
   * element in list of each row in the source columns to rows of lists in the result columns.
   * @param policy OutOfBoundsPolicy, `DONT_CHECK` leads to undefined behaviour; `NULLIFY`
   * replaces out of bounds with null.
   * @return the result.
   */
  public ColumnVector segmentedGather(ColumnView gatherMap, OutOfBoundsPolicy policy) {
    return new ColumnVector(segmentedGather(getNativeView(), gatherMap.getNativeView(),
        policy.equals(OutOfBoundsPolicy.NULLIFY)));
  }

  /**
   * Do a reduction on the values in a list. The output type will be the type of the data column
   * of this list.
   *
   * Example:
   * <pre>{@code
   * // col    = [[1, 2, 3], [4, 5], [6]], DType = LIST of INT32
   * ColumnVector result = col.listReduce(SegmentedReductionAggregation.sum());
   * // result = [6, 9, 6],                DType = INT32
   * }</pre>
   *
   * @param aggregation the aggregation to perform
   */
  public ColumnVector listReduce(SegmentedReductionAggregation aggregation) {
    if (!getType().equals(DType.LIST)) {
      throw new IllegalArgumentException("listReduce only works on list types");
    }
    try (ColumnView offsets = getListOffsetsView();
         ColumnView data = getChildColumnView(0)) {
      return data.segmentedReduce(offsets, aggregation);
    }
  }

  /**
   * Do a reduction on the values in a list.
   *
   * Example:
   * <pre>{@code
   * // col    = [[1, 2, 3], [4, 5], [6]], DType = LIST of INT32
   * ColumnVector result = col.listReduce(SegmentedReductionAggregation.sum(), DType.INT64);
   * // result = [6, 9, 6],                DType = INT64
   * }</pre>
   *
   * @param aggregation the aggregation to perform
   * @param outType the type of the output. Typically, this should match with the child type
   *                of the list.
   */
  public ColumnVector listReduce(SegmentedReductionAggregation aggregation, DType outType) {
    return listReduce(aggregation, NullPolicy.EXCLUDE, outType);
  }

  /**
   * Do a reduction on the values in a list.
   *
   * Example:
   * <pre>{@code
   * // col    = [[1, null, 3], [4, 5], [6]], DType = LIST of INT32
   * ColumnVector result = col.listReduce(SegmentedReductionAggregation.sum(),
   *     NullPolicy.INCLUDE, DType.INT64);
   * // result = [null, 9, 6],                DType = INT64
   * }</pre>
   *
   * @param aggregation the aggregation to perform
   * @param nullPolicy should nulls be included or excluded from the aggregation.
   * @param outType the type of the output. Typically, this should match with the child type
   *                of the list.
   */
  public ColumnVector listReduce(SegmentedReductionAggregation aggregation, NullPolicy nullPolicy,
      DType outType) {
    if (!getType().equals(DType.LIST)) {
      throw new IllegalArgumentException("listReduce only works on list types");
    }
    try (ColumnView offsets = getListOffsetsView();
         ColumnView data = getChildColumnView(0)) {
      return data.segmentedReduce(offsets, aggregation, nullPolicy, outType);
    }
  }

  /**
   * Calculate various percentiles of this ColumnVector, which must contain centroids produced by
   * a t-digest aggregation.
   *
   * Example:
   * <pre>{@code
   * // Create t-digest centroids via groupBy aggregation:
   * // Table tdigestTable = inputTable.groupBy(0)
   * //     .aggregate(GroupByAggregation.createTDigest(100).onColumn(1));
   * // ColumnVector tdigestCol = tdigestTable.getColumn(1);
   *
   * // Compute approximate percentiles:
   * ColumnVector result = tdigestCol.approxPercentile(new double[]{0.25, 0.5, 0.75});
   * // result = LIST of FLOAT64, one list of percentile values per element in the `percentiles` array
   * }</pre>
   *
   * @param percentiles Required percentiles [0,1]
   * @return Column containing the approximate percentile values as a list of doubles, in
   *         the same order as the input percentiles
   */
  public final ColumnVector approxPercentile(double[] percentiles) {
    try (ColumnVector cv = ColumnVector.fromDoubles(percentiles)) {
      return approxPercentile(cv);
    }
  }

  /**
   * Calculate various percentiles of this ColumnVector, which must contain centroids produced by
   * a t-digest aggregation.
   *
   * Example:
   * <pre>{@code
   * // Create t-digest centroids via groupBy aggregation:
   * // Table tdigestTable = inputTable.groupBy(0)
   * //     .aggregate(GroupByAggregation.createTDigest(100).onColumn(1));
   * // ColumnVector tdigestCol = tdigestTable.getColumn(1);
   *
   * // percentiles = [0.25, 0.5, 0.75], DType = FLOAT64
   * ColumnVector result = tdigestCol.approxPercentile(percentiles);
   * // result = LIST of FLOAT64, one list of percentile values per element in the `percentiles` array
   * }</pre>
   *
   * @param percentiles Column containing percentiles [0,1]
   * @return Column containing the approximate percentile values as a list of doubles, in
   *         the same order as the input percentiles
   */
  public final ColumnVector approxPercentile(ColumnVector percentiles) {
    return new ColumnVector(approxPercentile(getNativeView(), percentiles.getNativeView()));
  }

  /**
   * Calculate various quantiles of this ColumnVector.  It is assumed that this is already sorted
   * in the desired order.
   *
   * Example:
   * <pre>{@code
   * // col    = [-1, 0, 1, 1, 2, 3, 4, 6, 7, 9], DType = INT32
   * ColumnVector result = col.quantile(QuantileMethod.LINEAR, new double[]{0.0, 0.25, 0.33, 0.5, 1.0});
   * // result = [-1.0, 1.0, 1.0, 2.5, 9.0],      DType = FLOAT64
   * }</pre>
   *
   * @param method   the method used to calculate the quantiles
   * @param quantiles the quantile values [0,1]
   * @return Column containing the approximate percentile values as a list of doubles, in
   *         the same order as the input percentiles
   */
  public final ColumnVector quantile(QuantileMethod method, double[] quantiles) {
    return new ColumnVector(quantile(getNativeView(), method.nativeId, quantiles));
  }

  /**
   * This function aggregates values in a window around each element i of the input
   * column. Please refer to WindowsOptions for various options that can be passed.
   * Note: Only rows-based windows are supported.
   *
   * Example:
   * <pre>{@code
   * // col    = [5, 4, 7, 6, 8],     DType = INT32
   * Scalar one = Scalar.fromInt(1);
   * Scalar two = Scalar.fromInt(2);
   * WindowOptions options = WindowOptions.builder().minPeriods(2).window(two, one).build();
   * ColumnVector result = col.rollingWindow(RollingAggregation.sum(), options);
   * // result = [9, 16, 17, 21, 14], DType = INT64
   * }</pre>
   *
   * @param op the operation to perform.
   * @param options various window function arguments.
   * @return Column containing aggregate function result.
   * @throws IllegalArgumentException if unsupported window specification * (i.e. other than {@link WindowOptions.FrameType#ROWS} is used.
   */
  public final ColumnVector rollingWindow(RollingAggregation op, WindowOptions options) {
    // Check that only row-based windows are used.
    if (!options.getFrameType().equals(WindowOptions.FrameType.ROWS)) {
      throw new IllegalArgumentException("Expected ROWS-based window specification. Unexpected window type: "
          + options.getFrameType());
    }

    long nativePtr = op.createNativeInstance();
    try {
      Scalar p = options.getPrecedingScalar();
      Scalar f = options.getFollowingScalar();
      return new ColumnVector(
          rollingWindow(this.getNativeView(),
              op.getDefaultOutput(),
              options.getMinPeriods(),
              nativePtr,
              p == null || !p.isValid() ? 0 : p.getInt(),
              f == null || !f.isValid() ? 0 : f.getInt(),
              options.getPrecedingCol() == null ? 0 : options.getPrecedingCol().getNativeView(),
              options.getFollowingCol() == null ? 0 : options.getFollowingCol().getNativeView()));
    } finally {
      Aggregation.close(nativePtr);
    }
  }

  /**
   * Compute the prefix sum (aka cumulative sum) of the values in this column.
   * This is just a convenience method for an inclusive scan with a SUM aggregation.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, 3, 5, 8, 10],   DType = INT64
   * ColumnVector result = col.prefixSum();
   * // result = [1, 3, 6, 11, 19, 29], DType = INT64
   * }</pre>
   */
  public final ColumnVector prefixSum() {
    return scan(ScanAggregation.sum());
  }

  /**
   * Computes a scan for a column. This is very similar to a running window on the column.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, null, 3, 5, 8, 10],   DType = INT32
   * ColumnVector result = col.scan(ScanAggregation.sum(), ScanType.INCLUSIVE, NullPolicy.EXCLUDE);
   * // result = [1, 3, null, 6, 11, 19, 29], DType = INT32
   * }</pre>
   *
   * @param aggregation the aggregation to perform
   * @param scanType should the scan be inclusive, include the current row, or exclusive.
   * @param nullPolicy how should nulls be treated. Note that some aggregations also include a
   *                   null policy too. Currently none of those aggregations are supported so
   *                   it is undefined how they would interact with each other.
   */
  public final ColumnVector scan(ScanAggregation aggregation, ScanType scanType, NullPolicy nullPolicy) {
    long nativeId = aggregation.createNativeInstance();
    try {
      return new ColumnVector(scan(getNativeView(), nativeId,
          scanType.isInclusive, nullPolicy.includeNulls));
    } finally {
      Aggregation.close(nativeId);
    }
  }

  /**
   * Computes a scan for a column that excludes nulls.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, null, 3, 5, 8, 10],  DType = INT32
   * ColumnVector result = col.scan(ScanAggregation.sum(), ScanType.EXCLUSIVE);
   * // result = [0, 1, null, 3, 6, 11, 19], DType = INT32
   * }</pre>
   *
   * @param aggregation the aggregation to perform
   * @param scanType should the scan be inclusive, include the current row, or exclusive.
   */
  public final ColumnVector scan(ScanAggregation aggregation, ScanType scanType) {
    return scan(aggregation, scanType, NullPolicy.EXCLUDE);
  }

  /**
   * Computes an inclusive scan for a column that excludes nulls.
   *
   * Example:
   * <pre>{@code
   * // col    = [1, 2, null, 3, 5, 8, 10],   DType = INT32
   * ColumnVector result = col.scan(ScanAggregation.sum());
   * // result = [1, 3, null, 6, 11, 19, 29], DType = INT32
   * }</pre>
   *
   * @param aggregation the aggregation to perform
   */
  public final ColumnVector scan(ScanAggregation aggregation) {
    return scan(aggregation, ScanType.INCLUSIVE, NullPolicy.EXCLUDE);
  }



  /////////////////////////////////////////////////////////////////////////////
  // LOGICAL
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Returns a vector of the logical `not` of each value in the input
   * column (this)
   */
  public final ColumnVector not() {
    return unaryOp(UnaryOp.NOT);
  }

  /////////////////////////////////////////////////////////////////////////////
  // SEARCH
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Find if the `needle` is present in this col
   *
   * example:
   *
   *  Single Column:
   *      idx      0   1   2   3   4
   *      col = { 10, 20, 20, 30, 50 }
   *  Scalar:
   *   value = { 20 }
   *   result = true
   *
   * @param needle
   * @return true if needle is present else false
   */
  public boolean contains(Scalar needle) {
    return containsScalar(getNativeView(), needle.getScalarHandle());
  }

  /**
   * Returns a new column of {@link DType#BOOL8} elements having the same size as this column,
   * each row value is true if the corresponding entry in this column is contained in the
   * given searchSpace column and false if it is not.
   * The caller will be responsible for the lifecycle of the new vector.
   *
   * example:
   *
   *   col         = { 10, 20, 30, 40, 50 }
   *   searchSpace = { 20, 40, 60, 80 }
   *
   *   result = { false, true, false, true, false }
   *
   * @param searchSpace
   * @return A new ColumnVector of type {@link DType#BOOL8}
   */
  public final ColumnVector contains(ColumnView searchSpace) {
    return new ColumnVector(containsVector(getNativeView(), searchSpace.getNativeView()));
  }

  /**
   * Returns a column of strings where, for each string row in the input,
   * the first character after spaces is modified to upper-case,
   * while all the remaining characters in a word are modified to lower-case.
   *
   * Any null string entries return corresponding null output column entries
   */
  public final ColumnVector toTitle() {
    assert type.equals(DType.STRING);
    return new ColumnVector(title(getNativeView()));
  }

  /**
   * Returns a column of capitalized strings.
   *
   * If the `delimiters` is an empty string, then only the first character of each
   * row is capitalized. Otherwise, a non-delimiter character is capitalized after
   * any delimiter character is found.
   *
   * Example:
   *     input = ["tesT1", "a Test", "Another Test", "a\tb"];
   *     delimiters = ""
   *     output is ["Test1", "A test", "Another test", "A\tb"]
   *     delimiters = " "
   *     output is ["Test1", "A Test", "Another Test", "A\tb"]
   *
   * Any null string entries return corresponding null output column entries.
   *
   * @param delimiters Used if identifying words to capitalize. Should not be null.
   * @return a column of capitalized strings. Users should close the returned column.
   */
  public final ColumnVector capitalize(Scalar delimiters) {
    if (DType.STRING.equals(type) && DType.STRING.equals(delimiters.getType())) {
      return new ColumnVector(capitalize(getNativeView(), delimiters.getScalarHandle()));
    }
    throw new IllegalArgumentException("Both input column and delimiters scalar should be" +
        " string type. But got column: " + type + ", scalar: " + delimiters.getType());
  }

  /**
   * Concatenates all strings in the column into one new string delimited
   * by an optional separator string.
   *
   * This returns a column with one string. Any null entries are ignored unless
   * the narep parameter specifies a replacement string (not a null value).
   *
   * @param separator what to insert to separate each row.
   * @param narep what to replace nulls with
   * @return a ColumnVector with a single string in it.
   */
  public final ColumnVector joinStrings(Scalar separator, Scalar narep) {
    if (DType.STRING.equals(type) &&
        DType.STRING.equals(separator.getType()) &&
        DType.STRING.equals(narep.getType())) {
      return new ColumnVector(joinStrings(getNativeView(), separator.getScalarHandle(),
          narep.getScalarHandle()));
    }
    throw new IllegalArgumentException("The column, separator, and narep all need to be STRINGs");
  }
  /////////////////////////////////////////////////////////////////////////////
  // TYPE CAST
  /////////////////////////////////////////////////////////////////////////////

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
  public ColumnVector castTo(DType type) {
    return new ColumnVector(castTo(getNativeView(), type.typeId.getNativeId(), type.getScale()));
  }

  /**
   * This method takes in a nested type and replaces its children with the given views
   * Note: Make sure the numbers of rows in the leaf node are the same as the child replacing it
   * otherwise the list can point to elements outside of the column values.
   *
   * Note: this method returns a ColumnView that won't live past the ColumnVector that it's
   * pointing to.
   *
   * Ex: List<Int> list = col{{1,3}, {9,3,5}}
   *
   * validNewChild = col{8, 3, 9, 2, 0}
   *
   * list.replaceChildrenWithViews(1, validNewChild) => col{{8, 3}, {9, 2, 0}}
   *
   * invalidNewChild = col{3, 2}
   * list.replaceChildrenWithViews(1, invalidNewChild) => col{{3, 2}, {invalid, invalid, invalid}}
   *
   * invalidNewChild = col{8, 3, 9, 2, 0, 0, 7}
   * list.replaceChildrenWithViews(1, invalidNewChild) => col{{8, 3}, {9, 2, 0}} // undefined result
   */
  public ColumnView replaceChildrenWithViews(int[] indices,
                                             ColumnView[] views) {
    assert (type.isNestedType());
    assert (indices.length == views.length);
    if (type == DType.LIST) {
      assert (indices.length == 1);
    }
    if (indices.length != views.length) {
      throw new IllegalArgumentException("The indices size and children size should match");
    }
    Map<Integer, ColumnView> map = new HashMap<>();
    IntStream.range(0, indices.length).forEach(index -> {
      if (map.containsKey(indices[index])) {
        throw new IllegalArgumentException("Duplicate mapping found for replacing child index");
      }
      map.put(indices[index], views[index]);
    });
    List<ColumnView> newChildren = new ArrayList<>(getNumChildren());
    List<ColumnView> toClose = new ArrayList<>(getNumChildren());
    try {
      IntStream.range(0, getNumChildren()).forEach(i -> {
        ColumnView view = map.remove(i);
        ColumnView child = getChildColumnView(i);
        toClose.add(child);
        if (view == null) {
          newChildren.add(child);
        } else {
          if (child.getRowCount() != view.getRowCount()) {
            throw new IllegalArgumentException("Child row count doesn't match the old child");
          }
          newChildren.add(view);
        }
      });
      if (!map.isEmpty()) {
        throw new IllegalArgumentException("One or more invalid child indices passed to be " +
            "replaced");
      }
      return new ColumnView(type, getRowCount(), Optional.of(getNullCount()), getValid(),
          getOffsets(), newChildren.stream().toArray(n -> new ColumnView[n]));
    } finally {
      for (ColumnView columnView: toClose) {
        columnView.close();
      }
    }
  }

  /**
   * This method takes in a list and returns a new list with the leaf node replaced with the given
   * view. Make sure the numbers of rows in the leaf node are the same as the child replacing it
   * otherwise the list can point to elements outside of the column values.
   *
   * Note: this method returns a ColumnView that won't live past the ColumnVector that it's
   * pointing to.
   *
   * Ex: List<Int> list = col{{1,3}, {9,3,5}}
   *
   * validNewChild = col{8, 3, 9, 2, 0}
   *
   * list.replaceChildrenWithViews(1, validNewChild) => col{{8, 3}, {9, 2, 0}}
   *
   * invalidNewChild = col{3, 2}
   * list.replaceChildrenWithViews(1, invalidNewChild) =>
   *        col{{3, 2}, {invalid, invalid, invalid}} throws an exception
   *
   * invalidNewChild = col{8, 3, 9, 2, 0, 0, 7}
   * list.replaceChildrenWithViews(1, invalidNewChild) =>
   *       col{{8, 3}, {9, 2, 0}} throws an exception
   */
  public ColumnView replaceListChild(ColumnView child) {
    assert(type == DType.LIST);
    return replaceChildrenWithViews(new int[]{0}, new ColumnView[]{child});
  }

  /**
   * Zero-copy cast between types with the same underlying representation.
   *
   * Similar to reinterpret_cast or bit_cast in C++. This will essentially take the underlying data
   * and update the metadata to reflect a new type. Not all types are supported the width of the
   * types must match.
   * @param type the type you want to go to.
   * @return a ColumnView that cannot outlive the Column that owns the actual data it points to.
   * @deprecated this has changed to bit_cast in C++ so use that name instead
   */
  @Deprecated
  public ColumnView logicalCastTo(DType type) {
    return bitCastTo(type);
  }

  /**
   * Zero-copy cast between types with the same underlying length.
   *
   * Similar to bit_cast in C++. This will take the underlying data and create new metadata
   * so it is interpreted as a new type. Not all types are supported the width of the
   * types must match.
   * @param type the type you want to go to.
   * @return a ColumnView that cannot outlive the Column that owns the actual data it points to.
   */
  public ColumnView bitCastTo(DType type) {
    return new ColumnView(bitCastTo(getNativeView(),
        type.typeId.getNativeId(), type.getScale()));
  }

  /**
   * Cast to Byte - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to byte
   * When casting from a Date, Timestamp, or Boolean to a byte type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asBytes() {
    return castTo(DType.INT8);
  }

  /**
   * Cast to list of bytes
   * This method converts the rows provided by the ColumnVector and casts each row to a list of
   * bytes with endinanness reversed. Numeric and string types supported, but not timestamps.
   *
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asByteList() {
    return new ColumnVector(byteListCast(getNativeView(), true));
  }

  /**
   * Cast to list of bytes
   * This method converts the rows provided by the ColumnVector and casts each row to a list
   * of bytes. Numeric and string types supported, but not timestamps.
   *
   * @param config Flips the byte order (endianness) if true, retains byte order otherwise
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asByteList(boolean config) {
    return new ColumnVector(byteListCast(getNativeView(), config));
  }

  /**
   * Cast to unsigned Byte - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to byte
   * When casting from a Date, Timestamp, or Boolean to a byte type the underlying numerical
   * representation of the data will be used for the cast.
   * <p>
   * Java does not have an unsigned byte type, so properly decoding these values
   * will require extra steps on the part of the application.  See
   * {@link Byte#toUnsignedInt(byte)}.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asUnsignedBytes() {
    return castTo(DType.UINT8);
  }

  /**
   * Cast to Short - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to short
   * When casting from a Date, Timestamp, or Boolean to a short type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asShorts() {
    return castTo(DType.INT16);
  }

  /**
   * Cast to unsigned Short - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to short
   * When casting from a Date, Timestamp, or Boolean to a short type the underlying numerical
   * representation of the data will be used for the cast.
   * <p>
   * Java does not have an unsigned short type, so properly decoding these values
   * will require extra steps on the part of the application.  See
   * {@link Short#toUnsignedInt(short)}.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asUnsignedShorts() {
    return castTo(DType.UINT16);
  }

  /**
   * Cast to Int - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to int
   * When casting from a Date, Timestamp, or Boolean to a int type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asInts() {
    return castTo(DType.INT32);
  }

  /**
   * Cast to unsigned Int - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to int
   * When casting from a Date, Timestamp, or Boolean to a int type the underlying numerical
   * representation of the data will be used for the cast.
   * <p>
   * Java does not have an unsigned int type, so properly decoding these values
   * will require extra steps on the part of the application.  See
   * {@link Integer#toUnsignedLong(int)}.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asUnsignedInts() {
    return castTo(DType.UINT32);
  }

  /**
   * Cast to Long - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to long
   * When casting from a Date, Timestamp, or Boolean to a long type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asLongs() {
    return castTo(DType.INT64);
  }

  /**
   * Cast to unsigned Long - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to long
   * When casting from a Date, Timestamp, or Boolean to a long type the underlying numerical
   * representation of the data will be used for the cast.
   * <p>
   * Java does not have an unsigned long type, so properly decoding these values
   * will require extra steps on the part of the application.  See
   * {@link Long#toUnsignedString(long)}.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asUnsignedLongs() {
    return castTo(DType.UINT64);
  }

  /**
   * Cast to Float - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to float
   * When casting from a Date, Timestamp, or Boolean to a float type the underlying numerical
   * representatio of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asFloats() {
    return castTo(DType.FLOAT32);
  }

  /**
   * Cast to Double - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to double
   * When casting from a Date, Timestamp, or Boolean to a double type the underlying numerical
   * representation of the data will be used for the cast.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asDoubles() {
    return castTo(DType.FLOAT64);
  }

  /**
   * Cast to TIMESTAMP_DAYS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_DAYS
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampDays() {
    if (type.equals(DType.STRING)) {
      return asTimestamp(DType.TIMESTAMP_DAYS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_DAYS);
  }

  /**
   * Cast to TIMESTAMP_DAYS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_DAYS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampDays(String format) {
    assert type.equals(DType.STRING) : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_DAYS, format);
  }

  /**
   * Cast to TIMESTAMP_SECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_SECONDS
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampSeconds() {
    if (type.equals(DType.STRING)) {
      return asTimestamp(DType.TIMESTAMP_SECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_SECONDS);
  }

  /**
   * Cast to TIMESTAMP_SECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_SECONDS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampSeconds(String format) {
    assert type.equals(DType.STRING) : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_SECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_MICROSECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_MICROSECONDS
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampMicroseconds() {
    if (type.equals(DType.STRING)) {
      return asTimestamp(DType.TIMESTAMP_MICROSECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_MICROSECONDS);
  }

  /**
   * Cast to TIMESTAMP_MICROSECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_MICROSECONDS
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampMicroseconds(String format) {
    assert type.equals(DType.STRING) : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_MICROSECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_MILLISECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_MILLISECONDS.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampMilliseconds() {
    if (type.equals(DType.STRING)) {
      return asTimestamp(DType.TIMESTAMP_MILLISECONDS, "%Y-%m-%dT%H:%M:%SZ%f");
    }
    return castTo(DType.TIMESTAMP_MILLISECONDS);
  }

  /**
   * Cast to TIMESTAMP_MILLISECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_MILLISECONDS.
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampMilliseconds(String format) {
    assert type.equals(DType.STRING) : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_MILLISECONDS, format);
  }

  /**
   * Cast to TIMESTAMP_NANOSECONDS - ColumnVector
   * This method takes the value provided by the ColumnVector and casts to TIMESTAMP_NANOSECONDS.
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampNanoseconds() {
    if (type.equals(DType.STRING)) {
      return asTimestamp(DType.TIMESTAMP_NANOSECONDS, "%Y-%m-%dT%H:%M:%SZ%9f");
    }
    return castTo(DType.TIMESTAMP_NANOSECONDS);
  }

  /**
   * Cast to TIMESTAMP_NANOSECONDS - ColumnVector
   * This method takes the string value provided by the ColumnVector and casts to TIMESTAMP_NANOSECONDS.
   * @param format timestamp string format specifier, ignored if the column type is not string
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asTimestampNanoseconds(String format) {
    assert type.equals(DType.STRING) : "A column of type string is required when using a format string";
    return asTimestamp(DType.TIMESTAMP_NANOSECONDS, format);
  }

  /**
   * Parse a string to a timestamp. Strings that fail to parse will default to 0, corresponding
   * to 1970-01-01 00:00:00.000.
   * @param timestampType timestamp DType that includes the time unit to parse the timestamp into.
   * @param format strptime format specifier string of the timestamp. Used to parse and convert
   *               the timestamp with. Supports %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z format specifiers.
   *               See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
   *               for parsing format specification and documentation.
   * @return A new ColumnVector containing the long representations of the timestamps in the
   *         original column vector.
   */
  public final ColumnVector asTimestamp(DType timestampType, String format) {
    assert type.equals(DType.STRING) : "A column of type string " +
        "is required for .to_timestamp() operation";
    assert format != null : "Format string may not be NULL";
    assert timestampType.isTimestampType() : "unsupported conversion to non-timestamp DType";
    // Only nativeID is passed in the below function as timestamp type does not have `scale`.
    return new ColumnVector(stringTimestampToTimestamp(getNativeView(),
        timestampType.typeId.getNativeId(), format));
  }

  /**
   * Cast to Strings.
   * In case of timestamps it follows the following formats
   *    {@link DType#TIMESTAMP_DAYS} - "%Y-%m-%d"
   *    {@link DType#TIMESTAMP_SECONDS} - "%Y-%m-%d %H:%M:%S"
   *    {@link DType#TIMESTAMP_MICROSECONDS} - "%Y-%m-%d %H:%M:%S.%f"
   *    {@link DType#TIMESTAMP_MILLISECONDS} - "%Y-%m-%d %H:%M:%S.%f"
   *    {@link DType#TIMESTAMP_NANOSECONDS} - "%Y-%m-%d %H:%M:%S.%f"
   *
   * @return A new vector allocated on the GPU.
   */
  public final ColumnVector asStrings() {
    switch(type.typeId) {
      case TIMESTAMP_SECONDS:
        return asStrings("%Y-%m-%d %H:%M:%S");
      case TIMESTAMP_DAYS:
        return asStrings("%Y-%m-%d");
      case TIMESTAMP_MICROSECONDS:
      case TIMESTAMP_MILLISECONDS:
      case TIMESTAMP_NANOSECONDS:
        return asStrings("%Y-%m-%d %H:%M:%S.%f");
      default:
        return castTo(DType.STRING);
    }
  }

  /**
   * Method to parse and convert a timestamp column vector to string column vector. A unix
   * timestamp is a long value representing how many units since 1970-01-01 00:00:00:000 in either
   * positive or negative direction.

   * No checking is done for invalid formats or invalid timestamp units.
   *
   * @param format - strftime format specifier string of the timestamp. Its used to parse and convert
   *               the timestamp with. Supports the following format specifiers:
   *               %d 	Day of the month: 01-31
   *               %m 	Month of the year: 01-12
   *               %y 	Year without century: 00-99
   *               %Y 	Year with century: 0001-9999
   *               %H 	24-hour of the day: 00-23
   *               %I 	12-hour of the day: 01-12
   *               %M 	Minute of the hour: 00-59
   *               %S 	Second of the minute: 00-59
   *               %f 	6-digit microsecond: 000000-999999
   *               %p 	One of: 'AM', 'PM', 'am', 'pm'
   *               %j 	Day of the year: 001-366
   *               See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
   *
   * @return A new vector allocated on the GPU
   */
  public final ColumnVector asStrings(String format) {
    assert type.isTimestampType() : "unsupported conversion from non-timestamp DType";
    assert format != null || format.isEmpty(): "Format string may not be NULL or empty";

    return new ColumnVector(timestampToStringTimestamp(this.getNativeView(), format));
  }

  /**
   * Verifies that a string column can be parsed to timestamps using the provided format
   * pattern.
   *
   * The format pattern can include the following specifiers: "%Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z"
   *
   * | Specifier | Description |
   * | :-------: | ----------- |
   * | \%d | Day of the month: 01-31 |
   * | \%m | Month of the year: 01-12 |
   * | \%y | Year without century: 00-99 |
   * | \%Y | Year with century: 0001-9999 |
   * | \%H | 24-hour of the day: 00-23 |
   * | \%I | 12-hour of the day: 01-12 |
   * | \%M | Minute of the hour: 00-59|
   * | \%S | Second of the minute: 00-59 |
   * | \%f | 6-digit microsecond: 000000-999999 |
   * | \%z | UTC offset with format ±HHMM Example +0500 |
   * | \%j | Day of the year: 001-366 |
   * | \%p | Only 'AM', 'PM' or 'am', 'pm' are recognized |
   *
   * Other specifiers are not currently supported.
   * The "%f" supports a precision value to read the numeric digits. Specify the
   * precision with a single integer value (1-9) as follows:
   * use "%3f" for milliseconds, "%6f" for microseconds and "%9f" for nanoseconds.
   *
   * Any null string entry will result in a corresponding null row in the output column.
   *
   * This will return a column of type boolean where a `true` row indicates the corresponding
   * input string can be parsed correctly with the given format.
   *
   * @param format String specifying the timestamp format in strings.
   * @return New boolean ColumnVector.
   */
  public final ColumnVector isTimestamp(String format) {
    return new ColumnVector(isTimestamp(getNativeView(), format));
  }

  /////////////////////////////////////////////////////////////////////////////
  // LISTS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * For each list in this column pull out the entry at the given index. If the entry would
   * go off the end of the list a NULL is returned instead.
   * @param index 0 based offset into the list. Negative values go backwards from the end of the
   *              list.
   * @return a new column of the values at those indexes.
   */
  public final ColumnVector extractListElement(int index) {
    assert type.equals(DType.LIST) : "A column of type LIST is required for .extractListElement()";
    return new ColumnVector(extractListElement(getNativeView(), index));
  }

  /**
   * For each list in this column pull out the entry at the corresponding index specified in
   * the index column. If the entry goes off the end of the list a NULL is returned instead.
   *
   * The index column should have the same row count with the list column.
   *
   * @param indices a column of 0 based offsets into the list. Negative values go backwards from
   *                the end of the list.
   * @return a new column of the values at those indexes.
   */
  public final ColumnVector extractListElement(ColumnView indices) {
    assert type.equals(DType.LIST) : "A column of type LIST is required for .extractListElement()";
    assert indices != null && DType.INT32.equals(indices.type)
        : "indices should be non-null and integer type";
    assert indices.getRowCount() == rows
        : "indices must have the same row count with list column";
    return new ColumnVector(extractListElementV(getNativeView(), indices.getNativeView()));
  }

  /**
   * Create a new LIST column by copying elements from the current LIST column ignoring duplicate,
   * producing a LIST column in which each list contain only unique elements.
   *
   * Relative ordering elements will be kept the same, by default can keep any of the duplicates
   * Example: [0,3,4,0] may produce either [0,3,4] or [3,4,0], both of which are valid here
   *
   * @return A new LIST column having unique list elements.
   */
  public final ColumnVector dropListDuplicates() {
    return new ColumnVector(dropListDuplicates(getNativeView(), DuplicateKeepOption.KEEP_ANY.nativeId));
  }

  /**
   * Create a new LIST column by copying elements from the current LIST column ignoring duplicate,
   * producing a LIST column in which each list contain only unique elements.
   *
   * Order of the output elements within each list will be preserved as in the input
   *
   * @param keep_option Flag to specify which element to keep (first, last, any)
   * @return A new LIST column having unique list elements.
   */
  public final ColumnVector dropListDuplicates(DuplicateKeepOption keepOption) {
    return new ColumnVector(dropListDuplicates(getNativeView(), keepOption.nativeId));
  }

  /**
   * Given a LIST column in which each element is a struct containing a <key, value> pair. An output
   * LIST column is generated by copying elements of the current column in a way such that if a list
   * contains multiple elements having the same key then only the last element will be copied.
   *
   * @return A new LIST column having list elements with unique keys.
   */
  public final ColumnVector dropListDuplicatesWithKeysValues() {
    return new ColumnVector(dropListDuplicatesWithKeysValues(getNativeView()));
  }

  /**
   * Flatten each list of lists into a single list.
   *
   * The column must have rows that are lists of lists.
   * Any row containing null list elements will result in a null output row.
   *
   * @return A new column vector containing the flattened result
   */
  public ColumnVector flattenLists() {
    return flattenLists(false);
  }

  /**
   * Flatten each list of lists into a single list.
   *
   * The column must have rows that are lists of lists.
   *
   * @param ignoreNull Whether to ignore null list elements in the input column from the operation,
   *                   or any row containing null list elements will result in a null output row
   * @return A new column vector containing the flattened result
   */
  public ColumnVector flattenLists(boolean ignoreNull) {
    return new ColumnVector(flattenLists(getNativeView(), ignoreNull));
  }

  /////////////////////////////////////////////////////////////////////////////
  // STRINGS
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Copy the current column to a new column, each string or list of the output column will have
   * reverse order of characters or elements.
   *
   * @return A new column with lists or strings having reverse order.
   */
  public final ColumnVector reverseStringsOrLists() {
    assert type.equals(DType.STRING) || type.equals(DType.LIST) :
        "A column of type string or list is required, actual: " + type;
    return new ColumnVector(reverseStringsOrLists(getNativeView()));
  }

  /**
   * Convert a string to upper case.
   */
  public final ColumnVector upper() {
    assert type.equals(DType.STRING) : "A column of type string is required for .upper() operation";
    return new ColumnVector(upperStrings(getNativeView()));
  }

  /**
   * Convert a string to lower case.
   */
  public final ColumnVector lower() {
    assert type.equals(DType.STRING) : "A column of type string is required for .lower() operation";
    return new ColumnVector(lowerStrings(getNativeView()));
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Overloading stringLocate to support
   * default values for start (0) and end index.
   * @param substring scalar containing the string to locate within each row.
   */
  public final ColumnVector stringLocate(Scalar substring) {
    return stringLocate(substring, 0);
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Overloading stringLocate to support
   * default value for end index (-1, the end of each string).
   * @param substring scalar containing the string to locate within each row.
   * @param start character index to start the search from (inclusive).
   */
  public final ColumnVector stringLocate(Scalar substring, int start) {
    return stringLocate(substring, start, -1);
  }

  /**
   * Locates the starting index of the first instance of the given string in each row of a column.
   * 0 indexing, returns -1 if the substring is not found. Can be be configured to start or end
   * the search mid string.
   * @param substring scalar containing the string scalar to locate within each row.
   * @param start character index to start the search from (inclusive).
   * @param end character index to end the search on (exclusive).
   */
  public final ColumnVector stringLocate(Scalar substring, int start, int end) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert substring != null : "target string may not be null";
    assert substring.getType().equals(DType.STRING) : "substring scalar must be a string scalar";
    assert start >= 0 : "start index must be a positive value";
    assert end >= start || end == -1 : "end index must be -1 or >= the start index";

    return new ColumnVector(substringLocate(getNativeView(), substring.getScalarHandle(),
        start, end));
  }

  /**
   * Returns a list of columns by splitting each string using the specified pattern. The number of
   * rows in the output columns will be the same as the input column. Null entries are added for a
   * row where split results have been exhausted. Null input entries result in all nulls in the
   * corresponding rows of the output columns.
   *
   * @param pattern UTF-8 encoded string identifying the split pattern for each input string.
   * @param limit the maximum size of the list resulting from splitting each input string,
   *              or -1 for all possible splits. Note that limit = 0 (all possible splits without
   *              trailing empty strings) and limit = 1 (no split at all) are not supported.
   * @param splitByRegex a boolean flag indicating whether the input strings will be split by a
   *                     regular expression pattern or just by a string literal delimiter.
   * @return list of strings columns as a table.
   */
  @Deprecated
  public final Table stringSplit(String pattern, int limit, boolean splitByRegex) {
    if (splitByRegex) {
      return stringSplit(new RegexProgram(pattern, CaptureGroups.NON_CAPTURE), limit);
    } else {
      return stringSplit(pattern, limit);
    }
  }

  /**
   * Returns a list of columns by splitting each string using the specified regex program pattern.
   * The number of rows in the output columns will be the same as the input column. Null entries
   * are added for the rows where split results have been exhausted. Null input entries result in
   * all nulls in the corresponding rows of the output columns.
   *
   * @param regexProg the regex program with UTF-8 encoded string identifying the split pattern
   *                  for each input string.
   * @param limit the maximum size of the list resulting from splitting each input string,
   *              or -1 for all possible splits. Note that limit = 0 (all possible splits without
   *              trailing empty strings) and limit = 1 (no split at all) are not supported.
   * @return list of strings columns as a table.
   */
  public final Table stringSplit(RegexProgram regexProg, int limit) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert regexProg != null : "regex program is null";
    assert limit != 0 && limit != 1 : "split limit == 0 and limit == 1 are not supported";
    return new Table(stringSplitRe(this.getNativeView(), regexProg.pattern(), regexProg.combinedFlags(),
                                   regexProg.capture().nativeId, limit));
  }

  /**
   * Returns a list of columns by splitting each string using the specified pattern. The number of
   * rows in the output columns will be the same as the input column. Null entries are added for a
   * row where split results have been exhausted. Null input entries result in all nulls in the
   * corresponding rows of the output columns.
   *
   * @param pattern UTF-8 encoded string identifying the split pattern for each input string.
   * @param splitByRegex a boolean flag indicating whether the input strings will be split by a
   *                     regular expression pattern or just by a string literal delimiter.
   * @return list of strings columns as a table.
   */
  @Deprecated
  public final Table stringSplit(String pattern, boolean splitByRegex) {
    return stringSplit(pattern, -1, splitByRegex);
  }

  /**
   * Returns a list of columns by splitting each string using the specified string literal
   * delimiter. The number of rows in the output columns will be the same as the input column.
   * Null entries are added for a row where split results have been exhausted. Null input entries
   * result in all nulls in the corresponding rows of the output columns.
   *
   * @param delimiter UTF-8 encoded string identifying the split delimiter for each input string.
   * @param limit the maximum size of the list resulting from splitting each input string,
   *              or -1 for all possible splits. Note that limit = 0 (all possible splits without
   *              trailing empty strings) and limit = 1 (no split at all) are not supported.
   * @return list of strings columns as a table.
   */
  public final Table stringSplit(String delimiter, int limit) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert delimiter != null : "delimiter is null";
    assert limit != 0 && limit != 1 : "split limit == 0 and limit == 1 are not supported";
    return new Table(stringSplit(this.getNativeView(), delimiter, limit));
  }

  /**
   * Returns a list of columns by splitting each string using the specified string literal
   * delimiter. The number of rows in the output columns will be the same as the input column.
   * Null entries are added for a row where split results have been exhausted. Null input entries
   * result in all nulls in the corresponding rows of the output columns.
   *
   * @param delimiter UTF-8 encoded string identifying the split delimiter for each input string.
   * @return list of strings columns as a table.
   */
  public final Table stringSplit(String delimiter) {
    return stringSplit(delimiter, -1);
  }

  /**
   * Returns a list of columns by splitting each string using the specified regex program pattern.
   * The number of rows in the output columns will be the same as the input column. Null entries
   * are added for the rows where split results have been exhausted. Null input entries result in
   * all nulls in the corresponding rows of the output columns.
   *
   * @param regexProg the regex program with UTF-8 encoded string identifying the split pattern
   *                  for each input string.
   * @return list of strings columns as a table.
   */
  public final Table stringSplit(RegexProgram regexProg) {
    return stringSplit(regexProg, -1);
  }

  /**
   * Returns a column that are lists of strings in which each list is made by splitting the
   * corresponding input string using the specified pattern.
   *
   * @param pattern UTF-8 encoded string identifying the split pattern for each input string.
   * @param limit the maximum size of the list resulting from splitting each input string,
   *              or -1 for all possible splits. Note that limit = 0 (all possible splits without
   *              trailing empty strings) and limit = 1 (no split at all) are not supported.
   * @param splitByRegex a boolean flag indicating whether the input strings will be split by a
   *                     regular expression pattern or just by a string literal delimiter.
   * @return a LIST column of string elements.
   */
  @Deprecated
  public final ColumnVector stringSplitRecord(String pattern, int limit, boolean splitByRegex) {
    if (splitByRegex) {
      return stringSplitRecord(new RegexProgram(pattern, CaptureGroups.NON_CAPTURE), limit);
    } else {
      return stringSplitRecord(pattern, limit);
    }
  }

  /**
   * Returns a column that are lists of strings in which each list is made by splitting the
   * corresponding input string using the specified regex program pattern.
   *
   * @param regexProg the regex program with UTF-8 encoded string identifying the split pattern
   *                  for each input string.
   * @param limit the maximum size of the list resulting from splitting each input string,
   *              or -1 for all possible splits. Note that limit = 0 (all possible splits without
   *              trailing empty strings) and limit = 1 (no split at all) are not supported.
   * @return a LIST column of string elements.
   */
  public final ColumnVector stringSplitRecord(RegexProgram regexProg, int limit) {
    assert type.equals(DType.STRING) : "column type must be String";
    assert regexProg != null : "regex program is null";
    assert limit != 0 && limit != 1 : "split limit == 0 and limit == 1 are not supported";
    return new ColumnVector(
        stringSplitRecordRe(this.getNativeView(), regexProg.pattern(), regexProg.combinedFlags(),
                            regexProg.capture().nativeId, limit));
  }

  /**
   * Returns a column that are lists of strings in which each list is made by splitting the
   * corresponding input string using the specified pattern.
   *
   * @param pattern UTF-8 encoded string identifying the split pattern for each input string.
   * @param splitByRegex a boolean flag indicating whether the input strings will be split by a
   *                     regular expression pattern or just by a string literal delimiter.
   * @return a LIST column of string elements.
   */
  @Deprecated
  public final ColumnVector stringSplitRecord(String pattern, boolean splitByRegex) {
    return stringSplitRecord(pattern, -1, splitByRegex);
  }

  /**
   * Returns a column that are lists of strings in which each list is made by splitting the
   * corresponding input string using the specified string literal delimiter.
   *
   * @param delimiter UTF-8 encoded string identifying the split delimiter for each input string.
   * @param limit the maximum size of the list resulting from splitting each input string,
   *              or -1 for all possible splits. Note that limit = 0 (all possible splits without
   *              trailing empty strings) and limit = 1 (no split at all) are not supported.
   * @return a LIST column of string elements.
   */
  public final ColumnVector stringSplitRecord(String delimiter, int limit) {
    assert type.equals(DType.STRING) : "column type must be String";
    assert delimiter != null : "delimiter is null";
    assert limit != 0 && limit != 1 : "split limit == 0 and limit == 1 are not supported";
    return new ColumnVector(stringSplitRecord(this.getNativeView(), delimiter, limit));
  }

  /**
   * Returns a column that are lists of strings in which each list is made by splitting the
   * corresponding input string using the specified string literal delimiter.
   *
   * @param delimiter UTF-8 encoded string identifying the split delimiter for each input string.
   * @return a LIST column of string elements.
   */
  public final ColumnVector stringSplitRecord(String delimiter) {
    return stringSplitRecord(delimiter, -1);
  }

  /**
   * Returns a column that are lists of strings in which each list is made by splitting the
   * corresponding input string using the specified regex program pattern.
   *
   * @param regexProg the regex program with UTF-8 encoded string identifying the split pattern
   *                  for each input string.
   * @return a LIST column of string elements.
   */
  public final ColumnVector stringSplitRecord(RegexProgram regexProg) {
    return stringSplitRecord(regexProg, -1);
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column.
   * The character positions to retrieve in each string are `[start, <the string end>)`..
   *
   * @param start first character index to begin the substring(inclusive).
   */
  public final ColumnVector substring(int start) {
    assert type.equals(DType.STRING) : "column type must be a String";
    return new ColumnVector(substringS(getNativeView(), start));
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column.
   * 0-based indexing, If the stop position is past end of a string's length, then end of string is
   * used as stop position for that string.
   * @param start first character index to begin the substring(inclusive).
   * @param end   last character index to stop the substring(exclusive)
   * @return A new java column vector containing the substrings.
   */
  public final ColumnVector substring(int start, int end) {
    assert type.equals(DType.STRING) : "column type must be a String";
    return new ColumnVector(substring(getNativeView(), start, end));
  }

  /**
   * Returns a new strings column that contains substrings of the strings in the provided column
   * which uses unique ranges for each string
   * @param start Vector containing start indices of each string
   * @param end   Vector containing end indices of each string. -1 indicated to read until end of string.
   * @return A new java column vector containing the substrings/
   */
  public final ColumnVector substring(ColumnView start, ColumnView end) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert (rows == start.getRowCount() && rows == end.getRowCount()) : "Number of rows must be equal";
    assert (start.getType().equals(DType.INT32) && end.getType().equals(DType.INT32)) : "start and end " +
        "vectors must be of integer type";
    return new ColumnVector(substringColumn(getNativeView(), start.getNativeView(), end.getNativeView()));
  }

  /**
   * Given a lists column of strings (each row is a list of strings), concatenates the strings
   * within each row and returns a single strings column result. Each new string is created by
   * concatenating the strings from the same row (same list element) delimited by the separator
   * provided. This version of the function relaces nulls with empty string and returns null
   * for empty list.
   * @param sepCol strings column that provides separators for concatenation.
   * @return A new java column vector containing the concatenated strings with separator between.
   */
  public final ColumnVector stringConcatenateListElements(ColumnView sepCol) {
    try (Scalar nullString = Scalar.fromString(null);
         Scalar emptyString = Scalar.fromString("")) {
      return stringConcatenateListElements(sepCol, nullString, emptyString,
          false, false);
    }
  }

  /**
   * Given a lists column of strings (each row is a list of strings), concatenates the strings
   * within each row and returns a single strings column result.
   * Each new string is created by concatenating the strings from the same row (same list element)
   * delimited by the row separator provided in the sepCol strings column.
   * @param sepCol strings column that provides separators for concatenation.
   * @param separatorNarep string scalar indicating null behavior when a separator is null.
   *                        If set to null and the separator is null the resulting string will
   *                        be null. If not null, this string will be used in place of a null
   *                        separator.
   * @param stringNarep string that should be used to replace null strings in any non-null list
   *                     row. If set to null and the string is null the resulting string will
   *                     be null. If not null, this string will be used in place of a null value.
   * @param separateNulls if true, then the separator is included for null rows if
   *                       `stringNarep` is valid.
   * @param emptyStringOutputIfEmptyList if set to true, any input row that is an empty list
   *                          will result in an empty string. Otherwise, it will result in a null.
   * @return A new java column vector containing the concatenated strings with separator between.
   */
  public final ColumnVector stringConcatenateListElements(ColumnView sepCol,
      Scalar separatorNarep, Scalar stringNarep, boolean separateNulls,
      boolean emptyStringOutputIfEmptyList) {
    assert type.equals(DType.LIST) : "column type must be a list";
    assert separatorNarep != null : "separator narep scalar provided may not be null";
    assert stringNarep != null : "string narep scalar provided may not be null";
    assert separatorNarep.getType().equals(DType.STRING) : "separator naprep scalar must be a string scalar";
    assert stringNarep.getType().equals(DType.STRING) : "string narep scalar must be a string scalar";

    return new ColumnVector(stringConcatenationListElementsSepCol(getNativeView(),
      sepCol.getNativeView(), separatorNarep.getScalarHandle(), stringNarep.getScalarHandle(),
      separateNulls, emptyStringOutputIfEmptyList));
  }

  /**
   * Given a lists column of strings (each row is a list of strings), concatenates the strings
   * within each row and returns a single strings column result. Each new string is created by
   * concatenating the strings from the same row (same list element) delimited by the
   * separator provided.
   * @param separator string scalar inserted between each string being merged.
   * @param narep string scalar indicating null behavior. If set to null and any string in the row
   *              is null the resulting string will be null. If not null, null values in any
   *              column will be replaced by the specified string. The underlying value in the
   *              string scalar may be null, but the object passed in may not.
   * @param separateNulls if true, then the separator is included for null rows if
   *                       `narep` is valid.
   * @param emptyStringOutputIfEmptyList if set to true, any input row that is an empty list
   *                          will result in an empty string. Otherwise, it will result in a null.
   * @return A new java column vector containing the concatenated strings with separator between.
   */
  public final ColumnVector stringConcatenateListElements(Scalar separator,
      Scalar narep, boolean separateNulls, boolean emptyStringOutputIfEmptyList) {
    assert type.equals(DType.LIST) : "column type must be a list";
    assert separator != null : "separator scalar provided may not be null";
    assert narep != null : "column narep scalar provided may not be null";
    assert narep.getType().equals(DType.STRING) : "narep scalar must be a string scalar";

    return new ColumnVector(stringConcatenationListElements(getNativeView(),
        separator.getScalarHandle(), narep.getScalarHandle(), separateNulls,
        emptyStringOutputIfEmptyList));
  }

  /**
   * Given a strings column, each string in it is repeated a number of times specified by the
   * <code>repeatTimes</code> parameter.
   *
   * In special cases:
   *  - If <code>repeatTimes</code> is not a positive number, a non-null input string will always
   *    result in an empty output string.
   *  - A null input string will always result in a null output string regardless of the value of
   *    the <code>repeatTimes</code> parameter.
   *
   * @param repeatTimes The number of times each input string is repeated.
   * @return A new java column vector containing repeated strings.
   */
  public final ColumnVector repeatStrings(int repeatTimes) {
    assert type.equals(DType.STRING) : "column type must be String";
    return new ColumnVector(repeatStrings(getNativeView(), repeatTimes));
  }

  /**
   * Given a strings column, an output strings column is generated by repeating each of the input
   * string by a number of times given by the corresponding row in a <code>repeatTimes</code>
   * numeric column.
   *
   * In special cases:
   *  - Any null row (from either the input strings column or the <code>repeatTimes</code> column)
   *    will always result in a null output string.
   *  - If any value in the <code>repeatTimes</code> column is not a positive number and its
   *    corresponding input string is not null, the output string will be an empty string.
   *
   * @param repeatTimes The column containing numbers of times each input string is repeated.
   * @return A new java column vector containing repeated strings.
   */
  public final ColumnVector repeatStrings(ColumnView repeatTimes) {
    assert type.equals(DType.STRING) : "column type must be String";
    return new ColumnVector(repeatStringsWithColumnRepeatTimes(getNativeView(),
            repeatTimes.getNativeView()));
  }

   /**
   * Apply a JSONPath string to all rows in an input strings column.
   *
   * Applies a JSONPath string to an incoming strings column where each row in the column
   * is a valid json string.  The output is returned by row as a strings column.
   *
   * For reference, https://datatracker.ietf.org/doc/id/draft-goessner-dispatch-jsonpath-00.html
   * Note: Only implements the operators: $ . [] *
   *
   * @param path The JSONPath string to be applied to each row
   * @param path The GetJsonObjectOptions to control get_json_object behaviour
   * @return new strings ColumnVector containing the retrieved json object strings
   */
  public final ColumnVector getJSONObject(Scalar path, GetJsonObjectOptions options) {
    assert(type.equals(DType.STRING)) : "column type must be a String";
    return new ColumnVector(getJSONObject(getNativeView(), path.getScalarHandle(), options.isAllowSingleQuotes(), options.isStripQuotesFromSingleStrings(), options.isMissingFieldsAsNulls()));
  }

   /**
   * Apply a JSONPath string to all rows in an input strings column.
   *
   * Applies a JSONPath string to an incoming strings column where each row in the column
   * is a valid json string.  The output is returned by row as a strings column.
   *
   * For reference, https://datatracker.ietf.org/doc/id/draft-goessner-dispatch-jsonpath-00.html
   * Note: Only implements the operators: $ . [] *
   *
   * @param path The JSONPath string to be applied to each row
   * @return new strings ColumnVector containing the retrieved json object strings
   */
  public final ColumnVector getJSONObject(Scalar path) {
    assert(type.equals(DType.STRING)) : "column type must be a String";
    return getJSONObject(path, GetJsonObjectOptions.DEFAULT);
  }

  /**
   * Returns a new strings column where target string within each string is replaced with the specified
   * replacement string.
   * The replacement proceeds from the beginning of the string to the end, for example,
   * replacing "aa" with "b" in the string "aaa" will result in "ba" rather than "ab".
   * Specifying an empty string for replace will essentially remove the target string if found in each string.
   * Null string entries will return null output string entries.
   * target Scalar should be string and should not be empty or null.
   *
   * @param target String to search for within each string.
   * @param replace Replacement string if target is found.
   * @return A new java column vector containing replaced strings
   */
  public final ColumnVector stringReplace(Scalar target, Scalar replace) {

    assert type.equals(DType.STRING) : "column type must be a String";
    assert target != null : "target string may not be null";
    assert target.getType().equals(DType.STRING) : "target string must be a string scalar";
    assert target.getJavaString().isEmpty() == false : "target scalar may not be empty";

    return new ColumnVector(stringReplace(getNativeView(), target.getScalarHandle(),
        replace.getScalarHandle()));
  }

  /**
   * Returns a new strings column where target strings with each string are replaced with
   * corresponding replacement strings. For each string in the column, the list of targets
   * is searched within that string. If a target string is found, it is replaced by the
   * corresponding entry in the repls column. All occurrences found in each string are replaced.
   * The repls argument can optionally contain a single string. In this case, all matching
   * target substrings will be replaced by that single string.
   *
   * Example:
   * cv = ["hello", "goodbye"]
   * targets = ["e","o"]
   * repls = ["EE","OO"]
   * r1 = cv.stringReplace(targets, repls)
   * r1 is now ["hEEllO", "gOOOOdbyEE"]
   *
   * targets = ["e", "o"]
   * repls = ["_"]
   * r2 = cv.stringReplace(targets, repls)
   * r2 is now ["h_ll_", "g__dby_"]
   *
   * @param targets Strings to search for in each string.
   * @param repls Corresponding replacement strings for target strings.
   * @return A new java column vector containing the replaced strings.
   */
  public final ColumnVector stringReplace(ColumnView targets, ColumnView repls) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert targets != null : "target list may not be null";
    assert targets.getType().equals(DType.STRING) : "target list must be a string column";
    assert repls != null : "replacement list may not be null";
    assert repls.getType().equals(DType.STRING) : "replacement list must be a string column";

    return new ColumnVector(stringReplaceMulti(getNativeView(), targets.getNativeView(),
        repls.getNativeView()));
  }

  /**
   * Returns a new strings column where, for each row {@code i}, all occurrences of
   * {@code targets[i]} within {@code input[i]} are replaced with {@code repls[i]}.
   *
   * Unlike {@link #stringReplace(ColumnView, ColumnView)}, which searches every row for a
   * shared list of target strings, this method pairs each input row with its own
   * target/replacement pair.
   *
   * The {@code targets} and {@code repls} columns must have the same number of rows as this
   * column. Output row {@code i} is null if any of {@code input[i]}, {@code targets[i]},
   * or {@code repls[i]} is null. If {@code targets[i]} is an empty string, {@code input[i]}
   * is copied unchanged.
   *
   * @param targets Per-row strings to search for within each input string.
   * @param repls Per-row replacement strings used when the corresponding target is found.
   * @return A new column vector containing the replaced strings.
   */
  public final ColumnVector stringReplacePerRow(ColumnView targets, ColumnView repls) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert targets != null : "targets column may not be null";
    assert targets.getType().equals(DType.STRING) : "targets column must be a string column";
    assert repls != null : "repls column may not be null";
    assert repls.getType().equals(DType.STRING) : "repls column must be a string column";
    assert targets.getRowCount() == getRowCount() : "targets must have the same number of rows as this column";
    assert repls.getRowCount() == getRowCount() : "repls must have the same number of rows as this column";

    return new ColumnVector(stringReplacePerRow(getNativeView(), targets.getNativeView(),
        repls.getNativeView()));
  }

  /**
   * For each string, replaces any character sequence matching the given pattern using the
   * replacement string scalar.
   *
   * @param pattern The regular expression pattern to search within each string.
   * @param repl The string scalar to replace for each pattern match.
   * @return A new column vector containing the string results.
   */
  @Deprecated
  public final ColumnVector replaceRegex(String pattern, Scalar repl) {
    return replaceRegex(new RegexProgram(pattern, CaptureGroups.NON_CAPTURE), repl);
  }

  /**
   * For each string, replaces any character sequence matching the given regex program pattern
   * using the replacement string scalar.
   *
   * @param regexProg The regex program with pattern to search within each string.
   * @param repl The string scalar to replace for each pattern match.
   * @return A new column vector containing the string results.
   */
  public final ColumnVector replaceRegex(RegexProgram regexProg, Scalar repl) {
    return replaceRegex(regexProg, repl, -1);
  }

  /**
   * For each string, replaces any character sequence matching the given pattern using the
   * replacement string scalar.
   *
   * @param pattern The regular expression pattern to search within each string.
   * @param repl The string scalar to replace for each pattern match.
   * @param maxRepl The maximum number of times a replacement should occur within each string.
   * @return A new column vector containing the string results.
   */
  @Deprecated
  public final ColumnVector replaceRegex(String pattern, Scalar repl, int maxRepl) {
    return replaceRegex(new RegexProgram(pattern, CaptureGroups.NON_CAPTURE), repl, maxRepl);
  }

  /**
   * For each string, replaces any character sequence matching the given regex program pattern
   * using the replacement string scalar.
   *
   * @param regexProg The regex program with pattern to search within each string.
   * @param repl The string scalar to replace for each pattern match.
   * @param maxRepl The maximum number of times a replacement should occur within each string.
   * @return A new column vector containing the string results.
   */
  public final ColumnVector replaceRegex(RegexProgram regexProg, Scalar repl, int maxRepl) {
    if (!repl.getType().equals(DType.STRING)) {
      throw new IllegalArgumentException("Replacement must be a string scalar");
    }
    assert regexProg != null : "regex program may not be null";
    return new ColumnVector(replaceRegex(getNativeView(), regexProg.pattern(), regexProg.combinedFlags(),
                                         regexProg.capture().nativeId, repl.getScalarHandle(), maxRepl));
  }

  /**
   * For each string, replaces any character sequence matching the given pattern
   * using the replace template for back-references.
   *
   * Any null string entries return corresponding null output column entries.
   *
   * @param pattern The regular expression patterns to search within each string.
   * @param replace The replacement template for creating the output string.
   * @return A new java column vector containing the string results.
   */
  @Deprecated
  public final ColumnVector stringReplaceWithBackrefs(String pattern, String replace) {
    return stringReplaceWithBackrefs(new RegexProgram(pattern), replace);
  }

  /**
   * For each string, replaces any character sequence matching the given regex program
   * pattern using the replace template for back-references.
   *
   * Any null string entries return corresponding null output column entries.
   *
   * @param regexProg The regex program with pattern to search within each string.
   * @param replace The replacement template for creating the output string.
   * @return A new java column vector containing the string results.
   */
  public final ColumnVector stringReplaceWithBackrefs(RegexProgram regexProg, String replace) {
    assert regexProg != null : "regex program may not be null";
    return new ColumnVector(
        stringReplaceWithBackrefs(getNativeView(), regexProg.pattern(), regexProg.combinedFlags(),
                                  regexProg.capture().nativeId, replace));
  }

  /**
   * Add '0' as padding to the left of each string.
   *
   * If the string is already width or more characters, no padding is performed.
   * No strings are truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width The minimum number of characters for each string.
   * @return New column of strings.
   */
  public final ColumnVector zfill(int width) {
    return new ColumnVector(zfill(getNativeView(), width));
  }

  /**
   * Pad the Strings column until it reaches the desired length with spaces " " on the right.
   *
   * If the string is already width or more characters, no padding is performed.
   * No strings are truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width the minimum number of characters for each string.
   * @return the new strings column.
   */
  public final ColumnVector pad(int width) {
    return pad(width, PadSide.RIGHT, " ");
  }

  /**
   * Pad the Strings column until it reaches the desired length with spaces " ".
   *
   * If the string is already width or more characters, no padding is performed.
   * No strings are truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width the minimum number of characters for each string.
   * @param side where to add new characters.
   * @return the new strings column.
   */
  public final ColumnVector pad(int width, PadSide side) {
    return pad(width, side, " ");
  }

  /**
   * Pad the Strings column until it reaches the desired length.
   *
   * If the string is already width or more characters, no padding is performed.
   * No strings are truncated.
   *
   * Null string entries result in null entries in the output column.
   *
   * @param width the minimum number of characters for each string.
   * @param side where to add new characters.
   * @param fillChar a single character string that holds what should be added.
   * @return the new strings column.
   */
  public final ColumnVector pad(int width, PadSide side, String fillChar) {
    assert fillChar != null;
    assert fillChar.length() == 1;
    return new ColumnVector(pad(getNativeView(), width, side.getNativeId(), fillChar));
  }

  /**
   * Checks if each string in a column starts with a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param pattern scalar containing the string being searched for at the beginning of the column's strings.
   * @return A new java column vector containing the boolean results.
   */
  public final ColumnVector startsWith(Scalar pattern) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert pattern != null : "pattern scalar may not be null";
    assert pattern.getType().equals(DType.STRING) : "pattern scalar must be a string scalar";
    return new ColumnVector(stringStartWith(getNativeView(), pattern.getScalarHandle()));
  }

  /**
   * Checks if each string in a column ends with a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param pattern scalar containing the string being searched for at the end of the column's strings.
   * @return A new java column vector containing the boolean results.
   */
  public final ColumnVector endsWith(Scalar pattern) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert pattern != null : "pattern scalar may not be null";
    assert pattern.getType().equals(DType.STRING) : "pattern scalar must be a string scalar";
    return new ColumnVector(stringEndWith(getNativeView(), pattern.getScalarHandle()));
  }

  /**
   * Removes whitespace from the beginning and end of a string.
   * @return A new java column vector containing the stripped strings.
   */
  public final ColumnVector strip() {
    assert type.equals(DType.STRING) : "column type must be a String";
    try (Scalar emptyString = Scalar.fromString("")) {
      return new ColumnVector(stringStrip(getNativeView(), StripType.BOTH.nativeId,
          emptyString.getScalarHandle()));
    }
  }

  /**
   * Removes the specified characters from the beginning and end of each string.
   * @param toStrip UTF-8 encoded characters to strip from each string.
   * @return A new java column vector containing the stripped strings.
   */
  public final ColumnVector strip(Scalar toStrip) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert toStrip != null : "toStrip scalar may not be null";
    assert toStrip.getType().equals(DType.STRING) : "toStrip must be a string scalar";
    return new ColumnVector(stringStrip(getNativeView(), StripType.BOTH.nativeId, toStrip.getScalarHandle()));
  }

  /**
   * Removes whitespace from the beginning of a string.
   * @return A new java column vector containing the stripped strings.
   */
  public final ColumnVector lstrip() {
    assert type.equals(DType.STRING) : "column type must be a String";
    try (Scalar emptyString = Scalar.fromString("")) {
      return new ColumnVector(stringStrip(getNativeView(), StripType.LEFT.nativeId,
          emptyString.getScalarHandle()));
    }
  }

  /**
   * Removes the specified characters from the beginning of each string.
   * @param toStrip UTF-8 encoded characters to strip from each string.
   * @return A new java column vector containing the stripped strings.
   */
  public final ColumnVector lstrip(Scalar toStrip) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert toStrip != null : "toStrip  Scalar may not be null";
    assert toStrip.getType().equals(DType.STRING) : "toStrip must be a string scalar";
    return new ColumnVector(stringStrip(getNativeView(), StripType.LEFT.nativeId, toStrip.getScalarHandle()));
  }

  /**
   * Removes whitespace from the end of a string.
   * @return A new java column vector containing the stripped strings.
   */
  public final ColumnVector rstrip() {
    assert type.equals(DType.STRING) : "column type must be a String";
    try (Scalar emptyString = Scalar.fromString("")) {
      return new ColumnVector(stringStrip(getNativeView(), StripType.RIGHT.nativeId,
          emptyString.getScalarHandle()));
    }
  }

  /**
   * Removes the specified characters from the end of each string.
   * @param toStrip UTF-8 encoded characters to strip from each string.
   * @return A new java column vector containing the stripped strings.
   */
  public final ColumnVector rstrip(Scalar toStrip) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert toStrip != null : "toStrip  Scalar may not be null";
    assert toStrip.getType().equals(DType.STRING) : "toStrip must be a string scalar";
    return new ColumnVector(stringStrip(getNativeView(), StripType.RIGHT.nativeId, toStrip.getScalarHandle()));
  }

  /**
   * Checks if each string in a column contains a specified comparison string, resulting in a
   * parallel column of the boolean results.
   * @param compString scalar containing the string being searched for.
   * @return A new java column vector containing the boolean results.
   */

  public final ColumnVector stringContains(Scalar compString) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert compString != null : "compString scalar may not be null";
    assert compString.getType().equals(DType.STRING) : "compString scalar must be a string scalar";
    return new ColumnVector(stringContains(getNativeView(), compString.getScalarHandle()));
  }

  /**
   * @brief Searches for the given target strings within each string in the provided column
   *
   * Each column in the result table corresponds to the result for the target string at the same
   * ordinal. i.e. 0th column is the BOOL8 column result for the 0th target string, 1th for 1th,
   * etc.
   *
   * If the target is not found for a string, false is returned for that entry in the output column.
   * If the target is an empty string, true is returned for all non-null entries in the output column.
   *
   * Any null input strings return corresponding null entries in the output columns.
   *
   * input = ["a", "b", "c"]
   * targets = ["a", "c"]
   * output is a table with two boolean columns:
   *   column 0: [true, false, false]
   *   column 1: [false, false, true]
   *
   * @param targets UTF-8 encoded strings to search for in each string in `input`
   * @return BOOL8 columns
   */
  public final ColumnVector[] stringContains(ColumnView targets) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert targets.getType().equals(DType.STRING) : "targets type must be a string";
    assert targets.getNullCount() == 0 : "targets must not contain nulls";
    assert targets.getRowCount() > 0 : "targets must not be empty";
    long[] resultPointers = stringContainsMulti(getNativeView(), targets.getNativeView());
    return Arrays.stream(resultPointers).mapToObj(ColumnVector::new).toArray(ColumnVector[]::new);
  }

  /**
   * For each row {@code i}, returns whether {@code input[i]} contains the literal substring
   * {@code targets[i]} (UTF-8 byte sequence, not regex).
   *
   * <p>This column and {@code targets} must have the same row count. Null {@code targets[i]} yields
   * {@code false} for that row; null {@code input[i]} yields null in the output.
   *
   * <p>Contrast with {@link #stringContains(Scalar)} (one needle for all rows) and
   * {@link #stringContains(ColumnView)} ({@code contains_multiple}: each needle checked against
   * every row, returning multiple boolean columns).
   *
   * <p>Example:
   * <pre>{@code
   * // input   = ["apple", "banana", null, "date"], DType = STRING
   * // targets = ["pl",    "xyz",    "a",  null],   DType = STRING
   * ColumnVector result = input.stringContainsPerRow(targets);
   * // result  = [true,    false,    null, false],   DType = BOOL8
   * }</pre>
   *
   * @param targets string column aligned row-for-row with this column
   * @return a new BOOL8 column
   */
  public final ColumnVector stringContainsPerRow(ColumnView targets) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert targets.getType().equals(DType.STRING) : "targets type must be a string";
    assert getRowCount() == targets.getRowCount()
        : "column and targets must have the same number of rows";
    return new ColumnVector(stringContainsPerRow(getNativeView(), targets.getNativeView()));
  }

  /**
   * Replaces values less than `lo` in `input` with `lo`,
   * and values greater than `hi` with `hi`.
   *
   * if `lo` is invalid, then lo will not be considered while
   * evaluating the input (Essentially considered minimum value of that type).
   * if `hi` is invalid, then hi will not be considered while
   * evaluating the input (Essentially considered maximum value of that type).
   *
   * ```
   * Example:
   * input: {1, 2, 3, NULL, 5, 6, 7}
   *
   * valid lo and hi
   * lo: 3, hi: 5, lo_replace : 0, hi_replace : 16
   * output:{0, 0, 3, NULL, 5, 16, 16}
   *
   * invalid lo
   * lo: NULL, hi: 5, lo_replace : 0, hi_replace : 16
   * output:{1, 2, 3, NULL, 5, 16, 16}
   *
   * invalid hi
   * lo: 3, hi: NULL, lo_replace : 0, hi_replace : 16
   * output:{0, 0, 3, NULL, 5, 6, 7}
   * ```
   * @param lo - Minimum clamp value. All elements less than `lo` will be replaced by `lo`.
   *           Ignored if null.
   * @param hi - Maximum clamp value. All elements greater than `hi` will be replaced by `hi`.
   *           Ignored if null.
   * @return Returns a new clamped column as per `lo` and `hi` boundaries
   */
  public final ColumnVector clamp(Scalar lo, Scalar hi) {
    return new ColumnVector(clamper(this.getNativeView(), lo.getScalarHandle(),
        lo.getScalarHandle(), hi.getScalarHandle(), hi.getScalarHandle()));
  }

  /**
   * Replaces values less than `lo` in `input` with `lo_replace`,
   * and values greater than `hi` with `hi_replace`.
   *
   * if `lo` is invalid, then lo will not be considered while
   * evaluating the input (Essentially considered minimum value of that type).
   * if `hi` is invalid, then hi will not be considered while
   * evaluating the input (Essentially considered maximum value of that type).
   *
   * @note: If `lo` is valid then `lo_replace` should be valid
   *        If `hi` is valid then `hi_replace` should be valid
   *
   * ```
   * Example:
   *    input: {1, 2, 3, NULL, 5, 6, 7}
   *
   *    valid lo and hi
   *    lo: 3, hi: 5, lo_replace : 0, hi_replace : 16
   *    output:{0, 0, 3, NULL, 5, 16, 16}
   *
   *    invalid lo
   *    lo: NULL, hi: 5, lo_replace : 0, hi_replace : 16
   *    output:{1, 2, 3, NULL, 5, 16, 16}
   *
   *    invalid hi
   *    lo: 3, hi: NULL, lo_replace : 0, hi_replace : 16
   *    output:{0, 0, 3, NULL, 5, 6, 7}
   * ```
   *
   * @param lo - Minimum clamp value. All elements less than `lo` will be replaced by `loReplace`. Ignored if null.
   * @param loReplace - All elements less than `lo` will be replaced by `loReplace`.
   * @param hi - Maximum clamp value. All elements greater than `hi` will be replaced by `hiReplace`. Ignored if null.
   * @param hiReplace - All elements greater than `hi` will be replaced by `hiReplace`.
   * @return - a new clamped column as per `lo` and `hi` boundaries
   */
  public final ColumnVector clamp(Scalar lo, Scalar loReplace, Scalar hi, Scalar hiReplace) {
    return new ColumnVector(clamper(this.getNativeView(), lo.getScalarHandle(),
        loReplace.getScalarHandle(), hi.getScalarHandle(), hiReplace.getScalarHandle()));
  }

  /**
   * Returns a boolean ColumnVector identifying rows which
   * match the given regex pattern but only at the beginning of the string.
   *
   * ```
   * cv = ["abc", "123", "def456"]
   * result = cv.matchesRe("\\d+")
   * r is now [false, true, false]
   * ```
   * Any null string entries return corresponding null output column entries.
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   *
   * @param pattern Regex pattern to match to each string.
   * @return New ColumnVector of boolean results for each string.
   */
  @Deprecated
  public final ColumnVector matchesRe(String pattern) {
    return matchesRe(new RegexProgram(pattern, CaptureGroups.NON_CAPTURE));
  }

  /**
   * Returns a boolean ColumnVector identifying rows which
   * match the given regex program pattern but only at the beginning of the string.
   *
   * ```
   * cv = ["abc", "123", "def456"]
   * p = new RegexProgram("\\d+", CaptureGroups.NON_CAPTURE)
   * r = cv.matchesRe(p)
   * r is now [false, true, false]
   * ```
   * Any null string entries return corresponding null output column entries.
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   *
   * @param regexProg Regex program to match to each string.
   * @return New ColumnVector of boolean results for each string.
   */
  public final ColumnVector matchesRe(RegexProgram regexProg) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert regexProg != null : "regex program may not be null";
    assert !regexProg.pattern().isEmpty() : "pattern string may not be empty";
    return new ColumnVector(matchesRe(getNativeView(), regexProg.pattern(),
                                      regexProg.combinedFlags(), regexProg.capture().nativeId));
  }

  /**
   * Returns a boolean ColumnVector identifying rows which
   * match the given regex pattern starting at any location.
   *
   * ```
   * cv = ["abc", "123", "def456"]
   * r = cv.containsRe("\\d+")
   * r is now [false, true, true]
   * ```
   * Any null string entries return corresponding null output column entries.
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   *
   * @param pattern Regex pattern to match to each string.
   * @return New ColumnVector of boolean results for each string.
   */
  @Deprecated
  public final ColumnVector containsRe(String pattern) {
    return containsRe(new RegexProgram(pattern, CaptureGroups.NON_CAPTURE));
  }

  /**
   * Returns a boolean ColumnVector identifying rows which
   * match the given RegexProgram pattern starting at any location.
   *
   * ```
   * cv = ["abc", "123", "def456"]
   * p = new RegexProgram("\\d+", CaptureGroups.NON_CAPTURE)
   * r = cv.containsRe(p)
   * r is now [false, true, true]
   * ```
   * Any null string entries return corresponding null output column entries.
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   *
   * @param regexProg Regex program to match to each string.
   * @return New ColumnVector of boolean results for each string.
   */
  public final ColumnVector containsRe(RegexProgram regexProg) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert regexProg != null : "regex program may not be null";
    assert !regexProg.pattern().isEmpty() : "pattern string may not be empty";
    return new ColumnVector(containsRe(getNativeView(), regexProg.pattern(),
                                       regexProg.combinedFlags(), regexProg.capture().nativeId));
  }

  /**
   * For each captured group specified in the given regular expression
   * return a column in the table. Null entries are added if the string
   * does not match. Any null inputs also result in null output entries.
   *
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   * @param pattern the pattern to use
   * @return the table of extracted matches
   * @throws CudfException if any error happens including if the RE does
   * not contain any capture groups.
   */
  @Deprecated
  public final Table extractRe(String pattern) throws CudfException {
    return extractRe(new RegexProgram(pattern));
  }

  /**
   * For each captured group specified in the given regex program
   * return a column in the table. Null entries are added if the string
   * does not match. Any null inputs also result in null output entries.
   *
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   * @param regexProg the regex program to use
   * @return the table of extracted matches
   * @throws CudfException if any error happens including if the regex
   * program does not contain any capture groups.
   */
  public final Table extractRe(RegexProgram regexProg) throws CudfException {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert regexProg != null : "regex program may not be null";
    return new Table(extractRe(this.getNativeView(), regexProg.pattern(),
                               regexProg.combinedFlags(), regexProg.capture().nativeId));
  }

  /**
   * Extracts all strings that match the given regular expression and corresponds to the
   * regular expression group index. Any null inputs also result in null output entries.
   *
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   * @param pattern The regex pattern
   * @param idx The regex group index
   * @return A new column vector of extracted matches
   */
  @Deprecated
  public final ColumnVector extractAllRecord(String pattern, int idx) {
    if (idx == 0) {
      return extractAllRecord(new RegexProgram(pattern, CaptureGroups.NON_CAPTURE), idx);
    }
    return extractAllRecord(new RegexProgram(pattern), idx);
  }

  /**
   * Extracts all strings that match the given regex program pattern and corresponds to the
   * regular expression group index. Any null inputs also result in null output entries.
   *
   * For supported regex patterns refer to:
   * @link https://docs.rapids.ai/api/libcudf/nightly/md_regex.html
   * @param regexProg The regex program
   * @param idx The regex group index
   * @return A new column vector of extracted matches
   */
  public final ColumnVector extractAllRecord(RegexProgram regexProg, int idx) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert idx >= 0 : "group index must be at least 0";
    assert regexProg != null : "regex program may not be null";
    return new ColumnVector(
        extractAllRecord(this.getNativeView(), regexProg.pattern(), regexProg.combinedFlags(),
                         regexProg.capture().nativeId, idx));
  }

  /**
   * Returns a boolean ColumnVector identifying rows which
   * match the given like pattern.
   *
   * The like pattern expects only 2 wildcard special characters
   * - `%` any number of any character (including no characters)
   * - `_` any single character
   *
   * ```
   * cv = ["azaa", "ababaabba", "aaxa"]
   * r = cv.like("%a_aa%", "\\")
   * r is now [true, true, false]
   * r = cv.like("a__a", "\\")
   * r is now [true, false, true]
   * ```
   *
   * The escape character is specified to include either `%` or `_` in the search,
   * which is expected to be either 0 or 1 character.
   * If more than one character is specified, only the first character is used.
   *
   * ```
   * cv = ["abc_def", "abc1def", "abc_"]
   * r = cv.like("abc/_d%", "/")
   * r is now [true, false, false]
   * ```
   * Any null string entries return corresponding null output column entries.
   *
   * @param pattern Like pattern to match to each string.
   * @param escapeChar Character specifies the escape prefix; default is "\\".
   * @return New ColumnVector of boolean results for each string.
   */
  public final ColumnVector like(Scalar pattern, Scalar escapeChar) {
    assert type.equals(DType.STRING) : "column type must be a String";
    assert pattern != null : "pattern scalar must not be null";
    assert pattern.getType().equals(DType.STRING) : "pattern scalar must be a string scalar";
    assert escapeChar != null : "escapeChar scalar must not be null";
    assert escapeChar.getType().equals(DType.STRING) : "escapeChar scalar must be a string scalar";
    return new ColumnVector(like(getNativeView(), pattern.getScalarHandle(), escapeChar.getScalarHandle()));
  }


  /**
   * Converts all character sequences starting with '%' into character code-points
   * interpreting the 2 following characters as hex values to create the code-point.
   * For example, the sequence '%20' is converted into byte (0x20) which is a single
   * space character. Another example converts '%C3%A9' into 2 sequential bytes
   * (0xc3 and 0xa9 respectively) which is the é character. Overall, 3 characters
   * are converted into one char byte whenever a '%%' (single percent) character
   * is encountered in the string.
   * <p>
   * Any null entries will result in corresponding null entries in the output column.
   *
   * @return a new column instance containing the decoded strings
   */
  public final ColumnVector urlDecode() throws CudfException {
    assert type.equals(DType.STRING) : "column type must be a String";
    return new ColumnVector(urlDecode(getNativeView()));
  }

  /**
   * Converts mostly non-ascii characters and control characters into UTF-8 hex code-points
   * prefixed with '%'. For example, the space character must be converted to characters '%20' where
   * the '20' indicates the hex value for space in UTF-8. Likewise, multi-byte characters are
   * converted to multiple hex characters. For example, the é character is converted to characters
   * '%C3%A9' where 'C3A9' is the UTF-8 bytes 0xC3A9 for this character.
   * <p>
   * Any null entries will result in corresponding null entries in the output column.
   *
   * @return a new column instance containing the encoded strings
   */
  public final ColumnVector urlEncode() throws CudfException {
    assert type.equals(DType.STRING) : "column type must be a String";
    return new ColumnVector(urlEncode(getNativeView()));
  }

  private static void assertIsSupportedMapKeyType(DType keyType) {
    boolean isSupportedKeyType =
      !keyType.equals(DType.EMPTY) && !keyType.equals(DType.LIST) && !keyType.equals(DType.STRUCT);
    assert isSupportedKeyType : "Map lookup by STRUCT and LIST keys is not supported.";
  }

  /**
   * Given a column of type List<Struct<X, Y>> and a key column of type X, return a column of type Y,
   * where each row in the output column is the Y value corresponding to the X key.
   * If the key is not found, the corresponding output value is null.
   * @param keys the column view with keys to lookup in the column
   * @return a column of values or nulls based on the lookup result
   */
  public final ColumnVector getMapValue(ColumnView keys) {
    assert type.equals(DType.LIST) : "column type must be a LIST";
    assert keys != null : "Lookup key may not be null";
    return new ColumnVector(mapLookupForKeys(getNativeView(), keys.getNativeView()));
  }

  /**
   * Given a column of type List<Struct<X, Y>> and a key of type X, return a column of type Y,
   * where each row in the output column is the Y value corresponding to the X key.
   * If the key is not found, the corresponding output value is null.
   * @param key the scalar key to lookup in the column
   * @return a column of values or nulls based on the lookup result
   */
  public final ColumnVector getMapValue(Scalar key) {
    assert type.equals(DType.LIST) : "column type must be a LIST";
    assert key != null : "Lookup key may not be null";
    assertIsSupportedMapKeyType(key.getType());
    return new ColumnVector(mapLookup(getNativeView(), key.getScalarHandle()));
  }

  /** For a column of type List<Struct<String, String>> and a passed in String key, return a boolean
   * column for all keys in the structs, It is true if the key exists in the corresponding map for
   * that row, false otherwise. It will never return null for a row.
   * @param key the String scalar to lookup in the column
   * @return a boolean column based on the lookup result
   */
  public final ColumnVector getMapKeyExistence(Scalar key) {
    assert type.equals(DType.LIST) : "column type must be a LIST";
    assert key != null : "Lookup key may not be null";
    assertIsSupportedMapKeyType(key.getType());
    return new ColumnVector(mapContains(getNativeView(), key.getScalarHandle()));
  }

  /** For a column of type List<Struct<_, _>> and a passed in key column, return a boolean
   * column for all keys in the map. Each output row is true if the key exists in the corresponding map for
   * that row, false otherwise. It will never return null for a row.
   * @param keys the keys to lookup in the column
   * @return a boolean column based on the lookup result
   */
  public final ColumnVector getMapKeyExistence(ColumnView keys) {
    assert type.equals(DType.LIST) : "column type must be a LIST";
    assert keys != null : "Lookup key may not be null";
    assertIsSupportedMapKeyType(keys.getType());
    return new ColumnVector(mapContainsKeys(getNativeView(), keys.getNativeView()));
  }

  /**
   * Create a new struct column view of existing column views. Note that this will NOT copy
   * the contents of the input columns to make a new vector, but makes a view that must not
   * outlive the child views that it references. The resulting column cannot be null.
   * @param rows the number of rows in the struct column. This is needed if no columns
   *             are provided.
   * @param columns the columns to add to the struct in the order they should be added
   * @return the new column view. It is the responsibility of the caller to close this.
   */
  public static ColumnView makeStructView(long rows, ColumnView... columns) {
    long[] handles = new long[columns.length];
    for (int i = 0; i < columns.length; i++) {
      ColumnView cv = columns[i];
      if (rows != cv.getRowCount()) {
        throw new IllegalArgumentException("All columns must have the same number of rows");
      }
      handles[i] = cv.getNativeView();
    }
    return new ColumnView(makeStructView(handles, rows));
  }

  /**
   * Create a new struct column view of existing column views. Note that this will NOT copy
   * the contents of the input columns to make a new vector, but makes a view that must not
   * outlive the child views that it references. The resulting column cannot be null.
   * @param columns the columns to add to the struct in the order they should be added
   * @return the new column view. It is the responsibility of the caller to close this.
   */
  public static ColumnView makeStructView(ColumnView... columns) {
    if (columns.length <= 0) {
      throw new IllegalArgumentException("At least one column is needed to get the row count");
    }
    return makeStructView(columns[0].rows, columns);
  }

  /**
   * Create a new column view from a raw device buffer. Note that this will NOT copy
   * the contents of the buffer but only creates a view. The view MUST NOT outlive
   * the underlying device buffer. The column view will be created without a validity
   * vector, so it is not possible to create a view containing null elements. Additionally
   * only fixed-width primitive types are supported.
   *
   * @param buffer device memory that will back the column view
   * @param startOffset byte offset into the device buffer where the column data starts
   * @param type type of data in the column view
   * @param rows number of data elements in the column view
   * @return new column view instance that must not outlive the backing device buffer
   */
  public static ColumnView fromDeviceBuffer(BaseDeviceMemoryBuffer buffer,
                                            long startOffset,
                                            DType type,
                                            int rows) {
    if (buffer == null) {
      throw new NullPointerException("buffer is null");
    }
    int typeSize = type.getSizeInBytes();
    if (typeSize <= 0) {
      throw new IllegalArgumentException("Unsupported type: " + type);
    }
    if (startOffset < 0) {
      throw new IllegalArgumentException("Invalid start offset: " + startOffset);
    }
    if (rows < 0) {
      throw new IllegalArgumentException("Invalid row count: " + rows);
    }
    long dataSize = typeSize * rows;
    if (startOffset + dataSize > buffer.length) {
      throw new IllegalArgumentException("View extends beyond buffer range");
    }
    long dataAddress = buffer.getAddress() + startOffset;
    if (dataAddress % typeSize != 0) {
      throw new IllegalArgumentException("Data address " + Long.toHexString(dataAddress) +
          " is misaligned relative to type size of " + typeSize + " bytes");
    }
    return new ColumnView(makeCudfColumnView(type.typeId.getNativeId(), type.getScale(),
        dataAddress, dataSize, 0, 0, 0, rows, null));
  }

  /**
   * Create a column of bool values indicating whether the specified scalar
   * is an element of each row of a list column.
   * Output `column[i]` is set to null if one or more of the following are true:
   * 1. The key is null
   * 2. The column vector list value is null
   * @param key the scalar to look up
   * @return a Boolean ColumnVector with the result of the lookup
   */
  public final ColumnVector listContains(Scalar key) {
    assert type.equals(DType.LIST) : "column type must be a LIST";
    return new ColumnVector(listContains(getNativeView(), key.getScalarHandle()));
  }

  /**
   * Create a column of bool values indicating whether the list rows of the first
   * column contain the corresponding values in the second column.
   * Output `column[i]` is set to null if one or more of the following are true:
   * 1. The key value is null
   * 2. The column vector list value is null
   * @param key the ColumnVector with look up values
   * @return a Boolean ColumnVector with the result of the lookup
   */
  public final ColumnVector listContainsColumn(ColumnView key) {
    assert type.equals(DType.LIST) : "column type must be a LIST";
    return new ColumnVector(listContainsColumn(getNativeView(), key.getNativeView()));
  }

  /**
   * Create a column of bool values indicating whether the list rows of the specified
   * column contain null elements.
   * Output `column[i]` is set to null iff the input list row is null.
   * @return a Boolean ColumnVector with the result of the lookup
   */
  public final ColumnVector listContainsNulls() {
    assert type.equals(DType.LIST) : "column type must be a LIST";
    return new ColumnVector(listContainsNulls(getNativeView()));
  }

  /**
   * Enum to choose behaviour of listIndexOf functions:
   *   1. FIND_FIRST finds the first occurrence of a search key.
   *   2. FIND_LAST finds the last occurrence of a search key.
   */
  public enum FindOptions {FIND_FIRST, FIND_LAST};

  /**
   * Create a column of int32 indices, indicating the position of the scalar search key
   * in each list row.
   * All indices are 0-based. If a search key is not found, the index is set to -1.
   * The index is set to null if one of the following is true:
   * 1. The search key is null.
   * 2. The list row is null.
   * @param key The scalar search key
   * @param findOption Whether to find the first index of the key, or the last.
   * @return The resultant column of int32 indices
   */
  public final ColumnVector listIndexOf(Scalar key, FindOptions findOption) {
    assert type.equals(DType.LIST) : "column type must be a LIST";
    boolean isFindFirst = findOption == FindOptions.FIND_FIRST;
    return new ColumnVector(listIndexOfScalar(getNativeView(), key.getScalarHandle(), isFindFirst));
  }

  /**
   * Create a column of int32 indices, indicating the position of each row in the
   * search key column in the corresponding row of the lists column.
   * All indices are 0-based. If a search key is not found, the index is set to -1.
   * The index is set to null if one of the following is true:
   * 1. The search key row is null.
   * 2. The list row is null.
   * @param keys ColumnView of search keys.
   * @param findOption Whether to find the first index of the key, or the last.
   * @return The resultant column of int32 indices
   */
  public final ColumnVector listIndexOf(ColumnView keys, FindOptions findOption) {
    assert type.equals(DType.LIST) : "column type must be a LIST";
    boolean isFindFirst = findOption == FindOptions.FIND_FIRST;
    return new ColumnVector(listIndexOfColumn(getNativeView(), keys.getNativeView(), isFindFirst));
  }

  /**
   * Segmented sort of the elements within a list in each row of a list column.
   * The list child may be of any type whose leaf elements are relationally comparable, including
   * arbitrarily nested LIST and STRUCT children.
   *
   * @param isDescending   whether sorting each row with descending order (or ascending order)
   * @param isNullSmallest whether to regard the null value as the min value (or the max value)
   * @return a List ColumnVector with elements in each list sorted
   */
  public final ColumnVector listSortRows(boolean isDescending, boolean isNullSmallest) {
    assert type.equals(DType.LIST) : "column type must be a LIST";
    return new ColumnVector(listSortRows(getNativeView(), isDescending, isNullSmallest));
  }

  /**
   * For each pair of lists from the input lists columns, check if they have any common non-null
   * elements.
   *
   * A null input row in any of the input columns will result in a null output row. During checking
   * for common elements, nulls within each list are considered as different values while
   * floating-point NaN values are considered as equal.
   *
   * The input lists columns must have the same size and same data type.
   *
   * @param lhs The input lists column for one side
   * @param rhs The input lists column for the other side
   * @return A column of type BOOL8 containing the check result
   */
  public static ColumnVector listsHaveOverlap(ColumnView lhs, ColumnView rhs) {
    assert lhs.getType().equals(DType.LIST) && rhs.getType().equals(DType.LIST) :
        "Input columns type must be of type LIST";
    assert lhs.getRowCount() == rhs.getRowCount() : "Input columns must have the same size";
    return new ColumnVector(listsHaveOverlap(lhs.getNativeView(), rhs.getNativeView()));
  }

  /**
   * Find the intersection without duplicate between lists at each row of the given lists columns.
   *
   * A null input row in any of the input lists columns will result in a null output row. During
   * finding list intersection, nulls and floating-point NaN values within each list are
   * considered as equal values.
   *
   * The input lists columns must have the same size and same data type.
   *
   * @param lhs The input lists column for one side
   * @param rhs The input lists column for the other side
   * @return A lists column containing the intersection result
   */
  public static ColumnVector listsIntersectDistinct(ColumnView lhs, ColumnView rhs) {
    assert lhs.getType().equals(DType.LIST) && rhs.getType().equals(DType.LIST) :
        "Input columns type must be of type LIST";
    assert lhs.getRowCount() == rhs.getRowCount() : "Input columns must have the same size";
    return new ColumnVector(listsIntersectDistinct(lhs.getNativeView(), rhs.getNativeView()));
  }

  /**
   * Find the union without duplicate between lists at each row of the given lists columns.
   *
   * A null input row in any of the input lists columns will result in a null output row. During
   * finding list union, nulls and floating-point NaN values within each list are considered as
   * equal values.
   *
   * The input lists columns must have the same size and same data type.
   *
   * @param lhs The input lists column for one side
   * @param rhs The input lists column for the other side
   * @return A lists column containing the union result
   */
  public static ColumnVector listsUnionDistinct(ColumnView lhs, ColumnView rhs) {
    assert lhs.getType().equals(DType.LIST) && rhs.getType().equals(DType.LIST) :
        "Input columns type must be of type LIST";
    assert lhs.getRowCount() == rhs.getRowCount() : "Input columns must have the same size";
    return new ColumnVector(listsUnionDistinct(lhs.getNativeView(), rhs.getNativeView()));
  }

  /**
   * Find the difference of lists of the left column against lists of the right column.
   * Specifically, find the elements (without duplicates) from each list of the left column that
   * do not exist in the corresponding list of the right column.
   *
   * A null input row in any of the input lists columns will result in a null output row. During
   * finding, nulls and floating-point NaN values within each list are considered as equal values.
   *
   * The input lists columns must have the same size and same data type.
   *
   * @param lhs The input lists column for one side
   * @param rhs The input lists column for the other side
   * @return A lists column containing the difference result
   */
  public static ColumnVector listsDifferenceDistinct(ColumnView lhs, ColumnView rhs) {
    assert lhs.getType().equals(DType.LIST) && rhs.getType().equals(DType.LIST) :
        "Input columns type must be of type LIST";
    assert lhs.getRowCount() == rhs.getRowCount() : "Input columns must have the same size";
    return new ColumnVector(listsDifferenceDistinct(lhs.getNativeView(), rhs.getNativeView()));
  }

  /**
   * Generate list offsets from sizes of each list.
   * NOTICE: This API only works for INT32. Otherwise, the behavior is undefined. And no null and negative value is allowed.
   *
   * @return a column of list offsets whose size is N + 1
   */
  public final ColumnVector generateListOffsets() {
    return new ColumnVector(generateListOffsets(getNativeView()));
  }

  /**
   * Get a single item from the column at the specified index as a Scalar.
   *
   * Be careful. This is expensive and may involve running a kernel to copy the data out.
   *
   * @param index the index to look at
   * @return the value at that index as a scalar.
   * @throws CudfException if the index is out of bounds.
   */
  public final Scalar getScalarElement(int index) {
    return new Scalar(getType(), getElement(getNativeView(), index));
  }

  /**
   * Filters elements in each row of this LIST column using `booleanMaskView`
   * LIST of booleans as a mask.
   * <p>
   * Given a list-of-bools column, the function produces
   * a new `LIST` column of the same type as this column, where each element is copied
   * from the row *only* if the corresponding `boolean_mask` is non-null and `true`.
   * <p>
   * E.g.
   * column       = { {0,1,2}, {3,4}, {5,6,7}, {8,9} };
   * boolean_mask = { {0,1,1}, {1,0}, {1,1,1}, {0,0} };
   * results      = { {1,2},   {3},   {5,6,7}, {} };
   * <p>
   * This column and `boolean_mask` must have the same number of rows.
   * The output column has the same number of rows as this column.
   * An element is copied to an output row *only*
   * if the corresponding boolean_mask element is `true`.
   * An output row is invalid only if the row is invalid.
   *
   * @param booleanMaskView A nullable list of bools column used to filter elements in this column
   * @return List column of the same type as this column, containing filtered list rows
   * @throws CudfException if `boolean_mask` is not a "lists of bools" column
   * @throws CudfException if this column and `boolean_mask` have different number of rows
   */
  public final ColumnVector applyBooleanMask(ColumnView booleanMaskView) {
    assert (getType().equals(DType.LIST));
    assert (booleanMaskView.getType().equals(DType.LIST));
    assert (getRowCount() == booleanMaskView.getRowCount());
    return new ColumnVector(applyBooleanMask(getNativeView(), booleanMaskView.getNativeView()));
  }

  /**
   * Get the number of bytes needed to allocate a validity buffer for the given number of rows.
   * According to cudf::bitmask_allocation_size_bytes, the padding boundary for null mask is 64 bytes.
   */
  static long getValidityBufferSize(int numRows) {
    // number of bytes required = Math.ceil(number of bits / 8)
    long actualBytes = ((long) numRows + 7) >> 3;
    // padding to the multiplies of the padding boundary(64 bytes)
    return ((actualBytes + 63) >> 6) << 6;
  }

  /**
   * Count how many rows in the column are distinct from one another.
   * @param nullPolicy if nulls should be included or not.
   */
  public int distinctCount(NullPolicy nullPolicy) {
    return distinctCount(getNativeView(), nullPolicy.includeNulls);
  }

  /**
   * Count how many rows in the column are distinct from one another.
   * Nulls are included.
   */
  public int distinctCount() {
    return distinctCount(getNativeView(), true);
  }

  /////////////////////////////////////////////////////////////////////////////
  // INTERNAL/NATIVE ACCESS
  /////////////////////////////////////////////////////////////////////////////

  static DeviceMemoryBufferView getDataBuffer(long viewHandle) {
    long address = getNativeDataAddress(viewHandle);
    if (address == 0) {
      return null;
    }
    long length = getNativeDataLength(viewHandle);
    return new DeviceMemoryBufferView(address, length);
  }

  static DeviceMemoryBufferView getValidityBuffer(long viewHandle) {
    long address = getNativeValidityAddress(viewHandle);
    if (address == 0) {
      return null;
    }
    long length = getNativeValidityLength(viewHandle);
    return new DeviceMemoryBufferView(address, length);
  }

  static DeviceMemoryBufferView getOffsetsBuffer(long viewHandle) {
    long address = getNativeOffsetsAddress(viewHandle);
    if (address == 0) {
      return null;
    }
    long length = getNativeOffsetsLength(viewHandle);
    return new DeviceMemoryBufferView(address, length);
  }

  // Native Methods
  private static native int distinctCount(long handle, boolean nullsIncluded);

  /**
   * Native method to parse and convert a string column vector to unix timestamp. A unix
   * timestamp is a long value representing how many units since 1970-01-01 00:00:00.000 in either
   * positive or negative direction. This mirrors the functionality spark sql's to_unix_timestamp.
   * Strings that fail to parse will default to 0. Supported time units are second, millisecond,
   * microsecond, and nanosecond. Larger time units for column vectors are not supported yet in cudf.
   * No checking is done for invalid formats or invalid timestamp units.
   *
   * @param unit integer native ID of the time unit to parse the timestamp into.
   * @param format strptime format specifier string of the timestamp. Used to parse and convert
   *               the timestamp with. Supports %Y,%y,%m,%d,%H,%I,%p,%M,%S,%f,%z format specifiers.
   *               See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
   *               for parsing format specification and documentation.
   * @return native handle of the resulting cudf column, used to construct the Java column vector
   *         by the timestampToLong method.
   */
  private static native long stringTimestampToTimestamp(long viewHandle, int unit, String format);


  private static native long isFixedPoint(long viewHandle, int nativeTypeId, int scale);

  private static native long toHex(long viewHandle);

  /**
   * Native method to concatenate a list column of strings (each row is a list of strings),
   * concatenates the strings within each row and returns a single strings column result.
   * Each new string is created by concatenating the strings from the same row (same list element)
   * delimited by the row separator provided in the `separators` strings column.
   * @param listColumnHandle long holding the native handle of the column containing lists of strings
   *                         to concatenate.
   * @param sepColumn long holding the native handle of the strings column that provides separators
   *                  for concatenation.
   * @param separatorNarep string scalar indicating null behavior when a separator is null.
   *                       If set to null and the separator is null the resulting string will
   *                       be null. If not null, this string will be used in place of a null
   *                       separator.
   * @param colNarep string String scalar that should be used in place of any null strings
   *                 found in any column.
   * @param separateNulls boolean if true, then the separator is included for null rows if
   *                     `colNarep` is valid.
   * @param emptyStringOutputIfEmptyList boolean if true, any input row that is an empty list
   *                                     will result in an empty string. Otherwise, it will
   *                                     result in a null.
   * @return native handle of the resulting cudf column, used to construct the Java column.
   */
  private static native long stringConcatenationListElementsSepCol(long listColumnHandle,
                                                                   long sepColumn,
                                                                   long separatorNarep,
                                                                   long colNarep,
                                                                   boolean separateNulls,
                                                                   boolean emptyStringOutputIfEmptyList);

  /**
   * Native method to concatenate a list column of strings (each row is a list of strings),
   * concatenates the strings within each row and returns a single strings column result.
   * Each new string is created by concatenating the strings from the same row (same list element)
   * delimited by the separator provided.
   * @param listColumnHandle long holding the native handle of the column containing lists of strings
   *                     to concatenate.
   * @param separator string scalar inserted between each string being merged, may not be null.
   * @param narep string scalar indicating null behavior. If set to null and any string in the row
   *              is null the resulting string will be null. If not null, null values in any
   *              column will be replaced by the specified string. The underlying value in the
   *              string scalar may be null, but the object passed in may not.
   * @param separateNulls boolean if true, then the separator is included for null rows if
   *                      `narep` is valid.
   * @param emptyStringOutputIfEmptyList boolean if true, any input row that is an empty list
   *                                     will result in an empty string. Otherwise, it will
   *                                     result in a null.
   * @return native handle of the resulting cudf column, used to construct the Java column.
   */
  private static native long stringConcatenationListElements(long listColumnHandle,
                                                             long separator,
                                                             long narep,
                                                             boolean separateNulls,
                                                             boolean emptyStringOutputIfEmptyList);

  /**
   * Native method to repeat each string in the given input strings column a number of times
   * specified by the <code>repeatTimes</code> parameter.
   *
   * In special cases:
   *  - If <code>repeatTimes</code> is not a positive number, a non-null input string will always
   *    result in an empty output string.
   *  - A null input string will always result in a null output string regardless of the value of
   *    the <code>repeatTimes</code> parameter.
   *
   * @param viewHandle long holding the native handle of the column containing strings to repeat.
   * @param repeatTimes The number of times each input string is repeated.
   * @return native handle of the resulting cudf strings column containing repeated strings.
   */
  private static native long repeatStrings(long viewHandle, int repeatTimes);

  /**
   * Native method to repeat strings in the given input strings column, each string is repeated
   * by a different number of times given by the corresponding row in a <code>repeatTimes</code>
   * numeric column.
   *
   * In special cases:
   *  - Any null row (from either the input strings column or the <code>repeatTimes</code> column)
   *    will always result in a null output string.
   *  - If any value in the <code>repeatTimes</code> column is not a positive number and its
   *    corresponding input string is not null, the output string will be an empty string.
   *
   * If the input <code>repeatTimesHandle</code> column does not have a numeric type, or it has a
   * size that is different from size of the input strings column, an exception will be thrown.
   *
   * @param stringsHandle long holding the native handle of the column containing strings to repeat.
   * @param repeatTimesHandle long holding the native handle of the column containing the numbers
   *        of times each input string is repeated.
   * @return native handle of the resulting cudf strings column containing repeated strings.
   */
  private static native long repeatStringsWithColumnRepeatTimes(long stringsHandle,
    long repeatTimesHandle);


  private static native long getJSONObject(long viewHandle, long scalarHandle, boolean allowSingleQuotes, boolean stripQuotesFromSingleStrings, boolean missingFieldsAsNulls) throws CudfException;

  /**
   * Native method to parse and convert a timestamp column vector to string column vector. A unix
   * timestamp is a long value representing how many units since 1970-01-01 00:00:00:000 in either
   * positive or negative direction. This mirrors the functionality spark sql's from_unixtime.
   * No checking is done for invalid formats or invalid timestamp units.
   *
   * @param format - strftime format specifier string of the timestamp. Its used to parse and convert
   *               the timestamp with. Supports the following format specifiers:
   *               %d 	Day of the month: 01-31
   *               %m 	Month of the year: 01-12
   *               %y 	Year without century: 00-99
   *               %Y 	Year with century: 0001-9999
   *               %H 	24-hour of the day: 00-23
   *               %I 	12-hour of the day: 01-12
   *               %M 	Minute of the hour: 00-59
   *               %S 	Second of the minute: 00-59
   *               %f 	6-digit microsecond: 000000-999999
   *               %p 	One of: 'AM', 'PM', 'am', 'pm'
   *               %j 	Day of the year: 001-366
   *               See https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
   *
   * @return - native handle of the resulting cudf column used to construct the Java column vector
   */
  private static native long timestampToStringTimestamp(long viewHandle, String format);

  /**
   * Native method for locating the starting index of the first instance of a given substring
   * in each string in the column. 0 indexing, returns -1 if the substring is not found. Can be
   * be configured to start or end the search mid string.
   * @param columnView native handle of the cudf::column_view containing strings being operated on.
   * @param substringScalar string scalar handle containing the string to locate within each row.
   * @param start character index to start the search from (inclusive).
   * @param end character index to end the search on (exclusive).
   */
  private static native long substringLocate(long columnView, long substringScalar, int start, int end);

  /**
   * Returns a list of columns by splitting each string using the specified string literal
   * delimiter. The number of rows in the output columns will be the same as the input column.
   * Null entries are added for the rows where split results have been exhausted. Null input entries
   * result in all nulls in the corresponding rows of the output columns.
   *
   * @param nativeHandle native handle of the input strings column that being operated on.
   * @param delimiter UTF-8 encoded string identifying the split delimiter for each input string.
   * @param limit the maximum size of the list resulting from splitting each input string,
   *              or -1 for all possible splits. Note that limit = 0 (all possible splits without
   *              trailing empty strings) and limit = 1 (no split at all) are not supported.
   */
  private static native long[] stringSplit(long nativeHandle, String delimiter, int limit);

  /**
   * Returns a list of columns by splitting each string using the specified regular expression
   * pattern. The number of rows in the output columns will be the same as the input column.
   * Null entries are added for the rows where split results have been exhausted. Null input entries
   * result in all nulls in the corresponding rows of the output columns.
   *
   * @param nativeHandle native handle of the input strings column that being operated on.
   * @param pattern UTF-8 encoded string identifying the split regular expression pattern for
   *                each input string.
   * @param flags regex flags setting.
   * @param capture capture groups setting.
   * @param limit the maximum size of the list resulting from splitting each input string,
   *              or -1 for all possible splits. Note that limit = 0 (all possible splits without
   *              trailing empty strings) and limit = 1 (no split at all) are not supported.
   */
  private static native long[] stringSplitRe(long nativeHandle, String pattern, int flags,
                                             int capture, int limit);

  /**
   * Returns a column that are lists of strings in which each list is made by splitting the
   * corresponding input string using the specified string literal delimiter.
   *
   * @param nativeHandle native handle of the input strings column that being operated on.
   * @param delimiter UTF-8 encoded string identifying the split delimiter for each input string.
   * @param limit the maximum size of the list resulting from splitting each input string,
   *              or -1 for all possible splits. Note that limit = 0 (all possible splits without
   *              trailing empty strings) and limit = 1 (no split at all) are not supported.
   */
  private static native long stringSplitRecord(long nativeHandle, String delimiter, int limit);

  /**
   * Returns a column that are lists of strings in which each list is made by splitting the
   * corresponding input string using the specified regular expression pattern.
   *
   * @param nativeHandle native handle of the input strings column that being operated on.
   * @param pattern UTF-8 encoded string identifying the split regular expression pattern for
   *                each input string.
   * @param flags regex flags setting.
   * @param capture capture groups setting.
   * @param limit the maximum size of the list resulting from splitting each input string,
   *              or -1 for all possible splits. Note that limit = 0 (all possible splits without
   *              trailing empty strings) and limit = 1 (no split at all) are not supported.
   */
  private static native long stringSplitRecordRe(long nativeHandle, String pattern, int flags,
                                                 int capture, int limit);

  /**
   * Native method to calculate substring from a given string column. 0 indexing.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param start      first character index to begin the substring(inclusive).
   * @param end        last character index to stop the substring(exclusive).
   */
  private static native long substring(long columnView, int start, int end) throws CudfException;

  /**
   * Native method to extract substrings from a given strings column.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param start      first character index to begin the substrings (inclusive).
   */
  private static native long substringS(long columnView, int start) throws CudfException;

  /**
   * Native method to calculate substring from a given string column.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param startColumn handle of cudf::column_view which has start indices of each string.
   * @param endColumn handle of cudf::column_view which has end indices of each string.
   */
  private static native long substringColumn(long columnView, long startColumn, long endColumn)
      throws CudfException;

  /**
   * Native method to replace target string by repl string.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param target handle of scalar containing the string being searched.
   * @param repl handle of scalar containing the string to replace.
   */
  private static native long stringReplace(long columnView, long target, long repl) throws CudfException;

  /**
   * Native method to replace target strings by corresponding repl strings.
   * @param inputCV native handle of the cudf::column_view being operated on.
   * @param targetsCV handle of column containing the strings being searched.
   * @param replsCV handle of column containing the strings to replace (can optionally contain a single string).
   */
  private static native long stringReplaceMulti(long inputCV, long targetsCV, long replsCV) throws CudfException;

  /**
   * Native method for per-row string replacement.
   * @param inputCV native handle of the cudf::column_view being operated on.
   * @param targetsCV handle of column containing the per-row target strings.
   * @param replsCV handle of column containing the per-row replacement strings.
   */
  private static native long stringReplacePerRow(long inputCV, long targetsCV, long replsCV) throws CudfException;

  /**
   * Native method for replacing each regular expression pattern match with the specified
   * replacement string.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param pattern regular expression pattern to search within each string.
   * @param flags regex flags setting.
   * @param capture capture groups setting.
   * @param repl native handle of the cudf::scalar containing the replacement string.
   * @param maxRepl maximum number of times to replace the pattern within a string
   * @return native handle of the resulting cudf column containing the string results.
   */
  private static native long replaceRegex(long columnView, String pattern, int flags, int capture,
                                          long repl, long maxRepl) throws CudfException;

  /**
   * Native method for replacing any character sequence matching the given regex program
   * pattern using the replace template for back-references.
   * @param columnView native handle of the cudf::column_view being operated on.
   * @param pattern The regular expression patterns to search within each string.
   * @param flags Regex flags setting.
   * @param capture Capture groups setting.
   * @param replace The replacement template for creating the output string.
   * @return native handle of the resulting cudf column containing the string results.
   */
  private static native long stringReplaceWithBackrefs(long columnView, String pattern, int flags,
                                                       int capture, String replace) throws CudfException;

  /**
   * Native method for checking if strings in a column starts with a specified comparison string.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param compString handle of scalar containing the string being searched for at the beginning
  of each string in the column.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long stringStartWith(long cudfViewHandle, long compString) throws CudfException;

  /**
   * Native method for checking if strings in a column ends with a specified comparison string.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param compString handle of scalar containing the string being searched for at the end
  of each string in the column.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long stringEndWith(long cudfViewHandle, long compString) throws CudfException;

  /**
   * Native method to strip whitespace from the start and end of a string.
   * @param columnView native handle of the cudf::column_view being operated on.
   */
  private static native long stringStrip(long columnView, int type, long toStrip) throws CudfException;

  /**
   * Native method for checking if strings match the passed in regex program pattern from the
   * beginning of the string.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param pattern string regex pattern.
   * @param flags regex flags setting.
   * @param capture capture groups setting.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long matchesRe(long cudfViewHandle, String pattern, int flags, int capture) throws CudfException;

  /**
   * Native method for checking if strings match the passed in regex program pattern starting at any location.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param pattern string regex pattern.
   * @param flags regex flags setting.
   * @param capture capture groups setting.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long containsRe(long cudfViewHandle, String pattern, int flags, int capture) throws CudfException;

  /**
   * Native method for checking if strings match the passed in like pattern
   * and escape character.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param patternHandle handle of scalar containing the string like pattern.
   * @param escapeCharHandle handle of scalar containing the string escape character.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long like(long cudfViewHandle, long patternHandle, long escapeCharHandle) throws CudfException;

  /**
   * Native method for checking if strings in a column contains a specified comparison string.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param compString handle of scalar containing the string being searched for.
   * @return native handle of the resulting cudf column containing the boolean results.
   */
  private static native long stringContains(long cudfViewHandle, long compString) throws CudfException;

  /**
   * Row-aligned substring contains: row {@code i} searches for {@code targets[i]} in {@code input[i]}.
   */
  private static native long stringContainsPerRow(long cudfViewHandle, long targetsViewHandle)
      throws CudfException;

  /**
   * Native method for searching for the given target strings within each string in the provided column.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @param targetViewHandle handle of the column view containing the strings being searched for.
   */
  private static native long[] stringContainsMulti(long cudfViewHandle, long targetViewHandle) throws CudfException;

  /**
   * Native method for extracting results from a regex program pattern. Returns a table handle.
   *
   * @param cudfViewHandle Native handle of the cudf::column_view being operated on.
   * @param pattern String regex pattern.
   * @param flags Regex flags setting.
   * @param capture Capture groups setting.
   */
  private static native long[] extractRe(long cudfViewHandle, String pattern, int flags, int capture) throws CudfException;

  /**
   * Native method for extracting all results corresponding to group idx from a regex program pattern.
   *
   * @param nativeHandle Native handle of the cudf::column_view being operated on.
   * @param pattern String regex pattern.
   * @param flags Regex flags setting.
   * @param capture Capture groups setting.
   * @param idx Regex group index. A 0 value means matching the entire regex.
   * @return Native handle of a string column of the result.
   */
  private static native long extractAllRecord(long nativeHandle, String pattern, int flags, int capture, int idx);

  private static native long urlDecode(long cudfViewHandle);

  private static native long urlEncode(long cudfViewHandle);

  /**
   * Native method for map lookup over a column of List<Struct<String,String>>
   * @param columnView the column view handle of the map
   * @param key the string scalar that is the key for lookup
   * @return a string column handle of the resultant
   * @throws CudfException
   */
  private static native long mapLookup(long columnView, long key) throws CudfException;

  /**
   * Native method for map lookup over a column of List<Struct<String,String>>
   * The lookup column must have as many rows as the map column,
   * and must match the key-type of the map.
   * A column of values is returned, with the same number of rows as the map column.
   * If a key is repeated in a map row, the value corresponding to the last matching
   * key is returned.
   * If a lookup key is null or not found, the corresponding value is null.
   * @param columnView the column view handle of the map
   * @param keys       the column view holding the keys
   * @return a column of values corresponding the value of the lookup key.
   * @throws CudfException
   */
  private static native long mapLookupForKeys(long columnView, long keys) throws CudfException;

  /**
   * Native method for check the existence of a key over a column of List<Struct<_, _>>
   * @param columnView the column view handle of the map
   * @param key the column view holding the keys
   * @return boolean column handle of the result
   * @throws CudfException
   */
  private static native long mapContainsKeys(long columnView, long key) throws CudfException;

  /**
   * Native method for check the existence of a key over a column of List<Struct<String,String>>
   * @param columnView the column view handle of the map
   * @param key the string scalar that is the key for lookup
   * @return boolean column handle of the result
   * @throws CudfException
   */
  private static native long mapContains(long columnView, long key) throws CudfException;
  /**
   * Native method to add zeros as padding to the left of each string.
   */
  private static native long zfill(long nativeHandle, int width);

  private static native long pad(long nativeHandle, int width, int side, String fillChar);

  private static native long binaryOpVS(long lhs, long rhs, int op, int dtype, int scale);

  private static native long binaryOpVV(long lhs, long rhs, int op, int dtype, int scale);

  private static native long countElements(long viewHandle);

  private static native long byteCount(long viewHandle) throws CudfException;

  private static native long codePoints(long viewHandle);

  private static native long extractListElement(long nativeView, int index);

  private static native long extractListElementV(long nativeView, long indicesView);

  private static native long dropListDuplicates(long nativeView, int keep_option);

  private static native long dropListDuplicatesWithKeysValues(long nativeHandle);

  private static native long flattenLists(long inputHandle, boolean ignoreNull);

  /**
   * Native method for list lookup
   * @param nativeView the column view handle of the list
   * @param key the scalar key handle
   * @return column handle of the resultant
   */
  private static native long listContains(long nativeView, long key);

  /**
   * Native method for list lookup
   * @param nativeView the column view handle of the list
   * @param keyColumn the column handle of look up keys
   * @return column handle of the resultant
   */
  private static native long listContainsColumn(long nativeView, long keyColumn);

  /**
   * Native method to search list rows for null elements.
   * @param nativeView the column view handle of the list
   * @return column handle of the resultant boolean column
   */
  private static native long listContainsNulls(long nativeView);

  /**
   * Native method to find the first (or last) index of a specified scalar key,
   * in each row of a list column.
   * @param nativeView the column view handle of the list
   * @param scalarKeyHandle handle to the scalar search key
   * @param isFindFirst Whether to find the first index of the key, or the last.
   * @return column handle of the resultant column of int32 indices
   */
  private static native long listIndexOfScalar(long nativeView, long scalarKeyHandle, boolean isFindFirst);

  /**
   * Native method to find the first (or last) index of each search key in the specified column,
   * in each row of a list column.
   * @param nativeView the column view handle of the list
   * @param keyColumnHandle handle to the search key column
   * @param isFindFirst Whether to find the first index of the key, or the last.
   * @return column handle of the resultant column of int32 indices
   */
  private static native long listIndexOfColumn(long nativeView, long keyColumnHandle, boolean isFindFirst);

  private static native long listSortRows(long nativeView, boolean isDescending, boolean isNullSmallest);

  private static native long listsHaveOverlap(long lhsViewHandle, long rhsViewHandle);

  private static native long listsIntersectDistinct(long lhsViewHandle, long rhsViewHandle);

  private static native long listsUnionDistinct(long lhsViewHandle, long rhsViewHandle);

  private static native long listsDifferenceDistinct(long lhsViewHandle, long rhsViewHandle);

  private static native long getElement(long nativeView, int index);

  private static native long castTo(long nativeHandle, int type, int scale);

  private static native long bitCastTo(long nativeHandle, int type, int scale);

  private static native long byteListCast(long nativeHandle, boolean config);

  private static native long[] slice(long nativeHandle, int[] indices) throws CudfException;

  private static native long[] split(long nativeHandle, int[] indices) throws CudfException;

  private static native long findAndReplaceAll(long valuesHandle, long replaceHandle, long myself) throws CudfException;


  private static native long reverseStringsOrLists(long inputHandle);

  /**
   * Native method to switch all characters in a column of strings to lowercase characters.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @return native handle of the resulting cudf column, used to construct the Java column
   *         by the lower method.
   */
  private static native long lowerStrings(long cudfViewHandle);

  /**
   * Native method to switch all characters in a column of strings to uppercase characters.
   * @param cudfViewHandle native handle of the cudf::column_view being operated on.
   * @return native handle of the resulting cudf column, used to construct the Java column
   *         by the upper method.
   */
  private static native long upperStrings(long cudfViewHandle);

  /**
   * Native method to compute approx percentiles.
   * @param cudfColumnHandle T-Digest column
   * @param percentilesHandle Percentiles
   * @return native handle of the resulting cudf column, used to construct the Java column
   *         by the approxPercentile method.
   */
  private static native long approxPercentile(long cudfColumnHandle, long percentilesHandle) throws CudfException;

  private static native long quantile(long cudfColumnHandle, int quantileMethod, double[] quantiles) throws CudfException;

  private static native long rollingWindow(
      long viewHandle,
      long defaultOutputHandle,
      int min_periods,
      long aggPtr,
      int preceding,
      int following,
      long preceding_col,
      long following_col);

  private static native long scan(long viewHandle, long aggregation,
      boolean isInclusive, boolean includeNulls) throws CudfException;

  private static native long nansToNulls(long viewHandle) throws CudfException;

  private static native long charLengths(long viewHandle) throws CudfException;

  private static native long replaceNullsScalar(long viewHandle, long scalarHandle) throws CudfException;

  private static native long replaceNullsColumn(long viewHandle, long replaceViewHandle) throws CudfException;

  private static native long replaceNullsPolicy(long nativeView, boolean isPreceding) throws CudfException;

  private static native long ifElseVV(long predVec, long trueVec, long falseVec) throws CudfException;

  private static native long ifElseVS(long predVec, long trueVec, long falseScalar) throws CudfException;

  private static native long ifElseSV(long predVec, long trueScalar, long falseVec) throws CudfException;

  private static native long ifElseSS(long predVec, long trueScalar, long falseScalar) throws CudfException;

  private static native long reduce(long viewHandle, long aggregation, int dtype, int scale) throws CudfException;

  private static native long segmentedReduce(long dataViewHandle, long offsetsViewHandle,
      long aggregation, boolean includeNulls, int dtype, int scale) throws CudfException;

  private static native long segmentedGather(long sourceColumnHandle, long gatherMapListHandle,
      boolean isNullifyOutBounds) throws CudfException;

  private static native long isNullNative(long viewHandle);

  private static native long isNanNative(long viewHandle);

  private static native long isFloat(long viewHandle);

  private static native long isInteger(long viewHandle);

  private static native long isIntegerWithType(long viewHandle, int typeId, int typeScale);

  private static native long isNotNanNative(long viewHandle);

  private static native long isNotNullNative(long viewHandle);

  private static native long unaryOperation(long viewHandle, int op);

  private static native long extractDateTimeComponent(long viewHandle, int component);

  private static native long lastDayOfMonth(long viewHandle) throws CudfException;

  private static native long dayOfYear(long viewHandle) throws CudfException;

  private static native long quarterOfYear(long viewHandle) throws CudfException;

  private static native long addCalendricalMonths(long tsViewHandle, long monthsViewHandle);

  private static native long addScalarCalendricalMonths(long tsViewHandle, long scalarHandle);

  private static native long isLeapYear(long viewHandle) throws CudfException;

  private static native long daysInMonth(long viewHandle) throws CudfException;

  private static native long dateTimeCeil(long viewHandle, int freq);

  private static native long dateTimeFloor(long viewHandle, int freq);

  private static native long dateTimeRound(long viewHandle, int freq);

  private static native boolean containsScalar(long columnViewHaystack, long scalarHandle) throws CudfException;

  private static native long containsVector(long valuesHandle, long searchSpaceHandle) throws CudfException;

  private static native long transform(long viewHandle, String udf, boolean isPtx);

  private static native long clamper(long nativeView, long loScalarHandle, long loScalarReplaceHandle,
                                     long hiScalarHandle, long hiScalarReplaceHandle);

  protected static native long title(long handle);

  private static native long capitalize(long strsColHandle, long delimitersHandle);

  private static native long joinStrings(long strsHandle, long sepHandle, long narepHandle);

  private static native long makeStructView(long[] handles, long rowCount);

  private static native long isTimestamp(long nativeView, String format);
  /**
   * Native method to normalize the various bitwise representations of NAN and zero.
   *
   * All occurrences of -NaN are converted to NaN. Likewise, all -0.0 are converted to 0.0.
   *
   * @param viewHandle `long` representation of pointer to input column_view.
   * @return Pointer to a new `column` of normalized values.
   * @throws CudfException On failure to normalize.
   */
  private static native long normalizeNANsAndZeros(long viewHandle) throws CudfException;

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

  ////////
  // Native cudf::column_view life cycle and metadata access methods. Life cycle methods
  // should typically only be called from the OffHeap inner class.
  ////////

  static native int getNativeTypeId(long viewHandle) throws CudfException;

  static native int getNativeTypeScale(long viewHandle) throws CudfException;

  static native int getNativeRowCount(long viewHandle) throws CudfException;

  static native int getNativeNullCount(long viewHandle) throws CudfException;

  static native void deleteColumnView(long viewHandle) throws CudfException;

  private static native long getNativeDataAddress(long viewHandle) throws CudfException;
  private static native long getNativeDataLength(long viewHandle) throws CudfException;

  private static native long getNativeOffsetsAddress(long viewHandle) throws CudfException;
  private static native long getNativeOffsetsLength(long viewHandle) throws CudfException;

  private static native long getNativeValidityAddress(long viewHandle) throws CudfException;
  private static native long getNativeValidityLength(long viewHandle) throws CudfException;

  static native long makeCudfColumnView(int type, int scale, long data, long dataSize, long offsets,
                                                long valid, int nullCount, int size, long[] childHandle);


  static native long getChildCvPointer(long viewHandle, int childIndex) throws CudfException;

  private static native long getListOffsetCvPointer(long viewHandle) throws CudfException;

  static native int getNativeNumChildren(long viewHandle) throws CudfException;

  // calculate the amount of device memory used by this column including any child columns
  static native long getDeviceMemorySize(long viewHandle, boolean shouldPadForCpu) throws CudfException;

  static native long copyColumnViewToCV(long viewHandle) throws CudfException;

  static native long generateListOffsets(long handle) throws CudfException;

  static native long applyBooleanMask(long arrayColumnView, long booleanMaskHandle) throws CudfException;

  static native boolean hasNonEmptyNulls(long handle) throws CudfException;

  static native long purgeNonEmptyNulls(long handle) throws CudfException;

  /**
   * A utility class to create column vector like objects without refcounts and other APIs when
   * creating the device side vector from host side nested vectors. Eventually this can go away or
   * be refactored to hold less state like just the handles and the buffers to close.
   */
  static class NestedColumnVector {

    private final DeviceMemoryBuffer data;
    private final DeviceMemoryBuffer valid;
    private final DeviceMemoryBuffer offsets;
    private final DType dataType;
    private final long rows;
    private final Optional<Long> nullCount;
    List<NestedColumnVector> children;

    private NestedColumnVector(DType type, long rows, Optional<Long> nullCount,
        DeviceMemoryBuffer data, DeviceMemoryBuffer valid,
        DeviceMemoryBuffer offsets, List<NestedColumnVector> children) {
      this.dataType = type;
      this.rows = rows;
      this.nullCount = nullCount;
      this.data = data;
      this.valid = valid;
      this.offsets = offsets;
      this.children = children;
    }

    /**
     * Returns a LIST ColumnVector, for now, after constructing the NestedColumnVector from the host side
     * nested Column Vector - children. This is used for host side to device side copying internally.
     * @param type top level dtype, which is LIST currently
     * @param rows top level number of rows in the LIST column
     * @param valid validity buffer
     * @param offsets offsets buffer
     * @param nullCount nullCount for the LIST column
     * @param child the host side nested column vector list
     * @return new ColumnVector of type LIST at the moment
     */
    static ColumnVector createColumnVector(DType type, int rows, HostMemoryBuffer data,
        HostMemoryBuffer valid, HostMemoryBuffer offsets, Optional<Long> nullCount, List<HostColumnVectorCore> child) {
      List<NestedColumnVector> devChildren = new ArrayList<>();
      for (HostColumnVectorCore c : child) {
        devChildren.add(createNewNestedColumnVector(c));
      }
      int mainColRows = rows;
      DType mainColType = type;
      HostMemoryBuffer mainColValid = valid;
      HostMemoryBuffer mainColOffsets = offsets;
      DeviceMemoryBuffer mainDataDevBuff = null;
      DeviceMemoryBuffer mainValidDevBuff = null;
      DeviceMemoryBuffer mainOffsetsDevBuff = null;
      if (mainColValid != null) {
        long validLen = getValidityBufferSize(mainColRows);
        mainValidDevBuff = DeviceMemoryBuffer.allocate(validLen);
        mainValidDevBuff.copyFromHostBuffer(mainColValid, 0, validLen);
      }
      if (data != null) {
        long dataLen = data.length;
        mainDataDevBuff = DeviceMemoryBuffer.allocate(dataLen);
        mainDataDevBuff.copyFromHostBuffer(data, 0, dataLen);
      }
      if (mainColOffsets != null) {
        // The offset buffer has (no. of rows + 1) entries, where each entry is INT32.sizeInBytes
        long offsetsLen = OFFSET_SIZE * (((long)mainColRows) + 1);
        mainOffsetsDevBuff = DeviceMemoryBuffer.allocate(offsetsLen);
        mainOffsetsDevBuff.copyFromHostBuffer(mainColOffsets, 0, offsetsLen);
      }
      List<DeviceMemoryBuffer> toClose = new ArrayList<>();
      long[] childHandles = new long[devChildren.size()];
      try {
        for (ColumnView.NestedColumnVector ncv : devChildren) {
          toClose.addAll(ncv.getBuffersToClose());
        }
        for (int i = 0; i < devChildren.size(); i++) {
          childHandles[i] = devChildren.get(i).createViewHandle();
        }
        return new ColumnVector(mainColType, mainColRows, nullCount, mainDataDevBuff,
            mainValidDevBuff, mainOffsetsDevBuff, toClose, childHandles);
      } finally {
        for (int i = 0; i < childHandles.length; i++) {
          if (childHandles[i] != 0) {
            ColumnView.deleteColumnView(childHandles[i]);
            childHandles[i] = 0;
          }
        }
      }
    }

    private static NestedColumnVector createNewNestedColumnVector(
        HostColumnVectorCore nestedChildren) {
      if (nestedChildren == null) {
        return null;
      }
      DType colType = nestedChildren.getType();
      Optional<Long> nullCount = Optional.of(nestedChildren.getNullCount());
      long colRows = nestedChildren.getRowCount();
      HostMemoryBuffer colData = nestedChildren.getNestedChildren().isEmpty() ? nestedChildren.getData() : null;
      HostMemoryBuffer colValid = nestedChildren.getValidity();
      HostMemoryBuffer colOffsets = nestedChildren.getOffsets();

      List<NestedColumnVector> children = new ArrayList<>();
      for (HostColumnVectorCore nhcv : nestedChildren.getNestedChildren()) {
        children.add(createNewNestedColumnVector(nhcv));
      }
      return createNestedColumnVector(colType, colRows, nullCount, colData, colValid, colOffsets,
        children);
    }

    private long createViewHandle() {
      long[] childrenColViews = null;
      try {
        if (children != null) {
          childrenColViews = new long[children.size()];
          for (int i = 0; i < children.size(); i++) {
            childrenColViews[i] = children.get(i).createViewHandle();
          }
        }
        long dataAddr = data == null ? 0 : data.address;
        long dataLen = data == null ? 0 : data.length;
        long offsetAddr = offsets == null ? 0 : offsets.address;
        long validAddr = valid == null ? 0 : valid.address;
        int nc = nullCount.orElse(ColumnVector.OffHeapState.UNKNOWN_NULL_COUNT).intValue();
        return makeCudfColumnView(dataType.typeId.getNativeId(), dataType.getScale(), dataAddr, dataLen,
            offsetAddr, validAddr, nc, (int) rows, childrenColViews);
      } finally {
        if (childrenColViews != null) {
          for (int i = 0; i < childrenColViews.length; i++) {
            if (childrenColViews[i] != 0) {
              deleteColumnView(childrenColViews[i]);
              childrenColViews[i] = 0;
            }
          }
        }
      }
    }

    List<DeviceMemoryBuffer> getBuffersToClose() {
      List<DeviceMemoryBuffer> buffers = new ArrayList<>();
      if (children != null) {
        for (NestedColumnVector ncv : children) {
          buffers.addAll(ncv.getBuffersToClose());
        }
      }
      if (data != null) {
        buffers.add(data);
      }
      if (valid != null) {
        buffers.add(valid);
      }
      if (offsets != null) {
        buffers.add(offsets);
      }
      return buffers;
    }

    private static long getEndStringOffset(long totalRows, long index, HostMemoryBuffer offsets) {
      assert index < totalRows;
      return offsets.getInt((index + 1) * 4);
    }

    private static NestedColumnVector createNestedColumnVector(DType type, long rows, Optional<Long> nullCount,
        HostMemoryBuffer dataBuffer, HostMemoryBuffer validityBuffer,
        HostMemoryBuffer offsetBuffer, List<NestedColumnVector> child) {
      DeviceMemoryBuffer data = null;
      DeviceMemoryBuffer valid = null;
      DeviceMemoryBuffer offsets = null;
      if (dataBuffer != null) {
        long dataLen = rows * type.getSizeInBytes();
        if (type.equals(DType.STRING)) {
          // This needs a different type
          dataLen = getEndStringOffset(rows, rows - 1, offsetBuffer);
          if (dataLen == 0 && nullCount.get() == 0) {
            // This is a work around to an issue where a column of all empty strings must have at
            // least one byte or it will not be interpreted correctly.
            dataLen = 1;
          }
        }
        data = DeviceMemoryBuffer.allocate(dataLen);
        data.copyFromHostBuffer(dataBuffer, 0, dataLen);
      }
      if (validityBuffer != null) {
        long validLen = getValidityBufferSize((int)rows);
        valid = DeviceMemoryBuffer.allocate(validLen);
        valid.copyFromHostBuffer(validityBuffer, 0, validLen);
      }
      if (offsetBuffer != null) {
        long offsetsLen = OFFSET_SIZE * (rows + 1);
        offsets = DeviceMemoryBuffer.allocate(offsetsLen);
        offsets.copyFromHostBuffer(offsetBuffer, 0, offsetsLen);
      }
      NestedColumnVector ret = new NestedColumnVector(type, rows, nullCount, data, valid, offsets,
        child);
      return ret;
    }
  }


  /////////////////////////////////////////////////////////////////////////////
  // DATA MOVEMENT
  /////////////////////////////////////////////////////////////////////////////

  /**
   * Drain {@code stream} and clean up host-side resources after a failure in
   * {@link #copyToHostAsync} or {@link #copyToHostAsyncNestedHelper}. Sync
   * failures and close failures are recorded as suppressed on {@code primary}
   * so the original cause is preserved.
   *
   * <p>If the recovery sync itself fails, the CUDA stream is in an error state
   * and pending DMA writes into the host buffers may not have completed.
   * Closing them then would risk a use-after-free (the DMA engine writing into
   * freed memory), so the host buffers are intentionally leaked — the process
   * is already in a broken state (sticky CUDA error usually requires a
   * restart), and a leak beats silent corruption.
   */
  private static void syncAndCleanup(Cuda.Stream stream, Throwable primary,
      List<HostColumnVectorCore> children,
      HostMemoryBuffer hostData, HostMemoryBuffer hostOffsets, HostMemoryBuffer hostValid) {
    try {
      stream.sync();
    } catch (Throwable syncFailure) {
      // Stream is in an error state; pending DMA writes may not have landed.
      // Intentionally leak the host buffers rather than risk closing them while
      // the DMA engine could still write into freed memory.
      primary.addSuppressed(syncFailure);
      return;
    }
    if (children != null) {
      for (HostColumnVectorCore child : children) {
        CleanupHelpers.closeAndSuppress(child, primary);
      }
    }
    CleanupHelpers.closeAndSuppress(hostData, primary);
    CleanupHelpers.closeAndSuppress(hostOffsets, primary);
    CleanupHelpers.closeAndSuppress(hostValid, primary);
  }

  private static HostColumnVectorCore copyToHostAsyncNestedHelper(
      Cuda.Stream stream, ColumnView deviceCvPointer, HostMemoryAllocator hostMemoryAllocator) {
    if (deviceCvPointer == null) {
      return null;
    }
    HostMemoryBuffer hostOffsets = null;
    HostMemoryBuffer hostValid = null;
    HostMemoryBuffer hostData = null;
    List<HostColumnVectorCore> children = new ArrayList<>();
    BaseDeviceMemoryBuffer currData = null;
    BaseDeviceMemoryBuffer currOffsets = null;
    BaseDeviceMemoryBuffer currValidity = null;
    long currNullCount = 0l;
    try {
      long currRows = deviceCvPointer.getRowCount();
      DType currType = deviceCvPointer.getType();
      currData = deviceCvPointer.getData();
      currOffsets = deviceCvPointer.getOffsets();
      currValidity = deviceCvPointer.getValid();
      if (currData != null) {
        hostData = hostMemoryAllocator.allocate(currData.length);
        hostData.copyFromDeviceBufferAsync(currData, stream);
      }
      if (currValidity != null) {
        hostValid = hostMemoryAllocator.allocate(currValidity.length);
        hostValid.copyFromDeviceBufferAsync(currValidity, stream);
      }
      if (currOffsets != null) {
        hostOffsets = hostMemoryAllocator.allocate(currOffsets.length);
        hostOffsets.copyFromDeviceBufferAsync(currOffsets, stream);
      }
      int numChildren = deviceCvPointer.getNumChildren();
      for (int i = 0; i < numChildren; i++) {
        try(ColumnView childDevPtr = deviceCvPointer.getChildColumnView(i)) {
          children.add(copyToHostAsyncNestedHelper(stream, childDevPtr, hostMemoryAllocator));
        }
      }
      currNullCount = deviceCvPointer.getNullCount();
      Optional<Long> nullCount = Optional.of(currNullCount);
      return new HostColumnVectorCore(currType, currRows, nullCount, hostData,
          hostValid, hostOffsets, children);
    } catch (Throwable t) {
      syncAndCleanup(stream, t, children, hostData, hostOffsets, hostValid);
      throw t;
    } finally {
      // These are refcount handles to the parent ColumnView's device memory;
      // closing them only decrements a refcount and does not deallocate, so
      // they're safe to close regardless of stream state and do not need the
      // sync-then-suppress treatment applied to host buffers above.
      if (currData != null) {
        currData.close();
      }
      if (currOffsets != null) {
        currOffsets.close();
      }
      if (currValidity != null) {
        currValidity.close();
      }
    }
  }

  /** Copy the data to the host synchronously. */
  public HostColumnVector copyToHost(HostMemoryAllocator hostMemoryAllocator) {
    HostColumnVector result = copyToHostAsync(Cuda.DEFAULT_STREAM, hostMemoryAllocator);
    Cuda.DEFAULT_STREAM.sync();
    return result;
  }

  /**
   * Copy the data to the host asynchronously. The caller MUST synchronize on the stream
   * before examining the result.
   */
  public HostColumnVector copyToHostAsync(Cuda.Stream stream,
                                          HostMemoryAllocator hostMemoryAllocator) {
    try (NvtxRange toHost = new NvtxRange("toHostAsync", NvtxColor.BLUE)) {
      HostMemoryBuffer hostDataBuffer = null;
      HostMemoryBuffer hostValidityBuffer = null;
      HostMemoryBuffer hostOffsetsBuffer = null;
      List<HostColumnVectorCore> children = null;
      BaseDeviceMemoryBuffer valid = getValid();
      BaseDeviceMemoryBuffer offsets = getOffsets();
      BaseDeviceMemoryBuffer data = null;
      DType type = this.type;
      long rows = this.rows;
      if (!type.isNestedType()) {
        data = getData();
      }
      try {
        // We don't have a good way to tell if it is cached on the device or recalculate it on
        // the host for now, so take the hit here.
        getNullCount();
        if (!type.isNestedType()) {
          if (valid != null) {
            hostValidityBuffer = hostMemoryAllocator.allocate(valid.getLength());
            hostValidityBuffer.copyFromDeviceBufferAsync(valid, stream);
          }
          if (offsets != null) {
            hostOffsetsBuffer = hostMemoryAllocator.allocate(offsets.length);
            hostOffsetsBuffer.copyFromDeviceBufferAsync(offsets, stream);
          }
          // If a strings column is all null values there is no data buffer allocated
          if (data != null) {
            hostDataBuffer = hostMemoryAllocator.allocate(data.length);
            hostDataBuffer.copyFromDeviceBufferAsync(data, stream);
          }
          return new HostColumnVector(type, rows, Optional.of(nullCount),
              hostDataBuffer, hostValidityBuffer, hostOffsetsBuffer);
        } else {
          if (data != null) {
            hostDataBuffer = hostMemoryAllocator.allocate(data.length);
            hostDataBuffer.copyFromDeviceBufferAsync(data, stream);
          }

          if (valid != null) {
            hostValidityBuffer = hostMemoryAllocator.allocate(valid.getLength());
            hostValidityBuffer.copyFromDeviceBufferAsync(valid, stream);
          }
          if (offsets != null) {
            hostOffsetsBuffer = hostMemoryAllocator.allocate(offsets.getLength());
            hostOffsetsBuffer.copyFromDeviceBufferAsync(offsets, stream);
          }
          children = new ArrayList<>();
          for (int i = 0; i < getNumChildren(); i++) {
            try (ColumnView childDevPtr = getChildColumnView(i)) {
              children.add(copyToHostAsyncNestedHelper(stream, childDevPtr, hostMemoryAllocator));
            }
          }
          return new HostColumnVector(type, rows, Optional.of(nullCount),
              hostDataBuffer, hostValidityBuffer, hostOffsetsBuffer, children);
        }
      } catch (Throwable t) {
        syncAndCleanup(stream, t, children, hostDataBuffer, hostOffsetsBuffer, hostValidityBuffer);
        throw t;
      } finally {
        // These are refcount handles to this ColumnView's device memory;
        // closing them only decrements a refcount and does not deallocate, so
        // they're safe to close regardless of stream state and do not need the
        // sync-then-suppress treatment applied to host buffers above.
        if (data != null) {
          data.close();
        }
        if (offsets != null) {
          offsets.close();
        }
        if (valid != null) {
          valid.close();
        }
      }
    }
  }

  /** Copy the data to host memory synchronously */
  public HostColumnVector copyToHost() {
    return copyToHost(DefaultHostMemoryAllocator.get());
  }

  /**
   * Copy the data to the host asynchronously. The caller MUST synchronize on the stream
   * before examining the result.
   */
  public HostColumnVector copyToHostAsync(Cuda.Stream stream) {
    return copyToHostAsync(stream, DefaultHostMemoryAllocator.get());
  }

  /**
   * Calculate the total space required to copy the data to the host. This should be padded to
   * the alignment that the CPU requires.
   */
  public long getHostBytesRequired() {
    return getDeviceMemorySize(getNativeView(), true);
  }

  /**
   * Get the size that the host will align memory allocations to in bytes.
   */
  public static native long hostPaddingSizeInBytes();

  /**
   * Exact check if a column or its descendants have non-empty null rows
   *
   * @return Whether the column or its descendants have non-empty null rows
   */
  public boolean hasNonEmptyNulls() {
    return hasNonEmptyNulls(viewHandle);
  }

  /**
   * Copies this column into output while purging any non-empty null rows in the column or its
   * descendants.
   *
   * If this column is not of compound type (LIST/STRING/STRUCT/DICTIONARY), the output will be
   * the same as input.
   *
   * The purge operation only applies directly to LIST and STRING columns, but it applies indirectly
   * to STRUCT/DICTIONARY columns as well, since these columns may have child columns that
   * are LIST or STRING.
   *
   * Examples:
   * lists = data: [{{0,1}, {2,3}, {4,5}} validity: {true, false, true}]
   * lists[1] is null, but the list's child column still stores `{2,3}`.
   *
   * After purging the contents of the list's null rows, the column's contents will be:
   * lists = [data: {{0,1}, {4,5}} validity: {true, false, true}]
   *
   * @return A new column with equivalent contents to `input`, but with null rows purged
   */
  public ColumnVector purgeNonEmptyNulls() {
    return new ColumnVector(purgeNonEmptyNulls(viewHandle));
  }

  static ColumnView[] getColumnViewsFromPointers(long[] nativeHandles) {
    ColumnView[] columns = new ColumnView[nativeHandles.length];
    try {
      for (int i = 0; i < nativeHandles.length; i++) {
        long nativeHandle = nativeHandles[i];
        // setting address to zero, so we don't clean it in case of an exception as it
        // will be cleaned up by the constructor
        nativeHandles[i] = 0;
        columns[i] = new ColumnView(nativeHandle);
      }
      return columns;
    } catch (Throwable t) {
      try {
        cleanupColumnViews(nativeHandles, columns, t);
      } catch (Throwable s) {
        t.addSuppressed(s);
      } finally {
        throw t;
      }
    }
  }

  /**
   * Convert this integer column to hexadecimal column and return a new strings column
   *
   * Any null entries will result in corresponding null entries in the output column.
   *
   * The output character set is '0'-'9' and 'A'-'F'. The output string width will
   * be a multiple of 2 depending on the size of the integer type. A single leading
   * zero is applied to the first non-zero output byte if it is less than 0x10.
   *
   * Example:
   * input = [123, -1, 0, 27, 342718233]
   * s = input.toHex()
   * s is [ '04D2', 'FFFFFFFF', '00', '1B', '146D7719']
   *
   * The example above shows an `INT32` type column where each integer is 4 bytes.
   * Leading zeros are suppressed unless filling out a complete byte as in
   * `123 -> '04D2'` instead of `000004D2` or `4D2`.
   *
   * @return new string ColumnVector
   */
  public ColumnVector toHex() {
    assert getType().isIntegral() : "Only integers are supported";
    return new ColumnVector(toHex(this.getNativeView()));
  }
}

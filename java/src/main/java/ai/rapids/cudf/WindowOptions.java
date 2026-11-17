/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.Arrays;

/**
  * Options for rolling windows.
 */
public class WindowOptions implements AutoCloseable {

  enum FrameType {ROWS, RANGE}

  /**
   * Extent of (range) window bounds.
   * Analogous to cudf::range_window_bounds::extent_type.
   */
  enum RangeExtentType {
    CURRENT_ROW(0), // Bounds defined as the first/last row that matches the current row.
    BOUNDED(1),     // Bounds defined as the first/last row that falls within
                    // a specified range from the current row.
    UNBOUNDED(2);   // Bounds stretching to the first/last row in the entire group.

    public final int nominalValue;

    RangeExtentType(int n) {
      this.nominalValue = n;
    }
  }

  private final int minPeriods;
  private final Scalar precedingScalar;
  private final Scalar followingScalar;
  private final ColumnVector precedingCol;
  private final ColumnVector followingCol;
  // Single internal representation of the order-by columns: three parallel, non-empty arrays.
  // They are null only when no order-by was set, which selects a ROWS frame. The single-column
  // builder setters normalize to length-1 arrays in build(), so both construction paths produce
  // the same internal state.
  private final int[] orderByColumnIndices;
  private final boolean[] orderByAscendingFlags;
  private final boolean[] orderByNullsFirstFlags;
  private final FrameType frameType;
  private final RangeExtentType precedingBoundsExtent;
  private final RangeExtentType followingBoundsExtent;

  private WindowOptions(Builder builder) {
    this.minPeriods = builder.minPeriods;
    this.precedingScalar = builder.precedingScalar;
    if (precedingScalar != null) {
      precedingScalar.incRefCount();
    }
    this.followingScalar = builder.followingScalar;
    if (followingScalar != null) {
      followingScalar.incRefCount();
    }
    this.precedingCol = builder.precedingCol;
    if (precedingCol != null) {
      precedingCol.incRefCount();
    }
    this.followingCol = builder.followingCol;
    if (followingCol != null) {
      followingCol.incRefCount();
    }
    this.orderByColumnIndices = builder.normalizedOrderByColumnIndices();
    this.orderByAscendingFlags = builder.normalizedOrderByAscendingFlags();
    this.orderByNullsFirstFlags = builder.normalizedOrderByNullsFirstFlags();
    this.frameType = orderByColumnIndices != null ? FrameType.RANGE : FrameType.ROWS;
    this.precedingBoundsExtent = builder.precedingBoundsExtent;
    this.followingBoundsExtent = builder.followingBoundsExtent;
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    } else if (other instanceof WindowOptions) {
      WindowOptions o = (WindowOptions) other;
      boolean ret = this.minPeriods == o.minPeriods &&
              Arrays.equals(this.orderByColumnIndices, o.orderByColumnIndices) &&
              Arrays.equals(this.orderByAscendingFlags, o.orderByAscendingFlags) &&
              Arrays.equals(this.orderByNullsFirstFlags, o.orderByNullsFirstFlags) &&
              this.frameType == o.frameType &&
              this.precedingBoundsExtent == o.precedingBoundsExtent &&
              this.followingBoundsExtent == o.followingBoundsExtent;
      if (precedingCol != null) {
        ret = ret && precedingCol.equals(o.precedingCol);
      }
      if (followingCol != null) {
        ret = ret && followingCol.equals(o.followingCol);
      }
      if (precedingScalar != null) {
        ret = ret && precedingScalar.equals(o.precedingScalar);
      }
      if (followingScalar != null) {
        ret = ret && followingScalar.equals(o.followingScalar);
      }
      return ret;
    }
    return false;
  }

  @Override
  public int hashCode() {
    int ret = 7;
    ret = 31 * ret + minPeriods;
    ret = 31 * ret + Arrays.hashCode(orderByColumnIndices);
    ret = 31 * ret + Arrays.hashCode(orderByAscendingFlags);
    ret = 31 * ret + Arrays.hashCode(orderByNullsFirstFlags);
    ret = 31 * ret + frameType.hashCode();
    if (precedingCol != null) {
      ret = 31 * ret + precedingCol.hashCode();
    }
    if (followingCol != null) {
      ret = 31 * ret + followingCol.hashCode();
    }
    if (precedingScalar != null) {
      ret = 31 * ret + precedingScalar.hashCode();
    }
    if (followingScalar != null) {
      ret = 31 * ret + followingScalar.hashCode();
    }
    ret = 31 * ret + precedingBoundsExtent.hashCode();
    ret = 31 * ret + followingBoundsExtent.hashCode();
    return ret;
  }

  public static Builder builder(){
    return new Builder();
  }

  int getMinPeriods() { return  this.minPeriods; }

  Scalar getPrecedingScalar() { return this.precedingScalar; }

  Scalar getFollowingScalar() { return this.followingScalar; }

  ColumnVector getPrecedingCol() { return precedingCol; }

  ColumnVector getFollowingCol() { return this.followingCol; }

  @Deprecated
  int getTimestampColumnIndex() { return getOrderByColumnIndex(); }

  int getOrderByColumnIndex() {
    return orderByColumnIndices != null ? orderByColumnIndices[0] : -1;
  }

  @Deprecated
  boolean isTimestampOrderAscending() { return isOrderByAscending(); }

  boolean isOrderByAscending() {
    return orderByAscendingFlags != null ? orderByAscendingFlags[0] : true;
  }

  /**
   * Order-by column indices for this RANGE window. The single-column builder setters normalize
   * to a length-1 array, so this is non-null and non-empty for a RANGE window.
   * @return a copy of the order-by column indices, or null when no order-by was set (ROWS frame).
   */
  int[] getOrderByColumnIndices() {
    return orderByColumnIndices != null ? Arrays.copyOf(orderByColumnIndices, orderByColumnIndices.length) : null;
  }

  /**
   * Per-order-by-column ascending flags, parallel to {@link #getOrderByColumnIndices()}.
   * @return a copy of the ascending flags, or null when no order-by was set (ROWS frame).
   */
  boolean[] getOrderByAscending() {
    return orderByAscendingFlags != null ? Arrays.copyOf(orderByAscendingFlags, orderByAscendingFlags.length) : null;
  }

  /**
   * Per-order-by-column null-placement flags (true == nulls first), parallel to
   * {@link #getOrderByColumnIndices()}. Only consumed for multi-column RANGE windows; the
   * single-column path deduces null placement natively, so the returned value is unused there.
   * @return a copy of the null-placement flags, or null when no order-by was set (ROWS frame).
   */
  boolean[] getOrderByNullsFirst() {
    return orderByNullsFirstFlags != null ? Arrays.copyOf(orderByNullsFirstFlags, orderByNullsFirstFlags.length) : null;
  }

  boolean isUnboundedPreceding() { return this.precedingBoundsExtent == RangeExtentType.UNBOUNDED; }

  boolean isUnboundedFollowing() { return this.followingBoundsExtent == RangeExtentType.UNBOUNDED; }

  boolean isCurrentRowPreceding() { return this.precedingBoundsExtent == RangeExtentType.CURRENT_ROW; }

  boolean isCurrentRowFollowing() { return this.followingBoundsExtent == RangeExtentType.CURRENT_ROW; }

  RangeExtentType getPrecedingBoundsExtent() { return this.precedingBoundsExtent; }
  RangeExtentType getFollowingBoundsExtent() { return this.followingBoundsExtent; }

  FrameType getFrameType() { return frameType; }

  public static class Builder {
    private int minPeriods = 1;
    // for range window
    private Scalar precedingScalar = null;
    private Scalar followingScalar = null;
    private ColumnVector precedingCol = null;
    private ColumnVector followingCol = null;
    // Single-column order-by accumulators. orderByColumnIndex defaults to -1, but -1 alone does
    // NOT mark the order-by as set; singleColumnOrderBySet (flipped only by the index setter) is
    // the explicit RANGE discriminator, replacing the old "index == -1" sentinel. A bare
    // direction call without an index keeps the ROWS frame, matching the prior behavior.
    private int orderByColumnIndex = -1;
    private boolean orderByOrderAscending = true;
    private boolean singleColumnOrderBySet = false;
    // Tracks whether ANY single-column order-by setter (index or direction) was used, so build()
    // can reject mixing the single-column and multi-column APIs.
    private boolean singleColumnSetterUsed = false;
    // Multi-column order-by accumulators, populated only by orderByColumns().
    private int[] orderByColumnIndices = null;
    private boolean[] orderByAscendingFlags = null;
    private boolean[] orderByNullsFirstFlags = null;
    private RangeExtentType precedingBoundsExtent = RangeExtentType.BOUNDED;
    private RangeExtentType followingBoundsExtent = RangeExtentType.BOUNDED;

    /**
     * Set the minimum number of observation required to evaluate an element.  If there are not
     * enough elements for a given window a null is placed in the result instead.
     */
    public Builder minPeriods(int minPeriods) {
      if (minPeriods < 0 ) {
        throw  new IllegalArgumentException("Minimum observations must be non negative");
      }
      this.minPeriods = minPeriods;
      return this;
    }

    /**
     * Set the size of the window, one entry per row. This does not take ownership of the
     * columns passed in so you have to be sure that the lifetime of the column outlives
     * this operation.
     * @param precedingCol the number of rows in the window before and including the current row.
     *                     precedingCol will be live outside of WindowOptions.
     * @param followingCol the number of rows in the window after the current row.
     *                     followingCol will be live outside of WindowOptions.
     */
    public Builder window(ColumnVector precedingCol, ColumnVector followingCol) {
      if (precedingCol == null || precedingCol.hasNulls()) {
        throw new IllegalArgumentException("preceding cannot be null or have nulls");
      }
      if (followingCol == null || followingCol.hasNulls()) {
        throw new IllegalArgumentException("following cannot be null or have nulls");
      }
      if (precedingBoundsExtent != RangeExtentType.BOUNDED || precedingScalar != null) {
        throw new IllegalStateException("preceding has already been set a different way");
      }
      if (followingBoundsExtent != RangeExtentType.BOUNDED || followingScalar != null) {
        throw new IllegalStateException("following has already been set a different way");
      }
      this.precedingCol = precedingCol;
      this.followingCol = followingCol;
      return this;
    }

    /**
     * Set the size of the range window. This does not take ownership of the
     * scalars passed in so you have to be sure that the lifetime of the scalars outlives
     * this operation.
     * @param precedingScalar the number of rows in the window before and including the current row.
     *                        precedingScalar will be live outside of WindowOptions.
     * @param followingScalar the number of rows in the window after the current row.
     *                        followingScalar will be live outside of WindowOptions.
     */
    public Builder window(Scalar precedingScalar, Scalar followingScalar) {
      return preceding(precedingScalar).following(followingScalar);
    }

    /**
     * @deprecated Use orderByColumnIndex(int index)
     */
    @Deprecated
    public Builder timestampColumnIndex(int index) {
      return orderByColumnIndex(index);
    }

    public Builder orderByColumnIndex(int index) {
      this.orderByColumnIndex = index;
      this.singleColumnOrderBySet = true;
      this.singleColumnSetterUsed = true;
      return this;
    }

    /**
     * Specify multiple order-by columns for a multi-column RANGE window. All three arrays must be
     * non-empty and of equal length; entry {@code i} describes the i-th order-by column.
     *
     * <p>Multi-column RANGE windows support only peer-frame bounds ({@code UNBOUNDED} and
     * {@code CURRENT_ROW}); bounded scalar ranges are not supported across multiple order-by
     * columns. Unlike the single-column order-by methods, null placement is not deduced from the
     * data and must be stated explicitly here. This API is mutually exclusive with the
     * single-column order-by setters; mixing them in one builder is rejected by {@link #build()}.
     *
     * @param indices    input-table column indices of the order-by columns, in order.
     * @param ascending  per-column sort direction (true == ascending).
     * @param nullsFirst per-column null placement (true == nulls ordered before non-null values).
     */
    public Builder orderByColumns(int[] indices, boolean[] ascending, boolean[] nullsFirst) {
      if (indices == null || ascending == null || nullsFirst == null) {
        throw new IllegalArgumentException("order-by column arrays cannot be null");
      }
      if (indices.length == 0) {
        throw new IllegalArgumentException("at least one order-by column is required");
      }
      if (indices.length != ascending.length || indices.length != nullsFirst.length) {
        throw new IllegalArgumentException(
            "order-by index, ascending, and nullsFirst arrays must have the same length");
      }
      this.orderByColumnIndices = Arrays.copyOf(indices, indices.length);
      this.orderByAscendingFlags = Arrays.copyOf(ascending, ascending.length);
      this.orderByNullsFirstFlags = Arrays.copyOf(nullsFirst, nullsFirst.length);
      return this;
    }

    /**
     * @deprecated Use orderByAscending()
     */
    @Deprecated
    public Builder timestampAscending() {
      return orderByAscending();
    }

    public Builder orderByAscending() {
      this.orderByOrderAscending = true;
      this.singleColumnSetterUsed = true;
      return this;
    }

    public Builder orderByDescending() {
      this.orderByOrderAscending = false;
      this.singleColumnSetterUsed = true;
      return this;
    }

    /**
     * @deprecated Use orderByDescending()
     */
    @Deprecated
    public Builder timestampDescending() {
      return orderByDescending();
    }

    public Builder currentRowPreceding() {
      if (precedingCol != null || precedingScalar != null) {
        throw new IllegalStateException("preceding has already been set a different way");
      }
      this.precedingBoundsExtent = RangeExtentType.CURRENT_ROW;
      return this;
    }

    public Builder currentRowFollowing() {
      if (followingCol != null || followingScalar != null) {
        throw new IllegalStateException("following has already been set a different way");
      }
      this.followingBoundsExtent = RangeExtentType.CURRENT_ROW;
      return this;
    }

    public Builder unboundedPreceding() {
      if (precedingCol != null || precedingScalar != null) {
        throw new IllegalStateException("preceding has already been set a different way");
      }
      this.precedingBoundsExtent = RangeExtentType.UNBOUNDED;
      return this;
    }

    public Builder unboundedFollowing() {
      if (followingCol != null || followingScalar != null) {
        throw new IllegalStateException("following has already been set a different way");
      }
      this.followingBoundsExtent = RangeExtentType.UNBOUNDED;
      return this;
    }

    /**
     * Set the relative number preceding the current row for range window
     * @return this for chaining
     */
    public Builder preceding(Scalar preceding) {
      if (preceding == null || !preceding.isValid()) {
        throw new IllegalArgumentException("preceding cannot be null");
      }
      if (precedingBoundsExtent != RangeExtentType.BOUNDED || precedingCol != null) {
        throw new IllegalStateException("preceding has already been set a different way");
      }
      this.precedingScalar = preceding;
      return this;
    }

    /**
     * Set the relative number following the current row for range window
     * @return this for chaining
     */
    public Builder following(Scalar following) {
      if (following == null || !following.isValid()) {
        throw new IllegalArgumentException("following cannot be null");
      }
      if (followingBoundsExtent != RangeExtentType.BOUNDED || followingCol != null) {
        throw new IllegalStateException("following has already been set a different way");
      }
      this.followingScalar = following;
      return this;
    }

    private boolean usesMultiColumnOrderBy() {
      return orderByColumnIndices != null;
    }

    /**
     * Normalize the order-by columns into a single internal representation: the three parallel
     * arrays. Returns the multi-column arrays when {@code orderByColumns()} was used, length-1
     * arrays when a single-column index was set, or null when no order-by was set (ROWS frame).
     */
    private int[] normalizedOrderByColumnIndices() {
      if (usesMultiColumnOrderBy()) {
        return orderByColumnIndices;
      }
      return singleColumnOrderBySet ? new int[]{orderByColumnIndex} : null;
    }

    private boolean[] normalizedOrderByAscendingFlags() {
      if (usesMultiColumnOrderBy()) {
        return orderByAscendingFlags;
      }
      return singleColumnOrderBySet ? new boolean[]{orderByOrderAscending} : null;
    }

    private boolean[] normalizedOrderByNullsFirstFlags() {
      if (usesMultiColumnOrderBy()) {
        return orderByNullsFirstFlags;
      }
      // The single-column path deduces null placement natively, so this value is unused; a
      // length-1 default keeps the three arrays parallel.
      return singleColumnOrderBySet ? new boolean[]{true} : null;
    }

    public WindowOptions build() {
      if (usesMultiColumnOrderBy() && singleColumnSetterUsed) {
        throw new IllegalStateException(
            "Cannot mix orderByColumns(...) with the single-column order-by setters " +
            "(orderByColumnIndex/orderByAscending/orderByDescending); use one API or the other");
      }
      return new WindowOptions(this);
    }
  }

  public synchronized WindowOptions incRefCount() {
    if (precedingScalar != null) {
      precedingScalar.incRefCount();
    }
    if (followingScalar != null) {
      followingScalar.incRefCount();
    }
    if (precedingCol != null) {
      precedingCol.incRefCount();
    }
    if (followingCol != null) {
      followingCol.incRefCount();
    }
    return this;
  }

  @Override
  public void close() {
    if (precedingScalar != null) {
      precedingScalar.close();
    }
    if (followingScalar != null) {
      followingScalar.close();
    }
    if (precedingCol != null) {
      precedingCol.close();
    }
    if (followingCol != null) {
      followingCol.close();
    }
  }
}

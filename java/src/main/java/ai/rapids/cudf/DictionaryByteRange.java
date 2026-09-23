/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.util.Objects;

/**
 * Byte range of a column chunk's dictionary page, and how closely that range describes the page.
 *
 * <p>Mirrors {@code cudf::io::parquet::experimental::dictionary_byte_range}.
 *
 * <p>The APIs in this file are experimental and subject to change.
 */
@Experimental
public final class DictionaryByteRange {
  /** How closely a range describes the dictionary page it points at. */
  public enum Extent {
    /** The range is exactly the dictionary page. */
    EXACT,
    /**
     * The range begins at the dictionary page if the column chunk has one, and ends no earlier
     * than that page does. A writer is allowed to leave out where the page ends, and to say that a
     * chunk is dictionary encoded when it holds no dictionary page at all, so a range of this kind
     * is a bound on a page that may not be there.
     */
    UPPER_BOUND_IF_PRESENT
  }

  private final ByteRange byteRange;
  private final Extent extent;

  /**
   * @param byteRange byte range to read from the file
   * @param extent    how closely {@code byteRange} describes the dictionary page
   */
  public DictionaryByteRange(ByteRange byteRange, Extent extent) {
    this.byteRange = Objects.requireNonNull(byteRange, "byteRange must not be null");
    this.extent = Objects.requireNonNull(extent, "extent must not be null");
  }

  /** @return the byte range this dictionary page lies in. */
  public ByteRange byteRange() {
    return byteRange;
  }

  /** @return how closely {@link #byteRange()} describes the dictionary page. */
  public Extent extent() {
    return extent;
  }

  /**
   * The byte range to read, reading no more than {@code maxUpperBoundSize} bytes of a range that
   * only bounds its dictionary page. That caps what a caller spends looking for a page that may not
   * be there. What is read of such a range still has to be cut down to the dictionary page before
   * it is handed to the reader, which wants a buffer holding exactly one page; see
   * {@link HybridScanReader#dictionaryPageLengths}.
   *
   * @param maxUpperBoundSize most bytes to read of a range that only bounds its dictionary page
   * @return the byte range to read
   */
  public ByteRange byteRangeToRead(long maxUpperBoundSize) {
    if (maxUpperBoundSize < 0) {
      throw new IllegalArgumentException("maxUpperBoundSize must be >= 0, got " + maxUpperBoundSize);
    }
    if (extent == Extent.EXACT) {
      return byteRange;
    }
    return new ByteRange(byteRange.offset(), Math.min(byteRange.size(), maxUpperBoundSize));
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof DictionaryByteRange)) return false;
    DictionaryByteRange other = (DictionaryByteRange) o;
    return byteRange.equals(other.byteRange) && extent == other.extent;
  }

  @Override
  public int hashCode() {
    return Objects.hash(byteRange, extent);
  }

  @Override
  public String toString() {
    return "DictionaryByteRange{" + byteRange + ", extent=" + extent + "}";
  }
}

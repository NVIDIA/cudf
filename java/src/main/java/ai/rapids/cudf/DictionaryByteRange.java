/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.util.Objects;

/**
 * Byte range of a column chunk's dictionary page, and whether that range is exactly the page.
 *
 * <p>A range that is not exact starts at the chunk's dictionary page, if it has one, and may
 * extend past it.
 *
 * <p>Mirrors {@code cudf::io::parquet::experimental::dictionary_byte_range}.
 *
 * <p>The APIs in this file are experimental and subject to change.
 */
@Experimental
public final class DictionaryByteRange {
  private final ByteRange byteRange;
  private final boolean isExact;

  /**
   * @param byteRange byte range to read from the file
   * @param isExact   whether {@code byteRange} is exactly the dictionary page, not an upper bound
   */
  public DictionaryByteRange(ByteRange byteRange, boolean isExact) {
    this.byteRange = Objects.requireNonNull(byteRange, "byteRange must not be null");
    this.isExact = isExact;
  }

  /** @return the byte range this dictionary page lies in. */
  public ByteRange byteRange() {
    return byteRange;
  }

  /** @return whether {@link #byteRange()} is exactly the dictionary page, not an upper bound. */
  public boolean isExact() {
    return isExact;
  }

  /**
   * The byte range to read, reading no more than {@code maxUpperBoundSize} bytes of a range that
   * is not exact. What is read of such a range must still be trimmed to the dictionary page before
   * it is handed to the reader; see {@link HybridScanReader#dictionaryPageLengths}.
   *
   * @param maxUpperBoundSize most bytes to read of a range that is not exact
   * @return the byte range to read
   */
  public ByteRange byteRangeToRead(long maxUpperBoundSize) {
    if (maxUpperBoundSize < 0) {
      throw new IllegalArgumentException("maxUpperBoundSize must be >= 0, got " + maxUpperBoundSize);
    }
    if (isExact) {
      return byteRange;
    }
    return new ByteRange(byteRange.offset(), Math.min(byteRange.size(), maxUpperBoundSize));
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof DictionaryByteRange)) return false;
    DictionaryByteRange other = (DictionaryByteRange) o;
    return byteRange.equals(other.byteRange) && isExact == other.isExact;
  }

  @Override
  public int hashCode() {
    return Objects.hash(byteRange, isExact);
  }

  @Override
  public String toString() {
    return "DictionaryByteRange{" + byteRange + ", isExact=" + isExact + "}";
  }
}

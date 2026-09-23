/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.util.Objects;

/**
 * Immutable byte range describing an offset and size within a file or buffer.
 *
 * <p>Mirrors {@code cudf::io::text::byte_range_info}.
 *
 * <p>The APIs in this file are experimental and subject to change.
 */
@Experimental
public final class ByteRange {
  private final long offset;
  private final long size;

  /**
   * @param offset starting offset, in bytes, from the beginning of the source
   * @param size   length of the range, in bytes
   */
  public ByteRange(long offset, long size) {
    if (offset < 0) {
      throw new IllegalArgumentException("offset must be >= 0, got " + offset);
    }
    if (size < 0) {
      throw new IllegalArgumentException("size must be >= 0, got " + size);
    }
    this.offset = offset;
    this.size = size;
  }

  /** @return starting byte offset within the source. */
  public long offset() {
    return offset;
  }

  /** @return length of the byte range, in bytes. */
  public long size() {
    return size;
  }

  /** @return {@code true} when the range has zero size. */
  public boolean isEmpty() {
    return size == 0;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof ByteRange)) return false;
    ByteRange other = (ByteRange) o;
    return offset == other.offset && size == other.size;
  }

  @Override
  public int hashCode() {
    return Objects.hash(offset, size);
  }

  @Override
  public String toString() {
    return "ByteRange{offset=" + offset + ", size=" + size + "}";
  }
}

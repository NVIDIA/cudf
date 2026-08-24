/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.util.Arrays;
import java.util.Objects;

/**
 * Pair of byte-range arrays returned by
 * {@link HybridScanReader#secondaryFiltersByteRanges(int[])}.
 *
 * <p>The two arrays describe, for the input row groups:
 * <ul>
 *   <li>bloom-filter ranges — file byte ranges containing per-column-chunk Parquet bloom
 *       filter blobs (no getter yet; see constructor parameter TODO);</li>
 *   <li>{@link #dictionaryPageRanges()} — file byte ranges of the column-chunk
 *       dictionary pages used for row-group pruning of (in)equality predicates.</li>
 * </ul>
 *
 * <p>Both arrays may be empty. The ordering follows the C++ reader's ordering and is
 * meaningful: the i-th entry corresponds to the i-th column-chunk needing the
 * respective filter.
 *
 * <p>Mirrors the {@code std::pair<std::vector<byte_range_info>, std::vector<byte_range_info>>}
 * returned by {@code hybrid_scan_reader::secondary_filters_byte_ranges}.
 *
 * <p>The APIs in this file are experimental and subject to change.
 */
@Experimental
public final class SecondaryFilterRanges {
  private final ByteRange[] bloomFilterRanges;
  private final ByteRange[] dictionaryPageRanges;

  /**
   * @param bloomFilterRanges   bloom-filter byte ranges (stored but not yet exposed via a getter;
   *                            TODO: add {@code bloomFilterRanges()} once bloom filter writing is
   *                            supported in {@link ParquetWriterOptions})
   * @param dictionaryPageRanges dictionary-page byte ranges
   */
  public SecondaryFilterRanges(ByteRange[] bloomFilterRanges,
                               ByteRange[] dictionaryPageRanges) {
    this.bloomFilterRanges =
        bloomFilterRanges == null ? new ByteRange[0] : bloomFilterRanges.clone();
    this.dictionaryPageRanges =
        dictionaryPageRanges == null ? new ByteRange[0] : dictionaryPageRanges.clone();
  }

  /** @return a defensive copy of the dictionary-page byte ranges. */
  public ByteRange[] dictionaryPageRanges() {
    return dictionaryPageRanges.clone();
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (!(o instanceof SecondaryFilterRanges)) return false;
    SecondaryFilterRanges other = (SecondaryFilterRanges) o;
    return Arrays.equals(bloomFilterRanges, other.bloomFilterRanges) &&
           Arrays.equals(dictionaryPageRanges, other.dictionaryPageRanges);
  }

  @Override
  public int hashCode() {
    return Objects.hash(Arrays.hashCode(bloomFilterRanges),
                        Arrays.hashCode(dictionaryPageRanges));
  }

  @Override
  public String toString() {
    return "SecondaryFilterRanges{bloom=" + bloomFilterRanges.length +
           " ranges, dict=" + dictionaryPageRanges.length + " ranges}";
  }
}

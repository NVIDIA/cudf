/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.params.provider.Arguments.arguments;

import ai.rapids.cudf.ast.BinaryOperation;
import ai.rapids.cudf.ast.BinaryOperator;
import ai.rapids.cudf.ast.ColumnNameReference;
import ai.rapids.cudf.ast.CompiledExpression;
import ai.rapids.cudf.ast.Literal;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.Consumer;
import java.util.stream.IntStream;
import java.util.stream.Stream;

/**
 * Tests for {@link HybridScanReader}.
 */
public class HybridScanReaderTest extends CudfTestBase {

  private static final String[] DEFAULT_COLS = {"id", "zip_code", "num_units"};
  private static final int[] ALL_ROW_GROUPS = {0, 1, 2};

  // --------------------------------------------------------------------
  // Tests: HybridScanReader (constructor)
  // --------------------------------------------------------------------

  /** Verifies that passing a null footer buffer to the constructor throws IllegalArgumentException. */
  @Test
  void testNullFooterThrows() {
    assertThrows(IllegalArgumentException.class,
        () -> new HybridScanReader(null, optsForColumns("id")));
  }

  // --------------------------------------------------------------------
  // Tests: setFilter()
  // --------------------------------------------------------------------

  /**
   * Verifies setFilter() invalidates the C++ reader's cached column-selection state:
   * calling filterRowGroupsWithStats() after replacing filterA with filterB yields the
   * same result as a freshly-constructed reader with filterB. If reset_column_selection
   * did not fire, the second call would silently reuse filterA's cached filter columns.
   */
  @Test
  void testSetFilterInvalidatesCachedColumnSelection(@TempDir Path tmp) throws IOException {
    // filterA prunes group 0 (max zip 109,900); filterB prunes groups 0 and 1 (max zip 209,900).
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 150000)) {
      int[] afterA = open.reader.filterRowGroupsWithStats(open.reader.allRowGroups());
      assertArrayEquals(new int[]{1, 2}, afterA, "filterA prunes group 0 only");
      open.withFilter("zip_code", BinaryOperator.GREATER, 250000);
      int[] afterB = open.reader.filterRowGroupsWithStats(open.reader.allRowGroups());
      assertArrayEquals(new int[]{2}, afterB,
          "filterB must produce fresh pruning; stale cached filter-columns would return afterA");
    }
  }

  /**
   * Verifies setFilter(null) clears the filter: after clearing, filter-based APIs behave
   * as though no filter was ever installed. The most direct signal is that the C++
   * reader's "Empty input filter expression" precondition fires on
   * filterRowGroupsWithStats(), which happily succeeded (returning a subset) with the
   * filter installed. If setFilter(null) had left the filter in place, the second call
   * would silently return the same pruned subset instead of throwing.
   */
  @Test
  void testSetFilterNullClearsFilter(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 150000)) {
      assertArrayEquals(new int[]{1, 2},
          open.reader.filterRowGroupsWithStats(open.reader.allRowGroups()),
          "sanity: filter prunes group 0");
      open.reader.setFilter(null);
      open.filter.close();
      open.filter = null;
      assertThrows(CudfException.class,
          () -> open.reader.filterRowGroupsWithStats(open.reader.allRowGroups()),
          "after setFilter(null), filter-based APIs must reject with 'Empty input filter expression'");
    }
  }

  // --------------------------------------------------------------------
  // Tests: pageIndexByteRange()
  // --------------------------------------------------------------------

  /**
   * Verifies pageIndexByteRange() returns a structurally valid range for a COLUMN-stats file:
   * non-zero size, positive offset, and ending exactly at the Parquet footer boundary
   * (per the spec: the page index region is contiguous and immediately precedes the footer).
   */
  @Test
  void testPageIndexByteRangeContiguousAndBeforeFooter(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.pageIndex(tmp)) {
      long fileLength = open.file.getLength();
      int footerLength = open.file.getInt(fileLength - 8);
      long footerStart = fileLength - 8 - footerLength;

      ByteRange piRange = open.reader.pageIndexByteRange();
      assertTrue(piRange.size() > 0,
          "COLUMN-stats file must contain a non-empty page index");
      assertTrue(piRange.offset() > 0,
          "Page index region must start after byte 0 (which holds the PAR1 magic)");
      assertEquals(footerStart, piRange.offset() + piRange.size(),
          "Page index region must end exactly at the Parquet footer boundary");
    }
  }

  /**
   * Verifies pageIndexByteRange() returns an empty range for a file written with ROWGROUP
   * statistics: such a file has no ColumnIndex/OffsetIndex structs and therefore no page
   * index region.
   */
  @Test
  void testPageIndexByteRangeEmptyForRowGroupStats(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.rowGroupStats(tmp)) {
      ByteRange piRange = open.reader.pageIndexByteRange();
      assertEquals(0L, piRange.size(),
          "A ROWGROUP-stats file has no page index region");
    }
  }

  // --------------------------------------------------------------------
  // Tests: setupPageIndex()
  // --------------------------------------------------------------------

  /**
   * Verifies setupPageIndex() correctly populates the page-index metadata: feed it the
   * bytes returned by pageIndexByteRange(), then assert PAGE_INDEX_STATS produces the
   * exact row count expected from the fixture (group 2 alone, 5,000 rows). All 5,000
   * rows in group 2 satisfy the filter zip_code > 99,999 (zip_code 100,000–104,999).
   */
  @Test
  void testSetupPageIndexPopulatesMetadata(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.pageIndex(tmp).withFilter("zip_code", BinaryOperator.GREATER, 99999)) {
      HybridScanReader reader = open.reader;
      ByteRange piRange = reader.pageIndexByteRange();
      try (HostMemoryBuffer pi = open.file.slice(piRange.offset(), piRange.size())) {
        reader.setupPageIndex(pi);
      }
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          open.file, reader.filterColumnChunksByteRanges(survived));
      try (HybridScanReader.FilterMaterializationResult fr =
               reader.materializeFilterColumns(survived, filterCols, true)) {
        assertEquals(5000L, fr.table().getRowCount(),
            "Group 2 (zip_code 100,000–104,999) entirely satisfies zip_code > 99,999");
      } finally {
        closeAll(filterCols);
      }
    }
  }

  /**
   * Verifies that materializeFilterColumns(..., usePageLevelPruning=true) throws when
   * invoked without a prior setupPageIndex() call: the page-index metadata must be
   * materialised before page-level row-mask construction can succeed.
   */
  @Test
  void testPageIndexStatsRequiresSetupPageIndex(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.pageIndex(tmp).withFilter("zip_code", BinaryOperator.GREATER, 99999)) {
      HybridScanReader reader = open.reader;
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          open.file, reader.filterColumnChunksByteRanges(survived));
      try {
        assertThrows(CudfException.class, () ->
            reader.materializeFilterColumns(survived, filterCols, true));
      } finally {
        closeAll(filterCols);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: allRowGroups()
  // --------------------------------------------------------------------

  /** Verifies allRowGroups() returns the exact contiguous indices {0, 1, 2} for a 3-group fixture. */
  @Test
  void testAllRowGroupsReturnsExactIndices(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      assertArrayEquals(ALL_ROW_GROUPS, open.reader.allRowGroups());
    }
  }

  // --------------------------------------------------------------------
  // Tests: totalRowsInRowGroups()
  // --------------------------------------------------------------------

  /** Verifies totalRowsInRowGroups() returns 3000 for all 3 groups (1000 rows × 3 groups). */
  @Test
  void testTotalRowsInRowGroupsAllGroups(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      assertEquals(3000L, open.reader.totalRowsInRowGroups(ALL_ROW_GROUPS));
    }
  }

  /** Verifies totalRowsInRowGroups() returns 1000 for a single row group. */
  @Test
  void testTotalRowsInRowGroupsSingleGroup(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      assertEquals(1000L, open.reader.totalRowsInRowGroups(new int[]{0}));
    }
  }

  /** Verifies totalRowsInRowGroups() returns 0 for an empty input array (spec-defined edge case). */
  @Test
  void testTotalRowsInRowGroupsEmpty(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      assertEquals(0L, open.reader.totalRowsInRowGroups(new int[0]));
    }
  }

  // --------------------------------------------------------------------
  // Tests: filterRowGroupsWithStats()
  // --------------------------------------------------------------------

  /**
   * Verifies filterRowGroupsWithStats() prunes the exact expected row groups: with filter
   * zip_code > 150,000, group 0 (max zip_code 109,900) is pruned and groups 1, 2 survive.
   */
  @Test
  void testFilterRowGroupsWithStatsExactSurvivors(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 150000)) {
      int[] survived = open.reader.filterRowGroupsWithStats(open.reader.allRowGroups());
      assertArrayEquals(new int[]{1, 2}, survived);
    }
  }

  // --------------------------------------------------------------------
  // Tests: secondaryFiltersByteRanges()
  // --------------------------------------------------------------------

  /**
   * Verifies secondaryFiltersByteRanges() returns one non-empty dictionary-page range per
   * row group when all three required conditions are met:
   * <ul>
   *   <li>The filter contains an (in)equality predicate (num_units == 2);
   *       dictionary_literals_collector only collects literals from EQUAL/NOT_EQUAL AST nodes.</li>
   *   <li>The fixture has a page index (COLUMN stats) and setupPageIndex() has been called.</li>
   *   <li>The writer's ADAPTIVE dictionary policy emits dictionaries for the filter column;
   *       num_units is low-cardinality in every row group, easily passing the
   *       plain_data_size > dict_enc_size heuristic in writer_impl.cu.</li>
   * </ul>
   * Result: 3 row groups × 1 dict-eligible filter col = 3 non-empty ranges.
   */
  @Test
  void testSecondaryFiltersByteRangesPresentForLowCardinality(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.pageIndex(tmp).withFilter("num_units", BinaryOperator.EQUAL, 2)) {
      open.withPageIndex();
      HybridScanReader reader = open.reader;
      SecondaryFilterRanges sfr = reader.secondaryFiltersByteRanges(reader.allRowGroups());
      ByteRange[] dict = sfr.dictionaryPageRanges();
      assertEquals(3, dict.length, "3 row groups × 1 dict-eligible filter column");
      for (ByteRange r : dict) {
        assertTrue(r.size() > 0, "Dictionary page range must be non-empty");
      }
    }
  }

  /**
   * Verifies secondaryFiltersByteRanges() returns no dictionary-page ranges for a
   * high-cardinality int filter column even when the conditions for dictionary lookup
   * are otherwise satisfied: the fixture has a page index (COLUMN stats) and the
   * predicate is EQUAL. The empty result must be attributable solely to the writer's
   * ADAPTIVE dictionary policy skipping dictionary emission for high-cardinality columns
   * (zip_code is unique per row).
   */
  @Test
  void testSecondaryFiltersByteRangesEmptyForHighCardinalityInts(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.pageIndex(tmp).withFilter("zip_code", BinaryOperator.EQUAL, 12345)) {
      open.withPageIndex();
      SecondaryFilterRanges sfr = open.reader.secondaryFiltersByteRanges(open.reader.allRowGroups());
      assertEquals(0, sfr.dictionaryPageRanges().length,
          "High-cardinality zip_code: ADAPTIVE policy skips dictionary emission "
              + "even with page index present and an EQUAL predicate requesting it");
    }
  }

  /**
   * Verifies secondaryFiltersByteRanges() finds dictionary pages even without page
   * index. Dictionary pruning relies on encoding statistics and dictionary-page metadata
   * rather than page indexes, so a low-cardinality column with an EQUAL predicate remains
   * eligible.
   */
  @Test
  void testSecondaryFiltersByteRangesForRowGroupStats(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.rowGroupStats(tmp).withFilter("num_units", BinaryOperator.EQUAL, 2)) {
      SecondaryFilterRanges sfr = open.reader.secondaryFiltersByteRanges(new int[]{0});
      assertEquals(1, sfr.dictionaryPageRanges().length,
          "A row group has only one dictionary-page per (filter) column");
      assertTrue(sfr.dictionaryPageRanges()[0].size() > 0,
          "Dictionary page range must be non-empty");
    }
  }

  // --------------------------------------------------------------------
  // Tests: filterRowGroupsWithDictionaryPages()
  // --------------------------------------------------------------------

  /**
   * Verifies filterRowGroupsWithDictionaryPages() prunes all row groups when the equality
   * literal is not present in any group's dictionary. With the pageIndex fixture, num_units
   * dictionaries are {1,2} (g0), {2,3} (g1), {3,4} (g2); filtering on num_units == 5 must
   * yield an empty surviving-group array.
   * <p>setupPageIndex() must be called before secondaryFiltersByteRanges() so that
   * has_page_index_and_only_dict_encoded_pages is true and dictionary page ranges are emitted.</p>
   */
  @Test
  void testFilterRowGroupsWithDictionaryPagesPrunesAllGroups(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.pageIndex(tmp).withFilter("num_units", BinaryOperator.EQUAL, 5)) {
      open.withPageIndex();
      HybridScanReader reader = open.reader;
      int[] rgs = reader.allRowGroups();
      SecondaryFilterRanges sfr = reader.secondaryFiltersByteRanges(rgs);
      DeviceMemoryBuffer[] dictBufs = copyRangesToDevice(open.file, sfr.dictionaryPageRanges());
      try {
        int[] result = reader.filterRowGroupsWithDictionaryPages(dictBufs, rgs);
        assertEquals(0, result.length,
            "num_units == 5 is not in any group's dictionary ({1,2}, {2,3}, {3,4}); all pruned");
      } finally {
        closeAll(dictBufs);
      }
    }
  }

  /**
   * Verifies filterRowGroupsWithDictionaryPages() returns a strict subset of input row
   * groups when an EQUAL literal is present in some but not all groups' dictionaries. The
   * pageIndex fixture's num_units dictionaries are {1,2} (g0), {2,3} (g1), {3,4} (g2); all
   * three column chunks are dictionary-encoded under COLUMN stats. With filter
   * num_units == 2, the literal is in g0 and g1 but not g2, so the result is exactly {0, 1}.
   */
  @Test
  void testFilterRowGroupsWithDictionaryPagesPrunesSubsetOfGroups(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.pageIndex(tmp).withFilter("num_units", BinaryOperator.EQUAL, 2)) {
      open.withPageIndex();
      HybridScanReader reader = open.reader;
      int[] rgs = reader.allRowGroups();
      SecondaryFilterRanges sfr = reader.secondaryFiltersByteRanges(rgs);
      DeviceMemoryBuffer[] dictBufs = copyRangesToDevice(open.file, sfr.dictionaryPageRanges());
      try {
        int[] result = reader.filterRowGroupsWithDictionaryPages(dictBufs, rgs);
        assertArrayEquals(new int[]{0, 1}, result,
            "num_units == 2 ∈ g0 dict {1,2} and g1 dict {2,3}, ∉ g2 dict {3,4}");
      } finally {
        closeAll(dictBufs);
      }
    }
  }

  /**
   * Verifies filterRowGroupsWithDictionaryPages() throws when the upstream
   * secondaryFiltersByteRanges yields no dict ranges due to ADAPTIVE policy skipping dict
   * emission on a high-cardinality column. With EQUAL on zip_code (5,000 distinct values
   * per group), no dict pages exist, so no buffers can be supplied. The C++ CUDF_EXPECTS
   * at prepare_dictionaries enforces buffers.size() == row_groups × dict-eligible cols.
   */
  @Test
  void testFilterRowGroupsWithDictionaryPagesThrowsForHighCardinality(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.pageIndex(tmp).withFilter("zip_code", BinaryOperator.EQUAL, 12345)) {
      open.withPageIndex();
      int[] rgs = open.reader.allRowGroups();
      SecondaryFilterRanges sfr = open.reader.secondaryFiltersByteRanges(rgs);
      DeviceMemoryBuffer[] dictBufs = copyRangesToDevice(open.file, sfr.dictionaryPageRanges());
      try {
        assertEquals(0, dictBufs.length, "ADAPTIVE skips dict for high-cardinality zip_code");
        assertThrows(CudfException.class,
            () -> open.reader.filterRowGroupsWithDictionaryPages(dictBufs, rgs),
            "CUDF_EXPECTS at prepare_dictionaries: 0 buffers != 3 row groups × 1 dict-eligible col");
      } finally {
        closeAll(dictBufs);
      }
    }
  }

  /**
   * Verifies dictionary-page pruning works without a page index. With a ROWGROUP-stats
   * fixture and {@code num_units == 2}, the sole row group's dictionary contains the
   *  literal, so it survives.
   */
  @Test
  void testFilterRowGroupsWithDictionaryPagesWithoutPageIndex(@TempDir Path tmp)
      throws IOException {
    try (OpenReader open = OpenReader.rowGroupStats(tmp).withFilter("num_units", BinaryOperator.EQUAL, 2)) {
      int[] rgs = new int[]{0};
      SecondaryFilterRanges sfr = open.reader.secondaryFiltersByteRanges(rgs);
      DeviceMemoryBuffer[] dictBufs = copyRangesToDevice(open.file, sfr.dictionaryPageRanges());
      try {
        assertEquals(1, dictBufs.length,
            "The dictionary page is discoverable even without a page index");
        assertArrayEquals(rgs, open.reader.filterRowGroupsWithDictionaryPages(dictBufs, rgs),
            "num_units == 2 is present in the row group's dictionary");
      } finally {
        closeAll(dictBufs);
      }
    }
  }

  // TODO: add testFilterRowGroupsWithBloomFilters once ParquetWriterOptions exposes
  //       bloom filter writing (set_column_chunks_bloom_filter_params). See
  //       HybridScanReader.java for details.

  // --------------------------------------------------------------------
  // Tests: filterColumnChunksByteRanges()
  // --------------------------------------------------------------------

  /**
   * Verifies filterColumnChunksByteRanges() returns one range per surviving row group for
   * the single filter column ("zip_code"). With filter zip_code > 150,000, group 0 (max
   * zip 109,900) is pruned by filterRowGroupsWithStats and groups 1, 2 survive:
   * 1 filter column × 2 surviving groups = 2 non-empty ranges. The reduction from
   * allRowGroups (3) to survived (2) demonstrates that the method's output tracks its
   * row-group input.
   */
  @Test
  void testFilterColumnChunksByteRangesOnePerSurvivingGroup(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 150000)) {
      int[] survived = open.reader.filterRowGroupsWithStats(open.reader.allRowGroups());
      ByteRange[] ranges = open.reader.filterColumnChunksByteRanges(survived);
      assertEquals(2, ranges.length,
          "1 filter column × 2 surviving row groups (group 0 pruned by filterRowGroupsWithStats)");
      for (ByteRange r : ranges) {
        assertTrue(r.size() > 0, "Each filter column-chunk range must be non-empty");
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: payloadColumnChunksByteRanges()
  // --------------------------------------------------------------------

  /**
   * Verifies payloadColumnChunksByteRanges() returns one range per projected column per row
   * group when no filter is set: with 3 projected columns × 3 row groups, the result has
   * 9 ranges.
   */
  @Test
  void testPayloadColumnChunksByteRangesNoFilter(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      ByteRange[] ranges = open.reader.payloadColumnChunksByteRanges(open.reader.allRowGroups());
      assertEquals(9, ranges.length, "3 payload columns × 3 row groups");
      for (ByteRange r : ranges) {
        assertTrue(r.size() > 0);
      }
    }
  }

  /**
   * Verifies payloadColumnChunksByteRanges() returns ranges for all projected columns when
   * called BEFORE any filter-column operation has populated the reader's filter
   * column-name cache (_filter_columns_names in C++). In this pre-filter-pipeline state,
   * the C++ select_payload_columns receives an empty filter-column set and skips the
   * filter-column exclusion step. Result: 3 projected columns × 3 row groups = 9 ranges.
   * See {@link #testPayloadColumnChunksByteRangesAfterFilterColumnsCall} for the
   * post-pipeline contract (filter column excluded → 6 ranges).
   */
  @Test
  void testPayloadColumnChunksByteRangesWithFilter(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 50000)) {
      ByteRange[] ranges = open.reader.payloadColumnChunksByteRanges(open.reader.allRowGroups());
      assertEquals(9, ranges.length, "3 projected columns × 3 row groups (filter column not excluded)");
      for (ByteRange r : ranges) {
        assertTrue(r.size() > 0);
      }
    }
  }

  /**
   * Verifies payloadColumnChunksByteRanges() excludes the filter column when called AFTER
   * filterColumnChunksByteRanges has populated _filter_columns_names. This is the typical
   * pipeline order (filter columns resolved first, materialized, then payload columns
   * resolved for the surviving rows). With filter zip_code > 50,000, _filter_columns_names
   * = {"zip_code"} after the first call; payloadColumnChunksByteRanges then returns ranges
   * for {id, num_units} only: 2 cols × 3 row groups = 6 non-empty ranges.
   */
  @Test
  void testPayloadColumnChunksByteRangesAfterFilterColumnsCall(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 50000)) {
      int[] rgs = open.reader.allRowGroups();
      open.reader.filterColumnChunksByteRanges(rgs);
      ByteRange[] ranges = open.reader.payloadColumnChunksByteRanges(rgs);
      assertEquals(6, ranges.length,
          "After filterColumnChunksByteRanges populates _filter_columns_names, "
              + "the filter column zip_code is excluded: 2 cols × 3 row groups");
      for (ByteRange r : ranges) {
        assertTrue(r.size() > 0);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: allColumnChunksByteRanges()
  // --------------------------------------------------------------------

  /**
   * Verifies allColumnChunksByteRanges() includes all projected columns even after the
   * filter pipeline has populated _filter_columns_names. The C++ select_columns(ALL_COLUMNS)
   * never consults the filter-column cache, so the filter column zip_code is NOT excluded —
   * a deliberate contrast with payloadColumnChunksByteRanges, which DOES exclude in the
   * same state (see testPayloadColumnChunksByteRangesAfterFilterColumnsCall returning 6).
   * Result: 3 cols × 3 row groups = 9 non-empty ranges; the post-pipeline state implies
   * the no-filter state for this method.
   */
  @Test
  void testAllColumnChunksByteRangesExactCount(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 50000)) {
      HybridScanReader reader = open.reader;
      int[] rgs = reader.allRowGroups();
      reader.filterColumnChunksByteRanges(rgs);
      ByteRange[] ranges = reader.allColumnChunksByteRanges(rgs);
      assertEquals(9, ranges.length,
          "All 3 columns × 3 row groups; filter column NOT excluded (unlike payload).");
      for (ByteRange r : ranges) {
        assertTrue(r.size() > 0);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: materializeFilterColumns()
  // --------------------------------------------------------------------

  /**
   * Verifies materializeFilterColumns() produces a filter table with the exact expected row
   * count and column count: filter zip_code > 150,000 prunes group 0 (max 109,900) and
   * keeps 599 + 1000 = 1599 rows from groups 1+2; the filter table contains the single
   * filter column ("zip_code").
   */
  @Test
  void testMaterializeFilterColumnsExactRowCount(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 150000)) {
      HybridScanReader reader = open.reader;
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          open.file, reader.filterColumnChunksByteRanges(survived));
      try (HybridScanReader.FilterMaterializationResult fr =
               reader.materializeFilterColumns(survived, filterCols, false)) {
        assertEquals(1599L, fr.table().getRowCount());
        assertEquals(1, fr.table().getNumberOfColumns(), "filter table contains only zip_code");
      } finally {
        closeAll(filterCols);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: materializePayloadColumns()
  // --------------------------------------------------------------------

  /**
   * Verifies materializePayloadColumns() produces a payload table with the exact expected
   * row count and column count: 1599 rows survive the filter; the payload table contains
   * the two non-filter columns ("id", "num_units").
   */
  @Test
  void testMaterializePayloadColumnsExactRowCount(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 150000)) {
      HybridScanReader reader = open.reader;
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          open.file, reader.filterColumnChunksByteRanges(survived));
      DeviceMemoryBuffer[] payloadCols = copyRangesToDevice(
          open.file, reader.payloadColumnChunksByteRanges(survived));
      try (HybridScanReader.FilterMaterializationResult fr =
               reader.materializeFilterColumns(survived, filterCols, false);
           Table payload = reader.materializePayloadColumns(survived, payloadCols,
               fr.rowMask(), false)) {
        assertEquals(1599L, payload.getRowCount());
        assertEquals(2, payload.getNumberOfColumns(), "payload table contains id + num_units");
      } finally {
        closeAll(filterCols);
        closeAll(payloadCols);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: materializeAllColumns()
  // --------------------------------------------------------------------

  /**
   * Verifies materializeAllColumns() returns a table with all 3 projected columns and the
   * exact total row count (3,000) for the fixture.
   */
  @Test
  void testMaterializeAllColumnsExactRowCount(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      HybridScanReader reader = open.reader;
      int[] rgs = reader.allRowGroups();
      ByteRange[] ranges = reader.allColumnChunksByteRanges(rgs);
      DeviceMemoryBuffer[] devs = copyRangesToDevice(open.file, ranges);
      try (Table t = reader.materializeAllColumns(rgs, devs)) {
        assertEquals(3, t.getNumberOfColumns());
        assertEquals(3000L, t.getRowCount());
      } finally {
        closeAll(devs);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: setupChunkingForFilterColumns()
  // --------------------------------------------------------------------

  /**
   * Verifies setupChunkingForFilterColumns() activates the filter-column chunked pipeline:
   * hasNextTableChunk() reports true after setup. (Calling hasNextTableChunk() before any
   * chunking has been set up is invalid in the C++ contract — see the dedicated
   * hasNextTableChunk tests.)
   */
  @Test
  void testSetupChunkingForFilterColumnsActivatesPipeline(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 150000)) {
      HybridScanReader reader = open.reader;
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          open.file, reader.filterColumnChunksByteRanges(survived));
      try {
        reader.setupChunkingForFilterColumns(0L, 0L, survived, false, filterCols);
        assertTrue(reader.hasNextTableChunk(), "chunking active after setup");
      } finally {
        closeAll(filterCols);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: materializeFilterColumnsChunk()
  // --------------------------------------------------------------------

  /**
   * Verifies materializeFilterColumnsChunk() drains the exact expected row count when
   * chained with setupChunkingForFilterColumns(): 1599 rows (filter zip_code > 150,000
   * survives 599 + 1000 rows from groups 1+2). Also asserts the post-drain row mask:
   * setup establishes length 2000 (preserved by drain), and drain refines per-row
   * truth so the final true-count = 1599.
   */
  @Test
  void testMaterializeFilterColumnsChunkExactTotal(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 150000)) {
      HybridScanReader reader = open.reader;
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          open.file, reader.filterColumnChunksByteRanges(survived));
      try {
        reader.setupChunkingForFilterColumns(0L, 0L, survived, false, filterCols);
        long total = 0;
        while (reader.hasNextTableChunk()) {
          try (Table chunk = reader.materializeFilterColumnsChunk()) {
            total += chunk.getRowCount();
          }
        }
        assertEquals(1599L, total,
            "Filter chunks contain only the rows that survive the filter expression");
        try (ColumnVector rowMask = reader.takeFilterRowMask()) {
          assertEquals(2000L, rowMask.getRowCount(),
              "Drain does not change mask length");
          assertEquals(1599L, countTrue(rowMask),
              "After drain, true-count = surviving row count");
        }
      } finally {
        closeAll(filterCols);
      }
    }
  }

  /**
   * Verifies materializeFilterColumnsChunk() throws IllegalArgumentException when invoked
   * without an active chunked filter pipeline. The JNI layer guards this state and rejects
   * the call before reaching the C++ implementation.
   */
  @Test
  void testMaterializeFilterColumnsChunkBeforeSetupThrows(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      assertThrows(IllegalArgumentException.class, open.reader::materializeFilterColumnsChunk);
    }
  }

  // --------------------------------------------------------------------
  // Tests: takeFilterRowMask()
  // --------------------------------------------------------------------

  /**
   * Verifies takeFilterRowMask() exposes the all-true mask immediately after setup, before
   * any chunks have been materialized. The mask is fully constructed by build_all_true_row_mask
   * at setup time: length = total input rows in surviving row groups, all entries = true.
   * For survived = {1, 2}: length = 2 × 1000 = 2000, true-count = 2000.
   */
  @Test
  void testTakeFilterRowMaskAllTrueExposesPreallocatedMask(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 150000)) {
      HybridScanReader reader = open.reader;
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          open.file, reader.filterColumnChunksByteRanges(survived));
      try {
        reader.setupChunkingForFilterColumns(0L, 0L, survived, false, filterCols);
        try (ColumnVector rowMask = reader.takeFilterRowMask()) {
          assertEquals(2000L, rowMask.getRowCount(),
              "Mask spans input rows of surviving row groups (groups 1+2)");
          assertEquals(2000L, countTrue(rowMask),
              "All entries true: all-true seed, no filter evaluation has run");
        }
      } finally {
        closeAll(filterCols);
      }
    }
  }

  /**
   * Verifies takeFilterRowMask() exposes the page-index-stats-pre-evaluated mask after setup,
   * before any chunks have been materialized. By passing all 3 row groups into setup (skipping
   * the upstream filterRowGroupsWithStats step), the page-index pre-evaluation is observable:
   * groups 0 and 1 (max zip 4,999 and 54,999) are pruned to false; group 2 (zip 100,000+) is
   * preserved as true. Length = 3 × 5000 = 15,000; true-count = 5,000.
   */
  @Test
  void testTakeFilterRowMaskPageIndexStatsExposesPagePrunedMask(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.pageIndex(tmp).withFilter("zip_code", BinaryOperator.GREATER, 99999)) {
      open.withPageIndex();
      HybridScanReader reader = open.reader;
      int[] allRgs = reader.allRowGroups();
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          open.file, reader.filterColumnChunksByteRanges(allRgs));
      try {
        reader.setupChunkingForFilterColumns(0L, 0L, allRgs, true, filterCols);
        try (ColumnVector rowMask = reader.takeFilterRowMask()) {
          assertEquals(15000L, rowMask.getRowCount(),
              "Mask spans all 3 row groups: 3 × 5000 = 15,000");
          assertEquals(5000L, countTrue(rowMask),
              "Only group 2 (5000 rows) survives page-index pruning of zip_code > 99,999");
        }
      } finally {
        closeAll(filterCols);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: setupChunkingForPayloadColumns()
  // --------------------------------------------------------------------

  /**
   * Verifies setupChunkingForPayloadColumns() activates payload-column chunking:
   * hasNextTableChunk() reports true after setup.
   */
  @Test
  void testSetupChunkingForPayloadColumnsActivatesPipeline(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 150000)) {
      HybridScanReader reader = open.reader;
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          open.file, reader.filterColumnChunksByteRanges(survived));
      DeviceMemoryBuffer[] payloadCols = copyRangesToDevice(
          open.file, reader.payloadColumnChunksByteRanges(survived));
      try {
        reader.setupChunkingForFilterColumns(0L, 0L, survived, false, filterCols);
        try (ColumnVector rowMask = reader.takeFilterRowMask()) {
          reader.setupChunkingForPayloadColumns(0L, 0L, survived, rowMask, false, payloadCols);
          assertTrue(reader.hasNextTableChunk(), "payload chunking active after setup");
        }
      } finally {
        closeAll(filterCols);
        closeAll(payloadCols);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: materializePayloadColumnsChunk()
  // --------------------------------------------------------------------

  /**
   * Verifies materializePayloadColumnsChunk() drains the exact expected row count when
   * chained after setupChunkingForPayloadColumns(): 1599 rows surviving the filter
   * zip_code > 150,000.
   */
  @Test
  void testMaterializePayloadColumnsChunkExactTotal(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp).withFilter("zip_code", BinaryOperator.GREATER, 150000)) {
      HybridScanReader reader = open.reader;
      int[] survived = reader.filterRowGroupsWithStats(reader.allRowGroups());
      DeviceMemoryBuffer[] filterCols = copyRangesToDevice(
          open.file, reader.filterColumnChunksByteRanges(survived));
      DeviceMemoryBuffer[] payloadCols = copyRangesToDevice(
          open.file, reader.payloadColumnChunksByteRanges(survived));
      try {
        reader.setupChunkingForFilterColumns(0L, 0L, survived, false, filterCols);
        while (reader.hasNextTableChunk()) {
          reader.materializeFilterColumnsChunk().close();
        }
        try (ColumnVector rowMask = reader.takeFilterRowMask()) {
          reader.setupChunkingForPayloadColumns(0L, 0L, survived, rowMask, false, payloadCols);
          long total = 0;
          while (reader.hasNextTableChunk()) {
            try (Table chunk = reader.materializePayloadColumnsChunk(rowMask)) {
              total += chunk.getRowCount();
            }
          }
          assertEquals(1599L, total);
        }
      } finally {
        closeAll(filterCols);
        closeAll(payloadCols);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: setupChunkingForAllColumns() / materializeAllColumnsChunk()
  //
  // These two methods are tightly coupled (every meaningful scenario calls them in
  // sequence with hasNextTableChunk() driving the loop). One integration test below
  // exercises both together. Their post-close and null-argument behaviors are
  // covered by the parameterized tests at the end of the class.
  // --------------------------------------------------------------------

  /**
   * Verifies the all-columns chunked pipeline drains the exact total row count (3,000)
   * with all 3 projected columns in every chunk.
   */
  @Test
  void testChunkedAllColumnsExactTotal(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      HybridScanReader reader = open.reader;
      int[] rgs = reader.allRowGroups();
      ByteRange[] ranges = reader.allColumnChunksByteRanges(rgs);
      DeviceMemoryBuffer[] devs = copyRangesToDevice(open.file, ranges);
      try {
        reader.setupChunkingForAllColumns(0L, 0L, rgs, devs);
        long totalRows = 0;
        int chunks = 0;
        while (reader.hasNextTableChunk()) {
          try (Table chunk = reader.materializeAllColumnsChunk()) {
            assertEquals(3, chunk.getNumberOfColumns());
            totalRows += chunk.getRowCount();
            chunks++;
          }
        }
        assertTrue(chunks >= 1);
        assertEquals(3000L, totalRows);
      } finally {
        closeAll(devs);
      }
    }
  }

  // --------------------------------------------------------------------
  // Tests: hasNextTableChunk()
  // --------------------------------------------------------------------

  /**
   * Verifies hasNextTableChunk() reports the active lifecycle of a chunked pipeline: true
   * after setup, then false once chunks have been drained. With (0L, 0L) read limits, the
   * C++ contract guarantees a single chunk per pass, so the lifecycle is:
   * setup → true → one materialize → false.
   */
  @Test
  void testHasNextTableChunkActiveAndAfterDrain(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      HybridScanReader reader = open.reader;
      int[] rgs = reader.allRowGroups();
      DeviceMemoryBuffer[] devs = copyRangesToDevice(
          open.file, reader.allColumnChunksByteRanges(rgs));
      try {
        reader.setupChunkingForAllColumns(0L, 0L, rgs, devs);
        assertTrue(reader.hasNextTableChunk(), "chunking active after setup");
        reader.materializeAllColumnsChunk().close();
        assertFalse(reader.hasNextTableChunk(),
            "(0,0) read limits emit a single chunk; iterator drained after one materialize call");
      } finally {
        closeAll(devs);
      }
    }
  }

  /**
   * Verifies hasNextTableChunk() throws when called before any setupChunkingFor* method has
   * established an active pipeline. The C++ guard (CUDF_EXPECTS at hybrid_scan_impl.cpp
   * "Chunking not yet setup") propagates to Java as a CudfException.
   */
  @Test
  void testHasNextTableChunkRequiresSetupChunkingFirst(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      assertThrows(CudfException.class, () -> open.reader.hasNextTableChunk(),
          "Pre-setup hasNextTableChunk() must throw 'Chunking not yet setup'");
    }
  }

  // --------------------------------------------------------------------
  // Tests: constructRowGroupPasses()
  // --------------------------------------------------------------------

  /**
   * Verifies constructRowGroupPasses() with no read-limit (passReadLimit = 0) packs all
   * input row groups into a single pass that is structurally equal to the input.
   */
  @Test
  void testConstructRowGroupPassesUnlimitedReturnsSinglePass(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      int[] all = open.reader.allRowGroups();
      int[][] passes = open.reader.constructRowGroupPasses(all, 0L);
      assertEquals(1, passes.length, "passReadLimit = 0 must return one pass");
      assertArrayEquals(all, passes[0]);
    }
  }

  /**
   * Verifies constructRowGroupPasses() partitions input row groups across multiple passes,
   * preserving order, when passReadLimit is small enough to force splitting. With
   * passReadLimit = 1, comp_read_limit = floor(1 * 0.3) = 0, so compute_row_group_passes
   * closes a pass at every row group boundary: {[0]}, {[1]}, {[2]}. The exact partition
   * simultaneously verifies the multi-pass capability and that the result is covering,
   * disjoint, and order-preserving.
   */
  @Test
  void testConstructRowGroupPassesMultiPassPartition(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      int[] all = open.reader.allRowGroups();
      int[][] passes = open.reader.constructRowGroupPasses(all, 1L);
      assertEquals(3, passes.length, "passReadLimit = 1 forces each row group into its own pass");
      assertArrayEquals(new int[]{0}, passes[0]);
      assertArrayEquals(new int[]{1}, passes[1]);
      assertArrayEquals(new int[]{2}, passes[2]);
    }
  }

  // --------------------------------------------------------------------
  // Tests: close()
  // --------------------------------------------------------------------

  /**
   * Verifies close() flips the reader from operational to closed-and-rejecting state, and
   * that subsequent close() calls are idempotent. The pre-close probe asserts the reader
   * is usable (not just that the constructor succeeded); the post-close probe asserts that
   * assertNotClosed() now fires; the second close() asserts the no-throw idempotency
   * contract.
   */
  @Test
  void testCloseTransitionsAndIsIdempotent(@TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      HybridScanReader reader = open.reader;
      assertArrayEquals(new int[]{0, 1, 2}, reader.allRowGroups(),
          "Reader operational pre-close");
      reader.close();
      assertThrows(IllegalStateException.class, reader::allRowGroups,
          "close() flips state; further use throws");
      reader.close();
    }
  }

  // --------------------------------------------------------------------
  // Tests: null-argument rejection (parameterized)
  //
  // Every public method that accepts a non-null reference argument validates it with a
  // Java-side null check before reaching native code. The check is uniform across the API
  // (all throw IllegalArgumentException), so a single parameterized test exercises every
  // method's null-arg contract. Discoverability is preserved via the {0}RejectsNull
  // display name in the surefire report.
  // --------------------------------------------------------------------

  /** Verifies every public method rejects null arguments with IllegalArgumentException. */
  @ParameterizedTest(name = "{0}RejectsNull")
  @MethodSource("nullArgInvocations")
  void testRejectsNullArg(String name, Consumer<HybridScanReader> action,
                          @TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      assertThrows(IllegalArgumentException.class, () -> action.accept(open.reader));
    }
  }

  static Stream<Arguments> nullArgInvocations() {
    return Stream.of(
        invocation("setupPageIndex", r -> r.setupPageIndex(null)),
        invocation("totalRowsInRowGroups", r -> r.totalRowsInRowGroups(null)),
        invocation("filterRowGroupsWithStats", r -> r.filterRowGroupsWithStats(null)),
        invocation("secondaryFiltersByteRanges", r -> r.secondaryFiltersByteRanges(null)),
        invocation("filterRowGroupsWithDictionaryPagesNullBuffers",
            r -> r.filterRowGroupsWithDictionaryPages(null, new int[]{0})),
        invocation("filterRowGroupsWithDictionaryPagesNullRowGroups",
            r -> r.filterRowGroupsWithDictionaryPages(new DeviceMemoryBuffer[0], null)),
        invocation("filterColumnChunksByteRanges", r -> r.filterColumnChunksByteRanges(null)),
        invocation("payloadColumnChunksByteRanges", r -> r.payloadColumnChunksByteRanges(null)),
        invocation("allColumnChunksByteRanges", r -> r.allColumnChunksByteRanges(null)),
        invocation("materializeFilterColumns", r -> r.materializeFilterColumns(null,
            new DeviceMemoryBuffer[0], false)),
        invocation("materializeFilterColumnsNullBuffers", r -> r.materializeFilterColumns(
            new int[]{0}, null, false)),
        invocation("materializeFilterColumnsNullBufferElement", r -> r.materializeFilterColumns(
            new int[]{0}, new DeviceMemoryBuffer[]{null}, false)),
        invocation("materializePayloadColumnsNullRowGroups", r -> {
          try (ColumnVector mask = ColumnVector.fromBooleans(true)) {
            r.materializePayloadColumns(null, new DeviceMemoryBuffer[0], mask, false);
          }
        }),
        invocation("materializePayloadColumnsNullRowMask", r -> r.materializePayloadColumns(
            new int[]{0}, new DeviceMemoryBuffer[0], null, false)),
        invocation("materializePayloadColumnsNullBuffers", r -> {
          try (ColumnVector mask = ColumnVector.fromBooleans(true)) {
            r.materializePayloadColumns(new int[]{0}, null, mask, false);
          }
        }),
        invocation("materializePayloadColumnsNullBufferElement", r -> {
          try (ColumnVector mask = ColumnVector.fromBooleans(true)) {
            r.materializePayloadColumns(new int[]{0}, new DeviceMemoryBuffer[]{null}, mask, false);
          }
        }),
        invocation("materializeAllColumns", r -> r.materializeAllColumns(null,
            new DeviceMemoryBuffer[0])),
        invocation("materializeAllColumnsNullBuffers", r -> r.materializeAllColumns(
            new int[]{0}, null)),
        invocation("materializeAllColumnsNullBufferElement", r -> r.materializeAllColumns(
            new int[]{0}, new DeviceMemoryBuffer[]{null})),
        invocation("setupChunkingForFilterColumns", r -> r.setupChunkingForFilterColumns(
            0L, 0L, null, false, new DeviceMemoryBuffer[0])),
        invocation("setupChunkingForFilterColumnsNullBuffers", r ->
            r.setupChunkingForFilterColumns(0L, 0L, new int[]{0}, false, null)),
        invocation("setupChunkingForFilterColumnsNullBufferElement", r ->
            r.setupChunkingForFilterColumns(0L, 0L, new int[]{0}, false,
                new DeviceMemoryBuffer[]{null})),
        invocation("setupChunkingForPayloadColumnsNullRowGroups", r -> {
          try (ColumnVector mask = ColumnVector.fromBooleans(true)) {
            r.setupChunkingForPayloadColumns(0L, 0L, null, mask, false,
                new DeviceMemoryBuffer[0]);
          }
        }),
        invocation("setupChunkingForPayloadColumnsNullRowMask", r ->
            r.setupChunkingForPayloadColumns(0L, 0L, new int[]{0}, null, false,
                new DeviceMemoryBuffer[0])),
        invocation("setupChunkingForPayloadColumnsNullBuffers", r -> {
          try (ColumnVector mask = ColumnVector.fromBooleans(true)) {
            r.setupChunkingForPayloadColumns(0L, 0L, new int[]{0}, mask, false, null);
          }
        }),
        invocation("setupChunkingForPayloadColumnsNullBufferElement", r -> {
          try (ColumnVector mask = ColumnVector.fromBooleans(true)) {
            r.setupChunkingForPayloadColumns(0L, 0L, new int[]{0}, mask, false,
                new DeviceMemoryBuffer[]{null});
          }
        }),
        invocation("materializePayloadColumnsChunk", r -> r.materializePayloadColumnsChunk(null)),
        invocation("setupChunkingForAllColumns", r -> r.setupChunkingForAllColumns(
            0L, 0L, null, new DeviceMemoryBuffer[0])),
        invocation("setupChunkingForAllColumnsNullBuffers", r ->
            r.setupChunkingForAllColumns(0L, 0L, new int[]{0}, null)),
        invocation("setupChunkingForAllColumnsNullBufferElement", r ->
            r.setupChunkingForAllColumns(0L, 0L, new int[]{0}, new DeviceMemoryBuffer[]{null})),
        invocation("constructRowGroupPasses", r -> r.constructRowGroupPasses(null, 0L))
    );
  }

  // --------------------------------------------------------------------
  // Tests: negative read-limit rejection (parameterized)
  // --------------------------------------------------------------------

  /** Verifies every setupChunking* / constructRowGroupPasses entry point rejects a negative chunkReadLimit or passReadLimit. */
  @ParameterizedTest(name = "{0}RejectsNegativeReadLimits")
  @MethodSource("negativeLimitInvocations")
  void testRejectsNegativeReadLimits(String name, Consumer<HybridScanReader> action,
                                     @TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      assertThrows(IllegalArgumentException.class, () -> action.accept(open.reader));
    }
  }

  static Stream<Arguments> negativeLimitInvocations() {
    // Default cases use chunk=-1, pass=0; the swapped case (chunk=0, pass=-1) confirms
    // both arguments are validated independently.
    return Stream.of(
        invocation("setupChunkingForFilterColumns", r -> r.setupChunkingForFilterColumns(
            -1L, 0L, new int[]{0}, false, new DeviceMemoryBuffer[0])),
        invocation("setupChunkingForPayloadColumns", r -> {
          try (ColumnVector mask = ColumnVector.fromBooleans(true)) {
            r.setupChunkingForPayloadColumns(-1L, 0L, new int[]{0}, mask, false,
                new DeviceMemoryBuffer[0]);
          }
        }),
        invocation("setupChunkingForAllColumns", r ->
            r.setupChunkingForAllColumns(-1L, 0L, new int[]{0}, new DeviceMemoryBuffer[0])),
        invocation("constructRowGroupPasses", r ->
            r.constructRowGroupPasses(new int[]{0}, -1L)),
        invocation("setupChunkingForFilterColumnsPassLimit", r ->
            r.setupChunkingForFilterColumns(0L, -1L, new int[]{0}, false,
                new DeviceMemoryBuffer[0]))
    );
  }

  // --------------------------------------------------------------------
  // Tests: post-close behavior (parameterized)
  //
  // Every public method on HybridScanReader calls assertNotClosed() before reaching native
  // code, which throws IllegalStateException when the reader has been closed. The contract
  // is uniform across the API, so a single parameterized test exercises every method's
  // post-close behavior. Discoverability is preserved via the {0}AfterCloseThrows display
  // name in the surefire report.
  // --------------------------------------------------------------------

  /** Verifies every public method throws IllegalStateException after the reader is closed. */
  @ParameterizedTest(name = "{0}AfterCloseThrows")
  @MethodSource("postCloseInvocations")
  void testAfterCloseThrows(String name, Consumer<HybridScanReader> action,
                            @TempDir Path tmp) throws IOException {
    try (OpenReader open = OpenReader.standard(tmp)) {
      open.reader.close();
      assertThrows(IllegalStateException.class, () -> action.accept(open.reader));
    }
  }

  static Stream<Arguments> postCloseInvocations() {
    return Stream.of(
        invocation("pageIndexByteRange", HybridScanReader::pageIndexByteRange),
        invocation("setupPageIndex", r -> {
          try (HostMemoryBuffer empty = HostMemoryBuffer.allocate(1)) {
            r.setupPageIndex(empty);
          }
        }),
        invocation("allRowGroups", HybridScanReader::allRowGroups),
        invocation("totalRowsInRowGroups", r -> r.totalRowsInRowGroups(new int[]{0})),
        invocation("filterRowGroupsWithStats", r -> r.filterRowGroupsWithStats(new int[]{0})),
        invocation("secondaryFiltersByteRanges", r -> r.secondaryFiltersByteRanges(new int[]{0})),
        invocation("filterRowGroupsWithDictionaryPages", r ->
            r.filterRowGroupsWithDictionaryPages(new DeviceMemoryBuffer[0], new int[]{0})),
        invocation("filterColumnChunksByteRanges", r -> r.filterColumnChunksByteRanges(new int[]{0})),
        invocation("payloadColumnChunksByteRanges", r -> r.payloadColumnChunksByteRanges(new int[]{0})),
        invocation("allColumnChunksByteRanges", r -> r.allColumnChunksByteRanges(new int[]{0})),
        invocation("materializeFilterColumns", r -> r.materializeFilterColumns(new int[]{0},
            new DeviceMemoryBuffer[0], false)),
        invocation("materializePayloadColumns", r -> {
          try (ColumnVector mask = ColumnVector.fromBooleans(true)) {
            r.materializePayloadColumns(new int[]{0}, new DeviceMemoryBuffer[0], mask, false);
          }
        }),
        invocation("materializeAllColumns", r -> r.materializeAllColumns(new int[]{0},
            new DeviceMemoryBuffer[0])),
        invocation("setupChunkingForFilterColumns", r -> r.setupChunkingForFilterColumns(
            0L, 0L, new int[]{0}, false, new DeviceMemoryBuffer[0])),
        invocation("materializeFilterColumnsChunk", HybridScanReader::materializeFilterColumnsChunk),
        invocation("takeFilterRowMask", HybridScanReader::takeFilterRowMask),
        invocation("setupChunkingForPayloadColumns", r -> {
          try (ColumnVector mask = ColumnVector.fromBooleans(true)) {
            r.setupChunkingForPayloadColumns(0L, 0L, new int[]{0}, mask, false,
                new DeviceMemoryBuffer[0]);
          }
        }),
        invocation("materializePayloadColumnsChunk", r -> {
          try (ColumnVector mask = ColumnVector.fromBooleans(true)) {
            r.materializePayloadColumnsChunk(mask);
          }
        }),
        invocation("setupChunkingForAllColumns", r -> r.setupChunkingForAllColumns(
            0L, 0L, new int[]{0}, new DeviceMemoryBuffer[0])),
        invocation("materializeAllColumnsChunk", HybridScanReader::materializeAllColumnsChunk),
        invocation("hasNextTableChunk", HybridScanReader::hasNextTableChunk),
        invocation("constructRowGroupPasses", r -> r.constructRowGroupPasses(new int[]{0}, 0L)),
        invocation("setFilter", r -> r.setFilter(null))
    );
  }

  // --------------------------------------------------------------------
  // Fixture helpers
  // --------------------------------------------------------------------

  /**
   * Bundles a Parquet fixture (file bytes + extracted footer) with an open
   * {@link HybridScanReader} and an optional compiled filter, all under one
   * try-with-resources lifecycle. Use the {@link #standard(Path)},
   * {@link #pageIndex(Path)}, or {@link #rowGroupStats(Path)} factories, optionally chained
   * with {@link #withFilter(String, BinaryOperator, int)}.
   */
  private static final class OpenReader implements AutoCloseable {
    final HostMemoryBuffer file;
    final HostMemoryBuffer footer;
    HybridScanReader reader;
    CompiledExpression filter;

    private OpenReader(HostMemoryBuffer file, HostMemoryBuffer footer,
                       HybridScanReader reader, CompiledExpression filter) {
      this.file = file;
      this.footer = footer;
      this.reader = reader;
      this.filter = filter;
    }

    static OpenReader standard(Path tmp) throws IOException {
      File pq = tmp.resolve("fixture.parquet").toFile();
      writeFixtureParquet(pq);
      return openFromFile(pq, DEFAULT_COLS);
    }

    static OpenReader pageIndex(Path tmp) throws IOException {
      File pq = tmp.resolve("fixture.parquet").toFile();
      writePageIndexParquet(pq);
      return openFromFile(pq, DEFAULT_COLS);
    }

    static OpenReader rowGroupStats(Path tmp) throws IOException {
      File pq = tmp.resolve("fixture.parquet").toFile();
      writeRowGroupStatsParquet(pq);
      return openFromFile(pq, DEFAULT_COLS);
    }

    private static OpenReader openFromFile(File pq, String[] cols) throws IOException {
      HostMemoryBuffer file = readFileToHostBuffer(pq);
      HostMemoryBuffer footer = null;
      HybridScanReader reader = null;
      try {
        footer = extractFooter(file);
        reader = new HybridScanReader(footer, optsForColumns(cols));
      } catch (Throwable t) {
        for (AutoCloseable c : new AutoCloseable[]{reader, footer, file}) {
          if (c != null) {
            try {
              c.close();
            } catch (Throwable closeEx) {
              t.addSuppressed(closeEx);
            }
          }
        }
        throw t;
      }
      return new OpenReader(file, footer, reader, null);
    }

    /**
     * Install a fresh filter expression on the existing reader via
     * {@link HybridScanReader#setFilter(CompiledExpression)}. Closes any previously
     * installed filter after the new one is committed so this fixture retains a single
     * live filter at any time.
     */
    OpenReader withFilter(String col, BinaryOperator op, int literal) {
      CompiledExpression newFilter = new BinaryOperation(op, new ColumnNameReference(col),
          Literal.ofInt(literal)).compile();
      try {
        reader.setFilter(newFilter);
      } catch (Throwable t) {
        newFilter.close();
        throw t;
      }
      CompiledExpression oldFilter = this.filter;
      this.filter = newFilter;
      if (oldFilter != null) oldFilter.close();
      return this;
    }

    /** Loads the file's page-index region into the reader; returns {@code this} for chaining. */
    OpenReader withPageIndex() throws IOException {
      ByteRange piRange = reader.pageIndexByteRange();
      try (HostMemoryBuffer pi = file.slice(piRange.offset(), piRange.size())) {
        reader.setupPageIndex(pi);
      }
      return this;
    }

    @Override
    public void close() {
      Throwable err = null;
      try { if (reader != null) reader.close(); } catch (Throwable t) { err = t; }
      try { if (filter != null) filter.close(); } catch (Throwable t) { if (err == null) err = t; }
      try { if (footer != null) footer.close(); } catch (Throwable t) { if (err == null) err = t; }
      try { if (file != null) file.close(); } catch (Throwable t) { if (err == null) err = t; }
      if (err instanceof RuntimeException) throw (RuntimeException) err;
      if (err instanceof Error) throw (Error) err;
      if (err != null) throw new RuntimeException(err);
    }
  }

  private static Arguments invocation(String name, Consumer<HybridScanReader> action) {
    return arguments(name, action);
  }

  /**
   * Writes a 3-row-group Parquet file: int columns {@code id} (globally sequential),
   * {@code zip_code} (10000 + id * 100), and {@code num_units} (1..3 cycle).
   * Uses {@code PAGE} statistics; 1,000 rows per group (3,000 total).
   *
   * @return total row count (3,000)
   */
  private static int writeFixtureParquet(File path) {
    int rowsPerGroup = 1000;
    int numGroups = 3;
    int rows = rowsPerGroup * numGroups;
    ParquetWriterOptions opts = ParquetWriterOptions.builder()
        .withNonNullableColumns("id", "zip_code", "num_units")
        .withRowGroupSizeRows(rowsPerGroup)
        .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.PAGE)
        .build();
    try (TableWriter writer = Table.writeParquetChunked(opts, path)) {
      for (int g = 0; g < numGroups; g++) {
        int start = g * rowsPerGroup;
        try (ColumnVector id = ColumnVector.fromInts(
                 IntStream.range(start, start + rowsPerGroup).toArray());
             ColumnVector zipCode = ColumnVector.fromInts(
                 IntStream.range(start, start + rowsPerGroup)
                     .map(i -> 10000 + i * 100).toArray());
             ColumnVector numUnits = ColumnVector.fromInts(
                 IntStream.range(start, start + rowsPerGroup)
                     .map(i -> 1 + (i % 3)).toArray());
             Table t = new Table(id, zipCode, numUnits)) {
          writer.write(t);
        }
      }
    }
    return rows;
  }

  /**
   * Writes a small Parquet file with {@code ROWGROUP}-level statistics: row-group min/max
   * are recorded but no page index (no {@code ColumnIndex}/{@code OffsetIndex}) is emitted.
   * Includes a low-cardinality {@code num_units} column ({1, 2, 3} cycle) so the writer's
   * ADAPTIVE dictionary policy emits a dictionary; this lets tests exercise the
   * "no page index, dict exists" path (see
   * {@link #testSecondaryFiltersByteRangesEmptyForRowGroupStats}).
   */
  private static void writeRowGroupStatsParquet(File path) {
    int rows = 100;
    ParquetWriterOptions opts = ParquetWriterOptions.builder()
        .withNonNullableColumns("id", "zip_code", "num_units")
        .withRowGroupSizeRows(rows)
        .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.ROWGROUP)
        .build();
    try (TableWriter writer = Table.writeParquetChunked(opts, path);
         ColumnVector id = ColumnVector.fromInts(IntStream.range(0, rows).toArray());
         ColumnVector zipCode = ColumnVector.fromInts(
             IntStream.range(0, rows).map(i -> 10000 + i).toArray());
         ColumnVector numUnits = ColumnVector.fromInts(
             IntStream.range(0, rows).map(i -> 1 + (i % 3)).toArray());
         Table t = new Table(id, zipCode, numUnits)) {
      writer.write(t);
    }
  }

  /**
   * Writes a 3-row-group Parquet file with {@code COLUMN}-level statistics, guaranteeing
   * a non-empty page index. Each group gets a non-overlapping {@code zip_code} range
   * (group {@code g}: base = g × 50,000):
   * <ul>
   *   <li>Group 0: zip_code 0–4,999</li>
   *   <li>Group 1: zip_code 50,000–54,999</li>
   *   <li>Group 2: zip_code 100,000–104,999</li>
   * </ul>
   * Each group also has a distinct low-cardinality {@code num_units} dictionary, enabling
   * strict-subset row-group pruning tests for {@code filterRowGroupsWithDictionaryPages}:
   * <ul>
   *   <li>Group 0: num_units ∈ {1, 2}</li>
   *   <li>Group 1: num_units ∈ {2, 3}</li>
   *   <li>Group 2: num_units ∈ {3, 4}</li>
   * </ul>
   * 5,000 rows per group (15,000 total).
   *
   * @return total row count (15,000)
   */
  private static int writePageIndexParquet(File path) {
    int rowsPerGroup = 5_000;
    int numGroups = 3;
    int rows = rowsPerGroup * numGroups;
    int[] zipBases = {0, 50_000, 100_000};
    int[][] numUnitsValues = {{1, 2}, {2, 3}, {3, 4}};
    ParquetWriterOptions opts = ParquetWriterOptions.builder()
        .withNonNullableColumns("id", "zip_code", "num_units")
        .withRowGroupSizeRows(rowsPerGroup)
        .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.COLUMN)
        .build();
    try (TableWriter writer = Table.writeParquetChunked(opts, path)) {
      for (int g = 0; g < numGroups; g++) {
        int start = g * rowsPerGroup;
        int zipBase = zipBases[g];
        int[] numUnitsVals = numUnitsValues[g];
        try (ColumnVector id = ColumnVector.fromInts(
                 IntStream.range(start, start + rowsPerGroup).toArray());
             ColumnVector zipCode = ColumnVector.fromInts(
                 IntStream.range(0, rowsPerGroup).map(i -> zipBase + i).toArray());
             ColumnVector numUnits = ColumnVector.fromInts(
                 IntStream.range(0, rowsPerGroup)
                     .map(i -> numUnitsVals[i % numUnitsVals.length]).toArray());
             Table t = new Table(id, zipCode, numUnits)) {
          writer.write(t);
        }
      }
    }
    return rows;
  }

  /** Read the entire file into a {@link HostMemoryBuffer}. */
  private static HostMemoryBuffer readFileToHostBuffer(File file) throws IOException {
    byte[] fileBytes = Files.readAllBytes(file.toPath());
    HostMemoryBuffer buffer = HostMemoryBuffer.allocate(fileBytes.length);
    buffer.setBytes(0, fileBytes, 0, fileBytes.length);
    return buffer;
  }

  /**
   * Extract the Parquet file footer from a host buffer.
   * Format: {@code [footer_bytes][4-byte LE footer_length][4-byte magic "PAR1"]}.
   */
  private static HostMemoryBuffer extractFooter(HostMemoryBuffer fileBuffer) {
    long fileLen = fileBuffer.getLength();
    int footerLength = fileBuffer.getInt(fileLen - 8);
    long footerStart = fileLen - 8 - footerLength;
    byte[] footerBytes = new byte[footerLength];
    fileBuffer.getBytes(footerBytes, 0, footerStart, footerLength);
    HostMemoryBuffer footer = HostMemoryBuffer.allocate(footerLength);
    footer.setBytes(0, footerBytes, 0, footerLength);
    return footer;
  }

  /** Copy byte ranges from a host buffer into device buffers (one per range). */
  private static DeviceMemoryBuffer[] copyRangesToDevice(HostMemoryBuffer fileBuffer,
                                                         ByteRange[] ranges) {
    DeviceMemoryBuffer[] out = new DeviceMemoryBuffer[ranges.length];
    try {
      for (int i = 0; i < ranges.length; i++) {
        ByteRange r = ranges[i];
        DeviceMemoryBuffer dev = DeviceMemoryBuffer.allocate(r.size());
        // Store before the copy so the catch handler frees it if slice/copy throws.
        out[i] = dev;
        try (HostMemoryBuffer slice = fileBuffer.slice(r.offset(), r.size())) {
          dev.copyFromHostBuffer(slice);
        }
      }
      return out;
    } catch (Throwable t) {
      for (DeviceMemoryBuffer b : out) {
        if (b != null) b.close();
      }
      throw t;
    }
  }

  private static void closeAll(DeviceMemoryBuffer[] buffers) {
    if (buffers == null) return;
    for (DeviceMemoryBuffer b : buffers) {
      if (b != null) b.close();
    }
  }

  /** Count true entries in a boolean ColumnVector by host-side iteration. */
  private static long countTrue(ColumnVector mask) {
    try (HostColumnVector host = mask.copyToHost()) {
      long count = 0;
      for (long i = 0; i < host.getRowCount(); i++) {
        if (host.getBoolean(i)) count++;
      }
      return count;
    }
  }

  private static ParquetOptions optsForColumns(String... cols) {
    ParquetOptions.Builder b = ParquetOptions.builder();
    for (String c : cols) {
      b.includeColumn(c);
    }
    return b.build();
  }
}

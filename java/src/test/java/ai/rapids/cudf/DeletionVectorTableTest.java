/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.TableTestUtils.HostMemoryBufferArray;
import ai.rapids.cudf.DeletionVector.DeletionVectorInfo;
import ai.rapids.cudf.DeletionVector.ParquetChunkedReader;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import static ai.rapids.cudf.AssertUtils.assertTableTypes;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;


class DeletionVectorTableTest extends CudfTestBase {
  private static final File TEST_FILE1 = TestUtils.getResourceAsFile("acq.parquet");
  private static final File TEST_FILE2 = TestUtils.getResourceAsFile("splittable-multi-rgs.parquet"); // 4 row groups, 10000 rows per rg
  private static final File DELETED_ROWS_FILE1 = TestUtils.getResourceAsFile("acq-deleted.bin");
  private static final int DELETED_ROWS_COUNT1 = 50;
  private static final File DELETED_ROWS_FILE2 = TestUtils.getResourceAsFile("splittable-deleted.bin");
  private static final int DELETED_ROWS_COUNT2 = 3959;
  private static final int DELETED_ROWS_COUNT2_RGS_1_AND_3 = 1974;

  @ParameterizedTest(name = "isRetention={0}")
  @CsvSource({"false", "true"})
  void testReadParquetReadAllRowGroups(boolean isRetention) throws IOException {
    ParquetOptions opts = ParquetOptions.builder()
            .includeColumn("loan_id")
            .includeColumn("zip")
            .includeColumn("num_units")
            .build();
    byte[][] data = TableTestUtils.sliceBytes(TableTestUtils.arrayFrom(TEST_FILE1), 10);
    byte[] bitmapData = TableTestUtils.arrayFrom(DELETED_ROWS_FILE1);
    try (HostMemoryBufferArray array = TableTestUtils.buffersFrom(data);
         HostMemoryBufferArray bitmapArray = TableTestUtils.buffersFrom(new byte[][] { bitmapData })) {
      DeletionVectorInfo dvInfo = new DeletionVectorInfo(
          bitmapArray.buffers[0], isRetention, null, null);
      try (Table table = DeletionVector.readParquet(opts, array.buffers, new DeletionVectorInfo[] { dvInfo })) {
        long expectedRows =
            isRetention ? DELETED_ROWS_COUNT1 : 1000 - DELETED_ROWS_COUNT1;
        assertEquals(expectedRows, table.getRowCount());
        assertTableTypes(new DType[]{DType.UINT64, DType.INT64, DType.INT32, DType.INT32}, table);
      }
    }
  }

  @Test
  void testReadParquetReadSomeRowGroups() throws IOException {
    ParquetOptions opts = ParquetOptions.DEFAULT;
    byte[][] data = TableTestUtils.sliceBytes(TableTestUtils.arrayFrom(TEST_FILE2), 1);
    byte[] bitmapData = TableTestUtils.arrayFrom(DELETED_ROWS_FILE2);
    int[][] rowGroups = new int[][] { {1, 3} };
    try (HostMemoryBufferArray array = TableTestUtils.buffersFrom(data);
         HostMemoryBufferArray bitmapArray = TableTestUtils.buffersFrom(new byte[][] { bitmapData })) {
      long[] rowGroupOffsets = Arrays.stream(rowGroups[0]).mapToLong(i -> i * 10000L).toArray();
      int[] rowGroupNumRows = Arrays.stream(rowGroups[0]).map(i -> 10000).toArray();
      DeletionVectorInfo dvInfo = new DeletionVectorInfo(bitmapArray.buffers[0], false, rowGroupOffsets, rowGroupNumRows);
      try (Table table = DeletionVector.readParquet(opts, array.buffers, rowGroups, new DeletionVectorInfo[] { dvInfo })) {
        long rows = table.getRowCount();
        assertEquals(20000 - DELETED_ROWS_COUNT2_RGS_1_AND_3, rows);
        System.err.println("Table schema: " + Arrays.stream(table.getColumns()).map(c -> c.getType().toString()).reduce((a, b) -> a + ", " + b).orElse(""));
        assertTableTypes(new DType[]{DType.UINT64, DType.INT32, DType.INT64}, table);
      }
    }
  }

  @Test
  void testChunkedReadParquetAllRowGroups() throws Exception {
    byte[][] data = TableTestUtils.sliceBytes(TableTestUtils.arrayFrom(TEST_FILE2), 2);
    byte[] bitmapData = TableTestUtils.arrayFrom(DELETED_ROWS_FILE2);
    try (HostMemoryBufferArray array = TableTestUtils.buffersFrom(data);
         HostMemoryBufferArray bitmapArray = TableTestUtils.buffersFrom(new byte[][] { bitmapData })) {
      ParquetOptions opts = ParquetOptions.DEFAULT;
      DeletionVectorInfo dvInfo = new DeletionVectorInfo(bitmapArray.buffers[0], false, null, null);
      try (ParquetChunkedReader reader = DeletionVector.newParquetChunkedReader(240000, 0, opts, array.buffers,
              new DeletionVectorInfo[] { dvInfo })) {
        int numChunks = 0;
        long totalRows = 0;
        while(reader.hasNext()) {
          ++numChunks;
          try(Table chunk = reader.readChunk()) {
            totalRows += chunk.getRowCount();
            assertTableTypes(new DType[]{DType.UINT64, DType.INT32, DType.INT64}, chunk);
          }
        }
        assertEquals(2, numChunks);
        assertEquals(40000 - DELETED_ROWS_COUNT2, totalRows);
      }
    }
  }

  @ParameterizedTest(name = "isRetention={0}")
  @CsvSource({"false", "true"})
  void testChunkedReadParquetSomeRowGroups(boolean isRetention) throws Exception {
    byte[][] data = TableTestUtils.sliceBytes(TableTestUtils.arrayFrom(TEST_FILE2), 2);
    byte[] bitmapData = TableTestUtils.arrayFrom(DELETED_ROWS_FILE2);
    int[][] rowGroups = new int[][] { {1, 3} };
    try (HostMemoryBufferArray array = TableTestUtils.buffersFrom(data);
         HostMemoryBufferArray bitmapArray = TableTestUtils.buffersFrom(new byte[][] { bitmapData })) {
      ParquetOptions opts = ParquetOptions.DEFAULT;
      long[] rowGroupOffsets = Arrays.stream(rowGroups[0]).mapToLong(i -> i * 10000L).toArray();
      int[] rowGroupNumRows = Arrays.stream(rowGroups[0]).map(i -> 10000).toArray();
      DeletionVectorInfo dvInfo = new DeletionVectorInfo(
          bitmapArray.buffers[0], isRetention, rowGroupOffsets, rowGroupNumRows);
      try (ParquetChunkedReader reader = DeletionVector.newParquetChunkedReader(120000, 0, opts, array.buffers,
        rowGroups, new DeletionVectorInfo[] { dvInfo })) {
        int numChunks = 0;
        long totalRows = 0;
        while(reader.hasNext()) {
          ++numChunks;
          try(Table chunk = reader.readChunk()) {
            totalRows += chunk.getRowCount();
            assertTableTypes(new DType[]{DType.UINT64, DType.INT32, DType.INT64}, chunk);
          }
        }
        assertEquals(2, numChunks);
        long expectedRows = isRetention
            ? DELETED_ROWS_COUNT2_RGS_1_AND_3
            : 20000 - DELETED_ROWS_COUNT2_RGS_1_AND_3;
        assertEquals(expectedRows, totalRows);
      }
    }
  }

  @Test
  void testChunkedReadParquetMultiFiles() throws Exception {
    byte[][] bitmaps = new byte[2][];
    bitmaps[0] = TableTestUtils.arrayFrom(DELETED_ROWS_FILE2);
    bitmaps[1] = TableTestUtils.arrayFrom(DELETED_ROWS_FILE2);
    int[][] rowGroups = new int[][] { {3}, {1} };
    try (HostMemoryBufferArray bitmapArray = TableTestUtils.buffersFrom(bitmaps)) {
      ParquetOptions opts = ParquetOptions.DEFAULT;
      long[] rowGroupOffsets = new long[] { 30000L, 10000L };
      int[] rowGroupNumRows = new int[] { 10000, 10000 };
      DeletionVectorInfo dvInfo = new DeletionVectorInfo(bitmapArray.buffers[0], false, rowGroupOffsets, rowGroupNumRows);
      try (ParquetChunkedReader reader = DeletionVector.newParquetChunkedReader(120000, 0, opts, new String[] {
              TEST_FILE2.getAbsolutePath(),
              TEST_FILE2.getAbsolutePath()
          },
        rowGroups, new DeletionVectorInfo[] { dvInfo })) {
        int numChunks = 0;
        long totalRows = 0;
        while(reader.hasNext()) {
          ++numChunks;
          try(Table chunk = reader.readChunk()) {
            totalRows += chunk.getRowCount();
            assertTableTypes(new DType[]{DType.UINT64, DType.INT32, DType.INT64}, chunk);
          }
        }
        assertEquals(2, numChunks);
        assertEquals(20000 - DELETED_ROWS_COUNT2_RGS_1_AND_3, totalRows);
      }
    }
  }

  @Test
  void testMixedDeletionAndRetentionVectorsRejected() throws IOException {
    byte[][] data = TableTestUtils.sliceBytes(TableTestUtils.arrayFrom(TEST_FILE1), 10);
    byte[] bitmapData = TableTestUtils.arrayFrom(DELETED_ROWS_FILE1);
    try (HostMemoryBufferArray array = TableTestUtils.buffersFrom(data);
         HostMemoryBufferArray bitmapArray =
             TableTestUtils.buffersFrom(new byte[][] { bitmapData })) {
      DeletionVectorInfo[] mixedVectorInfos = new DeletionVectorInfo[] {
          new DeletionVectorInfo(bitmapArray.buffers[0], false, null, null),
          new DeletionVectorInfo(bitmapArray.buffers[0], true, null, null)
      };

      assertThrows(IllegalArgumentException.class,
          () -> DeletionVector.readParquet(
              ParquetOptions.DEFAULT, array.buffers, mixedVectorInfos));
      assertThrows(IllegalArgumentException.class,
          () -> DeletionVector.newParquetChunkedReader(
              0, 0, ParquetOptions.DEFAULT, array.buffers, mixedVectorInfos));
    }
  }
}

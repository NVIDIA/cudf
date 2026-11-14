/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Provides an interface for writing out Table information in multiple steps.
 * A TableWriter will be returned from one of various factory functions in the Table class that
 * let you set the format of the data and its destination. After that write can be called one or
 * more times. When you are done writing call close to finish.
 */
public abstract class TableWriter implements AutoCloseable {
  protected long writerHandle;

  TableWriter(long writerHandle) { this.writerHandle = writerHandle; }

  /**
   * Write out a table. Note that all columns must be in the same order each time this is called
   * and the format of each table cannot change.
   * @param table what to write out.
   */
  abstract public void write(Table table) throws CudfException;

  @Override
  abstract public void close() throws CudfException;

  /**
   * Throw if any column of the table, including nested children, is DICTIONARY32. The file
   * writers (Parquet, ORC, CSV) do not support dictionary columns; callers must decode them with
   * {@link ColumnView#dictionaryDecode()} first.
   */
  static void rejectDictionaryColumns(Table table, String format) {
    for (int i = 0; i < table.getNumberOfColumns(); i++) {
      rejectDictionaryColumns(table.getColumn(i), format);
    }
  }

  static void rejectDictionaryColumns(ColumnView column, String format) {
    DType type = column.getType();
    if (DType.DICTIONARY32.equals(type)) {
      throw new IllegalArgumentException("The " + format + " writer does not support " +
          "DICTIONARY32 columns; decode them with ColumnView.dictionaryDecode() first");
    }
    // Only LIST and STRUCT columns can contain a dictionary child.
    if (type.isNestedType()) {
      for (int i = 0; i < column.getNumChildren(); i++) {
        try (ColumnView child = column.getChildColumnView(i)) {
          rejectDictionaryColumns(child, format);
        }
      }
    }
  }

  public static class WriteStatistics {
    public final long numCompressedBytes; // The number of bytes that were successfully compressed
    public final long numFailedBytes;     // The number of bytes that failed to compress
    public final long numSkippedBytes;    // The number of bytes that were skipped during compression
    public final double compressionRatio; // The compression ratio for the successfully compressed data

    public WriteStatistics(long numCompressedBytes, long numFailedBytes, long numSkippedBytes,
        double compressionRatio) {
      this.numCompressedBytes = numCompressedBytes;
      this.numFailedBytes = numFailedBytes;
      this.numSkippedBytes = numSkippedBytes;
      this.compressionRatio = compressionRatio;
    }
  }

  /**
   * Get the write statistics for the writer up to the last write call.
   * Currently, only ORC and Parquet writers support write statistics.
   * Calling this method on other writers will return null.
   * @return The write statistics.
   */
  public WriteStatistics getWriteStatistics() {
    double[] statsData = getWriteStatistics(writerHandle);
    assert statsData.length == 4 : "Unexpected write statistics data length";
    return new WriteStatistics((long) statsData[0], (long) statsData[1], (long) statsData[2],
        statsData[3]);
  }

  /**
   * Get the write statistics for the writer up to the last write call.
   * The data returned from native method is encoded as an array of doubles.
   * @param writerHandle The handle to the writer.
   * @return The write statistics.
   */
  private static native double[] getWriteStatistics(long writerHandle);
}

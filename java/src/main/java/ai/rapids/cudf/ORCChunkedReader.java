/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Provide an interface for reading an ORC file in an iterative manner.
 */
public class ORCChunkedReader implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Construct the reader instance from read limits, output row granularity,
   * and a file already loaded in a memory buffer.
   *
   * @param chunkReadLimit Limit on total number of bytes to be returned per read,
   *                       or 0 if there is no limit.
   * @param passReadLimit  Limit on the amount of memory used by the chunked reader,
   *                       or 0 if there is no limit.
   * @param opts           The options for ORC reading.
   * @param buffer         Raw ORC file content.
   * @param offset         The starting offset into buffer.
   * @param len            The number of bytes to parse the given buffer.
   */
  public ORCChunkedReader(long chunkReadLimit, long passReadLimit,
      ORCOptions opts, HostMemoryBuffer buffer, long offset, long len) {
    handle = createReader(chunkReadLimit, passReadLimit,
        opts.getIncludeColumnNames(), buffer.getAddress() + offset, len,
        opts.usingNumPyTypes(), opts.timeUnit().typeId.getNativeId(),
        opts.getDecimal128Columns(),
        opts.ignoreTimezoneInStripeFooter(), opts.convertDatesToProleptic());
    if (handle == 0) {
      throw new IllegalStateException("Cannot create native chunked ORC reader object.");
    }
  }

  /**
   * Construct a chunked ORC reader instance, similar to
   * {@link ORCChunkedReader#ORCChunkedReader(long, long, ORCOptions, HostMemoryBuffer, long, long)},
   * with an additional parameter to control the granularity of the output table.
   * When reading a chunk table, with respect to the given size limits, a subset of stripes may
   * be loaded, decompressed and decoded into a large intermediate table. The reader will then
   * subdivide that table into smaller tables for final output using
   * {@code outputRowSizingGranularity} as the subdivision step. If the chunked reader is
   * constructed without this parameter, the default value of 10k rows will be used.
   *
   * @param outputRowSizingGranularity The change step in number of rows in the output table.
   * @see ORCChunkedReader#ORCChunkedReader(long, long, ORCOptions, HostMemoryBuffer, long, long)
   */
  public ORCChunkedReader(long chunkReadLimit, long passReadLimit, long outputRowSizingGranularity,
      ORCOptions opts, HostMemoryBuffer buffer, long offset, long len) {
    handle = createReaderWithOutputGranularity(chunkReadLimit, passReadLimit, outputRowSizingGranularity,
        opts.getIncludeColumnNames(), buffer.getAddress() + offset, len,
        opts.usingNumPyTypes(), opts.timeUnit().typeId.getNativeId(),
        opts.getDecimal128Columns(), opts.ignoreTimezoneInStripeFooter(),
        opts.convertDatesToProleptic());
    if (handle == 0) {
      throw new IllegalStateException("Cannot create native chunked ORC reader object.");
    }
  }

  /**
   * Check if the given file has anything left to read.
   *
   * @return A boolean value indicating if there is more data to read from file.
   */
  public boolean hasNext() {
    if (handle == 0) {
      throw new IllegalStateException("Native chunked ORC reader object may have been closed.");
    }

    if (firstCall) {
      // This function needs to return true at least once, so an empty table
      // (but having empty columns instead of no column) can be returned by readChunk()
      // if the input file has no row.
      firstCall = false;
      return true;
    }
    return hasNext(handle);
  }

  /**
   * Read a chunk of rows in the given ORC file such that the returning data has total size
   * does not exceed the given read limit. If the given file has no data, or all data has been read
   * before by previous calls to this function, a null Table will be returned.
   *
   * @return A table of new rows reading from the given file.
   */
  public Table readChunk() {
    if (handle == 0) {
      throw new IllegalStateException("Native chunked ORC reader object may have been closed.");
    }

    long[] columnPtrs = readChunk(handle);
    return columnPtrs != null ? new Table(columnPtrs) : null;
  }

  @Override
  public void close() {
    if (handle != 0) {
      close(handle);
      handle = 0;
    }
  }


  /**
   * Auxiliary variable to help {@link #hasNext()} returning true at least once.
   */
  private boolean firstCall = true;

  /**
   * Handle for memory address of the native ORC chunked reader class.
   */
  private long handle;

  /**
   * Create a native chunked ORC reader object on heap and return its memory address.
   *
   * @param chunkReadLimit    Limit on total number of bytes to be returned per read,
   *                          or 0 if there is no limit.
   * @param passReadLimit     Limit on the amount of memory used by the chunked reader,
   *                          or 0 if there is no limit.
   * @param filterColumnNames Name of the columns to read, or an empty array if we want to read all.
   * @param bufferAddrs       The address of a buffer to read from, or 0 if we are not using that buffer.
   * @param length            The length of the buffer to read from.
   * @param usingNumPyTypes   Whether the parser should implicitly promote TIMESTAMP
   *                          columns to TIMESTAMP_MILLISECONDS for compatibility with NumPy.
   * @param timeUnit          return type of TimeStamp in units
   * @param decimal128Columns name of the columns which are read as Decimal128 rather than Decimal64
   */
  private static native long createReader(long chunkReadLimit, long passReadLimit,
      String[] filterColumnNames, long bufferAddrs, long length,
      boolean usingNumPyTypes, int timeUnit, String[] decimal128Columns,
      boolean ignoreTimezoneInStripeFooter, boolean convertDatesToProleptic);

  /**
   * Create a native chunked ORC reader object, similar to
   * {@link ORCChunkedReader#createReader(long, long, String[], long, long, boolean, int, String[], boolean, boolean)},
   * with an additional parameter to control the granularity of the output table.
   *
   * @param outputRowSizingGranularity The change step in number of rows in the output table.
   * @see ORCChunkedReader#createReader(long, long, String[], long, long, boolean, int, String[], boolean, boolean)
   */
  private static native long createReaderWithOutputGranularity(
      long chunkReadLimit, long passReadLimit, long outputRowSizingGranularity,
      String[] filterColumnNames, long bufferAddrs, long length,
      boolean usingNumPyTypes, int timeUnit, String[] decimal128Columns,
      boolean ignoreTimezoneInStripeFooter, boolean convertDatesToProleptic);

  private static native boolean hasNext(long handle);

  private static native long[] readChunk(long handle);

  private static native void close(long handle);
}

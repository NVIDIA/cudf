/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.io.File;

/**
 * Provide an interface for reading a Parquet file in an iterative manner.
 */
public class ParquetChunkedReader implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Construct the reader instance from a read limit and a file path.
   *
   * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
   *                           or 0 if there is no limit.
   * @param filePath Full path of the input Parquet file to read.
   */
  public ParquetChunkedReader(long chunkSizeByteLimit, File filePath) {
    this(chunkSizeByteLimit, ParquetOptions.DEFAULT, filePath);
  }

  /**
   * Construct the reader instance from a read limit, a ParquetOptions object, and a file path.
   *
   * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
   *                           or 0 if there is no limit.
   * @param opts The options for Parquet reading.
   * @param filePath Full path of the input Parquet file to read.
   */
  public ParquetChunkedReader(long chunkSizeByteLimit, ParquetOptions opts, File filePath) {
    this(chunkSizeByteLimit, 0, opts, filePath);
  }

  /**
   * Construct the reader instance from a read limit, a ParquetOptions object, and a file path.
   *
   * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
   *                           or 0 if there is no limit.
   * @param passReadLimit Limit on the amount of memory used for reading and decompressing data or
   *                      0 if there is no limit
   * @param opts The options for Parquet reading.
   * @param filePath Full path of the input Parquet file to read.
   */
  public ParquetChunkedReader(long chunkSizeByteLimit, long passReadLimit, ParquetOptions opts, File filePath) {
    long[] handles = create(chunkSizeByteLimit, passReadLimit, opts.getIncludeColumnNames(), opts.getReadBinaryAsString(),
        filePath.getAbsolutePath(), null, opts.timeUnit().typeId.getNativeId());
    handle = handles[0];
    if (handle == 0) {
      throw new IllegalStateException("Cannot create native chunked Parquet reader object.");
    }
    multiHostBufferSourceHandle = handles[1];
  }

  /**
   * Construct the reader instance from a read limit and a file already read in a memory buffer.
   *
   * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
   *                           or 0 if there is no limit.
   * @param opts The options for Parquet reading.
   * @param buffer Raw Parquet file content.
   * @param offset The starting offset into buffer.
   * @param len The number of bytes to parse the given buffer.
   */
  public ParquetChunkedReader(long chunkSizeByteLimit, ParquetOptions opts, HostMemoryBuffer buffer,
                              long offset, long len) {
    this(chunkSizeByteLimit, 0L, opts, buffer, offset, len);
  }

  /**
   * Construct the reader instance from a read limit and a file already read in a memory buffer.
   *
   * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
   *                           or 0 if there is no limit.
   * @param passReadLimit Limit on the amount of memory used for reading and decompressing data or
   *                      0 if there is no limit
   * @param opts The options for Parquet reading.
   * @param buffer Raw Parquet file content.
   * @param offset The starting offset into buffer.
   * @param len The number of bytes to parse the given buffer.
   */
  public ParquetChunkedReader(long chunkSizeByteLimit, long passReadLimit,
                              ParquetOptions opts, HostMemoryBuffer buffer,
                              long offset, long len) {
    long[] addrsSizes = new long[]{ buffer.getAddress() + offset, len };
    long[] handles = create(chunkSizeByteLimit,passReadLimit,  opts.getIncludeColumnNames(), opts.getReadBinaryAsString(), null,
        addrsSizes, opts.timeUnit().typeId.getNativeId());
    handle = handles[0];
    if (handle == 0) {
      throw new IllegalStateException("Cannot create native chunked Parquet reader object.");
    }
    multiHostBufferSourceHandle = handles[1];
  }

  /**
   * Construct the reader instance from a read limit and data in host memory buffers.
   *
   * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
   *                           or 0 if there is no limit.
   * @param passReadLimit Limit on the amount of memory used for reading and decompressing data or
   *                      0 if there is no limit
   * @param opts The options for Parquet reading.
   * @param buffers Array of buffers containing the file data. The buffers are logically
   *                concatenated to construct the file being read.
   */
  public ParquetChunkedReader(long chunkSizeByteLimit, long passReadLimit,
                              ParquetOptions opts, HostMemoryBuffer... buffers) {
    long[] addrsSizes = new long[buffers.length * 2];
    for (int i = 0; i < buffers.length; i++) {
      addrsSizes[i * 2] = buffers[i].getAddress();
      addrsSizes[(i * 2) + 1] = buffers[i].getLength();
    }
    long[] handles = create(chunkSizeByteLimit,passReadLimit,  opts.getIncludeColumnNames(), opts.getReadBinaryAsString(), null,
        addrsSizes, opts.timeUnit().typeId.getNativeId());
    handle = handles[0];
    if (handle == 0) {
      throw new IllegalStateException("Cannot create native chunked Parquet reader object.");
    }
    multiHostBufferSourceHandle = handles[1];
  }

  /**
   * Construct a reader instance from a DataSource
   * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
   *                           or 0 if there is no limit.
   * @param opts The options for Parquet reading.
   * @param ds the data source to read from
   */
  public ParquetChunkedReader(long chunkSizeByteLimit, ParquetOptions opts, DataSource ds) {
    dataSourceHandle = DataSourceHelper.createWrapperDataSource(ds);
    if (dataSourceHandle == 0) {
      throw new IllegalStateException("Cannot create native datasource object");
    }

    boolean passed = false;
    try {
      handle = createWithDataSource(chunkSizeByteLimit, opts.getIncludeColumnNames(),
              opts.getReadBinaryAsString(), opts.timeUnit().typeId.getNativeId(),
              dataSourceHandle);
      passed = true;
    } finally {
      if (!passed) {
        DataSourceHelper.destroyWrapperDataSource(dataSourceHandle);
        dataSourceHandle = 0;
      }
    }
  }

  /**
   * Check if the given file has anything left to read.
   *
   * @return A boolean value indicating if there is more data to read from file.
   */
  public boolean hasNext() {
    if (handle == 0) {
      throw new IllegalStateException("Native chunked Parquet reader object may have been closed.");
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
   * Read a chunk of rows in the given Parquet file such that the returning data has total size
   * does not exceed the given read limit. If the given file has no data, or all data has been read
   * before by previous calls to this function, a null Table will be returned.
   *
   * @return A table of new rows reading from the given file.
   */
  public Table readChunk() {
    if (handle == 0) {
      throw new IllegalStateException("Native chunked Parquet reader object may have been closed.");
    }

    long[] columnPtrs = readChunk(handle);
    return columnPtrs != null ? Table.fromRowCountAndColumns(columnPtrs) : null;
  }

  @Override
  public void close() {
    if (handle != 0) {
      close(handle);
      handle = 0;
    }
    if (dataSourceHandle != 0) {
      DataSourceHelper.destroyWrapperDataSource(dataSourceHandle);
      dataSourceHandle = 0;
    }
    if (multiHostBufferSourceHandle != 0) {
      destroyMultiHostBufferSource(multiHostBufferSourceHandle);
      multiHostBufferSourceHandle = 0;
    }
  }


  /**
   * Auxiliary variable to help {@link #hasNext()} returning true at least once.
   */
  private boolean firstCall = true;

  /**
   * Handle for memory address of the native Parquet chunked reader class.
   */
  private long handle;

  private long dataSourceHandle = 0;

  private long multiHostBufferSourceHandle = 0;

  /**
   * Create a native chunked Parquet reader object on heap and return its memory address.
   *
   * @param chunkSizeByteLimit Limit on total number of bytes to be returned per read,
   *                           or 0 if there is no limit.
   * @param passReadLimit Limit on the amount of memory used for reading and decompressing
   *                      data or 0 if there is no limit.
   * @param filterColumnNames Name of the columns to read, or an empty array if we want to read all.
   * @param binaryToString Whether to convert the corresponding column to String if it is binary.
   * @param filePath Full path of the file to read, or given as null if reading from a buffer.
   * @param bufferAddrsSizes The address and size pairs of buffers to read from, or null if we are not using buffers.
   * @param timeUnit Return type of time unit for timestamps.
   */
  private static native long[] create(long chunkSizeByteLimit, long passReadLimit,
                                      String[] filterColumnNames, boolean[] binaryToString,
                                      String filePath, long[] bufferAddrsSizes, int timeUnit);

  private static native long createWithDataSource(long chunkedSizeByteLimit,
      String[] filterColumnNames, boolean[] binaryToString, int timeUnit, long dataSourceHandle);

  private static native boolean hasNext(long handle);

  private static native long[] readChunk(long handle);

  private static native void close(long handle);

  private static native void destroyMultiHostBufferSource(long handle);
}

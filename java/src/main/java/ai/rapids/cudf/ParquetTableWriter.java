/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.io.File;

/** A chunked Parquet writer that can return its footer metadata when closed. */
public final class ParquetTableWriter extends TableWriter {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private final HostBufferConsumer consumer;
  private final HostMemoryAllocator hostMemoryAllocator;

  ParquetTableWriter(ParquetWriterOptions options, File outputFile) {
    super(writeParquetFileBegin(options.getFlatColumnNames(),
        options.getTopLevelChildren(),
        options.getFlatNumChildren(),
        options.getFlatIsNullable(),
        options.getMetadataKeys(),
        options.getMetadataValues(),
        options.getCompressionType().nativeId,
        options.getRowGroupSizeRows(),
        options.getRowGroupSizeBytes(),
        options.getMaxDictionarySize(),
        options.getDictionaryPolicy().nativeId,
        options.getStatisticsFrequency().nativeId,
        options.getFlatIsTimeTypeInt96(),
        options.getFlatPrecision(),
        options.getFlatIsMap(),
        options.getFlatIsBinary(),
        options.getFlatHasParquetFieldId(),
        options.getFlatParquetFieldId(),
        outputFile.getAbsolutePath()));
    this.consumer = null;
    this.hostMemoryAllocator = DefaultHostMemoryAllocator.get();
  }

  ParquetTableWriter(ParquetWriterOptions options, HostBufferConsumer consumer,
      HostMemoryAllocator hostMemoryAllocator) {
    super(writeParquetBufferBegin(options.getFlatColumnNames(),
        options.getTopLevelChildren(),
        options.getFlatNumChildren(),
        options.getFlatIsNullable(),
        options.getMetadataKeys(),
        options.getMetadataValues(),
        options.getCompressionType().nativeId,
        options.getRowGroupSizeRows(),
        options.getRowGroupSizeBytes(),
        options.getMaxDictionarySize(),
        options.getDictionaryPolicy().nativeId,
        options.getStatisticsFrequency().nativeId,
        options.getFlatIsTimeTypeInt96(),
        options.getFlatPrecision(),
        options.getFlatIsMap(),
        options.getFlatIsBinary(),
        options.getFlatHasParquetFieldId(),
        options.getFlatParquetFieldId(),
        consumer, hostMemoryAllocator));
    this.consumer = consumer;
    this.hostMemoryAllocator = hostMemoryAllocator;
  }

  @Override
  public void write(Table table) {
    write(table.getNativeView(), table.getDeviceMemorySize());
  }

  // Used by Table.writeColumnViewsToParquet, which only has a native table-view handle.
  void write(long tableHandle, long tableMemSize) {
    if (writerHandle == 0) {
      throw new IllegalStateException("Writer was already closed");
    }
    writeParquetChunk(writerHandle, tableHandle, tableMemSize);
  }

  @Override
  public void close() throws CudfException {
    if (writerHandle != 0) {
      finish(false);
    }
  }

  /**
   * Finish writing and return the Parquet footer metadata.
   *
   * <p>The returned buffer is a metadata-only Parquet file containing the leading {@code PAR1}
   * magic, the serialized file metadata, the footer length, and the trailing {@code PAR1}
   * magic. The caller must close the returned buffer.
   *
   * <p>This is a complete metadata-only Parquet file rather than raw footer bytes. Callers can
   * pass it to a Parquet reader to obtain row-group metadata, offsets, and column metrics without
   * rereading the output file. libcudf creates the buffer from the writer's in-memory metadata
   * during successful finalization; the downstream Parquet reader is responsible for validating
   * the serialized metadata it consumes.
   *
   * @return an owned host buffer containing the Parquet footer metadata
   * @throws CudfException if finalizing the Parquet writer fails
   * @throws IllegalStateException if the writer was already closed
   */
  public HostMemoryBuffer closeAndGetFooter() throws CudfException {
    if (writerHandle == 0) {
      throw new IllegalStateException("Writer was already closed");
    }
    return finish(true);
  }

  private HostMemoryBuffer finish(boolean returnFooter) {
    long handle = writerHandle;
    writerHandle = 0;
    HostMemoryBuffer footer = null;
    try {
      if (returnFooter) {
        footer = writeParquetEndAndGetFooter(handle, hostMemoryAllocator);
      } else {
        writeParquetEnd(handle);
      }
    } catch (RuntimeException | Error e) {
      if (consumer != null) {
        try {
          consumer.done();
        } catch (RuntimeException | Error doneError) {
          e.addSuppressed(doneError);
        }
      }
      throw e;
    }

    try {
      if (consumer != null) {
        consumer.done();
      }
    } catch (RuntimeException | Error e) {
      CleanupHelpers.closeAndSuppress(footer, e);
      throw e;
    }
    return footer;
  }

  private static native long writeParquetFileBegin(String[] columnNames,
                                                    int numChildren,
                                                    int[] flatNumChildren,
                                                    boolean[] nullable,
                                                    String[] metadataKeys,
                                                    String[] metadataValues,
                                                    int compression,
                                                    int rowGroupSizeRows,
                                                    long rowGroupSizeBytes,
                                                    long maxDictionarySize,
                                                    int dictionaryPolicy,
                                                    int statsFreq,
                                                    boolean[] isInt96,
                                                    int[] precisions,
                                                    boolean[] isMapValues,
                                                    boolean[] isBinaryValues,
                                                    boolean[] hasParquetFieldIds,
                                                    int[] parquetFieldIds,
                                                    String filename) throws CudfException;

  private static native long writeParquetBufferBegin(String[] columnNames,
                                                      int numChildren,
                                                      int[] flatNumChildren,
                                                      boolean[] nullable,
                                                      String[] metadataKeys,
                                                      String[] metadataValues,
                                                      int compression,
                                                      int rowGroupSizeRows,
                                                      long rowGroupSizeBytes,
                                                      long maxDictionarySize,
                                                      int dictionaryPolicy,
                                                      int statsFreq,
                                                      boolean[] isInt96,
                                                      int[] precisions,
                                                      boolean[] isMapValues,
                                                      boolean[] isBinaryValues,
                                                      boolean[] hasParquetFieldIds,
                                                      int[] parquetFieldIds,
                                                      HostBufferConsumer consumer,
                                                      HostMemoryAllocator hostMemoryAllocator)
      throws CudfException;

  private static native void writeParquetChunk(long handle, long table, long tableMemSize);

  private static native void writeParquetEnd(long handle);

  private static native HostMemoryBuffer writeParquetEndAndGetFooter(long handle,
      HostMemoryAllocator hostMemoryAllocator);
}

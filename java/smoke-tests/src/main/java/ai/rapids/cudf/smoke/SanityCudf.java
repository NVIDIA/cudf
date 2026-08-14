/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf.smoke;

import ai.rapids.cudf.BinaryOp;
import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DType;
import ai.rapids.cudf.DefaultHostMemoryAllocator;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostBufferConsumer;
import ai.rapids.cudf.HostColumnVector;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.MultiBufferDataSource;
import ai.rapids.cudf.ParquetChunkedReader;
import ai.rapids.cudf.ParquetOptions;
import ai.rapids.cudf.ParquetWriterOptions;
import ai.rapids.cudf.Scalar;
import ai.rapids.cudf.Table;
import ai.rapids.cudf.TableWriter;
import ai.rapids.cudf.nvcomp.BatchedLZ4Compressor;
import ai.rapids.cudf.nvcomp.BatchedLZ4Decompressor;

/**
 * Core cudf-java smoke test exercised once per published classifier JAR.
 * Includes a small nvcomp LZ4 round-trip so static-libcudf JARs exercise
 * nvcomp symbols linked into libcudf (not a separate libnvcomp.so), and a
 * chunked Parquet DataSource read that triggers CUDF_LOG_WARN (native
 * rapids_logger / spdlog path) when pass_read_limit is derived.
 */
public final class SanityCudf {
  private SanityCudf() {}

  private static void step(int n, String label) {
    System.out.printf("[%d/9] %s ...%n", n, label);
  }

  private static void ok(String label) {
    System.out.println("OK: " + label);
  }

  private static void check(boolean cond, String msg) {
    if (!cond) {
      throw new IllegalStateException("ASSERT: " + msg);
    }
  }

  /** Collect parquet bytes written via HostBufferConsumer. */
  private static final class CollectingConsumer implements HostBufferConsumer, AutoCloseable {
    private final HostMemoryBuffer buffer = HostMemoryBuffer.allocate(1024 * 1024);
    private long offset = 0;

    @Override
    public void handleBuffer(HostMemoryBuffer src, long len) {
      try {
        buffer.copyFromHostBuffer(offset, src, 0, len);
        offset += len;
      } finally {
        src.close();
      }
    }

    long length() {
      return offset;
    }

    HostMemoryBuffer buffer() {
      return buffer;
    }

    @Override
    public void close() {
      buffer.close();
    }
  }

  /**
   * Write a tiny parquet table, then open it with the DataSource-only
   * ParquetChunkedReader ctor (chunk limit set, no pass limit). That native
   * path calls derive_pass_read_limit() and emits CUDF_LOG_WARN - exercising
   * the linked rapids_logger/spdlog stack without needing external libspdlog.
   */
  private static void parquetChunkedLoggerSmoke() {
    ParquetWriterOptions writeOpts =
        ParquetWriterOptions.builder().withColumns(false, "a").build();
    try (CollectingConsumer consumer = new CollectingConsumer()) {
      try (ColumnVector col = ColumnVector.fromInts(1, 2, 3, 4, 5);
           Table table = new Table(col);
           TableWriter writer = Table.writeParquetChunked(writeOpts, consumer)) {
        writer.write(table);
      }
      check(consumer.length() > 0, "expected non-empty parquet bytes");
      try (HostMemoryBuffer slice = consumer.buffer().slice(0, consumer.length());
           MultiBufferDataSource ds = new MultiBufferDataSource(slice);
           ParquetChunkedReader reader =
               new ParquetChunkedReader(64 * 1024L, ParquetOptions.DEFAULT, ds)) {
        long rows = 0;
        while (reader.hasNext()) {
          try (Table chunk = reader.readChunk()) {
            if (chunk != null) {
              rows += chunk.getRowCount();
            }
          }
        }
        check(rows == 5, "expected 5 rows from chunked parquet logger smoke");
      }
    }
  }

  /** Minimal LZ4 compress/decompress round-trip via ai.rapids.cudf.nvcomp. */
  private static void nvcompLz4RoundTrip() {
    final long chunkSize = 64 * 1024;
    final Cuda.Stream stream = Cuda.DEFAULT_STREAM;
    final long[] data = new long[4096];
    for (int i = 0; i < data.length; i++) {
      data[i] = i;
    }

    DeviceMemoryBuffer original = null;
    DeviceMemoryBuffer[] compressed = null;
    DeviceMemoryBuffer decompressed = null;
    try (HostMemoryBuffer hostIn =
             DefaultHostMemoryAllocator.get().allocate(data.length * 8L)) {
      hostIn.setLongs(0, data, 0, data.length);
      original = DeviceMemoryBuffer.allocate(hostIn.getLength());
      original.copyFromHostBuffer(hostIn);
      // compress() takes ownership / closes inputs; keep a live ref for compare.
      original.incRefCount();

      BatchedLZ4Compressor comp =
          new BatchedLZ4Compressor(chunkSize, Long.MAX_VALUE);
      compressed = comp.compress(new DeviceMemoryBuffer[]{original}, stream);
      check(compressed != null && compressed.length == 1, "expected 1 compressed buffer");
      check(compressed[0] != null && compressed[0].getLength() > 0,
          "compressed buffer should be non-empty");

      decompressed = DeviceMemoryBuffer.allocate(hostIn.getLength());
      BatchedLZ4Decompressor decomp = new BatchedLZ4Decompressor(chunkSize);
      // decompressAsync takes ownership of compressed buffers.
      decomp.decompressAsync(compressed, new DeviceMemoryBuffer[]{decompressed}, stream);
      compressed = null; // owned/closed by decompressAsync
      stream.sync();

      try (HostMemoryBuffer hostOut =
               DefaultHostMemoryAllocator.get().allocate(decompressed.getLength())) {
        hostOut.copyFromDeviceBuffer(decompressed);
        check(hostOut.getLength() == hostIn.getLength(), "decompressed size mismatch");
        for (int i = 0; i < data.length; i++) {
          check(hostOut.getLong(i * 8L) == data[i], "nvcomp mismatch at long[" + i + "]");
        }
      }
    } finally {
      if (original != null) {
        original.close();
      }
      if (compressed != null) {
        for (DeviceMemoryBuffer b : compressed) {
          if (b != null) {
            b.close();
          }
        }
      }
      if (decompressed != null) {
        decompressed.close();
      }
    }
  }

  public static void main(String[] args) {
    step(1, "Native deps load");
    try (ColumnVector ints = ColumnVector.fromInts(1, 2, 3, 4, 5)) {
      check(ints.getRowCount() == 5, "expected 5 rows after fromInts");
      ok("Native deps load");

      step(2, "ColumnVector + Table");
      try (ColumnVector more = ColumnVector.fromInts(10, 20, 30, 40, 50);
           Table table = new Table(ints, more)) {
        check(table.getNumberOfColumns() == 2, "expected 2 columns");
        check(table.getRowCount() == 5, "expected 5 table rows");
        ok("ColumnVector + Table");
      }

      step(3, "Filter");
      try (Scalar three = Scalar.fromInt(3);
           ColumnVector mask = ints.binaryOp(BinaryOp.GREATER, three, DType.BOOL8);
           Table table = new Table(ints);
           Table filtered = table.filter(mask)) {
        check(filtered.getRowCount() == 2, "expected 2 rows after filter (>3)");
        ok("Filter");
      }

      step(4, "Aggregation");
      try (Scalar sum = ints.sum(DType.INT64)) {
        check(sum.isValid(), "sum scalar should be valid");
        check(sum.getLong() == 15L, "sum should be 15");
        ok("Aggregation (sum)");
      }

      step(5, "String column create + length");
      try (ColumnVector strs = ColumnVector.fromStrings("a", "bb", "ccc");
           ColumnVector lengths = strs.getCharLengths();
           HostColumnVector hostLens = lengths.copyToHost()) {
        check(strs.getRowCount() == 3, "expected 3 string rows");
        check(hostLens.getInt(0) == 1, "len[0]==1");
        check(hostLens.getInt(1) == 2, "len[1]==2");
        check(hostLens.getInt(2) == 3, "len[2]==3");
        ok("String column create + length");
      }

      step(6, "Host round-trip");
      try (HostColumnVector host = ints.copyToHost()) {
        check(host.getInt(0) == 1, "host[0]==1");
        check(host.getInt(4) == 5, "host[4]==5");
        ok("Host round-trip");
      }

      step(7, "nvcomp LZ4 round-trip");
      nvcompLz4RoundTrip();
      ok("nvcomp LZ4 round-trip");

      step(8, "Parquet chunked logger smoke");
      parquetChunkedLoggerSmoke();
      ok("Parquet chunked logger smoke (CUDF_LOG_WARN path)");

      step(9, "Resource close");
      ok("Clean close via try-with-resources");
    }
    System.out.println("ALL STEPS PASSED");
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.nvcomp;

import ai.rapids.cudf.*;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

public class NvcompTest {
  private static final HostMemoryAllocator hostMemoryAllocator = DefaultHostMemoryAllocator.get();

  private static final Logger log = LoggerFactory.getLogger(ColumnVector.class);

  private final long chunkSize = 64 * 1024;
  private final long targetIntermediteSize = Long.MAX_VALUE;

  @Test
  void testBatchedLZ4RoundTripAsync() {
    testBatchedRoundTripAsync(new BatchedLZ4Compressor(chunkSize, targetIntermediteSize),
        new BatchedLZ4Decompressor(chunkSize), Cuda.DEFAULT_STREAM);
  }

  @Test
  void testBatchedZstdRoundTripAsync() {
    testBatchedRoundTripAsync(new BatchedZstdCompressor(chunkSize, targetIntermediteSize),
        new BatchedZstdDecompressor(chunkSize), Cuda.DEFAULT_STREAM);
  }

  @Test
  void testBatchedLZ4RoundTripOnNonDefaultStream() {
    try (Cuda.Stream stream = new Cuda.Stream(true)) {
      testBatchedRoundTripAsync(new BatchedLZ4Compressor(chunkSize, targetIntermediteSize),
          new BatchedLZ4Decompressor(chunkSize), stream);
    }
  }

  void testBatchedRoundTripAsync(BatchedCompressor comp, BatchedDecompressor decomp,
      Cuda.Stream stream) {
    final int maxElements = 1024 * 1024 + 1;
    final int numBuffers = 200;
    long[] data = new long[maxElements];
    for (int i = 0; i < maxElements; ++i) {
      data[i] = i;
    }

    try (CloseableArray<DeviceMemoryBuffer> originalBuffers =
             CloseableArray.wrap(new DeviceMemoryBuffer[numBuffers])) {
      // create the batched buffers to compress
      for (int i = 0; i < originalBuffers.size(); i++) {
        originalBuffers.set(i, initBatchBuffer(data, i, stream));
        // Increment the refcount since compression will try to close it
        originalBuffers.get(i).incRefCount();
      }

      // compress and decompress the buffers
      try (CloseableArray<DeviceMemoryBuffer> compressedBuffers =
               CloseableArray.wrap(comp.compress(originalBuffers.getArray(), stream));
           CloseableArray<DeviceMemoryBuffer> uncompressedBuffers =
               CloseableArray.wrap(new DeviceMemoryBuffer[numBuffers])) {
        for (int i = 0; i < numBuffers; i++) {
          uncompressedBuffers.set(i,
              DeviceMemoryBuffer.allocate(originalBuffers.get(i).getLength(), stream));
        }

        // decompress takes ownership of the compressed buffers and will close them
        decomp.decompressAsync(compressedBuffers.release(), uncompressedBuffers.getArray(),
            stream);
        stream.sync();

        // check the decompressed results against the original
        for (int i = 0; i < numBuffers; ++i) {
          try (HostMemoryBuffer expected =
                   hostMemoryAllocator.allocate(originalBuffers.get(i).getLength());
               HostMemoryBuffer actual =
                   hostMemoryAllocator.allocate(uncompressedBuffers.get(i).getLength())) {
            Assertions.assertTrue(expected.getLength() <= Integer.MAX_VALUE);
            Assertions.assertTrue(actual.getLength() <= Integer.MAX_VALUE);
            Assertions.assertEquals(expected.getLength(), actual.getLength(),
                "uncompressed size mismatch at buffer " + i);
            expected.copyFromDeviceBuffer(originalBuffers.get(i), stream);
            actual.copyFromDeviceBuffer(uncompressedBuffers.get(i), stream);
            byte[] expectedBytes = new byte[(int) expected.getLength()];
            expected.getBytes(expectedBytes, 0, 0, expected.getLength());
            byte[] actualBytes = new byte[(int) actual.getLength()];
            actual.getBytes(actualBytes, 0, 0, actual.getLength());
            Assertions.assertArrayEquals(expectedBytes, actualBytes,
                "mismatch in batch buffer " + i);
          }
        }
      }
    }
  }

  private void closeBuffer(MemoryBuffer buffer) {
    if (buffer != null) {
      buffer.close();
    }
  }

  private DeviceMemoryBuffer initBatchBuffer(long[] data, int bufferId, Cuda.Stream stream) {
    // grab a subsection of the data based on buffer ID
    int dataStart = 0;
    int dataLength = data.length / (bufferId + 1);
    switch (bufferId % 3) {
      case 0:
        // take a portion of the first half
        dataLength /= 2;
        break;
      case 1:
        // take a portion of the last half
        dataStart = data.length / 2;
        dataLength /= 2;
        break;
      default:
        break;
    }
    long[] bufferData = Arrays.copyOfRange(data, dataStart, dataStart + dataLength + 1);
    DeviceMemoryBuffer devBuffer = null;
    try (HostMemoryBuffer hmb = hostMemoryAllocator.allocate(bufferData.length * 8)) {
      hmb.setLongs(0, bufferData, 0, bufferData.length);
      devBuffer = DeviceMemoryBuffer.allocate(hmb.getLength(), stream);
      devBuffer.copyFromHostBuffer(hmb, stream);
      return devBuffer;
    } catch (Throwable t) {
      closeBuffer(devBuffer);
      throw new RuntimeException(t);
    }
  }
}

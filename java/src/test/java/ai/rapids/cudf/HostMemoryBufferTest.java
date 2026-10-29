/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.AfterEach;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class HostMemoryBufferTest extends CudfTestBase {
  @AfterEach
  void teardown() {
    if (PinnedMemoryPool.isInitialized()) {
      PinnedMemoryPool.shutdown();
    }
  }

  @Test
  void testRefCountLeak() throws InterruptedException {
    assumeTrue(Boolean.getBoolean("ai.rapids.cudf.flaky-tests-enabled"));
    long expectedLeakCount = MemoryCleaner.leakCount.get() + 1;
    HostMemoryBuffer.allocate(1);
    long maxTime = System.currentTimeMillis() + 10_000;
    long leakNow;
    do {
      System.gc();
      Thread.sleep(50);
      leakNow = MemoryCleaner.leakCount.get();
    } while (leakNow != expectedLeakCount && System.currentTimeMillis() < maxTime);
    assertEquals(expectedLeakCount, MemoryCleaner.leakCount.get());
  }

  @Test
  void asByteBuffer() {
    final long size = 1024;
    try (HostMemoryBuffer buff = HostMemoryBuffer.allocate(size)) {
      ByteBuffer dbuff = buff.asByteBuffer();
      assertEquals(size, dbuff.capacity());
      assertEquals(ByteOrder.nativeOrder(), dbuff.order());
      dbuff.putInt(101);
      dbuff.putDouble(101.1);
      assertEquals(101, buff.getInt(0));
      assertEquals(101.1, buff.getDouble(4));
    }
  }

  @Test
  void testDoubleFree() {
    HostMemoryBuffer buffer = HostMemoryBuffer.allocate(1);
    buffer.close();
    assertThrows(IllegalStateException.class, () -> buffer.close() );
  }

  @Test
  public void testGetInt() {
    try (HostMemoryBuffer hostMemoryBuffer = HostMemoryBuffer.allocate(16)) {
      long offset = 1;
      hostMemoryBuffer.setInt(offset * DType.INT32.getSizeInBytes(), 2);
      assertEquals(2, hostMemoryBuffer.getInt(offset * DType.INT32.getSizeInBytes()));
    }
  }

  @Test
  public void testGetInts() {
    try (HostMemoryBuffer hostMemoryBuffer = HostMemoryBuffer.allocate(16)) {
      hostMemoryBuffer.setInt(0, 1);
      hostMemoryBuffer.setInt(4, 2);
      hostMemoryBuffer.setInt(8, 3);
      hostMemoryBuffer.setInt(12, 4);
      int[] expectedInts = new int[] {1, 2, 3, 4};
      int[] result = new int[expectedInts.length];
      hostMemoryBuffer.getInts(result, 0, 0, 4);
      assertArrayEquals(expectedInts, result);
    }
  }

  @Test
  public void testGetByte() {
    try (HostMemoryBuffer hostMemoryBuffer = HostMemoryBuffer.allocate(16)) {
      long offset = 1;
      hostMemoryBuffer.setByte(offset * DType.INT8.getSizeInBytes(), (byte) 2);
      assertEquals((byte) 2, hostMemoryBuffer.getByte(offset * DType.INT8.getSizeInBytes()));
    }
  }

  @Test
  public void testGetLong() {
    try (HostMemoryBuffer hostMemoryBuffer = HostMemoryBuffer.allocate(16)) {
      long offset = 1;
      hostMemoryBuffer.setLong(offset * DType.INT64.getSizeInBytes(), 3);
      assertEquals(3, hostMemoryBuffer.getLong(offset * DType.INT64.getSizeInBytes()));
    }
  }

  @Test
  public void testGetLongs() {
    try (HostMemoryBuffer hostMemoryBuffer = HostMemoryBuffer.allocate(16)) {
      hostMemoryBuffer.setLong(0, 3);
      hostMemoryBuffer.setLong(DType.INT64.getSizeInBytes(), 10);
      long[] results = new long[2];
      hostMemoryBuffer.getLongs(results, 0, 0, 2);
      assertEquals(3, results[0]);
      assertEquals(10, results[1]);
    }
  }

  @Test
  public void testGetLength() {
    try (HostMemoryBuffer hostMemoryBuffer = HostMemoryBuffer.allocate(16)) {
      long length = hostMemoryBuffer.getLength();
      assertEquals(16, length);
    }
  }

  @Test
  public void testCopyFromDeviceBuffer() {
    try (HostMemoryBuffer init = HostMemoryBuffer.allocate(16);
         DeviceMemoryBuffer tmp = DeviceMemoryBuffer.allocate(16);
         HostMemoryBuffer to = HostMemoryBuffer.allocate(16)) {
      init.setLong(0, 123456789);
      tmp.copyFromHostBuffer(init);
      to.copyFromDeviceBuffer(tmp);
      assertEquals(123456789, to.getLong(0));
    }
  }

  @Test
  public void testFilemap() throws Exception {
    Random random = new Random(12345L);
    final int pageSize = UnsafeMemoryAccessor.pageSize();
    final int bufferSize = pageSize * 5;
    byte[] testbuf = new byte[bufferSize];
    random.nextBytes(testbuf);
    Path tempFile = Files.createTempFile("mmaptest", ".data");
    try {
      Files.write(tempFile, testbuf);

      // verify we can map the whole file
      try (HostMemoryBuffer hmb = HostMemoryBuffer.mapFile(tempFile.toFile(),
          FileChannel.MapMode.READ_ONLY, 0,bufferSize)) {
        assertEquals(bufferSize, hmb.length);
        byte[] bytes = new byte[(int) hmb.length];
        hmb.getBytes(bytes, 0, 0, hmb.length);
        assertArrayEquals(testbuf, bytes);
      }

      // verify we can map at offsets that aren't a page boundary
      int mapOffset = pageSize + 1;
      int mapLength = pageSize * 2 + 7;
      try (HostMemoryBuffer hmb = HostMemoryBuffer.mapFile(tempFile.toFile(),
          FileChannel.MapMode.READ_ONLY, mapOffset, mapLength)) {
        assertEquals(mapLength, hmb.length);
        byte[] expected = Arrays.copyOfRange(testbuf, mapOffset, mapOffset + mapLength);
        byte[] bytes = new byte[(int) hmb.length];
        hmb.getBytes(bytes, 0, 0, hmb.length);
        assertArrayEquals(expected, bytes);
      }

      // verify we can modify the file via a writable mapping
      mapOffset = pageSize * 3 + 123;
      mapLength = bufferSize - mapOffset - 456;
      byte[] newData = new byte[mapLength];
      random.nextBytes(newData);
      try (HostMemoryBuffer hmb = HostMemoryBuffer.mapFile(tempFile.toFile(),
          FileChannel.MapMode.READ_WRITE, mapOffset, mapLength)) {
        hmb.setBytes(0, newData, 0, newData.length);
      }
      byte[] data = Files.readAllBytes(tempFile);
      System.arraycopy(newData, 0, testbuf, mapOffset, mapLength);
      assertArrayEquals(testbuf, data);
    } finally {
      Files.delete(tempFile);
    }
  }

  public static void initPinnedPoolIfNeeded(long size) {
    long available = PinnedMemoryPool.getTotalPoolSizeBytes();
    if (available < size) {
      if (PinnedMemoryPool.isInitialized()) {
        PinnedMemoryPool.shutdown();
      }
      PinnedMemoryPool.initialize(size + 2048);
    }
  }

  public static byte[] rba(int size, long seed) {
    Random random = new Random(12345L);
    byte[] data = new byte[size];
    random.nextBytes(data);
    return data;
  }

  public static byte[] rba(int size) {
    return rba(size, 12345L);
  }

  @Test
  public void testCopyWithStream() {
    long length = 1 * 1024 * 1024;
    initPinnedPoolIfNeeded(length * 2);
    byte[] data = rba((int)length);
    byte[] result = new byte[data.length];
    try (Cuda.Stream stream1 = new Cuda.Stream(true);
         Cuda.Stream stream2 = new Cuda.Stream(true);
         HostMemoryBuffer hostBuffer = PinnedMemoryPool.allocate(data.length);
         DeviceMemoryBuffer devBuffer = DeviceMemoryBuffer.allocate(data.length);
         HostMemoryBuffer hostBuffer2 = PinnedMemoryPool.allocate(data.length)) {
      hostBuffer.setBytes(0, data, 0, data.length);
      devBuffer.copyFromHostBuffer(hostBuffer, stream1);
      hostBuffer2.copyFromDeviceBuffer(devBuffer, stream2);
      hostBuffer2.getBytes(result, 0, 0, result.length);
      assertArrayEquals(data, result);
    }
  }

  @Test
  public void deviceCopyHandlerCanDeclineAllCopyMethods() {
    byte[] expected = new byte[] {1, 2, 3, 4};
    AtomicInteger attempts = new AtomicInteger();
    try (Cuda.Stream stream = new Cuda.Stream(true);
         HostMemoryBuffer input = HostMemoryBuffer.allocateRaw(expected.length);
         DeviceMemoryBuffer device = DeviceMemoryBuffer.allocate(expected.length);
         HostMemoryBuffer output = HostMemoryBuffer.allocateRaw(expected.length)) {
      HostMemoryBuffer.setDeviceToHostCopyHandler((dst, dstOffset, src, srcOffset,
          length, copyStream) -> {
        if (dst == output) {
          attempts.incrementAndGet();
        }
        return false;
      });
      input.setBytes(0, expected, 0, expected.length);
      device.copyFromHostBuffer(input);
      byte[] actual = new byte[expected.length];
      output.copyFromDeviceBuffer(device);
      output.getBytes(actual, 0, 0, actual.length);
      assertArrayEquals(expected, actual);
      output.copyFromDeviceBuffer(device, stream);
      output.getBytes(actual, 0, 0, actual.length);
      assertArrayEquals(expected, actual);
      output.copyFromDeviceBufferAsync(device, stream);
      stream.sync();
      output.getBytes(actual, 0, 0, actual.length);
      assertArrayEquals(expected, actual);
      assertEquals(3, attempts.get());
    } finally {
      HostMemoryBuffer.setDeviceToHostCopyHandler(null);
    }
  }

  @Test
  public void simpleEventTest() {
    long length = 1 * 1024 * 1024;
    initPinnedPoolIfNeeded(length * 2);
    byte[] data = rba((int)length);
    byte[] result = new byte[data.length];
    try (Cuda.Stream stream1 = new Cuda.Stream(true);
         Cuda.Stream stream2 = new Cuda.Stream(true);
         Cuda.Event event1 = new Cuda.Event();
         Cuda.Event event2 = new Cuda.Event();
         HostMemoryBuffer hostBuffer = PinnedMemoryPool.allocate(data.length);
         DeviceMemoryBuffer devBuffer = DeviceMemoryBuffer.allocate(data.length);
         HostMemoryBuffer hostBuffer2 = PinnedMemoryPool.allocate(data.length)) {
      hostBuffer.setBytes(0, data, 0, data.length);
      devBuffer.copyFromHostBufferAsync(hostBuffer, stream1);
      event1.record(stream1);
      stream2.waitOn(event1);
      hostBuffer2.copyFromDeviceBufferAsync(devBuffer, stream2);
      event2.record(stream2);
      event2.sync();
      hostBuffer2.getBytes(result, 0, 0, result.length);
      assertArrayEquals(data, result);
    }
  }

  @Test
  public void simpleEventQueryTest() throws InterruptedException {
    long length = 1 * 1024 * 1024;
    initPinnedPoolIfNeeded(length * 2);
    byte[] data = rba((int)length);
    byte[] result = new byte[data.length];
    try (Cuda.Stream stream1 = new Cuda.Stream(true);
         Cuda.Stream stream2 = new Cuda.Stream(true);
         Cuda.Event event1 = new Cuda.Event();
         Cuda.Event event2 = new Cuda.Event();
         HostMemoryBuffer hostBuffer = PinnedMemoryPool.allocate(data.length);
         DeviceMemoryBuffer devBuffer = DeviceMemoryBuffer.allocate(data.length);
         HostMemoryBuffer hostBuffer2 = PinnedMemoryPool.allocate(data.length)) {
      hostBuffer.setBytes(0, data, 0, data.length);
      devBuffer.copyFromHostBufferAsync(hostBuffer, stream1);
      event1.record(stream1);
      stream2.waitOn(event1);
      hostBuffer2.copyFromDeviceBufferAsync(devBuffer, stream2);
      event2.record(stream2);
      while (!event2.hasCompleted()) {
        Thread.sleep(100);
      }
      hostBuffer2.getBytes(result, 0, 0, result.length);
      assertArrayEquals(data, result);
    }
  }

  @Test
  public void simpleStreamSynchTest() {
    long length = 1 * 1024 * 1024;
    initPinnedPoolIfNeeded(length * 2);
    byte[] data = rba((int)length);
    byte[] result = new byte[data.length];
    try (Cuda.Stream stream1 = new Cuda.Stream(true);
         Cuda.Stream stream2 = new Cuda.Stream(true);
         HostMemoryBuffer hostBuffer = PinnedMemoryPool.allocate(data.length);
         DeviceMemoryBuffer devBuffer = DeviceMemoryBuffer.allocate(data.length);
         HostMemoryBuffer hostBuffer2 = PinnedMemoryPool.allocate(data.length)) {
      hostBuffer.setBytes(0, data, 0, data.length);
      devBuffer.copyFromHostBufferAsync(hostBuffer, stream1);
      stream1.sync();
      hostBuffer2.copyFromDeviceBufferAsync(devBuffer, stream2);
      stream2.sync();
      hostBuffer2.getBytes(result, 0, 0, result.length);
      assertArrayEquals(data, result);
    }
  }
}

/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.nio.ByteBuffer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.jupiter.api.Assertions.*;

class PinnedMemoryPoolTest extends CudfTestBase {
  private static final Logger log = LoggerFactory.getLogger(PinnedMemoryPoolTest.class);

  @AfterEach
  void teardown() {
    if (PinnedMemoryPool.isInitialized()) {
      PinnedMemoryPool.shutdown();
    }
  }

  @Test
  void init() {
    assertFalse(PinnedMemoryPool.isInitialized());
    PinnedMemoryPool.initialize(1024*1024*500L);
    assertTrue(PinnedMemoryPool.isInitialized());
    PinnedMemoryPool.shutdown();
    assertFalse(PinnedMemoryPool.isInitialized());
  }

  @Test
  void initWithParallelFirstTouch() {
    final long poolSize = 16L * 1024 * 1024;
    assertFalse(PinnedMemoryPool.isInitialized());
    PinnedMemoryPool.initialize(poolSize, 0, true, 4);
    assertTrue(PinnedMemoryPool.isInitialized());
    assertEquals(poolSize, PinnedMemoryPool.getTotalPoolSizeBytes());

    try (HostMemoryBuffer buffer = PinnedMemoryPool.tryAllocate(poolSize)) {
      assertNotNull(buffer);
      buffer.setByte(0, (byte) 0x5a);
      assertEquals((byte) 0x5a, buffer.getByte(0));
      assertNull(PinnedMemoryPool.tryAllocate(1));
    }

    long fallbackPtr = Rmm.allocFromFallbackPinnedPool(1024);
    Rmm.freeFromFallbackPinnedPool(fallbackPtr, 1024);
  }

  @Test
  void initParallelPoolWithNonPageAlignedSize() {
    final long poolSize = 16L * 1024 * 1024 + 256;
    PinnedMemoryPool.initialize(poolSize, 0, true, 4);
    assertEquals(poolSize, PinnedMemoryPool.getTotalPoolSizeBytes());

    try (HostMemoryBuffer buffer = PinnedMemoryPool.tryAllocate(poolSize)) {
      assertNotNull(buffer);
      buffer.setByte(poolSize - 1, (byte) 0x5a);
      assertEquals((byte) 0x5a, buffer.getByte(poolSize - 1));
      assertNull(PinnedMemoryPool.tryAllocate(1));
    }
  }

  @Test
  void validateInitializationArguments() {
    assertThrows(IllegalArgumentException.class,
        () -> PinnedMemoryPool.initialize(16L * 1024 * 1024, 0, false, 0));
    assertFalse(PinnedMemoryPool.isInitialized());
  }

  @Test
  void failedInitFallsBackAndCanRetry() {
    // Negative pool size to intentionally fail initialization.
    PinnedMemoryPool.initialize(-1, 0, false, 2);
    try (HostMemoryBuffer buffer = PinnedMemoryPool.allocate(1024)) {
      assertNotNull(buffer);  // this falls back
    }
    assertFalse(PinnedMemoryPool.isInitialized());
    // Initialization can be retried.
    PinnedMemoryPool.initialize(16L * 1024 * 1024, 0, false, 2);
    assertTrue(PinnedMemoryPool.isInitialized());
  }

  @Test
  void allocate() {
    PinnedMemoryPool.initialize(1024*1024*500L);
    for (int i = 2048000; i < 1024*1024*1024; i = i * 2) {
      log.warn("STARTING TEST FOR size = " + i);
      HostMemoryBuffer buff = null;
      HostMemoryBuffer buff2 = null;
      HostMemoryBuffer buff3 = null;
      try {
        buff = PinnedMemoryPool.allocate(i);
        assertEquals(i, buff.length);
        buff2 = PinnedMemoryPool.allocate(i / 2);
        assertEquals(i/2, buff2.length);
        buff.close();
        buff = null;
        buff3 = PinnedMemoryPool.allocate(i * 2);
        assertEquals(i * 2, buff3.length);
      } finally {
        if (buff != null) {
          buff.close();
        }
        if (buff3 != null) {
          buff3.close();
        }
        if (buff2 != null) {
          buff2.close();
        }
      }
      log.warn("DONE TEST FOR size = " + i + "\n");
    }
  }

  @Test
  void testFragmentationAndExhaustion() {
    final long poolSize = 15 * 1024L;
    PinnedMemoryPool.initialize(poolSize);
    assertEquals(poolSize, PinnedMemoryPool.getTotalPoolSizeBytes());
    HostMemoryBuffer[] buffers = new HostMemoryBuffer[5];
    try {
      buffers[0] = PinnedMemoryPool.tryAllocate(1024);
      assertNotNull(buffers[0]);
      buffers[1] = PinnedMemoryPool.tryAllocate(2048);
      assertNotNull(buffers[1]);
      buffers[2] = PinnedMemoryPool.tryAllocate(4096);
      assertNotNull(buffers[2]);
      buffers[1].close();
      buffers[1] = null;
      buffers[1] = PinnedMemoryPool.tryAllocate(8192);
      assertNotNull(buffers[1]);
      buffers[3] = PinnedMemoryPool.tryAllocate(2048);
      assertNotNull(buffers[3]);
      buffers[4] = PinnedMemoryPool.tryAllocate(64);
      assertNull(buffers[4]);
      buffers[0].close();
      buffers[0] = null;
      buffers[4] = PinnedMemoryPool.tryAllocate(64);
      assertNotNull(buffers[4]);
    } finally {
      for (HostMemoryBuffer buffer : buffers) {
        if (buffer != null) {
          buffer.close();
        }
      }
    }
  }

  @Test
  void testTouchPinnedMemory() {
    final long poolSize = 15 * 1024L;
    PinnedMemoryPool.initialize(poolSize);
    int bufLength = 256;
    try(HostMemoryBuffer hmb = PinnedMemoryPool.allocate(bufLength);
        HostMemoryBuffer hmb2 = PinnedMemoryPool.allocate(bufLength)) {
      ByteBuffer bb = hmb.asByteBuffer(0, bufLength);
      for (int i = 0; i < bufLength; i++) {
        bb.put(i, (byte)i);
      }
      hmb2.copyFromHostBuffer(0, hmb, 0, bufLength);
      ByteBuffer bb2 = hmb2.asByteBuffer(0, bufLength);
      for (int i = 0; i < bufLength; i++) {
        assertEquals(bb.get(i), bb2.get(i));
      }
    }
  }

  @Test
  void testZeroSizedAllocation() {
    final long poolSize = 4 * 1024L;
    PinnedMemoryPool.initialize(poolSize);
    assertEquals(poolSize, PinnedMemoryPool.getTotalPoolSizeBytes());
    try (HostMemoryBuffer buffer = PinnedMemoryPool.tryAllocate(0)) {
      assertNotNull(buffer);
      assertEquals(0, buffer.getLength());
    }
  }

  // This test simulates cuIO using our fallback pinned pool wrapper
  // we should be able to either go to the pool, in this case 15KB in size
  // or we should be falling back to pinned cudaMallocHost/cudaFreeHost.
  @Test
  void testFallbackPinnedPool() {
    final long poolSize = 15 * 1024L;
    PinnedMemoryPool.initialize(poolSize);
    assertEquals(poolSize, PinnedMemoryPool.getTotalPoolSizeBytes());

    long ptr = Rmm.allocFromFallbackPinnedPool(1347);  // this doesn't fallback
    long ptr2 = Rmm.allocFromFallbackPinnedPool(15 * 1024L);  // this does
    Rmm.freeFromFallbackPinnedPool(ptr, 1347); // free from pool
    Rmm.freeFromFallbackPinnedPool(ptr2, 15*1024); // free from fallback

    ptr = Rmm.allocFromFallbackPinnedPool(15*1024L); // this doesn't fallback
    ptr2 = Rmm.allocFromFallbackPinnedPool(15*1024L); // this does
    Rmm.freeFromFallbackPinnedPool(ptr, 15*1024L); // free from pool
    Rmm.freeFromFallbackPinnedPool(ptr2, 15*1024L); // free from fallback
  }
}

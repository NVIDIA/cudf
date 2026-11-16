/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;
import static org.junit.jupiter.api.Assumptions.assumeFalse;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

public class RmmTest {
  private static final long TOO_MUCH_MEMORY = 3L * 1024 * 1024 * 1024 * 1024 * 1024 * 1024;

  @BeforeEach
  public void setup() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
  }

  @AfterEach
  public void teardown() {
    if (Rmm.isInitialized()) {
      Rmm.shutdown();
    }
  }

  @Test
  public void testCreateAdaptors() {
    final long poolSize = 32 * 1024 * 1024; // 32 MiB
    try (RmmCudaMemoryResource r = new RmmCudaMemoryResource()) {
      assert(r.getHandle() != 0);
    }
    try (RmmCudaAsyncMemoryResource r = new RmmCudaAsyncMemoryResource(poolSize, poolSize)) {
      assert(r.getHandle() != 0);
    }
    try (RmmManagedMemoryResource r = new RmmManagedMemoryResource()) {
      assert(r.getHandle() != 0);
    }
    try (RmmArenaMemoryResource<RmmCudaMemoryResource> r =
             new RmmArenaMemoryResource<>(new RmmCudaMemoryResource(), poolSize, false)) {
      assert(r.getHandle() != 0);
    }
    try (RmmPoolMemoryResource<RmmCudaMemoryResource> r =
             new RmmPoolMemoryResource<>(new RmmCudaMemoryResource(), poolSize, poolSize)) {
      assert(r.getHandle() != 0);
    }
    try (RmmLimitingResourceAdaptor<RmmCudaMemoryResource> r =
             new RmmLimitingResourceAdaptor<>(new RmmCudaMemoryResource(), poolSize, 64)) {
      assert(r.getHandle() != 0);
    }
    try (RmmLoggingResourceAdaptor<RmmCudaMemoryResource> r =
             new RmmLoggingResourceAdaptor<>(new RmmCudaMemoryResource(), Rmm.logToStderr(), true)) {
      assert(r.getHandle() != 0);
    }
    try (RmmTrackingResourceAdaptor<RmmCudaMemoryResource> r =
             new RmmTrackingResourceAdaptor<>(new RmmCudaMemoryResource(), 64)) {
      assert(r.getHandle() != 0);
      assert(r.getTotalBytesAllocated() == 0);
      assert(r.getMaxTotalBytesAllocated() == 0);
      assert(r.getScopedMaxTotalBytesAllocated() == 0);
      r.resetScopedMaxTotalBytesAllocated(1024);
      assert(r.getScopedMaxTotalBytesAllocated() == 1024);
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {
      RmmAllocationMode.CUDA_DEFAULT,
      RmmAllocationMode.POOL,
      RmmAllocationMode.ARENA})
  public void testTotalAllocated(int rmmAllocMode) {
    Rmm.initialize(rmmAllocMode, Rmm.logToStderr(), 512 * 1024 * 1024);
    assertEquals(0, Rmm.getTotalBytesAllocated());
    try (DeviceMemoryBuffer ignored = Rmm.alloc(1024)) {
      assertEquals(1024, Rmm.getTotalBytesAllocated());
    }
    assertEquals(0, Rmm.getTotalBytesAllocated());
  }

  @ParameterizedTest
  @ValueSource(ints = {
      RmmAllocationMode.CUDA_DEFAULT,
      RmmAllocationMode.POOL,
      RmmAllocationMode.ARENA})
  public void testMaxOutstanding(int rmmAllocMode) {
    Rmm.initialize(rmmAllocMode, Rmm.logToStderr(), 512 * 1024 * 1024);
    assertEquals(0, Rmm.getMaximumTotalBytesAllocated());
    try (DeviceMemoryBuffer ignored = Rmm.alloc(1024)) {
      assertEquals(1024, Rmm.getMaximumTotalBytesAllocated());
    }
    assertEquals(0, Rmm.getTotalBytesAllocated());
    assertEquals(1024, Rmm.getMaximumTotalBytesAllocated());
  }

  @ParameterizedTest
  @ValueSource(ints = {
      RmmAllocationMode.CUDA_DEFAULT,
      RmmAllocationMode.POOL,
      RmmAllocationMode.ARENA})
  public void testScopedMaxOutstanding(int rmmAllocMode) {
    Rmm.initialize(rmmAllocMode, Rmm.logToStderr(), 512 * 1024 * 1024);
    assertEquals(0, Rmm.getMaximumTotalBytesAllocated());
    try (DeviceMemoryBuffer ignored = Rmm.alloc(1024);
         DeviceMemoryBuffer ignored2 = Rmm.alloc(1024)) {
      assertEquals(2048, Rmm.getScopedMaximumBytesAllocated());
    }
    assertEquals(0, Rmm.getTotalBytesAllocated());
    assertEquals(2048, Rmm.getScopedMaximumBytesAllocated());

    Rmm.resetScopedMaximumBytesAllocated();
    assertEquals(0, Rmm.getScopedMaximumBytesAllocated());
    assertEquals(2048, Rmm.getMaximumTotalBytesAllocated());

    DeviceMemoryBuffer ignored = Rmm.alloc(1024);
    ignored.close();
    assertEquals(1024, Rmm.getScopedMaximumBytesAllocated());
    assertEquals(2048, Rmm.getMaximumTotalBytesAllocated());
    assertEquals(0, Rmm.getTotalBytesAllocated());

    // a non-zero value is the new minimum
    DeviceMemoryBuffer ignored2 = Rmm.alloc(1024);
    ignored2.close();
    Rmm.resetScopedMaximumBytesAllocated(10000);
    assertEquals(10000, Rmm.getScopedMaximumBytesAllocated());
    assertEquals(2048, Rmm.getMaximumTotalBytesAllocated());

    try(DeviceMemoryBuffer ignored3 = Rmm.alloc(1024)) {
      Rmm.resetScopedMaximumBytesAllocated(1024);
      try (DeviceMemoryBuffer ignored4 = Rmm.alloc(20480)) {
        assertEquals(21504, Rmm.getScopedMaximumBytesAllocated());
        assertEquals(21504, Rmm.getMaximumTotalBytesAllocated());
      }
    }
  }

  @ParameterizedTest
  @ValueSource(ints = {
      RmmAllocationMode.CUDA_DEFAULT,
      RmmAllocationMode.POOL,
      RmmAllocationMode.ARENA})
  public void testScopedMaxOutstandingNegative(int rmmAllocMode) {
    Rmm.initialize(rmmAllocMode, Rmm.logToStderr(), 512 * 1024 * 1024);
    assertEquals(0, Rmm.getMaximumTotalBytesAllocated());
    try (DeviceMemoryBuffer ignored = Rmm.alloc(1024);
         DeviceMemoryBuffer ignored2 = Rmm.alloc(1024)) {
      assertEquals(2048, Rmm.getScopedMaximumBytesAllocated());
      Rmm.resetScopedMaximumBytesAllocated();
      assertEquals(0, Rmm.getScopedMaximumBytesAllocated());
    }
    // because we allocated a net -2048 Bytes since reset
    assertEquals(0, Rmm.getScopedMaximumBytesAllocated());
    DeviceMemoryBuffer ignored = Rmm.alloc(1024);
    ignored.close();
    assertEquals(0, Rmm.getScopedMaximumBytesAllocated());

    // if we allocate 2KB and then 256B we start seeing a positive local maximum
    try (DeviceMemoryBuffer ignored2 = Rmm.alloc(2048);
         DeviceMemoryBuffer ignored3 = Rmm.alloc(256)) {
      assertEquals(256, Rmm.getScopedMaximumBytesAllocated());
    }
  }

  @Tag("noSanitizer")
  @ParameterizedTest
  @ValueSource(ints = {
      RmmAllocationMode.CUDA_DEFAULT,
      RmmAllocationMode.POOL,
      RmmAllocationMode.ARENA})
  public void testEventHandler(int rmmAllocMode) {
    AtomicInteger invokedCount = new AtomicInteger();
    AtomicLong amountRequested = new AtomicLong();
    AtomicInteger timesRetried = new AtomicInteger();
    AtomicLong totalAllocated = new AtomicLong();
    AtomicLong totalDeallocated = new AtomicLong();

    RmmEventHandler handler = new BaseRmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested, int retryCount) {
        int count = invokedCount.incrementAndGet();
        timesRetried.set(retryCount);
        amountRequested.set(sizeRequested);
        return count != 3;
      }

      @Override
      public void onAllocated(long sizeAllocated) {
        totalAllocated.addAndGet(sizeAllocated);
      }

      @Override
      public void onDeallocated(long sizeDeallocated) {
        totalDeallocated.addAndGet(sizeDeallocated);
      }
    };

    Rmm.initialize(rmmAllocMode, Rmm.logToStderr(), 512 * 1024 * 1024);
    Rmm.setEventHandler(handler);
    DeviceMemoryBuffer addr = Rmm.alloc(1024);
    addr.close();
    assertTrue(addr.address != 0);
    assertEquals(0, invokedCount.get());

    // by default, we don't get callbacks on allocated or deallocated
    assertEquals(0, totalAllocated.get());
    assertEquals(0, totalDeallocated.get());

    // Try to allocate too much
    long requested = TOO_MUCH_MEMORY;
    try {
      addr = Rmm.alloc(requested);
      addr.close();
      fail("should have failed to allocate");
    } catch (OutOfMemoryError | RmmException ignored) {
    }

    assertEquals(3, invokedCount.get());
    assertEquals(2, timesRetried.get());
    assertEquals(requested, amountRequested.get());

    // verify after a failure we can still allocate something more reasonable
    requested = 8192;
    addr = Rmm.alloc(requested);
    addr.close();

    // test the debug event handler
    Rmm.clearEventHandler();
    Rmm.setEventHandler(handler, /*enableDebug*/ true);
    addr = Rmm.alloc(1024);
    addr.close();
    assertEquals(1024, totalAllocated.get());
    assertEquals(1024, totalDeallocated.get());
  }

  @Tag("noSanitizer")
  @Test
  public void testSetEventHandlerTwice() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, Rmm.logToStderr(), 0L);
    // installing an event handler the first time should not be an error
    Rmm.setEventHandler(new BaseRmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested, int retryCount) {
        return false;
      }
    });

    // installing a second event handler is an error
    RmmEventHandler otherHandler = new BaseRmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested, int retryCount) {
        return true;
      }
    };
    assertThrows(RmmException.class, () -> Rmm.setEventHandler(otherHandler));
  }

  @Tag("noSanitizer")
  @Test
  public void testClearEventHandler() {
    Rmm.initialize(RmmAllocationMode.CUDA_DEFAULT, Rmm.logToStderr(), 0L);
    // clearing the event handler when it isn't set is not an error
    Rmm.clearEventHandler();

    // create an event handler that will always retry
    RmmEventHandler retryHandler = new BaseRmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested, int retryCount) {
        return true;
      }
    };

    Rmm.setEventHandler(retryHandler);
    Rmm.clearEventHandler();

    // verify handler is no longer installed, alloc should fail
    try {
      DeviceMemoryBuffer addr = Rmm.alloc(TOO_MUCH_MEMORY);
      addr.close();
      fail("should have failed to allocate");
    } catch (OutOfMemoryError | RmmException ignored) {
    }
  }

  @Tag("noSanitizer")
  @Test
  public void testAllocOnlyThresholds() {
    final AtomicInteger allocInvocations = new AtomicInteger(0);
    final AtomicInteger deallocInvocations = new AtomicInteger(0);
    final AtomicLong allocated = new AtomicLong(0);

    Rmm.initialize(RmmAllocationMode.POOL, Rmm.logToStderr(), 1024 * 1024L);

    RmmEventHandler handler = new RmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested, int retryCount) {
        return false;
      }

      @Override
      public long[] getAllocThresholds() {
        return new long[] { 32 * 1024, 8 * 1024 };
      }

      @Override
      public long[] getDeallocThresholds() {
        return null;
      }

      @Override
      public void onAllocThreshold(long totalAllocSize) {
        allocInvocations.getAndIncrement();
        allocated.set(totalAllocSize);
      }

      @Override
      public void onDeallocThreshold(long totalAllocSize) {
        deallocInvocations.getAndIncrement();
      }
    };

    Rmm.setEventHandler(handler);
    DeviceMemoryBuffer[] addrs = new DeviceMemoryBuffer[5];
    try {
      addrs[0] = Rmm.alloc(6 * 1024);
      assertEquals(0, allocInvocations.get());
      addrs[1] = Rmm.alloc(2 * 1024);
      assertEquals(1, allocInvocations.get());
      assertEquals(8 * 1024, allocated.get());
      addrs[2] = Rmm.alloc(21 * 1024);
      assertEquals(1, allocInvocations.get());
      addrs[3] = Rmm.alloc(8 * 1024);
      assertEquals(2, allocInvocations.get());
      assertEquals(37 * 1024, allocated.get());
      addrs[4] = Rmm.alloc(8 * 1024);
      assertEquals(2, allocInvocations.get());
    } finally {
      for (DeviceMemoryBuffer addr : addrs) {
        if (addr != null) {
          addr.close();
        }
      }
    }

    assertEquals(2, allocInvocations.get());
    assertEquals(0, deallocInvocations.get());
  }

  @Tag("noSanitizer")
  @Test
  public void testThresholds() {
    final AtomicInteger allocInvocations = new AtomicInteger(0);
    final AtomicInteger deallocInvocations = new AtomicInteger(0);
    final AtomicLong allocated = new AtomicLong(0);

    Rmm.initialize(RmmAllocationMode.POOL, Rmm.logToStderr(), 1024 * 1024L);

    RmmEventHandler handler = new RmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested, int retryCount) {
        return false;
      }

      @Override
      public long[] getAllocThresholds() {
        return new long[] { 8 * 1024 };
      }

      @Override
      public long[] getDeallocThresholds() {
        return new long[] { 6 * 1024 };
      }

      @Override
      public void onAllocThreshold(long totalAllocSize) {
        allocInvocations.getAndIncrement();
        allocated.set(totalAllocSize);
      }

      @Override
      public void onDeallocThreshold(long totalAllocSize) {
        deallocInvocations.getAndIncrement();
        allocated.set(totalAllocSize);
      }
    };

    Rmm.setEventHandler(handler);
    DeviceMemoryBuffer[] addrs = new DeviceMemoryBuffer[5];
    try {
      addrs[0] = Rmm.alloc(6 * 1024);
      assertEquals(0, allocInvocations.get());
      assertEquals(0, deallocInvocations.get());
      addrs[0].close();
      addrs[0] = null;
      assertEquals(0, allocInvocations.get());
      assertEquals(1, deallocInvocations.get());
      assertEquals(0, allocated.get());
      addrs[0] = Rmm.alloc(12 * 1024);
      assertEquals(1, allocInvocations.get());
      assertEquals(1, deallocInvocations.get());
      assertEquals(12 * 1024, allocated.get());
      addrs[1] = Rmm.alloc(6 * 1024);
      assertEquals(1, allocInvocations.get());
      assertEquals(1, deallocInvocations.get());
      addrs[0].close();
      addrs[0] = null;
      assertEquals(1, allocInvocations.get());
      assertEquals(1, deallocInvocations.get());
      addrs[0] = Rmm.alloc(4 * 1024);
      assertEquals(2, allocInvocations.get());
      assertEquals(1, deallocInvocations.get());
      assertEquals(10 * 1024, allocated.get());
      addrs[1].close();
      addrs[1] = null;
      assertEquals(2, allocInvocations.get());
      assertEquals(2, deallocInvocations.get());
      assertEquals(4 * 1024, allocated.get());
      addrs[0].close();
      addrs[0] = null;
      assertEquals(2, allocInvocations.get());
      assertEquals(2, deallocInvocations.get());
    } finally {
      for (DeviceMemoryBuffer addr : addrs) {
        if (addr != null) {
          addr.close();
        }
      }
    }

    assertEquals(2, allocInvocations.get());
    assertEquals(2, deallocInvocations.get());
  }

  @Tag("noSanitizer")
  @Test
  public void testExceptionHandling() {
    Rmm.initialize(RmmAllocationMode.POOL, Rmm.logToStderr(), 1024 * 1024L);

    RmmEventHandler handler = new RmmEventHandler() {
      @Override
      public boolean onAllocFailure(long sizeRequested, int retryCount) {
        throw new AllocFailException();
      }

      @Override
      public long[] getAllocThresholds() {
        return new long[] { 8 * 1024 };
      }

      @Override
      public long[] getDeallocThresholds() {
        return new long[] { 6 * 1024 };
      }

      @Override
      public void onAllocThreshold(long totalAllocSize) {
        throw new AllocThresholdException();
      }

      @Override
      public void onDeallocThreshold(long totalAllocSize) {
        throw new DeallocThresholdException();
      }
    };

    Rmm.setEventHandler(handler);
    DeviceMemoryBuffer addr = Rmm.alloc(6 * 1024);
    assertThrows(DeallocThresholdException.class, addr::close);
    assertThrows(AllocThresholdException.class, () -> Rmm.alloc(12 * 1024));
    assertThrows(AllocFailException.class, () -> Rmm.alloc(TOO_MUCH_MEMORY));
  }

  @Test
  public void testThreadAutoDeviceSetup() throws Exception {
    // A smoke-test for automatic CUDA device setup for threads calling
    // into cudf. Hard to fully test without requiring multiple CUDA devices.
    Rmm.initialize(RmmAllocationMode.POOL, Rmm.logToStderr(), 1024 * 1024L);
    DeviceMemoryBuffer buff = Rmm.alloc(1024);
    try {
      ExecutorService executor = Executors.newSingleThreadExecutor();
      Future<Boolean> future = executor.submit(() -> {
        DeviceMemoryBuffer localBuffer = Rmm.alloc(2048);
        localBuffer.close();
        buff.close();
        return true;
      });
      assertTrue(future.get());
      executor.shutdown();
    } catch (Exception t) {
      buff.close();
      throw t;
    }
  }

  @Tag("noSanitizer")
  @ParameterizedTest
  @ValueSource(ints = {
      RmmAllocationMode.CUDA_DEFAULT,
      RmmAllocationMode.POOL,
      RmmAllocationMode.ARENA})
  public void testSetDeviceThrowsAfterRmmInit(int rmmAllocMode) {
    Rmm.initialize(rmmAllocMode, Rmm.logToStderr(), 1024 * 1024);
    assertThrows(CudfException.class, () -> Cuda.setDevice(Cuda.getDevice() + 1));
    // Verify that auto set device does not
    Cuda.autoSetDevice();
  }

  @Tag("noSanitizer")
  @Test
  public void testPoolSize() {
    Rmm.initialize(RmmAllocationMode.POOL, Rmm.logToStderr(), 1024);
    try (DeviceMemoryBuffer ignored1 = Rmm.alloc(1024)) {
      assertThrows(OutOfMemoryError.class,
          () -> {
            DeviceMemoryBuffer ignored2 = Rmm.alloc(1024);
            ignored2.close();
      });
    }
  }

  @Tag("noSanitizer")
  @Test
  public void testCudaAsyncMemoryResourceSize() {
    Rmm.initialize(RmmAllocationMode.CUDA_ASYNC, Rmm.logToStderr(), 1024);
    try (DeviceMemoryBuffer ignored1 = Rmm.alloc(1024)) {
      assertThrows(OutOfMemoryError.class,
          () -> {
            DeviceMemoryBuffer ignored2 = Rmm.alloc(1024);
            ignored2.close();
          });
    }
  }

  // Skipped (JUnit assumption) when the device does not support CUDA memory pools exported as
  // POSIX file descriptors; see Cuda.isMemPoolPosixFdSupported().
  @Test
  public void testCudaAsyncPosixFdPoolHandle() {
    assumeTrue(Cuda.isMemPoolPosixFdSupported(), "POSIX file descriptor pools not supported");
    Rmm.initialize(RmmAllocationMode.CUDA_ASYNC_POSIX_FD, Rmm.logToStderr(), 1024);
    try {
      RmmCudaAsyncMemoryResource asyncResource = currentAsyncResource();
      assertEquals(RmmCudaAsyncMemoryResource.HandleType.POSIX_FILE_DESCRIPTOR,
          asyncResource.getHandleType());
      assertNotEquals(0, asyncResource.getPoolHandle());
      try (DeviceMemoryBuffer ignored1 = Rmm.alloc(1024)) {
        assertThrows(OutOfMemoryError.class,
            () -> {
              DeviceMemoryBuffer ignored2 = Rmm.alloc(1024);
              ignored2.close();
            });
      }
    } finally {
      Rmm.shutdown();
    }
  }

  @Test
  public void testCudaAsyncPoolHandle() {
    Rmm.initialize(RmmAllocationMode.CUDA_ASYNC, Rmm.logToStderr(), 1024);
    try {
      RmmCudaAsyncMemoryResource asyncResource = currentAsyncResource();
      assertEquals(RmmCudaAsyncMemoryResource.HandleType.NONE, asyncResource.getHandleType());
      assertNotEquals(0, asyncResource.getPoolHandle());
    } finally {
      Rmm.shutdown();
    }
  }

  /** The RmmCudaAsyncMemoryResource wrapped by the current device resource. */
  private static RmmCudaAsyncMemoryResource currentAsyncResource() {
    RmmDeviceMemoryResource resource = Rmm.getCurrentDeviceResource();
    while (resource instanceof RmmWrappingDeviceMemoryResource) {
      resource = ((RmmWrappingDeviceMemoryResource<?>) resource).getWrapped();
    }
    assertTrue(resource instanceof RmmCudaAsyncMemoryResource,
        "current device resource does not wrap an RmmCudaAsyncMemoryResource");
    return (RmmCudaAsyncMemoryResource) resource;
  }

  @Test
  public void testCudaAsyncHandleTypeResource() {
    try (RmmCudaAsyncMemoryResource mr = new RmmCudaAsyncMemoryResource(1024, 1024, false)) {
      assertEquals(RmmCudaAsyncMemoryResource.HandleType.NONE, mr.getHandleType());
      assertNotEquals(0, mr.getPoolHandle());
    }
    assertThrows(IllegalArgumentException.class,
        () -> new RmmCudaAsyncMemoryResource(1024, 1024,
            (RmmCudaAsyncMemoryResource.HandleType) null));
  }

  // Skipped (JUnit assumption) when the device does not support CUDA memory pools exported as
  // POSIX file descriptors; see Cuda.isMemPoolPosixFdSupported().
  @Test
  public void testCudaAsyncPosixFdHandleTypeResource() {
    assumeTrue(Cuda.isMemPoolPosixFdSupported(), "POSIX file descriptor pools not supported");
    try (RmmCudaAsyncMemoryResource mr = new RmmCudaAsyncMemoryResource(1024, 1024,
        RmmCudaAsyncMemoryResource.HandleType.POSIX_FILE_DESCRIPTOR)) {
      assertEquals(RmmCudaAsyncMemoryResource.HandleType.POSIX_FILE_DESCRIPTOR,
          mr.getHandleType());
      assertNotEquals(0, mr.getPoolHandle());
    }
  }

  // Skipped (JUnit assumption) when the device does not support CUDA memory pools exported as
  // POSIX file descriptors; see Cuda.isMemPoolPosixFdSupported().
  @Test
  public void testMemPoolIpcRoundTrip() {
    assumeTrue(Cuda.isMemPoolPosixFdSupported(), "POSIX file descriptor pools not supported");
    Rmm.initialize(RmmAllocationMode.CUDA_ASYNC_POSIX_FD, Rmm.logToStderr(), 16 * 1024 * 1024);
    try {
      byte[] expected = new byte[4096];
      for (int i = 0; i < expected.length; i++) {
        expected[i] = (byte) (i * 31);
      }
      try (DeviceMemoryBuffer dev = DeviceMemoryBuffer.allocate(expected.length);
           HostMemoryBuffer host = HostMemoryBuffer.allocate(expected.length)) {
        host.setBytes(0, expected, 0, expected.length);
        dev.copyFromHostBuffer(host);
        // Export and import in one process, as a process that hosts both sides would. The pool
        // is exported before the pointer: cudaMemPoolExportPointer fails with
        // cudaErrorInvalidValue on a pool that has not been exported. The import closes the fd.
        int fd = Cuda.memPoolExportToPosixFd(currentAsyncResource().getPoolHandle());
        long imported = Cuda.memPoolImportFromPosixFd(fd);
        try {
          byte[] exportData = Cuda.memPoolExportPointer(dev.getAddress());
          long ptr = Cuda.memPoolImportPointer(imported, exportData);
          try (HostMemoryBuffer out = HostMemoryBuffer.allocate(expected.length)) {
            out.copyFromDeviceBuffer(new DeviceMemoryBufferView(ptr, expected.length));
            byte[] actual = new byte[expected.length];
            out.getBytes(actual, 0, 0, actual.length);
            assertArrayEquals(expected, actual);
          } finally {
            // The imported pointer is freed before the imported pool is destroyed and before
            // the exported allocation is freed.
            Cuda.freeAsync(ptr, Cuda.DEFAULT_STREAM);
            Cuda.DEFAULT_STREAM.sync();
          }
        } finally {
          Cuda.memPoolDestroy(imported);
        }
      }
    } finally {
      Rmm.shutdown();
    }
  }

  // Skipped (JUnit assumption) when the device does not support CUDA memory pools exported as
  // POSIX file descriptors; see Cuda.isMemPoolPosixFdSupported().
  @Test
  public void testMemPoolImportPointerRejectsWrongSize() {
    assumeTrue(Cuda.isMemPoolPosixFdSupported(), "POSIX file descriptor pools not supported");
    Rmm.initialize(RmmAllocationMode.CUDA_ASYNC_POSIX_FD, Rmm.logToStderr(), 1024 * 1024);
    try {
      assertThrows(IllegalArgumentException.class,
          () -> Cuda.memPoolImportPointer(currentAsyncResource().getPoolHandle(), new byte[3]));
    } finally {
      Rmm.shutdown();
    }
  }

  // The export fails with a CUDA error on purpose, which compute-sanitizer reports.
  @Tag("noSanitizer")
  @Test
  public void testMemPoolExportRequiresPosixFdPool() {
    try (RmmCudaAsyncMemoryResource mr = new RmmCudaAsyncMemoryResource(1024, 1024,
        RmmCudaAsyncMemoryResource.HandleType.NONE)) {
      CudaException e = assertThrows(CudaException.class,
          () -> Cuda.memPoolExportToPosixFd(mr.getPoolHandle()));
      assertFalse(e instanceof CudaFatalException, "export failure must not be fatal");
    }
    // The failure must not leave a sticky error behind.
    try (ColumnVector cv = ColumnVector.fromBoxedInts(1, 2, 3)) {
      assertEquals(3, cv.getRowCount());
    }
  }

  @Test
  public void testCudaAsyncPosixFdIsIncompatibleWithManaged() {
    assertThrows(IllegalArgumentException.class,
        () -> Rmm.initialize(
            RmmAllocationMode.CUDA_ASYNC_POSIX_FD | RmmAllocationMode.CUDA_MANAGED_MEMORY,
            Rmm.logToStderr(), 1024));
  }

  @Tag("noSanitizer")
  @Test
  public void testCudaAsyncIsIncompatibleWithManaged() {
    assertThrows(IllegalArgumentException.class,
        () -> Rmm.initialize(
            RmmAllocationMode.CUDA_ASYNC | RmmAllocationMode.CUDA_MANAGED_MEMORY,
            Rmm.logToStderr(), 1024));
  }

  @Test
  public void testCudaMemoryBuffer() {
    Rmm.initialize(RmmAllocationMode.ARENA, Rmm.logToStderr(), 8 * 1024 * 1024);
    try (CudaMemoryBuffer one = CudaMemoryBuffer.allocate(512);
         CudaMemoryBuffer two = CudaMemoryBuffer.allocate(1024)) {
      assertEquals(512, one.length);
      assertEquals(1024, two.length);
      assertEquals(0, Rmm.getTotalBytesAllocated());
    }
  }

  private static class AllocFailException extends RuntimeException {
  }

  private static class AllocThresholdException extends RuntimeException {
  }

  private static class DeallocThresholdException extends RuntimeException {
  }

  private static abstract class BaseRmmEventHandler implements RmmEventHandler {
    @Override
    public long[] getAllocThresholds() {
      return null;
    }

    @Override
    public long[] getDeallocThresholds() {
      return null;
    }

    @Override
    public void onAllocThreshold(long totalAllocSize) {
    }

    @Override
    public void onDeallocThreshold(long totalAllocSize) {
    }
  }
}

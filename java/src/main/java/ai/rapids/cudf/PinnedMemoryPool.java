/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;


import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

/**
 * This is the JNI interface to an RMM pool backed by pinned host memory.
 */
public final class PinnedMemoryPool implements AutoCloseable {
  private static final Logger log = LoggerFactory.getLogger(PinnedMemoryPool.class);

  // These static fields should only ever be accessed when class-synchronized.
  // Do NOT use singleton_ directly!  Use the getSingleton accessor instead.
  private static volatile PinnedMemoryPool singleton_ = null;
  private static volatile Future<PinnedMemoryPool> initFuture = null;
  private long poolHandle;
  private long poolSize;

  private static final class PinnedHostBufferCleaner extends MemoryBuffer.MemoryBufferCleaner {
    private long address;
    private final long origLength;

    PinnedHostBufferCleaner(long address, long length) {
      this.address = address;
      origLength = length;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long origAddress = 0;
      if (address != -1) {
        origAddress = address;
        try {
          PinnedMemoryPool.freeInternal(address, origLength);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          address = -1;
        }
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A PINNED HOST BUFFER WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress) + ")");
        logRefCountDebug("Leaked pinned host buffer");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return address == -1;
    }
  }

  private static PinnedMemoryPool getSingleton() {
    if (singleton_ == null && initFuture != null) {
      // There is an initFuture whose result is not yet retrieved.
      synchronized (PinnedMemoryPool.class) {
        if (singleton_ == null && initFuture != null) {
          try {
            singleton_ = initFuture.get();
            initFuture = null;
          } catch (InterruptedException e) {
            // Interruption does not cancel initialization; keep the future.
            Thread.currentThread().interrupt();
            throw new RuntimeException("Interrupted initializing pinned memory pool", e);
          } catch (Exception e) {
            // Null the future so this and subsequent callers can fall back or retry initialization.
            initFuture = null;
            log.error("Error initializing pinned memory pool",
                e.getCause() != null ? e.getCause() : e);
          }
        }
      }
    }
    return singleton_;
  }

  private static void freeInternal(long address, long origLength) {
    Objects.requireNonNull(getSingleton()).free(address, origLength);
  }

  /**
   * Initialize the pool.
   *
   * @param poolSize size of the pool to initialize.
   * @note when using this method, the pinned pool will be shared with cuIO
   */
  public static synchronized void initialize(long poolSize) {
    initialize(poolSize, -1, true);
  }

  /**
   * Initialize the pool.
   *
   * @param poolSize size of the pool to initialize.
   * @param gpuId    gpu id to set to get memory pool from, -1 means to use default
   * @note when using this method, the pinned pool will be shared with cuIO
   */
  public static synchronized void initialize(long poolSize, int gpuId) {
    initialize(poolSize, gpuId, true);
  }

  /**
   * Initialize the pool.
   *
   * @param poolSize size of the pool to initialize.
   * @param gpuId    gpu id to set to get memory pool from, -1 means to use default
   * @param setCudfPinnedPoolMemoryResource true if this pinned pool should be used by cuDF for pinned memory
   */
  public static synchronized void initialize(long poolSize, int gpuId, boolean setCudfPinnedPoolMemoryResource) {
    initialize(poolSize, gpuId, setCudfPinnedPoolMemoryResource, 1);
  }

  /**
   * Initialize the pool.
   *
   * @param poolSize size of the pool to initialize.
   * @param gpuId gpu id to set to get memory pool from, -1 means to use default
   * @param setCudfPinnedPoolMemoryResource true if this pinned pool should be used by cuDF for pinned memory
   * @param initializationThreads requested number of threads used to initialize the pool's backing memory,
   *                              capped at reported hardware concurrency. A value of 1 initializes the
   *                              backing memory using {@code cudaHostAlloc}. Values greater than 1 instead
   *                              request huge pages and pre-touch pages concurrently before pinning for
   *                              faster initialization. This does not affect subsequent suballocator behavior.
   * @note on multi-NUMA systems, multithreaded initialization may scatter pages across nodes if placement
   *       is not constrained in advance. Pages cannot be migrated once pinned.
   */
  public static synchronized void initialize(long poolSize, int gpuId, boolean setCudfPinnedPoolMemoryResource,
      int initializationThreads) {
    if (initializationThreads <= 0) {
      throw new IllegalArgumentException("Initialization thread count must be positive");
    }
    if (isInitialized()) {
      throw new IllegalStateException("Pinned memory pool is already initialized.");
    }
    ExecutorService initService = Executors.newSingleThreadExecutor(runnable -> {
      Thread t = new Thread(runnable, "pinned pool init");
      t.setDaemon(true);
      return t;
    });
    initFuture = initService.submit(() ->
        new PinnedMemoryPool(poolSize, gpuId, setCudfPinnedPoolMemoryResource,
            initializationThreads));
    initService.shutdown();
  }

  /**
   * Check if the pool has been initialized or not.
   */
  public static boolean isInitialized() {
    return getSingleton() != null;
  }

  /**
   * Shut down the RMM pool_memory_resource, nulling out our reference. Any allocation
   * or free that is in flight will fail after this.
   */
  public static synchronized void shutdown() {
    PinnedMemoryPool pool = getSingleton();
    if (pool != null) {
      pool.close();
      pool = null;
    }
    initFuture = null;
    singleton_ = null;
  }

  /**
   * Factory method to create a pinned host memory buffer.
   *
   * @param bytes size in bytes to allocate
   * @return newly created buffer or null if insufficient pinned memory
   */
  public static HostMemoryBuffer tryAllocate(long bytes) {
    HostMemoryBuffer result = null;
    PinnedMemoryPool pool = getSingleton();
    if (pool != null) {
      result = pool.tryAllocateInternal(bytes);
    }
    return result;
  }

  /**
   * Factory method to create a host buffer but preferably pointing to pinned memory.
   * It is not guaranteed that the returned buffer will be pointer to pinned memory.
   *
   * @param bytes size in bytes to allocate
   * @return newly created buffer
   */
  public static HostMemoryBuffer allocate(long bytes, HostMemoryAllocator hostMemoryAllocator) {
    HostMemoryBuffer result = tryAllocate(bytes);
    if (result == null) {
      result = hostMemoryAllocator.allocate(bytes, false);
    }
    return result;
  }

  /**
   * Factory method to create a host buffer but preferably pointing to pinned memory.
   * It is not guaranteed that the returned buffer will be pointer to pinned memory.
   *
   * @param bytes size in bytes to allocate
   * @return newly created buffer
   */
  public static HostMemoryBuffer allocate(long bytes) {
    return allocate(bytes, DefaultHostMemoryAllocator.get());
  }

  /**
   * Get the number of bytes that the pinned memory pool was allocated with.
   */
  public static long getTotalPoolSizeBytes() {
    PinnedMemoryPool pool = getSingleton();
    if (pool != null) {
      return pool.poolSize;
    }
    return 0;
  }

  private PinnedMemoryPool(long poolSize, int gpuId, boolean setCudfPinnedPoolMemoryResource,
      int initializationThreads) {
    if (gpuId > -1) {
      // set the gpu device to use
      Cuda.setDevice(gpuId);
      Cuda.freeZero();
    }
    if (initializationThreads == 1) {
      this.poolHandle = Rmm.newPinnedPoolMemoryResource(poolSize, poolSize);
    } else {
      this.poolHandle = Rmm.newParallelPinnedPoolMemoryResource(poolSize, initializationThreads);
    }
    if (setCudfPinnedPoolMemoryResource) {
      Rmm.setCudfPinnedPoolMemoryResource(this.poolHandle);
    }
    this.poolSize = poolSize;
  }

  @Override
  public void close() {
    Rmm.releasePinnedPoolMemoryResource(this.poolHandle);
    this.poolHandle = -1;
  }

  /**
   * This makes an attempt to allocate pinned memory, and if the pinned memory allocation fails
   * it will return null, instead of throw.
   */
  private synchronized HostMemoryBuffer tryAllocateInternal(long bytes) {
    long allocated = Rmm.allocFromPinnedPool(this.poolHandle, bytes);
    if (allocated == -1) {
      return null;
    } else {
      return new HostMemoryBuffer(allocated, bytes,
              new PinnedHostBufferCleaner(allocated, bytes));
    }
  }

  private synchronized void free(long address, long size) {
    Rmm.freeFromPinnedPool(this.poolHandle, address, size);
  }

  /**
   * Sets the size of the cuDF default pinned pool.
   *
   * @note This has to be called before cuDF functions are executed.
   *
   * @param size initial and maximum size for the cuDF default pinned pool.
   *        Pass size=0 to disable the default pool.
   *
   * @return true if we were able to setup the default resource, false if there was
   *         a resource already set.
   */
  public static synchronized boolean configureDefaultCudfPinnedPoolSize(long size) {
    return Rmm.configureDefaultCudfPinnedPoolSize(size);
  }

}

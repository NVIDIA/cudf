/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

import java.io.File;
import java.util.concurrent.TimeUnit;

import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This is the binding class for rmm lib.
 */
public class Rmm {
  private static volatile RmmTrackingResourceAdaptor<RmmDeviceMemoryResource> tracker = null;
  private static volatile RmmDeviceMemoryResource deviceResource = null;
  private static volatile boolean initialized = false;
  private static volatile long poolSize = -1;
  private static volatile boolean poolingEnabled = false;
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final ReadWriteLock rmmLock = new ReentrantReadWriteLock();

  /*
   * Advanced users that want to make sure they can protect against Rmm
   * shutting down for example, while doing work with memory resources
   * could use the reader and writer locks provided here (publicy)
   * to protect their state.
   *
   * `writeLock`:
   *    1. Covers mutation of `tracker` and `initialized`.
   *    2. [set/clear]EventHandler (unlikely) race
   *
   * `readLock`: All other APIs.
   */
  public static final Lock readLock = rmmLock.readLock();
  public static final Lock writeLock = rmmLock.writeLock();

  enum LogLoc {
    NONE(0),
    FILE(1),
    STDOUT(2),
    STDERR(3);

    final int internalId;

    LogLoc(int internalId) {
      this.internalId = internalId;
    }
  }

  /**
   * What to send RMM alloc and free logs to.
   */
  public static class LogConf {
    final File file;
    final LogLoc loc;

    private LogConf(File file, LogLoc loc) {
      this.file = file;
      this.loc = loc;
    }
  }

  /**
   * Create a config that will write alloc/free logs to a file.
   */
  public static LogConf logTo(File location) {
    return new LogConf(location, LogLoc.FILE);
  }

  /**
   * Create a config that will write alloc/free logs to stdout.
   */
  public static LogConf logToStdout() {
    return new LogConf(null, LogLoc.STDOUT);
  }

  /**
   * Create a config that will write alloc/free logs to stderr.
   */
  public static LogConf logToStderr() {
    return new LogConf(null, LogLoc.STDERR);
  }


  /**
   * Get the RmmDeviceMemoryResource that was last set through the java APIs. This will
   * not return the correct value if the resource was not set using the java APIs. It will
   * return a null if the resource was never set through the java APIs.
   */
  public static RmmDeviceMemoryResource getCurrentDeviceResource() {
    readLock.lock();
    try {
      return deviceResource;
    } finally {
      readLock.unlock();
    }
  }

  /**
   * Get the currently set RmmTrackingResourceAdaptor that is set. This might return null if
   * RMM has nto been initialized.
   */
  public static RmmTrackingResourceAdaptor<RmmDeviceMemoryResource> getTracker() {
    readLock.lock();
    try {
      return tracker;
    } finally {
      readLock.unlock();
    }
  }

  /**
   * Set the current device resource that RMM should use for all allocations and de-allocations.
   * This should only be done if you feel comfortable that the current device resource has no
   * pending allocations. Note that the caller of this is responsible for closing the current
   * RmmDeviceMemoryResource that is returned by this. Assuming that it was not used to create
   * the newResource. Please use the `shutdown` API to clear the resource as it does best
   * effort clean up before shutting it down. If `newResource` is not null this will initialize
   * the CUDA context for the calling thread if it is not already set. The caller is responsible
   * for setting the desired CUDA device prior to this call if a specific device is already set.
   * <p>NOTE: All cudf methods will set the chosen CUDA device in the CUDA context of the calling
   * thread after this returns and `newResource` was not null.
   * <p>If `newResource` is null this will unset the default CUDA device and mark RMM as not
   * initialized.
   * <p>Be aware that for many of these APIs to work the RmmDeviceMemoryResource will need an
   * `RmmTrackingResourceAdaptor`. If one is not found and `newResource` is not null it will
   * be added to `newResource`.
   * <p>Also be very careful with how you set this up. It is possible to set up an
   * RmmDeviceMemoryResource that is just bad, like multiple pools or pools on top of an
   * RmmAsyncMemoryResource, that does pooling already. Unless you know what you are doing it is
   * best to just use the `initialize` API instead.
   *
   * @param newResource the new resource to set. If it is null an RmmCudaMemoryResource will be
   *                    used, and RMM will be set as not initialized.
   * @param expectedResource the resource that we expect to be set. This is to let us avoid race
   *                         conditions with multiple things trying to set this at once. It should
   *                         never happen, but just to be careful.
   * @param forceChange if true then the expectedResource check is not done.
   */
  public static RmmDeviceMemoryResource setCurrentDeviceResource(
      RmmDeviceMemoryResource newResource,
      RmmDeviceMemoryResource expectedResource,
      boolean forceChange) {

    writeLock.lock();
    try {
      boolean shouldInit = false;
      boolean shouldDeinit = false;
      RmmDeviceMemoryResource newResourceToSet = newResource;
      if (newResourceToSet == null) {
        // We always want it to be set to something or else it can cause problems...
        newResourceToSet = new RmmCudaMemoryResource();
        if (initialized) {
          shouldDeinit = true;
        }
      } else if (!initialized) {
        shouldInit = true;
      }

      RmmDeviceMemoryResource oldResource = deviceResource;
      if (!forceChange && expectedResource != null && deviceResource != null) {
        long expectedOldHandle = expectedResource.getHandle();
        long oldHandle = deviceResource.getHandle();
        if (oldHandle != expectedOldHandle) {
          throw new RmmException("The expected device resource is not correct " +
              Long.toHexString(oldHandle) + " != " + Long.toHexString(expectedOldHandle));
        }
      }

      poolSize = -1;
      poolingEnabled = false;
      setGlobalValsFromResource(newResourceToSet);
      if (newResource != null && tracker == null) {
        // No tracker was set, but we need one
        tracker = new RmmTrackingResourceAdaptor<>(newResourceToSet, 256);
        newResourceToSet = tracker;
      }
      long newHandle = newResourceToSet.getHandle();
      setCurrentDeviceResourceInternal(newHandle);
      deviceResource = newResource;
      if (shouldInit) {
        initDefaultCudaDevice();
        MemoryCleaner.setDefaultGpu(Cuda.getDevice());
        initialized = true;
      }

      if (shouldDeinit) {
        cleanupDefaultCudaDevice();
        initialized = false;
      }
      return oldResource;
    } finally {
      writeLock.unlock();
    }
  }

  private static void setGlobalValsFromResource(RmmDeviceMemoryResource resource) {
    if (resource instanceof RmmTrackingResourceAdaptor) {
      Rmm.tracker = (RmmTrackingResourceAdaptor<RmmDeviceMemoryResource>) resource;
    } else if (resource instanceof RmmPoolMemoryResource) {
      Rmm.poolSize = Math.max(((RmmPoolMemoryResource)resource).getMaxSize(), Rmm.poolSize);
      Rmm.poolingEnabled = true;
    } else if (resource instanceof RmmArenaMemoryResource) {
      Rmm.poolSize = Math.max(((RmmArenaMemoryResource)resource).getSize(), Rmm.poolSize);
      Rmm.poolingEnabled = true;
    } else if (resource instanceof RmmCudaAsyncMemoryResource) {
      Rmm.poolSize = Math.max(((RmmCudaAsyncMemoryResource)resource).getSize(), Rmm.poolSize);
      Rmm.poolingEnabled = true;
    }

    // Recurse as needed
    if (resource instanceof RmmWrappingDeviceMemoryResource) {
      setGlobalValsFromResource(((RmmWrappingDeviceMemoryResource<RmmDeviceMemoryResource>)resource).getWrapped());
    }
  }

  /**
   * Initialize memory manager state and storage. This will always initialize
   * the CUDA context for the calling thread if it is not already set. The
   * caller is responsible for setting the desired CUDA device prior to this
   * call if a specific device is already set.
   * <p>NOTE: All cudf methods will set the chosen CUDA device in the CUDA
   * context of the calling thread after this returns.
   * @param allocationMode Allocation strategy to use. Bit set using
   *                       {@link RmmAllocationMode#CUDA_DEFAULT},
   *                       {@link RmmAllocationMode#POOL},
   *                       {@link RmmAllocationMode#ARENA},
   *                       {@link RmmAllocationMode#CUDA_ASYNC},
   *                       {@link RmmAllocationMode#CUDA_ASYNC_FABRIC} and
   *                       {@link RmmAllocationMode#CUDA_MANAGED_MEMORY}
   * @param logConf        How to do logging or null if you don't want to
   * @param poolSize       The initial pool size in bytes
   * @throws IllegalStateException if RMM has already been initialized
   */
  public static void initialize(int allocationMode, LogConf logConf, long poolSize)
      throws RmmException {
    writeLock.lock();
    try {
      initializeUnderLock(allocationMode, logConf, poolSize);
    } finally {
      writeLock.unlock();
    }
  }

  // This function is called from initialize and must be called with a write lock held.
  private static void initializeUnderLock(int allocationMode, LogConf logConf, long poolSize) throws RmmException {
    if (initialized) {
      throw new IllegalStateException("RMM is already initialized");
    }

    boolean isPool = (allocationMode & RmmAllocationMode.POOL) != 0;
    boolean isArena = (allocationMode & RmmAllocationMode.ARENA) != 0;
    boolean isAsync = (allocationMode & RmmAllocationMode.CUDA_ASYNC) != 0;
    boolean isAsyncFabric = (allocationMode & RmmAllocationMode.CUDA_ASYNC_FABRIC) != 0;
    boolean isManaged = (allocationMode & RmmAllocationMode.CUDA_MANAGED_MEMORY) != 0;

    if (isAsync && isManaged) {
      throw new IllegalArgumentException(
          "CUDA Unified Memory is not supported in CUDA_ASYNC allocation mode");
    }

    RmmDeviceMemoryResource resource = null;
    boolean succeeded = false;
    try {
      if (isPool) {
        if (isManaged) {
          resource = new RmmPoolMemoryResource<>(new RmmManagedMemoryResource(), poolSize, poolSize);
        } else {
          resource = new RmmPoolMemoryResource<>(new RmmCudaMemoryResource(), poolSize, poolSize);
        }
      } else if (isArena) {
        if (isManaged) {
          resource = new RmmArenaMemoryResource<>(new RmmManagedMemoryResource(), poolSize, false);
        } else {
          resource = new RmmArenaMemoryResource<>(new RmmCudaMemoryResource(), poolSize, false);
        }
      } else if (isAsync) {
        resource = new RmmLimitingResourceAdaptor<>(
            new RmmCudaAsyncMemoryResource(poolSize, poolSize), poolSize, 512);
      } else if (isAsyncFabric) {
        resource = new RmmLimitingResourceAdaptor<>(
            new RmmCudaAsyncMemoryResource(poolSize, poolSize, true), poolSize, 512);
      } else if (isManaged) {
        resource = new RmmManagedMemoryResource();
      } else {
        resource = new RmmCudaMemoryResource();
      }

      if (logConf != null && logConf.loc != LogLoc.NONE) {
        resource = new RmmLoggingResourceAdaptor<>(resource, logConf, true);
      }

      resource = new RmmTrackingResourceAdaptor<>(resource, 256);
      setCurrentDeviceResource(resource, null, false);
      succeeded = true;
    } finally {
      if (!succeeded && resource != null) {
        resource.close();
      }
    }
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
  public static boolean configureDefaultCudfPinnedPoolSize(long size) {
    writeLock.lock();
    try {
      return configureDefaultCudfPinnedPoolSizeImpl(size);
    } finally {
      writeLock.unlock();
    }
  }

  private static native boolean configureDefaultCudfPinnedPoolSizeImpl(long size);

  /**
   * Get the most recently set pool size or -1 if RMM has not been initialized or pooling is
   * not enabled.
   */
  public static long getPoolSize() {
    return poolSize;
  }

  /**
   * Return true if rmm is initialized and pooling has been enabled, else false.
   */
  public static boolean isPoolingEnabled() {
    return poolingEnabled;
  }

  /**
   * Check if RMM has been initialized already or not.
   */
  public static boolean isInitialized() throws RmmException {
    return initialized;
  }

  /**
   * Return the amount of RMM memory allocated in bytes. Note that the result
   * may be less than the actual amount of allocated memory if underlying RMM
   * allocator decides to return more memory than what was requested. However,
   * the result will always be a lower bound on the amount allocated.
   */
  public static long getTotalBytesAllocated() {
    readLock.lock();
    try {
      if (tracker == null) {
        return 0;
      } else {
        return tracker.getTotalBytesAllocated();
      }
    } finally {
      readLock.unlock();
    }
  }


  /**
   * Returns the maximum amount of RMM memory (Bytes) outstanding during the
   * lifetime of the process.
   */
  public static long getMaximumTotalBytesAllocated() {
    readLock.lock();
    try {
      if (tracker == null) {
        return 0;
      } else {
        return tracker.getMaxTotalBytesAllocated();
      }
    } finally {
      readLock.unlock();
    }
  }

  /**
   * Resets a scoped maximum counter of RMM memory used to keep track of usage between
   * code sections while debugging.
   *
   * @param initialValue an initial value (in Bytes) to use for this scoped counter
   */
  public static void resetScopedMaximumBytesAllocated(long initialValue) {
    readLock.lock();
    try {
      if (tracker != null) {
        tracker.resetScopedMaxTotalBytesAllocated(initialValue);
      }
    } finally {
      readLock.unlock();
    }
  }

  /**
   * Resets a scoped maximum counter of RMM memory used to keep track of usage between
   * code sections while debugging.
   *
   * This resets the counter to 0 Bytes.
   */
  public static void resetScopedMaximumBytesAllocated() {
    readLock.lock();
    try {
      if (tracker != null) {
        tracker.resetScopedMaxTotalBytesAllocated(0L);
      }
    } finally {
      readLock.unlock();
    }
  }

  /**
   * Returns the maximum amount of RMM memory (Bytes) outstanding since the last
   * `resetScopedMaximumOutstanding` call was issued (it is "scoped" because it's the
   * maximum amount seen since the last reset).
   * <p>
   * If the memory used is net negative (for example if only frees happened since
   * reset, and we reset to 0), then result will be 0.
   * <p>
   * If `resetScopedMaximumBytesAllocated` is never called, the scope is the whole
   * program and is equivalent to `getMaximumTotalBytesAllocated`.
   *
   * @return the scoped maximum bytes allocated
   */
  public static long getScopedMaximumBytesAllocated() {
    readLock.lock();
    try {
      if (tracker == null) {
        return 0L;
      } else {
        return tracker.getScopedMaxTotalBytesAllocated();
      }
    } finally {
      readLock.unlock();
    }
  }

  /**
   * Sets the event handler to be called on RMM events (e.g.: allocation failure).
   * @param handler event handler to invoke on RMM events or null to clear an existing handler
   * @throws RmmException if an active handler is already set
   */
  public static void setEventHandler(RmmEventHandler handler) throws RmmException {
    setEventHandler(handler, false);
  }

  /**
   * Sets the event handler to be called on RMM events (e.g.: allocation failure) and
   * optionally enable debug mode (callbacks on every allocate and deallocate)
   * <p>
   * NOTE: Only enable debug mode when necessary, as code will run much slower!
   *
   * @param handler event handler to invoke on RMM events or null to clear an existing handler
   * @param enableDebug if true enable debug callbacks in RmmEventHandler
   *                    (onAllocated, onDeallocated)
   * @throws RmmException if an active handler is already set
   */
  public static void setEventHandler(RmmEventHandler handler,
                                     boolean enableDebug) throws RmmException {
    writeLock.lock();
    try {
      setEventHandlerUnderLock(handler, enableDebug);
    } finally {
      writeLock.unlock();
    }
  }

  private static void setEventHandlerUnderLock(RmmEventHandler handler, boolean enableDebug) throws RmmException {
    if (!initialized) {
      throw new RmmException("RMM has not been initialized");
    }
    if (deviceResource instanceof RmmEventHandlerResourceAdaptor) {
      throw new RmmException("Another event handler is already set");
    }
    if (tracker == null) {
      // This is just to be safe it should always be true if this is initialized.
      throw new RmmException("A tracker must be set for the event handler to work");
    }
    RmmEventHandlerResourceAdaptor<RmmDeviceMemoryResource> newResource =
        new RmmEventHandlerResourceAdaptor<>(deviceResource, tracker, handler, enableDebug);
    boolean success = false;
    try {
      setCurrentDeviceResource(newResource, deviceResource, false);
      success = true;
    } finally {
      if (!success) {
        newResource.releaseWrapped();
      }
    }
  }

  /** Clears the active RMM event handler if one is set. */
  public static void clearEventHandler() throws RmmException {
    writeLock.lock();
    try {
      clearEventHandlerUnderLock();
    } finally {
      writeLock.unlock();
    }
  }

  private static void clearEventHandlerUnderLock() throws RmmException {
    if (deviceResource != null && deviceResource instanceof RmmEventHandlerResourceAdaptor) {
      RmmEventHandlerResourceAdaptor<RmmDeviceMemoryResource> orig =
          (RmmEventHandlerResourceAdaptor<RmmDeviceMemoryResource>)deviceResource;
      boolean success = false;
      try {
        setCurrentDeviceResource(orig.wrapped, orig, false);
        success = true;
      } finally {
        if (success) {
          orig.releaseWrapped();
        }
      }
    }
  }

  public static native void initDefaultCudaDevice();

  public static native void cleanupDefaultCudaDevice();

  /**
   * Shut down any initialized RMM instance.  This should be used very rarely.  It does not need to
   * be used when shutting down your process because CUDA will handle releasing all of the
   * resources when your process exits.  This really should only be used if you want to turn off the
   * memory pool for some reasons.  As such we make an effort to be sure no resources have been
   * leaked before shutting down.  This may involve forcing a JVM GC to collect any leaked java
   * objects that still point to CUDA memory.  By default this will do a gc every 2 seconds and
   * wait for up to 4 seconds before throwing an RmmException if not all of the resources are freed.
   * @throws RmmException on any error. This includes if there are outstanding allocations that
   * could not be collected.
   */
  public static void shutdown() throws RmmException {
    shutdown(2, 4, TimeUnit.SECONDS);
  }

  /**
   * Shut down any initialized RMM instance.  This should be used very rarely.  It does not need to
   * be used when shutting down your process because CUDA will handle releasing all of the
   * resources when your process exits.  This really should only be used if you want to turn off the
   * memory pool for some reasons.  As such we make an effort to be sure no resources have been
   * leaked before shutting down.  This may involve forcing a JVM GC to collect any leaked java
   * objects that still point to CUDA memory.
   *
   * @param forceGCInterval how frequently should we force a JVM GC. This is just a recommendation
   *                        to the JVM to do a gc.
   * @param maxWaitTime the maximum amount of time to wait for all objects to be collected before
   *                    throwing an exception.
   * @param units the units for forceGcInterval and maxWaitTime.
   * @throws RmmException on any error. This includes if there are outstanding allocations that
   * could not be collected before maxWaitTime.
   */
  public static void shutdown(long forceGCInterval, long maxWaitTime, TimeUnit units)
      throws RmmException {
    writeLock.lock();
    try {
      long now = System.currentTimeMillis();
      final long endTime = now + units.toMillis(maxWaitTime);
      long nextGcTime = now;
      try {
        if (MemoryCleaner.bestEffortHasRmmBlockers()) {
          do {
            if (nextGcTime <= now) {
              System.gc();
              nextGcTime = nextGcTime + units.toMillis(forceGCInterval);
            }
            // Check if everything is ready about every 10 ms
            Thread.sleep(10);
            now = System.currentTimeMillis();
          } while (endTime > now && MemoryCleaner.bestEffortHasRmmBlockers());
        }
      } catch (InterruptedException e) {
        // Ignored
      }
      if (MemoryCleaner.bestEffortHasRmmBlockers()) {
        throw new RmmException("Could not shut down RMM there appear to be outstanding allocations");
      }
      if (initialized) {
        if (deviceResource != null) {
          setCurrentDeviceResource(null, deviceResource, true).close();
        }
      }
    } finally {
      writeLock.unlock();
    }
  }

  /**
   * Allocate device memory and return a pointer to device memory, using stream 0.
   * @param size   The size in bytes of the allocated memory region
   * @return Returned pointer to the allocated memory
   */
  public static DeviceMemoryBuffer alloc(long size) {
    return alloc(size, null);
  }

  /**
   * Allocate device memory and return a pointer to device memory.
   * @param size   The size in bytes of the allocated memory region
   * @param stream The stream in which to synchronize this command.
   * @return Returned pointer to the allocated memory
   */
  public static DeviceMemoryBuffer alloc(long size, Cuda.Stream stream) {
    long s = stream == null ? 0 : stream.getStream();
    return new DeviceMemoryBuffer(allocInternal(size, s), size, stream);
  }

  private static native long allocInternal(long size, long stream) throws RmmException;

  static native void free(long ptr, long length, long stream) throws RmmException;

  /**
   * Delete an rmm::device_buffer.
   */
  static native void freeDeviceBuffer(long rmmBufferAddress) throws RmmException;

  /**
   * Allocate device memory using `cudaMalloc` and return a pointer to device memory.
   * @param size   The size in bytes of the allocated memory region
   * @param stream The stream in which to synchronize this command.
   * @return Returned pointer to the allocated memory
   */
  public static CudaMemoryBuffer allocCuda(long size, Cuda.Stream stream) {
    long s = stream == null ? 0 : stream.getStream();
    return new CudaMemoryBuffer(allocCudaInternal(size, s), size, stream);
  }

  private static native long allocCudaInternal(long size, long stream) throws RmmException;

  static native void freeCuda(long ptr, long length, long stream) throws RmmException;

  static native long newCudaMemoryResource() throws RmmException;

  static native void releaseCudaMemoryResource(long handle);

  static native long newManagedMemoryResource() throws RmmException;

  static native void releaseManagedMemoryResource(long handle);

  static native long newPoolMemoryResource(long childHandle,
      long initSize, long maxSize) throws RmmException;

  static native void releasePoolMemoryResource(long handle);

  static native long newArenaMemoryResource(long childHandle,
      long size, boolean dumpOnOOM) throws RmmException;

  static native void releaseArenaMemoryResource(long handle);

  static native long newCudaAsyncMemoryResource(long size, long release, boolean fabric) throws RmmException;

  static native void releaseCudaAsyncMemoryResource(long handle);

  static native long newLimitingResourceAdaptor(long handle, long limit, long align) throws RmmException;

  static native void releaseLimitingResourceAdaptor(long handle);

  static native long newLoggingResourceAdaptor(long handle, int type, String path,
      boolean autoFlush) throws RmmException;

  static native void releaseLoggingResourceAdaptor(long handle);

  static native long newTrackingResourceAdaptor(long handle, long alignment) throws RmmException;

  static native void releaseTrackingResourceAdaptor(long handle);

  static native long nativeGetTotalBytesAllocated(long handle);

  static native long nativeGetMaxTotalBytesAllocated(long handle);

  static native void nativeResetScopedMaxTotalBytesAllocated(long handle, long initValue);

  static native long nativeGetScopedMaxTotalBytesAllocated(long handle);

  static native long newEventHandlerResourceAdaptor(long handle, long trackerHandle,
      RmmEventHandler handler, long[] allocThresholds, long[] deallocThresholds, boolean debug);

  static native long releaseEventHandlerResourceAdaptor(long handle, boolean debug);

  private static native void setCurrentDeviceResourceInternal(long newHandle);

  public static native long newPinnedPoolMemoryResource(long initSize, long maxSize);

  static native long newParallelPinnedPoolMemoryResource(
      long poolSize, int initializationThreads);

  public static native long setCudfPinnedPoolMemoryResource(long poolPtr);

  public static native void releasePinnedPoolMemoryResource(long poolPtr);

  public static native long allocFromPinnedPool(long poolPtr, long size);

  public static native void freeFromPinnedPool(long poolPtr, long ptr, long size);

  // only for tests
  public static native long allocFromFallbackPinnedPool(long size);

  // only for tests
  public static native void freeFromFallbackPinnedPool(long ptr, long size);
}

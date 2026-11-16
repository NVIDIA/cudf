/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * A device memory resource that uses `cudaMallocAsync` and `cudaFreeAsync` for allocation and
 * deallocation.
 */
public class RmmCudaAsyncMemoryResource implements RmmDeviceMemoryResource {
  /** The pool does not support exporting. */
  public static final int HANDLE_TYPE_NONE = 0;
  /** The pool supports exporting fabric handles. */
  public static final int HANDLE_TYPE_FABRIC = 1;
  /** The pool supports exporting POSIX file descriptors. */
  public static final int HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 2;

  private final long releaseThreshold;
  private final long size;
  private final int handleType;
  private final long poolHandle;
  private long handle = 0;

  /**
   * Create a new async memory resource
   * @param size the initial size of the pool
   * @param releaseThreshold size in bytes for when memory is released back to cuda
   */
  public RmmCudaAsyncMemoryResource(long size, long releaseThreshold) {
    this(size, releaseThreshold, false);
  }

  /**
   * Create a new async memory resource
   * @param size the initial size of the pool
   * @param releaseThreshold size in bytes for when memory is released back to cuda
   * @param fabric if true request peer read+write accessible fabric handles when
   *        creating the pool
   */
  public RmmCudaAsyncMemoryResource(long size, long releaseThreshold, boolean fabric) {
    this(size, releaseThreshold, fabric ? HANDLE_TYPE_FABRIC : HANDLE_TYPE_NONE);
  }

  /**
   * Create a new async memory resource whose CUDA memory pool supports the given export handle
   * type.
   * @param size the initial size of the pool
   * @param releaseThreshold size in bytes for when memory is released back to cuda
   * @param handleType one of {@link #HANDLE_TYPE_NONE}, {@link #HANDLE_TYPE_FABRIC} or
   *        {@link #HANDLE_TYPE_POSIX_FILE_DESCRIPTOR}
   */
  public RmmCudaAsyncMemoryResource(long size, long releaseThreshold, int handleType) {
    if (handleType != HANDLE_TYPE_NONE && handleType != HANDLE_TYPE_FABRIC &&
        handleType != HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) {
      throw new IllegalArgumentException("Unsupported handle type " + handleType);
    }
    this.size = size;
    this.releaseThreshold = releaseThreshold;
    this.handleType = handleType;
    long[] created = Rmm.newCudaAsyncMemoryResource(size, releaseThreshold, handleType);
    handle = created[0];
    poolHandle = created[1];
  }

  @Override
  public long getHandle() {
    return handle;
  }

  public long getSize() {
    return size;
  }

  /** The export handle type the underlying CUDA memory pool was created with. */
  public int getHandleType() {
    return handleType;
  }

  /**
   * Get the underlying {@code cudaMemPool_t}. It can be exported with
   * {@link Cuda#memPoolExportToPosixFd(long)} only if the resource was created with
   * {@link #HANDLE_TYPE_POSIX_FILE_DESCRIPTOR}.
   */
  public long getPoolHandle() {
    return poolHandle;
  }

  @Override
  public void close() {
    if (handle != 0) {
      Rmm.releaseCudaAsyncMemoryResource(handle);
      handle = 0;
    }
  }

  @Override
  public String toString() {
    return Long.toHexString(getHandle()) + "/ASYNC(" + size + ", " + releaseThreshold + ")";
  }
}

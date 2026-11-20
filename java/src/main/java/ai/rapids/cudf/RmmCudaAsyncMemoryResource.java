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
  /**
   * The export handle type of the CUDA memory pool. {@code nativeId} matches
   * {@code rmm::mr::cuda_async_memory_resource::allocation_handle_type}, whose values are the
   * CUDA {@code cudaMemAllocationHandleType} values. Like RMM, FABRIC is the literal 0x8 because
   * {@code cudaMemHandleTypeFabric} is not defined in CUDA versions before 12.4.
   */
  public enum HandleType {
    /** The pool does not support exporting ({@code cudaMemHandleTypeNone}). */
    NONE(0x0),
    /** The pool supports exporting POSIX file descriptors
     *  ({@code cudaMemHandleTypePosixFileDescriptor}). */
    POSIX_FILE_DESCRIPTOR(0x1),
    /** The pool supports exporting fabric handles ({@code cudaMemHandleTypeFabric}). */
    FABRIC(0x8);

    final int nativeId;

    HandleType(int nativeId) {
      this.nativeId = nativeId;
    }
  }

  private final long releaseThreshold;
  private final long size;
  private final HandleType handleType;
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
    this(size, releaseThreshold, fabric ? HandleType.FABRIC : HandleType.NONE);
  }

  /**
   * Create a new async memory resource whose CUDA memory pool supports the given export handle
   * type.
   * @param size the initial size of the pool
   * @param releaseThreshold size in bytes for when memory is released back to cuda
   * @param handleType the export handle type of the pool
   */
  public RmmCudaAsyncMemoryResource(long size, long releaseThreshold, HandleType handleType) {
    if (handleType == null) {
      throw new IllegalArgumentException("handleType must not be null");
    }
    this.size = size;
    this.releaseThreshold = releaseThreshold;
    this.handleType = handleType;
    long[] created =
        Rmm.newCudaAsyncMemoryResource(size, releaseThreshold, handleType.nativeId);
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
  public HandleType getHandleType() {
    return handleType;
  }

  /**
   * Get the underlying {@code cudaMemPool_t}. It can be exported with
   * {@link Cuda#memPoolExportToPosixFd(long)} only if the resource was created with
   * {@link HandleType#POSIX_FILE_DESCRIPTOR}.
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

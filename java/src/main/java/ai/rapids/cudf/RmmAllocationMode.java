/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * Bit flags that select the RMM memory allocation strategy.
 * Combine flags with bitwise OR and pass the result to
 * {@link Rmm#initialize(int, Rmm.LogConf, long)}.
 */
public class RmmAllocationMode {
  /**
   * Use cudaMalloc for allocation
   */
  public static final int CUDA_DEFAULT = 0x00000000;
  /**
   * Use pool suballocation strategy
   */
  public static final int POOL = 0x00000001;
  /**
   * Use cudaMallocManaged rather than cudaMalloc
   */
  public static final int CUDA_MANAGED_MEMORY = 0x00000002;
  /**
   * Use arena suballocation strategy
   */
  public static final int ARENA = 0x00000004;
  /**
   * Use CUDA async suballocation strategy
   */
  public static final int CUDA_ASYNC = 0x00000008;
  /**
   * Use CUDA async suballocation strategy with fabric handles that are
   * peer accessible with read-write access
   */
  public static final int CUDA_ASYNC_FABRIC = 0x00000010;
  /**
   * Use CUDA async suballocation strategy with a pool that supports POSIX file descriptor
   * handles, so the pool can be exported with cudaMemPoolExportToShareableHandle and imported
   * by another process on the same host. The pool handle is available from
   * {@link RmmCudaAsyncMemoryResource#getPoolHandle()}.
   */
  public static final int CUDA_ASYNC_POSIX_FD = 0x00000020;
}

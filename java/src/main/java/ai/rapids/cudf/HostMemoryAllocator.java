/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * An allocator of host memory buffers.
 * See {@link DefaultHostMemoryAllocator} for the default implementation.
 */
public interface HostMemoryAllocator {

  /**
   * Allocate memory, but be sure to close the returned buffer to avoid memory leaks.
   * @param bytes size in bytes to allocate
   * @param preferPinned If set to true, the pinned memory pool will be used if possible with a
   *                    fallback to off-heap memory.  If set to false, the allocation will always
   *                    be from off-heap memory.
   * @return the newly created buffer
   */
  HostMemoryBuffer allocate(long bytes, boolean preferPinned);

  /**
   * Allocate memory, but be sure to close the returned buffer to avoid memory leaks. Pinned memory
   * for allocations preference is up to the implementor
   *
   * @param bytes size in bytes to allocate
   * @return the newly created buffer
   */
  HostMemoryBuffer allocate(long bytes);
}

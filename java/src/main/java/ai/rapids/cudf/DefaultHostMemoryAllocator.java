/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * The default {@link HostMemoryAllocator}, which uses pinned memory when requested and
 * available and falls back to off-heap memory.
 * The process-wide default allocator can be changed with
 * {@link #set(HostMemoryAllocator)}.
 */
public class DefaultHostMemoryAllocator implements HostMemoryAllocator {
  private static volatile HostMemoryAllocator instance = new DefaultHostMemoryAllocator();

  /**
   * Retrieve current host memory allocator used by default if not passed directly to API
   *
   * @return current default HostMemoryAllocator implementation
   */
  public static HostMemoryAllocator get() {
    return instance;
  }

  /**
   * Sets a new default host memory allocator implementation by default.
   * @param hostMemoryAllocator the new allocator to use.
   */
  public static void set(HostMemoryAllocator hostMemoryAllocator) {
    instance = hostMemoryAllocator;
  }

  @Override
  public HostMemoryBuffer allocate(long bytes, boolean preferPinned) {
    if (preferPinned) {
      HostMemoryBuffer pinnedBuffer = PinnedMemoryPool.tryAllocate(bytes);
      if (pinnedBuffer != null) {
        return pinnedBuffer;
      }
    }
    return new HostMemoryBuffer(UnsafeMemoryAccessor.allocate(bytes), bytes);
  }

  @Override
  public HostMemoryBuffer allocate(long bytes) {
    return allocate(bytes, HostMemoryBuffer.defaultPreferPinned);
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * The device memory resources used by an operation, mirroring libcudf's
 * {@code cudf::memory_resources}. Memory returned to the caller is allocated from the output
 * resource, and memory used only for the duration of the operation is allocated from the
 * temporary resource.
 * <p>
 * A null resource means the current device resource at the time of the operation
 * (see {@link Rmm#getCurrentDeviceResource()}). A resource must stay open while any memory it
 * allocated is still in use.
 */
public final class MemoryResources {
  /** Allocate both output and temporary memory from the current device resource. */
  public static final MemoryResources CURRENT = new MemoryResources(null, null);

  private final RmmDeviceMemoryResource output;
  private final RmmDeviceMemoryResource temporary;

  /**
   * Allocate output memory from {@code output} and temporary memory from the current device
   * resource.
   * @param output the resource for memory returned to the caller, or null for the current device
   *               resource
   */
  public MemoryResources(RmmDeviceMemoryResource output) {
    this(output, null);
  }

  /**
   * @param output the resource for memory returned to the caller, or null for the current device
   *               resource
   * @param temporary the resource for memory used only during the operation, or null for the
   *                  current device resource
   */
  public MemoryResources(RmmDeviceMemoryResource output, RmmDeviceMemoryResource temporary) {
    this.output = output;
    this.temporary = temporary;
  }

  /** The resource for memory returned to the caller, or null for the current device resource. */
  public RmmDeviceMemoryResource getOutput() {
    return output;
  }

  /**
   * The resource for memory used only during the operation, or null for the current device
   * resource.
   */
  public RmmDeviceMemoryResource getTemporary() {
    return temporary;
  }

  long getOutputHandle() {
    return getHandle(output);
  }

  long getTemporaryHandle() {
    return getHandle(temporary);
  }

  private static long getHandle(RmmDeviceMemoryResource resource) {
    if (resource == null) {
      return 0;
    }
    long handle = resource.getHandle();
    if (handle == 0) {
      throw new IllegalStateException("memory resource " + resource + " is closed");
    }
    return handle;
  }
}

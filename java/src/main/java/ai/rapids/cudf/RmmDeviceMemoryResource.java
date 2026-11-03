/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

/**
 * A resource that allocates/deallocates device memory. This is not intended to be something that
 * a user will just subclass. This is intended to be a wrapper around a C++ class that RMM will
 * use directly.
 */
public interface RmmDeviceMemoryResource extends AutoCloseable {
  /**
   * Returns a pointer to the underlying C++ class that implements an RMM device memory resource
   */
  long getHandle();

  // Remove the exception...
  void close();
}

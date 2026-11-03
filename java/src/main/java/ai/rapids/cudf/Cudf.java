/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

public class Cudf {

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * cuDF copies that are smaller than the threshold will use a kernel to copy, instead
   * of cudaMemcpyAsync.
   */
  public static native void setKernelPinnedCopyThreshold(long kernelPinnedCopyThreshold);

  /**
   * cudf allocations that are smaller than the threshold will use the pinned host
   * memory resource.
   */
  public static native void setPinnedAllocationThreshold(long pinnedAllocationThreshold);
}

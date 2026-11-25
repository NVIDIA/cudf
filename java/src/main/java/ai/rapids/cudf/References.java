/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

final class References {
  private References() {}

  static void reachabilityFence(Object object) {
    if (object != null) {
      synchronized (object) {
        // The monitor operation is a Java 8 reachability fence.
        // TODO replace with java.lang.ref.Reference.reachabilityFence when we move to Java 9+.
      }
    }
  }
}

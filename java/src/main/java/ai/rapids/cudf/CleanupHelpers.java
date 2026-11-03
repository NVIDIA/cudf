/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/** Utility methods for cleaning up resources during exception handling. */
final class CleanupHelpers {
  private CleanupHelpers() {}

  /**
   * Close {@code resource} on a cleanup path, attaching any close-time failure
   * as a suppressed exception on {@code primary} so the original cause is
   * preserved. No-op if {@code resource} is null.
   */
  static void closeAndSuppress(AutoCloseable resource, Throwable primary) {
    if (resource == null) {
      return;
    }
    try {
      resource.close();
    } catch (Throwable closeFailure) {
      primary.addSuppressed(closeFailure);
    }
  }
}

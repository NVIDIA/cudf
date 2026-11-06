/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * A callback interface for RMM memory allocation events.
 * Notified of allocation failures and threshold crossings, and of every allocation and
 * deallocation when debug mode is enabled. Register a handler with
 * {@link Rmm#setEventHandler(RmmEventHandler)}.
 */
public interface RmmEventHandler {
  /**
   * Invoked on a memory allocation failure.
   * @param sizeRequested number of bytes that failed to allocate
   * @deprecated deprecated in favor of onAllocFailure(long, boolean)
   * @return true if the memory allocation should be retried or false if it should fail
   */
  default boolean onAllocFailure(long sizeRequested) {
    // this should not be called since it was the previous interface,
    // and it was abstract before, throwing by default for good measure.
    throw new UnsupportedOperationException(
        "Unexpected invocation of deprecated onAllocFailure without retry count.");
  }

  /**
   * Invoked after every memory allocation when debug mode is enabled.
   * @param size number of bytes allocated
   */
  default void onAllocated(long size) {}

  /**
   * Invoked after every memory deallocation when debug mode is enabled.
   * @param size number of bytes deallocated
   */
  default void onDeallocated(long size) {}

  /**
   * Invoked on a memory allocation failure.
   * @param sizeRequested number of bytes that failed to allocate
   * @param retryCount number of times this allocation has been retried after failure
   * @return true if the memory allocation should be retried or false if it should fail
   */
  default boolean onAllocFailure(long sizeRequested, int retryCount) {
    // newer code should override this implementation of `onAllocFailure` to handle
    // `retryCount`. Otherwise, we call the prior implementation to not
    // break existing code.
    return onAllocFailure(sizeRequested);
  }

  /**
   * Get the memory thresholds that will trigger {@link #onAllocThreshold(long)}
   * to be called when one or more of the thresholds is crossed during a memory allocation.
   * A threshold is crossed when the total memory allocated before the RMM allocate operation
   * is less than a threshold value and the threshold value is less than or equal to the
   * total memory allocated after the RMM memory allocate operation.
   * @return allocate memory thresholds or null for no thresholds.
   */
  long[] getAllocThresholds();

  /**
   * Get the memory thresholds that will trigger {@link #onDeallocThreshold(long)}
   * to be called when one or more of the thresholds is crossed during a memory deallocation.
   * A threshold is crossed when the total memory allocated before the RMM deallocate operation
   * is greater than or equal to a threshold value and the threshold value is greater than the
   * total memory allocated after the RMM memory deallocate operation.
   * @return deallocate memory thresholds or null for no thresholds.
   */
  long[] getDeallocThresholds();

  /**
   * Invoked after an RMM memory allocate operation when an allocate threshold is crossed.
   * See {@link #getAllocThresholds()} for details on allocate threshold crossing.
   * <p>NOTE: Any exception thrown by this method will cause the corresponding allocation
   * that triggered the threshold callback to be released before the exception is
   * propagated to the application.
   * @param totalAllocSize total amount of memory allocated after the crossing
   */
  void onAllocThreshold(long totalAllocSize);

  /**
   * Invoked after an RMM memory deallocation operation when a deallocate threshold is crossed.
   * See {@link #getDeallocThresholds()} for details on deallocate threshold crossing.
   * <p>NOTE: Any exception thrown by this method will be propagated to the application
   * after the resource that triggered the threshold was released.
   * @param totalAllocSize total amount of memory allocated after the crossing
   */
  void onDeallocThreshold(long totalAllocSize);
}

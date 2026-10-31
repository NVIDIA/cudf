/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Per-row status identifiers returned by Variant extraction and cast operations.
 *
 * <p>These values mirror the explicitly assigned numeric values of the experimental C++
 * {@code cudf::io::parquet::experimental::variant_operation_status} enum. The IDs are part of the
 * Java API contract and must not be derived from the Java enum ordinal.
 */
@Experimental
public enum VariantOperationStatus {
  SUCCESS(0),
  ROW_NULL(1),
  MISSING_PATH(2),
  VARIANT_NULL(3),
  TYPE_MISMATCH(4),
  MALFORMED_VARIANT(5),
  OVERFLOW(6),
  INVALID_CONVERSION(7);

  private static final VariantOperationStatus[] STATUSES = VariantOperationStatus.values();

  private final int nativeId;

  VariantOperationStatus(int nativeId) {
    this.nativeId = nativeId;
  }

  /**
   * Get the value stored in the {@code UINT8} status column.
   *
   * @return the native operation status ID
   */
  public int getNativeId() {
    return nativeId;
  }

  /**
   * Find the named operation status for a native ID.
   *
   * @param nativeId ID returned in a status row
   * @return the corresponding operation status
   * @throws IllegalArgumentException if {@code nativeId} is not recognized
   */
  public static VariantOperationStatus fromNative(int nativeId) {
    for (VariantOperationStatus status : STATUSES) {
      if (status.nativeId == nativeId) {
        return status;
      }
    }
    throw new IllegalArgumentException("Unknown Variant operation status ID: " + nativeId);
  }
}

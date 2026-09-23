/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Logical type identifiers returned by {@link VariantUtils#getVariantTypeId(ColumnView)}.
 *
 * <p>These values mirror the explicitly assigned numeric values of the experimental C++
 * {@code cudf::io::parquet::experimental::variant_logical_type} enum. The IDs are part of the
 * Java API contract and must not be derived from the Java enum ordinal.
 */
@Experimental
public enum VariantLogicalType {
  OBJECT(0),
  ARRAY(1),
  NULL_VALUE(2),
  BOOLEAN(3),
  LONG_VALUE(4),
  STRING(5),
  DOUBLE_VALUE(6),
  DECIMAL(7),
  DATE(8),
  TIMESTAMP(9),
  TIMESTAMP_NTZ(10),
  FLOAT_VALUE(11),
  BINARY(12),
  UUID(13),
  TIME_NTZ(14);

  private static final VariantLogicalType[] TYPES = VariantLogicalType.values();

  private final int nativeId;

  VariantLogicalType(int nativeId) {
    this.nativeId = nativeId;
  }

  /**
   * Get the value stored in the {@code UINT8} result column.
   *
   * @return the native logical type ID
   */
  public int getNativeId() {
    return nativeId;
  }

  /**
   * Find the named logical type for a native ID.
   *
   * @param nativeId ID returned in a valid result row
   * @return the corresponding logical type
   * @throws IllegalArgumentException if {@code nativeId} is not recognized
   */
  public static VariantLogicalType fromNative(int nativeId) {
    for (VariantLogicalType type : TYPES) {
      if (type.nativeId == nativeId) {
        return type;
      }
    }
    throw new IllegalArgumentException("Unknown Variant logical type ID: " + nativeId);
  }
}

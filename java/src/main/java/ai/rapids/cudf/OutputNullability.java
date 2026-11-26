/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Whether the output column of a transform may have a null mask. Mirrors
 * {@code cudf::output_nullability}.
 */
public enum OutputNullability {
  /** A null mask is produced if the output can have nulls. */
  PRESERVE(0),
  /**
   * No null mask is produced and every row is valid. It is undefined behavior to use it for an
   * output that has nulls.
   */
  ALL_VALID(1);

  final int nativeId;

  OutputNullability(int nativeId) {
    this.nativeId = nativeId;
  }
}

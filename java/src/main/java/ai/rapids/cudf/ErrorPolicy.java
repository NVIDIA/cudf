/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * How an operation handles rows that fail, such as a division by zero or an overflow.
 * PROPAGATE : the operation throws a {@link CudfException} if any row fails.
 * NULLIFY : rows that fail are null in the output.
 *
 * NOTE: This must be kept in sync with `cudf::error_policy` in cudf/types.hpp!
 */
public enum ErrorPolicy {
  PROPAGATE(0),
  NULLIFY(1);

  final int nativeId;

  ErrorPolicy(int nativeId) { this.nativeId = nativeId; }
}

/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * Widths of the intermediate value that decimal division rescales the dividend into.
 * STANDARD : The widest of the input and output types. A row is null if its rescaled dividend
 * overflows this width.
 * WIDE : 256 bits. A row is null only if the divisor is zero or the quotient doesn't fit in the
 * output type.
 */
public enum DecimalIntermediate {
  STANDARD(0),
  WIDE(1);

  final int nativeId;

  DecimalIntermediate(int nativeId) { this.nativeId = nativeId; }
}

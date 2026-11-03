/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

final class AstUtils {
  private AstUtils() {
  }

  static byte checkByte(int value) {
    byte result = (byte) value;
    if (result != value) {
      throw new IllegalArgumentException("value does not fit in a byte: " + value);
    }
    return result;
  }
}

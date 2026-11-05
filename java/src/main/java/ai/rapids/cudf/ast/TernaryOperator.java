/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;

/**
 * Enumeration of AST operators that can appear in a ternary operation.
 * NOTE: This must be kept in sync with `jni_to_ternary_operator` in CompiledExpression.cpp!
 */
public enum TernaryOperator {
  IF_ELSE(0);  // (condition, then, else): only the selected branch is evaluated. A null
               // condition selects the else branch.

  private final byte nativeId;

  TernaryOperator(int nativeId) {
    this.nativeId = AstUtils.checkByte(nativeId);
  }

  /** Get the size in bytes to serialize this operator */
  int getSerializedSize() {
    return Byte.BYTES;
  }

  /** Serialize this operator to the specified buffer */
  void serialize(ByteBuffer bb) {
    bb.put(nativeId);
  }
}

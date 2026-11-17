/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;

/**
 * Error handling policy for fallible JIT AST operations.
 *
 * NOTE: This must be kept in sync with `jni_to_jit_error_policy` in CompiledExpression.cpp!
 */
public enum JitErrorPolicy {
  /** Propagate an evaluation error to the caller. */
  PROPAGATE(0),
  /** Produce null for a row where evaluation fails. */
  NULLIFY(1);

  private final byte nativeId;

  JitErrorPolicy(int nativeId) {
    this.nativeId = AstUtils.checkByte(nativeId);
  }

  int getSerializedSize() {
    return Byte.BYTES;
  }

  void serialize(ByteBuffer bb) {
    bb.put(nativeId);
  }
}

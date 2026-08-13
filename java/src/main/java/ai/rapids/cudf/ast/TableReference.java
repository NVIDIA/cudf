/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;


import java.nio.ByteBuffer;

/**
 * Enumeration of tables that can be referenced in an AST.
 * NOTE: This must be kept in sync with `jni_to_table_reference` code in CompiledExpression.cpp!
 */
public enum TableReference {
  LEFT(0),
  RIGHT(1);
  // OUTPUT is an AST implementation detail and should not appear in user-built expressions.

  private final byte nativeId;

  TableReference(int nativeId) {
    this.nativeId = AstUtils.checkByte(nativeId);
  }

  /** Get the size in bytes to serialize this table reference */
  int getSerializedSize() {
    return Byte.BYTES;
  }

  /** Serialize this table reference to the specified buffer */
  void serialize(ByteBuffer bb) {
    bb.put(nativeId);
  }
}

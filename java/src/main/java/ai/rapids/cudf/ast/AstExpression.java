/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/** Base class of every node in an AST */
public abstract class AstExpression {
  /**
   * Enumeration for the types of AST nodes that can appear in a serialized AST.
   * NOTE: This must be kept in sync with `jni_serialized_expression_type` in
   * CompiledExpression.cpp!
   */
  protected enum ExpressionType {
    VALID_LITERAL(0),
    NULL_LITERAL(1),
    COLUMN_REFERENCE(2),
    UNARY_EXPRESSION(3),
    BINARY_EXPRESSION(4),
    COLUMN_NAME_REFERENCE(5),
    JIT_EXPRESSION(6);

    private final byte nativeId;

    ExpressionType(int nativeId) {
      this.nativeId = AstUtils.checkByte(nativeId);
    }

    /** Get the size in bytes to serialize this node type */
    int getSerializedSize() {
      return Byte.BYTES;
    }

    /** Serialize this node type to the specified buffer */
    void serialize(ByteBuffer bb) {
      bb.put(nativeId);
    }
  }

  public CompiledExpression compile() {
    int size = getSerializedSize();
    ByteBuffer bb = ByteBuffer.allocate(size);
    bb.order(ByteOrder.nativeOrder());
    serialize(bb);
    return new CompiledExpression(bb.array());
  }

  /** Get the size in bytes of the serialized form of this node and all child nodes */
  abstract int getSerializedSize();

  /**
   * Serialize this node and all child nodes.
   * @param bb buffer to receive the serialized data
   */
  abstract void serialize(ByteBuffer bb);
}

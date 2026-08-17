/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;

/**
 * Enumeration of AST operators that can appear in a binary operation.
 * NOTE: This must be kept in sync with `jni_to_binary_operator` in CompiledExpression.cpp!
 */
public enum BinaryOperator {
  ADD(0),                 // operator +
  SUB(1),                 // operator -
  MUL(2),                 // operator *
  DIV(3),                 // operator / using common type of lhs and rhs
  TRUE_DIV(4),            // operator / after promoting type to floating point
  FLOOR_DIV(5),           // ooperator / after promoting to the common type of lhs and rhs (integral or
                          // floating point), and then flooring the result
  MOD(6),                 // operator %
  PYMOD(7),               // operator % using Python's sign rules for negatives
  POW(8),                 // lhs ^ rhs
  EQUAL(9),               // operator ==
  NULL_EQUAL(10),         // operator == using Spark rules for null inputs
  NOT_EQUAL(11),          // operator !=
  LESS(12),               // operator <
  GREATER(13),            // operator >
  LESS_EQUAL(14),         // operator <=
  GREATER_EQUAL(15),      // operator >=
  BITWISE_AND(16),        // operator &
  BITWISE_OR(17),         // operator |
  BITWISE_XOR(18),        // operator ^
  LOGICAL_AND(19),        // operator &&
  NULL_LOGICAL_AND(20),   // operator && using Spark rules for null inputs
  LOGICAL_OR(21),         // operator ||
  NULL_LOGICAL_OR(22);    // operator || using Spark rules for null inputs

  private final byte nativeId;

  BinaryOperator(int nativeId) {
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

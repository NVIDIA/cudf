/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;

/**
 * Enumeration of AST operators that can appear in a unary operation.
 * NOTE: This must be kept in sync with `jni_to_unary_operator` in CompiledExpression.cpp!
 */
public enum UnaryOperator {
  IDENTITY(0),          // Identity function
  IS_NULL(1),           // Check if operand is null
  SIN(2),               // Trigonometric sine
  COS(3),               // Trigonometric cosine
  TAN(4),               // Trigonometric tangent
  ARCSIN(5),            // Trigonometric sine inverse
  ARCCOS(6),            // Trigonometric cosine inverse
  ARCTAN(7),            // Trigonometric tangent inverse
  SINH(8),              // Hyperbolic sine
  COSH(9),              // Hyperbolic cosine
  TANH(10),              // Hyperbolic tangent
  ARCSINH(11),          // Hyperbolic sine inverse
  ARCCOSH(12),          // Hyperbolic cosine inverse
  ARCTANH(13),          // Hyperbolic tangent inverse
  EXP(14),              // Exponential (base e, Euler number)
  LOG(15),              // Natural Logarithm (base e)
  SQRT(16),             // Square-root (x^0.5)
  CBRT(17),             // Cube-root (x^(1.0/3))
  CEIL(18),             // Smallest integer value not less than arg
  FLOOR(19),            // largest integer value not greater than arg
  ABS(20),              // Absolute value
  RINT(21),             // Rounds the floating-point argument arg to an integer value
  BIT_INVERT(22),       // Bitwise Not (~)
  NOT(23),              // Logical Not (!)
  CAST_TO_INT64(24),    // Cast value to int64_t
  CAST_TO_UINT64(25),   // Cast value to uint64_t
  CAST_TO_FLOAT64(26);  // Cast value to double

  private final byte nativeId;

  UnaryOperator(int nativeId) {
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;

/**
 * The subset of libcudf row-IR operators exposed through the Java AST API. Standard comparison,
 * logical, and mathematical operations remain available through {@link BinaryOperation} and
 * {@link UnaryOperation}.
 * Operators ending in {@code _OVERFLOW} and {@link #CHECK_PRECISION} are fallible and support
 * {@link JitErrorPolicy#NULLIFY}. Other operators require the default
 * {@link JitErrorPolicy#PROPAGATE} policy.
 *
 * NOTE: This must be kept in sync with `jni_to_jit_operator` in CompiledExpression.cpp!
 */
public enum JitOperator {
  /** Return the first non-null input. */
  COALESCE(0),
  /** Convert a nullable boolean input into an always-valid predicate. */
  PREDICATE(1),
  ADD(2),
  SUB(3),
  MUL(4),
  /** Divide without reporting arithmetic errors. */
  DIV(5),
  NEG(6),
  ABS(7),
  MOD(8),
  ADD_OVERFLOW(9),
  SUB_OVERFLOW(10),
  MUL_OVERFLOW(11),
  /** Divide and report division-by-zero and signed-overflow errors. */
  DIV_OVERFLOW(12),
  NEG_OVERFLOW(13),
  ABS_OVERFLOW(14),
  MOD_OVERFLOW(15),
  /** Verify that decimal input 0 fits the INT32 precision supplied by input 1. */
  CHECK_PRECISION(16),
  BITWISE_SHIFT_LEFT(17),
  BITWISE_SHIFT_RIGHT(18),
  CAST_TO_BOOL8(19),
  CAST_TO_INT8(20),
  CAST_TO_INT16(21),
  CAST_TO_INT32(22),
  CAST_TO_INT64(23),
  CAST_TO_UINT8(24),
  CAST_TO_UINT16(25),
  CAST_TO_UINT32(26),
  CAST_TO_UINT64(27),
  CAST_TO_FLOAT32(28),
  CAST_TO_FLOAT64(29),
  CAST_TO_DECIMAL32(30),
  CAST_TO_DECIMAL64(31),
  CAST_TO_DECIMAL128(32),
  /** Rescale a decimal input to the target scale supplied to {@link JitOperation}. */
  RESCALE(33),
  /** Select input 0 when input 2 is true, otherwise input 1. */
  IF_ELSE(34);

  private final byte nativeId;

  JitOperator(int nativeId) {
    this.nativeId = AstUtils.checkByte(nativeId);
  }

  int getSerializedSize() {
    return Byte.BYTES;
  }

  void serialize(ByteBuffer bb) {
    bb.put(nativeId);
  }
}

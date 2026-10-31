/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;

/** A ternary operation consisting of an operator and three operands. */
public final class TernaryOperation extends AstExpression {
  private final TernaryOperator op;
  private final AstExpression firstInput;
  private final AstExpression secondInput;
  private final AstExpression thirdInput;

  public TernaryOperation(TernaryOperator op, AstExpression firstInput,
      AstExpression secondInput, AstExpression thirdInput) {
    this.op = op;
    this.firstInput = firstInput;
    this.secondInput = secondInput;
    this.thirdInput = thirdInput;
  }

  @Override
  int getSerializedSize() {
    return ExpressionType.TERNARY_EXPRESSION.getSerializedSize() +
        op.getSerializedSize() +
        firstInput.getSerializedSize() +
        secondInput.getSerializedSize() +
        thirdInput.getSerializedSize();
  }

  @Override
  void serialize(ByteBuffer bb) {
    ExpressionType.TERNARY_EXPRESSION.serialize(bb);
    op.serialize(bb);
    firstInput.serialize(bb);
    secondInput.serialize(bb);
    thirdInput.serialize(bb);
  }

  @Override
  public String toString() {
    return op + "(" + firstInput + ", " + secondInput + ", " + thirdInput + ")";
  }
}

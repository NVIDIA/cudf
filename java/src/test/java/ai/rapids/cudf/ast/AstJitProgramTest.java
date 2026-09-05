/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.CudfException;
import ai.rapids.cudf.CudfTestBase;
import ai.rapids.cudf.Table;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;

public class AstJitProgramTest extends CudfTestBase {
  @Test
  void testReusesMultiOutputProgramAndOwnsLiterals() {
    AstExpression shared = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), new ColumnReference(1));
    AstExpression multiply = new JitOperation(JitOperator.MUL, shared, Literal.ofInt(2));
    AstExpression sum = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), new ColumnReference(1));

    AstJitProgram program;
    try (Table schemaTable = new Table.TestBuilder()
             .column(1, 2, 3)
             .column(10, 20, 30)
             .column(100, 200, 300)
             .build();
         CompiledExpression multiplyCompiled = multiply.compileJit();
         CompiledExpression sumCompiled = sum.compileJit()) {
      program = AstJitProgram.compile(schemaTable, multiplyCompiled, sumCompiled);
    }

    try (AstJitProgram closeableProgram = program;
         Table firstInput = new Table.TestBuilder()
             .column(4, 5)
             .column(40, 50)
             .build();
         Table firstResult = closeableProgram.computeTable(firstInput);
         ColumnVector firstMultiply = ColumnVector.fromInts(88, 110);
         ColumnVector firstSum = ColumnVector.fromInts(44, 55);
         Table secondInput = new Table.TestBuilder()
             .column(6, 7, 8, 9)
             .column(60, 70, 80, 90)
             .column(600L, 700L, 800L, 900L)
             .build();
         Table secondResult = closeableProgram.computeTable(secondInput);
         ColumnVector secondMultiply = ColumnVector.fromInts(132, 154, 176, 198);
         ColumnVector secondSum = ColumnVector.fromInts(66, 77, 88, 99)) {
      Assertions.assertEquals(2, firstResult.getNumberOfColumns());
      assertColumnsAreEqual(firstMultiply, firstResult.getColumn(0));
      assertColumnsAreEqual(firstSum, firstResult.getColumn(1));
      Assertions.assertEquals(2, secondResult.getNumberOfColumns());
      assertColumnsAreEqual(secondMultiply, secondResult.getColumn(0));
      assertColumnsAreEqual(secondSum, secondResult.getColumn(1));
    }
  }

  @Test
  void testReusesSingleOutputProgram() {
    AstExpression expression = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), Literal.ofInt(1));
    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = expression.compileJit();
         AstJitProgram program = AstJitProgram.compile(schemaTable, compiled);
         Table firstInput = new Table.TestBuilder().column(4, 5).build();
         Table firstResult = program.computeTable(firstInput);
         ColumnVector firstExpected = ColumnVector.fromInts(5, 6);
         Table secondInput = new Table.TestBuilder().column(6, 7, 8, 9).build();
         Table secondResult = program.computeTable(secondInput);
         ColumnVector secondExpected = ColumnVector.fromInts(7, 8, 9, 10)) {
      Assertions.assertEquals(1, firstResult.getNumberOfColumns());
      assertColumnsAreEqual(firstExpected, firstResult.getColumn(0));
      Assertions.assertEquals(1, secondResult.getNumberOfColumns());
      assertColumnsAreEqual(secondExpected, secondResult.getColumn(0));
    }
  }

  @Test
  void testRejectsIncompatibleReferencedColumnSchema() {
    AstExpression expression = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), Literal.ofInt(1));
    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = expression.compileJit();
         AstJitProgram program = AstJitProgram.compile(schemaTable, compiled);
         Table wrongType = new Table.TestBuilder().column(1L, 2L, 3L).build();
         Table nullable = new Table.TestBuilder().column(1, null, 3).build()) {
      Assertions.assertThrows(CudfException.class, () -> program.computeTable(wrongType).close());
      Assertions.assertThrows(CudfException.class, () -> program.computeTable(nullable).close());
    }
  }

  @Test
  void testValidation() {
    AstExpression expression = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), Literal.ofInt(1));
    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = expression.compileJit()) {
      Assertions.assertThrows(NullPointerException.class,
          () -> AstJitProgram.compile(null, compiled));
      Assertions.assertThrows(NullPointerException.class,
          () -> AstJitProgram.compile(schemaTable, (CompiledExpression[]) null));
      Assertions.assertThrows(IllegalArgumentException.class,
          () -> AstJitProgram.compile(schemaTable));
      Assertions.assertThrows(NullPointerException.class,
          () -> AstJitProgram.compile(schemaTable, compiled, null));
    }

    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build()) {
      CompiledExpression closedExpression = expression.compileJit();
      closedExpression.close();
      Assertions.assertThrows(IllegalStateException.class,
          () -> AstJitProgram.compile(schemaTable, closedExpression));
    }

    AstExpression defaultExpression = new BinaryOperation(BinaryOperator.ADD,
        new ColumnReference(0), Literal.ofInt(1));
    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = defaultExpression.compile()) {
      Assertions.assertThrows(IllegalArgumentException.class,
          () -> AstJitProgram.compile(schemaTable, compiled));
    }

    try (Table closedTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = expression.compileJit()) {
      closedTable.close();
      Assertions.assertThrows(IllegalStateException.class,
          () -> AstJitProgram.compile(closedTable, compiled));
    }
  }

  @Test
  void testClosedInputs() {
    AstExpression expression = new JitOperation(JitOperator.ADD,
        new ColumnReference(0), Literal.ofInt(1));
    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = expression.compileJit()) {
      AstJitProgram program = AstJitProgram.compile(schemaTable, compiled);
      Assertions.assertThrows(NullPointerException.class, () -> program.computeTable(null));
      program.close();
      Assertions.assertThrows(IllegalStateException.class,
          () -> program.computeTable(schemaTable));
      Assertions.assertThrows(IllegalStateException.class, program::close);
    }

    try (Table schemaTable = new Table.TestBuilder().column(1, 2, 3).build();
         CompiledExpression compiled = expression.compileJit();
         AstJitProgram program = AstJitProgram.compile(schemaTable, compiled);
         Table closedTable = new Table.TestBuilder().column(4, 5, 6).build()) {
      closedTable.close();
      Assertions.assertThrows(IllegalStateException.class,
          () -> program.computeTable(closedTable));
    }
  }
}

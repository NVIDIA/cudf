/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import ai.rapids.cudf.ast.BinaryOperation;
import ai.rapids.cudf.ast.BinaryOperator;
import ai.rapids.cudf.ast.ColumnReference;
import ai.rapids.cudf.ast.CompiledExpression;
import ai.rapids.cudf.ast.TableReference;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.function.Function;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.params.provider.Arguments.arguments;

/**
 * Join APIs taking a {@link CompiledExpression} condition do not support expressions produced by
 * {@link ai.rapids.cudf.ast.AstExpression#compileJit()}.
 */
public class JitConditionTest extends CudfTestBase {
  private static Stream<Arguments> joinCalls() {
    return Stream.of(
        call("conditionalLeftJoinRowCount",
            f -> f.left.conditionalLeftJoinRowCount(f.right, f.jitCondition)),
        call("conditionalLeftJoinGatherMaps",
            f -> f.left.conditionalLeftJoinGatherMaps(f.right, f.jitCondition)),
        call("conditionalLeftJoinGatherMaps with row count",
            f -> f.left.conditionalLeftJoinGatherMaps(f.right, f.jitCondition,
                f.left.conditionalLeftJoinRowCount(f.right, f.condition))),
        call("conditionalInnerJoinRowCount",
            f -> f.left.conditionalInnerJoinRowCount(f.right, f.jitCondition)),
        call("conditionalInnerJoinGatherMaps",
            f -> f.left.conditionalInnerJoinGatherMaps(f.right, f.jitCondition)),
        call("conditionalInnerJoinGatherMaps with row count",
            f -> f.left.conditionalInnerJoinGatherMaps(f.right, f.jitCondition,
                f.left.conditionalInnerJoinRowCount(f.right, f.condition))),
        call("conditionalFullJoinGatherMaps",
            f -> f.left.conditionalFullJoinGatherMaps(f.right, f.jitCondition)),
        call("conditionalLeftSemiJoinRowCount",
            f -> f.left.conditionalLeftSemiJoinRowCount(f.right, f.jitCondition)),
        call("conditionalLeftSemiJoinGatherMap",
            f -> f.left.conditionalLeftSemiJoinGatherMap(f.right, f.jitCondition)),
        call("conditionalLeftSemiJoinGatherMap with row count",
            f -> f.left.conditionalLeftSemiJoinGatherMap(f.right, f.jitCondition,
                f.left.conditionalLeftSemiJoinRowCount(f.right, f.condition))),
        call("conditionalLeftAntiJoinRowCount",
            f -> f.left.conditionalLeftAntiJoinRowCount(f.right, f.jitCondition)),
        call("conditionalLeftAntiJoinGatherMap",
            f -> f.left.conditionalLeftAntiJoinGatherMap(f.right, f.jitCondition)),
        call("conditionalLeftAntiJoinGatherMap with row count",
            f -> f.left.conditionalLeftAntiJoinGatherMap(f.right, f.jitCondition,
                f.left.conditionalLeftAntiJoinRowCount(f.right, f.condition))),
        call("mixedLeftJoinSize",
            f -> Table.mixedLeftJoinSize(f.leftKeys, f.rightKeys, f.left, f.right,
                f.jitCondition, NullEquality.UNEQUAL)),
        call("mixedLeftJoinGatherMaps",
            f -> Table.mixedLeftJoinGatherMaps(f.leftKeys, f.rightKeys, f.left, f.right,
                f.jitCondition, NullEquality.UNEQUAL)),
        call("mixedLeftJoinGatherMaps with size", f -> {
          try (MixedJoinSize size = Table.mixedLeftJoinSize(f.leftKeys, f.rightKeys, f.left,
              f.right, f.condition, NullEquality.UNEQUAL)) {
            return Table.mixedLeftJoinGatherMaps(f.leftKeys, f.rightKeys, f.left, f.right,
                f.jitCondition, NullEquality.UNEQUAL, size);
          }
        }),
        call("mixedInnerJoinSize",
            f -> Table.mixedInnerJoinSize(f.leftKeys, f.rightKeys, f.left, f.right,
                f.jitCondition, NullEquality.UNEQUAL)),
        call("mixedInnerJoinGatherMaps",
            f -> Table.mixedInnerJoinGatherMaps(f.leftKeys, f.rightKeys, f.left, f.right,
                f.jitCondition, NullEquality.UNEQUAL)),
        call("mixedInnerJoinGatherMaps with size", f -> {
          try (MixedJoinSize size = Table.mixedInnerJoinSize(f.leftKeys, f.rightKeys, f.left,
              f.right, f.condition, NullEquality.UNEQUAL)) {
            return Table.mixedInnerJoinGatherMaps(f.leftKeys, f.rightKeys, f.left, f.right,
                f.jitCondition, NullEquality.UNEQUAL, size);
          }
        }),
        call("mixedFullJoinGatherMaps",
            f -> Table.mixedFullJoinGatherMaps(f.leftKeys, f.rightKeys, f.left, f.right,
                f.jitCondition, NullEquality.UNEQUAL)),
        call("mixedLeftSemiJoinGatherMap",
            f -> Table.mixedLeftSemiJoinGatherMap(f.leftKeys, f.rightKeys, f.left, f.right,
                f.jitCondition, NullEquality.UNEQUAL)),
        call("mixedLeftAntiJoinGatherMap",
            f -> Table.mixedLeftAntiJoinGatherMap(f.leftKeys, f.rightKeys, f.left, f.right,
                f.jitCondition, NullEquality.UNEQUAL)),
        call("filterJoinGatherMaps", f -> {
          GatherMap[] maps = f.leftKeys.innerJoinGatherMaps(f.rightKeys, false);
          try {
            return Table.filterJoinGatherMaps(maps[0], maps[1], f.left, f.right,
                f.jitCondition, JoinKind.INNER);
          } finally {
            closeResult(maps);
          }
        }));
  }

  private static Arguments call(String name, Function<Fixture, Object> call) {
    return arguments(name, call);
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("joinCalls")
  void testJitConditionUnsupported(String name, Function<Fixture, Object> call) {
    try (Fixture f = new Fixture()) {
      CudfException e = assertThrows(CudfException.class, () -> closeResult(call.apply(f)));
      assertEquals("JIT-compiled expressions cannot be used by a default AST consumer",
          e.getMessage());
    }
  }

  private static void closeResult(Object result) {
    if (result instanceof Long) {
      // row count, nothing to close
    } else if (result instanceof AutoCloseable[]) {
      for (AutoCloseable c : (AutoCloseable[]) result) {
        closeResult(c);
      }
    } else if (result instanceof AutoCloseable) {
      try {
        ((AutoCloseable) result).close();
      } catch (Exception e) {
        throw new RuntimeException(e);
      }
    } else {
      throw new IllegalArgumentException("Unexpected result: " + result);
    }
  }

  private static final class Fixture implements AutoCloseable {
    final Table left;
    final Table right;
    final Table leftKeys;
    final Table rightKeys;
    /** The condition compiled for the default AST consumer, used to size outputs. */
    final CompiledExpression condition;
    /** The same condition compiled for JIT */
    final CompiledExpression jitCondition;

    Fixture() {
      BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
          new ColumnReference(1, TableReference.LEFT),
          new ColumnReference(1, TableReference.RIGHT));
      Table left = null;
      Table right = null;
      Table leftKeys = null;
      Table rightKeys = null;
      CompiledExpression condition = null;
      CompiledExpression jitCondition = null;
      try {
        left = new Table.TestBuilder()
            .column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8)
            .column(1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
            .build();
        right = new Table.TestBuilder()
            .column(6, 5, 9, 8, 10, 32)
            .column(0, 1, 2, 3, 4, 5)
            .build();
        leftKeys = new Table(left.getColumn(0));
        rightKeys = new Table(right.getColumn(0));
        condition = expr.compile();
        jitCondition = expr.compileJit();
      } catch (Throwable t) {
        for (AutoCloseable c : new AutoCloseable[]{
            jitCondition, condition, rightKeys, leftKeys, right, left}) {
          if (c != null) {
            try {
              c.close();
            } catch (Throwable closeEx) {
              t.addSuppressed(closeEx);
            }
          }
        }
        throw t;
      }
      this.left = left;
      this.right = right;
      this.leftKeys = leftKeys;
      this.rightKeys = rightKeys;
      this.condition = condition;
      this.jitCondition = jitCondition;
    }

    @Override
    public void close() {
      jitCondition.close();
      condition.close();
      rightKeys.close();
      leftKeys.close();
      right.close();
      left.close();
    }
  }
}

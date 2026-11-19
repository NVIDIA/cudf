/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.stream.Stream;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static ai.rapids.cudf.AssertUtils.assertGatherMapsEqualUnordered;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class JoinMatchContextTest {

  @ParameterizedTest
  @EnumSource(JoinKind.class)
  void testOutputSize(JoinKind kind) {
    try (Table build = new Table.TestBuilder().column(1, 2, 2, 3, 4).build();
         HashJoin hashJoin = new HashJoin(build, true);
         Table probe = new Table.TestBuilder().column(2, 3, 5).build();
         JoinMatchContext ctx = createContext(probe, hashJoin, kind)) {
      // A full join's output size only counts rows from the probe side, so it matches the left
      // join. Unmatched build rows are added later by finalizePartitionedFullJoin.
      long expected = kind == JoinKind.INNER
          ? probe.innerJoinRowCount(hashJoin)
          : probe.leftJoinRowCount(hashJoin);
      assertEquals(expected, ctx.getOutputSize());
    }
  }

  @Test
  void testGetJoinKind() {
    try (Table build = new Table.TestBuilder().column(1).build();
         HashJoin hashJoin = new HashJoin(build, false);
         Table probe = new Table.TestBuilder().column(1).build();
         JoinMatchContext inner = probe.innerJoinMatchContext(hashJoin);
         JoinMatchContext left = probe.leftJoinMatchContext(hashJoin);
         JoinMatchContext full = probe.fullJoinMatchContext(hashJoin)) {
      assertSame(JoinKind.INNER, inner.getJoinKind());
      assertSame(JoinKind.LEFT, left.getJoinKind());
      assertSame(JoinKind.FULL, full.getJoinKind());
    }
  }

  @Test
  void testMatchCounts() {
    try (Table build = new Table.TestBuilder().column(0, 1, 1, 2, 3).build();
         HashJoin hashJoin = new HashJoin(build, true);
         Table probe = new Table.TestBuilder().column(1, 2, 4).build()) {
      try (JoinMatchContext inner = probe.innerJoinMatchContext(hashJoin);
           ColumnVector counts = inner.getMatchCounts();
           ColumnVector expected = ColumnVector.fromInts(2, 1, 0)) {
        assertColumnsAreEqual(expected, counts);
      }
      try (JoinMatchContext left = probe.leftJoinMatchContext(hashJoin);
           ColumnVector counts = left.getMatchCounts();
           ColumnVector expected = ColumnVector.fromInts(2, 1, 1)) {
        assertColumnsAreEqual(expected, counts);
      }
    }
  }

  @Test
  void testEmptyBuildTable() {
    try (Table build = new Table.TestBuilder().column(new Integer[0]).build();
         HashJoin hashJoin = new HashJoin(build, false);
         Table probe = new Table.TestBuilder().column(1, 2, 3).build()) {
      try (JoinMatchContext inner = probe.innerJoinMatchContext(hashJoin)) {
        assertEquals(0, inner.getOutputSize());
      }
      try (JoinMatchContext left = probe.leftJoinMatchContext(hashJoin);
           ColumnVector counts = left.getMatchCounts();
           ColumnVector expected = ColumnVector.fromInts(1, 1, 1)) {
        assertEquals(3, left.getOutputSize());
        assertColumnsAreEqual(expected, counts);
      }
    }
  }

  private static Stream<Arguments> partitionCases() {
    final int inv = Integer.MIN_VALUE;
    return Stream.of(
        Arguments.of(JoinKind.INNER,
            new Integer[][]{{0, 0}, {1, 2}},
            new Integer[][]{{2}, {4}},
            null),
        Arguments.of(JoinKind.LEFT,
            new Integer[][]{{0, 0, 1}, {1, 2, inv}},
            new Integer[][]{{2, 3}, {4, inv}},
            null),
        // The finalized full join adds the unmatched build rows 0 and 3.
        Arguments.of(JoinKind.FULL,
            new Integer[][]{{0, 0, 1}, {1, 2, inv}},
            new Integer[][]{{2, 3}, {4, inv}},
            new Integer[][]{{0, 0, 1, 2, 3, inv, inv}, {1, 2, inv, 4, inv, 0, 3}}));
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("partitionCases")
  void testPartitionGatherMaps(JoinKind kind, Integer[][] expected0, Integer[][] expected1,
                               Integer[][] expectedFinalized) {
    try (Table build = new Table.TestBuilder().column(0, 1, 1, 2, 3).build();
         HashJoin hashJoin = new HashJoin(build, true);
         Table probe = new Table.TestBuilder().column(1, 5, 3, 7).build();
         JoinMatchContext ctx = createContext(probe, hashJoin, kind);
         Table expectedP0 = indexTable(expected0);
         Table expectedP1 = indexTable(expected1);
         CloseableArray<GatherMap> p0 = CloseableArray.wrap(ctx.partitionGatherMaps(0, 2));
         CloseableArray<GatherMap> p1 = CloseableArray.wrap(ctx.partitionGatherMaps(2, 4))) {
      assertGatherMapsEqualUnordered(expectedP0, p0.getArray());
      assertGatherMapsEqualUnordered(expectedP1, p1.getArray());
      if (kind == JoinKind.FULL) {
        try (Table expected = indexTable(expectedFinalized);
             CloseableArray<GatherMap> finalized = CloseableArray.wrap(
                 ctx.finalizePartitionedFullJoin(p0.getArray(), p1.getArray()))) {
          assertGatherMapsEqualUnordered(expected, finalized.getArray());
        }
      }
    }
  }

  @Test
  void testColumnCountMismatch() {
    try (Table build = new Table.TestBuilder().column(1, 2).build();
         HashJoin hashJoin = new HashJoin(build, false);
         Table probe = new Table.TestBuilder().column(1, 2).column(3, 4).build()) {
      assertThrows(IllegalArgumentException.class, () -> probe.innerJoinMatchContext(hashJoin));
      assertThrows(IllegalArgumentException.class, () -> probe.leftJoinMatchContext(hashJoin));
      assertThrows(IllegalArgumentException.class, () -> probe.fullJoinMatchContext(hashJoin));
    }
  }

  @Test
  void testClosedHashJoinRejected() {
    try (Table build = new Table.TestBuilder().column(1).build();
         Table probe = new Table.TestBuilder().column(1).build()) {
      HashJoin hashJoin = new HashJoin(build, false);
      hashJoin.close();
      assertThrows(IllegalStateException.class, () -> probe.innerJoinMatchContext(hashJoin));
    }
  }

  @Test
  void testDoubleCloseThrows() {
    try (Table build = new Table.TestBuilder().column(1).build();
         HashJoin hashJoin = new HashJoin(build, false);
         Table probe = new Table.TestBuilder().column(1).build()) {
      JoinMatchContext ctx = probe.innerJoinMatchContext(hashJoin);
      ctx.close();
      assertThrows(IllegalStateException.class, ctx::close);
    }
  }

  @Test
  void testClosedContextRejects() {
    try (Table build = new Table.TestBuilder().column(1).build();
         HashJoin hashJoin = new HashJoin(build, false);
         Table probe = new Table.TestBuilder().column(1).build()) {
      JoinMatchContext ctx = probe.innerJoinMatchContext(hashJoin);
      ctx.close();
      assertThrows(IllegalStateException.class, ctx::getOutputSize);
      assertThrows(IllegalStateException.class, ctx::getMatchCounts);
      assertThrows(IllegalStateException.class, () -> ctx.partitionGatherMaps(0, 1));
      assertThrows(IllegalStateException.class, () -> ctx.finalizePartitionedFullJoin());
    }
  }

  @Test
  void testRetainsLeftKeysAfterOriginalTableCloses() {
    try (Table build = new Table.TestBuilder().column(1, 2, 3).build();
         HashJoin hashJoin = new HashJoin(build, false);
         ColumnVector leftKeys = ColumnVector.fromInts(1, 2)) {
      final JoinMatchContext ctx;
      try (Table probe = new Table(leftKeys)) {
        ctx = probe.innerJoinMatchContext(hashJoin);
      }
      try (JoinMatchContext toClose = ctx) {
        assertEquals(2, leftKeys.getRefCount());
        assertEquals(2, ctx.getOutputSize());
      }
      assertEquals(1, leftKeys.getRefCount());
    }
  }

  @ParameterizedTest
  @EnumSource(value = JoinKind.class, names = {"INNER", "LEFT"})
  void testFinalizeRequiresFullJoin(JoinKind kind) {
    try (Table build = new Table.TestBuilder().column(1).build();
         HashJoin hashJoin = new HashJoin(build, false);
         Table probe = new Table.TestBuilder().column(1, 2).build();
         JoinMatchContext ctx = createContext(probe, hashJoin, kind);
         CloseableArray<GatherMap> p0 = CloseableArray.wrap(ctx.partitionGatherMaps(0, 2))) {
      assertThrows(IllegalStateException.class,
          () -> ctx.finalizePartitionedFullJoin(p0.getArray()));
    }
  }

  @Test
  void testFinalizeRejectsMalformedPartition() {
    try (Table build = new Table.TestBuilder().column(1).build();
         HashJoin hashJoin = new HashJoin(build, false);
         Table probe = new Table.TestBuilder().column(1, 2).build();
         JoinMatchContext ctx = probe.fullJoinMatchContext(hashJoin);
         CloseableArray<GatherMap> p0 = CloseableArray.wrap(ctx.partitionGatherMaps(0, 2))) {
      assertThrows(IllegalArgumentException.class,
          () -> ctx.finalizePartitionedFullJoin(new GatherMap[]{p0.get(0)}));
    }
  }

  @Test
  void testFinalizeRejectsMismatchedPairLengths() {
    try (Table build = new Table.TestBuilder().column(0, 1, 1, 2, 3).build();
         HashJoin hashJoin = new HashJoin(build, true);
         Table probe = new Table.TestBuilder().column(1, 5, 3, 7).build();
         JoinMatchContext ctx = probe.fullJoinMatchContext(hashJoin);
         CloseableArray<GatherMap> p0 = CloseableArray.wrap(ctx.partitionGatherMaps(0, 2));
         CloseableArray<GatherMap> p1 = CloseableArray.wrap(ctx.partitionGatherMaps(2, 4))) {
      // The p0 left map has 3 rows and the p1 right map has 2.
      assertThrows(IllegalArgumentException.class,
          () -> ctx.finalizePartitionedFullJoin(new GatherMap[]{p0.get(0), p1.get(1)}));
    }
  }

  private static JoinMatchContext createContext(Table probe, HashJoin hashJoin, JoinKind kind) {
    switch (kind) {
      case INNER: return probe.innerJoinMatchContext(hashJoin);
      case LEFT: return probe.leftJoinMatchContext(hashJoin);
      case FULL: return probe.fullJoinMatchContext(hashJoin);
      default: throw new IllegalArgumentException("Unsupported join kind " + kind);
    }
  }

  private static Table indexTable(Integer[][] pairs) {
    return new Table.TestBuilder().column(pairs[0]).column(pairs[1]).build();
  }
}

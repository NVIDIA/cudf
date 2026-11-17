/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static ai.rapids.cudf.AssertUtils.assertTablesAreEqual;
import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class JoinMatchContextTest {

  @Test
  void testOutputSizeMatchesInnerRowCount() {
    try (Table build = new Table.TestBuilder().column(1, 2, 2, 3, 4).build();
         HashJoin hashJoin = new HashJoin(build, true);
         Table probe = new Table.TestBuilder().column(2, 3, 5).build();
         JoinMatchContext ctx = probe.innerJoinMatchContext(hashJoin)) {
      assertEquals(probe.innerJoinRowCount(hashJoin), ctx.getOutputSize());
    }
  }

  @Test
  void testOutputSizeMatchesLeftRowCount() {
    try (Table build = new Table.TestBuilder().column(1, 2, 2, 3, 4).build();
         HashJoin hashJoin = new HashJoin(build, true);
         Table probe = new Table.TestBuilder().column(2, 3, 5).build();
         JoinMatchContext ctx = probe.leftJoinMatchContext(hashJoin)) {
      assertEquals(probe.leftJoinRowCount(hashJoin), ctx.getOutputSize());
    }
  }

  @Test
  void testFullOutputSizeExcludesUnmatchedRight() {
    try (Table build = new Table.TestBuilder().column(1, 2, 2, 3, 4).build();
         HashJoin hashJoin = new HashJoin(build, true);
         Table probe = new Table.TestBuilder().column(2, 3, 5).build();
         JoinMatchContext ctx = probe.fullJoinMatchContext(hashJoin)) {
      // A full join's output size only counts rows from the probe side, so it matches the left
      // join. Unmatched build rows are added later by finalizePartitionedFullJoin.
      assertEquals(probe.leftJoinRowCount(hashJoin), ctx.getOutputSize());
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

  @Test
  void testPartitionedInnerMatchesWholeJoin() {
    try (Table build = new Table.TestBuilder().column(0, 1, 1, 2, 3, 3, 4).build();
         HashJoin hashJoin = new HashJoin(build, true);
         Table probe = new Table.TestBuilder().column(1, 3, 2, 5, 3, 0).build();
         JoinMatchContext ctx = probe.innerJoinMatchContext(hashJoin)) {
      GatherMap[] p0 = null, p1 = null, p2 = null, whole = null;
      try {
        p0 = ctx.partitionGatherMaps(0, 2);
        p1 = ctx.partitionGatherMaps(2, 4);
        p2 = ctx.partitionGatherMaps(4, 6);
        whole = probe.innerJoinGatherMaps(hashJoin);
        try (Table expected = toSortedIndexTable(whole);
             Table actual = toSortedIndexTable(Arrays.asList(p0, p1, p2))) {
          assertTablesAreEqual(expected, actual);
        }
      } finally {
        closeAll(p0);
        closeAll(p1);
        closeAll(p2);
        closeAll(whole);
      }
    }
  }

  @Test
  void testPartitionedLeftMatchesWholeJoin() {
    try (Table build = new Table.TestBuilder().column(0, 1, 1, 2, 3).build();
         HashJoin hashJoin = new HashJoin(build, true);
         Table probe = new Table.TestBuilder().column(1, 5, 3, 7).build();
         JoinMatchContext ctx = probe.leftJoinMatchContext(hashJoin)) {
      GatherMap[] p0 = null, p1 = null, whole = null;
      try {
        p0 = ctx.partitionGatherMaps(0, 2);
        p1 = ctx.partitionGatherMaps(2, 4);
        whole = probe.leftJoinGatherMaps(hashJoin);
        try (Table expected = toSortedIndexTable(whole);
             Table actual = toSortedIndexTable(Arrays.asList(p0, p1))) {
          assertTablesAreEqual(expected, actual);
        }
      } finally {
        closeAll(p0);
        closeAll(p1);
        closeAll(whole);
      }
    }
  }

  @Test
  void testPartitionedFullMatchesWholeJoin() {
    try (Table build = new Table.TestBuilder().column(0, 1, 1, 2, 3).build();
         HashJoin hashJoin = new HashJoin(build, true);
         Table probe = new Table.TestBuilder().column(1, 5, 3, 7).build();
         JoinMatchContext ctx = probe.fullJoinMatchContext(hashJoin)) {
      GatherMap[] p0 = null, p1 = null, finalized = null, whole = null;
      try {
        p0 = ctx.partitionGatherMaps(0, 2);
        p1 = ctx.partitionGatherMaps(2, 4);
        finalized = JoinMatchContext.finalizePartitionedFullJoin(
            new GatherMap[]{p0[0], p1[0]},
            new GatherMap[]{p0[1], p1[1]},
            probe.getRowCount(), build.getRowCount());
        whole = probe.fullJoinGatherMaps(hashJoin);
        try (Table expected = toSortedIndexTable(whole);
             Table actual = toSortedIndexTable(finalized)) {
          assertTablesAreEqual(expected, actual);
        }
      } finally {
        closeAll(p0);
        closeAll(p1);
        closeAll(finalized);
        closeAll(whole);
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
    }
  }

  @Test
  void testLeftTableClosedBeforeContextStillWorks() {
    HashJoin hashJoin;
    JoinMatchContext ctx;
    try (Table build = new Table.TestBuilder().column(1, 2, 3).build()) {
      hashJoin = new HashJoin(build, false);
    }
    try (Table probe = new Table.TestBuilder().column(1, 2).build()) {
      ctx = probe.innerJoinMatchContext(hashJoin);
    }
    try {
      assertEquals(2, ctx.getOutputSize());
    } finally {
      ctx.close();
      hashJoin.close();
    }
  }

  @Test
  void testFinalizePartitionedFullJoinArrayMismatch() {
    try (Table build = new Table.TestBuilder().column(1).build();
         HashJoin hashJoin = new HashJoin(build, false);
         Table probe = new Table.TestBuilder().column(1, 2).build();
         JoinMatchContext ctx = probe.fullJoinMatchContext(hashJoin)) {
      GatherMap[] p0 = ctx.partitionGatherMaps(0, 2);
      try {
        assertThrows(IllegalArgumentException.class, () ->
            JoinMatchContext.finalizePartitionedFullJoin(
                new GatherMap[]{p0[0]}, new GatherMap[]{}, 2, 1));
      } finally {
        closeAll(p0);
      }
    }
  }

  @Test
  void testFinalizePartitionedFullJoinRowCountOutOfRange() {
    GatherMap[] none = new GatherMap[0];
    long tooBig = Integer.MAX_VALUE + 1L;
    assertThrows(IllegalArgumentException.class, () ->
        JoinMatchContext.finalizePartitionedFullJoin(none, none, tooBig, 1));
    assertThrows(IllegalArgumentException.class, () ->
        JoinMatchContext.finalizePartitionedFullJoin(none, none, 1, tooBig));
    assertThrows(IllegalArgumentException.class, () ->
        JoinMatchContext.finalizePartitionedFullJoin(none, none, -1, 1));
  }

  // Convert gather-map pairs to a two-column index table sorted by (left, right).
  private static Table toSortedIndexTable(GatherMap[] maps) {
    int rows = (int) maps[0].getRowCount();
    try (ColumnView leftView = maps[0].toColumnView(0, rows);
         ColumnView rightView = maps[1].toColumnView(0, rows);
         ColumnVector leftCol = leftView.copyToColumnVector();
         ColumnVector rightCol = rightView.copyToColumnVector();
         Table t = new Table(leftCol, rightCol)) {
      return t.orderBy(OrderByArg.asc(0, true), OrderByArg.asc(1, true));
    }
  }

  // Concatenate per-partition gather maps into one sorted index table.
  private static Table toSortedIndexTable(List<GatherMap[]> partitions) {
    List<ColumnView> leftViews = new ArrayList<>();
    List<ColumnView> rightViews = new ArrayList<>();
    try {
      for (GatherMap[] p : partitions) {
        int rows = (int) p[0].getRowCount();
        leftViews.add(p[0].toColumnView(0, rows));
        rightViews.add(p[1].toColumnView(0, rows));
      }
      try (ColumnVector leftCat =
               ColumnVector.concatenate(leftViews.toArray(new ColumnView[0]));
           ColumnVector rightCat =
               ColumnVector.concatenate(rightViews.toArray(new ColumnView[0]));
           Table t = new Table(leftCat, rightCat)) {
        return t.orderBy(OrderByArg.asc(0, true), OrderByArg.asc(1, true));
      }
    } finally {
      for (ColumnView v : leftViews) v.close();
      for (ColumnView v : rightViews) v.close();
    }
  }

  private static void closeAll(GatherMap[] maps) {
    if (maps == null) return;
    for (GatherMap m : maps) m.close();
  }
}

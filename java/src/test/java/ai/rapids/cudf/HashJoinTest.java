/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.function.BiFunction;
import java.util.stream.Stream;

import static ai.rapids.cudf.AssertUtils.assertGatherMapsEqualUnordered;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class HashJoinTest {
  @Test
  void testGetNumberOfColumns() {
    try (Table t = new Table.TestBuilder().column(1, 2).column(3, 4).column(5, 6).build();
         HashJoin hashJoin = new HashJoin(t, false)) {
      assertEquals(3, hashJoin.getNumberOfColumns());
    }
  }

  @Test
  void testGetCompareNullsEqual() {
    try (Table t = new Table.TestBuilder().column(1, 2, 3, 4).build()) {
      try (HashJoin hashJoin = new HashJoin(t, false)) {
        assertFalse(hashJoin.getCompareNullsEqual());
        assertFalse(hashJoin.getCompareNulls());
      }
      try (HashJoin hashJoin = new HashJoin(t, true)) {
        assertTrue(hashJoin.getCompareNullsEqual());
        assertTrue(hashJoin.getCompareNulls());
      }
    }
  }

  private static Stream<Arguments> joinGatherMapsReuseCases() {
    final int inv = Integer.MIN_VALUE;
    return Stream.of(
        Arguments.of("left", (BiFunction<Table, HashJoin, GatherMap[]>) Table::leftJoinGatherMaps,
            new Integer[][]{{0, 0, 1, 2}, {1, 2, 3, inv}},
            new Integer[][]{{0, 1, 2}, {4, 0, inv}}),
        Arguments.of("inner", (BiFunction<Table, HashJoin, GatherMap[]>) Table::innerJoinGatherMaps,
            new Integer[][]{{0, 0, 1}, {1, 2, 3}},
            new Integer[][]{{0, 1}, {4, 0}}),
        Arguments.of("full", (BiFunction<Table, HashJoin, GatherMap[]>) Table::fullJoinGatherMaps,
            new Integer[][]{{inv, inv, 0, 0, 1, 2}, {0, 4, 1, 2, 3, inv}},
            new Integer[][]{{inv, inv, inv, 0, 1, 2}, {1, 2, 3, 4, 0, inv}}));
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("joinGatherMapsReuseCases")
  void testJoinGatherMapsCanBeReusedAcrossProbeTables(String joinType,
      BiFunction<Table, HashJoin, GatherMap[]> join, Integer[][] expected1Indices,
      Integer[][] expected2Indices) {
    try (ColumnVector buildKeys = ColumnVector.fromInts(0, 1, 1, 2, 3);
         Table buildTable = new Table(buildKeys);
         HashJoin hashJoin = new HashJoin(buildTable, true);
         ColumnVector probe1Keys = ColumnVector.fromInts(1, 2, 4);
         Table probe1Table = new Table(probe1Keys);
         ColumnVector probe2Keys = ColumnVector.fromInts(3, 0, 5);
         Table probe2Table = new Table(probe2Keys);
         Table expected1 = new Table.TestBuilder()
             .column(expected1Indices[0]).column(expected1Indices[1]).build();
         Table expected2 = new Table.TestBuilder()
             .column(expected2Indices[0]).column(expected2Indices[1]).build();
         CloseableArray<GatherMap> maps1 =
             CloseableArray.wrap(join.apply(probe1Table, hashJoin));
         CloseableArray<GatherMap> maps2 =
             CloseableArray.wrap(join.apply(probe2Table, hashJoin))) {
      assertGatherMapsEqualUnordered(expected1, maps1.getArray());
      assertGatherMapsEqualUnordered(expected2, maps2.getArray());
    }
  }

  @Test
  void testColumnCountMismatch() {
    try (Table build = new Table.TestBuilder().column(7, 9).build();
         HashJoin hashJoin = new HashJoin(build, false);
         Table probe = new Table.TestBuilder().column(7, 8).column(1, 2).build()) {
      assertThrows(IllegalArgumentException.class, () -> probe.leftJoinRowCount(hashJoin));
      assertThrows(IllegalArgumentException.class, () -> probe.leftJoinGatherMaps(hashJoin));
      assertThrows(IllegalArgumentException.class, () -> probe.leftJoinGatherMaps(hashJoin, 0));
      assertThrows(IllegalArgumentException.class, () -> probe.innerJoinRowCount(hashJoin));
      assertThrows(IllegalArgumentException.class, () -> probe.innerJoinGatherMaps(hashJoin));
      assertThrows(IllegalArgumentException.class, () -> probe.innerJoinGatherMaps(hashJoin, 0));
      assertThrows(IllegalArgumentException.class, () -> probe.fullJoinRowCount(hashJoin));
      assertThrows(IllegalArgumentException.class, () -> probe.fullJoinGatherMaps(hashJoin));
      assertThrows(IllegalArgumentException.class, () -> probe.fullJoinGatherMaps(hashJoin, 0));
    }
  }

  @Test
  void testClosedHashJoin() {
    try (Table build = new Table.TestBuilder().column(7, 9).build();
         Table probe = new Table.TestBuilder().column(7, 8).build()) {
      HashJoin hashJoin = new HashJoin(build, false);
      hashJoin.close();
      assertEquals(1, hashJoin.getNumberOfColumns());
      assertFalse(hashJoin.getCompareNullsEqual());
      assertThrows(IllegalStateException.class, () -> probe.leftJoinGatherMaps(hashJoin));
      assertThrows(IllegalStateException.class, () -> probe.innerJoinGatherMaps(hashJoin));
      assertThrows(IllegalStateException.class, () -> probe.fullJoinGatherMaps(hashJoin));
      assertThrows(IllegalStateException.class, hashJoin::close);
    }
  }
}

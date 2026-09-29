/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertGatherMapEquals;
import static ai.rapids.cudf.AssertUtils.assertGatherMapsEqualUnordered;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class DistinctHashJoinTest {
  @Test
  void testGetNumberOfColumns() {
    try (Table t = new Table.TestBuilder().column(1, 2).column(3, 4).column(5, 6).build();
         DistinctHashJoin hashJoin = new DistinctHashJoin(t, false)) {
      assertEquals(3, hashJoin.getNumberOfColumns());
    }
  }

  @Test
  void testGetCompareNullsEqual() {
    try (Table t = new Table.TestBuilder().column(1, 2, 3, 4).build()) {
      try (DistinctHashJoin hashJoin = new DistinctHashJoin(t, false)) {
        assertFalse(hashJoin.getCompareNullsEqual());
      }
      try (DistinctHashJoin hashJoin = new DistinctHashJoin(t, true)) {
        assertTrue(hashJoin.getCompareNullsEqual());
      }
    }
  }

  @Test
  void testLeftDistinctJoinGatherMapCanBeReusedAcrossProbeTables() {
    final int inv = Integer.MIN_VALUE;
    try (ColumnVector buildKeys = ColumnVector.fromInts(0, 1, 2, 3);
         Table buildTable = new Table(buildKeys);
         DistinctHashJoin hashJoin = new DistinctHashJoin(buildTable, true);
         ColumnVector probe1Keys = ColumnVector.fromInts(1, 2, 4);
         Table probe1Table = new Table(probe1Keys);
         ColumnVector probe2Keys = ColumnVector.fromInts(3, 0, 5);
         Table probe2Table = new Table(probe2Keys);
         ColumnVector expected1 = ColumnVector.fromInts(1, 2, inv);
         ColumnVector expected2 = ColumnVector.fromInts(3, 0, inv);
         GatherMap map1 = probe1Table.leftDistinctJoinGatherMap(hashJoin);
         GatherMap map2 = probe2Table.leftDistinctJoinGatherMap(hashJoin)) {
      assertGatherMapEquals(expected1, map1);
      assertGatherMapEquals(expected2, map2);
    }
  }

  @Test
  void testInnerDistinctJoinGatherMapsCanBeReusedAcrossProbeTables() {
    try (ColumnVector buildKeys = ColumnVector.fromInts(0, 1, 2, 3);
         Table buildTable = new Table(buildKeys);
         DistinctHashJoin hashJoin = new DistinctHashJoin(buildTable, true);
         ColumnVector probe1Keys = ColumnVector.fromInts(1, 2, 4);
         Table probe1Table = new Table(probe1Keys);
         ColumnVector probe2Keys = ColumnVector.fromInts(3, 0, 5);
         Table probe2Table = new Table(probe2Keys);
         Table expected1 = new Table.TestBuilder().column(0, 1).column(1, 2).build();
         Table expected2 = new Table.TestBuilder().column(0, 1).column(3, 0).build();
         CloseableArray<GatherMap> maps1 =
             CloseableArray.wrap(probe1Table.innerDistinctJoinGatherMaps(hashJoin));
         CloseableArray<GatherMap> maps2 =
             CloseableArray.wrap(probe2Table.innerDistinctJoinGatherMaps(hashJoin))) {
      assertGatherMapsEqualUnordered(expected1, maps1.getArray());
      assertGatherMapsEqualUnordered(expected2, maps2.getArray());
    }
  }

  @Test
  void testColumnCountMismatch() {
    try (Table build = new Table.TestBuilder().column(7, 9).build();
         DistinctHashJoin hashJoin = new DistinctHashJoin(build, false);
         Table probe = new Table.TestBuilder().column(7, 8).column(1, 2).build()) {
      assertThrows(IllegalArgumentException.class, () -> probe.leftDistinctJoinGatherMap(hashJoin));
      assertThrows(IllegalArgumentException.class, () -> probe.innerDistinctJoinGatherMaps(hashJoin));
    }
  }

  @Test
  void testClosedDistinctHashJoin() {
    try (Table build = new Table.TestBuilder().column(7, 9).build();
         Table probe = new Table.TestBuilder().column(7, 8).build()) {
      DistinctHashJoin hashJoin = new DistinctHashJoin(build, false);
      hashJoin.close();
      assertEquals(1, hashJoin.getNumberOfColumns());
      assertFalse(hashJoin.getCompareNullsEqual());
      assertThrows(IllegalStateException.class, () -> probe.leftDistinctJoinGatherMap(hashJoin));
      assertThrows(IllegalStateException.class, () -> probe.innerDistinctJoinGatherMaps(hashJoin));
      assertThrows(IllegalStateException.class, hashJoin::close);
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

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

  @Test
  void testLeftJoinGatherMapsCanBeReusedAcrossProbeTables() {
    try (ColumnVector buildKeys = ColumnVector.fromInts(0, 1, 1, 2, 3);
         Table buildTable = new Table(buildKeys);
         HashJoin hashJoin = new HashJoin(buildTable, true);
         ColumnVector probe1Keys = ColumnVector.fromInts(1, 2, 4);
         Table probe1Table = new Table(probe1Keys);
         ColumnVector probe2Keys = ColumnVector.fromInts(3, 0, 5);
         Table probe2Table = new Table(probe2Keys);
         Table expected1 = new Table.TestBuilder()
             .column(0, 0, 1, 2).column(1, 2, 3, Integer.MIN_VALUE).build();
         Table expected2 = new Table.TestBuilder()
             .column(0, 1, 2).column(4, 0, Integer.MIN_VALUE).build();
         CloseableArray<GatherMap> maps1 =
             CloseableArray.wrap(probe1Table.leftJoinGatherMaps(hashJoin));
         CloseableArray<GatherMap> maps2 =
             CloseableArray.wrap(probe2Table.leftJoinGatherMaps(hashJoin))) {
      assertGatherMapsEqualUnordered(expected1, maps1.getArray());
      assertGatherMapsEqualUnordered(expected2, maps2.getArray());
    }
  }

  @Test
  void testInnerJoinGatherMapsCanBeReusedAcrossProbeTables() {
    try (ColumnVector buildKeys = ColumnVector.fromInts(0, 1, 1, 2, 3);
         Table buildTable = new Table(buildKeys);
         HashJoin hashJoin = new HashJoin(buildTable, true);
         ColumnVector probe1Keys = ColumnVector.fromInts(1, 2, 4);
         Table probe1Table = new Table(probe1Keys);
         ColumnVector probe2Keys = ColumnVector.fromInts(3, 0, 5);
         Table probe2Table = new Table(probe2Keys);
         Table expected1 = new Table.TestBuilder().column(0, 0, 1).column(1, 2, 3).build();
         Table expected2 = new Table.TestBuilder().column(0, 1).column(4, 0).build();
         CloseableArray<GatherMap> maps1 =
             CloseableArray.wrap(probe1Table.innerJoinGatherMaps(hashJoin));
         CloseableArray<GatherMap> maps2 =
             CloseableArray.wrap(probe2Table.innerJoinGatherMaps(hashJoin))) {
      assertGatherMapsEqualUnordered(expected1, maps1.getArray());
      assertGatherMapsEqualUnordered(expected2, maps2.getArray());
    }
  }

  @Test
  void testFullJoinGatherMapsCanBeReusedAcrossProbeTables() {
    try (ColumnVector buildKeys = ColumnVector.fromInts(0, 1, 1, 2, 3);
         Table buildTable = new Table(buildKeys);
         HashJoin hashJoin = new HashJoin(buildTable, true);
         ColumnVector probe1Keys = ColumnVector.fromInts(1, 2, 4);
         Table probe1Table = new Table(probe1Keys);
         ColumnVector probe2Keys = ColumnVector.fromInts(3, 0, 5);
         Table probe2Table = new Table(probe2Keys);
         Table expected1 = new Table.TestBuilder()
             .column(Integer.MIN_VALUE, Integer.MIN_VALUE, 0, 0, 1, 2)
             .column(0, 4, 1, 2, 3, Integer.MIN_VALUE).build();
         Table expected2 = new Table.TestBuilder()
             .column(Integer.MIN_VALUE, Integer.MIN_VALUE, Integer.MIN_VALUE, 0, 1, 2)
             .column(1, 2, 3, 4, 0, Integer.MIN_VALUE).build();
         CloseableArray<GatherMap> maps1 =
             CloseableArray.wrap(probe1Table.fullJoinGatherMaps(hashJoin));
         CloseableArray<GatherMap> maps2 =
             CloseableArray.wrap(probe2Table.fullJoinGatherMaps(hashJoin))) {
      assertGatherMapsEqualUnordered(expected1, maps1.getArray());
      assertGatherMapsEqualUnordered(expected2, maps2.getArray());
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

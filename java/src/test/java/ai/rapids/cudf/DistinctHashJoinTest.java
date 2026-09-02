/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class DistinctHashJoinTest {
  @Test
  void testGetNumberOfColumns() {
    try (Table buildTable = new Table.TestBuilder()
             .column(1, 2)
             .column(3, 4)
             .column(5, 6)
             .build();
         DistinctHashJoin hashJoin = new DistinctHashJoin(buildTable, false)) {
      assertEquals(3, hashJoin.getNumberOfColumns());
    }
  }

  @Test
  void testGetCompareNullsEqual() {
    try (Table buildTable = new Table.TestBuilder().column(1, 2, 3, 4).build()) {
      try (DistinctHashJoin hashJoin = new DistinctHashJoin(buildTable, false)) {
        assertFalse(hashJoin.getCompareNullsEqual());
      }
      try (DistinctHashJoin hashJoin = new DistinctHashJoin(buildTable, true)) {
        assertTrue(hashJoin.getCompareNullsEqual());
      }
    }
  }

  @Test
  void testLeftJoinGatherMapCanBeReusedAcrossProbeTables() {
    final int inv = Integer.MIN_VALUE;
    try (ColumnVector buildKeys = ColumnVector.fromInts(0, 1, 2, 3);
         Table buildTable = new Table(buildKeys);
         DistinctHashJoin hashJoin = new DistinctHashJoin(buildTable, true);
         ColumnVector probe1Keys = ColumnVector.fromInts(1, 2, 4);
         Table probe1Table = new Table(probe1Keys);
         ColumnVector probe2Keys = ColumnVector.fromInts(3, 0, 5);
         Table probe2Table = new Table(probe2Keys);
         ColumnVector expected1 = ColumnVector.fromInts(1, 2, inv);
         ColumnVector expected2 = ColumnVector.fromInts(3, 0, inv)) {
      assertGatherMapEquals(expected1, probe1Table.leftDistinctJoinGatherMap(hashJoin));
      assertGatherMapEquals(expected2, probe2Table.leftDistinctJoinGatherMap(hashJoin));
    }
  }

  private static void assertGatherMapEquals(ColumnView expected, GatherMap gatherMap) {
    try (GatherMap map = gatherMap;
         ColumnView actual = map.toColumnView(0, (int) map.getRowCount())) {
      assertEquals(expected.getRowCount(), map.getRowCount());
      assertColumnsAreEqual(expected, actual);
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.util.AbstractMap;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;

public class DistinctHashJoinTest {
  private static final int GATHER_MAP_SENTINEL = Integer.MIN_VALUE;

  @SafeVarargs
  private static Set<Map.Entry<Integer, Integer>> pairSet(Map.Entry<Integer, Integer>... pairs) {
    return new HashSet<>(Arrays.asList(pairs));
  }

  private static Map.Entry<Integer, Integer> pair(Integer left, Integer right) {
    return new AbstractMap.SimpleEntry<>(left, right);
  }

  private static Set<Map.Entry<Integer, Integer>> gatherMapPairToSet(
      HostColumnVector leftMap, HostColumnVector rightMap) {
    Set<Map.Entry<Integer, Integer>> result = new HashSet<>();
    for (int i = 0; i < leftMap.getRowCount(); i++) {
      Integer leftVal = leftMap.getInt(i);
      Integer rightVal = rightMap.getInt(i);
      result.add(pair(leftVal == GATHER_MAP_SENTINEL ? null : leftVal,
          rightVal == GATHER_MAP_SENTINEL ? null : rightVal));
    }
    return result;
  }

  private static List<Integer> gatherMapToList(HostColumnVector gatherMap) {
    Integer[] result = new Integer[(int) gatherMap.getRowCount()];
    for (int i = 0; i < gatherMap.getRowCount(); i++) {
      int index = gatherMap.getInt(i);
      result[i] = index == GATHER_MAP_SENTINEL ? null : index;
    }
    return Arrays.asList(result);
  }

  @Test
  void testInnerJoinGatherMapsCanBeReusedAcrossProbeTables() {
    try (ColumnVector buildKeys = ColumnVector.fromInts(0, 1, 2, 3);
         Table buildTable = new Table(buildKeys);
         DistinctHashJoin hashJoin = new DistinctHashJoin(buildTable, true);
         ColumnVector probe1Keys = ColumnVector.fromInts(1, 2, 4);
         Table probe1Table = new Table(probe1Keys);
         ColumnVector probe2Keys = ColumnVector.fromInts(3, 0, 5);
         Table probe2Table = new Table(probe2Keys)) {
      assertGatherPairs(probe1Table.innerJoinGatherMaps(hashJoin),
          pairSet(pair(0, 1), pair(1, 2)));
      assertGatherPairs(probe2Table.innerJoinGatherMaps(hashJoin),
          pairSet(pair(0, 3), pair(1, 0)));
    }
  }

  @Test
  void testLeftJoinGatherMapCanBeReusedAcrossProbeTables() {
    try (ColumnVector buildKeys = ColumnVector.fromInts(0, 1, 2, 3);
         Table buildTable = new Table(buildKeys);
         DistinctHashJoin hashJoin = new DistinctHashJoin(buildTable, true);
         ColumnVector probe1Keys = ColumnVector.fromInts(1, 4, 0);
         Table probe1Table = new Table(probe1Keys);
         ColumnVector probe2Keys = ColumnVector.fromBoxedInts(null, 2, 8);
         Table probe2Table = new Table(probe2Keys)) {
      assertGatherIndices(probe1Table.leftDistinctJoinGatherMap(hashJoin),
          Arrays.asList(1, null, 0));
      assertGatherIndices(probe2Table.leftDistinctJoinGatherMap(hashJoin),
          Arrays.asList(null, 2, null));
    }
  }

  @Test
  void testInnerJoinRespectsNullEquality() {
    try (ColumnVector buildKeys = ColumnVector.fromBoxedInts(null, 1, 2);
         Table buildTable = new Table(buildKeys);
         DistinctHashJoin hashJoin = new DistinctHashJoin(buildTable, false);
         ColumnVector probeKeys = ColumnVector.fromBoxedInts(null, 2);
         Table probeTable = new Table(probeKeys)) {
      assertGatherPairs(probeTable.innerJoinGatherMaps(hashJoin), pairSet(pair(1, 2)));
    }
  }

  private static void assertGatherPairs(GatherMap[] gatherMaps,
      Set<Map.Entry<Integer, Integer>> expected) {
    try (GatherMap leftMap = gatherMaps[0];
         GatherMap rightMap = gatherMaps[1];
         HostColumnVector leftHost =
             leftMap.toColumnView(0, (int) leftMap.getRowCount()).copyToHost();
         HostColumnVector rightHost =
             rightMap.toColumnView(0, (int) rightMap.getRowCount()).copyToHost()) {
      assertEquals(expected.size(), leftMap.getRowCount());
      assertEquals(expected.size(), rightMap.getRowCount());
      assertEquals(expected, gatherMapPairToSet(leftHost, rightHost));
    }
  }

  private static void assertGatherIndices(GatherMap gatherMap, List<Integer> expected) {
    try (GatherMap map = gatherMap;
         HostColumnVector host = map.toColumnView(0, (int) map.getRowCount()).copyToHost()) {
      assertEquals(expected.size(), map.getRowCount());
      assertEquals(expected, gatherMapToList(host));
    }
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static ai.rapids.cudf.AssertUtils.assertGatherMapEqualsUnordered;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class FilteredJoinTest {
  @Test
  void testGetNumberOfColumns() {
    try (Table t = new Table.TestBuilder().column(1, 2).column(3, 4).column(5, 6).build();
         FilteredJoin filter = new FilteredJoin(t, false)) {
      assertEquals(3, filter.getNumberOfColumns());
    }
  }

  @Test
  void testGetCompareNullsEqual() {
    try (Table t = new Table.TestBuilder().column(1, 2, 3, 4).build()) {
      try (FilteredJoin filter = new FilteredJoin(t, false)) {
        assertFalse(filter.getCompareNullsEqual());
      }
      try (FilteredJoin filter = new FilteredJoin(t, true)) {
        assertTrue(filter.getCompareNullsEqual());
      }
    }
  }

  @Test
  void testLeftSemiJoinGatherMapCanBeReusedAcrossProbeTables() {
    try (Table build = new Table.TestBuilder().column(7, 7, 7, 9).build();
         FilteredJoin filter = new FilteredJoin(build, false);
         Table first = new Table.TestBuilder().column(7, 7, 8, 9).build();
         Table second = new Table.TestBuilder().column(7, 10).build();
         GatherMap map1 = first.leftSemiJoinGatherMap(filter);
         GatherMap map2 = second.leftSemiJoinGatherMap(filter)) {
      assertGatherMapEqualsUnordered(map1, 0, 1, 3);
      assertGatherMapEqualsUnordered(map2, 0);
    }
  }

  @Test
  void testLeftAntiJoinGatherMapCanBeReusedAcrossProbeTables() {
    try (Table build = new Table.TestBuilder().column(7, 7, 7, 9).build();
         FilteredJoin filter = new FilteredJoin(build, false);
         Table first = new Table.TestBuilder().column(7, 7, 8, 9).build();
         Table second = new Table.TestBuilder().column(7, 10).build();
         GatherMap map1 = first.leftAntiJoinGatherMap(filter);
         GatherMap map2 = second.leftAntiJoinGatherMap(filter)) {
      assertGatherMapEqualsUnordered(map1, 2);
      assertGatherMapEqualsUnordered(map2, 1);
    }
  }

  @Test
  void testRetainsBuildKeysAfterOriginalTableCloses() {
    final FilteredJoin filter;
    try (Table build = new Table.TestBuilder().column("a", "a", "b").build()) {
      filter = new FilteredJoin(build, false);
    }
    try (FilteredJoin toClose = filter;
         Table probe = new Table.TestBuilder().column("b", "c", "a").build();
         GatherMap semi = probe.leftSemiJoinGatherMap(filter);
         GatherMap anti = probe.leftAntiJoinGatherMap(filter)) {
      assertGatherMapEqualsUnordered(semi, 0, 2);
      assertGatherMapEqualsUnordered(anti, 1);
    }
  }

  @Test
  void testClosingFilterPreservesOriginalBuildTable() {
    try (Table build = new Table.TestBuilder().column(7, 7, 9).build();
         Table probe = new Table.TestBuilder().column(7, 8).build()) {
      try (FilteredJoin filter = new FilteredJoin(build, false);
           GatherMap map = probe.leftSemiJoinGatherMap(filter)) {
        assertGatherMapEqualsUnordered(map, 0);
      }
      try (GatherMap map = probe.leftSemiJoinGatherMap(build, false)) {
        assertGatherMapEqualsUnordered(map, 0);
      }
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"false", "true"})
  void testCompositeKeysWithNulls(boolean compareNullsEqual) {
    try (Table build = new Table.TestBuilder()
        .column(7, 7, 7, null, null).column("a", "a", null, "b", "b").build();
         FilteredJoin filter = new FilteredJoin(build, compareNullsEqual);
         Table probe = new Table.TestBuilder()
             .column(7, 7, null, 7, null).column("a", null, "b", "b", null).build();
         GatherMap semi = probe.leftSemiJoinGatherMap(filter);
         GatherMap anti = probe.leftAntiJoinGatherMap(filter)) {
      if (compareNullsEqual) {
        assertGatherMapEqualsUnordered(semi, 0, 1, 2);
        assertGatherMapEqualsUnordered(anti, 3, 4);
      } else {
        assertGatherMapEqualsUnordered(semi, 0);
        assertGatherMapEqualsUnordered(anti, 1, 2, 3, 4);
      }
    }
  }

  @ParameterizedTest
  @ValueSource(strings = {"false", "true"})
  void testEmptyInputs(boolean emptyBuild) {
    Integer[] buildRows = emptyBuild ? new Integer[0] : new Integer[]{7, 7};
    try (Table build = new Table.TestBuilder().column(buildRows).build();
         FilteredJoin filter = new FilteredJoin(build, false);
         Table empty = new Table.TestBuilder().column(new Integer[0]).build();
         Table probe = new Table.TestBuilder().column(7, 7, null).build();
         GatherMap emptySemi = empty.leftSemiJoinGatherMap(filter);
         GatherMap emptyAnti = empty.leftAntiJoinGatherMap(filter);
         GatherMap semi = probe.leftSemiJoinGatherMap(filter);
         GatherMap anti = probe.leftAntiJoinGatherMap(filter)) {
      assertGatherMapEqualsUnordered(emptySemi);
      assertGatherMapEqualsUnordered(emptyAnti);
      // Empty probes must leave the lookup usable for later nonempty batches.
      assertGatherMapEqualsUnordered(semi, emptyBuild ? new int[0] : new int[]{0, 1});
      assertGatherMapEqualsUnordered(anti, emptyBuild ? new int[]{0, 1, 2} : new int[]{2});
    }
  }

  @Test
  void testColumnCountMismatch() {
    try (Table build = new Table.TestBuilder().column(7, 9).build();
         FilteredJoin filter = new FilteredJoin(build, false);
         Table probe = new Table.TestBuilder().column(7, 8).column(1, 2).build()) {
      assertThrows(IllegalArgumentException.class, () -> probe.leftSemiJoinGatherMap(filter));
      assertThrows(IllegalArgumentException.class, () -> probe.leftAntiJoinGatherMap(filter));
    }
  }

  @Test
  void testClosedFilter() {
    try (Table build = new Table.TestBuilder().column(7, 9).build();
         Table probe = new Table.TestBuilder().column(7, 8).build()) {
      FilteredJoin filter = new FilteredJoin(build, false);
      filter.close();
      assertEquals(1, filter.getNumberOfColumns());
      assertFalse(filter.getCompareNullsEqual());
      assertThrows(IllegalStateException.class, () -> probe.leftSemiJoinGatherMap(filter));
      assertThrows(IllegalStateException.class, () -> probe.leftAntiJoinGatherMap(filter));
      assertThrows(IllegalStateException.class, filter::close);
    }
  }
}

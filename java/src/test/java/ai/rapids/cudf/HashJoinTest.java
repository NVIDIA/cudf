/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
    try (Table t = new Table.TestBuilder().column(1, 2, 3, 4).column(5, 6, 7, 8).build()) {
      try (HashJoin hashJoin = new HashJoin(t, false)) {
        assertFalse(hashJoin.getCompareNullsEqual());
      }
      try (HashJoin hashJoin = new HashJoin(t, true)) {
        assertTrue(hashJoin.getCompareNullsEqual());
      }
    }
  }
}

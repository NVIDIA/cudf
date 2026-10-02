/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Basic tests to validate JNI plumbing for JoinFactorizer.
 * Comprehensive algorithm testing is done in C++ tests.
 */
public class JoinFactorizerTest extends CudfTestBase {

  @Test
  void testBasicIntegerKeys() {
    // Right table with some duplicate keys: [1, 2, 3, 2, 1]
    try (Table rightTable = new Table.TestBuilder()
            .column(1, 2, 3, 2, 1)
            .build();
         JoinFactorizer remap = new JoinFactorizer(rightTable)) {

      // Verify counts
      assertEquals(3, remap.getDistinctCount());
      assertEquals(2, remap.getMaxMultiplicity());

      // Verify we can remap and get non-negative IDs
      try (ColumnVector result = remap.factorizeRightKeys();
           HostColumnVector hostResult = result.copyToHost()) {
        assertEquals(5, hostResult.getRowCount());
        for (int i = 0; i < hostResult.getRowCount(); i++) {
          assertTrue(hostResult.getInt(i) >= 0, "Right IDs should be non-negative");
        }
      }
    }
  }

  @Test
  void testStringKeys() {
    try (Table rightTable = new Table.TestBuilder()
            .column("apple", "banana", null, "cherry", "banana", null)
            .build();
         JoinFactorizer remap = new JoinFactorizer(rightTable)) {

      // Distinct keys: "apple", "banana", null, "cherry" = 4 (with default EQUAL null handling)
      assertEquals(4, remap.getDistinctCount());
      // "banana" appears twice, null appears twice = max duplicate count 2
      assertEquals(2, remap.getMaxMultiplicity());

      try (ColumnVector result = remap.factorizeRightKeys();
           HostColumnVector hostResult = result.copyToHost()) {
        assertEquals(6, hostResult.getRowCount());
      }
    }
  }

  @Test
  void testLeftNotFound() {
    try (Table rightTable = new Table.TestBuilder()
            .column(1, 2, 3)
            .build();
         JoinFactorizer remap = new JoinFactorizer(rightTable);
         Table leftTable = new Table.TestBuilder()
            .column(4, 5)  // Keys not in right
            .build();
         ColumnVector result = remap.factorizeLeftKeys(leftTable);
         HostColumnVector hostResult = result.copyToHost()) {

      // All left keys should be NOT_FOUND
      for (int i = 0; i < hostResult.getRowCount(); i++) {
        assertEquals(JoinFactorizer.NOT_FOUND_SENTINEL, hostResult.getInt(i));
      }
    }
  }

  @Test
  void testNullEqualityEqual() {
    // With EQUAL, nulls are treated as equal and get valid IDs
    try (ColumnVector col = ColumnVector.fromBoxedInts(1, 2, null, 2);
         Table rightTable = new Table(col);
         JoinFactorizer remap = new JoinFactorizer(rightTable, NullEquality.EQUAL)) {

      // Distinct: 1, 2, null = 3
      assertEquals(3, remap.getDistinctCount());
      assertEquals(NullEquality.EQUAL, remap.getNullEquality());

      try (ColumnVector result = remap.factorizeRightKeys();
           HostColumnVector hostResult = result.copyToHost()) {
        // All IDs should be non-negative (including null row)
        for (int i = 0; i < hostResult.getRowCount(); i++) {
          assertTrue(hostResult.getInt(i) >= 0);
        }
      }
    }
  }

  @Test
  void testNullEqualityUnequal() {
    // With UNEQUAL, null rows get RIGHT_NULL_SENTINEL
    try (ColumnVector col = ColumnVector.fromBoxedInts(1, 2, null, 2);
         Table rightTable = new Table(col);
         JoinFactorizer remap = new JoinFactorizer(rightTable, NullEquality.UNEQUAL)) {

      // Distinct: 1, 2 = 2 (null is skipped)
      assertEquals(2, remap.getDistinctCount());
      assertEquals(NullEquality.UNEQUAL, remap.getNullEquality());

      try (ColumnVector result = remap.factorizeRightKeys();
           HostColumnVector hostResult = result.copyToHost()) {
        // Null row (index 2) should have RIGHT_NULL_SENTINEL
        assertEquals(JoinFactorizer.RIGHT_NULL_SENTINEL, hostResult.getInt(2));
        // Non-null rows should have non-negative IDs
        assertTrue(hostResult.getInt(0) >= 0);
        assertTrue(hostResult.getInt(1) >= 0);
        assertTrue(hostResult.getInt(3) >= 0);
      }
    }
  }

  @Test
  void testEmptyTable() {
    try (Table rightTable = new Table.TestBuilder()
            .column(new Integer[0])
            .build();
         JoinFactorizer remap = new JoinFactorizer(rightTable)) {

      assertEquals(0, remap.getDistinctCount());
      assertEquals(0, remap.getMaxMultiplicity());
    }
  }

  @Test
  void testReleaseRightKeys() {
    Table rightTable = new Table.TestBuilder()
            .column(1, 2, 3)
            .build();
    try (JoinFactorizer remap = new JoinFactorizer(rightTable)) {
      assertFalse(remap.isRightKeysReleased());

      // Release the right keys
      Table released = remap.releaseRightKeys();
      assertTrue(remap.isRightKeysReleased());

      // Can still use remap
      assertEquals(3, remap.getDistinctCount());

      // Must close the released table ourselves
      released.close();
    }
    // Original rightTable is still ours to close
    rightTable.close();
  }
}

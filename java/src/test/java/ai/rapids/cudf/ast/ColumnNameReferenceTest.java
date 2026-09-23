/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import ai.rapids.cudf.CudfTestBase;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;

/**
 * Unit tests for {@link ColumnNameReference}.
 */
public class ColumnNameReferenceTest extends CudfTestBase {

  // --------------------------------------------------------------------
  // Tests: ColumnNameReference (constructor)
  // --------------------------------------------------------------------

  /** Verifies that passing null to the constructor throws IllegalArgumentException. */
  @Test
  void testNullNameRejected() {
    assertThrows(IllegalArgumentException.class, () -> new ColumnNameReference(null));
  }

  /** Verifies that passing an empty string to the constructor throws IllegalArgumentException. */
  @Test
  void testEmptyNameRejected() {
    assertThrows(IllegalArgumentException.class, () -> new ColumnNameReference(""));
  }

  // --------------------------------------------------------------------
  // Tests: getColumnName()
  // --------------------------------------------------------------------

  /** Verifies that getColumnName() returns the name supplied to the constructor. */
  @Test
  void testGetColumnName() {
    ColumnNameReference colNameRef = new ColumnNameReference("my_col");
    assertEquals("my_col", colNameRef.getColumnName());
  }

  // --------------------------------------------------------------------
  // Tests: toString()
  // --------------------------------------------------------------------

  /** Verifies that toString() produces the expected COLUMN("name") representation. */
  @Test
  void testToString() {
    assertEquals("COLUMN(\"zip\")", new ColumnNameReference("zip").toString());
  }

  // --------------------------------------------------------------------
  // Tests: getSerializedSize()
  // --------------------------------------------------------------------

  /**
   * Verifies that getSerializedSize() equals 1 (node-type byte) + 4 (name-length int) +
   * the number of UTF-8 bytes in the name.
   */
  @Test
  void testGetSerializedSizeMatchesExpectedLayout() {
    // "zip" = 3 UTF-8 bytes → 1 + 4 + 3 = 8
    assertEquals(8, new ColumnNameReference("zip").getSerializedSize());
  }

  // --------------------------------------------------------------------
  // Tests: serialize()
  // --------------------------------------------------------------------

  /**
   * Verifies that serialize() writes the COLUMN_NAME_REFERENCE node-type byte (5), the
   * name length as a 4-byte int, and the UTF-8 name bytes, in that order.
   */
  @Test
  void testSerializeWritesNodeTypeIdNameLengthAndNameBytes() {
    ColumnNameReference colNameRef = new ColumnNameReference("zip");
    ByteBuffer buffer = ByteBuffer.allocate(colNameRef.getSerializedSize())
                                  .order(ByteOrder.nativeOrder());
    colNameRef.serialize(buffer);
    buffer.flip();
    assertEquals(5, buffer.get());       // COLUMN_NAME_REFERENCE nativeId
    assertEquals(3, buffer.getInt());    // "zip" is 3 UTF-8 bytes
    byte[] nameBytes = "zip".getBytes(StandardCharsets.UTF_8);
    for (byte b : nameBytes) {
      assertEquals(b, buffer.get());
    }
  }
}

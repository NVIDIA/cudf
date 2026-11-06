/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.ListType;
import ai.rapids.cudf.HostColumnVector.StructType;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.*;

public class ScalarTest extends CudfTestBase {
  @Test
  public void testDoubleClose() {
    Scalar s = Scalar.fromNull(DType.INT32);
    s.close();
    assertThrows(IllegalStateException.class, s::close);
  }

  @Test
  public void testIncRefAndDoubleFree() {
    Scalar s = Scalar.fromNull(DType.INT32);
    try (Scalar ignored1 = s) {
      try (Scalar ignored2 = s.incRefCount()) {
        try (Scalar ignored3 = s.incRefCount()) {
        }
      }
    }
    assertThrows(IllegalStateException.class, s::close);
  }

  @Test
  public void testNull() {
    for (DType.DTypeEnum dataType : DType.DTypeEnum.values()) {
      if (dataType == DType.DTypeEnum.DICTIONARY32) {
        continue; // there are no dictionary scalars, nor lists or structs of them built on the host
      }
      DType type;
      if (dataType.isDecimalType()) {
        type = DType.create(dataType, -3);
      } else {
        type = DType.create(dataType);
      }
      if (!type.isNestedType()) {
        try (Scalar s = Scalar.fromNull(type)) {
          assertEquals(type, s.getType());
          assertFalse(s.isValid(), "null validity for " + type);
        }
      }

      // create elementType for nested types
      HostColumnVector.DataType hDataType;
      if (DType.EMPTY.equals(type)) {
        continue;
      } else if (DType.LIST.equals(type)) {
        // list of list of int32
        hDataType = new ListType(true, new BasicType(true, DType.INT32));
      } else if (DType.STRUCT.equals(type)) {
        // list of struct of int32
        hDataType = new StructType(true, new BasicType(true, DType.INT32));
      } else {
        // list of non nested type
        hDataType = new BasicType(true, type);
      }

      // test list scalar with elementType(`type`)
      try (Scalar s = Scalar.listFromNull(hDataType); ColumnView listCv = s.getListAsColumnView()) {
        assertFalse(s.isValid(), "null validity for " + type);
        assertEquals(DType.LIST, s.getType());
        assertEquals(type, listCv.getType());
        assertEquals(0L, listCv.getRowCount());
        assertEquals(0L, listCv.getNullCount());
        if (type.isNestedType()) {
          try (ColumnView child = listCv.getChildColumnView(0)) {
            assertEquals(DType.INT32, child.getType());
            assertEquals(0L, child.getRowCount());
            assertEquals(0L, child.getNullCount());
          }
        }
      }

      // test struct scalar with elementType(`type`)
      try (Scalar s = Scalar.structFromNull(hDataType, hDataType, hDataType)) {
        assertFalse(s.isValid(), "null validity for " + type);
        assertEquals(DType.STRUCT, s.getType());

        ColumnView[] children = s.getChildrenFromStructScalar();
        try {
          for (ColumnView child : children) {
            assertEquals(hDataType.getType(), child.getType());
            assertEquals(1L, child.getRowCount());
            assertEquals(1L, child.getNullCount());
          }
        } finally {
          for (ColumnView child : children) child.close();
        }
      }
    }
  }

  @Test
  public void testBool() {
    try (Scalar s = Scalar.fromBool(false)) {
      assertEquals(DType.BOOL8, s.getType());
      assertTrue(s.isValid());
      assertFalse(s.getBoolean());
    }
  }

  @Test
  public void testByte() {
    try (Scalar s = Scalar.fromByte((byte) 1)) {
      assertEquals(DType.INT8, s.getType());
      assertTrue(s.isValid());
      assertEquals(1, s.getByte());
    }
  }

  @Test
  public void testShort() {
    try (Scalar s = Scalar.fromShort((short) 2)) {
      assertEquals(DType.INT16, s.getType());
      assertTrue(s.isValid());
      assertEquals(2, s.getShort());
    }
  }

  @Test
  public void testInt() {
    try (Scalar s = Scalar.fromInt(3)) {
      assertEquals(DType.INT32, s.getType());
      assertTrue(s.isValid());
      assertEquals(3, s.getInt());
    }
  }

  @Test
  public void testLong() {
    try (Scalar s = Scalar.fromLong(4)) {
      assertEquals(DType.INT64, s.getType());
      assertTrue(s.isValid());
      assertEquals(4L, s.getLong());
    }
  }

  @Test
  public void testFloat() {
    try (Scalar s = Scalar.fromFloat(5.1f)) {
      assertEquals(DType.FLOAT32, s.getType());
      assertTrue(s.isValid());
      assertEquals(5.1f, s.getFloat());
    }
  }

  @Test
  public void testDouble() {
    try (Scalar s = Scalar.fromDouble(6.2)) {
      assertEquals(DType.FLOAT64, s.getType());
      assertTrue(s.isValid());
      assertEquals(6.2, s.getDouble());
    }
  }

  @Test
  public void testDecimal() {
    BigDecimal[] bigDecimals = new BigDecimal[] {
        BigDecimal.valueOf(1234, 0),
        BigDecimal.valueOf(12345678, 2),
        BigDecimal.valueOf(1234567890123L, 6),
        new BigDecimal(new BigInteger("12312341234123412341234123412341234120"), 4)
    };
    for (BigDecimal dec : bigDecimals) {
      try (Scalar s = Scalar.fromDecimal(dec)) {
        DType dtype = DType.fromJavaBigDecimal(dec);
        assertEquals(dtype, s.getType());
        assertTrue(s.isValid());
        if (dtype.getTypeId() == DType.DTypeEnum.DECIMAL64) {
          assertEquals(dec.unscaledValue().longValueExact(), s.getLong());
        } else if (dtype.getTypeId() == DType.DTypeEnum.DECIMAL32) {
          assertEquals(dec.unscaledValue().intValueExact(), s.getInt());
        } else if (dtype.getTypeId() == DType.DTypeEnum.DECIMAL128) {
          assertEquals(dec.unscaledValue(), s.getBigDecimal().unscaledValue());
        }
        assertEquals(dec, s.getBigDecimal());
      }

      try (Scalar s = Scalar.fromDecimal(-dec.scale(), dec.unscaledValue().intValueExact())) {
        assertEquals(dec, s.getBigDecimal());
      } catch (java.lang.ArithmeticException ex) {
        try (Scalar s = Scalar.fromDecimal(-dec.scale(), dec.unscaledValue().longValueExact())) {
          assertEquals(dec, s.getBigDecimal());
          assertTrue(s.getType().isBackedByLong());
        } catch (java.lang.ArithmeticException e) {
          try (Scalar s = Scalar.fromDecimal(-dec.scale(), dec.unscaledValue())) {
            assertEquals(dec, s.getBigDecimal());
          }
        }
      }
    }
  }

  @Test
  public void testTimestampDays() {
    try (Scalar s = Scalar.timestampDaysFromInt(7)) {
      assertEquals(DType.TIMESTAMP_DAYS, s.getType());
      assertTrue(s.isValid());
      assertEquals(7, s.getInt());
    }
  }

  @Test
  public void testTimestampSeconds() {
    try (Scalar s = Scalar.timestampFromLong(DType.TIMESTAMP_SECONDS, 8)) {
      assertEquals(DType.TIMESTAMP_SECONDS, s.getType());
      assertTrue(s.isValid());
      assertEquals(8L, s.getLong());
    }
  }

  @Test
  public void testTimestampMilliseconds() {
    try (Scalar s = Scalar.timestampFromLong(DType.TIMESTAMP_MILLISECONDS, 9)) {
      assertEquals(DType.TIMESTAMP_MILLISECONDS, s.getType());
      assertTrue(s.isValid());
      assertEquals(9L, s.getLong());
    }
  }

  @Test
  public void testTimestampMicroseconds() {
    try (Scalar s = Scalar.timestampFromLong(DType.TIMESTAMP_MICROSECONDS, 10)) {
      assertEquals(DType.TIMESTAMP_MICROSECONDS, s.getType());
      assertTrue(s.isValid());
      assertEquals(10L, s.getLong());
    }
  }

  @Test
  public void testTimestampNanoseconds() {
    try (Scalar s = Scalar.timestampFromLong(DType.TIMESTAMP_NANOSECONDS, 11)) {
      assertEquals(DType.TIMESTAMP_NANOSECONDS, s.getType());
      assertTrue(s.isValid());
      assertEquals(11L, s.getLong());
    }
  }

  @Test
  public void testString() {
    try (Scalar s = Scalar.fromString("TEST")) {
      assertEquals(DType.STRING, s.getType());
      assertTrue(s.isValid());
      assertEquals("TEST", s.getJavaString());
      assertArrayEquals(new byte[] {'T', 'E', 'S', 'T'}, s.getUTF8());
    }
  }

  @Test
  public void testUTF8String() {
    try (Scalar s = Scalar.fromUTF8String("TEST".getBytes(StandardCharsets.UTF_8))) {
      assertEquals(DType.STRING, s.getType());
      assertTrue(s.isValid());
      assertEquals("TEST", s.getJavaString());
      assertArrayEquals(new byte[] {'T', 'E', 'S', 'T'}, s.getUTF8());
    }
    try (Scalar s = Scalar.fromUTF8String("".getBytes(StandardCharsets.UTF_8))) {
      assertEquals(DType.STRING, s.getType());
      assertTrue(s.isValid());
      assertEquals("", s.getJavaString());
      assertArrayEquals(new byte[] {}, s.getUTF8());
    }
  }

  @Test
  public void testList() {
    // list of int
    try (ColumnVector listInt = ColumnVector.fromInts(1, 2, 3, 4);
         Scalar s = Scalar.listFromColumnView(listInt)) {
      assertEquals(DType.LIST, s.getType());
      assertTrue(s.isValid());
      try (ColumnView v = s.getListAsColumnView()) {
        assertColumnsAreEqual(listInt, v);
      }
    }

    // list of list
    HostColumnVector.DataType listDT =
        new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true, DType.INT32));
    try (ColumnVector listList =
             ColumnVector.fromLists(listDT, Arrays.asList(1, 2, 3), Arrays.asList(4, 5, 6));
         Scalar s = Scalar.listFromColumnView(listList)) {
      assertEquals(DType.LIST, s.getType());
      assertTrue(s.isValid());
      try (ColumnView v = s.getListAsColumnView()) {
        assertColumnsAreEqual(listList, v);
      }
    }
  }

  @Test
  public void testStruct() {
    try (ColumnVector col0 = ColumnVector.fromInts(1);
         ColumnVector col1 = ColumnVector.fromBoxedDoubles(1.2);
         ColumnVector col2 = ColumnVector.fromStrings("a");
         ColumnVector col3 = ColumnVector.fromDecimals(BigDecimal.TEN);
         ColumnVector col4 = ColumnVector.daysFromInts(10);
         ColumnVector col5 = ColumnVector.durationSecondsFromLongs(12345L);
         Scalar s = Scalar.structFromColumnViews(col0, col1, col2, col3, col4, col5, col0, col1)) {
      assertEquals(DType.STRUCT, s.getType());
      assertTrue(s.isValid());
      ColumnView[] children = s.getChildrenFromStructScalar();
      try {
        assertColumnsAreEqual(col0, children[0]);
        assertColumnsAreEqual(col1, children[1]);
        assertColumnsAreEqual(col2, children[2]);
        assertColumnsAreEqual(col3, children[3]);
        assertColumnsAreEqual(col4, children[4]);
        assertColumnsAreEqual(col5, children[5]);
        assertColumnsAreEqual(col0, children[6]);
        assertColumnsAreEqual(col1, children[7]);
      } finally {
        for (ColumnView child : children) child.close();
      }
    }

    // test Struct Scalar with null members
    try (ColumnVector col0 = ColumnVector.fromInts(1);
         ColumnVector col1 = ColumnVector.fromBoxedDoubles((Double) null);
         ColumnVector col2 = ColumnVector.fromStrings((String) null);
         Scalar s1 = Scalar.structFromColumnViews(col0, col1, col2);
         Scalar s2 = Scalar.structFromColumnViews(col1, col2)) {
      ColumnView[] children = s1.getChildrenFromStructScalar();
      try {
        assertColumnsAreEqual(col0, children[0]);
        assertColumnsAreEqual(col1, children[1]);
        assertColumnsAreEqual(col2, children[2]);
      } finally {
        for (ColumnView child : children) child.close();
      }

      ColumnView[] children2 = s2.getChildrenFromStructScalar();
      try {
        assertColumnsAreEqual(col1, children2[0]);
        assertColumnsAreEqual(col2, children2[1]);
      } finally {
        for (ColumnView child : children2) child.close();
      }
    }

    // test Struct Scalar with single column
    try (ColumnVector col0 = ColumnVector.fromInts(1234);
         Scalar s = Scalar.structFromColumnViews(col0)) {
      ColumnView[] children = s.getChildrenFromStructScalar();
      try {
        assertColumnsAreEqual(col0, children[0]);
      } finally {
        children[0].close();
      }
    }

    // test Struct Scalar without column
    try (Scalar s = Scalar.structFromColumnViews()) {
      assertEquals(DType.STRUCT, s.getType());
      assertTrue(s.isValid());
      ColumnView[] children = s.getChildrenFromStructScalar();
      assertEquals(0, children.length);
    }

    // test Struct Scalar with nested types
    HostColumnVector.DataType listType =
        new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true, DType.INT32));
    HostColumnVector.DataType structType =
        new HostColumnVector.StructType(true, new HostColumnVector.BasicType(true, DType.INT32),
            new HostColumnVector.BasicType(true, DType.INT64));
    HostColumnVector.DataType nestedStructType = new HostColumnVector.StructType(
        true, new HostColumnVector.BasicType(true, DType.STRING), listType, structType);
    try (ColumnVector strCol = ColumnVector.fromStrings("AAAAAA");
         ColumnVector listCol = ColumnVector.fromLists(listType, Arrays.asList(1, 2, 3, 4, 5));
         ColumnVector structCol =
             ColumnVector.fromStructs(structType, new HostColumnVector.StructData(1, -1L));
         ColumnVector nestedStructCol = ColumnVector.fromStructs(nestedStructType,
             new HostColumnVector.StructData(
                 null, Arrays.asList(1, 2, null), new HostColumnVector.StructData(null, 10L)));
         Scalar s = Scalar.structFromColumnViews(strCol, listCol, structCol, nestedStructCol)) {
      assertEquals(DType.STRUCT, s.getType());
      assertTrue(s.isValid());
      ColumnView[] children = s.getChildrenFromStructScalar();
      try {
        assertColumnsAreEqual(strCol, children[0]);
        assertColumnsAreEqual(listCol, children[1]);
        assertColumnsAreEqual(structCol, children[2]);
        assertColumnsAreEqual(nestedStructCol, children[3]);
      } finally {
        for (ColumnView child : children) child.close();
      }
    }
  }

  @Test
  public void testRepeatString() {
    // Invalid scalar.
    try (Scalar nullString = Scalar.fromString(null);
         Scalar result = nullString.repeatString(5)) {
      assertFalse(result.isValid());
    }

    // Empty string.
    try (Scalar emptyString = Scalar.fromString("");
         Scalar result = emptyString.repeatString(5)) {
      assertTrue(result.isValid());
      assertEquals("", result.getJavaString());
    }

    // Negative repeatTimes.
    try (Scalar s = Scalar.fromString("Hello World");
         Scalar result = s.repeatString(-100)) {
      assertTrue(result.isValid());
      assertEquals("", result.getJavaString());
    }

    // Zero repeatTimes.
    try (Scalar s = Scalar.fromString("Hello World");
         Scalar result = s.repeatString(0)) {
      assertTrue(result.isValid());
      assertEquals("", result.getJavaString());
    }

    // Trivial input, output is copied exactly from input.
    try (Scalar s = Scalar.fromString("Hello World");
         Scalar result = s.repeatString(1)) {
      assertTrue(result.isValid());
      assertEquals(s.getJavaString(), result.getJavaString());
    }

    // Trivial input.
    try (Scalar s = Scalar.fromString("abcxyz-");
         Scalar result = s.repeatString(3)) {
      assertTrue(result.isValid());
      assertEquals("abcxyz-abcxyz-abcxyz-", result.getJavaString());
    }
  }
}

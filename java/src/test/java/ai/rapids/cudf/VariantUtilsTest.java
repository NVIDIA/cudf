/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.ListType;
import ai.rapids.cudf.HostColumnVector.StructData;
import ai.rapids.cudf.HostColumnVector.StructType;

import org.junit.jupiter.api.Test;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class VariantUtilsTest extends CudfTestBase {
  private static final ListType BINARY_TYPE =
      new ListType(true, new BasicType(false, DType.UINT8));
  private static final StructType VARIANT_TYPE =
      new StructType(true, Arrays.asList(BINARY_TYPE, BINARY_TYPE));

  private static List<Byte> bytes(int... values) {
    List<Byte> list = new ArrayList<>(values.length);
    for (int value : values) {
      list.add(VariantEncoder.oneByte(value, "byte literal"));
    }
    return list;
  }

  @SafeVarargs
  private static List<Byte> concat(List<Byte>... lists) {
    List<Byte> result = new ArrayList<>();
    for (List<Byte> list : lists) {
      result.addAll(list);
    }
    return result;
  }

  private static StructData variant(List<Byte> metadata, List<Byte> value) {
    return new StructData(metadata, value);
  }

  private static Field field(String name, List<Byte> value) {
    return new Field(name, value);
  }

  private static Map<String, List<Byte>> fields(Field... fields) {
    Map<String, List<Byte>> result = new LinkedHashMap<>();
    for (Field field : fields) {
      if (result.put(field.name, field.value) != null) {
        throw new IllegalArgumentException("duplicate Variant field: " + field.name);
      }
    }
    return result;
  }

  private static final class Field {
    private final String name;
    private final List<Byte> value;

    private Field(String name, List<Byte> value) {
      this.name = name;
      this.value = value;
    }
  }

  private static final class VariantEncoder {
    private static final int VERSION = 1;
    private static final int SORTED = 0;
    private static final int SMALL_OFFSET_SIZE = 1;
    private static final int SMALL_FIELD_ID_SIZE = 1;
    private static final int SMALL_CONTAINER = 0;

    private final String[] keys;
    private final Map<String, Integer> keyIndices = new HashMap<>();

    private VariantEncoder(String... keys) {
      this.keys = keys;
      for (int i = 0; i < keys.length; i++) {
        keyIndices.put(keys[i], i);
      }
    }

    private List<Byte> metadata() {
      List<Byte> offsets = new ArrayList<>();
      List<Byte> encodedKeys = new ArrayList<>();
      int offset = 0;
      for (String key : keys) {
        offsets.add(oneByte(offset, "metadata offset"));
        byte[] keyBytes = key.getBytes(StandardCharsets.UTF_8);
        offset += keyBytes.length;
        encodedKeys.addAll(box(keyBytes));
      }
      offsets.add(oneByte(offset, "metadata offset"));
      int header = (VERSION & 0x0f)
          | ((SORTED & 0x01) << 4)
          | (((SMALL_OFFSET_SIZE - 1) & 0x03) << 6);
      return concat(bytes(header, keys.length), offsets, encodedKeys);
    }

    private List<Byte> object(Map<String, List<Byte>> fields) {
      return object(fields, sortedFieldNames(fields));
    }

    private List<Byte> object(Map<String, List<Byte>> fields, String... valueOrder) {
      return object(fields, Arrays.asList(valueOrder));
    }

    private List<Byte> object(Map<String, List<Byte>> fields, List<String> valueOrder) {
      List<String> sortedNames = sortedFieldNames(fields);
      if (valueOrder.size() != fields.size()) {
        throw new IllegalArgumentException("value order must include every Variant field");
      }

      Map<String, Integer> valueOffsets = new HashMap<>();
      List<Byte> encodedValues = new ArrayList<>();
      int offset = 0;
      for (String name : valueOrder) {
        List<Byte> encodedValue = fields.get(name);
        if (encodedValue == null) {
          throw new IllegalArgumentException("unknown Variant field in value order: " + name);
        }
        if (valueOffsets.put(name, offset) != null) {
          throw new IllegalArgumentException("duplicate Variant field in value order: " + name);
        }
        offset += encodedValue.size();
        encodedValues.addAll(encodedValue);
      }
      if (valueOffsets.size() != fields.size()) {
        throw new IllegalArgumentException("value order must include every Variant field");
      }

      List<Byte> fieldIds = new ArrayList<>();
      List<Byte> offsets = new ArrayList<>();
      for (String name : sortedNames) {
        fieldIds.add(oneByte(keyIndex(name), "field id"));
        offsets.add(oneByte(valueOffsets.get(name), "object offset"));
      }
      offsets.add(oneByte(encodedValues.size(), "object offset"));

      int header = header(0x02,
          ((SMALL_FIELD_ID_SIZE - 1) & 0x03)
              | (((SMALL_OFFSET_SIZE - 1) & 0x03) << 2)
              | ((SMALL_CONTAINER & 0x01) << 4));
      return concat(bytes(header, fields.size()), fieldIds, offsets, encodedValues);
    }

    private static List<String> sortedFieldNames(Map<String, List<Byte>> fields) {
      List<String> sortedNames = new ArrayList<>(fields.keySet());
      sortedNames.sort(VariantEncoder::compareUtf8Unsigned);
      return sortedNames;
    }

    private static int compareUtf8Unsigned(String left, String right) {
      byte[] leftBytes = left.getBytes(StandardCharsets.UTF_8);
      byte[] rightBytes = right.getBytes(StandardCharsets.UTF_8);
      int commonLength = Math.min(leftBytes.length, rightBytes.length);
      for (int i = 0; i < commonLength; i++) {
        int difference = Byte.toUnsignedInt(leftBytes[i]) - Byte.toUnsignedInt(rightBytes[i]);
        if (difference != 0) {
          return difference;
        }
      }
      return Integer.compare(leftBytes.length, rightBytes.length);
    }

    private static List<Byte> int8(int value) {
      if (value < Byte.MIN_VALUE || value > Byte.MAX_VALUE) {
        throw new IllegalArgumentException("int8 value out of range: " + value);
      }
      return bytes(simple(0x03), value & 0xff);
    }

    private static List<Byte> int16(int value) {
      if (value < Short.MIN_VALUE || value > Short.MAX_VALUE) {
        throw new IllegalArgumentException("int16 value out of range: " + value);
      }
      return bytes(simple(0x04), value & 0xff, (value >>> 8) & 0xff);
    }

    private static List<Byte> int32(int value) {
      return bytes(simple(0x05),
          value & 0xff,
          (value >>> 8) & 0xff,
          (value >>> 16) & 0xff,
          (value >>> 24) & 0xff);
    }

    private static List<Byte> int64(long value) {
      return bytes(simple(0x06),
          (int) (value & 0xff),
          (int) ((value >>> 8) & 0xff),
          (int) ((value >>> 16) & 0xff),
          (int) ((value >>> 24) & 0xff),
          (int) ((value >>> 32) & 0xff),
          (int) ((value >>> 40) & 0xff),
          (int) ((value >>> 48) & 0xff),
          (int) ((value >>> 56) & 0xff));
    }

    private static List<Byte> string(String value) {
      byte[] bytes = value.getBytes(StandardCharsets.UTF_8);
      if (bytes.length >= (1 << 6)) {
        throw new IllegalArgumentException("only 6-bit string lengths are supported");
      }
      return concat(bytes(header(0x01, bytes.length)), box(bytes));
    }

    private static List<Byte> box(byte[] bytes) {
      List<Byte> result = new ArrayList<>(bytes.length);
      for (byte value : bytes) {
        result.add(value);
      }
      return result;
    }

    private int keyIndex(String key) {
      Integer index = keyIndices.get(key);
      if (index == null) {
        throw new IllegalArgumentException("unknown Variant metadata key: " + key);
      }
      return index;
    }

    private static byte oneByte(int value, String description) {
      if (value < 0 || value >= (1 << 8)) {
        throw new IllegalArgumentException(description + " does not fit in one byte: " + value);
      }
      return (byte) value;
    }

    private static int header(int kind, int value) {
      if (kind < 0 || kind >= (1 << 2)) {
        throw new IllegalArgumentException("Variant header kind does not fit in two bits: " + kind);
      }
      if (value < 0 || value >= (1 << 6)) {
        throw new IllegalArgumentException("Variant header value does not fit in six bits: " + value);
      }
      return (value << 2) | kind;
    }

    private static int simple(int type) {
      return header(0x00, type);
    }
  }

  // Test data is constructed according to Apache Parquet's Variant encoding spec:
  // https://github.com/apache/parquet-format/blob/master/VariantEncoding.md
  private static ColumnVector makeApacheObjectNestedVariantColumn() {
    VariantEncoder encoder = new VariantEncoder(
        "id", "species", "name", "population", "observation",
        "time", "location", "value", "temperature", "humidity");
    return ColumnVector.fromStructs(VARIANT_TYPE, variant(
        encoder.metadata(),
        encoder.object(fields(
            field("id", VariantEncoder.int8(1)),
            field("species", encoder.object(fields(
                field("name", VariantEncoder.string("lava monster")),
                field("population", VariantEncoder.int16(6789))))),
            field("observation", encoder.object(fields(
                field("time", VariantEncoder.string("12:34:56")),
                field("location", VariantEncoder.string("In the Volcano")),
                field("value", encoder.object(fields(
                    field("temperature", VariantEncoder.int8(123)),
                    field("humidity", VariantEncoder.int16(456))))))))),
            // Store values in a different order than field-id lookup order to cover
            // non-monotonic object offsets.
            "id", "species", "observation")));
  }

  private static ColumnVector makeXyzVariantColumn() {
    VariantEncoder xyEncoder = new VariantEncoder("x", "y");
    VariantEncoder xzEncoder = new VariantEncoder("x", "z");
    VariantEncoder yEncoder = new VariantEncoder("y");

    return ColumnVector.fromStructs(
        VARIANT_TYPE,
        variant(xyEncoder.metadata(), xyEncoder.object(fields(
            field("x", VariantEncoder.int32(7)),
            field("y", VariantEncoder.string("hi"))))),
        variant(xzEncoder.metadata(), xzEncoder.object(fields(
            field("x", VariantEncoder.int32(42)),
            field("z", VariantEncoder.int32(99))))),
        variant(yEncoder.metadata(), yEncoder.object(fields(
            field("y", VariantEncoder.string("zzz"))))));
  }

  private static ColumnVector makeExactWidthIntVariantColumn() {
    VariantEncoder encoder = new VariantEncoder("b", "l");
    return ColumnVector.fromStructs(VARIANT_TYPE, variant(
        encoder.metadata(),
        encoder.object(fields(
            field("b", VariantEncoder.int8(42)),
            field("l", VariantEncoder.int64(1234567890123456789L))))));
  }

  @Test
  void extractStringField() {
    try (ColumnVector variant = makeXyzVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "y", DType.STRING);
         ColumnVector expected = ColumnVector.fromStrings("hi", null, "zzz")) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractIntField() {
    try (ColumnVector variant = makeXyzVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "x", DType.INT32);
         ColumnVector expected = ColumnVector.fromBoxedInts(7, 42, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractDollarPrefixedPath() {
    try (ColumnVector variant = makeXyzVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "$.x", DType.INT32);
         ColumnVector expected = ColumnVector.fromBoxedInts(7, 42, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractNestedStringField() {
    try (ColumnVector variant = makeApacheObjectNestedVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(
             variant, "$.species.name", DType.STRING);
         ColumnVector expected = ColumnVector.fromStrings("lava monster")) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractNestedInt16Field() {
    try (ColumnVector variant = makeApacheObjectNestedVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(
             variant, "$.species.population", DType.INT16);
         ColumnVector expected = ColumnVector.fromBoxedShorts((short) 6789)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractNestedFieldWithNonMonotonicOffset() {
    try (ColumnVector variant = makeApacheObjectNestedVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(
             variant, "$.observation.value.temperature", DType.INT8);
         ColumnVector expected = ColumnVector.fromBoxedBytes((byte) 123)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractInt8Field() {
    try (ColumnVector variant = makeExactWidthIntVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "b", DType.INT8);
         ColumnVector expected = ColumnVector.fromBoxedBytes((byte) 42)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractInt64Field() {
    try (ColumnVector variant = makeExactWidthIntVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "l", DType.INT64);
         ColumnVector expected = ColumnVector.fromBoxedLongs(1234567890123456789L)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void getThenCastFieldValue() {
    try (ColumnVector variant = makeXyzVariantColumn();
         ColumnVector valueBytes = VariantUtils.getVariantFieldValue(variant, "z");
         ColumnVector result = VariantUtils.castVariantValue(valueBytes, DType.INT32);
         ColumnVector expected = ColumnVector.fromBoxedInts(null, 99, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void emptyInputProducesEmptyOutput() {
    try (ColumnVector variant = ColumnVector.fromStructs(VARIANT_TYPE);
         ColumnVector result = VariantUtils.extractVariantField(variant, "x", DType.INT32);
         ColumnVector expected = ColumnVector.fromBoxedInts()) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void emptyPathThrows() {
    try (ColumnVector variant = makeXyzVariantColumn()) {
      assertThrows(CudfException.class, () -> VariantUtils.getVariantFieldValue(variant, ""));
      assertThrows(CudfException.class,
          () -> VariantUtils.extractVariantField(variant, "", DType.INT32));
    }
  }

  @Test
  void malformedPathThrows() {
    try (ColumnVector variant = makeXyzVariantColumn()) {
      assertThrows(CudfException.class,
          () -> VariantUtils.getVariantFieldValue(variant, "$.x["));
      assertThrows(CudfException.class,
          () -> VariantUtils.extractVariantField(variant, "$.x[", DType.INT32));
    }
  }

  @Test
  void nullPathThrows() {
    try (ColumnVector variant = makeXyzVariantColumn()) {
      assertThrows(NullPointerException.class,
          () -> VariantUtils.getVariantFieldValue(variant, null));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.extractVariantField(variant, null, DType.INT32));
    }
  }

  @Test
  void nullVariantStructThrows() {
    assertThrows(NullPointerException.class,
        () -> VariantUtils.getVariantFieldValue(null, "x"));
    assertThrows(NullPointerException.class,
        () -> VariantUtils.extractVariantField(null, "x", DType.INT32));
  }

  @Test
  void unsupportedTargetTypeThrows() {
    try (ColumnVector variant = makeXyzVariantColumn();
         ColumnVector valueBytes = VariantUtils.getVariantFieldValue(variant, "x")) {
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.castVariantValue(valueBytes, DType.FLOAT64));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.extractVariantField(variant, "x", DType.FLOAT64));
    }
  }

  @Test
  void nullCastArgumentsThrow() {
    assertThrows(NullPointerException.class, () -> VariantUtils.castVariantValue(null, null));
    assertThrows(NullPointerException.class,
        () -> VariantUtils.castVariantValue(null, DType.INT32));
    assertThrows(NullPointerException.class,
        () -> VariantUtils.castVariantValue(null, DType.FLOAT64));
  }

  @Test
  void parentStructNullIsPreserved() {
    VariantEncoder encoder = new VariantEncoder("x");
    try (ColumnVector variant = ColumnVector.fromStructs(
             VARIANT_TYPE,
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.int32(7))))),
             null);
         ColumnVector result = VariantUtils.extractVariantField(variant, "x", DType.INT32);
         ColumnVector expected = ColumnVector.fromBoxedInts(7, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }
}

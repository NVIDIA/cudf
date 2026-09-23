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
import static org.junit.jupiter.api.Assertions.assertEquals;
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

    @SafeVarargs
    private static List<Byte> array(List<Byte>... values) {
      List<Byte> offsets = new ArrayList<>(values.length + 1);
      List<Byte> encodedValues = new ArrayList<>();
      int offset = 0;
      for (List<Byte> value : values) {
        offsets.add(oneByte(offset, "array offset"));
        offset += value.size();
        encodedValues.addAll(value);
      }
      offsets.add(oneByte(offset, "array offset"));

      int header = header(0x03,
          ((SMALL_OFFSET_SIZE - 1) & 0x03)
              | ((SMALL_CONTAINER & 0x01) << 2));
      return concat(bytes(header, values.length), offsets, encodedValues);
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

    private static List<Byte> float32(float value) {
      int bits = Float.floatToRawIntBits(value);
      return bytes(simple(0x0e),
          bits & 0xff,
          (bits >>> 8) & 0xff,
          (bits >>> 16) & 0xff,
          (bits >>> 24) & 0xff);
    }

    private static List<Byte> float64(double value) {
      long bits = Double.doubleToRawLongBits(value);
      return bytes(simple(0x07),
          (int) (bits & 0xff),
          (int) ((bits >>> 8) & 0xff),
          (int) ((bits >>> 16) & 0xff),
          (int) ((bits >>> 24) & 0xff),
          (int) ((bits >>> 32) & 0xff),
          (int) ((bits >>> 40) & 0xff),
          (int) ((bits >>> 48) & 0xff),
          (int) ((bits >>> 56) & 0xff));
    }

    private static List<Byte> bool(boolean value) {
      return bytes(simple(value ? 0x01 : 0x02));
    }

    private static List<Byte> nullValue() {
      return bytes(simple(0x00));
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

    private static List<Byte> primitiveHeader(int type) {
      return bytes(simple(type));
    }

    private static List<Byte> shortStringHeader(int length) {
      return bytes(header(0x01, length));
    }

    private static List<Byte> containerHeader(boolean array) {
      return bytes(header(array ? 0x03 : 0x02, 0));
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

  private static ColumnVector makeArrayVariantColumn() {
    VariantEncoder encoder = new VariantEncoder();
    return ColumnVector.fromStructs(
        VARIANT_TYPE,
        variant(encoder.metadata(), VariantEncoder.array(
            VariantEncoder.int8(2), VariantEncoder.int8(1), VariantEncoder.int8(5))),
        variant(encoder.metadata(), VariantEncoder.array(
            VariantEncoder.int8(9), VariantEncoder.int8(8))));
  }

  private static ColumnVector makeNestedArrayVariantColumn() {
    VariantEncoder encoder = new VariantEncoder();
    return ColumnVector.fromStructs(
        VARIANT_TYPE,
        variant(encoder.metadata(), VariantEncoder.array(
            VariantEncoder.array(VariantEncoder.int8(2), VariantEncoder.int8(1)),
            VariantEncoder.array(VariantEncoder.int8(5)))),
        variant(encoder.metadata(), VariantEncoder.array(
            VariantEncoder.array(VariantEncoder.int8(9)),
            VariantEncoder.array(VariantEncoder.int8(8), VariantEncoder.int8(7)))));
  }

  private static ColumnVector makeMixedArrayVariantColumn() {
    VariantEncoder encoder = new VariantEncoder("a", "b");
    return ColumnVector.fromStructs(VARIANT_TYPE, variant(
        encoder.metadata(),
        encoder.object(fields(field("a", VariantEncoder.array(
            encoder.object(fields(field("b", VariantEncoder.int32(7)))),
            encoder.object(fields(field("b", VariantEncoder.int32(42))))))))));
  }

  private static ColumnVector makeFloatVariantColumn() {
    VariantEncoder encoder = new VariantEncoder("f", "d");
    return ColumnVector.fromStructs(
        VARIANT_TYPE,
        variant(encoder.metadata(), encoder.object(fields(
            field("f", VariantEncoder.float32(1.25f)),
            field("d", VariantEncoder.float64(-2.5))))),
        variant(encoder.metadata(), encoder.object(fields(
            field("f", VariantEncoder.float32(-4.5f)),
            field("d", VariantEncoder.float64(8.125))))));
  }

  private static ColumnVector makeBoolVariantColumn() {
    VariantEncoder encoder = new VariantEncoder("b");
    return ColumnVector.fromStructs(
        VARIANT_TYPE,
        variant(encoder.metadata(), encoder.object(fields(
            field("b", VariantEncoder.int8(0))))),
        variant(encoder.metadata(), encoder.object(fields(
            field("b", VariantEncoder.bool(true))))),
        variant(encoder.metadata(), encoder.object(fields(
            field("b", VariantEncoder.bool(false))))),
        variant(encoder.metadata(), encoder.object(fields(
            field("b", VariantEncoder.nullValue())))),
        variant(encoder.metadata(), encoder.object(fields(
            field("b", VariantEncoder.int8(1))))),
        null);
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
  void extractRootArrayElement() {
    try (ColumnVector variant = makeArrayVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "$[0]", DType.INT8);
         ColumnVector expected = ColumnVector.fromBoxedBytes((byte) 2, (byte) 9)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void getThenCastArrayElement() {
    try (ColumnVector variant = makeArrayVariantColumn();
         ColumnVector valueBytes = VariantUtils.getVariantFieldValue(variant, "$[1]");
         ColumnVector result = VariantUtils.castVariantValue(valueBytes, DType.INT8);
         ColumnVector expected = ColumnVector.fromBoxedBytes((byte) 1, (byte) 8)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractNestedArrayElement() {
    try (ColumnVector variant = makeNestedArrayVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "$[0][0]", DType.INT8);
         ColumnVector expected = ColumnVector.fromBoxedBytes((byte) 2, (byte) 9)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractMixedObjectArrayPath() {
    try (ColumnVector variant = makeMixedArrayVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(
             variant, "$.a[1].b", DType.INT32);
         ColumnVector expected = ColumnVector.fromBoxedInts(42)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void arrayIndexResolutionFailuresProduceNulls() {
    try (ColumnVector variant = makeArrayVariantColumn();
         ColumnVector outOfBounds = VariantUtils.extractVariantField(
             variant, "$[99]", DType.INT8);
         ColumnVector containerMismatch = VariantUtils.extractVariantField(
             variant, "$[0][0]", DType.INT8);
         ColumnVector expected = ColumnVector.fromBoxedBytes(null, null)) {
      assertColumnsAreEqual(expected, outOfBounds);
      assertColumnsAreEqual(expected, containerMismatch);
    }
  }

  @Test
  void extractArrayElementFromSlice() {
    try (ColumnVector variant = makeArrayVariantColumn();
         ColumnVector sliced = variant.subVector(1, 2);
         ColumnVector result = VariantUtils.extractVariantField(sliced, "$[0]", DType.INT8);
         ColumnVector expected = ColumnVector.fromBoxedBytes((byte) 9)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void castFloatFields() {
    try (ColumnVector variant = makeFloatVariantColumn();
         ColumnVector floatBytes = VariantUtils.getVariantFieldValue(variant, "f");
         ColumnVector doubleBytes = VariantUtils.getVariantFieldValue(variant, "d");
         ColumnVector floats = VariantUtils.castVariantValue(floatBytes, DType.FLOAT32);
         ColumnVector doubles = VariantUtils.castVariantValue(doubleBytes, DType.FLOAT64);
         ColumnVector expectedFloats = ColumnVector.fromBoxedFloats(1.25f, -4.5f);
         ColumnVector expectedDoubles = ColumnVector.fromBoxedDoubles(-2.5, 8.125)) {
      assertColumnsAreEqual(expectedFloats, floats);
      assertColumnsAreEqual(expectedDoubles, doubles);
    }
  }

  @Test
  void extractFloatFields() {
    try (ColumnVector variant = makeFloatVariantColumn();
         ColumnVector floats = VariantUtils.extractVariantField(variant, "f", DType.FLOAT32);
         ColumnVector doubles = VariantUtils.extractVariantField(variant, "d", DType.FLOAT64);
         ColumnVector expectedFloats = ColumnVector.fromBoxedFloats(1.25f, -4.5f);
         ColumnVector expectedDoubles = ColumnVector.fromBoxedDoubles(-2.5, 8.125)) {
      assertColumnsAreEqual(expectedFloats, floats);
      assertColumnsAreEqual(expectedDoubles, doubles);
    }
  }

  @Test
  void floatWidthMismatchProducesNulls() {
    try (ColumnVector variant = makeFloatVariantColumn();
         ColumnVector result = VariantUtils.extractVariantField(variant, "f", DType.FLOAT64);
         ColumnVector expected = ColumnVector.fromBoxedDoubles(null, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void castFloatWidthMismatchesProduceNulls() {
    try (ColumnVector variant = makeFloatVariantColumn();
         ColumnVector floatBytes = VariantUtils.getVariantFieldValue(variant, "f");
         ColumnVector doubleBytes = VariantUtils.getVariantFieldValue(variant, "d");
         ColumnVector floatsAsDoubles = VariantUtils.castVariantValue(floatBytes, DType.FLOAT64);
         ColumnVector doublesAsFloats = VariantUtils.castVariantValue(doubleBytes, DType.FLOAT32);
         ColumnVector expectedDoubles = ColumnVector.fromBoxedDoubles(null, null);
         ColumnVector expectedFloats = ColumnVector.fromBoxedFloats(null, null)) {
      assertColumnsAreEqual(expectedDoubles, floatsAsDoubles);
      assertColumnsAreEqual(expectedFloats, doublesAsFloats);
    }
  }

  @Test
  void castFloatNullInputIsPreserved() {
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE, VariantEncoder.float32(1.25f), null);
         ColumnVector result = VariantUtils.castVariantValue(values, DType.FLOAT32);
         ColumnVector expected = ColumnVector.fromBoxedFloats(1.25f, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void castBooleanValues() {
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.int8(0),
             VariantEncoder.bool(true),
             VariantEncoder.bool(false),
             VariantEncoder.nullValue(),
             VariantEncoder.int8(1),
             null);
         ColumnVector sliced = values.subVector(1, 6);
         ColumnVector unslicedResult = VariantUtils.castVariantValue(values, DType.BOOL8);
         ColumnVector result = VariantUtils.castVariantValue(sliced, DType.BOOL8);
         ColumnVector unslicedExpected = ColumnVector.fromBoxedBooleans(
             null, true, false, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, null, null, null)) {
      assertColumnsAreEqual(unslicedExpected, unslicedResult);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractBooleanFieldFromSlice() {
    try (ColumnVector variant = makeBoolVariantColumn();
         ColumnVector sliced = variant.subVector(1, 6);
         ColumnVector unslicedResult = VariantUtils.extractVariantField(variant, "b", DType.BOOL8);
         ColumnVector result = VariantUtils.extractVariantField(sliced, "b", DType.BOOL8);
         ColumnVector unslicedExpected = ColumnVector.fromBoxedBooleans(
             null, true, false, null, null, null);
         ColumnVector expected = ColumnVector.fromBoxedBooleans(true, false, null, null, null)) {
      assertColumnsAreEqual(unslicedExpected, unslicedResult);
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void variantLogicalTypeIdsMatchNativeValues() {
    VariantLogicalType[] types = {
        VariantLogicalType.OBJECT,
        VariantLogicalType.ARRAY,
        VariantLogicalType.NULL_VALUE,
        VariantLogicalType.BOOLEAN,
        VariantLogicalType.LONG_VALUE,
        VariantLogicalType.STRING,
        VariantLogicalType.DOUBLE_VALUE,
        VariantLogicalType.DECIMAL,
        VariantLogicalType.DATE,
        VariantLogicalType.TIMESTAMP,
        VariantLogicalType.TIMESTAMP_NTZ,
        VariantLogicalType.FLOAT_VALUE,
        VariantLogicalType.BINARY,
        VariantLogicalType.UUID,
        VariantLogicalType.TIME_NTZ
    };

    for (int nativeId = 0; nativeId < types.length; nativeId++) {
      assertEquals(nativeId, types[nativeId].getNativeId());
      assertEquals(types[nativeId], VariantLogicalType.fromNative(nativeId));
    }
    assertThrows(IllegalArgumentException.class, () -> VariantLogicalType.fromNative(-1));
    assertThrows(IllegalArgumentException.class, () -> VariantLogicalType.fromNative(15));
  }

  @Test
  void getVariantTypeIdCoversLogicalTypesAndPhysicalAliases() {
    // get_variant_type_id intentionally classifies only the header byte. Supplying header-only
    // blobs here both pins every physical-to-logical mapping and verifies that truncated payloads
    // with recognized headers remain classifiable.
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.containerHeader(false),       // OBJECT
             VariantEncoder.containerHeader(true),        // ARRAY
             VariantEncoder.primitiveHeader(0),            // NULL_VALUE
             VariantEncoder.primitiveHeader(1),            // BOOLEAN_TRUE
             VariantEncoder.primitiveHeader(2),            // BOOLEAN_FALSE
             VariantEncoder.primitiveHeader(3),            // INT8
             VariantEncoder.primitiveHeader(4),            // INT16
             VariantEncoder.primitiveHeader(5),            // INT32
             VariantEncoder.primitiveHeader(6),            // INT64
             VariantEncoder.shortStringHeader(0),          // SHORT_STRING
             VariantEncoder.primitiveHeader(16),           // LONG_STRING
             VariantEncoder.primitiveHeader(7),            // FLOAT64
             VariantEncoder.primitiveHeader(8),            // DECIMAL4
             VariantEncoder.primitiveHeader(9),            // DECIMAL8
             VariantEncoder.primitiveHeader(10),           // DECIMAL16
             VariantEncoder.primitiveHeader(11),           // DATE
             VariantEncoder.primitiveHeader(12),           // TIMESTAMP_MICROS
             VariantEncoder.primitiveHeader(18),           // TIMESTAMP_NANOS
             VariantEncoder.primitiveHeader(13),           // TIMESTAMP_NTZ_MICROS
             VariantEncoder.primitiveHeader(19),           // TIMESTAMP_NTZ_NANOS
             VariantEncoder.primitiveHeader(14),           // FLOAT32
             VariantEncoder.primitiveHeader(15),           // BINARY
             VariantEncoder.primitiveHeader(20),           // UUID
             VariantEncoder.primitiveHeader(17));          // TIME_NTZ_MICROS
         ColumnVector result = VariantUtils.getVariantTypeId(values);
         ColumnVector expected = ColumnVector.fromUnsignedBytes(
             (byte) VariantLogicalType.OBJECT.getNativeId(),
             (byte) VariantLogicalType.ARRAY.getNativeId(),
             (byte) VariantLogicalType.NULL_VALUE.getNativeId(),
             (byte) VariantLogicalType.BOOLEAN.getNativeId(),
             (byte) VariantLogicalType.BOOLEAN.getNativeId(),
             (byte) VariantLogicalType.LONG_VALUE.getNativeId(),
             (byte) VariantLogicalType.LONG_VALUE.getNativeId(),
             (byte) VariantLogicalType.LONG_VALUE.getNativeId(),
             (byte) VariantLogicalType.LONG_VALUE.getNativeId(),
             (byte) VariantLogicalType.STRING.getNativeId(),
             (byte) VariantLogicalType.STRING.getNativeId(),
             (byte) VariantLogicalType.DOUBLE_VALUE.getNativeId(),
             (byte) VariantLogicalType.DECIMAL.getNativeId(),
             (byte) VariantLogicalType.DECIMAL.getNativeId(),
             (byte) VariantLogicalType.DECIMAL.getNativeId(),
             (byte) VariantLogicalType.DATE.getNativeId(),
             (byte) VariantLogicalType.TIMESTAMP.getNativeId(),
             (byte) VariantLogicalType.TIMESTAMP.getNativeId(),
             (byte) VariantLogicalType.TIMESTAMP_NTZ.getNativeId(),
             (byte) VariantLogicalType.TIMESTAMP_NTZ.getNativeId(),
             (byte) VariantLogicalType.FLOAT_VALUE.getNativeId(),
             (byte) VariantLogicalType.BINARY.getNativeId(),
             (byte) VariantLogicalType.UUID.getNativeId(),
             (byte) VariantLogicalType.TIME_NTZ.getNativeId())) {
      assertEquals(DType.UINT8, result.getType());
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void getVariantTypeIdNullAndInvalidHeaderBehavior() {
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.primitiveHeader(0),
             null,
             bytes(),
             bytes(0xfc),
             VariantEncoder.primitiveHeader(6));
         ColumnVector result = VariantUtils.getVariantTypeId(values);
         ColumnVector expected = ColumnVector.fromBoxedUnsignedBytes(
             (byte) VariantLogicalType.NULL_VALUE.getNativeId(),
             null,
             null,
             null,
             (byte) VariantLogicalType.LONG_VALUE.getNativeId())) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void getVariantTypeIdAcceptsExtractedValues() {
    try (ColumnVector variant = makeXyzVariantColumn();
         ColumnVector values = VariantUtils.getVariantFieldValue(variant, "x");
         ColumnVector result = VariantUtils.getVariantTypeId(values);
         ColumnVector expected = ColumnVector.fromBoxedUnsignedBytes(
             (byte) VariantLogicalType.LONG_VALUE.getNativeId(),
             (byte) VariantLogicalType.LONG_VALUE.getNativeId(),
             null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void getVariantTypeIdSupportsSlicedInput() {
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.primitiveHeader(0),
             VariantEncoder.containerHeader(true),
             VariantEncoder.primitiveHeader(14),
             VariantEncoder.shortStringHeader(0),
             VariantEncoder.containerHeader(false));
         ColumnVector slice = values.subVector(1, 4);
         ColumnVector result = VariantUtils.getVariantTypeId(slice);
         ColumnVector expected = ColumnVector.fromUnsignedBytes(
             (byte) VariantLogicalType.ARRAY.getNativeId(),
             (byte) VariantLogicalType.FLOAT_VALUE.getNativeId(),
             (byte) VariantLogicalType.STRING.getNativeId())) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void getVariantTypeIdHandlesEmptyAndAllNullInput() {
    try (ColumnVector empty = ColumnVector.fromLists(BINARY_TYPE);
         ColumnVector emptyResult = VariantUtils.getVariantTypeId(empty);
         ColumnVector expectedEmpty = ColumnVector.fromUnsignedBytes();
         ColumnVector allNull = ColumnVector.fromLists(
             BINARY_TYPE, (List<Byte>) null, (List<Byte>) null);
         ColumnVector allNullResult = VariantUtils.getVariantTypeId(allNull);
         ColumnVector expectedAllNull = ColumnVector.fromBoxedUnsignedBytes(null, null)) {
      assertEquals(DType.UINT8, emptyResult.getType());
      assertColumnsAreEqual(expectedEmpty, emptyResult);
      assertColumnsAreEqual(expectedAllNull, allNullResult);
    }
  }

  @Test
  void getVariantTypeIdRejectsInvalidInput() {
    ListType listOfInt = new ListType(true, new BasicType(false, DType.INT32));
    try (ColumnVector notAList = ColumnVector.fromInts(1);
         ColumnVector wrongChildType = ColumnVector.fromLists(listOfInt, Arrays.asList(1, 2))) {
      assertThrows(CudfException.class, () -> VariantUtils.getVariantTypeId(notAList));
      assertThrows(CudfException.class, () -> VariantUtils.getVariantTypeId(wrongChildType));
    }
    assertThrows(NullPointerException.class, () -> VariantUtils.getVariantTypeId(null));
  }

  @Test
  void emptyInputUnsupportedDirectCastThrows() {
    try (ColumnVector empty = ColumnVector.fromLists(BINARY_TYPE)) {
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.castVariantValue(empty, DType.UINT32));
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
  void emptyInputSupportsFloatOutput() {
    try (ColumnVector variant = ColumnVector.fromStructs(VARIANT_TYPE);
         ColumnVector result = VariantUtils.extractVariantField(variant, "x", DType.FLOAT32);
         ColumnVector expected = ColumnVector.fromBoxedFloats()) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void emptyInputSupportsBooleanOutput() {
    try (ColumnVector values = ColumnVector.fromLists(BINARY_TYPE);
         ColumnVector directResult = VariantUtils.castVariantValue(values, DType.BOOL8);
         ColumnVector variant = ColumnVector.fromStructs(VARIANT_TYPE);
         ColumnVector extractedResult = VariantUtils.extractVariantField(
             variant, "x", DType.BOOL8);
         ColumnVector expected = ColumnVector.fromBoxedBooleans()) {
      assertColumnsAreEqual(expected, directResult);
      assertColumnsAreEqual(expected, extractedResult);
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
          () -> VariantUtils.castVariantValue(valueBytes, DType.UINT32));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.extractVariantField(variant, "x", DType.UINT32));
    }
  }

  @Test
  void nullArgumentsThrow() {
    try (ColumnVector values = ColumnVector.fromLists(BINARY_TYPE, VariantEncoder.int32(1));
         ColumnVector variant = makeXyzVariantColumn()) {
      assertThrows(NullPointerException.class, () -> VariantUtils.castVariantValue(null, null));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.castVariantValue(null, DType.INT32));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.castVariantValue(null, DType.FLOAT64));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.castVariantValue(values, null));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.extractVariantField(variant, "x", null));
    }
  }

  @Test
  void invalidInputShapesThrow() {
    try (ColumnVector nonVariant = ColumnVector.fromInts(1, 2, 3)) {
      assertThrows(CudfException.class,
          () -> VariantUtils.getVariantFieldValue(nonVariant, "x"));
      assertThrows(CudfException.class,
          () -> VariantUtils.castVariantValue(nonVariant, DType.INT32));
      assertThrows(CudfException.class,
          () -> VariantUtils.extractVariantField(nonVariant, "x", DType.INT32));
    }
  }

  @Test
  void truncatedFloatPayloadProducesNull() {
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE, bytes(VariantEncoder.simple(0x07), 0x00, 0x00));
         ColumnVector result = VariantUtils.castVariantValue(values, DType.FLOAT64);
         ColumnVector expected = ColumnVector.fromBoxedDoubles((Double) null)) {
      assertColumnsAreEqual(expected, result);
    }
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

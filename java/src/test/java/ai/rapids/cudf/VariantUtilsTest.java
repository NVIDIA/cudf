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

import java.math.BigInteger;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

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

  private static byte statusId(VariantOperationStatus status) {
    return (byte) status.getNativeId();
  }

  private static void assertStatusColumnsAreEqual(ColumnView expected, ColumnView actual) {
    assertColumnsAreEqual(expected, actual);
    // Logical equality and null count alone do not exclude an allocated all-valid mask.
    assertNull(actual.getValid());
  }

  private static final class ScopedExecutor implements AutoCloseable {
    private final ExecutorService executor;

    private ScopedExecutor(int threads) {
      executor = Executors.newFixedThreadPool(threads);
    }

    @Override
    public void close() throws InterruptedException {
      executor.shutdownNow();
      assertTrue(executor.awaitTermination(10, TimeUnit.SECONDS));
    }
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

    private static List<Byte> decimal4(int unscaled, int fractionalDigits) {
      return decimal(BigInteger.valueOf(unscaled), fractionalDigits, 4, 0x08);
    }

    private static List<Byte> decimal8(long unscaled, int fractionalDigits) {
      return decimal(BigInteger.valueOf(unscaled), fractionalDigits, 8, 0x09);
    }

    private static List<Byte> decimal16(BigInteger unscaled, int fractionalDigits) {
      return decimal(unscaled, fractionalDigits, 16, 0x0a);
    }

    private static List<Byte> decimal(
        BigInteger unscaled, int fractionalDigits, int width, int primitiveType) {
      if (fractionalDigits < 0 || fractionalDigits > 255) {
        throw new IllegalArgumentException(
            "decimal fractional digits do not fit in one byte: " + fractionalDigits);
      }
      BigInteger limit = BigInteger.ONE.shiftLeft(width * Byte.SIZE - 1);
      if (unscaled.compareTo(limit.negate()) < 0 || unscaled.compareTo(limit) >= 0) {
        throw new IllegalArgumentException(
            "decimal value does not fit in " + width + " bytes: " + unscaled);
      }

      List<Byte> result = new ArrayList<>(width + 2);
      result.add(oneByte(simple(primitiveType), "decimal header"));
      result.add(oneByte(fractionalDigits, "decimal fractional digits"));
      for (int byteIndex = 0; byteIndex < width; byteIndex++) {
        result.add(unscaled.shiftRight(byteIndex * Byte.SIZE).byteValue());
      }
      return result;
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
  void castDecimalWidthsToEveryOutputWidthFromSlice() {
    DType decimal32 = DType.create(DType.DTypeEnum.DECIMAL32, -2);
    DType decimal64 = DType.create(DType.DTypeEnum.DECIMAL64, -2);
    DType decimal128 = DType.create(DType.DTypeEnum.DECIMAL128, -2);
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.int8(0),
             VariantEncoder.decimal4(1234, 2),
             VariantEncoder.decimal8(-5678, 2),
             VariantEncoder.decimal16(BigInteger.valueOf(9012), 2));
         ColumnVector sliced = values.subVector(1, 4);
         ColumnVector result32 = VariantUtils.castVariantValue(sliced, decimal32);
         ColumnVector result64 = VariantUtils.castVariantValue(sliced, decimal64);
         ColumnVector result128 = VariantUtils.castVariantValue(sliced, decimal128);
         ColumnVector expected32 = ColumnVector.decimalFromInts(-2, 1234, -5678, 9012);
         ColumnVector expected64 = ColumnVector.decimalFromLongs(-2, 1234, -5678, 9012);
         ColumnVector expected128 = ColumnVector.decimalFromBigInt(
             -2, BigInteger.valueOf(1234), BigInteger.valueOf(-5678), BigInteger.valueOf(9012))) {
      assertColumnsAreEqual(expected32, result32);
      assertColumnsAreEqual(expected64, result64);
      assertColumnsAreEqual(expected128, result128);
    }
  }

  @Test
  @SuppressWarnings("unchecked")
  void castDecimalAcceptsEveryEncodedScaleForEveryWidth() {
    List<List<Byte>> encoded = new ArrayList<>();
    List<Integer> expected32 = new ArrayList<>();
    List<Long> expected64 = new ArrayList<>();
    List<BigInteger> expected128 = new ArrayList<>();
    for (int fractionalDigits = 0; fractionalDigits <= 38; fractionalDigits++) {
      for (int sign : new int[] {1, -1}) {
        // The encoded scale is independent of the payload width, even for DECIMAL4/8.
        encoded.add(VariantEncoder.decimal4(sign, fractionalDigits));
        encoded.add(VariantEncoder.decimal8(sign, fractionalDigits));
        encoded.add(VariantEncoder.decimal16(BigInteger.valueOf(sign), fractionalDigits));
        for (int width = 0; width < 3; width++) {
          expected32.add(fractionalDigits <= 9 ?
              BigInteger.TEN.pow(9 - fractionalDigits).intValueExact() * sign : 0);
          expected64.add(fractionalDigits <= 18 ?
              BigInteger.TEN.pow(18 - fractionalDigits).longValueExact() * sign : 0L);
          expected128.add(BigInteger.TEN.pow(38 - fractionalDigits)
              .multiply(BigInteger.valueOf(sign)));
        }
      }
    }
    List<Byte>[] encodedRows = (List<Byte>[]) encoded.toArray(new List<?>[0]);
    try (ColumnVector values = ColumnVector.fromLists(BINARY_TYPE, encodedRows);
         ColumnVector result32 = VariantUtils.castVariantValue(
             values, DType.create(DType.DTypeEnum.DECIMAL32, -9));
         ColumnVector result64 = VariantUtils.castVariantValue(
             values, DType.create(DType.DTypeEnum.DECIMAL64, -18));
         ColumnVector result128 = VariantUtils.castVariantValue(
             values, DType.create(DType.DTypeEnum.DECIMAL128, -38));
         ColumnVector expected32Column = ColumnVector.decimalFromBoxedInts(
             -9, expected32.toArray(new Integer[0]));
         ColumnVector expected64Column = ColumnVector.decimalFromBoxedLongs(
             -18, expected64.toArray(new Long[0]));
         ColumnVector expected128Column = ColumnVector.decimalFromBigInt(
             -38, expected128.toArray(new BigInteger[0]))) {
      assertColumnsAreEqual(expected32Column, result32);
      assertColumnsAreEqual(expected64Column, result64);
      assertColumnsAreEqual(expected128Column, result128);
    }
  }

  @Test
  @SuppressWarnings("unchecked")
  void positiveDecimalTargetScaleUsesBase10ExponentForBothEntryPoints() {
    VariantEncoder encoder = new VariantEncoder("price");
    List<List<Byte>> encoded = Arrays.asList(
        VariantEncoder.decimal4(12345, 0),
        VariantEncoder.decimal8(-12345, 0),
        VariantEncoder.decimal16(BigInteger.valueOf(12345), 2));
    List<StructData> rows = new ArrayList<>();
    for (List<Byte> value : encoded) {
      rows.add(variant(encoder.metadata(), encoder.object(fields(field("price", value)))));
    }
    DType[] targets = {
        DType.create(DType.DTypeEnum.DECIMAL32, 2),
        DType.create(DType.DTypeEnum.DECIMAL64, 2),
        DType.create(DType.DTypeEnum.DECIMAL128, 2)
    };
    List<Byte>[] encodedRows = (List<Byte>[]) encoded.toArray(new List<?>[0]);
    try (ColumnVector values = ColumnVector.fromLists(BINARY_TYPE, encodedRows);
         ColumnVector variant = ColumnVector.fromStructs(VARIANT_TYPE, rows);
         ColumnVector expected32 = ColumnVector.decimalFromInts(2, 123, -123, 1);
         ColumnVector expected64 = ColumnVector.decimalFromLongs(2, 123, -123, 1);
         ColumnVector expected128 = ColumnVector.decimalFromBigInt(
             2, BigInteger.valueOf(123), BigInteger.valueOf(-123), BigInteger.ONE)) {
      ColumnVector[] expected = {expected32, expected64, expected128};
      for (int i = 0; i < targets.length; i++) {
        try (ColumnVector direct = VariantUtils.castVariantValue(values, targets[i]);
             ColumnVector extracted =
                 VariantUtils.extractVariantField(variant, "price", targets[i])) {
          assertColumnsAreEqual(expected[i], direct);
          assertColumnsAreEqual(expected[i], extracted);
        }
      }
    }
  }

  @Test
  void castDecimalRejectsOverflowWhenWideningTargetScale() {
    DType target = DType.create(DType.DTypeEnum.DECIMAL128, -38);
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.decimal4(0, 0),
             VariantEncoder.decimal4(1, 0),
             VariantEncoder.decimal4(-1, 0),
             VariantEncoder.decimal8(2, 0),
             VariantEncoder.decimal16(BigInteger.valueOf(-2), 0));
         ColumnVector result = VariantUtils.castVariantValue(values, target);
         ColumnVector expected = ColumnVector.decimalFromBigInt(
             -38, BigInteger.ZERO, BigInteger.TEN.pow(38), BigInteger.TEN.pow(38).negate(),
             null, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void castDecimal128UsesFullPayloadWidth() {
    BigInteger high = BigInteger.ONE.shiftLeft(100).add(BigInteger.valueOf(1234));
    DType decimal128 = DType.create(DType.DTypeEnum.DECIMAL128, 0);
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.decimal16(high, 0),
             VariantEncoder.decimal16(high.negate(), 0));
         ColumnVector result = VariantUtils.castVariantValue(values, decimal128);
         ColumnVector expected = ColumnVector.decimalFromBigInt(0, high, high.negate())) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void castDecimalRescalesAndTruncatesTowardZero() {
    DType target = DType.create(DType.DTypeEnum.DECIMAL32, -1);
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.decimal4(125, 2),
             VariantEncoder.decimal4(-125, 2),
             VariantEncoder.decimal4(7, 0),
             VariantEncoder.decimal8(30, 3),
             VariantEncoder.decimal16(BigInteger.ONE, 38));
         ColumnVector result = VariantUtils.castVariantValue(values, target);
         ColumnVector expected = ColumnVector.decimalFromInts(-1, 12, -12, 70, 0, 0)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void castDecimalInvalidRowsProduceNulls() {
    DType target = DType.create(DType.DTypeEnum.DECIMAL32, -3);
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.decimal4(1234, 2),
             VariantEncoder.decimal8((long) Integer.MAX_VALUE + 1, 3),
             VariantEncoder.decimal4(Integer.MAX_VALUE / 1000 + 1, 0),
             VariantEncoder.nullValue(),
             VariantEncoder.int32(1234),
             VariantEncoder.decimal4(1234, 39),
             bytes(VariantEncoder.simple(0x09), 0x03, 0x00),
             null);
         ColumnVector result = VariantUtils.castVariantValue(values, target);
         ColumnVector expected = ColumnVector.decimalFromBoxedInts(
             -3, 12340, null, null, null, null, null, null, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void extractDecimalFieldRescalesAndPreservesNulls() {
    VariantEncoder priceEncoder = new VariantEncoder("price");
    VariantEncoder otherEncoder = new VariantEncoder("other");
    DType target = DType.create(DType.DTypeEnum.DECIMAL64, -2);
    try (ColumnVector variant = ColumnVector.fromStructs(
             VARIANT_TYPE,
             variant(priceEncoder.metadata(), priceEncoder.object(fields(
                 field("price", VariantEncoder.decimal8(123456, 3))))),
             variant(priceEncoder.metadata(), priceEncoder.object(fields(
                 field("price", VariantEncoder.decimal4(-5, 1))))),
             variant(otherEncoder.metadata(), otherEncoder.object(fields(
                 field("other", VariantEncoder.decimal4(1, 0))))),
             variant(priceEncoder.metadata(), priceEncoder.object(fields(
                 field("price", VariantEncoder.nullValue())))),
             null);
         ColumnVector result = VariantUtils.extractVariantField(variant, "price", target);
         ColumnVector expected = ColumnVector.decimalFromBoxedLongs(
             -2, 12345L, -50L, null, null, null)) {
      assertColumnsAreEqual(expected, result);
    }
  }

  @Test
  void emptyInputSupportsEveryDecimalOutputWidthAndScale() {
    DType[] targets = {
        DType.create(DType.DTypeEnum.DECIMAL32, -1),
        DType.create(DType.DTypeEnum.DECIMAL64, -2),
        DType.create(DType.DTypeEnum.DECIMAL128, -3)
    };
    try (ColumnVector values = ColumnVector.fromLists(BINARY_TYPE);
         ColumnVector variant = ColumnVector.fromStructs(VARIANT_TYPE)) {
      for (DType target : targets) {
        try (ColumnVector directResult = VariantUtils.castVariantValue(values, target);
             ColumnVector extractedResult =
                 VariantUtils.extractVariantField(variant, "x", target)) {
          assertEquals(target, directResult.getType());
          assertEquals(target, extractedResult.getType());
          assertEquals(0, directResult.getRowCount());
          assertEquals(0, extractedResult.getRowCount());
        }
      }
    }
  }

  @Test
  void variantOperationStatusIdsMatchNativeValues() {
    VariantOperationStatus[] statuses = {
        VariantOperationStatus.SUCCESS,
        VariantOperationStatus.ROW_NULL,
        VariantOperationStatus.MISSING_PATH,
        VariantOperationStatus.VARIANT_NULL,
        VariantOperationStatus.TYPE_MISMATCH,
        VariantOperationStatus.MALFORMED_VARIANT,
        VariantOperationStatus.OVERFLOW,
        VariantOperationStatus.INVALID_CONVERSION
    };

    for (int nativeId = 0; nativeId < statuses.length; nativeId++) {
      assertEquals(nativeId, statuses[nativeId].getNativeId());
      assertEquals(statuses[nativeId], VariantOperationStatus.fromNative(nativeId));
    }
    assertThrows(IllegalArgumentException.class, () -> VariantOperationStatus.fromNative(-1));
    assertThrows(IllegalArgumentException.class, () -> VariantOperationStatus.fromNative(8));
  }

  @Test
  void getVariantFieldValueStatusDistinguishesRowOutcomes() {
    VariantEncoder encoder = new VariantEncoder("x");
    VariantEncoder missingEncoder = new VariantEncoder("y");
    try (ColumnVector variant = ColumnVector.fromStructs(
             VARIANT_TYPE,
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.int32(7))))),
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.nullValue())))),
             variant(bytes(0x02), encoder.object(fields(
                 field("x", VariantEncoder.int32(9))))),
             null,
             variant(missingEncoder.metadata(), missingEncoder.object(fields(
                 field("y", VariantEncoder.int32(11))))));
         VariantOperationResult result =
             VariantUtils.getVariantFieldValueWithStatus(variant, "x");
         ColumnVector expectedValues = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.int32(7),
             VariantEncoder.nullValue(),
             null,
             null,
             null);
         ColumnVector expectedStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.MALFORMED_VARIANT),
             statusId(VariantOperationStatus.ROW_NULL),
             statusId(VariantOperationStatus.MISSING_PATH))) {
      assertColumnsAreEqual(expectedValues, result.getValue());
      assertStatusColumnsAreEqual(expectedStatus, result.getStatus());
      assertEquals(DType.UINT8, result.getStatus().getType());
      assertEquals(0, result.getStatus().getNullCount());
      assertNull(result.getStatus().getValid());
      assertEquals(result.getValue().getRowCount(), result.getStatus().getRowCount());
    }
  }

  @Test
  void getVariantFieldValueStatusCoversMissingPathForms() {
    VariantEncoder encoder = new VariantEncoder("x");
    try (ColumnVector arrays = makeArrayVariantColumn();
         VariantOperationResult outOfBounds =
             VariantUtils.getVariantFieldValueWithStatus(arrays, "$[99]");
         ColumnVector expectedArrayStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.MISSING_PATH),
             statusId(VariantOperationStatus.MISSING_PATH));
         ColumnVector nullIntermediate = ColumnVector.fromStructs(
             VARIANT_TYPE,
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.nullValue())))));
         VariantOperationResult nested =
             VariantUtils.getVariantFieldValueWithStatus(nullIntermediate, "$.x.y");
         ColumnVector expectedNestedStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.MISSING_PATH))) {
      assertStatusColumnsAreEqual(expectedArrayStatus, outOfBounds.getStatus());
      assertStatusColumnsAreEqual(expectedNestedStatus, nested.getStatus());
    }
  }

  @Test
  void castVariantValueStatusDistinguishesRowOutcomes() {
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.int32(42),
             VariantEncoder.nullValue(),
             VariantEncoder.int8(5),
             bytes(VariantEncoder.simple(0x05), 0x01),
             bytes(),
             null);
         VariantOperationResult result =
             VariantUtils.castVariantValueWithStatus(values, DType.INT32);
         ColumnVector expectedValues = ColumnVector.fromBoxedInts(42, null, null, null, null, null);
         ColumnVector expectedStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.TYPE_MISMATCH),
             statusId(VariantOperationStatus.MALFORMED_VARIANT),
             statusId(VariantOperationStatus.MALFORMED_VARIANT),
             statusId(VariantOperationStatus.ROW_NULL))) {
      assertColumnsAreEqual(expectedValues, result.getValue());
      assertStatusColumnsAreEqual(expectedStatus, result.getStatus());
      assertEquals(0, result.getStatus().getNullCount());
    }
  }

  @Test
  void castBooleanAndStringStatusesUseTheirNativePaths() {
    try (ColumnVector boolValues = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.bool(true),
             VariantEncoder.nullValue(),
             VariantEncoder.int8(1),
             bytes(0xfc));
         VariantOperationResult boolResult =
             VariantUtils.castVariantValueWithStatus(boolValues, DType.BOOL8);
         ColumnVector expectedBools = ColumnVector.fromBoxedBooleans(true, null, null, null);
         ColumnVector expectedBoolStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.TYPE_MISMATCH),
             statusId(VariantOperationStatus.MALFORMED_VARIANT));
         ColumnVector stringValues = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.string("hi"),
             VariantEncoder.nullValue(),
             VariantEncoder.int32(1),
             bytes(VariantEncoder.header(0x01, 5)));
         VariantOperationResult stringResult =
             VariantUtils.castVariantValueWithStatus(stringValues, DType.STRING);
         ColumnVector expectedStrings = ColumnVector.fromStrings("hi", null, null, null);
         ColumnVector expectedStringStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.TYPE_MISMATCH),
             statusId(VariantOperationStatus.MALFORMED_VARIANT))) {
      assertColumnsAreEqual(expectedBools, boolResult.getValue());
      assertStatusColumnsAreEqual(expectedBoolStatus, boolResult.getStatus());
      assertColumnsAreEqual(expectedStrings, stringResult.getValue());
      assertStatusColumnsAreEqual(expectedStringStatus, stringResult.getStatus());
    }
  }

  @Test
  void castFloatingPointStatusesUseTheirNativePaths() {
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.float32(1.25f),
             VariantEncoder.float64(-2.5),
             VariantEncoder.nullValue(),
             bytes(VariantEncoder.simple(0x07), 0x01),
             null);
         VariantOperationResult floats =
             VariantUtils.castVariantValueWithStatus(values, DType.FLOAT32);
         VariantOperationResult doubles =
             VariantUtils.castVariantValueWithStatus(values, DType.FLOAT64);
         ColumnVector expectedFloats =
             ColumnVector.fromBoxedFloats(1.25f, null, null, null, null);
         ColumnVector expectedDoubles =
             ColumnVector.fromBoxedDoubles(null, -2.5, null, null, null);
         ColumnVector expectedFloatStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.TYPE_MISMATCH),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.TYPE_MISMATCH),
             statusId(VariantOperationStatus.ROW_NULL));
         ColumnVector expectedDoubleStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.TYPE_MISMATCH),
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.MALFORMED_VARIANT),
             statusId(VariantOperationStatus.ROW_NULL))) {
      assertColumnsAreEqual(expectedFloats, floats.getValue());
      assertColumnsAreEqual(expectedDoubles, doubles.getValue());
      assertStatusColumnsAreEqual(expectedFloatStatus, floats.getStatus());
      assertStatusColumnsAreEqual(expectedDoubleStatus, doubles.getStatus());
    }
  }

  @Test
  void variantPriorStatusRejectsAllValidMasks() {
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE, VariantEncoder.int32(7), VariantEncoder.int32(8));
         ColumnVector data = ColumnVector.fromUnsignedBytes((byte) 0, (byte) 0);
         ColumnVector maskOwner =
             ColumnVector.fromBoxedUnsignedBytes((byte) 0, (byte) 0, null);
         ColumnView allValidMaskedStatus = new ColumnView(
             DType.UINT8, 2, Optional.of(0L), data.getData(), maskOwner.getValid());
         CloseableArray<ColumnView> statusViews =
             CloseableArray.wrap(allValidMaskedStatus.splitAsViews(1));
         CloseableArray<ColumnView> valueViews = CloseableArray.wrap(values.splitAsViews(1))) {
      ColumnView slicedStatus = statusViews.get(1);
      ColumnView slicedValues = valueViews.get(1);
      assertEquals(0, allValidMaskedStatus.getNullCount());
      assertNotNull(allValidMaskedStatus.getValid());
      assertEquals(0, slicedStatus.getNullCount());
      assertNotNull(slicedStatus.getValid());
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.castVariantValueWithStatus(values, DType.INT32, allValidMaskedStatus));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.castVariantValueWithStatus(slicedValues, DType.INT32, slicedStatus));
    }
  }

  @Test
  void variantPriorStatusSupportsNonzeroOffsetAndEmptyInputs() {
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE, VariantEncoder.int32(0), VariantEncoder.int32(7),
             VariantEncoder.int32(8), VariantEncoder.int32(9));
         ColumnVector status = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.TYPE_MISMATCH),
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.MISSING_PATH),
             statusId(VariantOperationStatus.SUCCESS));
         CloseableArray<ColumnView> valueViews = CloseableArray.wrap(values.splitAsViews(1));
         CloseableArray<ColumnView> statusViews = CloseableArray.wrap(status.splitAsViews(1));
         VariantOperationResult sliced =
             VariantUtils.castVariantValueWithStatus(
                 valueViews.get(1), DType.INT32, statusViews.get(1));
         ColumnVector expectedValues = ColumnVector.fromBoxedInts(7, null, 9);
         ColumnVector expectedStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.MISSING_PATH),
             statusId(VariantOperationStatus.SUCCESS));
         ColumnVector emptyValues = ColumnVector.fromLists(BINARY_TYPE);
         ColumnVector emptyStatus = ColumnVector.fromUnsignedBytes();
         VariantOperationResult empty =
             VariantUtils.castVariantValueWithStatus(emptyValues, DType.INT32, emptyStatus)) {
      assertColumnsAreEqual(expectedValues, sliced.getValue());
      assertStatusColumnsAreEqual(expectedStatus, sliced.getStatus());
      assertColumnsAreEqual(expectedStatus, statusViews.get(1));
      assertEquals(0, empty.getValue().getRowCount());
      assertEquals(0, empty.getStatus().getRowCount());
      assertNull(empty.getStatus().getValid());
    }
  }

  @Test
  void variantResultCloseIsIdempotentAndRetainedColumnsRemainValid() {
    try (ColumnVector values = ColumnVector.fromLists(BINARY_TYPE, VariantEncoder.int32(7));
         VariantOperationResult result =
             VariantUtils.castVariantValueWithStatus(values, DType.INT32);
         ColumnVector retainedValue = result.getValue().incRefCount();
         ColumnVector retainedStatus = result.getStatus().incRefCount();
         ColumnVector expectedValue = ColumnVector.fromInts(7);
         ColumnVector expectedStatus = ColumnVector.fromUnsignedBytes((byte) 0)) {
      result.close();
      result.close();
      assertEquals(1, retainedValue.getRefCount());
      assertEquals(1, retainedStatus.getRefCount());
      assertColumnsAreEqual(expectedValue, retainedValue);
      assertColumnsAreEqual(expectedStatus, retainedStatus);
    }
  }

  @Test
  void castVariantValuePropagatesEveryPriorStatusWithoutMutatingInput() {
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.int32(7),
             VariantEncoder.int32(1),
             VariantEncoder.int32(2),
             VariantEncoder.int32(3),
             VariantEncoder.int32(4),
             VariantEncoder.int32(5),
             VariantEncoder.int32(6),
             VariantEncoder.int32(7),
             VariantEncoder.int8(8));
         ColumnVector priorStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.ROW_NULL),
             statusId(VariantOperationStatus.MISSING_PATH),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.TYPE_MISMATCH),
             statusId(VariantOperationStatus.MALFORMED_VARIANT),
             statusId(VariantOperationStatus.OVERFLOW),
             statusId(VariantOperationStatus.INVALID_CONVERSION),
             statusId(VariantOperationStatus.SUCCESS));
         ColumnVector expectedOriginalStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.ROW_NULL),
             statusId(VariantOperationStatus.MISSING_PATH),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.TYPE_MISMATCH),
             statusId(VariantOperationStatus.MALFORMED_VARIANT),
             statusId(VariantOperationStatus.OVERFLOW),
             statusId(VariantOperationStatus.INVALID_CONVERSION),
             statusId(VariantOperationStatus.SUCCESS));
         ColumnVector expectedFinalStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.ROW_NULL),
             statusId(VariantOperationStatus.MISSING_PATH),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.TYPE_MISMATCH),
             statusId(VariantOperationStatus.MALFORMED_VARIANT),
             statusId(VariantOperationStatus.OVERFLOW),
             statusId(VariantOperationStatus.INVALID_CONVERSION),
             statusId(VariantOperationStatus.TYPE_MISMATCH));
         VariantOperationResult result =
             VariantUtils.castVariantValueWithStatus(values, DType.INT32, priorStatus);
         ColumnVector expectedValues =
             ColumnVector.fromBoxedInts(7, null, null, null, null, null, null, null, null)) {
      assertColumnsAreEqual(expectedValues, result.getValue());
      assertStatusColumnsAreEqual(expectedFinalStatus, result.getStatus());
      assertColumnsAreEqual(expectedOriginalStatus, priorStatus);
    }
  }

  @Test
  void fusedVariantStatusMatchesStagedExtraction() {
    VariantEncoder encoder = new VariantEncoder("x");
    VariantEncoder missingEncoder = new VariantEncoder("y");
    try (ColumnVector variant = ColumnVector.fromStructs(
             VARIANT_TYPE,
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.int32(7))))),
             variant(missingEncoder.metadata(), missingEncoder.object(fields(
                 field("y", VariantEncoder.int32(8))))),
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.nullValue())))),
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.int8(9))))),
             variant(bytes(0x02), encoder.object(fields(
                 field("x", VariantEncoder.int32(10))))),
             null);
         VariantOperationResult raw =
             VariantUtils.getVariantFieldValueWithStatus(variant, "x");
         VariantOperationResult staged = VariantUtils.castVariantValueWithStatus(
             raw.getValue(), DType.INT32, raw.getStatus());
         VariantOperationResult fused =
             VariantUtils.extractVariantFieldWithStatus(variant, "x", DType.INT32);
         ColumnVector expectedValues =
             ColumnVector.fromBoxedInts(7, null, null, null, null, null);
         ColumnVector expectedStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.MISSING_PATH),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.TYPE_MISMATCH),
             statusId(VariantOperationStatus.MALFORMED_VARIANT),
             statusId(VariantOperationStatus.ROW_NULL));
         ColumnVector expectedRawStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.MISSING_PATH),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.MALFORMED_VARIANT),
             statusId(VariantOperationStatus.ROW_NULL))) {
      assertColumnsAreEqual(expectedValues, staged.getValue());
      assertColumnsAreEqual(expectedValues, fused.getValue());
      assertStatusColumnsAreEqual(expectedStatus, staged.getStatus());
      assertStatusColumnsAreEqual(expectedStatus, fused.getStatus());
      assertStatusColumnsAreEqual(expectedRawStatus, raw.getStatus());
    }
  }

  @Test
  void variantStatusSupportsSlicesAndEmptyInputs() {
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.int8(0),
             VariantEncoder.int32(7),
             null,
             VariantEncoder.int32(9));
         ColumnVector slice = values.subVector(1, 4);
         VariantOperationResult sliced =
             VariantUtils.castVariantValueWithStatus(slice, DType.INT32);
         ColumnVector expectedValues = ColumnVector.fromBoxedInts(7, null, 9);
         ColumnVector expectedStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.ROW_NULL),
             statusId(VariantOperationStatus.SUCCESS));
         ColumnVector emptyValues = ColumnVector.fromLists(BINARY_TYPE);
         ColumnVector emptyVariant = ColumnVector.fromStructs(VARIANT_TYPE);
         VariantOperationResult emptyCast =
             VariantUtils.castVariantValueWithStatus(emptyValues, DType.INT32);
         VariantOperationResult emptyGet =
             VariantUtils.getVariantFieldValueWithStatus(emptyVariant, "x");
         VariantOperationResult emptyExtract =
             VariantUtils.extractVariantFieldWithStatus(emptyVariant, "x", DType.INT32)) {
      assertColumnsAreEqual(expectedValues, sliced.getValue());
      assertStatusColumnsAreEqual(expectedStatus, sliced.getStatus());
      assertEquals(0, emptyCast.getValue().getRowCount());
      assertEquals(0, emptyCast.getStatus().getRowCount());
      assertEquals(0, emptyGet.getValue().getRowCount());
      assertEquals(0, emptyGet.getStatus().getRowCount());
      assertEquals(0, emptyExtract.getValue().getRowCount());
      assertEquals(0, emptyExtract.getStatus().getRowCount());
      assertEquals(DType.UINT8, emptyCast.getStatus().getType());
      assertEquals(DType.UINT8, emptyGet.getStatus().getType());
      assertEquals(DType.UINT8, emptyExtract.getStatus().getType());
      assertNull(emptyCast.getStatus().getValid());
      assertNull(emptyGet.getStatus().getValid());
      assertNull(emptyExtract.getStatus().getValid());
    }
  }

  @Test
  void variantStatusApisValidateArguments() {
    try (ColumnVector variant = makeXyzVariantColumn();
         ColumnVector values = VariantUtils.getVariantFieldValue(variant, "x");
         ColumnVector wrongType = ColumnVector.fromInts(0, 0, 0);
         ColumnVector wrongSize = ColumnVector.fromUnsignedBytes((byte) 0);
         ColumnVector nullableStatus =
             ColumnVector.fromBoxedUnsignedBytes((byte) 0, null, (byte) 0);
         ColumnVector nonVariant = ColumnVector.fromInts(1, 2, 3)) {
      assertThrows(NullPointerException.class,
          () -> VariantUtils.getVariantFieldValueWithStatus(null, "x"));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.getVariantFieldValueWithStatus(variant, null));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.castVariantValueWithStatus(null, DType.INT32));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.castVariantValueWithStatus(values, null));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.castVariantValueWithStatus(values, DType.INT32, null));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.extractVariantFieldWithStatus(null, "x", DType.INT32));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.extractVariantFieldWithStatus(variant, null, DType.INT32));
      assertThrows(NullPointerException.class,
          () -> VariantUtils.extractVariantFieldWithStatus(variant, "x", null));

      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.castVariantValueWithStatus(values, DType.UINT32));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.extractVariantFieldWithStatus(variant, "x", DType.UINT32));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.castVariantValueWithStatus(values, DType.INT32, wrongType));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.castVariantValueWithStatus(values, DType.INT32, wrongSize));
      assertThrows(IllegalArgumentException.class,
          () -> VariantUtils.castVariantValueWithStatus(values, DType.INT32, nullableStatus));

      assertThrows(CudfException.class,
          () -> VariantUtils.getVariantFieldValueWithStatus(variant, "$.x["));
      assertThrows(CudfException.class,
          () -> VariantUtils.extractVariantFieldWithStatus(variant, "$.x[", DType.INT32));
      assertThrows(CudfException.class,
          () -> VariantUtils.getVariantFieldValueWithStatus(nonVariant, "x"));
      assertThrows(CudfException.class,
          () -> VariantUtils.castVariantValueWithStatus(nonVariant, DType.INT32));
      assertThrows(CudfException.class,
          () -> VariantUtils.extractVariantFieldWithStatus(nonVariant, "x", DType.INT32));
    }
  }

  @Test
  void decimalStatusCoversAllWidthsAndScaleForwarding() {
    VariantEncoder encoder = new VariantEncoder("x");
    DType[] targets = {
        DType.create(DType.DTypeEnum.DECIMAL32, -2),
        DType.create(DType.DTypeEnum.DECIMAL64, -2),
        DType.create(DType.DTypeEnum.DECIMAL128, -2)
    };
    try (ColumnVector variant = ColumnVector.fromStructs(
             VARIANT_TYPE,
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.decimal4(1234, 2))))),
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.decimal8(-5678, 2))))),
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.decimal16(BigInteger.valueOf(9012), 2))))));
         VariantOperationResult raw = VariantUtils.getVariantFieldValueWithStatus(variant, "x");
         ColumnVector expectedStatus =
             ColumnVector.fromUnsignedBytes((byte) 0, (byte) 0, (byte) 0)) {
      for (DType target : targets) {
        try (ColumnVector expected = VariantUtils.castVariantValue(raw.getValue(), target);
             VariantOperationResult direct =
                 VariantUtils.castVariantValueWithStatus(raw.getValue(), target);
             VariantOperationResult staged =
                 VariantUtils.castVariantValueWithStatus(raw.getValue(), target, raw.getStatus());
             VariantOperationResult fused =
                 VariantUtils.extractVariantFieldWithStatus(variant, "x", target)) {
          assertEquals(target, direct.getValue().getType());
          assertColumnsAreEqual(expected, direct.getValue());
          assertColumnsAreEqual(expected, staged.getValue());
          assertColumnsAreEqual(expected, fused.getValue());
          assertStatusColumnsAreEqual(expectedStatus, direct.getStatus());
          assertStatusColumnsAreEqual(expectedStatus, staged.getStatus());
          assertStatusColumnsAreEqual(expectedStatus, fused.getStatus());
          assertStatusColumnsAreEqual(expectedStatus, raw.getStatus());
        }
      }
    }
  }

  @Test
  void decimalStatusPreservesOverflowAndMalformedOutcomes() {
    DType target = DType.create(DType.DTypeEnum.DECIMAL32, -2);
    try (ColumnVector values = ColumnVector.fromLists(
             BINARY_TYPE,
             VariantEncoder.decimal4(1234, 2),
             VariantEncoder.decimal8((long) Integer.MAX_VALUE + 1, 2),
             VariantEncoder.decimal16(BigInteger.ONE, 39),
             VariantEncoder.primitiveHeader(8),
             VariantEncoder.nullValue(), null, VariantEncoder.int32(1));
         VariantOperationResult direct = VariantUtils.castVariantValueWithStatus(values, target);
         ColumnVector expectedValues = ColumnVector.decimalFromBoxedInts(
             -2, 1234, null, null, null, null, null, null);
         ColumnVector expectedStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.OVERFLOW),
             statusId(VariantOperationStatus.MALFORMED_VARIANT),
             statusId(VariantOperationStatus.MALFORMED_VARIANT),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.ROW_NULL),
             statusId(VariantOperationStatus.TYPE_MISMATCH))) {
      assertColumnsAreEqual(expectedValues, direct.getValue());
      assertStatusColumnsAreEqual(expectedStatus, direct.getStatus());
    }
  }

  @Test
  void variantPathsPreserveEmbeddedNulAcrossAllExtractionApis() {
    String key = "a\0b";
    VariantEncoder encoder = new VariantEncoder("a", key);
    try (ColumnVector variant = ColumnVector.fromStructs(
             VARIANT_TYPE, variant(encoder.metadata(), encoder.object(fields(
                 field("a", VariantEncoder.int32(7)),
                 field(key, VariantEncoder.int32(42))))));
         ColumnVector expectedRaw = ColumnVector.fromLists(BINARY_TYPE, VariantEncoder.int32(42));
         ColumnVector expectedValue = ColumnVector.fromInts(42);
         ColumnVector expectedStatus = ColumnVector.fromUnsignedBytes((byte) 0)) {
      for (String path : new String[] {key, "$." + key}) {
        try (ColumnVector raw = VariantUtils.getVariantFieldValue(variant, path);
             ColumnVector fused = VariantUtils.extractVariantField(variant, path, DType.INT32);
             VariantOperationResult rawStatus =
                 VariantUtils.getVariantFieldValueWithStatus(variant, path);
             VariantOperationResult fusedStatus =
                 VariantUtils.extractVariantFieldWithStatus(variant, path, DType.INT32)) {
          assertColumnsAreEqual(expectedRaw, raw);
          assertColumnsAreEqual(expectedValue, fused);
          assertColumnsAreEqual(expectedRaw, rawStatus.getValue());
          assertStatusColumnsAreEqual(expectedStatus, rawStatus.getStatus());
          assertColumnsAreEqual(expectedValue, fusedStatus.getValue());
          assertStatusColumnsAreEqual(expectedStatus, fusedStatus.getStatus());
        }
      }
    }
  }

  @Test
  void variantResultCloseAttemptsBothColumnsWhenCallbacksThrow() {
    RuntimeException shared = new IllegalStateException("shared close failure");
    // The CUDA exception below is only a Java callback failure; it does not poison the GPU.
    Throwable[][] failures = {
        {new IllegalStateException("value close"), new IllegalArgumentException("status close")},
        {new AssertionError("value close"), new AssertionError("status close")},
        {new CudaFatalException("callback failure", 719),
            new IllegalStateException("status close")},
        {shared, shared}
    };
    for (Throwable[] failure : failures) {
      try (ColumnVector values = ColumnVector.fromLists(BINARY_TYPE, VariantEncoder.int32(7))) {
        VariantOperationResult result =
            VariantUtils.castVariantValueWithStatus(values, DType.INT32);
        ColumnVector value = result.getValue();
        ColumnVector status = result.getStatus();
        try {
          ColumnVector[] columns = {value, status};
          for (int i = 0; i < columns.length; i++) {
            Throwable callbackFailure = failure[i];
            columns[i].setEventHandler((column, refCount) -> {
              if (callbackFailure instanceof Error) {
                throw (Error) callbackFailure;
              }
              throw (RuntimeException) callbackFailure;
            });
          }
          Throwable thrown = assertThrows(Throwable.class, result::close);
          assertSame(failure[0], thrown);
          assertEquals(0, value.getRefCount());
          assertEquals(0, status.getRefCount());
          if (failure[0] == failure[1]) {
            assertEquals(0, thrown.getSuppressed().length);
          } else {
            assertEquals(1, thrown.getSuppressed().length);
            assertSame(failure[1], thrown.getSuppressed()[0]);
          }
          assertDoesNotThrow(result::close);
        } finally {
          // Also clean up correctly if this regression test is run against the old implementation.
          value.setEventHandler(null);
          status.setEventHandler(null);
          if (value.getRefCount() > 0) {
            value.close();
          }
          if (status.getRefCount() > 0) {
            status.close();
          }
        }
      }
    }
  }

  @Test
  void variantResultSupportsConcurrentClose() throws Exception {
    try (ColumnVector values = ColumnVector.fromLists(BINARY_TYPE, VariantEncoder.int32(7))) {
      for (int iteration = 0; iteration < 32; iteration++) {
        try (VariantOperationResult result =
                 VariantUtils.castVariantValueWithStatus(values, DType.INT32);
             ScopedExecutor workers = new ScopedExecutor(8)) {
          ColumnVector value = result.getValue();
          ColumnVector status = result.getStatus();
          AtomicInteger valueCloses = new AtomicInteger();
          AtomicInteger statusCloses = new AtomicInteger();
          value.setEventHandler((column, refCount) -> valueCloses.incrementAndGet());
          status.setEventHandler((column, refCount) -> statusCloses.incrementAndGet());
          CountDownLatch ready = new CountDownLatch(8);
          CountDownLatch start = new CountDownLatch(1);
          List<Future<?>> closes = new ArrayList<>();
          try {
            for (int i = 0; i < 8; i++) {
              closes.add(workers.executor.submit(() -> {
                ready.countDown();
                if (!start.await(10, TimeUnit.SECONDS)) {
                  throw new AssertionError("concurrent close was not started");
                }
                result.close();
                return null;
              }));
            }
            assertTrue(ready.await(10, TimeUnit.SECONDS));
            start.countDown();
            for (Future<?> close : closes) {
              close.get(10, TimeUnit.SECONDS);
            }
            assertEquals(1, valueCloses.get());
            assertEquals(1, statusCloses.get());
            assertEquals(0, value.getRefCount());
            assertEquals(0, status.getRefCount());
          } finally {
            start.countDown();
          }
        }
      }
    }
  }

  @Test
  void variantResultClosesWithoutHoldingMonitorAcrossCallbacks() throws Exception {
    CountDownLatch enteredCallback = new CountDownLatch(1);
    CountDownLatch releaseCallback = new CountDownLatch(1);
    try (ColumnVector values = ColumnVector.fromLists(BINARY_TYPE, VariantEncoder.int32(7));
         VariantOperationResult result =
             VariantUtils.castVariantValueWithStatus(values, DType.INT32);
         ScopedExecutor workers = new ScopedExecutor(2)) {
      ColumnVector value = result.getValue();
      ColumnVector status = result.getStatus();
      AtomicInteger valueCloses = new AtomicInteger();
      AtomicInteger statusCloses = new AtomicInteger();
      value.setEventHandler((column, refCount) -> {
        valueCloses.incrementAndGet();
        enteredCallback.countDown();
        assertFalse(Thread.holdsLock(result));
        try {
          assertTrue(releaseCallback.await(10, TimeUnit.SECONDS));
        } catch (InterruptedException failure) {
          Thread.currentThread().interrupt();
          throw new AssertionError(failure);
        }
        result.close();
      });
      status.setEventHandler((column, refCount) -> statusCloses.incrementAndGet());
      Future<?> owner = workers.executor.submit(result::close);
      try {
        assertTrue(enteredCallback.await(10, TimeUnit.SECONDS));
        // This close must not wait for the first caller's callback to finish.
        workers.executor.submit(result::close).get(10, TimeUnit.SECONDS);
        assertEquals(1, valueCloses.get());
        assertEquals(0, statusCloses.get());
      } finally {
        releaseCallback.countDown();
      }
      owner.get(10, TimeUnit.SECONDS);
      assertEquals(1, valueCloses.get());
      assertEquals(1, statusCloses.get());
      assertEquals(0, value.getRefCount());
      assertEquals(0, status.getRefCount());
    } finally {
      releaseCallback.countDown();
    }
  }

  @Test
  void variantStructStatusExtractionSupportsNonzeroOffsets() {
    VariantEncoder encoder = new VariantEncoder("x");
    try (ColumnVector variant = ColumnVector.fromStructs(
             VARIANT_TYPE,
             variant(encoder.metadata(), encoder.object(fields())),
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.int32(7))))),
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.int8(8))))),
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.nullValue())))),
             null,
             variant(encoder.metadata(), encoder.object(fields())));
         CloseableArray<ColumnView> views = CloseableArray.wrap(variant.splitAsViews(1));
         VariantOperationResult raw =
             VariantUtils.getVariantFieldValueWithStatus(views.get(1), "x");
         VariantOperationResult fused =
             VariantUtils.extractVariantFieldWithStatus(views.get(1), "x", DType.INT32);
         ColumnVector expectedRaw = ColumnVector.fromLists(BINARY_TYPE,
             VariantEncoder.int32(7), VariantEncoder.int8(8), VariantEncoder.nullValue(),
             null, null);
         ColumnVector expectedValues = ColumnVector.fromBoxedInts(7, null, null, null, null);
         ColumnVector expectedRawStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS), statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.ROW_NULL),
             statusId(VariantOperationStatus.MISSING_PATH));
         ColumnVector expectedFusedStatus = ColumnVector.fromUnsignedBytes(
             statusId(VariantOperationStatus.SUCCESS),
             statusId(VariantOperationStatus.TYPE_MISMATCH),
             statusId(VariantOperationStatus.VARIANT_NULL),
             statusId(VariantOperationStatus.ROW_NULL),
             statusId(VariantOperationStatus.MISSING_PATH))) {
      assertColumnsAreEqual(expectedRaw, raw.getValue());
      assertStatusColumnsAreEqual(expectedRawStatus, raw.getStatus());
      assertColumnsAreEqual(expectedValues, fused.getValue());
      assertStatusColumnsAreEqual(expectedFusedStatus, fused.getStatus());
    }
  }

  @Test
  void decimalStatusUsesPositiveScaleWithExplicitExpectedValues() {
    VariantEncoder encoder = new VariantEncoder("x");
    try (ColumnVector variant = ColumnVector.fromStructs(VARIANT_TYPE,
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.decimal4(12345, 0))))),
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.decimal8(-12345, 0))))),
             variant(encoder.metadata(), encoder.object(fields(
                 field("x", VariantEncoder.decimal16(BigInteger.valueOf(12345), 2))))));
         VariantOperationResult raw = VariantUtils.getVariantFieldValueWithStatus(variant, "x");
         ColumnVector expected32 = ColumnVector.decimalFromInts(2, 123, -123, 1);
         ColumnVector expected64 = ColumnVector.decimalFromLongs(2, 123, -123, 1);
         ColumnVector expected128 = ColumnVector.decimalFromBigInt(
             2, BigInteger.valueOf(123), BigInteger.valueOf(-123), BigInteger.ONE);
         ColumnVector expectedStatus =
             ColumnVector.fromUnsignedBytes((byte) 0, (byte) 0, (byte) 0)) {
      for (ColumnVector expected : new ColumnVector[] {expected32, expected64, expected128}) {
        DType target = expected.getType();
        try (VariantOperationResult direct =
                 VariantUtils.castVariantValueWithStatus(raw.getValue(), target);
             VariantOperationResult staged =
                 VariantUtils.castVariantValueWithStatus(raw.getValue(), target, raw.getStatus());
             VariantOperationResult fused =
                 VariantUtils.extractVariantFieldWithStatus(variant, "x", target)) {
          assertColumnsAreEqual(expected, direct.getValue());
          assertColumnsAreEqual(expected, staged.getValue());
          assertColumnsAreEqual(expected, fused.getValue());
          assertStatusColumnsAreEqual(expectedStatus, direct.getStatus());
          assertStatusColumnsAreEqual(expectedStatus, staged.getStatus());
          assertStatusColumnsAreEqual(expectedStatus, fused.getStatus());
          assertStatusColumnsAreEqual(expectedStatus, raw.getStatus());
        }
      }
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

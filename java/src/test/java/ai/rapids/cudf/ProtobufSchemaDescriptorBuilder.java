/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;

/**
 * Test-only fluent builder for {@link ProtobufSchemaDescriptor}.
 *
 * <p>The descriptor is a set of parallel arrays, which is error-prone to assemble by hand: the
 * reader has to count positions across a dozen arrays to see what a single field looks like. This
 * builder lets each field be described in one place, with only the non-default attributes named:
 *
 * <pre>{@code
 * ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
 *     .addField(1, DType.INT32)
 *     .addField(2, DType.STRING).required()
 *     .build();
 * }</pre>
 *
 * <p>Each attribute setter applies to the most recently added field. Wire types are derived from
 * the output type and encoding unless overridden via {@link #wireType(int)}.
 *
 * <p>For nested messages, {@link #down()} descends into the most recently added field so that
 * subsequent {@code addField} calls become its children (with parent index and depth derived
 * automatically); {@link #up()} returns to the enclosing level. This avoids hand-counting flat
 * parent indices as the nesting deepens:
 *
 * <pre>{@code
 * ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptorBuilder()
 *     .addField(1, DType.STRUCT).down()   // message Outer
 *         .addField(1, DType.INT32)       //   int32 a = 1
 *         .addField(2, DType.STRUCT).down()  // Inner b = 2
 *             .addField(1, DType.INT32)   //     int32 x = 1
 *         .up()
 *     .up()
 *     .build();
 * }</pre>
 *
 * <p>{@link #parent(int)} remains available as an explicit escape hatch, e.g. to construct the
 * malformed schemas that the validation tests deliberately feed to the descriptor.
 */
public final class ProtobufSchemaDescriptorBuilder {
  private static final class Field {
    int fieldNumber;
    int outputTypeId;
    int parentIndex = -1;
    int depth = 0;
    int encoding = Protobuf.ENC_DEFAULT;
    Integer wireType = null;  // null => derive from output type + encoding
    boolean isRepeated = false;
    boolean isRequired = false;
    boolean hasDefaultValue = false;
    boolean isOutput = true;
    long defaultInt = 0L;
    double defaultFloat = 0.0;
    boolean defaultBool = false;
    byte[] defaultString = null;
    int[] enumValidValues = null;
    byte[][] enumNames = null;
  }

  private final List<Field> fields = new ArrayList<>();
  // Indices of the fields we have descended into via down(); the top is the current parent.
  // A Deque is required rather than a single "current parent" pointer because parent() allows
  // arbitrary (non-ancestor) parent indices, so up() cannot recover the enclosing context by
  // walking fields.get(currentParent).parentIndex.
  private final Deque<Integer> parentStack = new ArrayDeque<>();

  /**
   * Add a field. By default it is top-level; inside a {@link #down()} scope it becomes a child of
   * the enclosing field, with parent index and depth derived automatically.
   */
  public ProtobufSchemaDescriptorBuilder addField(int fieldNumber, DType outputType) {
    Field f = new Field();
    f.fieldNumber = fieldNumber;
    f.outputTypeId = outputType.getTypeId().getNativeId();
    fields.add(f);
    if (parentStack.isEmpty()) return this;
    return parent(parentStack.peek());
  }

  /**
   * Descend into the most recently added field so that subsequent {@code addField} calls become
   * its children (until the matching {@link #up()}). This is the idiom for expressing message
   * nesting; prefer it over {@link #parent(int)}, which is a low-level primitive.
   */
  public ProtobufSchemaDescriptorBuilder down() {
    if (fields.isEmpty()) {
      throw new IllegalStateException("down() requires a field to descend into");
    }
    parentStack.push(fields.size() - 1);
    return this;
  }

  /** Return to the enclosing nesting level opened by {@link #down()}. */
  public ProtobufSchemaDescriptorBuilder up() {
    if (parentStack.isEmpty()) {
      throw new IllegalStateException("up() called without a matching down()");
    }
    parentStack.pop();
    return this;
  }

  /**
   * Low-level: assign a raw parent index to the current field (depth = parent depth + 1),
   * overriding any parent set by {@link #down()}. Prefer {@link #down()}/{@link #up()} for ordinary
   * nesting. Reserve this for (a) loop-built degenerate chains where {@code parent(i - 1)} reads
   * cleaner, and (b) malformed parent links that {@code down()}/{@code up()} cannot express (e.g. a
   * non-ancestor index) in validation tests.
   */
  public ProtobufSchemaDescriptorBuilder parent(int parentIndex) {
    Field f = current();
    f.parentIndex = parentIndex;
    f.depth = fields.get(parentIndex).depth + 1;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder encoding(int encoding) {
    current().encoding = encoding;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder wireType(int wireType) {
    current().wireType = wireType;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder repeated() {
    return repeated(true);
  }

  /** Set whether the current field is repeated. */
  public ProtobufSchemaDescriptorBuilder repeated(boolean value) {
    current().isRepeated = value;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder required() {
    current().isRequired = true;
    return this;
  }

  /** Mark the current field hidden: decoded for validation but dropped from the output struct. */
  public ProtobufSchemaDescriptorBuilder isOutput(boolean isOutput) {
    current().isOutput = isOutput;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder defaultValue(long value) {
    Field f = current();
    f.hasDefaultValue = true;
    f.defaultInt = value;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder defaultValue(double value) {
    Field f = current();
    f.hasDefaultValue = true;
    f.defaultFloat = value;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder defaultValue(boolean value) {
    Field f = current();
    f.hasDefaultValue = true;
    f.defaultBool = value;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder defaultValue(byte[] value) {
    Field f = current();
    f.hasDefaultValue = true;
    f.defaultString = value;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder defaultValue(String value) {
    return defaultValue(value.getBytes(StandardCharsets.UTF_8));
  }

  /**
   * Set the has-default flag without a value. Used by negative tests that pair a default with an
   * incompatible field (e.g. a repeated or STRUCT field) to exercise validation.
   */
  public ProtobufSchemaDescriptorBuilder hasDefault() {
    return hasDefault(true);
  }

  /** Set whether the current field has a default value. */
  public ProtobufSchemaDescriptorBuilder hasDefault(boolean value) {
    current().hasDefaultValue = value;
    return this;
  }

  /** Provide zero-based enum-as-string metadata for the current field. */
  public ProtobufSchemaDescriptorBuilder enumMetadata(String... names) {
    Field f = current();
    f.encoding = Protobuf.ENC_ENUM_STRING;
    f.enumValidValues = new int[names.length];
    f.enumNames = new byte[names.length][];
    for (int i = 0; i < names.length; i++) {
      f.enumValidValues[i] = i;
      f.enumNames[i] = names[i].getBytes(StandardCharsets.UTF_8);
    }
    return this;
  }

  /** Attach raw enum metadata without forcing the encoding (for validation-only tests). */
  public ProtobufSchemaDescriptorBuilder enumValidValues(int[] validValues) {
    current().enumValidValues = validValues;
    return this;
  }

  public ProtobufSchemaDescriptorBuilder enumNames(byte[][] names) {
    current().enumNames = names;
    return this;
  }

  public ProtobufSchemaDescriptor build() {
    if (!parentStack.isEmpty()) {
      throw new IllegalStateException("build() called inside a down() scope; missing up()");
    }
    int n = fields.size();
    int[] fieldNumbers = new int[n];
    int[] parentIndices = new int[n];
    int[] depthLevels = new int[n];
    int[] wireTypes = new int[n];
    int[] outputTypeIds = new int[n];
    int[] encodings = new int[n];
    boolean[] isRepeated = new boolean[n];
    boolean[] isRequired = new boolean[n];
    boolean[] hasDefaultValue = new boolean[n];
    boolean[] isOutput = new boolean[n];
    long[] defaultInts = new long[n];
    double[] defaultFloats = new double[n];
    boolean[] defaultBools = new boolean[n];
    byte[][] defaultStrings = new byte[n][];
    int[][] enumValidValues = new int[n][];
    byte[][][] enumNames = new byte[n][][];

    for (int i = 0; i < n; i++) {
      Field f = fields.get(i);
      fieldNumbers[i] = f.fieldNumber;
      parentIndices[i] = f.parentIndex;
      depthLevels[i] = f.depth;
      outputTypeIds[i] = f.outputTypeId;
      encodings[i] = f.encoding;
      wireTypes[i] = f.wireType != null ? f.wireType : deriveWireType(f.outputTypeId, f.encoding);
      isRepeated[i] = f.isRepeated;
      isRequired[i] = f.isRequired;
      hasDefaultValue[i] = f.hasDefaultValue;
      isOutput[i] = f.isOutput;
      defaultInts[i] = f.defaultInt;
      defaultFloats[i] = f.defaultFloat;
      defaultBools[i] = f.defaultBool;
      defaultStrings[i] = f.defaultString;
      enumValidValues[i] = f.enumValidValues;
      enumNames[i] = f.enumNames;
    }

    return new ProtobufSchemaDescriptor(fieldNumbers, parentIndices, depthLevels, wireTypes,
        outputTypeIds, encodings, isRepeated, isRequired, hasDefaultValue, isOutput,
        defaultInts, defaultFloats, defaultBools, defaultStrings, enumValidValues, enumNames);
  }

  private Field current() {
    if (fields.isEmpty()) {
      throw new IllegalStateException("addField must be called before setting field attributes");
    }
    return fields.get(fields.size() - 1);
  }

  static int deriveWireType(int typeId, int encoding) {
    if (encoding == Protobuf.ENC_ENUM_STRING) return Protobuf.WT_VARINT;
    if (typeId == DType.FLOAT32.getTypeId().getNativeId()) return Protobuf.WT_32BIT;
    if (typeId == DType.FLOAT64.getTypeId().getNativeId()) return Protobuf.WT_64BIT;
    if (typeId == DType.STRING.getTypeId().getNativeId()) return Protobuf.WT_LEN;
    if (typeId == DType.LIST.getTypeId().getNativeId()) return Protobuf.WT_LEN;
    if (typeId == DType.STRUCT.getTypeId().getNativeId()) return Protobuf.WT_LEN;
    if (encoding == Protobuf.ENC_FIXED) {
      if (typeId == DType.INT64.getTypeId().getNativeId()
          || typeId == DType.UINT64.getTypeId().getNativeId()) return Protobuf.WT_64BIT;
      return Protobuf.WT_32BIT;
    }
    return Protobuf.WT_VARINT;
  }
}

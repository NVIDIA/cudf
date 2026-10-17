/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class ProtobufSchemaDescriptorTest {
  private ProtobufSchemaDescriptor makeDescriptor(
      boolean isRepeated,
      boolean hasDefaultValue,
      int encoding,
      int[] enumValidValues,
      byte[][] enumNames) {
    DType outputType = (encoding == Protobuf.ENC_ENUM_STRING) ? DType.STRING : DType.INT32;
    return new ProtobufSchemaDescriptorBuilder()
        .addField(1, outputType).encoding(encoding)
            .repeated(isRepeated).hasDefault(hasDefaultValue)
            .enumValidValues(enumValidValues).enumNames(enumNames)
        .build();
  }

  @Test
  void testRepeatedFieldCannotCarryDefaultValue() {
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(true, true, Protobuf.ENC_DEFAULT, null, null));
  }

  @Test
  void testFieldCannotBeBothRepeatedAndRequired() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT32).repeated().required()
            .build());
  }

  @Test
  void testStructFieldCannotCarryDefaultValue() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.STRUCT).hasDefault()
            .build());
  }

  @Test
  void testListFieldCannotCarryDefaultValue() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.LIST).hasDefault()
            .build());
  }

  @Test
  void testEnumStringRequiresEnumMetadata() {
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(false, false, Protobuf.ENC_ENUM_STRING, null, null));
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(false, false, Protobuf.ENC_ENUM_STRING, new int[]{0, 1}, null));
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(false, false, Protobuf.ENC_ENUM_STRING, null,
            new byte[][]{"A".getBytes(), "B".getBytes()}));
  }

  @Test
  void testEnumStringRejectsEmptyEnumArrays() {
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(false, false, Protobuf.ENC_ENUM_STRING, new int[]{}, new byte[][]{}));
  }

  @Test
  void testEnumMetadataRejectsNonEnumTypeOrEncoding() {
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(false, false, Protobuf.ENC_FIXED, new int[]{0, 1}, null));
    assertThrows(IllegalArgumentException.class, () ->
        makeDescriptor(false, false, Protobuf.ENC_ZIGZAG, new int[]{0, 1}, null));
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT64).enumValidValues(new int[]{0, 1})
            .build());
  }

  @Test
  void testEnumDefaultMustBeARecognizedValue() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT32).defaultValue(99)
                .enumValidValues(new int[]{0, 1, 2})
            .build());
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT32).defaultValue(1L << 42)
                .enumValidValues(new int[]{0, 1, 2})
            .build());
    assertDoesNotThrow(() ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT32).defaultValue(2)
                .enumValidValues(new int[]{0, 1, 2})
            .build());
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.STRING)
                .enumMetadata("A", "B")
                .defaultValue(2)
            .build());
  }

  @Test
  void testEnumMetadataRequiresCompatibleOutputType() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT64).enumValidValues(new int[]{0, 1})
            .build());
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT32).wireType(Protobuf.WT_32BIT)
                .encoding(Protobuf.ENC_FIXED).enumValidValues(new int[]{0, 1})
            .build());
    assertDoesNotThrow(() ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT32).enumValidValues(new int[]{0, 1})
            .build());
    assertDoesNotThrow(() ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.STRING)
                .enumMetadata("A", "B")
            .build());
  }

  @Test
  void testDuplicateFieldNumbersUnderSameParentRejected() {
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.STRUCT).down()
                .addField(7, DType.INT32)
                .addField(7, DType.INT32)  // duplicate field number under same parent
            .up()
            .build());
  }

  @Test
  void testDuplicateFieldNumbersUnderDifferentParentsAllowed() {
    assertDoesNotThrow(() ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.STRUCT).down()
                .addField(7, DType.INT32)
            .up()
            .addField(2, DType.STRUCT).down()
                .addField(7, DType.INT32)  // same number, different parents -> allowed
            .up()
            .build());
  }

  @Test
  void testChildParentMustBeStruct() {
    // Field 2 is nested under field 1, but field 1 is INT32 (not STRUCT) -> illegal.
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT32).down()
                .addField(2, DType.INT32)
            .up()
            .build());
  }

  @Test
  void testEncodingCompatibilityValidation() {
    // INT32 with a 32-bit wire type under default (non-fixed) encoding is incompatible.
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.INT32).wireType(Protobuf.WT_32BIT)
            .build());

    // Enum-as-string must use the varint wire type; WT_LEN is incompatible.
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.STRING).wireType(Protobuf.WT_LEN)
                .enumMetadata("A", "B")
            .build());
  }

  @Test
  void testDepthAboveSupportedLimitRejected() {
    // Build a STRUCT chain reaching depth 10 (one past the limit), capped with an INT32 leaf.
    ProtobufSchemaDescriptorBuilder builder = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT);
    for (int i = 1; i <= 9; i++) {
      builder.addField(i + 1, DType.STRUCT).parent(i - 1);
    }
    builder.addField(11, DType.INT32).parent(9);  // depth 10 -> exceeds limit
    assertThrows(IllegalArgumentException.class, builder::build);
  }

  @Test
  void testDownUpNestingMatchesExplicitParent() {
    // A branching tree:
    //   message Outer { int32 a = 1; Mid b = 2; int32 f = 3; }
    //   message Mid   { int32 c = 1; Inner d = 2; }
    //   message Inner { int32 e = 1; }
    ProtobufSchemaDescriptor viaDownUp = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT).down()        // Outer
            .addField(1, DType.INT32)            //   a
            .addField(2, DType.STRUCT).down()    //   b: Mid
                .addField(1, DType.INT32)        //     c
                .addField(2, DType.STRUCT).down()//     d: Inner
                    .addField(1, DType.INT32)    //       e
                .up()
            .up()
            .addField(3, DType.INT32)            //   f
        .up()
        .build();

    ProtobufSchemaDescriptor viaExplicitParent = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRUCT)
        .addField(1, DType.INT32).parent(0)
        .addField(2, DType.STRUCT).parent(0)
        .addField(1, DType.INT32).parent(2)
        .addField(2, DType.STRUCT).parent(2)
        .addField(1, DType.INT32).parent(4)
        .addField(3, DType.INT32).parent(0)
        .build();

    assertArrayEquals(viaExplicitParent.parentIndices, viaDownUp.parentIndices);
    assertArrayEquals(viaExplicitParent.depthLevels, viaDownUp.depthLevels);
    assertArrayEquals(viaExplicitParent.fieldNumbers, viaDownUp.fieldNumbers);
    assertArrayEquals(viaExplicitParent.outputTypeIds, viaDownUp.outputTypeIds);
    // Check the derived flat indices directly.
    assertArrayEquals(new int[]{-1, 0, 0, 2, 2, 4, 0}, viaDownUp.parentIndices);
    assertArrayEquals(new int[]{0, 1, 1, 2, 2, 3, 1}, viaDownUp.depthLevels);
  }

  @Test
  void testUnbalancedNestingRejected() {
    // up() without a matching down()
    assertThrows(IllegalStateException.class, () ->
        new ProtobufSchemaDescriptorBuilder().addField(1, DType.STRUCT).up());
    // down() before any field
    assertThrows(IllegalStateException.class, () ->
        new ProtobufSchemaDescriptorBuilder().down());
    // build() left inside a down() scope (missing up())
    assertThrows(IllegalStateException.class, () ->
        new ProtobufSchemaDescriptorBuilder().addField(1, DType.STRUCT).down().build());
  }

  @Test
  void testSerializationRoundTripPreservesContentsAndIsolation() throws Exception {
    ProtobufSchemaDescriptor original = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.STRING)
            .enumMetadata("A", "B")
            .defaultValue(1)
        .addField(2, DType.STRING).defaultValue("def")
        .addField(3, DType.INT32).defaultValue(7)
        .build();

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
      oos.writeObject(original);
    }

    ProtobufSchemaDescriptor roundTrip;
    try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(baos.toByteArray()))) {
      roundTrip = (ProtobufSchemaDescriptor) ois.readObject();
    }

    assertEquals(original.numFields(), roundTrip.numFields());
    assertArrayEquals(original.fieldNumbers, roundTrip.fieldNumbers);
    assertArrayEquals(original.defaultInts, roundTrip.defaultInts);
    assertArrayEquals(original.defaultStrings[1], roundTrip.defaultStrings[1]);
    assertArrayEquals(original.enumValidValues[0], roundTrip.enumValidValues[0]);
    assertArrayEquals(original.enumNames[0][0], roundTrip.enumNames[0][0]);
    assertArrayEquals(original.enumNames[0][1], roundTrip.enumNames[0][1]);
    assertNotSame(original.defaultStrings, roundTrip.defaultStrings);
    assertNotSame(original.defaultStrings[1], roundTrip.defaultStrings[1]);
    assertNotSame(original.enumValidValues, roundTrip.enumValidValues);
    assertNotSame(original.enumValidValues[0], roundTrip.enumValidValues[0]);
    assertNotSame(original.enumNames, roundTrip.enumNames);
    assertNotSame(original.enumNames[0], roundTrip.enumNames[0]);
    assertNotSame(original.enumNames[0][0], roundTrip.enumNames[0][0]);
  }

  @Test
  void testMismatchedArrayLengthsRejected() {
    // parentIndices has length 2 while every other array has length 1.
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptor(
            new int[]{1},
            new int[]{-1, -1},
            new int[]{0},
            new int[]{Protobuf.WT_VARINT},
            new int[]{DType.INT32.getTypeId().getNativeId()},
            new int[]{Protobuf.ENC_DEFAULT},
            new boolean[]{false},
            new boolean[]{false},
            new boolean[]{false},
            new long[]{0},
            new double[]{0.0},
            new boolean[]{false},
            new byte[][]{null},
            new int[][]{null},
            new byte[][][]{null}));
  }

  @Test
  void testBackCompatConstructorMarksAllFieldsAsOutput() {
    // The 15-arg constructor fills isOutput with all-true.
    ProtobufSchemaDescriptor schema = new ProtobufSchemaDescriptor(
        new int[]{1, 2},
        new int[]{-1, -1},
        new int[]{0, 0},
        new int[]{Protobuf.WT_VARINT, Protobuf.WT_LEN},
        new int[]{DType.INT32.getTypeId().getNativeId(),
                  DType.STRING.getTypeId().getNativeId()},
        new int[]{Protobuf.ENC_DEFAULT, Protobuf.ENC_DEFAULT},
        new boolean[]{false, false},
        new boolean[]{false, false},
        new boolean[]{false, false},
        new long[]{0, 0},
        new double[]{0.0, 0.0},
        new boolean[]{false, false},
        new byte[][]{null, null},
        new int[][]{null, null},
        new byte[][][]{null, null});
    assertArrayEquals(new boolean[]{true, true}, schema.isOutput);
  }

  @Test
  void testNestedFieldMustShareOutputFlagWithParent() {
    // Hidden parent struct with a visible child -> illegal.
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.STRUCT).isOutput(false).down()
                .addField(1, DType.INT32).isOutput(true)
            .up()
            .build());

    // Reverse direction: visible parent struct with a hidden child -> also illegal.
    assertThrows(IllegalArgumentException.class, () ->
        new ProtobufSchemaDescriptorBuilder()
            .addField(1, DType.STRUCT).isOutput(true).down()
                .addField(1, DType.INT32).isOutput(false)
            .up()
            .build());
  }

  @Test
  void testHiddenFieldRoundTripsThroughSerialization() throws Exception {
    ProtobufSchemaDescriptor original = new ProtobufSchemaDescriptorBuilder()
        .addField(1, DType.INT32)
        .addField(2, DType.INT32).isOutput(false)  // second field hidden
        .build();

    ByteArrayOutputStream baos = new ByteArrayOutputStream();
    try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
      oos.writeObject(original);
    }
    ProtobufSchemaDescriptor roundTrip;
    try (ObjectInputStream ois = new ObjectInputStream(new ByteArrayInputStream(baos.toByteArray()))) {
      roundTrip = (ProtobufSchemaDescriptor) ois.readObject();
    }
    assertArrayEquals(original.isOutput, roundTrip.isOutput);
  }
}

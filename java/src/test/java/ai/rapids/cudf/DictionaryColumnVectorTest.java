/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.apache.arrow.memory.BufferAllocator;
import org.apache.arrow.memory.RootAllocator;
import org.apache.arrow.vector.FieldVector;
import org.apache.arrow.vector.IntVector;
import org.apache.arrow.vector.VectorSchemaRoot;
import org.apache.arrow.vector.dictionary.Dictionary;
import org.apache.arrow.vector.dictionary.DictionaryEncoder;
import org.apache.arrow.vector.dictionary.DictionaryProvider;
import org.apache.arrow.vector.ipc.ArrowStreamWriter;
import org.apache.arrow.vector.types.pojo.DictionaryEncoding;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static org.junit.jupiter.api.Assertions.*;

public class DictionaryColumnVectorTest extends CudfTestBase {
  // 4 distinct non-null values ("b", "a", "", "c") and 2 nulls.
  private static final String[] VALUES = {"b", null, "a", "", "b", "c", null, "a"};

  @Test
  void testEncodeDecodeRoundTrip() {
    try (ColumnVector input = ColumnVector.fromStrings(VALUES);
         ColumnVector dict = input.dictionaryEncode();
         ColumnView indices = dict.getDictionaryIndices();
         ColumnView keys = dict.getDictionaryKeys();
         ColumnVector decoded = dict.dictionaryDecode()) {
      DType type = dict.getType();
      assertEquals(DType.DICTIONARY32, type);
      assertTrue(type.hasChildren());
      assertFalse(type.isNestedType());
      assertFalse(type.hasOffsets());
      assertEquals(0, type.getSizeInBytes());
      assertEquals(VALUES.length, dict.getRowCount());
      assertEquals(2, dict.getNullCount());
      assertEquals(2, dict.getNumChildren());
      try (BaseDeviceMemoryBuffer data = dict.getData()) {
        assertNull(data);
      }
      assertEquals(DType.INT32, indices.getType());
      assertEquals(VALUES.length, indices.getRowCount());
      assertEquals(DType.STRING, keys.getType());
      assertEquals(4, keys.getRowCount());
      assertEquals(0, keys.getNullCount());
      assertColumnsAreEqual(input, decoded);
    }
  }

  @Test
  void testEncodeAndDecodeRejectWrongTypes() {
    try (ColumnVector ints = ColumnVector.fromInts(1, 2);
         ColumnVector strings = ColumnVector.fromStrings("a")) {
      assertThrows(IllegalArgumentException.class, () -> ints.dictionaryEncode().close());
      assertThrows(IllegalArgumentException.class, () -> strings.dictionaryDecode().close());
      assertThrows(IllegalArgumentException.class, () -> strings.getDictionaryKeys().close());
    }
  }

  @Test
  void testHostRoundTrip() {
    try (ColumnVector input = ColumnVector.fromStrings(VALUES);
         ColumnVector dict = input.dictionaryEncode();
         HostColumnVector host = dict.copyToHost()) {
      assertEquals(DType.DICTIONARY32, host.getType());
      assertEquals(2, host.getNumChildren());
      for (int i = 0; i < VALUES.length; i++) {
        assertEquals(VALUES[i], host.getElement(i), "row " + i);
      }
      try (ColumnVector back = host.copyToDevice();
           ColumnVector decoded = back.dictionaryDecode()) {
        assertEquals(DType.DICTIONARY32, back.getType());
        assertColumnsAreEqual(input, decoded);
      }
    }
  }

  @Test
  void testMakeDictionaryWithEachIndexWidth() {
    String[] expectedValues = {"c", null, "a", "c"};
    List<ColumnVector> indexColumns = new ArrayList<>();
    try (ColumnVector keys = ColumnVector.fromStrings("a", "b", "c");
         ColumnVector expected = ColumnVector.fromStrings(expectedValues)) {
      indexColumns.add(ColumnVector.fromBoxedBytes((byte) 2, null, (byte) 0, (byte) 2));
      indexColumns.add(ColumnVector.fromBoxedShorts((short) 2, null, (short) 0, (short) 2));
      indexColumns.add(ColumnVector.fromBoxedInts(2, null, 0, 2));
      indexColumns.add(ColumnVector.fromBoxedLongs(2L, null, 0L, 2L));
      for (ColumnVector indices : indexColumns) {
        try (ColumnVector dict = ColumnVector.makeDictionary(keys, indices);
             ColumnView dictIndices = dict.getDictionaryIndices();
             ColumnVector decoded = dict.dictionaryDecode();
             HostColumnVector host = dict.copyToHost();
             ColumnVector back = host.copyToDevice();
             ColumnVector backDecoded = back.dictionaryDecode()) {
          assertEquals(indices.getType(), dictIndices.getType());
          assertEquals(1, dict.getNullCount());
          assertColumnsAreEqual(expected, decoded);
          for (int i = 0; i < expectedValues.length; i++) {
            assertEquals(expectedValues[i], host.getElement(i),
                indices.getType() + " row " + i);
          }
          assertColumnsAreEqual(expected, backDecoded);
        }
      }
    } finally {
      for (ColumnVector cv : indexColumns) {
        cv.close();
      }
    }
  }

  @Test
  void testMakeDictionaryErrors() {
    try (ColumnVector keysWithNull = ColumnVector.fromStrings("a", null);
         ColumnVector intKeys = ColumnVector.fromInts(1, 2);
         ColumnVector keys = ColumnVector.fromStrings("a", "b");
         ColumnVector indices = ColumnVector.fromInts(0, 1);
         ColumnVector unsignedIndices = ColumnVector.fromUnsignedInts(0, 1)) {
      assertThrows(CudfException.class,
          () -> ColumnVector.makeDictionary(keysWithNull, indices).close());
      assertThrows(CudfException.class,
          () -> ColumnVector.makeDictionary(keys, unsignedIndices).close());
      assertThrows(IllegalArgumentException.class,
          () -> ColumnVector.makeDictionary(intKeys, indices).close());
    }
  }

  @Test
  void testEmptyDictionaries() {
    try (ColumnVector childless = ColumnVector.empty(
             new HostColumnVector.BasicType(true, DType.DICTIONARY32));
         ColumnVector emptyStrings = ColumnVector.fromStrings();
         ColumnVector withChildren = emptyStrings.dictionaryEncode()) {
      assertEquals(0, childless.getNumChildren());
      assertEquals(2, withChildren.getNumChildren());
      assertThrows(IllegalStateException.class, () -> childless.getDictionaryKeys().close());
      for (ColumnVector dict : Arrays.asList(childless, withChildren)) {
        assertEquals(DType.DICTIONARY32, dict.getType());
        assertEquals(0, dict.getRowCount());
        try (ColumnVector decoded = dict.dictionaryDecode();
             HostColumnVector host = dict.copyToHost();
             ColumnVector back = host.copyToDevice()) {
          assertEquals(DType.STRING, decoded.getType());
          assertEquals(0, decoded.getRowCount());
          assertEquals(DType.DICTIONARY32, back.getType());
          assertEquals(0, back.getRowCount());
          assertEquals(dict.getNumChildren(), back.getNumChildren());
        }
      }
    }
  }

  @Test
  void testSlicedDictionary() {
    try (ColumnVector input = ColumnVector.fromStrings(VALUES);
         ColumnVector dict = input.dictionaryEncode()) {
      ColumnView[] parts = dict.splitAsViews(3);
      ColumnVector[] expectedIndices = null;
      try (ColumnView fullIndices = dict.getDictionaryIndices();
           ColumnView tail = parts[1];
           ColumnView tailIndices = tail.getDictionaryIndices();
           ColumnView keys = dict.getDictionaryKeys();
           ColumnVector tailIndicesCopy = tailIndices.copyToColumnVector();
           ColumnVector tailCopy = tail.copyToColumnVector();
           ColumnVector decodedView = tail.dictionaryDecode();
           ColumnVector decodedCopy = tailCopy.dictionaryDecode();
           ColumnVector expected = ColumnVector.fromStrings(
               Arrays.copyOfRange(VALUES, 3, VALUES.length))) {
        expectedIndices = fullIndices.slice(3, VALUES.length);
        int tailRows = VALUES.length - 3;
        // The indices child is exposed for the slice's rows only.
        assertEquals(tailRows, tailIndices.getRowCount());
        assertColumnsAreEqual(expectedIndices[0], tailIndicesCopy);
        assertColumnsAreEqual(expected, decodedView);
        assertColumnsAreEqual(expected, decodedCopy);
        // Device memory counts the slice's indices, not the whole indices child.
        long expectedSize = ColumnView.getValidityBufferSize(tailRows) +
            tailRows * DType.INT32.getSizeInBytes() + keys.getDeviceMemorySize();
        assertEquals(expectedSize, tail.getDeviceMemorySize());
      } finally {
        parts[0].close();
        if (expectedIndices != null) {
          for (ColumnVector cv : expectedIndices) {
            cv.close();
          }
        }
      }
    }
  }

  @Test
  void testConcatenateDifferentKeys() {
    try (ColumnVector a = ColumnVector.fromStrings("x", "y", null);
         ColumnVector b = ColumnVector.fromStrings("z", "y");
         ColumnVector da = a.dictionaryEncode();
         ColumnVector db = b.dictionaryEncode();
         ColumnVector concat = ColumnVector.concatenate(da, db);
         ColumnVector decoded = concat.dictionaryDecode();
         ColumnVector expected = ColumnVector.fromStrings("x", "y", null, "z", "y")) {
      assertEquals(DType.DICTIONARY32, concat.getType());
      assertColumnsAreEqual(expected, decoded);
    }
  }

  @Test
  void testOrderByAndContiguousSplit() {
    try (ColumnVector strings = ColumnVector.fromStrings(VALUES);
         ColumnVector dict = strings.dictionaryEncode();
         ColumnVector rowIds = ColumnVector.fromInts(0, 1, 2, 3, 4, 5, 6, 7);
         Table plain = new Table(strings, rowIds);
         Table encoded = new Table(dict, rowIds);
         Table expectedSorted = plain.orderBy(OrderByArg.asc(0, true), OrderByArg.asc(1));
         Table sorted = encoded.orderBy(OrderByArg.asc(0, true), OrderByArg.asc(1));
         ColumnVector sortedDecoded = sorted.getColumn(0).dictionaryDecode()) {
      // Ordered by string value, not by index.
      assertColumnsAreEqual(expectedSorted.getColumn(0), sortedDecoded);
      assertColumnsAreEqual(expectedSorted.getColumn(1), sorted.getColumn(1));

      ContiguousTable[] parts = encoded.contiguousSplit(3);
      try {
        int start = 0;
        for (ContiguousTable part : parts) {
          Table t = part.getTable();
          int end = start + (int) t.getRowCount();
          try (ColumnVector decoded = t.getColumn(0).dictionaryDecode();
               ColumnVector expected = ColumnVector.fromStrings(
                   Arrays.copyOfRange(VALUES, start, end))) {
            assertEquals(DType.DICTIONARY32, t.getColumn(0).getType());
            assertColumnsAreEqual(expected, decoded);
          }
          start = end;
        }
        assertEquals(VALUES.length, start);
      } finally {
        for (ContiguousTable part : parts) {
          part.close();
        }
      }
    }
  }

  @Test
  void testArrowIPCRoundTrip() throws IOException {
    File file = File.createTempFile("dictionary", ".arrow");
    file.deleteOnExit();
    String[] first = {"b", null, "a", "b"};
    String[] second = {"c", "c", null};
    try (ColumnVector s1 = ColumnVector.fromStrings(first);
         ColumnVector s2 = ColumnVector.fromStrings(second);
         ColumnVector d1 = s1.dictionaryEncode();
         ColumnVector d2 = s2.dictionaryEncode();
         Table t1 = new Table(d1);
         Table t2 = new Table(d2)) {
      ArrowIPCWriterOptions options = ArrowIPCWriterOptions.builder()
          .withColumnNames("d")
          .build();
      try (TableWriter writer = Table.writeArrowIPCChunked(options, file)) {
        writer.write(t1);
        writer.write(t2);
      }
    }
    List<String> actual = new ArrayList<>();
    try (StreamedTableReader reader = Table.readArrowIPCChunked(file)) {
      Table t;
      while ((t = reader.getNextIfAvailable()) != null) {
        try (Table read = t;
             ColumnVector decoded = read.getColumn(0).dictionaryDecode();
             HostColumnVector host = decoded.copyToHost()) {
          assertEquals(DType.DICTIONARY32, read.getColumn(0).getType());
          for (int i = 0; i < host.getRowCount(); i++) {
            actual.add(host.isNull(i) ? null : host.getJavaString(i));
          }
        }
      }
    }
    List<String> expectedValues = new ArrayList<>(Arrays.asList("b", null, "a", "b"));
    expectedValues.addAll(Arrays.asList("c", "c", null));
    assertEquals(expectedValues, actual);
  }

  @Test
  void testArrowIPCReadRejectsNonStringDictionary() throws IOException {
    File file = File.createTempFile("int-dictionary", ".arrow");
    file.deleteOnExit();
    try (BufferAllocator allocator = new RootAllocator(Long.MAX_VALUE);
         IntVector dictionaryValues = new IntVector("dictionary", allocator);
         IntVector values = new IntVector("values", allocator)) {
      dictionaryValues.allocateNew(2);
      dictionaryValues.set(0, 10);
      dictionaryValues.set(1, 20);
      dictionaryValues.setValueCount(2);
      values.allocateNew(3);
      values.set(0, 20);
      values.set(1, 10);
      values.set(2, 20);
      values.setValueCount(3);
      Dictionary dictionary =
          new Dictionary(dictionaryValues, new DictionaryEncoding(1L, false, null));
      DictionaryProvider provider = new DictionaryProvider.MapDictionaryProvider(dictionary);
      try (FieldVector encoded = (FieldVector) DictionaryEncoder.encode(values, dictionary)) {
        VectorSchemaRoot root = new VectorSchemaRoot(
            Collections.singletonList(encoded.getField()), Collections.singletonList(encoded),
            encoded.getValueCount());
        try (OutputStream out = new FileOutputStream(file);
             ArrowStreamWriter writer = new ArrowStreamWriter(root, provider, out)) {
          writer.start();
          writer.writeBatch();
          writer.end();
        }
      }
    }
    try (StreamedTableReader reader = Table.readArrowIPCChunked(file)) {
      assertThrows(IllegalArgumentException.class, () -> {
        try (Table t = reader.getNextIfAvailable()) {
          // only reached if the read did not throw
        }
      });
    }
  }

  @Test
  void testFileWritersRejectDictionaries() {
    try (ColumnVector strings = ColumnVector.fromStrings(VALUES);
         ColumnVector dict = strings.dictionaryEncode();
         ColumnVector ints = ColumnVector.fromInts(0, 1, 2, 3, 4, 5, 6, 7);
         ColumnVector struct = ColumnVector.makeStruct(ints, dict);
         Table dictTable = new Table(dict);
         Table structTable = new Table(struct)) {
      ParquetWriterOptions parquetOptions = ParquetWriterOptions.builder()
          .withColumns(true, "d")
          .build();
      try (TableWriter writer =
               Table.writeParquetChunked(parquetOptions, new File("/dev/null"))) {
        assertThrows(IllegalArgumentException.class, () -> writer.write(dictTable));
      }
      ORCWriterOptions orcOptions = ORCWriterOptions.builder()
          .withColumns(true, "d")
          .build();
      try (TableWriter writer = Table.writeORCChunked(orcOptions, new File("/dev/null"))) {
        assertThrows(IllegalArgumentException.class, () -> writer.write(dictTable));
      }
      CSVWriterOptions csvOptions = CSVWriterOptions.builder()
          .withColumnNames("d")
          .build();
      assertThrows(IllegalArgumentException.class,
          () -> dictTable.writeCSVToFile(csvOptions, "/dev/null"));
      // A dictionary nested in a struct is rejected too.
      ParquetWriterOptions structOptions = ParquetWriterOptions.builder()
          .withStructColumn(ColumnWriterOptions.structBuilder("s", true)
              .withColumns(true, "i", "d")
              .build())
          .build();
      try (TableWriter writer =
               Table.writeParquetChunked(structOptions, new File("/dev/null"))) {
        assertThrows(IllegalArgumentException.class, () -> writer.write(structTable));
      }
    }
  }

  @Test
  void testJCudfSerializationRejectsDictionaries() {
    try (ColumnVector strings = ColumnVector.fromStrings(VALUES);
         ColumnVector dict = strings.dictionaryEncode();
         ColumnVector ints = ColumnVector.fromInts(0, 1, 2, 3, 4, 5, 6, 7);
         ColumnVector struct = ColumnVector.makeStruct(ints, dict)) {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      assertThrows(UnsupportedOperationException.class, () -> JCudfSerialization.writeToStream(
          new ColumnVector[]{dict}, out, 0, VALUES.length));
      assertThrows(UnsupportedOperationException.class, () -> JCudfSerialization.writeToStream(
          new ColumnVector[]{struct}, out, 0, VALUES.length));
    }
  }
}

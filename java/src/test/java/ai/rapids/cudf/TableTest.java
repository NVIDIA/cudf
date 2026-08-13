/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import ai.rapids.cudf.HostColumnVector.BasicType;
import ai.rapids.cudf.HostColumnVector.Builder;
import ai.rapids.cudf.HostColumnVector.DataType;
import ai.rapids.cudf.HostColumnVector.ListType;
import ai.rapids.cudf.HostColumnVector.StructData;
import ai.rapids.cudf.HostColumnVector.StructType;

import ai.rapids.cudf.ast.BinaryOperation;
import ai.rapids.cudf.ast.BinaryOperator;
import ai.rapids.cudf.ast.ColumnReference;
import ai.rapids.cudf.ast.CompiledExpression;
import ai.rapids.cudf.ast.TableReference;
import com.google.common.base.Charsets;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.parquet.hadoop.ParquetFileReader;
import org.apache.parquet.hadoop.util.HadoopInputFile;
import org.apache.parquet.schema.GroupType;
import org.apache.parquet.schema.MessageType;
import org.apache.parquet.schema.OriginalType;
import org.apache.parquet.schema.Type;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.io.*;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.math.RoundingMode;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.channels.SeekableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardOpenOption;
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

import static ai.rapids.cudf.AssertUtils.assertColumnsAreEqual;
import static ai.rapids.cudf.AssertUtils.assertPartialColumnsAreEqual;
import static ai.rapids.cudf.AssertUtils.assertPartialTablesAreEqual;
import static ai.rapids.cudf.AssertUtils.assertTableTypes;
import static ai.rapids.cudf.AssertUtils.assertTablesAreEqual;
import static ai.rapids.cudf.ColumnWriterOptions.mapColumn;
import static ai.rapids.cudf.ParquetWriterOptions.listBuilder;
import static ai.rapids.cudf.ParquetWriterOptions.structBuilder;
import static ai.rapids.cudf.Table.TestBuilder;
import static ai.rapids.cudf.Table.removeNullMasksIfNeeded;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class TableTest extends CudfTestBase {

  private static final HostMemoryAllocator hostMemoryAllocator = DefaultHostMemoryAllocator.get();

  private static final File TEST_PARQUET_FILE = TestUtils.getResourceAsFile("acq.parquet");
  private static final File TEST_PARQUET_FILE_CHUNKED_READ = TestUtils.getResourceAsFile("splittable.parquet");
  private static final File TEST_PARQUET_FILE_BINARY = TestUtils.getResourceAsFile("binary.parquet");
  private static final File TEST_ORC_FILE = TestUtils.getResourceAsFile("TestOrcFile.orc");
  private static final File TEST_ORC_FILE_CHUNKED_READ = TestUtils.getResourceAsFile("splittable.orc");
  private static final File TEST_ORC_TIMESTAMP_DATE_FILE = TestUtils.getResourceAsFile("timestamp-date-test.orc");
  private static final File TEST_DECIMAL_PARQUET_FILE = TestUtils.getResourceAsFile("decimal.parquet");
  private static final File TEST_ALL_TYPES_PLAIN_AVRO_FILE = TestUtils.getResourceAsFile("alltypes_plain.avro");
  private static final File TEST_SIMPLE_CSV_FILE = TestUtils.getResourceAsFile("simple.csv");
  private static final File TEST_SIMPLE_JSON_FILE = TestUtils.getResourceAsFile("people.json");
  private static final File TEST_JSON_ERROR_FILE = TestUtils.getResourceAsFile("people_with_invalid_lines.json");
  private static final File TEST_JSON_SINGLE_QUOTES_FILE = TestUtils.getResourceAsFile("single_quotes.json");
  private static final File TEST_JSON_WHITESPACES_FILE = TestUtils.getResourceAsFile("whitespaces.json");
  private static final File TEST_MIXED_TYPE_1_JSON = TestUtils.getResourceAsFile("mixed_types_1.json");
  private static final File TEST_MIXED_TYPE_2_JSON = TestUtils.getResourceAsFile("mixed_types_2.json");

  private static class TempFile implements AutoCloseable {
    private final File file;

    private TempFile(File file) {
      this.file = file;
    }

    static TempFile create(String prefix, String suffix) throws IOException {
      File file = File.createTempFile(prefix, suffix);
      file.deleteOnExit();
      return new TempFile(file);
    }

    File getFile() {
      return file;
    }

    @Override
    public void close() {
      file.delete();
    }
  }

  private static final Schema CSV_DATA_BUFFER_SCHEMA = Schema.builder()
      .column(DType.INT32, "A")
      .column(DType.FLOAT64, "B")
      .column(DType.INT64, "C")
      .build();

  private static final byte[] CSV_DATA_BUFFER = ("A|B|C\n" +
      "'0'|'110.0'|'120'\n" +
      "1|111.0|121\n" +
      "2|112.0|122\n" +
      "3|113.0|123\n" +
      "4|114.0|124\n" +
      "5|115.0|125\n" +
      "6|116.0|126\n" +
      "7|NULL|127\n" +
      "8|118.2|128\n" +
      "9|119.8|129").getBytes(StandardCharsets.UTF_8);

  void assertTablesHaveSameValues(HashMap<Object, Integer>[] expectedTable, Table table) {
    assertEquals(expectedTable.length, table.getNumberOfColumns());
    int numCols = table.getNumberOfColumns();
    long numRows = table.getRowCount();
    for (int col = 0; col < numCols; col++) {
      for (long row = 0; row < numRows; row++) {
        try (HostColumnVector cv = table.getColumn(col).copyToHost()) {
          Object key = 0;
          if (cv.getType().equals(DType.INT32)) {
            key = cv.getInt(row);
          } else {
            key = cv.getDouble(row);
          }
          assertTrue(expectedTable[col].containsKey(key));
          Integer count = expectedTable[col].get(key);
          if (count == 1) {
            expectedTable[col].remove(key);
          } else {
            expectedTable[col].put(key, count - 1);
          }
        }
      }
    }
    for (int i = 0 ; i < expectedTable.length ; i++) {
      assertTrue(expectedTable[i].isEmpty());
    }
  }

  @Test
  void testDistinctCount() {
    try (Table table1 = new Table.TestBuilder()
            .column(5, 3, null, null, 5)
            .build()) {
      assertEquals(3, table1.distinctCount());
      assertEquals(4, table1.distinctCount(NullEquality.UNEQUAL));
    }
  }

  @Test
  void testMergeSimple() {
    try (Table table1 = new Table.TestBuilder()
            .column(5, 3, 3, 1, 1)
            .column(5, 3, null, 1, 2)
            .column(1, 3, 5, 7, 9)
            .build();
         Table table2 = new Table.TestBuilder()
                 .column(1, 2, 7)
                 .column(3, 2, 2)
                 .column(1, 3, 10)
                 .build();
         Table expected = new Table.TestBuilder()
                 .column(1, 1, 1, 2, 3, 3, 5, 7)
                 .column(3, 2, 1, 2, null, 3, 5, 2)
                 .column(1, 9, 7, 3, 5, 3, 1, 10)
                 .build();
         Table sortedTable1 = table1.orderBy(OrderByArg.asc(0), OrderByArg.desc(1));
         Table sortedTable2 = table2.orderBy(OrderByArg.asc(0), OrderByArg.desc(1));
         Table merged = Table.merge(Arrays.asList(sortedTable1, sortedTable2), OrderByArg.asc(0), OrderByArg.desc(1))) {
      assertTablesAreEqual(expected, merged);
    }
  }

  @Test
  void testOrderByAD() {
    try (Table table = new Table.TestBuilder()
        .column(5, 3, 3, 1, 1)
        .column(5, 3, 4, 1, 2)
        .column(1, 3, 5, 7, 9)
        .build();
         Table expected = new Table.TestBuilder()
             .column(1, 1, 3, 3, 5)
             .column(2, 1, 4, 3, 5)
             .column(9, 7, 5, 3, 1)
             .build();
         Table sortedTable = table.orderBy(OrderByArg.asc(0), OrderByArg.desc(1))) {
      assertTablesAreEqual(expected, sortedTable);
    }
  }

  @Test
  void testSortOrderSimple() {
    try (Table table = new Table.TestBuilder()
        .column(5, 3, 3, 1, 1)
        .column(5, 3, 4, 1, 2)
        .column(1, 3, 5, 7, 9)
        .build();
         Table expected = new Table.TestBuilder()
             .column(1, 1, 3, 3, 5)
             .column(2, 1, 4, 3, 5)
             .column(9, 7, 5, 3, 1)
             .build();
         ColumnVector gatherMap = table.sortOrder(OrderByArg.asc(0), OrderByArg.desc(1));
         Table sortedTable = table.gather(gatherMap)) {
      assertTablesAreEqual(expected, sortedTable);
    }
  }

  @Test
  void testOrderByDD() {
    try (Table table = new Table.TestBuilder()
        .column(5, 3, 3, 1, 1)
        .column(5, 3, 4, 1, 2)
        .column(1, 3, 5, 7, 9)
        .build();
         Table expected = new Table.TestBuilder()
             .column(5, 3, 3, 1, 1)
             .column(5, 4, 3, 2, 1)
             .column(1, 5, 3, 9, 7)
             .build();
         Table sortedTable = table.orderBy(OrderByArg.desc(0), OrderByArg.desc(1))) {
      assertTablesAreEqual(expected, sortedTable);
    }
  }

  @Test
  void testOrderByWithNulls() {
    try (Table table = new Table.TestBuilder()
        .column(5, null, 3, 1, 1)
        .column(5, 3, 4, null, null)
        .column("4", "3", "2", "1", "0")
        .column(1, 3, 5, 7, 9)
        .build();
         Table expected = new Table.TestBuilder()
             .column(1, 1, 3, 5, null)
             .column(null, null, 4, 5, 3)
             .column("1", "0", "2", "4", "3")
             .column(7, 9, 5, 1, 3)
             .build();
         Table sortedTable = table.orderBy(OrderByArg.asc(0), OrderByArg.desc(1))) {
      assertTablesAreEqual(expected, sortedTable);
    }
  }

  @Test
  void testOrderByWithNullsAndStrings() {
    try (Table table = new Table.TestBuilder()
        .column("4", "3", "2", "1", "0")
        .column(5, null, 3, 1, 1)
        .column(5, 3, 4, null, null)
        .column(1, 3, 5, 7, 9)
        .build();
         Table expected = new Table.TestBuilder()
             .column("0", "1", "2", "3", "4")
             .column(1, 1, 3, null, 5)
             .column(null, null, 4, 3, 5)
             .column(9, 7, 5, 3, 1)
             .build();
         Table sortedTable = table.orderBy(OrderByArg.asc(0))) {
      assertTablesAreEqual(expected, sortedTable);
    }
  }

  @Test
  void testTableCreationIncreasesRefCountWithDoubleFree() {
    //tests the Table increases the refcount on column vectors
    assertThrows(IllegalStateException.class, () -> {
      try (ColumnVector v1 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
           ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5))) {
        assertDoesNotThrow(() -> {
          try (Table t = new Table(new ColumnVector[]{v1, v2})) {
            v1.close();
            v2.close();
          }
        });
      }
    });
  }

  @Test
  void testGetRows() {
    try (ColumnVector v1 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
         ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
         Table t = new Table(new ColumnVector[]{v1, v2})) {
      assertEquals(5, t.getRowCount());
    }
  }

  @Test
  void testSettingNullVectors() {
    ColumnVector[] columnVectors = null;
    assertThrows(AssertionError.class, () -> new Table(columnVectors));
  }

  @Test
  void testAllRowsSize() {
    try (ColumnVector v1 = ColumnVector.build(DType.INT32, 4, Range.appendInts(4));
         ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5))) {
      assertThrows(AssertionError.class, () -> {
        try (Table t = new Table(new ColumnVector[]{v1, v2})) {
        }
      });
    }
  }

  @Test
  void testGetNumberOfColumns() {
    try (ColumnVector v1 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
         ColumnVector v2 = ColumnVector.build(DType.INT32, 5, Range.appendInts(5));
         Table t = new Table(new ColumnVector[]{v1, v2})) {
      assertEquals(2, t.getNumberOfColumns());
    }
  }

  @Test
  void testReadJSONFile() {
    Schema schema = Schema.builder()
        .column(DType.STRING, "name")
        .column(DType.INT32, "age")
        .build();
    JSONOptions opts = JSONOptions.builder()
        .withLines(true)
        .build();
    try (Table expected = new Table.TestBuilder()
        .column("Michael", "Andy", "Justin")
        .column(null, 30, 19)
        .build();
        Table table = Table.readJSON(schema, opts, TEST_SIMPLE_JSON_FILE)) {
      assertTablesAreEqual(expected, table);
    }
  }

  private static final byte[] EMPTY_JSON_DATA_BUFFER = ("{}\n").getBytes(StandardCharsets.UTF_8);

  @Test
  void testReadEmptyJson() {
    Schema schema = Schema.builder()
        .column(DType.STRING, "name")
        .build();
    JSONOptions opts = JSONOptions.builder()
        .withKeepQuotes(true)
        .withRecoverWithNull(true)
        .withMixedTypesAsStrings(true)
        .withNormalizeSingleQuotes(true)
        .withNormalizeWhitespace(true)
        .withLines(true)
        .build();
    try (Table expected = new Table.TestBuilder()
        .column((String)null)
        .build();
         Table table = Table.readJSON(schema, opts, EMPTY_JSON_DATA_BUFFER, 0,
             EMPTY_JSON_DATA_BUFFER.length, 1)) {
      assertTablesAreEqual(expected, table);
    }
  }

  private static final byte[] EMPTY_ARRAY_JSON_DATA_BUFFER =
      ("{'a':[]}\n").getBytes(StandardCharsets.UTF_8);

  @Test
  void testReadEmptyArrayJson() {
    Schema.Builder builder = Schema.builder();
    Schema.Builder listBuilder = builder.addColumn(DType.LIST, "a");
    // INT8 is selected here because CUDF always returns INT8 for this no matter what we ask for.
    listBuilder.addColumn(DType.INT8, "child");
    Schema schema = builder.build();
    JSONOptions opts = JSONOptions.builder()
        .withKeepQuotes(true)
        .withRecoverWithNull(true)
        .withMixedTypesAsStrings(true)
        .withNormalizeSingleQuotes(true)
        .withNormalizeWhitespace(true)
        .withLines(true)
        .build();
    ListType lt = new ListType(true, new BasicType(true, DType.INT8));
    try (Table expected = new Table.TestBuilder()
        .column(lt, new ArrayList<Byte>())
        .build();
         Table table = Table.readJSON(schema, opts, EMPTY_ARRAY_JSON_DATA_BUFFER, 0,
             EMPTY_ARRAY_JSON_DATA_BUFFER.length, 1)) {
      TableDebug.get().debug("OUTPUT", table);
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadSingleQuotesJSONFile() throws IOException {
    Schema schema = Schema.builder()
            .column(DType.STRING, "A")
            .build();
    JSONOptions opts = JSONOptions.builder()
            .withLines(true)
            .withNormalizeSingleQuotes(true)
            .build();
    try (Table expected = new Table.TestBuilder()
            .column("TEST\"", "TESTER'")
            .build();
         MultiBufferDataSource source = sourceFrom(TEST_JSON_SINGLE_QUOTES_FILE);
         Table table = Table.readJSON(schema, opts, source)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadSingleQuotesJSONFileFeatureDisabled() throws IOException {
    Schema schema = Schema.builder()
      .column(DType.STRING, "A")
      .build();
    JSONOptions opts = JSONOptions.builder()
      .withLines(true)
      .withNormalizeSingleQuotes(false)
      .build();
    try (MultiBufferDataSource source = sourceFrom(TEST_JSON_SINGLE_QUOTES_FILE)) {
      assertThrows(CudfException.class, () ->
        Table.readJSON(schema, opts, source));
    }
  }

  @Test
  void testReadWhitespacesJSONFile() throws IOException {
    Schema schema = Schema.builder()
            .column(DType.STRING, "a")
            .build();
    JSONOptions opts = JSONOptions.builder()
            .withLines(true)
            .withMixedTypesAsStrings(true)
            .withNormalizeWhitespace(true)
            .build();
    try (Table expected = new Table.TestBuilder()
            .column("b", "50", "[1,2,3,4,5,6,7,8]", "{\"c\":\"d\"}", "b")
            .build();
         MultiBufferDataSource source = sourceFrom(TEST_JSON_WHITESPACES_FILE);
         Table table = Table.readJSON(schema, opts, source)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadSingleQuotesJSONFileKeepQuotes() throws IOException {
    Schema schema = Schema.builder()
        .column(DType.STRING, "A")
        .build();
    JSONOptions opts = JSONOptions.builder()
        .withLines(true)
        .withNormalizeSingleQuotes(true)
        .withKeepQuotes(true)
        .build();
    try (Table expected = new Table.TestBuilder()
        .column("\"TEST\"\"", "\"TESTER'\"") // Note that escapes are also processed
        .build();
         MultiBufferDataSource source = sourceFrom(TEST_JSON_SINGLE_QUOTES_FILE);
         Table table = Table.readJSON(schema, opts, source)) {
      assertTablesAreEqual(expected, table);
    }
  }

  private static final byte[] JSON_VALIDATION_BUFFER = (
      "{\"a\":true}\n" +
      "{\"a\":false}\n" +
      "{\"a\":null}\n" +
      "{\"a\":true, \"b\":truee}\n" +
      "{\"a\":true, \"b\":\"nulll\"}\n" +
      "{\"a\": 1}\n" +
      "{\"a\": 0}\n" +
      "{\"a\": -}\n" +
      "{\"a\": -0}\n" +
      "{\"a\": -01}\n" +

      "{\"a\": 01}\n" +
      "{\"a\": -0.1}\n" +
      "{\"a\": -00.1}\n" +
      "{\"a\": NaN}\n" +
      "{\"a\": INF}\n" +
      "{\"a\": +INF}\n" +
      "{\"a\": -INF}\n" +
      "{\"a\": +Infinity}\n" +
      "{\"a\": Infinity}\n" +
      "{\"a\": -Infinity}\n" +

      "{\"a\": INFinity}\n" +
      "{\"a\":\"3710-11-10T02:46:58.732Z\"}\n" +
      "{\"a\":12.}\n" +
      "{\"a\": -3.4e+38}\n" +
      "{\"a\": -3.4e-38}\n" +
      "{\"a\": 1.4e38}\n" +
      "{\"a\": -3.4E+38}\n" +
      "{\"a\": -3.4E-38}\n" +
      "{\"a\": 1.4E38}\n" +
      "{\"a\": -3.4E+}\n" +

      "{\"a\": -3.4E-}\n" +
      "{\"a\": \"A\u0000B\"}\n" +
      "{\"a\": \"A\\u0000B\"}\n" +
      "{\"a\": \"A\u0001B\"}\n" +
      "{\"a\": \"A\\u0001B\"}\n" +
      "{\"a\": \"A\u001FB\"}\n" +
      "{\"a\": \"A\\u001FB\"}\n" +
      "{\"a\": \"A\u0020B\"}\n" +
      "{\"a\": \"A\\u0020B\"}\n" +
      "{\"a\": \"\\u12\"}\n" +

      "{\"a\": \"\\z\"}\n" +
      "{\"a\": \"\\r\"}\n" +
      "{\"a\": \"something\", \"b\": \"\\z\"}\n"
  ).getBytes(StandardCharsets.UTF_8);

  @Test
  void testJSONValidationNoStrict() {
    Schema schema = Schema.builder()
        .column(DType.STRING, "a")
        .build();
    JSONOptions opts = JSONOptions.builder()
        .withRecoverWithNull(true)
        .withMixedTypesAsStrings(true)
        .withNormalizeWhitespace(true)
        .withKeepQuotes(true)
        .withNormalizeSingleQuotes(true)
        .withStrictValidation(false)
        .withLeadingZeros(false)
        .withNonNumericNumbers(false)
        .withUnquotedControlChars(true)
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(
            "true", "false", null, "true", "true", "1", "0", "-", "-0", "-01",
            "01", "-0.1", "-00.1", "NaN", "INF", "+INF", "-INF", "+Infinity", "Infinity", "-Infinity",
            "INFinity", "\"3710-11-10T02:46:58.732Z\"", "12.", "-3.4e+38", "-3.4e-38", "1.4e38", "-3.4E+38", "-3.4E-38", "1.4E38", "-3.4E+",
            "-3.4E-", "\"A\u0000B\"", "\"A\u0000B\"", "\"A\u0001B\"", "\"A\u0001B\"", "\"A\u001FB\"", "\"A\u001FB\"", "\"A B\"", "\"A B\"", null,
            null, "\"\r\"", "\"something\"")
        .build();
         MultiBufferDataSource source = sourceFrom(JSON_VALIDATION_BUFFER);
         Table table = Table.readJSON(schema, opts, source, (int)expected.getRowCount())) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testJSONValidation() {
    Schema schema = Schema.builder()
        .column(DType.STRING, "a")
        .build();
    JSONOptions opts = JSONOptions.builder()
        .withRecoverWithNull(true)
        .withMixedTypesAsStrings(true)
        .withNormalizeWhitespace(true)
        .withKeepQuotes(true)
        .withNormalizeSingleQuotes(true)
        .withStrictValidation(true)
        .withLeadingZeros(false)
        .withNonNumericNumbers(false)
        .withUnquotedControlChars(true)
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(
            "true", "false", null, null, "true", "1", "0", null, "-0", null,
            null, "-0.1", null, null, null, null, null, null, null, null,
            null, "\"3710-11-10T02:46:58.732Z\"", null, "-3.4e+38", "-3.4e-38", "1.4e38", "-3.4E+38", "-3.4E-38", "1.4E38", null,
            null, "\"A\u0000B\"", "\"A\u0000B\"", "\"A\u0001B\"", "\"A\u0001B\"", "\"A\u001FB\"", "\"A\u001FB\"", "\"A B\"", "\"A B\"", null,
            null, "\"\r\"", null)
        .build();
         MultiBufferDataSource source = sourceFrom(JSON_VALIDATION_BUFFER);
         Table table = Table.readJSON(schema, opts, source, (int)expected.getRowCount())) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testJSONValidationLeadingZeros() {
    Schema schema = Schema.builder()
        .column(DType.STRING, "a")
        .build();
    JSONOptions opts = JSONOptions.builder()
        .withRecoverWithNull(true)
        .withMixedTypesAsStrings(true)
        .withNormalizeWhitespace(true)
        .withKeepQuotes(true)
        .withNormalizeSingleQuotes(true)
        .withStrictValidation(true)
        .withLeadingZeros(true)
        .withNonNumericNumbers(false)
        .withUnquotedControlChars(true)
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(
            "true", "false", null, null, "true", "1", "0", null, "-0", "-01",
            "01", "-0.1", "-00.1", null, null, null, null, null, null, null,
            null, "\"3710-11-10T02:46:58.732Z\"", null, "-3.4e+38", "-3.4e-38", "1.4e38", "-3.4E+38", "-3.4E-38", "1.4E38", null,
            null, "\"A\u0000B\"", "\"A\u0000B\"", "\"A\u0001B\"", "\"A\u0001B\"", "\"A\u001FB\"", "\"A\u001FB\"", "\"A B\"", "\"A B\"", null,
            null, "\"\r\"", null)
        .build();
         MultiBufferDataSource source = sourceFrom(JSON_VALIDATION_BUFFER);
         Table table = Table.readJSON(schema, opts, source, (int)expected.getRowCount())) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testJSONValidationNonNumeric() {
    Schema schema = Schema.builder()
        .column(DType.STRING, "a")
        .build();
    JSONOptions opts = JSONOptions.builder()
        .withRecoverWithNull(true)
        .withMixedTypesAsStrings(true)
        .withNormalizeWhitespace(true)
        .withKeepQuotes(true)
        .withNormalizeSingleQuotes(true)
        .withStrictValidation(true)
        .withLeadingZeros(false)
        .withNonNumericNumbers(true)
        .withUnquotedControlChars(true)
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(
            "true", "false", null, null, "true", "1", "0", null, "-0", null,
            null, "-0.1", null, "NaN", null, "+INF", "-INF", "+Infinity", "Infinity", "-Infinity",
            null, "\"3710-11-10T02:46:58.732Z\"", null, "-3.4e+38", "-3.4e-38", "1.4e38", "-3.4E+38", "-3.4E-38", "1.4E38", null,
            null, "\"A\u0000B\"", "\"A\u0000B\"", "\"A\u0001B\"", "\"A\u0001B\"", "\"A\u001FB\"", "\"A\u001FB\"", "\"A B\"", "\"A B\"", null,
            null, "\"\r\"", null)
        .build();
         MultiBufferDataSource source = sourceFrom(JSON_VALIDATION_BUFFER);
         Table table = Table.readJSON(schema, opts, source, (int)expected.getRowCount())) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testJSONValidationUnquotedControl() {
    Schema schema = Schema.builder()
        .column(DType.STRING, "a")
        .build();
    JSONOptions opts = JSONOptions.builder()
        .withRecoverWithNull(true)
        .withMixedTypesAsStrings(true)
        .withNormalizeWhitespace(true)
        .withKeepQuotes(true)
        .withNormalizeSingleQuotes(true)
        .withStrictValidation(true)
        .withLeadingZeros(false)
        .withNonNumericNumbers(false)
        .withUnquotedControlChars(false)
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(
            "true", "false", null, null, "true", "1", "0", null, "-0", null,
            null, "-0.1", null, null, null, null, null, null, null, null,
            null, "\"3710-11-10T02:46:58.732Z\"", null, "-3.4e+38", "-3.4e-38", "1.4e38", "-3.4E+38", "-3.4E-38", "1.4E38", null,
            null, null, "\"A\u0000B\"", null, "\"A\u0001B\"", null, "\"A\u001FB\"", "\"A B\"", "\"A B\"", null,
            null, "\"\r\"", null)
        .build();
         MultiBufferDataSource source = sourceFrom(JSON_VALIDATION_BUFFER);
         Table table = Table.readJSON(schema, opts, source, (int)expected.getRowCount())) {
      assertTablesAreEqual(expected, table);
    }
  }

  private static final byte[] CR_JSON_TEST_BUFFER = ("{\"a\":\"12\n3\"}\0" +
      "{\"a\":\"AB\nC\"}\0").getBytes(StandardCharsets.UTF_8);

  @Test
  void testReadJSONDelim() {
    Schema schema = Schema.builder().addColumn(DType.STRING, "a").build();
    JSONOptions opts = JSONOptions.builder()
        .withLines(true)
        .withLineDelimiter('\0')
        .build();
    try (Table expected = new Table.TestBuilder()
        .column("12\n3", "AB\nC")
        .build();
        Table found = Table.readJSON(schema, opts, CR_JSON_TEST_BUFFER)) {
      assertTablesAreEqual(expected, found);
    }
  }

  private static final byte[] NESTED_JSON_DATA_BUFFER = ("{\"a\":{\"c\":\"C1\"}}\n" +
      "{\"a\":{\"c\":\"C2\", \"b\":\"B2\"}}\n" +
      "{\"d\":[1,2,3]}\n" +
      "{\"e\": [{\"g\": 1}, {\"f\": 2}, {\"f\": 3, \"g\": 4}], \"d\":[]}").getBytes(StandardCharsets.UTF_8);

  @Test
  void testReadJSONNestedTypes() {
    Schema.Builder root = Schema.builder();
    Schema.Builder a = root.addColumn(DType.STRUCT, "a");
    a.addColumn(DType.STRING, "b");
    a.addColumn(DType.STRING, "c");
    a.addColumn(DType.STRING, "missing");
    Schema.Builder d = root.addColumn(DType.LIST, "d");
    d.addColumn(DType.INT64, "ignored");
    root.addColumn(DType.INT64, "also_missing");
    Schema.Builder e = root.addColumn(DType.LIST, "e");
    Schema.Builder eChild = e.addColumn(DType.STRUCT, "ignored");
    eChild.addColumn(DType.INT64, "f");
    eChild.addColumn(DType.STRING, "missing_in_list");
    eChild.addColumn(DType.INT64, "g");
    Schema schema = root.build();
    JSONOptions opts = JSONOptions.builder()
        .withLines(true)
        .build();
    StructType aStruct = new StructType(true,
        new BasicType(true, DType.STRING),
        new BasicType(true, DType.STRING),
        new BasicType(true, DType.STRING));
    ListType dList = new ListType(true, new BasicType(true, DType.INT64));
    StructType eChildStruct = new StructType(true,
        new BasicType(true, DType.INT64),
        new BasicType(true, DType.STRING),
        new BasicType(true, DType.INT64));
    ListType eList = new ListType(true, eChildStruct);
    try (Table expected = new Table.TestBuilder()
        .column(aStruct,
            new StructData(null, "C1", null),
            new StructData("B2", "C2", null),
            null,
            null)
        .column(dList,
            null,
            null,
            Arrays.asList(1L,2L,3L),
            new ArrayList<Long>())
        .column((Long)null, null, null, null) // also_missing
        .column(eList,
            null,
            null,
            null,
            Arrays.asList(new StructData(null, null, 1L), new StructData(2L, null, null), new StructData(3L, null, 4L)))
        .build();
        Table table = Table.readJSON(schema, opts, NESTED_JSON_DATA_BUFFER)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadJSONNestedTypesVerySmallChanges() {
    Schema.Builder root = Schema.builder();
    Schema.Builder e = root.addColumn(DType.LIST, "e");
    Schema.Builder eChild = e.addColumn(DType.STRUCT, "ignored");
    eChild.addColumn(DType.INT64, "g");
    eChild.addColumn(DType.INT64, "f");
    Schema schema = root.build();
    JSONOptions opts = JSONOptions.builder()
        .withLines(true)
        .build();
    StructType eChildStruct = new StructType(true,
        new BasicType(true, DType.INT64),
        new BasicType(true, DType.INT64));
    ListType eList = new ListType(true, eChildStruct);
    try (Table expected = new Table.TestBuilder()
        .column(eList,
            null,
            null,
            null,
            Arrays.asList(new StructData(1L, null), new StructData(null, 2L), new StructData(4L, 3L)))
        .build();
         Table table = Table.readJSON(schema, opts, NESTED_JSON_DATA_BUFFER)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadJSONNestedTypesDataSource() {
    Schema.Builder root = Schema.builder();
    Schema.Builder a = root.addColumn(DType.STRUCT, "a");
    a.addColumn(DType.STRING, "b");
    a.addColumn(DType.STRING, "c");
    a.addColumn(DType.STRING, "missing");
    Schema.Builder d = root.addColumn(DType.LIST, "d");
    d.addColumn(DType.INT64, "ignored");
    root.addColumn(DType.INT64, "also_missing");
    Schema.Builder e = root.addColumn(DType.LIST, "e");
    Schema.Builder eChild = e.addColumn(DType.STRUCT, "ignored");
    eChild.addColumn(DType.INT64, "g");
    Schema schema = root.build();
    JSONOptions opts = JSONOptions.builder()
        .withLines(true)
        .build();
    StructType aStruct = new StructType(true,
        new BasicType(true, DType.STRING),
        new BasicType(true, DType.STRING),
        new BasicType(true, DType.STRING));
    ListType dList = new ListType(true, new BasicType(true, DType.INT64));
    StructType eChildStruct = new StructType(true,
        new BasicType(true, DType.INT64));
    ListType eList = new ListType(true, eChildStruct);
    try (Table expected = new Table.TestBuilder()
        .column(aStruct,
            new StructData(null, "C1", null),
            new StructData("B2", "C2", null),
            null,
            null)
        .column(dList,
            null,
            null,
            Arrays.asList(1L,2L,3L),
            new ArrayList<Long>())
        .column((Long)null, null, null, null) // also_missing
        .column(eList,
            null,
            null,
            null,
            Arrays.asList(new StructData(1L), new StructData((Long)null), new StructData(4L)))
        .build();
         MultiBufferDataSource source = sourceFrom(NESTED_JSON_DATA_BUFFER);
         Table table = Table.readJSON(schema, opts, source)) {
      assertTablesAreEqual(expected, table);
    }
  }

  void testReadMixedType2JSONFileFeatureDisabled() {
    Schema schema = Schema.builder()
            .column(DType.STRING, "a")
            .build();
    JSONOptions opts = JSONOptions.builder()
            .withLines(true)
            .withMixedTypesAsStrings(false)
            .build();
    assertThrows(CudfException.class, () ->
      Table.readJSON(schema, opts, TEST_MIXED_TYPE_2_JSON));
  }

  @Test
  void testReadMixedType1JSONFile() {
    Schema schema = Schema.builder()
            .column(DType.STRING, "a")
            .build();
    JSONOptions opts = JSONOptions.builder()
            .withLines(true)
            .withMixedTypesAsStrings(true)
            .build();
    try (Table expected = new Table.TestBuilder()
            .column("123", "123" )
            .build();
         Table table = Table.readJSON(schema, opts, TEST_MIXED_TYPE_1_JSON)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadMixedType2JSONFile() throws IOException {
    Schema schema = Schema.builder()
            .column(DType.STRING, "a")
            .build();
    JSONOptions opts = JSONOptions.builder()
            .withLines(true)
            .withMixedTypesAsStrings(true)
            .build();
    try (Table expected = new Table.TestBuilder()
            .column("[1,2,3]", "{ \"b\": 1 }" )
            .build();
         MultiBufferDataSource source = sourceFrom(TEST_MIXED_TYPE_2_JSON);
         Table table = Table.readJSON(schema, opts, source)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadJSONFromDataSource() throws IOException {
    Schema schema = Schema.builder()
            .column(DType.STRING, "name")
            .column(DType.INT32, "age")
            .build();
    JSONOptions opts = JSONOptions.builder()
            .withLines(true)
            .build();
    try (Table expected = new Table.TestBuilder()
            .column("Michael", "Andy", "Justin")
            .column(null, 30, 19)
            .build();
         MultiBufferDataSource source = sourceFrom(TEST_SIMPLE_JSON_FILE);
         Table table = Table.readJSON(schema, opts, source)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadJSONFileWithInvalidLines() {
    Schema schema = Schema.builder()
            .column(DType.STRING, "name")
            .column(DType.INT32, "age")
            .build();

    // test with recoverWithNulls=true
    {
      JSONOptions opts = JSONOptions.builder()
              .withLines(true)
              .withRecoverWithNull(true)
              .build();
      try (Table expected = new Table.TestBuilder()
              .column("Michael", "Andy", null, "Justin")
              .column(null, 30, null, 19)
              .build();
           Table table = Table.readJSON(schema, opts, TEST_JSON_ERROR_FILE)) {
        assertTablesAreEqual(expected, table);
      }
    }

    // test with recoverWithNulls=false
    {
      JSONOptions opts = JSONOptions.builder()
              .withLines(true)
              .withRecoverWithNull(false)
              .build();
      assertThrows(CudfException.class, () ->
        Table.readJSON(schema, opts, TEST_JSON_ERROR_FILE));
    }
  }

  @Test
  void testReadJSONFileWithDifferentColumnOrder() {
    Schema schema = Schema.builder()
        .column(DType.INT32, "age")
        .column(DType.STRING, "name")
        .build();
    JSONOptions opts = JSONOptions.builder()
        .withLines(true)
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(null, 30, 19)
        .column("Michael", "Andy", "Justin")
        .build();
         Table table = Table.readJSON(schema, opts, TEST_SIMPLE_JSON_FILE)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadJSONBufferInferred() {
    JSONOptions opts = JSONOptions.builder()
        .withDayFirst(true)
        .build();
    byte[] data = ("[false,A,1,2]\n" +
        "[true,B,2,3]\n" +
        "[false,C,3,4]\n" +
        "[true,D,4,5]").getBytes(StandardCharsets.UTF_8);
    try (Table expected = new Table.TestBuilder()
        .column(false, true, false, true)
        .column("A", "B", "C", "D")
        .column(1L, 2L, 3L, 4L)
        .column(2L, 3L, 4L, 5L)
        .build();
         Table table = Table.readJSON(Schema.INFERRED, opts, data)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadJSONSubColumns() {
    // JSON file has 2 columns, here only read 1 column
    Schema schema = Schema.builder()
        .column(DType.INT32, "age")
        .build();
    JSONOptions opts = JSONOptions.builder()
        .withLines(true)
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(null, 30, 19)
        .build();
         Table table = Table.readJSON(schema, opts, TEST_SIMPLE_JSON_FILE)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadJSONBuffer() {
    // JSON reader will set the column according to the iterator if can't infer the name
    // So we must set the same name accordingly
    Schema schema = Schema.builder()
        .column(DType.STRING, "0")
        .column(DType.INT32, "1")
        .column(DType.INT32, "2")
        .build();
    JSONOptions opts = JSONOptions.builder()
        .build();
    byte[] data = ("[A,1,2]\n" +
        "[B,2,3]\n" +
        "[C,3,4]\n" +
        "[D,4,5]").getBytes(StandardCharsets.UTF_8);
    try (Table expected = new Table.TestBuilder()
        .column("A", "B", "C", "D")
        .column(1, 2, 3, 4)
        .column(2, 3, 4, 5)
        .build();
         Table table = Table.readJSON(schema, opts, data)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadJSONBufferWithOffset() {
    // JSON reader will set the column according to the iterator if can't infer the name
    // So we must set the same name accordingly
    Schema schema = Schema.builder()
        .column(DType.STRING, "0")
        .column(DType.INT32, "1")
        .column(DType.INT32, "2")
        .build();
    JSONOptions opts = JSONOptions.builder()
        .build();
    int bytesToIgnore = 8;
    byte[] data = ("[A,1,2]\n" +
        "[B,2,3]\n" +
        "[C,3,4]\n" +
        "[D,4,5]").getBytes(StandardCharsets.UTF_8);
    try (Table expected = new Table.TestBuilder()
        .column("B", "C", "D")
        .column(2, 3, 4)
        .column(3, 4, 5)
        .build();
         Table table = Table.readJSON(schema, opts, data,
             bytesToIgnore, data.length - bytesToIgnore)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadJSONTableWithMeta() {
    JSONOptions opts = JSONOptions.builder()
            .build();
    byte[] data = ("{ \"A\": 1, \"B\": 2, \"C\": \"X\"}\n" +
            "{ \"A\": 2, \"B\": 4, \"C\": \"Y\"}\n" +
            "{ \"A\": 3, \"B\": 6, \"C\": \"Z\"}\n" +
            "{ \"A\": 4, \"B\": 8, \"C\": \"W\"}\n").getBytes(StandardCharsets.UTF_8);
    final int numBytes = data.length;
    try (HostMemoryBuffer hostbuf = hostMemoryAllocator.allocate(numBytes)) {
      hostbuf.setBytes(0, data, 0, numBytes);
      try (Table expected = new Table.TestBuilder()
              .column(1L, 2L, 3L, 4L)
              .column(2L, 4L, 6L, 8L)
              .column("X", "Y", "Z", "W")
              .build();
         TableWithMeta tablemeta = Table.readJSON(opts, hostbuf, 0, numBytes);
         Table table = tablemeta.releaseTable()) {
        assertArrayEquals(new String[] { "A", "B", "C" }, tablemeta.getColumnNames());
        assertTablesAreEqual(expected, table);
      }
    }
  }

  @Test
  void testReadCSVPrune() {
    Schema schema = Schema.builder()
        .column(DType.INT32, "A")
        .column(DType.FLOAT64, "B")
        .column(DType.INT64, "C")
        .build();
    CSVOptions opts = CSVOptions.builder()
        .includeColumn("A")
        .includeColumn("B")
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.2, 119.8)
        .build();
         Table table = Table.readCSV(schema, opts, TEST_SIMPLE_CSV_FILE)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadCSVBufferInferred() {
    CSVOptions opts = CSVOptions.builder()
        .includeColumn("A")
        .includeColumn("B")
        .hasHeader()
        .withComment('#')
        .build();
    byte[] data = ("A,B,C\n" +
        "0,110.0,120'\n" +
        "#0.5,1.0,200\n" +
        "1,111.0,121\n" +
        "2,112.0,122\n" +
        "3,113.0,123\n" +
        "4,114.0,124\n" +
        "5,115.0,125\n" +
        "6,116.0,126\n" +
        "7,117.0,127\n" +
        "8,118.2,128\n" +
        "9,119.8,129").getBytes(StandardCharsets.UTF_8);
    try (Table expected = new Table.TestBuilder()
        .column(0L, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L)
        .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.2, 119.8)
        .build();
         Table table = Table.readCSV(Schema.INFERRED, opts, data)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadCSVBuffer() {
    CSVOptions opts = CSVOptions.builder()
        .includeColumn("A")
        .includeColumn("B")
        .hasHeader()
        .withDelim('|')
        .withQuote('\'')
        .withNullValue("NULL")
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, null, 118.2, 119.8)
        .build();
         Table table = Table.readCSV(TableTest.CSV_DATA_BUFFER_SCHEMA, opts,
             TableTest.CSV_DATA_BUFFER)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadCSVBufferMultiBuffer() {
    CSVOptions opts = CSVOptions.builder()
            .includeColumn("A")
            .includeColumn("B")
            .hasHeader()
            .withDelim('|')
            .withQuote('\'')
            .withNullValue("NULL")
            .build();
    byte[][] data = TableTestUtils.sliceBytes(CSV_DATA_BUFFER, 10);
    try (Table expected = new Table.TestBuilder()
            .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
            .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, null, 118.2, 119.8)
            .build();
         MultiBufferDataSource source = sourceFrom(data);
         Table table = Table.readCSV(TableTest.CSV_DATA_BUFFER_SCHEMA, opts, source)) {
      assertTablesAreEqual(expected, table);
    }
  }

  public static MultiBufferDataSource sourceFrom(File f) throws IOException {
    long len = f.length();
    byte[] tmp = new byte[(int)Math.min(32 * 1024, len)];
    try (HostMemoryBuffer buffer = HostMemoryBuffer.allocate(len)) {
      try (java.io.FileInputStream fin = new java.io.FileInputStream(f)) {
        long at = 0;
        while (at < len) {
          int amount = fin.read(tmp);
          buffer.setBytes(at, tmp, 0, amount);
          at += amount;
        }
      }
      return new MultiBufferDataSource(buffer);
    }
  }

  public static MultiBufferDataSource sourceFrom(byte[] data) {
    long len = data.length;
    try (HostMemoryBuffer buffer = HostMemoryBuffer.allocate(len)) {
      buffer.setBytes(0, data, 0, len);
      return new MultiBufferDataSource(buffer);
    }
  }

  public static MultiBufferDataSource sourceFrom(byte[][] data) {
    HostMemoryBuffer[] buffers = new HostMemoryBuffer[data.length];
    try {
      for (int i = 0; i < data.length; i++) {
        byte[] subData = data[i];
        buffers[i] = HostMemoryBuffer.allocate(subData.length);
        buffers[i].setBytes(0, subData, 0, subData.length);
      }
      return new MultiBufferDataSource(buffers);
    } finally {
      for (HostMemoryBuffer buffer: buffers) {
        if (buffer != null) {
          buffer.close();
        }
      }
    }
  }

  @Test
  void testReadCSVDataSource() {
    CSVOptions opts = CSVOptions.builder()
            .includeColumn("A")
            .includeColumn("B")
            .hasHeader()
            .withDelim('|')
            .withQuote('\'')
            .withNullValue("NULL")
            .build();
    try (Table expected = new Table.TestBuilder()
            .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
            .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, null, 118.2, 119.8)
            .build();
         MultiBufferDataSource source = sourceFrom(TableTest.CSV_DATA_BUFFER);
         Table table = Table.readCSV(TableTest.CSV_DATA_BUFFER_SCHEMA, opts, source)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadCSVWithOffset() {
    CSVOptions opts = CSVOptions.builder()
        .includeColumn("A")
        .includeColumn("B")
        .hasHeader(false)
        .withDelim('|')
        .withNullValue("NULL")
        .build();
    int bytesToIgnore = 24;
    try (Table expected = new Table.TestBuilder()
        .column(1, 2, 3, 4, 5, 6, 7, 8, 9)
        .column(111.0, 112.0, 113.0, 114.0, 115.0, 116.0, null, 118.2, 119.8)
        .build();
         Table table = Table.readCSV(TableTest.CSV_DATA_BUFFER_SCHEMA, opts,
             TableTest.CSV_DATA_BUFFER, bytesToIgnore, CSV_DATA_BUFFER.length - bytesToIgnore)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadCSVOtherTypes() {
    final byte[] CSV_DATA_WITH_TYPES = ("A,B,C,D\n" +
        "0,true,120,\"zero\"\n" +
        "1,True,121,\"one\"\n" +
        "2,false,122,\"two\"\n" +
        "3,false,123,\"three\"\n" +
        "4,TRUE,124,\"four\"\n" +
        "5,true,125,\"five\"\n" +
        "6,true,126,\"six\"\n" +
        "7,NULL,127,NULL\n" +
        "8,false,128,\"eight\"\n" +
        "9,false,129,\"nine\uD80C\uDC3F\"").getBytes(StandardCharsets.UTF_8);

    final Schema CSV_DATA_WITH_TYPES_SCHEMA = Schema.builder()
        .column(DType.INT32, "A")
        .column(DType.BOOL8, "B")
        .column(DType.INT64, "C")
        .column(DType.STRING, "D")
        .build();

    CSVOptions opts = CSVOptions.builder()
        .includeColumn("A", "B", "D")
        .hasHeader(true)
        .withNullValue("NULL")
        .withQuote('"')
        .withTrueValue("true", "True", "TRUE")
        .withFalseValue("false")
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        .column(true, true, false, false, true, true, true, null, false, false)
        .column("zero", "one", "two", "three", "four", "five", "six", null, "eight", "nine\uD80C\uDC3F")
        .build();
         Table table = Table.readCSV(CSV_DATA_WITH_TYPES_SCHEMA, opts, CSV_DATA_WITH_TYPES)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadCSV() {
    Schema schema = Schema.builder()
        .column(DType.INT32, "A")
        .column(DType.FLOAT64, "B")
        .column(DType.INT64, "C")
        .column(DType.STRING, "D")
        .build();
    try (Table expected = new Table.TestBuilder()
        .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
        .column(110.0, 111.0, 112.0, 113.0, 114.0, 115.0, 116.0, 117.0, 118.2, 119.8)
        .column(120L, 121L, 122L, 123L, 124L, 125L, 126L, 127L, 128L, 129L)
        .column("one", "two", "three", "four", "five", "six", "seven\ud801\uddb8", "eight\uBF68", "nine\u03E8", "ten")
        .build();
         Table table = Table.readCSV(schema, TEST_SIMPLE_CSV_FILE)) {
      assertTablesAreEqual(expected, table);
    }
  }

  private void testWriteCSVToFileImpl(char fieldDelim, boolean includeHeader,
                                      String trueValue, String falseValue) throws IOException {
    Schema schema = Schema.builder()
                          .column(DType.INT32, "i")
                          .column(DType.FLOAT64, "f")
                          .column(DType.BOOL8, "b")
                          .column(DType.STRING, "str")
                          .build();
    CSVWriterOptions writeOptions = CSVWriterOptions.builder()
                                               .withColumnNames(schema.getFlattenedColumnNames())
                                               .withIncludeHeader(includeHeader)
                                               .withFieldDelimiter((byte)fieldDelim)
                                               .withRowDelimiter("\n")
                                               .withNullValue("\\N")
                                               .withTrueValue(trueValue)
                                               .withFalseValue(falseValue)
                                               .build();
    CSVOptions readOptions = CSVOptions.builder()
                                       .includeColumn("i")
                                       .includeColumn("f")
                                       .includeColumn("b")
                                       .includeColumn("str")
                                       .hasHeader(includeHeader)
                                       .withDelim(fieldDelim)
                                       .withTrueValue(trueValue)
                                       .withFalseValue(falseValue)
                                       .build();
    try (TempFile outputTempFile = TempFile.create("testWriteCSVToFile", ".csv")) {
      File outputFile = outputTempFile.getFile();
      try (Table inputTable = new Table.TestBuilder()
          .column(0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
          .column(0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
          .column(false, true, false, true, false, true, false, true, false, true)
          .column("All", "the", "leaves", "are", "brown", "and", "the", "sky", "is", "grey")
          .build()) {
        inputTable.writeCSVToFile(writeOptions, outputFile.getAbsolutePath());

        // Read back.
        try (Table readTable = Table.readCSV(schema, readOptions, outputFile)) {
          assertTablesAreEqual(inputTable, readTable);
        }
      }
    }
  }

  @Test
  void testWriteCSVToFile() throws IOException {
    final boolean INCLUDE_HEADER = true;
    final boolean NO_HEADER = false;
    testWriteCSVToFileImpl(',', INCLUDE_HEADER, "true", "false");
    testWriteCSVToFileImpl(',', NO_HEADER, "TRUE", "FALSE");
    testWriteCSVToFileImpl('\u0001', INCLUDE_HEADER, "T", "F");
    testWriteCSVToFileImpl('\u0001', NO_HEADER, "True", "False");
  }

  private void testWriteUnquotedCSVToFileImpl(char fieldDelim) throws IOException {
    Schema schema = Schema.builder()
                          .column(DType.STRING, "str")
                          .build();
    CSVWriterOptions writeOptions = CSVWriterOptions.builder()
                                               .withColumnNames(schema.getFlattenedColumnNames())
                                               .withIncludeHeader(false)
                                               .withFieldDelimiter((byte)fieldDelim)
                                               .withRowDelimiter("\n")
                                               .withNullValue("\\N")
                                               .withQuoteStyle(QuoteStyle.NONE)
                                               .build();
    CSVOptions readOptions = CSVOptions.builder()
                                       .includeColumn("str")
                                       .hasHeader(false)
                                       .withDelim(fieldDelim)
                                       .withQuoteStyle(QuoteStyle.NONE)
                                       .build();
    try (TempFile outputTempFile = TempFile.create("testWriteUnquotedCSVToFile", ".csv")) {
      File outputFile = outputTempFile.getFile();
      try (Table inputTable = new Table.TestBuilder()
          .column("All" + fieldDelim + "the" + fieldDelim + "leaves",
                  "are\"brown",
                  "and\nthe\nsky\nis\ngrey")
          .build()) {
        inputTable.writeCSVToFile(writeOptions, outputFile.getAbsolutePath());
      }

      // Read back.
      try (Table readTable = Table.readCSV(schema, readOptions, outputFile);
           Table expected = new Table.TestBuilder()
             .column("All", "are\"brown", "and", "the", "sky", "is", "grey")
             .build()) {
        assertTablesAreEqual(expected, readTable);
      }
    }
  }

  @Test
  void testWriteUnquotedCSVToFile() throws IOException {
    testWriteUnquotedCSVToFileImpl(',');
    testWriteUnquotedCSVToFileImpl('\u0001');
  }

  private void testChunkedCSVWriterUnquotedImpl(char fieldDelim) throws IOException {
    Schema schema = Schema.builder()
                          .column(DType.STRING, "str")
                          .build();
    CSVWriterOptions writeOptions = CSVWriterOptions.builder()
                                               .withColumnNames(schema.getFlattenedColumnNames())
                                               .withIncludeHeader(false)
                                               .withFieldDelimiter((byte)fieldDelim)
                                               .withRowDelimiter("\n")
                                               .withNullValue("\\N")
                                               .withQuoteStyle(QuoteStyle.NONE)
                                               .build();
    try (Table inputTable
          = new Table.TestBuilder()
              .column("All" + fieldDelim + "the" + fieldDelim + "leaves",
                      "are\"brown",
                      "and\nthe\nsky\nis\ngrey")
              .build();
          MyBufferConsumer consumer = new MyBufferConsumer()) {

      try (TableWriter writer = Table.getCSVBufferWriter(writeOptions, consumer)) {
        writer.write(inputTable);
        writer.write(inputTable);
        writer.write(inputTable);
      }

      // Read back.
      CSVOptions readOptions = CSVOptions.builder()
                                         .includeColumn("str")
                                         .hasHeader(false)
                                         .withDelim(fieldDelim)
                                         .withNullValue("\\N")
                                         .withQuoteStyle(QuoteStyle.NONE)
                                         .build();
      try (Table readTable = Table.readCSV(schema, readOptions, consumer.buffer, 0, consumer.offset);
           Table section = new Table.TestBuilder()
             .column("All", "are\"brown", "and", "the", "sky", "is", "grey")
             .build();
           Table expected  = Table.concatenate(section, section, section)) {
        assertTablesAreEqual(expected, readTable);
      }
    }
  }

  @Test
  void testChunkedCSVWriterUnquoted() throws IOException {
    testChunkedCSVWriterUnquotedImpl(',');
    testChunkedCSVWriterUnquotedImpl('\u0001');
  }

  private void testChunkedCSVWriterImpl(char fieldDelim, boolean includeHeader,
                                        String trueValue, String falseValue) throws IOException {
    Schema schema = Schema.builder()
                          .column(DType.INT32, "i")
                          .column(DType.FLOAT64, "f")
                          .column(DType.BOOL8, "b")
                          .column(DType.STRING, "str")
                          .build();
    CSVWriterOptions writeOptions = CSVWriterOptions.builder()
                                               .withColumnNames(schema.getFlattenedColumnNames())
                                               .withIncludeHeader(includeHeader)
                                               .withFieldDelimiter((byte)fieldDelim)
                                               .withRowDelimiter("\n")
                                               .withNullValue("\\N")
                                               .withTrueValue(trueValue)
                                               .withFalseValue(falseValue)
                                               .build();
    try (Table inputTable
          = new Table.TestBuilder()
              .column(0, 1, 2, 3, 4, 5, 6, 7, 8, null)
              .column(0.0, 1.0, 2.0, 3.0, 4.0, null, 6.0, 7.0, 8.0, 9.0)
              .column(false, true, null, true, false, true, null, true, false, true)
              .column("All", "the", "leaves", "are", "brown", "and", "the", "sky", "is", null)
              .build();
          MyBufferConsumer consumer = new MyBufferConsumer()) {

      try (TableWriter writer = Table.getCSVBufferWriter(writeOptions, consumer)) {
        writer.write(inputTable);
        writer.write(inputTable);
        writer.write(inputTable);
      }

      // Read back.
      CSVOptions readOptions = CSVOptions.builder()
                                         .includeColumn("i")
                                         .includeColumn("f")
                                         .includeColumn("b")
                                         .includeColumn("str")
                                         .hasHeader(includeHeader)
                                         .withDelim(fieldDelim)
                                         .withNullValue("\\N")
                                         .withTrueValue(trueValue)
                                         .withFalseValue(falseValue)
                                         .build();
      try (Table readTable = Table.readCSV(schema, readOptions, consumer.buffer, 0, consumer.offset);
           Table expected  = Table.concatenate(inputTable, inputTable, inputTable)) {
        assertTablesAreEqual(expected, readTable);
      }
    }
  }

  @Test
  void testChunkedCSVWriter() throws IOException {
    final boolean INCLUDE_HEADER = true;
    final boolean NO_HEADER = false;
    testChunkedCSVWriterImpl(',', NO_HEADER, "true", "false");
    testChunkedCSVWriterImpl(',', INCLUDE_HEADER, "TRUE", "FALSE");
    testChunkedCSVWriterImpl('\u0001', NO_HEADER, "T", "F");
    testChunkedCSVWriterImpl('\u0001', INCLUDE_HEADER, "True", "False");
  }

  @Test
  void testReadParquet() {
    ParquetOptions opts = ParquetOptions.builder()
        .includeColumn("loan_id")
        .includeColumn("zip")
        .includeColumn("num_units")
        .build();
    try (Table table = Table.readParquet(opts, TEST_PARQUET_FILE)) {
      long rows = table.getRowCount();
      assertEquals(1000, rows);
      assertTableTypes(new DType[]{DType.INT64, DType.INT32, DType.INT32}, table);
    }
  }

  @Test
  void testReadParquetFromDataSource() throws IOException {
    ParquetOptions opts = ParquetOptions.builder()
            .includeColumn("loan_id")
            .includeColumn("zip")
            .includeColumn("num_units")
            .build();
    try (MultiBufferDataSource source = sourceFrom(TEST_PARQUET_FILE);
         Table table = Table.readParquet(opts, source)) {
      long rows = table.getRowCount();
      assertEquals(1000, rows);
      assertTableTypes(new DType[]{DType.INT64, DType.INT32, DType.INT32}, table);
    }
  }

  @Test
  void testReadParquetMultiBuffer() throws IOException {
    ParquetOptions opts = ParquetOptions.builder()
            .includeColumn("loan_id")
            .includeColumn("zip")
            .includeColumn("num_units")
            .build();
    byte [][] data = TableTestUtils.sliceBytes(TableTestUtils.arrayFrom(TEST_PARQUET_FILE), 10);
    try (MultiBufferDataSource source = sourceFrom(data);
         Table table = Table.readParquet(opts, source)) {
      long rows = table.getRowCount();
      assertEquals(1000, rows);
      assertTableTypes(new DType[]{DType.INT64, DType.INT32, DType.INT32}, table);
    }
  }

  @Test
  void testReadParquetBinary() {
    ParquetOptions opts = ParquetOptions.builder()
        .includeColumn("value1", true)
        .includeColumn("value2", false)
        .build();
    try (Table table = Table.readParquet(opts, TEST_PARQUET_FILE_BINARY)) {
      assertTableTypes(new DType[]{DType.STRING, DType.STRING}, table);
      ColumnView columnView = table.getColumn(0);
      assertEquals(DType.STRING, columnView.getType());
    }
  }

  List<Byte> asList(String str) {
    byte[] bytes = str.getBytes(Charsets.UTF_8);
    List<Byte> ret = new ArrayList<>(bytes.length);
    for(int i = 0; i < bytes.length; i++) {
      ret.add(bytes[i]);
    }
    return ret;
  }

  @Test
  void testParquetWriteToBufferChunkedBinary() {
    // We create a String table and a Binary table with the same data in them to
    // avoid trying to read the binary data back in the same way. At least until the
    // API for that is stable
    String string1 = "ABC";
    String string2 = "DEF";
    List<Byte> bin1 = asList(string1);
    List<Byte> bin2 = asList(string2);

    try (Table binTable = new Table.TestBuilder()
        .column(new ListType(true, new BasicType(false, DType.UINT8)),
            bin1, bin2)
        .build();
         Table stringTable = new Table.TestBuilder()
             .column(string1, string2)
             .build();
         MyBufferConsumer consumer = new MyBufferConsumer()) {
      ParquetWriterOptions options = ParquetWriterOptions.builder()
          .withBinaryColumn("_c0", true)
          .build();

      try (TableWriter writer = Table.writeParquetChunked(options, consumer)) {
        writer.write(binTable);
        writer.write(binTable);
        writer.write(binTable);
      }
      ParquetOptions opts = ParquetOptions.builder()
          .includeColumn("_c0")
          .build();
      try (Table table1 = Table.readParquet(opts, consumer.buffer, 0, consumer.offset);
           Table concat = Table.concatenate(stringTable, stringTable, stringTable)) {
        assertTablesAreEqual(concat, table1);
      }
    }
  }

  @Test
  void testReadParquetBuffer() throws IOException {
    ParquetOptions opts = ParquetOptions.builder()
        .includeColumn("loan_id")
        .includeColumn("coborrow_credit_score")
        .includeColumn("borrower_credit_score")
        .build();

    byte[] buffer = new byte[(int) TEST_PARQUET_FILE.length() + 1024];
    int bufferLen = 0;
    try (FileInputStream in = new FileInputStream(TEST_PARQUET_FILE)) {
      bufferLen = in.read(buffer);
    }
    try (Table table = Table.readParquet(opts, buffer, 0, bufferLen)) {
      long rows = table.getRowCount();
      assertEquals(1000, rows);
      assertTableTypes(new DType[]{DType.INT64, DType.FLOAT64, DType.FLOAT64}, table);
    }
  }

  @Test
  void testReadParquetFull() {
    try (Table table = Table.readParquet(TEST_PARQUET_FILE)) {
      long rows = table.getRowCount();
      assertEquals(1000, rows);

      DType[] expectedTypes = new DType[]{
          DType.INT64, // loan_id
          DType.INT32, // orig_channel
          DType.FLOAT64, // orig_interest_rate
          DType.INT32, // orig_upb
          DType.INT32, // orig_loan_term
          DType.TIMESTAMP_DAYS, // orig_date
          DType.TIMESTAMP_DAYS, // first_pay_date
          DType.FLOAT64, // orig_ltv
          DType.FLOAT64, // orig_cltv
          DType.FLOAT64, // num_borrowers
          DType.FLOAT64, // dti
          DType.FLOAT64, // borrower_credit_score
          DType.INT32, // first_home_buyer
          DType.INT32, // loan_purpose
          DType.INT32, // property_type
          DType.INT32, // num_units
          DType.INT32, // occupancy_status
          DType.INT32, // property_state
          DType.INT32, // zip
          DType.FLOAT64, // mortgage_insurance_percent
          DType.INT32, // product_type
          DType.FLOAT64, // coborrow_credit_score
          DType.FLOAT64, // mortgage_insurance_type
          DType.INT32, // relocation_mortgage_indicator
          DType.INT32, // quarter
          DType.INT32 // seller_id
      };

      assertTableTypes(expectedTypes, table);
    }
  }

  @Test
  void testReadParquetContainsDecimalData() {
    try (Table table = Table.readParquet(TEST_DECIMAL_PARQUET_FILE)) {
      long rows = table.getRowCount();
      assertEquals(100, rows);
      DType[] expectedTypes = new DType[]{
          DType.create(DType.DTypeEnum.DECIMAL64, 0), // Decimal(18, 0)
          DType.create(DType.DTypeEnum.DECIMAL32, -3), // Decimal(7, 3)
          DType.create(DType.DTypeEnum.DECIMAL64, -10),  // Decimal(10, 10)
          DType.create(DType.DTypeEnum.DECIMAL32, 0),  // Decimal(1, 0)
          DType.create(DType.DTypeEnum.DECIMAL64, -15),  // Decimal(18, 15)
          DType.create(DType.DTypeEnum.DECIMAL128, -10),  // Decimal(20, 10)
          DType.INT64,
          DType.FLOAT32
      };
      assertTableTypes(expectedTypes, table);
    }
  }

  @Test
  void testChunkedReadParquet() {
    try (ParquetChunkedReader reader = new ParquetChunkedReader(240000,
        TEST_PARQUET_FILE_CHUNKED_READ)) {
      int numChunks = 0;
      long totalRows = 0;
      while(reader.hasNext()) {
        ++numChunks;
        try(Table chunk = reader.readChunk()) {
          totalRows += chunk.getRowCount();
        }
      }
      assertEquals(2, numChunks);
      assertEquals(40000, totalRows);
    }
  }

  @Test
  void testChunkedReadParquetHostBuffers() throws Exception {
    long size = TEST_PARQUET_FILE_CHUNKED_READ.length();
    java.nio.file.Path path = TEST_PARQUET_FILE_CHUNKED_READ.toPath();
    try (HostMemoryBuffer buf1 = HostMemoryBuffer.allocate(size / 2);
         HostMemoryBuffer buf2 = HostMemoryBuffer.allocate(size - buf1.getLength())) {
      try (SeekableByteChannel channel = Files.newByteChannel(path, StandardOpenOption.READ)) {
        ByteBuffer bb1 = buf1.asByteBuffer();
        while (bb1.hasRemaining()) {
          if (channel.read(bb1) == -1) {
            throw new EOFException("error reading first buffer");
          }
        }
        ByteBuffer bb2 = buf2.asByteBuffer();
        while (bb2.hasRemaining()) {
          if (channel.read(bb2) == -1) {
            throw new EOFException("error reading second buffer");
          }
        }
      }
      ParquetOptions opts = ParquetOptions.DEFAULT;
      try (ParquetChunkedReader reader = new ParquetChunkedReader(240000, 0, opts, buf1, buf2)) {
        int numChunks = 0;
        long totalRows = 0;
        while(reader.hasNext()) {
          ++numChunks;
          try(Table chunk = reader.readChunk()) {
            totalRows += chunk.getRowCount();
          }
        }
        assertEquals(2, numChunks);
        assertEquals(40000, totalRows);
      }
    }
  }

  @Test
  void testChunkedReadParquetFromDataSource() throws IOException {
    try (MultiBufferDataSource source = sourceFrom(TEST_PARQUET_FILE_CHUNKED_READ);
         ParquetChunkedReader reader = new ParquetChunkedReader(240000, ParquetOptions.DEFAULT, source)) {
      int numChunks = 0;
      long totalRows = 0;
      while(reader.hasNext()) {
        ++numChunks;
        try(Table chunk = reader.readChunk()) {
          totalRows += chunk.getRowCount();
        }
      }
      assertEquals(2, numChunks);
      assertEquals(40000, totalRows);
    }
  }

  @Test
  void testReadAvro() {
    AvroOptions opts = AvroOptions.builder()
        .includeColumn("bool_col")
        .includeColumn("int_col")
        .includeColumn("timestamp_col")
        .build();

    try (Table expected = new Table.TestBuilder()
        .column(true, false, true, false, true, false, true, false)
        .column(0, 1, 0, 1, 0, 1, 0, 1)
        .column(1235865600000000L, 1235865660000000L, 1238544000000000L, 1238544060000000L,
            1233446400000000L, 1233446460000000L, 1230768000000000L, 1230768060000000L)
        .build();
        Table table = Table.readAvro(opts, TEST_ALL_TYPES_PLAIN_AVRO_FILE)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadAvroFromDataSource() throws IOException {
    AvroOptions opts = AvroOptions.builder()
            .includeColumn("bool_col")
            .includeColumn("int_col")
            .includeColumn("timestamp_col")
            .build();

    try (Table expected = new Table.TestBuilder()
            .column(true, false, true, false, true, false, true, false)
            .column(0, 1, 0, 1, 0, 1, 0, 1)
            .column(1235865600000000L, 1235865660000000L, 1238544000000000L, 1238544060000000L,
                    1233446400000000L, 1233446460000000L, 1230768000000000L, 1230768060000000L)
            .build();
         MultiBufferDataSource source = sourceFrom(TEST_ALL_TYPES_PLAIN_AVRO_FILE);
         Table table = Table.readAvro(opts, source)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadAvroBuffer() throws IOException{
    AvroOptions opts = AvroOptions.builder()
        .includeColumn("bool_col")
        .includeColumn("timestamp_col")
        .build();

    byte[] buffer = Files.readAllBytes(TEST_ALL_TYPES_PLAIN_AVRO_FILE.toPath());
    int bufferLen = buffer.length;
    try (Table expected = new Table.TestBuilder()
        .column(true, false, true, false, true, false, true, false)
        .column(1235865600000000L, 1235865660000000L, 1238544000000000L, 1238544060000000L,
            1233446400000000L, 1233446460000000L, 1230768000000000L, 1230768060000000L)
        .build();
        Table table = Table.readAvro(opts, buffer, 0, bufferLen)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadAvroFull() {
    try (Table expected = new Table.TestBuilder()
        .column(4, 5, 6, 7, 2, 3, 0, 1)
        .column(true, false, true, false, true, false, true, false)
        .column(0, 1, 0, 1, 0, 1, 0, 1)
        .column(0, 1, 0, 1, 0, 1, 0, 1)
        .column(0, 1, 0, 1, 0, 1, 0, 1)
        .column(0L, 10L, 0L, 10L, 0L, 10L, 0L, 10L)
        .column(0.0f, 1.100000023841858f, 0.0f, 1.100000023841858f, 0.0f, 1.100000023841858f, 0.0f, 1.100000023841858f)
        .column(0.0d, 10.1d, 0.0d, 10.1d, 0.0d, 10.1d, 0.0d, 10.1d)
        .column("03/01/09", "03/01/09", "04/01/09", "04/01/09", "02/01/09", "02/01/09", "01/01/09", "01/01/09")
        .column("0", "1", "0", "1", "0", "1", "0", "1")
        .column(1235865600000000L, 1235865660000000L, 1238544000000000L, 1238544060000000L,
            1233446400000000L, 1233446460000000L, 1230768000000000L, 1230768060000000L)
        .build();
        Table table = Table.readAvro(TEST_ALL_TYPES_PLAIN_AVRO_FILE)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadORC() {
    ORCOptions opts = ORCOptions.builder()
        .includeColumn("string1")
        .includeColumn("float1")
        .includeColumn("int1")
        .build();
    try (Table expected = new Table.TestBuilder()
        .column("hi","bye")
        .column(1.0f,2.0f)
        .column(65536,65536)
        .build();
         Table table = Table.readORC(opts, TEST_ORC_FILE)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadORCFromDataSource() throws IOException {
    ORCOptions opts = ORCOptions.builder()
            .includeColumn("string1")
            .includeColumn("float1")
            .includeColumn("int1")
            .build();
    try (Table expected = new Table.TestBuilder()
            .column("hi","bye")
            .column(1.0f,2.0f)
            .column(65536,65536)
            .build();
         MultiBufferDataSource source = sourceFrom(TEST_ORC_FILE);
         Table table = Table.readORC(opts, source)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadORCBuffer() throws IOException {
    ORCOptions opts = ORCOptions.builder()
        .includeColumn("string1")
        .includeColumn("float1")
        .includeColumn("int1")
        .build();

    int bufferLen = 0;
    byte[] buffer = Files.readAllBytes(TEST_ORC_FILE.toPath());
    bufferLen = buffer.length;
    try (Table expected = new Table.TestBuilder()
        .column("hi","bye")
        .column(1.0f,2.0f)
        .column(65536,65536)
        .build();
         Table table = Table.readORC(opts, buffer, 0, bufferLen)) {
      assertTablesAreEqual(expected, table);
    }
  }

  @Test
  void testReadORCFull() {
    try (Table expected = new Table.TestBuilder()
        .column(false, true)
        .column((byte)1, (byte)100)
        .column((short)1024, (short)2048)
        .column(65536, 65536)
        .column(9223372036854775807L,9223372036854775807L)
        .column(1.0f, 2.0f)
        .column(-15.0, -5.0)
        .column("hi", "bye")
        .build();
         Table table = Table.readORC(TEST_ORC_FILE)) {
      assertTablesAreEqual(expected,  table);
    }
  }

  @Test
  void testReadORCNumPyTypes() {
    // by default ORC will promote TIMESTAMP_DAYS to TIMESTAMP_MILLISECONDS
    DType found;
    try (Table table = Table.readORC(TEST_ORC_TIMESTAMP_DATE_FILE)) {
      assertEquals(2, table.getNumberOfColumns());
      found = table.getColumn(0).getType();
      assertTrue(found.isTimestampType());
      assertEquals(DType.TIMESTAMP_MILLISECONDS, table.getColumn(1).getType());
    }

    // specifying no NumPy types should load them as TIMESTAMP_DAYS
    ORCOptions opts = ORCOptions.builder().withNumPyTypes(false).build();
    try (Table table = Table.readORC(opts, TEST_ORC_TIMESTAMP_DATE_FILE)) {
      assertEquals(2, table.getNumberOfColumns());
      assertEquals(found, table.getColumn(0).getType());
      assertEquals(DType.TIMESTAMP_DAYS, table.getColumn(1).getType());
    }
  }

  @Test
  void testReadORCTimeUnit() {
    // specifying no NumPy types should load them as TIMESTAMP_DAYS.
    // specifying a specific type will return the result in that unit
    ORCOptions opts = ORCOptions.builder()
        .withNumPyTypes(false)
        .withTimeUnit(DType.TIMESTAMP_SECONDS)
        .build();
    try (Table table = Table.readORC(opts, TEST_ORC_TIMESTAMP_DATE_FILE)) {
      assertEquals(2, table.getNumberOfColumns());
      assertEquals(DType.TIMESTAMP_SECONDS, table.getColumn(0).getType());
      assertEquals(DType.TIMESTAMP_DAYS, table.getColumn(1).getType());
    }
  }

  @Test
  void testORCChunkedReader() throws IOException {
    byte[] buffer = Files.readAllBytes(TEST_ORC_FILE_CHUNKED_READ.toPath());
    long len = buffer.length;

    try (HostMemoryBuffer hostBuf = hostMemoryAllocator.allocate(len)) {
      hostBuf.setBytes(0, buffer, 0, len);
      try (ORCChunkedReader reader = new ORCChunkedReader(0, 2 * 1024 * 1024, 10000,
          ORCOptions.DEFAULT, hostBuf, 0, len)) {
        int numChunks = 0;
        long totalRows = 0;
        while (reader.hasNext()) {
          ++numChunks;
          try (Table chunk = reader.readChunk()) {
            totalRows += chunk.getRowCount();
          }
        }
        assertEquals(10, numChunks);
        assertEquals(1000000, totalRows);
      }
    }
  }

  @Test
  void testCrossJoin() {
    try (Table leftTable = new Table.TestBuilder()
            .column(100, 101, 102)
            .build();
         Table rightTable = new Table.TestBuilder()
                 .column(200, null)
                 .build();
         Table expected = new Table.TestBuilder()
                 .column(  100, 100,  101, 101,  102, 102) // left
                 .column( null, 200, null, 200, null, 200) // right
                 .build();
         Table joinedTable = leftTable.crossJoin(rightTable);
         Table orderedJoinedTable =
                 joinedTable.orderBy(
                     OrderByArg.asc(0, true),
                     OrderByArg.asc(1, true))) {
      assertTablesAreEqual(expected, orderedJoinedTable);
    }
  }

  private void verifyJoinGatherMaps(GatherMap[] maps, Table expected) {
    assertEquals(2, maps.length);
    int numRows = (int) expected.getRowCount();
    assertEquals(numRows, maps[0].getRowCount());
    assertEquals(numRows, maps[1].getRowCount());
    try (ColumnVector leftMap = maps[0].toColumnView(0, numRows).copyToColumnVector();
         ColumnVector rightMap = maps[1].toColumnView(0, numRows).copyToColumnVector();
         Table result = new Table(leftMap, rightMap);
         Table orderedResult = result.orderBy(OrderByArg.asc(0, true))) {
      assertTablesAreEqual(expected, orderedResult);
    }
  }

  private void verifySemiJoinGatherMap(GatherMap map, Table expected) {
    int numRows = (int) expected.getRowCount();
    assertEquals(numRows, map.getRowCount());
    try (ColumnVector leftMap = map.toColumnView(0, numRows).copyToColumnVector();
         Table result = new Table(leftMap);
         Table orderedResult = result.orderBy(OrderByArg.asc(0, true))) {
      assertTablesAreEqual(expected, orderedResult);
    }
  }

  @Test
  void testLeftJoinGatherMaps() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, 32).build();
         Table expected = new Table.TestBuilder()
             .column(  0,   1, 2,   3,   4,   5,   6, 7, 8, 9)
             .column(inv, inv, 2, inv, inv, inv, inv, 0, 1, 3)
             .build()) {
      GatherMap[] maps = leftKeys.leftJoinGatherMaps(rightKeys, false);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testLeftJoinGatherMapsNulls() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder()
            .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
            .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(  0,   1, 2,   3,   4,   5,   6, 7, 7, 8, 8, 9) // left
             .column(inv, inv, 2, inv, inv, inv, inv, 0, 1, 0, 1, 3) // right
             .build()) {
      GatherMap[] maps = leftKeys.leftJoinGatherMaps(rightKeys, true);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  private void checkLeftDistinctJoin(Table leftKeys, Table rightKeys, ColumnView expected,
                                     boolean compareNullsEqual) {
    try (GatherMap map = leftKeys.leftDistinctJoinGatherMap(rightKeys, compareNullsEqual)) {
      int numRows = (int) expected.getRowCount();
      assertEquals(numRows, map.getRowCount());
      try (ColumnView view = map.toColumnView(0, numRows)) {
        assertColumnsAreEqual(expected, view);
      }
    }
  }

  @Test
  void testLeftDistinctJoinGatherMaps() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8, 6).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, 32).build();
         ColumnVector expected = ColumnVector.fromInts(inv, inv, 2, inv, inv, inv, inv, 0, 1, 3, 0)) {
      checkLeftDistinctJoin(leftKeys, rightKeys, expected, false);
    }
  }

  @Test
  void testLeftDistinctJoinGatherMapsWithNested() {
    final int inv = Integer.MIN_VALUE;
    StructType structType = new StructType(false,
        new BasicType(false, DType.STRING),
        new BasicType(false, DType.INT32));
    StructData[] leftData = new StructData[]{
        new StructData("abc", 1),
        new StructData("xyz", 1),
        new StructData("abc", 2),
        new StructData("xyz", 2),
        new StructData("abc", 1),
        new StructData("abc", 3),
        new StructData("xyz", 3)
    };
    StructData[] rightData = new StructData[]{
        new StructData("abc", 1),
        new StructData("xyz", 4),
        new StructData("xyz", 2),
        new StructData("abc", -1),
    };
    try (Table leftKeys = new Table.TestBuilder().column(structType, leftData).build();
         Table rightKeys = new Table.TestBuilder().column(structType, rightData).build();
         ColumnVector expected = ColumnVector.fromInts(0, inv, inv, 2, 0, inv, inv)) {
      checkLeftDistinctJoin(leftKeys, rightKeys, expected, false);
    }
  }

  @Test
  void testLeftDistinctJoinGatherMapsNullsEqual() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, 9, 8, 10, 32)
             .build();
         ColumnVector expected = ColumnVector.fromInts(inv, inv, 1, inv, inv, inv, inv, 0, 0, 2)) {
      checkLeftDistinctJoin(leftKeys, rightKeys, expected, true);
    }
  }

  @Test
  void testLeftDistinctJoinGatherMapsWithNestedNullsEqual() {
    final int inv = Integer.MIN_VALUE;
    StructType structType = new StructType(true,
        new BasicType(true, DType.STRING),
        new BasicType(true, DType.INT32));
    StructData[] leftData = new StructData[]{
        new StructData("abc", 1),
        null,
        new StructData("xyz", 1),
        new StructData("abc", 2),
        new StructData("xyz", null),
        null,
        new StructData("abc", 1),
        new StructData("abc", 3),
        new StructData("xyz", 3),
        new StructData(null, null),
        new StructData(null, 1)
    };
    StructData[] rightData = new StructData[]{
        null,
        new StructData("abc", 1),
        new StructData("xyz", 4),
        new StructData("xyz", 2),
        new StructData(null, null),
        new StructData(null, 2),
        new StructData(null, 1),
        new StructData("xyz", null),
        new StructData("abc", null),
        new StructData("abc", -1)
    };
    try (Table leftKeys = new Table.TestBuilder().column(structType, leftData).build();
         Table rightKeys = new Table.TestBuilder().column(structType, rightData).build();
         ColumnVector expected = ColumnVector.fromInts(1, 0, inv, inv, 7, 0, 1, inv, inv, 4, 6)) {
      checkLeftDistinctJoin(leftKeys, rightKeys, expected, true);
    }
  }

  @Test
  void testLeftHashJoinGatherMaps() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, 32).build();
         HashJoin rightHash = new HashJoin(rightKeys, false);
         Table expected = new Table.TestBuilder()
             .column(  0,   1, 2,   3,   4,   5,   6, 7, 8, 9)
             .column(inv, inv, 2, inv, inv, inv, inv, 0, 1, 3)
             .build()) {
      GatherMap[] maps = leftKeys.leftJoinGatherMaps(rightHash);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testLeftHashJoinGatherMapsWithCount() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, 32).build();
         HashJoin rightHash = new HashJoin(rightKeys, false);
         Table expected = new Table.TestBuilder()
             .column(  0,   1, 2,   3,   4,   5,   6, 7, 8, 9)
             .column(inv, inv, 2, inv, inv, inv, inv, 0, 1, 3)
             .build()) {
      long rowCount = leftKeys.leftJoinRowCount(rightHash);
      assertEquals(expected.getRowCount(), rowCount);
      GatherMap[] maps = leftKeys.leftJoinGatherMaps(rightHash, rowCount);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testLeftHashJoinGatherMapsNulls() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         HashJoin rightHash = new HashJoin(rightKeys, true);
         Table expected = new Table.TestBuilder()
             .column(  0,   1, 2,   3,   4,   5,   6, 7, 7, 8, 8, 9) // left
             .column(inv, inv, 2, inv, inv, inv, inv, 0, 1, 0, 1, 3) // right
             .build()) {
      GatherMap[] maps = leftKeys.leftJoinGatherMaps(rightHash);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testLeftHashJoinGatherMapsNullsWithCount() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         HashJoin rightHash = new HashJoin(rightKeys,true);
         Table expected = new Table.TestBuilder()
             .column(  0,   1, 2,   3,   4,   5,   6, 7, 7, 8, 8, 9) // left
             .column(inv, inv, 2, inv, inv, inv, inv, 0, 1, 0, 1, 3) // right
             .build()) {
      long rowCount = leftKeys.leftJoinRowCount(rightHash);
      assertEquals(expected.getRowCount(), rowCount);
      GatherMap[] maps = leftKeys.leftJoinGatherMaps(rightHash, rowCount);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testConditionalLeftJoinGatherMaps() {
    final int inv = Integer.MIN_VALUE;
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5).build();
         Table expected = new Table.TestBuilder()
             .column(  0,   1, 2, 2, 2,   3,   4, 5, 5,   6, 7,   8, 9, 9)
             .column(inv, inv, 0, 1, 3, inv, inv, 0, 1, inv, 1, inv, 0, 1)
             .build();
         CompiledExpression condition = expr.compile()) {
      GatherMap[] maps = left.conditionalLeftJoinGatherMaps(right, condition);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testConditionalLeftJoinGatherMapsNulls() {
    final int inv = Integer.MIN_VALUE;
    BinaryOperation expr = new BinaryOperation(BinaryOperator.NULL_EQUAL,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table right = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(  0,   1, 2,   3,   4,   5,   6, 7, 7, 8, 8, 9) // left
             .column(inv, inv, 2, inv, inv, inv, inv, 0, 1, 0, 1, 3) // right
             .build();
         CompiledExpression condition = expr.compile()) {
      GatherMap[] maps = left.conditionalLeftJoinGatherMaps(right, condition);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testConditionalLeftJoinGatherMapsWithCount() {
    final int inv = Integer.MIN_VALUE;
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5).build();
         Table expected = new Table.TestBuilder()
             .column(  0,   1, 2, 2, 2,   3,   4, 5, 5,   6, 7,   8, 9, 9)
             .column(inv, inv, 0, 1, 3, inv, inv, 0, 1, inv, 1, inv, 0, 1)
             .build();
         CompiledExpression condition = expr.compile()) {
      long rowCount = left.conditionalLeftJoinRowCount(right, condition);
      assertEquals(expected.getRowCount(), rowCount);
      GatherMap[] maps = left.conditionalLeftJoinGatherMaps(right, condition, rowCount);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testConditionalLeftJoinGatherMapsNullsWithCount() {
    final int inv = Integer.MIN_VALUE;
    BinaryOperation expr = new BinaryOperation(BinaryOperator.NULL_EQUAL,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table right = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(  0,   1, 2,   3,   4,   5,   6, 7, 7, 8, 8, 9) // left
             .column(inv, inv, 2, inv, inv, inv, inv, 0, 1, 0, 1, 3) // right
             .build();
         CompiledExpression condition = expr.compile()) {
      long rowCount = left.conditionalLeftJoinRowCount(right, condition);
      assertEquals(expected.getRowCount(), rowCount);
      GatherMap[] maps = left.conditionalLeftJoinGatherMaps(right, condition, rowCount);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testMixedLeftJoinGatherMaps() {
    final int inv = Integer.MIN_VALUE;
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
            new ColumnReference(1, TableReference.LEFT),
            new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8)
             .column(1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3,  4,  5)
             .column(7, 8, 9, 0,  1,  2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(  0,   1, 2,   3,   4,   5,   6, 7, 8,   9)
             .column(inv, inv, 2, inv, inv, inv, inv, 0, 1, inv)
             .build()) {
      GatherMap[] maps = Table.mixedLeftJoinGatherMaps(leftKeys, rightKeys, left, right, condition,
          NullEquality.UNEQUAL);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testMixedLeftJoinGatherMapsNulls() {
    final int inv = Integer.MIN_VALUE;
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
            new ColumnReference(1, TableReference.LEFT),
            new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(null, 3, 9, 0, 1, 7, 4, null, 5, 8)
             .column(   1, 2, 3, 4, 5, 6, 7,    8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(null, 5, null, 8, 10, 32)
             .column(   0, 1,    2, 3,  4,  5)
             .column(   7, 8,    9, 0,  1,  2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(0,   1,   2,   3,   4,   5,   6, 7, 7, 8,   9)
             .column(0, inv, inv, inv, inv, inv, inv, 0, 2, 1, inv)
             .build()) {
      GatherMap[] maps = Table.mixedLeftJoinGatherMaps(leftKeys, rightKeys, left, right, condition,
          NullEquality.EQUAL);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testMixedLeftJoinGatherMapsWithSize() {
    final int inv = Integer.MIN_VALUE;
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
            new ColumnReference(1, TableReference.LEFT),
            new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8)
             .column(1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5)
             .column(7, 8, 9, 0, 1, 2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(  0,   1, 2,   3,   4,   5,   6, 7, 8,   9)
             .column(inv, inv, 2, inv, inv, inv, inv, 0, 1, inv)
             .build();
         MixedJoinSize sizeInfo = Table.mixedLeftJoinSize(leftKeys, rightKeys, left, right,
             condition, NullEquality.UNEQUAL)) {
      assertEquals(expected.getRowCount(), sizeInfo.getOutputRowCount());
      GatherMap[] maps = Table.mixedLeftJoinGatherMaps(leftKeys, rightKeys, left, right, condition,
          NullEquality.UNEQUAL, sizeInfo);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testMixedLeftJoinGatherMapsNullsWithSize() {
    final int inv = Integer.MIN_VALUE;
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
            new ColumnReference(1, TableReference.LEFT),
            new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(null, 3, 9, 0, 1, 7, 4, null, 5, 8)
             .column(   1, 2, 3, 4, 5, 6, 7,    8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(null, 5, null, 8, 10, 32)
             .column(   0, 1,    2, 3,  4,  5)
             .column(   7, 8,    9, 0,  1,  2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(0,   1,   2,   3,   4,   5,   6, 7, 7, 8,   9)
             .column(0, inv, inv, inv, inv, inv, inv, 0, 2, 1, inv)
             .build();
         MixedJoinSize sizeInfo = Table.mixedLeftJoinSize(leftKeys, rightKeys, left, right,
             condition, NullEquality.EQUAL)) {
      assertEquals(expected.getRowCount(), sizeInfo.getOutputRowCount());
      GatherMap[] maps = Table.mixedLeftJoinGatherMaps(leftKeys, rightKeys, left, right, condition,
              NullEquality.EQUAL, sizeInfo);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testInnerJoinGatherMaps() {
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, 32).build();
         Table expected = new Table.TestBuilder()
             .column(2, 7, 8, 9) // left
             .column(2, 0, 1, 3) // right
             .build()) {
      GatherMap[] maps = leftKeys.innerJoinGatherMaps(rightKeys, false);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testInnerJoinGatherMapsNulls() {
    try (Table leftKeys = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(2, 7, 7, 8, 8, 9) // left
             .column(2, 0, 1, 0, 1, 3) // right
             .build()) {
      GatherMap[] maps = leftKeys.innerJoinGatherMaps(rightKeys, true);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  private void checkInnerDistinctJoin(Table leftKeys, Table rightKeys, Table expected,
                                      boolean compareNullsEqual) {
    GatherMap[] maps = leftKeys.innerDistinctJoinGatherMaps(rightKeys, compareNullsEqual);
    try {
      verifyJoinGatherMaps(maps, expected);
    } finally {
      for (GatherMap map : maps) {
        map.close();
      }
    }
  }

  @Test
  void testInnerDistinctJoinGatherMaps() {
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8, 6).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, 32).build();
         Table expected = new Table.TestBuilder()
             .column(2, 7, 8, 9, 10) // left
             .column(2, 0, 1, 3, 0) // right
             .build()) {
      checkInnerDistinctJoin(leftKeys, rightKeys, expected, false);
    }
  }

  @Test
  void testInnerDistinctJoinGatherMapsWithNested() {
    StructType structType = new StructType(false,
        new BasicType(false, DType.STRING),
        new BasicType(false, DType.INT32));
    StructData[] leftData = new StructData[]{
        new StructData("abc", 1),
        new StructData("xyz", 1),
        new StructData("abc", 2),
        new StructData("xyz", 2),
        new StructData("abc", 1),
        new StructData("abc", 3),
        new StructData("xyz", 3)
    };
    StructData[] rightData = new StructData[]{
        new StructData("abc", 1),
        new StructData("xyz", 4),
        new StructData("xyz", 2),
        new StructData("abc", -1),
    };
    try (Table leftKeys = new Table.TestBuilder().column(structType, leftData).build();
         Table rightKeys = new Table.TestBuilder().column(structType, rightData).build();
         Table expected = new Table.TestBuilder()
             .column(0, 3, 4)
             .column(0, 2, 0)
             .build()) {
      checkInnerDistinctJoin(leftKeys, rightKeys, expected, false);
    }
  }

  @Test
  void testInnerDistinctJoinGatherMapsNullsEqual() {
    try (Table leftKeys = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(2, 7, 8, 9) // left
             .column(1, 0, 0, 2) // right
             .build()) {
      checkInnerDistinctJoin(leftKeys, rightKeys, expected, true);
    }
  }

  @Test
  void testInnerDistinctJoinGatherMapsWithNestedNullsEqual() {
    StructType structType = new StructType(true,
        new BasicType(true, DType.STRING),
        new BasicType(true, DType.INT32));
    StructData[] leftData = new StructData[]{
        new StructData("abc", 1),
        null,
        new StructData("xyz", 1),
        new StructData("abc", 2),
        new StructData("xyz", null),
        null,
        new StructData("abc", 1),
        new StructData("abc", 3),
        new StructData("xyz", 3),
        new StructData(null, null),
        new StructData(null, 1)
    };
    StructData[] rightData = new StructData[]{
        null,
        new StructData("abc", 1),
        new StructData("xyz", 4),
        new StructData("xyz", 2),
        new StructData(null, null),
        new StructData(null, 2),
        new StructData(null, 1),
        new StructData("xyz", null),
        new StructData("abc", null),
        new StructData("abc", -1)
    };
    try (Table leftKeys = new Table.TestBuilder().column(structType, leftData).build();
         Table rightKeys = new Table.TestBuilder().column(structType, rightData).build();
         Table expected = new Table.TestBuilder()
             .column(0, 1, 4, 5, 6, 9, 10)
             .column(1, 0, 7, 0, 1, 4, 6)
             .build()) {
      checkInnerDistinctJoin(leftKeys, rightKeys, expected, true);
    }
  }

  @Test
  void testInnerHashJoinGatherMaps() {
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, 32).build();
         HashJoin rightHash = new HashJoin(rightKeys, false);
         Table expected = new Table.TestBuilder()
             .column(2, 7, 8, 9) // left
             .column(2, 0, 1, 3) // right
             .build()) {
      GatherMap[] maps = leftKeys.innerJoinGatherMaps(rightHash);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testInnerHashJoinGatherMapsWithCount() {
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, 32).build();
         HashJoin rightHash = new HashJoin(rightKeys, false);
         Table expected = new Table.TestBuilder()
             .column(2, 7, 8, 9) // left
             .column(2, 0, 1, 3) // right
             .build()) {
      long rowCount = leftKeys.innerJoinRowCount(rightHash);
      assertEquals(expected.getRowCount(), rowCount);
      GatherMap[] maps = leftKeys.innerJoinGatherMaps(rightHash, rowCount);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testInnerHashJoinGatherMapsNulls() {
    try (Table leftKeys = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         HashJoin rightHash = new HashJoin(rightKeys, true);
         Table expected = new Table.TestBuilder()
             .column(2, 7, 7, 8, 8, 9) // left
             .column(2, 0, 1, 0, 1, 3) // right
             .build()) {
      GatherMap[] maps = leftKeys.innerJoinGatherMaps(rightHash);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testInnerHashJoinGatherMapsNullsWithCount() {
    try (Table leftKeys = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         HashJoin rightHash = new HashJoin(rightKeys, true);
         Table expected = new Table.TestBuilder()
             .column(2, 7, 7, 8, 8, 9) // left
             .column(2, 0, 1, 0, 1, 3) // right
             .build()) {
      long rowCount = leftKeys.innerJoinRowCount(rightHash);
      assertEquals(expected.getRowCount(), rowCount);
      GatherMap[] maps = leftKeys.innerJoinGatherMaps(rightHash, rowCount);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testConditionalInnerJoinGatherMaps() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5).build();
         Table expected = new Table.TestBuilder()
             .column(2, 2, 2, 5, 5, 7, 9, 9)
             .column(0, 1, 3, 0, 1, 1, 0, 1)
             .build();
         CompiledExpression condition = expr.compile()) {
      GatherMap[] maps = left.conditionalInnerJoinGatherMaps(right, condition);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  // Test non-null-supporting equality at least once.
  @Test
  void testConditionalInnerJoinGatherMapsEqual() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.EQUAL,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table right = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(2, 9) // left
             .column(2, 3) // right
             .build();
         CompiledExpression condition = expr.compile()) {
      GatherMap[] maps = left.conditionalInnerJoinGatherMaps(right, condition);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testConditionalInnerJoinGatherMapsNulls() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.NULL_EQUAL,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table right = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(2, 7, 7, 8, 8, 9) // left
             .column(2, 0, 1, 0, 1, 3) // right
             .build();
         CompiledExpression condition = expr.compile()) {
      GatherMap[] maps = left.conditionalInnerJoinGatherMaps(right, condition);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testConditionalInnerJoinGatherMapsWithCount() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5).build();
         Table expected = new Table.TestBuilder()
             .column(2, 2, 2, 5, 5, 7, 9, 9)
             .column(0, 1, 3, 0, 1, 1, 0, 1)
             .build();
         CompiledExpression condition = expr.compile()) {
      long rowCount = left.conditionalInnerJoinRowCount(right, condition);
      assertEquals(expected.getRowCount(), rowCount);
      GatherMap[] maps = left.conditionalInnerJoinGatherMaps(right, condition, rowCount);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testConditionalInnerJoinGatherMapsNullsWithCount() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.NULL_EQUAL,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table right = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(2, 7, 7, 8, 8, 9) // left
             .column(2, 0, 1, 0, 1, 3) // right
             .build();
         CompiledExpression condition = expr.compile()) {
      long rowCount = left.conditionalInnerJoinRowCount(right, condition);
      assertEquals(expected.getRowCount(), rowCount);
      GatherMap[] maps = left.conditionalInnerJoinGatherMaps(right, condition, rowCount);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testMixedInnerJoinGatherMaps() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(1, TableReference.LEFT),
        new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8)
             .column(1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5)
             .column(7, 8, 9, 0, 1, 2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(2, 7, 8)
             .column(2, 0, 1)
             .build()) {
      GatherMap[] maps = Table.mixedInnerJoinGatherMaps(leftKeys, rightKeys, left, right, condition,
          NullEquality.UNEQUAL);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testMixedInnerJoinGatherMapsNulls() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(1, TableReference.LEFT),
        new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(null, 3, 9, 0, 1, 7, 4, null, 5, 8)
             .column(   1, 2, 3, 4, 5, 6, 7,    8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(null, 5, null, 8, 10, 32)
             .column(   0, 1,    2, 3,  4,  5)
             .column(   7, 8,    9, 0,  1,  2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(0, 7, 7, 8)
             .column(0, 0, 2, 1)
             .build()) {
      GatherMap[] maps = Table.mixedInnerJoinGatherMaps(leftKeys, rightKeys, left, right, condition,
          NullEquality.EQUAL);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testMixedInnerJoinGatherMapsWithSize() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(1, TableReference.LEFT),
        new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8)
             .column(1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5)
             .column(7, 8, 9, 0, 1, 2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(2, 7, 8)
             .column(2, 0, 1)
             .build();
         MixedJoinSize sizeInfo = Table.mixedInnerJoinSize(leftKeys, rightKeys, left, right,
             condition, NullEquality.UNEQUAL)) {
      assertEquals(expected.getRowCount(), sizeInfo.getOutputRowCount());
      GatherMap[] maps = Table.mixedInnerJoinGatherMaps(leftKeys, rightKeys, left, right, condition,
          NullEquality.UNEQUAL, sizeInfo);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testMixedInnerJoinGatherMapsNullsWithSize() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(1, TableReference.LEFT),
        new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(null, 3, 9, 0, 1, 7, 4, null, 5, 8)
             .column(   1, 2, 3, 4, 5, 6, 7,    8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(null, 5, null, 8, 10, 32)
             .column(   0, 1,    2, 3,  4,  5)
             .column(   7, 8,    9, 0,  1,  2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(0, 7, 7, 8)
             .column(0, 0, 2, 1)
             .build();
         MixedJoinSize sizeInfo = Table.mixedInnerJoinSize(leftKeys, rightKeys, left, right,
             condition, NullEquality.EQUAL)) {
      assertEquals(expected.getRowCount(), sizeInfo.getOutputRowCount());
      GatherMap[] maps = Table.mixedInnerJoinGatherMaps(leftKeys, rightKeys, left, right, condition,
          NullEquality.EQUAL, sizeInfo);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testFullJoinGatherMaps() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, null, 1, 7, 4, 6, 5, 8).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, null).build();
         Table expected = new Table.TestBuilder()
             .column(inv, inv,   0,   1, 2,   3,   4,   5,   6, 7, 8, 9) // left
             .column(  4,   5, inv, inv, 2, inv, inv, inv, inv, 0, 1, 3) // right
             .build()) {
      GatherMap[] maps = leftKeys.fullJoinGatherMaps(rightKeys, false);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testFullJoinGatherMapsNulls() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder()
             .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
             .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(inv, inv,   0,   1, 2,   3,   4,   5,   6, 7, 7, 8, 8, 9) // left
             .column(  4,   5, inv, inv, 2, inv, inv, inv, inv, 0, 1, 0, 1, 3) // right
             .build()) {
      GatherMap[] maps = leftKeys.fullJoinGatherMaps(rightKeys, true);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testFullHashJoinGatherMaps() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, null, 1, 7, 4, 6, 5, 8).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, null).build();
         HashJoin rightHash = new HashJoin(rightKeys, false);
         Table expected = new Table.TestBuilder()
             .column(inv, inv,   0,   1, 2,   3,   4,   5,   6, 7, 8, 9) // left
             .column(  4,   5, inv, inv, 2, inv, inv, inv, inv, 0, 1, 3) // right
             .build()) {
      GatherMap[] maps = leftKeys.fullJoinGatherMaps(rightHash);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testFullHashJoinGatherMapsWithCount() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, null, 1, 7, 4, 6, 5, 8).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, null).build();
         HashJoin rightHash = new HashJoin(rightKeys, false);
         Table expected = new Table.TestBuilder()
             .column(inv, inv,   0,   1, 2,   3,   4,   5,   6, 7, 8, 9) // left
             .column(  4,   5, inv, inv, 2, inv, inv, inv, inv, 0, 1, 3) // right
             .build()) {
      long rowCount = leftKeys.fullJoinRowCount(rightHash);
      assertEquals(expected.getRowCount(), rowCount);
      GatherMap[] maps = leftKeys.fullJoinGatherMaps(rightHash, rowCount);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testFullHashJoinGatherMapsNulls() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         HashJoin rightHash = new HashJoin(rightKeys, true);
         Table expected = new Table.TestBuilder()
             .column(inv, inv,   0,   1, 2,   3,   4,   5,   6, 7, 7, 8, 8, 9) // left
             .column(  4,   5, inv, inv, 2, inv, inv, inv, inv, 0, 1, 0, 1, 3) // right
             .build()) {
      GatherMap[] maps = leftKeys.fullJoinGatherMaps(rightHash);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testFullHashJoinGatherMapsNullsWithCount() {
    final int inv = Integer.MIN_VALUE;
    try (Table leftKeys = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         HashJoin rightHash = new HashJoin(rightKeys, true);
         Table expected = new Table.TestBuilder()
             .column(inv, inv,   0,   1, 2,   3,   4,   5,   6, 7, 7, 8, 8, 9) // left
             .column(  4,   5, inv, inv, 2, inv, inv, inv, inv, 0, 1, 0, 1, 3) // right
             .build()) {
      long rowCount = leftKeys.fullJoinRowCount(rightHash);
      assertEquals(expected.getRowCount(), rowCount);
      GatherMap[] maps = leftKeys.fullJoinGatherMaps(rightHash, rowCount);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testConditionalFullJoinGatherMaps() {
    final int inv = Integer.MIN_VALUE;
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5).build();
         Table expected = new Table.TestBuilder()
             .column(inv, inv, inv,   0,   1, 2, 2, 2,   3,   4, 5, 5,   6, 7,   8, 9, 9)
             .column(  2,   4,   5, inv, inv, 0, 1, 3, inv, inv, 0, 1, inv, 1, inv, 0, 1)
             .build();
         CompiledExpression condition = expr.compile()) {
      GatherMap[] maps = left.conditionalFullJoinGatherMaps(right, condition);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testConditionalFullJoinGatherMapsNulls() {
    final int inv = Integer.MIN_VALUE;
    BinaryOperation expr = new BinaryOperation(BinaryOperator.NULL_EQUAL,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table right = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(inv, inv,   0,   1, 2,   3,   4,   5,   6, 7, 7, 8, 8, 9) // left
             .column(  4,   5, inv, inv, 2, inv, inv, inv, inv, 0, 1, 0, 1, 3) // right
             .build();
         CompiledExpression condition = expr.compile()) {
      GatherMap[] maps = left.conditionalFullJoinGatherMaps(right, condition);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testMixedFullJoinGatherMaps() {
    final int inv = Integer.MIN_VALUE;
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
            new ColumnReference(1, TableReference.LEFT),
            new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8)
             .column(1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5)
             .column(7, 8, 9, 0, 1, 2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(inv, inv, inv,   0,   1, 2,   3,   4,   5,   6, 7, 8,   9)
             .column(  3,   4,   5, inv, inv, 2, inv, inv, inv, inv, 0, 1, inv)
             .build()) {
      GatherMap[] maps = Table.mixedFullJoinGatherMaps(leftKeys, rightKeys, left, right, condition,
          NullEquality.UNEQUAL);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testMixedFullJoinGatherMapsNulls() {
    final int inv = Integer.MIN_VALUE;
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(1, TableReference.LEFT),
        new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(null, 3, 9, 0, 1, 7, 4, null, 5, 8)
             .column(   1, 2, 3, 4, 5, 6, 7,    8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(null, 5, null, 8, 10, 32)
             .column(   0, 1,    2, 3,  4,  5)
             .column(   7, 8,    9, 0,  1,  2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(inv, inv, inv, 0,   1,   2,   3,   4,   5,   6, 7, 7, 8,   9)
             .column(  3,   4,   5, 0, inv, inv, inv, inv, inv, inv, 0, 2, 1, inv)
             .build()) {
      GatherMap[] maps = Table.mixedFullJoinGatherMaps(leftKeys, rightKeys, left, right, condition,
          NullEquality.EQUAL);
      try {
        verifyJoinGatherMaps(maps, expected);
      } finally {
        for (GatherMap map : maps) {
          map.close();
        }
      }
    }
  }

  @Test
  void testMixedLeftSemiJoinGatherMap() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(1, TableReference.LEFT),
        new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8)
             .column(1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5)
             .column(7, 8, 9, 0, 1, 2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(2, 7, 8)
             .build();
         GatherMap map = Table.mixedLeftSemiJoinGatherMap(leftKeys, rightKeys, left, right,
             condition, NullEquality.UNEQUAL)) {
      verifySemiJoinGatherMap(map, expected);
    }
  }

  @Test
  void testMixedLeftSemiJoinGatherMapNulls() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(1, TableReference.LEFT),
        new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(null, 3, 9, 0, 1, 7, 4, null, 5, 8)
             .column(   1, 2, 3, 4, 5, 6, 7,    8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(null, 5, null, 8, 10, 32)
             .column(   0, 1,    2, 3,  4,  5)
             .column(   7, 8,    9, 0,  1,  2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(0, 7, 8)
             .build();
         GatherMap map = Table.mixedLeftSemiJoinGatherMap(leftKeys, rightKeys, left, right,
             condition, NullEquality.EQUAL)) {
      verifySemiJoinGatherMap(map, expected);
    }
  }

  @Test
  void testMixedLeftAntiJoinGatherMap() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(1, TableReference.LEFT),
        new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8)
             .column(1, 2, 3, 4, 5, 6, 7, 8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5)
             .column(7, 8, 9, 0, 1, 2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(0, 1, 3, 4, 5, 6, 9)
             .build();
         GatherMap map = Table.mixedLeftAntiJoinGatherMap(leftKeys, rightKeys, left, right,
             condition, NullEquality.UNEQUAL)) {
      verifySemiJoinGatherMap(map, expected);
    }
  }

  @Test
  void testMixedLeftAntiJoinGatherMapNulls() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(1, TableReference.LEFT),
        new ColumnReference(1, TableReference.RIGHT));
    try (CompiledExpression condition = expr.compile();
         Table left = new Table.TestBuilder()
             .column(null, 3, 9, 0, 1, 7, 4, null, 5, 8)
             .column(   1, 2, 3, 4, 5, 6, 7,    8, 9, 0)
             .build();
         Table leftKeys = new Table(left.getColumn(0));
         Table right = new Table.TestBuilder()
             .column(null, 5, null, 8, 10, 32)
             .column(   0, 1,    2, 3,  4,  5)
             .column(   7, 8,    9, 0,  1,  2).build();
         Table rightKeys = new Table(right.getColumn(0));
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3, 4, 5, 6, 9)
             .build();
         GatherMap map = Table.mixedLeftAntiJoinGatherMap(leftKeys, rightKeys, left, right,
             condition, NullEquality.EQUAL)) {
      verifySemiJoinGatherMap(map, expected);
    }
  }

  @Test
  void testLeftSemiJoinGatherMap() {
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, 32).build();
         Table expected = new Table.TestBuilder()
             .column(2, 7, 8, 9) // left
             .build();
         GatherMap map = leftKeys.leftSemiJoinGatherMap(rightKeys, false)) {
      verifySemiJoinGatherMap(map, expected);
    }
  }

  @Test
  void testLeftSemiJoinGatherMapNulls() {
    try (Table leftKeys = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(2, 7, 8, 9) // left
             .build();
         GatherMap map = leftKeys.leftSemiJoinGatherMap(rightKeys, true)) {
      verifySemiJoinGatherMap(map, expected);
    }
  }

  @Test
  void testConditionalLeftSemiJoinGatherMap() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5).build();
         Table expected = new Table.TestBuilder()
             .column(2, 5, 7, 9) // left
             .build();
         CompiledExpression condition = expr.compile();
         GatherMap map = left.conditionalLeftSemiJoinGatherMap(right, condition)) {
      verifySemiJoinGatherMap(map, expected);
    }
  }

  @Test
  void testConditionalLeftSemiJoinGatherMapNulls() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.NULL_EQUAL,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table right = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(2, 7, 8, 9) // left
             .build();
         CompiledExpression condition = expr.compile();
         GatherMap map = left.conditionalLeftSemiJoinGatherMap(right, condition)) {
      verifySemiJoinGatherMap(map, expected);
    }
  }

  @Test
  void testConditionalLeftSemiJoinGatherMapWithCount() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5).build();
         Table expected = new Table.TestBuilder()
             .column(2, 5, 7, 9) // left
             .build();
         CompiledExpression condition = expr.compile()) {
      long rowCount = left.conditionalLeftSemiJoinRowCount(right, condition);
      assertEquals(expected.getRowCount(), rowCount);
      try (GatherMap map =
               left.conditionalLeftSemiJoinGatherMap(right, condition, rowCount)) {
        verifySemiJoinGatherMap(map, expected);
      }
    }
  }

  @Test
  void testConditionalLeftSemiJoinGatherMapNullsWithCount() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.NULL_EQUAL,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table right = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(2, 7, 8, 9) // left
             .build();
         CompiledExpression condition = expr.compile()) {
      long rowCount = left.conditionalLeftSemiJoinRowCount(right, condition);
      assertEquals(expected.getRowCount(), rowCount);
      try (GatherMap map =
               left.conditionalLeftSemiJoinGatherMap(right, condition, rowCount)) {
        verifySemiJoinGatherMap(map, expected);
      }
    }
  }

  @Test
  void testAntiSemiJoinGatherMap() {
    try (Table leftKeys = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table rightKeys = new Table.TestBuilder().column(6, 5, 9, 8, 10, 32).build();
         Table expected = new Table.TestBuilder()
             .column(0, 1, 3, 4, 5, 6) // left
             .build();
         GatherMap map = leftKeys.leftAntiJoinGatherMap(rightKeys, false)) {
      verifySemiJoinGatherMap(map, expected);
    }
  }

  @Test
  void testAntiSemiJoinGatherMapNulls() {
    try (Table leftKeys = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table rightKeys = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(0, 1, 3, 4, 5, 6) // left
             .build();
         GatherMap map = leftKeys.leftAntiJoinGatherMap(rightKeys, true)) {
      verifySemiJoinGatherMap(map, expected);
    }
  }

  @Test
  void testConditionalLeftAntiJoinGatherMap() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5).build();
         Table expected = new Table.TestBuilder()
             .column(0, 1, 3, 4, 6, 8) // left
             .build();
         CompiledExpression condition = expr.compile();
         GatherMap map = left.conditionalLeftAntiJoinGatherMap(right, condition)) {
      verifySemiJoinGatherMap(map, expected);
    }
  }

  @Test
  void testConditionalAntiSemiJoinGatherMapNulls() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.NULL_EQUAL,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table right = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(0, 1, 3, 4, 5, 6) // left
             .build();
         CompiledExpression condition = expr.compile();
         GatherMap map = left.conditionalLeftAntiJoinGatherMap(right, condition)) {
      verifySemiJoinGatherMap(map, expected);
    }
  }

  @Test
  void testConditionalLeftAntiJoinGatherMapWithCount() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.GREATER,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder().column(2, 3, 9, 0, 1, 7, 4, 6, 5, 8).build();
         Table right = new Table.TestBuilder()
             .column(6, 5, 9, 8, 10, 32)
             .column(0, 1, 2, 3, 4, 5).build();
         Table expected = new Table.TestBuilder()
             .column(0, 1, 3, 4, 6, 8) // left
             .build();
         CompiledExpression condition = expr.compile()) {
      long rowCount = left.conditionalLeftAntiJoinRowCount(right, condition);
      assertEquals(expected.getRowCount(), rowCount);
      try (GatherMap map =
               left.conditionalLeftAntiJoinGatherMap(right, condition, rowCount)) {
        verifySemiJoinGatherMap(map, expected);
      }
    }
  }

  @Test
  void testConditionalAntiSemiJoinGatherMapNullsWithCount() {
    BinaryOperation expr = new BinaryOperation(BinaryOperator.NULL_EQUAL,
        new ColumnReference(0, TableReference.LEFT),
        new ColumnReference(0, TableReference.RIGHT));
    try (Table left = new Table.TestBuilder()
        .column(2, 3, 9, 0, 1, 7, 4, null, null, 8)
        .build();
         Table right = new Table.TestBuilder()
             .column(null, null, 9, 8, 10, 32)
             .build();
         Table expected = new Table.TestBuilder()
             .column(0, 1, 3, 4, 5, 6) // left
             .build();
         CompiledExpression condition = expr.compile()) {
      long rowCount = left.conditionalLeftAntiJoinRowCount(right, condition);
      assertEquals(expected.getRowCount(), rowCount);
      try (GatherMap map =
               left.conditionalLeftAntiJoinGatherMap(right, condition, rowCount)) {
        verifySemiJoinGatherMap(map, expected);
      }
    }
  }

  @Test
  void testBoundsNulls() {
    boolean[] descFlags = new boolean[1];
    try (Table table = new TestBuilder()
            .column(null, 20, 20, 20, 30)
            .build();
        Table values = new TestBuilder()
            .column(15)
            .build();
         ColumnVector expected = ColumnVector.fromBoxedInts(1)) {
      try (ColumnVector columnVector = getBoundsCv(descFlags, true, table, values)) {
        assertColumnsAreEqual(expected, columnVector);
      }
      try (ColumnVector columnVector = getBoundsCv(descFlags, false, table, values)) {
        assertColumnsAreEqual(expected, columnVector);
      }
    }
  }

  @Test
  void testBoundsValuesSizeBigger() {
    boolean[] descFlags = new boolean[2];
    try(Table table = new TestBuilder()
            .column(90, 100, 120, 130, 135)
            .column(.5, .5, .5, .7, .7)
            .build();
        Table values = new TestBuilder()
            .column(120)
            .column(.3)
            .column(.7)
            .build()) {
      assertThrows(CudfException.class, () -> getBoundsCv(descFlags, true, table, values));
      assertThrows(CudfException.class, () ->  getBoundsCv(descFlags, false, table, values));
    }
  }

  @Test
  void testBoundsInputSizeBigger() {
    boolean[] descFlags = new boolean[3];
    try(Table table = new TestBuilder()
            .column(90, 100, 120, 130, 135)
            .column(.5, .5, .5, .7, .7)
            .column(90, 100, 120, 130, 135)
            .build();
        Table values = new TestBuilder()
            .column(120)
            .column(.3)
            .build()) {
      assertThrows(CudfException.class, () -> getBoundsCv(descFlags, true, table, values));
      assertThrows(CudfException.class, () -> getBoundsCv(descFlags, false, table, values));
    }
  }

  @Test
  void testBoundsMultiCol() {
    boolean[] descFlags = new boolean[4];
    try (Table table = new TestBuilder()
            .column(10, 20, 20, 20, 20)
            .column(5.0, .5, .5, .7, .7)
            .column("1","2","3","4","4")
            .column(90, 77, 78, 61, 61)
            .build();
        Table values = new TestBuilder()
            .column(20)
            .column(0.7)
            .column("4")
            .column(61)
            .build()) {
      try (ColumnVector columnVector = getBoundsCv(descFlags, true, table, values);
           ColumnVector expected = ColumnVector.fromBoxedInts(5)) {
        assertColumnsAreEqual(expected, columnVector);
      }
      try (ColumnVector columnVector = getBoundsCv(descFlags, false, table, values);
           ColumnVector expected = ColumnVector.fromBoxedInts(3)) {
        assertColumnsAreEqual(expected, columnVector);
      }
    }
  }

  @Test
  void testBoundsFloatsMultiVal() {
    boolean[] descFlags = new boolean[1];
    try (Table table = new TestBuilder()
            .column(10.0, 20.6, 20.7)
            .build();
        Table values = new TestBuilder()
            .column(20.3, 20.8)
            .build();
         ColumnVector expected = ColumnVector.fromBoxedInts(1, 3)) {
      try (ColumnVector columnVector = getBoundsCv(descFlags, true, table, values)) {
        assertColumnsAreEqual(expected, columnVector);
      }
      try (ColumnVector columnVector = getBoundsCv(descFlags, false, table, values)) {
        assertColumnsAreEqual(expected, columnVector);
      }
    }
  }

  @Test
  void testBoundsFloatsSingleCol() {
    boolean[] descFlags = {false};
    try(Table table = new TestBuilder()
            .column(10.0, 20.6, 20.7)
            .build();
        Table values = new TestBuilder()
            .column(20.6)
            .build()) {
      try (ColumnVector columnVector = getBoundsCv(descFlags, true, table, values);
           ColumnVector expected = ColumnVector.fromBoxedInts(2)) {
        assertColumnsAreEqual(expected, columnVector);
      }
      try (ColumnVector columnVector = getBoundsCv(descFlags, false, table, values);
           ColumnVector expected = ColumnVector.fromBoxedInts(1)) {
        assertColumnsAreEqual(expected, columnVector);
      }
    }
  }

  @Test
  void testBoundsFloatsSingleColDesc() {
    boolean[] descFlags = new boolean[] {true};
    try(Table table = new TestBuilder()
        .column(20.7, 20.6, 10.0)
        .build();
        Table values = new TestBuilder()
            .column(20.6)
            .build()) {
      try (ColumnVector columnVector = getBoundsCv(descFlags, true, table, values);
           ColumnVector expected = ColumnVector.fromBoxedInts(2)) {
        assertColumnsAreEqual(expected, columnVector);
      }
      try (ColumnVector columnVector = getBoundsCv(descFlags, false, table, values);
           ColumnVector expected = ColumnVector.fromBoxedInts(1)) {
        assertColumnsAreEqual(expected, columnVector);
      }
    }
  }

  @Test
  void testBoundsIntsSingleCol() {
    boolean[] descFlags = new boolean[1];
    try(Table table = new TestBuilder()
            .column(10, 20, 20, 20, 20)
            .build();
        Table values = new TestBuilder()
            .column(20)
            .build()) {
      try (ColumnVector columnVector = getBoundsCv(descFlags, true, table, values);
           ColumnVector expected = ColumnVector.fromBoxedInts(5)) {
        assertColumnsAreEqual(expected, columnVector);
      }
      try (ColumnVector columnVector = getBoundsCv(descFlags, false, table, values);
           ColumnVector expected = ColumnVector.fromBoxedInts(1)) {
        assertColumnsAreEqual(expected, columnVector);
      }
    }
  }

  @Test
  void testBoundsIntsSingleColDesc() {
    boolean[] descFlags = new boolean[]{true};
    try (Table table = new TestBuilder()
        .column(20, 20, 20, 20, 10)
        .build();
         Table values = new TestBuilder()
             .column(5)
             .build();
         ColumnVector expected = ColumnVector.fromBoxedInts(5)) {
      try (ColumnVector columnVector = getBoundsCv(descFlags, true, table, values)) {
        assertColumnsAreEqual(expected, columnVector);
      }
      try (ColumnVector columnVector = getBoundsCv(descFlags, false, table, values)) {
        assertColumnsAreEqual(expected, columnVector);
      }
    }
  }

  @Test
  void testBoundsString() {
    boolean[] descFlags = new boolean[1];
    try (ColumnVector cIn = ColumnVector.build(DType.STRING, 4, (b) -> {
           for (int i = 0; i < 4; i++) {
             b.appendUTF8String(String.valueOf(i).getBytes());
           }
        });
        Table table = new Table(cIn);
        ColumnVector cVal = ColumnVector.fromStrings("0");
        Table values = new Table(cVal)) {
      try (ColumnVector cv = getBoundsCv(descFlags, true, table, values);
           ColumnVector expected = ColumnVector.fromInts(1)) {
        assertColumnsAreEqual(expected, cv);
      }
      try (ColumnVector cv = getBoundsCv(descFlags, false, table, values);
           ColumnVector expected = ColumnVector.fromInts(0)) {
        assertColumnsAreEqual(expected, cv);
      }
    }
  }

  @Test
  void testBoundsEmptyValues() {
    boolean[] descFlags = new boolean[1];
    try (ColumnVector cv = ColumnVector.fromBoxedLongs();
         Table table = new TestBuilder()
             .column(10, 20, 20, 20, 20)
             .build();
         Table values = new Table(cv)) {
      assertThrows(AssertionError.class,
          () -> getBoundsCv(descFlags, true, table, values).close());
      assertThrows(AssertionError.class,
          () -> getBoundsCv(descFlags, false, table, values).close());
    }
  }

  @Test
  void testBoundsEmptyInput() {
    boolean[] descFlags = new boolean[1];
    try (ColumnVector cv =  ColumnVector.fromBoxedLongs();
         Table table = new Table(cv);
         Table values = new TestBuilder()
             .column(20)
             .build()) {
      assertThrows(AssertionError.class,
          () -> getBoundsCv(descFlags, true, table, values).close());
      assertThrows(AssertionError.class,
          () -> getBoundsCv(descFlags, false, table, values).close());
    }
  }

  private ColumnVector getBoundsCv(boolean[] descFlags, boolean isUpperBound,
      Table table, Table values) {
    boolean[] nullsAreSmallest = new boolean[descFlags.length];
    Arrays.fill(nullsAreSmallest, true);
    return isUpperBound ?
        table.upperBound(nullsAreSmallest, values, descFlags) :
        table.lowerBound(nullsAreSmallest, values, descFlags);
  }

  @Test
  void testRepeat() {
    try (Table t = new Table.TestBuilder()
            .column(1, 2)
            .column("a", "b")
            .decimal32Column(-3, 12, -25)
            .decimal64Column(2, 11111L, -22222L)
            .build();
         Table expected = new Table.TestBuilder()
                 .column(1, 1, 1, 2, 2, 2)
                 .column("a", "a", "a", "b", "b", "b")
                 .decimal32Column(-3, 12, 12, 12, -25, -25, -25)
                 .decimal64Column(2, 11111L, 11111L, 11111L, -22222L, -22222L, -22222L)
                 .build();
         Table repeated = t.repeat(3)) {
      assertTablesAreEqual(expected, repeated);
    }
  }

  @Test
  void testRepeatColumn() {
    try (Table t = new Table.TestBuilder()
            .column(1, 2)
            .column("a", "b")
            .decimal32Column(-3, 12, -25)
            .decimal64Column(2, 11111L, -22222L)
            .build();
         ColumnVector repeats = ColumnVector.fromBytes((byte)1, (byte)4);
         Table expected = new Table.TestBuilder()
                 .column(1, 2, 2, 2, 2)
                 .column("a", "b", "b", "b", "b")
                 .decimal32Column(-3, 12, -25, -25, -25, -25)
                 .decimal64Column(2, 11111L, -22222L, -22222L, -22222L, -22222L)
                 .build();
         Table repeated = t.repeat(repeats)) {
      assertTablesAreEqual(expected, repeated);
    }
  }

  @Test
  void testInterleaveIntColumns() {
    try (Table t = new Table.TestBuilder()
          .column(1,2,3,4,5)
          .column(6,7,8,9,10)
          .build();
         ColumnVector expected = ColumnVector.fromInts(1,6,2,7,3,8,4,9,5,10);
         ColumnVector actual = t.interleaveColumns()) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testInterleaveFloatColumns() {
    try (Table t = new Table.TestBuilder()
        .column(1f,2f,3f,4f,5f)
        .column(6f,7f,8f,9f,10f)
        .build();
         ColumnVector expected = ColumnVector.fromFloats(1f,6f,2f,7f,3f,8f,4f,9f,5f,10f);
         ColumnVector actual = t.interleaveColumns()) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testInterleaveDecimalColumns() {
    try (Table t = new Table.TestBuilder()
        .decimal32Column(-2, 123, 456, 789)
        .decimal32Column(-2,-100, -200, -300)
        .build();
         ColumnVector expected = ColumnVector.decimalFromInts(-2, 123, -100, 456, -200, 789, -300);
         ColumnVector actual = t.interleaveColumns()) {
      assertColumnsAreEqual(expected, actual);
    }
    try (Table t = new Table.TestBuilder()
        .decimal64Column(-5, 123456790L, 987654321L)
        .decimal64Column(-5,-123456790L, -987654321L)
        .build();
         ColumnVector expected = ColumnVector.decimalFromLongs(-5, 123456790L, -123456790L, 987654321L, -987654321L);
         ColumnVector actual = t.interleaveColumns()) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testInterleaveStringColumns() {
    try (Table t = new Table.TestBuilder()
        .column("a", "b", "c")
        .column("d", "e", "f")
        .build();
         ColumnVector expected = ColumnVector.fromStrings("a", "d", "b", "e", "c", "f");
         ColumnVector actual = t.interleaveColumns()) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testInterleaveMixedColumns() {
    try (Table t = new Table.TestBuilder()
        .column(1f,2f,3f,4f,5f)
        .column(6,7,8,9,10)
        .build()) {
      assertThrows(CudfException.class, () -> t.interleaveColumns(),
          "All columns must have the same data type in interleave_columns");
    }
  }

  @Test
  void testConcatNoNulls() {
    try (Table t1 = new Table.TestBuilder()
        .column(1, 2, 3)
        .column("1", "2", "3")
        .timestampMicrosecondsColumn(1L, 2L, 3L)
        .column(11.0, 12.0, 13.0)
        .decimal32Column(-3, 1, 2, 3)
        .decimal64Column(-10, 1L, 2L, 3L)
        .build();
         Table t2 = new Table.TestBuilder()
             .column(4, 5)
             .column("4", "3")
             .timestampMicrosecondsColumn(4L, 3L)
             .column(14.0, 15.0)
             .decimal32Column(-3, 4, 5)
             .decimal64Column(-10, 4L, 5L)
             .build();
         Table t3 = new Table.TestBuilder()
             .column(6, 7, 8, 9)
             .column("4", "1", "2", "2")
             .timestampMicrosecondsColumn(4L, 1L, 2L, 2L)
             .column(16.0, 17.0, 18.0, 19.0)
             .decimal32Column(-3, 6, 7, 8, 9)
             .decimal64Column(-10, 6L, 7L, 8L, 9L)
             .build();
         Table concat = Table.concatenate(t1, t2, t3);
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3, 4, 5, 6, 7, 8, 9)
             .column("1", "2", "3", "4", "3", "4", "1", "2", "2")
             .timestampMicrosecondsColumn(1L, 2L, 3L, 4L, 3L, 4L, 1L, 2L, 2L)
             .column(11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0)
             .decimal32Column(-3, 1, 2, 3, 4, 5, 6, 7, 8, 9)
             .decimal64Column(-10, 1L, 2L, 3L, 4L, 5L, 6L, 7L, 8L, 9L)
             .build()) {
      assertTablesAreEqual(expected, concat);
    }
  }

  @Test
  void testConcatWithNulls() {
    try (Table t1 = new Table.TestBuilder()
        .column(1, null, 3)
        .column(11.0, 12.0, 13.0)
        .decimal32Column(-3, 1, null, 3)
        .decimal64Column(-10, 11L, 12L, 13L)
        .build();
         Table t2 = new Table.TestBuilder()
             .column(4, null)
             .column(14.0, 15.0)
             .decimal32Column(-3, 4, null)
             .decimal64Column(-10, 14L, 15L)
             .build();
         Table t3 = new Table.TestBuilder()
             .column(6, 7, 8, 9)
             .column(null, null, 18.0, 19.0)
             .decimal32Column(-3, 6, 7, 8, 9)
             .decimal64Column(-10, null, null, 18L, 19L)
             .build();
         Table concat = Table.concatenate(t1, t2, t3);
         Table expected = new Table.TestBuilder()
             .column(1, null, 3, 4, null, 6, 7, 8, 9)
             .column(11.0, 12.0, 13.0, 14.0, 15.0, null, null, 18.0, 19.0)
             .decimal32Column(-3, 1, null, 3, 4, null, 6, 7, 8, 9)
             .decimal64Column(-10, 11L, 12L, 13L, 14L, 15L, null, null, 18L, 19L)
             .build()) {
      assertTablesAreEqual(expected, concat);
    }
  }

  @Test
  void testContiguousSplit() {
    ContiguousTable[] splits = null;
    try (Table t1 = new Table.TestBuilder()
        .column(10, 12, 14, 16, 18, 20, 22, 24, null, 28)
        .column(50, 52, 54, 56, 58, 60, 62, 64, 66, null)
        .decimal32Column(-3, 10, 12, 14, 16, 18, 20, 22, 24, null, 28)
        .decimal64Column(-8, 50L, 52L, 54L, 56L, 58L, 60L, 62L, 64L, 66L, null)
        .build()) {
      splits = t1.contiguousSplit(2, 5, 9);
      assertEquals(4, splits.length);
      assertEquals(2, splits[0].getRowCount());
      assertEquals(2, splits[0].getTable().getRowCount());
      assertEquals(3, splits[1].getRowCount());
      assertEquals(3, splits[1].getTable().getRowCount());
      assertEquals(4, splits[2].getRowCount());
      assertEquals(4, splits[2].getTable().getRowCount());
      assertEquals(1, splits[3].getRowCount());
      assertEquals(1, splits[3].getTable().getRowCount());
    } finally {
      if (splits != null) {
        for (int i = 0; i < splits.length; i++) {
          splits[i].close();
        }
      }
    }
  }

  @Test
  void testChunkedPackBasic() {
    try (Table t1 = new Table.TestBuilder()
        .column(10, 12, 14, 16, 18, 20, 22, 24, null, 28)
        .column(50, 52, 54, 56, 58, 60, 62, 64, 66, null)
        .decimal32Column(-3, 10, 12, 14, 16, 18, 20, 22, 24, null, 28)
        .decimal64Column(-8, 50L, 52L, 54L, 56L, 58L, 60L, 62L, 64L, 66L, null)
        .build();
        DeviceMemoryBuffer bounceBuffer = DeviceMemoryBuffer.allocate(10L*1024*1024);
        ChunkedPack cp = t1.makeChunkedPack(10L*1024*1024);
        PackedColumnMetadata meta = cp.buildMetadata()) {

      // unpack to bounce buffer
      assertEquals(true, cp.hasNext());
      assertEquals(cp.getTotalContiguousSize(), cp.next(bounceBuffer));
      assertEquals(false, cp.hasNext());

      try (Table unpacked = Table.fromPackedTable(meta.getMetadataDirectBuffer(), bounceBuffer)) {
        assertTablesAreEqual(t1, unpacked);
      }
    }
  }

  @Test
  void testChunkedPackTwoPasses() {
    // this test packes ~2MB worth of long into a 1MB bounce buffer
    // this is 3 iterations because of the validity buffer
    Long[] longs = new Long[256*1024];
    // Initialize elements at odd-numbered indices
    for (int i = 1; i < longs.length; i += 2) {
      longs[i] = (long)i;
    }
    try (Table t1 = new Table.TestBuilder().column(longs).build();
         DeviceMemoryBuffer bounceBuffer = DeviceMemoryBuffer.allocate(1L*1024*1024);
         ChunkedPack cp = t1.makeChunkedPack(1L*1024*1024);
         PackedColumnMetadata meta = cp.buildMetadata();
         DeviceMemoryBuffer target = DeviceMemoryBuffer.allocate(cp.getTotalContiguousSize())) {
      long offset = 0;

      // unpack to bounce buffer
      assertEquals(true, cp.hasNext());
      while (cp.hasNext()) {
        long copied = cp.next(bounceBuffer);
        target.copyFromDeviceBufferAsync(
          offset, bounceBuffer, 0, copied, Cuda.DEFAULT_STREAM);
        offset += copied;
      }

      assertEquals(offset, cp.getTotalContiguousSize());

      try (Table unpacked = Table.fromPackedTable(meta.getMetadataDirectBuffer(), target)) {
        assertTablesAreEqual(t1, unpacked);
      }
    }
  }

  @Test
  void testContiguousSplitWithStrings() {
    ContiguousTable[] splits = null;
    try (Table t1 = new Table.TestBuilder()
        .column(10, 12, 14, 16, 18, 20, 22, 24, null, 28)
        .column(50, 52, 54, 56, 58, 60, 62, 64, 66, null)
        .column("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
        .decimal32Column(-3, 10, 12, 14, 16, 18, 20, 22, 24, null, 28)
        .decimal64Column(-8, 50L, 52L, 54L, 56L, 58L, 60L, 62L, 64L, 66L, null)
        .build()) {
      splits = t1.contiguousSplit(2, 5, 9);
      assertEquals(4, splits.length);
      assertEquals(2, splits[0].getRowCount());
      assertEquals(2, splits[0].getTable().getRowCount());
      assertEquals(3, splits[1].getRowCount());
      assertEquals(3, splits[1].getTable().getRowCount());
      assertEquals(4, splits[2].getRowCount());
      assertEquals(4, splits[2].getTable().getRowCount());
      assertEquals(1, splits[3].getRowCount());
      assertEquals(1, splits[3].getTable().getRowCount());
    } finally {
      if (splits != null) {
        for (int i = 0; i < splits.length; i++) {
          splits[i].close();
        }
      }
    }
  }

  @Test
  void testContiguousSplitWithStringsChunked() {
    try (Table t1 = new Table.TestBuilder()
        .column(10, 12, 14, 16, 18, 20, 22, 24, null, 28)
        .column(50, 52, 54, 56, 58, 60, 62, 64, 66, null)
        .column("A", "B", "C", "D", "E", "F", "G", "H", "I", "J")
        .decimal32Column(-3, 10, 12, 14, 16, 18, 20, 22, 24, null, 28)
        .decimal64Column(-8, 50L, 52L, 54L, 56L, 58L, 60L, 62L, 64L, 66L, null)
        .build();
        DeviceMemoryBuffer bounceBuffer = DeviceMemoryBuffer.allocate(2L*1024*1024);
        ChunkedPack cp = t1.makeChunkedPack(2L*1024*1024);
        PackedColumnMetadata meta = cp.buildMetadata()) {

      // unpack to bounce buffer
      assertEquals(true, cp.hasNext());
      assertEquals(cp.getTotalContiguousSize(), cp.next(bounceBuffer));
      assertEquals(false, cp.hasNext());

      try (Table unpacked = Table.fromPackedTable(meta.getMetadataDirectBuffer(), bounceBuffer)) {
        assertTablesAreEqual(t1, unpacked);
      }
    }
  }

  @Test
  void testPartStability() {
    final int PARTS = 5;
    int expectedPart = -1;
    try (Table start = new Table.TestBuilder().column(0).build();
         PartitionedTable out = start.onColumns(0).hashPartition(PARTS)) {
      // Lets figure out what partition this is a part of.
      // With num_partitions + 1 offsets, partition i has size = parts[i+1] - parts[i]
      int[] parts = out.getPartitions();
      for (int i = 0; i < PARTS; i++) {
        if (parts[i + 1] - parts[i] > 0) {
          expectedPart = i;
          break;
        }
      }
    }
    final int COUNT = 20;
    for (int numEntries = 1; numEntries < COUNT; numEntries++) {
      try (ColumnVector data = ColumnVector.build(DType.INT32, numEntries, Range.appendInts(0, numEntries));
           Table t = new Table(data);
           PartitionedTable out = t.onColumns(0).hashPartition(PARTS);
           HostColumnVector tmp = out.getColumn(0).copyToHost()) {
        // Now we need to get the range out for the partition we expect
        // With num_partitions + 1 offsets, partition i spans [parts[i], parts[i+1])
        int[] parts = out.getPartitions();
        int start = parts[expectedPart];
        int end = parts[expectedPart + 1];
        boolean found = false;
        for (int i = start; i < end; i++) {
          if (tmp.getInt(i) == 0) {
            found = true;
            break;
          }
        }
        assertTrue(found);
      }
    }
  }

  @Test
  void testPartition() {
    try (Table t = new Table.TestBuilder()
        .column(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        .build();
         ColumnVector parts = ColumnVector
             .fromInts(1, 2, 1, 2, 1, 2, 1, 2, 1, 2);
         PartitionedTable pt = t.partition(parts, 3)) {
      assertArrayEquals(new int[]{0, 0, 5, 10}, pt.getPartitions());
      // order within partitions is not guaranteed, so sort each partition to compare
      ColumnVector[] slicedColumns = pt.getTable().getColumn(0).slice(0, 5, 5, 10);
      try (Table part1 = new Table(slicedColumns[0]);
           Table part1Sorted = part1.orderBy(OrderByArg.asc(0));
           Table part1Expected = new Table.TestBuilder().column(1, 3, 5, 7, 9).build();
           Table part2 = new Table(slicedColumns[1]);
           Table part2Sorted = part2.orderBy(OrderByArg.asc(0));
           Table part2Expected = new Table.TestBuilder().column(2, 4, 6, 8, 10).build()) {
        assertTablesAreEqual(part1Expected, part1Sorted);
        assertTablesAreEqual(part2Expected, part2Sorted);
      } finally {
        for (ColumnVector c : slicedColumns) {
          c.close();
        }
      }
    }
  }

  @Test
  void testIdentityHashPartition() {
    final int count = 1024 * 1024;
    try (ColumnVector aIn = ColumnVector.build(DType.INT64, count, Range.appendLongs(count));
         ColumnVector bIn = ColumnVector.build(DType.INT32, count, (b) -> {
           for (int i = 0; i < count; i++) {
             b.append(i / 2);
           }
         });
         ColumnVector cIn = ColumnVector.build(DType.STRING, count, (b) -> {
           for (int i = 0; i < count; i++) {
             b.appendUTF8String(String.valueOf(i).getBytes());
           }
         })) {

      HashSet<Long> expected = new HashSet<>();
      for (long i = 0; i < count; i++) {
        expected.add(i);
      }
      try (Table input = new Table(new ColumnVector[]{aIn, bIn, cIn});
           PartitionedTable output = input.onColumns(0).hashPartition(HashType.IDENTITY, 5)) {
        int[] parts = output.getPartitions();
        assertEquals(6, parts.length);
        assertEquals(0, parts[0]);
        int previous = 0;
        long rows = 0;
        for (int i = 1; i < parts.length; i++) {
          assertTrue(parts[i] >= previous);
          rows += parts[i] - previous;
          previous = parts[i];
        }
        assertTrue(rows <= count);
        try (HostColumnVector aOut = output.getColumn(0).copyToHost();
             HostColumnVector bOut = output.getColumn(1).copyToHost();
             HostColumnVector cOut = output.getColumn(2).copyToHost()) {

          for (int i = 0; i < count; i++) {
            long fromA = aOut.getLong(i);
            long fromB = bOut.getInt(i);
            String fromC = cOut.getJavaString(i);
            assertTrue(expected.remove(fromA));
            assertEquals(fromA / 2, fromB);
            assertEquals(String.valueOf(fromA), fromC, "At Index " + i);
          }
          assertTrue(expected.isEmpty());
        }
      }
    }
  }

  @Test
  void testHashPartition() {
    final int count = 1024 * 1024;
    try (ColumnVector aIn = ColumnVector.build(DType.INT64, count, Range.appendLongs(count));
         ColumnVector bIn = ColumnVector.build(DType.INT32, count, (b) -> {
           for (int i = 0; i < count; i++) {
             b.append(i / 2);
           }
         });
         ColumnVector cIn = ColumnVector.build(DType.STRING, count, (b) -> {
           for (int i = 0; i < count; i++) {
             b.appendUTF8String(String.valueOf(i).getBytes());
           }
         })) {

      HashSet<Long> expected = new HashSet<>();
      for (long i = 0; i < count; i++) {
        expected.add(i);
      }
      try (Table input = new Table(new ColumnVector[]{aIn, bIn, cIn});
           PartitionedTable output = input.onColumns(0).hashPartition(5)) {
        int[] parts = output.getPartitions();
        assertEquals(6, parts.length);
        assertEquals(0, parts[0]);
        int previous = 0;
        long rows = 0;
        for (int i = 1; i < parts.length; i++) {
          assertTrue(parts[i] >= previous);
          rows += parts[i] - previous;
          previous = parts[i];
        }
        assertTrue(rows <= count);
        try (HostColumnVector aOut = output.getColumn(0).copyToHost();
             HostColumnVector bOut = output.getColumn(1).copyToHost();
             HostColumnVector cOut = output.getColumn(2).copyToHost()) {

          for (int i = 0; i < count; i++) {
            long fromA = aOut.getLong(i);
            long fromB = bOut.getInt(i);
            String fromC = cOut.getJavaString(i);
            assertTrue(expected.remove(fromA));
            assertEquals(fromA / 2, fromB);
            assertEquals(String.valueOf(fromA), fromC, "At Index " + i);
          }
          assertTrue(expected.isEmpty());
        }
      }
    }
  }

  @Test
  void testSerializationRoundTripEmpty() throws IOException {
    DataType listStringsType = new ListType(true, new BasicType(true, DType.STRING));
    DataType mapType = new ListType(true,
        new StructType(true,
            new BasicType(false, DType.STRING),
            new BasicType(false, DType.STRING)));
    DataType structType = new StructType(true,
        new BasicType(true, DType.INT8),
        new BasicType(false, DType.FLOAT32));
    try (ColumnVector emptyInt = ColumnVector.fromInts();
         ColumnVector emptyDouble = ColumnVector.fromDoubles();
         ColumnVector emptyString = ColumnVector.fromStrings();
         ColumnVector emptyListString = ColumnVector.fromLists(listStringsType);
         ColumnVector emptyMap = ColumnVector.fromLists(mapType);
         ColumnVector emptyStruct = ColumnVector.fromStructs(structType);
         Table t = new Table(emptyInt, emptyInt, emptyDouble, emptyString,
             emptyListString, emptyMap, emptyStruct)) {
      ByteArrayOutputStream bout = new ByteArrayOutputStream();
      JCudfSerialization.writeToStream(t, bout, 0, 0);
      ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
      DataInputStream din = new DataInputStream(bin);
      try (JCudfSerialization.TableAndRowCountPair result = JCudfSerialization.readTableFrom(din)) {
        assertTablesAreEqual(t, result.getTable());
        assertEquals(result.getTable(), result.getContiguousTable().getTable());
        assertNotNull(result.getContiguousTable().getBuffer());
      }
    }
  }

  @Test
  void testSerializationZeroColumns() throws IOException {
    ByteArrayOutputStream bout = new ByteArrayOutputStream();
    JCudfSerialization.writeRowsToStream(bout, 10);
    ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
    try (JCudfSerialization.TableAndRowCountPair result = JCudfSerialization.readTableFrom(bin)) {
      assertNull(result.getTable());
      assertNull(result.getContiguousTable());
      assertEquals(10, result.getNumRows());
    }
  }

  @Test
  void testSerializationZeroColsZeroRows() throws IOException {
    ByteArrayOutputStream bout = new ByteArrayOutputStream();
    JCudfSerialization.writeRowsToStream(bout, 0);
    ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
    try (JCudfSerialization.TableAndRowCountPair result = JCudfSerialization.readTableFrom(bin)) {
      assertNull(result.getTable());
      assertNull(result.getContiguousTable());
      assertEquals(0, result.getNumRows());
    }
  }

  @Test
  void testSerializationRoundTripConcatOnHostEmpty() throws IOException {
    DataType listStringsType = new ListType(true, new BasicType(true, DType.STRING));
    DataType mapType = new ListType(true,
        new StructType(true,
            new BasicType(false, DType.STRING),
            new BasicType(false, DType.STRING)));
    DataType structType = new StructType(true,
        new BasicType(true, DType.INT8),
        new BasicType(false, DType.FLOAT32));
    try (ColumnVector emptyInt = ColumnVector.fromInts();
         ColumnVector emptyDouble = ColumnVector.fromDoubles();
         ColumnVector emptyString = ColumnVector.fromStrings();
         ColumnVector emptyListString = ColumnVector.fromLists(listStringsType);
         ColumnVector emptyMap = ColumnVector.fromLists(mapType);
         ColumnVector emptyStruct = ColumnVector.fromStructs(structType);
         Table t = new Table(emptyInt, emptyInt, emptyDouble, emptyString,
             emptyListString, emptyMap, emptyStruct)) {
      ByteArrayOutputStream bout = new ByteArrayOutputStream();
      JCudfSerialization.writeToStream(t, bout, 0, 0);
      JCudfSerialization.writeToStream(t, bout, 0, 0);
      ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
      DataInputStream din = new DataInputStream(bin);

      ArrayList<JCudfSerialization.SerializedTableHeader> headers = new ArrayList<>();
      List<HostMemoryBuffer> buffers = new ArrayList<>();
      try {
        JCudfSerialization.SerializedTableHeader head;
        long numRows = 0;
        do {
          head = new JCudfSerialization.SerializedTableHeader(din);
          if (head.wasInitialized()) {
            HostMemoryBuffer buff = hostMemoryAllocator.allocate(head.getDataLen());
            buffers.add(buff);
            JCudfSerialization.readTableIntoBuffer(din, head, buff);
            assert head.wasDataRead();
            numRows += head.getNumRows();
            assert numRows <= Integer.MAX_VALUE;
            headers.add(head);
          }
        } while (head.wasInitialized());
        assert numRows == t.getRowCount();
        try (Table found = JCudfSerialization.readAndConcat(
            headers.toArray(new JCudfSerialization.SerializedTableHeader[headers.size()]),
            buffers.toArray(new HostMemoryBuffer[buffers.size()]))) {
          assertTablesAreEqual(t, found);
        }
      } finally {
        for (HostMemoryBuffer buff: buffers) {
          buff.close();
        }
      }
    }
  }

  @Test
  void testSerializationRoundTripToHostEmpty() throws IOException {
    DataType listStringsType = new ListType(true, new BasicType(true, DType.STRING));
    DataType mapType = new ListType(true,
            new StructType(true,
                    new BasicType(false, DType.STRING),
                    new BasicType(false, DType.STRING)));
    DataType structType = new StructType(true,
            new BasicType(true, DType.INT8),
            new BasicType(false, DType.FLOAT32));
    try (ColumnVector emptyInt = ColumnVector.fromInts();
         ColumnVector emptyDouble = ColumnVector.fromDoubles();
         ColumnVector emptyString = ColumnVector.fromStrings();
         ColumnVector emptyListString = ColumnVector.fromLists(listStringsType);
         ColumnVector emptyMap = ColumnVector.fromLists(mapType);
         ColumnVector emptyStruct = ColumnVector.fromStructs(structType);
         Table t = new Table(emptyInt, emptyInt, emptyDouble, emptyString,
                 emptyListString, emptyMap, emptyStruct)) {
      testSerializationRoundTripToHost(t);
    }
  }

  @Test
  void testRoundRobinPartition() {
    try (Table t = new Table.TestBuilder()
        .column(     100,      202,      3003,    40004,        5,      -60,       1,      null,        3,  null,        5,     null,        7, null,        9,      null,       11,      null,        13,      null,       15)
        .column(    true,     true,     false,    false,     true,     null,     true,     true,     null, false,    false,     null,     true, true,     null,     false,    false,      null,      true,      true,     null)
        .column( (byte)1,  (byte)2,      null,  (byte)4,  (byte)5,  (byte)6,  (byte)1,  (byte)2,  (byte)3,  null,  (byte)5,  (byte)6,  (byte)7, null,  (byte)9,  (byte)10, (byte)11,      null,  (byte)13,  (byte)14, (byte)15)
        .column((short)6, (short)5,  (short)4,     null, (short)2, (short)1, (short)1, (short)2, (short)3,  null, (short)5, (short)6, (short)7, null, (short)9, (short)10,     null, (short)12, (short)13, (short)14,     null)
        .column(      1L,     null,     1001L,      50L,   -2000L,     null,       1L,       2L,       3L,    4L,     null,       6L,       7L,   8L,       9L,      null,      11L,       12L,       13L,       14L,     null)
        .column(   10.1f,      20f, Float.NaN,  3.1415f,     -60f,     null,       1f,       2f,       3f,    4f,       5f,     null,       7f,   8f,       9f,       10f,      11f,      null,       13f,       14f,      15f)
        .column(    10.1,     20.0,      33.1,   3.1415,    -60.5,     null,       1.,       2.,       3.,    4.,       5.,       6.,     null,   8.,       9.,       10.,      11.,       12.,      null,       14.,      15.)
        .timestampDayColumn(99, 100,      101,      102,      103,      104,        1,        2,        3,     4,        5,        6,        7, null,        9,        10,       11,        12,        13,      null,       15)
        .timestampMillisecondsColumn(9L, 1006L, 101L, 5092L, null,      88L,       1L,       2L,       3L,    4L,       5L,       6L,       7L,   8L,     null,       10L,      11L,       12L,       13L,       14L,      15L)
        .timestampSecondsColumn(1L, null,  3L,       4L,       5L,       6L,       1L,       2L,       3L,    4L,       5L,       6L,       7L,   8L,       9L,      null,      11L,       12L,       13L,       14L,      15L)
        .column(     "A",      "B",       "C",      "D",     null, "TESTING",     "1",      "2",      "3",   "4",      "5",      "6",      "7", null,      "9",      "10",     "11",      "12",      "13",      null,     "15")
        .column(     "A",      "A",       "C",      "C",     null, "TESTING",     "1",      "2",      "3",   "4",      "5",      "6",      "7", null,      "9",      "10",     "11",      "12",      "13",      null,     "15")
        .decimal32Column(-3, 100,      202,      3003,    40004,        5,      -60,       1,      null,        3,  null,        5,     null,        7, null,        9,      null,       11,      null,        13,      null,       15)
        .decimal64Column(      -8, 1L,     null,     1001L,      50L,   -2000L,     null,       1L,       2L,       3L,    4L,     null,       6L,       7L,   8L,       9L,      null,      11L,       12L,       13L,       14L,     null)
        .build()) {
      try (Table expectedTable = new Table.TestBuilder()
          .column(     100,   40004,        1,  null,        7,      null,        13,      202,        5,     null,        5, null,       11,      null,      3003,       -60,        3,     null,        9,      null,       15)
          .column(    true,   false,     true, false,     true,     false,      true,     true,     true,     true,    false, true,    false,      true,     false,      null,     null,     null,     null,      null,     null)
          .column( (byte)1, (byte)4,  (byte)1,  null,  (byte)7,  (byte)10,  (byte)13,  (byte)2,  (byte)5,  (byte)2,  (byte)5, null, (byte)11,  (byte)14,      null,   (byte)6,  (byte)3,  (byte)6,  (byte)9,      null, (byte)15)
          .column((short)6,    null, (short)1,  null, (short)7, (short)10, (short)13, (short)5, (short)2, (short)2, (short)5, null,     null, (short)14,  (short)4,  (short)1, (short)3, (short)6, (short)9, (short)12,     null)
          .column(      1L,     50L,       1L,    4L,       7L,      null,       13L,     null,   -2000L,       2L,     null,   8L,      11L,       14L,     1001L,      null,       3L,       6L,       9L,       12L,     null)
          .column(   10.1f, 3.1415f,       1f,    4f,       7f,       10f,       13f,      20f,     -60f,       2f,       5f,   8f,      11f,       14f, Float.NaN,      null,       3f,     null,       9f,      null,      15f)
          .column(    10.1,  3.1415,       1.,    4.,     null,       10.,      null,     20.0,    -60.5,       2.,       5.,   8.,      11.,       14.,      33.1,      null,       3.,       6.,       9.,       12.,      15.)
          .timestampDayColumn(99, 102,      1,     4,        7,        10,        13,      100,      103,        2,        5, null,       11,      null,       101,       104,        3,        6,        9,        12,       15)
          .timestampMillisecondsColumn(9L, 5092L, 1L, 4L,   7L,       10L,       13L,    1006L,     null,       2L,       5L,   8L,      11L,       14L,      101L,       88L,       3L,       6L,     null,       12L,      15L)
          .timestampSecondsColumn(1L, 4L,   1L,   4L,       7L,      null,       13L,     null,       5L,       2L,       5L,   8L,      11L,       14L,        3L,        6L,       3L,       6L,       9L,       12L,      15L)
          .column(     "A",     "D",       "1",  "4",      "7",      "10",      "13",      "B",     null,      "2",      "5", null,     "11",      null,       "C", "TESTING",      "3",      "6",      "9",      "12",     "15")
          .column(     "A",     "C",       "1",  "4",      "7",      "10",      "13",      "A",     null,      "2",      "5", null,     "11",      null,       "C", "TESTING",      "3",      "6",      "9",      "12",     "15")
          .decimal32Column(-3,     100,   40004,        1,  null,        7,      null,        13,      202,        5,     null,        5, null,       11,      null,      3003,       -60,        3,     null,        9,      null,       15)
          .decimal64Column(-8, 1L,     50L,       1L,    4L,       7L,      null,       13L,     null,   -2000L,       2L,     null,   8L,      11L,       14L,     1001L,      null,       3L,       6L,       9L,       12L,     null)
          .build();
           PartitionedTable pt = t.roundRobinPartition(3, 0)) {
        assertTablesAreEqual(expectedTable, pt.getTable());
        int[] parts = pt.getPartitions();
        assertEquals(4, parts.length);
        assertEquals(0, parts[0]);
        assertEquals(7, parts[1]);
        assertEquals(14, parts[2]);
        assertEquals(21, parts[3]);
      }

      try (Table expectedTable = new Table.TestBuilder()
          .column(      3003,       -60,        3,     null,        9,      null,       15,     100,   40004,        1,  null,        7,      null,        13,      202,        5,     null,        5, null,       11,      null)
          .column(     false,      null,     null,     null,     null,      null,     null,    true,   false,     true, false,     true,     false,      true,     true,     true,     true,    false, true,    false,      true)
          .column(      null,   (byte)6,  (byte)3,  (byte)6,  (byte)9,      null, (byte)15, (byte)1, (byte)4,  (byte)1,  null,  (byte)7,  (byte)10,  (byte)13,  (byte)2,  (byte)5,  (byte)2,  (byte)5, null, (byte)11,  (byte)14)
          .column(  (short)4,  (short)1, (short)3, (short)6, (short)9, (short)12,     null,(short)6,    null, (short)1,  null, (short)7, (short)10, (short)13, (short)5, (short)2, (short)2, (short)5, null,     null, (short)14)
          .column(     1001L,      null,       3L,       6L,       9L,       12L,     null,      1L,     50L,       1L,    4L,       7L,      null,       13L,     null,   -2000L,       2L,     null,   8L,      11L,       14L)
          .column( Float.NaN,      null,       3f,     null,       9f,      null,      15f,   10.1f, 3.1415f,       1f,    4f,       7f,       10f,       13f,      20f,     -60f,       2f,       5f,   8f,      11f,       14f)
          .column(      33.1,      null,       3.,       6.,       9.,       12.,      15.,    10.1,  3.1415,       1.,    4.,     null,       10.,      null,     20.0,    -60.5,       2.,       5.,   8.,      11.,       14.)
          .timestampDayColumn(101, 104,         3,        6,        9,        12,       15,      99,     102,        1,     4,        7,        10,        13,      100,      103,        2,        5, null,       11,      null)
          .timestampMillisecondsColumn(101L, 88L, 3L,    6L,     null,       12L,      15L,      9L,   5092L,       1L,    4L,       7L,       10L,       13L,    1006L,     null,       2L,       5L,   8L,      11L,       14L)
          .timestampSecondsColumn(3L, 6L,      3L,       6L,       9L,       12L,      15L,      1L,      4L,       1L,    4L,       7L,      null,       13L,     null,       5L,       2L,       5L,   8L,      11L,       14L)
          .column(       "C", "TESTING",      "3",      "6",      "9",      "12",     "15",     "A",     "D",       "1",  "4",      "7",      "10",      "13",      "B",     null,      "2",      "5", null,     "11",      null)
          .column(       "C", "TESTING",      "3",      "6",      "9",      "12",     "15",     "A",     "C",       "1",  "4",      "7",      "10",      "13",      "A",     null,      "2",      "5", null,     "11",      null)
          .decimal32Column(-3,      3003,       -60,        3,     null,        9,      null,       15,     100,   40004,        1,  null,        7,      null,        13,      202,        5,     null,        5, null,       11,      null)
          .decimal64Column(-8,     1001L,      null,       3L,       6L,       9L,       12L,     null,      1L,     50L,       1L,    4L,       7L,      null,       13L,     null,   -2000L,       2L,     null,   8L,      11L,       14L)
          .build();
           PartitionedTable pt = t.roundRobinPartition(3, 1)) {
        assertTablesAreEqual(expectedTable, pt.getTable());
        int[] parts = pt.getPartitions();
        assertEquals(4, parts.length);
        assertEquals(0, parts[0]);
        assertEquals(7, parts[1]);
        assertEquals(14, parts[2]);
        assertEquals(21, parts[3]);
      }

      try (Table expectedTable = new Table.TestBuilder()
          .column(      202,        5,     null,        5, null,       11,      null,      3003,       -60,        3,     null,        9,      null,       15,     100,   40004,        1,  null,        7,      null,        13)
          .column(     true,     true,     true,    false, true,    false,      true,     false,      null,     null,     null,     null,      null,     null,    true,   false,     true, false,     true,     false,      true)
          .column(  (byte)2,  (byte)5,  (byte)2,  (byte)5, null, (byte)11,  (byte)14,      null,   (byte)6,  (byte)3,  (byte)6,  (byte)9,      null, (byte)15, (byte)1, (byte)4,  (byte)1,  null,  (byte)7,  (byte)10,  (byte)13)
          .column( (short)5, (short)2, (short)2, (short)5, null,     null, (short)14,  (short)4,  (short)1, (short)3, (short)6, (short)9, (short)12,     null,(short)6,    null, (short)1,  null, (short)7, (short)10, (short)13)
          .column(     null,   -2000L,       2L,     null,   8L,      11L,       14L,     1001L,      null,       3L,       6L,       9L,       12L,     null,      1L,     50L,       1L,    4L,       7L,      null,       13L)
          .column(      20f,     -60f,       2f,       5f,   8f,      11f,       14f, Float.NaN,      null,       3f,     null,       9f,      null,      15f,   10.1f, 3.1415f,       1f,    4f,       7f,       10f,       13f)
          .column(     20.0,    -60.5,       2.,       5.,   8.,      11.,       14.,      33.1,      null,       3.,       6.,       9.,       12.,      15.,    10.1,  3.1415,       1.,    4.,     null,       10.,      null)
          .timestampDayColumn(100, 103,       2,        5, null,       11,      null,       101,       104,        3,        6,        9,        12,       15,      99,     102,        1,     4,        7,        10,        13)
          .timestampMillisecondsColumn(1006L, null, 2L, 5L,  8L,      11L,       14L,      101L,      88L,       3L,       6L,      null,       12L,      15L,      9L,   5092L,       1L,    4L,       7L,       10L,       13L)
          .timestampSecondsColumn(null, 5L,  2L,       5L,   8L,      11L,       14L,        3L,        6L,       3L,       6L,       9L,       12L,      15L,      1L,      4L,       1L,    4L,       7L,      null,       13L)
          .column(      "B",     null,      "2",      "5", null,     "11",      null,       "C", "TESTING",      "3",      "6",      "9",      "12",     "15",     "A",     "D",       "1",  "4",      "7",      "10",      "13")
          .column(      "A",     null,      "2",      "5", null,     "11",      null,       "C", "TESTING",      "3",      "6",      "9",      "12",     "15",     "A",     "C",       "1",  "4",      "7",      "10",      "13")
          .decimal32Column(-3,      202,        5,     null,        5, null,       11,      null,      3003,       -60,        3,     null,        9,      null,       15,     100,   40004,        1,  null,        7,      null,        13)
          .decimal64Column(-8,     null,   -2000L,       2L,     null,   8L,      11L,       14L,     1001L,      null,       3L,       6L,       9L,       12L,     null,      1L,     50L,       1L,    4L,       7L,      null,       13L)
          .build();
           PartitionedTable pt = t.roundRobinPartition(3, 2)) {
        assertTablesAreEqual(expectedTable, pt.getTable());
        int[] parts = pt.getPartitions();
        assertEquals(4, parts.length);
        assertEquals(0, parts[0]);
        assertEquals(7, parts[1]);
        assertEquals(14, parts[2]);
        assertEquals(21, parts[3]);
      }
    }
  }

  @Test
  void testSerializationRoundTripConcatHostSide() throws IOException {
    try (Table t = buildTestTable()) {
      for (int sliceAmount = 1; sliceAmount < t.getRowCount(); sliceAmount ++) {
        ByteArrayOutputStream bout = new ByteArrayOutputStream();
        for (int i = 0; i < t.getRowCount(); i += sliceAmount) {
          int len = (int) Math.min(t.getRowCount() - i, sliceAmount);
          JCudfSerialization.writeToStream(t, bout, i, len);
        }
        ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
        DataInputStream din = new DataInputStream(bin);
        ArrayList<JCudfSerialization.SerializedTableHeader> headers = new ArrayList<>();
        List<HostMemoryBuffer> buffers = new ArrayList<>();
        try {
          JCudfSerialization.SerializedTableHeader head;
          long numRows = 0;
          do {
            head = new JCudfSerialization.SerializedTableHeader(din);
            if (head.wasInitialized()) {
              HostMemoryBuffer buff = hostMemoryAllocator.allocate(100 * 1024);
              buffers.add(buff);
              JCudfSerialization.readTableIntoBuffer(din, head, buff);
              assert head.wasDataRead();
              numRows += head.getNumRows();
              assert numRows <= Integer.MAX_VALUE;
              headers.add(head);
            }
          } while (head.wasInitialized());
          assert numRows == t.getRowCount();
          try (Table found = JCudfSerialization.readAndConcat(
              headers.toArray(new JCudfSerialization.SerializedTableHeader[headers.size()]),
              buffers.toArray(new HostMemoryBuffer[buffers.size()]))) {
            assertPartialTablesAreEqual(t, 0, t.getRowCount(), found, true, false);
          }
        } finally {
          for (HostMemoryBuffer buff: buffers) {
            buff.close();
          }
        }
      }
    }
  }

  @Test
  void testSerializationRoundTripToHost() throws IOException {
    try (Table t = buildTestTable()) {
      testSerializationRoundTripToHost(t);
    }
  }

  private void testSerializationRoundTripToHost(Table t) throws IOException {
    long rowCount = t.getRowCount();
    ByteArrayOutputStream bout = new ByteArrayOutputStream();
    JCudfSerialization.writeToStream(t, bout, 0, rowCount);
    ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
    DataInputStream din = new DataInputStream(bin);

    JCudfSerialization.SerializedTableHeader header =
            new JCudfSerialization.SerializedTableHeader(din);
    assertTrue(header.wasInitialized());
    try (HostMemoryBuffer buffer = hostMemoryAllocator.allocate(header.getDataLen())) {
      JCudfSerialization.readTableIntoBuffer(din, header, buffer);
      assertTrue(header.wasDataRead());
      HostColumnVector[] hostColumns =
              JCudfSerialization.unpackHostColumnVectors(header, buffer);
      try {
        assertEquals(t.getNumberOfColumns(), hostColumns.length);
        for (int i = 0; i < hostColumns.length; i++) {
          HostColumnVector actual = hostColumns[i];
          assertEquals(rowCount, actual.getRowCount());
          try (HostColumnVector expected = t.getColumn(i).copyToHost()) {
            assertPartialColumnsAreEqual(expected, 0, rowCount, actual, "COLUMN " + i, true, false);
          }
        }
      } finally {
        for (HostColumnVector c: hostColumns) {
          // close child columns for multiple times should NOT throw exceptions
          for (int i = 0; i < c.getNumChildren(); i++) {
            c.getChildColumnView(i).close();
          }
          c.close();
        }
      }
    }
  }

  @Test
  void testConcatHost() throws IOException {
    try (Table t1 = new Table.TestBuilder()
        .column(
            1, 2, null, 4, 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null,
            1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null)
        .decimal32Column(-3,
            1, 2, null, 4, 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null,
            1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null)
        .build();
         Table expected = new Table.TestBuilder()
             .column(
                 null, 12, 13, 14, null, null,
                 1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null,
                 1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null,
                 1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null,
                 null, 12, 13, 14, null, null,
                 1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null,
                 1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null,
                 1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null)
             .decimal32Column(-3,
                 null, 12, 13, 14, null, null,
                 1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null,
                 1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null,
                 1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null,
                 null, 12, 13, 14, null, null,
                 1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null,
                 1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null,
                 1, 2, null, 4 , 5, 6, 7, 8, 9, 10, null, 12, 13, 14, null, null)
             .build();
         Table t2 = t1.concatenate(t1, t1)) {
      ByteArrayOutputStream out = new ByteArrayOutputStream();
      JCudfSerialization.writeToStream(t2, out, 10, t2.getRowCount() - 10);
      DataInputStream in = new DataInputStream(new ByteArrayInputStream(out.toByteArray()));
      JCudfSerialization.SerializedTableHeader header = new JCudfSerialization.SerializedTableHeader(in);
      assert header.wasInitialized();
      try (HostMemoryBuffer buff = hostMemoryAllocator.allocate(header.getDataLen())) {
        JCudfSerialization.readTableIntoBuffer(in, header, buff);
        assert header.wasDataRead();
        try (Table result = JCudfSerialization.readAndConcat(
            new JCudfSerialization.SerializedTableHeader[] {header, header},
            new HostMemoryBuffer[] {buff, buff})) {
          assertPartialTablesAreEqual(expected, 0, expected.getRowCount(), result, true, false);
        }
      }
    }
  }

  @Test
  void testSerializationRoundTripSlicedHostSide() throws IOException {
    try (Table t = buildTestTable()) {
      for (int sliceAmount = 1; sliceAmount < t.getRowCount(); sliceAmount ++) {
        ByteArrayOutputStream bout = new ByteArrayOutputStream();
        for (int i = 0; i < t.getRowCount(); i += sliceAmount) {
          int len = (int) Math.min(t.getRowCount() - i, sliceAmount);
          JCudfSerialization.writeToStream(t, bout, i, len);
        }
        ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
        DataInputStream din = new DataInputStream(bin);
        ArrayList<JCudfSerialization.SerializedTableHeader> headers = new ArrayList<>();
        List<HostMemoryBuffer> buffers = new ArrayList<>();
        try {
          JCudfSerialization.SerializedTableHeader head;
          long numRows = 0;
          do {
            head = new JCudfSerialization.SerializedTableHeader(din);
            if (head.wasInitialized()) {
              HostMemoryBuffer buff = hostMemoryAllocator.allocate(100 * 1024);
              buffers.add(buff);
              JCudfSerialization.readTableIntoBuffer(din, head, buff);
              assert head.wasDataRead();
              numRows += head.getNumRows();
              assert numRows <= Integer.MAX_VALUE;
              headers.add(head);
            }
          } while (head.wasInitialized());
          assert numRows == t.getRowCount();
          ByteArrayOutputStream bout2 = new ByteArrayOutputStream();
          JCudfSerialization.writeConcatedStream(
              headers.toArray(new JCudfSerialization.SerializedTableHeader[headers.size()]),
              buffers.toArray(new HostMemoryBuffer[buffers.size()]), bout2);
          ByteArrayInputStream bin2 = new ByteArrayInputStream(bout2.toByteArray());
          try (JCudfSerialization.TableAndRowCountPair found = JCudfSerialization.readTableFrom(bin2)) {
            assertPartialTablesAreEqual(t, 0, t.getRowCount(), found.getTable(), true, false);
            assertEquals(found.getTable(), found.getContiguousTable().getTable());
            assertNotNull(found.getContiguousTable().getBuffer());
          }
          JCudfSerialization.TableAndRowCountPair tp = JCudfSerialization.readTableFrom(bin2);
          assertNull(tp.getTable());
          assertNull(tp.getContiguousTable());
        } finally {
          for (HostMemoryBuffer buff: buffers) {
            buff.close();
          }
        }
      }
    }
  }

  @Test
  void testSerializationRoundTripSliced() throws IOException {
    try (Table t = buildTestTable()) {
      for (int sliceAmount = 1; sliceAmount < t.getRowCount(); sliceAmount ++) {
        for (int i = 0; i < t.getRowCount(); i += sliceAmount) {
          ByteArrayOutputStream bout = new ByteArrayOutputStream();
          int len = (int) Math.min(t.getRowCount() - i, sliceAmount);
          JCudfSerialization.writeToStream(t, bout, i, len);
          ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
          try (JCudfSerialization.TableAndRowCountPair found = JCudfSerialization.readTableFrom(bin)) {
            assertPartialTablesAreEqual(t, i, len, found.getTable(), i == 0 && len == t.getRowCount(), false);
            assertEquals(found.getTable(), found.getContiguousTable().getTable());
            assertNotNull(found.getContiguousTable().getBuffer());
          }
          JCudfSerialization.TableAndRowCountPair tp = JCudfSerialization.readTableFrom(bin);
          assertNull(tp.getTable());
          assertNull(tp.getContiguousTable());
        }
      }
    }
  }

  @Test
  void testSerializationReconstructFromMetadata() throws IOException {
    try (Table t = buildTestTable()) {
      ByteArrayOutputStream bout = new ByteArrayOutputStream();
      JCudfSerialization.writeToStream(t, bout, 0, t.getRowCount());
      ByteArrayInputStream bin = new ByteArrayInputStream(bout.toByteArray());
      try (JCudfSerialization.TableAndRowCountPair trcp = JCudfSerialization.readTableFrom(bin)) {
        ContiguousTable contigTable = trcp.getContiguousTable();
        DeviceMemoryBuffer oldbuf = contigTable.getBuffer();
        try (DeviceMemoryBuffer newbuf = oldbuf.sliceWithCopy(0, oldbuf.getLength())) {
          ByteBuffer metadata = contigTable.getMetadataDirectBuffer();
          try (Table newTable = Table.fromPackedTable(metadata, newbuf)) {
            assertTablesAreEqual(t, newTable);
          }
        }
      }
    }
  }

  @Test
  void testValidityFill() {
    byte[] buff = new byte[2];
    buff[0] = 0;
    int bitsToFill = (buff.length * 8) - 1;
    assertEquals(bitsToFill, JCudfSerialization.fillValidity(buff, 1, bitsToFill));
    assertEquals(buff[0], 0xFFFFFFFE);
    assertEquals(buff[1], 0xFFFFFFFF);
  }

  @Test
  void testGroupByScan() {
    try (Table t1 = new Table.TestBuilder()
        .column(  "1",  "1",  "1",  "1",  "1",  "1",  "1",  "2",  "2",  "2",  "2") // GBY Key#0
        .column(   0,    1,    3,    3,    5,    5,    5,    5,    5,    5,    5)  // GBY Key#1
        .column(12.0, 14.0, 13.0, 17.0, 17.0, 17.0, null, null, 11.0, null, 10.0)
        .column(  -9, null,   -5,   0,     4,    4,    8,    2,    2,    2, null)
        .build()) {
      try (Table result = t1
          .groupBy(GroupByOptions.builder()
              .withKeysSorted(true)
              .withKeysDescending(false, false)
              .build(), 0, 1)
          .scan(GroupByScanAggregation.sum().onColumn(2),
              GroupByScanAggregation.count(NullPolicy.INCLUDE).onColumn(2),
              GroupByScanAggregation.min().onColumn(2),
              GroupByScanAggregation.max().onColumn(2),
              GroupByScanAggregation.rank().onColumn(3),
              GroupByScanAggregation.denseRank().onColumn(3),
              GroupByScanAggregation.percentRank().onColumn(3));
           Table expected = new Table.TestBuilder()
               .column(  "1",  "1",  "1",  "1",  "1",  "1",  "1",  "2",  "2",  "2",  "2")
               .column(   0,    1,    3,    3,    5,    5,    5,    5,    5,    5,    5)
               .column(12.0, 14.0, 13.0, 30.0, 17.0, 34.0, null, null, 11.0, null, 21.0)
               .column(   1,    1,    1,    2,    1,    2,    3,    1,    2,    3,    4)
               .column(12.0, 14.0, 13.0, 13.0, 17.0, 17.0, null, null, 11.0, null, 10.0)
               .column(12.0, 14.0, 13.0, 17.0, 17.0, 17.0, null, null, 11.0, null, 11.0)
               .column(   1,    1,    1,    2,    1,    1,    3,    1,    1,    1,    4)
               .column(   1,    1,    1,    2,    1,    1,    2,    1,    1,    1,    2)
               .column( 0.0,  0.0,  0.0,  1.0,  0.0,  0.0,  1.0,  0.0,  0.0,  0.0,  1.0)
               .build()) {
        assertTablesAreEqual(expected, result);
      }
    }
  }

  @Test
  void testGroupByReplaceNulls() {
    try (Table t1 = new Table.TestBuilder()
        .column( "1",  "1",  "1",  "1",  "1",  "1",  "1",  "2",  "2",  "2",  "2")
        .column(   0,    1,    3,    3,    5,    5,    5,    5,    5,    5,    5)
        .column(null, 14.0, 13.0, 17.0, 17.0, 17.0, null, null, 11.0, null, null)
        .build()) {
      try (Table result = t1
          .groupBy(GroupByOptions.builder()
              .withKeysSorted(true)
              .withKeysDescending(false, false)
              .build(), 0, 1)
          .replaceNulls(ReplacePolicy.PRECEDING.onColumn(2),
              ReplacePolicy.FOLLOWING.onColumn(2));
           Table expected = new Table.TestBuilder()
               .column( "1",  "1",  "1",  "1",  "1",  "1",  "1",  "2",  "2",  "2",  "2")
               .column(   0,    1,    3,    3,    5,    5,    5,    5,    5,    5,    5)
               .column(null, 14.0, 13.0, 17.0, 17.0, 17.0, 17.0, null, 11.0, 11.0, 11.0)
               .column(null, 14.0, 13.0, 17.0, 17.0, 17.0, null, 11.0, 11.0, null, null)
               .build()) {
        assertTablesAreEqual(expected, result);
      }
    }
  }

  @Test
  void testGroupByApproxPercentileReproCase() {
    double[] percentiles = {0.25, 0.50, 0.75};
    try (Table t1 = new Table.TestBuilder()
            .column("a", "a", "b", "c", "d")
            .column(1084.0, 1719.0, 15948.0, 148029.0, 1269761.0)
            .build();
         Table t2 = t1
            .groupBy(0)
            .aggregate(GroupByAggregation.createTDigest(100).onColumn(1));
         Table sorted = t2.orderBy(OrderByArg.asc(0));
         ColumnVector actual = sorted.getColumn(1).approxPercentile(percentiles);
         ColumnVector expected = ColumnVector.fromLists(
             new ListType(false, new BasicType(false, DType.FLOAT64)),
             Arrays.asList(1084.0, 1084.0, 1719.0),
             Arrays.asList(15948.0, 15948.0, 15948.0),
             Arrays.asList(148029.0, 148029.0, 148029.0),
             Arrays.asList(1269761.0, 1269761.0, 1269761.0)
         )) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testGroupByApproxPercentile() {
    double[] percentiles = {0.25, 0.50, 0.75};
    try (Table t1 = new Table.TestBuilder()
            .column("a", "a", "a", "b", "b", "b")
            .column(100, 150, 160, 70, 110, 160)
            .build();
      Table t2 = t1
          .groupBy(0)
          .aggregate(GroupByAggregation.createTDigest(1000).onColumn(1));
      Table sorted = t2.orderBy(OrderByArg.asc(0));
      ColumnVector actual = sorted.getColumn(1).approxPercentile(percentiles);
      ColumnVector expected = ColumnVector.fromLists(
        new ListType(false, new BasicType(false, DType.FLOAT64)),
          Arrays.asList(100d, 150d, 160d),
          Arrays.asList(70d, 110d, 160d)
      )) {
        assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testMergeApproxPercentile() {
    double[] percentiles = {0.25, 0.50, 0.75};
    try (Table t1 = new Table.TestBuilder()
            .column("a", "a", "a", "b", "b", "b")
            .column(100, 150, 160, 70, 110, 160)
            .build();
         Table t2 = t1
                 .groupBy(0)
                 .aggregate(GroupByAggregation.createTDigest(1000).onColumn(1));
         Table t3 = t1
                 .groupBy(0)
                 .aggregate(GroupByAggregation.createTDigest(1000).onColumn(1));
         Table t4 = Table.concatenate(t2, t3);
         Table t5 = t4
                 .groupBy(0)
                 .aggregate(GroupByAggregation.mergeTDigest(1000).onColumn(1));
         Table sorted = t5.orderBy(OrderByArg.asc(0));
         ColumnVector actual = sorted.getColumn(1).approxPercentile(percentiles);
         ColumnVector expected = ColumnVector.fromLists(
                 new ListType(false, new BasicType(false, DType.FLOAT64)),
                 Arrays.asList(100d, 150d, 160d),
                 Arrays.asList(70d, 110d, 160d)
         )) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testMergeApproxPercentile2() {
    double[] percentiles = {0.25, 0.50, 0.75};
    try (Table t1 = new Table.TestBuilder()
            .column("a", "a", "a", "b", "b", "b")
            .column(70, 110, 160, 100, 150, 160)
            .build();
         Table t2 = t1
                 .groupBy(0)
                 .aggregate(GroupByAggregation.createTDigest(1000).onColumn(1));
         Table t3 = t1
                 .groupBy(0)
                 .aggregate(GroupByAggregation.createTDigest(1000).onColumn(1));
         Table t4 = Table.concatenate(t2, t3);
         Table t5 = t4
                 .groupBy(0)
                 .aggregate(GroupByAggregation.mergeTDigest(1000).onColumn(1));
         Table sorted = t5.orderBy(OrderByArg.asc(0));
         ColumnVector actual = sorted.getColumn(1).approxPercentile(percentiles);
         ColumnVector expected = ColumnVector.fromLists(
                 new ListType(false, new BasicType(false, DType.FLOAT64)),
                 Arrays.asList(70d, 110d, 160d),
                 Arrays.asList(100d, 150d, 160d)
         )) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testCreateTDigestReduction() {
    try (Table t1 = new Table.TestBuilder()
            .column(100, 150, 160, 70, 110, 160)
            .build();
         Scalar tdigest = t1.getColumn(0)
            .reduce(ReductionAggregation.createTDigest(1000), DType.STRUCT)) {
      assertEquals(DType.STRUCT, tdigest.getType());

      try (CloseableArray columns = CloseableArray.wrap(tdigest.getChildrenFromStructScalar())) {
        assertEquals(3, columns.size());
        try (HostColumnVector centroids = ((ColumnView) columns.get(0)).copyToHost();
           HostColumnVector min = ((ColumnView) columns.get(1)).copyToHost();
           HostColumnVector max = ((ColumnView) columns.get(2)).copyToHost()) {
          assertEquals(DType.LIST, centroids.getType());
          assertEquals(DType.FLOAT64, min.getType());
          assertEquals(DType.FLOAT64, max.getType());
          assertEquals(1, min.getRowCount());
          assertEquals(1, max.getRowCount());
          assertEquals(70, min.getDouble(0));
          assertEquals(160, max.getDouble(0));
        }
      }
    }
  }

  @Test
  void testMergeTDigestReduction() {
    StructType centroidStruct = new StructType(false,
            new BasicType(false, DType.FLOAT64), // mean
            new BasicType(false, DType.FLOAT64)); // weight

    ListType centroidList = new ListType(false, centroidStruct);

    StructType tdigestType = new StructType(false,
            centroidList,
            new BasicType(false, DType.FLOAT64), // min
            new BasicType(false, DType.FLOAT64)); // max

    try (ColumnVector tdigests = ColumnVector.fromStructs(tdigestType,
            new StructData(Arrays.asList(
                    new StructData(1.0, 100.0),
                    new StructData(2.0, 50.0)),
                    1.0, // min
                    2.0), // max
            new StructData(Arrays.asList(
                    new StructData(3.0, 200.0),
                    new StructData(4.0, 99.0)),
                    3.0, // min
                    4.0)); // max
      Scalar merged = tdigests.reduce(ReductionAggregation.mergeTDigest(1000), DType.STRUCT)) {

      assertEquals(DType.STRUCT, merged.getType());
      try (CloseableArray columns = CloseableArray.wrap(merged.getChildrenFromStructScalar())) {
        assertEquals(3, columns.size());
        try (HostColumnVector centroids = ((ColumnView) columns.get(0)).copyToHost();
             HostColumnVector min = ((ColumnView) columns.get(1)).copyToHost();
             HostColumnVector max = ((ColumnView) columns.get(2)).copyToHost()) {
          assertEquals(3, columns.size());
          assertEquals(DType.LIST, centroids.getType());
          assertEquals(DType.FLOAT64, min.getType());
          assertEquals(DType.FLOAT64, max.getType());
          assertEquals(1, min.getRowCount());
          assertEquals(1, max.getRowCount());
          assertEquals(1.0, min.getDouble(0));
          assertEquals(4.0, max.getDouble(0));
          assertEquals(1, centroids.rows);

          List list = centroids.getList(0);
          assertEquals(4, list.size());

          StructData data = (StructData) list.get(0);
          assertEquals(1.0, data.dataRecord.get(0));
          assertEquals(100.0, data.dataRecord.get(1));

          data = (StructData) list.get(1);
          assertEquals(2.0, data.dataRecord.get(0));
          assertEquals(50.0, data.dataRecord.get(1));

          data = (StructData) list.get(2);
          assertEquals(3.0, data.dataRecord.get(0));
          assertEquals(200.0, data.dataRecord.get(1));

          data = (StructData) list.get(3);
          assertEquals(4.0, data.dataRecord.get(0));
          assertEquals(99.0, data.dataRecord.get(1));
        }
      }
    }
  }

  @Test
  void testGroupbyHistogram() {
    StructType histogramStruct = new StructType(false,
        new BasicType(false, DType.INT32), // values
        new BasicType(false, DType.INT64)); // frequencies
    ListType histogramList = new ListType(false, histogramStruct);

    // key = 0: values = [2, 2, -3, -2, 2]
    // key = 1: values = [2, 0, 5, 2, 1]
    // key = 2: values = [-3, 1, 1, 2, 2]
    try (Table input = new Table.TestBuilder()
        .column(2, 0, 2, 1, 1, 1, 0, 0, 0, 1, 2, 2, 1, 0, 2)
        .column(-3, 2, 1, 2, 0, 5, 2, -3, -2, 2, 1, 2, 1, 2, 2)
        .build();
         Table result = input.groupBy(0)
             .aggregate(GroupByAggregation.histogram().onColumn(1));
         Table sortedResult = result.orderBy(OrderByArg.asc(0));
         ColumnVector sortedOutHistograms = sortedResult.getColumn(1).listSortRows(false, false);

         ColumnVector expectedKeys = ColumnVector.fromInts(0, 1, 2);
         ColumnVector expectedHistograms = ColumnVector.fromLists(histogramList,
             Arrays.asList(new StructData(-3, 1L), new StructData(-2, 1L), new StructData(2, 3L)),
             Arrays.asList(new StructData(0, 1L), new StructData(1, 1L), new StructData(2, 2L),
                 new StructData(5, 1L)),
             Arrays.asList(new StructData(-3, 1L), new StructData(1, 2L), new StructData(2, 2L)))
    ) {
      assertColumnsAreEqual(expectedKeys, sortedResult.getColumn(0));
      assertColumnsAreEqual(expectedHistograms, sortedOutHistograms);
    }
  }

  @Test
  void testGroupbyMergeHistogram() {
    StructType histogramStruct = new StructType(false,
        new BasicType(false, DType.INT32), // values
        new BasicType(false, DType.INT64)); // frequencies
    ListType histogramList = new ListType(false, histogramStruct);

    // key = 0: histograms = [[<-3, 1>, <-2, 1>, <2, 3>], [<0, 1>, <1, 1>], [<-3, 3>, <0, 1>, <1, 2>]]
    // key = 1: histograms = [[<-2, 1>, <1, 3>, <2, 2>], [<0, 2>, <1, 1>, <2, 2>]]
    try (Table input = new Table.TestBuilder()
        .column(0, 1, 0, 1, 0)
        .column(histogramStruct,
            new StructData[]{new StructData(-3, 1L), new StructData(-2, 1L), new StructData(2, 3L)},
            new StructData[]{new StructData(-2, 1L), new StructData(1, 3L), new StructData(2, 2L)},
            new StructData[]{new StructData(0, 1L), new StructData(1, 1L)},
            new StructData[]{new StructData(0, 2L), new StructData(1, 1L), new StructData(2, 2L)},
            new StructData[]{new StructData(-3, 3L), new StructData(0, 1L), new StructData(1, 2L)})
        .build();
         Table result = input.groupBy(0)
             .aggregate(GroupByAggregation.mergeHistogram().onColumn(1));
         Table sortedResult = result.orderBy(OrderByArg.asc(0));
         ColumnVector sortedOutHistograms = sortedResult.getColumn(1).listSortRows(false, false);

         ColumnVector expectedKeys = ColumnVector.fromInts(0, 1);
         ColumnVector expectedHistograms = ColumnVector.fromLists(histogramList,
             Arrays.asList(new StructData(-3, 4L), new StructData(-2, 1L), new StructData(0, 2L),
                           new StructData(1, 3L), new StructData(2, 3L)),
             Arrays.asList(new StructData(-2, 1L), new StructData(0, 2L), new StructData(1, 4L),
                           new StructData(2, 4L)))
    ) {
      assertColumnsAreEqual(expectedKeys, sortedResult.getColumn(0));
      assertColumnsAreEqual(expectedHistograms, sortedOutHistograms);
    }
  }

  @Test
  void testReductionHistogram() {
    StructType histogramStruct = new StructType(false,
        new BasicType(false, DType.INT32), // values
        new BasicType(false, DType.INT64)); // frequencies

    try (ColumnVector input = ColumnVector.fromInts(-3, 2, 1, 2, 0, 5, 2, -3, -2, 2, 1);
         Scalar result = input.reduce(ReductionAggregation.histogram(), DType.LIST);
         ColumnVector resultCV = result.getListAsColumnView().copyToColumnVector();
         Table resultTable = new Table(resultCV);
         Table sortedResult = resultTable.orderBy(OrderByArg.asc(0));

         ColumnVector expectedHistograms = ColumnVector.fromStructs(histogramStruct,
             new StructData(-3, 2L), new StructData(-2, 1L), new StructData(0, 1L),
             new StructData(1, 2L), new StructData(2, 4L), new StructData(5, 1L))
    ) {
      assertColumnsAreEqual(expectedHistograms, sortedResult.getColumn(0));
    }
  }

  @Test
  void testReductionMergeHistogram() {
    StructType histogramStruct = new StructType(false,
        new BasicType(false, DType.INT32), // values
        new BasicType(false, DType.INT64)); // frequencies

    try (ColumnVector input = ColumnVector.fromStructs(histogramStruct,
             new StructData(-3, 2L), new StructData(2, 1L), new StructData(1, 1L),
             new StructData(2, 2L), new StructData(0, 4L), new StructData(5, 1L),
             new StructData(2, 2L), new StructData(-3, 3L), new StructData(-2, 5L),
             new StructData(2, 3L), new StructData(1, 4L));
         Scalar result = input.reduce(ReductionAggregation.mergeHistogram(), DType.LIST);
         ColumnVector resultCV = result.getListAsColumnView().copyToColumnVector();
         Table resultTable = new Table(resultCV);
         Table sortedResult = resultTable.orderBy(OrderByArg.asc(0));

         ColumnVector expectedHistograms = ColumnVector.fromStructs(histogramStruct,
             new StructData(-3, 5L), new StructData(-2, 5L), new StructData(0, 4L),
             new StructData(1, 5L), new StructData(2, 8L), new StructData(5, 1L))
    ) {
      assertColumnsAreEqual(expectedHistograms, sortedResult.getColumn(0));
    }
  }
  @Test
  void testGroupByMinMaxDecimal() {
    try (Table t1 = new Table.TestBuilder()
        .column( "1",  "1", "1", "1", "2")
        .column(0, 1, 3 , 3, 4)
        .decimal128Column(-4, RoundingMode.HALF_UP,
            new BigInteger("123456789123456789"),
            new BigInteger("7979879879879798"),
            new BigInteger("17979879879879798"),
            new BigInteger("2234563472398472398"),
            null)
        .build()) {
      try (Table result = t1
          .groupBy(GroupByOptions.builder()
              .withKeysSorted(true)
              .withKeysDescending(false, false)
              .build(), 0, 1)
          .scan(GroupByScanAggregation.min().onColumn(2),
              GroupByScanAggregation.max().onColumn(2));
           Table expected = new Table.TestBuilder()
               .column( "1",  "1", "1", "1", "2")
               .column(0, 1, 3, 3, 4)
               .decimal128Column(-4, RoundingMode.HALF_UP,
                   new BigInteger("123456789123456789"),
                   new BigInteger("7979879879879798"),
                   new BigInteger("17979879879879798"),
                   new BigInteger("17979879879879798"),
                   null)
               .decimal128Column(-4, RoundingMode.HALF_UP,
                   new BigInteger("123456789123456789"),
                   new BigInteger("7979879879879798"),
                   new BigInteger("17979879879879798"),
                   new BigInteger("2234563472398472398"),
                   null)
               .build()) {
        assertTablesAreEqual(expected, result);
      }
    }
  }

  @Test
  void testGroupByMinMaxDecimalAgg() {
    try (Table t1 = new Table.TestBuilder()
        .column(-341142443, 48424546)
        .decimal128Column(-2, RoundingMode.HALF_DOWN,
            new BigInteger("2978603952268112009"),
            new BigInteger("571526248386900094"))
        .build()) {
      try (Table result = t1
          .groupBy(GroupByOptions.builder()
              .build(), 0)
          .aggregate(GroupByAggregation.max().onColumn(1));
           Table expected = new Table.TestBuilder()
               .column(-341142443, 48424546)
               .decimal128Column(-2, RoundingMode.HALF_DOWN,
                   new BigInteger("2978603952268112009"),
                   new BigInteger("571526248386900094"))
               .build()) {
        assertTablesAreEqual(expected, result);
      }
    }
  }

  @Test
  void testGroupByCountDecimal() {
    try (Table t1 = new Table.TestBuilder()
        .column( "1",  "1", "1", "1", "2")
        .column(0, 1, 3 , 3, 4)
        .decimal128Column(-4, RoundingMode.HALF_UP,
            new BigInteger("123456789123456789"),
            new BigInteger("7979879879879798"),
            new BigInteger("17979879879879798"),
            new BigInteger("2234563472398472398"),
            null)
        .build()) {
      try (Table result = t1
          .groupBy(GroupByOptions.builder()
              .withKeysSorted(true)
              .withKeysDescending(false, false)
              .build(), 0, 1)
          .aggregate(GroupByAggregation.count().onColumn(2));
           Table expected = new Table.TestBuilder()
               .column( "1",  "1", "1", "2")
               .column(0, 1, 3, 4)
               .column(1, 1, 2, 0)
               .build()) {
        assertTablesAreEqual(expected, result);
      }
    }
  }

  @Test
  void testGroupByUniqueCount() {
    try (Table t1 = new Table.TestBuilder()
            .column( "1",  "1",  "1",  "1",  "1",  "1")
            .column(   1,    3,    3,    5,    5,    0)
            .column(12.0, 14.0, 13.0, 17.0, 17.0, 17.0)
            .build()) {
      try (Table t3 = t1
              .groupBy(0, 1)
              .aggregate(GroupByAggregation.nunique().onColumn(0));
           Table sorted = t3.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           Table expected = new Table.TestBuilder()
                   .column( "1",  "1",  "1",  "1")
                   .column(   0,    1,    3,    5)
                   .column(   1,    1,    1,    1)
                   .build()) {
        assertTablesAreEqual(expected, sorted);
      }
    }
  }

  @Test
  void testOrderByDecimal() {
    try (Table t1 = new Table.TestBuilder()
        .column( "1",  "1", "1", "1")
        .column(0, 1, 3 , 3)
        .decimal64Column(4,
            123456L,
            124567L,
            125678L,
            126789L)
        .build()) {
      try (Table sorted = t1.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           Table expected = new Table.TestBuilder()
               .column( "1",  "1", "1", "1")
               .column(   0,    1, 3, 3)
               .decimal64Column(4,
                   123456L,
                   124567L,
                   125678L,
                   126789L)
               .build()) {
        assertTablesAreEqual(expected, sorted);

      }
    }
  }

  @Test
  void testGroupByUniqueCountNulls() {
    try (Table t1 = new Table.TestBuilder()
            .column( "1",  "1",  "1",  "1",  "1",  "1")
            .column(   1,    3,    3,    5,    5,    0)
            .column(null, null, 13.0, null, null, null)
            .build()) {
      try (Table t3 = t1
              .groupBy(0, 1)
              .aggregate(GroupByAggregation.nunique(NullPolicy.INCLUDE).onColumn(0));
           Table sorted = t3.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           Table expected = new Table.TestBuilder()
                   .column( "1",  "1",  "1",  "1")
                   .column(   0,    1,    3,    5)
                   .column(   1,    1,    1,    1)
                   .build()) {
        assertTablesAreEqual(expected, sorted);
      }
    }
  }

  @Test
  void testGroupByCount() {
    try (Table t1 = new Table.TestBuilder().column( "1",  "1",  "1",  "1",  "1",  "1")
                                           .column(   1,    3,    3,    5,    5,    0)
                                           .column(12.0, 14.0, 13.0, 17.0, 17.0, 17.0)
                                           .build()) {
      try (Table t3 = t1.groupBy(0, 1)
          .aggregate(GroupByAggregation.count().onColumn(0));
           HostColumnVector aggOut1 = t3.getColumn(2).copyToHost()) {
        // verify t3
        assertEquals(4, t3.getRowCount());
        Map<Object, Integer> expectedAggregateResult = new HashMap() {
          {
            // value, count
            put(1, 2);
            put(2, 2);
          }
        };
        for (int i = 0; i < 4; ++i) {
          int key = aggOut1.getInt(i);
          assertTrue(expectedAggregateResult.containsKey(key));
          Integer count = expectedAggregateResult.get(key);
          if (count == 1) {
            expectedAggregateResult.remove(key);
          } else {
            expectedAggregateResult.put(key, count - 1);
          }
        }
      }
    }
  }

  @Test
  void testWindowingCount() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
        .column(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6) // OBY Key
        .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6) // Agg Column
        .decimal32Column(-1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // Decimal GBY Key
        .decimal64Column(1, 1L, 1L, 2L, 2L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L) // Decimal OBY Key
        .build()) {
      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           Table decSorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(4), OrderByArg.asc(5));
           ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
           Scalar two = Scalar.fromInt(2);
           Scalar one = Scalar.fromInt(1)) {
        ColumnVector sortedAggColumn = sorted.getColumn(3);
        assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);
        ColumnVector decSortedAggColumn = decSorted.getColumn(3);
        assertColumnsAreEqual(expectSortedAggColumn, decSortedAggColumn);

        try (WindowOptions window = WindowOptions.builder()
            .minPeriods(1)
            .window(two, one)
            .build()) {

          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(RollingAggregation.count().onColumn(3).overWindow(window));
               Table decWindowAggResults = decSorted.groupBy(0, 4)
                   .aggregateWindows(RollingAggregation.count().onColumn(3).overWindow(window));
               ColumnVector expect = ColumnVector.fromBoxedInts(2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3, 2)) {
            assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
            assertColumnsAreEqual(expect, decWindowAggResults.getColumn(0));
          }
        }
      }
    }
  }

  @Test
  void testWindowingMin() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
        .column(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6) // OBY Key
        .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6) // Agg Column
        .decimal32Column(-1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // Decimal GBY Key
        .decimal64Column(1, 1L, 1L, 2L, 2L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L) // Decimal OBY Key
        .decimal64Column(2, 7L, 5L, 1L, 9L, 7L, 9L, 8L, 2L, 8L, 0L, 6L, 6L) // Decimal Agg Column
        .build()) {
      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           Table decSorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(4), OrderByArg.asc(5));
           ColumnVector expectSortedAggCol = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
           ColumnVector expectDecSortedAggCol = ColumnVector.decimalFromLongs(2, 7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
           Scalar two = Scalar.fromInt(2);
           Scalar one = Scalar.fromInt(1)) {
        ColumnVector sortedAggColumn = sorted.getColumn(3);
        assertColumnsAreEqual(expectSortedAggCol, sortedAggColumn);
        ColumnVector decSortedAggColumn = decSorted.getColumn(6);
        assertColumnsAreEqual(expectDecSortedAggCol, decSortedAggColumn);

        try (WindowOptions window = WindowOptions.builder()
            .minPeriods(1)
            .window(two, one)
            .build()) {

          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(RollingAggregation.min().onColumn(3).overWindow(window));
               Table decWindowAggResults = decSorted.groupBy(0, 4)
                   .aggregateWindows(RollingAggregation.min().onColumn(6).overWindow(window));
               ColumnVector expect = ColumnVector.fromBoxedInts(5, 1, 1, 1, 7, 7, 2, 2, 0, 0, 0, 6);
               ColumnVector decExpect = ColumnVector.decimalFromLongs(2, 5, 1, 1, 1, 7, 7, 2, 2, 0, 0, 0, 6)) {
            assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
            assertColumnsAreEqual(decExpect, decWindowAggResults.getColumn(0));
          }
        }
      }
    }
  }

  @Test
  void testWindowingMax() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
        .column(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6) // OBY Key
        .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6) // Agg Column
        .decimal32Column(-1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // Decimal GBY Key
        .decimal64Column(1, 1L, 1L, 2L, 2L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L) // Decimal OBY Key
        .decimal64Column(2, 7L, 5L, 1L, 9L, 7L, 9L, 8L, 2L, 8L, 0L, 6L, 6L) // Decimal Agg Column
        .build()) {
      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           Table decSorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(4), OrderByArg.asc(5));
           ColumnVector expectSortedAggCol = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
           ColumnVector expectDecSortedAggCol = ColumnVector.decimalFromLongs(2, 7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
           Scalar two = Scalar.fromInt(2);
           Scalar one = Scalar.fromInt(1)) {
        ColumnVector sortedAggColumn = sorted.getColumn(3);
        assertColumnsAreEqual(expectSortedAggCol, sortedAggColumn);
        ColumnVector decSortedAggColumn = decSorted.getColumn(6);
        assertColumnsAreEqual(expectDecSortedAggCol, decSortedAggColumn);

        try (WindowOptions window = WindowOptions.builder()
            .minPeriods(1)
            .window(two, one)
            .build()) {

          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(RollingAggregation.max().onColumn(3).overWindow(window));
               Table decWindowAggResults = decSorted.groupBy(0, 4)
                   .aggregateWindows(RollingAggregation.max().onColumn(6).overWindow(window));
               ColumnVector expect = ColumnVector.fromBoxedInts(7, 7, 9, 9, 9, 9, 9, 8, 8, 8, 6, 6);
               ColumnVector decExpect = ColumnVector.decimalFromLongs(2, 7, 7, 9, 9, 9, 9, 9, 8, 8, 8, 6, 6)) {
            assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
            assertColumnsAreEqual(decExpect, decWindowAggResults.getColumn(0));
          }
        }
      }
    }
  }

  @Test
  void testWindowingSum() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
        .column(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6) // OBY Key
        .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6) // Agg Column
        .build()) {
      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
           Scalar two = Scalar.fromInt(2);
           Scalar one = Scalar.fromInt(1)) {
        ColumnVector sortedAggColumn = sorted.getColumn(3);
        assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

        try (WindowOptions window = WindowOptions.builder()
            .minPeriods(1)
            .window(two, one)
            .build()) {

          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(RollingAggregation.sum().onColumn(3).overWindow(window));
               ColumnVector expectAggResult = ColumnVector.fromBoxedLongs(12L, 13L, 15L, 10L, 16L, 24L, 19L, 10L, 8L, 14L, 12L, 12L)) {
            assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
          }
        }
      }
    }
  }

  @Test
  void testWindowingRowNumber() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
        .column(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6) // OBY Key
        .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6) // Agg Column
        .decimal32Column(-1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // Decimal GBY Key
        .decimal64Column(1, 1L, 1L, 2L, 2L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L) // Decimal OBY Key
        .decimal64Column(2, 7L, 5L, 1L, 9L, 7L, 9L, 8L, 2L, 8L, 0L, 6L, 6L) // Decimal Agg Column
        .build()) {
      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           Table decSorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(4), OrderByArg.asc(5));
           ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
           ColumnVector expectDecSortedAggColumn = ColumnVector.decimalFromLongs(2, 7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6)) {
        ColumnVector sortedAggColumn = sorted.getColumn(3);
        assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);
        ColumnVector decSortedAggColumn = decSorted.getColumn(6);
        assertColumnsAreEqual(expectDecSortedAggColumn, decSortedAggColumn);

        WindowOptions.Builder windowBuilder = WindowOptions.builder().minPeriods(1);

        try (Scalar two = Scalar.fromInt(2);
             Scalar one = Scalar.fromInt(1);
             WindowOptions options = windowBuilder.window(two, one).build();
             WindowOptions options1 = windowBuilder.window(two, one).build()) {
          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(RollingAggregation
                  .rowNumber()
                  .onColumn(3)
                  .overWindow(options));
               Table decWindowAggResults = decSorted.groupBy(0, 4)
                   .aggregateWindows(RollingAggregation
                       .rowNumber()
                       .onColumn(6)
                       .overWindow(options1));
               ColumnVector expectAggResult = ColumnVector.fromBoxedInts(1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2)) {
            assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
            assertColumnsAreEqual(expectAggResult, decWindowAggResults.getColumn(0));
          }
        }

        try (Scalar three = Scalar.fromInt(3);
             Scalar two = Scalar.fromInt(2);
             WindowOptions options = windowBuilder.window(three, two).build();
             WindowOptions options1 = windowBuilder.window(three, two).build()) {
          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(RollingAggregation
                  .rowNumber()
                  .onColumn(3)
                  .overWindow(options));
               Table decWindowAggResults = decSorted.groupBy(0, 4)
                   .aggregateWindows(RollingAggregation
                       .rowNumber()
                       .onColumn(6)
                       .overWindow(options1));
               ColumnVector expectAggResult = ColumnVector.fromBoxedInts(1, 2, 3, 3, 1, 2, 3, 3, 1, 2, 3, 3)) {
            assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
            assertColumnsAreEqual(expectAggResult, decWindowAggResults.getColumn(0));
          }
        }

        try (Scalar four = Scalar.fromInt(4);
             Scalar three = Scalar.fromInt(3);
             WindowOptions options = windowBuilder.window(four, three).build();
             WindowOptions options1 = windowBuilder.window(four, three).build()) {
          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(RollingAggregation
                  .rowNumber()
                  .onColumn(3)
                  .overWindow(options));
               Table decWindowAggResults = decSorted.groupBy(0, 4)
                   .aggregateWindows(RollingAggregation
                       .rowNumber()
                       .onColumn(6)
                       .overWindow(options1));
               ColumnVector expectAggResult = ColumnVector.fromBoxedInts(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4)) {
            assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
            assertColumnsAreEqual(expectAggResult, decWindowAggResults.getColumn(0));
          }
        }
       }
    }
  }

  @Test
  void testWindowingCollectList() {
    RollingAggregation aggCollectWithNulls = RollingAggregation.collectList(NullPolicy.INCLUDE);
    RollingAggregation aggCollect = RollingAggregation.collectList();
    try (Scalar two = Scalar.fromInt(2);
         Scalar one = Scalar.fromInt(1);
         WindowOptions winOpts = WindowOptions.builder()
             .minPeriods(1)
             .window(two, one)
             .build()) {
      StructType nestedType = new StructType(false,
          new BasicType(false, DType.INT32), new BasicType(false, DType.STRING));
      try (Table raw = new Table.TestBuilder()
          .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
          .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
          .column(1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8) // OBY Key
          .column(7, 5, 1, 9, 7, 9, 8, 2, null, 0, 6, null) // Agg Column of INT32
          .column(nestedType,                          // Agg Column of Struct
              new StructData(1, "s1"), new StructData(2, "s2"), new StructData(3, "s3"),
              new StructData(4, "s4"), new StructData(11, "s11"), new StructData(22, "s22"),
              new StructData(33, "s33"), new StructData(44, "s44"), new StructData(111, "s111"),
              new StructData(222, "s222"), new StructData(333, "s333"), new StructData(444, "s444")
          ).build();
           ColumnVector expectSortedAggColumn = ColumnVector
               .fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, null, 0, 6, null)) {
        try (Table sorted = raw.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2))) {
          ColumnVector sortedAggColumn = sorted.getColumn(3);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          // Primitive type: INT32
          //  a) including nulls
          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(aggCollectWithNulls.onColumn(3).overWindow(winOpts));
               ColumnVector expected = ColumnVector.fromLists(
                   new ListType(false, new BasicType(false, DType.INT32)),
                   Arrays.asList(7, 5), Arrays.asList(7, 5, 1), Arrays.asList(5, 1, 9), Arrays.asList(1, 9),
                   Arrays.asList(7, 9), Arrays.asList(7, 9, 8), Arrays.asList(9, 8, 2), Arrays.asList(8, 2),
                   Arrays.asList(null, 0), Arrays.asList(null, 0, 6), Arrays.asList(0, 6, null), Arrays.asList(6, null))) {
            assertColumnsAreEqual(expected, windowAggResults.getColumn(0));
          }
          //  b) excluding nulls
          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(aggCollect.onColumn(3).overWindow(winOpts));
               ColumnVector expected = ColumnVector.fromLists(
                   new ListType(false, new BasicType(false, DType.INT32)),
                   Arrays.asList(7, 5), Arrays.asList(7, 5, 1), Arrays.asList(5, 1, 9), Arrays.asList(1, 9),
                   Arrays.asList(7, 9), Arrays.asList(7, 9, 8), Arrays.asList(9, 8, 2), Arrays.asList(8, 2),
                   Arrays.asList(0), Arrays.asList(0, 6), Arrays.asList(0, 6), Arrays.asList(6))) {
            assertColumnsAreEqual(expected, windowAggResults.getColumn(0));
          }

          // Nested type: Struct
          List<StructData>[] expectedNestedData = new List[12];
          expectedNestedData[0] = Arrays.asList(new StructData(1, "s1"), new StructData(2, "s2"));
          expectedNestedData[1] = Arrays.asList(new StructData(1, "s1"), new StructData(2, "s2"), new StructData(3, "s3"));
          expectedNestedData[2] = Arrays.asList(new StructData(2, "s2"), new StructData(3, "s3"), new StructData(4, "s4"));
          expectedNestedData[3] = Arrays.asList(new StructData(3, "s3"), new StructData(4, "s4"));
          expectedNestedData[4] = Arrays.asList(new StructData(11, "s11"), new StructData(22, "s22"));
          expectedNestedData[5] = Arrays.asList(new StructData(11, "s11"), new StructData(22, "s22"), new StructData(33, "s33"));
          expectedNestedData[6] = Arrays.asList(new StructData(22, "s22"), new StructData(33, "s33"), new StructData(44, "s44"));
          expectedNestedData[7] = Arrays.asList(new StructData(33, "s33"), new StructData(44, "s44"));
          expectedNestedData[8] = Arrays.asList(new StructData(111, "s111"), new StructData(222, "s222"));
          expectedNestedData[9] = Arrays.asList(new StructData(111, "s111"), new StructData(222, "s222"), new StructData(333, "s333"));
          expectedNestedData[10] = Arrays.asList(new StructData(222, "s222"), new StructData(333, "s333"), new StructData(444, "s444"));
          expectedNestedData[11] = Arrays.asList(new StructData(333, "s333"), new StructData(444, "s444"));
          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(aggCollect.onColumn(4).overWindow(winOpts));
               ColumnVector expected = ColumnVector.fromLists(
                   new ListType(false, nestedType), expectedNestedData)) {
            assertColumnsAreEqual(expected, windowAggResults.getColumn(0));
          }
        }
      }
    }
  }

  @Test
  void testWindowingCollectSet() {
    RollingAggregation aggCollect = RollingAggregation.collectSet();
    RollingAggregation aggCollectWithEqNulls = RollingAggregation.collectSet(NullPolicy.INCLUDE,
        NullEquality.EQUAL, NaNEquality.UNEQUAL);
    RollingAggregation aggCollectWithUnEqNulls = RollingAggregation.collectSet(NullPolicy.INCLUDE,
        NullEquality.UNEQUAL, NaNEquality.UNEQUAL);
    RollingAggregation aggCollectWithEqNaNs = RollingAggregation.collectSet(NullPolicy.INCLUDE,
        NullEquality.EQUAL, NaNEquality.ALL_EQUAL);

    try (Scalar two = Scalar.fromInt(2);
         Scalar one = Scalar.fromInt(1);
         WindowOptions winOpts = WindowOptions.builder()
             .minPeriods(1)
             .window(two, one)
             .build()) {
      try (Table raw = new Table.TestBuilder()
          .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
          .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
          .column(1, 2, 3, 4, 1, 2, 3, 4, 5, 6, 7, 8) // OBY Key
          .column(5, 5, 1, 1, 1, 4, 3, 4, null, null, 6, 7) // Agg Column of INT32
          .column(1.1, 1.1, null, 2.2, -3.0, 1.3e-7, -3.0, Double.NaN, 1e-3, null, Double.NaN, Double.NaN) // Agg Column of FLOAT64
          .build();
           ColumnVector expectSortedAggColumn = ColumnVector
               .fromBoxedInts(5, 5, 1, 1, 1, 4, 3, 4, null, null, 6, 7)) {
        try (Table sorted = raw.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2))) {
          ColumnVector sortedAggColumn = sorted.getColumn(3);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          // Primitive type: INT32
          //  a) excluding NULLs
          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(aggCollect.onColumn(3).overWindow(winOpts));
               ColumnVector resultSorted = windowAggResults.getColumn(0).listSortRows(false, false);
               ColumnVector expected = ColumnVector.fromLists(
                   new ListType(false, new BasicType(false, DType.INT32)),
                   Arrays.asList(5), Arrays.asList(1, 5), Arrays.asList(1, 5), Arrays.asList(1),
                   Arrays.asList(1, 4), Arrays.asList(1, 3, 4), Arrays.asList(3, 4), Arrays.asList(3, 4),
                   Arrays.asList(), Arrays.asList(6), Arrays.asList(6, 7), Arrays.asList(6, 7))) {
            assertColumnsAreEqual(expected, resultSorted);
          }
          //  b) including NULLs AND NULLs are equal
          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(aggCollectWithEqNulls.onColumn(3).overWindow(winOpts));
               ColumnVector resultSorted = windowAggResults.getColumn(0).listSortRows(false, false);
               ColumnVector expected = ColumnVector.fromLists(
                   new ListType(false, new BasicType(false, DType.INT32)),
                   Arrays.asList(5), Arrays.asList(1, 5), Arrays.asList(1, 5), Arrays.asList(1),
                   Arrays.asList(1, 4), Arrays.asList(1, 3, 4), Arrays.asList(3, 4), Arrays.asList(3, 4),
                   Arrays.asList((Integer) null), Arrays.asList(6, null), Arrays.asList(6, 7, null), Arrays.asList(6, 7))) {
            assertColumnsAreEqual(expected, resultSorted);
          }
          //  c) including NULLs AND NULLs are unequal
          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(aggCollectWithUnEqNulls.onColumn(3).overWindow(winOpts));
               ColumnVector resultSorted = windowAggResults.getColumn(0).listSortRows(false, false);
               ColumnVector expected = ColumnVector.fromLists(
                   new ListType(false, new BasicType(false, DType.INT32)),
                   Arrays.asList(5), Arrays.asList(1, 5), Arrays.asList(1, 5), Arrays.asList(1),
                   Arrays.asList(1, 4), Arrays.asList(1, 3, 4), Arrays.asList(3, 4), Arrays.asList(3, 4),
                   Arrays.asList(null, null), Arrays.asList(6, null, null), Arrays.asList(6, 7, null), Arrays.asList(6, 7))) {
            assertColumnsAreEqual(expected, resultSorted);
          }

          // Primitive type: FLOAT64
          //  a) excluding NULLs
          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(aggCollect.onColumn(4).overWindow(winOpts));
               ColumnVector resultSorted = windowAggResults.getColumn(0).listSortRows(false, false);
               ColumnVector expected = ColumnVector.fromLists(
                   new ListType(false, new BasicType(false, DType.FLOAT64)),
                   Arrays.asList(1.1), Arrays.asList(1.1), Arrays.asList(1.1, 2.2), Arrays.asList(2.2),
                   Arrays.asList(-3.0, 1.3e-7), Arrays.asList(-3.0, 1.3e-7),
                   Arrays.asList(-3.0, 1.3e-7, Double.NaN), Arrays.asList(-3.0, Double.NaN),
                   Arrays.asList(1e-3), Arrays.asList(1e-3, Double.NaN),
                   Arrays.asList(Double.NaN, Double.NaN), Arrays.asList(Double.NaN, Double.NaN))) {
            assertColumnsAreEqual(expected, resultSorted);
          }
          //  b) including NULLs AND NULLs are equal
          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(aggCollectWithEqNulls.onColumn(4).overWindow(winOpts));
               ColumnVector resultSorted = windowAggResults.getColumn(0).listSortRows(false, false);
               ColumnVector expected = ColumnVector.fromLists(
                   new ListType(false, new BasicType(false, DType.FLOAT64)),
                   Arrays.asList(1.1), Arrays.asList(1.1, null), Arrays.asList(1.1, 2.2, null), Arrays.asList(2.2, null),
                   Arrays.asList(-3.0, 1.3e-7), Arrays.asList(-3.0, 1.3e-7),
                   Arrays.asList(-3.0, 1.3e-7, Double.NaN), Arrays.asList(-3.0, Double.NaN),
                   Arrays.asList(1e-3, null), Arrays.asList(1e-3, Double.NaN, null),
                   Arrays.asList(Double.NaN, Double.NaN, null), Arrays.asList(Double.NaN, Double.NaN))) {
            assertColumnsAreEqual(expected, resultSorted);
          }
          //  c) including NULLs AND NULLs are equal AND NaNs are equal
          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(aggCollectWithEqNaNs.onColumn(4).overWindow(winOpts));
               ColumnVector resultSorted = windowAggResults.getColumn(0).listSortRows(false, false);
               ColumnVector expected = ColumnVector.fromLists(
                   new ListType(false, new BasicType(false, DType.FLOAT64)),
                   Arrays.asList(1.1), Arrays.asList(1.1, null), Arrays.asList(1.1, 2.2, null), Arrays.asList(2.2, null),
                   Arrays.asList(-3.0, 1.3e-7), Arrays.asList(-3.0, 1.3e-7),
                   Arrays.asList(-3.0, 1.3e-7, Double.NaN), Arrays.asList(-3.0, Double.NaN),
                   Arrays.asList(1e-3, null), Arrays.asList(1e-3, Double.NaN, null),
                   Arrays.asList(Double.NaN, null), Arrays.asList(Double.NaN))) {
            assertColumnsAreEqual(expected, resultSorted);
          }
        }
      }
    }
  }

  @Test
  void testWindowingLead() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
        .column(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6) // OBY Key
        .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6) // Int Agg Column
        .decimal32Column(-1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // Decimal GBY Key
        .decimal64Column(1, 1L, 1L, 2L, 2L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L) // Decimal OBY Key
        .decimal64Column(-2, 7L, 5L, 1L, 9L, 7L, 9L, 8L, 2L, 8L, 0L, 6L, 6L) // Decimal Agg Column
        .column(new ListType(false, new BasicType(true, DType.INT32)),
            Arrays.asList(11, 12, null, 13), Arrays.asList(14, null, 15, null), Arrays.asList((Integer) null),  Arrays.asList(16),
            Arrays.asList(21, null, null, 22), Arrays.asList(23, 24), Arrays.asList(25, 26, 27),  Arrays.asList(28, 29, null),
            Arrays.asList(null, 31), Arrays.asList(32, 33, 34), Arrays.asList(35, 36),  Arrays.asList(37, 38, 39)) // List Agg COLUMN
        .column(new StructType(true,
                new BasicType(true, DType.INT32),
                new BasicType(true, DType.STRING)),
            new StructData(1, "s1"), new StructData(null, "s2"), new StructData(2, null), new StructData(3, "s3"),
            new StructData(11, "s11"), null, new StructData(13, "s13"), new StructData(14, "s14"),
            new StructData(111, "s111"), new StructData(null, "s112"), new StructData(2, "s222"), new StructData(3, "s333")) //STRUCT Agg COLUMN
        .build()) {

      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           Table decSorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(4), OrderByArg.asc(5));
           ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
           ColumnVector expectDecSortedAggColumn = ColumnVector.decimalFromLongs(-2, 7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6)) {
        ColumnVector sortedAggColumn = sorted.getColumn(3);
        assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);
        ColumnVector decSortedAggColumn = decSorted.getColumn(6);
        assertColumnsAreEqual(expectDecSortedAggColumn, decSortedAggColumn);

        WindowOptions.Builder windowBuilder = WindowOptions.builder().minPeriods(1);

        try (Scalar two = Scalar.fromInt(2);
             Scalar one = Scalar.fromInt(1);
             WindowOptions options = windowBuilder.window(two, one).build();
             Table windowAggResults = sorted.groupBy(0, 1)
                 .aggregateWindows(RollingAggregation
                     .lead(0)
                     .onColumn(3) // Int Agg Column
                     .overWindow(options));
             Table decWindowAggResults = decSorted.groupBy(0, 4)
                 .aggregateWindows(RollingAggregation
                     .lead(0)
                     .onColumn(6) // Decimal Agg Column
                     .overWindow(options));
             Table listWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lead(0)
                     .onColumn(7) // List Agg COLUMN
                     .overWindow(options));
             Table structWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lead(0)
                     .onColumn(8) //STRUCT Agg COLUMN
                     .overWindow(options));
             ColumnVector expectAggResult = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
             ColumnVector decExpectAggResult = ColumnVector.decimalFromLongs(-2, 7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
             ColumnVector listExpectAggResult = ColumnVector.fromLists(
                 new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true, DType.INT32)),
                 Arrays.asList(11, 12, null, 13), Arrays.asList(14, null, 15, null), Arrays.asList((Integer) null), Arrays.asList(16),
                 Arrays.asList(21, null, null, 22), Arrays.asList(23, 24), Arrays.asList(25, 26, 27), Arrays.asList(28, 29, null),
                 Arrays.asList(null, 31), Arrays.asList(32, 33, 34), Arrays.asList(35, 36), Arrays.asList(37, 38, 39));
             ColumnVector structExpectAggResult = ColumnVector.fromStructs(
                 new StructType(true,
                     new BasicType(true, DType.INT32),
                     new BasicType(true, DType.STRING)),
                 new StructData(1, "s1"), new StructData(null, "s2"), new StructData(2, null), new StructData(3, "s3"),
                 new StructData(11, "s11"), null, new StructData(13, "s13"), new StructData(14, "s14"),
                 new StructData(111, "s111"), new StructData(null, "s112"), new StructData(2, "s222"), new StructData(3, "s333"))) {

          assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
          assertColumnsAreEqual(decExpectAggResult, decWindowAggResults.getColumn(0));
          assertColumnsAreEqual(listExpectAggResult, listWindowAggResults.getColumn(0));
          assertColumnsAreEqual(structExpectAggResult, structWindowAggResults.getColumn(0));
        }

        try (Scalar zero = Scalar.fromInt(0);
             Scalar one = Scalar.fromInt(1);
             WindowOptions options = windowBuilder.window(zero, one).build();
             Table windowAggResults = sorted.groupBy(0, 1)
                 .aggregateWindows(RollingAggregation
                     .lead(1)
                     .onColumn(3) //Int Agg COLUMN
                     .overWindow(options));
             Table decWindowAggResults = sorted.groupBy(0, 4)
                 .aggregateWindows(RollingAggregation
                     .lead(1)
                     .onColumn(6) //Decimal Agg COLUMN
                     .overWindow(options));
             Table listWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lead(1)
                     .onColumn(7) //LIST Agg COLUMN
                     .overWindow(options));
             Table structWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lead(1)
                     .onColumn(8) //STRUCT Agg COLUMN
                     .overWindow(options));
             ColumnVector expectAggResult = ColumnVector.fromBoxedInts(5, 1, 9, null, 9, 8, 2, null, 0, 6, 6, null);
             ColumnVector decExpectAggResult = decimalFromBoxedInts(true, -2, 5, 1, 9, null, 9, 8, 2, null, 0, 6, 6, null);
             ColumnVector listExpectAggResult = ColumnVector.fromLists(
                 new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true, DType.INT32)),
                 Arrays.asList(14, null, 15, null), Arrays.asList((Integer) null), Arrays.asList(16), null,
                 Arrays.asList(23, 24), Arrays.asList(25, 26, 27), Arrays.asList(28, 29, null), null,
                 Arrays.asList(32, 33, 34), Arrays.asList(35, 36), Arrays.asList(37, 38, 39), null);
             ColumnVector structExpectAggResult = ColumnVector.fromStructs(
                 new StructType(true,
                     new BasicType(true, DType.INT32),
                     new BasicType(true, DType.STRING)),
                 new StructData(null, "s2"), new StructData(2, null), new StructData(3, "s3"), null,
                 null, new StructData(13, "s13"), new StructData(14, "s14"), null,
                 new StructData(null, "s112"), new StructData(2, "s222"), new StructData(3, "s333"), null)) {
          assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
          assertColumnsAreEqual(decExpectAggResult, decWindowAggResults.getColumn(0));
          assertColumnsAreEqual(listExpectAggResult, listWindowAggResults.getColumn(0));
          assertColumnsAreEqual(structExpectAggResult, structWindowAggResults.getColumn(0));
        }

        try (Scalar zero = Scalar.fromInt(0);
             Scalar one = Scalar.fromInt(1);
             WindowOptions options = windowBuilder.window(zero, one).build();
             ColumnVector defaultOutput = ColumnVector.fromBoxedInts(0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11);
             ColumnVector decDefaultOutput = ColumnVector.decimalFromLongs(-2, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11);
             ColumnVector listDefaultOutput = ColumnVector.fromLists(
                 new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true, DType.INT32)),
                 Arrays.asList(111), Arrays.asList(222), Arrays.asList(333), Arrays.asList(444, null, 555),
                 Arrays.asList(-11), Arrays.asList(-22), Arrays.asList(-33), Arrays.asList(-44),
                 Arrays.asList(6), Arrays.asList(6), Arrays.asList(6), Arrays.asList(6, null, null));
             ColumnVector structDefaultOutput = ColumnVector.fromStructs(
                 new StructType(true,
                     new BasicType(true, DType.INT32),
                     new BasicType(true, DType.STRING)),
                 new StructData(-1, "s1"), new StructData(null, "s2"), new StructData(-2, null), new StructData(-3, "s3"),
                 new StructData(-11, "s11"), null, new StructData(-13, "s13"), new StructData(-14, "s14"),
                 new StructData(-111, "s111"), new StructData(null, "s112"), new StructData(-222, "s222"), new StructData(-333, "s333"));

             Table windowAggResults = sorted.groupBy(0, 1)
                 .aggregateWindows(RollingAggregation
                     .lead(1, defaultOutput)
                     .onColumn(3) //Int Agg COLUMN
                     .overWindow(options));
             Table decWindowAggResults = sorted.groupBy(0, 4)
                 .aggregateWindows(RollingAggregation
                     .lead(1, decDefaultOutput)
                     .onColumn(6) //Decimal Agg COLUMN
                     .overWindow(options));
             Table listWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lead(1, listDefaultOutput)
                     .onColumn(7) //LIST Agg COLUMN
                     .overWindow(options));
             Table structWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lead(1, structDefaultOutput)
                     .onColumn(8) //STRUCT Agg COLUMN
                     .overWindow(options));
             ColumnVector expectAggResult = ColumnVector.fromBoxedInts(5, 1, 9, -3, 9, 8, 2, -7, 0, 6, 6, -11);
             ColumnVector decExpectAggResult = ColumnVector.decimalFromLongs(-2, 5, 1, 9, -3, 9, 8, 2, -7, 0, 6, 6, -11);
             ColumnVector listExpectAggResult = ColumnVector.fromLists(
                 new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true, DType.INT32)),
                 Arrays.asList(14, null, 15, null), Arrays.asList((Integer) null), Arrays.asList(16), Arrays.asList(444, null, 555),
                 Arrays.asList(23, 24), Arrays.asList(25, 26, 27), Arrays.asList(28, 29, null), Arrays.asList(-44),
                 Arrays.asList(32, 33, 34), Arrays.asList(35, 36), Arrays.asList(37, 38, 39), Arrays.asList(6, null, null));
             ColumnVector structExpectAggResult = ColumnVector.fromStructs(
                 new StructType(true,
                     new BasicType(true, DType.INT32),
                     new BasicType(true, DType.STRING)),
                 new StructData(null, "s2"), new StructData(2, null), new StructData(3, "s3"), new StructData(-3, "s3"),
                 null, new StructData(13, "s13"), new StructData(14, "s14"), new StructData(-14, "s14"),
                 new StructData(null, "s112"), new StructData(2, "s222"), new StructData(3, "s333"), new StructData(-333, "s333"))) {
          assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
          assertColumnsAreEqual(decExpectAggResult, decWindowAggResults.getColumn(0));
          assertColumnsAreEqual(listExpectAggResult, listWindowAggResults.getColumn(0));
          assertColumnsAreEqual(structExpectAggResult, structWindowAggResults.getColumn(0));
        }

        // Outside bounds
        try (Scalar zero = Scalar.fromInt(0);
             Scalar one = Scalar.fromInt(1);
             WindowOptions options = windowBuilder.window(zero, one).build();
             Table windowAggResults = sorted.groupBy(0, 1)
                 .aggregateWindows(RollingAggregation
                     .lead(3)
                     .onColumn(3) //Int Agg COLUMN
                     .overWindow(options));
             Table decWindowAggResults = sorted.groupBy(0, 4)
                 .aggregateWindows(RollingAggregation
                     .lead(3)
                     .onColumn(6) //Decimal Agg COLUMN
                     .overWindow(options));
             Table listWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lead(3)
                     .onColumn(7) //LIST Agg COLUMN
                     .overWindow(options));
             Table structWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lead(3)
                     .onColumn(8) //STRUCT Agg COLUMN
                     .overWindow(options));
             ColumnVector expectAggResult = ColumnVector.fromBoxedInts(null, null, null, null, null, null, null, null, null, null, null, null);
             ColumnVector decExpectAggResult = decimalFromBoxedInts(true, -2, null, null, null, null, null, null, null, null, null, null, null, null);
             ColumnVector listExpectAggResult = ColumnVector.fromLists(
                 new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true, DType.INT32)),
                 null, null, null, null, null, null, null, null, null, null, null, null);
             ColumnVector structExpectAggResult = ColumnVector.fromStructs(
                 new StructType(true,
                     new BasicType(true, DType.INT32),
                     new BasicType(true, DType.STRING)),
                 null, null, null, null, null, null, null, null, null, null, null, null)){
          assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
          assertColumnsAreEqual(decExpectAggResult, decWindowAggResults.getColumn(0));
          assertColumnsAreEqual(listExpectAggResult, listWindowAggResults.getColumn(0));
          assertColumnsAreEqual(structExpectAggResult, structWindowAggResults.getColumn(0));
        }
      }
    }
  }

  @Test
  void testWindowingLag() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
        .column(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6) // OBY Key
        .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6) // Agg Column
        .decimal32Column(-1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // Decimal GBY Key
        .decimal64Column(1, 1L, 1L, 2L, 2L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L) // Decimal OBY Key
        .decimal64Column(-2, 7L, 5L, 1L, 9L, 7L, 9L, 8L, 2L, 8L, 0L, 6L, 6L) // Decimal Agg Column
        .column(new ListType(false, new BasicType(true, DType.INT32)),
            Arrays.asList(11, 12, null, 13), Arrays.asList(14, null, 15, null), Arrays.asList((Integer) null),  Arrays.asList(16),
            Arrays.asList(21, null, null, 22), Arrays.asList(23, 24), Arrays.asList(25, 26, 27),  Arrays.asList(28, 29, null),
            Arrays.asList(null, 31), Arrays.asList(32, 33, 34), Arrays.asList(35, 36),  Arrays.asList(37, 38, 39)) // List Agg COLUMN
        .column(new StructType(true,
                new BasicType(true, DType.INT32),
                new BasicType(true, DType.STRING)),
            new StructData(1, "s1"), new StructData(null, "s2"), new StructData(2, null), new StructData(3, "s3"),
            new StructData(11, "s11"), null, new StructData(13, "s13"), new StructData(14, "s14"),
            new StructData(111, "s111"), new StructData(null, "s112"), new StructData(2, "s222"), new StructData(3, "s333")) //STRUCT Agg COLUMN
        .build()) {

      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           Table decSorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(4), OrderByArg.asc(5));
           ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
           ColumnVector decExpectSortedAggColumn = ColumnVector.decimalFromLongs(-2, 7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6)) {
        ColumnVector sortedAggColumn = sorted.getColumn(3);
        assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);
        ColumnVector decSortedAggColumn = decSorted.getColumn(6);
        assertColumnsAreEqual(decExpectSortedAggColumn, decSortedAggColumn);

        WindowOptions.Builder windowBuilder = WindowOptions.builder().minPeriods(1);

        try (Scalar two = Scalar.fromInt(2);
             Scalar one = Scalar.fromInt(1);
             WindowOptions options = windowBuilder.window(two, one).build();
             Table windowAggResults = sorted.groupBy(0, 1)
                 .aggregateWindows(RollingAggregation
                     .lag(0)
                     .onColumn(3) //Int Agg COLUMN
                     .overWindow(options));
             Table decWindowAggResults = sorted.groupBy(0, 4)
                 .aggregateWindows(RollingAggregation
                     .lag(0)
                     .onColumn(6) //Decimal Agg COLUMN
                     .overWindow(options));
             Table listWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lag(0)
                     .onColumn(7) //LIST Agg COLUMN
                     .overWindow(options));
             Table structWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lag(0)
                     .onColumn(8) //STRUCT Agg COLUMN
                     .overWindow(options));
             ColumnVector expectAggResult = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
             ColumnVector decExpectAggResult = ColumnVector.decimalFromLongs(-2, 7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6);
             ColumnVector listExpectAggResult = ColumnVector.fromLists(
                 new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true, DType.INT32)),
                 Arrays.asList(11, 12, null, 13), Arrays.asList(14, null, 15, null), Arrays.asList((Integer) null), Arrays.asList(16),
                 Arrays.asList(21, null, null, 22), Arrays.asList(23, 24), Arrays.asList(25, 26, 27), Arrays.asList(28, 29, null),
                 Arrays.asList(null, 31), Arrays.asList(32, 33, 34), Arrays.asList(35, 36), Arrays.asList(37, 38, 39));
             ColumnVector structExpectAggResult = ColumnVector.fromStructs(
                 new StructType(true,
                     new BasicType(true, DType.INT32),
                     new BasicType(true, DType.STRING)),
                 new StructData(1, "s1"), new StructData(null, "s2"), new StructData(2, null), new StructData(3, "s3"),
                 new StructData(11, "s11"), null, new StructData(13, "s13"), new StructData(14, "s14"),
                 new StructData(111, "s111"), new StructData(null, "s112"), new StructData(2, "s222"), new StructData(3, "s333"))) {
          assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
          assertColumnsAreEqual(decExpectAggResult, decWindowAggResults.getColumn(0));
          assertColumnsAreEqual(listExpectAggResult, listWindowAggResults.getColumn(0));
          assertColumnsAreEqual(structExpectAggResult, structWindowAggResults.getColumn(0));
        }

        try (Scalar zero = Scalar.fromInt(0);
             Scalar two = Scalar.fromInt(2);
             WindowOptions options = windowBuilder.window(two, zero).build();
             Table windowAggResults = sorted.groupBy(0, 1)
                 .aggregateWindows(RollingAggregation
                     .lag(1)
                     .onColumn(3) //Int Agg COLUMN
                     .overWindow(options));
             Table decWindowAggResults = sorted.groupBy(0, 4)
                 .aggregateWindows(RollingAggregation
                     .lag(1)
                     .onColumn(6) //Decimal Agg COLUMN
                     .overWindow(options));
             Table listWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lag(1)
                     .onColumn(7) //LIST Agg COLUMN
                     .overWindow(options));
             Table structWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lag(1)
                     .onColumn(8) //STRUCT Agg COLUMN
                     .overWindow(options));
             ColumnVector expectAggResult = ColumnVector.fromBoxedInts(null, 7, 5, 1, null, 7, 9, 8, null, 8, 0, 6);
             ColumnVector decExpectAggResult = decimalFromBoxedInts(true, -2, null, 7, 5, 1, null, 7, 9, 8, null, 8, 0, 6);
             ColumnVector listExpectAggResult = ColumnVector.fromLists(
                 new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true, DType.INT32)),
                 null, Arrays.asList(11, 12, null, 13), Arrays.asList(14, null, 15, null), Arrays.asList((Integer) null),
                 null, Arrays.asList(21, null, null, 22), Arrays.asList(23, 24), Arrays.asList(25, 26, 27),
                 null, Arrays.asList(null, 31), Arrays.asList(32, 33, 34), Arrays.asList(35, 36));
             ColumnVector structExpectAggResult = ColumnVector.fromStructs(
                 new StructType(true,
                     new BasicType(true, DType.INT32),
                     new BasicType(true, DType.STRING)),
                 null, new StructData(1, "s1"), new StructData(null, "s2"), new StructData(2, null),
                 null, new StructData(11, "s11"), null, new StructData(13, "s13"),
                 null, new StructData(111, "s111"), new StructData(null, "s112"), new StructData(2, "s222"))) {
          assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
          assertColumnsAreEqual(decExpectAggResult, decWindowAggResults.getColumn(0));
          assertColumnsAreEqual(listExpectAggResult, listWindowAggResults.getColumn(0));
          assertColumnsAreEqual(structExpectAggResult, structWindowAggResults.getColumn(0));
        }

        try (Scalar zero = Scalar.fromInt(0);
             Scalar two = Scalar.fromInt(2);
             WindowOptions options = windowBuilder.window(two, zero).build();
             ColumnVector defaultOutput = ColumnVector.fromBoxedInts(0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11);
             ColumnVector decDefaultOutput = ColumnVector.decimalFromLongs(-2, 0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11);
             ColumnVector listDefaultOutput = ColumnVector.fromLists(
                 new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true, DType.INT32)),
                 Arrays.asList(111), Arrays.asList(222), Arrays.asList(333), Arrays.asList(444, null, 555),
                 Arrays.asList(-11), Arrays.asList(-22), Arrays.asList(-33), Arrays.asList(-44),
                 Arrays.asList(6), Arrays.asList(6), Arrays.asList(6), Arrays.asList(6, null, null));
             ColumnVector structDefaultOutput = ColumnVector.fromStructs(
                 new StructType(true,
                     new BasicType(true, DType.INT32),
                     new BasicType(true, DType.STRING)),
                 new StructData(-1, "s1"), new StructData(null, "s2"), new StructData(-2, null), new StructData(-3, "s3"),
                 new StructData(-11, "s11"), null, new StructData(-13, "s13"), new StructData(-14, "s14"),
                 new StructData(-111, "s111"), new StructData(null, "s112"), new StructData(-222, "s222"), new StructData(-333, "s333"));
             Table windowAggResults = sorted.groupBy(0, 1)
                 .aggregateWindows(RollingAggregation
                     .lag(1, defaultOutput)
                     .onColumn(3) //Int Agg COLUMN
                     .overWindow(options));
             Table decWindowAggResults = sorted.groupBy(0, 4)
                 .aggregateWindows(RollingAggregation
                     .lag(1, decDefaultOutput)
                     .onColumn(6) //Decimal Agg COLUMN
                     .overWindow(options));
             Table listWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lag(1, listDefaultOutput)
                     .onColumn(7) //LIST Agg COLUMN
                     .overWindow(options));
             Table structWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lag(1, structDefaultOutput)
                     .onColumn(8) //STRUCT Agg COLUMN
                     .overWindow(options));
             ColumnVector expectAggResult = ColumnVector.fromBoxedInts(0, 7, 5, 1, -4, 7, 9, 8, -8, 8, 0, 6);
             ColumnVector decExpectAggResult = ColumnVector.decimalFromLongs(-2, 0, 7, 5, 1, -4, 7, 9, 8, -8, 8, 0, 6);
             ColumnVector listExpectAggResult = ColumnVector.fromLists(
                 new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true, DType.INT32)),
                 Arrays.asList(111), Arrays.asList(11, 12, null, 13), Arrays.asList(14, null, 15, null), Arrays.asList((Integer) null),
                 Arrays.asList(-11), Arrays.asList(21, null, null, 22), Arrays.asList(23, 24), Arrays.asList(25, 26, 27),
                 Arrays.asList(6), Arrays.asList(null, 31), Arrays.asList(32, 33, 34), Arrays.asList(35, 36));
             ColumnVector structExpectAggResult = ColumnVector.fromStructs(
                 new StructType(true,
                     new BasicType(true, DType.INT32),
                     new BasicType(true, DType.STRING)),
                 new StructData(-1, "s1"), new StructData(1, "s1"), new StructData(null, "s2"), new StructData(2, null),
                 new StructData(-11, "s11"), new StructData(11, "s11"), null, new StructData(13, "s13"),
                 new StructData(-111, "s111"), new StructData(111, "s111"), new StructData(null, "s112"), new StructData(2, "s222"))) {
          assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
          assertColumnsAreEqual(decExpectAggResult, decWindowAggResults.getColumn(0));
          assertColumnsAreEqual(listExpectAggResult, listWindowAggResults.getColumn(0));
          assertColumnsAreEqual(structExpectAggResult, structWindowAggResults.getColumn(0));
        }

        // Outside bounds
        try (Scalar zero = Scalar.fromInt(0);
             Scalar one = Scalar.fromInt(1);
             WindowOptions options = windowBuilder.window(one, zero).build();
             Table windowAggResults = sorted.groupBy(0, 1)
                 .aggregateWindows(RollingAggregation
                     .lag(3)
                     .onColumn(3) //Int Agg COLUMN
                     .overWindow(options));
             Table decWindowAggResults = sorted.groupBy(0, 4)
                 .aggregateWindows(RollingAggregation
                     .lag(3)
                     .onColumn(6) //Decimal Agg COLUMN
                     .overWindow(options));
             Table listWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lag(3)
                     .onColumn(7) //LIST Agg COLUMN
                     .overWindow(options));
             Table structWindowAggResults = sorted.groupBy(0, 1).aggregateWindows(
                 RollingAggregation
                     .lag(3)
                     .onColumn(8) //STRUCT Agg COLUMN
                     .overWindow(options));
             ColumnVector expectAggResult = ColumnVector.fromBoxedInts(null, null, null, null, null, null, null, null, null, null, null, null);
             ColumnVector decExpectAggResult = decimalFromBoxedInts(true, -2, null, null, null, null, null, null, null, null, null, null, null, null);
             ColumnVector listExpectAggResult = ColumnVector.fromLists(
                 new HostColumnVector.ListType(true, new HostColumnVector.BasicType(true, DType.INT32)),
                 null, null, null, null, null, null, null, null, null, null, null, null);
             ColumnVector structExpectAggResult = ColumnVector.fromStructs(
                 new StructType(true,
                     new BasicType(true, DType.INT32),
                     new BasicType(true, DType.STRING)),
                 null, null, null, null, null, null, null, null, null, null, null, null);) {
          assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
          assertColumnsAreEqual(decExpectAggResult, decWindowAggResults.getColumn(0));
          assertColumnsAreEqual(listExpectAggResult, listWindowAggResults.getColumn(0));
          assertColumnsAreEqual(structExpectAggResult, structWindowAggResults.getColumn(0));
        }
      }
    }
  }

  @Test
  void testWindowingMean() {
    try (Table unsorted = new Table.TestBuilder().column( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column( 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
        .column( 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6) // OBY Key
        .column( 7, 5, 3, 7, 7, 9, 8, 4, 8, 0, 4, 8) // Agg Column
        .build()) {
      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           ColumnVector expectedSortedAggCol = ColumnVector.fromBoxedInts(7, 5, 3, 7, 7, 9, 8, 4, 8, 0, 4, 8)) {
        ColumnVector sortedAggColumn = sorted.getColumn(3);
        assertColumnsAreEqual(expectedSortedAggCol, sortedAggColumn);

        try (Scalar one = Scalar.fromInt(1);
             Scalar two = Scalar.fromInt(2);
             WindowOptions window = WindowOptions.builder()
                 .minPeriods(1)
                 .window(two, one)
                 .build()) {

          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(RollingAggregation.mean().onColumn(3).overWindow(window));
               ColumnVector expect = ColumnVector.fromBoxedDoubles(6.0d, 5.0d, 5.0d, 5.0d, 8.0d, 8.0d, 7.0d, 6.0d, 4.0d, 4.0d, 4.0d, 6.0d)) {
            assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
          }
        }
      }
    }
  }

  @Test
  void testWindowingNthElement() {
    final Integer X = null;
    try (Table unsorted = new Table.TestBuilder()
        .column(  1,  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // 0: GBY Key
        .column(  3,  3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1) // 1: GBY Key
        .column( 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0) // 2: OBY Key
        .column(  X,  4, 0, X, 4, X, 9, 7, 7, 3, 5, 7) // 3: Agg Column
        .build()) {
      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           ColumnVector expectedSortedAggCol = ColumnVector.fromBoxedInts(7, 5, 3, 7, 7, 9, X, 4, X, 0, 4, X)) {
        ColumnVector sortedAggColumn = sorted.getColumn(3);
        assertColumnsAreEqual(expectedSortedAggCol, sortedAggColumn);

        try (Scalar one = Scalar.fromInt(1);
             Scalar two = Scalar.fromInt(2);
             WindowOptions window = WindowOptions.builder()
                 .minPeriods(1)
                 .window(two, one)
                 .build()) {

          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(
                RollingAggregation.nth(0, NullPolicy.INCLUDE).onColumn(3).overWindow(window),
                RollingAggregation.nth(-1, NullPolicy.INCLUDE).onColumn(3).overWindow(window),
                RollingAggregation.nth(1, NullPolicy.INCLUDE).onColumn(3).overWindow(window),
                RollingAggregation.nth(0, NullPolicy.EXCLUDE).onColumn(3).overWindow(window),
                RollingAggregation.nth(-1, NullPolicy.EXCLUDE).onColumn(3).overWindow(window),
                RollingAggregation.nth(1, NullPolicy.EXCLUDE).onColumn(3).overWindow(window));
               ColumnVector expect_first = ColumnVector.fromBoxedInts(7, 7, 5, 3, 7, 7, 9, X, X, X, 0, 4);
               ColumnVector expect_last = ColumnVector.fromBoxedInts(5, 3, 7, 7, 9, X, 4, 4, 0, 4, X, X);
               ColumnVector expect_1th  = ColumnVector.fromBoxedInts(5, 5, 3, 7, 9, 9, X, 4, 0, 0, 4, X);
               ColumnVector expect_first_skip_null  =
                 ColumnVector.fromBoxedInts(7, 7, 5, 3, 7, 7, 9, 4, 0, 0, 0, 4);
               ColumnVector expect_last_skip_null  =
                 ColumnVector.fromBoxedInts(5, 3, 7, 7, 9, 9, 4, 4, 0, 4, 4, 4);
               ColumnVector expect_1th_skip_null  =
                 ColumnVector.fromBoxedInts(5, 5, 3, 7, 9, 9, 4, X, X, 4, 4, X)) {
            assertColumnsAreEqual(expect_first, windowAggResults.getColumn(0));
            assertColumnsAreEqual(expect_last, windowAggResults.getColumn(1));
            assertColumnsAreEqual(expect_1th, windowAggResults.getColumn(2));
            assertColumnsAreEqual(expect_first_skip_null, windowAggResults.getColumn(3));
            assertColumnsAreEqual(expect_last_skip_null, windowAggResults.getColumn(4));
            assertColumnsAreEqual(expect_1th_skip_null, windowAggResults.getColumn(5));
          }
        }
      }
    }
  }

  @Test
  void testWindowingOnMultipleDifferentColumns() {
    try (Table unsorted = new Table.TestBuilder()
        .column( 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column( 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
        .column( 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6) // OBY Key
        .column( 7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6) // Agg Column
        .build()) {
      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           ColumnVector expectedSortedAggCol = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6)) {
        ColumnVector sortedAggColumn = sorted.getColumn(3);
        assertColumnsAreEqual(expectedSortedAggCol, sortedAggColumn);

        try (Scalar one = Scalar.fromInt(1);
             Scalar two = Scalar.fromInt(2);
             Scalar three = Scalar.fromInt(3);
             // Window (1,1), with a minimum of 1 reading.
             WindowOptions window_1 = WindowOptions.builder()
                 .minPeriods(1)
                 .window(two, one)
                 .build();

             // Window (2,2), with a minimum of 2 readings.
             WindowOptions window_2 = WindowOptions.builder()
                 .minPeriods(2)
                 .window(three, two)
                 .build();

             // Window (1,1), with a minimum of 3 readings.
             WindowOptions window_3 = WindowOptions.builder()
                 .minPeriods(3)
                 .window(two, one)
                 .build()) {

          try (Table windowAggResults = sorted.groupBy(0, 1)
              .aggregateWindows(
                  RollingAggregation.sum().onColumn(3).overWindow(window_1),
                  RollingAggregation.max().onColumn(3).overWindow(window_1),
                  RollingAggregation.sum().onColumn(3).overWindow(window_2),
                  RollingAggregation.min().onColumn(2).overWindow(window_3)
              );
               ColumnVector expect_0 = ColumnVector.fromBoxedLongs(12L, 13L, 15L, 10L, 16L, 24L, 19L, 10L, 8L, 14L, 12L, 12L);
               ColumnVector expect_1 = ColumnVector.fromBoxedInts(7, 7, 9, 9, 9, 9, 9, 8, 8, 8, 6, 6);
               ColumnVector expect_2 = ColumnVector.fromBoxedLongs(13L, 22L, 22L, 15L, 24L, 26L, 26L, 19L, 14L, 20L, 20L, 12L);
               ColumnVector expect_3 = ColumnVector.fromBoxedInts(null, 1, 1, null, null, 3, 3, null, null, 5, 5, null)) {
            assertColumnsAreEqual(expect_0, windowAggResults.getColumn(0));
            assertColumnsAreEqual(expect_1, windowAggResults.getColumn(1));
            assertColumnsAreEqual(expect_2, windowAggResults.getColumn(2));
            assertColumnsAreEqual(expect_3, windowAggResults.getColumn(3));
          }
        }
      }
    }
  }

  @Test
  void testWindowingWithoutGroupByColumns() {
    try (Table unsorted = new Table.TestBuilder().column( 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6) // OBY Key
        .column( 7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6) // Agg Column
        .build();
         ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6)) {

      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0))) {
        ColumnVector sortedAggColumn = sorted.getColumn(1);
        assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

        try (Scalar one = Scalar.fromInt(1);
             Scalar two = Scalar.fromInt(2);
             WindowOptions window = WindowOptions.builder()
                 .minPeriods(1)
                 .window(two, one)
                 .build()) {

          try (Table windowAggResults = sorted.groupBy().aggregateWindows(
              RollingAggregation.sum().onColumn(1).overWindow(window));
               ColumnVector expectAggResult = ColumnVector.fromBoxedLongs(12L, 13L, 15L, 17L, 25L, 24L, 19L, 18L, 10L, 14L, 12L, 12L)
          ) {
            assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
          }
        }
      }
    }
  }

  @Test
  void testWindowWithUnboundedPrecedingUnboundedFollowing() {
    try (Table unsorted = new Table.TestBuilder()
            .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
            .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
            .column(1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6) // OBY Key
            .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6) // Agg Column
            .build()) {
      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6)) {
        ColumnVector sortedAggColumn = sorted.getColumn(3);
        assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

        try (WindowOptions window = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .unboundedFollowing()
                .build()) {

          try (Table windowAggResults = sorted.groupBy(0, 1)
                  .aggregateWindows(RollingAggregation.sum().onColumn(3).overWindow(window));
               ColumnVector expectAggResult = ColumnVector.fromBoxedLongs(22L, 22L, 22L, 22L, 26L, 26L, 26L, 26L, 20L, 20L, 20L, 20L)) {
            assertColumnsAreEqual(expectAggResult, windowAggResults.getColumn(0));
          }
        }
      }
    }
  }

  private Scalar getScalar(DType type, long value) {
    if (type.equals(DType.INT32)) {
      return Scalar.fromInt((int) value);
    } else if (type.equals(DType.INT64)) {
      return Scalar.fromLong(value);
    } else if (type.equals(DType.INT16)) {
      return Scalar.fromShort((short) value);
    } else if (type.equals(DType.INT8)) {
      return Scalar.fromByte((byte) value);
    } else if (type.equals(DType.UINT8)) {
      return Scalar.fromUnsignedByte((byte) value);
    } else if (type.equals(DType.UINT16)) {
      return Scalar.fromUnsignedShort((short) value);
    } else if (type.equals(DType.UINT32)) {
      return Scalar.fromUnsignedInt((int) value);
    } else if (type.equals(DType.UINT64)) {
      return Scalar.fromUnsignedLong(value);
    } else if (type.equals(DType.TIMESTAMP_DAYS)) {
      return Scalar.durationFromLong(DType.DURATION_DAYS, value);
    } else if (type.equals(DType.TIMESTAMP_SECONDS)) {
      return Scalar.durationFromLong(DType.DURATION_SECONDS, value);
    } else if (type.equals(DType.TIMESTAMP_MILLISECONDS)) {
      return Scalar.durationFromLong(DType.DURATION_MILLISECONDS, value);
    } else if (type.equals(DType.TIMESTAMP_MICROSECONDS)) {
      return Scalar.durationFromLong(DType.DURATION_MICROSECONDS, value);
    } else if (type.equals(DType.TIMESTAMP_NANOSECONDS)) {
      return Scalar.durationFromLong(DType.DURATION_NANOSECONDS, value);
    } else {
      return Scalar.fromNull(type);
    }
  }

  @Test
  void testRangeWindowingCount() {
    try (
        Table unsorted = new Table.TestBuilder()
            .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
            .column(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2) // GBY Key
            .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8) // Agg Column
            .column(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L) // orderBy Key
            .column((short) 1, (short)1, (short)2, (short)3, (short)3, (short)3, (short)4, (short)4, (short)5, (short)5, (short)6, (short)6, (short)7) // orderBy Key
            .column(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // orderBy Key
            .column((byte) 1, (byte)1, (byte)2, (byte)3, (byte)3, (byte)3, (byte)4, (byte)4, (byte)5, (byte)5, (byte)6, (byte)6, (byte)7) // orderBy Key
            .timestampDayColumn(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // timestamp orderBy Key
            .timestampSecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
            .timestampMicrosecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
            .timestampMillisecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
            .timestampNanosecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
            .build()) {

      for (int orderIndex = 3; orderIndex < unsorted.getNumberOfColumns(); orderIndex++) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(orderIndex));
             ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8)) {
          ColumnVector sortedAggColumn = sorted.getColumn(2);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          DType type = unsorted.getColumn(orderIndex).getType();
          try (Scalar preceding = getScalar(type, 1L);
               Scalar following = getScalar(type, 1L)) {
            try (WindowOptions window = WindowOptions.builder()
                .minPeriods(1)
                .window(preceding, following)
                .orderByColumnIndex(orderIndex)
                .build()) {
              try (Table windowAggResults = sorted.groupBy(0, 1).aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(window));
                   ColumnVector expect = ColumnVector.fromBoxedInts(3, 3, 4, 2, 4, 4, 4, 4, 4, 4, 5, 5, 3)) {
                assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testRangeWindowingLead() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2) // GBY Key
        .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8) // Agg Column
        .column(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L) // orderBy Key
        .column((short) 1, (short)1, (short)2, (short)3, (short)3, (short)3, (short)4, (short)4, (short)5, (short)5, (short)6, (short)6, (short)7) // orderBy Key
        .column(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // orderBy Key
        .column((byte) 1, (byte)1, (byte)2, (byte)3, (byte)3, (byte)3, (byte)4, (byte)4, (byte)5, (byte)5, (byte)6, (byte)6, (byte)7) // orderBy Key
        .timestampDayColumn(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // Timestamp orderBy Key
        .timestampSecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampMicrosecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampMillisecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampNanosecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .build()) {

      for (int orderIndex = 3; orderIndex < unsorted.getNumberOfColumns(); orderIndex++) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(orderIndex));
             ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8)) {
          ColumnVector sortedAggColumn = sorted.getColumn(2);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          DType type = unsorted.getColumn(orderIndex).getType();
          try (Scalar preceding = getScalar(type, 1L);
               Scalar following = getScalar(type, 1L)) {
            try (WindowOptions window = WindowOptions.builder()
                .minPeriods(1)
                .window(preceding, following)
                .orderByColumnIndex(orderIndex)
                .build()) {

              try (Table windowAggResults = sorted.groupBy(0, 1)
                  .aggregateWindowsOverRanges(RollingAggregation.lead(1)
                      .onColumn(2)
                      .overWindow(window));
                   ColumnVector expect = ColumnVector.fromBoxedInts(5, 1, 9, null, 9, 8, 2, null, 0, 6, 6, 8, null)) {
                assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testRangeWindowingMax() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2) // GBY Key
        .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8) // Agg Column
        .column(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L) // orderBy Key
        .column((short) 1, (short)1, (short)2, (short)3, (short)3, (short)3, (short)4, (short)4, (short)5, (short)5, (short)6, (short)6, (short)7) // orderBy Key
        .column(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // orderBy Key
        .column((byte) 1, (byte)1, (byte)2, (byte)3, (byte)3, (byte)3, (byte)4, (byte)4, (byte)5, (byte)5, (byte)6, (byte)6, (byte)7) // orderBy Key
        .timestampDayColumn(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // Timestamp orderBy Key
        .timestampSecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampMicrosecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampMillisecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampNanosecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .build()) {

      for (int orderIndex = 3; orderIndex < unsorted.getNumberOfColumns(); orderIndex++) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(orderIndex));
             ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8)) {
          ColumnVector sortedAggColumn = sorted.getColumn(2);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          DType type = unsorted.getColumn(orderIndex).getType();
          try (Scalar preceding = getScalar(type, 1L);
               Scalar following = getScalar(type, 1L)) {
            try (WindowOptions window = WindowOptions.builder()
                .minPeriods(1)
                .window(preceding, following)
                .orderByColumnIndex(orderIndex)
                .build()) {

              try (Table windowAggResults = sorted.groupBy(0, 1)
                  .aggregateWindowsOverRanges(RollingAggregation.max().onColumn(2).overWindow(window));
                   ColumnVector expect = ColumnVector.fromBoxedInts(7, 7, 9, 9, 9, 9, 9, 9, 8, 8, 8, 8, 8)) {
                assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
              }
            }

            try (Scalar one = Scalar.fromInt(1);
                 Scalar two = Scalar.fromInt(2);
                 WindowOptions window = WindowOptions.builder()
                     .minPeriods(1)
                     .window(two, one)
                     .build()) {

              try (Table windowAggResults = sorted.groupBy(0, 1)
                  .aggregateWindows(RollingAggregation.max().onColumn(2).overWindow(window));
                   ColumnVector expect = ColumnVector.fromBoxedInts(7, 7, 9, 9, 9, 9, 9, 8, 8, 8, 6, 8, 8)) {
                assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testRangeWindowingRowNumber() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2) // GBY Key
        .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8) // Agg Column
        .column(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L) // orderBy Key
        .column((short) 1, (short)1, (short)2, (short)3, (short)3, (short)3, (short)4, (short)4, (short)5, (short)5, (short)6, (short)6, (short)7) // orderBy Key
        .column(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // orderBy Key
        .column((byte) 1, (byte)1, (byte)2, (byte)3, (byte)3, (byte)3, (byte)4, (byte)4, (byte)5, (byte)5, (byte)6, (byte)6, (byte)7) // orderBy Key
        .timestampDayColumn(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // Timestamp orderBy Key
        .timestampSecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampMicrosecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampMillisecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampNanosecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .build()) {

      for (int orderIndex = 3; orderIndex < unsorted.getNumberOfColumns(); orderIndex++) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(orderIndex));
             ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8)) {
          ColumnVector sortedAggColumn = sorted.getColumn(2);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          DType type = unsorted.getColumn(orderIndex).getType();
          try (Scalar preceding = getScalar(type, 2L);
               Scalar following = getScalar(type, 0L)) {
            try (WindowOptions window = WindowOptions.builder()
                .minPeriods(1)
                .window(preceding, following)
                .orderByColumnIndex(orderIndex)
                .build()) {

              try (Table windowAggResults = sorted.groupBy(0, 1)
                  .aggregateWindowsOverRanges(RollingAggregation.rowNumber().onColumn(2).overWindow(window));
                   ColumnVector expect = ColumnVector.fromBoxedInts(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 5)) {
                assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testRangeWindowingCountDescendingTimestamps() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1, 1) // GBY Key
        .column(0, 0, 0, 0,  1, 1, 1, 1,  2, 2, 2, 2, 2) // GBY Key
        .column(7, 5, 1, 9,  7, 9, 8, 2,  8, 0, 6, 6, 8) // Agg Column
        .column((short)7, (short)6, (short)6, (short)5, (short)5, (short)4, (short)4, (short)3, (short)3, (short)3, (short)2, (short)1, (short)1)
        .column(7L, 6L, 6L, 5L, 5L, 4L, 4L, 3L, 3L, 3L, 2L, 1L, 1L)
        .column(7, 6, 6, 5,  5, 4, 4, 3,  3, 3, 2, 1, 1)
        .column((byte)7, (byte)6, (byte)6, (byte)5,  (byte)5, (byte)4, (byte)4, (byte)3,  (byte)3, (byte)3, (byte)2, (byte)1, (byte)1)
        .timestampDayColumn(7, 6, 6, 5,  5, 4, 4, 3,  3, 3, 2, 1, 1) // Timestamp Key
        .timestampSecondsColumn(7L, 6L, 6L, 5L, 5L, 4L, 4L, 3L, 3L, 3L, 2L, 1L, 1L)
        .timestampMicrosecondsColumn(7L, 6L, 6L, 5L, 5L, 4L, 4L, 3L, 3L, 3L, 2L, 1L, 1L)
        .timestampMillisecondsColumn(7L, 6L, 6L, 5L, 5L, 4L, 4L, 3L, 3L, 3L, 2L, 1L, 1L)
        .timestampNanosecondsColumn(7L, 6L, 6L, 5L, 5L, 4L, 4L, 3L, 3L, 3L, 2L, 1L, 1L)
        .build()) {

      for (int orderIndex = 3; orderIndex < unsorted.getNumberOfColumns(); orderIndex++) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.desc(orderIndex));
             ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8)) {
          ColumnVector sortedAggColumn = sorted.getColumn(2);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          DType type = unsorted.getColumn(orderIndex).getType();
          try (Scalar preceding_0 = getScalar(type, 2L);
               Scalar following_0 = getScalar(type, 1L);
               Scalar preceding_1 = getScalar(type, 3L);
               Scalar following_1 = getScalar(type, 0L)) {

            try (WindowOptions window_0 = WindowOptions.builder()
                  .minPeriods(1)
                  .window(preceding_0, following_0)
                  .orderByColumnIndex(orderIndex)
                  .orderByDescending()
                  .build();

                 WindowOptions window_1 = WindowOptions.builder()
                  .minPeriods(1)
                  .window(preceding_1, following_1)
                  .orderByColumnIndex(orderIndex)
                  .orderByDescending()
                  .build()) {

              try (Table windowAggResults = sorted.groupBy(0, 1)
                  .aggregateWindowsOverRanges(
                      RollingAggregation.count().onColumn(2).overWindow(window_0),
                      RollingAggregation.sum().onColumn(2).overWindow(window_1));
                   ColumnVector expect_0 = ColumnVector.fromBoxedInts(3, 4, 4, 4, 3, 4, 4, 4, 3, 3, 5, 5, 5);
                   ColumnVector expect_1 = ColumnVector.fromBoxedLongs(7L, 13L, 13L, 22L, 7L, 24L, 24L, 26L, 8L, 8L, 14L, 28L, 28L)) {
                assertColumnsAreEqual(expect_0, windowAggResults.getColumn(0));
                assertColumnsAreEqual(expect_1, windowAggResults.getColumn(1));
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testRangeWindowingWithoutGroupByColumns() {
    try (Table unsorted = new Table.TestBuilder()
        .column(             7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8) // Agg Column
        .column(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L) // orderBy Key
        .column((short) 1, (short)1, (short)2, (short)3, (short)3, (short)3, (short)4, (short)4, (short)5, (short)5, (short)6, (short)6, (short)7) // orderBy Key
        .column(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // orderBy Key
        .column((byte) 1, (byte)1, (byte)2, (byte)3, (byte)3, (byte)3, (byte)4, (byte)4, (byte)5, (byte)5, (byte)6, (byte)6, (byte)7) // orderBy Key
        .timestampDayColumn(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // Timestamp orderBy Key
        .timestampSecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampMicrosecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampMillisecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampNanosecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .build()) {

      for (int orderIndex = 3; orderIndex < unsorted.getNumberOfColumns(); orderIndex++) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(orderIndex));
             ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8)) {
          ColumnVector sortedAggColumn = sorted.getColumn(0);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          DType type = unsorted.getColumn(orderIndex).getType();
          try (Scalar preceding = getScalar(type, 1L);
               Scalar following = getScalar(type, 1L)) {
            try (WindowOptions window = WindowOptions.builder()
                .minPeriods(1)
                .window(preceding, following)
                .orderByColumnIndex(orderIndex)
                .build();) {

              try (Table windowAggResults = sorted.groupBy()
                  .aggregateWindowsOverRanges(RollingAggregation.count().onColumn(1).overWindow(window));
                   ColumnVector expect = ColumnVector.fromBoxedInts(3, 3, 6, 6, 6, 6, 7, 7, 6, 6, 5, 5, 3)) {
                assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderByUnboundedToCurrentRow() {
    // PARTITION BY part ORDER BY oby_int ASC, oby_str DESC
    // RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW. Input is pre-sorted.
    try (Table sorted = new Table.TestBuilder()
        .column(0, 0, 0, 0, 0, 1, 1, 1)                  // part (GBY key)
        .column(10, 10, 20, 20, 30, 5, 5, 7)             // oby_int (ASC)
        .column("b", "a", "z", "z", "m", "y", "y", "x")  // oby_str (DESC)
        .column(1, 2, 3, 4, 5, 6, 7, 8)                  // agg column (non-null)
        .build()) {
      try (WindowOptions window = WindowOptions.builder()
              .minPeriods(1)
              .unboundedPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{1, 2},
                              new boolean[]{true, false},
                              new boolean[]{true, true})
              .build();
           Table result = sorted.groupBy(0)
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(3).overWindow(window));
           ColumnVector expect = ColumnVector.fromBoxedInts(1, 2, 4, 4, 5, 2, 2, 3)) {
        assertColumnsAreEqual(expect, result.getColumn(0));
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderByCurrentRowPeers() {
    // ORDER BY oby_int ASC, oby_str DESC  RANGE BETWEEN CURRENT ROW AND CURRENT ROW.
    // The window for each row is its peer group (rows equal on every order-by column).
    try (Table sorted = new Table.TestBuilder()
        .column(10, 10, 20, 20, 30)              // oby_int (ASC)
        .column("b", "a", "z", "z", "m")         // oby_str (DESC)
        .column(1, 2, 3, 4, 5)                   // agg column
        .build()) {
      try (WindowOptions window = WindowOptions.builder()
              .minPeriods(1)
              .currentRowPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{0, 1},
                              new boolean[]{true, false},
                              new boolean[]{true, true})
              .build();
           Table result = sorted.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(window));
           ColumnVector expect = ColumnVector.fromBoxedInts(1, 1, 2, 2, 1)) {
        assertColumnsAreEqual(expect, result.getColumn(0));
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderByCurrentRowToUnbounded() {
    // ORDER BY oby_int DESC, oby_str ASC  RANGE BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING.
    try (Table sorted = new Table.TestBuilder()
        .column(30, 20, 20, 10, 10)              // oby_int (DESC)
        .column("m", "z", "z", "a", "b")         // oby_str (ASC)
        .column(1, 2, 3, 4, 5)                   // agg column
        .build()) {
      try (WindowOptions window = WindowOptions.builder()
              .minPeriods(1)
              .currentRowPreceding()
              .unboundedFollowing()
              .orderByColumns(new int[]{0, 1},
                              new boolean[]{false, true},
                              new boolean[]{true, true})
              .build();
           Table result = sorted.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(window));
           ColumnVector expect = ColumnVector.fromBoxedInts(5, 4, 4, 2, 1)) {
        assertColumnsAreEqual(expect, result.getColumn(0));
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderByNullsFirstPeers() {
    // ORDER BY oby_int ASC NULLS FIRST, oby_str DESC  RANGE BETWEEN CURRENT ROW AND CURRENT ROW.
    // Null order-by values sort before non-null values, and null rows that match on the remaining
    // order-by columns are grouped together as peers.
    try (Table sorted = new Table.TestBuilder()
        .column((Integer) null, (Integer) null, 10, 20)  // oby_int (ASC, nulls first)
        .column("x", "x", "y", "z")                       // oby_str (DESC)
        .column(1, 2, 3, 4)                               // agg column
        .build()) {
      try (WindowOptions window = WindowOptions.builder()
              .minPeriods(1)
              .currentRowPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{0, 1},
                              new boolean[]{true, false},
                              new boolean[]{true, true})   // nulls first on oby_int
              .build();
           Table result = sorted.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(window));
           ColumnVector expect = ColumnVector.fromBoxedInts(2, 2, 1, 1)) {
        assertColumnsAreEqual(expect, result.getColumn(0));
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderByNullsLastPeers() {
    // ORDER BY oby_int ASC NULLS LAST, oby_str DESC  RANGE BETWEEN CURRENT ROW AND CURRENT ROW.
    // Null order-by values sort after non-null values; null rows that match on the remaining
    // order-by columns are grouped together as peers.
    try (Table sorted = new Table.TestBuilder()
        .column(10, 20, (Integer) null, (Integer) null)  // oby_int (ASC, nulls last)
        .column("y", "z", "x", "x")                       // oby_str (DESC)
        .column(1, 2, 3, 4)                               // agg column
        .build()) {
      try (WindowOptions window = WindowOptions.builder()
              .minPeriods(1)
              .currentRowPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{0, 1},
                              new boolean[]{true, false},
                              new boolean[]{false, true})  // nulls last on oby_int
              .build();
           Table result = sorted.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(window));
           ColumnVector expect = ColumnVector.fromBoxedInts(1, 1, 2, 2)) {
        assertColumnsAreEqual(expect, result.getColumn(0));
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderByNullsLastUnboundedToCurrentRow() {
    // ORDER BY oby_int ASC NULLS LAST, oby_str DESC  RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT
    // ROW. The cumulative window extends through the last peer of the current row, so the trailing
    // null peers see the whole group.
    try (Table sorted = new Table.TestBuilder()
        .column(10, 20, (Integer) null, (Integer) null)  // oby_int (ASC, nulls last)
        .column("y", "z", "x", "x")                       // oby_str (DESC)
        .column(1, 2, 3, 4)                               // agg column
        .build()) {
      try (WindowOptions window = WindowOptions.builder()
              .minPeriods(1)
              .unboundedPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{0, 1},
                              new boolean[]{true, false},
                              new boolean[]{false, true})  // nulls last on oby_int
              .build();
           Table result = sorted.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(window));
           ColumnVector expect = ColumnVector.fromBoxedInts(1, 2, 4, 4)) {
        assertColumnsAreEqual(expect, result.getColumn(0));
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderByBoundedScalarThrows() {
    // Multi-column RANGE windows do not support bounded scalar ranges.
    try (Table table = new Table.TestBuilder()
        .column(10, 10, 20)
        .column("b", "a", "z")
        .column(1, 2, 3)
        .build();
         Scalar one = Scalar.fromInt(1);
         WindowOptions window = WindowOptions.builder()
             .minPeriods(1)
             .window(one, one)
             .orderByColumns(new int[]{0, 1},
                             new boolean[]{true, false},
                             new boolean[]{true, true})
             .build()) {
      assertThrows(IllegalArgumentException.class,
          () -> table.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(window)));
    }
  }

  @Test
  void testRangeWindowingMultiOrderByUnsupportedTypeThrows() {
    // An unsupported type on any order-by column is rejected (validation loops over all columns).
    try (Table table = new Table.TestBuilder()
        .column(10, 20, 30)              // valid int order-by
        .column(true, false, true)       // unsupported BOOL8 order-by
        .column(1, 2, 3)                 // agg column
        .build();
         WindowOptions window = WindowOptions.builder()
             .minPeriods(1)
             .currentRowPreceding()
             .currentRowFollowing()
             .orderByColumns(new int[]{0, 1},
                             new boolean[]{true, true},
                             new boolean[]{true, true})
             .build()) {
      assertThrows(IllegalArgumentException.class,
          () -> table.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(window)));
    }
  }

  @Test
  void testRangeWindowingMultiOrderByMismatchedArraysThrows() {
    assertThrows(IllegalArgumentException.class,
        () -> WindowOptions.builder()
            .orderByColumns(new int[]{0, 1},
                            new boolean[]{true},
                            new boolean[]{true, true}));
  }

  @Test
  void testRangeWindowingMultiOrderByNullArraysThrows() {
    // Any null order-by array is rejected by the builder.
    assertThrows(IllegalArgumentException.class,
        () -> WindowOptions.builder()
            .orderByColumns(null, new boolean[]{true}, new boolean[]{true}));
    assertThrows(IllegalArgumentException.class,
        () -> WindowOptions.builder()
            .orderByColumns(new int[]{0}, null, new boolean[]{true}));
    assertThrows(IllegalArgumentException.class,
        () -> WindowOptions.builder()
            .orderByColumns(new int[]{0}, new boolean[]{true}, null));
  }

  @Test
  void testRangeWindowingMultiOrderByEmptyArraysThrows() {
    // At least one order-by column is required.
    assertThrows(IllegalArgumentException.class,
        () -> WindowOptions.builder()
            .orderByColumns(new int[]{}, new boolean[]{}, new boolean[]{}));
  }

  @Test
  void testRangeWindowingMultiOrderByMixingSettersThrows() {
    // Mixing orderByColumns(...) with the single-column order-by setters is rejected by build().
    assertThrows(IllegalStateException.class,
        () -> WindowOptions.builder()
            .currentRowPreceding()
            .currentRowFollowing()
            .orderByColumns(new int[]{0, 1},
                            new boolean[]{true, true},
                            new boolean[]{true, true})
            .orderByColumnIndex(0)
            .build());
    assertThrows(IllegalStateException.class,
        () -> WindowOptions.builder()
            .currentRowPreceding()
            .currentRowFollowing()
            .orderByColumns(new int[]{0, 1},
                            new boolean[]{true, true},
                            new boolean[]{true, true})
            .orderByAscending()
            .build());
    assertThrows(IllegalStateException.class,
        () -> WindowOptions.builder()
            .currentRowPreceding()
            .currentRowFollowing()
            .orderByColumns(new int[]{0, 1},
                            new boolean[]{true, true},
                            new boolean[]{true, true})
            .orderByDescending()
            .build());
  }

  @Test
  void testRangeWindowingMultiOrderByOutOfRangeIndexThrows() {
    // An order-by index outside the input table is rejected during marshalling, before the
    // native call, with an IllegalArgumentException naming the offending index.
    try (Table table = new Table.TestBuilder()
        .column(10, 10, 20)
        .column("b", "a", "z")
        .column(1, 2, 3)
        .build();
         WindowOptions window = WindowOptions.builder()
             .minPeriods(1)
             .currentRowPreceding()
             .currentRowFollowing()
             .orderByColumns(new int[]{0, 99},   // 99 is out of range
                             new boolean[]{true, true},
                             new boolean[]{true, true})
             .build()) {
      assertThrows(IllegalArgumentException.class,
          () -> table.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(window)));
    }
  }

  @Test
  void testRangeWindowingMultiOrderBySumMinMax() {
    // PARTITION BY part ORDER BY oby_int ASC, oby_str DESC
    // RANGE BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW. Input is pre-sorted. Each row's window
    // runs from the group start through the last peer of the current row (peers match on every
    // order-by column), so it mirrors the COUNT layout in
    // testRangeWindowingMultiOrderByUnboundedToCurrentRow.
    try (Table sorted = new Table.TestBuilder()
        .column(0, 0, 0, 0, 0, 1, 1, 1)                  // part (GBY key)
        .column(10, 10, 20, 20, 30, 5, 5, 7)             // oby_int (ASC)
        .column("b", "a", "z", "z", "m", "y", "y", "x")  // oby_str (DESC)
        .column(1, 2, 3, 4, 5, 6, 7, 8)                  // agg column (non-null)
        .build()) {
      try (WindowOptions window = WindowOptions.builder()
              .minPeriods(1)
              .unboundedPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{1, 2},
                              new boolean[]{true, false},
                              new boolean[]{true, true})
              .build();
           Table result = sorted.groupBy(0)
              .aggregateWindowsOverRanges(
                  RollingAggregation.sum().onColumn(3).overWindow(window),
                  RollingAggregation.min().onColumn(3).overWindow(window),
                  RollingAggregation.max().onColumn(3).overWindow(window));
           // Windows: {1},{1,2},{1,2,3,4},{1,2,3,4},{1,2,3,4,5},{6,7},{6,7},{6,7,8}.
           ColumnVector expectSum = ColumnVector.fromBoxedLongs(1L, 3L, 10L, 10L, 15L, 13L, 13L, 21L);
           ColumnVector expectMin = ColumnVector.fromBoxedInts(1, 1, 1, 1, 1, 6, 6, 6);
           ColumnVector expectMax = ColumnVector.fromBoxedInts(1, 2, 4, 4, 5, 7, 7, 8)) {
        assertColumnsAreEqual(expectSum, result.getColumn(0));
        assertColumnsAreEqual(expectMin, result.getColumn(1));
        assertColumnsAreEqual(expectMax, result.getColumn(2));
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderByAggColumnWithNulls() {
    // Same peer-window layout as testRangeWindowingMultiOrderBySumMinMax, but the agg column has
    // nulls. COUNT defaults to NullPolicy.EXCLUDE and SUM skips nulls.
    try (Table sorted = new Table.TestBuilder()
        .column(0, 0, 0, 0, 0, 1, 1, 1)                  // part (GBY key)
        .column(10, 10, 20, 20, 30, 5, 5, 7)             // oby_int (ASC)
        .column("b", "a", "z", "z", "m", "y", "y", "x")  // oby_str (DESC)
        .column(1, null, 3, null, 5, 6, null, 8)         // agg column (with nulls)
        .build()) {
      try (WindowOptions window = WindowOptions.builder()
              .minPeriods(1)
              .unboundedPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{1, 2},
                              new boolean[]{true, false},
                              new boolean[]{true, true})
              .build();
           Table result = sorted.groupBy(0)
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(3).overWindow(window),
                  RollingAggregation.sum().onColumn(3).overWindow(window));
           // Windows: {1},{1,n},{1,n,3,n},{1,n,3,n},{1,n,3,n,5},{6,n},{6,n},{6,n,8}.
           ColumnVector expectCount = ColumnVector.fromBoxedInts(1, 1, 2, 2, 3, 1, 1, 2);
           ColumnVector expectSum = ColumnVector.fromBoxedLongs(1L, 1L, 4L, 4L, 9L, 6L, 6L, 14L)) {
        assertColumnsAreEqual(expectCount, result.getColumn(0));
        assertColumnsAreEqual(expectSum, result.getColumn(1));
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderBySingleColumnEquivalence() {
    // A single-column order-by expressed via orderByColumns({i},{asc},{nf}) must produce the same
    // result as the legacy orderByColumnIndex(i) + orderByAscending() path; both route to the
    // single-column native overload. The order-by column has no nulls, so null placement is moot.
    try (Table sorted = new Table.TestBuilder()
        .column(10, 10, 20, 30)  // oby_int (ASC)
        .column(1, 2, 3, 4)      // agg column
        .build()) {
      try (WindowOptions legacy = WindowOptions.builder()
              .minPeriods(1)
              .unboundedPreceding()
              .currentRowFollowing()
              .orderByColumnIndex(0)
              .orderByAscending()
              .build();
           WindowOptions viaArrays = WindowOptions.builder()
              .minPeriods(1)
              .unboundedPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{0}, new boolean[]{true}, new boolean[]{true})
              .build();
           Table legacyResult = sorted.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(1).overWindow(legacy));
           Table arraysResult = sorted.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(1).overWindow(viaArrays));
           // Peers on oby_int: rows 0,1 (=10); window UNBOUNDED PRECEDING -> CURRENT ROW.
           ColumnVector expect = ColumnVector.fromBoxedInts(2, 2, 3, 4)) {
        assertColumnsAreEqual(expect, legacyResult.getColumn(0));
        assertColumnsAreEqual(expect, arraysResult.getColumn(0));
        assertColumnsAreEqual(legacyResult.getColumn(0), arraysResult.getColumn(0));
      }
    }
    // The two builder paths must also compare equal via WindowOptions.equals/hashCode.
    try (WindowOptions legacy = WindowOptions.builder()
            .orderByColumnIndex(3)
            .orderByAscending()
            .build();
         WindowOptions viaArrays = WindowOptions.builder()
            .orderByColumns(new int[]{3}, new boolean[]{true}, new boolean[]{true})
            .build()) {
      assertEquals(legacy, viaArrays);
      assertEquals(legacy.hashCode(), viaArrays.hashCode());
    }
  }

  @Test
  void testRangeWindowingMultiOrderBySingleColumnEquivalenceWithNulls() {
    // Like testRangeWindowingMultiOrderBySingleColumnEquivalence, but the order-by column has
    // nulls -- the one place the null-ordering handling could diverge between the legacy
    // single-column path and the length-1 orderByColumns({i}) array path. Both route to the
    // single-column native overload, which deduces null placement natively, so the results must
    // match. A CURRENT ROW peer frame keeps the expectation independent of the deduced null
    // placement: two nulls are mutual peers, so the answer depends only on peer-equality.
    try (Table sorted = new Table.TestBuilder()
        .column((Integer) null, (Integer) null, 10, 20)  // oby_int (with nulls)
        .column(1, 2, 3, 4)                               // agg column
        .build()) {
      try (WindowOptions legacy = WindowOptions.builder()
              .minPeriods(1)
              .currentRowPreceding()
              .currentRowFollowing()
              .orderByColumnIndex(0)
              .orderByAscending()
              .build();
           WindowOptions viaArrays = WindowOptions.builder()
              .minPeriods(1)
              .currentRowPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{0}, new boolean[]{true}, new boolean[]{true})
              .build();
           Table legacyResult = sorted.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(1).overWindow(legacy));
           Table arraysResult = sorted.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(1).overWindow(viaArrays));
           // Peer groups on oby_int: {null,null}={0,1}, {10}={2}, {20}={3}.
           ColumnVector expect = ColumnVector.fromBoxedInts(2, 2, 1, 1)) {
        assertColumnsAreEqual(expect, legacyResult.getColumn(0));
        assertColumnsAreEqual(expect, arraysResult.getColumn(0));
        assertColumnsAreEqual(legacyResult.getColumn(0), arraysResult.getColumn(0));
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderByTimestampDateDecimalLeading() {
    // Multi-column RANGE with the leading order-by column being, in turn, a TIMESTAMP_SECONDS,
    // a TIMESTAMP_DAYS (date), and a DECIMAL64 column, tie-broken by an int column.
    // ORDER BY lead ASC, tiebreak ASC  RANGE BETWEEN CURRENT ROW AND CURRENT ROW (peer groups).
    // Leading values 1,1,2,2,2 and tiebreak 5,6,7,7,8 give peer groups {0},{1},{2,3},{4}.
    try (Table tsTable = new Table.TestBuilder()
            .timestampSecondsColumn(1L, 1L, 2L, 2L, 2L)  // leading order-by (TIMESTAMP_SECONDS)
            .column(5, 6, 7, 7, 8)                        // tiebreak order-by (INT32)
            .column(1, 2, 3, 4, 5)                        // agg column
            .build();
         Table dateTable = new Table.TestBuilder()
            .timestampDayColumn(1, 1, 2, 2, 2)            // leading order-by (TIMESTAMP_DAYS == date)
            .column(5, 6, 7, 7, 8)
            .column(1, 2, 3, 4, 5)
            .build();
         Table decimalTable = new Table.TestBuilder()
            .decimal64Column(0, 1L, 1L, 2L, 2L, 2L)       // leading order-by (DECIMAL64)
            .column(5, 6, 7, 7, 8)
            .column(1, 2, 3, 4, 5)
            .build()) {
      WindowOptions.Builder windowBuilder = WindowOptions.builder()
          .minPeriods(1)
          .currentRowPreceding()
          .currentRowFollowing()
          .orderByColumns(new int[]{0, 1},
                          new boolean[]{true, true},
                          new boolean[]{true, true});
      // Peer groups {0},{1},{2,3},{4} -> COUNT {1,1,2,2,1} for every leading type.
      try (WindowOptions tsWindow = windowBuilder.build();
           WindowOptions dateWindow = windowBuilder.build();
           WindowOptions decimalWindow = windowBuilder.build();
           ColumnVector expect = ColumnVector.fromBoxedInts(1, 1, 2, 2, 1);
           Table tsResult = tsTable.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(tsWindow));
           Table dateResult = dateTable.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(dateWindow));
           Table decimalResult = decimalTable.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(decimalWindow))) {
        assertColumnsAreEqual(expect, tsResult.getColumn(0));
        assertColumnsAreEqual(expect, dateResult.getColumn(0));
        assertColumnsAreEqual(expect, decimalResult.getColumn(0));
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderByThreeColumns() {
    // Three order-by columns exercise the CSR order-by slice for ob_count > 2.
    // ORDER BY a ASC, b ASC, c ASC  RANGE BETWEEN CURRENT ROW AND CURRENT ROW (peer groups).
    try (Table sorted = new Table.TestBuilder()
        .column(1, 1, 1, 1)   // oby a
        .column(1, 1, 2, 2)   // oby b
        .column(5, 5, 6, 7)   // oby c
        .column(1, 2, 3, 4)   // agg column
        .build()) {
      try (WindowOptions window = WindowOptions.builder()
              .minPeriods(1)
              .currentRowPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{0, 1, 2},
                              new boolean[]{true, true, true},
                              new boolean[]{true, true, true})
              .build();
           Table result = sorted.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(3).overWindow(window));
           // Peer groups: {0,1} (1,1,5), {2} (1,2,6), {3} (1,2,7).
           ColumnVector expect = ColumnVector.fromBoxedInts(2, 2, 1, 1)) {
        assertColumnsAreEqual(expect, result.getColumn(0));
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderByNullsInOrderByColumns() {
    // Nulls in the second order-by column, then nulls in both order-by columns. Null order-by
    // values that are equal (both null) on a column and match on the others are peers.
    // ORDER BY a ASC NULLS FIRST, b ASC NULLS FIRST  RANGE BETWEEN CURRENT ROW AND CURRENT ROW.
    try (Table secondColNulls = new Table.TestBuilder()
        .column(1, 1, 2, 2)                              // oby a (no nulls)
        .column((Integer) null, (Integer) null, 3, 4)    // oby b (nulls in 2nd column)
        .column(1, 2, 3, 4)                              // agg column
        .build()) {
      try (WindowOptions window = WindowOptions.builder()
              .minPeriods(1)
              .currentRowPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{0, 1},
                              new boolean[]{true, true},
                              new boolean[]{true, true})
              .build();
           Table result = secondColNulls.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(window));
           // Peer groups: {0,1} (1,null), {2} (2,3), {3} (2,4).
           ColumnVector expect = ColumnVector.fromBoxedInts(2, 2, 1, 1)) {
        assertColumnsAreEqual(expect, result.getColumn(0));
      }
    }
    try (Table bothColsNulls = new Table.TestBuilder()
        .column((Integer) null, (Integer) null, 2)       // oby a (nulls)
        .column((Integer) null, (Integer) null, 3)       // oby b (nulls)
        .column(1, 2, 3)                                 // agg column
        .build()) {
      try (WindowOptions window = WindowOptions.builder()
              .minPeriods(1)
              .currentRowPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{0, 1},
                              new boolean[]{true, true},
                              new boolean[]{true, true})
              .build();
           Table result = bothColsNulls.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(window));
           // Peer groups: {0,1} (null,null), {2} (2,3).
           ColumnVector expect = ColumnVector.fromBoxedInts(2, 2, 1)) {
        assertColumnsAreEqual(expect, result.getColumn(0));
      }
    }
  }

  @Test
  void testRangeWindowingMultiOrderByFloatEdgeValuePeers() {
    // ORDER BY oby_f32 ASC NULLS FIRST (FLOAT32), oby_f64 ASC NULLS FIRST (FLOAT64)
    // RANGE BETWEEN CURRENT ROW AND CURRENT ROW. Each row's window is its peer group, so the
    // expected COUNT exercises float peer-equality on the tricky edge values: NaN == NaN, signed
    // zero (-0.0f == +0.0f), and null == null on either order-by column. Input is pre-sorted on
    // (oby_f32, oby_f64): nulls first, then ascending finite values, then NaN last.
    try (Table sorted = new Table.TestBuilder()
        // oby_f32 (FLOAT32): null,null | -0.0,+0.0 | 2.5,2.5,2.5 | NaN,NaN
        .column((Float) null, (Float) null, -0.0f, 0.0f, 2.5f, 2.5f, 2.5f, Float.NaN, Float.NaN)
        // oby_f64 (FLOAT64), tie-breaks within equal oby_f32 (nulls first)
        .column(1.0, 1.0, 5.0, 5.0, (Double) null, (Double) null, 9.0, 4.0, 4.0)
        .column(1, 2, 3, 4, 5, 6, 7, 8, 9)  // agg column (non-null)
        .build()) {
      try (WindowOptions window = WindowOptions.builder()
              .minPeriods(1)
              .currentRowPreceding()
              .currentRowFollowing()
              .orderByColumns(new int[]{0, 1},
                              new boolean[]{true, true},
                              new boolean[]{true, true})   // ASC, nulls first on both
              .build();
           Table result = sorted.groupBy()
              .aggregateWindowsOverRanges(
                  RollingAggregation.count().onColumn(2).overWindow(window));
           // Peer groups: {0,1} (null,1.0), {2,3} (+/-0.0,5.0), {4,5} (2.5,null), {6} (2.5,9.0),
           // {7,8} (NaN,4.0).
           ColumnVector expect = ColumnVector.fromBoxedInts(2, 2, 2, 2, 2, 2, 1, 2, 2)) {
        assertColumnsAreEqual(expect, result.getColumn(0));
      }
    }
  }

  @Test
  void testRangeWindowingOrderByUnsupportedDataTypeExceptions() {
    try (Table table = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2) // GBY Key
        .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8) // Agg Column
        .column(true, false, true, false, true, false, true, false, false, false, false, false, false) // orderBy Key
        .build()) {

      try (Scalar one = Scalar.fromInt(1);
           WindowOptions rangeBasedWindow = WindowOptions.builder()
               .minPeriods(1)
               .window(one, one)
               .orderByColumnIndex(3)
               .build()) {

        assertThrows(IllegalArgumentException.class,
            () -> table
                .groupBy(0, 1)
                .aggregateWindowsOverRanges(RollingAggregation.max().onColumn(2).overWindow(rangeBasedWindow)));
      }
    }
  }

  @Test
  void testInvalidWindowTypeExceptions() {
    try (Scalar one = Scalar.fromInt(1);
         Table table = new Table.TestBuilder()
             .column(             1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
             .column(             0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2) // GBY Key
             .timestampDayColumn( 1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // Timestamp Key
             .column(             7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8) // Agg Column
             .build()) {


      try (WindowOptions rowBasedWindow = WindowOptions.builder()
          .minPeriods(1)
          .window(one, one)
          .build()) {
        assertThrows(IllegalArgumentException.class, () -> table.groupBy(0, 1).aggregateWindowsOverRanges(RollingAggregation.max().onColumn(3).overWindow(rowBasedWindow)));
      }

      try (WindowOptions rangeBasedWindow = WindowOptions.builder()
          .minPeriods(1)
          .window(one, one)
          .orderByColumnIndex(2)
          .build()) {
        assertThrows(IllegalArgumentException.class, () -> table.groupBy(0, 1).aggregateWindows(RollingAggregation.max().onColumn(3).overWindow(rangeBasedWindow)));
      }
    }
  }

  @Test
  void testRangeWindowingCountUnboundedPreceding() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2) // GBY Key
        .column(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8) // Agg Column
        .column(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L) // orderBy Key
        .column((short) 1, (short)1, (short)2, (short)3, (short)3, (short)3, (short)4, (short)4, (short)5, (short)5, (short)6, (short)6, (short)7) // orderBy Key
        .column(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // orderBy Key
        .column((byte) 1, (byte)1, (byte)2, (byte)3, (byte)3, (byte)3, (byte)4, (byte)4, (byte)5, (byte)5, (byte)6, (byte)6, (byte)7) // orderBy Key
        .timestampDayColumn(1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 7) // Timestamp orderBy Key
        .timestampSecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampMicrosecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampMillisecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .timestampNanosecondsColumn(1L, 1L, 2L, 3L, 3L, 3L, 4L, 4L, 5L, 5L, 6L, 6L, 7L)
        .build()) {

      for (int orderIndex = 3; orderIndex < unsorted.getNumberOfColumns(); orderIndex++) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(orderIndex));
             ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8)) {
          ColumnVector sortedAggColumn = sorted.getColumn(2);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          DType type = unsorted.getColumn(orderIndex).getType();
          try (Scalar following = getScalar(type, 1L)) {
            try (WindowOptions window = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .following(following)
                .orderByColumnIndex(orderIndex)
                .build();) {

              try (Table windowAggResults = sorted.groupBy(0, 1)
                  .aggregateWindowsOverRanges(RollingAggregation.count().onColumn(2).overWindow(window));
                   ColumnVector expect = ColumnVector.fromBoxedInts(3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5)) {
                assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testRangeWindowingWithStringOrderByColumn() {
    final String X = null;
    final int orderIndex = 3; // Index of order-by column.
    try (Table unsorted = new Table.TestBuilder()
            .column(1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1) // GBY Key
            .column(0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1) // GBY Key
            .column(7, 5, 1, 9, 7, 9,  8, 2, 8, 0, 6, 6, 8) // Agg Column
            .column("0", "1", "2", "3", "4", "5", X, X, "1", "2", "4", "5", "7") // String orderBy Key
            .build()) {
      try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(3, true));
           ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8)) {
        ColumnVector sortedAggColumn = sorted.getColumn(2);
        assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

        try (WindowOptions unboundedPrecedingAndFollowing = WindowOptions.builder()
                     .minPeriods(1)
                     .unboundedPreceding()
                     .unboundedFollowing()
                     .orderByColumnIndex(orderIndex)
                     .build();
             WindowOptions unboundedPrecedingAndCurrentRow = WindowOptions.builder()
                     .minPeriods(1)
                     .unboundedPreceding()
                     .currentRowFollowing()
                     .orderByColumnIndex(orderIndex)
                     .build();
             WindowOptions currentRowAndUnboundedFollowing = WindowOptions.builder()
                     .minPeriods(1)
                     .currentRowPreceding()
                     .unboundedFollowing()
                     .orderByColumnIndex(orderIndex)
                     .build()) {

          try (Table windowAggResults = sorted.groupBy(0, 1)
                  .aggregateWindowsOverRanges(
                          RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingAndFollowing),
                          RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingAndCurrentRow),
                          RollingAggregation.count().onColumn(2).overWindow(currentRowAndUnboundedFollowing));
               ColumnVector expect_0 = ColumnVector.fromBoxedInts(6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7);
               ColumnVector expect_1 = ColumnVector.fromBoxedInts(1, 2, 3, 4, 5, 6, 2, 2, 3, 4, 5, 6, 7);
               ColumnVector expect_2 = ColumnVector.fromBoxedInts(6, 5, 4, 3, 2, 1, 7, 7, 5, 4, 3, 2, 1)) {

            assertColumnsAreEqual(expect_0, windowAggResults.getColumn(0));
            assertColumnsAreEqual(expect_1, windowAggResults.getColumn(1));
            assertColumnsAreEqual(expect_2, windowAggResults.getColumn(2));
          }
        }
      }
    }
  }

  @Test
  void testRangeWindowingCountUnboundedASCWithNullsFirst() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(7, 5, 1, 9, 7, 9,  8, 2, 8, 0, 6, 6, 8) // Agg Column
        .column( null, null, null, 2, 3, 5, null, null, 1, 2, 4, 5, 7) // Timestamp Key
        .column( null, null, null, 2L, 3L, 5L, null, null, 1L, 2L, 4L, 5L, 7L) // orderBy Key
        .column( null, null, null, (short)2, (short)3, (short)5, null, null, (short)1, (short)2, (short)4, (short)5, (short)7) // orderBy Key
        .column( null, null, null, (byte)2, (byte)3, (byte)5, null, null, (byte)1, (byte)2, (byte)4, (byte)5, (byte)7) // orderBy Key
        .timestampDayColumn( null, null, null, 2, 3, 5, null, null, 1, 2, 4, 5, 7) // Timestamp orderBy Key
        .timestampSecondsColumn( null, null, null, 2L, 3L, 5L, null, null, 1L, 2L, 4L, 5L, 7L)
        .timestampMicrosecondsColumn( null, null, null, 2L, 3L, 5L, null, null, 1L, 2L, 4L, 5L, 7L)
        .timestampMillisecondsColumn( null, null, null, 2L, 3L, 5L, null, null, 1L, 2L, 4L, 5L, 7L)
        .timestampNanosecondsColumn( null, null, null, 2L, 3L, 5L, null, null, 1L, 2L, 4L, 5L, 7L)
        .build()) {

      for (int orderIndex = 3; orderIndex < unsorted.getNumberOfColumns(); orderIndex++) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(orderIndex, true));
             ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8)) {
          ColumnVector sortedAggColumn = sorted.getColumn(2);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          DType type = unsorted.getColumn(orderIndex).getType();
          try (Scalar following1 = getScalar(type, 1L);
               Scalar preceding1 = getScalar(type, 1L);
               Scalar following0 = getScalar(type, 0L);
               Scalar preceding0 = getScalar(type, 0L);) {
            try (WindowOptions unboundedPrecedingOneFollowing = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .following(following1)
                .orderByColumnIndex(orderIndex)
                .build();

            WindowOptions onePrecedingUnboundedFollowing = WindowOptions.builder()
                .minPeriods(1)
                .preceding(preceding1)
                .unboundedFollowing()
                .orderByColumnIndex(orderIndex)
                .build();

            WindowOptions unboundedPrecedingAndFollowing = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .unboundedFollowing()
                .orderByColumnIndex(orderIndex)
                .build();

            WindowOptions unboundedPrecedingAndCurrentRow = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .following(following0)
                .orderByColumnIndex(orderIndex)
                .build();

            WindowOptions currentRowAndUnboundedFollowing = WindowOptions.builder()
                .minPeriods(1)
                .preceding(preceding0)
                .unboundedFollowing()
                .orderByColumnIndex(orderIndex)
                .build();) {

              try (Table windowAggResults = sorted.groupBy(0, 1)
                  .aggregateWindowsOverRanges(
                      RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingOneFollowing),
                      RollingAggregation.count().onColumn(2).overWindow(onePrecedingUnboundedFollowing),
                      RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingAndFollowing),
                      RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingAndCurrentRow),
                      RollingAggregation.count().onColumn(2).overWindow(currentRowAndUnboundedFollowing));
                   ColumnVector expect_0 = ColumnVector.fromBoxedInts(3, 3, 3, 5, 5, 6, 2, 2, 4, 4, 6, 6, 7);
                   ColumnVector expect_1 = ColumnVector.fromBoxedInts(6, 6, 6, 3, 3, 1, 7, 7, 5, 5, 3, 3, 1);
                   ColumnVector expect_2 = ColumnVector.fromBoxedInts(6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7);
                   ColumnVector expect_3 = ColumnVector.fromBoxedInts(3, 3, 3, 4, 5, 6, 2, 2, 3, 4, 5, 6, 7);
                   ColumnVector expect_4 = ColumnVector.fromBoxedInts(6, 6, 6, 3, 2, 1, 7, 7, 5, 4, 3, 2, 1)) {

                assertColumnsAreEqual(expect_0, windowAggResults.getColumn(0));
                assertColumnsAreEqual(expect_1, windowAggResults.getColumn(1));
                assertColumnsAreEqual(expect_2, windowAggResults.getColumn(2));
                assertColumnsAreEqual(expect_3, windowAggResults.getColumn(3));
                assertColumnsAreEqual(expect_4, windowAggResults.getColumn(4));
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testRangeWindowingCountUnboundedDESCWithNullsFirst() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(7, 5, 1, 9, 7, 9,  8, 2, 8, 0, 6, 6, 8) // Agg Column
        .column(null, null, null, 5, 3, 2,  null, null, 7, 5, 4, 2, 1) // Timestamp Key
        .column(null, null, null, 5L, 3L, 2L,  null, null, 7L, 5L, 4L, 2L, 1L) // orderby Key
        .column(null, null, null, (short)5, (short)3, (short)2,  null, null, (short)7, (short)5, (short)4, (short)2, (short)1) // orderby Key
        .column(null, null, null, (byte)5, (byte)3, (byte)2,  null, null, (byte)7, (byte)5, (byte)4, (byte)2, (byte)1) // orderby Key
        .timestampDayColumn(null, null, null, 5, 3, 2, null, null, 7, 5, 4, 2, 1) // Timestamp orderby Key
        .timestampSecondsColumn( null, null, null, 5L, 3L, 2L,  null, null, 7L, 5L, 4L, 2L, 1L)
        .timestampMicrosecondsColumn( null, null, null, 5L, 3L, 2L,  null, null, 7L, 5L, 4L, 2L, 1L)
        .timestampMillisecondsColumn( null, null, null, 5L, 3L, 2L,  null, null, 7L, 5L, 4L, 2L, 1L)
        .timestampNanosecondsColumn( null, null, null, 5L, 3L, 2L,  null, null, 7L, 5L, 4L, 2L, 1L)
        .build()) {

      for (int orderIndex = 3; orderIndex < unsorted.getNumberOfColumns(); orderIndex++) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.desc(orderIndex, false));
             ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8)) {
          ColumnVector sortedAggColumn = sorted.getColumn(2);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          DType type = unsorted.getColumn(orderIndex).getType();
          try (Scalar following1 = getScalar(type, 1L);
               Scalar preceding1 = getScalar(type, 1L);
               Scalar following0 = getScalar(type, 0L);
               Scalar preceding0 = getScalar(type, 0L);) {

            try (WindowOptions unboundedPrecedingOneFollowing = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .following(following1)
                .orderByColumnIndex(orderIndex)
                .orderByDescending()
                .build();

            WindowOptions onePrecedingUnboundedFollowing = WindowOptions.builder()
                .minPeriods(1)
                .preceding(preceding1)
                .unboundedFollowing()
                .orderByColumnIndex(orderIndex)
                .orderByDescending()
                .build();

            WindowOptions unboundedPrecedingAndFollowing = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .unboundedFollowing()
                .orderByColumnIndex(orderIndex)
                .orderByDescending()
                .build();

            WindowOptions unboundedPrecedingAndCurrentRow = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .following(following0)
                .orderByColumnIndex(orderIndex)
                .orderByDescending()
                .build();

            WindowOptions currentRowAndUnboundedFollowing = WindowOptions.builder()
                .minPeriods(1)
                .preceding(preceding0)
                .unboundedFollowing()
                .orderByColumnIndex(orderIndex)
                .orderByDescending()
                .build();) {

              try (Table windowAggResults = sorted.groupBy(0, 1)
                  .aggregateWindowsOverRanges(
                      RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingOneFollowing),
                      RollingAggregation.count().onColumn(2).overWindow(onePrecedingUnboundedFollowing),
                      RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingAndFollowing),
                      RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingAndCurrentRow),
                      RollingAggregation.count().onColumn(2).overWindow(currentRowAndUnboundedFollowing));
                   ColumnVector expect_0 = ColumnVector.fromBoxedInts(3, 3, 3, 4, 6, 6, 2, 2, 3, 5, 5, 7, 7);
                   ColumnVector expect_1 = ColumnVector.fromBoxedInts(6, 6, 6, 3, 2, 2, 7, 7, 5, 4, 4, 2, 2);
                   ColumnVector expect_2 = ColumnVector.fromBoxedInts(6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7);
                   ColumnVector expect_3 = ColumnVector.fromBoxedInts(3, 3, 3, 4, 5, 6, 2, 2, 3, 4, 5, 6, 7);
                   ColumnVector expect_4 = ColumnVector.fromBoxedInts(6, 6, 6, 3, 2, 1, 7, 7, 5, 4, 3, 2, 1)) {

                assertColumnsAreEqual(expect_0, windowAggResults.getColumn(0));
                assertColumnsAreEqual(expect_1, windowAggResults.getColumn(1));
                assertColumnsAreEqual(expect_2, windowAggResults.getColumn(2));
                assertColumnsAreEqual(expect_3, windowAggResults.getColumn(3));
                assertColumnsAreEqual(expect_4, windowAggResults.getColumn(4));
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testRangeWindowingCountUnboundedASCWithNullsLast() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(7, 5, 1, 9, 7, 9,  8, 2, 8, 0, 6, 6, 8) // Agg Column
        .column(2, 3, 5, null, null, null,  1, 2, 4, 5, 7, null, null) // Timestamp Key
        .column(2L, 3L, 5L, null, null, null, 1L, 2L, 4L, 5L, 7L, null, null) // order by Key
        .column((short)2, (short)3, (short)5, null, null, null, (short)1, (short)2, (short)4, (short)5, (short)7, null, null) // order by Key
        .column((byte)2, (byte)3, (byte)5, null, null, null, (byte)1, (byte)2, (byte)4, (byte)5, (byte)7, null, null) // order by Key
        .timestampDayColumn( 2, 3, 5, null, null, null,  1, 2, 4, 5, 7, null, null) // Timestamp order by Key
        .timestampSecondsColumn( 2L, 3L, 5L, null, null, null, 1L, 2L, 4L, 5L, 7L, null, null)
        .timestampMicrosecondsColumn( 2L, 3L, 5L, null, null, null, 1L, 2L, 4L, 5L, 7L, null, null)
        .timestampMillisecondsColumn( 2L, 3L, 5L, null, null, null, 1L, 2L, 4L, 5L, 7L, null, null)
        .timestampNanosecondsColumn( 2L, 3L, 5L, null, null, null, 1L, 2L, 4L, 5L, 7L, null, null)
        .build()) {
      for (int orderIndex = 3; orderIndex < unsorted.getNumberOfColumns(); orderIndex++) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(orderIndex, false));
             ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8)) {
          ColumnVector sortedAggColumn = sorted.getColumn(2);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          DType type = unsorted.getColumn(orderIndex).getType();
          try (Scalar following1 = getScalar(type, 1L);
               Scalar preceding1 = getScalar(type, 1L);
               Scalar following0 = getScalar(type, 0L);
               Scalar preceding0 = getScalar(type, 0L);) {
            try (WindowOptions unboundedPrecedingOneFollowing = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .following(following1)
                .orderByColumnIndex(orderIndex)
                .build();

            WindowOptions onePrecedingUnboundedFollowing = WindowOptions.builder()
                .minPeriods(1)
                .preceding(preceding1)
                .unboundedFollowing()
                .orderByColumnIndex(orderIndex)
                .build();

            WindowOptions unboundedPrecedingAndFollowing = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .unboundedFollowing()
                .orderByColumnIndex(orderIndex)
                .build();

            WindowOptions unboundedPrecedingAndCurrentRow = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .following(following0)
                .orderByColumnIndex(orderIndex)
                .build();

            WindowOptions currentRowAndUnboundedFollowing = WindowOptions.builder()
                .minPeriods(1)
                .preceding(preceding0)
                .unboundedFollowing()
                .orderByColumnIndex(orderIndex)
                .build();) {

              try (Table windowAggResults = sorted.groupBy(0, 1)
                  .aggregateWindowsOverRanges(
                      RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingOneFollowing),
                      RollingAggregation.count().onColumn(2).overWindow(onePrecedingUnboundedFollowing),
                      RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingAndFollowing),
                      RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingAndCurrentRow),
                      RollingAggregation.count().onColumn(2).overWindow(currentRowAndUnboundedFollowing));
                   ColumnVector expect_0 = ColumnVector.fromBoxedInts(2, 2, 3, 6, 6, 6, 2, 2, 4, 4, 5, 7, 7);
                   ColumnVector expect_1 = ColumnVector.fromBoxedInts(6, 6, 4, 3, 3, 3, 7, 7, 5, 5, 3, 2, 2);
                   ColumnVector expect_2 = ColumnVector.fromBoxedInts(6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7);
                   ColumnVector expect_3 = ColumnVector.fromBoxedInts(1, 2, 3, 6, 6, 6, 1, 2, 3, 4, 5, 7, 7);
                   ColumnVector expect_4 = ColumnVector.fromBoxedInts(6, 5, 4, 3, 3, 3, 7, 6, 5, 4, 3, 2, 2)) {

                assertColumnsAreEqual(expect_0, windowAggResults.getColumn(0));
                assertColumnsAreEqual(expect_1, windowAggResults.getColumn(1));
                assertColumnsAreEqual(expect_2, windowAggResults.getColumn(2));
                assertColumnsAreEqual(expect_3, windowAggResults.getColumn(3));
                assertColumnsAreEqual(expect_4, windowAggResults.getColumn(4));
              }
            }
          }
        }
      }
    }
  }

  @Test
  void testRangeWindowingCountUnboundedDESCWithNullsLast() {
    Integer X = null;
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1,  1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(0, 0, 0, 0, 0, 0,  1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(7, 5, 1, 9, 7, 9,  8, 2, 8, 0, 6, 6, 8) // Agg Column
        .column( 5, 3, 2, null, null, null, 7, 5, 4, 2, 1, null, null) // Timestamp Key
        .column(5L, 3L, 2L, null, null, null, 7L, 5L, 4L, 2L, 1L, null, null) // Timestamp Key
        .column((short)5, (short)3, (short)2, null, null, null, (short)7, (short)5, (short)4, (short)2, (short)1, null, null) // Timestamp Key
        .column((byte)5, (byte)3, (byte)2, null, null, null, (byte)7, (byte)5, (byte)4, (byte)2, (byte)1, null, null) // Timestamp Key
        .timestampDayColumn( 5, 3, 2, X, X, X,  7, 5, 4, 2, 1, X, X) // Timestamp Key
        .timestampSecondsColumn( 5L, 3L, 2L, null, null, null, 7L, 5L, 4L, 2L, 1L, null, null)
        .timestampMicrosecondsColumn( 5L, 3L, 2L, null, null, null, 7L, 5L, 4L, 2L, 1L, null, null)
        .timestampMillisecondsColumn( 5L, 3L, 2L, null, null, null, 7L, 5L, 4L, 2L, 1L, null, null)
        .timestampNanosecondsColumn( 5L, 3L, 2L, null, null, null, 7L, 5L, 4L, 2L, 1L, null, null)
        .build()) {
      for (int orderIndex = 3; orderIndex < unsorted.getNumberOfColumns(); orderIndex++) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.desc(orderIndex, true));
             ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6, 8)) {
          ColumnVector sortedAggColumn = sorted.getColumn(2);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          DType type = unsorted.getColumn(orderIndex).getType();
          try (Scalar following1 = getScalar(type, 1L);
               Scalar preceding1 = getScalar(type, 1L);
               Scalar following0 = getScalar(type, 0L);
               Scalar preceding0 = getScalar(type, 0L);) {
            try (WindowOptions unboundedPrecedingOneFollowing = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .following(following1)
                .orderByColumnIndex(orderIndex)
                .orderByDescending()
                .build();

            WindowOptions onePrecedingUnboundedFollowing = WindowOptions.builder()
                .minPeriods(1)
                .preceding(preceding1)
                .unboundedFollowing()
                .orderByColumnIndex(orderIndex)
                .orderByDescending()
                .build();

            WindowOptions unboundedPrecedingAndFollowing = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .unboundedFollowing()
                .orderByColumnIndex(orderIndex)
                .orderByDescending()
                .build();

            WindowOptions unboundedPrecedingAndCurrentRow = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .following(following0)
                .orderByColumnIndex(orderIndex)
                .orderByDescending()
                .build();

            WindowOptions currentRowAndUnboundedFollowing = WindowOptions.builder()
                .minPeriods(1)
                .preceding(preceding0)
                .unboundedFollowing()
                .orderByColumnIndex(orderIndex)
                .orderByDescending()
                .build();) {

              try (Table windowAggResults = sorted.groupBy(0, 1)
                  .aggregateWindowsOverRanges(
                      RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingOneFollowing),
                      RollingAggregation.count().onColumn(2).overWindow(onePrecedingUnboundedFollowing),
                      RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingAndFollowing),
                      RollingAggregation.count().onColumn(2).overWindow(unboundedPrecedingAndCurrentRow),
                      RollingAggregation.count().onColumn(2).overWindow(currentRowAndUnboundedFollowing));
                   ColumnVector expect_0 = ColumnVector.fromBoxedInts(1, 3, 3, 6, 6, 6, 1, 3, 3, 5, 5, 7, 7);
                   ColumnVector expect_1 = ColumnVector.fromBoxedInts(6, 5, 5, 3, 3, 3, 7, 6, 6, 4, 4, 2, 2);
                   ColumnVector expect_2 = ColumnVector.fromBoxedInts(6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7);
                   ColumnVector expect_3 = ColumnVector.fromBoxedInts(1, 2, 3, 6, 6, 6, 1, 2, 3, 4, 5, 7, 7);
                   ColumnVector expect_4 = ColumnVector.fromBoxedInts(6, 5, 4, 3, 3, 3, 7, 6, 5, 4, 3, 2, 2)) {

                assertColumnsAreEqual(expect_0, windowAggResults.getColumn(0));
                assertColumnsAreEqual(expect_1, windowAggResults.getColumn(1));
                assertColumnsAreEqual(expect_2, windowAggResults.getColumn(2));
                assertColumnsAreEqual(expect_3, windowAggResults.getColumn(3));
                assertColumnsAreEqual(expect_4, windowAggResults.getColumn(4));
              }
            }
          }
        }
      }
    }
  }

  /**
   * Helper for constructing BigInteger from int
   * @param x Integer value
   * @return BigInteger equivalent of x
   */
  private static BigInteger big(int x)
  {
    return new BigInteger("" + x);
  }

  /**
   * Helper to get scalar for preceding == Decimal(value),
   * with data width depending upon the order-by
   * column index:
   *   orderby_col_idx = 2 -> Decimal32
   *   orderby_col_idx = 3 -> Decimal64
   *   orderby_col_idx = 4 -> Decimal128
   */
  private static Scalar getDecimalScalarRangeBounds(int scale, int unscaledValue, int orderby_col_idx)
  {
    switch(orderby_col_idx)
    {
      case 2: return Scalar.fromDecimal(scale, unscaledValue);
      case 3: return Scalar.fromDecimal(scale, Long.valueOf(unscaledValue));
      case 4: return Scalar.fromDecimal(scale, big(unscaledValue));
      default:
        throw new IllegalStateException("Unexpected order by column index: "
                                        + orderby_col_idx);
    }
  }

  @Test
  void testRangeWindowsWithDecimalOrderBy() {
    try (Table unsorted = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
        .decimal32Column(-1, 4000, 3000, 2000, 1000,
                             4000, 3000, 2000, 1000,
                             4000, 3000, 2000, 1000) // Decimal OBY Key
        .decimal64Column(-1, 4000l, 3000l, 2000l, 1000l,
                             4000l, 3000l, 2000l, 1000l,
                             4000l, 3000l, 2000l, 1000l) // Decimal OBY Key
        .decimal128Column(-1, RoundingMode.UNNECESSARY,
                              big(4000), big(3000), big(2000), big(1000),
                              big(4000), big(3000), big(2000), big(1000),
                              big(4000), big(3000), big(2000), big(1000))
        .column(9, 1, 5, 7, 2, 8, 9, 7, 6, 6, 0, 8) // Agg Column
        .build()) {

      // Columns 2,3,4 are decimal order-by columns of type DECIMAL32, DECIMAL64,
      // and DECIMAL128 respectively, with similarly ordered values.
      // In the following loop, each decimal type is tested as the order-by column,
      // producing the same results with similar range bounds.
      for (int decimal_oby_col_idx = 2; decimal_oby_col_idx <= 4; ++decimal_oby_col_idx) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(0),
                                             OrderByArg.asc(1),
                                             OrderByArg.asc(decimal_oby_col_idx));
            ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6)) {
          ColumnVector sortedAggColumn = sorted.getColumn(5);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          // Test Window functionality with range window (200 PRECEDING and 100 FOLLOWING)
          try (Scalar preceding200 = getDecimalScalarRangeBounds(0, 200, decimal_oby_col_idx);
               Scalar following100 = getDecimalScalarRangeBounds(2, 1, decimal_oby_col_idx);
               WindowOptions window = WindowOptions.builder()
                .minPeriods(1)
                .window(preceding200, following100)
                .orderByColumnIndex(decimal_oby_col_idx)
                .build()) {

            try (Table windowAggResults = sorted.groupBy(0, 1)
                                                .aggregateWindowsOverRanges(RollingAggregation.count()
                                                                                              .onColumn(5)
                                                                                              .overWindow(window));
                ColumnVector expect = ColumnVector.fromBoxedInts(2, 3, 4, 3, 2, 3, 4, 3, 2, 3, 4, 3)) {
              assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
            }
          }

          // Test Window functionality with range window (UNBOUNDED PRECEDING and CURRENT ROW)
          try (Scalar current_row = getDecimalScalarRangeBounds(0, 0, decimal_oby_col_idx);
               WindowOptions window = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .following(current_row)
                .orderByColumnIndex(decimal_oby_col_idx)
                .build()) {

            try (Table windowAggResults = sorted.groupBy(0, 1)
                                                .aggregateWindowsOverRanges(RollingAggregation.count()
                                                                                              .onColumn(5)
                                                                                              .overWindow(window));
                ColumnVector expect = ColumnVector.fromBoxedInts(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4)) {
              assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
            }
          }

          // Test Window functionality with range window (UNBOUNDED PRECEDING and UNBOUNDED FOLLOWING)
          try (WindowOptions window = WindowOptions.builder()
                .minPeriods(1)
                .unboundedPreceding()
                .unboundedFollowing()
                .orderByColumnIndex(decimal_oby_col_idx)
                .build()) {

            try (Table windowAggResults = sorted.groupBy(0, 1)
                                                .aggregateWindowsOverRanges(RollingAggregation.count()
                                                                                              .onColumn(5)
                                                                                              .overWindow(window));
                ColumnVector expect = ColumnVector.fromBoxedInts(4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4)) {
              assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
            }
          }
        }
      }
    }
  }

  /**
   * Helper to get scalar for preceding == Decimal(value),
   * with data width depending upon the order-by column index:
   *   orderby_col_idx = 2 -> FLOAT32
   *   orderby_col_idx = 3 -> FLOAT64
   */
  private static Scalar getFloatingPointScalarRangeBounds(float value, int orderby_col_idx)
  {
    switch(orderby_col_idx)
    {
      case 2: return Scalar.fromFloat(value);
      case 3: return Scalar.fromDouble(Double.valueOf(value));
      default:
        throw new IllegalStateException("Unexpected order by column index: "
                + orderby_col_idx);
    }
  }

  @Test
  void testRangeWindowsWithFloatOrderBy() {
    try (Table unsorted = new Table.TestBuilder()
            .column(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1) // GBY Key
            .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3) // GBY Key
            .column(400f, 300f, 200f, 100f,
                    400f, 300f, 200f, 100f,
                    400f, 300f, 200f, 100f) // Float OBY Key
            .column(400.0, 300.0, 200.0, 100.0,
                    400.0, 300.0, 200.0, 100.0,
                    400.0, 300.0, 200.0, 100.0) // Double OBY Key
            .column(9, 1, 5, 7, 2, 8, 9, 7, 6, 6, 0, 8) // Agg Column
            .build()) {

      // Columns 2-3 are order-by columns of type FLOAT32 and FLOAT64 respectively, with similarly ordered values.
      // In the following loop, each float type is tested as the order-by column,
      // producing the same results with similar range bounds.
      for (int float_oby_col_idx = 2; float_oby_col_idx <= 3; ++float_oby_col_idx) {
        try (Table sorted = unsorted.orderBy(OrderByArg.asc(0),
                OrderByArg.asc(1),
                OrderByArg.asc(float_oby_col_idx));
             ColumnVector expectSortedAggColumn = ColumnVector.fromBoxedInts(7, 5, 1, 9, 7, 9, 8, 2, 8, 0, 6, 6)) {
          ColumnVector sortedAggColumn = sorted.getColumn(4);
          assertColumnsAreEqual(expectSortedAggColumn, sortedAggColumn);

          // Test Window functionality with range window (200 PRECEDING and 100 FOLLOWING)
          try (Scalar preceding200 = getFloatingPointScalarRangeBounds(200, float_oby_col_idx);
               Scalar following100 = getFloatingPointScalarRangeBounds(100, float_oby_col_idx);
               WindowOptions window = WindowOptions.builder()
                       .minPeriods(1)
                       .window(preceding200, following100)
                       .orderByColumnIndex(float_oby_col_idx)
                       .build()) {

            try (Table windowAggResults = sorted.groupBy(0, 1)
                    .aggregateWindowsOverRanges(RollingAggregation.count()
                            .onColumn(4)
                            .overWindow(window));
                 ColumnVector expect = ColumnVector.fromBoxedInts(2, 3, 4, 3, 2, 3, 4, 3, 2, 3, 4, 3)) {
              assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
            }
          }

          // Test Window functionality with range window (UNBOUNDED PRECEDING and CURRENT ROW)
          try (Scalar current_row = getFloatingPointScalarRangeBounds(0, float_oby_col_idx);
               WindowOptions window = WindowOptions.builder()
                       .minPeriods(1)
                       .unboundedPreceding()
                       .following(current_row)
                       .orderByColumnIndex(float_oby_col_idx)
                       .build()) {

            try (Table windowAggResults = sorted.groupBy(0, 1)
                    .aggregateWindowsOverRanges(RollingAggregation.count()
                            .onColumn(4)
                            .overWindow(window));
                 ColumnVector expect = ColumnVector.fromBoxedInts(1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4)) {
              assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
            }
          }

          // Test Window functionality with range window (UNBOUNDED PRECEDING and UNBOUNDED FOLLOWING)
          try (WindowOptions window = WindowOptions.builder()
                  .minPeriods(1)
                  .unboundedPreceding()
                  .unboundedFollowing()
                  .orderByColumnIndex(float_oby_col_idx)
                  .build()) {

            try (Table windowAggResults = sorted.groupBy(0, 1)
                    .aggregateWindowsOverRanges(RollingAggregation.count()
                            .onColumn(4)
                            .overWindow(window));
                 ColumnVector expect = ColumnVector.fromBoxedInts(4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4)) {
              assertColumnsAreEqual(expect, windowAggResults.getColumn(0));
            }
          }
        }
      }
    }
  }

  @Test
  void testGroupByCountWithNulls() {
    try (Table t1 = new Table.TestBuilder().column(null, null,    1,    1,    1,    1)
                                           .column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    1, null, null,    1,    1)
                                           .column(   1,    1,    1, null,    1,    1)
                                           .build()) {
      try (Table tmp = t1.groupBy(0).aggregate(
          GroupByAggregation.count().onColumn(1),
          GroupByAggregation.count().onColumn(2),
          GroupByAggregation.count().onColumn(3));
           Table t3 = tmp.orderBy(OrderByArg.asc(0, true));
           HostColumnVector groupCol = t3.getColumn(0).copyToHost();
           HostColumnVector countCol = t3.getColumn(1).copyToHost();
           HostColumnVector nullCountCol = t3.getColumn(2).copyToHost();
           HostColumnVector nullCountCol2 = t3.getColumn(3).copyToHost()) {
        // verify t3
        assertEquals(2, t3.getRowCount());

        // compare the grouping columns
        assertTrue(groupCol.isNull(0));
        assertEquals(groupCol.getInt(1), 1);

        // compare the agg columns
        // count(1)
        assertEquals(countCol.getInt(0), 2);
        assertEquals(countCol.getInt(1), 4);

        // count(2)
        assertEquals(nullCountCol.getInt(0), 2);
        assertEquals(nullCountCol.getInt(1), 2); // counts only the non-nulls

        // count(3)
        assertEquals(nullCountCol2.getInt(0), 2);
        assertEquals(nullCountCol2.getInt(1), 3); // counts only the non-nulls
      }
    }
  }

  @Test
  void testGroupByCountWithNullsIncluded() {
    try (Table t1 = new Table.TestBuilder()
            .column(null, null,    1,    1,    1,    1)
            .column(   1,    1,    1,    1,    1,    1)
            .column(   1,    1, null, null,    1,    1)
            .column(   1,    1,    1, null,    1,    1)
            .build()) {
      try (Table tmp = t1.groupBy(0).aggregate(
          GroupByAggregation.count(NullPolicy.INCLUDE).onColumn(1),
          GroupByAggregation.count(NullPolicy.INCLUDE).onColumn(2),
          GroupByAggregation.count(NullPolicy.INCLUDE).onColumn(3),
          GroupByAggregation.count().onColumn(3));
           Table t3 = tmp.orderBy(OrderByArg.asc(0, true));
           HostColumnVector groupCol = t3.getColumn(0).copyToHost();
           HostColumnVector countCol = t3.getColumn(1).copyToHost();
           HostColumnVector nullCountCol = t3.getColumn(2).copyToHost();
           HostColumnVector nullCountCol2 = t3.getColumn(3).copyToHost();
           HostColumnVector nullCountCol3 = t3.getColumn(4).copyToHost()) {
        // verify t3
        assertEquals(2, t3.getRowCount());

        // compare the grouping columns
        assertTrue(groupCol.isNull(0));
        assertEquals(groupCol.getInt(1), 1);

        // compare the agg columns
        // count(1, true)
        assertEquals(countCol.getInt(0), 2);
        assertEquals(countCol.getInt(1), 4);

        // count(2, true)
        assertEquals(nullCountCol.getInt(0), 2);
        assertEquals(nullCountCol.getInt(1), 4); // counts including nulls

        // count(3, true)
        assertEquals(nullCountCol2.getInt(0), 2);
        assertEquals(nullCountCol2.getInt(1), 4); // counts including nulls

        // count(3)
        assertEquals(nullCountCol3.getInt(0), 2);
        assertEquals(nullCountCol3.getInt(1), 3); // counts only the non-nulls
      }
    }
  }

  @Test
  void testGroupByCountWithCollapsingNulls() {
    try (Table t1 = new Table.TestBuilder()
        .column(null, null,    1,    1,    1,    1)
        .column(   1,    1,    1,    1,    1,    1)
        .column(   1,    1, null, null,    1,    1)
        .column(   1,    1,    1, null,    1,    1)
        .build()) {

      GroupByOptions options = GroupByOptions.builder()
          .withIgnoreNullKeys(true)
          .build();

      try (Table tmp = t1.groupBy(options, 0).aggregate(
          GroupByAggregation.count().onColumn(1),
          GroupByAggregation.count().onColumn(2),
          GroupByAggregation.count().onColumn(3));
           Table t3 = tmp.orderBy(OrderByArg.asc(0, true));
           HostColumnVector groupCol = t3.getColumn(0).copyToHost();
           HostColumnVector countCol = t3.getColumn(1).copyToHost();
           HostColumnVector nullCountCol = t3.getColumn(2).copyToHost();
           HostColumnVector nullCountCol2 = t3.getColumn(3).copyToHost()) {
        // (null, 1) => became (1) because we are ignoring nulls
        assertEquals(1, t3.getRowCount());

        // compare the grouping columns
        assertEquals(groupCol.getInt(0), 1);

        // compare the agg columns
        // count(1)
        assertEquals(countCol.getInt(0), 4);

        // count(2)
        assertEquals(nullCountCol.getInt(0), 2); // counts only the non-nulls

        // count(3)
        assertEquals(nullCountCol2.getInt(0), 3); // counts only the non-nulls
      }
    }
  }

  @Test
  void testGroupByMax() {
    try (Table t1 = new Table.TestBuilder().column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    3,    3,    5,    5,    0)
                                           .column(12.0, 14.0, 13.0, 17.0, 17.0, 17.0)
                                           .build()) {
      try (Table t3 = t1.groupBy(0, 1).aggregate(GroupByAggregation.max().onColumn(2));
           HostColumnVector aggOut1 = t3.getColumn(2).copyToHost()) {
        // verify t3
        assertEquals(4, t3.getRowCount());
        Map<Double, Integer> expectedAggregateResult = new HashMap() {
          {
            // value, count
            put(12.0, 1);
            put(14.0, 1);
            put(17.0, 2);
          }
        };
        for (int i = 0; i < 4; ++i) {
          Double key = aggOut1.getDouble(i);
          assertTrue(expectedAggregateResult.containsKey(key));
          Integer count = expectedAggregateResult.get(key);
          if (count == 1) {
            expectedAggregateResult.remove(key);
          } else {
            expectedAggregateResult.put(key, count - 1);
          }
        }
      }
    }
  }

  @Test
  void testGroupByArgMax() {
    // ArgMax is a sort based aggregation.
    try (Table t1 = new Table.TestBuilder()
            .column(   1,    1,    1,    1,    1,    1)
            .column(   0,    1,    2,    2,    3,    3)
            .column(17.0, 14.0, 14.0, 17.0, 17.1, 17.0)
            .build()) {
      try (Table t3 = t1.groupBy(0, 1)
              .aggregate(GroupByAggregation.argMax().onColumn(2));
           Table sorted = t3
              .orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           Table expected = new Table.TestBuilder()
                   .column(1, 1, 1, 1)
                   .column(0, 1, 2, 3)
                   .column(0, 1, 3, 4)
                   .build()) {
        assertTablesAreEqual(expected, sorted);
      }
    }
  }

  @Test
  void testGroupByArgMin() {
    // ArgMin is a sort based aggregation
    try (Table t1 = new Table.TestBuilder()
            .column(   1,    1,    1,    1,    1,    1)
            .column(   0,    1,    2,    2,    3,    3)
            .column(17.0, 14.0, 14.0, 17.0, 17.1, 17.0)
            .build()) {
      try (Table t3 = t1.groupBy(0, 1)
              .aggregate(GroupByAggregation.argMin().onColumn(2));
           Table sorted = t3
                   .orderBy(OrderByArg.asc(0), OrderByArg.asc(1), OrderByArg.asc(2));
           Table expected = new Table.TestBuilder()
                   .column(1, 1, 1, 1)
                   .column(0, 1, 2, 3)
                   .column(0, 1, 2, 5)
                   .build()) {
        assertTablesAreEqual(expected, sorted);
      }
    }
  }

  @Test
  void testGroupByMinBool() {
    try (Table t1 = new Table.TestBuilder()
        .column(true, null, false, true, null, null)
        .column(   1,    1,     2,    2,    3,    3).build();
         Table other = t1.groupBy(1).aggregate(GroupByAggregation.min().onColumn(0));
         Table ordered = other.orderBy(OrderByArg.asc(0));
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3)
             .column (true, false, null)
             .build()) {
      assertTablesAreEqual(expected, ordered);
    }
  }

  @Test
  void testGroupByMaxBool() {
    try (Table t1 = new Table.TestBuilder()
        .column(false, null, false, true, null, null)
        .column(   1,    1,     2,    2,    3,    3).build();
         Table other = t1.groupBy(1).aggregate(GroupByAggregation.max().onColumn(0));
         Table ordered = other.orderBy(OrderByArg.asc(0));
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3)
             .column (false, true, null)
             .build()) {
      assertTablesAreEqual(expected, ordered);
    }
  }

  @Test
  void testGroupByDuplicateAggregates() {
    try (Table t1 = new Table.TestBuilder().column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    3,    3,    5,    5,    0)
                                           .column(12.0, 14.0, 13.0, 15.0, 17.0, 18.0)
                                           .build();
         Table expected = new Table.TestBuilder()
             .column(1, 1, 1, 1)
             .column(1, 3, 5, 0)
             .column(12.0, 14.0, 17.0, 18.0)
             .column(12.0, 13.0, 15.0, 18.0)
             .column(12.0, 13.0, 15.0, 18.0)
             .column(12.0, 14.0, 17.0, 18.0)
             .column(12.0, 13.0, 15.0, 18.0)
             .column(   1,    2,    2,    1).build()) {
      try (Table t3 = t1.groupBy(0, 1)
          .aggregate(
              GroupByAggregation.max().onColumn(2),
              GroupByAggregation.min().onColumn(2),
              GroupByAggregation.min().onColumn(2),
              GroupByAggregation.max().onColumn(2),
              GroupByAggregation.min().onColumn(2),
              GroupByAggregation.count().onColumn(1));
          Table t4 = t3.orderBy(OrderByArg.asc(2))) {
        // verify t4
        assertEquals(4, t4.getRowCount());
        assertTablesAreEqual(t4, expected);

        assertEquals(t3.getColumn(0).getRefCount(), 1);
        assertEquals(t3.getColumn(1).getRefCount(), 1);
        assertEquals(t3.getColumn(2).getRefCount(), 2);
        assertEquals(t3.getColumn(3).getRefCount(), 3);
        assertEquals(t3.getColumn(4).getRefCount(), 3);
        assertEquals(t3.getColumn(5).getRefCount(), 2);
        assertEquals(t3.getColumn(6).getRefCount(), 3);
      }
    }
  }

  @Test
  void testGroupByMin() {
    try (Table t1 = new Table.TestBuilder().column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    3,    3,    5,    5,    0)
                                           .column(  12,   14,   13,   17,   17,   17)
                                           .build()) {
      try (Table t3 = t1.groupBy(0, 1).aggregate(GroupByAggregation.min().onColumn(2));
           HostColumnVector aggOut0 = t3.getColumn(2).copyToHost()) {
        // verify t3
        assertEquals(4, t3.getRowCount());
        Map<Integer, Integer> expectedAggregateResult = new HashMap() {
          {
            // value, count
            put(12, 1);
            put(13, 1);
            put(17, 2);
          }
        };
        // check to see the aggregate column type depends on the source column
        // in this case the source column is Integer, therefore the result should be Integer type
        assertEquals(DType.INT32, aggOut0.getType());
        for (int i = 0; i < 4; ++i) {
          int key = aggOut0.getInt(i);
          assertTrue(expectedAggregateResult.containsKey(key));
          Integer count = expectedAggregateResult.get(key);
          if (count == 1) {
            expectedAggregateResult.remove(key);
          } else {
            expectedAggregateResult.put(key, count - 1);
          }
        }
      }
    }
  }

  @Test
  void testGroupBySum() {
    try (Table t1 = new Table.TestBuilder().column(   1,    1,    1,    1,    1,    1)
                                           .column(   1,    3,    3,    5,    5,    0)
                                           .column(12.0, 14.0, 13.0, 17.0, 17.0, 17.0)
                                           .build()) {
      try (Table t3 = t1.groupBy(0, 1).aggregate(GroupByAggregation.sum().onColumn(2));
           HostColumnVector aggOut1 = t3.getColumn(2).copyToHost()) {
        // verify t3
        assertEquals(4, t3.getRowCount());
        Map<Double, Integer> expectedAggregateResult = new HashMap() {
          {
            // value, count
            put(12.0, 1);
            put(27.0, 1);
            put(34.0, 1);
            put(17.0, 1);
          }
        };
        for (int i = 0; i < 4; ++i) {
          Double key = aggOut1.getDouble(i);
          assertTrue(expectedAggregateResult.containsKey(key));
          Integer count = expectedAggregateResult.get(key);
          if (count == 1) {
            expectedAggregateResult.remove(key);
          } else {
            expectedAggregateResult.put(key, count - 1);
          }
        }
      }
    }
  }

  @Test
  void testGroupByM2() {
    // A trivial test:
    try (Table input = new Table.TestBuilder().column(1, 2, 3, 1, 2, 2, 1, 3, 3, 2)
             .column(0, 1, -2, 3, -4, -5, -6, 7, -8, 9)
             .build();
         Table results = input.groupBy(0).aggregate(GroupByAggregation.M2()
               .onColumn(1));
         Table resultsSorted = results.orderBy(OrderByArg.asc(0));
         Table expected = new Table.TestBuilder().column(1, 2, 3)
             .column(42.0, 122.75, 114.0)
             .build()) {
      assertTablesAreEqual(expected, resultsSorted);
    }

    // Test with values have nulls (the values associated with key=2 has both nulls and non-nulls,
    // while the values associated with key=5 are all nulls):
    try (Table input = new Table.TestBuilder().column(1, 2, 5, 3, 4, 5, 2, 3, 2, 5)
             .column(0, null, null, 2, 3, null, 5, 6, 7, null)
             .build();
         Table results = input.groupBy(0).aggregate(GroupByAggregation.M2()
             .onColumn(1));
         Table resultsSorted = results.orderBy(OrderByArg.asc(0));
         Table expected = new Table.TestBuilder().column(1, 2, 3, 4, 5)
             .column(0.0, 2.0, 8.0, 0.0, 0.0)
             .build()) {
      assertTablesAreEqual(expected, resultsSorted);
    }

    // Test with floating-point values having NaN:
    try (Table input = new Table.TestBuilder().column(4, 3, 1, 2, 3, 1, 2, 2, 1, null, 3, 2, 4, 4)
             .column(null, null, 0.0, 1.0, 2.0, 3.0, 4.0, Double.NaN, 6.0, 7.0, 8.0, 9.0, 10.0, Double.NaN)
             .build();
         Table results = input.groupBy(0).aggregate(GroupByAggregation.M2()
             .onColumn(1));
         Table resultsSorted = results.orderBy(OrderByArg.asc(0));
         Table expected = new Table.TestBuilder().column(1, 2, 3, 4, null)
             .column(18.0, Double.NaN, 18.0, Double.NaN, 0.0)
             .build()) {
      assertTablesAreEqual(expected, resultsSorted);
    }

    // Test with floating-point values having NaN and +/- Inf
    // (The values associated with:
    //   key=1: have only NaN
    //   key=2: have only +Inf
    //   key=3: have only -Inf
    //   key=4: have NaN and +/- Inf,
    //   key=5: have normal numbers):
    try (Table input = new Table.TestBuilder().column(1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4)
             .column(Double.NaN,
                     Double.POSITIVE_INFINITY,
                     Double.NEGATIVE_INFINITY,
                     Double.POSITIVE_INFINITY,
                     5.0,
                     //
                     Double.NaN,
                     Double.POSITIVE_INFINITY,
                     Double.NEGATIVE_INFINITY,
                     Double.NEGATIVE_INFINITY,
                     10.0,
                     //
                     Double.NaN,
                     Double.POSITIVE_INFINITY,
                     Double.NEGATIVE_INFINITY,
                     Double.POSITIVE_INFINITY)
             .build();
         Table results = input.groupBy(0).aggregate(GroupByAggregation.M2()
             .onColumn(1));
         Table resultsSorted = results.orderBy(OrderByArg.asc(0));
         Table expected = new Table.TestBuilder().column(1, 2, 3, 4, 5)
             .column(Double.NaN, Double.NaN, Double.NaN, Double.NaN, 12.5)
             .build()) {
      assertTablesAreEqual(expected, resultsSorted);
    }
  }

  private static void assertSumOverflowResult(Table results,
                                                  int[] expectedKeys,
                                                  long[] expectedSums,
                                                  boolean[] expectedOvf) {
    ColumnVector structCol = results.getColumn(1);
    assertEquals(DType.STRUCT, structCol.getType());
    try (ColumnView ovfChild = structCol.getChildColumnView(1);
         ColumnVector ovfCol = ovfChild.copyToColumnVector();
         ColumnVector expectedKeyCol = ColumnVector.fromInts(expectedKeys);
         ColumnVector expectedOvfCol = ColumnVector.fromBooleans(expectedOvf)) {
      assertColumnsAreEqual(expectedKeyCol, results.getColumn(0));
      assertColumnsAreEqual(expectedOvfCol, ovfCol);
      if (expectedSums != null) {
        try (ColumnView sumChild = structCol.getChildColumnView(0);
             ColumnVector sumCol = sumChild.copyToColumnVector();
             ColumnVector expectedSumCol = ColumnVector.fromLongs(expectedSums)) {
          assertEquals(DType.INT64, sumCol.getType());
          assertColumnsAreEqual(expectedSumCol, sumCol);
        }
      }
    }
  }

  @Test
  void testGroupByHashSumOverflow() {
    try (Table input = new Table.TestBuilder()
             .column(1, 2, 3, 1, 2, 2, 1, 3, 3, 2)
             .column(10L, 20L, 30L, 11L, 21L, 22L, 12L, 31L, 32L, 23L)
             .build();
         Table results = input.groupBy(0).aggregate(
             GroupByAggregation.sumOverflow().onColumn(1));
         Table sorted = results.orderBy(OrderByArg.asc(0))) {
      assertEquals(2, sorted.getNumberOfColumns());
      assertEquals(3, sorted.getRowCount());
      assertSumOverflowResult(sorted,
          new int[]{1, 2, 3}, new long[]{33L, 86L, 93L}, new boolean[]{false, false, false});
    }
  }

  @Test
  void testGroupByHashSumOverflowDetectsOverflow() {
    try (Table input = new Table.TestBuilder()
             .column(1, 1, 2, 2)
             .column(Long.MAX_VALUE, Long.MAX_VALUE, 3L, 4L)
             .build();
         Table results = input.groupBy(0).aggregate(
             GroupByAggregation.sumOverflow().onColumn(1));
         Table sorted = results.orderBy(OrderByArg.asc(0))) {
      assertSumOverflowResult(sorted, new int[]{1, 2}, null, new boolean[]{true, false});
    }
  }

  @Test
  void testGroupByHashSumOverflowInt32() {
    // Int32 input: hash groupby supports it (reduction does not).
    try (Table input = new Table.TestBuilder()
             .column(1, 2, 1, 2)
             .column(1, 10, 2, 20)
             .build();
         Table results = input.groupBy(0).aggregate(
             GroupByAggregation.sumOverflow().onColumn(1));
         Table sorted = results.orderBy(OrderByArg.asc(0))) {
      ColumnVector structCol = sorted.getColumn(1);
      assertEquals(DType.STRUCT, structCol.getType());
      try (ColumnView sumChild = structCol.getChildColumnView(0);
           ColumnView ovfChild = structCol.getChildColumnView(1);
           ColumnVector sumCol = sumChild.copyToColumnVector();
           ColumnVector ovfCol = ovfChild.copyToColumnVector();
           ColumnVector expectedSum = ColumnVector.fromInts(3, 30);
           ColumnVector expectedOvf = ColumnVector.fromBooleans(false, false)) {
        assertEquals(DType.INT32, sumCol.getType());
        assertColumnsAreEqual(expectedSum, sumCol);
        assertColumnsAreEqual(expectedOvf, ovfCol);
      }
    }
  }

  @Test
  void testGroupBySortSumOverflow() {
    try (Table input = new Table.TestBuilder()
             .column(1, 1, 2, 2)
             .column(1L, 2L, 3L, 4L)
             .build();
         // median() is sort-only, so it forces the sort-based groupby path
         Table results = input.groupBy(0).aggregate(
             GroupByAggregation.sumOverflow().onColumn(1),
             GroupByAggregation.median().onColumn(1));
         Table sorted = results.orderBy(OrderByArg.asc(0))) {
      assertSumOverflowResult(sorted,
          new int[]{1, 2}, new long[]{3L, 7L}, new boolean[]{false, false});
    }
  }

  @Test
  void testGroupBySortSumOverflowDetectsOverflow() {
    try (Table input = new Table.TestBuilder()
             .column(1, 1, 2, 2)
             .column(Long.MAX_VALUE, Long.MAX_VALUE, 3L, 4L)
             .build();
         // median() is sort-only, so it forces the sort-based groupby path
         Table results = input.groupBy(0).aggregate(
             GroupByAggregation.sumOverflow().onColumn(1),
             GroupByAggregation.median().onColumn(1));
         Table sorted = results.orderBy(OrderByArg.asc(0))) {
      assertSumOverflowResult(sorted, new int[]{1, 2}, null, new boolean[]{true, false});
    }
  }

  // For SUM_OVERFLOW on fixed-point inputs, cudf's hash groupby produces a
  // STRUCT whose sum-child has the same decimal type AND scale as the input
  // (target_type_impl<fixed_point, SUM_OVERFLOW> uses Source as the sum
  // type at cpp/include/cudf/detail/aggregation/aggregation.hpp). The three
  // tests below assert both the values and that scale is preserved end-to-end
  // through the JNI struct-column extraction path.
  @Test
  void testGroupByHashSumOverflowDecimal32() {
    final int scale = -2;
    try (Table input = new Table.TestBuilder()
             .column(1, 2, 1, 2, 1)
             .decimal32Column(scale, 100, 200, 150, 250, 50)  // 1.00, 2.00, 1.50, 2.50, 0.50
             .build();
         Table results = input.groupBy(0).aggregate(
             GroupByAggregation.sumOverflow().onColumn(1));
         Table sorted = results.orderBy(OrderByArg.asc(0))) {
      ColumnVector keyCol = sorted.getColumn(0);
      ColumnVector structCol = sorted.getColumn(1);
      assertEquals(DType.STRUCT, structCol.getType());

      try (ColumnView sumChild = structCol.getChildColumnView(0);
           ColumnView ovfChild = structCol.getChildColumnView(1);
           ColumnVector sumCol = sumChild.copyToColumnVector();
           ColumnVector ovfCol = ovfChild.copyToColumnVector();
           ColumnVector expectedKeys = ColumnVector.fromInts(1, 2);
           ColumnVector expectedSum = ColumnVector.decimalFromInts(scale, 300, 450);
           ColumnVector expectedOvf = ColumnVector.fromBooleans(false, false)) {
        assertEquals(DType.DTypeEnum.DECIMAL32, sumCol.getType().getTypeId());
        assertEquals(scale, sumCol.getType().getScale());
        assertColumnsAreEqual(expectedKeys, keyCol);
        assertColumnsAreEqual(expectedSum, sumCol);
        assertColumnsAreEqual(expectedOvf, ovfCol);
      }
    }
  }

  @Test
  void testGroupByHashSumOverflowDecimal64() {
    final int scale = -4;
    try (Table input = new Table.TestBuilder()
             .column(1, 2, 1, 2)
             .decimal64Column(scale, 10000L, 20000L, 30000L, 40000L)  // 1.0000, 2.0000, ...
             .build();
         Table results = input.groupBy(0).aggregate(
             GroupByAggregation.sumOverflow().onColumn(1));
         Table sorted = results.orderBy(OrderByArg.asc(0))) {
      ColumnVector structCol = sorted.getColumn(1);
      assertEquals(DType.STRUCT, structCol.getType());

      try (ColumnView sumChild = structCol.getChildColumnView(0);
           ColumnView ovfChild = structCol.getChildColumnView(1);
           ColumnVector sumCol = sumChild.copyToColumnVector();
           ColumnVector ovfCol = ovfChild.copyToColumnVector();
           ColumnVector expectedSum = ColumnVector.decimalFromLongs(scale, 40000L, 60000L);
           ColumnVector expectedOvf = ColumnVector.fromBooleans(false, false)) {
        assertEquals(DType.DTypeEnum.DECIMAL64, sumCol.getType().getTypeId());
        assertEquals(scale, sumCol.getType().getScale());
        assertColumnsAreEqual(expectedSum, sumCol);
        assertColumnsAreEqual(expectedOvf, ovfCol);
      }
    }
  }

  @Test
  void testGroupByHashSumOverflowDecimal128() {
    final int scale = -10;
    BigInteger v1 = BigInteger.valueOf(123456789L).multiply(BigInteger.TEN.pow(10));
    BigInteger v2 = BigInteger.valueOf(987654321L).multiply(BigInteger.TEN.pow(10));
    BigInteger v3 = BigInteger.valueOf(111111111L).multiply(BigInteger.TEN.pow(10));
    BigInteger v4 = BigInteger.valueOf(222222222L).multiply(BigInteger.TEN.pow(10));
    try (Table input = new Table.TestBuilder()
             .column(1, 2, 1, 2)
             .decimal128Column(scale, RoundingMode.UNNECESSARY, v1, v2, v3, v4)
             .build();
         Table results = input.groupBy(0).aggregate(
             GroupByAggregation.sumOverflow().onColumn(1));
         Table sorted = results.orderBy(OrderByArg.asc(0))) {
      ColumnVector structCol = sorted.getColumn(1);
      assertEquals(DType.STRUCT, structCol.getType());

      try (ColumnView sumChild = structCol.getChildColumnView(0);
           ColumnView ovfChild = structCol.getChildColumnView(1);
           ColumnVector sumCol = sumChild.copyToColumnVector();
           ColumnVector ovfCol = ovfChild.copyToColumnVector();
           ColumnVector expectedSum = ColumnVector.decimalFromBigInt(scale,
               v1.add(v3), v2.add(v4));
           ColumnVector expectedOvf = ColumnVector.fromBooleans(false, false)) {
        assertEquals(DType.DTypeEnum.DECIMAL128, sumCol.getType().getTypeId());
        assertEquals(scale, sumCol.getType().getScale());
        assertColumnsAreEqual(expectedSum, sumCol);
        assertColumnsAreEqual(expectedOvf, ovfCol);
      }
    }
  }

  @Test
  void testGroupByHashSumOverflowDecimal128DetectsOverflow() {
    // Pick a value with at most 38 significant digits (the MathContext cap in
    // Table.TestBuilder.decimal128Column) that still overflows int128_max when
    // summed with itself. 10^38 - 1 is 38 nines; 2 * (10^38 - 1) ~= 2e38 which
    // exceeds int128_max ~= 1.7e38, so cudf's kernel flags overflow. The sum
    // value for the overflowed group is unspecified so only the flag is checked.
    final int scale = 0;
    BigInteger nearMax = BigInteger.TEN.pow(38).subtract(BigInteger.ONE);
    try (Table input = new Table.TestBuilder()
             .column(1, 1, 2, 2)
             .decimal128Column(scale, RoundingMode.UNNECESSARY,
                 nearMax, nearMax, BigInteger.valueOf(3), BigInteger.valueOf(4))
             .build();
         Table results = input.groupBy(0).aggregate(
             GroupByAggregation.sumOverflow().onColumn(1));
         Table sorted = results.orderBy(OrderByArg.asc(0))) {
      ColumnVector structCol = sorted.getColumn(1);
      assertEquals(DType.STRUCT, structCol.getType());

      try (ColumnView sumChild = structCol.getChildColumnView(0);
           ColumnView ovfChild = structCol.getChildColumnView(1);
           ColumnVector ovfCol = ovfChild.copyToColumnVector();
           ColumnVector expectedOvf = ColumnVector.fromBooleans(true, false)) {
        assertEquals(DType.DTypeEnum.DECIMAL128, sumChild.getType().getTypeId());
        assertEquals(scale, sumChild.getType().getScale());
        assertColumnsAreEqual(expectedOvf, ovfCol);
      }
    }
  }

  @Test
  void testGroupByMergeM2() {
    StructType nestedType = new StructType(false,
        new BasicType(true, DType.INT64),
        new BasicType(true, DType.FLOAT64),
        new BasicType(true, DType.FLOAT64));

    try (Table partialResults1 = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(nestedType,
                 struct(1L, 0.0, 0.0),
                 struct(1L, 1.0, 0.0),
                 struct(0L, null, null),
                 struct(0L, null, null))
             .build();
         Table partialResults2 = new Table.TestBuilder()
             .column(1, 2, 3)
             .column(nestedType,
                 struct(1L, 3.0, 0.0),
                 struct(1L, 4.0, 0.0),
                 struct(1L, 2.0, 0.0))
             .build();
         Table partialResults3 = new Table.TestBuilder()
             .column(1, 2)
             .column(nestedType,
                 struct(1L, 6.0, 0.0),
                 struct(1L, Double.NaN, Double.NaN))
             .build();
         Table partialResults4 = new Table.TestBuilder()
             .column(2, 3, 4)
             .column(nestedType,
                 struct(1L, 9.0, 0.0),
                 struct(1L, 8.0, 0.0),
                 struct(2L, Double.NaN, Double.NaN))
             .build();
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(nestedType,
                 struct(3L, 3.0, 18.0),
                 struct(4L, Double.NaN, Double.NaN),
                 struct(2L, 5.0, 18.0),
                 struct(2L, Double.NaN, Double.NaN))
             .build()) {
      try (Table concatenatedResults = Table.concatenate(
             partialResults1,
             partialResults2,
             partialResults3,
             partialResults4);
           Table finalResults = concatenatedResults.groupBy(0).aggregate(
               GroupByAggregation.mergeM2().onColumn(1))
           ) {
        assertTablesAreEqual(expected, finalResults);
      }
    }
  }

  @Test
  void testGroupByFirstExcludeNulls() {
    try (Table input = new Table.TestBuilder()
            .column(  1,   1,    1,  1,  2,    2,  2,    2)
            .column(null, 13, null, 12, 14, null, 15, null)
            .build();
         Table expected = new Table.TestBuilder()
                 .column(1, 2)
                 .column(13, 14)
                 .build();
         Table found = input.groupBy(0).aggregate(
             GroupByAggregation.nth(0, NullPolicy.EXCLUDE).onColumn(1))) {
      assertTablesAreEqual(expected, found);
    }
  }

  @Test
  void testGroupByLastExcludeNulls() {
    try (Table input = new Table.TestBuilder()
            .column(  1,   1,    1,  1,  2,    2,  2,    2)
            .column(null, 13, null, 12, 14, null, 15, null)
            .build();
         Table expected = new Table.TestBuilder()
                 .column(1, 2)
                 .column(12, 15)
                 .build();
         Table found = input.groupBy(0).aggregate(
             GroupByAggregation.nth(-1, NullPolicy.EXCLUDE).onColumn(1))) {
      assertTablesAreEqual(expected, found);
    }
  }

  @Test
  void testGroupByFirstIncludeNulls() {
    try (Table input = new Table.TestBuilder()
            .column(  1,   1,    1,  1,  2,    2,  2,    2)
            .column(null, 13, null, 12, 14, null, 15, null)
            .build();
         Table expected = new Table.TestBuilder()
                 .column(1, 2)
                 .column(null, 14)
                 .build();
         Table found = input.groupBy(0).aggregate(
             GroupByAggregation.nth(0, NullPolicy.INCLUDE).onColumn(1))) {
      assertTablesAreEqual(expected, found);
    }
  }

  @Test
  void testGroupByLastIncludeNulls() {
    try (Table input = new Table.TestBuilder()
            .column(  1,   1,    1,  1,  2,    2,  2,    2)
            .column(null, 13, null, 12, 14, null, 15, null)
            .build();
         Table expected = new Table.TestBuilder()
                 .column(1, 2)
                 .column(12, null)
                 .build();
         Table found = input.groupBy(0).aggregate(
             GroupByAggregation.nth(-1, NullPolicy.INCLUDE).onColumn(1))) {
      assertTablesAreEqual(expected, found);
    }
  }

  @Test
  void testGroupByAvg() {
    try (Table t1 = new Table.TestBuilder().column( 1,  1,  1,  1,  1,  1)
                                           .column( 1,  3,  3,  5,  5,  0)
                                           .column(12, 14, 13,  1, 17, 17)
                                           .build()) {
      try (Table t3 = t1.groupBy(0, 1).aggregate(GroupByAggregation.mean().onColumn(2));
           HostColumnVector aggOut1 = t3.getColumn(2).copyToHost()) {
        // verify t3
        assertEquals(4, t3.getRowCount());
        Map<Double, Integer> expectedAggregateResult = new HashMap() {
          {
            // value, count
            put(12.0, 1);
            put(13.5, 1);
            put(17.0, 1);
            put(9.0, 1);
          }
        };
        for (int i = 0; i < 4; ++i) {
          Double key = aggOut1.getDouble(i);
          assertTrue(expectedAggregateResult.containsKey(key));
          Integer count = expectedAggregateResult.get(key);
          if (count == 1) {
            expectedAggregateResult.remove(key);
          } else {
            expectedAggregateResult.put(key, count - 1);
          }
        }
      }
    }
  }

  @Test
  void testMultiAgg() {
    try (Table t1 = new Table.TestBuilder().column(  1,   1,   1,   1,   1,    1)
                                           .column(  2,   2,   2,   3,   3,    3)
                                           .column(5.0, 2.3, 3.4, 2.3, 1.3, 12.2)
                                           .column(  3,   1,   7,  -1,   9,    0)
                                           .build()) {
      try (Table t2 = t1.groupBy(0, 1).aggregate(
          GroupByAggregation.count().onColumn(0),
          GroupByAggregation.max().onColumn(3),
          GroupByAggregation.min().onColumn(2),
          GroupByAggregation.mean().onColumn(2),
          GroupByAggregation.sum().onColumn(2));
           HostColumnVector countOut = t2.getColumn(2).copyToHost();
           HostColumnVector maxOut = t2.getColumn(3).copyToHost();
           HostColumnVector minOut = t2.getColumn(4).copyToHost();
           HostColumnVector avgOut = t2.getColumn(5).copyToHost();
           HostColumnVector sumOut = t2.getColumn(6).copyToHost()) {
        assertEquals(2, t2.getRowCount());

        // verify count
        assertEquals(3, countOut.getInt(0));
        assertEquals(3, countOut.getInt(1));

        // verify mean
        List<Double> sortedMean = new ArrayList<>();
        sortedMean.add(avgOut.getDouble(0));
        sortedMean.add(avgOut.getDouble(1));
        sortedMean = sortedMean.stream()
            .sorted(Comparator.naturalOrder())
            .collect(Collectors.toList());

        assertEqualsWithinPercentage(3.5666f, sortedMean.get(0), 0.0001);
        assertEqualsWithinPercentage(5.2666f, sortedMean.get(1), 0.0001);

        // verify sum
        List<Double> sortedSum = new ArrayList<>();
        sortedSum.add(sumOut.getDouble(0));
        sortedSum.add(sumOut.getDouble(1));
        sortedSum = sortedSum.stream()
            .sorted(Comparator.naturalOrder())
            .collect(Collectors.toList());

        assertEqualsWithinPercentage(10.7f, sortedSum.get(0), 0.0001);
        assertEqualsWithinPercentage(15.8f, sortedSum.get(1), 0.0001);

        // verify min
        List<Double> sortedMin = new ArrayList<>();
        sortedMin.add(minOut.getDouble(0));
        sortedMin.add(minOut.getDouble(1));
        sortedMin = sortedMin.stream()
            .sorted(Comparator.naturalOrder())
            .collect(Collectors.toList());

        assertEqualsWithinPercentage(1.3f, sortedMin.get(0), 0.0001);
        assertEqualsWithinPercentage(2.3f, sortedMin.get(1), 0.0001);

        // verify max
        List<Integer> sortedMax = new ArrayList<>();
        sortedMax.add(maxOut.getInt(0));
        sortedMax.add(maxOut.getInt(1));
        sortedMax = sortedMax.stream()
            .sorted(Comparator.naturalOrder())
            .collect(Collectors.toList());

        assertEquals(7, sortedMax.get(0));
        assertEquals(9, sortedMax.get(1));
      }
    }
  }

  @Test
  void testSumWithStrings() {
    try (Table t = new Table.TestBuilder()
        .column("1-URGENT", "3-MEDIUM", "1-URGENT", "3-MEDIUM")
        .column(5289L, 5203L, 5303L, 5206L)
        .build();
         Table result = t.groupBy(0).aggregate(
             GroupByAggregation.sum().onColumn(1));
         Table sorted = result.orderBy(OrderByArg.asc(0));
         Table expected = new Table.TestBuilder()
             .column("1-URGENT", "3-MEDIUM")
             .column(5289L + 5303L, 5203L + 5206L)
             .build()) {
      assertTablesAreEqual(expected, sorted);
    }
  }

  @Test
  void testGroupByNoAggs() {
    try (Table t1 = new Table.TestBuilder().column(   1,    1,    1,    1,    1,    1)
        .column(   1,    3,    3,    5,    5,    0)
        .column(  12,   14,   13,   17,   17,   17)
        .decimal32Column(-3,   12,   14,   13,   111,   222,   333)
        .decimal64Column(-3,   12L,   14L,   13L,   111L,   222L,   333L)
        .build()) {
      try (Table t3 = t1.groupBy(0, 1).aggregate()) {
        // verify t3
        assertEquals(4, t3.getRowCount());
      }
    }
  }

  /**
   * A wrapper for ContiguousTable[] to implement AutoCloseable
   */
  private static class ContiguousSplitRes implements AutoCloseable {
    // to be closed
    private ContiguousTable[] splits;

    public ContiguousSplitRes(ContiguousTable[] splits) {
      this.splits = splits;
    }

    public ContiguousTable[] getSplits() {
      return splits;
    }

    @Override
    public void close() throws Exception {
      if (splits != null) {
        for (ContiguousTable t : splits) { t.close(); }
      }
    }
  }

  @Test
  void testGroupByContiguousSplitGroups() throws Exception {
    try (Table table = new Table.TestBuilder()
        .column(   1,    1,    1,    1,    1,    1)
        .column(   1,    3,    3,    5,    5,    5)
        .column(  12,   14,   13,   17,   16,   18)
        .column("s1", "s2", "s3", "s4", "s5", "s6")
        .build()) {
      // Normal case with primitive types.
      try (Table expected1 = new Table.TestBuilder()
              .column(   1)
              .column(   1)
              .column(  12)
              .column("s1").build();
           Table expected2 = new Table.TestBuilder()
              .column(   1,    1)
              .column(   3,    3)
              .column(  14,   13)
              .column("s2", "s3").build();
           Table expected3 = new Table.TestBuilder()
              .column(   1,    1,    1)
              .column(   5,    5,    5)
              .column(  17,   16,   18)
              .column("s4", "s5", "s6").build();
           Table expected4 = new Table.TestBuilder()
              .column(   1,    1,    1)
              .column(   1,    3,    5).build();
           ContiguousSplitRes splitsRes = new ContiguousSplitRes(
               table.groupBy(0, 1).contiguousSplitGroups());
           ContigSplitGroupByResult r =
               table.groupBy(0, 1).contiguousSplitGroupsAndGenUniqKeys()) {
        ContiguousTable[] splits = splitsRes.getSplits();
        ContiguousTable[] splits2 = r.getGroups();
        Table uniqKeys = r.getUniqKeyTable();

        for (ContiguousTable[] currSplits : Arrays.asList(splits, splits2)) {
          assertEquals(3, currSplits.length);
          for (ContiguousTable ct : currSplits) {
            if (ct.getRowCount() == 1) {
              assertTablesAreEqual(expected1, ct.getTable());
            } else if (ct.getRowCount() == 2) {
              assertTablesAreEqual(expected2, ct.getTable());
            } else if (ct.getRowCount() == 3) {
              assertTablesAreEqual(expected3, ct.getTable());
            } else {
              throw new RuntimeException("unexpected behavior: contiguousSplitGroups");
            }
          }
        }

        // verify uniq keys table
        assertTablesAreEqual(expected4, uniqKeys);
      }

      // Empty key columns, the whole table is a group.
      try(ContiguousSplitRes splitsRes = new ContiguousSplitRes(
          table.groupBy().contiguousSplitGroups());
          ContigSplitGroupByResult r = table.groupBy().contiguousSplitGroupsAndGenUniqKeys()) {
        ContiguousTable[] splits = splitsRes.getSplits();
        ContiguousTable[] splits2 = r.getGroups();
        Table uniqKeys = r.getUniqKeyTable();

        assertEquals(1, splits.length);
        assertTablesAreEqual(table, splits[0].getTable());

        assertEquals(1, splits2.length);
        assertTablesAreEqual(table, splits2[0].getTable());

        // Table should contain 1 or more columns,
        // If group by empty, keys table should be null;
        assertNull(uniqKeys);
      }

      // Row count is 0
      try(
          Table emptyTable = new Table.TestBuilder()
              .column(new Integer[0])
              .column(new Integer[0])
              .column(new Integer[0])
              .column(new String[0]).build();
          ContiguousSplitRes splitsRes = new ContiguousSplitRes(
              emptyTable.groupBy(0, 1).contiguousSplitGroups());
          ContigSplitGroupByResult r =
              emptyTable.groupBy(0, 1).contiguousSplitGroupsAndGenUniqKeys()) {
        ContiguousTable[] splits = splitsRes.getSplits();
        ContiguousTable[] splits2 = r.getGroups();
        Table uniqKeys = r.getUniqKeyTable();

        // the first of tmpSplits is empty split
        assertEquals(0, emptyTable.getRowCount());

        assertEquals(1, splits.length);
        assertEquals(0, splits[0].getTable().getRowCount());

        assertEquals(1, splits2.length);
        assertEquals(0, splits2[0].getTable().getRowCount());
        assertEquals(0, uniqKeys.getRowCount());
      }

      // table has 4 columns, but input keys has 5 columns, should throw exception
      assertThrows(IllegalArgumentException.class,
          () -> table.groupBy(0, 1, 1, 2, 3).contiguousSplitGroupsAndGenUniqKeys());
    }
  }

  @Test
  void testGroupByContiguousSplitGroupsSpecifyProjectionIndices() throws Exception {
    try (Table table = new Table.TestBuilder()
        .column(1, 1, 1, 1, 1, 1)
        .column(1, 3, 3, 5, 5, 5)
        .column(12, 14, 13, 17, 16, 18)
        .column("s1", "s2", "s3", "s4", "s5", "s6")
        .build()) {
      // Normal case with primitive types.
      try (Table expected1 = new Table.TestBuilder()
          .column(12)
          .column("s1").build();
           Table expected2 = new Table.TestBuilder()
               .column(14, 13)
               .column("s2", "s3").build();
           Table expected3 = new Table.TestBuilder()
               .column(17, 16, 18)
               .column("s4", "s5", "s6").build();
           Table expectedUniqKeys = new Table.TestBuilder()
               .column(1, 1, 1)
               .column(1, 3, 5).build();
           ContigSplitGroupByResult r =
               table.groupBy(0, 1).contiguousSplitGroupsAndGenUniqKeys(new int[]{2, 3})) {
        ContiguousTable[] splits = r.getGroups();
        Table uniqKeys = r.getUniqKeyTable();

        for (ContiguousTable ct : splits) {
          if (ct.getRowCount() == 1) {
            assertTablesAreEqual(expected1, ct.getTable());
          } else if (ct.getRowCount() == 2) {
            assertTablesAreEqual(expected2, ct.getTable());
          } else if (ct.getRowCount() == 3) {
            assertTablesAreEqual(expected3, ct.getTable());
          } else {
            throw new RuntimeException("unexpected behavior: contiguousSplitGroups");
          }
        }

        // verify uniq keys table
        assertTablesAreEqual(expectedUniqKeys, uniqKeys);
      }

      // Row count is 0
      try (
          Table emptyTable = new Table.TestBuilder()
              .column(new Integer[0])
              .column(new Integer[0])
              .column(new Integer[0])
              .column(new String[0]).build();
          ContigSplitGroupByResult r =
              emptyTable.groupBy(0, 1).contiguousSplitGroupsAndGenUniqKeys(new int[]{2, 3})) {
        ContiguousTable[] splits = r.getGroups();
        Table uniqKeys = r.getUniqKeyTable();

        assertEquals(0, emptyTable.getRowCount());
        assertEquals(1, splits.length);
        assertEquals(0, splits[0].getTable().getRowCount());
        assertEquals(0, uniqKeys.getRowCount());
      }
    }
  }

  @Test
  void testGroupByCollectListIncludeNulls() {
    try (Table input = new Table.TestBuilder()
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 4)
        .column(null, 13, null, 12, 14, null, 15, null, null, 0)
        .build();
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(new ListType(false, new BasicType(true, DType.INT32)),
                 Arrays.asList(null, 13, null, 12),
                 Arrays.asList(14, null, 15, null),
                 Arrays.asList((Integer) null),
                 Arrays.asList(0))
             .build();
         Table found = input.groupBy(0).aggregate(
             GroupByAggregation.collectList(NullPolicy.INCLUDE).onColumn(1))) {
      assertTablesAreEqual(expected, found);
    }
  }

  @Test
  void testGroupByMergeLists() {
    ListType listOfInts = new ListType(false, new BasicType(false, DType.INT32));
    ListType listOfStructs = new ListType(false, new StructType(false,
        new BasicType(false, DType.INT32), new BasicType(false, DType.STRING)));
    try (Table input = new Table.TestBuilder()
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 4)
        .column(listOfInts,
            Arrays.asList(1, 2), Arrays.asList(3), Arrays.asList(7, 8), Arrays.asList(4, 5, 6),
            Arrays.asList(8, 9), Arrays.asList(8, 9, 10), Arrays.asList(10, 11), Arrays.asList(11, 12),
            Arrays.asList(13, 13), Arrays.asList(14, 15, 15))
        .column(listOfStructs,
            Arrays.asList(new StructData(1, "s1"), new StructData(2, "s2")),
            Arrays.asList(new StructData(2, "s3"), new StructData(3, "s4")),
            Arrays.asList(new StructData(2, "s2")),
            Arrays.asList(),
            Arrays.asList(new StructData(11, "s11")),
            Arrays.asList(new StructData(22, "s22"), new StructData(33, "s33")),
            Arrays.asList(),
            Arrays.asList(new StructData(22, "s22"), new StructData(33, "s33"), new StructData(44, "s44")),
            Arrays.asList(new StructData(333, "s333"), new StructData(222, "s222"), new StructData(111, "s111")),
            Arrays.asList(new StructData(222, "s222"), new StructData(444, "s444")))
        .build();
         Table expectedListOfInts = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(listOfInts,
                 Arrays.asList(1, 2, 3, 7 ,8, 4, 5, 6),
                 Arrays.asList(8, 9, 8, 9, 10, 10, 11, 11, 12),
                 Arrays.asList(13, 13),
                 Arrays.asList(14, 15, 15))
             .build();
         Table expectedListOfStructs = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(listOfStructs,
                 Arrays.asList(new StructData(1, "s1"), new StructData(2, "s2"),
                     new StructData(2, "s3"), new StructData(3, "s4"), new StructData(2, "s2")),
                 Arrays.asList(new StructData(11, "s11"), new StructData(22, "s22"), new StructData(33, "s33"),
                     new StructData(22, "s22"), new StructData(33, "s33"), new StructData(44, "s44")),
                 Arrays.asList(new StructData(333, "s333"), new StructData(222, "s222"), new StructData(111, "s111")),
                 Arrays.asList(new StructData(222, "s222"), new StructData(444, "s444")))
             .build();
         Table retListOfInts = input.groupBy(0).aggregate(GroupByAggregation.mergeLists().onColumn(1));
         Table retListOfStructs = input.groupBy(0).aggregate(GroupByAggregation.mergeLists().onColumn(2))) {
      assertTablesAreEqual(expectedListOfInts, retListOfInts);
      assertTablesAreEqual(expectedListOfStructs, retListOfStructs);
    }
  }

  @Test
  void testGroupByCollectSetIncludeNulls() {
    // test with null unequal and nan unequal
    GroupByAggregation collectSet = GroupByAggregation.collectSet(NullPolicy.INCLUDE,
        NullEquality.UNEQUAL, NaNEquality.UNEQUAL);
    try (Table input = new Table.TestBuilder()
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
        .column(null, 13, null, 13, 14, null, 15, null, 4, 1, 1, 4, 0, 0, 0, 0)
        .build();
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(new ListType(false, new BasicType(true, DType.INT32)),
                 Arrays.asList(13, null, null), Arrays.asList(14, 15, null, null),
                 Arrays.asList(1, 4), Arrays.asList(0))
             .build();
         Table found = input.groupBy(0).aggregate(collectSet.onColumn(1));
         ColumnVector listsSorted = found.getColumn(1).listSortRows(false, false)) {
      assertColumnsAreEqual(expected.getColumn(0), found.getColumn(0));
      assertColumnsAreEqual(expected.getColumn(1), listsSorted);
    }
    // test with null equal and nan unequal
    collectSet = GroupByAggregation.collectSet(NullPolicy.INCLUDE,
        NullEquality.EQUAL, NaNEquality.UNEQUAL);
    try (Table input = new Table.TestBuilder()
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
        .column(null, 13.0, null, 13.0,
            14.1, Double.NaN, 13.9, Double.NaN,
            Double.NaN, null, 1.0, null,
            null, null, null, null)
        .build();
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(new ListType(false, new BasicType(true, DType.FLOAT64)),
                 Arrays.asList(13.0, null),
                 Arrays.asList(13.9, 14.1, Double.NaN, Double.NaN),
                 Arrays.asList(1.0, Double.NaN, null),
                 Arrays.asList((Integer) null))
             .build();
         Table found = input.groupBy(0).aggregate(collectSet.onColumn(1));
         ColumnVector listsSorted = found.getColumn(1).listSortRows(false, false)) {
      assertColumnsAreEqual(expected.getColumn(0), found.getColumn(0));
      assertColumnsAreEqual(expected.getColumn(1), listsSorted);
    }
    // test with null equal and nan equal
    collectSet = GroupByAggregation.collectSet(NullPolicy.INCLUDE,
        NullEquality.EQUAL, NaNEquality.ALL_EQUAL);
    try (Table input = new Table.TestBuilder()
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4)
        .column(null, 13.0, null, 13.0,
            14.1, Double.NaN, 13.9, Double.NaN,
            0.0, 0.0, 0.00, 0.0,
            Double.NaN, Double.NaN, null, null)
        .build();
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(new ListType(false, new BasicType(true, DType.FLOAT64)),
                 Arrays.asList(13.0, null),
                 Arrays.asList(13.9, 14.1, Double.NaN),
                 Arrays.asList(0.0),
                 Arrays.asList(Double.NaN, (Integer) null))
             .build();
         Table found = input.groupBy(0).aggregate(collectSet.onColumn(1));
         ColumnVector listsSorted = found.getColumn(1).listSortRows(false, false)) {
      assertColumnsAreEqual(expected.getColumn(0), found.getColumn(0));
      assertColumnsAreEqual(expected.getColumn(1), listsSorted);
    }
  }

  @Test
  void testGroupByMergeSets() {
    ListType listOfInts = new ListType(false, new BasicType(false, DType.INT32));
    ListType listOfDoubles = new ListType(false, new BasicType(false, DType.FLOAT64));
    try (Table input = new Table.TestBuilder()
        .column(1, 1, 1, 1, 2, 2, 2, 2, 3, 4)
        .column(listOfInts,
            Arrays.asList(1, 2), Arrays.asList(3), Arrays.asList(7, 8), Arrays.asList(4, 5, 6),
            Arrays.asList(8, 9), Arrays.asList(8, 9, 10), Arrays.asList(10, 11), Arrays.asList(11, 12),
            Arrays.asList(13, 13), Arrays.asList(14, 15, 15))
        .column(listOfDoubles,
            Arrays.asList(Double.NaN, 1.2), Arrays.asList(), Arrays.asList(Double.NaN), Arrays.asList(-3e10),
            Arrays.asList(1.1, 2.2, 3.3), Arrays.asList(3.3, 2.2), Arrays.asList(), Arrays.asList(),
            Arrays.asList(1e3, Double.NaN, 1e-3, Double.NaN), Arrays.asList())
        .build();
         Table expectedListOfInts = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(listOfInts,
                 Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8),
                 Arrays.asList(8, 9, 10, 11, 12),
                 Arrays.asList(13),
                 Arrays.asList(14, 15))
             .build();
         Table expectedListOfDoubles = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(listOfDoubles,
                 Arrays.asList(-3e10, 1.2, Double.NaN, Double.NaN),
                 Arrays.asList(1.1, 2.2, 3.3),
                 Arrays.asList(1e-3, 1e3, Double.NaN, Double.NaN),
                 Arrays.asList())
             .build();
         Table expectedListOfDoublesNaNEq = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(listOfDoubles,
                 Arrays.asList(-3e10, 1.2, Double.NaN),
                 Arrays.asList(1.1, 2.2, 3.3),
                 Arrays.asList(1e-3, 1e3, Double.NaN),
                 Arrays.asList())
             .build();
         Table retListOfInts = input.groupBy(0).aggregate(GroupByAggregation.mergeSets().onColumn(1));
         Table retListOfDoubles = input.groupBy(0).aggregate(GroupByAggregation.mergeSets().onColumn(2));
         Table retListOfDoublesNaNEq = input.groupBy(0).aggregate(
             GroupByAggregation.mergeSets(NullEquality.UNEQUAL, NaNEquality.ALL_EQUAL).onColumn(2));
         ColumnVector listsIntsSorted = retListOfInts.getColumn(1).listSortRows(false, false);
         ColumnVector listsDoublesSorted = retListOfDoubles.getColumn(1).listSortRows(false, false);
         ColumnVector listsDoublesNaNEqSorted = retListOfDoublesNaNEq.getColumn(1).listSortRows(false, false)) {
      assertColumnsAreEqual(expectedListOfInts.getColumn(0), retListOfInts.getColumn(0));
      assertColumnsAreEqual(expectedListOfDoubles.getColumn(0), retListOfDoubles.getColumn(0));
      assertColumnsAreEqual(expectedListOfDoublesNaNEq.getColumn(0), retListOfDoublesNaNEq.getColumn(0));

      assertColumnsAreEqual(expectedListOfInts.getColumn(1), listsIntsSorted);
      assertColumnsAreEqual(expectedListOfDoubles.getColumn(1), listsDoublesSorted);
      assertColumnsAreEqual(expectedListOfDoublesNaNEq.getColumn(1), listsDoublesNaNEqSorted);

    }
  }

  @Test
  void testGroupByBitAnd() {
    try (Table t1 = new Table.TestBuilder()
        .column(0x1F, 0x0F, 0x33, 0x55, 0x3F, 0x2F, 0x0F, 0x42)
        .column(1, 1, 2, 2, 2, 3, 3, 4).build();
         Table output = t1.groupBy(1).aggregate(GroupByAggregation.bitAnd().onColumn(0));
         Table ordered = output.orderBy(OrderByArg.asc(0));
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(0x1F & 0x0F,
                 0x33 & 0x55 & 0x3F,
                 0x2F & 0x0F,
                 0x42)
             .build()) {
      assertTablesAreEqual(expected, ordered);
    }
  }

  @Test
  void testGroupByBitOr() {
    try (Table t1 = new Table.TestBuilder()
        .column(0x1F, 0x0F, 0x33, 0x55, 0x3F, 0x2F, 0x0F, 0x42)
        .column(1, 1, 2, 2, 2, 3, 3, 4).build();
         Table output = t1.groupBy(1).aggregate(GroupByAggregation.bitOr().onColumn(0));
         Table ordered = output.orderBy(OrderByArg.asc(0));
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(0x1F | 0x0F,
                 0x33 | 0x55 | 0x3F,
                 0x2F | 0x0F,
                 0x42)
             .build()) {
      assertTablesAreEqual(expected, ordered);
    }
  }

  @Test
  void testGroupByBitXor() {
    try (Table t1 = new Table.TestBuilder()
        .column(0x1F, 0x0F, 0x33, 0x55, 0x3F, 0x2F, 0x0F, 0x42)
        .column(1, 1, 2, 2, 2, 3, 3, 4).build();
         Table output = t1.groupBy(1).aggregate(GroupByAggregation.bitXor().onColumn(0));
         Table ordered = output.orderBy(OrderByArg.asc(0));
         Table expected = new Table.TestBuilder()
             .column(1, 2, 3, 4)
             .column(0x1F ^ 0x0F,
                 0x33 ^ 0x55 ^ 0x3F,
                 0x2F ^ 0x0F,
                 0x42)
             .build()) {
      assertTablesAreEqual(expected, ordered);
    }
  }

  @Test
  void testReductionBitAnd() {
    try (ColumnVector input = ColumnVector.fromInts(0x1F, 0x0F, 0x33, 0xFF, 0x2F, 0x3F);
         Scalar result = input.reduce(ReductionAggregation.bitAnd())) {
      int expected = 0x1F & 0x0F & 0x33 & 0xFF & 0x2F & 0x3F;
      assertEquals(expected, result.getInt());
    }
  }

  @Test
  void testReductionBitOr() {
    try (ColumnVector input = ColumnVector.fromInts(0x1F, 0x0F, 0x33, 0xFF, 0x2F, 0x3F);
         Scalar result = input.reduce(ReductionAggregation.bitOr())) {
      int expected = 0x1F | 0x0F | 0x33 | 0xFF | 0x2F | 0x3F;
      assertEquals(expected, result.getInt());
    }
  }

  @Test
  void testReductionBitXor() {
    try (ColumnVector input = ColumnVector.fromInts(0x1F, 0x0F, 0x33, 0xFF, 0x2F, 0x3F);
         Scalar result = input.reduce(ReductionAggregation.bitXor())) {
      int expected = 0x1F ^ 0x0F ^ 0x33 ^ 0xFF ^ 0x2F ^ 0x3F;
      assertEquals(expected, result.getInt());
    }
  }

  @Test
  void testRowBitCount() {
    try (Table t = new Table.TestBuilder()
        .column(0, 1, null, 3)                 // 33 bits per row (4 bytes + valid bit)
        .column(0.0, null, 2.0, 3.0)           // 65 bits per row (8 bytes + valid bit)
        .column("zero", null, "two", "three")  // 33 bits (4 byte offset + valid bit) + char bits
        .build();
         ColumnVector expected = ColumnVector.fromInts(163, 131, 155, 171);
         ColumnVector actual = t.rowBitCount()) {
      assertColumnsAreEqual(expected, actual);
    }
  }

  @Test
  void testRowBitCountEmpty() {
    try (Table t = new Table.TestBuilder()
            .column(new Integer[0])
            .column(new Double[0])
            .column(new String[0])
            .build();
         ColumnVector c = t.rowBitCount()) {
      assertEquals(DType.INT32, c.getType());
      assertEquals(0, c.getRowCount());
    }
  }

  @Test
  void testSimpleGather() {
    try (Table testTable = new Table.TestBuilder()
            .column(1, 2, 3, 4, 5)
            .column("A", "AA", "AAA", "AAAA", "AAAAA")
            .decimal32Column(-3, 1, 2, 3, 4, 5)
            .decimal64Column(-8, 100001L, 200002L, 300003L, 400004L, 500005L)
            .build();
         ColumnVector gatherMap = ColumnVector.fromInts(0, 2, 4, -2);
         Table expected = new Table.TestBuilder()
                 .column(1, 3, 5, 4)
                 .column("A", "AAA", "AAAAA", "AAAA")
                 .decimal32Column(-3, 1, 3, 5, 4)
                 .decimal64Column(-8, 100001L, 300003L, 500005L, 400004L)
                 .build();
         Table found = testTable.gather(gatherMap)) {
      assertTablesAreEqual(expected, found);
    }
  }

  @Test
  void testBoundsCheckedGather() {
    try (Table testTable = new Table.TestBuilder()
            .column(1, 2, 3, 4, 5)
            .column("A", "AA", "AAA", "AAAA", "AAAAA")
            .decimal32Column(-3, 1, 2, 3, 4, 5)
            .decimal64Column(-8, 100001L, 200002L, 300003L, 400004L, 500005L)
            .build();
         ColumnVector gatherMap = ColumnVector.fromInts(0, 100, 4, -2);
         Table expected = new Table.TestBuilder()
                 .column(1, null, 5, 4)
                 .column("A", null, "AAAAA", "AAAA")
                 .decimal32Column(-3, 1, null, 5, 4)
                 .decimal64Column(-8, 100001L, null, 500005L, 400004L)
                 .build();
         Table found = testTable.gather(gatherMap)) {
      assertTablesAreEqual(expected, found);
    }
  }


  @Test
  void testScatterTable() {
    try (Table srcTable = new Table.TestBuilder()
            .column(1, 2, 3, 4, 5)
            .column("A", "AA", "AAA", "AAAA", "AAAAA")
            .decimal32Column(-3, 1, 2, 3, 4, 5)
            .decimal64Column(-8, 100001L, 200002L, 300003L, 400004L, 500005L)
            .build();
         ColumnVector scatterMap = ColumnVector.fromInts(0, 2, 4, -2);
         Table targetTable = new Table.TestBuilder()
            .column(-1, -2, -3, -4, -5)
            .column("B", "BB", "BBB", "BBBB", "BBBBB")
            .decimal32Column(-3, -1, -2, -3, -4, -5)
            .decimal64Column(-8, -100001L, -200002L, -300003L, -400004L, -500005L)
            .build();
         Table expected = new Table.TestBuilder()
            .column(1, -2, 2, 4, 3)
            .column("A", "BB", "AA", "AAAA", "AAA")
            .decimal32Column(-3, 1, -2, 2, 4, 3)
            .decimal64Column(-8, 100001L, -200002L, 200002L, 400004L, 300003L)
            .build();
         Table result = srcTable.scatter(scatterMap, targetTable)) {
      assertTablesAreEqual(expected, result);
    }
  }

  @Test
  void testScatterScalars() {
    try (Scalar s1 = Scalar.fromInt(0);
         Scalar s2 = Scalar.fromString("A");
         ColumnVector scatterMap = ColumnVector.fromInts(0, 2, -1);
         Table targetTable = new Table.TestBuilder()
            .column(-1, -2, -3, -4, -5)
            .column("B", "BB", "BBB", "BBBB", "BBBBB")
            .build();
         Table expected = new Table.TestBuilder()
            .column(0, -2, 0, -4, 0)
            .column("A", "BB", "A", "BBBB", "A")
            .build();
         Table result = Table.scatter(new Scalar[] { s1, s2 }, scatterMap, targetTable)) {
       assertTablesAreEqual(expected, result);
     }
  }

  @Test
  void testMaskWithoutValidity() {
    try (ColumnVector mask = ColumnVector.fromBoxedBooleans(true, false, true, false, true);
         ColumnVector fromInts = ColumnVector.fromInts(1, 2, 3, 4, 5);
         ColumnVector fromStrings = ColumnVector.fromStrings("1", "2", "3", "4", "5");
         ColumnVector fromDecimals = ColumnVector.decimalFromLongs(-3, 123L, -234L, 345L, 1000L, -2000L);
         Table input = new Table(fromInts, fromStrings, fromDecimals);
         Table filteredTable = input.filter(mask);
         ColumnVector expectedInts = ColumnVector.fromInts(1, 3, 5);
         ColumnVector expectedStrings = ColumnVector.fromStrings("1", "3", "5");
         ColumnVector expectedDecimals = ColumnVector.decimalFromLongs(-3, 123L, 345L, -2000L);
         Table expected = new Table(expectedInts, expectedStrings, expectedDecimals)) {
      assertTablesAreEqual(expected, filteredTable);
    }
  }

  @Test
  void testMaskWithValidity() {
    final int numRows = 5;
    try (Builder builder = HostColumnVector.builder(DType.BOOL8, numRows)) {
      for (int i = 0; i < numRows; ++i) {
        builder.append((byte) 1);
        if (i % 2 != 0) {
          builder.setNullAt(i);
        }
      }
      try (ColumnVector mask = builder.buildAndPutOnDevice();
           ColumnVector fromInts = ColumnVector.fromBoxedInts(1, null, 2, 3, null);
           Table input = new Table(fromInts);
           Table filteredTable = input.filter(mask);
           HostColumnVector filtered = filteredTable.getColumn(0).copyToHost()) {
        assertEquals(DType.INT32, filtered.getType());
        assertEquals(3, filtered.getRowCount());
        assertEquals(1, filtered.getInt(0));
        assertEquals(2, filtered.getInt(1));
        assertTrue(filtered.isNull(2));
      }
    }
  }

  @Test
  void testMaskDataOnly() {
    byte[] maskVals = new byte[]{0, 1, 0, 1, 1};
    try (ColumnVector mask = ColumnVector.boolFromBytes(maskVals);
         ColumnVector fromBytes = ColumnVector.fromBoxedBytes((byte) 1, null, (byte) 2, (byte) 3, null);
         Table input = new Table(fromBytes);
         Table filteredTable = input.filter(mask);
         HostColumnVector filtered = filteredTable.getColumn(0).copyToHost()) {
      assertEquals(DType.INT8, filtered.getType());
      assertEquals(3, filtered.getRowCount());
      assertTrue(filtered.isNull(0));
      assertEquals(3, filtered.getByte(1));
      assertTrue(filtered.isNull(2));
    }
  }


  @Test
  void testAllFilteredFromData() {
    Boolean[] maskVals = new Boolean[5];
    Arrays.fill(maskVals, false);
    try (ColumnVector mask = ColumnVector.fromBoxedBooleans(maskVals);
         ColumnVector fromInts = ColumnVector.fromBoxedInts(1, null, 2, 3, null);
         ColumnVector fromDecimal32s = ColumnVector.decimalFromInts(-3, 1, 2, 3, 4, 5);
         ColumnVector fromDecimal64s = ColumnVector.decimalFromLongs(-11, 1L, 2L, 3L, 4L, 5L);
         Table input = new Table(fromInts, fromDecimal32s, fromDecimal64s);
         Table filteredTable = input.filter(mask)) {
      ColumnVector filtered = filteredTable.getColumn(0);
      assertEquals(DType.INT32, filtered.getType());
      assertEquals(0, filtered.getRowCount());
      filtered = filteredTable.getColumn(1);
      assertEquals(DType.create(DType.DTypeEnum.DECIMAL32, -3), filtered.getType());
      assertEquals(0, filtered.getRowCount());
      filtered = filteredTable.getColumn(2);
      assertEquals(DType.create(DType.DTypeEnum.DECIMAL64, -11), filtered.getType());
      assertEquals(0, filtered.getRowCount());
    }
  }

  @Test
  void testAllFilteredFromValidity() {
    final int numRows = 5;
    try (Builder builder = HostColumnVector.builder(DType.BOOL8, numRows)) {
      for (int i = 0; i < numRows; ++i) {
        builder.append((byte) 1);
        builder.setNullAt(i);
      }
      try (ColumnVector mask = builder.buildAndPutOnDevice();
           ColumnVector fromInts = ColumnVector.fromBoxedInts(1, null, 2, 3, null);
           ColumnVector fromDecimal32s = ColumnVector.decimalFromInts(-3, 1, 2, 3, 4, 5);
           ColumnVector fromDecimal64s = ColumnVector.decimalFromLongs(-11, 1L, 2L, 3L, 4L, 5L);
           Table input = new Table(fromInts, fromDecimal32s, fromDecimal64s);
           Table filteredTable = input.filter(mask)) {
        ColumnVector filtered = filteredTable.getColumn(0);
        assertEquals(DType.INT32, filtered.getType());
        assertEquals(0, filtered.getRowCount());
        filtered = filteredTable.getColumn(1);
        assertEquals(DType.create(DType.DTypeEnum.DECIMAL32, -3), filtered.getType());
        assertEquals(0, filtered.getRowCount());
        filtered = filteredTable.getColumn(2);
        assertEquals(DType.create(DType.DTypeEnum.DECIMAL64, -11), filtered.getType());
        assertEquals(0, filtered.getRowCount());
      }
    }
  }

  ColumnView replaceValidity(ColumnView cv, DeviceMemoryBuffer validity, long nullCount) {
    assert (validity.length >= BitVectorHelper.getValidityAllocationSizeInBytes(cv.rows));
    if (cv.type.isNestedType()) {
      ColumnView[] children = cv.getChildColumnViews();
      try {
        return new ColumnView(cv.type,
            cv.rows,
            Optional.of(nullCount),
            validity,
            cv.getOffsets(),
            children);
      } finally {
        for (ColumnView v : children) {
          if (v != null) {
            v.close();
          }
        }
      }
    } else {
      return new ColumnView(cv.type, cv.rows, Optional.of(nullCount), cv.getData(), validity, cv.getOffsets());
    }
  }

  @Test
  void testRemoveNullMasksIfNeeded() {
    ListType nestedType = new ListType(true, new StructType(false,
        new BasicType(true, DType.INT32),
        new BasicType(true, DType.INT64)));

    List data1 = Arrays.asList(10, 20L);
    List data2 = Arrays.asList(50, 60L);
    HostColumnVector.StructData structData1 = new HostColumnVector.StructData(data1);
    HostColumnVector.StructData structData2 = new HostColumnVector.StructData(data2);

    //First we create ColumnVectors
    try (ColumnVector nonNullVector0 = ColumnVector.fromBoxedInts(1, 2, 3);
         ColumnVector nonNullVector2 = ColumnVector.fromStrings("1", "2", "3");
         ColumnVector nonNullVector1 = ColumnVector.fromLists(nestedType,
             Arrays.asList(structData1, structData2),
             Arrays.asList(structData1, structData2),
             Arrays.asList(structData1, structData2))) {
      //Then we take the created ColumnVectors and add validity masks even though the nullCount = 0
      long allocSize = BitVectorHelper.getValidityAllocationSizeInBytes(nonNullVector0.rows);
      try (DeviceMemoryBuffer dm0 = DeviceMemoryBuffer.allocate(allocSize);
           DeviceMemoryBuffer dm1 = DeviceMemoryBuffer.allocate(allocSize);
           DeviceMemoryBuffer dm2 = DeviceMemoryBuffer.allocate(allocSize);
           DeviceMemoryBuffer dm3_child =
               DeviceMemoryBuffer.allocate(BitVectorHelper.getValidityAllocationSizeInBytes(2))) {
        Cuda.memset(dm0.address, (byte) 0xFF, allocSize);
        Cuda.memset(dm1.address, (byte) 0xFF, allocSize);
        Cuda.memset(dm2.address, (byte) 0xFF, allocSize);
        Cuda.memset(dm3_child.address, (byte) 0xFF,
            BitVectorHelper.getValidityAllocationSizeInBytes(2));

        try (ColumnView cv0View = replaceValidity(nonNullVector0, dm0, 0);
             ColumnVector cv0 = cv0View.copyToColumnVector();
             ColumnView struct = nonNullVector1.getChildColumnView(0);
             ColumnView structChild0 = struct.getChildColumnView(0);
             ColumnView newStructChild0 = replaceValidity(structChild0, dm3_child, 0);
             ColumnView newStruct = struct.replaceChildrenWithViews(new int[]{0}, new ColumnView[]{newStructChild0});
             ColumnView list = nonNullVector1.replaceChildrenWithViews(new int[]{0}, new ColumnView[]{newStruct});
             ColumnView cv1View = replaceValidity(list, dm1, 0);
             ColumnVector cv1 = cv1View.copyToColumnVector();
             ColumnView cv2View = replaceValidity(nonNullVector2, dm2, 0);
             ColumnVector cv2 = cv2View.copyToColumnVector()) {

          try (Table t = new Table(new ColumnVector[]{cv0, cv1, cv2});
               Table tableWithoutNullMask = removeNullMasksIfNeeded(t);
               ColumnView tableStructChild0 = t.getColumn(1).getChildColumnView(0).getChildColumnView(0);
               ColumnVector tableStructChild0Cv = tableStructChild0.copyToColumnVector();
               Table expected = new Table(new ColumnVector[]{nonNullVector0, nonNullVector1,
                nonNullVector2})) {
            assertTrue(t.getColumn(0).hasValidityVector());
            assertTrue(t.getColumn(1).hasValidityVector());
            assertTrue(t.getColumn(2).hasValidityVector());
            assertTrue(tableStructChild0Cv.hasValidityVector());

            assertPartialTablesAreEqual(expected,
                0,
                expected.getRowCount(),
                tableWithoutNullMask,
                true,
                true);
          }
        }
      }
    }
  }

  @Test
  void testRemoveNullMasksIfNeededWithNulls() {
    ListType nestedType = new ListType(true, new StructType(true,
        new BasicType(true, DType.INT32),
        new BasicType(true, DType.INT64)));

    List data1 = Arrays.asList(0, 10L);
    List data2 = Arrays.asList(50, null);
    HostColumnVector.StructData structData1 = new HostColumnVector.StructData(data1);
    HostColumnVector.StructData structData2 = new HostColumnVector.StructData(data2);

    //First we create ColumnVectors
    try (ColumnVector nonNullVector0 = ColumnVector.fromBoxedInts(1, null, 2, 3);
         ColumnVector nonNullVector1 = ColumnVector.fromStrings("1", "2", null, "3");
         ColumnVector nonNullVector2 = ColumnVector.fromLists(nestedType,
             Arrays.asList(structData1, structData2),
             null,
             Arrays.asList(structData1, structData2),
             Arrays.asList(structData1, structData2))) {
      try (Table expected = new Table(new ColumnVector[]{nonNullVector0, nonNullVector1, nonNullVector2});
           Table unchangedTable = removeNullMasksIfNeeded(expected)) {
        assertTablesAreEqual(expected, unchangedTable);
      }
    }
  }

  @Test
  void testMismatchedSizesForFilter() {
    Boolean[] maskVals = new Boolean[3];
    Arrays.fill(maskVals, true);
    try (ColumnVector mask = ColumnVector.fromBoxedBooleans(maskVals);
         ColumnVector fromInts = ColumnVector.fromBoxedInts(1, null, 2, 3, null);
         Table input = new Table(fromInts)) {
      assertThrows(AssertionError.class, () -> input.filter(mask).close());
    }
  }

  @Test
  void testTableBasedFilter() {
    byte[] maskVals = new byte[]{0, 1, 0, 1, 1};
    try (ColumnVector mask = ColumnVector.boolFromBytes(maskVals);
         ColumnVector fromInts = ColumnVector.fromBoxedInts(1, null, 2, 3, null);
         ColumnVector fromStrings = ColumnVector.fromStrings("one", "two", "three", null, "five");
         ColumnVector fromDecimals = ColumnVector.fromDecimals(BigDecimal.ZERO, null, BigDecimal.ONE, null, BigDecimal.TEN);
         Table input = new Table(fromInts, fromStrings, fromDecimals);
         Table filtered = input.filter(mask);
         ColumnVector expectedFromInts = ColumnVector.fromBoxedInts(null, 3, null);
         ColumnVector expectedFromStrings = ColumnVector.fromStrings("two", null, "five");
         ColumnVector expectedFromDecimals = ColumnVector.fromDecimals(null, null, BigDecimal.TEN);
         Table expected = new Table(expectedFromInts, expectedFromStrings, expectedFromDecimals)) {
      assertTablesAreEqual(expected, filtered);
    }
  }

  @Test
  void testDropDuplicates() {
    int[] keyColumns = new int[]{ 1 };

    try (ColumnVector col1 = ColumnVector.fromBoxedInts(5, null, 3, 5, 8, 1);
         ColumnVector col2 = ColumnVector.fromBoxedInts(20, null, null, 19, 21, 19);
         Table input = new Table(col1, col2)) {

      // Keep the first duplicate element.
      try (Table result = input.dropDuplicates(keyColumns, Table.DuplicateKeepOption.KEEP_FIRST, true);
           Table resultSorted = result.orderBy(OrderByArg.asc(1, true));
           ColumnVector expectedCol1 = ColumnVector.fromBoxedInts(null, 5, 5, 8);
           ColumnVector expectedCol2 = ColumnVector.fromBoxedInts(null, 19, 20, 21);
           Table expected = new Table(expectedCol1, expectedCol2)) {
        assertTablesAreEqual(expected, resultSorted);
      }

      // Keep the last duplicate element.
      try (Table result = input.dropDuplicates(keyColumns, Table.DuplicateKeepOption.KEEP_LAST, true);
           Table resultSorted = result.orderBy(OrderByArg.asc(1, true));
           ColumnVector expectedCol1 = ColumnVector.fromBoxedInts(3, 1, 5, 8);
           ColumnVector expectedCol2 = ColumnVector.fromBoxedInts(null, 19, 20, 21);
           Table expected = new Table(expectedCol1, expectedCol2)) {
        assertTablesAreEqual(expected, resultSorted);
      }
    }
  }

  private enum Columns {
    BOOL("BOOL"),
    INT("INT"),
    BYTE("BYTE"),
    LONG("LONG"),
    STRING("STRING"),
    FLOAT("FLOAT"),
    DOUBLE("DOUBLE"),
    DECIMAL64("DECIMAL64"),
    DECIMAL128("DECIMAL128"),
    STRUCT("STRUCT"),
    STRUCT_DEC128("STRUCT_DEC128"),
    LIST("LIST"),
    LIST_STRUCT("LIST_STRUCT"),
    LIST_DEC128("LIST_DEC128");

    final String name;

    Columns(String columnName) {
      this.name = columnName;
    }
  }

  private static class WriteUtils {

    private static final Map<Columns, Function<TestBuilder, TestBuilder>> addColumnFn = Maps.newHashMap();

    static {
      addColumnFn.put(Columns.BOOL, (t) -> t.column(true, false, false, true, false));
      addColumnFn.put(Columns.INT, (t) -> t.column(5, 1, 0, 2, 7));
      addColumnFn.put(Columns.LONG, (t) -> t.column(3l, 9l, 4l, 2l, 20l));
      addColumnFn.put(Columns.BYTE, (t) -> t.column(new Byte[]{2, 3, 4, 5, 9}));
      addColumnFn.put(Columns.STRING, (t) -> t.column("this", "is", "a", "test", "string"));
      addColumnFn.put(Columns.FLOAT, (t) -> t.column(1.0f, 3.5f, 5.9f, 7.1f, 9.8f));
      addColumnFn.put(Columns.DOUBLE, (t) -> t.column(5.0d, 9.5d, 0.9d, 7.23d, 2.8d));
      addColumnFn.put(Columns.DECIMAL64, (t) ->
          t.decimal64Column(-5, 1L, 323L, 12398423L, -231312412L, 239893414231L));
      addColumnFn.put(Columns.DECIMAL128, (t) ->
          t.decimal128Column(-10, RoundingMode.UNNECESSARY, BigInteger.ONE, BigInteger.ZERO,
              BigInteger.TEN, new BigInteger("100000000000000000000000000000"),
              new BigInteger("-1234567890123456789012345678")));

      BasicType dec64Type = new BasicType(true, DType.create(DType.DTypeEnum.DECIMAL64, 0));
      StructType structType = new StructType(true,
          new BasicType(true, DType.INT32), new BasicType(true, DType.STRING), dec64Type);
      addColumnFn.put(Columns.STRUCT, (t) -> t.column(structType,
          struct(1, "k1", BigDecimal.ONE),
          struct(2, "k2", BigDecimal.ZERO),
          struct(3, "k3", BigDecimal.TEN),
          struct(4, "k4", BigDecimal.valueOf(Long.MAX_VALUE)),
          new HostColumnVector.StructData((List) null)));
      BasicType dec128Type = new BasicType(true, DType.create(DType.DTypeEnum.DECIMAL128, -5));
      addColumnFn.put(Columns.STRUCT_DEC128, (t) ->
          t.column(new StructType(false, dec128Type),
              struct(BigDecimal.valueOf(Integer.MAX_VALUE, 5)),
              struct(BigDecimal.valueOf(Long.MAX_VALUE, 5)),
              struct(new BigDecimal("111111111122222222223333333333").setScale(5)),
              struct(new BigDecimal("123456789123456789123456789").setScale(5)),
              struct((BigDecimal) null)));

      addColumnFn.put(Columns.LIST, (t) ->
          t.column(new ListType(false, new BasicType(false, DType.INT32)),
              Arrays.asList(1, 2),
              Arrays.asList(3, 4),
              Arrays.asList(5),
              Arrays.asList(6, 7),
              Arrays.asList(8, 9, 10)));
      addColumnFn.put(Columns.LIST_STRUCT, (t) ->
          t.column(new ListType(true, structType),
              Arrays.asList(struct(1, "k1", BigDecimal.ONE), struct(2, "k2", BigDecimal.ONE),
                  struct(3, "k3", BigDecimal.ONE)),
              Arrays.asList(struct(4, "k4", BigDecimal.ONE), struct(5, "k5", BigDecimal.ONE)),
              Arrays.asList(struct(6, "k6", BigDecimal.ONE)),
              Arrays.asList(new HostColumnVector.StructData((List) null)),
              (List) null));
      addColumnFn.put(Columns.LIST_DEC128, (t) ->
          t.column(new ListType(true, new StructType(false, dec128Type)),
              Arrays.asList(struct(BigDecimal.valueOf(Integer.MAX_VALUE, 5)),
                  struct(BigDecimal.valueOf(Integer.MIN_VALUE, 5))),
              Arrays.asList(struct(BigDecimal.valueOf(Long.MAX_VALUE, 5)),
                  struct(BigDecimal.valueOf(0, 5)), struct(BigDecimal.valueOf(-1, 5))),
              Arrays.asList(struct(new BigDecimal("111111111122222222223333333333").setScale(5))),
              Arrays.asList(struct(new BigDecimal("123456789123456789123456789").setScale(5))),
              Arrays.asList(struct((BigDecimal) null))));
    }

    static TestBuilder addColumn(TestBuilder tb, String colName) {
      if (!addColumnFn.containsKey(Columns.valueOf(colName))) {
        throw new IllegalArgumentException("Unknown column name: " + colName);
      }
      return addColumnFn.get(Columns.valueOf(colName)).apply(tb);
    }

    static String[] getAllColumns(boolean withDecimal128) {
      List<String> columns = Lists.newArrayList(
          Columns.BOOL.name, Columns.INT.name, Columns.BYTE.name, Columns.LONG.name,
          Columns.STRING.name, Columns.FLOAT.name, Columns.DOUBLE.name, Columns.DECIMAL64.name,
          Columns.STRUCT.name, Columns.LIST.name, Columns.LIST_STRUCT.name);
      if (withDecimal128) {
        columns.add(Columns.DECIMAL128.name);
        columns.add(Columns.STRUCT_DEC128.name);
        columns.add(Columns.LIST_DEC128.name);
      }
      String[] ret = new String[columns.size()];
      columns.toArray(ret);
      return ret;
    }

    static String[] getNonNestedColumns(boolean withDecimal128) {
      List<String> columns = Lists.newArrayList(
          Columns.BOOL.name, Columns.INT.name, Columns.BYTE.name, Columns.LONG.name,
          Columns.STRING.name, Columns.FLOAT.name, Columns.DOUBLE.name, Columns.DECIMAL64.name);
      if (withDecimal128) {
        columns.add(Columns.DECIMAL128.name);
      }
      String[] ret = new String[columns.size()];
      columns.toArray(ret);
      return ret;
    }

    static void buildWriterOptions(ColumnWriterOptions.NestedBuilder builder, List<String> columns) {
      for (String colName : columns) {
        buildWriterOptions(builder, colName);
      }
    }

    static void buildWriterOptions(ColumnWriterOptions.NestedBuilder builder, String... columns) {
      for (String colName : columns) {
        buildWriterOptions(builder, colName);
      }
    }

    static void buildWriterOptions(ColumnWriterOptions.NestedBuilder builder, String colName) {
      switch (Columns.valueOf(colName)) {
      case BOOL:
      case INT:
      case LONG:
      case FLOAT:
      case DOUBLE:
      case BYTE:
      case STRING:
        builder.withColumns(false, colName);
        break;
      case DECIMAL64:
        builder.withDecimalColumn(colName, DType.DECIMAL64_MAX_PRECISION);
        break;
      case DECIMAL128:
        builder.withDecimalColumn(colName, DType.DECIMAL128_MAX_PRECISION);
        break;
      case STRUCT:
        builder.withStructColumn(structBuilder(colName)
            .withNullableColumns("ch_int")
            .withNullableColumns("ch_str")
            .withDecimalColumn("ch_dec64", DType.DECIMAL64_MAX_PRECISION, true)
            .build());
        break;
      case LIST:
        builder.withListColumn(listBuilder(colName, false)
            .withNonNullableColumns("ch_int")
            .build());
        break;
      case LIST_STRUCT:
        builder.withListColumn(listBuilder(colName)
            .withStructColumn(structBuilder(colName)
                .withNullableColumns("ch_int")
                .withNullableColumns("ch_str")
                .withDecimalColumn("ch_dec64", DType.DECIMAL64_MAX_PRECISION, true)
                .build())
            .build());
        break;
      case STRUCT_DEC128:
        builder.withStructColumn(structBuilder(colName, false)
            .withDecimalColumn("ch_dec128", DType.DECIMAL128_MAX_PRECISION, true)
            .build());
        break;
      case LIST_DEC128:
        builder.withListColumn(listBuilder(colName)
            .withStructColumn(structBuilder(colName, false)
                .withDecimalColumn("ch_dec128", DType.DECIMAL128_MAX_PRECISION, true)
                .build())
            .build());
        break;
      default:
        throw new IllegalArgumentException("should NOT reach here");
      }
    }
  }

  private Table getExpectedFileTable(String... selectColumns) {
    return getExpectedFileTable(Lists.newArrayList(selectColumns));
  }

  private Table getExpectedFileTable(List<String> selectColumns) {
    TestBuilder tb = new TestBuilder();
    for (String c : selectColumns) {
      WriteUtils.addColumn(tb, c);
    }
    return tb.build();
  }

  private Table getExpectedFileTableWithDecimals() {
    return new TestBuilder()
        .column(true, false, false, true, false)
        .column(5, 1, 0, 2, 7)
        .column(new Byte[]{2, 3, 4, 5, 9})
        .column(3l, 9l, 4l, 2l, 20l)
        .column("this", "is", "a", "test", "string")
        .column(1.0f, 3.5f, 5.9f, 7.1f, 9.8f)
        .column(5.0d, 9.5d, 0.9d, 7.23d, 2.8d)
        .decimal32Column(3, 298, 2473, 2119, 1273, 9879)
        .decimal64Column(4, 398l, 1322l, 983237l, 99872l, 21337l)
        .build();
  }

  private final class MyBufferConsumer implements HostBufferConsumer, AutoCloseable {
    public final HostMemoryBuffer buffer;
    long offset = 0;

    public MyBufferConsumer() {
      buffer = hostMemoryAllocator.allocate(10 * 1024 * 1024);
    }

    @Override
    public void handleBuffer(HostMemoryBuffer src, long len) {
      try {
        this.buffer.copyFromHostBuffer(offset, src, 0, len);
        offset += len;
      } finally {
        src.close();
      }
    }

    @Override
    public void close() {
      buffer.close();
    }
  }

  private final class MyBufferProvider implements HostBufferProvider {
    private final MyBufferConsumer wrapped;
    long offset = 0;

    private MyBufferProvider(MyBufferConsumer wrapped) {
      this.wrapped = wrapped;
    }

    @Override
    public long readInto(HostMemoryBuffer buffer, long len) {
      long amountLeft = wrapped.offset - offset;
      long amountToCopy = Math.max(0, Math.min(len, amountLeft));
      if (amountToCopy > 0) {
        buffer.copyFromHostBuffer(0, wrapped.buffer, offset, amountToCopy);
        offset += amountToCopy;
      }
      return amountToCopy;
    }
  }

  @Test
  void testParquetWriteToBufferChunkedInt96() {
    try (Table table0 = getExpectedFileTableWithDecimals();
         MyBufferConsumer consumer = new MyBufferConsumer()) {
      ParquetWriterOptions options = ParquetWriterOptions.builder()
          .withNonNullableColumns("_c0", "_c1", "_c2", "_c3", "_c4", "_c5", "_c6")
          .withDecimalColumn("_c7", 5)
          .withDecimalColumn("_c8", 5)
          .build();

      TableDebug.get().debug("default stderr table0", table0);
      TableDebug.builder()
        .withOutput(TableDebug.Output.STDOUT)
        .build().debug("stdout table0", table0);
      TableDebug.builder()
          .withOutput(TableDebug.Output.LOG)
          .build().debug("slf4j default debug table0", table0);
      TableDebug.builder()
          .withOutput(TableDebug.Output.LOG_ERROR)
          .build().debug("slf4j error table0", table0);

      try (TableWriter writer = Table.writeParquetChunked(options, consumer)) {
        writer.write(table0);
        writer.write(table0);
        writer.write(table0);
      }
      try (Table table1 = Table.readParquet(ParquetOptions.DEFAULT, consumer.buffer, 0, consumer.offset);
           Table concat = Table.concatenate(table0, table0, table0)) {
        assertTablesAreEqual(concat, table1);
      }
    }
  }

  @Test
  void testParquetWriteMap() throws IOException {
    ParquetWriterOptions options = ParquetWriterOptions.builder()
        .withMapColumn(mapColumn("my_map",
            new ColumnWriterOptions("key0", false),
            new ColumnWriterOptions("value0"),
            true)).build();
    try (TempFile tempFile = TempFile.create("test-map", ".parquet")) {
      File f = tempFile.getFile();
      List<HostColumnVector.StructData> list1 =
          Arrays.asList(new HostColumnVector.StructData(Arrays.asList("a", "b")));
      List<HostColumnVector.StructData> list2 =
          Arrays.asList(new HostColumnVector.StructData(Arrays.asList("a", "c")));
      List<HostColumnVector.StructData> list3 =
          Arrays.asList(new HostColumnVector.StructData(Arrays.asList("e", "d")));
      HostColumnVector.StructType structType = new HostColumnVector.StructType(true,
          Arrays.asList(new HostColumnVector.BasicType(true, DType.STRING),
              new HostColumnVector.BasicType(true, DType.STRING)));
      try (ColumnVector listColumn = ColumnVector.fromLists(new HostColumnVector.ListType(true,
              structType), list1, list2, list3);
           Table t0 = new Table(listColumn)) {
        try (TableWriter writer = Table.writeParquetChunked(options, f)) {
          writer.write(t0);
        }
        try (ParquetFileReader reader =
             ParquetFileReader.open(HadoopInputFile.fromPath(new Path(f.getAbsolutePath()),
                 new Configuration()))) {
          MessageType schema = reader.getFooter().getFileMetaData().getSchema();
          assertEquals(OriginalType.MAP, schema.getType("my_map").getOriginalType());
        }
      }
      try (Table table = Table.readParquet(f);
           Scalar aString = Scalar.fromString("a");
           ColumnVector res = table.getColumn(0).getMapValue(aString);
           ColumnVector expected = ColumnVector.fromStrings("b", "c", null)) {
        assertColumnsAreEqual(expected, res);
      }
    }
  }

  @Test
  void testParquetWriteToBufferChunkedWithNested() {
    ParquetWriterOptions.Builder optBuilder = ParquetWriterOptions.builder();
    WriteUtils.buildWriterOptions(optBuilder, WriteUtils.getAllColumns(false));
    ParquetWriterOptions options = optBuilder.build();
    try (Table table0 = getExpectedFileTable(WriteUtils.getAllColumns(false));
         MyBufferConsumer consumer = new MyBufferConsumer()) {
      try (TableWriter writer = Table.writeParquetChunked(options, consumer)) {
        writer.write(table0);
        writer.write(table0);
        writer.write(table0);
      }
      try (Table table1 = Table.readParquet(ParquetOptions.DEFAULT, consumer.buffer, 0,
          consumer.offset);
           Table concat = Table.concatenate(table0, table0, table0)) {
        assertTablesAreEqual(concat, table1);
      }
    }
  }

  @Test
  void testParquetWriteToBufferChunked() {
    ParquetWriterOptions.Builder optBuilder = ParquetWriterOptions.builder();
    List<String> columns = Lists.newArrayList(WriteUtils.getNonNestedColumns(false));
    columns.add(Columns.STRUCT.name);
    WriteUtils.buildWriterOptions(optBuilder, columns);
    ParquetWriterOptions options = optBuilder.build();
    ParquetWriterOptions optionsNoCompress =
      optBuilder.withCompressionType(CompressionType.NONE)
      .withRowGroupSizeRows(10000)
      .withRowGroupSizeBytes(10000)
      .withMaxDictionarySize(2 * 1024 * 1024)
      .withDictionaryPolicy(ParquetWriterOptions.DictionaryPolicy.ALWAYS)
      .build();
    try (Table table0 = getExpectedFileTable(columns);
         MyBufferConsumer consumer = new MyBufferConsumer()) {
      try (TableWriter writer = Table.writeParquetChunked(options, consumer)) {
        writer.write(table0);
        writer.write(table0);
        writer.write(table0);

        TableWriter.WriteStatistics statistics = writer.getWriteStatistics();
        assertNotEquals(0, statistics.numCompressedBytes);
        assertEquals(0, statistics.numFailedBytes);
        assertEquals(0, statistics.numSkippedBytes);
        assertNotEquals(Double.NaN, statistics.compressionRatio);
      }
      try (Table table1 = Table.readParquet(ParquetOptions.DEFAULT, consumer.buffer, 0, consumer.offset);
           Table concat = Table.concatenate(table0, table0, table0)) {
        assertTablesAreEqual(concat, table1);
      }
      try (TableWriter writer = Table.writeParquetChunked(optionsNoCompress, consumer)) {
        writer.write(table0);
        writer.write(table0);
        writer.write(table0);

        TableWriter.WriteStatistics statistics = writer.getWriteStatistics();
        assertEquals(0, statistics.numCompressedBytes);
        assertEquals(0, statistics.numFailedBytes);
        assertEquals(0, statistics.numSkippedBytes);
        assertEquals(Double.NaN, statistics.compressionRatio);
      }
    }
  }

  /**
   * Exercises the file-chunked Parquet write path with the dictionary configuration
   * setters so any signature/forwarding drift between {@code writeParquetFileBegin}
   * and {@code writeParquetBufferBegin} is caught.
   */
  @Test
  void testParquetWriteToFileChunkedWithDictionaryOptions() throws IOException {
    try (TempFile tempFile = TempFile.create("test-dict-options", ".parquet");
         Table table0 = getExpectedFileTableWithDecimals()) {
      File file = tempFile.getFile();
      ParquetWriterOptions options = ParquetWriterOptions.builder()
          .withNonNullableColumns("first", "second", "third", "fourth", "fifth", "sixth", "seventh")
          .withDecimalColumn("eighth", 5)
          .withDecimalColumn("ninth", 6)
          .withCompressionType(CompressionType.NONE)
          .withMaxDictionarySize(2 * 1024 * 1024)
          .withDictionaryPolicy(ParquetWriterOptions.DictionaryPolicy.ALWAYS)
          .build();
      try (TableWriter writer = Table.writeParquetChunked(options, file.getAbsoluteFile())) {
        writer.write(table0);
      }
      try (Table table2 = Table.readParquet(file.getAbsoluteFile())) {
        assertTablesAreEqual(table0, table2);
      }
    }
  }

  @Test
  void testParquetWriteToFileWithNames() throws IOException {
    try (TempFile tempFile = TempFile.create("test-names", ".parquet");
         Table table0 = getExpectedFileTableWithDecimals()) {
      File file = tempFile.getFile();
      ParquetWriterOptions options = ParquetWriterOptions.builder()
          .withNonNullableColumns("first", "second", "third", "fourth", "fifth", "sixth", "seventh")
          .withDecimalColumn("eighth", 5)
          .withDecimalColumn("ninth", 6)
          .withCompressionType(CompressionType.NONE)
          .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.NONE)
          .build();
      try (TableWriter writer = Table.writeParquetChunked(options, file.getAbsoluteFile())) {
        writer.write(table0);
      }
      try (Table table2 = Table.readParquet(file.getAbsoluteFile())) {
        assertTablesAreEqual(table0, table2);
      }
    }
  }

  @Test
  void testParquetWriteToFileWithNamesAndMetadata() throws IOException {
    try (TempFile tempFile = TempFile.create("test-names-metadata", ".parquet");
         Table table0 = getExpectedFileTableWithDecimals()) {
      File file = tempFile.getFile();
      ParquetWriterOptions options = ParquetWriterOptions.builder()
          .withNonNullableColumns("first", "second", "third", "fourth", "fifth", "sixth", "seventh")
          .withDecimalColumn("eighth", 6)
          .withDecimalColumn("ninth", 8)
          .withMetadata("somekey", "somevalue")
          .withCompressionType(CompressionType.NONE)
          .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.NONE)
          .build();
      try (TableWriter writer = Table.writeParquetChunked(options, file.getAbsoluteFile())) {
        writer.write(table0);
      }
      try (Table table2 = Table.readParquet(file.getAbsoluteFile())) {
        assertTablesAreEqual(table0, table2);
      }
    }
  }

  @Test
  void testParquetWriteToFileUncompressedNoStats() throws IOException {
    try (TempFile tempFile = TempFile.create("test-uncompressed", ".parquet");
         Table table0 = getExpectedFileTableWithDecimals()) {
      File file = tempFile.getFile();
      ParquetWriterOptions options = ParquetWriterOptions.builder()
          .withNonNullableColumns("_c0", "_c1", "_c2", "_c3", "_c4", "_c5", "_c6")
          .withDecimalColumn("_c7", 4)
          .withDecimalColumn("_c8", 6)
          .withCompressionType(CompressionType.NONE)
          .withRowGroupSizeRows(10000)
          .withRowGroupSizeBytes(10000)
          .withStatisticsFrequency(ParquetWriterOptions.StatisticsFrequency.NONE)
          .build();
      try (TableWriter writer = Table.writeParquetChunked(options, file.getAbsoluteFile())) {
        writer.write(table0);
      }
      try (Table table2 = Table.readParquet(file.getAbsoluteFile())) {
        assertTablesAreEqual(table0, table2);
      }
    }
  }

  @Test
  void testParquetWriteWithFieldId() throws IOException {
    // field IDs are:
    // c1: -1, c2: 2, c3: 3, c31: 31, c32: 32, c4: -4, c5: not specified
    ColumnWriterOptions.StructBuilder sBuilder =
        structBuilder("c3", true, 3)
            .withColumn(true, "c31", 31)
            .withColumn(true, "c32", 32);
    ParquetWriterOptions options = ParquetWriterOptions.builder()
        .withColumn(true, "c1", -1)
        .withDecimalColumn("c2", 9, true, 2)
        .withStructColumn(sBuilder.build())
        .withTimestampColumn("c4", true, true, -4)
        .withColumns( true, "c5")
        .build();

    try (TempFile tempFile = TempFile.create("test-field-id", ".parquet")) {
      File file = tempFile.getFile();
      HostColumnVector.StructType structType = new HostColumnVector.StructType(
          true,
          new HostColumnVector.BasicType(true, DType.STRING),
          new HostColumnVector.BasicType(true, DType.STRING));

      try (Table table0 = new Table.TestBuilder()
          .column(true, false) // c1
          .decimal32Column(0, 298, 2473) // c2
          .column(structType, // c3
              new HostColumnVector.StructData("a", "b"), new HostColumnVector.StructData("a", "b"))
          .timestampMicrosecondsColumn(1000L, 2000L) // c4
          .column("a", "b") // c5
          .build()) {
        try (TableWriter writer = Table.writeParquetChunked(options, file.getAbsoluteFile())) {
          writer.write(table0);
        }
      }

      try (ParquetFileReader reader = ParquetFileReader.open(HadoopInputFile.fromPath(
          new Path(file.getAbsolutePath()),
          new Configuration()))) {
        MessageType schema = reader.getFooter().getFileMetaData().getSchema();
        assert (schema.getFields().get(0).getId().intValue() == -1);
        assert (schema.getFields().get(1).getId().intValue() == 2);
        assert (schema.getFields().get(2).getId().intValue() == 3);
        assert (((GroupType) schema.getFields().get(2)).getFields().get(0).getId().intValue() == 31);
        assert (((GroupType) schema.getFields().get(2)).getFields().get(1).getId().intValue() == 32);
        assert (schema.getFields().get(3).getId().intValue() == -4);
        assert (schema.getFields().get(4).getId() == null);
      }
    }
  }

  @Test
  void testParquetWriteWithFieldIdNestNotSpecified() throws IOException {
    // field IDs are:
    // c0: no field ID
    // c1: 1
    // c2: no field ID
    //   c21: 21
    //   c22: no field ID
    // c3: 3
    //   c31: 31
    //   c32: no field ID
    // c4: 0
    ColumnWriterOptions.StructBuilder c2Builder =
        structBuilder("c2", true)
            .withColumn(true, "c21", 21)
            .withColumns(true, "c22");
    ColumnWriterOptions.StructBuilder c3Builder =
        structBuilder("c3", true, 3)
            .withColumn(true, "c31", 31)
            .withColumns(true, "c32");
    ParquetWriterOptions options = ParquetWriterOptions.builder()
        .withColumns(true, "c0")
        .withDecimalColumn("c1", 9, true, 1)
        .withStructColumn(c2Builder.build())
        .withStructColumn(c3Builder.build())
        .withColumn(true, "c4", 0)
        .build();

    try (TempFile tempFile = TempFile.create("test-field-id", ".parquet")) {
      File file = tempFile.getFile();
      HostColumnVector.StructType structType = new HostColumnVector.StructType(
          true,
          new HostColumnVector.BasicType(true, DType.STRING),
          new HostColumnVector.BasicType(true, DType.STRING));

      try (Table table0 = new Table.TestBuilder()
          .column(true, false) // c0
          .decimal32Column(0, 298, 2473) // c1
          .column(structType, // c2
              new HostColumnVector.StructData("a", "b"), new HostColumnVector.StructData("a", "b"))
          .column(structType, // c3
              new HostColumnVector.StructData("a", "b"), new HostColumnVector.StructData("a", "b"))
          .column("a", "b") // c4
          .build()) {
        try (TableWriter writer = Table.writeParquetChunked(options, file.getAbsoluteFile())) {
          writer.write(table0);
        }
      }

      try (ParquetFileReader reader = ParquetFileReader.open(HadoopInputFile.fromPath(
          new Path(file.getAbsolutePath()),
          new Configuration()))) {
        MessageType schema = reader.getFooter().getFileMetaData().getSchema();
        assert (schema.getFields().get(0).getId() == null);
        assert (schema.getFields().get(1).getId().intValue() == 1);
        assert (schema.getFields().get(2).getId() == null);
        assert (((GroupType) schema.getFields().get(2)).getFields().get(0).getId().intValue() == 21);
        assert (((GroupType) schema.getFields().get(2)).getFields().get(1).getId() == null);
        assert (((GroupType) schema.getFields().get(3)).getFields().get(0).getId().intValue() == 31);
        assert (((GroupType) schema.getFields().get(3)).getFields().get(1).getId() == null);
        assert (schema.getFields().get(4).getId().intValue() == 0);
      }
    }
  }

  @Test
  void testParquetWriteFieldIdOnOuterListBinaryAndMap() throws IOException {
    // Regression test for cudf #22347: parquet field IDs supplied via
    // withBinaryColumn(name, nullable, id), listBuilder(name, nullable, id) and
    // mapColumn(name, key, value, isNullable, id) must end up on the outer
    // (list/map) column on disk, not on the inner element/key_value child.
    ParquetWriterOptions options = ParquetWriterOptions.builder()
        .withBinaryColumn("_c0", true, 1)
        .withListColumn(listBuilder("_c1", true, 2)
            .withColumn(false, "element", 22)
            .build())
        .withMapColumn(mapColumn("_c2",
            new ColumnWriterOptions("key", false),
            new ColumnWriterOptions("value"),
            true, 3))
        .build();

    try (TempFile tempFile = TempFile.create("test-field-id-outer", ".parquet")) {
      File file = tempFile.getFile();
      List<Byte> bin1 = asList("ABC");
      List<Byte> bin2 = asList("DEF");
      HostColumnVector.ListType binType = new HostColumnVector.ListType(true,
          new HostColumnVector.BasicType(false, DType.UINT8));
      HostColumnVector.ListType listOfInt = new HostColumnVector.ListType(true,
          new HostColumnVector.BasicType(false, DType.INT32));
      HostColumnVector.StructType kvType = new HostColumnVector.StructType(true,
          Arrays.asList(new HostColumnVector.BasicType(true, DType.STRING),
                        new HostColumnVector.BasicType(true, DType.STRING)));
      HostColumnVector.ListType mapType = new HostColumnVector.ListType(true, kvType);

      try (ColumnVector binCol = ColumnVector.fromLists(binType, bin1, bin2);
           ColumnVector listCol = ColumnVector.fromLists(listOfInt,
               Arrays.asList(1, 2, 3), Arrays.asList(4, 5));
           ColumnVector mapCol = ColumnVector.fromLists(mapType,
               Arrays.asList(new HostColumnVector.StructData(Arrays.asList("a", "b"))),
               Arrays.asList(new HostColumnVector.StructData(Arrays.asList("c", "d"))));
           Table t0 = new Table(binCol, listCol, mapCol)) {
        try (TableWriter writer = Table.writeParquetChunked(options, file.getAbsoluteFile())) {
          writer.write(t0);
        }
      }

      try (ParquetFileReader reader = ParquetFileReader.open(HadoopInputFile.fromPath(
          new Path(file.getAbsolutePath()), new Configuration()))) {
        MessageType schema = reader.getFooter().getFileMetaData().getSchema();
        // The id supplied to the API must land on the outer (BINARY/LIST/MAP) column.
        assertEquals(1, schema.getFields().get(0).getId().intValue());
        assertEquals(2, schema.getFields().get(1).getId().intValue());
        assertEquals(3, schema.getFields().get(2).getId().intValue());

        // No nested descendant must inherit the outer id (catches "id placed on
        // both outer and inner" regressions, including the original bug where
        // withBinary(...id) attached the id to the inner BINARY_DATA child).
        assertNoNestedIdEquals(schema.getFields().get(0), 1);
        assertNoNestedIdEquals(schema.getFields().get(1), 2);
        assertNoNestedIdEquals(schema.getFields().get(2), 3);

        // The explicit child id on the LIST<INT> "element" must be preserved.
        Type listElement = findDescendantByName(schema.getFields().get(1), "element");
        assertNotNull(listElement);
        assertEquals(22, listElement.getId().intValue());
      }
    }
  }

  private static void assertNoNestedIdEquals(Type type, int outerId) {
    if (type.isPrimitive()) return;
    for (Type child : type.asGroupType().getFields()) {
      if (child.getId() != null) {
        assertNotEquals(outerId, child.getId().intValue(),
            "Field " + child.getName() + " unexpectedly inherited outer id " + outerId);
      }
      assertNoNestedIdEquals(child, outerId);
    }
  }

  private static Type findDescendantByName(Type type, String name) {
    if (type.isPrimitive()) {
      return name.equals(type.getName()) ? type : null;
    }
    for (Type child : type.asGroupType().getFields()) {
      if (name.equals(child.getName())) return child;
      Type found = findDescendantByName(child, name);
      if (found != null) return found;
    }
    return null;
  }

  @Test
  void testParquetWriteFieldIdOnNestedArrayOfBinaryAndString() throws IOException {
    // Regression test for cudf #22347: the original Spark-Rapids/Iceberg
    // failure ("Missing required field ... optional binary element is not in
    // the store") came from array<binary> / array<string> columns where
    // iceberg looks the leaf up by its parquet field id. Lock that nested path
    // down here in addition to the top-level cases.
    ParquetWriterOptions options = ParquetWriterOptions.builder()
        .withListColumn(listBuilder("arr_bin", true, 10)
            .withBinaryColumn("element", true, 11)
            .build())
        .withListColumn(listBuilder("arr_str", true, 20)
            .withColumn(true, "element", 21)
            .build())
        .build();

    try (TempFile tempFile = TempFile.create("test-field-id-nested", ".parquet")) {
      File file = tempFile.getFile();
      HostColumnVector.ListType binEl = new HostColumnVector.ListType(true,
          new HostColumnVector.BasicType(false, DType.UINT8));
      HostColumnVector.ListType arrBinType = new HostColumnVector.ListType(true, binEl);
      HostColumnVector.ListType arrStrType = new HostColumnVector.ListType(true,
          new HostColumnVector.BasicType(true, DType.STRING));

      try (ColumnVector arrBinCol = ColumnVector.fromLists(arrBinType,
               Arrays.asList(asList("ABC"), asList("DEF")),
               Arrays.asList(asList("GHI")));
           ColumnVector arrStrCol = ColumnVector.fromLists(arrStrType,
               Arrays.asList("a", "b"),
               Arrays.asList("c"));
           Table t0 = new Table(arrBinCol, arrStrCol)) {
        try (TableWriter writer = Table.writeParquetChunked(options, file.getAbsoluteFile())) {
          writer.write(t0);
        }
      }

      try (ParquetFileReader reader = ParquetFileReader.open(HadoopInputFile.fromPath(
          new Path(file.getAbsolutePath()), new Configuration()))) {
        MessageType schema = reader.getFooter().getFileMetaData().getSchema();
        // Outer LIST ids must land on the outer list groups.
        assertEquals(10, schema.getFields().get(0).getId().intValue());
        assertEquals(20, schema.getFields().get(1).getId().intValue());

        // Outer ids must not leak into nested descendants.
        assertNoNestedIdEquals(schema.getFields().get(0), 10);
        assertNoNestedIdEquals(schema.getFields().get(1), 20);

        // Inner element ids must reach the leaf — this is the iceberg lookup
        // path that broke before the fix on array<binary>.
        Type binElement = findDescendantByName(schema.getFields().get(0), "element");
        assertNotNull(binElement);
        assertEquals(11, binElement.getId().intValue());
        Type strElement = findDescendantByName(schema.getFields().get(1), "element");
        assertNotNull(strElement);
        assertEquals(21, strElement.getId().intValue());
      }
    }
  }

  /** Return a column where DECIMAL64 has been up-casted to DECIMAL128 */
  private ColumnVector castDecimal64To128(ColumnView c) {
    DType dtype = c.getType();
    switch (dtype.getTypeId()) {
      case DECIMAL64:
        return c.castTo(DType.create(DType.DTypeEnum.DECIMAL128, dtype.getScale()));
      case STRUCT:
      case LIST:
      {
        ColumnView[] oldViews = c.getChildColumnViews();
        assert oldViews != null;
        ColumnVector[] newChildren = new ColumnVector[oldViews.length];
        try {
          for (int i = 0; i < oldViews.length; i++) {
            newChildren[i] = castDecimal64To128(oldViews[i]);
          }
          try (ColumnView newView = new ColumnView(dtype, c.getRowCount(),
              Optional.of(c.getNullCount()), c.getValid(), c.getOffsets(), newChildren)) {
            return newView.copyToColumnVector();
          }
        } finally {
          for (ColumnView v : oldViews) {
            v.close();
          }
          for (ColumnVector v : newChildren) {
            if (v != null) {
              v.close();
            }
          }
        }
      }
      default:
        if (c instanceof ColumnVector) {
          return ((ColumnVector) c).incRefCount();
        } else {
          return c.copyToColumnVector();
        }
    }
  }

  /** Return a new Table with any DECIMAL64 columns up-casted to DECIMAL128 */
  private Table castDecimal64To128(Table t) {
    final int numCols = t.getNumberOfColumns();
    ColumnVector[] cols = new ColumnVector[numCols];
    try {
      for (int i = 0; i < numCols; i++) {
        cols[i] = castDecimal64To128(t.getColumn(i));
      }
      return new Table(cols);
    } finally {
      for (ColumnVector c : cols) {
        if (c != null) {
          c.close();
        }
      }
    }
  }

  @Test
  @Disabled("arrow-java does not yet support Decimal32/Decimal64, so now that" +
    "we don't automatically upcast to decimal128 on conversion to arrow, we have" +
    "to wait until it supports those types, then upgrade")
  void testArrowIPCWriteToFileWithNamesAndMetadata() throws IOException {
    try (TempFile tempFile = TempFile.create("test-names-metadata", ".arrow")) {
      File file = tempFile.getFile();
      String[] columnNames = WriteUtils.getNonNestedColumns(false);
      try (Table table0 = getExpectedFileTable(columnNames)) {
        ArrowIPCWriterOptions options = ArrowIPCWriterOptions.builder()
                .withColumnNames(columnNames)
                .build();
        try (TableWriter writer = Table.writeArrowIPCChunked(options, file.getAbsoluteFile())) {
          writer.write(table0);
        }
        // Reading from Arrow converts decimals to DECIMAL128
        try (StreamedTableReader reader = Table.readArrowIPCChunked(file);
             Table expected = castDecimal64To128(table0)) {
          int count;
          for (count = 0; ; count++) {
            try (Table t = reader.getNextIfAvailable()) {
              if (t == null) break;
              assertTablesAreEqual(expected, t);
            }
          }
          assertEquals(1, count);
        }
      }
    }
  }

  @Test
  @Disabled("arrow-java does not yet support Decimal32/Decimal64, " +
    "this can be re-enabled once it does and we upgrade")
  void testArrowIPCWriteToBufferChunked() {
    String[] nonNestedCols = WriteUtils.getNonNestedColumns(false);
    List<String> columns = Lists.newArrayList(nonNestedCols);
    columns.add(Columns.STRUCT.name);
    columns.add(Columns.LIST.name);
    columns.add(Columns.LIST_STRUCT.name);
    try (Table table0 = getExpectedFileTable(columns);
         MyBufferConsumer consumer = new MyBufferConsumer()) {
      ArrowIPCWriterOptions options = ArrowIPCWriterOptions.builder()
              .withColumnNames(nonNestedCols)
              .withColumnNames(Columns.STRUCT.name, "int", "str", "dec64")
              .withColumnNames(Columns.LIST.name)
              .withColumnNames(Columns.LIST_STRUCT.name, "int", "str", "dec64")
              .build();
      try (TableWriter writer = Table.writeArrowIPCChunked(options, consumer)) {
        writer.write(table0);
        writer.write(table0);
        writer.write(table0);
      }
      // Reading from Arrow converts decimals to DECIMAL128
      try (StreamedTableReader reader = Table.readArrowIPCChunked(new MyBufferProvider(consumer));
           Table expected = castDecimal64To128(table0)) {
        boolean done = false;
        int count = 0;
        while (!done) {
          try (Table t = reader.getNextIfAvailable()) {
            if (t == null) {
              done = true;
            } else {
              assertTablesAreEqual(expected, t);
              count++;
            }
          }
        }
        assertEquals(3, count);
      }
    }
  }

  @Test
  void testArrowIPCWriteEmptyToBufferChunked() {
    try (Table emptyTable = new Table.TestBuilder().timestampDayColumn().build();
         MyBufferConsumer consumer = new MyBufferConsumer()) {
      ArrowIPCWriterOptions options = ArrowIPCWriterOptions.builder()
              .withColumnNames("day")
              .build();
      try (TableWriter writer = Table.writeArrowIPCChunked(options, consumer)) {
        writer.write(emptyTable);
      }
      try (StreamedTableReader reader = Table.readArrowIPCChunked(new MyBufferProvider(consumer))) {
        boolean done = false;
        int count = 0;
        while (!done) {
          try (Table t = reader.getNextIfAvailable()) {
            if (t == null) {
              done = true;
            } else {
              assertTablesAreEqual(emptyTable, t);
              count++;
            }
          }
        }
        // Expect one empty batch for the empty table.
        assertEquals(1, count);
      }
    }
  }

  @Test
  void testORCWriteToBufferChunked() {
    String[] selectedColumns = WriteUtils.getAllColumns(false);
    try (Table table0 = getExpectedFileTable(selectedColumns);
         MyBufferConsumer consumer = new MyBufferConsumer()) {
      ORCWriterOptions.Builder builder = ORCWriterOptions.builder();
      WriteUtils.buildWriterOptions(builder, selectedColumns);
      ORCWriterOptions opts = builder.build();
      ORCWriterOptions optsNoCompress = builder.withCompressionType(CompressionType.NONE).build();
      try (TableWriter writer = Table.writeORCChunked(opts, consumer)) {
        writer.write(table0);
        writer.write(table0);
        writer.write(table0);

        TableWriter.WriteStatistics statistics = writer.getWriteStatistics();
        assertNotEquals(0, statistics.numCompressedBytes);
        assertEquals(0, statistics.numFailedBytes);
        assertEquals(0, statistics.numSkippedBytes);
        assertNotEquals(Double.NaN, statistics.compressionRatio);
      }
      try (Table table1 = Table.readORC(ORCOptions.DEFAULT, consumer.buffer, 0, consumer.offset);
           Table concat = Table.concatenate(table0, table0, table0)) {
        assertTablesAreEqual(concat, table1);
      }
      try (TableWriter writer = Table.writeORCChunked(optsNoCompress, consumer)) {
        writer.write(table0);
        writer.write(table0);
        writer.write(table0);

        TableWriter.WriteStatistics statistics = writer.getWriteStatistics();
        assertEquals(0, statistics.numCompressedBytes);
        assertEquals(0, statistics.numFailedBytes);
        assertEquals(0, statistics.numSkippedBytes);
        assertEquals(Double.NaN, statistics.compressionRatio);
      }
    }
  }

  @Test
  void testORCWriteToFileChunked() throws IOException {
    try (TempFile tempFile = TempFile.create("test", ".orc")) {
      File file = tempFile.getFile();
      String[] selectedColumns = WriteUtils.getAllColumns(false);
      try (Table table0 = getExpectedFileTable(selectedColumns)) {
        ORCWriterOptions.Builder builder = ORCWriterOptions.builder();
        WriteUtils.buildWriterOptions(builder, selectedColumns);
        ORCWriterOptions opts = builder.build();
        try (TableWriter writer = Table.writeORCChunked(opts, file.getAbsoluteFile())) {
          writer.write(table0);
        }
        try (Table table1 = Table.readORC(file.getAbsoluteFile())) {
          assertTablesAreEqual(table0, table1);
        }
      }
    }
  }

  @Test
  void testORCWriteMapChunked() throws IOException {
    ORCWriterOptions options = ORCWriterOptions.builder()
            .withMapColumn(mapColumn("my_map",
                    new ColumnWriterOptions("key0", false),
                    new ColumnWriterOptions("value0"),
                    true)).build();
    List<HostColumnVector.StructData> list1 =
            Arrays.asList(new HostColumnVector.StructData(Arrays.asList("a", "b")));
    List<HostColumnVector.StructData> list2 =
            Arrays.asList(new HostColumnVector.StructData(Arrays.asList("a", "c")));
    List<HostColumnVector.StructData> list3 =
            Arrays.asList(new HostColumnVector.StructData(Arrays.asList("e", "d")));
    HostColumnVector.StructType structType = new HostColumnVector.StructType(true,
            Arrays.asList(new HostColumnVector.BasicType(true, DType.STRING),
                    new HostColumnVector.BasicType(true, DType.STRING)));
    try (TempFile tempFile = TempFile.create("test-map", ".orc")) {
      File f = tempFile.getFile();
      try (ColumnVector listColumn = ColumnVector.fromLists(new HostColumnVector.ListType(true,
              structType), list1, list2, list3);
           Table t0 = new Table(listColumn)) {
        try (TableWriter writer = Table.writeORCChunked(options, f)) {
          writer.write(t0);
        }
        try (Table res = Table.readORC(f)) {
          assertTablesAreEqual(t0, res);
        }
      }
    }
  }

  @Test
  void testORCWriteToFileWithColNames() throws IOException {
    try (TempFile tempFile = TempFile.create("test", ".orc")) {
      File file = tempFile.getFile();
      String[] colNames = WriteUtils.getNonNestedColumns(false);
      try (Table table0 = getExpectedFileTable(colNames)) {
        ORCWriterOptions.Builder optBuilder = ORCWriterOptions.builder();
        WriteUtils.buildWriterOptions(optBuilder, colNames);
        ORCWriterOptions options = optBuilder.build();
        try (TableWriter writer = Table.writeORCChunked(options, file.getAbsoluteFile())) {
          writer.write(table0);
        }
        ORCOptions opts = ORCOptions.builder().includeColumn(colNames).build();
        try (Table table1 = Table.readORC(opts, file.getAbsoluteFile())) {
          assertTablesAreEqual(table0, table1);
        }
      }
    }
  }

  @Test
  void testORCReadAndWriteForDecimal128() throws IOException {
    try (TempFile tempFile = TempFile.create("test", ".orc")) {
      File file = tempFile.getFile();
      String[] colNames = new String[]{Columns.DECIMAL64.name,
          Columns.DECIMAL128.name, Columns.STRUCT_DEC128.name, Columns.LIST_DEC128.name};
      try (Table table0 = getExpectedFileTable(colNames)) {
        ORCWriterOptions.Builder optBuilder = ORCWriterOptions.builder();
        WriteUtils.buildWriterOptions(optBuilder, colNames);
        ORCWriterOptions options = optBuilder.build();
        try (TableWriter writer = Table.writeORCChunked(options, file.getAbsoluteFile())) {
          writer.write(table0);
        }
        ORCOptions opts = ORCOptions.builder()
            .includeColumn(colNames)
            .decimal128Column(Columns.DECIMAL128.name,
                String.format("%s.%s", Columns.STRUCT_DEC128.name, "ch_dec128"),
                String.format("%s.1.%s", Columns.LIST_DEC128.name, "ch_dec128"))
            .build();
        try (Table table1 = Table.readORC(opts, file.getAbsoluteFile())) {
          assertTablesAreEqual(table0, table1);
        }
      }
    }
  }

  @Test
  void testORCWriteToFileUncompressed() throws IOException {
    try (TempFile tempFileUncompressed = TempFile.create("test-uncompressed", ".orc");
         Table table0 = getExpectedFileTable(WriteUtils.getNonNestedColumns(false))) {
      File file = tempFileUncompressed.getFile();
      String[] colNames = WriteUtils.getNonNestedColumns(false);
      ORCWriterOptions.Builder optsBuilder = ORCWriterOptions.builder();
      WriteUtils.buildWriterOptions(optsBuilder, colNames);
      optsBuilder.withCompressionType(CompressionType.NONE);
      ORCWriterOptions opts = optsBuilder.build();
      try (TableWriter writer =
               Table.writeORCChunked(opts, file.getAbsoluteFile())) {
        writer.write(table0);
      }
      try (Table table2 = Table.readORC(file.getAbsoluteFile())) {
        assertTablesAreEqual(table0, table2);
      }
    }
  }

  @Test
  void testStructColumnFilter() {
    List<HostColumnVector.DataType> children =
        Arrays.asList(new HostColumnVector.BasicType(true, DType.INT32),
            new HostColumnVector.BasicType(true, DType.INT64));
    HostColumnVector.StructType type = new HostColumnVector.StructType(true, children);
    HostColumnVector.StructType expectedType = new HostColumnVector.StructType(true, children);
    List data1 = Arrays.asList(10, 20L);
    List data2 = Arrays.asList(50, 60L);
    List data3 = Arrays.asList(null, 80L);
    List data4 = null;
    HostColumnVector.StructData structData1 = new HostColumnVector.StructData(data1);
    HostColumnVector.StructData structData2 = new HostColumnVector.StructData(data2);
    HostColumnVector.StructData structData3 = new HostColumnVector.StructData(data3);
    HostColumnVector.StructData structData4 = new HostColumnVector.StructData(data4);
    try (ColumnVector mask = ColumnVector.fromBoxedBooleans(true, false, true, false);
         ColumnVector fromStructs = ColumnVector.fromStructs(type, Arrays.asList(structData1, structData2, structData3, structData4));
         Table input = new Table(fromStructs);
         Table filteredTable = input.filter(mask);
         ColumnVector expectedStructs = ColumnVector.fromStructs(expectedType, Arrays.asList(structData1, structData3));
         Table expected = new Table(expectedStructs)) {
      assertTablesAreEqual(expected, filteredTable);
    }
  }

  @Test
  void testStructColumnFilterStrings() {
    List<HostColumnVector.DataType> children =
        Arrays.asList(new HostColumnVector.BasicType(true, DType.STRING),
            new HostColumnVector.BasicType(true, DType.STRING));
    HostColumnVector.StructType type = new HostColumnVector.StructType(true, children);
    HostColumnVector.StructType expectedType = new HostColumnVector.StructType(true, children);
    List data1 = Arrays.asList("10", "aliceAndBob");
    List data2 = Arrays.asList("50", "foobar");
    List data3 = Arrays.asList(null, "zombies");
    List data4 = null;
    HostColumnVector.StructData structData1 = new HostColumnVector.StructData(data1);
    HostColumnVector.StructData structData2 = new HostColumnVector.StructData(data2);
    HostColumnVector.StructData structData3 = new HostColumnVector.StructData(data3);
    HostColumnVector.StructData structData4 = new HostColumnVector.StructData(data4);
    try (ColumnVector mask = ColumnVector.fromBoxedBooleans(true, false, true, true);
         ColumnVector fromStructs = ColumnVector.fromStructs(type, Arrays.asList(structData1, structData2, structData3, structData4));
         Table input = new Table(fromStructs);
         Table filteredTable = input.filter(mask);
         ColumnVector expectedStructs = ColumnVector.fromStructs(expectedType, Arrays.asList(structData1, structData3, structData4));
         Table expected = new Table(expectedStructs)) {
      assertEquals(expected.getRowCount(), 3L, "Expected column row count is incorrect");
      assertTablesAreEqual(expected, filteredTable);
    }
  }

  // utility methods to reduce typing

  private static StructData struct(Object... values) {
    return new StructData(values);
  }

  private StructData[] structs(StructData... values) {
    return values;
  }

  private String[] strings(String... values) {
    return values;
  }

  private static ColumnVector decimalFromBoxedInts(boolean isDec64, int scale, Integer... values) {
    BigDecimal[] decimals = new BigDecimal[values.length];
    for (int i = 0; i < values.length; i++) {
      if (values[i] == null) {
        decimals[i] = null;
      } else {
        decimals[i] = BigDecimal.valueOf(values[i], -scale);
      }
    }
    DType type = isDec64 ? DType.create(DType.DTypeEnum.DECIMAL64, scale) : DType.create(DType.DTypeEnum.DECIMAL32, scale);
    return ColumnVector.build(type, decimals.length, (b) -> b.appendBoxed(decimals));
  }

  private Table buildTestTable() {
    StructType mapStructType = new StructType(true,
        new BasicType(false, DType.STRING),
        new BasicType(false, DType.STRING));
    StructType structType = new StructType(true,
        new BasicType(true, DType.INT32),
        new BasicType(false, DType.FLOAT32));
    return new Table.TestBuilder()
        .column(     100,      202,      3003,    40004,        5,      -60,    1, null,    3,  null,     5, null,    7, null,   9,   null,    11, null,   13, null,  15)
        .column(    true,     true,     false,    false,     true,     null, true, true, null, false, false, null, true, true, null, false, false, null, true, true, null)
        .column( (byte)1,  (byte)2,      null,  (byte)4,  (byte)5,  (byte)6, (byte)1, (byte)2, (byte)3, null, (byte)5, (byte)6, (byte)7, null, (byte)9, (byte)10, (byte)11, null, (byte)13, (byte)14, (byte)15)
        .column((short)6, (short)5,  (short)4,     null, (short)2, (short)1, (short)1, (short)2, (short)3, null, (short)5, (short)6, (short)7, null, (short)9, (short)10, null, (short)12, (short)13, (short)14, null)
        .column(      1L,     null,     1001L,      50L,   -2000L,     null, 1L, 2L, 3L, 4L, null, 6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, null)
        .column(   10.1f,      20f, Float.NaN,  3.1415f,     -60f,     null, 1f, 2f, 3f, 4f, 5f, null, 7f, 8f, 9f, 10f, 11f, null, 13f, 14f, 15f)
        .column(   10.1f,      20f, Float.NaN,  3.1415f,     -60f,     -50f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, 10f, 11f, 12f, 13f, 14f, 15f)
        .column(    10.1,     20.0,      33.1,   3.1415,    -60.5,     null, 1., 2., 3., 4., 5., 6., null, 8., 9., 10., 11., 12., null, 14., 15.)
        .timestampDayColumn(99,      100,      101,      102,      103,      104, 1, 2, 3, 4, 5, 6, 7, null, 9, 10, 11, 12, 13, null, 15)
        .timestampMillisecondsColumn(9L,    1006L,     101L,    5092L,     null,      88L, 1L, 2L, 3L, 4L, 5L ,6L, 7L, 8L, null, 10L, 11L, 12L, 13L, 14L, 15L)
        .timestampSecondsColumn(1L, null, 3L, 4L, 5L, 6L, 1L, 2L, 3L, 4L, 5L ,6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, 15L)
        .decimal32Column(-3, 100,      202,      3003,    40004,        5,      -60,    1, null,    3,  null,     5, null,    7, null,   9,   null,    11, null,   13, null,  15)
        .decimal64Column(-8,      1L,     null,     1001L,      50L,   -2000L,     null, 1L, 2L, 3L, 4L, null, 6L, 7L, 8L, 9L, null, 11L, 12L, 13L, 14L, null)
        .column(     "A",      "B",      "C",      "D",     null,   "TESTING", "1", "2", "3", "4", "5", "6", "7", null, "9", "10", "11", "12", "13", null, "15")
        .column(
            strings("1", "2", "3"), strings("4"), strings("5"), strings("6, 7"),
            strings("", "9", null), strings("11"), strings(""), strings(null, null),
            strings("15", null), null, null, strings("18", "19", "20"),
            null, strings("22"), strings("23", ""), null,
            null, null, null, strings(),
            strings("the end"))
        .column(mapStructType,
            structs(struct("1", "2")), structs(struct("3", "4")),
            null, null,
            structs(struct("key", "value"), struct("a", "b")), null,
            null, structs(struct("3", "4"), struct("1", "2")),
            structs(), structs(null, struct("foo", "bar")),
            structs(null, null, null), null,
            null, null,
            null, null,
            null, null,
            null, null,
            structs(struct("the", "end")))
        .column(structType,
            struct(1, 1f), null, struct(2, 3f), null, struct(8, 7f),
            struct(0, 0f), null, null, struct(-1, -1f), struct(-100, -100f),
            struct(Integer.MAX_VALUE, Float.MAX_VALUE), null, null, null, null,
            null, null, null, null, null,
            struct(Integer.MIN_VALUE, Float.MIN_VALUE))
        .column(     "A",      "A",      "C",      "C",     null,   "TESTING", "1", "2", "3", "4", "5", "6", "7", null, "9", "10", "11", "12", "13", null, "15")
        .build();
  }

  @Test
  void testBuilderWithColumn() {
    try (Table t1 = new Table.TestBuilder()
        .decimal32Column(-3, 120, -230, null, 340)
        .decimal64Column(-8, 1000L, 200L, null, 30L).build()) {
      try (Table t2 = new Table.TestBuilder()
          .decimal32Column(-3, RoundingMode.UNNECESSARY, 0.12, -0.23, null, 0.34)
          .decimal64Column(-8, RoundingMode.UNNECESSARY, 1e-5, 2e-6, null, 3e-7).build()) {
        try (Table t3 = new Table.TestBuilder()
            .decimal32Column(-3, RoundingMode.UNNECESSARY, "0.12", "-000.23", null, ".34")
            .decimal64Column(-8, RoundingMode.UNNECESSARY, "1e-5", "2e-6", null, "3e-7").build()) {
          assertTablesAreEqual(t1, t2);
          assertTablesAreEqual(t1, t3);
        }
      }
    }
  }

  private Table[] buildExplodeTestTableWithPrimitiveTypes(boolean pos, boolean outer) {
    try (Table input = new Table.TestBuilder()
        .column(new ListType(true, new BasicType(true, DType.INT32)),
            Arrays.asList(1, 2, 3),
            Arrays.asList(4, 5),
            Arrays.asList(6),
            null,
            Arrays.asList())
        .column("s1", "s2", "s3", "s4", "s5")
        .column(1, 3, 5, 7, 9)
        .column(12.0, 14.0, 13.0, 11.0, 15.0)
        .build()) {
      Table.TestBuilder expectedBuilder = new Table.TestBuilder();
      if (pos) {
        Integer[] posData = outer ? new Integer[]{0, 1, 2, 0, 1, 0, null, null} : new Integer[]{0, 1, 2, 0, 1, 0};
        expectedBuilder.column(posData);
      }
      List<Object[]> expectedData = new ArrayList<Object[]>(){{
        if (!outer) {
          this.add(new Integer[]{1, 2, 3, 4, 5, 6});
          this.add(new String[]{"s1", "s1", "s1", "s2", "s2", "s3"});
          this.add(new Integer[]{1, 1, 1, 3, 3, 5});
          this.add(new Double[]{12.0, 12.0, 12.0, 14.0, 14.0, 13.0});
        } else {
          this.add(new Integer[]{1, 2, 3, 4, 5, 6, null, null});
          this.add(new String[]{"s1", "s1", "s1", "s2", "s2", "s3", "s4", "s5"});
          this.add(new Integer[]{1, 1, 1, 3, 3, 5, 7, 9});
          this.add(new Double[]{12.0, 12.0, 12.0, 14.0, 14.0, 13.0, 11.0, 15.0});
        }
      }};
      try (Table expected = expectedBuilder.column((Integer[]) expectedData.get(0))
          .column((String[]) expectedData.get(1))
          .column((Integer[]) expectedData.get(2))
          .column((Double[]) expectedData.get(3))
          .build()) {
        return new Table[]{new Table(input.getColumns()), new Table(expected.getColumns())};
      }
    }
  }

  private Table[] buildExplodeTestTableWithNestedTypes(boolean pos, boolean outer) {
    StructType nestedType = new StructType(true,
        new BasicType(false, DType.INT32), new BasicType(false, DType.STRING));
    try (Table input = new Table.TestBuilder()
        .column(new ListType(false, nestedType),
            Arrays.asList(struct(1, "k1"), struct(2, "k2"), struct(3, "k3")),
            Arrays.asList(struct(4, "k4"), struct(5, "k5")),
            Arrays.asList(struct(6, "k6")),
            Arrays.asList(new HostColumnVector.StructData((List) null)),
            null)
        .column("s1", "s2", "s3", "s4", "s5")
        .column(1, 3, 5, 7, 9)
        .column(12.0, 14.0, 13.0, 11.0, 15.0)
        .build()) {
      Table.TestBuilder expectedBuilder = new Table.TestBuilder();
      if (pos) {
        if (outer) {
          expectedBuilder.column(0, 1, 2, 0, 1, 0, 0, null);
        } else {
          expectedBuilder.column(0, 1, 2, 0, 1, 0, 0);
        }
      }
      List<Object[]> expectedData = new ArrayList<Object[]>(){{
        if (!outer) {
          this.add(new HostColumnVector.StructData[]{
              struct(1, "k1"), struct(2, "k2"), struct(3, "k3"),
              struct(4, "k4"), struct(5, "k5"), struct(6, "k6"),
              new HostColumnVector.StructData((List) null)});
          this.add(new String[]{"s1", "s1", "s1", "s2", "s2", "s3", "s4"});
          this.add(new Integer[]{1, 1, 1, 3, 3, 5, 7});
          this.add(new Double[]{12.0, 12.0, 12.0, 14.0, 14.0, 13.0, 11.0});
        } else {
          this.add(new HostColumnVector.StructData[]{
              struct(1, "k1"), struct(2, "k2"), struct(3, "k3"),
              struct(4, "k4"), struct(5, "k5"), struct(6, "k6"),
              new HostColumnVector.StructData((List) null), null});
          this.add(new String[]{"s1", "s1", "s1", "s2", "s2", "s3", "s4", "s5"});
          this.add(new Integer[]{1, 1, 1, 3, 3, 5, 7, 9});
          this.add(new Double[]{12.0, 12.0, 12.0, 14.0, 14.0, 13.0, 11.0, 15.0});
        }
      }};
      try (Table expected = expectedBuilder
          .column(nestedType, (HostColumnVector.StructData[]) expectedData.get(0))
          .column((String[]) expectedData.get(1))
          .column((Integer[]) expectedData.get(2))
          .column((Double[]) expectedData.get(3))
          .build()) {
        return new Table[]{new Table(input.getColumns()), new Table(expected.getColumns())};
      }
    }
  }

  @Test
  void testExplode() {
    // Child is primitive type
    Table[] testTables = buildExplodeTestTableWithPrimitiveTypes(false, false);
    try (Table input = testTables[0];
         Table expected = testTables[1]) {
      try (Table exploded = input.explode(0)) {
        assertTablesAreEqual(expected, exploded);
      }
    }

    // Child is nested type
    Table[] testTables2 = buildExplodeTestTableWithNestedTypes(false, false);
    try (Table input = testTables2[0];
         Table expected = testTables2[1]) {
      try (Table exploded = input.explode(0)) {
        assertTablesAreEqual(expected, exploded);
      }
    }
  }

  @Test
  void testExplodePosition() {
    // Child is primitive type
    Table[] testTables = buildExplodeTestTableWithPrimitiveTypes(true, false);
    try (Table input = testTables[0];
         Table expected = testTables[1]) {
      try (Table exploded = input.explodePosition(0)) {
        assertTablesAreEqual(expected, exploded);
      }
    }

    // Child is nested type
    Table[] testTables2 = buildExplodeTestTableWithNestedTypes(true, false);
    try (Table input = testTables2[0];
         Table expected = testTables2[1]) {
      try (Table exploded = input.explodePosition(0)) {
        assertTablesAreEqual(expected, exploded);
      }
    }
  }

  @Test
  void testExplodeOuter() {
    // Child is primitive type
    Table[] testTables = buildExplodeTestTableWithPrimitiveTypes(false, true);
    try (Table input = testTables[0];
         Table expected = testTables[1]) {
      try (Table exploded = input.explodeOuter(0)) {
        assertTablesAreEqual(expected, exploded);
      }
    }

    // Child is nested type
    Table[] testTables2 = buildExplodeTestTableWithNestedTypes(false, true);
    try (Table input = testTables2[0];
         Table expected = testTables2[1]) {
      try (Table exploded = input.explodeOuter(0)) {
        assertTablesAreEqual(expected, exploded);
      }
    }
  }

  @Test
  void testExplodeOuterPosition() {
    // Child is primitive type
    Table[] testTables = buildExplodeTestTableWithPrimitiveTypes(true, true);
    try (Table input = testTables[0];
         Table expected = testTables[1]) {
      try (Table exploded = input.explodeOuterPosition(0)) {
        assertTablesAreEqual(expected, exploded);
      }
    }

    // Child is nested type
    Table[] testTables2 = buildExplodeTestTableWithNestedTypes(true, true);
    try (Table input = testTables2[0];
         Table expected = testTables2[1]) {
      try (Table exploded = input.explodeOuterPosition(0)) {
        assertTablesAreEqual(expected, exploded);
      }
    }
  }

  @Test
  void testSample() {
    try (Table t = new Table.TestBuilder().column("s1", "s2", "s3", "s4", "s5").build()) {
      try (Table ret = t.sample(3, false, 0)) {
        assertEquals(ret.getRowCount(), 3);
      }

      try (Table ret = t.sample(5, false, 0)) {
        assertEquals(ret.getRowCount(), 5);
      }

      try (Table ret = t.sample(8, true, 0)) {
        assertEquals(ret.getRowCount(), 8);
      }
    }
  }

  @Test
  void testWindowOptionsRowsFrameOrderByGettersReturnNull() {
    // A ROWS-frame WindowOptions sets no order-by, so the parallel order-by arrays stay null and
    // their getters must return null rather than NPE on an unguarded copy.
    try (Scalar one = Scalar.fromInt(1);
         WindowOptions rowsWindow = WindowOptions.builder()
             .minPeriods(1)
             .window(one, one)
             .build()) {
      assertEquals(WindowOptions.FrameType.ROWS, rowsWindow.getFrameType());
      assertNull(rowsWindow.getOrderByColumnIndices());
      assertNull(rowsWindow.getOrderByAscending());
      assertNull(rowsWindow.getOrderByNullsFirst());
    }
  }
}

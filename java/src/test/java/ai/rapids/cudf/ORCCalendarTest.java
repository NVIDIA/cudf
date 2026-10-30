/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.hive.ql.exec.vector.ListColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.LongColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.StructColumnVector;
import org.apache.hadoop.hive.ql.exec.vector.VectorizedRowBatch;
import org.apache.orc.DateColumnStatistics;
import org.apache.orc.OrcFile;
import org.apache.orc.Reader;
import org.apache.orc.RecordReader;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.time.LocalDate;
import java.util.Arrays;
import java.util.Collections;

import static ai.rapids.cudf.AssertUtils.assertTablesAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ORCCalendarTest {
  private static final Integer[] DAYS = Arrays.stream(new String[] {
      "1001-01-01", "1200-01-01", "1582-10-04", "1582-10-10", "1582-10-15",
      "1970-01-01", "2024-05-06", null
  }).map(value -> value == null ? null : Math.toIntExact(LocalDate.parse(value).toEpochDay()))
      .toArray(Integer[]::new);

  @Test
  void rejectsUnsupportedCalendars() {
    assertEquals(ORCWriterOptions.CalendarKind.PROLEPTIC_GREGORIAN,
        ORCWriterOptions.builder().build().getCalendarKind());
    assertThrows(IllegalArgumentException.class,
        () -> ORCWriterOptions.builder().withCalendarKind(null));
    assertEquals(ORCWriterOptions.CalendarKind.JULIAN_GREGORIAN, ORCWriterOptions.builder()
        .withCalendarKind(ORCWriterOptions.CalendarKind.JULIAN_GREGORIAN).build().getCalendarKind());
  }

  @Test
  @SuppressWarnings("unchecked")
  void datesInteroperateWithApacheOrc() throws IOException {
    checkDates(DAYS, false);
  }

  @Test
  void legacyDatesInteroperateWithApacheOrc() throws IOException {
    Integer[] days = Arrays.stream(new String[] {
        "0001-01-01", "0100-03-01", "0400-02-29", "1001-01-01", "1200-01-01", "1500-02-28",
        "1500-03-01", "1582-10-04", "1582-10-15", "1970-01-01", "2024-05-06", null
    }).map(value -> value == null ? null : Math.toIntExact(LocalDate.parse(value).toEpochDay()))
        .toArray(Integer[]::new);
    checkDates(days, true);
  }

  @Test
  void apacheLegacyDatesReadAsProleptic() throws IOException {
    // Independent Apache ORC conversion provides the encoded day oracle, including BCE dates
    // and a deterministic wide sample within Apache HybridChronology's negative-year range.
    java.util.Random random = new java.util.Random(23794);
    Integer[] proleptic = new Integer[4096];
    Integer[] hybrid = new Integer[proleptic.length];
    // Hive 2.8.1's vector calendar conversion uses SimpleDateFormat without an era field.
    // Check BCE encoding against Apache ORC's DateUtils directly, which preserves the era.
    int[] anchors = Arrays.stream(new String[] {
        "-10000-02-28", "-0400-02-29", "-0001-03-01", "0000-02-29", "0001-01-01"
    }).mapToInt(value -> Math.toIntExact(LocalDate.parse(value).toEpochDay())).toArray();
    for (int i = 0; i < proleptic.length; ++i) {
      int day = i < anchors.length ? anchors[i] : random.nextInt();
      if (day < -365000000) { day %= 365000000; }
      if (day < Integer.MIN_VALUE + 100000 || (day >= -141437 && day < -141427)) {
        day = 0;
      }
      proleptic[i] = day;
      hybrid[i] = org.apache.orc.impl.DateUtils.convertDateToHybrid(day);
    }
    java.nio.file.Path file = Files.createTempFile("apache-legacy", ".orc");
    try {
      org.apache.orc.TypeDescription schema = org.apache.orc.TypeDescription.fromString("struct<d:date>");
      try (org.apache.orc.Writer writer = OrcFile.createWriter(new Path(file.toUri()),
          OrcFile.writerOptions(new Configuration()).overwrite(true).setSchema(schema).setProlepticGregorian(false)
              .compress(org.apache.orc.CompressionKind.NONE))) {
        VectorizedRowBatch batch = schema.createRowBatch();
        for (int i = 0; i < hybrid.length; ++i) {
          ((LongColumnVector) batch.cols[0]).vector[batch.size++] = hybrid[i];
          if (batch.size == batch.getMaxSize()) {
            writer.addRowBatch(batch);
            batch.reset();
          }
        }
        if (batch.size > 0) { writer.addRowBatch(batch); }
      }
      try (ColumnVector dates = ColumnVector.timestampDaysFromBoxedInts(proleptic);
           ColumnVector encoded = ColumnVector.timestampDaysFromBoxedInts(hybrid);
           Table expected = new Table(dates);
           Table rawExpected = new Table(encoded);
           Table actual = Table.readORC(ORCOptions.builder().withNumPyTypes(false)
               .convertDatesToProleptic(true).build(), file.toFile());
           Table raw = Table.readORC(ORCOptions.builder().withNumPyTypes(false).build(), file.toFile())) {
        assertTablesAreEqual(expected, actual);
        assertTablesAreEqual(rawExpected, raw);
        java.nio.file.Path gpuFile = Files.createTempFile("gpu-legacy", ".orc");
        try {
          try (TableWriter writer = Table.writeORCChunked(ORCWriterOptions.builder()
              .withColumns(true, "d").withCalendarKind(ORCWriterOptions.CalendarKind.JULIAN_GREGORIAN)
              .build(), gpuFile.toFile())) {
            writer.write(expected);
          }
          try (Table gpuRaw = Table.readORC(ORCOptions.builder().withNumPyTypes(false).build(),
              gpuFile.toFile())) {
            assertTablesAreEqual(rawExpected, gpuRaw);
          }
        } finally { Files.deleteIfExists(gpuFile); }
      }
      byte[] originalBytes = Files.readAllBytes(file);
      for (org.apache.orc.OrcProto.CalendarKind calendar : new org.apache.orc.OrcProto.CalendarKind[] {
          null, org.apache.orc.OrcProto.CalendarKind.UNKNOWN_CALENDAR,
          org.apache.orc.OrcProto.CalendarKind.PROLEPTIC_GREGORIAN,
          org.apache.orc.OrcProto.CalendarKind.JULIAN_GREGORIAN}) {
        byte[] variant = withCalendar(originalBytes, calendar);
        try (ColumnVector dates = ColumnVector.timestampDaysFromBoxedInts(
                 calendar == org.apache.orc.OrcProto.CalendarKind.JULIAN_GREGORIAN ? proleptic : hybrid);
             Table expected = new Table(dates);
             Table actual = Table.readORC(ORCOptions.builder().withNumPyTypes(false)
                 .convertDatesToProleptic(true).build(), variant)) {
          assertTablesAreEqual(expected, actual);
        }
      }
      // Unknown protobuf enum numbers must not alias recognized calendars after narrowing.
      for (int calendar : new int[] {3, 257, 258, -1}) {
        byte[] variant = withCalendarValue(originalBytes, calendar);
        try (ColumnVector dates = ColumnVector.timestampDaysFromBoxedInts(hybrid);
             Table expected = new Table(dates);
             Table actual = Table.readORC(ORCOptions.builder().withNumPyTypes(false)
                 .convertDatesToProleptic(true).build(), variant)) {
          assertTablesAreEqual(expected, actual);
        }
      }
      Long[] millis = Arrays.stream(proleptic).map(day -> day.longValue() * 86400000L)
          .toArray(Long[]::new);
      try (ColumnVector dates = ColumnVector.timestampMilliSecondsFromBoxedLongs(millis);
           Table expected = new Table(dates);
           Table actual = Table.readORC(ORCOptions.builder().convertDatesToProleptic(true).build(),
               Files.readAllBytes(file))) {
        assertTablesAreEqual(expected, actual);
      }
    } finally { Files.deleteIfExists(file); }
  }

  @Test
  void rejectsJulianOnlyLeapDayOnConversion() throws IOException {
    int day = org.apache.orc.impl.DateUtils.parseDate("1500-02-29", false);
    java.nio.file.Path file = Files.createTempFile("apache-julian-leap", ".orc");
    try {
      org.apache.orc.TypeDescription schema = org.apache.orc.TypeDescription.fromString("struct<d:date>");
      try (org.apache.orc.Writer writer = OrcFile.createWriter(new Path(file.toUri()),
          OrcFile.writerOptions(new Configuration()).overwrite(true).setSchema(schema)
              .setProlepticGregorian(false))) {
        VectorizedRowBatch batch = schema.createRowBatch();
        ((LongColumnVector) batch.cols[0]).vector[0] = day;
        batch.size = 1;
        writer.addRowBatch(batch);
      }
      try (Table raw = Table.readORC(ORCOptions.builder().withNumPyTypes(false).build(), file.toFile())) {
        assertEquals(1, raw.getRowCount());
      }
      assertThrows(CudfException.class, () -> Table.readORC(ORCOptions.builder()
          .convertDatesToProleptic(true).build(), file.toFile()));
    } finally { Files.deleteIfExists(file); }
  }

  @SuppressWarnings("unchecked")
  private void checkDates(Integer[] days, boolean legacy) throws IOException {
    HostColumnVector.ListType listType = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.TIMESTAMP_DAYS));
    try (ColumnVector dates = ColumnVector.timestampDaysFromBoxedInts(days);
         ColumnVector structs = ColumnVector.makeStruct(dates);
         ColumnVector lists = ColumnVector.fromLists(listType,
             Arrays.stream(days).map(Collections::singletonList).toArray(java.util.List[]::new));
         Table input = new Table(dates, structs, lists)) {
      for (CompressionType compression : new CompressionType[] {
          CompressionType.NONE, CompressionType.SNAPPY}) {
        for (boolean bufferSink : new boolean[] {false, true}) {
          for (boolean explicitCalendar : new boolean[] {false, true}) {
            if (legacy && !explicitCalendar) { continue; }
            java.nio.file.Path file = Files.createTempFile("orc-calendar", ".orc");
            try {
              ORCWriterOptions.Builder builder = ORCWriterOptions.builder()
                  .withColumns(true, "d")
                  .withStructColumn(ColumnWriterOptions.structBuilder("s")
                      .withColumns(true, "d").build())
                  .withListColumn(ColumnWriterOptions.listBuilder("l")
                      .withColumns(true, "d").build())
                  .withCompressionType(compression);
              if (legacy || explicitCalendar) {
                builder.withCalendarKind(legacy ? ORCWriterOptions.CalendarKind.JULIAN_GREGORIAN
                    : ORCWriterOptions.CalendarKind.PROLEPTIC_GREGORIAN);
              }
              ORCWriterOptions options = builder.build();
              if (bufferSink) {
                ByteArrayOutputStream output = new ByteArrayOutputStream();
                HostBufferConsumer consumer = (buffer, length) -> {
                  try {
                    byte[] bytes = new byte[Math.toIntExact(length)];
                    buffer.getBytes(bytes, 0, 0, bytes.length);
                    output.write(bytes, 0, bytes.length);
                  } finally {
                    buffer.close();
                  }
                };
                try (TableWriter writer = Table.writeORCChunked(options, consumer)) {
                  writer.write(input);
                  writer.write(input);
                }
                Files.write(file, output.toByteArray());
              } else {
                try (TableWriter writer = Table.writeORCChunked(options, file.toFile())) {
                  writer.write(input);
                  writer.write(input);
                }
              }
              try (Table expected = Table.concatenate(input, input);
                   Table actual = Table.readORC(
                       ORCOptions.builder().withNumPyTypes(false).convertDatesToProleptic(true).build(), file.toFile())) {
                assertTablesAreEqual(expected, actual);
              }
              // Both missing-field defaults must be overridden by the file's explicit calendar.
              for (boolean defaultProleptic : new boolean[] {false, true}) {
                Configuration conf = new Configuration();
                conf.setBoolean("orc.proleptic.gregorian.default", defaultProleptic);
                try (Reader reader = OrcFile.createReader(new Path(file.toUri()),
                    OrcFile.readerOptions(conf).convertToProlepticGregorian(true))) {
                  assertEquals(!legacy, reader.writerUsedProlepticGregorian());
                  assertEquals(days.length * 2, reader.getNumberOfRows());
                  // DATE leaves in the top-level column, STRUCT, and LIST.
                  for (int columnId : new int[] {1, 3, 5}) {
                    DateColumnStatistics stats =
                        (DateColumnStatistics) reader.getStatistics()[columnId];
                    assertEquals(days[0].longValue(), stats.getMinimumDayOfEpoch());
                    assertEquals(Arrays.stream(days).filter(java.util.Objects::nonNull).mapToLong(Integer::longValue)
                        .max().getAsLong(), stats.getMaximumDayOfEpoch());
                    assertEquals((days.length - 1) * 2, stats.getNumberOfValues());
                  }
                  VectorizedRowBatch batch = reader.getSchema().createRowBatch();
                  int rowCount = 0;
                  try (RecordReader rows = reader.rows()) {
                    while (rows.nextBatch(batch)) {
                      LongColumnVector top = (LongColumnVector) batch.cols[0];
                      LongColumnVector nested = (LongColumnVector)
                          ((StructColumnVector) batch.cols[1]).fields[0];
                      ListColumnVector list = (ListColumnVector) batch.cols[2];
                      LongColumnVector child = (LongColumnVector) list.child;
                      for (int row = 0; row < batch.size; row++, rowCount++) {
                        Integer expected = days[rowCount % days.length];
                        assertDate(top, row, expected);
                        assertDate(nested, row, expected);
                        assertEquals(1, list.lengths[row]);
                        assertDate(child, Math.toIntExact(list.offsets[row]), expected);
                      }
                    }
                  }
                  assertEquals(days.length * 2, rowCount);
                }
              }
            } finally {
              Files.deleteIfExists(file);
            }
          }
        }
      }
    }
  }

  // Rewrite only the uncompressed footer using Apache's independently generated protobuf classes.
  private static byte[] withCalendar(byte[] file, org.apache.orc.OrcProto.CalendarKind calendar)
      throws IOException {
    return withCalendarValue(file, calendar == null ? null : calendar.getNumber());
  }

  private static byte[] withCalendarValue(byte[] file, Integer calendar) throws IOException {
    int postscriptLength = Byte.toUnsignedInt(file[file.length - 1]);
    int postscriptStart = file.length - 1 - postscriptLength;
    org.apache.orc.OrcProto.PostScript ps = org.apache.orc.OrcProto.PostScript.parseFrom(
        Arrays.copyOfRange(file, postscriptStart, file.length - 1));
    int footerStart = postscriptStart - Math.toIntExact(ps.getFooterLength());
    org.apache.orc.OrcProto.Footer.Builder footer = org.apache.orc.OrcProto.Footer.parseFrom(
        Arrays.copyOfRange(file, footerStart, postscriptStart)).toBuilder();
    footer.clearCalendar();
    ByteArrayOutputStream rewrittenFooter = new ByteArrayOutputStream();
    rewrittenFooter.write(footer.build().toByteArray());
    if (calendar != null) {
      rewrittenFooter.write(0x58); // Footer.calendar, field 11, varint.
      long value = Integer.toUnsignedLong(calendar);
      while ((value & ~0x7fL) != 0) {
        rewrittenFooter.write((int) (value & 0x7f) | 0x80);
        value >>>= 7;
      }
      rewrittenFooter.write((int) value);
    }
    byte[] footerBytes = rewrittenFooter.toByteArray();
    byte[] psBytes = ps.toBuilder().setFooterLength(footerBytes.length).build().toByteArray();
    ByteArrayOutputStream output = new ByteArrayOutputStream();
    output.write(file, 0, footerStart);
    output.write(footerBytes);
    output.write(psBytes);
    output.write(psBytes.length);
    return output.toByteArray();
  }

  private static void assertDate(LongColumnVector column, int row, Integer expected) {
    int index = column.isRepeating ? 0 : row;
    if (expected == null) {
      assertFalse(column.noNulls);
      assertTrue(column.isNull[index]);
    } else {
      assertTrue(column.noNulls || !column.isNull[index]);
      assertEquals(expected.longValue(), column.vector[index]);
    }
  }
}

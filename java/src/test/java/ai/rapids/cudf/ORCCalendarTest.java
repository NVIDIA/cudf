/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.nio.file.Files;
import java.time.LocalDate;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

import static ai.rapids.cudf.AssertUtils.assertTablesAreEqual;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ORCCalendarTest {
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
  void readerConversionIsOptIn() {
    assertFalse(ORCOptions.builder().build().convertDatesToProleptic());
    assertTrue(ORCOptions.builder().convertDatesToProleptic(true).build()
        .convertDatesToProleptic());
    assertFalse(ORCOptions.builder().convertDatesToProleptic(true)
        .convertDatesToProleptic(false).build().convertDatesToProleptic());
  }

  @Test
  void prolepticDatesRoundTrip() throws IOException {
    Integer[] days = epochDays("1001-01-01", "1200-01-01", "1582-10-04", "1582-10-10",
        "1582-10-15", "1970-01-01", "2024-05-06", null);
    checkRoundTrip(ORCWriterOptions.CalendarKind.PROLEPTIC_GREGORIAN, days, days);
  }

  @Test
  void legacyDatesRoundTrip() throws IOException {
    Integer[] days = epochDays("1001-01-01", "1200-01-01", "1582-10-04", "1582-10-15",
        "1970-01-01", "2024-05-06", null);
    // Independent hybrid epoch-day anchors: raw legacy reads must not silently rebase.
    Integer[] encoded = {-353914, -281230, -141428, -141427, 0, 19849, null};
    checkRoundTrip(ORCWriterOptions.CalendarKind.JULIAN_GREGORIAN, days, encoded);
  }

  private static Integer[] epochDays(String... values) {
    return Arrays.stream(values)
        .map(value -> value == null ? null : Math.toIntExact(LocalDate.parse(value).toEpochDay()))
        .toArray(Integer[]::new);
  }

  @SuppressWarnings("unchecked")
  private static void checkRoundTrip(ORCWriterOptions.CalendarKind calendar, Integer[] days,
      Integer[] encoded) throws IOException {
    HostColumnVector.ListType listType = new HostColumnVector.ListType(true,
        new HostColumnVector.BasicType(true, DType.TIMESTAMP_DAYS));
    java.nio.file.Path file = Files.createTempFile("orc-calendar-roundtrip", ".orc");
    try (ColumnVector dates = ColumnVector.timestampDaysFromBoxedInts(days);
         ColumnVector structs = ColumnVector.makeStruct(dates);
         ColumnVector lists = ColumnVector.fromLists(listType,
             Arrays.stream(days).map(Collections::singletonList).toArray(java.util.List[]::new));
         Table input = new Table(dates, structs, lists)) {
      ORCWriterOptions options = ORCWriterOptions.builder()
          .withColumns(true, "d")
          .withStructColumn(ColumnWriterOptions.structBuilder("s")
              .withColumns(true, "d").build())
          .withListColumn(ColumnWriterOptions.listBuilder("l")
              .withColumns(true, "d").build())
          .withCalendarKind(calendar)
          .build();
      try (TableWriter writer = Table.writeORCChunked(options, file.toFile())) {
        writer.write(input);
      }
      try (Table actual = Table.readORC(ORCOptions.builder().withNumPyTypes(false)
          .convertDatesToProleptic(true).build(), file.toFile())) {
        assertTablesAreEqual(input, actual);
        checkChunkedRead(actual, ORCOptions.builder().withNumPyTypes(false)
            .convertDatesToProleptic(true).build(), file);
      }
      try (ColumnVector rawDates = ColumnVector.timestampDaysFromBoxedInts(encoded);
           Table expected = new Table(rawDates);
           Table raw = Table.readORC(ORCOptions.builder().withNumPyTypes(false)
               .includeColumn("d").build(), file.toFile())) {
        assertTablesAreEqual(expected, raw);
        checkChunkedRead(raw, ORCOptions.builder().withNumPyTypes(false)
            .includeColumn("d").build(), file);
      }
    } finally {
      Files.deleteIfExists(file);
    }
  }

  private static void checkChunkedRead(Table expected, ORCOptions options,
      java.nio.file.Path file) throws IOException {
    byte[] bytes = Files.readAllBytes(file);
    try (HostMemoryBuffer buffer = HostMemoryBuffer.allocate(bytes.length)) {
      buffer.setBytes(0, bytes, 0, bytes.length);
      for (boolean explicitGranularity : new boolean[]{false, true}) {
        ArrayList<Table> chunks = new ArrayList<>();
        try (ORCChunkedReader reader = explicitGranularity
            ? new ORCChunkedReader(0, 0, 1, options, buffer, 0, bytes.length)
            : new ORCChunkedReader(0, 0, options, buffer, 0, bytes.length)) {
          try {
            while (reader.hasNext()) {
              Table chunk = reader.readChunk();
              if (chunk != null) {
                chunks.add(chunk);
              }
            }
            assertFalse(chunks.isEmpty());
            if (chunks.size() == 1) {
              assertTablesAreEqual(expected, chunks.get(0));
            } else {
              try (Table actual = Table.concatenate(chunks.toArray(new Table[0]))) {
                assertTablesAreEqual(expected, actual);
              }
            }
          } finally {
            chunks.forEach(Table::close);
          }
        }
      }
    }
  }
}

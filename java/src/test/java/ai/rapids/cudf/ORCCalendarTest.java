/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.junit.jupiter.api.Test;

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
      }
      try (ColumnVector rawDates = ColumnVector.timestampDaysFromBoxedInts(encoded);
           Table expected = new Table(rawDates);
           Table raw = Table.readORC(ORCOptions.builder().withNumPyTypes(false)
               .includeColumn("d").build(), file.toFile())) {
        assertTablesAreEqual(expected, raw);
      }
    } finally {
      Files.deleteIfExists(file);
    }
  }
}

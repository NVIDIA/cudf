/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

/**
 * This class represents settings for writing ORC files. It includes meta data information
 * that will be used by the ORC writer to write the file.
 */
public class ORCWriterOptions extends CompressionMetadataWriterOptions {
  /** Calendar recorded in the standard ORC file footer. */
  public enum CalendarKind {
    JULIAN_GREGORIAN(1),
    PROLEPTIC_GREGORIAN(2);

    final int nativeId;

    CalendarKind(int nativeId) {
      this.nativeId = nativeId;
    }
  }

  private int stripeSizeRows;
  private String writerTimezone;
  private CalendarKind calendarKind;

  private ORCWriterOptions(Builder builder) {
    super(builder);
    this.stripeSizeRows = builder.stripeSizeRows;
    this.writerTimezone = builder.writerTimezone;
    this.calendarKind = builder.calendarKind;
  }

  public static Builder builder() {
    return new Builder();
  }

  public int getStripeSizeRows() {
    return stripeSizeRows;
  }

  public String getWriterTimezone() {
    return writerTimezone;
  }

  public CalendarKind getCalendarKind() {
    return calendarKind;
  }

  public static class Builder extends CompressionMetadataWriterOptions.Builder
          <Builder, ORCWriterOptions> {
    // < 1M rows default orc stripe rows, defined in cudf/cpp/include/cudf/io/orc.hpp
    private int stripeSizeRows = 1000000;
    private String writerTimezone = "UTC";
    private CalendarKind calendarKind = CalendarKind.PROLEPTIC_GREGORIAN;

    public Builder withStripeSizeRows(int stripeSizeRows) {
      // maximum stripe size cannot be smaller than 512
      if (stripeSizeRows < 512) {
        throw new IllegalArgumentException("Maximum stripe size cannot be smaller than 512");
      }
      this.stripeSizeRows = stripeSizeRows;
      return this;
    }

    /**
     * Sets the timezone that the written timestamps are relative to, recorded in the stripe
     * footers. cuDF timestamps are UTC instants, so the default of "UTC" writes them unshifted.
     * Set this to the timezone that gave the values their meaning to interoperate with writers
     * that record a local timezone, such as Hive and Spark.
     *
     * <p>A non-UTC file is meant for a reader whose timezone matches; it does not round-trip
     * through {@link Table#readORC}, which has no session timezone and returns the writer's wall
     * clock.
     *
     * @param writerTimezone timezone name, for example "America/Los_Angeles"
     * @throws IllegalArgumentException if {@code writerTimezone} is null or empty
     */
    public Builder withWriterTimezone(String writerTimezone) {
      if (writerTimezone == null || writerTimezone.isEmpty()) {
        throw new IllegalArgumentException("Writer timezone cannot be null or empty");
      }
      this.writerTimezone = writerTimezone;
      return this;
    }

    /**
     * Sets the calendar used for DATE encoding and recorded in the file footer.
     * Legacy mode rebases nested and top-level DATE values. It rejects TIMESTAMP columns,
     * dates in October 5-14, 1582, and dates outside the int32 hybrid domain.
     * Use {@link ORCOptions.Builder#convertDatesToProleptic(boolean)} to read legacy output
     * back as cuDF proleptic Gregorian dates.
     *
     * @param calendarKind calendar kind
     * @throws IllegalArgumentException if the calendar is null
     */
    public Builder withCalendarKind(CalendarKind calendarKind) {
      if (calendarKind == null) {
        throw new IllegalArgumentException(
            "ORC calendar must not be null");
      }
      this.calendarKind = calendarKind;
      return this;
    }

    public ORCWriterOptions build() {
      return new ORCWriterOptions(this);
    }
  }
}

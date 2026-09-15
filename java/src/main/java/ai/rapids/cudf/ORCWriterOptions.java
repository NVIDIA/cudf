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
  private int stripeSizeRows;
  private String writerTimezone;

  private ORCWriterOptions(Builder builder) {
    super(builder);
    this.stripeSizeRows = builder.stripeSizeRows;
    this.writerTimezone = builder.writerTimezone;
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

  public static class Builder extends CompressionMetadataWriterOptions.Builder
          <Builder, ORCWriterOptions> {
    // < 1M rows default orc stripe rows, defined in cudf/cpp/include/cudf/io/orc.hpp
    private int stripeSizeRows = 1000000;
    private String writerTimezone = "UTC";

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
     * @param writerTimezone timezone name, for example "America/Los_Angeles"
     */
    public Builder withWriterTimezone(String writerTimezone) {
      if (writerTimezone == null || writerTimezone.isEmpty()) {
        throw new IllegalArgumentException("Writer timezone cannot be null or empty");
      }
      this.writerTimezone = writerTimezone;
      return this;
    }

    public ORCWriterOptions build() {
      return new ORCWriterOptions(this);
    }
  }
}

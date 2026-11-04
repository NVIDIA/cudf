/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * Options for reading a parquet file
 */
public class ParquetOptions extends ColumnFilterOptions {

  public static ParquetOptions DEFAULT = new ParquetOptions(new Builder());

  private final DType unit;
  private final boolean[] readBinaryAsString;

  private ParquetOptions(Builder builder) {
    super(builder);
    unit = builder.unit;
    readBinaryAsString = new boolean[builder.binaryAsStringColumns.size()];
    for (int i = 0 ; i < builder.binaryAsStringColumns.size() ; i++) {
      readBinaryAsString[i] = builder.binaryAsStringColumns.get(i);
    }
  }

  DType timeUnit() {
    return unit;
  }

  boolean[] getReadBinaryAsString() {
    return readBinaryAsString;
  }

  public static ParquetOptions.Builder builder() {
    return new Builder();
  }

  public static class Builder extends ColumnFilterOptions.Builder<Builder> {
    private DType unit = DType.EMPTY;
    final List<Boolean> binaryAsStringColumns = new ArrayList<>();

    /**
     * Specify the time unit to use when returning timestamps.
     * DType.EMPTY (the default), TIMESTAMP_SECONDS, TIMESTAMP_MILLISECONDS,
     * TIMESTAMP_MICROSECONDS and TIMESTAMP_NANOSECONDS are valid; for TIMESTAMP_DAYS,
     * cast the returned column instead.
     * @param unit default unit of time specified by the user
     * @return builder for chaining
     * @throws IllegalArgumentException if {@code unit} is not one of those types
     */
    public Builder withTimeUnit(DType unit) {
      if (unit == null || (unit != DType.EMPTY && !unit.hasTimeResolution())) {
        throw new IllegalArgumentException(
            "Time unit must be EMPTY or a second-based timestamp type, got " + unit);
      }
      this.unit = unit;
      return this;
    }

    /**
     * Include one or more specific columns.  Any column not included will not be read.
     * @param names the name of the column, or more than one if you want.
     */
    @Override
    public Builder includeColumn(String... names) {
      super.includeColumn(names);
      for (int i = 0 ; i < names.length ; i++) {
        binaryAsStringColumns.add(true);
      }
      return this;
    }

    /**
     * Include this column.
     * @param name the name of the column
     * @param isBinary whether this column is to be read in as binary
     */
    public Builder includeColumn(String name, boolean isBinary) {
      includeColumnNames.add(name);
      binaryAsStringColumns.add(!isBinary);
      return this;
    }

    /**
     * Include one or more specific columns.  Any column not included will not be read.
     * @param names the name of the column, or more than one if you want.
     */
    @Override
    public Builder includeColumn(Collection<String> names) {
      super.includeColumn(names);
      for (int i = 0 ; i < names.size() ; i++) {
        binaryAsStringColumns.add(true);
      }
      return this;
    }

    public ParquetOptions build() {
      return new ParquetOptions(this);
    }
  }
}

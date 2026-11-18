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
  private final boolean outputDictColumns;

  private ParquetOptions(Builder builder) {
    super(builder);
    unit = builder.unit;
    outputDictColumns = builder.outputDictColumns;
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

  /**
   * Whether flat (non-nested) STRING columns are returned as DICTIONARY32 columns.
   * @see Builder#withOutputDictColumns(boolean)
   */
  public boolean isOutputDictColumns() {
    return outputDictColumns;
  }

  public static ParquetOptions.Builder builder() {
    return new Builder();
  }

  public static class Builder extends ColumnFilterOptions.Builder<Builder> {
    private DType unit = DType.EMPTY;
    final List<Boolean> binaryAsStringColumns = new ArrayList<>();
    private boolean outputDictColumns = false;

    /**
     * Specify the time unit to use when returning timestamps.
     * @param unit default unit of time specified by the user
     * @return builder for chaining
     */
    public Builder withTimeUnit(DType unit) {
      assert unit.isTimestampType();
      this.unit = unit;
      return this;
    }

    /**
     * Specify whether flat STRING columns should be returned as dictionary columns.
     * When true, flat (non-nested) STRING columns are returned as DICTIONARY32 columns with
     * INT32 indices and STRING keys. Strings nested inside LIST or STRUCT columns are unchanged
     * and are still returned as STRING. Defaults to false. {@link HybridScanReader} does not
     * support this option and throws if it is set.
     * @param outputDictColumns true to return flat STRING columns as DICTIONARY32
     * @return builder for chaining
     */
    public Builder withOutputDictColumns(boolean outputDictColumns) {
      this.outputDictColumns = outputDictColumns;
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

/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.util.Objects;

/**
 * This class represents settings for writing Parquet files. It includes meta data information
 * that will be used by the Parquet writer to write the file
 */
public final class ParquetWriterOptions extends CompressionMetadataWriterOptions {
  private final StatisticsFrequency statsGranularity;
  private int rowGroupSizeRows;
  private long rowGroupSizeBytes;
  private long maxDictionarySize;
  private DictionaryPolicy dictionaryPolicy;

  private ParquetWriterOptions(Builder builder) {
    super(builder);
    this.rowGroupSizeRows = builder.rowGroupSizeRows;
    this.rowGroupSizeBytes = builder.rowGroupSizeBytes;
    this.maxDictionarySize = builder.maxDictionarySize;
    this.dictionaryPolicy = builder.dictionaryPolicy;
    this.statsGranularity = builder.statsGranularity;
  }

  public enum StatisticsFrequency {
    /** Do not generate statistics */
    NONE(0),

    /** Generate column statistics for each rowgroup */
    ROWGROUP(1),

    /** Generate column statistics for each page */
    PAGE(2),

    /**
     * Generate full column and offset indices (page index). Implies ROWGROUP statistics.
     * Required for page-level pruning in {@link HybridScanReader}.
     */
    COLUMN(3);

    final int nativeId;

    StatisticsFrequency(int nativeId) {
      this.nativeId = nativeId;
    }
  }

  /**
   * Mirror of cudf::io::dictionary_policy. Controls when the Parquet writer is allowed
   * to use dictionary encoding.
   */
  public enum DictionaryPolicy {
    /** Never use dictionary encoding. */
    NEVER(0),

    /**
     * Use dictionary encoding when it does not impact compression. The writer disables
     * dictionary encoding for any column chunk whose dictionary would exceed
     * {@link Builder#withMaxDictionarySize(long)}.
     */
    ADAPTIVE(1),

    /**
     * Use dictionary encoding even if it disables compression for affected columns.
     */
    ALWAYS(2);

    final int nativeId;

    DictionaryPolicy(int nativeId) {
      this.nativeId = nativeId;
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  public int getRowGroupSizeRows() {
    return rowGroupSizeRows;
  }

  public long getRowGroupSizeBytes() {
    return rowGroupSizeBytes;
  }

  /**
   * @return the maximum per-column-chunk dictionary size in bytes. Only used when the
   * dictionary policy is {@link DictionaryPolicy#ADAPTIVE}.
   */
  public long getMaxDictionarySize() {
    return maxDictionarySize;
  }

  /**
   * @return the dictionary policy in effect for the Parquet writer.
   */
  public DictionaryPolicy getDictionaryPolicy() {
    return dictionaryPolicy;
  }

  public StatisticsFrequency getStatisticsFrequency() {
    return statsGranularity;
  }

  public static class Builder extends CompressionMetadataWriterOptions.Builder
        <Builder, ParquetWriterOptions> {
    private int rowGroupSizeRows = 1000000; //Max of 1 million rows per row group
    private long rowGroupSizeBytes = 128 * 1024 * 1024; //Max of 128MB per row group
    // Matches cudf::io::default_max_dictionary_size (1 MiB).
    private long maxDictionarySize = 1024 * 1024;
    // Matches the cudf::io::parquet_writer_options default.
    private DictionaryPolicy dictionaryPolicy = DictionaryPolicy.ADAPTIVE;
    private StatisticsFrequency statsGranularity = StatisticsFrequency.ROWGROUP;

    public Builder() {
      super();
    }

    public Builder withRowGroupSizeRows(int rowGroupSizeRows) {
      this.rowGroupSizeRows = rowGroupSizeRows;
      return this;
    }

    public Builder withRowGroupSizeBytes(long rowGroupSizeBytes) {
      this.rowGroupSizeBytes = rowGroupSizeBytes;
      return this;
    }

    /**
     * Sets the maximum dictionary size, in bytes. Only used when the dictionary policy is
     * {@link DictionaryPolicy#ADAPTIVE}. Must be in [0, Integer.MAX_VALUE].
     */
    public Builder withMaxDictionarySize(long maxDictionarySize) {
      if (maxDictionarySize < 0 || maxDictionarySize > Integer.MAX_VALUE) {
        throw new IllegalArgumentException(
            "maxDictionarySize must be in [0, " + Integer.MAX_VALUE + "], got " + maxDictionarySize);
      }
      this.maxDictionarySize = maxDictionarySize;
      return this;
    }

    /**
     * Sets the dictionary policy. Must not be null.
     */
    public Builder withDictionaryPolicy(DictionaryPolicy dictionaryPolicy) {
      this.dictionaryPolicy = Objects.requireNonNull(dictionaryPolicy, "dictionaryPolicy");
      return this;
    }

    public Builder withStatisticsFrequency(StatisticsFrequency statsGranularity) {
      this.statsGranularity = statsGranularity;
      return this;
    }

    public ParquetWriterOptions build() {
      return new ParquetWriterOptions(this);
    }
  }
}

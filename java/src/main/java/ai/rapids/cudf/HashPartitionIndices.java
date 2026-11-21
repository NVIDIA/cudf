/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * The result of {@link Table.TableOperation#hashPartitionIndices}: an INT32 gather map whose
 * rows are grouped by partition, and the offsets of each partition within the gather map.
 */
public final class HashPartitionIndices implements AutoCloseable {
  private final ColumnVector gatherMap;
  private final int[] partitionOffsets;

  HashPartitionIndices(ColumnVector gatherMap, int[] partitionOffsets) {
    this.gatherMap = gatherMap;
    this.partitionOffsets = partitionOffsets;
  }

  /** The INT32 gather map of row indices into the input table, grouped by partition. */
  public ColumnVector getGatherMap() {
    return gatherMap;
  }

  /**
   * The numberOfPartitions + 1 offsets into the gather map. Partition 'i' spans
   * [partitionOffsets[i], partitionOffsets[i + 1]).
   */
  public int[] getPartitionOffsets() {
    return partitionOffsets;
  }

  @Override
  public void close() {
    gatherMap.close();
  }
}

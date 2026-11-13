/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Utility methods for breaking apart and reassembling 128-bit values during aggregations
 * to enable hash-based aggregations and detect overflows.
 */
public class Aggregation128Utils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * Extract a 32-bit chunk from a 128-bit value.
   * @param col column of 128-bit values (e.g.: DECIMAL128)
   * @param outType integer type to use for the output column (e.g.: UINT32 or INT32)
   * @param chunkIdx index of the 32-bit chunk to extract where 0 is the least significant chunk
   *                 and 3 is the most significant chunk
   * @return column containing the specified 32-bit chunk of the input column values. A null input
   *                row will result in a corresponding null output row.
   */
  public static ColumnVector extractInt32Chunk(ColumnView col, DType outType, int chunkIdx) {
    return new ColumnVector(extractInt32Chunk(col.getNativeView(),
        outType.getTypeId().getNativeId(), chunkIdx));
  }

  /**
   * Reassemble a column of 128-bit values from a table of four 64-bit integer columns and check
   * for overflow. The 128-bit value is reconstructed by overlapping the 64-bit values by 32-bits.
   * The least significant 32-bits of the least significant 64-bit value are used directly as the
   * least significant 32-bits of the final 128-bit value, and the remaining 32-bits are added to
   * the next most significant 64-bit value. The lower 32-bits of that sum become the next most
   * significant 32-bits in the final 128-bit value, and the remaining 32-bits are added to the
   * next most significant 64-bit input value, and so on.
   *
   * @param chunks table of four 64-bit integer columns with the columns ordered from least
   *               significant to most significant. The last column must be of type INT64.
   * @param type the type to use for the resulting 128-bit value column
   * @return table containing a boolean column and a 128-bit value column of the requested type.
   *         The boolean value will be true if an overflow was detected for that row's value when
   *         it was reassembled. A null input row will result in a corresponding null output row.
   */
  public static Table combineInt64SumChunks(Table chunks, DType type) {
    return Table.fromRowCountAndColumns(combineInt64SumChunks(chunks.getNativeView(),
        type.getTypeId().getNativeId(),
        type.getScale()));
  }

  private static native long extractInt32Chunk(long columnView, int outTypeId, int chunkIdx);

  private static native long[] combineInt64SumChunks(long chunksTableView, int dtype, int scale);
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.nvcomp;

/** Multi-buffer LZ4 compressor */
public class BatchedLZ4Compressor extends BatchedCompressor {

  /**
   * Construct a batched LZ4 compressor instance
   * @param chunkSize maximum amount of uncompressed data to compress as a single chunk.
   *                  Inputs larger than this will be compressed in multiple chunks.
   * @param maxIntermediateBufferSize desired maximum size of intermediate device buffers
   *                                  used during compression.
   */
  public BatchedLZ4Compressor(long chunkSize, long maxIntermediateBufferSize) {
    super(chunkSize, NvcompJni.batchedLZ4CompressGetMaxOutputChunkSize(chunkSize),
        maxIntermediateBufferSize);
  }

  @Override
  protected long batchedCompressGetTempSize(long batchSize, long maxChunkSize, long totalSize,
      long stream) {
    return NvcompJni.batchedLZ4CompressGetTempSize(batchSize, maxChunkSize, totalSize, stream);
  }

  @Override
  protected void batchedCompressAsync(long devInPtrs, long devInSizes, long chunkSize,
      long batchSize, long tempPtr, long tempSize, long devOutPtrs,
      long compressedSizesOutPtr, long stream) {
    NvcompJni.batchedLZ4CompressAsync(devInPtrs, devInSizes, chunkSize, batchSize,
        tempPtr, tempSize, devOutPtrs, compressedSizesOutPtr, stream);
  }
}

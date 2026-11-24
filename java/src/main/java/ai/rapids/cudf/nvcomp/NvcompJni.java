/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.nvcomp;

import ai.rapids.cudf.NativeDepsLoader;

/** Raw JNI interface to the nvcomp library. */
class NvcompJni {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  // For lz4
  /**
   * Get the temporary workspace size required to perform compression of entire LZ4 batch.
   * @param batchSize number of chunks in the batch
   * @param maxChunkSize maximum size of an uncompressed chunk in bytes
   * @param maxTotalSize Upper bound on the total uncompressed size of all chunks
   * @param stream CUDA stream to use
   * @return The size of required temporary workspace in bytes to compress the batch.
   */
  static native long batchedLZ4CompressGetTempSize(long batchSize, long maxChunkSize, long maxTotalSize,
      long stream);

  /**
   * Get the maximum size any chunk could compress to in a LZ4 batch. This is the minimum amount of
   * output memory to allocate per chunk when batch compressing.
   * @param maxChunkSize maximum size of an uncompressed chunk size in bytes
   * @return maximum compressed output size of a chunk
   */
  static native long batchedLZ4CompressGetMaxOutputChunkSize(long maxChunkSize);

  /**
   * Asynchronously compress a batch of buffers with LZ4. Note that
   * compressedSizesOutPtr must point to pinned memory for this operation
   * to be asynchronous.
   * @param devInPtrs device address of uncompressed buffer addresses vector
   * @param devInSizes device address of uncompressed buffer sizes vector
   * @param chunkSize maximum size of an uncompressed chunk in bytes
   * @param batchSize number of chunks in the batch
   * @param tempPtr device address of the temporary workspace buffer
   * @param tempSize size of the temporary workspace buffer in bytes
   * @param devOutPtrs device address of output buffer addresses vector
   * @param compressedSizesOutPtr device address where to write the sizes of the
   *                              compressed data written to the corresponding
   *                              output buffers. Must point to a buffer with
   *                              at least 8 bytes of memory per output buffer
   *                              in the batch.
   * @param stream CUDA stream to use
   */
  static native void batchedLZ4CompressAsync(
      long devInPtrs,
      long devInSizes,
      long chunkSize,
      long batchSize,
      long tempPtr,
      long tempSize,
      long devOutPtrs,
      long compressedSizesOutPtr,
      long stream);

  /**
   * Computes the temporary storage size in bytes needed to decompress a LZ4-compressed batch.
   * @param numChunks number of chunks in the batch
   * @param maxUncompressedChunkBytes maximum uncompressed size of any chunk in bytes
   * @param maxTotalSize Upper bound on the total uncompressed size of all chunks
   * @param stream CUDA stream to use
   * @return number of temporary storage bytes needed to decompress the batch
   */
  static native long batchedLZ4DecompressGetTempSize(
      long numChunks,
      long maxUncompressedChunkBytes,
      long maxTotalSize,
      long stream);

  /**
   * Asynchronously decompress a batch of LZ4-compressed data buffers.
   * @param devInPtrs device address of compressed input buffer addresses vector
   * @param devInSizes device address of compressed input buffer sizes vector
   * @param devOutSizes device address of uncompressed buffer sizes vector
   * @param batchSize number of buffers in the batch
   * @param tempPtr device address of the temporary decompression space
   * @param tempSize size of the temporary decompression space in bytes
   * @param devOutPtrs device address of uncompressed output buffer addresses vector
   * @param stream CUDA stream to use
   */
  static native void batchedLZ4DecompressAsync(
      long devInPtrs,
      long devInSizes,
      long devOutSizes,
      long batchSize,
      long tempPtr,
      long tempSize,
      long devOutPtrs,
      long stream);

  /**
   * Asynchronously calculates the decompressed size needed for each chunk.
   * @param devInPtrs device address of compressed input buffer addresses vector
   * @param devInSizes device address of compressed input buffer sizes vector
   * @param devOutSizes device address of calculated decompress sizes vector
   * @param batchSize number of buffers in the batch
   * @param stream CUDA stream to use
   */
  static native void batchedLZ4GetDecompressSizeAsync(
      long devInPtrs,
      long devInSizes,
      long devOutSizes,
      long batchSize,
      long stream);

  // For zstd
  /**
   * Get the temporary workspace size required to perform compression of entire zstd batch.
   * @param batchSize number of chunks in the batch
   * @param maxChunkSize maximum size of an uncompressed chunk in bytes
   * @param maxTotalSize Upper bound on the total uncompressed size of all chunks
   * @param stream CUDA stream to use
   * @return The size of required temporary workspace in bytes to compress the batch.
   */
  static native long batchedZstdCompressGetTempSize(long batchSize, long maxChunkSize, long maxTotalSize,
      long stream);

  /**
   * Get the maximum size any chunk could compress to in a ZSTD batch. This is the minimum
   * amount of output memory to allocate per chunk when batch compressing.
   * @param maxChunkSize maximum size of an uncompressed chunk size in bytes
   * @return maximum compressed output size of a chunk
   */
  static native long batchedZstdCompressGetMaxOutputChunkSize(long maxChunkSize);

  /**
   * Asynchronously compress a batch of buffers with ZSTD. Note that
   * compressedSizesOutPtr must point to pinned memory for this operation
   * to be asynchronous.
   * @param devInPtrs device address of uncompressed buffer addresses vector
   * @param devInSizes device address of uncompressed buffer sizes vector
   * @param chunkSize maximum size of an uncompressed chunk in bytes
   * @param batchSize number of chunks in the batch
   * @param tempPtr device address of the temporary workspace buffer
   * @param tempSize size of the temporary workspace buffer in bytes
   * @param devOutPtrs device address of output buffer addresses vector
   * @param compressedSizesOutPtr device address where to write the sizes of the
   *                              compressed data written to the corresponding
   *                              output buffers. Must point to a buffer with
   *                              at least 8 bytes of memory per output buffer
   *                              in the batch.
   * @param stream CUDA stream to use
   */
  static native void batchedZstdCompressAsync(
      long devInPtrs,
      long devInSizes,
      long chunkSize,
      long batchSize,
      long tempPtr,
      long tempSize,
      long devOutPtrs,
      long compressedSizesOutPtr,
      long stream);

  /**
   * Computes the temporary storage size in bytes needed to decompress a
   * ZSTD-compressed batch.
   * @param numChunks number of chunks in the batch
   * @param maxUncompressedChunkBytes maximum uncompressed size of any chunk in bytes
   * @param maxTotalSize Upper bound on the total uncompressed size of all chunks
   * @param stream CUDA stream to use
   * @return number of temporary storage bytes needed to decompress the batch
   */
  static native long batchedZstdDecompressGetTempSize(
      long numChunks,
      long maxUncompressedChunkBytes,
      long maxTotalSize,
      long stream);

  /**
   * Asynchronously decompress a batch of ZSTD-compressed data buffers.
   * @param devInPtrs device address of compressed input buffer addresses vector
   * @param devInSizes device address of compressed input buffer sizes vector
   * @param devOutSizes device address of uncompressed buffer sizes vector
   * @param batchSize number of buffers in the batch
   * @param tempPtr device address of the temporary decompression space
   * @param tempSize size of the temporary decompression space in bytes
   * @param devOutPtrs device address of uncompressed output buffer addresses vector
   * @param stream CUDA stream to use
   */
  static native void batchedZstdDecompressAsync(
      long devInPtrs,
      long devInSizes,
      long devOutSizes,
      long batchSize,
      long tempPtr,
      long tempSize,
      long devOutPtrs,
      long stream);

  /**
   * Asynchronously calculates the decompressed size needed for each chunk.
   * @param devInPtrs device address of compressed input buffer addresses vector
   * @param devInSizes device address of compressed input buffer sizes vector
   * @param devOutSizes device address of calculated decompress sizes vector
   * @param batchSize number of buffers in the batch
   * @param stream CUDA stream to use
   */
  static native void batchedZstdGetDecompressSizeAsync(
      long devInPtrs,
      long devInSizes,
      long devOutSizes,
      long batchSize,
      long stream);
}

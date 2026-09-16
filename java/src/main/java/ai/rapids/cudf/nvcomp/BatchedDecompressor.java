/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.nvcomp;

import ai.rapids.cudf.CloseableArray;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.NvtxColor;
import ai.rapids.cudf.NvtxRange;

import java.util.Arrays;

/** Decompressor that operates on multiple input buffers in a batch */
public abstract class BatchedDecompressor {

  private final long chunkSize;

  /**
   * Construct a batched decompressor instance
   * @param chunkSize maximum uncompressed block size, must match value used
   *                  during compression
   */
  public BatchedDecompressor(long chunkSize) {
    this.chunkSize = chunkSize;
  }

  /**
   * Asynchronously decompress a batch of buffers
   * @param origInputs buffers to decompress, will be closed by this operation
   * @param outputs output buffers that will contain the decompressed results, each must
   *                be sized to the exact decompressed size of the corresponding input
   * @param stream CUDA stream to use
   */
  public void decompressAsync(BaseDeviceMemoryBuffer[] origInputs,
      BaseDeviceMemoryBuffer[] outputs, Cuda.Stream stream) {
    try (CloseableArray<BaseDeviceMemoryBuffer> inputs =
            CloseableArray.wrap(Arrays.copyOf(origInputs, origInputs.length))) {
      BatchedCompressor.validateChunkSize(chunkSize);
      if (origInputs.length != outputs.length) {
        throw new IllegalArgumentException("number of inputs must match number of outputs");
      }
      final int numInputs = inputs.size();
      if (numInputs == 0) {
        return;
      }

      int[] chunksPerInput = new int[numInputs];
      long totalChunks = 0;
      for (int i = 0; i < numInputs; i++) {
        // use output size to determine number of chunks in the input, as the output buffer
        // must be exactly sized to the uncompressed data
        BaseDeviceMemoryBuffer buffer = outputs[i];
        int numBufferChunks = getNumChunksInBuffer(chunkSize, buffer);
        chunksPerInput[i] = numBufferChunks;
        totalChunks += numBufferChunks;
      }

      final long tempBufferSize = batchedDecompressGetTempSize(totalChunks, chunkSize,
        totalChunks * chunkSize);
      try (DeviceMemoryBuffer devAddrsSizes = buildAddrsSizesBuffer(chunkSize, totalChunks,
              inputs.getArray(), chunksPerInput, outputs, stream);
           DeviceMemoryBuffer devTemp = DeviceMemoryBuffer.allocate(tempBufferSize, stream)) {
        // buffer containing addresses and sizes contains four vectors of longs in this order:
        // - compressed chunk input addresses
        // - chunk output buffer addresses
        // - compressed chunk sizes
        // - uncompressed chunk sizes
        final long inputAddrsPtr = devAddrsSizes.getAddress();
        final long outputAddrsPtr = inputAddrsPtr + totalChunks * 8;
        final long inputSizesPtr = outputAddrsPtr + totalChunks * 8;
        final long outputSizesPtr = inputSizesPtr + totalChunks * 8;
        batchedDecompressAsync(inputAddrsPtr, inputSizesPtr, outputSizesPtr, totalChunks,
            devTemp.getAddress(), devTemp.getLength(), outputAddrsPtr, stream.getStream());
      }
    }
  }

  private static int getNumChunksInBuffer(long chunkSize, BaseDeviceMemoryBuffer buffer) {
    return (int) ((buffer.getLength() + chunkSize - 1) / chunkSize);
  }

  /**
   * Build a device memory buffer containing four vectors of longs in the following order:
   * <ul>
   *   <li>compressed chunk input addresses</li>
   *   <li>uncompressed chunk output addresses</li>
   *   <li>compressed chunk sizes</li>
   *   <li>uncompressed chunk sizes</li>
   * </ul>
   * Each vector contains as many longs as the number of chunks being decompressed
   * @param chunkSize maximum uncompressed size of a chunk
   * @param totalChunks total number of chunks to be decompressed
   * @param inputs device buffers containing the compressed data
   * @param chunksPerInput number of compressed chunks per input buffer
   * @param outputs device buffers that will hold the uncompressed output
   * @param stream CUDA stream to use
   * @return device buffer containing address and size vectors
   */
  private static DeviceMemoryBuffer buildAddrsSizesBuffer(long chunkSize, long totalChunks,
      BaseDeviceMemoryBuffer[] inputs, int[] chunksPerInput, BaseDeviceMemoryBuffer[] outputs,
      Cuda.Stream stream) {
    final long totalBufferSize = totalChunks * 8L * 4L;
    try (NvtxRange range = new NvtxRange("buildAddrSizesBuffer", NvtxColor.YELLOW)) {
      try (HostMemoryBuffer metadata = fetchMetadata(totalChunks, inputs, chunksPerInput, stream);
           HostMemoryBuffer hostAddrsSizes = HostMemoryBuffer.allocate(totalBufferSize);
           DeviceMemoryBuffer devAddrsSizes =
               DeviceMemoryBuffer.allocate(totalBufferSize, stream)) {
        // Build four long vectors in the AddrsSizes buffer:
        // - compressed input address (one per chunk)
        // - uncompressed output address (one per chunk)
        // - compressed input size (one per chunk)
        // - uncompressed input size (one per chunk)
        final long srcAddrsOffset = 0;
        final long destAddrsOffset = srcAddrsOffset + totalChunks * 8L;
        final long srcSizesOffset = destAddrsOffset + totalChunks * 8L;
        final long destSizesOffset = srcSizesOffset + totalChunks * 8L;
        long chunkIdx = 0;
        for (int inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
          final BaseDeviceMemoryBuffer input = inputs[inputIdx];
          final BaseDeviceMemoryBuffer output = outputs[inputIdx];
          final int numChunksInInput = chunksPerInput[inputIdx];
          long srcAddr = input.getAddress() +
              BatchedCompressor.METADATA_BYTES_PER_CHUNK * numChunksInInput;
          long destAddr = output.getAddress();
          final long chunkIdxEnd = chunkIdx + numChunksInInput;
          while (chunkIdx < chunkIdxEnd) {
            final long srcChunkSize = metadata.getLong(chunkIdx * 8);
            final long destChunkSize = (chunkIdx < chunkIdxEnd - 1) ? chunkSize
                : output.getAddress() + output.getLength() - destAddr;
            hostAddrsSizes.setLong(srcAddrsOffset + chunkIdx * 8, srcAddr);
            hostAddrsSizes.setLong(destAddrsOffset + chunkIdx * 8, destAddr);
            hostAddrsSizes.setLong(srcSizesOffset + chunkIdx * 8, srcChunkSize);
            hostAddrsSizes.setLong(destSizesOffset + chunkIdx * 8, destChunkSize);
            srcAddr += srcChunkSize;
            destAddr += destChunkSize;
            ++chunkIdx;
          }
        }
        devAddrsSizes.copyFromHostBuffer(hostAddrsSizes, stream);
        devAddrsSizes.incRefCount();
        return devAddrsSizes;
      }
    }
  }

  /**
   * Fetch the metadata at the front of each input in a single, contiguous host buffer.
   * @param totalChunks total number of compressed chunks
   * @param inputs buffers containing the compressed data
   * @param chunksPerInput number of compressed chunks for the corresponding input
   * @param stream CUDA stream to use
   * @return host buffer containing all of the metadata
   */
  private static HostMemoryBuffer fetchMetadata(long totalChunks, BaseDeviceMemoryBuffer[] inputs,
      int[] chunksPerInput, Cuda.Stream stream) {
    try (NvtxRange range = new NvtxRange("fetchMetadata", NvtxColor.PURPLE)) {
      // one long per chunk containing the compressed size
      final long totalMetadataSize = totalChunks * BatchedCompressor.METADATA_BYTES_PER_CHUNK;
      // Build corresponding vectors of destination addresses, source addresses and sizes.
      long[] destAddrs = new long[inputs.length];
      long[] srcAddrs = new long[inputs.length];
      long[] sizes = new long[inputs.length];
      try (HostMemoryBuffer hostMetadata = HostMemoryBuffer.allocate(totalMetadataSize);
           DeviceMemoryBuffer devMetadata =
               DeviceMemoryBuffer.allocate(totalMetadataSize, stream)) {
        long destCopyAddr = devMetadata.getAddress();
        for (int inputIdx = 0; inputIdx < inputs.length; inputIdx++) {
          final BaseDeviceMemoryBuffer input = inputs[inputIdx];
          final long copySize =
              chunksPerInput[inputIdx] * BatchedCompressor.METADATA_BYTES_PER_CHUNK;
          destAddrs[inputIdx] = destCopyAddr;
          srcAddrs[inputIdx] = input.getAddress();
          sizes[inputIdx] = copySize;
          destCopyAddr += copySize;
        }
        Cuda.multiBufferCopyAsync(destAddrs, srcAddrs, sizes, stream);
        hostMetadata.copyFromDeviceBuffer(devMetadata, stream);
        hostMetadata.incRefCount();
        return hostMetadata;
      }
    }
  }

  /**
   * Computes the temporary storage size in bytes needed to decompress a compressed batch.
   * @param numChunks number of chunks in the batch
   * @param maxUncompressedChunkBytes maximum uncompressed size of any chunk in bytes
   * @param maxTotalSize Upper bound on the total uncompressed size of all chunks
   * @return number of temporary storage bytes needed to decompress the batch
   */
  protected abstract long batchedDecompressGetTempSize(long numChunks,
      long maxUncompressedChunkBytes, long maxTotalSize);

    /**
   * Asynchronously decompress a batch of compressed data buffers.
   * @param devInPtrs device address of compressed input buffer addresses vector
   * @param devInSizes device address of compressed input buffer sizes vector
   * @param devOutSizes device address of uncompressed buffer sizes vector
   * @param batchSize number of buffers in the batch
   * @param tempPtr device address of the temporary decompression space
   * @param tempSize size of the temporary decompression space in bytes
   * @param devOutPtrs device address of uncompressed output buffer addresses vector
   * @param stream CUDA stream to use
   */
  protected abstract void batchedDecompressAsync(long devInPtrs, long devInSizes,
      long devOutSizes, long batchSize, long tempPtr, long tempSize, long devOutPtrs,
      long stream);
}

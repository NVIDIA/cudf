/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.nvcomp;

import ai.rapids.cudf.BaseDeviceMemoryBuffer;
import ai.rapids.cudf.CloseableArray;
import ai.rapids.cudf.Cuda;
import ai.rapids.cudf.DeviceMemoryBuffer;
import ai.rapids.cudf.HostMemoryBuffer;
import ai.rapids.cudf.MemoryBuffer;
import ai.rapids.cudf.NvtxColor;
import ai.rapids.cudf.NvtxRange;

/** Multi-buffer compressor */
public abstract class BatchedCompressor {

  static final long MAX_CHUNK_SIZE = 16777216;  // 16MiB in bytes
  // each chunk has a 64-bit integer value as metadata containing the compressed size
  static final long METADATA_BYTES_PER_CHUNK = 8;

  private final long chunkSize;
  private final long maxIntermediateBufferSize;
  private final long maxOutputChunkSize;

  /**
   * Construct a batched compressor instance
   * @param chunkSize maximum amount of uncompressed data to compress as a single chunk.
   *                  Inputs larger than this will be compressed in multiple chunks.
   * @param maxIntermediateBufferSize desired maximum size of intermediate device
   *                                  buffers used during compression.
   */
  public BatchedCompressor(long chunkSize, long maxOutputChunkSize,
      long maxIntermediateBufferSize) {
    validateChunkSize(chunkSize);
    assert maxOutputChunkSize < Integer.MAX_VALUE;
    this.chunkSize = chunkSize;
    this.maxOutputChunkSize = maxOutputChunkSize;
    this.maxIntermediateBufferSize = Math.max(maxOutputChunkSize, maxIntermediateBufferSize);
  }

  /**
   * Compress a batch of buffers. The input buffers will be closed.
   * @param origInputs buffers to compress
   * @param stream CUDA stream to use
   * @return compressed buffers corresponding to the input buffers
   */
  public DeviceMemoryBuffer[] compress(BaseDeviceMemoryBuffer[] origInputs, Cuda.Stream stream) {
    try (CloseableArray<BaseDeviceMemoryBuffer> inputs = CloseableArray.wrap(origInputs)) {
      if (chunkSize <= 0) {
        throw new IllegalArgumentException("Illegal chunk size: " + chunkSize);
      }
      final int numInputs = inputs.size();
      if (numInputs == 0) {
        return new DeviceMemoryBuffer[0];
      }

      // Each buffer is broken up into chunkSize chunks for compression.  Calculate how many
      // chunks are needed for each input buffer.
      int[] chunksPerInput = new int[numInputs];
      int numChunks = 0;
      for (int i = 0; i < numInputs; i++) {
        BaseDeviceMemoryBuffer buffer = inputs.get(i);
        int numBufferChunks = getNumChunksInBuffer(buffer);
        chunksPerInput[i] = numBufferChunks;
        numChunks += numBufferChunks;
      }

      // Allocate buffers for each chunk and generate parallel lists of chunk source addresses,
      // chunk destination addresses, and sizes.
      try (CloseableArray<DeviceMemoryBuffer> compressedBuffers =
              allocCompressedBuffers(numChunks, stream);
           DeviceMemoryBuffer compressedChunkSizes =
              DeviceMemoryBuffer.allocate(numChunks * 8L, stream)) {
        long[] inputChunkAddrs = new long[numChunks];
        long[] inputChunkSizes = new long[numChunks];
        long[] outputChunkAddrs = new long[numChunks];
        buildAddrsAndSizes(inputs, inputChunkAddrs, inputChunkSizes, compressedBuffers,
            outputChunkAddrs);

        final long tempBufferSize = batchedCompressGetTempSize(numChunks, chunkSize,
          numChunks * chunkSize);
        try (DeviceMemoryBuffer addrsAndSizes = putAddrsAndSizesOnDevice(inputChunkAddrs,
                inputChunkSizes, outputChunkAddrs, stream);
             DeviceMemoryBuffer tempBuffer =
                DeviceMemoryBuffer.allocate(tempBufferSize, stream)) {
          final long devOutputAddrsPtr = addrsAndSizes.getAddress() + numChunks * 8L;
          final long devInputSizesPtr = devOutputAddrsPtr + numChunks * 8L;
          batchedCompressAsync(addrsAndSizes.getAddress(), devInputSizesPtr, chunkSize,
              numChunks, tempBuffer.getAddress(), tempBufferSize, devOutputAddrsPtr,
              compressedChunkSizes.getAddress(), stream.getStream());
        }

        // Synchronously copy the resulting compressed sizes per chunk.
        long[] outputChunkSizes = getOutputChunkSizes(compressedChunkSizes, stream);

        // inputs are no longer needed at this point, so free them early
        inputs.close();

        // Combine compressed chunks into output buffers corresponding to each original input
        return stitchOutput(chunksPerInput, compressedChunkSizes, outputChunkAddrs,
            outputChunkSizes, stream);
      }
    }
  }

  static void validateChunkSize(long chunkSize) {
    if (chunkSize <= 0  || chunkSize > MAX_CHUNK_SIZE) {
      throw new IllegalArgumentException("Invalid chunk size: " + chunkSize +
          " Max chunk size is: " + MAX_CHUNK_SIZE + " bytes");
    }
  }

  private static long ceilingDivide(long x, long y) {
    return (x + y - 1) / y;
  }

  private int getNumChunksInBuffer(MemoryBuffer buffer) {
    return (int) ceilingDivide(buffer.getLength(), chunkSize);
  }

  private CloseableArray<DeviceMemoryBuffer> allocCompressedBuffers(long numChunks,
      Cuda.Stream stream) {
    final long chunksPerBuffer = maxIntermediateBufferSize / maxOutputChunkSize;
    final long numBuffers = ceilingDivide(numChunks, chunksPerBuffer);
    if (numBuffers > Integer.MAX_VALUE) {
      throw new IllegalStateException("Too many chunks");
    }
    try (NvtxRange range = new NvtxRange("allocCompressedBuffers", NvtxColor.YELLOW)) {
      CloseableArray<DeviceMemoryBuffer> buffers = CloseableArray.wrap(
          new DeviceMemoryBuffer[(int) numBuffers]);
      try {
        // allocate all of the max-chunks intermediate compressed buffers
        for (int i = 0; i < buffers.size() - 1; ++i) {
          buffers.set(i,
              DeviceMemoryBuffer.allocate(chunksPerBuffer * maxOutputChunkSize, stream));
        }
        // allocate the tail intermediate compressed buffer that may be smaller than the others
        buffers.set(buffers.size() - 1, DeviceMemoryBuffer.allocate(
            (numChunks - chunksPerBuffer * (buffers.size() - 1)) * maxOutputChunkSize, stream));
        return buffers;
      } catch (Exception e) {
        buffers.close(e);
        throw e;
      }
    }
  }

  // Fill in the inputChunkAddrs, inputChunkSizes, and outputChunkAddrs arrays to point
  // into the chunks in the input and output buffers.
  private void buildAddrsAndSizes(CloseableArray<BaseDeviceMemoryBuffer> inputs,
      long[] inputChunkAddrs, long[] inputChunkSizes,
      CloseableArray<DeviceMemoryBuffer> compressedBuffers, long[] outputChunkAddrs) {
    // setup the input addresses and sizes
    int chunkIdx = 0;
    for (BaseDeviceMemoryBuffer input : inputs.getArray()) {
      final int numChunksInBuffer = getNumChunksInBuffer(input);
      for (int i = 0; i < numChunksInBuffer; i++) {
        inputChunkAddrs[chunkIdx] = input.getAddress() + i * chunkSize;
        inputChunkSizes[chunkIdx] = (i != numChunksInBuffer - 1) ? chunkSize
            : (input.getLength() - (long) i * chunkSize);
        ++chunkIdx;
      }
    }
    assert chunkIdx == inputChunkAddrs.length;
    assert chunkIdx == inputChunkSizes.length;

    // setup output addresses
    chunkIdx = 0;
    for (DeviceMemoryBuffer buffer : compressedBuffers.getArray()) {
      assert buffer.getLength() % maxOutputChunkSize == 0;
      long numChunksInBuffer = buffer.getLength() / maxOutputChunkSize;
      long baseAddr = buffer.getAddress();
      for (int i = 0; i < numChunksInBuffer; i++) {
        outputChunkAddrs[chunkIdx++] = baseAddr + i * maxOutputChunkSize;
      }
    }
    assert chunkIdx == outputChunkAddrs.length;
  }

  // Write input addresses, output addresses and sizes contiguously into a DeviceMemoryBuffer.
  private DeviceMemoryBuffer putAddrsAndSizesOnDevice(long[] inputAddrs, long[] inputSizes,
        long[] outputAddrs, Cuda.Stream stream) {
    final long totalSize = inputAddrs.length * 8L * 3; // space for input, output, and size arrays
    final long outputAddrsOffset = inputAddrs.length * 8L;
    final long sizesOffset = outputAddrsOffset + inputAddrs.length * 8L;
    try (NvtxRange range = new NvtxRange("putAddrsAndSizesOnDevice", NvtxColor.YELLOW)) {
      try (HostMemoryBuffer hostbuf = HostMemoryBuffer.allocate(totalSize);
           DeviceMemoryBuffer result = DeviceMemoryBuffer.allocate(totalSize, stream)) {
        hostbuf.setLongs(0, inputAddrs, 0, inputAddrs.length);
        hostbuf.setLongs(outputAddrsOffset, outputAddrs, 0, outputAddrs.length);
        for (int i = 0; i < inputSizes.length; i++) {
          hostbuf.setLong(sizesOffset + i * 8L, inputSizes[i]);
        }
        result.copyFromHostBuffer(hostbuf, stream);
        result.incRefCount();
        return result;
      }
    }
  }

  // Synchronously copy the resulting compressed sizes from device memory to host memory.
  private long[] getOutputChunkSizes(BaseDeviceMemoryBuffer devChunkSizes, Cuda.Stream stream) {
    try (NvtxRange range = new NvtxRange("getOutputChunkSizes", NvtxColor.YELLOW)) {
      try (HostMemoryBuffer hostbuf = HostMemoryBuffer.allocate(devChunkSizes.getLength())) {
        hostbuf.copyFromDeviceBuffer(devChunkSizes, stream);
        int numChunks = (int) (devChunkSizes.getLength() / 8);
        long[] result = new long[numChunks];
        for (int i = 0; i < numChunks; i++) {
          long size = hostbuf.getLong(i * 8L);
          assert size < Integer.MAX_VALUE : "output size is too big";
          result[i] = size;
        }
        return result;
      }
    }
  }

  // Stitch together the individual chunks into the result buffers.
  // Each result buffer has metadata at the beginning, followed by compressed chunks.
  // This is done by building up parallel lists of source addr, dest addr and size and
  // then calling multiBufferCopyAsync()
  private DeviceMemoryBuffer[] stitchOutput(int[] chunksPerInput,
        DeviceMemoryBuffer compressedChunkSizes, long[] outputChunkAddrs,
        long[] outputChunkSizes, Cuda.Stream stream) {
    try (NvtxRange range = new NvtxRange("stitchOutput", NvtxColor.YELLOW)) {
      final int numOutputs = chunksPerInput.length;
      final long chunkSizesAddr = compressedChunkSizes.getAddress();
      long[] outputBufferSizes = calcOutputBufferSizes(chunksPerInput, outputChunkSizes);
      try (CloseableArray<DeviceMemoryBuffer> outputs =
              CloseableArray.wrap(new DeviceMemoryBuffer[numOutputs])) {
        // Each chunk needs to be copied, and each output needs a copy of the
        // compressed chunk size vector representing the metadata.
        final int totalBuffersToCopy = numOutputs + outputChunkAddrs.length;
        long[] destAddrs = new long[totalBuffersToCopy];
        long[] srcAddrs = new long[totalBuffersToCopy];
        long[] sizes = new long[totalBuffersToCopy];
        int copyBufferIdx = 0;
        int chunkIdx = 0;
        for (int outputIdx = 0; outputIdx < numOutputs; outputIdx++) {
          DeviceMemoryBuffer outputBuffer =
              DeviceMemoryBuffer.allocate(outputBufferSizes[outputIdx], stream);
          outputs.set(outputIdx, outputBuffer);
          final long outputBufferAddr = outputBuffer.getAddress();
          final long numChunks = chunksPerInput[outputIdx];
          final long metadataSize = numChunks * METADATA_BYTES_PER_CHUNK;

          // setup a copy of the metadata at the front of the output buffer
          srcAddrs[copyBufferIdx] = chunkSizesAddr + chunkIdx * 8;
          destAddrs[copyBufferIdx] = outputBufferAddr;
          sizes[copyBufferIdx] = metadataSize;
          ++copyBufferIdx;

          // setup copies of the compressed chunks for this output buffer
          long nextChunkAddr = outputBufferAddr + metadataSize;
          for (int i = 0; i < numChunks; ++i) {
            srcAddrs[copyBufferIdx] = outputChunkAddrs[chunkIdx];
            destAddrs[copyBufferIdx] = nextChunkAddr;
            final long chunkSize = outputChunkSizes[chunkIdx];
            sizes[copyBufferIdx] = chunkSize;
            copyBufferIdx++;
            chunkIdx++;
            nextChunkAddr += chunkSize;
          }
        }
        assert copyBufferIdx == totalBuffersToCopy;
        assert chunkIdx == outputChunkAddrs.length;
        assert chunkIdx == outputChunkSizes.length;

        Cuda.multiBufferCopyAsync(destAddrs, srcAddrs, sizes, stream);
        return outputs.release();
      }
    }
  }

  // Calculate the sizes for each output buffer (metadata plus size of compressed chunks)
  private long[] calcOutputBufferSizes(int[] chunksPerInput, long[] outputChunkSizes) {
    long[] sizes = new long[chunksPerInput.length];
    int chunkIdx = 0;
    for (int i = 0; i < sizes.length; i++) {
      final int chunksInBuffer = chunksPerInput[i];
      final int chunkEndIdx = chunkIdx + chunksInBuffer;
      // metadata stored in front of compressed data
      long bufferSize = METADATA_BYTES_PER_CHUNK * chunksInBuffer;
      // add in the compressed chunk sizes to get the total size
      while (chunkIdx < chunkEndIdx) {
        bufferSize += outputChunkSizes[chunkIdx++];
      }
      sizes[i] = bufferSize;
    }
    assert chunkIdx == outputChunkSizes.length;
    return sizes;
  }

  /**
   * Get the temporary workspace size required to perform compression of an entire batch.
   * @param batchSize number of chunks in the batch
   * @param maxChunkSize maximum size of an uncompressed chunk in bytes
   * @param totalSize Upper bound on the total uncompressed size of all chunks
   * @return The size of required temporary workspace in bytes to compress the batch.
   */
  protected abstract long batchedCompressGetTempSize(long batchSize, long maxChunkSize,
    long totalSize);

   /**
   * Asynchronously compress a batch of buffers. Note that compressedSizesOutPtr must
   * point to pinned memory for this operation to be asynchronous.
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
  protected abstract void batchedCompressAsync(long devInPtrs, long devInSizes, long chunkSize,
      long batchSize, long tempPtr, long tempSize, long devOutPtrs, long compressedSizesOutPtr,
      long stream);
}

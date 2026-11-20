/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2025, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.EOFException;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel.MapMode;


/**
 * This class holds an off-heap buffer in the host/CPU memory.
 * Please note that instances must be explicitly closed or native memory will be leaked!
 *
 * Internally this class will try to use PinnedMemoryPool to allocate and free the memory
 * it uses by default. To avoid using the pinned memory pool for allocations by default
 * set the Java system property ai.rapids.cudf.prefer-pinned to false.
 *
 * Be aware that the off heap memory limits set by Java do not apply to these buffers.
 */
public class HostMemoryBuffer extends MemoryBuffer {
  /** Optional handler for device-to-host copies into specialized host allocations. */
  public interface DeviceToHostCopyHandler {
    /**
     * Return true only if the copy has completed. Return false to use the default CUDA copy.
     * The handler must check whether it owns the destination and may receive any host buffer.
     */
    boolean copy(HostMemoryBuffer dst, long dstOffset, BaseDeviceMemoryBuffer src,
                 long srcOffset, long length, Cuda.Stream stream);
  }

  private static volatile DeviceToHostCopyHandler deviceToHostCopyHandler;

  /** Register a process-wide copy handler, or pass null to restore the default CUDA copy. */
  public static void setDeviceToHostCopyHandler(DeviceToHostCopyHandler handler) {
    deviceToHostCopyHandler = handler;
  }

  private boolean tryDeviceToHostCopy(BaseDeviceMemoryBuffer src, long length,
                                      Cuda.Stream stream) {
    DeviceToHostCopyHandler handler = deviceToHostCopyHandler;
    return handler != null && handler.copy(this, 0, src, 0, length, stream);
  }
  static final boolean defaultPreferPinned;
  private static final Logger log = LoggerFactory.getLogger(HostMemoryBuffer.class);

  static {
    boolean preferPinned = true;
    String propString = System.getProperty("ai.rapids.cudf.prefer-pinned");
    if (propString != null) {
      preferPinned = Boolean.parseBoolean(propString);
    }
    defaultPreferPinned = preferPinned;
  }

  private static final class HostBufferCleaner extends MemoryBufferCleaner {
    private long address;
    private final long length;

    HostBufferCleaner(long address, long length) {
      this.address = address;
      this.length = length;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      long origAddress = address;
      if (address != 0) {
        try {
          UnsafeMemoryAccessor.free(address);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          address = 0;
        }
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A HOST BUFFER WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress) + ")");
        logRefCountDebug("Leaked host buffer");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return address == 0;
    }
  }

  private static final class MmapCleaner extends MemoryBufferCleaner {
    private long address;
    private final long length;

    MmapCleaner(long address, long length) {
      this.address = address;
      this.length = length;
    }

    @Override
    protected boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = false;
      if (address != 0) {
        try {
          HostMemoryBufferNativeUtils.munmap(address, length);
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          address = 0;
        }
        neededCleanup = true;
      }
      if (neededCleanup && logErrorIfNotClean) {
        log.error("A MEMORY MAPPED BUFFER WAS LEAKED!!!!");
        logRefCountDebug("Leaked mmap buffer");
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return address == 0;
    }
  }

  /**
   * Allocate memory, but be sure to close the returned buffer to avoid memory leaks.
   * @param bytes size in bytes to allocate
   * @param preferPinned If set to true, the pinned memory pool will be used if possible with a
   *                    fallback to off-heap memory.  If set to false, the allocation will always
   *                    be from off-heap memory.
   * @return the newly created buffer
   */
  public static HostMemoryBuffer allocate(long bytes, boolean preferPinned) {
    return DefaultHostMemoryAllocator.get().allocate(bytes, preferPinned);
  }

  /**
   * Allocate memory, but be sure to close the returned buffer to avoid memory leaks. Pinned memory
   * will be preferred for allocations if the java system property ai.rapids.cudf.prefer-pinned is
   * set to true.
   * @param bytes size in bytes to allocate
   * @return the newly created buffer
   */
  public static HostMemoryBuffer allocate(long bytes) {
    return allocate(bytes, defaultPreferPinned);
  }

  /**
   * Allocate host memory bypassing the default allocator. This is intended to only be used by other allocators.
   * Pinned memory will not be used for these allocations.
   * @param bytes size in bytes to allocate
   * @return the newly created buffer
   */
  public static HostMemoryBuffer allocateRaw(long bytes) {
    return new HostMemoryBuffer(UnsafeMemoryAccessor.allocate(bytes), bytes);
  }

  /**
   * Create a host buffer that is memory-mapped to a file.
   * @param path path to the file to map into host memory
   * @param mode mapping type
   * @param offset file offset where the map will start
   * @param length the number of bytes to map
   * @return file-mapped buffer
   */
  public static HostMemoryBuffer mapFile(File path, MapMode mode,
      long offset, long length) throws IOException {
    // mapping offset must be a multiple of the system page size
    long offsetDelta = offset & (UnsafeMemoryAccessor.pageSize() - 1);
    long address;
    try {
      address = HostMemoryBufferNativeUtils.mmap(path.getPath(),
          modeAsInt(mode), offset - offsetDelta, length + offsetDelta);
    } catch (IOException e) {
      throw new IOException("Error creating memory map for " + path, e);
    }
    return new HostMemoryBuffer(address + offsetDelta, length,
        new MmapCleaner(address, length + offsetDelta));
  }

  private static int modeAsInt(MapMode mode) {
    if (MapMode.READ_ONLY.equals(mode)) {
      return 0;
    } else if (MapMode.READ_WRITE.equals(mode)) {
      return 1;
    } else {
      throw new UnsupportedOperationException("Unsupported mapping mode: " + mode);
    }
  }

  HostMemoryBuffer(long address, long length) {
    this(address, length, new HostBufferCleaner(address, length));
  }

  HostMemoryBuffer(long address, long length, MemoryBufferCleaner cleaner) {
    super(address, length, cleaner);
  }

  private HostMemoryBuffer(long address, long lengthInBytes, HostMemoryBuffer parent) {
    super(address, lengthInBytes, parent);
    // This is a slice so we are not going to mark it as allocated
  }

  /**
   * Return a ByteBuffer that provides access to the underlying memory.  Please note: if the buffer
   * is larger than a ByteBuffer can handle (2GB) an exception will be thrown.  Also
   * be aware that the ByteBuffer will be in native endian order, which is different from regular
   * ByteBuffers that are big endian by default.
   */
  public final ByteBuffer asByteBuffer() {
    assert length <= Integer.MAX_VALUE : "2GB limit on ByteBuffers";
    return asByteBuffer(0, (int) length);
  }

  /**
   * Return a ByteBuffer that provides access to the underlying memory.  Be aware that the
   * ByteBuffer will be in native endian order, which is different from regular
   * ByteBuffers that are big endian by default.
   * @param offset the offset to start at
   * @param length how many bytes to include.
   */
  public final ByteBuffer asByteBuffer(long offset, int length) {
    addressOutOfBoundsCheck(address + offset, length, "asByteBuffer");
    return HostMemoryBufferNativeUtils.wrapRangeInBuffer(address + offset, length)
        .order(ByteOrder.nativeOrder());
  }

  /**
   * Copy the contents of the given buffer to this buffer
   * @param destOffset offset in bytes in this buffer to start copying to
   * @param srcData    Buffer to be copied from
   * @param srcOffset  offset in bytes to start copying from in srcData
   * @param length     number of bytes to copy
   */
  public final void copyFromHostBuffer(long destOffset, HostMemoryBuffer srcData, long srcOffset,
                                       long length) {
    addressOutOfBoundsCheck(address + destOffset, length, "copy from dest");
    srcData.addressOutOfBoundsCheck(srcData.address + srcOffset, length, "copy from source");
    UnsafeMemoryAccessor.copyMemory(null, srcData.address + srcOffset, null,
        address + destOffset, length);
  }

  /**
   * Copy len bytes from in to this buffer.
   * @param destOffset  offset in bytes in this buffer to start copying to
   * @param in input stream to copy bytes from
   * @param byteLength number of bytes to copy
   * @throws EOFException If there are not enough bytes in the stream to copy.
   * @throws IOException If there is an error reading from the stream.
   */
  public final void copyFromStream(long destOffset, InputStream in, long byteLength) throws IOException {
    addressOutOfBoundsCheck(address + destOffset, byteLength, "copy from stream");
    byte[] arrayBuffer = new byte[(int) Math.min(1024 * 128, byteLength)];
    long left = byteLength;
    while (left > 0) {
      int amountToCopy = (int) Math.min(arrayBuffer.length, left);
      int amountRead = in.read(arrayBuffer, 0, amountToCopy);
      if (amountRead < 0) {
        throw new EOFException("Unexpected end of stream, expected " + left + " more bytes");
      }
      setBytes(destOffset, arrayBuffer, 0, amountRead);
      destOffset += amountRead;
      left -= amountRead;
    }
  }

  /**
   * Returns the byte value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final byte getByte(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 1, "getByte");
    return UnsafeMemoryAccessor.getByte(requestedAddress);
  }

  /**
   * Sets the byte value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setByte(long offset, byte value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 1, "setByte");
    UnsafeMemoryAccessor.setByte(requestedAddress, value);
  }

  /**
   * Copy a set of bytes to an array from the buffer starting at offset.
   * @param dst       destination byte array
   * @param dstOffset starting offset within the destination array
   * @param srcOffset starting offset within this buffer
   * @param len       number of bytes to copy
   */
  public final void getBytes(byte[] dst, long dstOffset, long srcOffset, long len) {
    assert len >= 0;
    assert len <= dst.length - dstOffset;
    assert srcOffset >= 0;
    long requestedAddress = this.address + srcOffset;
    addressOutOfBoundsCheck(requestedAddress, len, "getBytes");
    UnsafeMemoryAccessor.getBytes(dst, dstOffset, requestedAddress, len);
  }

  /**
   * Copy a set of bytes from an array into the buffer at offset.
   * @param offset the offset from the address to start copying to
   * @param data   the data to be copied.
   */
  public final void setBytes(long offset, byte[] data, long srcOffset, long len) {
    assert len >= 0 : "length is not allowed " + len;
    assert len <= data.length - srcOffset;
    assert srcOffset >= 0;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len, "setBytes");
    UnsafeMemoryAccessor.setBytes(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Short value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final short getShort(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 2, "getShort");
    return UnsafeMemoryAccessor.getShort(requestedAddress);
  }

  /**
   * Sets the Short value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setShort(long offset, short value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 2, "setShort");
    UnsafeMemoryAccessor.setShort(requestedAddress, value);
  }

  /**
   * Copy a set of shorts from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index in data to start at.
   */
  public final void setShorts(long offset, short[] data, long srcOffset, long len) {
    assert len >= 0 : "length is not allowed " + len;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 2, "setShorts");
    UnsafeMemoryAccessor.setShorts(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Integer value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final int getInt(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 4, "getInt");
    return UnsafeMemoryAccessor.getInt(requestedAddress);
  }

  /**
   * Copy a set of ints to an array from the buffer starting at offset.
   * @param dst       destination int array
   * @param dstIndex  starting index within the destination array
   * @param srcOffset starting offset within this buffer
   * @param count     number of ints to copy
   */
  public final void getInts(int[] dst, long dstIndex, long srcOffset, int count) {
    assert count >= 0;
    assert count <= dst.length - dstIndex;
    assert srcOffset >= 0;
    long requestedAddress = this.address + srcOffset;
    addressOutOfBoundsCheck(requestedAddress, count * 4L, "getInts");
    UnsafeMemoryAccessor.getInts(dst, dstIndex, requestedAddress, count);
  }


  /**
   * Sets the Integer value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setInt(long offset, int value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 4, "setInt");
    UnsafeMemoryAccessor.setInt(requestedAddress, value);
  }

  /**
   * Copy a set of ints from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index into data to start at
   */
  public final void setInts(long offset, int[] data, long srcOffset, long len) {
    assert len >= 0 : "length is not allowed " + len;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 4, "setInts");
    UnsafeMemoryAccessor.setInts(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Long value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final long getLong(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 8, "getLong");
    return UnsafeMemoryAccessor.getLong(requestedAddress);
  }

  /**
   * Sets the Long value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setLong(long offset, long value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 8, "setLong");
    UnsafeMemoryAccessor.setLong(requestedAddress, value);
  }

  /**
   * Copy a set of longs to an array from the buffer starting at offset.
   * @param dst       destination long array
   * @param dstIndex  starting index within the destination array
   * @param srcOffset starting offset within this buffer
   * @param count     number of longs to copy
   */
  public final void getLongs(long[] dst, long dstIndex, long srcOffset, int count) {
    assert count >= 0;
    assert count <= dst.length - dstIndex;
    assert srcOffset >= 0;
    long requestedAddress = this.address + srcOffset;
    addressOutOfBoundsCheck(requestedAddress, count * 8L, "getLongs");
    UnsafeMemoryAccessor.getLongs(dst, dstIndex, requestedAddress, count);
  }

  /**
   * Copy a set of longs from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index into data to start at.
   */
  public final void setLongs(long offset, long[] data, long srcOffset, long len) {
    assert len >= 0 : "length is not allowed " + len;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 8, "setLongs");
    UnsafeMemoryAccessor.setLongs(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Float value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final float getFloat(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 4, "getFloat");
    return UnsafeMemoryAccessor.getFloat(requestedAddress);
  }

  /**
   * Sets the Float value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setFloat(long offset, float value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 4, "setFloat");
    UnsafeMemoryAccessor.setFloat(requestedAddress, value);
  }

  /**
   * Copy a set of floats from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index into data to start at
   */
  public final void setFloats(long offset, float[] data, long srcOffset, long len) {
    assert len >= 0 : "length is not allowed " + len;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 4, "setFloats");
    UnsafeMemoryAccessor.setFloats(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Double value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final double getDouble(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 8, "getDouble");
    return UnsafeMemoryAccessor.getDouble(requestedAddress);
  }

  /**
   * Sets the Double value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setDouble(long offset, double value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 8, "setDouble");
    UnsafeMemoryAccessor.setDouble(requestedAddress, value);
  }

  /**
   * Copy a set of doubles from an array into the buffer at offset.
   * @param offset    the offset from the address to start copying to
   * @param data      the data to be copied.
   * @param srcOffset index into data to start at
   */
  public final void setDoubles(long offset, double[] data, long srcOffset, long len) {
    assert len >= 0 : "length is not allowed " + len;
    assert len <= data.length - srcOffset;
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, len * 8, "setDoubles");
    UnsafeMemoryAccessor.setDoubles(requestedAddress, data, srcOffset, len);
  }

  /**
   * Returns the Boolean value at that offset
   * @param offset - offset from the address
   * @return - value
   */
  public final boolean getBoolean(long offset) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 1, "getBoolean");
    return UnsafeMemoryAccessor.getBoolean(requestedAddress);
  }

  /**
   * Sets the Boolean value at that offset
   * @param offset - offset from the address
   * @param value  - value to be set
   */
  public final void setBoolean(long offset, boolean value) {
    long requestedAddress = this.address + offset;
    addressOutOfBoundsCheck(requestedAddress, 1, "setBoolean");
    UnsafeMemoryAccessor.setBoolean(requestedAddress, value);
  }

  /**
   * Sets the values in this buffer repeatedly
   * @param offset - offset from the address
   * @param length - number of bytes to set
   * @param value  - value to be set
   */
  public final void setMemory(long offset, long length, byte value) {
    addressOutOfBoundsCheck(address + offset, length, "set memory");
    UnsafeMemoryAccessor.setMemory(address + offset, length, value);
  }

  final void copyFromMemory(long fromAddress, long len) {
    addressOutOfBoundsCheck(address, len, "copy from memory");
    UnsafeMemoryAccessor.copyMemory(null, fromAddress, null, address, len);
  }

  /**
   * Copy data from this buffer to the given address.
   * @param toAddress where the data should go
   * @param len how much data to copy
   */
  final void copyToMemory(long toAddress, long len) {
    addressOutOfBoundsCheck(address, len, "copy to memory");
    UnsafeMemoryAccessor.copyMemory(null, address, null, toAddress, len);
  }

  /**
   * Synchronously copy from a DeviceMemoryBuffer to a HostMemoryBuffer
   * @param deviceMemoryBuffer buffer to copy data from
   */
  public final void copyFromDeviceBuffer(BaseDeviceMemoryBuffer deviceMemoryBuffer) {
    addressOutOfBoundsCheck(address, deviceMemoryBuffer.length, "copy range dest");
    assert !deviceMemoryBuffer.closed;
    if (!tryDeviceToHostCopy(deviceMemoryBuffer, deviceMemoryBuffer.length,
        Cuda.DEFAULT_STREAM)) {
      Cuda.memcpy(address, deviceMemoryBuffer.address, deviceMemoryBuffer.length,
          CudaMemcpyKind.DEVICE_TO_HOST);
    }
  }

  /**
   * Copy from a DeviceMemoryBuffer to a HostMemoryBuffer using the specified stream.
   * The copy has completed when this returns, but the memory copy could overlap with
   * operations occurring on other streams.
   * @param deviceMemoryBuffer buffer to copy data from
   * @param stream CUDA stream to use
   */
  public final void copyFromDeviceBuffer(BaseDeviceMemoryBuffer deviceMemoryBuffer,
                                         Cuda.Stream stream) {
    addressOutOfBoundsCheck(address, deviceMemoryBuffer.length, "copy range dest");
    assert !deviceMemoryBuffer.closed;
    if (!tryDeviceToHostCopy(deviceMemoryBuffer, deviceMemoryBuffer.length, stream)) {
      Cuda.memcpy(address, deviceMemoryBuffer.address, deviceMemoryBuffer.length,
          CudaMemcpyKind.DEVICE_TO_HOST, stream);
    }
  }

  /**
   * Copy from a DeviceMemoryBuffer to a HostMemoryBuffer using the specified stream.
   * The copy is async and may not have completed when this returns.
   * @param deviceMemoryBuffer buffer to copy data from
   * @param stream CUDA stream to use
   */
  public final void copyFromDeviceBufferAsync(BaseDeviceMemoryBuffer deviceMemoryBuffer,
                                              Cuda.Stream stream) {
    addressOutOfBoundsCheck(address, deviceMemoryBuffer.length, "copy range dest");
    assert !deviceMemoryBuffer.closed;
    if (!tryDeviceToHostCopy(deviceMemoryBuffer, deviceMemoryBuffer.length, stream)) {
      Cuda.asyncMemcpy(address, deviceMemoryBuffer.address, deviceMemoryBuffer.length,
          CudaMemcpyKind.DEVICE_TO_HOST, stream);
    }
  }

  /**
   * Slice off a part of the host buffer.
   * @param offset where to start the slice at.
   * @param len how many bytes to slice
   * @return a host buffer that will need to be closed independently from this buffer.
   */
  @Override
  public final synchronized HostMemoryBuffer slice(long offset, long len) {
    addressOutOfBoundsCheck(address + offset, len, "slice");
    refCount++;
    cleaner.addRef();
    return new HostMemoryBuffer(address + offset, len, this);
  }

  /**
   * Slice off a part of the host buffer, actually making a copy of the data.
   * @param offset where to start the slice at.
   * @param len how many bytes to slice
   * @return a host buffer that will need to be closed independently from this buffer.
   */
  public final HostMemoryBuffer sliceWithCopy(long offset, long len) {
    addressOutOfBoundsCheck(address + offset, len, "slice");

    HostMemoryBuffer ret = null;
    boolean success = false;
    try {
      ret = allocate(len);
      UnsafeMemoryAccessor.copyMemory(null, address + offset, null, ret.getAddress(), len);
      success = true;
      return ret;
    } finally {
      if (!success && ret != null) {
        ret.close();
      }
    }
  }

  /**
   * WARNING: Debug only method to print buffer. Does not work for buffers over 2GB.
   */
  public void printBuffer() {
    printBuffer(5);
  }

  /**
   * WARNING: Debug only method to print buffer.  Does not work for buffers over 2GB.
   * @param wordsPerRow the number of 32 bit works to print per row.
   */
  public void printBuffer(int wordsPerRow) {
    final int bytesPerWord = 4;
    final int bytesPerRow = bytesPerWord * wordsPerRow;
    assert (length == (int)length) : "The buffer is too large to be printed";
    byte[] data = new byte[(int)length];
    System.out.println("BUFFER length = " + data.length);
    getBytes(data, 0, 0, length);
    for (int i = 0; i < data.length; i++) {
      if (i % bytesPerWord == 0) {
        if (i % bytesPerRow == 0) {
          System.out.println();
        } else {
          System.out.print(" ");
        }
      }
      System.out.print(String.format("%02x",((long)data[i]) & 0xFF));
    }
    System.out.println();
  }
}

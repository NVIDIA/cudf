/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Abstract class for representing the Memory Buffer
 *
 * NOTE: MemoryBuffer is public to make it easier to work with the class hierarchy,
 * subclassing beyond what is included in CUDF is not recommended and not supported.
 */
abstract public class MemoryBuffer implements AutoCloseable {
  /**
   * Interface to handle events for this MemoryBuffer. Only invoked during
   * close, hence `onClosed` is the only event.
   */
  public interface EventHandler {
    /**
     * `onClosed` is invoked with the updated `refCount` during `close`.
     * The last invocation of `onClosed` will be with `refCount=0`.
     *
     * @note the callback is invoked with this `MemoryBuffer`'s lock held.
     *
     * @param refCount - the updated ref count for this MemoryBuffer at the time
     *                 of invocation
     */
    void onClosed(int refCount);
  }

  private static final Logger log = LoggerFactory.getLogger(MemoryBuffer.class);
  protected final long address;
  protected final long length;
  protected boolean closed = false;
  protected int refCount = 0;
  protected final MemoryBufferCleaner cleaner;
  protected final long id;

  private EventHandler eventHandler;

  public static abstract class MemoryBufferCleaner extends MemoryCleaner.Cleaner{}

  private static final class SlicedBufferCleaner extends MemoryBufferCleaner {
    private MemoryBuffer parent;

    SlicedBufferCleaner(MemoryBuffer parent) {
      this.parent = parent;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      if (parent != null) {
        if (logErrorIfNotClean) {
          log.error("A SLICED BUFFER WAS LEAKED(ID: " + id + " parent: " + parent + ")");
          logRefCountDebug("Leaked sliced buffer");
        }
        try {
          parent.close();
        } finally {
          // Always mark the resource as freed even if an exception is thrown.
          // We cannot know how far it progressed before the exception, and
          // therefore it is unsafe to retry.
          parent = null;
        }
        return true;
      }
      return false;
    }

    @Override
    public boolean isClean() {
      return parent == null;
    }
  }

  /**
   * This is a really ugly API, but it is possible that the lifecycle of a column of
   * data may not have a clear lifecycle thanks to java and GC. This API informs the leak
   * tracking code that this is expected for this column, and big scary warnings should
   * not be printed when this happens.
   */
  public void noWarnLeakExpected() {
    if (cleaner != null) {
      cleaner.noWarnLeakExpected();
    }
  }

  /**
   * Constructor
   * @param address location in memory
   * @param length  size of this buffer
   * @param cleaner used to clean up the memory. May be null if no cleanup is needed.
   */
  protected MemoryBuffer(long address, long length, MemoryBufferCleaner cleaner) {
    this.address = address;
    this.length = length;
    this.cleaner = cleaner;
    if (cleaner != null) {
      this.id = cleaner.id;
      incRefCount();
      MemoryCleaner.register(this, cleaner);
    } else {
      this.id = -1;
    }
  }

  /**
   * Constructor
   * @param address location in memory
   * @param length  size of this buffer
   */
  protected MemoryBuffer(long address, long length) {
    this(address, length, (MemoryBufferCleaner)null);
  }

  /**
   * Internal constructor used when creating a slice.
   * @param address location in memory
   * @param length size of this buffer
   * @param parent the buffer that should be closed instead of closing this one.
   */
  protected MemoryBuffer(long address, long length, MemoryBuffer parent) {
    this(address, length, new SlicedBufferCleaner(parent));
  }

  /**
   * Returns the size of this buffer
   * @return - size
   */
  public final long getLength() {
    return length;
  }

  protected final void addressOutOfBoundsCheck(long address, long size, String type) {
    assert !closed : "Buffer is already closed " + Long.toHexString(this.address);
    assert size >= 0 : "A positive size is required";
    assert address >= this.address : "Start address is too low for " + type +
        " 0x" + Long.toHexString(address) + " < 0x" + Long.toHexString(this.address);
    assert (address + size) <= (this.address + length) : "End address is too high for " + type +
        " 0x" + Long.toHexString(address + size) + " < 0x" + Long.toHexString(this.address + length);
  }

  /**
   * Returns the location of the data pointed to by this buffer
   * @return - data address
   */
  public final long getAddress() {
    return address;
  }

  /**
   * Copy a subset of src to this buffer starting at destOffset using the specified CUDA stream.
   * The copy has completed when this returns, but the memory copy could overlap with
   * operations occurring on other streams.
   * @param destOffset the offset in this to start copying from.
   * @param src what to copy from
   * @param srcOffset offset into src to start out
   * @param length how many bytes to copy
   * @param stream CUDA stream to use
   */
  public final void copyFromMemoryBuffer(
          long destOffset, MemoryBuffer src, long srcOffset, long length, Cuda.Stream stream) {
    addressOutOfBoundsCheck(address + destOffset, length, "copy range dest");
    src.addressOutOfBoundsCheck(src.address + srcOffset, length, "copy range src");
    Cuda.memcpy(address + destOffset, src.address + srcOffset, length, CudaMemcpyKind.DEFAULT, stream);
  }

  /**
   * Copy a subset of src to this buffer starting at destOffset using the specified CUDA stream.
   * The copy is async and may not have completed when this returns.
   * @param destOffset the offset in this to start copying from.
   * @param src what to copy from
   * @param srcOffset offset into src to start out
   * @param length how many bytes to copy
   * @param stream CUDA stream to use
   */
  public final void copyFromMemoryBufferAsync(
          long destOffset, MemoryBuffer src, long srcOffset, long length, Cuda.Stream stream) {
    addressOutOfBoundsCheck(address + destOffset, length, "copy range dest");
    src.addressOutOfBoundsCheck(src.address + srcOffset, length, "copy range src");
    Cuda.asyncMemcpy(address + destOffset, src.address + srcOffset, length, CudaMemcpyKind.DEFAULT, stream);
  }

  /**
   * Slice off a part of the buffer. Note that this is a zero copy operation and all
   * slices must be closed along with the original buffer before the memory is released.
   * So use this with some caution.
   *
   * Note that {@link DeviceMemoryBuffer} and {@link HostMemoryBuffer} support slicing, and override
   * this function.
   *
   * @param offset where to start the slice at.
   * @param len how many bytes to slice
   * @return a slice of the original buffer that will need to be closed independently
   */
  public abstract MemoryBuffer slice(long offset, long len);

  /**
   * Set an event handler for this buffer. This method can be invoked with null
   * to unset the handler.
   *
   * @param newHandler - the EventHandler to use from this point forward
   * @return the prior event handler, or null if not set.
   */
  public synchronized EventHandler setEventHandler(EventHandler newHandler) {
    EventHandler prev = this.eventHandler;
    this.eventHandler = newHandler;
    return prev;
  }

  /**
   * Returns the current event handler for this buffer or null if no handler
   * is associated or this buffer is closed.
   */
  public synchronized EventHandler getEventHandler() {
    return this.eventHandler;
  }

  /**
   * Close this buffer and free memory
   */
  public synchronized void close() {
    if (cleaner != null) {
      refCount--;
      cleaner.delRef();
      try {
        if (refCount == 0) {
          cleaner.clean(false);
          closed = true;
        } else if (refCount < 0) {
          cleaner.logRefCountDebug("double free " + this);
          throw new IllegalStateException("Close called too many times " + this);
        }
      } finally {
        if (eventHandler != null) {
          eventHandler.onClosed(refCount);
        }
      }
    }
  }

  @Override
  public String toString() {
    long id = -1;
    if (cleaner != null) {
      id = cleaner.id;
    }
    String name = this.getClass().getSimpleName();
    return name + "{" +
        "address=0x" + Long.toHexString(address) +
        ", length=" + length +
        ", id=" + id + "}";
  }

  /**
   * Increment the reference count for this column.  You need to call close on this
   * to decrement the reference count again.
   */
  public synchronized void incRefCount() {
    refCount++;
    cleaner.addRef();
  }

  /**
   * Get the current reference count for this buffer.
   */
  public synchronized int getRefCount() {
    return refCount;
  }
}

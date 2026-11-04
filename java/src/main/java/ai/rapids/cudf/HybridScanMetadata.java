/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A parsed Parquet file footer for the experimental hybrid-scan reader.
 *
 * <p>This class is the Java binding for
 * {@code cudf::io::parquet::experimental::hybrid_scan_metadata}. Parsing the footer apart from
 * constructing a {@link HybridScanReader} lets a caller inspect the footer before deciding
 * whether to build a reader at all, then build one with
 * {@link HybridScanReader#HybridScanReader(HybridScanMetadata, ParquetOptions)} without parsing
 * the footer again. Native code can read the parsed footer in place through
 * {@link #getParquetMetadataAddress(int)}.
 *
 * <p>The APIs in this file are experimental and subject to change.
 */
@Experimental
public class HybridScanMetadata implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(HybridScanMetadata.class);

  private static final class HybridScanMetadataCleaner extends MemoryCleaner.Cleaner {
    private long nativeHandle;

    HybridScanMetadataCleaner(long nativeHandle) {
      this.nativeHandle = nativeHandle;
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      long origAddress = nativeHandle;
      boolean neededCleanup = nativeHandle != 0;
      if (neededCleanup) {
        try {
          destroy(nativeHandle);
        } finally {
          nativeHandle = 0;
        }
        if (logErrorIfNotClean) {
          log.error("A HYBRID SCAN METADATA WAS LEAKED (ID: {} {})", id,
              Long.toHexString(origAddress));
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return nativeHandle == 0;
    }
  }

  private final HybridScanMetadataCleaner cleaner;

  private boolean isClosed = false;

  /**
   * Parse the bytes of a Parquet file footer.
   *
   * @param footerBuffer host-resident footer bytes (must remain valid until this constructor
   *                     returns; the JNI reads the bytes synchronously)
   * @param opts         Parquet reader options. {@link ParquetOptions#DEFAULT} by default.
   */
  public HybridScanMetadata(HostMemoryBuffer footerBuffer, ParquetOptions opts) {
    if (footerBuffer == null) {
      throw new IllegalArgumentException("footerBuffer must not be null");
    }
    if (opts == null) {
      opts = ParquetOptions.DEFAULT;
    }
    long handle = createFromFooter(
        footerBuffer.getAddress(),
        footerBuffer.getLength(),
        opts.getIncludeColumnNames(),
        opts.timeUnit().getTypeId().getNativeId());
    this.cleaner = new HybridScanMetadataCleaner(handle);
    MemoryCleaner.register(this, cleaner);
    cleaner.addRef();
  }

  /**
   * Get the address of the parsed native {@code cudf::io::parquet::FileMetaData}, for native
   * code to read in place.
   *
   * <p>The memory is owned by this object: the address is valid until {@link #close()}, and must
   * not be written through or freed. A {@link HybridScanReader} that calls
   * {@link HybridScanReader#setupPageIndex(HostMemoryBuffer)} fills in the page index of this
   * same footer.
   *
   * @return the address of the {@code FileMetaData}
   */
  public long getParquetMetadataAddress() {
    return parquetMetadataAddress(getNativeHandle());
  }

  long getNativeHandle() {
    if (isClosed) {
      throw new IllegalStateException("HybridScanMetadata has been closed");
    }
    return cleaner.nativeHandle;
  }

  @Override
  public synchronized void close() {
    if (isClosed) {
      return;
    }
    cleaner.delRef();
    cleaner.clean(false);
    isClosed = true;
  }

  private static native long createFromFooter(long footerAddress,
                                              long footerLength,
                                              String[] columnNames,
                                              int timeUnitTypeId);

  private static native long parquetMetadataAddress(long handle);

  private static native void destroy(long handle);
}

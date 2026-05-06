/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class represents a reusable hash table built from distinct join keys from the right-side
 * table for a join operation. The resulting handle can be reused across a series of left probe
 * tables when the right-side join keys are guaranteed to be distinct.
 */
public class DistinctHashJoin implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(DistinctHashJoin.class);

  private static class DistinctHashJoinCleaner extends MemoryCleaner.Cleaner {
    private Table buildKeys;
    private long nativeHandle;

    DistinctHashJoinCleaner(Table buildKeys, long nativeHandle) {
      this.buildKeys = buildKeys;
      this.nativeHandle = nativeHandle;
      addRef();
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      long origAddress = nativeHandle;
      boolean neededCleanup = nativeHandle != 0;
      if (neededCleanup) {
        try {
          destroy(nativeHandle);
          buildKeys.close();
          buildKeys = null;
        } finally {
          nativeHandle = 0;
        }
        if (logErrorIfNotClean) {
          log.error("A DISTINCT HASH TABLE WAS LEAKED (ID: " + id + " " +
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

  private final DistinctHashJoinCleaner cleaner;
  private final boolean compareNulls;
  private boolean isClosed = false;

  /**
   * Construct a reusable distinct hash table from the join key columns from the right-side table.
   * The build key rows must be distinct.
   *
   * @param buildKeys table view containing the join keys for the right-side join table
   * @param compareNulls true if null key values should match otherwise false
   */
  public DistinctHashJoin(Table buildKeys, boolean compareNulls) {
    this.compareNulls = compareNulls;
    Table buildTable = new Table(buildKeys.getColumns());
    try {
      long handle = create(buildTable.getNativeView(), compareNulls);
      this.cleaner = new DistinctHashJoinCleaner(buildTable, handle);
      MemoryCleaner.register(this, cleaner);
    } catch (Throwable t) {
      try {
        buildTable.close();
      } catch (Throwable t2) {
        t.addSuppressed(t2);
      }
      throw t;
    }
  }

  @Override
  public synchronized void close() {
    cleaner.delRef();
    if (isClosed) {
      cleaner.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times " + this);
    }
    cleaner.clean(false);
    isClosed = true;
  }

  /** Get the number of join key columns for the table used to generate the hash table. */
  public long getNumberOfColumns() {
    return cleaner.buildKeys.getNumberOfColumns();
  }

  /** Returns true if the hash table was built to match on nulls otherwise false. */
  public boolean getCompareNulls() {
    return compareNulls;
  }

  long getNativeView() {
    return cleaner.nativeHandle;
  }

  private static native long create(long tableView, boolean nullEqual);
  private static native void destroy(long handle);
}

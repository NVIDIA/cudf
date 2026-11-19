/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    private volatile Table buildKeys;
    private long nativeHandle;

    DistinctHashJoinCleaner(Table buildKeys) {
      this.buildKeys = new Table(buildKeys.getColumns());
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = buildKeys != null;
      if (neededCleanup) {
        long origAddress = nativeHandle;
        try (Table toClose = buildKeys) {
          destroy(nativeHandle);
        } finally {
          nativeHandle = 0;
          buildKeys = null;
        }
        if (logErrorIfNotClean) {
          log.error("A DISTINCT HASH TABLE WAS LEAKED (ID: {} {})", id,
              Long.toHexString(origAddress));
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return buildKeys == null;
    }
  }

  private final DistinctHashJoinCleaner cleaner;
  private final long numberOfColumns;
  private final boolean compareNullsEqual;
  private boolean isClosed = false;

  /**
   * Construct a reusable distinct hash table from the join key columns from the right-side table.
   * Behavior is undefined if the build key rows contain duplicates. All NaN values are considered
   * equal. The resulting instance must be closed to release the GPU resources associated with the
   * instance.
   *
   * @param buildKeys table view containing the join keys for the right-side join table
   * @param compareNullsEqual true if null key values should match otherwise false
   */
  public DistinctHashJoin(Table buildKeys, boolean compareNullsEqual) {
    this.numberOfColumns = buildKeys.getNumberOfColumns();
    this.compareNullsEqual = compareNullsEqual;
    this.cleaner = new DistinctHashJoinCleaner(buildKeys);
    try {
      cleaner.addRef();
      cleaner.nativeHandle = create(cleaner.buildKeys.getNativeView(), compareNullsEqual);
      MemoryCleaner.register(this, cleaner);
    } catch (Throwable t) {
      try {
        cleaner.clean(false);
      } catch (Throwable t2) {
        t.addSuppressed(t2);
      }
      throw t;
    }
  }

  @Override
  public synchronized void close() {
    if (isClosed) {
      cleaner.logRefCountDebug("double free " + this);
      throw new IllegalStateException("Close called too many times " + this);
    }
    cleaner.delRef();
    isClosed = true;
    cleaner.clean(false);
  }

  /** Get the number of join key columns for the table used to generate the hash table. */
  public long getNumberOfColumns() {
    return numberOfColumns;
  }

  /** Returns true if the hash table was built to match on nulls otherwise false. */
  public boolean getCompareNullsEqual() {
    return compareNullsEqual;
  }

  long getNativeView() {
    if (isClosed) {
      throw new IllegalStateException("DistinctHashJoin is already closed");
    }
    return cleaner.nativeHandle;
  }

  private static native long create(long tableView, boolean compareNullsEqual);
  private static native void destroy(long handle);
}

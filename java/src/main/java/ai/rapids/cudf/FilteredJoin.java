/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A reusable hash lookup built from the right-side join keys for left semi and left anti joins.
 * This object can be reused for multiple probe tables. Duplicate build keys are supported.
 * Each probe row is returned at most once by a semi join, regardless of how many build rows match.
 * An anti join returns probe rows with no match.
 */
public class FilteredJoin implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(FilteredJoin.class);

  private static class FilteredJoinCleaner extends MemoryCleaner.Cleaner {
    private volatile Table buildKeys;
    private long nativeHandle;

    FilteredJoinCleaner(Table buildKeys) {
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
          log.error("A FILTERED JOIN WAS LEAKED (ID: {} {})", id,
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

  private final FilteredJoinCleaner cleaner;
  private final long numberOfColumns;
  private final boolean compareNullsEqual;
  private boolean isClosed = false;

  /**
   * Construct a reusable lookup from the join key columns of the right-side table.
   * The key rows need not be distinct. All NaN values are considered equal. The
   * resulting instance must be closed to release the GPU resources associated with
   * this instance.
   *
   * @param buildKeys table containing the right-side join keys
   * @param compareNullsEqual true if null key values should match otherwise false
   */
  public FilteredJoin(Table buildKeys, boolean compareNullsEqual) {
    this.numberOfColumns = buildKeys.getNumberOfColumns();
    this.compareNullsEqual = compareNullsEqual;
    this.cleaner = new FilteredJoinCleaner(buildKeys);
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

  /** Get the number of join key columns used to build the lookup. */
  public long getNumberOfColumns() {
    return numberOfColumns;
  }

  /** Returns true if the lookup was built to match on null keys. */
  public boolean getCompareNullsEqual() {
    return compareNullsEqual;
  }

  long getNativeView() {
    if (isClosed) {
      throw new IllegalStateException("FilteredJoin is already closed");
    }
    return cleaner.nativeHandle;
  }

  private static native long create(long tableView, boolean compareNullsEqual);
  private static native void destroy(long handle);
}

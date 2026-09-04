/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * This class represents a hash table built from the join keys of the right-side table for a
 * join operation. This hash table can then be reused across a series of left probe tables
 * to compute gather maps for joins more efficiently when the right-side table is not changing.
 * It can also be used to query the output row count of a join before manifesting the join gather
 * maps. Passing that row count to the gather-map operation no longer avoids any work: the output
 * size is always computed internally and a supplied count is only validated against it, so an
 * incorrect count fails with a {@link CudfException}. Prefer the overloads without a row count;
 * the row-count overloads will be deprecated in a future release.
 */
public class HashJoin implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(HashJoin.class);

  private static class HashJoinCleaner extends MemoryCleaner.Cleaner {
    private Table buildKeys;
    private long nativeHandle;

    HashJoinCleaner(Table buildKeys, long nativeHandle) {
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
          log.error("A HASH TABLE WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress));
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return nativeHandle == 0;
    }
  }

  private final HashJoinCleaner cleaner;
  private final boolean compareNulls;
  private boolean isClosed = false;

  /**
   * Construct a hash table for a join from a table representing the join key columns from the
   * right-side table in the join. The resulting instance must be closed to release the
   * GPU resources associated with the instance.
   * @param buildKeys table view containing the join keys for the right-side join table
   * @param compareNulls true if null key values should match otherwise false
   */
  public HashJoin(Table buildKeys, boolean compareNulls) {
    this.compareNulls = compareNulls;
    Table buildTable = new Table(buildKeys.getColumns());
    try {
      long handle = create(buildTable.getNativeView(), compareNulls);
      this.cleaner = new HashJoinCleaner(buildTable, handle);
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

  long getNativeView() {
    return cleaner.nativeHandle;
  }

  /** Get the number of join key columns for the table that was used to generate the has table. */
  public long getNumberOfColumns() {
    return cleaner.buildKeys.getNumberOfColumns();
  }

  /** Returns true if the hash table was built to match on nulls otherwise false. */
  public boolean getCompareNulls() {
    return compareNulls;
  }

  private static native long create(long tableView, boolean nullEqual);
  private static native void destroy(long handle);
}

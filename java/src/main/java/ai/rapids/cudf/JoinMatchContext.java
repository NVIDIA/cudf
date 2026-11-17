/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The matches found by probing a left table against a {@link HashJoin}, kept so the join
 * results can be retrieved in partitions without probing again. Obtained from a Table factory
 * such as {@link Table#innerJoinMatchContext(HashJoin)}, used to read match counts and output
 * size, and used to retrieve a range of the left table with {@link #partitionGatherMaps(int, int)}.
 *
 * <p>The context must be closed, and the owning HashJoin must stay open for its lifetime.</p>
 *
 * <p>This class is not thread safe. A context should be created, used and closed on one thread.</p>
 */
public class JoinMatchContext implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(JoinMatchContext.class);

  private static class JoinMatchContextCleaner extends MemoryCleaner.Cleaner {
    private volatile Table leftKeys;
    private long nativeHandle;

    JoinMatchContextCleaner(Table leftKeys) {
      this.leftKeys = new Table(leftKeys.getColumns());
    }

    @Override
    protected synchronized boolean cleanImpl(boolean logErrorIfNotClean) {
      boolean neededCleanup = !isClean();
      if (neededCleanup) {
        long origAddress = nativeHandle;
        try (Table toClose = leftKeys) {
          destroy(nativeHandle);
        } finally {
          nativeHandle = 0;
          leftKeys = null;
        }
        if (logErrorIfNotClean) {
          log.error("A JOIN MATCH CONTEXT WAS LEAKED (ID: {} {})", id,
              Long.toHexString(origAddress));
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return leftKeys == null;
    }
  }

  private final JoinMatchContextCleaner cleaner;
  private final HashJoin hashJoin;
  private final JoinKind joinKind;
  private boolean isClosed = false;

  /**
   * Probes the left keys against the hash table. The probe runs asynchronously on the calling
   * thread's stream and is not synchronized, so the context should be used and closed on the
   * thread that created it.
   */
  JoinMatchContext(Table leftKeys, HashJoin hashJoin, JoinKind joinKind) {
    this.hashJoin = hashJoin;
    this.joinKind = joinKind;
    this.cleaner = new JoinMatchContextCleaner(leftKeys);
    try {
      cleaner.addRef();
      cleaner.nativeHandle = create(cleaner.leftKeys.getNativeView(),
          hashJoin.getNativeView(), joinKind.nativeId);
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

  /** The join kind that created this context. */
  public JoinKind getJoinKind() {
    return joinKind;
  }

  /**
   * The number of row pairs a full retrieval of the left table would produce.
   */
  public long getOutputSize() {
    ensureOpen();
    return outputSize(cleaner.nativeHandle);
  }

  /**
   * The per-row match counts as an INT32 column. The caller owns the returned column.
   * The column is copied asynchronously on the calling thread's stream, so the context must not
   * be closed from another thread until that stream has been synchronized.
   */
  public ColumnVector getMatchCounts() {
    ensureOpen();
    return new ColumnVector(matchCountsColumn(cleaner.nativeHandle));
  }

  /**
   * Gather maps for a contiguous range of left rows. For FULL joins the complement
   * of unmatched right rows is not included. Call
   * {@link #finalizePartitionedFullJoin(GatherMap[], GatherMap[], long, long)} once all
   * partitions have been retrieved.
   *
   * @param leftStart inclusive start row in the left table
   * @param leftEnd exclusive end row in the left table
   * @return left and right gather maps for this partition
   */
  public GatherMap[] partitionGatherMaps(int leftStart, int leftEnd) {
    ensureOpen();
    long[] data = partitionGatherMaps(cleaner.nativeHandle, hashJoin.getNativeView(),
        joinKind.nativeId, leftStart, leftEnd);
    return Table.buildJoinGatherMaps(data);
  }

  /**
   * Concatenate per-partition full join results and append the unmatched right rows.
   * The two input arrays must have the same length and must come from
   * {@link #partitionGatherMaps(int, int)} calls on FULL join contexts.
   *
   * @param leftPartials per-partition left gather maps
   * @param rightPartials per-partition right gather maps
   * @param leftRowCount total rows in the original left table
   * @param rightRowCount total rows in the right table
   * @return the full join left and right gather maps
   */
  public static GatherMap[] finalizePartitionedFullJoin(GatherMap[] leftPartials,
                                                        GatherMap[] rightPartials,
                                                        long leftRowCount,
                                                        long rightRowCount) {
    if (leftPartials.length != rightPartials.length) {
      throw new IllegalArgumentException("partition arrays must have the same length");
    }
    if (leftRowCount < 0 || leftRowCount > Integer.MAX_VALUE
        || rightRowCount < 0 || rightRowCount > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("row counts must be between 0 and " + Integer.MAX_VALUE
          + ", got left=" + leftRowCount + " right=" + rightRowCount);
    }
    int n = leftPartials.length;
    long[] leftAddrs = new long[n];
    long[] leftLens = new long[n];
    long[] rightAddrs = new long[n];
    long[] rightLens = new long[n];
    for (int i = 0; i < n; i++) {
      leftAddrs[i] = leftPartials[i].getBufferAddress();
      leftLens[i] = leftPartials[i].getRowCount();
      rightAddrs[i] = rightPartials[i].getBufferAddress();
      rightLens[i] = rightPartials[i].getRowCount();
    }
    long[] data = finalizePartitionedFullJoin(leftAddrs, leftLens, rightAddrs, rightLens,
        (int) leftRowCount, (int) rightRowCount);
    return Table.buildJoinGatherMaps(data);
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

  private void ensureOpen() {
    if (isClosed) {
      throw new IllegalStateException("JoinMatchContext is already closed");
    }
  }

  private static native long create(long leftView, long hashJoinHandle, int joinKindId);
  private static native void destroy(long handle);
  private static native long outputSize(long handle);
  private static native long matchCountsColumn(long handle);
  private static native long[] partitionGatherMaps(long handle, long hashJoinHandle,
                                                   int joinKindId, int leftStart, int leftEnd);
  private static native long[] finalizePartitionedFullJoin(long[] leftAddrs, long[] leftLens,
                                                           long[] rightAddrs, long[] rightLens,
                                                           int leftRowCount, int rightRowCount);
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * Owning result of a status-aware Variant extraction or cast.
 *
 * <p>The value and status columns have the same row count. The status column is a non-nullable
 * {@link DType#UINT8} column containing IDs defined by {@link VariantOperationStatus}. Both columns
 * are owned by this result and remain valid until it is closed. Callers must not close the returned
 * columns separately unless they first increment their reference counts with
 * {@link ColumnVector#incRefCount()}. A retained reference must then be closed separately.
 * Repeated or concurrent closes do not release either column more than once. A concurrent close
 * can return before another caller finishes cleanup. Reading or retaining columns must not race
 * with closing this result.
 * Both columns are closed even if a close callback throws. The first failure is rethrown and
 * subsequent distinct failures are suppressed.
 */
@Experimental
public final class VariantOperationResult implements AutoCloseable {
  private static final int VALUE_COLUMN_INDEX = 0;
  private static final int STATUS_COLUMN_INDEX = 1;

  private ColumnVector[] columns;

  VariantOperationResult(long[] columnHandles) {
    if (columnHandles == null || columnHandles.length != 2) {
      throw new IllegalArgumentException("a Variant operation result must contain two columns");
    }
    columns = ColumnVector.getColumnVectorsFromPointers(columnHandles);
  }

  /**
   * Return the operation value column.
   *
   * @return the value column, owned by this result
   */
  public ColumnVector getValue() {
    return columns[VALUE_COLUMN_INDEX];
  }

  /**
   * Return the non-nullable UINT8 operation status column.
   *
   * @return the status column, owned by this result
   */
  public ColumnVector getStatus() {
    return columns[STATUS_COLUMN_INDEX];
  }

  @Override
  public void close() {
    // Detach atomically, but never hold the result monitor while invoking column callbacks.
    ColumnVector[] toClose;
    synchronized (this) {
      toClose = columns;
      columns = null;
    }
    if (toClose == null) {
      return;
    }
    Throwable error = null;
    for (ColumnVector column : toClose) {
      try {
        column.close();
      } catch (RuntimeException | Error failure) {
        if (error == null) {
          error = failure;
        } else if (error != failure) {
          error.addSuppressed(failure);
        }
      }
    }
    if (error instanceof RuntimeException) {
      throw (RuntimeException) error;
    } else if (error != null) {
      throw (Error) error;
    }
  }
}

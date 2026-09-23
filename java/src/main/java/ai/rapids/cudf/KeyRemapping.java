/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Remaps keys to unique integer IDs.
 * <p>
 * Each distinct key in the build table is assigned a unique non-negative integer ID.
 * Rows with equal keys will map to the same ID. Keys that cannot be mapped (e.g., not found
 * in probe, or null keys when nulls are unequal) receive negative sentinel values.
 * The specific ID values are stable for the lifetime of this object but are otherwise unspecified.
 * </p>
 * <p>
 * <b>Ownership:</b> This class increments the reference counts on the columns from the provided
 * build keys table. The underlying column data is shared, not copied. When this object is closed,
 * it will decrement those reference counts. The original table passed to the constructor is not
 * affected and the caller retains ownership of it.
 * </p>
 * <p>
 * For advanced memory management (e.g., spilling), use {@link #releaseBuildKeys()} to take
 * ownership of the internal build keys table. After calling this method, the caller is
 * responsible for ensuring the returned table remains valid for the lifetime of this object
 * and for closing it when appropriate.
 * </p>
 * <p>
 * <b>Usage pattern:</b>
 * <pre>{@code
 * try (KeyRemapping remap = new KeyRemapping(buildKeys, true)) {
 *   // Remap build keys (recomputes from cached build table)
 *   try (ColumnVector remappedBuild = remap.remapBuildKeys()) {
 *     // Remap probe keys
 *     try (ColumnVector remappedProbe = remap.remapProbeKeys(probeKeys)) {
 *       // Use remapped integer keys
 *     }
 *   }
 * }
 * }</pre>
 * </p>
 */
public class KeyRemapping implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(KeyRemapping.class);

  /**
   * Sentinel value for probe-side keys not found in build table.
   * <p>
   * This constant is primarily exposed for testing purposes.
   * It must be kept in sync with KEY_REMAP_NOT_FOUND in cudf/join/key_remapping.hpp.
   * </p>
   */
  public static final int NOT_FOUND_SENTINEL = -1;

  /**
   * Sentinel value for build-side rows with null keys (when nulls are not equal).
   * <p>
   * This constant is primarily exposed for testing purposes.
   * It must be kept in sync with KEY_REMAP_RIGHT_NULL in cudf/join/key_remapping.hpp.
   * </p>
   */
  public static final int BUILD_NULL_SENTINEL = -2;

  private static class KeyRemappingCleaner extends MemoryCleaner.Cleaner {
    private Table buildKeys;
    private long nativeHandle;
    private boolean buildKeysReleased = false;

    KeyRemappingCleaner(Table buildKeys, long nativeHandle) {
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
          // Only close buildKeys if it wasn't released to the caller
          if (!buildKeysReleased && buildKeys != null) {
            buildKeys.close();
          }
          buildKeys = null;
        } finally {
          nativeHandle = 0;
        }
        if (logErrorIfNotClean) {
          log.error("A KEY REMAPPING WAS LEAKED (ID: " + id + " " + Long.toHexString(origAddress));
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return nativeHandle == 0;
    }
  }

  private final KeyRemappingCleaner cleaner;
  private final NullEquality nullEquality;
  private final boolean computeMetrics;
  private boolean isClosed = false;

  /**
   * Construct a key remapping structure from build keys.
   * <p>
   * This constructor increments the reference counts on the columns from the provided table,
   * creating a shared reference to the underlying column data. The original table is not
   * affected and the caller retains ownership of it.
   * </p>
   *
   * @param buildKeys table containing the keys to build from. The column reference counts
   *        will be incremented; the caller retains ownership of this table.
   * @param nullEquality how null key values should be compared.
   *        When EQUAL, null keys are treated as equal and assigned a valid non-negative ID.
   *        When UNEQUAL, rows with null keys receive a negative sentinel value.
   * @param computeMetrics if true, compute distinctCount and maxDuplicateCount.
   *        If false, skip metrics computation for better performance; calling
   *        {@link #getDistinctCount()} or {@link #getMaxDuplicateCount()} will throw.
   */
  public KeyRemapping(Table buildKeys, NullEquality nullEquality, boolean computeMetrics) {
    this.nullEquality = nullEquality;
    this.computeMetrics = computeMetrics;
    Table buildTable = new Table(buildKeys.getColumns());
    try {
      long handle = create(buildTable.getNativeView(), nullEquality.nullsEqual, computeMetrics);
      this.cleaner = new KeyRemappingCleaner(buildTable, handle);
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

  /**
   * Construct a key remapping structure from build keys with metrics computation enabled.
   *
   * @param buildKeys table containing the keys to build from
   * @param nullEquality how null key values should be compared
   */
  public KeyRemapping(Table buildKeys, NullEquality nullEquality) {
    this(buildKeys, nullEquality, true);
  }

  /**
   * Construct a key remapping structure from build keys with nulls comparing equal
   * and metrics computation enabled.
   *
   * @param buildKeys table containing the keys to build from
   */
  public KeyRemapping(Table buildKeys) {
    this(buildKeys, NullEquality.EQUAL, true);
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

  /**
   * Get the native handle to the key remapping structure.
   * <p><b>Internal use only.</b></p>
   */
  long getNativeHandle() {
    if (isClosed) {
      throw new IllegalStateException("KeyRemapping is already closed");
    }
    return cleaner.nativeHandle;
  }

  /**
   * Returns the null equality setting used when building the hash table.
   *
   * @return the NullEquality setting
   */
  public NullEquality getNullEquality() {
    return nullEquality;
  }

  /**
   * Check if metrics (distinctCount, maxDuplicateCount) were computed.
   *
   * @return true if metrics are available, false if computeMetrics was false during construction
   */
  public boolean hasMetrics() {
    return computeMetrics;
  }

  /**
   * Get the number of distinct keys in the build table.
   *
   * @return The count of unique key combinations found during build
   * @throws IllegalStateException if computeMetrics was false during construction
   */
  public int getDistinctCount() {
    if (isClosed) {
      throw new IllegalStateException("KeyRemapping is already closed");
    }
    return getDistinctCount(cleaner.nativeHandle);
  }

  /**
   * Get the maximum number of times any single key appears in the build table.
   *
   * @return The maximum duplicate count across all distinct keys
   * @throws IllegalStateException if computeMetrics was false during construction
   */
  public int getMaxDuplicateCount() {
    if (isClosed) {
      throw new IllegalStateException("KeyRemapping is already closed");
    }
    return getMaxDuplicateCount(cleaner.nativeHandle);
  }

  /**
   * Release ownership of the internal build keys table to the caller.
   * <p>
   * <b>Advanced API for memory management (e.g., spilling).</b>
   * </p>
   * <p>
   * After calling this method:
   * <ul>
   *   <li>The caller owns the returned Table and is responsible for closing it</li>
   *   <li>The caller must ensure the returned Table remains valid (not closed, not spilled)
   *       for as long as this KeyRemapping object is in use</li>
   *   <li>When this KeyRemapping is closed, it will NOT close the build keys table</li>
   *   <li>This method can only be called once; subsequent calls will throw an exception</li>
   * </ul>
   * </p>
   * <p>
   * This is useful for scenarios where the caller wants to manage the build keys table
   * separately, such as spilling it to disk and restoring it later, while keeping the
   * native hash table alive.
   * </p>
   *
   * @return The build keys Table. The caller takes ownership and must close it when done.
   * @throws IllegalStateException if already closed or if build keys were already released
   */
  public synchronized Table releaseBuildKeys() {
    if (isClosed) {
      throw new IllegalStateException("KeyRemapping is already closed");
    }
    if (cleaner.buildKeysReleased) {
      throw new IllegalStateException("Build keys have already been released");
    }
    if (cleaner.buildKeys == null) {
      throw new IllegalStateException("Build keys are not available");
    }
    cleaner.buildKeysReleased = true;
    return cleaner.buildKeys;
  }

  /**
   * Check if the build keys have been released via {@link #releaseBuildKeys()}.
   *
   * @return true if build keys have been released, false otherwise
   */
  public boolean isBuildKeysReleased() {
    return cleaner.buildKeysReleased;
  }

  /**
   * Remap build keys to integer IDs.
   * <p>
   * Recomputes the remapped build table from the cached build keys. This does not cache
   * the remapped table; each call will recompute it from the key remapping.
   * </p>
   * <p>
   * For each row in the cached build table, returns the integer ID assigned to that key.
   * Non-negative integers represent valid mapped keys, while negative values represent
   * keys that cannot be mapped (e.g., null keys when nulls are unequal).
   * </p>
   *
   * @return A column of INT32 values with the remapped key IDs (caller must close)
   */
  public ColumnVector remapBuildKeys() {
    if (isClosed) {
      throw new IllegalStateException("KeyRemapping is already closed");
    }
    return new ColumnVector(remapBuildKeys(cleaner.nativeHandle));
  }

  /**
   * Remap probe keys to integer IDs.
   * <p>
   * For each row in the input, returns the integer ID assigned to that key.
   * The keys table must have the same schema (number and types of columns) as
   * the build table used to construct this object.
   * </p>
   * <p>
   * Non-negative integers represent keys found in the build table, while negative values
   * represent keys that were not found or cannot be matched (e.g., null keys when nulls
   * are unequal, or keys not present in the build table).
   * </p>
   *
   * @param keys The probe keys to remap (must have same schema as build table)
   * @return A column of INT32 values with the remapped key IDs (caller must close)
   * @throws IllegalArgumentException if keys has different number of columns than build table
   * @throws CudfException if keys has different column types than build table
   */
  public ColumnVector remapProbeKeys(Table keys) {
    if (isClosed) {
      throw new IllegalStateException("KeyRemapping is already closed");
    }
    return new ColumnVector(remapProbeKeys(cleaner.nativeHandle, keys.getNativeView()));
  }

  // Native methods
  private static native long create(long tableView, boolean compareNulls, boolean computeMetrics);
  private static native void destroy(long handle);
  private static native int getDistinctCount(long handle);
  private static native int getMaxDuplicateCount(long handle);
  private static native long remapBuildKeys(long handle);
  private static native long remapProbeKeys(long handle, long keysTableView);
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import ai.rapids.cudf.ast.CompiledExpression;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * Experimental Parquet hybrid-scan reader.
 *
 * <p>This class is the Java binding for
 * {@code cudf::io::parquet::experimental::hybrid_scan_reader}. It is designed for highly
 * selective filter expressions over Parquet files and exposes the multi-step pipeline that
 * the C++ reader uses internally:
 * <ol>
 *   <li>Read the Parquet footer (and optional page index) to drive row-group / page-level
 *       pruning.</li>
 *   <li>Filter the row groups using statistics, bloom filters, and dictionary pages.</li>
 *   <li>Materialize the <em>filter</em> columns; the reader builds an initial row mask
 *       (all true or seeded from page index stats when page-level pruning is enabled)
 *       and then mutates it down to only the rows that survive the AST filter.</li>
 *   <li>Materialize the <em>payload</em> columns using the surviving row mask.</li>
 * </ol>
 *
 * <p>A two-step convenience flow is provided for selective filters
 * ({@link #materializeFilterColumns(int[], DeviceMemoryBuffer[], boolean)} then
 * {@link #materializePayloadColumns(int[], DeviceMemoryBuffer[], ColumnVector, boolean)}),
 * and a one-shot {@link #materializeAllColumns(int[], DeviceMemoryBuffer[])} is provided
 * for small files or broad filters.
 *
 * <p>Chunked / streaming materialization is also supported via the
 * {@code setupChunkingFor*} + {@code materialize*Chunk} family of methods, mirroring the C++
 * chunked reader pipeline.
 *
 * <p>The filter and payload materialization paths accept a boolean that toggles
 * page-level pruning: skips decode of pages the filter (or row mask) proves empty, in
 * exchange for a per-page stats scan and a carried row-mask column. Enable when the
 * workload prunes many pages; requires prior {@link #setupPageIndex(HostMemoryBuffer)} to
 * prune filter column pages using page-level statistics.
 *
 * <p>The reader is created with no filter expression installed. Filter-related APIs
 * behave as though nothing has been filtered out unless a filter is first supplied via
 * {@link #setFilter(CompiledExpression)}.
 *
 * <p>The APIs in this file are experimental and subject to change.
 */
@Experimental
public class HybridScanReader implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  /**
   * The result of a combined row-mask-build + filter-column-materialization call.
   *
   * <p>Both the filter {@link Table} and the mutated row {@link ColumnVector} mask are
   * owned by this object. Close via try-with-resources to release both.
   */
  public static final class FilterMaterializationResult implements AutoCloseable {
    private final Table table;
    private final ColumnVector rowMask;
    private boolean closed = false;

    FilterMaterializationResult(Table table, ColumnVector rowMask) {
      this.table = table;
      this.rowMask = rowMask;
    }

    /** @return the materialized filter column table. */
    public Table table() { return table; }

    /** @return the (mutated) row mask after the filter expression was applied. */
    public ColumnVector rowMask() { return rowMask; }

    @Override
    public synchronized void close() {
      if (closed) return;
      try { table.close(); } finally {
        try { rowMask.close(); } finally { closed = true; }
      }
    }
  }

  private static final Logger log = LoggerFactory.getLogger(HybridScanReader.class);

  /**
   * Subclasses {@link MemoryCleaner.Cleaner} (the standard cudf-java pattern for native-backed
   * resources) so the native reader is destroyed exactly once on {@link #close()} and is
   * reported as a leak if reclaimed by the garbage collector without being closed.
   */
  private static final class HybridScanReaderCleaner extends MemoryCleaner.Cleaner {
    private long nativeHandle;

    HybridScanReaderCleaner(long nativeHandle) {
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
          log.error("A HYBRID SCAN READER WAS LEAKED (ID: {} {})", id,
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

  private final HybridScanReaderCleaner cleaner;

  private CompiledExpression filter = null;
  private boolean isClosed = false;

  /**
   * Create a hybrid scan reader from the bytes of a Parquet file footer.
   *
   * <p>The {@code footerBuffer} can be obtained by reading the last few bytes of a Parquet
   * file. See the {@code hybrid_scan_io} example for a helper that does exactly that.
   *
   * <p>The reader is created with no filter expression installed. Filter-related APIs
   * (e.g. {@link #filterRowGroupsWithStats(int[])},
   * {@link #materializeFilterColumns(int[], DeviceMemoryBuffer[], boolean)}) behave as
   * though nothing has been filtered out unless a filter is first supplied via
   * {@link #setFilter(CompiledExpression)}.
   *
   * @param footerBuffer host-resident footer bytes (must remain valid until this constructor
   *                     returns; the JNI reads the bytes synchronously)
   * @param opts         Parquet reader options. {@link ParquetOptions#DEFAULT} by default.
   */
  public HybridScanReader(HostMemoryBuffer footerBuffer, ParquetOptions opts) {
    if (footerBuffer == null) {
      throw new IllegalArgumentException("footerBuffer must not be null");
    }
    if (opts == null) {
      opts = ParquetOptions.DEFAULT;
    }
    String[] columnNames = opts.getIncludeColumnNames();
    boolean[] readBinaryAsString = opts.getReadBinaryAsString();
    DType timeUnit = opts.timeUnit();
    long handle = createFromFooter(
        footerBuffer.getAddress(),
        footerBuffer.getLength(),
        columnNames,
        readBinaryAsString,
        timeUnit.getTypeId().getNativeId());
    this.cleaner = new HybridScanReaderCleaner(handle);
    MemoryCleaner.register(this, cleaner);
    cleaner.addRef();
  }

  /**
   * Install or replace the filter expression used by all subsequent filter-related APIs on
   * this reader; pass {@code null} to clear. Cheap to call, so callers can sweep the same
   * file with several candidate predicates in a loop.
   *
   * <p>The caller retains ownership of any previously installed {@code CompiledExpression}
   * and must close it themselves — this method replaces this reader's strong reference only.
   *
   * @param filter the new filter expression, or {@code null} to clear
   */
  public void setFilter(CompiledExpression filter) {
    assertNotClosed();
    long filterHandle = (filter != null) ? filter.getNativeHandle() : 0;
    setFilter(cleaner.nativeHandle, filterHandle);
    this.filter = filter;
  }

  // ----------------------------------------------------------------------
  // Metadata
  // ----------------------------------------------------------------------

  /** @return the byte range of the page index in the Parquet file. */
  public ByteRange pageIndexByteRange() {
    assertNotClosed();
    long[] offsetSize = pageIndexByteRange(cleaner.nativeHandle);
    return new ByteRange(offsetSize[0], offsetSize[1]);
  }

  /**
   * Materialize the {@code ColumnIndex} / {@code OffsetIndex} structs (collectively, the page
   * index) from the supplied bytes. Required before any filter or payload column materialization
   * call with {@code usePageLevelPruning == true}.
   *
   * @param pageIndexBuffer host-resident page index bytes
   */
  public void setupPageIndex(HostMemoryBuffer pageIndexBuffer) {
    assertNotClosed();
    if (pageIndexBuffer == null) {
      throw new IllegalArgumentException("pageIndexBuffer must not be null");
    }
    setupPageIndex(cleaner.nativeHandle,
        pageIndexBuffer.getAddress(),
        pageIndexBuffer.getLength());
  }

  // ----------------------------------------------------------------------
  // Row group enumeration
  // ----------------------------------------------------------------------

  /** @return all row group indices in the Parquet file. */
  public int[] allRowGroups() {
    assertNotClosed();
    return allRowGroups(cleaner.nativeHandle);
  }

  /** @return the total number of top-level rows in the supplied row groups. */
  public long totalRowsInRowGroups(int[] rowGroupIndices) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    return totalRowsInRowGroups(cleaner.nativeHandle, rowGroupIndices);
  }

  // ----------------------------------------------------------------------
  // Row group filtering
  // ----------------------------------------------------------------------

  /** Filter row groups using column-chunk statistics from the file footer. */
  public int[] filterRowGroupsWithStats(int[] rowGroupIndices) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    return filterRowGroupsWithStats(cleaner.nativeHandle, rowGroupIndices);
  }

  /**
   * Get the byte ranges in the source file that hold the bloom filter and dictionary page
   * data needed for the next round of row-group pruning.
   */
  public SecondaryFilterRanges secondaryFiltersByteRanges(int[] rowGroupIndices) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    long[] packed = secondaryFiltersByteRanges(cleaner.nativeHandle, rowGroupIndices);
    // Layout: [numBloomRanges, bloom_o0, bloom_s0, ..., dict_o0, dict_s0, ...]
    int numBloom = (int) packed[0];
    int totalRanges = (packed.length - 1) / 2;
    int numDict = totalRanges - numBloom;
    ByteRange[] bloom = new ByteRange[numBloom];
    ByteRange[] dict = new ByteRange[numDict];
    int idx = 1;
    for (int i = 0; i < numBloom; i++) {
      bloom[i] = new ByteRange(packed[idx], packed[idx + 1]);
      idx += 2;
    }
    for (int i = 0; i < numDict; i++) {
      dict[i] = new ByteRange(packed[idx], packed[idx + 1]);
      idx += 2;
    }
    return new SecondaryFilterRanges(bloom, dict);
  }

  // TODO: add filterRowGroupsWithBloomFilters(int[] rowGroups) once the Java Parquet
  //       writer can emit bloom filter blocks. The C++ writer exposes this via
  //       parquet_writer_options::set_column_chunks_bloom_filter_params, which has not
  //       yet been surfaced in ParquetWriterOptions / TableJni.cpp. Until then, any
  //       file written from Java contains no bloom filter blocks and the method would
  //       always return the input row groups unchanged.

  /** Filter row groups using column-chunk dictionary pages loaded into device memory. */
  public int[] filterRowGroupsWithDictionaryPages(DeviceMemoryBuffer[] dictionaryPageData,
                                                  int[] rowGroupIndices) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    if (dictionaryPageData == null) {
      throw new IllegalArgumentException("dictionaryPageData must not be null");
    }
    long[] addrs = new long[dictionaryPageData.length];
    long[] lens = new long[dictionaryPageData.length];
    for (int i = 0; i < dictionaryPageData.length; i++) {
      addrs[i] = dictionaryPageData[i].getAddress();
      lens[i] = dictionaryPageData[i].getLength();
    }
    return filterRowGroupsWithDictionaryPages(cleaner.nativeHandle, addrs, lens,
        rowGroupIndices);
  }

  // ----------------------------------------------------------------------
  // Byte ranges
  // ----------------------------------------------------------------------

  /** @return byte ranges for the column chunks of <em>filter</em> columns. */
  public ByteRange[] filterColumnChunksByteRanges(int[] rowGroupIndices) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    return decodeRanges(filterColumnChunksByteRanges(cleaner.nativeHandle, rowGroupIndices));
  }

  /**
   * @return byte ranges for the column chunks of <em>payload</em> columns.
   *
   * <p>This result is order-dependent. If filter columns have already been processed on this
   * reader (e.g. via {@link #filterColumnChunksByteRanges(int[])} or
   * {@link #materializeFilterColumns(int[], DeviceMemoryBuffer[], boolean)}),
   * the filter columns are excluded and only the payload columns are returned. If filter columns
   * have not yet been processed, nothing is excluded and the ranges cover the full set of columns
   * that would be read i.e. the columns projected via {@link ParquetOptions}, or all columns in
   * the file when no projection was set.
   */
  public ByteRange[] payloadColumnChunksByteRanges(int[] rowGroupIndices) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    return decodeRanges(payloadColumnChunksByteRanges(cleaner.nativeHandle, rowGroupIndices));
  }

  /** @return byte ranges for all column chunks (filter + payload) of the selected columns. */
  public ByteRange[] allColumnChunksByteRanges(int[] rowGroupIndices) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    return decodeRanges(allColumnChunksByteRanges(cleaner.nativeHandle, rowGroupIndices));
  }

  // ----------------------------------------------------------------------
  // Two-step materialize (filter + payload)
  // ----------------------------------------------------------------------

  /**
   * Build the initial row mask, materialize the filter columns, and evaluate the compiled
   * filter expression against them. The row mask and the resulting filter table are returned
   * together as a {@link FilterMaterializationResult}; close it via try-with-resources.
   *
   * <p>Set {@code usePageLevelPruning = true} to skip decompression and decode of pages
   * the filter proves empty. Requires prior {@link #setupPageIndex(HostMemoryBuffer)}.
   *
   * <p>Cost: a per-page stats scan of filter columns and a carried row-mask column.
   * Payoff: pruned pages are skipped entirely, typically the dominant read cost. Enable
   * when a meaningful fraction of pages can be pruned; otherwise leave {@code false}.
   *
   * @param rowGroupIndices        row groups to read
   * @param columnChunkData        device buffers holding the filter column chunks, in the
   *                               order returned by {@link #filterColumnChunksByteRanges(int[])}
   * @param usePageLevelPruning    seed the row mask from page-index stats and enable the
   *                               data page mask; requires prior
   *                               {@link #setupPageIndex(HostMemoryBuffer)}
   * @return combined filter table and mutated row mask; caller must close this result
   */
  public FilterMaterializationResult materializeFilterColumns(int[] rowGroupIndices,
                                                              DeviceMemoryBuffer[] columnChunkData,
                                                              boolean usePageLevelPruning) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    long[] addrs = bufferAddrs(columnChunkData);
    long[] lens = bufferLens(columnChunkData);
    long[] handles = materializeFilterColumns(cleaner.nativeHandle, rowGroupIndices,
        addrs, lens, usePageLevelPruning);
    ColumnVector rowMask = new ColumnVector(handles[0]);
    try {
      long[] tableHandles = Arrays.copyOfRange(handles, 1, handles.length);
      Table table = new Table(tableHandles);
      return new FilterMaterializationResult(table, rowMask);
    } catch (Throwable t) {
      rowMask.close();
      throw t;
    }
  }

  /**
   * Materialize only the payload columns, applying the supplied row mask to the output.
   *
   * <p>Set {@code usePageLevelPruning = true} to skip decompression and decode of pages
   * containing no rows surviving {@code rowMask}. Adds a small mask-build cost; wins
   * when {@code rowMask} prunes a meaningful fraction of pages.
   *
   * @param rowGroupIndices  row groups to read
   * @param columnChunkData  device buffers holding the payload column chunks, in the order
   *                         returned by {@link #payloadColumnChunksByteRanges(int[])}
   * @param rowMask          row mask (read-only)
   * @param usePageLevelPruning  enable the data page mask to skip decode of pages the row
   *                             mask proves empty; requires prior
   *                             {@link #setupPageIndex(HostMemoryBuffer)} to avoid fall back path
   * @return the materialized payload column table
   */
  public Table materializePayloadColumns(int[] rowGroupIndices,
                                         DeviceMemoryBuffer[] columnChunkData,
                                         ColumnVector rowMask,
                                         boolean usePageLevelPruning) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    requireNonNullRowMask(rowMask);
    long[] addrs = bufferAddrs(columnChunkData);
    long[] lens = bufferLens(columnChunkData);
    long[] handles = materializePayloadColumns(cleaner.nativeHandle, rowGroupIndices,
        addrs, lens, rowMask.getNativeView(), usePageLevelPruning);
    return new Table(handles);
  }

  // ----------------------------------------------------------------------
  // One-shot materialize (all columns)
  // ----------------------------------------------------------------------

  /**
   * Materialize all (or selected) columns in a single pass. The filter expression is applied
   * <em>after</em> reading, so this is most efficient for small files or when most rows
   * survive the filter. For highly-selective filters, prefer the explicit two-step flow.
   */
  public Table materializeAllColumns(int[] rowGroupIndices,
                                     DeviceMemoryBuffer[] columnChunkData) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    long[] addrs = bufferAddrs(columnChunkData);
    long[] lens = bufferLens(columnChunkData);
    long[] handles = materializeAllColumns(cleaner.nativeHandle, rowGroupIndices, addrs, lens);
    return new Table(handles);
  }

  // ----------------------------------------------------------------------
  // Chunked materialize
  // ----------------------------------------------------------------------

  /**
   * Build the initial row mask and set up chunking state for filter-column materialization.
   * The row mask is owned internally by the reader for the duration of the chunked filter
   * pipeline; subsequent calls to {@link #materializeFilterColumnsChunk()} mutate it in
   * place. After all chunks are drained, call {@link #takeFilterRowMask()} to transfer
   * ownership of the row mask to the caller (typically to feed it into
   * {@link #materializePayloadColumns(int[], DeviceMemoryBuffer[], ColumnVector, boolean)}).
   *
   * <p>Set {@code usePageLevelPruning = true} to skip decompression and decode of pages
   * the filter proves empty. Requires prior {@link #setupPageIndex(HostMemoryBuffer)}.
   *
   * <p>Cost: a per-page stats scan of filter columns and a carried row-mask column.
   * Payoff: pruned pages are skipped entirely, typically the dominant read cost. Enable
   * when a meaningful fraction of pages can be pruned; otherwise leave {@code false}.
   *
   * <p>Calling this method again before {@link #takeFilterRowMask()} discards the previous
   * chunked-filter row mask.
   *
   * <p>Caller must keep {@code columnChunkData} open until {@link #takeFilterRowMask()} (or
   * a re-setup); the native reader holds references to it across chunk calls.
   *
   * @param chunkReadLimit         soft limit (in bytes) on each output chunk returned by
   *                               the matching {@code materialize*Chunk} call, or 0 for no limit
   * @param passReadLimit          soft limit (in bytes) on the working memory used to
   *                               decompress and decode a single subpass (a page-level slice
   *                               of the selected row groups), or 0 for no limit. Hybrid scan
   *                               hardcodes the number of row-group passes to 1, so this
   *                               parameter only bounds the subpass (decode) working set;
   *                               it does not repartition row groups. To split row groups
   *                               across multiple passes up front, call
   *                               {@link #constructRowGroupPasses(int[], long)} first and
   *                               issue a separate chunked run per returned partition.
   * @param rowGroupIndices        row groups to read
   * @param usePageLevelPruning    seed the row mask from page-index stats and enable the
   *                               data page mask; requires prior
   *                               {@link #setupPageIndex(HostMemoryBuffer)}
   * @param columnChunkData        device buffers holding the filter column chunks, in the
   *                               order returned by {@link #filterColumnChunksByteRanges(int[])}
   */
  public void setupChunkingForFilterColumns(long chunkReadLimit,
                                            long passReadLimit,
                                            int[] rowGroupIndices,
                                            boolean usePageLevelPruning,
                                            DeviceMemoryBuffer[] columnChunkData) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    long[] addrs = bufferAddrs(columnChunkData);
    long[] lens = bufferLens(columnChunkData);
    setupChunkingForFilterColumns(cleaner.nativeHandle,
        chunkReadLimit, passReadLimit, rowGroupIndices, usePageLevelPruning,
        addrs, lens);
  }

  /**
   * @return the next filter-column chunk; throws if no chunk is available or if no chunked
   *         filter pipeline is active. The internal row mask owned by this reader is
   *         updated in place.
   */
  public Table materializeFilterColumnsChunk() {
    assertNotClosed();
    long[] handles = materializeFilterColumnsChunk(cleaner.nativeHandle);
    return new Table(handles);
  }

  /**
   * Transfer ownership of the row mask produced by the chunked filter-column materialization
   * to the caller. Typically called after all {@link #materializeFilterColumnsChunk()} calls
   * complete and immediately before
   * {@link #materializePayloadColumns(int[], DeviceMemoryBuffer[], ColumnVector, boolean)}.
   *
   * <p>After this call the reader has no chunked-filter row mask; subsequent
   * {@link #materializeFilterColumnsChunk()} calls will fail until
   * {@link #setupChunkingForFilterColumns(long, long, int[], boolean, DeviceMemoryBuffer[])}
   * is invoked again. If never called, the row mask is freed when the reader is closed.
   *
   * @return the owned row mask {@link ColumnVector}; caller must close it.
   */
  public ColumnVector takeFilterRowMask() {
    assertNotClosed();
    return new ColumnVector(takeFilterRowMask(cleaner.nativeHandle));
  }

  /**
   * Set up chunking state for payload-column materialization. Subsequent calls to
   * {@link #materializePayloadColumnsChunk(ColumnVector)} will yield successive chunks.
   *
   * <p>Set {@code usePageLevelPruning = true} to skip decompression and decode of pages
   * containing no rows surviving {@code rowMask}. Adds a small mask-build cost; wins
   * when {@code rowMask} prunes a meaningful fraction of pages.
   *
   * <p>Caller must keep {@code columnChunkData} open until {@link #hasNextTableChunk()}
   * returns {@code false}; the native reader holds references to it across chunk calls.
   *
   * @param chunkReadLimit   soft limit (in bytes) on each output chunk returned by the
   *                         matching {@code materialize*Chunk} call, or 0 for no limit
   * @param passReadLimit    soft limit (in bytes) on the working memory used to decompress
   *                         and decode a single subpass (a page-level slice of the selected
   *                         row groups), or 0 for no limit. Hybrid scan hardcodes the
   *                         number of row-group passes to 1, so this parameter only bounds
   *                         the subpass (decode) working set; it does not repartition row
   *                         groups. To split row groups across multiple passes up front,
   *                         call {@link #constructRowGroupPasses(int[], long)} first and
   *                         issue a separate chunked run per returned partition.
   * @param rowGroupIndices  row groups to read
   * @param rowMask          row mask (read-only)
   * @param usePageLevelPruning  enable the data page mask to skip decode of pages the row
   *                             mask proves empty; requires prior
   *                             {@link #setupPageIndex(HostMemoryBuffer)} to avoid fall back path
   * @param columnChunkData  device buffers holding the payload column chunks, in the order
   *                         returned by {@link #payloadColumnChunksByteRanges(int[])}
   */
  public void setupChunkingForPayloadColumns(long chunkReadLimit,
                                             long passReadLimit,
                                             int[] rowGroupIndices,
                                             ColumnVector rowMask,
                                             boolean usePageLevelPruning,
                                             DeviceMemoryBuffer[] columnChunkData) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    requireNonNullRowMask(rowMask);
    long[] addrs = bufferAddrs(columnChunkData);
    long[] lens = bufferLens(columnChunkData);
    setupChunkingForPayloadColumns(cleaner.nativeHandle, chunkReadLimit, passReadLimit,
        rowGroupIndices, rowMask.getNativeView(), usePageLevelPruning, addrs, lens);
  }

  /** @return the next payload-column chunk; throws if no chunk is available. */
  public Table materializePayloadColumnsChunk(ColumnVector rowMask) {
    assertNotClosed();
    requireNonNullRowMask(rowMask);
    long[] handles = materializePayloadColumnsChunk(cleaner.nativeHandle,
        rowMask.getNativeView());
    return new Table(handles);
  }

  /**
   * Set up chunking state for all-column materialization (single-pass mode). Subsequent calls
   * to {@link #materializeAllColumnsChunk()} will yield successive chunks.
   *
   * <p>Caller must keep {@code columnChunkData} open until {@link #hasNextTableChunk()}
   * returns {@code false}; the native reader holds references to it across chunk calls.
   *
   * @param chunkReadLimit   soft limit (in bytes) on each output chunk returned by the
   *                         matching {@code materialize*Chunk} call, or 0 for no limit
   * @param passReadLimit    soft limit (in bytes) on the working memory used to decompress
   *                         and decode a single subpass (a page-level slice of the selected
   *                         row groups), or 0 for no limit. Hybrid scan hardcodes the
   *                         number of row-group passes to 1, so this parameter only bounds
   *                         the subpass (decode) working set; it does not repartition row
   *                         groups. To split row groups across multiple passes up front,
   *                         call {@link #constructRowGroupPasses(int[], long)} first and
   *                         issue a separate chunked run per returned partition.
   * @param rowGroupIndices  row groups to read
   * @param columnChunkData  device buffers holding all column chunks, in the order returned
   *                         by {@link #allColumnChunksByteRanges(int[])}
   */
  public void setupChunkingForAllColumns(long chunkReadLimit,
                                         long passReadLimit,
                                         int[] rowGroupIndices,
                                         DeviceMemoryBuffer[] columnChunkData) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    long[] addrs = bufferAddrs(columnChunkData);
    long[] lens = bufferLens(columnChunkData);
    setupChunkingForAllColumns(cleaner.nativeHandle, chunkReadLimit, passReadLimit,
        rowGroupIndices, addrs, lens);
  }

  /** @return the next all-columns chunk; throws if no chunk is available. */
  public Table materializeAllColumnsChunk() {
    assertNotClosed();
    long[] handles = materializeAllColumnsChunk(cleaner.nativeHandle);
    return new Table(handles);
  }

  /** @return {@code true} when a subsequent {@code materialize*Chunk} call would return data. */
  public boolean hasNextTableChunk() {
    assertNotClosed();
    return hasNextTableChunk(cleaner.nativeHandle);
  }

  /**
   * Partition the supplied row groups into passes whose total uncompressed size respects the
   * given limit. The returned array contains one inner array per pass.
   *
   * @param rowGroupIndices row groups to partition
   * @param passReadLimit   limit on the memory used by a single pass, or 0 for no limit.
   *                        Each returned pass can then be fed to a
   *                        {@code setupChunkingFor*} call, which will further stream that
   *                        pass in subpass-sized chunks bounded by its own
   *                        {@code passReadLimit} argument.
   * @return an array of arrays of row group indices, one per pass
   */
  public int[][] constructRowGroupPasses(int[] rowGroupIndices, long passReadLimit) {
    assertNotClosed();
    requireNonNullRowGroups(rowGroupIndices);
    return constructRowGroupPasses(cleaner.nativeHandle, rowGroupIndices, passReadLimit);
  }

  // ----------------------------------------------------------------------
  // Lifecycle
  // ----------------------------------------------------------------------

  @Override
  public synchronized void close() {
    if (isClosed) {
      return;
    }
    cleaner.delRef();
    cleaner.clean(false);
    isClosed = true;
  }

  // ----------------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------------

  private void assertNotClosed() {
    if (isClosed) {
      throw new IllegalStateException("HybridScanReader has been closed");
    }
  }

  private static void requireNonNullRowGroups(int[] rowGroupIndices) {
    if (rowGroupIndices == null) {
      throw new IllegalArgumentException("rowGroupIndices must not be null");
    }
  }

  private static void requireNonNullRowMask(ColumnVector rowMask) {
    if (rowMask == null) {
      throw new IllegalArgumentException("rowMask must not be null");
    }
  }

  private static long[] bufferAddrs(DeviceMemoryBuffer[] buffers) {
    if (buffers == null) {
      throw new IllegalArgumentException("buffers must not be null");
    }
    long[] addrs = new long[buffers.length];
    for (int i = 0; i < buffers.length; i++) {
      if (buffers[i] == null) {
        throw new IllegalArgumentException("buffers[" + i + "] must not be null");
      }
      addrs[i] = buffers[i].getAddress();
    }
    return addrs;
  }

  private static long[] bufferLens(DeviceMemoryBuffer[] buffers) {
    if (buffers == null) {
      throw new IllegalArgumentException("buffers must not be null");
    }
    long[] lens = new long[buffers.length];
    for (int i = 0; i < buffers.length; i++) {
      if (buffers[i] == null) {
        throw new IllegalArgumentException("buffers[" + i + "] must not be null");
      }
      lens[i] = buffers[i].getLength();
    }
    return lens;
  }

  private static ByteRange[] decodeRanges(long[] packed) {
    if (packed == null || packed.length == 0) {
      return new ByteRange[0];
    }
    int n = packed.length / 2;
    ByteRange[] out = new ByteRange[n];
    for (int i = 0; i < n; i++) {
      out[i] = new ByteRange(packed[2 * i], packed[2 * i + 1]);
    }
    return out;
  }

  // ----------------------------------------------------------------------
  // Native methods
  // ----------------------------------------------------------------------

  private static native long createFromFooter(long footerAddress,
                                              long footerLength,
                                              String[] columnNames,
                                              boolean[] readBinaryAsString,
                                              int timeUnitTypeId);

  private static native void setFilter(long handle, long filterHandle);

  private static native void destroy(long handle);

  // Metadata
  private static native long[] pageIndexByteRange(long handle);
  private static native void setupPageIndex(long handle, long bufferAddress, long bufferLength);

  // Row group enumeration
  private static native int[] allRowGroups(long handle);
  private static native long totalRowsInRowGroups(long handle, int[] rowGroupIndices);

  // Filtering
  private static native int[] filterRowGroupsWithStats(long handle, int[] rowGroupIndices);
  private static native long[] secondaryFiltersByteRanges(long handle, int[] rowGroupIndices);
  private static native int[] filterRowGroupsWithDictionaryPages(long handle,
                                                                 long[] bufferAddresses,
                                                                 long[] bufferLengths,
                                                                 int[] rowGroupIndices);

  // Byte ranges
  private static native long[] filterColumnChunksByteRanges(long handle, int[] rowGroupIndices);
  private static native long[] payloadColumnChunksByteRanges(long handle, int[] rowGroupIndices);
  private static native long[] allColumnChunksByteRanges(long handle, int[] rowGroupIndices);

  // Two-step materialize (filter + payload)
  // Returns: [row_mask_col_handle, table_col0_handle, ..., table_colN_handle]
  private static native long[] materializeFilterColumns(long handle,
                                                        int[] rowGroupIndices,
                                                        long[] bufferAddresses,
                                                        long[] bufferLengths,
                                                        boolean usePageLevelPruning);
  private static native long[] materializePayloadColumns(long handle,
                                                         int[] rowGroupIndices,
                                                         long[] bufferAddresses,
                                                         long[] bufferLengths,
                                                         long rowMaskViewHandle,
                                                         boolean usePageLevelPruning);

  // One-shot materialize (all columns)
  private static native long[] materializeAllColumns(long handle,
                                                     int[] rowGroupIndices,
                                                     long[] bufferAddresses,
                                                     long[] bufferLengths);

  // Chunked
  // Builds the row mask, sets up chunking state, and stores the owned row mask column on
  // the C++ wrapper. Subsequent materializeFilterColumnsChunk calls mutate that column in
  // place; takeFilterRowMask transfers it out to Java.
  private static native void setupChunkingForFilterColumns(long handle,
                                                           long chunkReadLimit,
                                                           long passReadLimit,
                                                           int[] rowGroupIndices,
                                                           boolean usePageLevelPruning,
                                                           long[] bufferAddresses,
                                                           long[] bufferLengths);
  private static native long[] materializeFilterColumnsChunk(long handle);
  private static native long takeFilterRowMask(long handle);
  private static native void setupChunkingForPayloadColumns(long handle,
                                                            long chunkReadLimit,
                                                            long passReadLimit,
                                                            int[] rowGroupIndices,
                                                            long rowMaskViewHandle,
                                                            boolean usePageLevelPruning,
                                                            long[] bufferAddresses,
                                                            long[] bufferLengths);
  private static native long[] materializePayloadColumnsChunk(long handle,
                                                              long rowMaskViewHandle);
  private static native void setupChunkingForAllColumns(long handle,
                                                        long chunkReadLimit,
                                                        long passReadLimit,
                                                        int[] rowGroupIndices,
                                                        long[] bufferAddresses,
                                                        long[] bufferLengths);
  private static native long[] materializeAllColumnsChunk(long handle);
  private static native boolean hasNextTableChunk(long handle);
  private static native int[][] constructRowGroupPasses(long handle,
                                                        int[] rowGroupIndices,
                                                        long passReadLimit);
}

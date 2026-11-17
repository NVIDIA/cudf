/*
 *
 *  SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 *  SPDX-License-Identifier: Apache-2.0
 *
 */

package ai.rapids.cudf;

import java.io.BufferedOutputStream;
import java.io.Closeable;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

/**
 * Serialize and deserialize CUDF tables and columns using a custom format.  The goal of this is
 * to provide a way to efficiently serialize and deserialize cudf data for distributed
 * processing within a single application. Typically after a partition like operation has happened.
 * It is not intended for inter-application communication or for long term storage of data, there
 * are much better standards based formats for all of that.
 * <p>
 * The goal is to transfer data from a local GPU to a remote GPU as quickly and efficiently as
 * possible using build in java communication channels.  There is no guarantee of compatibility
 * between different releases of CUDF.  This is to allow us to adapt if internal memory layouts
 * and formats change.
 * <p>
 * This version optimizes for reduced memory transfers, and as such will try to do the fewest number
 * of transfers possible when putting the data back onto the GPU.  This means that it will slice
 * a single large memory buffer into smaller buffers used by the resulting ColumnVectors.  The
 * downside of this is that generally none of the memory can be released until all of the
 * ColumnVectors are closed.  It is assumed that this will not be a problem because for processing
 * efficiency after the data is transferred it will likely be combined with other similar batches
 * from other processes into a single larger buffer.
 */
public class JCudfSerialization {
  /**
   * Magic number "CUDF" in ASCII, which is 1178883395 if read in LE from big endian, which is
   * too large for any reasonable metadata for arrow, so we should probably be okay detecting
   * this, and switching back/forth at a later time.
   */
  private static final int SER_FORMAT_MAGIC_NUMBER = 0x43554446;
  private static final short VERSION_NUMBER = 0x0000;

  private static final class ColumnOffsets {
    private final long validity;
    private final long offsets;
    private final long data;
    private final long dataLen;

    public ColumnOffsets(long validity, long offsets, long data, long dataLen) {
      this.validity = validity;
      this.offsets = offsets;
      this.data = data;
      this.dataLen = dataLen;
    }
  }

  /**
   * Holds the metadata about a serialized table. If this is being read from a stream
   * isInitialized will return true if the metadata was read correctly from the stream.
   * It will return false if an EOF was encountered at the beginning indicating that
   * there was no data to be read.
   */
  public static final class SerializedTableHeader {
    private SerializedColumnHeader[] columns;
    private int numRows;
    private long dataLen;

    private boolean initialized = false;
    private boolean dataRead = false;

    public SerializedTableHeader(DataInputStream din) throws IOException {
      readFrom(din);
    }

    SerializedTableHeader(SerializedColumnHeader[] columns, int numRows, long dataLen) {
      this.columns = columns;
      this.numRows = numRows;
      this.dataLen = dataLen;
      initialized = true;
      dataRead = true;
    }

    /** Constructor for a row-count only table (no columns) */
    public SerializedTableHeader(int numRows) {
      this(new SerializedColumnHeader[0], numRows, 0);
    }

    /** Get the column header for the corresponding column index */
    public SerializedColumnHeader getColumnHeader(int columnIndex) {
      return columns[columnIndex];
    }

    /**
     * Set to true once data is successfully read from a stream by readTableIntoBuffer.
     * @return true if data was read, else false.
     */
    public boolean wasDataRead() {
      return dataRead;
    }

    /**
     * Returns the size of a buffer needed to read data into the stream.
     */
    public long getDataLen() {
      return dataLen;
    }

    /**
     * Returns the number of rows stored in this table.
     */
    public int getNumRows() {
      return numRows;
    }

    /**
     * Returns the number of columns stored in this table
     */
    public int getNumColumns() {
      return columns != null ? columns.length : 0;
    }

    /**
     * Returns true if the metadata for this table was read, else false indicating an EOF was
     * encountered.
     */
    public boolean wasInitialized() {
      return initialized;
    }

    /**
     * Returns the number of bytes needed to serialize this table header.
     * Note that this is only the metadata for the table (i.e.: column types, row counts, etc.)
     * and does not include the bytes needed to serialize the table data.
     */
    public long getSerializedHeaderSizeInBytes() {
      // table header always has:
      // - 4-byte magic number
      // - 2-byte version number
      // - 4-byte column count
      // - 4-byte row count
      // - 8-byte data buffer length
      long total = 4 + 2 + 4 + 4 + 8;
      for (SerializedColumnHeader column : columns) {
        total += column.getSerializedHeaderSizeInBytes();
      }
      return total;
    }

    /** Returns the number of bytes needed to serialize this table header and the table data. */
    public long getTotalSerializedSizeInBytes() {
      return getSerializedHeaderSizeInBytes() + dataLen;
    }

    private void readFrom(DataInputStream din) throws IOException {
      try {
        int num = din.readInt();
        if (num != SER_FORMAT_MAGIC_NUMBER) {
          throw new IllegalStateException("THIS DOES NOT LOOK LIKE CUDF SERIALIZED DATA. " +
              "Expected magic number " + SER_FORMAT_MAGIC_NUMBER + " Found " + num);
        }
      } catch (EOFException e) {
        // If we get an EOF at the very beginning don't treat it as an error because we may
        // have finished reading everything...
        return;
      }
      short version = din.readShort();
      if (version != VERSION_NUMBER) {
        throw new IllegalStateException("READING THE WRONG SERIALIZATION FORMAT VERSION FOUND "
            + version + " EXPECTED " + VERSION_NUMBER);
      }
      int numColumns = din.readInt();
      numRows = din.readInt();

      columns = new SerializedColumnHeader[numColumns];
      for (int i = 0; i < numColumns; i++) {
        columns[i] = SerializedColumnHeader.readFrom(din, numRows);
      }

      dataLen = din.readLong();
      initialized = true;
    }

    public void writeTo(DataWriter dout) throws IOException {
      // Now write out the data
      dout.writeInt(SER_FORMAT_MAGIC_NUMBER);
      dout.writeShort(VERSION_NUMBER);
      dout.writeInt(columns.length);
      dout.writeInt(numRows);

      // Header for each column...
      for (SerializedColumnHeader column : columns) {
        column.writeTo(dout);
      }
      dout.writeLong(dataLen);
    }
  }

  /** Holds the metadata about a serialized column. */
  public static final class SerializedColumnHeader {
    public final DType dtype;
    public final long nullCount;
    public final long rowCount;
    public final SerializedColumnHeader[] children;

    SerializedColumnHeader(DType dtype, long rowCount, long nullCount,
                           SerializedColumnHeader[] children) {
      this.dtype = dtype;
      this.rowCount = rowCount;
      this.nullCount = nullCount;
      this.children = children;
    }

    SerializedColumnHeader(ColumnBufferProvider column, long rowOffset, long numRows) {
      this.dtype = column.getType();
      this.rowCount = numRows;
      long columnNullCount = column.getNullCount();
      // For a subset of the original column we do not know the null count unless
      // the original column is either all nulls or no nulls.
      if (column.getRowCount() == numRows
          || columnNullCount == 0 || columnNullCount == column.getRowCount()) {
        this.nullCount = Math.min(columnNullCount, numRows);
      } else {
        this.nullCount = ColumnView.UNKNOWN_NULL_COUNT;
      }
      ColumnBufferProvider[] childProviders = column.getChildProviders();
      if (childProviders != null) {
        children = new SerializedColumnHeader[childProviders.length];
        long childRowOffset = rowOffset;
        long childNumRows = numRows;
        if (dtype.equals(DType.LIST)) {
          if (numRows > 0) {
            childRowOffset = column.getOffset(rowOffset);
            childNumRows = column.getOffset(rowOffset + numRows) - childRowOffset;
          }
        }
        for (int i = 0; i < children.length; i++) {
          children[i] = new SerializedColumnHeader(childProviders[i], childRowOffset, childNumRows);
        }
      } else {
        children = null;
      }
    }

    /** Get the data type of the column */
    public DType getType() {
      return dtype;
    }

    /** Get the row count of the column */
    public long getRowCount() {
      return rowCount;
    }

    /** Get the null count of the column */
    public long getNullCount() {
      return nullCount;
    }

    /** Get the metadata for any child columns or null if there are no children */
    public SerializedColumnHeader[] getChildren() {
      return children;
    }

    /** Get the number of child columns */
    public int getNumChildren() {
      return children != null ? children.length : 0;
    }

    /** Return the number of bytes needed to store this column header in serialized form. */
    public long getSerializedHeaderSizeInBytes() {
      // column header always has:
      // - 4-byte type ID
      // - 4-byte type scale
      // - 4-byte null count
      long total = 4 + 4 + 4;

      if (dtype.isNestedType()) {
        assert children != null;
        if (dtype.equals(DType.LIST)) {
          total += 4;  // 4-byte child row count
        } else if (dtype.equals(DType.STRUCT)) {
          total += 4;  // 4-byte child count
        } else {
          throw new IllegalStateException("Unexpected nested type: " + dtype);
        }
        for (SerializedColumnHeader child : children) {
          total += child.getSerializedHeaderSizeInBytes();
        }
      }

      return total;
    }

    /** Write this column header to the specified writer */
    public void writeTo(DataWriter dout) throws IOException {
      dout.writeInt(dtype.typeId.getNativeId());
      dout.writeInt(dtype.getScale());
      dout.writeInt((int) nullCount);
      if (dtype.isNestedType()) {
        assert children != null;
        if (dtype.equals(DType.LIST)) {
          dout.writeInt((int) children[0].getRowCount());
        } else if (dtype.equals(DType.STRUCT)) {
          dout.writeInt(getNumChildren());
        } else {
          throw new IllegalStateException("Unexpected nested type: " + dtype);
        }
        for (SerializedColumnHeader child : children) {
          child.writeTo(dout);
        }
      }
    }

    static SerializedColumnHeader readFrom(DataInputStream din, long rowCount) throws IOException {
      DType dtype = DType.fromNative(din.readInt(), din.readInt());
      long nullCount = din.readInt();
      SerializedColumnHeader[] children = null;
      if (dtype.isNestedType()) {
        int numChildren;
        long childRowCount;
        if (dtype.equals(DType.LIST)) {
          numChildren = 1;
          childRowCount = din.readInt();
        } else if (dtype.equals(DType.STRUCT)) {
          numChildren = din.readInt();
          childRowCount = rowCount;
        } else {
          throw new IllegalStateException("Unexpected nested type: " + dtype);
        }
        children = new SerializedColumnHeader[numChildren];
        for (int i = 0; i < numChildren; i++) {
          children[i] = readFrom(din, childRowCount);
        }
      }
      return new SerializedColumnHeader(dtype, rowCount, nullCount, children);
    }
  }

  /** Class to hold the header and buffer pair result from host-side concatenation */
  public static final class HostConcatResult implements AutoCloseable {
    private final SerializedTableHeader tableHeader;
    private final HostMemoryBuffer hostBuffer;

    public HostConcatResult(SerializedTableHeader tableHeader, HostMemoryBuffer tableBuffer) {
      this.tableHeader = tableHeader;
      this.hostBuffer = tableBuffer;
    }

    public SerializedTableHeader getTableHeader() {
      return tableHeader;
    }

    public HostMemoryBuffer getHostBuffer() {
      return hostBuffer;
    }

    /** Build a contiguous table in device memory from this host-concatenated result */
    public ContiguousTable toContiguousTable() {
      DeviceMemoryBuffer devBuffer = DeviceMemoryBuffer.allocate(hostBuffer.length);
      try {
        if (hostBuffer.length > 0) {
          devBuffer.copyFromHostBuffer(hostBuffer);
        }
        Table table = sliceUpColumnVectors(tableHeader, devBuffer, hostBuffer);
        try {
          return new ContiguousTable(table, devBuffer);
        } catch (Exception e) {
          table.close();
          throw e;
        }
      } catch (Exception e) {
        devBuffer.close();
        throw e;
      }
    }

    @Override
    public void close() {
      hostBuffer.close();
    }
  }

  /**
   * Visible for testing
   */
  static abstract class ColumnBufferProvider implements AutoCloseable {

    public abstract DType getType();

    public abstract long getNullCount();

    public abstract long getOffset(long index);

    public abstract long getRowCount();

    public abstract HostMemoryBuffer getHostBufferFor(BufferType buffType);

    public abstract long getBufferStartOffset(BufferType buffType);

    public abstract ColumnBufferProvider[] getChildProviders();

    @Override
    public abstract void close();
  }

  /**
   * Visible for testing
   */
  static class ColumnProvider extends ColumnBufferProvider {
    private final HostColumnVectorCore column;
    private final boolean closeAtEnd;
    private final ColumnBufferProvider[] childProviders;

    ColumnProvider(HostColumnVectorCore column, boolean closeAtEnd) {
      this.column = column;
      this.closeAtEnd = closeAtEnd;
      if (getType().isNestedType()) {
        int numChildren = column.getNumChildren();
        childProviders = new ColumnBufferProvider[numChildren];
        for (int i = 0; i < numChildren; i++) {
          childProviders[i] = new ColumnProvider(column.getChildColumnView(i), false);
        }
      } else {
        childProviders = null;
      }
    }

    @Override
    public DType getType() {
      return column.getType();
    }

    @Override
    public long getNullCount() {
      return column.getNullCount();
    }

    @Override
    public long getOffset(long index) {
      return column.getOffsets().getInt(index * Integer.BYTES);
    }

    @Override
    public long getRowCount() {
      return column.getRowCount();
    }

    @Override
    public HostMemoryBuffer getHostBufferFor(BufferType buffType) {
      switch (buffType) {
        case VALIDITY: return column.getValidity();
        case OFFSET: return column.getOffsets();
        case DATA: return column.getData();
        default: throw new IllegalStateException("Unexpected buffer type: " + buffType);
      }
    }

    @Override
    public long getBufferStartOffset(BufferType buffType) {
      // All of the buffers start at 0 for this.
      return 0;
    }

    @Override
    public ColumnBufferProvider[] getChildProviders() {
      return childProviders;
    }

    @Override
    public void close() {
      if (closeAtEnd) {
        column.close();
      }
    }
  }

  private static class BufferOffsetProvider extends ColumnBufferProvider {
    private final SerializedColumnHeader header;
    private final ColumnOffsets offsets;
    private final HostMemoryBuffer buffer;
    private final ColumnBufferProvider[] childProviders;

    private BufferOffsetProvider(SerializedColumnHeader header,
                                 ColumnOffsets offsets,
                                 HostMemoryBuffer buffer,
                                 ColumnBufferProvider[] childProviders) {
      this.header = header;
      this.offsets = offsets;
      this.buffer = buffer;
      this.childProviders = childProviders;
    }

    @Override
    public DType getType() {
      return header.getType();
    }

    @Override
    public long getNullCount() {
      return header.getNullCount();
    }

    @Override
    public long getRowCount() {
      return header.getRowCount();
    }

    @Override
    public HostMemoryBuffer getHostBufferFor(BufferType buffType) {
      return buffer;
    }

    @Override
    public long getBufferStartOffset(BufferType buffType) {
      switch (buffType) {
        case DATA:
          return offsets.data;
        case OFFSET:
          return offsets.offsets;
        case VALIDITY:
          return offsets.validity;
        default:
          throw new IllegalArgumentException("Buffer type " + buffType + " is not supported");
      }
    }

    @Override
    public long getOffset(long index) {
      assert getType().hasOffsets();
      assert (index >= 0 && index <= getRowCount()) : "index is out of range 0 <= " + index + " <= " + getRowCount();
      return buffer.getInt(offsets.offsets + (index * Integer.BYTES));
    }

    @Override
    public ColumnBufferProvider[] getChildProviders() {
      return childProviders;
    }

    @Override
    public void close() {
      // NOOP
    }
  }

  /**
   * Visible for testing
   */
  static abstract class DataWriter {

    public abstract void writeByte(byte b) throws IOException;

    public abstract void writeShort(short s) throws IOException;

    public abstract void writeInt(int i) throws IOException;

    public abstract void writeIntNativeOrder(int i) throws IOException;

    public abstract void writeLong(long val) throws IOException;

    /**
     * Copy data from src starting at srcOffset and going for len bytes.
     * @param src where to copy from.
     * @param srcOffset offset to start at.
     * @param len amount to copy.
     */
    public abstract void copyDataFrom(HostMemoryBuffer src, long srcOffset, long len)
        throws IOException;

    public void copyDataFrom(ColumnBufferProvider column, BufferType buffType,
                             long offset, long length) throws IOException {
      HostMemoryBuffer buff = column.getHostBufferFor(buffType);
      long startOffset = column.getBufferStartOffset(buffType);
      copyDataFrom(buff, startOffset + offset, length);
    }

    public void flush() throws IOException {
      // NOOP by default
    }

    public abstract void write(byte[] arr, int offset, int length) throws IOException;
  }

  /**
   * Visible for testing
   */
  static final class DataOutputStreamWriter extends DataWriter {
    private final byte[] arrayBuffer = new byte[1024 * 128];
    private final DataOutputStream dout;

    public DataOutputStreamWriter(DataOutputStream dout) {
      this.dout = dout;
    }

    @Override
    public void writeByte(byte b) throws IOException {
      dout.writeByte(b);
    }

    @Override
    public void writeShort(short s) throws IOException {
      dout.writeShort(s);
    }

    @Override
    public void writeInt(int i) throws IOException {
      dout.writeInt(i);
    }

    @Override
    public void writeIntNativeOrder(int i) throws IOException {
      // TODO this only works on Little Endian Architectures, x86.  If we need
      // to support others we need to detect the endianness and switch on the right implementation.
      writeInt(Integer.reverseBytes(i));
    }

    @Override
    public void writeLong(long val) throws IOException {
      dout.writeLong(val);
    }

    @Override
    public void copyDataFrom(HostMemoryBuffer src, long srcOffset, long len) throws IOException {
      long dataLeft = len;
      while (dataLeft > 0) {
        int amountToCopy = (int)Math.min(arrayBuffer.length, dataLeft);
        src.getBytes(arrayBuffer, 0, srcOffset, amountToCopy);
        dout.write(arrayBuffer, 0, amountToCopy);
        srcOffset += amountToCopy;
        dataLeft -= amountToCopy;
      }
    }

    @Override
    public void flush() throws IOException {
      dout.flush();
    }

    @Override
    public void write(byte[] arr, int offset, int length) throws IOException {
      dout.write(arr, offset, length);
    }
  }

  private static final class HostDataWriter extends DataWriter {
    private final HostMemoryBuffer buffer;
    private long offset = 0;

    public HostDataWriter(HostMemoryBuffer buffer) {
      this.buffer = buffer;
    }

    @Override
    public void writeByte(byte b) {
      buffer.setByte(offset, b);
      offset += 1;
    }

    @Override
    public void writeShort(short s) {
      buffer.setShort(offset, s);
      offset += 2;
    }

    @Override
    public void writeInt(int i) {
      buffer.setInt(offset, i);
      offset += 4;
    }

    @Override
    public void writeIntNativeOrder(int i) {
      // This is already in the native order...
      writeInt(i);
    }

    @Override
    public void writeLong(long val) {
      buffer.setLong(offset, val);
      offset += 8;
    }

    @Override
    public void copyDataFrom(HostMemoryBuffer src, long srcOffset, long len) {
      buffer.copyFromHostBuffer(offset, src, srcOffset, len);
      offset += len;
    }

    @Override
    public void write(byte[] arr, int srcOffset, int length) {
      buffer.setBytes(offset, arr, srcOffset, length);
      offset += length;
    }
  }

  /////////////////////////////////////////////
  // METHODS
  /////////////////////////////////////////////


  /////////////////////////////////////////////
  // PADDING FOR ALIGNMENT
  /////////////////////////////////////////////
  private static long padFor64byteAlignment(long orig) {
    return ((orig + 63) / 64) * 64;
  }

  private static long padFor64byteAlignment(DataWriter out, long bytes) throws IOException {
    final long paddedBytes = padFor64byteAlignment(bytes);
    while (paddedBytes > bytes) {
      out.writeByte((byte)0);
      bytes++;
    }
    return paddedBytes;
  }

  /////////////////////////////////////////////
  // SERIALIZED SIZE
  /////////////////////////////////////////////

  private static long getRawStringDataLength(ColumnBufferProvider column, long rowOffset, long numRows) {
    if (numRows <= 0) {
      return 0;
    }
    long start = column.getOffset(rowOffset);
    long end = column.getOffset(rowOffset + numRows);
    return end - start;
  }

  private static long getSlicedSerializedDataSizeInBytes(ColumnBufferProvider[] columns, long rowOffset, long numRows) {
    long totalDataSize = 0;
    for (ColumnBufferProvider column: columns) {
      totalDataSize += getSlicedSerializedDataSizeInBytes(column, rowOffset, numRows);
    }
    return totalDataSize;
  }

  private static long getSlicedSerializedDataSizeInBytes(ColumnBufferProvider column, long rowOffset, long numRows) {
    long totalDataSize = 0;
    DType type = column.getType();
    if (needsValidityBuffer(column.getNullCount())) {
      totalDataSize += padFor64byteAlignment(BitVectorHelper.getValidityLengthInBytes(numRows));
    }

    if (type.hasOffsets()) {
      if (numRows > 0) {
        // Add in size of offsets vector
        totalDataSize += padFor64byteAlignment((numRows + 1) * Integer.BYTES);
        if (type.equals(DType.STRING)) {
          totalDataSize += padFor64byteAlignment(getRawStringDataLength(column, rowOffset, numRows));
        }
      }
    } else if (type.getSizeInBytes() > 0) {
      totalDataSize += padFor64byteAlignment(column.getType().getSizeInBytes() * numRows);
    }

    if (numRows > 0 && type.isNestedType()) {
      if (type.equals(DType.LIST)) {
        ColumnBufferProvider child = column.getChildProviders()[0];
        long childStartRow = column.getOffset(rowOffset);
        long childNumRows = column.getOffset(rowOffset + numRows) - childStartRow;
        totalDataSize += getSlicedSerializedDataSizeInBytes(child, childStartRow, childNumRows);
      } else if (type.equals(DType.STRUCT)) {
        for (ColumnBufferProvider childProvider : column.getChildProviders()) {
          totalDataSize += getSlicedSerializedDataSizeInBytes(childProvider, rowOffset, numRows);
        }
      } else {
        throw new IllegalStateException("Unexpected nested type: " + type);
      }
    }
    return totalDataSize;
  }

  /**
   * Get the size in bytes needed to serialize the given data.  The columns should be in host memory
   * before calling this.
   * @param columns columns to be serialized.
   * @param rowOffset the first row to serialize.
   * @param numRows the number of rows to serialize.
   * @return the size in bytes needed to serialize the data including the header.
   */
  public static long getSerializedSizeInBytes(HostColumnVector[] columns, long rowOffset, long numRows) {
    ColumnBufferProvider[] providers = providersFrom(columns, false);
    try {
      SerializedColumnHeader[] columnHeaders = new SerializedColumnHeader[providers.length];
      for (int i = 0; i < columnHeaders.length; i++) {
        columnHeaders[i] = new SerializedColumnHeader(providers[i], rowOffset, numRows);
      }
      long dataLen = getSlicedSerializedDataSizeInBytes(providers, rowOffset, numRows);
      SerializedTableHeader tableHeader = new SerializedTableHeader(columnHeaders,
          (int) numRows, dataLen);
      return tableHeader.getTotalSerializedSizeInBytes();
    } finally {
      closeAll(providers);
    }
  }

  /////////////////////////////////////////////
  // HELPER METHODS buildIndex
  /////////////////////////////////////////////

  /** Build a list of column offset descriptors using a pre-order traversal of the columns */
  static ArrayDeque<ColumnOffsets> buildIndex(SerializedTableHeader header,
                                              HostMemoryBuffer buffer) {
    int numTopColumns = header.getNumColumns();
    ArrayDeque<ColumnOffsets> offsetsList = new ArrayDeque<>();
    long bufferOffset = 0;
    for (int i = 0; i < numTopColumns; i++) {
      SerializedColumnHeader column = header.getColumnHeader(i);
      bufferOffset = buildIndex(column, buffer, offsetsList, bufferOffset);
    }
    return offsetsList;
  }

  /**
   * Append a list of column offset descriptors using a pre-order traversal of the column
   * @param column column offset descriptors will be built for this column and its child columns
   * @param buffer host buffer backing the column data
   * @param offsetsList list where column offset descriptors will be appended during traversal
   * @param bufferOffset offset in the host buffer where the column data begins
   * @return buffer offset at the end of this column's data including all child columns
   */
  private static long buildIndex(SerializedColumnHeader column, HostMemoryBuffer buffer,
                                 ArrayDeque<ColumnOffsets> offsetsList, long bufferOffset) {
    long validity = 0;
    long offsets = 0;
    long data = 0;
    long dataLen = 0;
    long rowCount = column.getRowCount();
    if (needsValidityBuffer(column.getNullCount())) {
      long validityLen = padFor64byteAlignment(BitVectorHelper.getValidityLengthInBytes(rowCount));
      validity = bufferOffset;
      bufferOffset += validityLen;
    }

    DType dtype = column.getType();
    if (dtype.hasOffsets()) {
      if (rowCount > 0) {
        long offsetsLen = (rowCount + 1) * Integer.BYTES;
        offsets = bufferOffset;
        int startOffset = buffer.getInt(bufferOffset);
        int endOffset = buffer.getInt(bufferOffset + (rowCount * Integer.BYTES));
        bufferOffset += padFor64byteAlignment(offsetsLen);
        if (dtype.equals(DType.STRING)) {
          dataLen = endOffset - startOffset;
          data = bufferOffset;
          bufferOffset += padFor64byteAlignment(dataLen);
        }
      }
    } else if (dtype.getSizeInBytes() > 0) {
      dataLen = dtype.getSizeInBytes() * rowCount;
      data = bufferOffset;
      bufferOffset += padFor64byteAlignment(dataLen);
    }
    offsetsList.add(new ColumnOffsets(validity, offsets, data, dataLen));

    SerializedColumnHeader[] children = column.getChildren();
    if (children != null) {
      for (SerializedColumnHeader child : children) {
        bufferOffset = buildIndex(child, buffer, offsetsList, bufferOffset);
      }
    }

    return bufferOffset;
  }

  /////////////////////////////////////////////
  // HELPER METHODS FOR PROVIDERS
  /////////////////////////////////////////////

  private static void closeAll(ColumnBufferProvider[] providers) {
    for (int i = 0; i < providers.length; i++) {
      providers[i].close();
    }
  }

  private static void closeAll(ColumnBufferProvider[][] providers) {
    for (int i = 0; i < providers.length; i++) {
      if (providers[i] != null) {
        closeAll(providers[i]);
      }
    }
  }

  private static ColumnBufferProvider[] providersFrom(ColumnVector[] columns) {
    HostColumnVector[] onHost = new HostColumnVector[columns.length];
    try {
      for (int i = 0; i < columns.length; i++) {
        onHost[i] = columns[i].copyToHostAsync(Cuda.DEFAULT_STREAM);
      }
      Cuda.DEFAULT_STREAM.sync();
      return providersFrom(onHost, true);
    } catch (Throwable t) {
      // Async D->H copies may still be in flight into these host buffers;
      // sync before closing to avoid a use-after-free on pinned memory.
      // Preserve the original cause; record secondary failures as suppressed.
      boolean drained = false;
      try {
        Cuda.DEFAULT_STREAM.sync();
        drained = true;
      } catch (Throwable syncFailure) {
        t.addSuppressed(syncFailure);
      }
      // If the recovery sync threw, the CUDA stream is in an error state
      // and pending DMA writes into the host buffers may not have completed.
      // Closing them now risks the DMA engine writing into freed memory.
      // Leak instead — the process is already in a broken state (sticky CUDA
      // error usually requires a restart), and a leak beats silent corruption.
      if (drained) {
        for (int i = 0; i < onHost.length; i++) {
          CleanupHelpers.closeAndSuppress(onHost[i], t);
        }
      }
      throw t;
    }
  }

  private static ColumnBufferProvider[] providersFrom(HostColumnVector[] columns, boolean closeAtEnd) {
    ColumnBufferProvider[] providers = new ColumnBufferProvider[columns.length];
    for (int i = 0; i < columns.length; i++) {
      providers[i] = new ColumnProvider(columns[i], closeAtEnd);
    }
    return providers;
  }

  /**
   * For a batch of tables described by a header and corresponding buffer, return a mapping of
   * top column index to the corresponding column providers for that column across all tables.
   */
  private static ColumnBufferProvider[][] providersFrom(SerializedTableHeader[] headers,
                                                        HostMemoryBuffer[] dataBuffers) {
    int numColumns = 0;
    int numTables = headers.length;
    int numNonEmptyTables = 0;
    ArrayList<ArrayList<ColumnBufferProvider>> providersPerColumn = null;
    for (int tableIdx = 0; tableIdx < numTables; tableIdx++) {
      SerializedTableHeader header = headers[tableIdx];
      if (tableIdx == 0) {
        numColumns = header.getNumColumns();
        providersPerColumn = new ArrayList<>(numColumns);
        for (int i = 0; i < numColumns; i++) {
          providersPerColumn.add(new ArrayList<>(numTables));
        }
      } else {
        checkCompatibleTypes(headers[0], header, tableIdx);
      }
      // filter out empty tables but keep at least one if all were empty
      if (headers[tableIdx].getNumRows() > 0 ||
          (numNonEmptyTables == 0 && tableIdx == numTables - 1)) {
        numNonEmptyTables++;
        HostMemoryBuffer dataBuffer = dataBuffers[tableIdx];
        ArrayDeque<ColumnOffsets> offsets = buildIndex(header, dataBuffer);
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
          ColumnBufferProvider provider = buildBufferOffsetProvider(
              header.getColumnHeader(columnIdx), offsets, dataBuffer);
          providersPerColumn.get(columnIdx).add(provider);
        }
        assert offsets.isEmpty();
      } else {
        assert headers[tableIdx].dataLen == 0;
      }
    }

    ColumnBufferProvider[][] result = new ColumnBufferProvider[numColumns][];
    for (int i = 0; i < numColumns; i++) {
      result[i] = providersPerColumn.get(i).toArray(new ColumnBufferProvider[0]);
    }
    return result;
  }

  private static void checkCompatibleTypes(SerializedTableHeader expected,
                                           SerializedTableHeader other,
                                           int tableIdx) {
    int numColumns = expected.getNumColumns();
    if (other.getNumColumns() != numColumns) {
      throw new IllegalArgumentException("The number of columns did not match " + tableIdx
          + " " + other.getNumColumns() + " != " + numColumns);
    }
    for (int i = 0; i < numColumns; i++) {
      checkCompatibleTypes(expected.getColumnHeader(i), other.getColumnHeader(i), tableIdx, i);
    }
  }

  private static void checkCompatibleTypes(SerializedColumnHeader expected,
                                           SerializedColumnHeader other,
                                           int tableIdx, int columnIdx) {
    DType dtype = expected.getType();
    if (!dtype.equals(other.getType())) {
      throw new IllegalArgumentException("Type mismatch at table " + tableIdx +
          "column " + columnIdx + " expected " + dtype + " but found " + other.getType());
    }
    if (dtype.isNestedType()) {
      SerializedColumnHeader[] expectedChildren = expected.getChildren();
      SerializedColumnHeader[] otherChildren = other.getChildren();
      if (expectedChildren.length != otherChildren.length) {
        throw new IllegalArgumentException("Child count mismatch at table " + tableIdx +
            "column " + columnIdx + " expected " + expectedChildren.length + " but found " +
            otherChildren.length);
      }
      for (int i = 0; i < expectedChildren.length; i++) {
        checkCompatibleTypes(expectedChildren[i], otherChildren[i], tableIdx, columnIdx);
      }
    }
  }

  private static BufferOffsetProvider buildBufferOffsetProvider(SerializedColumnHeader header,
                                                                ArrayDeque<ColumnOffsets> offsets,
                                                                HostMemoryBuffer dataBuffer) {
    ColumnOffsets columnOffsets = offsets.remove();
    ColumnBufferProvider[] childProviders = null;
    SerializedColumnHeader[] children = header.getChildren();
    if (children != null) {
      childProviders = new ColumnBufferProvider[children.length];
      for (int i = 0; i < children.length; i++) {
        childProviders[i] = buildBufferOffsetProvider(children[i], offsets, dataBuffer);
      }
    }
    return new BufferOffsetProvider(header, columnOffsets, dataBuffer, childProviders);
  }

  /////////////////////////////////////////////
  // HELPER METHODS FOR SerializedTableHeader
  /////////////////////////////////////////////

  private static SerializedTableHeader calcHeader(ColumnBufferProvider[] columns,
                                                  long rowOffset,
                                                  int numRows) {
    SerializedColumnHeader[] headers = new SerializedColumnHeader[columns.length];
    for (int i = 0; i < headers.length; i++) {
      headers[i] = new SerializedColumnHeader(columns[i], rowOffset, numRows);
    }
    long dataLength = getSlicedSerializedDataSizeInBytes(columns, rowOffset, numRows);
    return new SerializedTableHeader(headers, numRows, dataLength);
  }

  /**
   * Calculate the new header for a concatenated set of columns.
   * @param providersPerColumn first index is the column, second index is the table.
   * @return the new header.
   */
  private static SerializedTableHeader calcConcatHeader(ColumnBufferProvider[][] providersPerColumn) {
    int numColumns = providersPerColumn.length;
    long rowCount = 0;
    long totalDataSize = 0;
    ArrayList<SerializedColumnHeader> headers = new ArrayList<>(numColumns);
    for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
      totalDataSize += calcConcatColumnHeaderAndSize(headers, providersPerColumn[columnIdx]);
      if (columnIdx == 0) {
        rowCount = headers.get(0).getRowCount();
      } else {
        assert rowCount == headers.get(columnIdx).getRowCount();
      }
    }
    SerializedColumnHeader[] columnHeaders = headers.toArray(new SerializedColumnHeader[0]);
    return new SerializedTableHeader(columnHeaders, (int)rowCount, totalDataSize);
  }

  /**
   * Calculate a column header describing all of the columns concatenated together
   * @param outHeaders list that will be appended with the new column header
   * @param providers columns to be concatenated
   * @return total bytes needed to store the data for the result column and its children
   */
  private static long calcConcatColumnHeaderAndSize(ArrayList<SerializedColumnHeader> outHeaders,
                                                    ColumnBufferProvider[] providers) {
    long totalSize = 0;
    int numTables = providers.length;
    long rowCount = 0;
    long nullCount = 0;
    for (ColumnBufferProvider provider : providers) {
      rowCount += provider.getRowCount();
      if (nullCount != ColumnView.UNKNOWN_NULL_COUNT) {
        long providerNullCount = provider.getNullCount();
        if (providerNullCount == ColumnView.UNKNOWN_NULL_COUNT) {
          nullCount = ColumnView.UNKNOWN_NULL_COUNT;
        } else {
          nullCount += providerNullCount;
        }
      }
    }

    if (rowCount > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("Cannot build a batch larger than " + Integer.MAX_VALUE + " rows");
    }

    if (needsValidityBuffer(nullCount)) {
      totalSize += padFor64byteAlignment(BitVectorHelper.getValidityLengthInBytes(rowCount));
    }

    ColumnBufferProvider firstProvider = providers[0];
    DType dtype = firstProvider.getType();
    if (dtype.hasOffsets()) {
      if (rowCount > 0) {
        totalSize += padFor64byteAlignment((rowCount + 1) * Integer.BYTES);
        if (dtype.equals(DType.STRING)) {
          long stringDataLen = 0;
          for (ColumnBufferProvider provider : providers) {
            stringDataLen += getRawStringDataLength(provider, 0, provider.getRowCount());
          }
          totalSize += padFor64byteAlignment(stringDataLen);
        }
      }
    } else if (dtype.getSizeInBytes() > 0) {
      totalSize += padFor64byteAlignment(dtype.getSizeInBytes() * rowCount);
    }

    SerializedColumnHeader[] children = null;
    if (dtype.isNestedType()) {
      int numChildren = firstProvider.getChildProviders().length;
      ArrayList<SerializedColumnHeader> childHeaders = new ArrayList<>(numChildren);
      ColumnBufferProvider[] childColumnProviders = new ColumnBufferProvider[numTables];
      for (int childIdx = 0; childIdx < numChildren; childIdx++) {
        // collect all the providers for the current child and build the child's header
        for (int tableIdx = 0; tableIdx < numTables; tableIdx++) {
          childColumnProviders[tableIdx] = providers[tableIdx].getChildProviders()[childIdx];
        }
        totalSize += calcConcatColumnHeaderAndSize(childHeaders, childColumnProviders);
      }
      children = childHeaders.toArray(new SerializedColumnHeader[0]);
    }

    outHeaders.add(new SerializedColumnHeader(dtype, rowCount, nullCount, children));
    return totalSize;
  }

  /////////////////////////////////////////////
  // HELPER METHODS FOR DataWriters
  /////////////////////////////////////////////

  private static DataWriter writerFrom(OutputStream out) {
    if (!(out instanceof DataOutputStream)) {
      out = new DataOutputStream(new BufferedOutputStream(out));
    }
    return new DataOutputStreamWriter((DataOutputStream) out);
  }

  private static DataWriter writerFrom(HostMemoryBuffer buffer) {
    return new HostDataWriter(buffer);
  }

  /////////////////////////////////////////////
  // Serialize Data Methods
  /////////////////////////////////////////////

  private static long copySlicedAndPad(DataWriter out,
                                       ColumnBufferProvider column,
                                       BufferType buffer,
                                       long offset,
                                       long length) throws IOException {
    out.copyDataFrom(column, buffer, offset, length);
    return padFor64byteAlignment(out, length);
  }

  /////////////////////////////////////////////
  // VALIDITY
  /////////////////////////////////////////////

  private static boolean needsValidityBuffer(long nullCount) {
    return nullCount > 0 || nullCount == ColumnView.UNKNOWN_NULL_COUNT;
  }

  private static int copyPartialValidity(byte[] dest,
                                         int destBitOffset,
                                         ColumnBufferProvider provider,
                                         int srcBitOffset,
                                         int lengthBits) {
    HostMemoryBuffer src = provider.getHostBufferFor(BufferType.VALIDITY);
    long baseSrcByteOffset = provider.getBufferStartOffset(BufferType.VALIDITY);

    int destStartBytes = destBitOffset / 8;
    int destStartBitOffset = destBitOffset % 8;
    long srcStartBytes = baseSrcByteOffset + (srcBitOffset / 8);
    int srcStartBitOffset = srcBitOffset % 8;
    int availableDestBits = (dest.length * 8) - destBitOffset;
    int bitsToCopy = Math.min(lengthBits, availableDestBits);

    int lastIndex = (bitsToCopy + destStartBitOffset + 7) / 8;

    byte allBitsSet = ~0;
    byte firstSrcMask = (byte)(allBitsSet << destStartBitOffset);

    int srcShift = destStartBitOffset - srcStartBitOffset;
    if (srcShift > 0) {
      // Shift left. If we are going to shift this is the path typically taken.

      byte current = src.getByte(srcStartBytes);
      byte result = (byte)(current << srcShift);
      // The first time we need to include any data already in dest.
      result |= dest[destStartBytes] & ~firstSrcMask;
      dest[destStartBytes] = result;

      // Keep the previous bytes around so we don't have to keep reading from src, which is not free
      byte previous = current;

      for (int index = 1; index < lastIndex; index++) {
        current = src.getByte(index + srcStartBytes);
        result = (byte)(current << srcShift);
        result |= (previous & 0xFF) >>> (8 - srcShift);
        dest[index + destStartBytes] = result;
        previous = current;
      }
      return bitsToCopy;
    } else if (srcShift < 0) {
      srcShift = -srcShift;

      // shifting right only happens when the buffer runs out of space.

      byte result = src.getByte(srcStartBytes);
      result = (byte)((result & 0xFF) >>> srcShift);
      byte next = 0;
      if (srcStartBytes + 1 < src.length) {
        next = src.getByte(srcStartBytes + 1);
      }
      result |= (byte)(next << 8 - srcShift);
      result &= firstSrcMask;

      // The first time through we need to include the data already in dest.
      result |= dest[destStartBytes] & ~firstSrcMask;
      dest[destStartBytes] = result;

      for (int index = 1; index < lastIndex - 1; index++) {
        result = next;
        result = (byte)((result & 0xFF) >>> srcShift);
        next = src.getByte(srcStartBytes + index + 1);
        result |= (byte)(next << 8 - srcShift);
        dest[index + destStartBytes] = result;
      }

      int idx = lastIndex - 1;
      if (idx > 0) {
        result = next;
        result = (byte) ((result & 0xFF) >>> srcShift);
        next = 0;
        if (srcStartBytes + idx + 1 < src.length) {
          next = src.getByte(srcStartBytes + idx + 1);
        }
        result |= (byte) (next << 8 - srcShift);
        dest[idx + destStartBytes] = result;
      }
      return bitsToCopy;
    } else {
      src.getBytes(dest, destStartBytes, srcStartBytes, (bitsToCopy + 7) / 8);
      return bitsToCopy;
    }
  }

  // package-private for testing
  static long copySlicedValidity(DataWriter out,
                                 ColumnBufferProvider column,
                                 long rowOffset,
                                 long numRows) throws IOException {
    long validityLen = BitVectorHelper.getValidityLengthInBytes(numRows);
    long byteOffset = (rowOffset / 8);
    long bytesLeft = validityLen;

    int lshift = (int) rowOffset % 8;
    if (lshift == 0) {
      out.copyDataFrom(column, BufferType.VALIDITY, byteOffset, bytesLeft);
    } else {
      byte[] arrayBuffer = new byte[128 * 1024];
      int rowsStoredInArray = 0;
      int rowsLeftInBatch = (int) numRows;
      int validityBitOffset = (int) rowOffset;
      while(rowsLeftInBatch > 0) {
        int rowsStoredJustNow = copyPartialValidity(arrayBuffer, rowsStoredInArray, column, validityBitOffset, rowsLeftInBatch);
        assert rowsStoredJustNow > 0;
        rowsLeftInBatch -= rowsStoredJustNow;
        rowsStoredInArray += rowsStoredJustNow;
        validityBitOffset += rowsStoredJustNow;
        if (rowsStoredInArray == arrayBuffer.length * 8) {
          out.write(arrayBuffer, 0, arrayBuffer.length);
          rowsStoredInArray = 0;
        }
      }
      if (rowsStoredInArray > 0) {
        out.write(arrayBuffer, 0, (rowsStoredInArray + 7) / 8);
      }
    }
    return padFor64byteAlignment(out, validityLen);
  }

  // Package private for testing
  static int fillValidity(byte[] dest, int destBitOffset, int lengthBits) {
    int destStartBytes = destBitOffset / 8;
    int destStartBits = destBitOffset % 8;

    long lengthBytes = BitVectorHelper.getValidityLengthInBytes(lengthBits);
    int rshift = destStartBits;
    int totalCopied = 0;
    if (rshift != 0) {
      // Fill in what we need to make it copyable
      dest[destStartBytes] |= (0xFF << destStartBits);
      destStartBytes += 1;
      totalCopied = (8 - destStartBits);
      // Not used again, but just to be safe
      destStartBits = 0;
    }
    int amountToCopyBytes = (int) Math.min(lengthBytes, dest.length - destStartBytes);
    for (int i = 0; i < amountToCopyBytes; i++) {
      dest[i + destStartBytes] = (byte) 0xFF;
    }
    totalCopied += amountToCopyBytes * 8;
    return Math.min(totalCopied, lengthBits);
  }

  private static long concatValidity(DataWriter out, long numRows,
                                     ColumnBufferProvider[] providers) throws IOException {
    long validityLen = BitVectorHelper.getValidityLengthInBytes(numRows);
    byte[] arrayBuffer = new byte[128 * 1024];
    int rowsStoredInArray = 0;
    for (ColumnBufferProvider provider : providers) {
      int rowsLeftInBatch = (int) provider.getRowCount();
      int validityBitOffset = 0;
      while(rowsLeftInBatch > 0) {
        int rowsStoredJustNow;
        if (needsValidityBuffer(provider.getNullCount())) {
          rowsStoredJustNow = copyPartialValidity(arrayBuffer, rowsStoredInArray, provider, validityBitOffset, rowsLeftInBatch);
        } else {
          rowsStoredJustNow = fillValidity(arrayBuffer, rowsStoredInArray, rowsLeftInBatch);
        }
        assert rowsStoredJustNow > 0;
        assert rowsStoredJustNow <= rowsLeftInBatch;
        rowsLeftInBatch -= rowsStoredJustNow;
        rowsStoredInArray += rowsStoredJustNow;
        validityBitOffset += rowsStoredJustNow;
        if (rowsStoredInArray == arrayBuffer.length * 8) {
          out.write(arrayBuffer, 0, arrayBuffer.length);
          rowsStoredInArray = 0;
        }
      }
    }

    if (rowsStoredInArray > 0) {
      int len = (rowsStoredInArray + 7) / 8;
      out.write(arrayBuffer, 0, len);
    }
    return padFor64byteAlignment(out, validityLen);
  }

  /////////////////////////////////////////////
  // STRING
  /////////////////////////////////////////////

  private static long copySlicedStringData(DataWriter out, ColumnBufferProvider column, long rowOffset,
                                           long numRows) throws IOException {
    if (numRows > 0) {
      long startByteOffset = column.getOffset(rowOffset);
      long endByteOffset = column.getOffset(rowOffset + numRows);
      long bytesToCopy = endByteOffset - startByteOffset;
      long srcOffset = startByteOffset;
      return copySlicedAndPad(out, column, BufferType.DATA, srcOffset, bytesToCopy);
    }
    return 0;
  }

  private static void copyConcatStringData(DataWriter out,
                                           ColumnBufferProvider[] providers) throws IOException {
    long totalCopied = 0;

    for (ColumnBufferProvider provider : providers) {
      long rowCount = provider.getRowCount();
      if (rowCount > 0) {
        HostMemoryBuffer dataBuffer = provider.getHostBufferFor(BufferType.DATA);
        long currentOffset = provider.getBufferStartOffset(BufferType.DATA);
        long dataLeft = provider.getOffset(rowCount);
        out.copyDataFrom(dataBuffer, currentOffset, dataLeft);
        totalCopied += dataLeft;
      }
    }
    padFor64byteAlignment(out, totalCopied);
  }

  private static long copySlicedOffsets(DataWriter out, ColumnBufferProvider column, long rowOffset,
                                        long numRows) throws IOException {
    if (numRows <= 0) {
      // Don't copy anything, there are no rows
      return 0;
    }
    long bytesToCopy = (numRows + 1) * Integer.BYTES;
    long srcOffset = rowOffset * Integer.BYTES;
    if (rowOffset == 0) {
      return copySlicedAndPad(out, column, BufferType.OFFSET, srcOffset, bytesToCopy);
    }
    HostMemoryBuffer buff = column.getHostBufferFor(BufferType.OFFSET);
    long startOffset = column.getBufferStartOffset(BufferType.OFFSET) + srcOffset;
    if (bytesToCopy >= Integer.MAX_VALUE) {
      throw new IllegalStateException("Copy is too large, need to do chunked copy");
    }
    ByteBuffer bb = buff.asByteBuffer(startOffset, (int)bytesToCopy);
    int start = bb.getInt();
    out.writeIntNativeOrder(0);
    long total = Integer.BYTES;
    for (int i = 1; i < (numRows + 1); i++) {
      int offset = bb.getInt();
      out.writeIntNativeOrder(offset - start);
      total += Integer.BYTES;
    }
    assert total == bytesToCopy;
    long ret = padFor64byteAlignment(out, total);
    return ret;
  }

  private static void copyConcatOffsets(DataWriter out,
                                        ColumnBufferProvider[] providers) throws IOException {
    long totalCopied = 0;
    int offsetToAdd = 0;
    for (ColumnBufferProvider provider : providers) {
      long rowCount = provider.getRowCount();
      if (rowCount > 0) {
        HostMemoryBuffer offsetsBuffer = provider.getHostBufferFor(BufferType.OFFSET);
        long currentOffset = provider.getBufferStartOffset(BufferType.OFFSET);
        if (totalCopied == 0) {
          // first chunk of offsets can be copied verbatim
          totalCopied = (rowCount + 1) * Integer.BYTES;
          out.copyDataFrom(offsetsBuffer, currentOffset, totalCopied);
          offsetToAdd = offsetsBuffer.getInt(currentOffset + (rowCount * Integer.BYTES));
        } else {
          int localOffset = 0;
          // first row's offset has already been written when processing the previous table
          for (int row = 1; row < rowCount + 1; row++) {
            localOffset = offsetsBuffer.getInt(currentOffset + (row * Integer.BYTES));
            out.writeIntNativeOrder(localOffset + offsetToAdd);
          }
          // last local offset of this chunk is the length of data referenced by offsets
          offsetToAdd += localOffset;
          totalCopied += rowCount * Integer.BYTES;
        }
      }
    }
    padFor64byteAlignment(out, totalCopied);
  }

  /////////////////////////////////////////////
  // BASIC DATA
  /////////////////////////////////////////////

  private static long sliceBasicData(DataWriter out,
                                     ColumnBufferProvider column,
                                     long rowOffset,
                                     long numRows) throws IOException {
    DType type = column.getType();
    long bytesToCopy = numRows * type.getSizeInBytes();
    long srcOffset = rowOffset * type.getSizeInBytes();
    return copySlicedAndPad(out, column, BufferType.DATA, srcOffset, bytesToCopy);
  }

  private static void concatBasicData(DataWriter out,
                                      DType type,
                                      ColumnBufferProvider[] providers) throws IOException {
    long totalCopied = 0;
    for (ColumnBufferProvider provider : providers) {
      long rowCount = provider.getRowCount();
      if (rowCount > 0) {
        HostMemoryBuffer dataBuffer = provider.getHostBufferFor(BufferType.DATA);
        long currentOffset = provider.getBufferStartOffset(BufferType.DATA);
        long dataLeft = rowCount * type.getSizeInBytes();
        out.copyDataFrom(dataBuffer, currentOffset, dataLeft);
        totalCopied += dataLeft;
      }
    }
    padFor64byteAlignment(out, totalCopied);
  }

  /////////////////////////////////////////////
  // COLUMN AND TABLE WRITE
  /////////////////////////////////////////////

  private static void writeConcat(DataWriter out, SerializedColumnHeader header,
                                  ColumnBufferProvider[] providers) throws IOException {
    if (needsValidityBuffer(header.getNullCount())) {
      concatValidity(out, header.getRowCount(), providers);
    }

    DType dtype = header.getType();
    if (dtype.hasOffsets()) {
      if (header.getRowCount() > 0) {
        copyConcatOffsets(out, providers);
        if (dtype.equals(DType.STRING)) {
          copyConcatStringData(out, providers);
        }
      }
    } else if (dtype.getSizeInBytes() > 0) {
      concatBasicData(out, dtype, providers);
    }

    if (dtype.isNestedType()) {
      int numTables = providers.length;
      SerializedColumnHeader[] childHeaders = header.getChildren();
      ColumnBufferProvider[] childColumnProviders = new ColumnBufferProvider[numTables];
      for (int childIdx = 0; childIdx < childHeaders.length; childIdx++) {
        // collect all the providers for the current child column
        for (int tableIdx = 0; tableIdx < numTables; tableIdx++) {
          childColumnProviders[tableIdx] = providers[tableIdx].getChildProviders()[childIdx];
        }
        writeConcat(out, childHeaders[childIdx], childColumnProviders);
      }
    }
  }

  private static void writeSliced(DataWriter out,
                                  ColumnBufferProvider column,
                                  long rowOffset,
                                  long numRows) throws IOException {
    if (needsValidityBuffer(column.getNullCount())) {
      try (NvtxRange range = new NvtxRange("Write Validity", NvtxColor.DARK_GREEN)) {
        copySlicedValidity(out, column, rowOffset, numRows);
      }
    }

    DType type = column.getType();
    if (type.hasOffsets()) {
      if (numRows > 0) {
        try (NvtxRange offsetRange = new NvtxRange("Write Offset Data", NvtxColor.ORANGE)) {
          copySlicedOffsets(out, column, rowOffset, numRows);
          if (type.equals(DType.STRING)) {
            try (NvtxRange dataRange = new NvtxRange("Write String Data", NvtxColor.RED)) {
              copySlicedStringData(out, column, rowOffset, numRows);
            }
          }
        }
      }
    } else if (type.getSizeInBytes() > 0){
      try (NvtxRange range = new NvtxRange("Write Data", NvtxColor.BLUE)) {
        sliceBasicData(out, column, rowOffset, numRows);
      }
    }

    if (numRows > 0 && type.isNestedType()) {
      if (type.equals(DType.LIST)) {
        try (NvtxRange range = new NvtxRange("Write List Child", NvtxColor.PURPLE)) {
          ColumnBufferProvider child = column.getChildProviders()[0];
          long childStartRow = column.getOffset(rowOffset);
          long childNumRows = column.getOffset(rowOffset + numRows) - childStartRow;
          writeSliced(out, child, childStartRow, childNumRows);
        }
      } else if (type.equals(DType.STRUCT)) {
        try (NvtxRange range = new NvtxRange("Write Struct Children", NvtxColor.PURPLE)) {
          for (ColumnBufferProvider child : column.getChildProviders()) {
            writeSliced(out, child, rowOffset, numRows);
          }
        }
      } else {
        throw new IllegalStateException("Unexpected nested type: " + type);
      }
    }
  }

  private static void writeSliced(ColumnBufferProvider[] columns,
                                  DataWriter out,
                                  long rowOffset,
                                  long numRows) throws IOException {
    assert rowOffset >= 0;
    assert numRows >= 0;
    for (int i = 0; i < columns.length; i++) {
      long rows = columns[i].getRowCount();
      assert rowOffset + numRows <= rows;
      assert rows == (int) rows : "can only support an int for indexes";
    }

    SerializedTableHeader header = calcHeader(columns, rowOffset, (int) numRows);
    header.writeTo(out);

    try (NvtxRange range = new NvtxRange("Write Sliced", NvtxColor.GREEN)) {
      for (int i = 0; i < columns.length; i++) {
        writeSliced(out, columns[i], rowOffset, numRows);
      }
    }
    out.flush();
  }

  /**
   * Write all or part of a table out in an internal format.
   * @param t the table to be written.
   * @param out the stream to write the serialized table out to.
   * @param rowOffset the first row to write out.
   * @param numRows the number of rows to write out.
   */
  public static void writeToStream(Table t, OutputStream out, long rowOffset, long numRows)
      throws IOException {
    writeToStream(t.getColumns(), out, rowOffset, numRows);
  }

  /**
   * Write all or part of a set of columns out in an internal format.
   * @param columns the columns to be written.
   * @param out the stream to write the serialized table out to.
   * @param rowOffset the first row to write out.
   * @param numRows the number of rows to write out.
   */
  public static void writeToStream(ColumnVector[] columns, OutputStream out, long rowOffset,
                                   long numRows) throws IOException {

    ColumnBufferProvider[] providers = providersFrom(columns);
    try {
      DataWriter writer = writerFrom(out);
      writeSliced(providers, writer, rowOffset, numRows);
    } finally {
      closeAll(providers);
    }
  }

  /**
   * Write all or part of a set of columns out in an internal format.
   * @param columns the columns to be written.
   * @param out the stream to write the serialized table out to.
   * @param rowOffset the first row to write out.
   * @param numRows the number of rows to write out.
   */
  public static void writeToStream(HostColumnVector[] columns, OutputStream out, long rowOffset,
                                   long numRows) throws IOException {

    ColumnBufferProvider[] providers = providersFrom(columns, false);
    try {
      DataWriter writer = writerFrom(out);
      writeSliced(providers, writer, rowOffset, numRows);
    } finally {
      closeAll(providers);
    }
  }

  /**
   * Write a rowcount only header to the output stream in a case
   * where a columnar batch with no columns but a non zero row count is received
   * @param out the stream to write the serialized table out to.
   * @param numRows the number of rows to write out.
   */
  public static void writeRowsToStream(OutputStream out, long numRows) throws IOException {
    DataWriter writer = writerFrom(out);
    SerializedTableHeader header = new SerializedTableHeader((int) numRows);
    header.writeTo(writer);
    writer.flush();
  }

  /**
   * Take the data from multiple batches stored in the parsed headers and the dataBuffer and write
   * it out to out as if it were a single buffer.
   * @param headers the headers parsed from multiple streams.
   * @param dataBuffers an array of buffers that hold the data, one per header.
   * @param out what to write the data out to.
   * @throws IOException on any error.
   */
  public static void writeConcatedStream(SerializedTableHeader[] headers,
                                         HostMemoryBuffer[] dataBuffers,
                                         OutputStream out) throws IOException {
    ColumnBufferProvider[][] providersPerColumn = providersFrom(headers, dataBuffers);
    try {
      SerializedTableHeader combined = calcConcatHeader(providersPerColumn);
      DataWriter writer = writerFrom(out);
      combined.writeTo(writer);
      try (NvtxRange range = new NvtxRange("Concat Host Side", NvtxColor.GREEN)) {
        int numColumns = combined.getNumColumns();
        for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
          ColumnBufferProvider[] providers = providersPerColumn[columnIdx];
          writeConcat(writer, combined.getColumnHeader(columnIdx), providersPerColumn[columnIdx]);
        }
      }
      writer.flush();
    } finally {
      closeAll(providersPerColumn);
    }
  }

  /////////////////////////////////////////////
  // COLUMN AND TABLE READ
  /////////////////////////////////////////////

  private static HostColumnVectorCore buildHostColumn(SerializedColumnHeader column,
                                                      ArrayDeque<ColumnOffsets> columnOffsets,
                                                      HostMemoryBuffer buffer,
                                                      boolean isRootColumn) {
    ColumnOffsets offsetsInfo = columnOffsets.remove();
    SerializedColumnHeader[] children = column.getChildren();
    int numChildren = children != null ? children.length : 0;
    List<HostColumnVectorCore> childColumns = new ArrayList<>(numChildren);
    try {
      if (children != null) {
        for (SerializedColumnHeader child : children) {
          childColumns.add(buildHostColumn(child, columnOffsets, buffer, false));
        }
      }
      DType dtype = column.getType();
      long rowCount = column.getRowCount();
      long nullCount = column.getNullCount();
      HostMemoryBuffer dataBuffer = null;
      HostMemoryBuffer validityBuffer = null;
      HostMemoryBuffer offsetsBuffer = null;
      if (!dtype.isNestedType()) {
        dataBuffer = buffer.slice(offsetsInfo.data, offsetsInfo.dataLen);
      }
      if (needsValidityBuffer(nullCount)) {
        long validitySize = BitVectorHelper.getValidityLengthInBytes(rowCount);
        validityBuffer = buffer.slice(offsetsInfo.validity, validitySize);
      }
      if (dtype.hasOffsets()) {
        // one 32-bit integer offset per row plus one additional offset at the end
        long offsetsSize = rowCount > 0 ? (rowCount + 1) * Integer.BYTES : 0;
        offsetsBuffer = buffer.slice(offsetsInfo.offsets, offsetsSize);
      }
      HostColumnVectorCore result;
      // Only creates HostColumnVector for root columns, since child columns are managed by their parents.
      if (isRootColumn) {
        result = new HostColumnVector(dtype, rowCount,
            Optional.of(nullCount), dataBuffer, validityBuffer, offsetsBuffer,
            childColumns);
      } else {
        result = new HostColumnVectorCore(dtype, rowCount,
            Optional.of(nullCount), dataBuffer, validityBuffer, offsetsBuffer,
            childColumns);
      }
      childColumns = null;
      return result;
    } finally {
      if (childColumns != null) {
        for (HostColumnVectorCore c : childColumns) {
          c.close();
        }
      }
    }
  }

  private static long buildColumnView(SerializedColumnHeader column,
                                      ArrayDeque<ColumnOffsets> columnOffsets,
                                      DeviceMemoryBuffer combinedBuffer) {
    ColumnOffsets offsetsInfo = columnOffsets.remove();
    long[] childViews = null;
    try {
      SerializedColumnHeader[] children = column.getChildren();
      if (children != null) {
        childViews = new long[children.length];
        for (int i = 0; i < childViews.length; i++) {
          childViews[i] = buildColumnView(children[i], columnOffsets, combinedBuffer);
        }
      }
      DType dtype = column.getType();
      long bufferAddress = combinedBuffer.getAddress();
      long dataAddress = offsetsInfo.dataLen == 0 ? 0 : bufferAddress + offsetsInfo.data;
      long validityAddress = needsValidityBuffer(column.getNullCount())
          ? bufferAddress + offsetsInfo.validity : 0;
      long offsetsAddress = dtype.hasOffsets() ? bufferAddress + offsetsInfo.offsets : 0;
      return ColumnView.makeCudfColumnView(
          dtype.typeId.getNativeId(), dtype.getScale(),
          dataAddress, offsetsInfo.dataLen,
          offsetsAddress, validityAddress,
          (int) column.getNullCount(), (int) column.getRowCount(),
          childViews);
    } finally {
      if (childViews != null) {
        for (long childView : childViews) {
          ColumnView.deleteColumnView(childView);
        }
      }
    }
  }

  private static Table sliceUpColumnVectors(SerializedTableHeader header,
                                            DeviceMemoryBuffer combinedBuffer,
                                            HostMemoryBuffer combinedBufferOnHost) {
    try (NvtxRange range = new NvtxRange("bufferToTable", NvtxColor.PURPLE)) {
      ArrayDeque<ColumnOffsets> columnOffsets = buildIndex(header, combinedBufferOnHost);
      int numColumns = header.getNumColumns();
      ColumnVector[] vectors = new ColumnVector[numColumns];
      try {
        for (int i = 0; i < numColumns; i++) {
          SerializedColumnHeader column = header.getColumnHeader(i);
          long columnView = buildColumnView(column, columnOffsets, combinedBuffer);
          vectors[i] = ColumnVector.fromViewWithContiguousAllocation(columnView, combinedBuffer);
        }
        assert columnOffsets.isEmpty();
        return new Table(vectors);
      } finally {
        for (ColumnVector cv: vectors) {
          if (cv != null) {
            cv.close();
          }
        }
      }
    }
  }

  public static Table readAndConcat(SerializedTableHeader[] headers,
                                    HostMemoryBuffer[] dataBuffers) throws IOException {
    ContiguousTable ct = concatToContiguousTable(headers, dataBuffers);
    ct.getBuffer().close();
    return ct.getTable();
  }

  /**
   * Concatenate multiple tables in host memory into a contiguous table in device memory.
   * @param headers table headers corresponding to the host table buffers
   * @param dataBuffers host table buffer for each input table to be concatenated
   * @return contiguous table in device memory
   */
  public static ContiguousTable concatToContiguousTable(SerializedTableHeader[] headers,
                                                        HostMemoryBuffer[] dataBuffers) throws IOException {
    try (HostConcatResult concatResult = concatToHostBuffer(headers, dataBuffers)) {
      return concatResult.toContiguousTable();
    }
  }

  /**
   * Concatenate multiple tables in host memory into a single host table buffer.
   * @param headers table headers corresponding to the host table buffers
   * @param dataBuffers host table buffer for each input table to be concatenated
   * @param hostMemoryAllocator allocator for host memory buffers
   * @return host table header and buffer
   */
  public static HostConcatResult concatToHostBuffer(SerializedTableHeader[] headers,
                                                    HostMemoryBuffer[] dataBuffers,
                                                    HostMemoryAllocator hostMemoryAllocator
                                                    ) throws IOException {
    ColumnBufferProvider[][] providersPerColumn = providersFrom(headers, dataBuffers);
    try {
      SerializedTableHeader combined = calcConcatHeader(providersPerColumn);
      HostMemoryBuffer hostBuffer = hostMemoryAllocator.allocate(combined.dataLen);
      try {
        try (NvtxRange range = new NvtxRange("Concat Host Side", NvtxColor.GREEN)) {
          DataWriter writer = writerFrom(hostBuffer);
          int numColumns = combined.getNumColumns();
          for (int columnIdx = 0; columnIdx < numColumns; columnIdx++) {
            writeConcat(writer, combined.getColumnHeader(columnIdx), providersPerColumn[columnIdx]);
          }
        }
      } catch (Exception e) {
        hostBuffer.close();
        throw e;
      }

      return new HostConcatResult(combined, hostBuffer);
    } finally {
      closeAll(providersPerColumn);
    }
  }

    public static HostConcatResult concatToHostBuffer(SerializedTableHeader[] headers,
                                                      HostMemoryBuffer[] dataBuffers
                                                      ) throws IOException {
      return concatToHostBuffer(headers, dataBuffers, DefaultHostMemoryAllocator.get());
    }

  /**
   * Deserialize a serialized contiguous table into an array of host columns.
   *
   * @param header     serialized table header
   * @param hostBuffer buffer containing the data for all columns in the serialized table
   * @return array of host columns representing the data from the serialized table
   */
  public static HostColumnVector[] unpackHostColumnVectors(SerializedTableHeader header,
                                                           HostMemoryBuffer hostBuffer) {
    ArrayDeque<ColumnOffsets> columnOffsets = buildIndex(header, hostBuffer);
    int numColumns = header.getNumColumns();
    HostColumnVector[] columns = new HostColumnVector[numColumns];
    boolean succeeded = false;
    try {
      for (int i = 0; i < numColumns; i++) {
        SerializedColumnHeader column = header.getColumnHeader(i);
        columns[i] = (HostColumnVector) buildHostColumn(column, columnOffsets, hostBuffer, true);
      }
      assert columnOffsets.isEmpty();
      succeeded = true;
    } finally {
      if (!succeeded) {
        for (HostColumnVector c : columns) {
          if (c != null) {
            c.close();
          }
        }
      }
    }
    return columns;
  }

  /**
   * After reading a header for a table read the data portion into a host side buffer.
   * @param in the stream to read the data from.
   * @param header the header that finished just moments ago.
   * @param buffer the buffer to write the data into.  If there is not enough room to store
   *               the data in buffer it will not be read and header will still have dataRead
   *               set to false.
   * @throws IOException
   */
  public static void readTableIntoBuffer(InputStream in,
                                         SerializedTableHeader header,
                                         HostMemoryBuffer buffer) throws IOException {
    if (header.initialized &&
        (buffer.length >= header.dataLen)) {
      try (NvtxRange range = new NvtxRange("Read Data", NvtxColor.RED)) {
        buffer.copyFromStream(0, in, header.dataLen);
      }
      header.dataRead = true;
    }
  }

  public static TableAndRowCountPair readTableFrom(SerializedTableHeader header,
                                                   HostMemoryBuffer hostBuffer) {
    ContiguousTable contigTable = null;
    DeviceMemoryBuffer devBuffer = DeviceMemoryBuffer.allocate(hostBuffer.length);
    try {
      if (hostBuffer.length > 0) {
        try (NvtxRange range = new NvtxRange("Copy Data To Device", NvtxColor.WHITE)) {
          devBuffer.copyFromHostBuffer(hostBuffer);
        }
      }
      if (header.getNumColumns() > 0) {
        Table table = sliceUpColumnVectors(header, devBuffer, hostBuffer);
        contigTable = new ContiguousTable(table, devBuffer);
      }
    } finally {
      if (contigTable == null) {
        devBuffer.close();
      }
    }

    return new TableAndRowCountPair(header.numRows, contigTable);
  }

  /**
   * Read a serialize table from the given InputStream.
   * @param in the stream to read the table data from.
   * @param hostMemoryAllocator a host memory allocator for an intermediate host memory buffer
   * @return the deserialized table in device memory, or null if the stream has no table to read
   * from, an end of the stream at the very beginning.
   * @throws IOException on any error.
   * @throws EOFException if the data stream ended unexpectedly in the middle of processing.
   */
  public static TableAndRowCountPair readTableFrom(InputStream in,
      HostMemoryAllocator hostMemoryAllocator) throws IOException {
    DataInputStream din;
    if (in instanceof DataInputStream) {
      din = (DataInputStream) in;
    } else {
      din = new DataInputStream(in);
    }

    SerializedTableHeader header = new SerializedTableHeader(din);
    if (!header.initialized) {
      return new TableAndRowCountPair(0, null);
    }

    try (HostMemoryBuffer hostBuffer = hostMemoryAllocator.allocate(header.dataLen)) {
      if (header.dataLen > 0) {
        readTableIntoBuffer(din, header, hostBuffer);
      }
      return readTableFrom(header, hostBuffer);
    }
  }

  public static TableAndRowCountPair readTableFrom(InputStream in) throws IOException {
    return readTableFrom(in, DefaultHostMemoryAllocator.get());
  }

  /** Holds the result of deserializing a table. */
  public static final class TableAndRowCountPair implements Closeable {
    private final int numRows;
    private final ContiguousTable contigTable;

    public TableAndRowCountPair(int numRows, ContiguousTable table) {
      this.numRows = numRows;
      this.contigTable = table;
    }

    @Override
    public void close() {
      if (contigTable != null) {
        contigTable.close();
      }
    }

    /** Get the number of rows that were deserialized. */
    public int getNumRows() {
          return numRows;
      }

    /**
     * Get the Table that was deserialized or null if there was no data
     * (e.g.: rows without columns).
     * <p>NOTE: Ownership of the table is not transferred by this method.
     * The table is still owned by this instance and will be closed when this
     * instance is closed.
     */
    public Table getTable() {
      if (contigTable != null) {
        return contigTable.getTable();
      }
      return null;
    }

    /**
     * Get the ContiguousTable that was deserialized or null if there was no
     * data (e.g.: rows without columns).
     * <p>NOTE: Ownership of the contiguous table is not transferred by this
     * method. The contiguous table is still owned by this instance and will
     * be closed when this instance is closed.
     */
    public ContiguousTable getContiguousTable() {
      return contigTable;
    }
  }
}

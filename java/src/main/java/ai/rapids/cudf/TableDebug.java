/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
package ai.rapids.cudf;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Locale;
import java.util.function.Consumer;

/**
 * A debugging utility that prints the contents of tables and columns.
 * Obtain an instance with {@link #get()} or {@link #builder()}; by default the output goes to
 * the destination named by the {@value #OUTPUT_STREAM} system property, or standard error if it
 * is not set. This is slow and is meant for debugging only.
 */
public class TableDebug {

  /**
   * Specify one of
   * -Dai.rapids.cudf.debug.output=stderr       to print directly to standard error (default)
   * -Dai.rapids.cudf.debug.output=stdout       to print directly to standard output
   * -Dai.rapids.cudf.debug.output=log[_level]  to redirect to a logging subsystem that can
   * further be
   * configured.
   * Supported log levels:
   * debug (default)
   * info
   * warn
   * error
   */
  public static final String OUTPUT_STREAM = "ai.rapids.cudf.debug.output";
  private static final Logger log = LoggerFactory.getLogger(TableDebug.class);

  public enum Output {
    STDOUT(System.out::println),
    STDERR(System.err::println),
    LOG(log::debug),
    LOG_DEBUG(log::debug),
    LOG_INFO(log::info),
    LOG_WARN(log::warn),
    LOG_ERROR(log::error);

    private final Consumer<String> printFunc;

    Output(Consumer<String> pf) {
      this.printFunc = pf;
    }

    final void println(String s) {
      printFunc.accept(s);
    }
  }


  public static class Builder {
    private Output outputMode = Output.STDERR;

    public Builder() {
      try {
        outputMode = Output.valueOf(
            System.getProperty(OUTPUT_STREAM, Output.STDERR.name())
                .toUpperCase(Locale.US));
      } catch (Throwable e) {
        log.warn("Failed to parse the output mode", e);
      }
    }

    public Builder withOutput(Output outputMode) {
      this.outputMode = outputMode;
      return this;
    }

    public final TableDebug build() {
      return new TableDebug(outputMode);
    }
  }

  public static Builder builder() {
    return new Builder();
  }

  private static final TableDebug DEFAULT_DEBUG = builder().build();

  public static TableDebug get() {
    return DEFAULT_DEBUG;
  }

  private final Output output;

  private TableDebug(Output output) {
    this.output = output;
  }

  /**
   * Print the contents of a table. Note that this should never be
   * called from production code, as it is very slow.  Also note that this is not production
   * code.  You might need/want to update how the data shows up or add in support for more
   * types as this really is just for debugging.
   * @param name  the name of the table to print out.
   * @param table the table to print out.
   */
  public synchronized void debug(String name, Table table) {
    output.println("DEBUG " + name + " " + table);
    for (int col = 0; col < table.getNumberOfColumns(); col++) {
      debug(String.valueOf(col), table.getColumn(col));
    }
  }

  /**
   * Print the contents of a column. Note that this should never be
   * called from production code, as it is very slow.  Also note that this is not production
   * code.  You might need/want to update how the data shows up or add in support for more
   * types as this really is just for debugging.
   * @param name the name of the column to print out.
   * @param col  the column to print out.
   */
  public synchronized void debug(String name, ColumnView col) {
    debugGPUAddrs(name, col);
    try (HostColumnVector hostCol = col.copyToHost()) {
      debug(name, hostCol);
    }
  }

  private synchronized void debugGPUAddrs(String name, ColumnView col) {
    try (BaseDeviceMemoryBuffer data = col.getData();
         BaseDeviceMemoryBuffer validity = col.getValid()) {
      output.println("GPU COLUMN " + name + " - NC: " + col.getNullCount()
          + " DATA: " + data + " VAL: " + validity);
    }
    if (col.getType() == DType.STRUCT) {
      for (int i = 0; i < col.getNumChildren(); i++) {
        try (ColumnView child = col.getChildColumnView(i)) {
          debugGPUAddrs(name + ":CHILD_" + i, child);
        }
      }
    } else if (col.getType() == DType.LIST) {
      try (ColumnView child = col.getChildColumnView(0)) {
        debugGPUAddrs(name + ":DATA", child);
      }
    }
  }


  /**
   * Print the contents of a column. Note that this should never be
   * called from production code, as it is very slow.  Also note that this is not production
   * code.  You might need/want to update how the data shows up or add in support for more
   * types as this really is just for debugging.
   * @param name    the name of the column to print out.
   * @param hostCol the column to print out.
   */
  public synchronized void debug(String name, HostColumnVectorCore hostCol) {
    DType type = hostCol.getType();
    output.println("COLUMN " + name + " - " + type);
    if (type.isDecimalType()) {
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          output.println(i + " NULL");
        } else {
          output.println(i + " " + hostCol.getBigDecimal(i));
        }
      }
    } else if (DType.STRING.equals(type)) {
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          output.println(i + " NULL");
        } else {
          output.println(i + " \"" + hostCol.getJavaString(i) + "\" " +
              hexString(hostCol.getUTF8(i)));
        }
      }
    } else if (DType.INT32.equals(type)
        || DType.INT8.equals(type)
        || DType.INT16.equals(type)
        || DType.INT64.equals(type)
        || DType.TIMESTAMP_DAYS.equals(type)
        || DType.TIMESTAMP_SECONDS.equals(type)
        || DType.TIMESTAMP_MICROSECONDS.equals(type)
        || DType.TIMESTAMP_MILLISECONDS.equals(type)
        || DType.TIMESTAMP_NANOSECONDS.equals(type)
        || DType.UINT8.equals(type)
        || DType.UINT16.equals(type)
        || DType.UINT32.equals(type)
        || DType.UINT64.equals(type)) {
      debugInteger(hostCol, type);
    } else if (DType.BOOL8.equals(type)) {
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          output.println(i + " NULL");
        } else {
          output.println(i + " " + hostCol.getBoolean(i));
        }
      }
    } else if (DType.FLOAT64.equals(type)) {
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          output.println(i + " NULL");
        } else {
          output.println(i + " " + hostCol.getDouble(i));
        }
      }
    } else if (DType.FLOAT32.equals(type)) {
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          output.println(i + " NULL");
        } else {
          output.println(i + " " + hostCol.getFloat(i));
        }
      }
    } else if (DType.STRUCT.equals(type)) {
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          output.println(i + " NULL");
        } // The struct child columns are printed out later on.
      }
      for (int i = 0; i < hostCol.getNumChildren(); i++) {
        debug(name + ":CHILD_" + i, hostCol.getChildColumnView(i));
      }
    } else if (DType.LIST.equals(type)) {
      output.println("OFFSETS");
      for (int i = 0; i < hostCol.getRowCount(); i++) {
        if (hostCol.isNull(i)) {
          output.println(i + " NULL");
        } else {
          output.println(i + " [" + hostCol.getStartListOffset(i) + " - " +
              hostCol.getEndListOffset(i) + ")");
        }
      }
      debug(name + ":DATA", hostCol.getChildColumnView(0));
    } else {
      output.println("TYPE " + type + " NOT SUPPORTED FOR DEBUG PRINT");
    }
  }


  private void debugInteger(HostColumnVectorCore hostCol, DType intType) {
    for (int i = 0; i < hostCol.getRowCount(); i++) {
      if (hostCol.isNull(i)) {
        output.println(i + " NULL");
      } else {
        final int sizeInBytes = intType.getSizeInBytes();
        final Object value;
        switch (sizeInBytes) {
          case Byte.BYTES:
            value = hostCol.getByte(i);
            break;
          case Short.BYTES:
            value = hostCol.getShort(i);
            break;
          case Integer.BYTES:
            value = hostCol.getInt(i);
            break;
          case Long.BYTES:
            value = hostCol.getLong(i);
            break;
          default:
            throw new IllegalArgumentException("INFEASIBLE: Unsupported integer-like type " + intType);
        }
        output.println(i + " " + value);
      }
    }
  }


  private static String hexString(byte[] bytes) {
    StringBuilder str = new StringBuilder();
    for (byte b : bytes) {
      str.append(String.format("%02x", b & 0xff));
    }
    return str.toString();
  }
}

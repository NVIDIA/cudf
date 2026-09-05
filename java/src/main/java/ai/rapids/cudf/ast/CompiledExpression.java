/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import ai.rapids.cudf.ColumnVector;
import ai.rapids.cudf.MemoryCleaner;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

/** This class wraps a native compiled AST and must be closed to avoid native memory leaks. */
public class CompiledExpression implements AutoCloseable {
  enum CompilationMode {
    DEFAULT,
    JIT
  }

  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(CompiledExpression.class);

  private static class CompiledExpressionCleaner extends MemoryCleaner.Cleaner {
    private long nativeHandle;

    CompiledExpressionCleaner(long nativeHandle) {
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
          log.error("AN AST COMPILED EXPRESSION WAS LEAKED (ID: " +
              id + " " + Long.toHexString(origAddress));
        }
      }
      return neededCleanup;
    }

    @Override
    public boolean isClean() {
      return nativeHandle == 0;
    }
  }

  private final CompiledExpressionCleaner cleaner;
  private final CompilationMode mode;
  private boolean isClosed = false;

  /** Construct a compiled expression from a serialized AST */
  CompiledExpression(byte[] serializedExpression, CompilationMode mode) {
    this(mode == CompilationMode.JIT ? compileJit(serializedExpression) :
        compile(serializedExpression), mode);
  }

  /** Construct a compiled expression from a native compiled AST pointer */
  private CompiledExpression(long nativeHandle, CompilationMode mode) {
    this.cleaner = new CompiledExpressionCleaner(nativeHandle);
    this.mode = mode;
    MemoryCleaner.register(this, cleaner);
    cleaner.addRef();
  }

  /**
   * Compute a new column by applying this AST expression to the specified table. All column
   * references must use {@link TableReference#LEFT}; references to {@link TableReference#RIGHT}
   * are rejected because this operation has only one input table.
   * An expression produced by {@link AstExpression#compileJit()} always uses the JIT backend.
   * Otherwise, execution uses the process-level backend selection.
   *
   * @param table input table for this expression
   * @return new column computed from this expression applied to the input table
   * @throws ai.rapids.cudf.CudfException if the expression refers to
   *         {@link TableReference#RIGHT}, or if compilation or evaluation fails
   */
  public ColumnVector computeColumn(Table table) {
    long result;
    try {
      result = computeColumn(cleaner.nativeHandle, table.getNativeView());
    } finally {
      reachabilityFence(this);
      reachabilityFence(table);
    }
    return new ColumnVector(result);
  }

  /**
   * Compute a new table by applying expressions to the input table in one multi-output JIT
   * transform. Output column {@code i} contains the result of {@code expressions[i]}.
   *
   * @param table input table for the expressions
   * @param expressions non-empty JIT-compiled expressions to evaluate in output order
   * @return table containing one output column per expression
   * @throws NullPointerException if the table, expression array, or an expression is null
   * @throws IllegalArgumentException if no expressions are provided or an expression was not
   *         produced by {@link AstExpression#compileJit()}
   * @throws IllegalStateException if the table or an expression is closed
   * @throws ai.rapids.cudf.CudfException if JIT compilation or evaluation fails
   */
  public static Table computeTableJit(Table table, CompiledExpression... expressions) {
    Objects.requireNonNull(table, "table");
    Objects.requireNonNull(expressions, "expressions");
    if (expressions.length == 0) {
      throw new IllegalArgumentException("At least one expression is required");
    }

    long tableHandle = table.getNativeView();
    if (tableHandle == 0) {
      throw new IllegalStateException("Table is closed");
    }

    CompiledExpression[] expressionRefs = expressions.clone();
    long[] nativeHandles = getJitNativeHandles(expressionRefs);
    long[] result;
    try {
      result = computeTableJitNative(nativeHandles, tableHandle);
    } finally {
      reachabilityFence(table);
      reachabilityFence(expressionRefs);
    }
    return new Table(result);
  }

  static long[] getJitNativeHandles(CompiledExpression[] expressions) {
    long[] nativeHandles = new long[expressions.length];
    for (int i = 0; i < expressions.length; i++) {
      CompiledExpression expression = Objects.requireNonNull(
          expressions[i], "expression " + i + " is null");
      if (expression.mode != CompilationMode.JIT) {
        throw new IllegalArgumentException("Expression " + i + " was not compiled for JIT");
      }
      nativeHandles[i] = expression.cleaner.nativeHandle;
      if (nativeHandles[i] == 0) {
        throw new IllegalStateException("Expression " + i + " is closed");
      }
    }
    return nativeHandles;
  }

  static void reachabilityFence(Object object) {
    if (object != null) {
      synchronized (object) {
        // The monitor operation is a Java 8 reachability fence.
      }
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

  /**
   * Returns the native address of a default-compatible compiled expression.
   * Intended for internal cudf use only.
   *
   * @throws IllegalStateException if this expression was produced by
   *         {@link AstExpression#compileJit()}
   */
  public long getNativeHandle() {
    if (mode == CompilationMode.JIT) {
      throw new IllegalStateException(
          "JIT-compiled expressions cannot be used by a default AST consumer");
    }
    return cleaner.nativeHandle;
  }

  private static native long compile(byte[] serializedExpression);
  private static native long compileJit(byte[] serializedExpression);
  private static native long computeColumn(long astHandle, long tableHandle);
  private static native long[] computeTableJitNative(long[] astHandles, long tableHandle);
  private static native void destroy(long handle);
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import ai.rapids.cudf.MemoryCleaner;
import ai.rapids.cudf.NativeDepsLoader;
import ai.rapids.cudf.Table;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;

/**
 * A reusable AST JIT program specialized to an input schema.
 * Construction lowers the expressions and retrieves their JIT kernel. Subsequent calls reuse that
 * kernel with tables whose referenced columns have compatible types and nullability.
 * Callers must ensure that {@link #close()} does not overlap with {@link #computeTable(Table)}.
 */
public final class AstJitProgram implements AutoCloseable {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  private static final Logger log = LoggerFactory.getLogger(AstJitProgram.class);

  private static final class AstJitProgramCleaner extends MemoryCleaner.Cleaner {
    private long nativeHandle;

    AstJitProgramCleaner(long nativeHandle) {
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
          log.error("AN AST JIT PROGRAM WAS LEAKED (ID: " +
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

  private final AstJitProgramCleaner cleaner;
  private boolean isClosed = false;

  private AstJitProgram(long nativeHandle) {
    cleaner = new AstJitProgramCleaner(nativeHandle);
    MemoryCleaner.register(this, cleaner);
    cleaner.addRef();
  }

  /**
   * Compile a reusable program from one or more JIT-compiled expressions.
   * The schema table and expressions are inspected during construction but are not retained. The
   * returned program owns any literal values required by later evaluations.
   *
   * @param schemaTable table whose referenced column schema is used to compile the program
   * @param expressions non-empty JIT-compiled expressions in output order
   * @return a reusable AST JIT program
   * @throws NullPointerException if the table, expression array, or an expression is null
   * @throws IllegalArgumentException if no expressions are provided or an expression was not
   *         produced by {@link AstExpression#compileJit()}
   * @throws IllegalStateException if the table or an expression is closed
   * @throws ai.rapids.cudf.CudfException if JIT compilation fails
   */
  public static AstJitProgram compile(Table schemaTable, CompiledExpression... expressions) {
    Objects.requireNonNull(schemaTable, "schemaTable");
    Objects.requireNonNull(expressions, "expressions");
    if (expressions.length == 0) {
      throw new IllegalArgumentException("At least one expression is required");
    }

    long tableHandle = schemaTable.getNativeView();
    if (tableHandle == 0) {
      throw new IllegalStateException("Table is closed");
    }

    CompiledExpression[] expressionRefs = expressions.clone();
    long[] nativeHandles = CompiledExpression.getJitNativeHandles(expressionRefs);
    long programHandle;
    try {
      programHandle = create(nativeHandles, tableHandle);
    } finally {
      CompiledExpression.reachabilityFence(schemaTable);
      CompiledExpression.reachabilityFence(expressionRefs);
    }
    return new AstJitProgram(programHandle);
  }

  /**
   * Evaluate this program on a table with a compatible referenced-column schema.
   * The row count and unreferenced columns may differ from the schema table used at compilation.
   * Calling {@link #close()} while an evaluation is in progress is unsupported.
   *
   * @param table input table for expression evaluation
   * @return table containing the program outputs in expression order
   * @throws NullPointerException if the table is null
   * @throws IllegalStateException if the program or table is closed
   * @throws ai.rapids.cudf.CudfException if the referenced-column schema is incompatible or
   *         evaluation fails
   */
  public Table computeTable(Table table) {
    Objects.requireNonNull(table, "table");
    long programHandle = cleaner.nativeHandle;
    if (programHandle == 0) {
      throw new IllegalStateException("AST JIT program is closed");
    }
    long tableHandle = table.getNativeView();
    if (tableHandle == 0) {
      throw new IllegalStateException("Table is closed");
    }

    long[] result;
    try {
      result = computeTableNative(programHandle, tableHandle);
    } finally {
      CompiledExpression.reachabilityFence(this);
      CompiledExpression.reachabilityFence(table);
    }
    return new Table(result);
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

  private static native long create(long[] astHandles, long tableHandle);
  private static native long[] computeTableNative(long programHandle, long tableHandle);
  private static native void destroy(long handle);
}

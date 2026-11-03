/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf.ast;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;

/**
 * A reference to a column in an input table by name.
 *
 * <p>Unlike {@link ColumnReference}, which references a column by its integer position
 * within the input table, this node carries the column name as a string. The name is
 * resolved to an actual column when the AST is evaluated. This is useful in contexts
 * where the column index is not known at AST-construction time, e.g. when the filter
 * expression is built before the file's schema has been read. The Parquet hybrid scan
 * reader relies on this node for filter expressions.
 */
public final class ColumnNameReference extends AstExpression {
  private final String columnName;

  /**
   * Construct a column reference using the column name.
   *
   * @param columnName the name of the column to reference. Must not be null or empty.
   */
  public ColumnNameReference(String columnName) {
    if (columnName == null) {
      throw new IllegalArgumentException("Column name cannot be null");
    }
    if (columnName.isEmpty()) {
      throw new IllegalArgumentException("Column name cannot be empty");
    }
    this.columnName = columnName;
  }

  /** @return the column name */
  public String getColumnName() {
    return columnName;
  }

  @Override
  int getSerializedSize() {
    byte[] nameBytes = columnName.getBytes(StandardCharsets.UTF_8);
    // node type + string length (int) + string bytes
    return ExpressionType.COLUMN_NAME_REFERENCE.getSerializedSize() +
        Integer.BYTES +
        nameBytes.length;
  }

  @Override
  void serialize(ByteBuffer bb) {
    ExpressionType.COLUMN_NAME_REFERENCE.serialize(bb);

    byte[] nameBytes = columnName.getBytes(StandardCharsets.UTF_8);
    bb.putInt(nameBytes.length);
    bb.put(nameBytes);
  }

  @Override
  public String toString() {
    return "COLUMN(\"" + columnName + "\")";
  }
}

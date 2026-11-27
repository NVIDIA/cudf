/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

/**
 * The type and nullability of one output column of
 * {@link ColumnView#transform(ColumnView[], String, boolean, boolean, TransformOutput...)}.
 * Mirrors {@code cudf::transform_output}.
 */
public final class TransformOutput {
  private final DType type;
  private final OutputNullability nullability;

  /**
   * @param type the type of the output column, fixed-width or STRING
   * @param nullability whether the output column may have a null mask
   */
  public TransformOutput(DType type, OutputNullability nullability) {
    if (type == null || nullability == null) {
      throw new IllegalArgumentException("type and nullability must not be null");
    }
    if (type.getSizeInBytes() == 0 && !type.equals(DType.STRING)) {
      throw new IllegalArgumentException(
          "transform only supports fixed-width and STRING output types, got " + type);
    }
    this.type = type;
    this.nullability = nullability;
  }

  /**
   * An output of the given type with {@link OutputNullability#PRESERVE}.
   * @param type the type of the output column, fixed-width or STRING
   */
  public TransformOutput(DType type) {
    this(type, OutputNullability.PRESERVE);
  }

  public DType getType() {
    return type;
  }

  public OutputNullability getNullability() {
    return nullability;
  }
}

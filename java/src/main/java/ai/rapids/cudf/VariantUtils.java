/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package ai.rapids.cudf;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * Utility methods for cuDF's experimental Parquet Variant extraction APIs.
 */
public class VariantUtils {
  static {
    NativeDepsLoader.loadNativeDeps();
  }

  // Keep in sync with the target types accepted by cuDF Variant extraction/cast:
  // cpp/include/cudf/io/experimental/variant.hpp and
  // cpp/src/io/parquet/experimental/variant_extract.cu:is_variant_castable.
  private static final List<DType> SUPPORTED_TYPES = Arrays.asList(
      DType.STRING, DType.INT8, DType.INT16, DType.INT32, DType.INT64);

  private VariantUtils() {}

  private static void validateTargetType(DType targetType) {
    Objects.requireNonNull(targetType, "targetType");
    if (!SUPPORTED_TYPES.contains(targetType)) {
      throw new IllegalArgumentException("unsupported Variant target type: " + targetType +
          "; supported types are " + SUPPORTED_TYPES);
    }
  }

  /**
   * Extract raw Variant-encoded value bytes at {@code path} from a Variant struct column.
   *
   * @param variantStruct Variant materialization: STRUCT(metadata LIST&lt;UINT8&gt;,
   *                      value LIST&lt;UINT8&gt;, optional shredded children...)
   * @param path JSONPath-like path accepted by cuDF's Variant extractor. Paths are expected to
   *             be ASCII object-field paths like {@code x}, {@code $.x}, or {@code $.x.y}.
   * @return LIST&lt;UINT8&gt; column of raw encoded Variant values
   */
  public static ColumnVector getVariantFieldValue(ColumnView variantStruct, String path) {
    Objects.requireNonNull(variantStruct, "variantStruct");
    Objects.requireNonNull(path, "path");
    return new ColumnVector(getVariantFieldValue(variantStruct.getNativeView(), path));
  }

  /**
   * Decode raw Variant-encoded value bytes into {@code targetType}. Supported target types are
   * {@link DType#STRING}, {@link DType#INT8}, {@link DType#INT16}, {@link DType#INT32}, and
   * {@link DType#INT64}.
   */
  public static ColumnVector castVariantValue(ColumnView valueBytes, DType targetType) {
    Objects.requireNonNull(valueBytes, "valueBytes");
    validateTargetType(targetType);
    return new ColumnVector(castVariantValue(
        valueBytes.getNativeView(), targetType.getTypeId().getNativeId()));
  }

  /**
   * Extract a Variant field and decode it into {@code targetType} in one native call.
   * Supported target types are {@link DType#STRING}, {@link DType#INT8}, {@link DType#INT16},
   * {@link DType#INT32}, and {@link DType#INT64}.
   */
  public static ColumnVector extractVariantField(
      ColumnView variantStruct, String path, DType targetType) {
    Objects.requireNonNull(variantStruct, "variantStruct");
    Objects.requireNonNull(path, "path");
    validateTargetType(targetType);
    return new ColumnVector(extractVariantField(
        variantStruct.getNativeView(), path, targetType.getTypeId().getNativeId()));
  }

  private static native long getVariantFieldValue(long variantStructHandle, String path);

  private static native long castVariantValue(long valueBytesHandle, int cudfTypeId);

  private static native long extractVariantField(
      long variantStructHandle, String path, int cudfTypeId);
}

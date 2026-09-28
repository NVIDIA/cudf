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
      DType.STRING, DType.INT8, DType.INT16, DType.INT32, DType.INT64,
      DType.FLOAT32, DType.FLOAT64, DType.BOOL8);

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
   * @param path JSONPath-like path accepted by cuDF's Variant extractor. Object-field steps use
   *             dot notation and zero-based array-index steps use bracket notation, for example
   *             {@code x}, {@code $.x.y}, {@code $[0]}, or {@code $.a[0].b}. Missing fields,
   *             out-of-bounds indices, and container mismatches produce null rows. Wildcards,
   *             negative indices, and quoted names inside brackets are not supported.
   * @return LIST&lt;UINT8&gt; column of raw encoded Variant values
   */
  public static ColumnVector getVariantFieldValue(ColumnView variantStruct, String path) {
    Objects.requireNonNull(variantStruct, "variantStruct");
    Objects.requireNonNull(path, "path");
    return new ColumnVector(getVariantFieldValue(variantStruct.getNativeView(), path));
  }

  /**
   * Return the logical type ID of each raw Variant-encoded value.
   *
   * <p>The input must be a LIST&lt;UINT8&gt; column. The result is a UINT8 column containing the
   * IDs defined by {@link VariantLogicalType}. A result row is null when the input row is null,
   * the value blob is empty, or its header is unrecognized. An encoded Variant null is represented
   * by a valid {@link VariantLogicalType#NULL_VALUE} row. Only the header byte is inspected, so a
   * recognized header is classified even if the remaining payload is truncated.
   *
   * <p>This API mirrors an experimental libcudf API and is subject to change.
   *
   * @param valueBytes LIST&lt;UINT8&gt; column of raw Variant-encoded values
   * @return owning UINT8 column of logical type IDs
   */
  public static ColumnVector getVariantTypeId(ColumnView valueBytes) {
    Objects.requireNonNull(valueBytes, "valueBytes");
    return new ColumnVector(getVariantTypeId(valueBytes.getNativeView()));
  }

  /**
   * Decode raw Variant-encoded value bytes into {@code targetType}. Supported target types are
   * {@link DType#STRING}, {@link DType#INT8}, {@link DType#INT16}, {@link DType#INT32},
   * {@link DType#INT64}, {@link DType#FLOAT32}, {@link DType#FLOAT64}, and {@link DType#BOOL8}.
   * Decoding requires the encoded physical type to exactly match {@code targetType}; no numeric
   * conversions are performed. Input nulls, encoded Variant nulls, and physical-type mismatches
   * produce null output rows.
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
   * {@link DType#INT32}, {@link DType#INT64}, {@link DType#FLOAT32}, {@link DType#FLOAT64}, and
   * {@link DType#BOOL8}.
   * Decoding requires the encoded physical type to exactly match {@code targetType}; no numeric
   * conversions are performed. Missing fields, input nulls, encoded Variant nulls, and
   * physical-type mismatches produce null output rows.
   *
   * @param variantStruct Variant materialization: STRUCT(metadata LIST&lt;UINT8&gt;,
   *                      value LIST&lt;UINT8&gt;, optional shredded children...)
   * @param path JSONPath-like path accepted by cuDF's Variant extractor. Object-field steps use
   *             dot notation and zero-based array-index steps use bracket notation, for example
   *             {@code x}, {@code $.x.y}, {@code $[0]}, or {@code $.a[0].b}. Missing fields,
   *             out-of-bounds indices, and container mismatches produce null rows. Wildcards,
   *             negative indices, and quoted names inside brackets are not supported.
   * @param targetType decoded output type
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

  private static native long getVariantTypeId(long valueBytesHandle);

  private static native long castVariantValue(long valueBytesHandle, int cudfTypeId);

  private static native long extractVariantField(
      long variantStructHandle, String path, int cudfTypeId);
}

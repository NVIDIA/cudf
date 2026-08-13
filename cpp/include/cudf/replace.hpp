/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

/**
 * @file
 * @brief APIs for replacing null and specified values within columns.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup transformation_replace
 * @{
 */

/**
 * @brief Policy to specify the position of replacement values relative to null rows
 *
 * `PRECEDING` means the replacement value is the first non-null value preceding the null row.
 * `FOLLOWING` means the replacement value is the first non-null value following the null row.
 */
enum class replace_policy : bool { PRECEDING, FOLLOWING };

/**
 * @brief Replaces all null values in a column with corresponding values of another column
 *
 * If `input[i]` is NULL, then `output[i]` will contain `replacement[i]`.
 * `input` and `replacement` must be of the same type and size.
 *
 * @param[in] input A column whose null values will be replaced
 * @param[in] replacement A cudf::column whose values will replace null values in input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate device memory of the returned column
 *
 * @returns A copy of `input` with the null values replaced with corresponding values from
 * `replacement`.
 */
std::unique_ptr<column> replace_nulls(
  column_view const& input,
  column_view const& replacement,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Replaces all null values in a column with a scalar.
 *
 * If `input[i]` is NULL, then `output[i]` will contain `replacement`.
 * `input` and `replacement` must have the same type.
 *
 * @param[in] input A column whose null values will be replaced
 * @param[in] replacement Scalar used to replace null values in `input`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate device memory of the returned column
 *
 * @returns Copy of `input` with null values replaced by `replacement`
 */
std::unique_ptr<column> replace_nulls(
  column_view const& input,
  scalar const& replacement,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Replaces all null values in a column with the first non-null value that precedes/follows.
 *
 * If `input[i]` is NULL, then `output[i]` will contain the first non-null value that precedes or
 * follows the null value, based on `replace_policy`.
 *
 * @param[in] input A column whose null values will be replaced
 * @param[in] replace_policy Specify the position of replacement values relative to null values
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate device memory of the returned column
 *
 * @returns Copy of `input` with null values replaced based on `replace_policy`
 */
std::unique_ptr<column> replace_nulls(
  column_view const& input,
  replace_policy const& replace_policy,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Replaces all NaN values in a column with corresponding values from another column
 *
 * If `input[i]` is NaN, then `output[i]` will contain `replacement[i]`.
 * @code{.pseudo}
 * input        = {1.0, NaN, 4.0}
 * replacement  = {3.0, 9.0, 7.0}
 * output       = {1.0, 9.0, 4.0}
 * @endcode
 *
 * @note Nulls are not considered as NaN
 *
 * @throws cudf::logic_error If `input` and `replacement` are of different type or size.
 * @throws cudf::logic_error If `input` or `replacement` are not of floating-point dtype.
 *
 * @param input A column whose NaN values will be replaced
 * @param replacement A cudf::column whose values will replace NaN values in input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A copy of `input` with the NaN values replaced with corresponding values from
 * `replacement`.
 */
std::unique_ptr<column> replace_nans(
  column_view const& input,
  column_view const& replacement,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Replaces all NaN values in a column with a scalar
 *
 * If `input[i]` is NaN, then `output[i]` will contain `replacement`.
 * @code{.pseudo}
 * input        = {1.0, NaN, 4.0}
 * replacement  = 7.0
 * output       = {1.0, 7.0, 4.0}
 * @endcode
 *
 * @note Nulls are not considered as NaN
 *
 * @throws cudf::logic_error If `input` and `replacement` are of different type.
 * @throws cudf::logic_error If `input` or `replacement` are not of floating-point dtype.
 *
 * @param input A column whose NaN values will be replaced
 * @param replacement A cudf::scalar whose value will replace NaN values in input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return A copy of `input` with the NaN values replaced by `replacement`
 */
std::unique_ptr<column> replace_nans(
  column_view const& input,
  scalar const& replacement,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Return a copy of `input_col` replacing any `values_to_replace[i]`
 * found with `replacement_values[i]`.
 *
 * @param input_col The column to find and replace values in
 * @param values_to_replace The values to replace
 * @param replacement_values The values to replace with
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 *
 * @returns Copy of `input_col` with specified values replaced
 */
std::unique_ptr<column> find_and_replace_all(
  column_view const& input_col,
  column_view const& values_to_replace,
  column_view const& replacement_values,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Replaces values less than `lo` in `input` with `lo_replace`,
 * and values greater than `hi` with `hi_replace`.
 *
 * if `lo` is invalid, then lo will not be considered while
 * evaluating the input (Essentially considered minimum value of that type).
 * if `hi` is invalid, then hi will not be considered while
 * evaluating the input (Essentially considered maximum value of that type).
 *
 * @note: If `lo` is valid then `lo_replace` should be valid
 *        If `hi` is valid then `hi_replace` should be valid
 *
 * ```
 * Example:
 *    input: {1, 2, 3, NULL, 5, 6, 7}
 *
 *    valid lo and hi
 *    lo: 3, hi: 5, lo_replace : 0, hi_replace : 16
 *    output:{0, 0, 3, NULL, 5, 16, 16}
 *
 *    invalid lo
 *    lo: NULL, hi: 5, lo_replace : 0, hi_replace : 16
 *    output:{1, 2, 3, NULL, 5, 16, 16}
 *
 *    invalid hi
 *    lo: 3, hi: NULL, lo_replace : 0, hi_replace : 16
 *    output:{0, 0, 3, NULL, 5, 6, 7}
 * ```
 *
 * @throws cudf::logic_error if `lo.type() != hi.type()`
 * @throws cudf::logic_error if `lo_replace.type() != hi_replace.type()`
 * @throws cudf::logic_error if `lo.type() != lo_replace.type()`
 * @throws cudf::logic_error if `lo.type() != input.type()`
 *
 * @param[in] input Column whose elements will be clamped
 * @param[in] lo Minimum clamp value. All elements less than `lo` will be replaced by `lo_replace`
 * Ignored if null.
 * @param[in] lo_replace All elements less than `lo` will be replaced by `lo_replace`
 * @param[in] hi Maximum clamp value. All elements greater than `hi` will be replaced by
 * `hi_replace`. Ignored if null.
 * @param[in] hi_replace All elements greater than `hi` will be replaced by `hi_replace`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate device memory of the returned column
 *
 * @return Returns a clamped column as per `lo` and `hi` boundaries
 */
std::unique_ptr<column> clamp(
  column_view const& input,
  scalar const& lo,
  scalar const& lo_replace,
  scalar const& hi,
  scalar const& hi_replace,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Replaces values less than `lo` in `input` with `lo`,
 * and values greater than `hi` with `hi`.
 *
 * if `lo` is invalid, then lo will not be considered while
 * evaluating the input (Essentially considered minimum value of that type).
 * if `hi` is invalid, then hi will not be considered while
 * evaluating the input (Essentially considered maximum value of that type).
 *
 * ```
 * Example:
 *    input: {1, 2, 3, NULL, 5, 6, 7}
 *
 *    valid lo and hi
 *    lo: 3, hi: 5
 *    output:{3, 3, 3, NULL, 5, 5, 5}
 *
 *    invalid lo
 *    lo: NULL, hi:5
 *    output:{1, 2, 3, NULL, 5, 5, 5}
 *
 *    invalid hi
 *    lo: 3, hi:NULL
 *    output:{3, 3, 3, NULL, 5, 6, 7}
 * ```
 *
 * @throws cudf::logic_error if `lo.type() != hi.type()`
 * @throws cudf::logic_error if `lo.type() != input.type()`
 *
 * @param[in] input Column whose elements will be clamped
 * @param[in] lo Minimum clamp value. All elements less than `lo` will be replaced by `lo` Ignored
 * if null.
 * @param[in] hi Maximum clamp value. All elements greater than `hi` will be replaced by `hi`
 * Ignored if null.
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource used to allocate device memory of the returned column
 *
 * @return Returns a clamped column as per `lo` and `hi` boundaries
 */
std::unique_ptr<column> clamp(
  column_view const& input,
  scalar const& lo,
  scalar const& hi,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Copies from a column of floating-point elements and replaces `-NaN` and `-0.0` with `+NaN`
 * and `+0.0`, respectively.
 *
 * Converts floating point values from @p input using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in] input column_view of floating-point elements to copy and normalize
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Device memory resource for allocating output data
 *
 * @returns new column with the modified data
 */
std::unique_ptr<column> normalize_nans_and_zeros(
  column_view const& input,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Modifies a column of floating-point elements to replace all `-NaN` and `-0.0` with `+NaN`
 * and `+0.0`, respectively.
 *
 * Converts floating point values from @p in_out using the following rules:
 *        Convert  -NaN  -> NaN
 *        Convert  -0.0  -> 0.0
 *
 * @throws cudf::logic_error if column does not have floating point data type.
 * @param[in, out] in_out of floating-point elements to normalize
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void normalize_nans_and_zeros(mutable_column_view& in_out,
                              rmm::cuda_stream_view stream = cudf::get_default_stream());

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <optional>

namespace cudf {
namespace reduction::detail {

/**
 * @brief Compute sum of each segment in the input column
 *
 * If an input segment is empty, the segment result is null.
 *
 * If `null_handling==null_policy::INCLUDE`, all elements in a segment must be valid
 * for the reduced value to be valid.
 * If `null_handling==null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`.
 * @throw cudf::logic_error if `output_dtype` is not an arithmetic type.
 *
 * @param col Input column data
 * @param offsets Indices to identify segment boundaries within input `col`
 * @param output_dtype Data type of the output column
 * @param null_handling Specifies how null elements are processed for each segment
 * @param init Initial value of each segment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Sums of segments as type `output_dtype`
 */
std::unique_ptr<column> segmented_sum(column_view const& col,
                                      device_span<size_type const> offsets,
                                      data_type const output_dtype,
                                      null_policy null_handling,
                                      std::optional<std::reference_wrapper<scalar const>> init,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Computes product of each segment in the input column
 *
 * If an input segment is empty, the segment result is null.
 *
 * If `null_handling==null_policy::INCLUDE`, all elements in a segment must be valid
 * for the reduced value to be valid.
 * If `null_handling==null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`.
 * @throw cudf::logic_error if `output_dtype` is not an arithmetic type.
 *
 * @param col Input column data
 * @param offsets Indices to identify segment boundaries within input `col`
 * @param output_dtype Data type of the output column
 * @param null_handling Specifies how null elements are processed for each segment
 * @param init Initial value of each segment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Product of segments as type `output_dtype`
 */
std::unique_ptr<column> segmented_product(column_view const& col,
                                          device_span<size_type const> offsets,
                                          data_type const output_dtype,
                                          null_policy null_handling,
                                          std::optional<std::reference_wrapper<scalar const>> init,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @brief Compute minimum of each segment in the input column
 *
 * If an input segment is empty, the segment result is null.
 *
 * If `null_handling==null_policy::INCLUDE`, all elements in a segment must be valid
 * for the reduced value to be valid.
 * If `null_handling==null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`.
 *
 * @param col Input column data
 * @param offsets Indices to identify segment boundaries within input `col`
 * @param output_dtype Data type of the output column
 * @param null_handling Specifies how null elements are processed for each segment
 * @param init Initial value of each segment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Minimums of segments as type `output_dtype`
 */
std::unique_ptr<column> segmented_min(column_view const& col,
                                      device_span<size_type const> offsets,
                                      data_type const output_dtype,
                                      null_policy null_handling,
                                      std::optional<std::reference_wrapper<scalar const>> init,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Compute maximum of each segment in the input column
 *
 * If an input segment is empty, the segment result is null.
 *
 * If `null_handling==null_policy::INCLUDE`, all elements in a segment must be valid
 * for the reduced value to be valid.
 * If `null_handling==null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 *
 * @throw cudf::logic_error if input column type is not convertible to `output_dtype`.
 *
 * @param col Input column data
 * @param offsets Indices to identify segment boundaries within input `col`
 * @param output_dtype Data type of the output column
 * @param null_handling Specifies how null elements are processed for each segment
 * @param init Initial value of each segment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Maximums of segments as type `output_dtype`
 */
std::unique_ptr<column> segmented_max(column_view const& col,
                                      device_span<size_type const> offsets,
                                      data_type const output_dtype,
                                      null_policy null_handling,
                                      std::optional<std::reference_wrapper<scalar const>> init,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Compute if any of the values in the segment are true when typecasted to bool
 *
 * If an input segment is empty, the segment result is null.
 *
 * If `null_handling==null_policy::INCLUDE`, all elements in a segment must be valid
 * for the reduced value to be valid.
 * If `null_handling==null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 *
 * @throw cudf::logic_error if input column type is not convertible to bool.
 * @throw cudf::logic_error if `output_dtype` is not BOOL8.
 *
 * @param col Input column data
 * @param offsets Indices to identify segment boundaries within input `col`
 * @param output_dtype Data type of the output column
 * @param null_handling Specifies how null elements are processed for each segment
 * @param init Initial value of each segment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of type BOOL8 for the results of the segments
 */
std::unique_ptr<column> segmented_any(column_view const& col,
                                      device_span<size_type const> offsets,
                                      data_type const output_dtype,
                                      null_policy null_handling,
                                      std::optional<std::reference_wrapper<scalar const>> init,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Compute if all of the values in the segment are true when typecasted to bool
 *
 * If an input segment is empty, the segment result is null.
 *
 * If `null_handling==null_policy::INCLUDE`, all elements in a segment must be valid
 * for the reduced value to be valid.
 * If `null_handling==null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 *
 * @throw cudf::logic_error if input column type is not convertible to bool.
 * @throw cudf::logic_error if `output_dtype` is not BOOL8.
 *
 * @param col Input column data
 * @param offsets Indices to identify segment boundaries within input `col`
 * @param output_dtype Data type of the output column
 * @param null_handling Specifies how null elements are processed for each segment
 * @param init Initial value of each segment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of BOOL8 for the results of the segments
 */
std::unique_ptr<column> segmented_all(column_view const& col,
                                      device_span<size_type const> offsets,
                                      data_type const output_dtype,
                                      null_policy null_handling,
                                      std::optional<std::reference_wrapper<scalar const>> init,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @brief Computes mean of elements of segments in the input column
 *
 * If input segment is empty, the segment result is null.
 *
 * If `null_handling==null_policy::INCLUDE`, all elements in a segment must be valid
 * for the reduced value to be valid.
 * If `null_handling==null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not floating point type
 *
 * @param col Input column data
 * @param offsets Indices to identify segment boundaries within input `col`
 * @param output_dtype Data type of the output column
 * @param null_handling Specifies how null elements are processed for each segment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of `output_dtype` for the reduction results of the segments
 */
std::unique_ptr<column> segmented_mean(column_view const& col,
                                       device_span<size_type const> offsets,
                                       data_type const output_dtype,
                                       null_policy null_handling,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @brief Computes sum of squares of elements of segments in the input column
 *
 * If input segment is empty, the segment result is null.
 *
 * If `null_handling==null_policy::INCLUDE`, all elements in a segment must be valid
 * for the reduced value to be valid.
 * If `null_handling==null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not an arithmetic type
 *
 * @param col Input column data
 * @param offsets Indices to identify segment boundaries within input `col`
 * @param output_dtype Data type of the output column
 * @param null_handling Specifies how null elements are processed for each segment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of `output_dtype` for the reduction results of the segments
 */
std::unique_ptr<column> segmented_sum_of_squares(column_view const& col,
                                                 device_span<size_type const> offsets,
                                                 data_type const output_dtype,
                                                 null_policy null_handling,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr);

/**
 * @brief Computes the standard deviation of elements of segments in the input column
 *
 * If input segment is empty, the segment result is null.
 *
 * If `null_handling==null_policy::INCLUDE`, all elements in a segment must be valid
 * for the reduced value to be valid.
 * If `null_handling==null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not floating point type
 *
 * @param col Input column data
 * @param offsets Indices to identify segment boundaries within input `col`
 * @param output_dtype Data type of the output column
 * @param null_handling Specifies how null elements are processed for each segment
 * @param ddof Delta degrees of freedom.
 *             The divisor used is N - ddof, where N the number of elements in each segment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of `output_dtype` for the reduction results of the segments
 */
std::unique_ptr<column> segmented_standard_deviation(column_view const& col,
                                                     device_span<size_type const> offsets,
                                                     data_type const output_dtype,
                                                     null_policy null_handling,
                                                     size_type ddof,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr);

/**
 * @brief Computes the variance of elements of segments in the input column
 *
 * If input segment is empty, the segment result is null.
 *
 * If `null_handling==null_policy::INCLUDE`, all elements in a segment must be valid
 * for the reduced value to be valid.
 * If `null_handling==null_policy::EXCLUDE`, the reduced value is valid if any element
 * in the segment is valid.
 *
 * @throw cudf::logic_error if input column type is not arithmetic type
 * @throw cudf::logic_error if `output_dtype` is not floating point type
 *
 * @param col Input column data
 * @param offsets Indices to identify segment boundaries within input `col`
 * @param output_dtype Data type of the output column
 * @param null_handling Specifies how null elements are processed for each segment
 * @param ddof Delta degrees of freedom.
 *             The divisor used is N - ddof, where N the number of elements in each segment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of `output_dtype` for the reduction results of the segments
 */
std::unique_ptr<column> segmented_variance(column_view const& col,
                                           device_span<size_type const> offsets,
                                           data_type const output_dtype,
                                           null_policy null_handling,
                                           size_type ddof,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr);

/**
 * @brief Counts the number of unique values within each segment of a column
 *
 * Unique entries are counted by comparing adjacent values so the column segments
 * are expected to be sorted before calling this function otherwise the results
 * are undefined.
 *
 * If any input segment is empty, that segment's result is null.
 *
 * If `null_handling==null_policy::INCLUDE`, the segment count is the number of
 * unique values +1 which includes all the null entries in that segment.
 * If `null_handling==null_policy::EXCLUDE`, the segment count does not include nulls.
 *
 * @throw cudf::logic_error if input column type is a nested type
 *
 * @param col Input column data
 * @param offsets Indices to identify segment boundaries within input `col`
 * @param null_handling Specifies how null elements are processed for each segment
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return Column of unique counts per segment
 */
std::unique_ptr<column> segmented_nunique(column_view const& col,
                                          device_span<size_type const> offsets,
                                          null_policy null_handling,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr);

}  // namespace reduction::detail
}  // namespace cudf

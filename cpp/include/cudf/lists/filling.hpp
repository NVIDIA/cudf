/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

/**
 * @file
 * @brief Column APIs for individual list sequence
 */

namespace CUDF_EXPORT cudf {
namespace lists {
/**
 * @addtogroup lists_filling
 * @{
 */

/**
 * @brief Create a lists column in which each row contains a sequence of values specified by a tuple
 * of (`start`, `size`) parameters.
 *
 * Create a lists column in which each row is a sequence of values starting from a `start` value,
 * incrementing by one, and its cardinality is specified by a `size` value. The `start` and `size`
 * values used to generate each list is taken from the corresponding row of the input @p starts and
 * @p sizes columns.
 *
 *  - @p sizes must be a column of integer types.
 *  - All the input columns must not have nulls.
 *  - If any row of the @p sizes column contains negative value, the output is undefined.
 *
 * @code{.pseudo}
 * starts = [0, 1, 2, 3, 4]
 * sizes  = [0, 2, 2, 1, 3]
 *
 * output = [ [], [1, 2], [2, 3], [3], [4, 5, 6] ]
 * @endcode
 *
 * @throws cudf::logic_error if @p sizes column is not of integer types.
 * @throws cudf::logic_error if any input column has nulls.
 * @throws cudf::logic_error if @p starts and @p sizes columns do not have the same size.
 * @throws std::overflow_error if the output column would exceed the column size limit.
 *
 * @param starts First values in the result sequences.
 * @param sizes Numbers of values in the result sequences.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return The result column containing generated sequences.
 */
std::unique_ptr<column> sequences(
  column_view const& starts,
  column_view const& sizes,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Create a lists column in which each row contains a sequence of values specified by a tuple
 * of (`start`, `step`, `size`) parameters.
 *
 * Create a lists column in which each row is a sequence of values starting from a `start` value,
 * incrementing by a `step` value, and its cardinality is specified by a `size` value. The values
 * `start`, `step`, and `size` used to generate each list is taken from the corresponding row of the
 * input @p starts, @p steps, and @p sizes columns.
 *
 *  - @p sizes must be a column of integer types.
 *  - @p starts and @p steps columns must have the same type.
 *  - All the input columns must not have nulls.
 *  - If any row of the @p sizes column contains negative value, the output is undefined.
 *
 * @code{.pseudo}
 * starts = [0, 1, 2, 3, 4]
 * steps  = [2, 1, 1, 1, -3]
 * sizes  = [0, 2, 2, 1, 3]
 *
 * output = [ [], [1, 2], [2, 3], [3], [4, 1, -2] ]
 * @endcode
 *
 * @throws cudf::logic_error if @p sizes column is not of integer types.
 * @throws cudf::logic_error if any input column has nulls.
 * @throws cudf::logic_error if @p starts and @p steps columns have different types.
 * @throws cudf::logic_error if @p starts, @p steps, and @p sizes columns do not have the same size.
 * @throws std::overflow_error if the output column would exceed the column size limit.
 *
 * @param starts First values in the result sequences.
 * @param steps Increment values for the result sequences.
 * @param sizes Numbers of values in the result sequences.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return The result column containing generated sequences.
 */
std::unique_ptr<column> sequences(
  column_view const& starts,
  column_view const& steps,
  column_view const& sizes,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group
}  // namespace lists
}  // namespace CUDF_EXPORT cudf

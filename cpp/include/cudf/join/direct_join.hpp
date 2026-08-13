/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <memory>
#include <utility>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup column_join
 * @{
 * @file
 * @brief Direct join APIs for pre-hashed integer keys
 */

/**
 * @brief Returns the row indices that can be used to construct the result of performing an inner
 * join between two key columns whose values directly determine the matched row index
 *
 * The right keys are treated as a perfect hash of the right rows: a lookup table of `capacity`
 * entries maps each key value to its row index, and each left key probes that table directly. No
 * hashing or key comparison is performed. Left keys that do not occur in the right keys produce no
 * output pair.
 *
 * @note Behavior is undefined if any key value is not less than `capacity`, or if the right keys
 * contain duplicates.
 *
 * @throw cudf::data_type_error if the key columns are not of type UINT32
 * @throw std::invalid_argument if the key columns contain nulls
 * @throw std::invalid_argument if `capacity` is less than the number of right keys
 *
 * @param left_keys The left key column containing pre-hashed keys in `[0, capacity)`, from which
 * the keys are probed
 * @param right_keys The right key column containing distinct pre-hashed keys in `[0, capacity)`
 * @param capacity The number of entries in the lookup table
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned indices' device memory
 *
 * @return A pair of vectors [`left_indices`, `right_indices`] that can be used to construct the
 * result of performing an inner join between two tables with `left_keys` and `right_keys` as the
 * join keys
 */
[[nodiscard]] std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                        std::unique_ptr<rmm::device_uvector<size_type>>>
direct_inner_join(column_view const& left_keys,
                  column_view const& right_keys,
                  std::size_t capacity,
                  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf

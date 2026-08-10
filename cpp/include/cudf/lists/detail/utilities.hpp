/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace CUDF_EXPORT cudf {
namespace lists::detail {

/**
 * @brief Returns one normalized value from an INT32 or INT64 list offsets column.
 */
int64_t get_offset_value(column_view const& offsets,
                         size_type index,
                         rmm::cuda_stream_view stream);

/**
 * @brief Returns the wider of two valid list offset types.
 */
data_type promoted_offsets_type(data_type lhs, data_type rhs);

/**
 * @brief Converts generated list offsets to the narrowest type capable of indexing the child.
 */
std::unique_ptr<column> normalize_offsets(std::unique_ptr<column> offsets,
                                          size_type child_size,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr);

}  // namespace lists::detail
}  // namespace CUDF_EXPORT cudf

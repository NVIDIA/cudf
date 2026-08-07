/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>
#include <rmm/resource_ref.hpp>

#include <nanoarrow/nanoarrow.hpp>

namespace cudf {
namespace detail {

/**
 * @brief Utility to handle STRING, LARGE_STRINGS, and STRING_VIEW types
 *
 * @param schema Arrow schema includes the column type
 * @param input Column data, nulls, offset
 * @param mask Mask to apply to the output column
 * @param null_count Number of nulls in mask
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource to use for all device memory allocations
 */
std::unique_ptr<column> string_column_from_arrow_host(ArrowSchemaView const* schema,
                                                      ArrowArray const* input,
                                                      std::unique_ptr<rmm::device_buffer>&& mask,
                                                      size_type null_count,
                                                      rmm::cuda_stream_view stream,
                                                      rmm::device_async_resource_ref mr);

/**
 * @brief Create offsets column for list or strings column
 *
 *
 * @param schema Arrow schema includes the column type
 * @param input Column data, nulls, offset
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource to use for all device memory allocations
 * @return Column plus offset and size bounds for copying data column
 */
std::tuple<std::unique_ptr<column>, int64_t, int64_t> get_offsets_column(
  ArrowSchemaView const* schema,
  ArrowArray const* input,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Create the offsets column for a fixed-size-list array
 *
 * Arrow fixed-size-list arrays carry no offsets buffer; the offsets are implicit.
 * This generates `size` offsets of the form `{0, width, 2*width, ...}`.
 *
 * @param size Number of offsets to generate (normally num_rows + 1)
 * @param width Number of child elements per list row
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource to use for all device memory allocations
 * @return INT32 offsets column
 */
std::unique_ptr<column> make_fixed_size_list_offsets(size_type size,
                                                     size_type width,
                                                     rmm::cuda_stream_view stream,
                                                     rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf

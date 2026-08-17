/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/binary/binary_column_view.hpp>
#include <cudf/column/column.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf::binary::detail {

/**
 * @brief Copies rows `[start, end)` from a BINARY column into a new owning column.
 *
 * The output offsets are normalized to begin at zero and retain the input
 * offsets width.
 */
CUDF_EXPORT std::unique_ptr<column> copy_slice(
  binary_column_view const& input,
  size_type start,
  size_type end,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace cudf::binary::detail

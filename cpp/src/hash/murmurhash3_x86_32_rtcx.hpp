/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <rmm/resource_ref.hpp>

#include <cuda/stream>

#include <cstddef>
#include <cstdint>
#include <memory>

namespace cudf::detail::row::equality {
struct preprocessed_table;
}

namespace cudf::hashing::detail {

std::unique_ptr<column> murmurhash3_x86_32_rtcx_generic(
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& input,
  size_type num_rows,
  uint32_t seed,
  bool nullable,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

[[nodiscard]] CUDF_EXPORT bool murmurhash3_x86_32_rtcx_enabled(table_view const& input);

std::unique_ptr<column> murmurhash3_x86_32_rtcx(table_view const& input,
                                                uint32_t seed,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr);

[[nodiscard]] CUDF_EXPORT std::size_t murmurhash3_x86_32_rtcx_cache_size();

}  // namespace cudf::hashing::detail

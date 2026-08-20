/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/resource_ref.hpp>

#include <cstdint>
#include <memory>

namespace cudf {
class column;

namespace detail::row::equality {
struct preprocessed_table;
}

namespace hashing::detail {

std::unique_ptr<column> murmurhash3_x86_32(
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& input,
  size_type num_rows,
  uint32_t seed,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace hashing::detail
}  // namespace cudf

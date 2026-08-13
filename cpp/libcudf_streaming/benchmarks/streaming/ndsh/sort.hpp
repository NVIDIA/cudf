/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once
#include <cudf/types.hpp>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <memory>
#include <vector>

namespace rapidsmpf::ndsh {

/**
 * @brief Sort chunks in a channel
 *
 * @param ctx Streaming context
 * @param ch_in Input channel of `table_chunk`s
 * @param ch_out Output channel of sorted `table_chunk`s
 * @param keys Indices of key columns in the input channel
 * @param values Indices of value columns in the input channel
 * @param order Sort order for each column named in `keys`
 * @param null_order Null precedence for each column named in `keys`
 *
 * @return Coroutine representing the sort
 */
[[nodiscard]] rapidsmpf::streaming::Actor chunkwise_sort_by(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
  std::vector<cudf::size_type> keys,
  std::vector<cudf::size_type> values,
  std::vector<cudf::order> order,
  std::vector<cudf::null_order> null_order);
}  // namespace rapidsmpf::ndsh

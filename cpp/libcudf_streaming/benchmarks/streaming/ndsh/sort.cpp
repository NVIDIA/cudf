/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort.hpp"

#include <cudf/sorting.hpp>
#include <cudf/types.hpp>

#include <cudf_streaming/table_chunk.hpp>

#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <algorithm>
#include <memory>
#include <vector>

namespace rapidsmpf::ndsh {
using cudf_streaming::table_chunk;
using cudf_streaming::to_message;

rapidsmpf::streaming::Actor chunkwise_sort_by(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                              std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
                                              std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                              std::vector<cudf::size_type> keys,
                                              std::vector<cudf::size_type> values,
                                              std::vector<cudf::order> order,
                                              std::vector<cudf::null_order> null_order)
{
  streaming::ShutdownAtExit c{ch_in, ch_out};
  co_await ctx->executor()->schedule();
  auto make_table = [&](cudf_streaming::table_chunk& chunk) {
    if (std::ranges::equal(keys, values)) {
      return cudf::sort(
        chunk.table_view().select(keys), order, null_order, chunk.stream(), ctx->br()->device_mr());
    } else {
      return cudf::sort_by_key(chunk.table_view().select(values),
                               chunk.table_view().select(keys),
                               order,
                               null_order,
                               chunk.stream(),
                               ctx->br()->device_mr());
    }
  };
  while (!ch_out->is_shutdown()) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto chunk = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    co_await ch_out->send(
      to_message(msg.sequence_number(),
                 std::make_unique<cudf_streaming::table_chunk>(make_table(chunk), chunk.stream())));
  }
  co_await ch_out->drain(ctx->executor());
}
}  // namespace rapidsmpf::ndsh

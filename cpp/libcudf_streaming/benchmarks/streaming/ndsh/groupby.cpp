/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/types.hpp>

#include <cudf_streaming/table_chunk.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <iterator>
#include <memory>
#include <vector>

namespace rapidsmpf::ndsh {

streaming::Actor chunkwise_group_by(std::shared_ptr<streaming::Context> ctx,
                                    std::shared_ptr<streaming::Channel> ch_in,
                                    std::shared_ptr<streaming::Channel> ch_out,
                                    std::vector<cudf::size_type> keys,
                                    std::vector<groupby_request> requests,
                                    cudf::null_policy null_policy)
{
  streaming::ShutdownAtExit c{ch_in, ch_out};
  co_await ctx->executor()->schedule();
  while (!ch_out->is_shutdown()) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto chunk        = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    auto stream       = chunk.stream();
    auto table        = chunk.table_view();
    auto agg_requests = std::vector<cudf::groupby::aggregation_request>();
    agg_requests.reserve(requests.size());
    std::ranges::transform(requests, std::back_inserter(agg_requests), [&table](auto&& req) {
      std::vector<std::unique_ptr<cudf::groupby_aggregation>> reqs;
      for (auto&& x : req.requests) {
        reqs.push_back(x());
      }
      return cudf::groupby::aggregation_request{table.column(req.column_idx), std::move(reqs)};
    });
    auto grouper = cudf::groupby::groupby(table.select(keys), null_policy, cudf::sorted::NO);

    auto [keys, aggregated] = grouper.aggregate(agg_requests, stream, ctx->br()->device_mr());
    std::ignore             = std::move(chunk);
    auto result             = keys->release();
    for (auto&& a : aggregated) {
      std::ranges::move(a.results, std::back_inserter(result));
    }
    co_await ch_out->send(
      cudf_streaming::to_message(msg.sequence_number(),
                                 std::make_unique<cudf_streaming::table_chunk>(
                                   std::make_unique<cudf::table>(std::move(result)), stream)));
  }
  co_await ch_out->drain(ctx->executor());
}
}  // namespace rapidsmpf::ndsh

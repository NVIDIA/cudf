/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "concatenate.hpp"

#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <cudf_streaming/table_chunk.hpp>

#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

#include <memory>
#include <ranges>

namespace rapidsmpf::ndsh {

streaming::Actor concatenate(std::shared_ptr<streaming::Context> ctx,
                             std::shared_ptr<streaming::Channel> ch_in,
                             std::shared_ptr<streaming::Channel> ch_out,
                             ConcatOrder order)
{
  streaming::ShutdownAtExit c{ch_in, ch_out};
  CudaEvent event;
  std::vector<streaming::Message> messages;
  ctx->logger()->print("Concatenate");
  auto concat_stream = ctx->br()->stream_pool()->get_stream();
  while (!ch_out->is_shutdown()) {
    co_await ctx->executor()->schedule();
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    messages.push_back(std::move(msg));
  }
  if (messages.size() == 0) {
    co_await ch_out->send(
      cudf_streaming::to_message(0,
                                 std::make_unique<cudf_streaming::table_chunk>(
                                   std::make_unique<cudf::table>(), concat_stream)));
  } else if (messages.size() == 1) {
    co_await ch_out->send(std::move(messages[0]));
  } else {
    std::vector<cudf_streaming::table_chunk> chunks;
    std::vector<cudf::table_view> views;
    if (order == ConcatOrder::LINEARIZE) {
      std::ranges::sort(messages, std::less{}, [](auto&& msg) { return msg.sequence_number(); });
    }
    chunks.reserve(messages.size());
    views.reserve(messages.size());
    for (auto&& msg : messages) {
      auto chunk = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
      cuda_stream_join(concat_stream, chunk.stream(), &event);
      views.push_back(chunk.table_view());
      chunks.push_back(std::move(chunk));
    }
    auto result = std::make_unique<cudf_streaming::table_chunk>(
      cudf::concatenate(views, concat_stream, ctx->br()->device_mr()), concat_stream);
    cuda_stream_join(chunks | std::views::transform([](auto&& chunk) { return chunk.stream(); }),
                     std::ranges::single_view(concat_stream),
                     &event);
    chunks.clear();
    co_await ch_out->send(cudf_streaming::to_message(0, std::move(result)));
  }
  co_await ch_out->drain(ctx->executor());
}

}  // namespace rapidsmpf::ndsh

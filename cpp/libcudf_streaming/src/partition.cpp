/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/partitioning.hpp>

#include <cudf_streaming/partition.hpp>
#include <cudf_streaming/partition_utils.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/memory/spill.hpp>
#include <rapidsmpf/streaming/chunks/partition.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>

#include <memory>

namespace cudf_streaming::actor {

rapidsmpf::streaming::Actor partition_and_pack(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
  std::vector<cudf::size_type> columns_to_hash,
  int num_partitions,
  cudf::hash_id hash_function,
  std::uint32_t seed)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

  co_await ctx->executor()->schedule();
  while (true) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto table       = msg.release<table_chunk>();
    auto reservation = ctx->br()->reserve_device_memory_and_spill(table.make_available_cost(),
                                                                  rapidsmpf::AllowOverbooking::NO);
    auto tbl         = table.make_available(reservation);

    rapidsmpf::streaming::PartitionMapChunk partition_map{
      .data = cudf_streaming::partition_and_pack(tbl.table_view(),
                                                 columns_to_hash,
                                                 num_partitions,
                                                 hash_function,
                                                 seed,
                                                 tbl.stream(),
                                                 ctx->br().get())};

    co_await ch_out->send(to_message(
      msg.sequence_number(),
      std::make_unique<rapidsmpf::streaming::PartitionMapChunk>(std::move(partition_map))));
  }
  co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Actor unpack_and_concat(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                              std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
                                              std::shared_ptr<rapidsmpf::streaming::Channel> ch_out)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
  co_await ctx->executor()->schedule();
  while (true) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }

    // If receiving a partition map, we convert it to a vector and discard
    // partition IDs.
    std::uint64_t seq = msg.sequence_number();
    std::vector<rapidsmpf::PackedData> data;
    if (msg.holds<rapidsmpf::streaming::PartitionMapChunk>()) {
      auto partition_map = msg.release<rapidsmpf::streaming::PartitionMapChunk>();
      data               = rapidsmpf::to_vector(std::move(partition_map.data));
    } else {
      auto partition_vec = msg.release<rapidsmpf::streaming::PartitionVectorChunk>();
      data               = std::move(partition_vec.data);
    }
    // Get a stream for the concatenated table chunk.
    auto stream = ctx->br()->stream_pool()->get_stream();

    std::unique_ptr<cudf::table> ret = cudf_streaming::unpack_and_concat(
      rapidsmpf::unspill_partitions(
        std::move(data), ctx->br().get(), rapidsmpf::AllowOverbooking::NO),
      stream,
      ctx->br().get());
    co_await ch_out->send(to_message(seq, std::make_unique<table_chunk>(std::move(ret), stream)));
  }
  co_await ch_out->drain(ctx->executor());
}

}  // namespace cudf_streaming::actor

/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cudf_streaming/partition_utils.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/memory/spill.hpp>
#include <rapidsmpf/shuffler/chunk.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/coll/shuffler.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <algorithm>
#include <memory>
#include <vector>

namespace rapidsmpf::ndsh {
using cudf_streaming::partition_and_pack;
using cudf_streaming::partition_and_split;
using cudf_streaming::table_chunk;
using cudf_streaming::to_message;
using cudf_streaming::unpack_and_concat;

coro::task<streaming::Message> broadcast(std::shared_ptr<streaming::Context> ctx,
                                         std::shared_ptr<Communicator> comm,
                                         std::shared_ptr<streaming::Channel> ch_in,
                                         OpID tag,
                                         streaming::AllGather::Ordered ordered)
{
  streaming::ShutdownAtExit c{ch_in};
  co_await ctx->executor()->schedule();
  CudaEvent event;
  comm->logger()->print("Broadcast ", static_cast<int>(tag));
  if (comm->nranks() == 1) {
    std::vector<cudf_streaming::table_chunk> chunks;
    std::vector<cudf::table_view> views;
    auto gather_stream = ctx->br()->stream_pool()->get_stream();
    while (true) {
      auto msg = co_await ch_in->receive();
      if (msg.empty()) { break; }
      auto chunk = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
      cuda_stream_join(gather_stream, chunk.stream(), &event);
      views.push_back(chunk.table_view());
      chunks.push_back(std::move(chunk));
    }
    if (chunks.size() == 1) {
      co_return to_message(0, std::make_unique<cudf_streaming::table_chunk>(std::move(chunks[0])));
    } else {
      RAPIDSMPF_EXPECTS(chunks.size() > 0, "No chunks in broadcast");
      auto result = cudf::concatenate(views, gather_stream, ctx->br()->device_mr());
      // So that deallocation of the consitutent tables is stream-ordered wrt the
      // concatenation.
      cuda_stream_join(chunks | std::views::transform([](auto&& chunk) { return chunk.stream(); }),
                       std::ranges::single_view(gather_stream),
                       &event);
      co_return to_message(
        0, std::make_unique<cudf_streaming::table_chunk>(std::move(result), gather_stream));
    }
  } else {
    streaming::AllGather gatherer{ctx, comm, tag};
    while (true) {
      auto msg = co_await ch_in->receive();
      if (msg.empty()) { break; }
      // TODO: If this chunk is already in pack form, this is unnecessary.
      auto chunk       = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
      auto pack        = cudf::pack(chunk.table_view(), chunk.stream(), ctx->br()->device_mr());
      auto packed_data = PackedData(std::move(pack.metadata),
                                    ctx->br()->move(std::move(pack.gpu_data), chunk.stream()));
      gatherer.insert(msg.sequence_number(), {std::move(packed_data)});
    }
    gatherer.insert_finished();
    auto result = co_await gatherer.extract_all(ordered);
    if (result.size() == 1) {
      co_return to_message(0,
                           std::make_unique<cudf_streaming::table_chunk>(
                             std::make_unique<PackedData>(std::move(result[0]))));
    } else {
      auto stream = ctx->br()->stream_pool()->get_stream();
      co_return to_message(
        0,
        std::make_unique<cudf_streaming::table_chunk>(
          unpack_and_concat(rapidsmpf::unspill_partitions(
                              std::move(result), ctx->br().get(), AllowOverbooking::YES),
                            stream,
                            ctx->br().get()),
          stream));
    }
  }
}

streaming::Actor broadcast(std::shared_ptr<streaming::Context> ctx,
                           std::shared_ptr<Communicator> comm,
                           std::shared_ptr<streaming::Channel> ch_in,
                           std::shared_ptr<streaming::Channel> ch_out,
                           OpID tag,
                           streaming::AllGather::Ordered ordered)
{
  streaming::ShutdownAtExit c{ch_in, ch_out};
  co_await ctx->executor()->schedule();
  co_await ch_out->send(co_await broadcast(ctx, comm, ch_in, tag, ordered));
  co_await ch_out->drain(ctx->executor());
}

/**
 * @brief Join a table chunk against a build hash table returning a message of the result.
 *
 * @param ctx Streaming context
 * @param left_chunk Chunk to join. Used as the probe table in a filtered join.
 * @param right_chunk Chunk to join. Used as the build table in a filtered join.
 * @param left_carrier Columns from `left_chunk` to include in the output.
 * @param left_on Key column indices in `left_chunk`.
 * @param right_on Key column indices in `right_chunk`.
 * @param sequence Sequence number of the output
 * @param left_event Event recording the availability of `left_chunk`.
 *
 * @return Message of `table_chunk` containing the result of the semi join.
 */
streaming::Message semi_join_chunk(std::shared_ptr<streaming::Context> ctx,
                                   cudf_streaming::table_chunk const& left_chunk,
                                   cudf_streaming::table_chunk&& right_chunk,
                                   cudf::table_view left_carrier,
                                   std::vector<cudf::size_type> left_on,
                                   std::vector<cudf::size_type> right_on,
                                   std::uint64_t sequence,
                                   CudaEvent* left_event)
{
  auto chunk_stream = right_chunk.stream();

  left_event->stream_wait(chunk_stream);

  // At this point, both left_chunk and right_chunk are valid on
  // either stream. We'll do everything from here out on the
  // right_chunk.stream(), so that we don't introduce false dependencies
  // between the different chunks.

  auto joiner = cudf::filtered_join(
    right_chunk.table_view().select(right_on), cudf::null_equality::UNEQUAL, chunk_stream);

  auto match =
    joiner.semi_join(left_chunk.table_view().select(left_on), chunk_stream, ctx->br()->device_mr());

  ctx->logger()->debug("semi_join_chunk: left.num_rows()=", left_chunk.table_view().num_rows());
  ctx->logger()->debug("semi_join_chunk: match.size()=", match->size());

  cudf::column_view indices = cudf::device_span<cudf::size_type const>(*match);
  auto result_columns       = cudf::gather(left_carrier,
                                     indices,
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     chunk_stream,
                                     ctx->br()->device_mr())
                          ->release();

  auto result_table = std::make_unique<cudf::table>(std::move(result_columns));
  // Deallocation of the join indices will happen on chunk_stream, so add stream dep
  cuda_stream_join(left_chunk.stream(), chunk_stream);

  return to_message(
    sequence, std::make_unique<cudf_streaming::table_chunk>(std::move(result_table), chunk_stream));
}

/**
 * @brief Join a table chunk against a build hash table returning a message of the result.
 *
 * @param ctx Streaming context.
 * @param right_chunk Chunk to join. Must be on device e.g. use make_available() on the
 * chunk.
 * @param sequence Sequence number of the output
 * @param joiner hash_join object, representing the build table.
 * @param build_carrier Columns from the build-side table to be included in the output.
 * @param right_on Key column indiecs in `right_chunk`.
 * @param build_stream Stream the `joiner` will be deallocated on.
 * @param build_event Event recording the creation of the `joiner`.
 * @param tmp_event Preallocated event used for internal stream ordering.
 *
 * @return Message of `table_chunk` containing the result of the inner join.
 */
streaming::Message inner_join_chunk(std::shared_ptr<streaming::Context> ctx,
                                    cudf_streaming::table_chunk&& right_chunk,
                                    std::uint64_t sequence,
                                    cudf::hash_join& joiner,
                                    cudf::table_view build_carrier,
                                    std::vector<cudf::size_type> right_on,
                                    rmm::cuda_stream_view build_stream,
                                    CudaEvent* build_event,
                                    CudaEvent* tmp_event

)
{
  auto chunk_stream = right_chunk.stream();
  build_event->stream_wait(chunk_stream);
  auto probe_table = right_chunk.table_view();
  auto probe_keys  = probe_table.select(right_on);
  auto [probe_match, build_match] =
    joiner.inner_join(probe_keys, std::nullopt, chunk_stream, ctx->br()->device_mr());

  cudf::column_view build_indices = cudf::device_span<cudf::size_type const>(*build_match);
  cudf::column_view probe_indices = cudf::device_span<cudf::size_type const>(*probe_match);
  // build_carrier is valid on build_stream, but chunk_stream is
  // waiting for build_stream work to be done, so running this on
  // chunk_stream is fine.
  auto result_columns = cudf::gather(build_carrier,
                                     build_indices,
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     chunk_stream,
                                     ctx->br()->device_mr())
                          ->release();
  // drop key columns from probe table.
  std::vector<cudf::size_type> to_keep;
  std::ranges::copy_if(std::ranges::iota_view(0, probe_table.num_columns()),
                       std::back_inserter(to_keep),
                       [&](auto i) { return std::ranges::find(right_on, i) == right_on.end(); });
  std::ranges::move(cudf::gather(probe_table.select(to_keep),
                                 probe_indices,
                                 cudf::out_of_bounds_policy::DONT_CHECK,
                                 chunk_stream,
                                 ctx->br()->device_mr())
                      ->release(),
                    std::back_inserter(result_columns));
  // Deallocation of the join indices will happen on build_stream, so add stream dep
  // This also ensure deallocation of the hash_join object waits for completion.
  cuda_stream_join(build_stream, chunk_stream, tmp_event);
  return to_message(sequence,
                    std::make_unique<cudf_streaming::table_chunk>(
                      std::make_unique<cudf::table>(std::move(result_columns)), chunk_stream));
}

streaming::Actor inner_join_broadcast(
  std::shared_ptr<streaming::Context> ctx,
  std::shared_ptr<Communicator> comm,
  // We will always choose left as build table and do "broadcast" joins
  std::shared_ptr<streaming::Channel> left,
  std::shared_ptr<streaming::Channel> right,
  std::shared_ptr<streaming::Channel> ch_out,
  std::vector<cudf::size_type> left_on,
  std::vector<cudf::size_type> right_on,
  OpID tag,
  KeepKeys keep_keys)
{
  streaming::ShutdownAtExit c{left, right, ch_out};
  co_await ctx->executor()->schedule();
  comm->logger()->print("Inner broadcast join ", static_cast<int>(tag));
  auto build_table =
    co_await ((co_await broadcast(ctx, comm, left, tag, streaming::AllGather::Ordered::NO))
                .release<cudf_streaming::table_chunk>()
                .make_available(ctx));
  comm->logger()->print("Build table has ", build_table.table_view().num_rows(), " rows");

  auto joiner = cudf::hash_join(
    build_table.table_view().select(left_on), cudf::null_equality::UNEQUAL, build_table.stream());
  CudaEvent build_event;
  build_event.record(build_table.stream());
  CudaEvent tmp_event;
  cudf::table_view build_carrier;
  if (keep_keys == KeepKeys::YES) {
    build_carrier = build_table.table_view();
  } else {
    std::vector<cudf::size_type> to_keep;
    std::ranges::copy_if(std::ranges::iota_view(0, build_table.table_view().num_columns()),
                         std::back_inserter(to_keep),
                         [&](auto i) { return std::ranges::find(left_on, i) == left_on.end(); });
    build_carrier = build_table.table_view().select(to_keep);
  }
  while (!ch_out->is_shutdown()) {
    auto right_msg = co_await right->receive();
    if (right_msg.empty()) { break; }
    co_await ch_out->send(inner_join_chunk(ctx,
                                           right_msg.release<cudf_streaming::table_chunk>(),
                                           right_msg.sequence_number(),
                                           joiner,
                                           build_carrier,
                                           right_on,
                                           build_table.stream(),
                                           &build_event,
                                           &tmp_event));
  }

  co_await ch_out->drain(ctx->executor());
}

streaming::Actor inner_join_shuffle(std::shared_ptr<streaming::Context> ctx,
                                    std::shared_ptr<Communicator> comm,
                                    std::shared_ptr<streaming::Channel> left,
                                    std::shared_ptr<streaming::Channel> right,
                                    std::shared_ptr<streaming::Channel> ch_out,
                                    std::vector<cudf::size_type> left_on,
                                    std::vector<cudf::size_type> right_on,
                                    KeepKeys keep_keys)
{
  streaming::ShutdownAtExit c{left, right, ch_out};
  comm->logger()->print("Inner shuffle join");
  co_await ctx->executor()->schedule();
  CudaEvent build_event;
  CudaEvent tmp_event;
  while (!ch_out->is_shutdown()) {
    // Requirement: two shuffles kick out partitions in the same order
    auto left_msg  = co_await left->receive();
    auto right_msg = co_await right->receive();
    if (left_msg.empty()) {
      RAPIDSMPF_EXPECTS(right_msg.empty(), "Left does not have same number of partitions as right");
      break;
    }
    RAPIDSMPF_EXPECTS(left_msg.sequence_number() == right_msg.sequence_number(),
                      "Mismatching sequence numbers");
    // TODO: currently always using left as build table.
    auto build_chunk = co_await left_msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    auto build_stream = build_chunk.stream();
    auto joiner       = cudf::hash_join(
      build_chunk.table_view().select(left_on), cudf::null_equality::UNEQUAL, build_stream);
    build_event.record(build_stream);
    cudf::table_view build_carrier;
    if (keep_keys == KeepKeys::YES) {
      build_carrier = build_chunk.table_view();
    } else {
      std::vector<cudf::size_type> to_keep;
      std::ranges::copy_if(std::ranges::iota_view(0, build_chunk.table_view().num_columns()),
                           std::back_inserter(to_keep),
                           [&](auto i) { return std::ranges::find(left_on, i) == left_on.end(); });
      build_carrier = build_chunk.table_view().select(to_keep);
    }
    co_await ch_out->send(inner_join_chunk(ctx,
                                           right_msg.release<cudf_streaming::table_chunk>(),
                                           left_msg.sequence_number(),
                                           joiner,
                                           build_carrier,
                                           right_on,
                                           build_stream,
                                           &build_event,
                                           &tmp_event));
  }
  co_await ch_out->drain(ctx->executor());
}

streaming::Actor left_semi_join_broadcast_left(std::shared_ptr<streaming::Context> ctx,
                                               std::shared_ptr<Communicator> comm,
                                               std::shared_ptr<streaming::Channel> left,
                                               std::shared_ptr<streaming::Channel> right,
                                               std::shared_ptr<streaming::Channel> ch_out,
                                               std::vector<cudf::size_type> left_on,
                                               std::vector<cudf::size_type> right_on,
                                               OpID tag,
                                               KeepKeys keep_keys)
{
  streaming::ShutdownAtExit c{left, right, ch_out};
  co_await ctx->executor()->schedule();
  comm->logger()->print("Left semi broadcast join ", static_cast<int>(tag));
  auto left_table = co_await (co_await broadcast(ctx, comm, left, tag))
                      .release<cudf_streaming::table_chunk>()
                      .make_available(ctx);
  comm->logger()->print("Left (probe) table has ", left_table.table_view().num_rows(), " rows");
  CudaEvent left_event;
  left_event.record(left_table.stream());

  cudf::table_view left_carrier;
  if (keep_keys == KeepKeys::YES) {
    left_carrier = left_table.table_view();
  } else {
    std::vector<cudf::size_type> to_keep;
    std::ranges::copy_if(std::ranges::iota_view(0, left_table.table_view().num_columns()),
                         std::back_inserter(to_keep),
                         [&](auto i) { return std::ranges::find(left_on, i) == left_on.end(); });
    left_carrier = left_table.table_view().select(to_keep);
  }

  while (!ch_out->is_shutdown()) {
    auto right_msg = co_await right->receive();
    if (right_msg.empty()) { break; }
    // The ``right`` table has been hash-partitioned (via a shuffle) on
    // the join key. Thanks to the hash-partitioning, we don't need to worry
    // about deduplicating matches across partitions. Anything that matches
    // in the semi-join belongs in the output.
    auto right_chunk =
      co_await right_msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    co_await ch_out->send(semi_join_chunk(ctx,
                                          left_table,
                                          std::move(right_chunk),
                                          left_carrier,
                                          left_on,
                                          right_on,
                                          right_msg.sequence_number(),
                                          &left_event));
  }

  co_await ch_out->drain(ctx->executor());
}

streaming::Actor left_semi_join_shuffle(std::shared_ptr<streaming::Context> ctx,
                                        std::shared_ptr<Communicator> comm,
                                        std::shared_ptr<streaming::Channel> left,
                                        std::shared_ptr<streaming::Channel> right,
                                        std::shared_ptr<streaming::Channel> ch_out,
                                        std::vector<cudf::size_type> left_on,
                                        std::vector<cudf::size_type> right_on,
                                        KeepKeys keep_keys)
{
  streaming::ShutdownAtExit c{left, right, ch_out};
  comm->logger()->print("Shuffle left semi join");

  co_await ctx->executor()->schedule();
  CudaEvent left_event;

  while (!ch_out->is_shutdown()) {
    // Requirement: two shuffles kick out partitions in the same order
    auto left_msg  = co_await left->receive();
    auto right_msg = co_await right->receive();

    if (left_msg.empty()) {
      RAPIDSMPF_EXPECTS(right_msg.empty(), "Left does not have same number of partitions as right");
      break;
    }
    RAPIDSMPF_EXPECTS(left_msg.sequence_number() == right_msg.sequence_number(),
                      "Mismatching sequence numbers");

    auto left_chunk = co_await left_msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    auto right_chunk =
      co_await right_msg.release<cudf_streaming::table_chunk>().make_available(ctx);

    left_event.record(left_chunk.stream());

    cudf::table_view left_carrier;
    if (keep_keys == KeepKeys::YES) {
      left_carrier = left_chunk.table_view();
    } else {
      std::vector<cudf::size_type> to_keep;
      std::ranges::copy_if(std::ranges::iota_view(0, left_chunk.table_view().num_columns()),
                           std::back_inserter(to_keep),
                           [&](auto i) { return std::ranges::find(left_on, i) == left_on.end(); });
      left_carrier = left_chunk.table_view().select(to_keep);
    }

    co_await ch_out->send(semi_join_chunk(ctx,
                                          left_chunk,
                                          std::move(right_chunk),
                                          left_carrier,
                                          left_on,
                                          right_on,
                                          left_msg.sequence_number(),
                                          &left_event));
  }
}

streaming::Actor shuffle(std::shared_ptr<streaming::Context> ctx,
                         std::shared_ptr<Communicator> comm,
                         std::shared_ptr<streaming::Channel> ch_in,
                         std::shared_ptr<streaming::Channel> ch_out,
                         std::vector<cudf::size_type> keys,
                         std::uint32_t num_partitions,
                         OpID tag)
{
  streaming::ShutdownAtExit c{ch_in, ch_out};
  co_await ctx->executor()->schedule();
  comm->logger()->print("Shuffle ", static_cast<int>(tag));
  streaming::ShufflerAsync shuffler(ctx, comm, tag, num_partitions);
  while (true) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) {
      comm->logger()->debug("Shuffle: no more input");
      break;
    }
    auto chunk  = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    auto packed = partition_and_pack(chunk.table_view(),
                                     keys,
                                     static_cast<int>(num_partitions),
                                     cudf::hash_id::HASH_MURMUR3,
                                     0,
                                     chunk.stream(),
                                     ctx->br().get());
    shuffler.insert(std::move(packed));
  }
  co_await shuffler.insert_finished();
  for (auto pid : shuffler.local_partitions()) {
    auto packed_data = shuffler.extract(pid);
    auto stream      = ctx->br()->stream_pool()->get_stream();
    co_await ch_out->send(to_message(
      pid,
      std::make_unique<cudf_streaming::table_chunk>(
        unpack_and_concat(rapidsmpf::unspill_partitions(
                            std::move(packed_data), ctx->br().get(), AllowOverbooking::YES),
                          stream,
                          ctx->br().get()),
        stream)));
  }
  co_await ch_out->drain(ctx->executor());
}

}  // namespace rapidsmpf::ndsh

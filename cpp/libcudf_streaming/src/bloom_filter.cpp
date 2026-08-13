/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/stream_compaction.hpp>

#include <cudf_streaming/bloom_filter.hpp>
#include <cudf_streaming/detail/device_bloom_filter.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <cuda_runtime_api.h>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/streaming/coll/allreduce.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

namespace cudf_streaming {

std::size_t bloom_filter::aligned_size(std::size_t size) noexcept
{
  return detail::device_bloom_filter::aligned_size(size);
}

bloom_filter::bloom_filter(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                           std::shared_ptr<rapidsmpf::Communicator> comm,
                           std::uint64_t seed,
                           std::size_t filter_size)
  : ctx_{std::move(ctx)}, comm_{std::move(comm)}, seed_{seed}, filter_size_{filter_size}
{
  RAPIDSMPF_EXPECTS(filter_size_ > 0, "Bloom filter storage size must be positive");
  RAPIDSMPF_EXPECTS(filter_size_ == aligned_size(filter_size_),
                    "Bloom filter storage size must be a multiple of the filter block size");
  RAPIDSMPF_EXPECTS(filter_size_ <= detail::device_bloom_filter::max_size(),
                    "Bloom filter storage exceeds the maximum size supported by its policy");
}

rapidsmpf::streaming::Actor bloom_filter::build(
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
  rapidsmpf::OpID tag)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
  co_await ctx_->executor()->schedule();
  co_await ch_in->shutdown_metadata();
  co_await ch_out->shutdown_metadata();
  auto const& br     = ctx_->br();
  auto mr            = br->device_mr();
  auto filter_stream = br->stream_pool()->get_stream();
  rapidsmpf::CudaEvent event;
  auto storage =
    cudf_streaming::detail::device_bloom_filter::storage(filter_size_, filter_stream, mr);
  RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(storage->data(), 0, storage->size(), filter_stream));
  auto filter = cudf_streaming::detail::device_bloom_filter(filter_size_, seed_, storage->data());
  rapidsmpf::CudaEvent build_event;
  build_event.record(filter_stream);
  while (!ch_out->is_shutdown()) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto chunk = msg.release<table_chunk>();
    chunk      = co_await chunk.make_available(
      ctx_,
      -rapidsmpf::safe_cast<std::int64_t>(chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE)));

    // Reservation for the hash values in add.
    auto res = co_await ctx_->memory(rapidsmpf::MemoryType::DEVICE)
                 ->reserve_or_wait(rapidsmpf::safe_cast<std::size_t>(chunk.table_view().num_rows())
                                     // TODO: no magic numbers: the hashing algorithm in
                                     // `add` below returns an int64 column.
                                     * sizeof(std::int64_t),
                                   0);
    // Filter is allocated on `filter_stream`, but we run the additions on the chunk's
    // stream. The addition modifies global memory but we can safely launch two
    // kernels doing that concurrently because the updates are atomic.
    build_event.stream_wait(chunk.stream());
    filter.add(chunk.table_view(), chunk.stream(), mr);
    rapidsmpf::cuda_stream_join(filter_stream, chunk.stream(), &event);
  }
  if (comm_->nranks() > 1) {
    auto reducer = rapidsmpf::streaming::AllReduce(
      ctx_,
      comm_,
      br->move(std::move(storage), filter_stream),
      br->move(
        cudf_streaming::detail::device_bloom_filter::storage(filter_size_, filter_stream, mr),
        filter_stream),
      tag,
      [filter_size = filter_size_, seed = seed_](rapidsmpf::Buffer const* left,
                                                 rapidsmpf::Buffer* right) {
        right->write_access([&](std::byte* out_bytes, rmm::cuda_stream_view stream) {
          auto const in =
            cudf_streaming::detail::device_bloom_filter::view(filter_size, seed, left->data());
          cudf_streaming::detail::device_bloom_filter(filter_size, seed, out_bytes)
            .merge(in, stream);
        });
      });
    auto result   = co_await reducer.extract();
    auto [res, _] = br->reserve(rapidsmpf::MemoryType::DEVICE, 0, rapidsmpf::AllowOverbooking::YES);
    storage       = br->move_to_device_buffer(std::move(result.second), res);
  }
  co_await ch_out->send(rapidsmpf::streaming::Message{0, std::move(storage), {}, {}});
  co_await ch_out->drain(ctx_->executor());
}

rapidsmpf::streaming::Actor bloom_filter::apply(
  std::shared_ptr<rapidsmpf::streaming::Channel> bloom_filter,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
  std::vector<cudf::size_type> keys)
{
  rapidsmpf::streaming::ShutdownAtExit c{bloom_filter, ch_in, ch_out};
  co_await ctx_->executor()->schedule();
  auto storage = (co_await bloom_filter->receive()).release<rmm::device_buffer>();
  RAPIDSMPF_EXPECTS((co_await bloom_filter->receive()).empty(),
                    "Bloom filter channel contained more than one message");
  auto stream = storage.stream();
  rapidsmpf::CudaEvent event;
  auto filter = cudf_streaming::detail::device_bloom_filter(filter_size_, seed_, storage.data());
  auto meta   = co_await ch_in->receive_metadata();
  if (!meta.empty()) { co_await ch_out->send_metadata(std::move(meta)); }
  while (!ch_out->is_shutdown()) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto chunk = msg.release<table_chunk>();
    chunk      = co_await chunk.make_available(
      ctx_,
      -rapidsmpf::safe_cast<std::int64_t>(chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE)));
    auto chunk_stream = chunk.stream();
    rapidsmpf::cuda_stream_join(chunk_stream, stream, &event);
    // Reservation for the mask construction and guess at output size.
    auto res = co_await ctx_->memory(rapidsmpf::MemoryType::DEVICE)
                 ->reserve_or_wait(rapidsmpf::safe_cast<std::size_t>(chunk.table_view().num_rows())
                                       // TODO: no magic numbers: the hashing algorithm in
                                       // `contains` below returns an int64 column.
                                       * (1 + sizeof(std::int64_t))
                                     // Guess at how selective the filter is.
                                     + chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE) / 2,
                                   0);
    auto mask =
      filter.contains(chunk.table_view().select(keys), chunk_stream, ctx_->br()->device_mr());
    rapidsmpf::cuda_stream_join(stream, chunk_stream, &event);
    RAPIDSMPF_EXPECTS(mask.size() == static_cast<std::size_t>(chunk.table_view().num_rows()),
                      "Invalid mask size");
    auto mask_view = cudf::column_view{cudf::data_type{cudf::type_id::BOOL8},
                                       rapidsmpf::safe_cast<cudf::size_type>(mask.size()),
                                       mask.data(),
                                       {},
                                       0};
    auto result    = cudf::apply_boolean_mask(
      chunk.table_view(), mask_view, chunk_stream, ctx_->br()->device_mr());
    std::ignore = std::move(chunk);
    std::ignore = std::move(res);
    co_await ch_out->send(
      to_message(msg.sequence_number(),
                 std::make_unique<cudf_streaming::table_chunk>(std::move(result), chunk_stream)));
  }
  co_await ch_out->drain(ctx_->executor());
}
}  // namespace cudf_streaming

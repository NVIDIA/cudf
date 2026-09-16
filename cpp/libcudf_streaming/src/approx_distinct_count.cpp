/**
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/reduction/approx_distinct_count.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>

#include <cudf_streaming/approx_distinct_count.hpp>
#include <cudf_streaming/detail/approx_distinct_count.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <cuda/stream>
#include <cuda_runtime_api.h>

#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/error.hpp>
#include <rapidsmpf/memory/buffer_resource.hpp>
#include <rapidsmpf/memory/memory_type.hpp>
#include <rapidsmpf/streaming/coll/allreduce.hpp>
#include <rapidsmpf/streaming/core/lineariser.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace cudf_streaming {
namespace {

constexpr auto null_handling = cudf::null_policy::INCLUDE;
constexpr auto nan_handling  = cudf::nan_policy::NAN_IS_VALID;

}  // namespace

rapidsmpf::streaming::Message to_message(std::uint64_t sequence_number,
                                         std::unique_ptr<cardinality_estimate> estimate)
{
  return {sequence_number, std::move(estimate), {}, {}};
}

cardinality_estimator::cardinality_estimator(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                             std::shared_ptr<rapidsmpf::Communicator> comm,
                                             rapidsmpf::OpID tag,
                                             std::int32_t precision)
  : ctx_{std::move(ctx)}, comm_{std::move(comm)}, tag_{tag}, precision_{precision}
{
  RAPIDSMPF_EXPECTS(ctx_ != nullptr, "Cardinality estimator context must not be null");
  RAPIDSMPF_EXPECTS(comm_ != nullptr, "Cardinality estimator communicator must not be null");
  RAPIDSMPF_EXPECTS(precision_ >= 4 && precision_ <= 18,
                    "Cardinality estimator precision must be in range [4, 18]",
                    std::invalid_argument);
}

rapidsmpf::streaming::Actor cardinality_estimator::estimate(
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_sampled,
  std::vector<cudf::size_type> column_indices)
{
  RAPIDSMPF_EXPECTS(ch_in != nullptr, "Input channel must not be null");
  RAPIDSMPF_EXPECTS(ch_out != nullptr, "Estimate output channel must not be null");
  std::vector<std::shared_ptr<rapidsmpf::streaming::Channel>> shutdown_channels;
  shutdown_channels.push_back(ch_in);
  if (ch_sampled != nullptr) { shutdown_channels.push_back(ch_sampled); }
  shutdown_channels.push_back(ch_out);
  rapidsmpf::streaming::ShutdownAtExit shutdown{std::move(shutdown_channels)};

  co_await ctx_->executor()->schedule();
  co_await ch_in->shutdown_metadata();
  if (ch_sampled != nullptr) { co_await ch_sampled->shutdown_metadata(); }
  co_await ch_out->shutdown_metadata();

  auto const& br              = ctx_->br();
  auto const sketch_stream    = br->stream_pool()->get_stream();
  auto const sketch_bytes     = cudf::approx_distinct_count::sketch_bytes(precision_);
  auto const row_count_offset = sketch_bytes;
  auto const storage_bytes    = sketch_bytes + sizeof(std::uint64_t);
  auto reservation =
    co_await ctx_->memory(rapidsmpf::MemoryType::DEVICE)->reserve_or_wait(storage_bytes, 0);
  auto buf = rmm::device_buffer(
    storage_bytes, cudf::approx_distinct_count::sketch_alignment(), sketch_stream, br->device_mr());
  RAPIDSMPF_CUDA_TRY(cudaMemsetAsync(buf.data(), 0, storage_bytes, sketch_stream.get()));
  reservation.clear();
  rapidsmpf::CudaEvent init_event;
  rapidsmpf::CudaEvent add_event;
  init_event.record(sketch_stream);
  std::uint64_t rows_seen{};
  while (!ch_out->is_shutdown()) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }

    auto const sequence_number = msg.sequence_number();
    auto chunk                 = msg.release<table_chunk>();
    rows_seen += rapidsmpf::safe_cast<std::uint64_t>(chunk.shape().first);
    chunk = co_await chunk.make_available(
      ctx_,
      -rapidsmpf::safe_cast<std::int64_t>(chunk.data_alloc_size(rapidsmpf::MemoryType::DEVICE)));
    init_event.stream_wait(chunk.stream());
    // Reserve a conservative hash-sized scratch allowance for the libcudf operation.
    reservation =
      co_await ctx_->memory(rapidsmpf::MemoryType::DEVICE)
        ->reserve_or_wait(
          rapidsmpf::safe_cast<std::size_t>(chunk.table_view().num_rows()) * sizeof(std::uint64_t),
          0);
    auto sketch =
      cudf::approx_distinct_count({reinterpret_cast<cuda::std::byte*>(buf.data()), sketch_bytes},
                                  precision_,
                                  null_handling,
                                  nan_handling);
    auto const table =
      column_indices.empty() ? chunk.table_view() : chunk.table_view().select(column_indices);
    sketch.add(table, chunk.stream());
    rapidsmpf::cuda_stream_join(sketch_stream, chunk.stream(), &add_event);
    reservation.clear();
    if (ch_sampled != nullptr) {
      co_await ch_sampled->send(
        to_message(sequence_number, std::make_unique<table_chunk>(std::move(chunk))));
    }
  }

  detail::set_value(
    reinterpret_cast<std::uint64_t*>(reinterpret_cast<std::byte*>(buf.data()) + row_count_offset),
    rows_seen,
    sketch_stream);

  auto storage = br->move(std::make_unique<rmm::device_buffer>(std::move(buf)), sketch_stream);
  if (comm_->nranks() > 1) {
    reservation =
      co_await ctx_->memory(rapidsmpf::MemoryType::DEVICE)->reserve_or_wait(storage_bytes, 0);
    auto reducer = rapidsmpf::streaming::AllReduce(
      ctx_,
      comm_,
      std::move(storage),
      br->make_buffer(sketch_stream, std::move(reservation)),
      tag_,
      [precision = precision_, sketch_bytes, row_count_offset](rapidsmpf::Buffer const* left,
                                                               rapidsmpf::Buffer* right) {
        right->write_access([&](std::byte* out, cuda::stream_ref stream) {
          auto sketch =
            cudf::approx_distinct_count({reinterpret_cast<cuda::std::byte*>(out), sketch_bytes},
                                        precision,
                                        null_handling,
                                        nan_handling);
          sketch.merge({reinterpret_cast<cuda::std::byte const*>(left->data()), sketch_bytes},
                       stream);
          detail::add_values(
            reinterpret_cast<std::uint64_t const*>(left->data() + row_count_offset),
            reinterpret_cast<std::uint64_t*>(out + row_count_offset),
            stream);
        });
      });
    auto result = co_await reducer.extract();
    storage     = std::move(result.second);
  }

  auto const [distinct_count, row_count] =
    storage->write_access([&](std::byte* data, cuda::stream_ref stream) {
      auto sketch =
        cudf::approx_distinct_count({reinterpret_cast<cuda::std::byte*>(data), sketch_bytes},
                                    precision_,
                                    null_handling,
                                    nan_handling);
      auto const distinct_count = sketch.estimate(stream);
      auto tmp                  = cudf::detail::make_host_vector<std::uint64_t>(1, stream);
      cudf::detail::cuda_memcpy(
        cudf::host_span<std::uint64_t>(tmp),
        cudf::device_span{reinterpret_cast<std::uint64_t const*>(data + row_count_offset), 1},
        stream);
      return std::pair{distinct_count, tmp[0]};
    });

  co_await ch_out->send(
    to_message(0,
               std::make_unique<cardinality_estimate>(cardinality_estimate{
                 rapidsmpf::safe_cast<std::size_t>(row_count), distinct_count})));
  if (ch_sampled != nullptr) { co_await ch_sampled->drain(ctx_->executor()); }
  co_await ch_out->drain(ctx_->executor());
}

}  // namespace cudf_streaming

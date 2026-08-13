/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "concatenate.hpp"
#include "groupby.hpp"
#include "join.hpp"
#include "parquet_writer.hpp"
#include "sort.hpp"
#include "utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/context.hpp>
#include <cudf/copying.hpp>
#include <cudf/hashing.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <cudf_streaming/bloom_filter.hpp>
#include <cudf_streaming/parquet.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <coro/when_all.hpp>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/rmm_resource_adaptor.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/streaming/core/coro_utils.hpp>
#include <rapidsmpf/streaming/core/fanout.hpp>
#include <rapidsmpf/streaming/core/message.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>

using rapidsmpf::safe_cast;

namespace {

rapidsmpf::streaming::Actor read_lineitem(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                          std::shared_ptr<rapidsmpf::Communicator> comm,
                                          std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                          std::size_t num_producers,
                                          cudf::size_type num_rows_per_chunk,
                                          std::string const input_directory,
                                          std::vector<std::string> columns,
                                          std::shared_ptr<coro::latch> latch = nullptr)
{
  auto files = rapidsmpf::ndsh::detail::list_parquet_files(
    rapidsmpf::ndsh::detail::get_table_path(input_directory, "lineitem"));
  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                   .column_names(columns)
                   .build();
  if (latch != nullptr) { co_await *latch; }
  co_return co_await cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk);
}

rapidsmpf::streaming::Actor read_nation(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                        std::shared_ptr<rapidsmpf::Communicator> comm,
                                        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                        std::size_t num_producers,
                                        cudf::size_type num_rows_per_chunk,
                                        std::string const& input_directory)
{
  auto files = rapidsmpf::ndsh::detail::list_parquet_files(
    rapidsmpf::ndsh::detail::get_table_path(input_directory, "nation"));
  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                   .column_names({"n_nationkey"})
                   .build();
  // filter: "n_name" == "SAUDI ARABIA"
  auto filter_expr = [&]() -> std::unique_ptr<cudf_streaming::filter> {
    auto stream         = ctx->br()->stream_pool()->get_stream();
    auto owner          = new std::vector<std::any>;
    constexpr auto name = "SAUDI ARABIA";
    owner->push_back(std::make_shared<cudf::string_scalar>(
      name, /* is_valid = */ true, stream, ctx->br()->device_mr()));
    owner->push_back(std::make_shared<cudf::ast::literal>(
      *std::any_cast<std::shared_ptr<cudf::string_scalar>>(owner->at(0))));
    owner->push_back(std::make_shared<cudf::ast::column_name_reference>("n_name"));
    owner->push_back(std::make_shared<cudf::ast::operation>(
      cudf::ast::ast_operator::EQUAL,
      *std::any_cast<std::shared_ptr<cudf::ast::column_name_reference>>(owner->at(2)),
      *std::any_cast<std::shared_ptr<cudf::ast::literal>>(owner->at(1))));
    return std::make_unique<cudf_streaming::filter>(
      stream,
      *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->back()),
      rapidsmpf::OwningWrapper(static_cast<void*>(owner),
                               [](void* p) { delete static_cast<std::vector<std::any>*>(p); }));
  }();
  return cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk, std::move(filter_expr));
}

rapidsmpf::streaming::Actor read_orders(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                        std::shared_ptr<rapidsmpf::Communicator> comm,
                                        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                        std::size_t num_producers,
                                        cudf::size_type num_rows_per_chunk,
                                        std::string const& input_directory)
{
  auto files = rapidsmpf::ndsh::detail::list_parquet_files(
    rapidsmpf::ndsh::detail::get_table_path(input_directory, "orders"));
  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                   .column_names({"o_orderkey"})
                   .build();
  // filter: "o_orderstatus" == "F"
  auto filter_expr = [&]() -> std::unique_ptr<cudf_streaming::filter> {
    auto stream           = ctx->br()->stream_pool()->get_stream();
    auto owner            = new std::vector<std::any>;
    constexpr auto status = "F";
    owner->push_back(std::make_shared<cudf::string_scalar>(
      status, /* is_valid = */ true, stream, ctx->br()->device_mr()));
    owner->push_back(std::make_shared<cudf::ast::literal>(
      *std::any_cast<std::shared_ptr<cudf::string_scalar>>(owner->at(0))));
    owner->push_back(std::make_shared<cudf::ast::column_name_reference>("o_orderstatus"));
    owner->push_back(std::make_shared<cudf::ast::operation>(
      cudf::ast::ast_operator::EQUAL,
      *std::any_cast<std::shared_ptr<cudf::ast::column_name_reference>>(owner->at(2)),
      *std::any_cast<std::shared_ptr<cudf::ast::literal>>(owner->at(1))));
    return std::make_unique<cudf_streaming::filter>(
      stream,
      *std::any_cast<std::shared_ptr<cudf::ast::operation>>(owner->back()),
      rapidsmpf::OwningWrapper(static_cast<void*>(owner),
                               [](void* p) { delete static_cast<std::vector<std::any>*>(p); }));
  }();
  return cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk, std::move(filter_expr));
}

rapidsmpf::streaming::Actor read_orders_with_bloom_filter(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::Communicator> comm,
  std::shared_ptr<rapidsmpf::streaming::Channel> bloom_filter_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
  std::vector<cudf::size_type> filter_keys,
  cudf_streaming::bloom_filter bloom_filter,
  std::size_t num_producers,
  cudf::size_type num_rows_per_chunk,
  std::string const input_directory)
{
  auto filter_passthrough = ctx->create_channel();
  auto orders_passthrough = ctx->create_channel();
  rapidsmpf::streaming::ShutdownAtExit c{
    bloom_filter_in, ch_out, filter_passthrough, orders_passthrough};
  co_await ctx->executor()->schedule();
  // We want to await the bloom_filter being ready before kicking off the tasks to read
  // orders and apply the filter. This way, the read won't start until the bloom filter
  // is ready and we won't stack up num_producers chunks waiting for ages.
  auto filter      = co_await bloom_filter_in->receive();
  auto passthrough = [&]() -> coro::task<void> {
    co_await filter_passthrough->send(std::move(filter));
    co_await filter_passthrough->drain(ctx->executor());
  };
  rapidsmpf::streaming::coro_results(co_await coro::when_all(
    passthrough(),
    read_orders(ctx, comm, orders_passthrough, num_producers, num_rows_per_chunk, input_directory),
    bloom_filter.apply(filter_passthrough, orders_passthrough, ch_out, filter_keys)));
  co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Actor read_supplier(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                          std::shared_ptr<rapidsmpf::Communicator> comm,
                                          std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                          std::size_t num_producers,
                                          cudf::size_type num_rows_per_chunk,
                                          std::string const& input_directory)
{
  auto files = rapidsmpf::ndsh::detail::list_parquet_files(
    rapidsmpf::ndsh::detail::get_table_path(input_directory, "supplier"));
  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                   .column_names({"s_suppkey", "s_nationkey", "s_name"})
                   .build();
  return cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk);
}

rapidsmpf::streaming::Actor filter_lineitem(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                            std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
                                            std::shared_ptr<rapidsmpf::streaming::Channel> ch_out)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

  while (!ch_out->is_shutdown()) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto chunk = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);

    auto mask = cudf::binary_operation(chunk.table_view().column(2),
                                       chunk.table_view().column(3),
                                       cudf::binary_operator::GREATER,
                                       cudf::data_type(cudf::type_id::BOOL8),
                                       chunk.stream(),
                                       ctx->br()->device_mr());
    co_await ch_out->send(cudf_streaming::to_message(
      msg.sequence_number(),
      std::make_unique<cudf_streaming::table_chunk>(
        cudf::apply_boolean_mask(
          chunk.table_view().select({0, 1}), mask->view(), chunk.stream(), ctx->br()->device_mr()),
        chunk.stream())));
  }
  co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Actor filter_grouped_greater(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
  std::shared_ptr<coro::latch> latch)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
  bool released_lineitem_read = false;

  while (!ch_out->is_shutdown()) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto chunk = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);

    auto mask =
      cudf::binary_operation(chunk.table_view().column(1),
                             cudf::numeric_scalar<cudf::size_type>(
                               1, /* is_valid = */ true, chunk.stream(), ctx->br()->device_mr()),
                             cudf::binary_operator::GREATER,
                             cudf::data_type(cudf::type_id::BOOL8),
                             chunk.stream(),
                             ctx->br()->device_mr());
    co_await ch_out->send(cudf_streaming::to_message(
      msg.sequence_number(),
      std::make_unique<cudf_streaming::table_chunk>(
        cudf::apply_boolean_mask(
          chunk.table_view().select({0}), mask->view(), chunk.stream(), ctx->br()->device_mr()),
        chunk.stream())));
    if (!released_lineitem_read) {
      latch->count_down();
      released_lineitem_read = true;
    }
  }
  if (!released_lineitem_read) { latch->count_down(); }
  co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Actor filter_grouped_equal(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

  while (true) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto chunk = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);

    auto mask =
      cudf::binary_operation(chunk.table_view().column(1),
                             cudf::numeric_scalar<cudf::size_type>(
                               1, /* is_valid = */ true, chunk.stream(), ctx->br()->device_mr()),
                             cudf::binary_operator::EQUAL,
                             cudf::data_type(cudf::type_id::BOOL8),
                             chunk.stream(),
                             ctx->br()->device_mr());
    co_await ch_out->send(cudf_streaming::to_message(
      msg.sequence_number(),
      std::make_unique<cudf_streaming::table_chunk>(
        cudf::apply_boolean_mask(
          chunk.table_view().select({0}), mask->view(), chunk.stream(), ctx->br()->device_mr()),
        chunk.stream())));
  }
  co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Actor fanout_bounded(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                           std::shared_ptr<rapidsmpf::Communicator> comm,
                                           std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
                                           std::shared_ptr<rapidsmpf::streaming::Channel> ch1_out,
                                           std::vector<cudf::size_type> ch1_cols,
                                           std::shared_ptr<rapidsmpf::streaming::Channel> ch2_out)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch1_out, ch2_out};

  co_await ctx->executor()->schedule();
  while (true) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto chunk = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    // Here, we know that copying ch1_cols (a single col) is better than copying
    // ch2_cols (the whole table)
    std::vector<coro::task<bool>> tasks;
    if (!ch1_out->is_shutdown()) {
      auto msg1 = cudf_streaming::to_message(
        msg.sequence_number(),
        std::make_unique<cudf_streaming::table_chunk>(
          std::make_unique<cudf::table>(
            chunk.table_view().select(ch1_cols), chunk.stream(), ctx->br()->device_mr()),
          chunk.stream()));
      tasks.push_back(ch1_out->send(std::move(msg1)));
    }
    if (!ch2_out->is_shutdown()) {
      // TODO: We know here that ch2 wants the whole table.
      tasks.push_back(ch2_out->send(cudf_streaming::to_message(
        msg.sequence_number(), std::make_unique<cudf_streaming::table_chunk>(std::move(chunk)))));
    }
    if (!std::ranges::any_of(
          rapidsmpf::streaming::coro_results(co_await coro::when_all(std::move(tasks))),
          std::identity{})) {
      comm->logger()->print("Breaking after ", msg.sequence_number());
      break;
    };
  }

  rapidsmpf::streaming::coro_results(
    co_await coro::when_all(ch1_out->drain(ctx->executor()), ch2_out->drain(ctx->executor())));
}

rapidsmpf::streaming::Actor slice(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
                                  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                  std::int64_t global_start,
                                  std::int64_t global_end)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
  co_await ctx->executor()->schedule();
  std::int64_t current_row = 0;
  while (true) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto chunk = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);

    if (global_start == global_end) {
      co_await ch_out->send(
        cudf_streaming::to_message(msg.sequence_number(),
                                   std::make_unique<cudf_streaming::table_chunk>(
                                     cudf::empty_like(chunk.table_view()), chunk.stream())));
      break;
    }

    auto num_rows = chunk.table_view().num_rows();

    std::int64_t chunk_start = current_row;
    std::int64_t chunk_end   = current_row + num_rows;

    std::int64_t slice_start = std::max(chunk_start, global_start);
    std::int64_t slice_end   = std::min(chunk_end, global_end);

    if (slice_start < slice_end) {
      auto local_start = static_cast<cudf::size_type>(slice_start - chunk_start);
      auto local_end   = static_cast<cudf::size_type>(slice_end - chunk_start);

      if (local_start == 0 && local_end == num_rows) {
        co_await ch_out->send(cudf_streaming::to_message(
          msg.sequence_number(), std::make_unique<cudf_streaming::table_chunk>(std::move(chunk))));
      } else {
        auto sliced_table = std::make_unique<cudf::table>(
          cudf::slice(chunk.table_view(), {local_start, local_end})[0],
          chunk.stream(),
          ctx->br()->device_mr());
        co_await ch_out->send(cudf_streaming::to_message(
          msg.sequence_number(),
          std::make_unique<cudf_streaming::table_chunk>(std::move(sliced_table), chunk.stream())));
      }
    }
    current_row += num_rows;
    if (current_row >= global_end) { break; }
  }
  co_await ch_out->drain(ctx->executor());
}

std::vector<rapidsmpf::ndsh::groupby_request> count_groupby_request()
{
  auto requests = std::vector<rapidsmpf::ndsh::groupby_request>();
  std::vector<std::function<std::unique_ptr<cudf::groupby_aggregation>()>> aggs;
  // count(*)
  aggs.emplace_back([]() {
    return cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
  });
  requests.emplace_back(0, std::move(aggs));
  return requests;
}

std::vector<rapidsmpf::ndsh::groupby_request> sum_groupby_request(cudf::size_type column)
{
  auto requests = std::vector<rapidsmpf::ndsh::groupby_request>();
  std::vector<std::function<std::unique_ptr<cudf::groupby_aggregation>()>> aggs;
  // count(*)
  aggs.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>);
  requests.emplace_back(column, std::move(aggs));
  return requests;
}

rapidsmpf::streaming::Actor populate_bloom_filter(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::Communicator> comm,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
  std::vector<cudf::size_type> keys,
  rapidsmpf::OpID tag,
  cudf_streaming::bloom_filter bloom_filter)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
  auto passthrough = ctx->create_channel();
  auto selector    = [&]() -> rapidsmpf::streaming::Actor {
    rapidsmpf::streaming::ShutdownAtExit c{passthrough};
    co_await ctx->executor()->schedule();
    while (!passthrough->is_shutdown()) {
      auto msg = co_await ch_in->receive();
      if (msg.empty()) { break; }
      auto chunk  = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
      auto stream = chunk.stream();
      auto out    = std::make_unique<cudf::table>(
        chunk.table_view().select(keys), stream, ctx->br()->device_mr());
      std::ignore = std::move(chunk);
      co_await passthrough->send(cudf_streaming::to_message(
        msg.sequence_number(),
        std::make_unique<cudf_streaming::table_chunk>(std::move(out), stream)));
    }
    comm->logger()->print("Sent all things through filter");
    co_await passthrough->drain(ctx->executor());
  };
  rapidsmpf::streaming::coro_results(
    co_await coro::when_all(selector(), bloom_filter.build(passthrough, ch_out, tag)));
  co_await ch_out->drain(ctx->executor());
}
}  // namespace

/**
 * @brief Run a derived version of TPC-H query 21.
 *
 * The SQL form of the query is:
 * @code{.sql}
 * select
 *     s_name,
 *     count(*) as numwait
 * from
 *     supplier,
 *     lineitem l1,
 *     orders,
 *     nation
 * where
 *     s_suppkey = l1.l_suppkey
 *     and o_orderkey = l1.l_orderkey
 *     and o_orderstatus = 'F'
 *     and l1.l_receiptdate > l1.l_commitdate
 *     and exists (
 *         select
 *             *
 *         from
 *             lineitem l2
 *         where
 *             l2.l_orderkey = l1.l_orderkey
 *             and l2.l_suppkey <> l1.l_suppkey
 *     )
 *     and not exists (
 *         select
 *             *
 *         from
 *             lineitem l3
 *         where
 *             l3.l_orderkey = l1.l_orderkey
 *             and l3.l_suppkey <> l1.l_suppkey
 *             and l3.l_receiptdate > l3.l_commitdate
 *     )
 *     and s_nationkey = n_nationkey
 *     and n_name = 'SAUDI ARABIA'
 * group by
 *     s_name
 * order by
 *     numwait desc,
 *     s_name
 * limit 100
 * @endcode{}
 */
int main(int argc, char** argv)
{
  rapidsmpf::ndsh::FinalizeMPI finalize{};
  CUDF_CUDA_TRY(cudaFree(nullptr));
  cudf::initialize();
  auto mr                 = rmm::mr::cuda_async_memory_resource{};
  auto arguments          = rapidsmpf::ndsh::parse_arguments(argc, argv);
  auto [ctx, comm]        = rapidsmpf::ndsh::create_context(arguments, std::move(mr));
  std::string output_path = arguments.output_file;
  std::vector<double> timings;
  int l2size;
  int device;
  RAPIDSMPF_CUDA_TRY(cudaGetDevice(&device));
  RAPIDSMPF_CUDA_TRY(cudaDeviceGetAttribute(&l2size, cudaDevAttrL2CacheSize, device));
  auto const filter_size =
    cudf_streaming::bloom_filter::aligned_size(static_cast<std::size_t>(l2size) * 2 / 3);
  for (int i = 0; i < arguments.num_iterations; i++) {
    int op_id{0};
    std::vector<rapidsmpf::streaming::Actor> actors;
    auto start = std::chrono::steady_clock::now();
    // TODO: configurable/adaptive
    std::uint32_t num_shuffle_partitions = 16;
    {
      // Idea:
      // We need to shuffle lineitem, but then we always join/groupby on l_orderkey
      // The join against orders will be done via shuffle join. The selectivity of
      // the supplier x nation join is about 1/25, so that can be a broadcast join.
      //
      // We commit to reading the lineitem table twice because one of the reads only
      // needs a single column and we can do all of the processing of it before
      // reading the second time, keeping memory under control.

      RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing Q21 pipeline");
      auto lineitem_orderkey = ctx->create_channel();
      actors.push_back(read_lineitem(ctx,
                                     comm,
                                     lineitem_orderkey,
                                     /* num_tickets */ 2,
                                     arguments.num_rows_per_chunk,
                                     arguments.input_directory,
                                     {"l_orderkey"}));  // "l_orderkey"
      auto lineitem_orderkey_grouped = ctx->create_channel();
      actors.push_back(
        rapidsmpf::ndsh::chunkwise_group_by(ctx,
                                            lineitem_orderkey,
                                            lineitem_orderkey_grouped,
                                            {0},
                                            count_groupby_request(),
                                            cudf::null_policy::INCLUDE));  // l_orderkey, count(*)
      auto lineitem_orderkey_shuffled = ctx->create_channel();
      actors.push_back(
        rapidsmpf::ndsh::shuffle(ctx,
                                 comm,
                                 lineitem_orderkey_grouped,
                                 lineitem_orderkey_shuffled,
                                 {0},
                                 num_shuffle_partitions,
                                 static_cast<rapidsmpf::OpID>(10 * i) +
                                   op_id++));  // l_orderkey, count(*) [shuffled on l_orderkey]
      auto lineitem_orderkey_shuffled_grouped = ctx->create_channel();
      actors.push_back(rapidsmpf::ndsh::chunkwise_group_by(
        ctx,
        lineitem_orderkey_shuffled,
        lineitem_orderkey_shuffled_grouped,
        {0},
        sum_groupby_request(1),
        cudf::null_policy::INCLUDE));  // l_orderkey, sum(count(*)) [groupby done]
      auto lineitem_orderkey_filtered = ctx->create_channel();
      auto latch                      = std::make_shared<coro::latch>(1);
      actors.push_back(
        filter_grouped_greater(ctx,
                               lineitem_orderkey_shuffled_grouped,
                               lineitem_orderkey_filtered,
                               latch));  // l_orderkey [sum(count(*)) > 1, releases lineitem read]
      auto lineitem_suppkey = ctx->create_channel();
      actors.push_back(read_lineitem(ctx,
                                     comm,
                                     lineitem_suppkey,
                                     /* num_tickets */ 2,
                                     arguments.num_rows_per_chunk,
                                     arguments.input_directory,
                                     {"l_orderkey", "l_suppkey", "l_receiptdate", "l_commitdate"},
                                     latch));  // l_orderkey, l_suppkey, l_receiptdate, l_commitdate
      // [released once filter_grouped_greater has seen an input]
      auto lineitem_suppkey_filtered = ctx->create_channel();
      actors.push_back(filter_lineitem(
        ctx, lineitem_suppkey, lineitem_suppkey_filtered));  // l_orderkey, l_suppkey
      auto lineitem_suppkey_shuffled = ctx->create_channel();
      actors.push_back(
        rapidsmpf::ndsh::shuffle(ctx,
                                 comm,
                                 lineitem_suppkey_filtered,
                                 lineitem_suppkey_shuffled,
                                 {0},
                                 num_shuffle_partitions,
                                 static_cast<rapidsmpf::OpID>(10 * i) +
                                   op_id++));  // l_orderkey, l_suppkey [shuffled on l_orderkey]
      auto lineitem_self_joined = ctx->create_channel();
      actors.push_back(
        rapidsmpf::ndsh::inner_join_shuffle(ctx,
                                            comm,
                                            lineitem_orderkey_filtered,
                                            lineitem_suppkey_shuffled,
                                            lineitem_self_joined,
                                            {0},
                                            {0}));  // l_orderkey, l_suppkey [join complete]

      auto joined_grouped_input = ctx->create_channel();
      auto joined_input         = ctx->create_channel();
      actors.push_back(fanout_bounded(ctx,
                                      comm,
                                      lineitem_self_joined,
                                      joined_grouped_input,
                                      {0},
                                      joined_input));  // l_orderkey (in joined_grouped_input),
      // l_orderkey l_suppkey (in joined_input)
      auto joined_grouped_len = ctx->create_channel();
      actors.push_back(rapidsmpf::ndsh::chunkwise_group_by(
        ctx,
        joined_grouped_input,
        joined_grouped_len,
        {0},
        count_groupby_request(),
        cudf::null_policy::INCLUDE));  // l_orderkey, count(*) [complete, because partitioned on
                                       // l_orderkey]
      auto joined_grouped_filter = ctx->create_channel();
      actors.push_back(filter_grouped_equal(
        ctx, joined_grouped_len, joined_grouped_filter));  // l_orderkey [count(*) == 1]
      auto lineitem_joined = ctx->create_channel();
      actors.push_back(rapidsmpf::ndsh::inner_join_shuffle(
        ctx, comm, joined_grouped_filter, joined_input, lineitem_joined, {0}, {0}));  // l_orderkey,
                                                                                      // l_suppkey
      auto supplier = ctx->create_channel();
      auto nation   = ctx->create_channel();
      actors.push_back(read_supplier(ctx,
                                     comm,
                                     supplier,
                                     2,
                                     arguments.num_rows_per_chunk,
                                     arguments.input_directory));  // s_suppkey, s_nationkey, s_name
      actors.push_back(read_nation(ctx,
                                   comm,
                                   nation,
                                   1,
                                   arguments.num_rows_per_chunk,
                                   arguments.input_directory));  // n_nationkey
      auto supp_x_nation = ctx->create_channel();
      actors.push_back(
        rapidsmpf::ndsh::inner_join_broadcast(ctx,
                                              comm,
                                              nation,
                                              supplier,
                                              supp_x_nation,
                                              {0},
                                              {1},
                                              static_cast<rapidsmpf::OpID>(10 * i + op_id++),
                                              rapidsmpf::ndsh::KeepKeys::NO));  // s_suppkey, s_name
      auto supp_nation_lineitem = ctx->create_channel();
      actors.push_back(rapidsmpf::ndsh::inner_join_broadcast(
        ctx,
        comm,
        supp_x_nation,
        lineitem_joined,
        supp_nation_lineitem,
        {0},
        {1},
        static_cast<rapidsmpf::OpID>(10 * i + op_id++),
        rapidsmpf::ndsh::KeepKeys::NO));  // s_name, l_orderkey [this join is quite selective]
      // OK, we're going to pre-filter the orders table before joining using a bloom
      // filter.
      // This has two consequences:
      // 1. We shuffle less data
      // 2. This acts as a latch: the orders table is not read and shuffled until
      // the other side is "ready", reducing memory pressure.
      auto bloom_input     = ctx->create_channel();
      auto snl_passthrough = ctx->create_channel();
      // Bloom filter needs to see all the input before we can release the orders
      // read, so need unbounded fanout.
      actors.push_back(
        rapidsmpf::streaming::actor::fanout(ctx,
                                            supp_nation_lineitem,
                                            {bloom_input, snl_passthrough},
                                            rapidsmpf::streaming::actor::FanoutPolicy::UNBOUNDED));
      auto bloom_output = ctx->create_channel();
      auto bloom_filter =
        cudf_streaming::bloom_filter(ctx, comm, cudf::DEFAULT_HASH_SEED, filter_size);
      // Select the relevant key column(s) and build filter.
      actors.push_back(populate_bloom_filter(ctx,
                                             comm,
                                             bloom_input,
                                             bloom_output,
                                             {1},
                                             static_cast<rapidsmpf::OpID>(10 * i + op_id++),
                                             bloom_filter));
      auto orders          = ctx->create_channel();
      auto shuffled_orders = ctx->create_channel();
      // OK, now we obtain the filter, and release the orders read which we apply
      // the filter to before sending on to the shuffle.
      actors.push_back(read_orders_with_bloom_filter(ctx,
                                                     comm,
                                                     bloom_output,
                                                     orders,
                                                     {0},
                                                     bloom_filter,
                                                     2,
                                                     arguments.num_rows_per_chunk,
                                                     arguments.input_directory));  // o_orderkey
      actors.push_back(rapidsmpf::ndsh::shuffle(
        ctx,
        comm,
        orders,
        shuffled_orders,
        {0},
        num_shuffle_partitions,
        static_cast<rapidsmpf::OpID>(10 * i + op_id++)));  // o_orderkey [shuffled on o_orderkey]
      auto all_joined = ctx->create_channel();
      actors.push_back(
        rapidsmpf::ndsh::inner_join_shuffle(ctx,
                                            comm,
                                            snl_passthrough,
                                            shuffled_orders,
                                            all_joined,
                                            {1},
                                            {0},
                                            rapidsmpf::ndsh::KeepKeys::NO));  // s_name
      auto chunked_groupby = ctx->create_channel();
      actors.push_back(
        rapidsmpf::ndsh::chunkwise_group_by(ctx,
                                            all_joined,
                                            chunked_groupby,
                                            {0},
                                            count_groupby_request(),
                                            cudf::null_policy::INCLUDE));  // s_name, count(*)
      auto final_groupby_input = ctx->create_channel();
      if (comm->nranks() > 1) {
        actors.push_back(rapidsmpf::ndsh::broadcast(ctx,
                                                    comm,
                                                    chunked_groupby,
                                                    final_groupby_input,
                                                    static_cast<rapidsmpf::OpID>(10 * i + op_id++),
                                                    rapidsmpf::streaming::AllGather::Ordered::NO));
      } else {
        actors.push_back(rapidsmpf::ndsh::concatenate(ctx, chunked_groupby, final_groupby_input));
      }
      if (comm->rank() == 0) {
        auto final_groupby_output = ctx->create_channel();
        actors.push_back(rapidsmpf::ndsh::chunkwise_group_by(
          ctx,
          final_groupby_input,
          final_groupby_output,
          {0},
          sum_groupby_request(1),
          cudf::null_policy::INCLUDE));  // s_name, sum(count(*)) [only a single partition now due
                                         // to the broadcast]
        auto sorted_output = ctx->create_channel();
        actors.push_back(
          rapidsmpf::ndsh::chunkwise_sort_by(ctx,
                                             final_groupby_output,
                                             sorted_output,
                                             {1, 0},
                                             {0, 1},
                                             {cudf::order::DESCENDING, cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE, cudf::null_order::BEFORE}));
        auto sliced = ctx->create_channel();
        actors.push_back(slice(ctx, sorted_output, sliced, 0, 100));
        actors.push_back(rapidsmpf::ndsh::write_parquet(
          ctx, sliced, cudf::io::sink_info{arguments.output_file}, {"s_name", "numwait"}));
      } else {
        actors.push_back(rapidsmpf::ndsh::sink_channel(ctx, final_groupby_input));
      }
    }
    auto end                               = std::chrono::steady_clock::now();
    std::chrono::duration<double> pipeline = end - start;
    start                                  = std::chrono::steady_clock::now();
    {
      RAPIDSMPF_NVTX_SCOPED_RANGE("Q21 Iteration");
      rapidsmpf::streaming::run_actor_network(std::move(actors));
    }
    end                                   = std::chrono::steady_clock::now();
    std::chrono::duration<double> compute = end - start;
    timings.push_back(pipeline.count());
    timings.push_back(compute.count());
    auto statistics = ctx->statistics();
    comm->logger()->print(
      statistics->report({.mr = ctx->br()->device_mr(), .pinned_mr = ctx->br()->try_pinned_mr()}));
    statistics->clear();
  }

  if (comm->rank() == 0) {
    for (std::size_t i = 0; i < safe_cast<std::size_t>(arguments.num_iterations); i++) {
      comm->logger()->print("Iteration ", i, " pipeline construction time [s]: ", timings[2 * i]);
      comm->logger()->print("Iteration ", i, " compute time [s]: ", timings[2 * i + 1]);
    }
  }
  return 0;
}

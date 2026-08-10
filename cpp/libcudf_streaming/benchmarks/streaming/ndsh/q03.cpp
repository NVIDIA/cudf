/**
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "concatenate.hpp"
#include "groupby.hpp"
#include "join.hpp"
#include "parquet_writer.hpp"
#include "utils.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/ast/expressions.hpp>
#include <cudf/context.hpp>
#include <cudf/copying.hpp>
#include <cudf/groupby.hpp>
#include <cudf/hashing.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/merge.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cudf_streaming/bloom_filter.hpp>
#include <cudf_streaming/parquet.hpp>
#include <cudf_streaming/partition_utils.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <driver_types.h>
#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/cuda_event.hpp>
#include <rapidsmpf/cuda_stream.hpp>
#include <rapidsmpf/memory/packed_data.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>
#include <rapidsmpf/utils/misc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

namespace {

rapidsmpf::streaming::Actor read_customer(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                          std::shared_ptr<rapidsmpf::Communicator> comm,
                                          std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                          std::size_t num_producers,
                                          cudf::size_type num_rows_per_chunk,
                                          std::string const& input_directory)
{
  auto files = rapidsmpf::ndsh::detail::list_parquet_files(
    rapidsmpf::ndsh::detail::get_table_path(input_directory, "customer"));
  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                   .column_names({"c_custkey"})  // 0
                   .build();
  auto filter_expr = [&]() -> std::unique_ptr<cudf_streaming::filter> {
    auto stream = ctx->br()->stream_pool()->get_stream();
    auto owner  = new std::vector<std::any>;
    owner->push_back(std::make_shared<cudf::string_scalar>("BUILDING", true, stream));
    owner->push_back(std::make_shared<cudf::ast::literal>(
      *std::any_cast<std::shared_ptr<cudf::string_scalar>>(owner->at(0))));
    owner->push_back(std::make_shared<cudf::ast::column_name_reference>("c_mktsegment"));
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

rapidsmpf::streaming::Actor read_lineitem(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                          std::shared_ptr<rapidsmpf::Communicator> comm,
                                          std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                          std::size_t num_producers,
                                          cudf::size_type num_rows_per_chunk,
                                          std::string const& input_directory,
                                          bool use_date32)
{
  auto files = rapidsmpf::ndsh::detail::list_parquet_files(
    rapidsmpf::ndsh::detail::get_table_path(input_directory, "lineitem"));
  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                   .column_names({
                     "l_orderkey",       // 0
                     "l_extendedprice",  // 1
                     "l_discount",       // 2
                   })
                   .build();
  auto stream = ctx->br()->stream_pool()->get_stream();
  // l_shipdate > DATE '1995-03-15'
  constexpr auto date = cuda::std::chrono::year_month_day(
    cuda::std::chrono::year(1995), cuda::std::chrono::month(3), cuda::std::chrono::day(15));
  auto filter_expr = use_date32 ? rapidsmpf::ndsh::make_date_filter<cudf::timestamp_D>(
                                    stream, date, "l_shipdate", cudf::ast::ast_operator::GREATER)
                                : rapidsmpf::ndsh::make_date_filter<cudf::timestamp_ms>(
                                    stream, date, "l_shipdate", cudf::ast::ast_operator::GREATER);
  return cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk, std::move(filter_expr));
}

rapidsmpf::streaming::Actor read_orders(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                        std::shared_ptr<rapidsmpf::Communicator> comm,
                                        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                        std::size_t num_producers,
                                        cudf::size_type num_rows_per_chunk,
                                        std::string const& input_directory,
                                        bool use_date32)
{
  auto files = rapidsmpf::ndsh::detail::list_parquet_files(
    rapidsmpf::ndsh::detail::get_table_path(input_directory, "orders"));
  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                   .column_names({
                     "o_orderkey",      // 0
                     "o_orderdate",     // 1
                     "o_shippriority",  // 2
                     "o_custkey"        // 3
                   })
                   .build();
  auto stream = ctx->br()->stream_pool()->get_stream();
  // o_orderdate < DATE '1995-03-15'
  constexpr auto date = cuda::std::chrono::year_month_day(
    cuda::std::chrono::year(1995), cuda::std::chrono::month(3), cuda::std::chrono::day(15));
  auto filter_expr = use_date32 ? rapidsmpf::ndsh::make_date_filter<cudf::timestamp_D>(
                                    stream, date, "o_orderdate", cudf::ast::ast_operator::LESS)
                                : rapidsmpf::ndsh::make_date_filter<cudf::timestamp_ms>(
                                    stream, date, "o_orderdate", cudf::ast::ast_operator::LESS);
  return cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk, std::move(filter_expr));
}

std::vector<rapidsmpf::ndsh::groupby_request> chunkwise_groupby_requests()
{
  auto requests = std::vector<rapidsmpf::ndsh::groupby_request>();
  std::vector<std::function<std::unique_ptr<cudf::groupby_aggregation>()>> aggs;
  // sum(revenue)
  aggs.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>);
  requests.emplace_back(3, std::move(aggs));
  return requests;
}

// In: o_orderkey, o_orderdate, o_shippriority, l_extendedprice, l_discount
// Out: o_orderkey, o_orderdate, o_shippriority, revenue = (l_extendedprice - (1 -
// l_discount))
rapidsmpf::streaming::Actor select_columns_for_groupby(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

  co_await ctx->executor()->schedule();
  while (true) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto chunk           = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    auto chunk_stream    = chunk.stream();
    auto sequence_number = msg.sequence_number();
    auto table           = chunk.table_view();
    std::vector<std::unique_ptr<cudf::column>> result;
    result.reserve(4);

    // o_orderkey
    result.push_back(
      std::make_unique<cudf::column>(table.column(0), chunk_stream, ctx->br()->device_mr()));
    // o_orderdate
    result.push_back(
      std::make_unique<cudf::column>(table.column(1), chunk_stream, ctx->br()->device_mr()));
    // o_shippriority
    result.push_back(
      std::make_unique<cudf::column>(table.column(2), chunk_stream, ctx->br()->device_mr()));
    auto extendedprice = table.column(3);
    auto discount      = table.column(4);
    std::string udf =
      R"***(
static __device__ void calculate_revenue(double *revenue, double extprice, double discount) {
    *revenue = extprice * (1 - discount);
}
           )***";

    // revenue
    result.push_back(std::move(
      cudf::transform(udf,
                      cudf::udf_source_type::CUDA,
                      cudf::null_aware::NO,
                      std::nullopt,
                      std::vector<cudf::transform_input>{extendedprice, discount},
                      std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT64),
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt,
                      chunk_stream,
                      ctx->br()->device_mr())
        ->release()
        .front()));
    co_await ch_out->send(cudf_streaming::to_message(
      sequence_number,
      std::make_unique<cudf_streaming::table_chunk>(
        std::make_unique<cudf::table>(std::move(result)), chunk_stream)));
  }
  co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Actor top_k_by(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                     std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
                                     std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                     std::vector<cudf::size_type> keys,
                                     std::vector<cudf::size_type> values,
                                     std::vector<cudf::order> order,
                                     cudf::size_type k)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

  co_await ctx->executor()->schedule();
  std::vector<std::unique_ptr<cudf::table>> partials;
  std::vector<rmm::cuda_stream_view> chunk_streams;
  while (true) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto chunk         = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    auto const indices = cudf::sorted_order(
      chunk.table_view().select(keys), order, {}, chunk.stream(), ctx->br()->device_mr());
    partials.push_back(cudf::gather(chunk.table_view().select(values),
                                    cudf::split(indices->view(), {k}, chunk.stream()).front(),
                                    cudf::out_of_bounds_policy::DONT_CHECK,
                                    chunk.stream(),
                                    ctx->br()->device_mr()));
    chunk_streams.push_back(chunk.stream());
  }

  // TODO: multi-node
  RAPIDSMPF_EXPECTS(chunk_streams.size() > 0, "No chunks to sort");
  auto out_stream = chunk_streams.front();
  rapidsmpf::CudaEvent event;
  rapidsmpf::cuda_stream_join(std::ranges::single_view{out_stream}, chunk_streams, &event);
  std::vector<cudf::table_view> views;
  std::ranges::transform(partials, std::back_inserter(views), [](auto& t) { return t->view(); });
  auto merged = cudf::merge(views, keys, order, {}, out_stream, ctx->br()->device_mr());
  auto result = std::make_unique<cudf::table>(
    cudf::slice(merged->view(), {0, 10}, out_stream), out_stream, ctx->br()->device_mr());
  co_await ch_out->send(cudf_streaming::to_message(
    0, std::make_unique<cudf_streaming::table_chunk>(std::move(result), out_stream)));
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
}  // namespace

/**
 * @brief Run a derived version of TPC-H query 3.
 *
 * The SQL form of the query is:
 * @code{.sql}
 * select
 *     l_orderkey,
 *     sum(l_extendedprice * (1 - l_discount)) as revenue,
 *     o_orderdate,
 *     o_shippriority
 * from
 *     customer,
 *     orders,
 *     lineitem
 * where
 *     c_mktsegment = 'BUILDING'
 *     and c_custkey = o_custkey
 *     and l_orderkey = o_orderkey
 *     and o_orderdate < '1995-03-15'
 *     and l_shipdate > '1995-03-15'
 * group by
 *     l_orderkey,
 *     o_orderdate,
 *     o_shippriority
 * order by
 *     revenue desc,
 *     o_orderdate
 * limit 10
 * @endcode{}
 */
int main(int argc, char** argv)
{
  rapidsmpf::ndsh::FinalizeMPI finalize{};
  CUDF_CUDA_TRY(cudaFree(nullptr));
  // work around https://github.com/rapidsai/cudf/issues/20849
  cudf::initialize();
  auto mr                 = rmm::mr::cuda_async_memory_resource{};
  auto arguments          = rapidsmpf::ndsh::parse_arguments(argc, argv);
  auto [ctx, comm]        = rapidsmpf::ndsh::create_context(arguments, std::move(mr));
  std::string output_path = arguments.output_file;

  // Detect date column types from parquet metadata before timed section
  auto const lineitem_types =
    rapidsmpf::ndsh::detail::get_column_types(arguments.input_directory, "lineitem");
  bool const lineitem_use_date32 =
    lineitem_types.at("l_shipdate").id() == cudf::type_id::TIMESTAMP_DAYS;
  auto const orders_types =
    rapidsmpf::ndsh::detail::get_column_types(arguments.input_directory, "orders");
  bool const orders_use_date32 =
    orders_types.at("o_orderdate").id() == cudf::type_id::TIMESTAMP_DAYS;

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
    {
      RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing Q3 pipeline");
      auto customer = ctx->create_channel();
      auto lineitem = ctx->create_channel();
      auto orders   = ctx->create_channel();

      auto customer_x_orders            = ctx->create_channel();
      auto customer_x_orders_x_lineitem = ctx->create_channel();

      // Out: "c_custkey"
      actors.push_back(read_customer(ctx,
                                     comm,
                                     customer,
                                     /* num_tickets */ 2,
                                     arguments.num_rows_per_chunk,
                                     arguments.input_directory));
      // Out: o_orderkey, o_orderdate, o_shippriority, o_custkey
      actors.push_back(read_orders(ctx,
                                   comm,
                                   orders,
                                   6,
                                   arguments.num_rows_per_chunk,
                                   arguments.input_directory,
                                   orders_use_date32));
      // join c_custkey = o_custkey
      // Out: o_orderkey, o_orderdate, o_shippriority
      actors.push_back(
        rapidsmpf::ndsh::inner_join_broadcast(ctx,
                                              comm,
                                              customer,
                                              orders,
                                              customer_x_orders,
                                              {0},
                                              {3},
                                              static_cast<rapidsmpf::OpID>(10 * i + op_id++),
                                              rapidsmpf::ndsh::KeepKeys::NO));
      auto bloom_filter_input      = ctx->create_channel();
      auto bloom_filter_output     = ctx->create_channel();
      auto customer_x_orders_input = ctx->create_channel();
      actors.push_back(fanout_bounded(
        ctx, comm, customer_x_orders, bloom_filter_input, {0}, customer_x_orders_input));
      auto bloom_filter =
        cudf_streaming::bloom_filter(ctx, comm, cudf::DEFAULT_HASH_SEED, filter_size);
      actors.push_back(bloom_filter.build(
        bloom_filter_input, bloom_filter_output, static_cast<rapidsmpf::OpID>(10 * i + op_id++)));
      // Out: l_orderkey, l_extendedprice, l_discount
      actors.push_back(read_lineitem(ctx,
                                     comm,
                                     lineitem,
                                     /* num_tickets */ 4,
                                     arguments.num_rows_per_chunk,
                                     arguments.input_directory,
                                     lineitem_use_date32));
      auto lineitem_output = ctx->create_channel();
      actors.push_back(bloom_filter.apply(bloom_filter_output, lineitem, lineitem_output, {0}));
      // join o_orderkey = l_orderkey
      // Out: o_orderkey, o_orderdate, o_shippriority, l_extendedprice,
      // l_discount
      if (arguments.use_shuffle_join) {
        auto lineitem_shuffled          = ctx->create_channel();
        auto customer_x_orders_shuffled = ctx->create_channel();
        std::uint32_t num_partitions    = 16;
        actors.push_back(rapidsmpf::ndsh::shuffle(ctx,
                                                  comm,
                                                  lineitem_output,
                                                  lineitem_shuffled,
                                                  {0},
                                                  num_partitions,
                                                  static_cast<rapidsmpf::OpID>(10 * i + op_id++)));
        actors.push_back(rapidsmpf::ndsh::shuffle(ctx,
                                                  comm,
                                                  customer_x_orders_input,
                                                  customer_x_orders_shuffled,
                                                  {0},
                                                  num_partitions,
                                                  static_cast<rapidsmpf::OpID>(10 * i + op_id++)));
        actors.push_back(rapidsmpf::ndsh::inner_join_shuffle(ctx,
                                                             comm,
                                                             customer_x_orders_shuffled,
                                                             lineitem_shuffled,
                                                             customer_x_orders_x_lineitem,
                                                             {0},
                                                             {0},
                                                             rapidsmpf::ndsh::KeepKeys::YES));
      } else {
        actors.push_back(
          rapidsmpf::ndsh::inner_join_broadcast(ctx,
                                                comm,
                                                customer_x_orders_input,
                                                lineitem_output,
                                                customer_x_orders_x_lineitem,
                                                {0},
                                                {0},
                                                static_cast<rapidsmpf::OpID>(10 * i + op_id++),
                                                rapidsmpf::ndsh::KeepKeys::YES));
      }
      auto groupby_input = ctx->create_channel();
      // Out: o_orderkey, o_orderdate, o_shippriority, revenue
      actors.push_back(
        select_columns_for_groupby(ctx, customer_x_orders_x_lineitem, groupby_input));
      auto chunkwise_groupby_output = ctx->create_channel();
      // Out: o_orderkey, o_orderdate, o_shippriority, revenue
      actors.push_back(rapidsmpf::ndsh::chunkwise_group_by(ctx,
                                                           groupby_input,
                                                           chunkwise_groupby_output,
                                                           {0, 1, 2},
                                                           chunkwise_groupby_requests(),
                                                           cudf::null_policy::INCLUDE));
      auto final_groupby_input = ctx->create_channel();
      if (comm->nranks() > 1) {
        actors.push_back(rapidsmpf::ndsh::broadcast(ctx,
                                                    comm,
                                                    chunkwise_groupby_output,
                                                    final_groupby_input,
                                                    static_cast<rapidsmpf::OpID>(10 * i + op_id++),
                                                    rapidsmpf::streaming::AllGather::Ordered::NO));
      } else {
        actors.push_back(
          rapidsmpf::ndsh::concatenate(ctx, chunkwise_groupby_output, final_groupby_input));
      }
      if (comm->rank() == 0) {
        auto final_groupby_output = ctx->create_channel();
        // Out: o_orderkey, o_orderdate, o_shippriority, revenue
        actors.push_back(rapidsmpf::ndsh::chunkwise_group_by(ctx,
                                                             final_groupby_input,
                                                             final_groupby_output,
                                                             {0, 1, 2},
                                                             chunkwise_groupby_requests(),
                                                             cudf::null_policy::INCLUDE

                                                             ));
        auto topk = ctx->create_channel();
        // Out: o_orderkey, revenue, o_orderdate, o_shippriority
        actors.push_back(top_k_by(ctx,
                                  final_groupby_output,
                                  topk,
                                  {3, 1},
                                  {0, 3, 1, 2},
                                  {cudf::order::DESCENDING, cudf::order::ASCENDING},
                                  10));
        actors.push_back(rapidsmpf::ndsh::write_parquet(
          ctx,
          topk,
          cudf::io::sink_info(output_path),
          {"l_orderkey", "revenue", "o_orderdate", "o_shippriority"}));
      } else {
        actors.push_back(rapidsmpf::ndsh::sink_channel(ctx, final_groupby_input));
      }
    }
    auto end                               = std::chrono::steady_clock::now();
    std::chrono::duration<double> pipeline = end - start;
    start                                  = std::chrono::steady_clock::now();
    {
      RAPIDSMPF_NVTX_SCOPED_RANGE("Q3 Iteration");
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
    for (int i = 0; i < arguments.num_iterations; i++) {
      comm->logger()->print("Iteration ",
                            i,
                            " pipeline construction time [s]: ",
                            timings[rapidsmpf::safe_cast<std::size_t>(2 * i)]);
      comm->logger()->print("Iteration ",
                            i,
                            " compute time [s]: ",
                            timings[rapidsmpf::safe_cast<std::size_t>(2 * i + 1)]);
    }
  }
  return 0;
}

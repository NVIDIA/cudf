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
#include <cudf/context.hpp>
#include <cudf/datetime.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/round.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>

#include <cudf_streaming/parquet.hpp>
#include <cudf_streaming/table_chunk.hpp>

#include <rmm/mr/cuda_async_memory_resource.hpp>

#include <cuda_runtime_api.h>

#include <rapidsmpf/communicator/communicator.hpp>
#include <rapidsmpf/nvtx.hpp>
#include <rapidsmpf/streaming/coll/allgather.hpp>
#include <rapidsmpf/streaming/core/actor.hpp>
#include <rapidsmpf/streaming/core/channel.hpp>
#include <rapidsmpf/streaming/core/context.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

using rapidsmpf::safe_cast;

namespace {

rapidsmpf::streaming::Actor read_lineitem(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                          std::shared_ptr<rapidsmpf::Communicator> comm,
                                          std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                          std::size_t num_producers,
                                          cudf::size_type num_rows_per_chunk,
                                          std::string const& input_directory)
{
  auto files = rapidsmpf::ndsh::detail::list_parquet_files(
    rapidsmpf::ndsh::detail::get_table_path(input_directory, "lineitem"));
  auto options =
    cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
      .column_names(
        {"l_discount", "l_extendedprice", "l_orderkey", "l_partkey", "l_quantity", "l_suppkey"})
      .build();
  return cudf_streaming::actor::read_parquet(
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
                   .column_names({"n_name", "n_nationkey"})
                   .build();
  return cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk);
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
                   .column_names({"o_orderdate", "o_orderkey"})
                   .build();
  return cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk);
}

rapidsmpf::streaming::Actor read_part(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                      std::shared_ptr<rapidsmpf::Communicator> comm,
                                      std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                      std::size_t num_producers,
                                      cudf::size_type num_rows_per_chunk,
                                      std::string const& input_directory)
{
  auto files = rapidsmpf::ndsh::detail::list_parquet_files(
    rapidsmpf::ndsh::detail::get_table_path(input_directory, "part"));
  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                   .column_names({"p_partkey", "p_name"})
                   .build();
  return cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk);
}

rapidsmpf::streaming::Actor read_partsupp(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                          std::shared_ptr<rapidsmpf::Communicator> comm,
                                          std::shared_ptr<rapidsmpf::streaming::Channel> ch_out,
                                          std::size_t num_producers,
                                          cudf::size_type num_rows_per_chunk,
                                          std::string const& input_directory)
{
  auto files = rapidsmpf::ndsh::detail::list_parquet_files(
    rapidsmpf::ndsh::detail::get_table_path(input_directory, "partsupp"));
  auto options = cudf::io::parquet_reader_options::builder(cudf::io::source_info(files))
                   .column_names({"ps_partkey", "ps_suppkey", "ps_supplycost"})
                   .build();
  return cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk);
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
                   .column_names({"s_nationkey", "s_suppkey"})
                   .build();
  return cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk);
}

rapidsmpf::streaming::Actor filter_part(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                        std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
                                        std::shared_ptr<rapidsmpf::streaming::Channel> ch_out)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
  auto mr = ctx->br()->device_mr();
  while (true) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    co_await ctx->executor()->schedule();
    auto chunk        = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    auto chunk_stream = chunk.stream();
    auto table        = chunk.table_view();
    auto p_name       = table.column(1);
    auto target       = cudf::make_string_scalar("green", chunk_stream, mr);
    auto mask         = cudf::strings::contains(
      p_name, *static_cast<cudf::string_scalar*>(target.get()), chunk_stream, mr);
    co_await ch_out->send(cudf_streaming::to_message(
      msg.sequence_number(),
      std::make_unique<cudf_streaming::table_chunk>(
        cudf::apply_retention_mask(table.select({0}), mask->view(), chunk_stream, mr),
        chunk_stream)));
  }
  co_await ch_out->drain(ctx->executor());
}

rapidsmpf::streaming::Actor select_columns(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                           std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
                                           std::shared_ptr<rapidsmpf::streaming::Channel> ch_out)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
  // n_name, ps_supplycost, l_discount, l_extendedprice, l_quantity,  o_orderdate

  // Select n_name, year_part_of(o_orderdate), amount = (extendedprice * (1
  // - discount)) - (ps_supplycost * l_quantity) group by n_name year agg
  // sum(amount).round(2) sort by n_name, o_year descending = true, false
  while (true) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    co_await ctx->executor()->schedule();
    auto chunk           = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    auto chunk_stream    = chunk.stream();
    auto sequence_number = msg.sequence_number();
    auto table           = chunk.table_view();
    std::vector<std::unique_ptr<cudf::column>> result;
    result.reserve(3);
    // n_name
    result.push_back(
      std::make_unique<cudf::column>(table.column(0), chunk_stream, ctx->br()->device_mr()));
    result.push_back(
      cudf::datetime::extract_datetime_component(table.column(5),
                                                 cudf::datetime::datetime_component::YEAR,
                                                 chunk_stream,
                                                 ctx->br()->device_mr()));
    auto discount      = table.column(2);
    auto extendedprice = table.column(3);
    auto supplycost    = table.column(1);
    auto quantity      = table.column(4);
    std::string udf =
      R"***(
static __device__ void calculate_amount(double *amount, double discount, double extprice, double supplycost, double quantity) {
    *amount = extprice * (1 - discount) - supplycost * quantity;
}
           )***";
    result.push_back(
      std::move(cudf::transform(
                  udf,
                  cudf::udf_source_type::CUDA,
                  cudf::null_aware::NO,
                  std::nullopt,
                  std::vector<cudf::transform_input>{discount, extendedprice, supplycost, quantity},
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

std::vector<rapidsmpf::ndsh::groupby_request> chunkwise_groupby_requests()
{
  auto requests = std::vector<rapidsmpf::ndsh::groupby_request>();
  std::vector<std::function<std::unique_ptr<cudf::groupby_aggregation>()>> aggs;
  aggs.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>);
  requests.emplace_back(2, std::move(aggs));
  return requests;
}

rapidsmpf::streaming::Actor round_sum_profit(std::shared_ptr<rapidsmpf::streaming::Context> ctx,
                                             std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
                                             std::shared_ptr<rapidsmpf::streaming::Channel> ch_out)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
  co_await ctx->executor()->schedule();
  auto msg = co_await ch_in->receive();
  RAPIDSMPF_EXPECTS(!msg.empty(), "Expecting to see a single chunk");
  auto next = co_await ch_in->receive();
  RAPIDSMPF_EXPECTS(next.empty(), "Not expecting to see a second chunk");
  auto chunk = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
  auto table = chunk.table_view();
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  // cudf::round_decimal does not support float types
  auto rounded = cudf::round(
    table.column(2), 2, cudf::rounding_method::HALF_EVEN, chunk.stream(), ctx->br()->device_mr());
#pragma GCC diagnostic pop
  auto result = cudf_streaming::to_message(
    0,
    std::make_unique<cudf_streaming::table_chunk>(
      std::make_unique<cudf::table>(
        cudf::table_view({table.column(0), table.column(1), rounded->view()}),
        chunk.stream(),
        ctx->br()->device_mr()),
      chunk.stream()));
  co_await ch_out->send(std::move(result));
  co_await ch_out->drain(ctx->executor());
}

}  // namespace

/**
 * @brief Run a derived version of TPC-H query 9.
 *
 * The SQL form of the query is:
 * @code{.sql}
 * select
 *     nation,
 *     o_year,
 *     round(sum(amount), 2) as sum_profit
 * from
 *     (
 *         select
 *             n_name as nation,
 *             year(o_orderdate) as o_year,
 *             l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
 *         from
 *             part,
 *             supplier,
 *             lineitem,
 *             partsupp,
 *             orders,
 *             nation
 *         where
 *             s_suppkey = l_suppkey
 *             and ps_suppkey = l_suppkey
 *             and ps_partkey = l_partkey
 *             and p_partkey = l_partkey
 *             and o_orderkey = l_orderkey
 *             and s_nationkey = n_nationkey
 *             and p_name like '%green%'
 *     ) as profit
 * group by
 *     nation,
 *     o_year
 * order by
 *     nation,
 *     o_year desc
 * @endcode{}
 */
int main(int argc, char** argv)
{
  rapidsmpf::ndsh::FinalizeMPI finalize{};
  CUDF_CUDA_TRY(cudaFree(nullptr));
  auto mr                 = rmm::mr::cuda_async_memory_resource{};
  auto arguments          = rapidsmpf::ndsh::parse_arguments(argc, argv);
  auto [ctx, comm]        = rapidsmpf::ndsh::create_context(arguments, std::move(mr));
  std::string output_path = arguments.output_file;
  std::vector<double> timings;
  for (int i = 0; i < arguments.num_iterations; i++) {
    rapidsmpf::OpID op_id{0};
    std::vector<rapidsmpf::streaming::Actor> actors;
    auto start = std::chrono::steady_clock::now();
    {
      RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing Q9 pipeline");
      auto part                                  = ctx->create_channel();
      auto filtered_part                         = ctx->create_channel();
      auto partsupp                              = ctx->create_channel();
      auto part_x_partsupp                       = ctx->create_channel();
      auto supplier                              = ctx->create_channel();
      auto lineitem                              = ctx->create_channel();
      auto supplier_x_part_x_partsupp            = ctx->create_channel();
      auto supplier_x_part_x_partsupp_x_lineitem = ctx->create_channel();
      actors.push_back(read_part(ctx,
                                 comm,
                                 part,
                                 /* num_tickets */ 4,
                                 arguments.num_rows_per_chunk,
                                 arguments.input_directory));   // p_partkey, p_name
      actors.push_back(filter_part(ctx, part, filtered_part));  // p_partkey
      actors.push_back(
        read_partsupp(ctx,
                      comm,
                      partsupp,
                      /* num_tickets */ 4,
                      arguments.num_rows_per_chunk,
                      arguments.input_directory));  // ps_partkey, ps_suppkey, ps_supplycost
      actors.push_back(
        // p_partkey x ps_partkey
        rapidsmpf::ndsh::inner_join_broadcast(
          ctx,
          comm,
          filtered_part,
          partsupp,
          part_x_partsupp,
          {0},
          {0},
          rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(
            10 * i + op_id++)})  // p_partkey/ps_partkey, ps_suppkey, ps_supplycost
      );
      actors.push_back(read_supplier(ctx,
                                     comm,
                                     supplier,
                                     /* num_tickets */ 4,
                                     arguments.num_rows_per_chunk,
                                     arguments.input_directory));  // s_nationkey, s_suppkey
      actors.push_back(
        // s_suppkey x ps_suppkey
        rapidsmpf::ndsh::inner_join_broadcast(
          ctx,
          comm,
          supplier,
          part_x_partsupp,
          supplier_x_part_x_partsupp,
          {1},
          {1},
          rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}

          )  // s_nationkey, s_suppkey/ps_suppkey, p_partkey/ps_partkey,
             // ps_supplycost
      );
      actors.push_back(read_lineitem(
        ctx,
        comm,
        lineitem,
        /* num_tickets */ 4,
        arguments.num_rows_per_chunk,
        arguments
          .input_directory));  // l_discount, l_extendedprice, l_orderkey, l_partkey, l_quantity,
      // l_suppkey
      actors.push_back(
        // [p_partkey, ps_suppkey] x [l_partkey, l_suppkey]
        rapidsmpf::ndsh::inner_join_broadcast(
          ctx,
          comm,
          supplier_x_part_x_partsupp,
          lineitem,
          supplier_x_part_x_partsupp_x_lineitem,
          {2, 1},
          {3, 5},
          rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)},
          rapidsmpf::ndsh::KeepKeys::NO)  // s_nationkey, ps_supplycost,
                                          // l_discount, l_extendedprice, l_orderkey, l_quantity
      );
      auto nation = ctx->create_channel();
      auto orders = ctx->create_channel();
      actors.push_back(read_nation(ctx,
                                   comm,
                                   nation,
                                   /* num_tickets */ 4,
                                   arguments.num_rows_per_chunk,
                                   arguments.input_directory)  // n_name, n_nationkey
      );
      actors.push_back(read_orders(ctx,
                                   comm,
                                   orders,
                                   /* num_tickets */ 4,
                                   arguments.num_rows_per_chunk,
                                   arguments.input_directory)  // o_orderdate, o_orderkey
      );
      auto all_joined                                     = ctx->create_channel();
      auto supplier_x_part_x_partsupp_x_lineitem_x_orders = ctx->create_channel();
      if (arguments.use_shuffle_join) {
        auto supplier_x_part_x_partsupp_x_lineitem_shuffled = ctx->create_channel();
        auto orders_shuffled                                = ctx->create_channel();
        // TODO: customisable
        std::uint32_t num_partitions = 16;
        actors.push_back(rapidsmpf::ndsh::shuffle(
          ctx,
          comm,
          supplier_x_part_x_partsupp_x_lineitem,
          supplier_x_part_x_partsupp_x_lineitem_shuffled,
          {4},
          num_partitions,
          rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}));
        actors.push_back(rapidsmpf::ndsh::shuffle(
          ctx,
          comm,
          orders,
          orders_shuffled,
          {1},
          num_partitions,
          rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)}));
        actors.push_back(
          // l_orderkey x o_orderkey
          rapidsmpf::ndsh::inner_join_shuffle(
            ctx,
            comm,
            supplier_x_part_x_partsupp_x_lineitem_shuffled,
            orders_shuffled,
            supplier_x_part_x_partsupp_x_lineitem_x_orders,
            {4},
            {1},
            rapidsmpf::ndsh::KeepKeys::NO)  // s_nationkey, ps_supplycost, l_discount,
                                            // l_extendedprice, l_quantity, o_orderdate
        );
      } else {
        actors.push_back(
          // l_orderkey x o_orderkey
          rapidsmpf::ndsh::inner_join_broadcast(
            ctx,
            comm,
            supplier_x_part_x_partsupp_x_lineitem,
            orders,
            supplier_x_part_x_partsupp_x_lineitem_x_orders,
            {4},
            {1},
            rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)},
            rapidsmpf::ndsh::KeepKeys::NO)  // s_nationkey, ps_supplycost, l_discount,
                                            // l_extendedprice, l_quantity, o_orderdate
        );
      }
      actors.push_back(
        // n_nationkey x s_nationkey
        rapidsmpf::ndsh::inner_join_broadcast(
          ctx,
          comm,
          nation,
          supplier_x_part_x_partsupp_x_lineitem_x_orders,
          all_joined,
          {1},
          {0},
          rapidsmpf::OpID{static_cast<rapidsmpf::OpID>(10 * i + op_id++)},
          rapidsmpf::ndsh::KeepKeys::NO)  // n_name, ps_supplycost, l_discount, l_extendedprice,
                                          // l_quantity, o_orderdate
      );
      auto chunkwise_groupby_input = ctx->create_channel();
      actors.push_back(select_columns(ctx, all_joined, chunkwise_groupby_input));
      auto chunkwise_groupby_output = ctx->create_channel();
      actors.push_back(rapidsmpf::ndsh::chunkwise_group_by(ctx,
                                                           chunkwise_groupby_input,
                                                           chunkwise_groupby_output,
                                                           {0, 1},
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
        actors.push_back(rapidsmpf::ndsh::chunkwise_group_by(ctx,
                                                             final_groupby_input,
                                                             final_groupby_output,
                                                             {0, 1},
                                                             chunkwise_groupby_requests(),
                                                             cudf::null_policy::INCLUDE));
        auto sorted_input = ctx->create_channel();
        actors.push_back(round_sum_profit(ctx, final_groupby_output, sorted_input));
        auto sorted_output = ctx->create_channel();
        actors.push_back(
          rapidsmpf::ndsh::chunkwise_sort_by(ctx,
                                             sorted_input,
                                             sorted_output,
                                             {0, 1},
                                             {0, 1, 2},
                                             {cudf::order::ASCENDING, cudf::order::DESCENDING},
                                             {cudf::null_order::BEFORE, cudf::null_order::BEFORE}));
        actors.push_back(rapidsmpf::ndsh::write_parquet(ctx,
                                                        sorted_output,
                                                        cudf::io::sink_info{output_path},
                                                        {"nation", "o_year", "sum_profit"}));
      } else {
        actors.push_back(rapidsmpf::ndsh::sink_channel(ctx, final_groupby_input));
      }
    }
    auto end                               = std::chrono::steady_clock::now();
    std::chrono::duration<double> pipeline = end - start;
    start                                  = std::chrono::steady_clock::now();
    {
      RAPIDSMPF_NVTX_SCOPED_RANGE("Q9 Iteration");
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

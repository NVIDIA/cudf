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
#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/context.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/timestamps.hpp>

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
#include <rapidsmpf/utils/misc.hpp>

#include <array>
#include <chrono>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>

namespace {

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
                     "l_returnflag",     // 0
                     "l_linestatus",     // 1
                     "l_quantity",       // 2
                     "l_extendedprice",  // 3
                     "l_discount",       // 4
                     "l_tax"             // 5
                   })
                   .build();
  auto stream = ctx->br()->stream_pool()->get_stream();
  // l_shipdate <= DATE '1998-09-02'
  constexpr auto date = cuda::std::chrono::year_month_day(
    cuda::std::chrono::year(1998), cuda::std::chrono::month(9), cuda::std::chrono::day(2));
  auto filter_expr = use_date32
                       ? rapidsmpf::ndsh::make_date_filter<cudf::timestamp_D>(
                           stream, date, "l_shipdate", cudf::ast::ast_operator::LESS_EQUAL)
                       : rapidsmpf::ndsh::make_date_filter<cudf::timestamp_ms>(
                           stream, date, "l_shipdate", cudf::ast::ast_operator::LESS_EQUAL);
  return cudf_streaming::actor::read_parquet(
    ctx, comm, ch_out, num_producers, options, num_rows_per_chunk, std::move(filter_expr));
}

std::vector<rapidsmpf::ndsh::groupby_request> chunkwise_groupby_requests()
{
  auto requests = std::vector<rapidsmpf::ndsh::groupby_request>();
  std::vector<std::function<std::unique_ptr<cudf::groupby_aggregation>()>> aggs;
  // sum(l_quantity), sum(l_extendedprice), sum(disc_price), sum(charge),
  // sum(l_discount)
  for (cudf::size_type idx = 2; idx < 7; idx++) {
    aggs.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>);
    requests.emplace_back(idx, std::move(aggs));
  }
  // count(*)
  aggs.emplace_back([]() {
    return cudf::make_count_aggregation<cudf::groupby_aggregation>(cudf::null_policy::INCLUDE);
  });
  requests.emplace_back(0, std::move(aggs));
  return requests;
}

std::vector<rapidsmpf::ndsh::groupby_request> final_groupby_requests()
{
  auto requests = std::vector<rapidsmpf::ndsh::groupby_request>();
  std::vector<std::function<std::unique_ptr<cudf::groupby_aggregation>()>> aggs;
  // sum(l_quantity), sum(l_extendedprice), sum(disc_price), sum(charge),
  // sum(l_discount), sum(count(*))
  for (cudf::size_type idx = 2; idx < 8; idx++) {
    aggs.emplace_back(cudf::make_sum_aggregation<cudf::groupby_aggregation>);
    requests.emplace_back(idx, std::move(aggs));
  }
  return requests;
}

rapidsmpf::streaming::Actor postprocess_group_by(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};
  co_await ctx->executor()->schedule();
  auto msg = co_await ch_in->receive();
  RAPIDSMPF_EXPECTS((co_await ch_in->receive()).empty(),
                    "Expecting concatenated input at this point");
  auto chunk   = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
  auto stream  = chunk.stream();
  auto columns = cudf::table{chunk.table_view(), stream, ctx->br()->device_mr()}.release();
  std::ignore  = std::move(chunk);
  auto count   = std::move(columns.back());
  columns.pop_back();
  auto discount = std::move(columns.back());
  columns.pop_back();
  for (std::size_t i = 2; i < 4; i++) {
    columns.push_back(cudf::binary_operation(columns[i]->view(),
                                             count->view(),
                                             cudf::binary_operator::TRUE_DIV,
                                             cudf::data_type(cudf::type_id::FLOAT64),
                                             stream,
                                             ctx->br()->device_mr()));
  }
  columns.push_back(cudf::binary_operation(discount->view(),
                                           count->view(),
                                           cudf::binary_operator::TRUE_DIV,
                                           cudf::data_type(cudf::type_id::FLOAT64),
                                           stream,
                                           ctx->br()->device_mr()));
  columns.push_back(std::move(count));
  co_await ch_out->send(
    cudf_streaming::to_message(msg.sequence_number(),
                               std::make_unique<cudf_streaming::table_chunk>(
                                 std::make_unique<cudf::table>(std::move(columns)), stream)));
  co_await ch_out->drain(ctx->executor());
}

// In: l_returnflag, l_linestatus, l_quantity, l_extendedprice,
// l_discount, l_tax
// Out: l_returnflag, l_linestatus, l_quantity, l_extendedprice,
// disc_price = (l_extendedprice * (1 - l_discount)),
// charge = (l_extendedprice * (1 - l_discount) * (1 + l_tax)),
// l_discount
rapidsmpf::streaming::Actor select_columns_for_groupby(
  std::shared_ptr<rapidsmpf::streaming::Context> ctx,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_in,
  std::shared_ptr<rapidsmpf::streaming::Channel> ch_out)
{
  rapidsmpf::streaming::ShutdownAtExit c{ch_in, ch_out};

  co_await ctx->executor()->schedule();
  while (!ch_out->is_shutdown()) {
    auto msg = co_await ch_in->receive();
    if (msg.empty()) { break; }
    auto chunk           = co_await msg.release<cudf_streaming::table_chunk>().make_available(ctx);
    auto chunk_stream    = chunk.stream();
    auto sequence_number = msg.sequence_number();
    auto table           = chunk.table_view();
    // l_returnflag, l_linestatus, l_quantity, l_extendedprice
    auto result =
      cudf::table(table.select({0, 1, 2, 3}), chunk_stream, ctx->br()->device_mr()).release();
    result.reserve(7);
    auto extendedprice = table.column(3);
    auto discount      = table.column(4);
    auto tax           = table.column(5);
    std::string udf_disc_price =
      R"***(
static __device__ void calculate_disc_price(double *disc_price, double extprice, double discount) {
    *disc_price = extprice * (1 - discount);
}
           )***";
    std::string udf_charge =
      R"***(
static __device__ void calculate_charge(double *charge, double discprice, double tax) {
    *charge = discprice * (1 + tax);
}
           )***";

    // disc_price
    result.push_back(std::move(
      cudf::transform(udf_disc_price,
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
    // charge
    result.push_back(std::move(
      cudf::transform(udf_charge,
                      cudf::udf_source_type::CUDA,
                      cudf::null_aware::NO,
                      std::nullopt,
                      std::vector<cudf::transform_input>{result.back()->view(), tax},
                      std::array{cudf::transform_output{cudf::data_type(cudf::type_id::FLOAT64),
                                                        cudf::output_nullability::PRESERVE}},
                      {},
                      std::nullopt,
                      chunk_stream,
                      ctx->br()->device_mr())
        ->release()
        .front()));
    // l_discount
    result.push_back(
      std::make_unique<cudf::column>(discount, chunk_stream, ctx->br()->device_mr()));
    co_await ch_out->send(cudf_streaming::to_message(
      sequence_number,
      std::make_unique<cudf_streaming::table_chunk>(
        std::make_unique<cudf::table>(std::move(result)), chunk_stream)));
  }
  co_await ch_out->drain(ctx->executor());
}
}  // namespace

/**
 * @brief Run a derived version of TPC-H query 1.
 *
 * The SQL form of the query is:
 * @code{.sql}
 * select
 *     l_returnflag,
 *     l_linestatus,
 *     sum(l_quantity) as sum_qty,
 *     sum(l_extendedprice) as sum_base_price,
 *     sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
 *     sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
 *     avg(l_quantity) as avg_qty,
 *     avg(l_extendedprice) as avg_price,
 *     avg(l_discount) as avg_disc,
 *     count(*) as count_order
 * from
 *     lineitem
 * where
 *     l_shipdate <= DATE '1998-09-02'
 * group by
 *     l_returnflag,
 *     l_linestatus
 * order by
 *     l_returnflag,
 *     l_linestatus
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

  // Detect date column type from parquet metadata before timed section
  auto const column_types =
    rapidsmpf::ndsh::detail::get_column_types(arguments.input_directory, "lineitem");
  bool const use_date32 = column_types.at("l_shipdate").id() == cudf::type_id::TIMESTAMP_DAYS;

  std::vector<double> timings;
  for (int i = 0; i < arguments.num_iterations; i++) {
    int op_id = 0;
    std::vector<rapidsmpf::streaming::Actor> actors;
    auto start = std::chrono::steady_clock::now();
    {
      RAPIDSMPF_NVTX_SCOPED_RANGE("Constructing Q1 pipeline");

      // Input data channels
      auto lineitem = ctx->create_channel();
      // Out: l_returnflag, l_linestatus, l_quantity, l_extendedprice,
      // l_discount, l_tax
      actors.push_back(read_lineitem(ctx,
                                     comm,
                                     lineitem,
                                     /* num_tickets */ 4,
                                     arguments.num_rows_per_chunk,
                                     arguments.input_directory,
                                     use_date32));

      auto chunkwise_groupby_input = ctx->create_channel();
      // Out: l_returnflag, l_linestatus, l_quantity, l_extendedprice,
      // disc_price = (l_extendedprice * (1 - l_discount)),
      // charge = (l_extendedprice * (1 - l_discount) * (1 + l_tax))
      // l_discount
      actors.push_back(select_columns_for_groupby(ctx, lineitem, chunkwise_groupby_input));
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
                                                             final_groupby_requests(),
                                                             cudf::null_policy::INCLUDE));
        auto sorted_input = ctx->create_channel();
        actors.push_back(postprocess_group_by(ctx, final_groupby_output, sorted_input));
        auto sorted_output = ctx->create_channel();
        actors.push_back(
          rapidsmpf::ndsh::chunkwise_sort_by(ctx,
                                             sorted_input,
                                             sorted_output,
                                             {0, 1},
                                             {0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
                                             {cudf::order::ASCENDING, cudf::order::ASCENDING},
                                             {cudf::null_order::BEFORE, cudf::null_order::BEFORE}));
        actors.push_back(rapidsmpf::ndsh::write_parquet(ctx,
                                                        sorted_output,
                                                        cudf::io::sink_info(output_path),
                                                        {"l_returnflag",
                                                         "l_linestatus",
                                                         "sum_qty",
                                                         "sum_base_price",
                                                         "sum_disc_price",
                                                         "sum_charge",
                                                         "avg_qty",
                                                         "avg_price",
                                                         "avg_disc",
                                                         "count_order"}

                                                        ));
      } else {
        actors.push_back(rapidsmpf::ndsh::sink_channel(ctx, final_groupby_input));
      }
    }
    auto end                               = std::chrono::steady_clock::now();
    std::chrono::duration<double> pipeline = end - start;
    start                                  = std::chrono::steady_clock::now();
    {
      RAPIDSMPF_NVTX_SCOPED_RANGE("Q1 Iteration");
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

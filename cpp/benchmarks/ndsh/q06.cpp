/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q06.cpp
 * @brief Implement query 6 of the NDS-H benchmark.
 *
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 *
 * select
 *    sum(l_extendedprice * l_discount) as revenue
 * from
 *    lineitem
 * where
 *    l_shipdate >= date '1994-01-01'
 *    and l_shipdate < date '1995-01-01'
 *    and l_discount >= 0.05
 *    and l_discount <= 0.07
 *    and l_quantity < 24;
 */

/**
 * @brief Calculate the revenue column
 *
 * @param extendedprice The extended price column
 * @param discount The discount column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> calculate_revenue(
  cudf::column_view const& extendedprice,
  cudf::column_view const& discount,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto const revenue_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto revenue            = cudf::binary_operation(
    extendedprice, discount, cudf::binary_operator::MUL, revenue_type, stream, mr);
  return revenue;
}

void run_ndsh_q6(nvbench::state& state,
                 std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  // Read out the `lineitem` table from parquet file
  std::vector<std::string> const lineitem_cols = {
    "l_extendedprice", "l_discount", "l_shipdate", "l_quantity"};
  auto const shipdate_ref = cudf::ast::column_reference(std::distance(
    lineitem_cols.begin(), std::find(lineitem_cols.begin(), lineitem_cols.end(), "l_shipdate")));
  auto shipdate_lower =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1994, 1, 1), true);
  auto const shipdate_lower_literal = cudf::ast::literal(shipdate_lower);
  auto shipdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1995, 1, 1), true);
  auto const shipdate_upper_literal = cudf::ast::literal(shipdate_upper);
  auto const shipdate_pred_a        = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, shipdate_ref, shipdate_lower_literal);
  auto const shipdate_pred_b =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, shipdate_ref, shipdate_upper_literal);
  auto const lineitem_pred = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LOGICAL_AND, shipdate_pred_a, shipdate_pred_b);
  auto lineitem = read_parquet(
    sources.at("lineitem").make_source_info(), lineitem_cols, std::move(lineitem_pred));

  // Cast the discount and quantity columns to float32 and append to lineitem table
  auto discout_float =
    cudf::cast(lineitem->column("l_discount"), cudf::data_type{cudf::type_id::FLOAT32});
  auto quantity_float =
    cudf::cast(lineitem->column("l_quantity"), cudf::data_type{cudf::type_id::FLOAT32});

  (*lineitem).append(discout_float, "l_discount_float").append(quantity_float, "l_quantity_float");

  // Apply the filters
  auto const discount_ref = cudf::ast::column_reference(lineitem->column_id("l_discount_float"));
  auto const quantity_ref = cudf::ast::column_reference(lineitem->column_id("l_quantity_float"));

  auto discount_lower               = cudf::numeric_scalar<float_t>(0.05);
  auto const discount_lower_literal = cudf::ast::literal(discount_lower);
  auto discount_upper               = cudf::numeric_scalar<float_t>(0.07);
  auto const discount_upper_literal = cudf::ast::literal(discount_upper);
  auto quantity_upper               = cudf::numeric_scalar<float_t>(24);
  auto const quantity_upper_literal = cudf::ast::literal(quantity_upper);

  auto const discount_pred_a = cudf::ast::operation(
    cudf::ast::ast_operator::GREATER_EQUAL, discount_ref, discount_lower_literal);

  auto const discount_pred_b =
    cudf::ast::operation(cudf::ast::ast_operator::LESS_EQUAL, discount_ref, discount_upper_literal);
  auto const discount_pred =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, discount_pred_a, discount_pred_b);
  auto const quantity_pred =
    cudf::ast::operation(cudf::ast::ast_operator::LESS, quantity_ref, quantity_upper_literal);
  auto const discount_quantity_pred =
    cudf::ast::operation(cudf::ast::ast_operator::LOGICAL_AND, discount_pred, quantity_pred);
  auto const filtered_table = apply_filter(lineitem, discount_quantity_pred);

  // Calculate the `revenue` column
  auto revenue = calculate_revenue(filtered_table->column("l_extendedprice"),
                                   filtered_table->column("l_discount"));

  // Sum the `revenue` column
  auto const revenue_view = revenue->view();
  auto const result_table = apply_reduction(revenue_view, cudf::aggregation::Kind::SUM, "revenue");

  // Write query result to a parquet file
  result_table->to_parquet("q6.parquet");
}

void ndsh_q6(nvbench::state& state)
{
  // Generate the required parquet files in device buffers
  double const scale_factor = state.get_float64("scale_factor");
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"lineitem"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { run_ndsh_q6(state, sources); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q6).set_name("ndsh_q6").add_float64_axis("scale_factor", {0.01, 0.1, 1});

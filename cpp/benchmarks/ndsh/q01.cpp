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
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

/**
 * @file q01.cpp
 * @brief Implement query 1 of the NDS-H benchmark.
 *
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 *
 * select
 *    l_returnflag,
 *    l_linestatus,
 *    sum(l_quantity) as sum_qty,
 *    sum(l_extendedprice) as sum_base_price,
 *    sum(l_extendedprice * (1 - l_discount)) as sum_disc_price,
 *    sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) as sum_charge,
 *    avg(l_quantity) as avg_qty,
 *    avg(l_extendedprice) as avg_price,
 *    avg(l_discount) as avg_disc,
 *    count(*) as count_order
 * from
 *    lineitem
 * where
 *    l_shipdate <= date '1998-09-02'
 * group by
 *    l_returnflag,
 *    l_linestatus
 * order by
 *    l_returnflag,
 *    l_linestatus;
 */

/**
 * @brief Calculate the discount price column
 *
 * @param discount The discount column
 * @param extendedprice The extended price column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> calculate_disc_price(
  cudf::column_view const& discount,
  cudf::column_view const& extendedprice,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto const one = discount.type().id() == cudf::type_id::DECIMAL64
                     ? cudf::make_fixed_point_scalar<numeric::decimal64>(1L, numeric::scale_type{0})
                     : cudf::make_fixed_width_scalar<double>(1);
  auto const one_minus_discount =
    cudf::binary_operation(*one, discount, cudf::binary_operator::SUB, discount.type(), stream, mr);
  return cudf::binary_operation(extendedprice,
                                one_minus_discount->view(),
                                cudf::binary_operator::MUL,
                                discount.type(),
                                stream,
                                mr);
}

/**
 * @brief Calculate the charge column
 *
 * @param tax The tax column
 * @param disc_price The discount price column
 * @param stream The CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 */
[[nodiscard]] std::unique_ptr<cudf::column> calculate_charge(
  cudf::column_view const& tax,
  cudf::column_view const& disc_price,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto const one = tax.type().id() == cudf::type_id::DECIMAL64
                     ? cudf::make_fixed_point_scalar<numeric::decimal64>(1L, numeric::scale_type{0})
                     : cudf::make_fixed_width_scalar<double>(1);
  auto const one_plus_tax =
    cudf::binary_operation(*one, tax, cudf::binary_operator::ADD, tax.type(), stream, mr);
  return cudf::binary_operation(
    disc_price, one_plus_tax->view(), cudf::binary_operator::MUL, tax.type(), stream, mr);
}

void run_ndsh_q1(nvbench::state& state, cudf::io::source_info const& source)
{
  // Define the column projections and filter predicate for `lineitem` table
  std::vector<std::string> const lineitem_cols = {"l_returnflag",
                                                  "l_linestatus",
                                                  "l_quantity",
                                                  "l_extendedprice",
                                                  "l_discount",
                                                  "l_shipdate",
                                                  "l_orderkey",
                                                  "l_tax"};
  auto const shipdate_ref                      = cudf::ast::column_reference(std::distance(
    lineitem_cols.begin(), std::find(lineitem_cols.begin(), lineitem_cols.end(), "l_shipdate")));
  auto shipdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1998, 9, 2), true);
  auto const shipdate_upper_literal = cudf::ast::literal(shipdate_upper);
  auto const lineitem_pred          = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LESS_EQUAL, shipdate_ref, shipdate_upper_literal);

  // Read out the `lineitem` table from parquet file
  auto lineitem = read_parquet(source, lineitem_cols, std::move(lineitem_pred));

  // Calculate the discount price and charge columns and append to lineitem table
  auto disc_price =
    calculate_disc_price(lineitem->column("l_discount"), lineitem->column("l_extendedprice"));
  auto charge = calculate_charge(lineitem->column("l_tax"), disc_price->view());
  (*lineitem).append(disc_price, "disc_price").append(charge, "charge");

  // Perform the group by operation
  auto const groupedby_table = apply_groupby(
    lineitem,
    groupby_context_t{
      {"l_returnflag", "l_linestatus"},
      {
        {"l_extendedprice",
         {{cudf::aggregation::Kind::SUM, "sum_base_price"},
          {cudf::aggregation::Kind::MEAN, "avg_price"}}},
        {"l_quantity",
         {{cudf::aggregation::Kind::SUM, "sum_qty"}, {cudf::aggregation::Kind::MEAN, "avg_qty"}}},
        {"l_discount",
         {
           {cudf::aggregation::Kind::MEAN, "avg_disc"},
         }},
        {"disc_price",
         {
           {cudf::aggregation::Kind::SUM, "sum_disc_price"},
         }},
        {"charge",
         {{cudf::aggregation::Kind::SUM, "sum_charge"},
          {cudf::aggregation::Kind::COUNT_ALL, "count_order"}}},
      }});

  // Perform the order by operation
  auto const orderedby_table = apply_orderby(groupedby_table,
                                             {"l_returnflag", "l_linestatus"},
                                             {cudf::order::ASCENDING, cudf::order::ASCENDING});

  // Write query result to a parquet file
  orderedby_table->to_parquet("q1.parquet");
}

void ndsh_q1(nvbench::state& state)
{
  // Generate the required parquet files in device buffers
  auto const scale_factor = state.get_float64("scale_factor");
  auto const filename     = state.get_string("filename");
  if (!filename.empty() && scale_factor != 1.0) {
    state.skip("Only scale_factor=1 supported with filename input");
    return;
  }
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  auto source = [&] {
    if (filename.empty()) {
      generate_parquet_data_sources(scale_factor, {"lineitem"}, sources);
      return sources.at("lineitem").make_source_info();
    }
    return cudf::io::source_info(filename);
  }();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) { run_ndsh_q1(state, source); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q1)
  .set_name("ndsh_q1")
  .add_string_axis("filename", {""})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});

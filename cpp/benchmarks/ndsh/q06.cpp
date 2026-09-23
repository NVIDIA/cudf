/*
 * SPDX-FileCopyrightText: Copyright the Vortex contributors
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet/parquet_io.hpp"
#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

#ifdef CUDF_WITH_VORTEX
#include "local_io.hpp"
#include "parquet/parquet_fixture.hpp"
#include "reference/q6_reference.hpp"
#include "vortex/vortex_io.hpp"

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cmath>
#include <map>
#include <optional>
#endif

namespace {
std::vector<std::string> const q6_columns{
  "l_extendedprice", "l_discount", "l_shipdate", "l_quantity"};
}

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
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  auto const revenue_type = cudf::data_type{cudf::type_id::FLOAT64};
  auto revenue            = cudf::binary_operation(
    extendedprice, discount, cudf::binary_operator::MUL, revenue_type, stream, mr);
  return revenue;
}

/**
 * read returns an owning projected table, applying its predicate if filter_shipdate is false.
 * Otherwise filtering happens here. consume receives the result owner by reference and may
 * move it out; its return value is forwarded. This helper adds no final stream synchronization.
 */
template <typename Read, typename Consume>
auto execute_q6(Read&& read, bool filter_shipdate, Consume&& consume)
{
  auto const shipdate_ref = cudf::ast::column_reference(std::distance(
    q6_columns.begin(), std::find(q6_columns.begin(), q6_columns.end(), "l_shipdate")));
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
  auto lineitem = read(q6_columns, lineitem_pred);
  if (filter_shipdate) { lineitem = apply_filter(lineitem, *lineitem_pred); }

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
  auto result_table       = apply_reduction(revenue_view, cudf::aggregation::Kind::SUM, "revenue");
  return consume(result_table);
}

void run_ndsh_q6(nvbench::state& state,
                 std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  execute_q6(
    [&](auto const& columns, auto const& predicate) {
      return read_parquet(sources.at("lineitem").make_source_info(), columns, predicate);
    },
    false,
    [](auto const& result) { write_parquet(*result, "q6.parquet"); });
}

void ndsh_q6(nvbench::state& state)
{
  // Generate the required parquet files in device buffers
  double const scale_factor = state.get_float64("scale_factor");
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(scale_factor, {"lineitem"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { run_ndsh_q6(state, sources); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q6).set_name("ndsh_q6").add_float64_axis("scale_factor", {0.01, 0.1, 1});

#ifdef CUDF_WITH_VORTEX
namespace {

std::optional<double> revenue_value(table_with_names const& result)
{
  CUDF_EXPECTS(result.table().num_rows() == 1 && result.table().num_columns() == 1 &&
                 result.column_names() == std::vector<std::string>{"revenue"},
               "Q6 result must have one row and one revenue column");
  CUDF_EXPECTS(result.table().column(0).type() == cudf::data_type{cudf::type_id::FLOAT64},
               "Q6 revenue must be FLOAT64");
  auto value = cudf::get_element(result.column("revenue"), 0);
  if (!value->is_valid()) { return std::nullopt; }
  return static_cast<cudf::numeric_scalar<double> const&>(*value).value();
}

void check_q6_result(ndsh::q6_reference_result const& expected, table_with_names const& result)
{
  auto const value = revenue_value(result);
  if (expected.matched == 0) {
    CUDF_EXPECTS(!value, "Q6 SUM over no matching rows must be null");
  } else {
    CUDF_EXPECTS(value && ndsh::detail::reference_equal(*value, expected.revenue),
                 "Q6 GPU revenue differs from CPU reference");
  }
}

void check_q6_result_schema()
{
  cudf::test::fixed_width_column_wrapper<double> revenue{{18.0}};
  cudf::test::fixed_width_column_wrapper<double> extra_rows{{18.0, 999.0}};
  cudf::test::fixed_width_column_wrapper<double> null_rows{{0.0, 999.0}, {false, true}};
  cudf::test::fixed_width_column_wrapper<int64_t> wrong_type{{18}};
  auto reject = [](cudf::table_view view, std::vector<std::string> names) {
    table_with_names result{std::make_unique<cudf::table>(view), std::move(names)};
    bool rejected = false;
    try {
      (void)revenue_value(result);
    } catch (cudf::logic_error const&) {
      rejected = true;
    }
    CUDF_EXPECTS(rejected, "Q6 validator accepted an invalid result schema");
  };
  reject(cudf::table_view{{extra_rows}}, {"revenue"});
  reject(cudf::table_view{{null_rows}}, {"revenue"});
  reject(cudf::table_view{{revenue, revenue}}, {"revenue", "extra"});
  reject(cudf::table_view{{wrong_type}}, {"revenue"});
  reject(cudf::table_view{{revenue}}, {"wrong_name"});
  reject(cudf::slice(cudf::table_view{{revenue}}, {0, 0}).front(), {"revenue"});

  table_with_names valid{std::make_unique<cudf::table>(cudf::table_view{{revenue}}), {"revenue"}};
  check_q6_result({1, 18.0}, valid);
  table_with_names empty_sum{
    std::make_unique<cudf::table>(cudf::slice(cudf::table_view{{null_rows}}, {0, 1}).front()),
    {"revenue"}};
  check_q6_result({}, empty_sum);
}

void check_q6_reference_boundaries()
{
  using cudf::test::fixed_width_column_wrapper;
  cuda::stream_ref const stream = cudf::get_default_stream();
  // Exclude a matching sentinel; near-boundary doubles qualify only after the float32 cast.
  fixed_width_column_wrapper<double> price{{1000, 100, 100, 100, 100, 100, 100, 100, 100}};
  fixed_width_column_wrapper<double> discount{
    {0.06, 0.06, 0.05 - 1e-10, 0.07 + 1e-10, 0.06, 0.06, 0.049, 0.071, 0.06}};
  fixed_width_column_wrapper<cudf::timestamp_D, int32_t> shipdate{
    {8766, 8766, 8766, 9130, 8765, 9131, 8766, 8766, 8766}};
  fixed_width_column_wrapper<int8_t> quantity{{23, 23, 23, 23, 23, 23, 23, 23, 24}};
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  auto const input = cudf::table_view{{price, discount, shipdate, quantity}};
  for (auto const& projected : cudf::slice(input, {1, 9, 4, 6, 1, 1}, stream)) {
    auto const expected =
      projected.num_rows() == 8 ? ndsh::q6_reference_result{3, 18.0} : ndsh::q6_reference_result{};
    auto const cpu = ndsh::q6_cpu_reference(projected, stream);
    CUDF_EXPECTS(
      cpu.matched == expected.matched && std::abs(cpu.revenue - expected.revenue) < 1e-12,
      "Q6 CPU reference boundary/slice regression");
    for (bool filter_shipdate : {true, false}) {
      auto result = execute_q6(
        [&](auto const& columns, auto const& predicate) {
          if (!filter_shipdate) {
            return ndsh::read_parquet_fixture(projected, columns, predicate);
          }
          return std::make_unique<table_with_names>(std::make_unique<cudf::table>(projected),
                                                    columns);
        },
        filter_shipdate,
        ndsh::take_result);
      check_q6_result(expected, *result);
    }
  }
}

void check_q6_boundaries()
{
  cudf::test::fixed_width_column_wrapper<double> price{
    {100, 100, 100, 100, 100, 100, 100, 100, 100}};
  cudf::test::fixed_width_column_wrapper<double> discount{
    {0.06, 0.05, 0.07, 0.06, 0.049, 0.071, 0.06, 0.06, 0.06},
    {true, true, true, true, true, true, true, true, false}};
  // Epoch days: 1993-12-31, 1994-01-01, 1994-12-31, 1995-01-01.
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> shipdate{
    {8765, 8766, 9130, 9131, 8766, 8766, 8766, 8766, 8766}};
  cudf::test::fixed_width_column_wrapper<double> quantity{{23, 23, 23, 23, 23, 23, 24, 23, 23}};
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  auto const input = cudf::table_view{{price, discount, shipdate, quantity}};
  for (bool filter_shipdate : {true, false}) {
    auto result = execute_q6(
      [&](auto const& columns, auto const& predicate) {
        if (!filter_shipdate) { return ndsh::read_parquet_fixture(input, columns, predicate); }
        return std::make_unique<table_with_names>(std::make_unique<cudf::table>(input), columns);
      },
      filter_shipdate,
      ndsh::take_result);
    auto value = revenue_value(*result);
    CUDF_EXPECTS(value && std::abs(*value - 18.0) < 1e-8, "Q6 boundary/null regression");
  }
}

struct q6_files {
  ndsh::local_table_files tables;
  ndsh::q6_reference_result reference;

  explicit q6_files(double scale_factor)
  {
    check_q6_result_schema();
    check_q6_boundaries();
    check_q6_reference_boundaries();
    ndsh::vortex_io io{cuda::stream_ref{cudf::get_default_stream()}.get()};
    rmm::mr::pool_memory_resource managed_pool_mr{rmm::mr::managed_memory_resource{},
                                                  rmm::percent_of_free_device_memory(50)};
    auto generated = generate_lineitem(scale_factor, managed_pool_mr);
    tables.write("lineitem", *generated, io);
    reference = ndsh::q6_cpu_reference(generated->select(q6_columns), cudf::get_default_stream());
    CUDF_EXPECTS(std::isfinite(reference.revenue), "Q6 CPU reference revenue must be finite");
    for (bool use_vortex : {false, true}) {
      auto input =
        ndsh::read_local_file(tables.path("lineitem", use_vortex), use_vortex, io, q6_columns);
      ndsh::check_projection(generated->select(q6_columns), *input, q6_columns);
      auto result = execute_q6(
        [&](auto const&, auto const&) { return std::move(input); }, true, ndsh::take_result);
      check_q6_result(reference, *result);
    }
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
};

void ndsh_q6_local(nvbench::state& state)
{
  auto const [use_vortex, read_only, cold, direct_io] = ndsh::local_options{state, 6};
  if (direct_io && !use_vortex) {
    state.skip("io=direct is supported only for Vortex");
    return;
  }
  auto const& files             = ndsh::local_fixture<q6_files>(state.get_float64("scale_factor"));
  cuda::stream_ref const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto memory = cudf::memory_stats_logger();
  ndsh::vortex_io io{stream.get()};
  auto read = [&](auto const&...) {
    return ndsh::read_local_file(
      files.tables.path("lineitem", use_vortex), use_vortex, io, q6_columns, direct_io);
  };
  {
    auto input = read();
    ndsh::check_local_projection(
      files.tables.path("lineitem", use_vortex), use_vortex, io, q6_columns, *input);
    auto result = execute_q6(
      [&](auto const&, auto const&) { return std::move(input); }, true, ndsh::take_result);
    check_q6_result(files.reference, *result);
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  ndsh::warm_local_inputs(cold, read);
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  memory.reset_counters();
  ndsh::exec_local_benchmark(state, files.tables, use_vortex, cold, [&] {
    auto result = read_only ? read() : execute_q6(read, true, ndsh::take_result);
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  });
  state.add_buffer_size(files.tables.bytes(use_vortex), "file_size", "File size");
  state.add_buffer_size(memory.peak_memory_usage(), "rmm_peak", "RMM peak (excludes Vortex)");
  auto& summary = state.add_summary("ndsh/q6/revenue");
  summary.set_string("name", "Verified revenue");
  if (files.reference.matched == 0) {
    summary.set_string("value", "NULL");
  } else {
    summary.set_float64("value", files.reference.revenue);
  }
  ndsh::add_count(state, "ndsh/q6/matched_rows", "Q6 matched rows", files.reference.matched);
}

}  // namespace

NVBENCH_BENCH(ndsh_q6_local)
  .set_name("ndsh_q6_local")
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10})
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q6"})
  .add_string_axis("cache", {"warm", "cold"})
  .add_string_axis("io", {"buffered"});
#endif

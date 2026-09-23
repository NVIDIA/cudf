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
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

#ifdef CUDF_WITH_VORTEX
#include "local_io.hpp"
#include "parquet/parquet_fixture.hpp"
#include "reference/q1_reference.hpp"
#include "vortex/vortex_io.hpp"

#include <benchmarks/common/nvtx_ranges.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/copying.hpp>

#include <rmm/cuda_device.hpp>
#include <rmm/mr/managed_memory_resource.hpp>
#include <rmm/mr/pool_memory_resource.hpp>

#include <cstdint>
#include <map>
#endif

namespace {
std::vector<std::string> const q1_columns{"l_returnflag",
                                          "l_linestatus",
                                          "l_quantity",
                                          "l_extendedprice",
                                          "l_discount",
                                          "l_shipdate",
                                          "l_orderkey",
                                          "l_tax"};
}  // namespace

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
  cuda::stream_ref stream           = cudf::get_default_stream(),
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
  cuda::stream_ref stream           = cudf::get_default_stream(),
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

/**
 * read returns an owning projected table, applying its predicate if filter_shipdate is false.
 * Otherwise filtering happens here. consume receives the result owner by reference and may
 * move it out; its return value is forwarded. This helper adds no final stream synchronization.
 */
template <typename Read, typename Consume>
auto execute_q1(Read&& read, bool filter_shipdate, Consume&& consume)
{
  auto const shipdate_ref = cudf::ast::column_reference(std::distance(
    q1_columns.begin(), std::find(q1_columns.begin(), q1_columns.end(), "l_shipdate")));
  auto shipdate_upper =
    cudf::timestamp_scalar<cudf::timestamp_D>(days_since_epoch(1998, 9, 2), true);
  auto const shipdate_upper_literal = cudf::ast::literal(shipdate_upper);
  auto const lineitem_pred          = std::make_unique<cudf::ast::operation>(
    cudf::ast::ast_operator::LESS_EQUAL, shipdate_ref, shipdate_upper_literal);

  auto lineitem = read(q1_columns, lineitem_pred);
  if (filter_shipdate) { lineitem = apply_filter(lineitem, *lineitem_pred); }

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
  auto orderedby_table = apply_orderby(groupedby_table,
                                       {"l_returnflag", "l_linestatus"},
                                       {cudf::order::ASCENDING, cudf::order::ASCENDING});
  return consume(orderedby_table);
}

void run_ndsh_q1(nvbench::state& state, cudf::io::source_info const& source)
{
  execute_q1([&](auto const& columns,
                 auto const& predicate) { return read_parquet(source, columns, predicate); },
             false,
             [](auto const& result) { write_parquet(*result, "q1.parquet"); });
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
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) { run_ndsh_q1(state, source); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q1)
  .set_name("ndsh_q1")
  .add_string_axis("filename", {""})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1});

#ifdef CUDF_WITH_VORTEX
namespace {

void check_q1_boundaries()
{
  cudf::test::strings_column_wrapper returnflag{"R", "A", "R", "A", "R", "X", "Z"};
  cudf::test::strings_column_wrapper linestatus{"O", "F", "F", "F", "O", "X", "Z"};
  cudf::test::fixed_width_column_wrapper<int8_t> quantity{{4, 2, 3, 6, 8, 50, 50}};
  cudf::test::fixed_width_column_wrapper<double> price{{100, 80, 40, 120, 200, 1000, 1000}};
  cudf::test::fixed_width_column_wrapper<double> discount{{0.25, 0.25, 0.5, 0.0, 0.5, 0.0, 0.0}};
  // Epoch days: 1998-09-01, inclusive cutoff 1998-09-02, 1998-09-03.
  cudf::test::fixed_width_column_wrapper<cudf::timestamp_D, int32_t> shipdate{
    {10470, 10471, 10470, 10470, 10471, 10472, 10471}, {true, true, true, true, true, true, false}};
  cudf::test::fixed_width_column_wrapper<int32_t> orderkey{{1, 2, 3, 4, 5, 6, 7}};
  cudf::test::fixed_width_column_wrapper<double> tax{{0.1, 0.2, 0.1, 0.0, 0.2, 0.0, 0.0}};
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  cudf::table_view const input{
    {returnflag, linestatus, quantity, price, discount, shipdate, orderkey, tax}};
  ndsh::q1_reference_result const expected{{{{"A", "F"}, {8, 200, 180, 192, 4, 100, 0.125, 2}},
                                            {{"R", "F"}, {3, 40, 20, 22, 3, 40, 0.5, 1}},
                                            {{"R", "O"}, {12, 300, 175, 202.5, 6, 150, 0.375, 2}}},
                                           5};
  cuda::stream_ref const stream = cudf::get_default_stream();
  auto const cpu        = ndsh::q1_cpu_reference(cudf::slice(input, {0, 6}).front(), stream);
  auto const sliced_cpu = ndsh::q1_cpu_reference(cudf::slice(input, {1, 6}).front(), stream);
  auto sliced_expected  = expected;
  sliced_expected.groups.at({"R", "O"}) = {8, 200, 100, 120, 8, 200, 0.5, 1};
  sliced_expected.matched               = 4;
  // Full/sliced input, rejected/null dates, and empty input; exercise both reader contracts.
  for (auto const& slice : cudf::slice(input, {0, 7, 1, 7, 5, 7, 0, 0})) {
    for (bool filter_shipdate : {true, false}) {
      auto result = execute_q1(
        [&](auto const& columns, auto const& predicate) {
          if (!filter_shipdate) { return ndsh::read_parquet_fixture(slice, columns, predicate); }
          return std::make_unique<table_with_names>(std::make_unique<cudf::table>(slice), columns);
        },
        filter_shipdate,
        ndsh::take_result);
      ndsh::check_q1_result(slice.num_rows() == 7   ? expected
                            : slice.num_rows() == 6 ? sliced_expected
                                                    : ndsh::q1_reference_result{},
                            *result,
                            stream);
      if (slice.num_rows() >= 6) {
        ndsh::check_q1_result(slice.num_rows() == 7 ? cpu : sliced_cpu, *result, stream);
      }
    }
  }
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
}

struct q1_files {
  ndsh::local_table_files tables;
  ndsh::q1_reference_result reference;

  explicit q1_files(double scale_factor)
  {
    check_q1_boundaries();
    cuda::stream_ref const stream = cudf::get_default_stream();
    ndsh::vortex_io io{stream.get()};
    rmm::mr::pool_memory_resource managed_pool_mr{rmm::mr::managed_memory_resource{},
                                                  rmm::percent_of_free_device_memory(50)};
    auto generated = generate_lineitem(scale_factor, managed_pool_mr);
    CUDF_EXPECTS(generated->table().num_columns() == 16, "Q1 fixture requires full lineitem");
    tables.write("lineitem", *generated, io);
    reference = ndsh::q1_cpu_reference(generated->select(q1_columns), stream);

    for (bool use_vortex : {false, true}) {
      auto input =
        ndsh::read_local_file(tables.path("lineitem", use_vortex), use_vortex, io, q1_columns);
      ndsh::check_projection(generated->select(q1_columns), *input, q1_columns);
      auto result = execute_q1(
        [&](auto const&, auto const&) { return std::move(input); }, true, ndsh::take_result);
      ndsh::check_q1_result(reference, *result, stream);
    }
    generated.reset();
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
};

void ndsh_q1_local(nvbench::state& state)
{
  auto const [use_vortex, read_only, cold, direct_io] = ndsh::local_options{state, 1};
  if (direct_io && !use_vortex) {
    state.skip("io=direct is supported only for Vortex");
    return;
  }
  auto const& files             = ndsh::local_fixture<q1_files>(state.get_float64("scale_factor"));
  cuda::stream_ref const stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto memory = cudf::memory_stats_logger();
  ndsh::vortex_io io{stream.get()};
  auto read = [&](auto const&...) {
    return ndsh::read_local_file(
      files.tables.path("lineitem", use_vortex), use_vortex, io, q1_columns, direct_io);
  };

  {
    auto input = read();
    ndsh::check_local_projection(
      files.tables.path("lineitem", use_vortex), use_vortex, io, q1_columns, *input);
    auto result = execute_q1(
      [&](auto const&, auto const&) { return std::move(input); }, true, ndsh::take_result);
    ndsh::check_q1_result(files.reference, *result, stream);
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  ndsh::warm_local_inputs(cold, read);
  CUDF_CUDA_TRY(cudaDeviceSynchronize());
  memory.reset_counters();
  ndsh::exec_local_benchmark(state, files.tables, use_vortex, cold, [&] {
    cudf::benchmark::scoped_range timed_range{"ndsh_q1_local_timed"};
    auto result = read_only ? read() : execute_q1(read, true, ndsh::take_result);
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
  });
  state.add_buffer_size(files.tables.bytes(use_vortex), "file_size", "File size");
  state.add_buffer_size(memory.peak_memory_usage(), "rmm_peak", "RMM peak (excludes Vortex)");
  ndsh::add_count(state, "ndsh/q1/matched_rows", "Q1 matched rows", files.reference.matched);
  ndsh::add_count(state, "ndsh/q1/groups", "Q1 groups", files.reference.groups.size());
}

}  // namespace

NVBENCH_BENCH(ndsh_q1_local)
  .set_name("ndsh_q1_local")
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10})
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q1"})
  .add_string_axis("cache", {"warm", "cold"})
  .add_string_axis("io", {"buffered"});
#endif

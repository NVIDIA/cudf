/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet/parquet_io.hpp"
#include "q05_query.hpp"
#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef CUDF_WITH_VORTEX
#include "local_io.hpp"
#include "reference/q5_reference.hpp"
#include "vortex/vortex_io.hpp"
#endif

using ndsh::q5::execute_q5;
using ndsh::q5::q5_projections;

/**
 * @file q05.cpp
 * @brief Implement query 5 of the NDS-H benchmark.
 *
 * create view customer as select * from '/tables/scale-1/customer.parquet';
 * create view orders as select * from '/tables/scale-1/orders.parquet';
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 * create view supplier as select * from '/tables/scale-1/supplier.parquet';
 * create view nation as select * from '/tables/scale-1/nation.parquet';
 * create view region as select * from '/tables/scale-1/region.parquet';
 *
 * select
 *    n_name,
 *    sum(l_extendedprice * (1 - l_discount)) as revenue
 * from
 *    customer,
 *    orders,
 *    lineitem,
 *    supplier,
 *    nation,
 *    region
 * where
 *     c_custkey = o_custkey
 *     and l_orderkey = o_orderkey
 *     and l_suppkey = s_suppkey
 *     and c_nationkey = s_nationkey
 *     and s_nationkey = n_nationkey
 *     and n_regionkey = r_regionkey
 *     and r_name = 'ASIA'
 *     and o_orderdate >= date '1994-01-01'
 *     and o_orderdate < date '1995-01-01'
 * group by
 *    n_name
 * order by
 *    revenue desc;
 */

void run_ndsh_q5(nvbench::state& state,
                 std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  execute_q5(
    [&](std::string const& name,
        std::vector<std::string> const& columns,
        std::unique_ptr<cudf::ast::operation> const& predicate) {
      return read_parquet(sources.at(name).make_source_info(), columns, predicate);
    },
    false,
    [](auto const& result) { result->to_parquet("q5.parquet"); });
}

void ndsh_q5(nvbench::state& state)
{
  // Generate the required parquet files in device buffers
  double const scale_factor = state.get_float64("scale_factor");
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"customer", "orders", "lineitem", "supplier", "nation", "region"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { run_ndsh_q5(state, sources); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q5).set_name("ndsh_q5").add_float64_axis("scale_factor", {0.01, 0.1, 1});

#ifdef CUDF_WITH_VORTEX
namespace {

std::vector<std::string> const q5_tables{
  "customer", "orders", "lineitem", "supplier", "nation", "region"};

struct q5_files {
  ndsh::local_table_files tables;
  ndsh::q5_reference_result reference;

  explicit q5_files(double scale_factor)
  {
    ndsh::make_reference_files<ndsh::q5_reference_builder>(
      scale_factor,
      tables,
      reference,
      q5_tables,
      q5_projections,
      [&](auto&& read, cuda::stream_ref stream) {
        auto result = execute_q5(read, true, ndsh::take_result);
        ndsh::check_q5_result(reference, *result, stream);
      });
  }
};

void ndsh_q5_local(nvbench::state& state)
{
  auto const options = ndsh::local_options{state, 5};
  if (!options.supported(state)) { return; }
  auto const& files = ndsh::local_fixture<q5_files>(state.get_float64("scale_factor"));
  ndsh::local_benchmark benchmark{state, files.tables, options};
  auto read = [&](auto const& name, auto const& columns, auto const&) {
    return benchmark.read(name, columns);
  };
  auto read_inputs = [&] {
    return ndsh::read_local_tables(q5_tables, q5_projections, read, options.use_vortex);
  };
  auto query = [&] {
    if (!options.use_vortex) { return execute_q5(read, true, ndsh::take_result); }
    auto inputs = read_inputs();
    return execute_q5(
      [&](std::string const& name, auto const&...) {
        auto const index =
          std::distance(q5_tables.begin(), std::find(q5_tables.begin(), q5_tables.end(), name));
        return std::move(inputs.at(index));
      },
      true,
      ndsh::take_result);
  };
  {
    {
      auto inputs = read_inputs();
      for (std::size_t i = 0; i < q5_tables.size(); ++i) {
        auto const& name = q5_tables[i];
        benchmark.check_projection(name, q5_projections.at(name), *inputs[i]);
      }
      CUDF_CUDA_TRY(cudaStreamSynchronize(benchmark.stream.get()));
    }
    auto result = query();
    ndsh::check_q5_result(files.reference, *result, benchmark.stream);
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  benchmark.exec(read_inputs, query);
  ndsh::add_count(state, "ndsh/q5/matched_rows", "Q5 matched rows", files.reference.matched);
  ndsh::add_count(state, "ndsh/q5/countries", "Q5 countries", files.reference.revenue.size());
}

}  // namespace

// NVBench varies the first axis fastest; keep scale last to reuse the one-scale fixture cache.
NVBENCH_BENCH(ndsh_q5_local)
  .set_name("ndsh_q5_local")
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q5"})
  .add_string_axis("cache", {"warm", "cold"})
  .add_string_axis("io", {"buffered"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10});
#endif

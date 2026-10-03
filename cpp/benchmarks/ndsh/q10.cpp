/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet/parquet_io.hpp"
#include "q10_query.hpp"
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
#include "reference/q10_reference.hpp"
#include "vortex/vortex_io.hpp"
#endif

using ndsh::q10::execute_q10;
using ndsh::q10::q10_projections;

/**
 * @file q10.cpp
 * @brief Implement query 10 of the NDS-H benchmark.
 *
 * create view customer as select * from '/tables/scale-1/customer.parquet';
 * create view orders as select * from '/tables/scale-1/orders.parquet';
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 * create view nation as select * from '/tables/scale-1/nation.parquet';
 *
 * select
 *    c_custkey,
 *    c_name,
 *    sum(l_extendedprice * (1 - l_discount)) as revenue,
 *    c_acctbal,
 *    n_name,
 *    c_address,
 *    c_phone,
 *    c_comment
 * from
 *    customer,
 *    orders,
 *    lineitem,
 *    nation
 * where
 *     c_custkey = o_custkey
 *     and l_orderkey = o_orderkey
 *     and o_orderdate >= date '1993-10-01'
 *     and o_orderdate < date '1994-01-01'
 *     and l_returnflag = 'R'
 *     and c_nationkey = n_nationkey
 * group by
 *     c_custkey,
 *     c_name,
 *     c_acctbal,
 *     c_phone,
 *     n_name,
 *     c_address,
 *     c_comment
 * order by
 *     revenue desc;
 */

void run_ndsh_q10(nvbench::state& state,
                  std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  execute_q10(
    [&](std::string const& name,
        std::vector<std::string> const& columns,
        std::unique_ptr<cudf::ast::operation> const& predicate) {
      return read_parquet(sources.at(name).make_source_info(), columns, predicate);
    },
    false,
    [](auto const& result) { result->to_parquet("q10.parquet"); });
}

void ndsh_q10(nvbench::state& state)
{
  // Generate the required parquet files in device buffers
  double const scale_factor = state.get_float64("scale_factor");
  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"customer", "orders", "lineitem", "nation"}, sources);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { run_ndsh_q10(state, sources); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q10).set_name("ndsh_q10").add_float64_axis("scale_factor", {0.01, 0.1, 1});

#ifdef CUDF_WITH_VORTEX
namespace {

std::vector<std::string> const q10_tables{"customer", "orders", "lineitem", "nation"};

struct q10_files {
  ndsh::local_table_files tables;
  ndsh::q10_reference_result reference;

  explicit q10_files(double scale_factor)
  {
    ndsh::make_reference_files<ndsh::q10_reference_builder>(
      scale_factor,
      tables,
      reference,
      q10_tables,
      q10_projections,
      [&](auto&& read, cuda::stream_ref stream) {
        auto result = execute_q10(read, true, ndsh::take_result);
        ndsh::check_q10_result(reference, *result, stream);
      });
  }
};

void ndsh_q10_local(nvbench::state& state)
{
  auto const options = ndsh::local_options{state, 10};
  if (!options.supported(state)) { return; }
  auto const& files = ndsh::local_fixture<q10_files>(state.get_float64("scale_factor"));
  ndsh::local_benchmark benchmark{state, files.tables, options};
  auto read = [&](auto const& name, auto const& columns, auto const&) {
    return benchmark.read(name, columns);
  };
  auto read_inputs = [&] {
    return ndsh::read_local_tables(q10_tables, q10_projections, read, options.use_vortex);
  };
  auto query = [&] {
    if (!options.use_vortex) { return execute_q10(read, true, ndsh::take_result); }
    auto inputs = read_inputs();
    return execute_q10(
      [&](std::string const& name, auto const&...) {
        auto const index =
          std::distance(q10_tables.begin(), std::find(q10_tables.begin(), q10_tables.end(), name));
        return std::move(inputs.at(index));
      },
      true,
      ndsh::take_result);
  };
  {
    {
      auto inputs = read_inputs();
      for (std::size_t i = 0; i < q10_tables.size(); ++i) {
        auto const& name = q10_tables[i];
        benchmark.check_projection(name, q10_projections.at(name), *inputs[i]);
      }
      CUDF_CUDA_TRY(cudaStreamSynchronize(benchmark.stream.get()));
    }
    auto result = query();
    ndsh::check_q10_result(files.reference, *result, benchmark.stream);
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  benchmark.exec(read_inputs, query, "Total file size", "ndsh_q10_local_timed");
  ndsh::add_count(state, "ndsh/q10/matched_rows", "Q10 matched rows", files.reference.matched);
  ndsh::add_count(state, "ndsh/q10/customers", "Q10 customers", files.reference.customers.size());
}

}  // namespace

// NVBench varies the first axis fastest; keep scale last to reuse the one-scale fixture cache.
NVBENCH_BENCH(ndsh_q10_local)
  .set_name("ndsh_q10_local")
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q10"})
  .add_string_axis("cache", {"warm", "cold"})
  .add_string_axis("io", {"buffered"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10});
#endif

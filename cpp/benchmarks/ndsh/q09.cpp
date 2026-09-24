/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet/parquet_io.hpp"
#include "q09_query.hpp"
#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef CUDF_WITH_VORTEX
#include "local_io.hpp"
#include "reference/q9_reference.hpp"
#include "vortex/vortex_io.hpp"
#endif

using ndsh::q9::compute_amount;
using ndsh::q9::compute_profit;
using ndsh::q9::engine_type;
using ndsh::q9::join_data;
using ndsh::q9::load_data;
using ndsh::q9::q9_data;
using ndsh::q9::q9_projections;

engine_type engine_from_string(std::string const& str)
{
  if (str == "binaryop") {
    return engine_type::BINARYOP;
  } else if (str == "ast") {
    return engine_type::AST;
  } else if (str == "transform") {
    return engine_type::TRANSFORM;
  } else {
    CUDF_FAIL("unrecognized engine enum: " + str);
  }
}

/**
 * @file q09.cpp
 * @brief Implement query 9 of the NDS-H benchmark.
 *
 * create view part as select * from '/tables/scale-1/part.parquet';
 * create view supplier as select * from '/tables/scale-1/supplier.parquet';
 * create view lineitem as select * from '/tables/scale-1/lineitem.parquet';
 * create view partsupp as select * from '/tables/scale-1/partsupp.parquet';
 * create view orders as select * from '/tables/scale-1/orders.parquet';
 * create view nation as select * from '/tables/scale-1/nation.parquet';
 *
 * select
 *    nation,
 *    o_year,
 *    sum(amount) as sum_profit
 * from
 *     (
 *        select
 *            n_name as nation,
 *            extract(year from o_orderdate) as o_year,
 *            l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
 *        from
 *            part,
 *            supplier,
 *            lineitem,
 *            partsupp,
 *            orders,
 *            nation
 *        where
 *           s_suppkey = l_suppkey
 *           and ps_suppkey = l_suppkey
 *           and ps_partkey = l_partkey
 *           and p_partkey = l_partkey
 *           and o_orderkey = l_orderkey
 *           and s_nationkey = n_nationkey
 *           and p_name like '%green%'
 *     ) as profit
 * group by
 *     nation,
 *     o_year
 * order by
 *     nation,
 *     o_year desc;
 */

q9_data load_data(std::unordered_map<std::string, cuio_source_sink_pair>& sources)
{
  return load_data([&](std::string const& name, std::vector<std::string> const& columns) {
    return read_parquet(sources.at(name).make_source_info(), columns);
  });
}

void ndsh_q9(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const engine       = engine_from_string(state.get_string("engine"));

  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"part", "supplier", "lineitem", "partsupp", "orders", "nation"}, sources);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    q9_data const data = load_data(sources);
    auto const result  = compute_profit(
      engine, data, launch.get_stream().get_stream(), cudf::get_current_device_resource_ref());
    result->to_parquet("q9.parquet");
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void ndsh_q9_noio(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const engine       = engine_from_string(state.get_string("engine"));

  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"part", "supplier", "lineitem", "partsupp", "orders", "nation"}, sources);

  q9_data const data = load_data(sources);

  std::unique_ptr<table_with_names> result;

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    result = compute_profit(
      engine, data, launch.get_stream().get_stream(), cudf::get_current_device_resource_ref());
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  if (result) { result->to_parquet("q9_noio.parquet"); }
}

// unlike `ndsh_q9`, `ndsh_q9_amount` benchmarks only the amount calculation part of the benchmark
void ndsh_q9_amount(nvbench::state& state)
{
  auto const scale_factor = state.get_float64("scale_factor");
  auto const engine       = engine_from_string(state.get_string("engine"));

  std::unordered_map<std::string, cuio_source_sink_pair> sources;
  generate_parquet_data_sources(
    scale_factor, {"part", "supplier", "lineitem", "partsupp", "orders", "nation"}, sources);

  q9_data const data      = load_data(sources);
  auto const joined_table = join_data(data);

  auto const size = joined_table->column("l_extendedprice").size();

  state.add_global_memory_reads<double>(size * 4);
  state.add_global_memory_writes<double>(size);
  state.add_element_count(size);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto amount = compute_amount(joined_table->column("l_discount"),
                                 joined_table->column("l_extendedprice"),
                                 joined_table->column("ps_supplycost"),
                                 joined_table->column("l_quantity"),
                                 engine,
                                 launch.get_stream().get_stream(),
                                 cudf::get_current_device_resource_ref());
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(ndsh_q9)
  .set_name("ndsh_q9")
  .add_float64_axis("scale_factor", {0.01, 0.1, 1})
  .add_string_axis("engine", {"binaryop", "ast", "transform"});

NVBENCH_BENCH(ndsh_q9_noio)
  .set_name("ndsh_q9_noio")
  .add_float64_axis("scale_factor", {0.01, 0.1, 1})
  .add_string_axis("engine", {"binaryop", "ast", "transform"});

NVBENCH_BENCH(ndsh_q9_amount)
  .set_name("ndsh_q9_amount")
  .add_float64_axis("scale_factor", {0.01, 0.1, 1})
  .add_string_axis("engine", {"binaryop", "ast", "transform"});

#ifdef CUDF_WITH_VORTEX
namespace {

std::vector<std::string> const q9_tables{
  "part", "supplier", "lineitem", "partsupp", "orders", "nation"};

struct q9_files {
  ndsh::local_table_files tables;
  ndsh::q9_reference_result reference;

  explicit q9_files(double scale_factor)
  {
    ndsh::make_reference_files<ndsh::q9_reference_builder>(
      scale_factor,
      tables,
      reference,
      q9_tables,
      q9_projections,
      [&](auto&& read, cuda::stream_ref stream) {
        auto result = compute_profit(engine_type::BINARYOP, load_data(read));
        ndsh::check_q9_result(reference, *result, stream);
      });
  }
};

void ndsh_q9_local(nvbench::state& state)
{
  auto const engine  = engine_from_string(state.get_string("engine"));
  auto const options = ndsh::local_options{state, 9};
  if (!options.supported(state)) { return; }
  auto const& files = ndsh::local_fixture<q9_files>(state.get_float64("scale_factor"));
  ndsh::local_benchmark benchmark{state, files.tables, options};
  auto read = [&](auto const& name, auto const& columns, auto const&...) {
    return benchmark.read(name, columns);
  };
  auto load_inputs = [&](bool verify = false) {
    if (!options.use_vortex) {
      return load_data([&](std::string const& name, std::vector<std::string> const& columns) {
        auto input = read(name, columns);
        if (verify) { benchmark.check_projection(name, columns, *input); }
        return input;
      });
    }
    auto inputs = ndsh::read_local_tables(q9_tables, q9_projections, read, true);
    return load_data([&](std::string const& name, std::vector<std::string> const& columns) {
      auto const index =
        std::distance(q9_tables.begin(), std::find(q9_tables.begin(), q9_tables.end(), name));
      if (verify) { benchmark.check_projection(name, columns, *inputs.at(index)); }
      return std::move(inputs.at(index));
    });
  };
  auto query = [&] {
    auto inputs = load_inputs();
    return compute_profit(
      engine, inputs, benchmark.stream, cudf::get_current_device_resource_ref());
  };
  {
    auto inputs = load_inputs(true);
    auto result =
      compute_profit(engine, inputs, benchmark.stream, cudf::get_current_device_resource_ref());
    ndsh::check_q9_result(files.reference, *result, benchmark.stream);
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  benchmark.exec(load_inputs, query);
  ndsh::add_count(state, "ndsh/q9/matched_rows", "Q9 matched rows", files.reference.matched);
  ndsh::add_count(
    state, "ndsh/q9/groups", "Q9 nation/year groups", files.reference.sum_profit.size());
}

}  // namespace

// NVBench varies the first axis fastest; keep scale last to reuse the one-scale fixture cache.
NVBENCH_BENCH(ndsh_q9_local)
  .set_name("ndsh_q9_local")
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q9"})
  .add_string_axis("engine", {"binaryop", "ast", "transform"})
  .add_string_axis("cache", {"warm", "cold"})
  .add_string_axis("io", {"buffered"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10});
#endif

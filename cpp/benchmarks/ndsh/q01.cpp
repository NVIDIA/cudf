/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet/parquet_io.hpp"
#include "q01_query.hpp"
#include "utilities.hpp"

#include <benchmarks/common/memory_stats.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef CUDF_WITH_VORTEX
#include "local_io.hpp"
#include "reference/q1_reference.hpp"
#include "vortex/vortex_io.hpp"
#endif

using ndsh::q1::execute_q1;
using ndsh::q1::q1_columns;

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

void run_ndsh_q1(nvbench::state& state, cudf::io::source_info const& source)
{
  execute_q1([&](auto const& columns,
                 auto const& predicate) { return read_parquet(source, columns, predicate); },
             false,
             [](auto const& result) { result->to_parquet("q1.parquet"); });
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

struct q1_files {
  ndsh::local_table_files tables;
  ndsh::q1_reference_result reference;

  explicit q1_files(double scale_factor)
  {
    cuda::stream_ref const stream = cudf::get_default_stream();
    ndsh::vortex_io io{stream.get()};
    for_each_generated_table(
      scale_factor, {"lineitem"}, [&](auto const& name, table_with_names const& generated) {
        CUDF_EXPECTS(generated.table().num_columns() == 16, "Q1 fixture requires full lineitem");
        tables.write(name, generated, io);
        reference = ndsh::q1_cpu_reference(generated.select(q1_columns), stream);
        for (bool use_vortex : {false, true}) {
          auto input =
            ndsh::read_local_file(tables.path(name, use_vortex), use_vortex, io, q1_columns);
          ndsh::check_projection(generated.select(q1_columns), *input, q1_columns);
          auto result = execute_q1(
            [&](auto const&, auto const&) { return std::move(input); }, true, ndsh::take_result);
          ndsh::check_q1_result(reference, *result, stream);
        }
        CUDF_CUDA_TRY(cudaDeviceSynchronize());
      });
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
};

void ndsh_q1_local(nvbench::state& state)
{
  auto const options = ndsh::local_options{state, 1};
  if (!options.supported(state)) { return; }
  auto const& files = ndsh::local_fixture<q1_files>(state.get_float64("scale_factor"));
  ndsh::local_benchmark benchmark{state, files.tables, options};
  auto read = [&](auto const&...) { return benchmark.read("lineitem", q1_columns); };

  {
    auto input = read();
    benchmark.check_projection("lineitem", q1_columns, *input);
    auto result = execute_q1(
      [&](auto const&, auto const&) { return std::move(input); }, true, ndsh::take_result);
    ndsh::check_q1_result(files.reference, *result, benchmark.stream);
    CUDF_CUDA_TRY(cudaDeviceSynchronize());
  }
  benchmark.exec(
    read,
    [&] { return execute_q1(read, true, ndsh::take_result); },
    "File size",
    "ndsh_q1_local_timed");
  ndsh::add_count(state, "ndsh/q1/matched_rows", "Q1 matched rows", files.reference.matched);
  ndsh::add_count(state, "ndsh/q1/groups", "Q1 groups", files.reference.groups.size());
}

}  // namespace

// NVBench varies the first axis fastest; keep scale last to reuse the one-scale fixture cache.
NVBENCH_BENCH(ndsh_q1_local)
  .set_name("ndsh_q1_local")
  .add_string_axis("format", {"parquet", "vortex"})
  .add_string_axis("workload", {"read", "q1"})
  .add_string_axis("cache", {"warm", "cold"})
  .add_string_axis("io", {"buffered"})
  .add_float64_axis("scale_factor", {0.01, 0.1, 1, 10});
#endif

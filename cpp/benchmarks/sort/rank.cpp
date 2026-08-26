/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_rank(nvbench::state& state)
{
  auto const n_rows = static_cast<cudf::size_type>(state.get_int64("n_rows"));
  auto const nulls  = state.get_float64("nulls");

  // Create columns with values in the range [0,100)
  data_profile const profile =
    data_profile_builder().cardinality(0).null_probability(nulls).distribution(
      cudf::type_id::INT32, distribution_id::UNIFORM, 0, 10);

  auto input = create_random_column(cudf::type_id::INT32, row_count{n_rows}, profile);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.add_element_count(n_rows, "n_rows");
  state.add_global_memory_reads<nvbench::int32_t>(n_rows);
  state.add_global_memory_writes<nvbench::int32_t>(n_rows);
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = cudf::rank(input->view(),
                             cudf::rank_method::FIRST,
                             cudf::order::ASCENDING,
                             nulls ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE,
                             cudf::null_order::AFTER,
                             false);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_rank)
  .set_name("rank")
  .add_float64_axis("nulls", {0, 0.1})
  .add_int64_axis("n_rows", {32768, 262144, 2097152, 16777216, 67108864});

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

static void bench_groupby_shift(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  int const num_groups = 100;

  data_profile const profile =
    data_profile_builder().cardinality(0).null_probability(0.01).distribution(
      cudf::type_to_id<int64_t>(), distribution_id::UNIFORM, 0, num_groups);

  auto keys_table =
    create_random_table({cudf::type_to_id<int64_t>()}, row_count{num_rows}, profile);
  auto vals_table =
    create_random_table({cudf::type_to_id<int64_t>()}, row_count{num_rows}, profile);

  std::vector<cudf::size_type> offsets{
    static_cast<cudf::size_type>(num_rows / float(num_groups) * 0.5)};  // forward shift half way
  // null fill value
  auto fill_value = cudf::make_default_constructed_scalar(cudf::data_type(cudf::type_id::INT64));

  state.add_global_memory_reads<nvbench::int8_t>(keys_table->alloc_size());
  state.add_global_memory_writes<nvbench::int8_t>(vals_table->alloc_size());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::groupby::groupby gb_obj(keys_table->view());
    auto result = gb_obj.shift(*vals_table, offsets, {*fill_value});
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_groupby_shift)
  .set_name("shift")
  .add_int64_axis("num_rows", {1'000'000, 10'000'000, 100'000'000});

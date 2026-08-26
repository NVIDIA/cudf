/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/reduction/distinct_count.hpp>

#include <nvbench/nvbench.cuh>

template <typename Type>
static void bench_distinct_count(nvbench::state& state, nvbench::type_list<Type>)
{
  auto const dtype            = cudf::type_to_id<Type>();
  auto const size             = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const null_probability = state.get_float64("null_probability");

  data_profile profile =
    data_profile_builder().distribution(dtype, distribution_id::UNIFORM, 0, size / 100);
  if (null_probability > 0) {
    profile.set_null_probability({null_probability});
  } else {
    profile.set_null_probability(std::nullopt);
  }

  auto const data_table   = create_random_table({dtype}, row_count{size}, profile);
  auto const& data_column = data_table->get_column(0);
  auto const input_table  = cudf::table_view{{data_column, data_column, data_column}};

  // Collect memory statistics for input and output.
  state.add_global_memory_reads<Type>(input_table.num_rows() * input_table.num_columns());
  state.add_global_memory_writes<cudf::size_type>(1);
  if (null_probability > 0) {
    state.add_global_memory_reads<nvbench::int8_t>(
      input_table.num_columns() * cudf::bitmask_allocation_size_bytes(input_table.num_rows()));
  }

  auto mem_stats_logger = cudf::memory_stats_logger();  // init stats logger
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::distinct_count(input_table, cudf::null_equality::EQUAL);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

using data_type = nvbench::type_list<int32_t, int64_t, float, double>;

NVBENCH_BENCH_TYPES(bench_distinct_count, NVBENCH_TYPE_AXES(data_type))
  .set_name("distinct_count")
  .add_int64_axis("num_rows",
                  {
                    10000,      // 10k
                    100000,     // 100k
                    1000000,    // 1M
                    10000000,   // 10M
                    100000000,  // 100M
                  })
  .add_float64_axis("null_probability", {0, 0.5});

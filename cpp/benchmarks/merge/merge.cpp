/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/merge.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

#include <random>

using IntColWrap = cudf::test::fixed_width_column_wrapper<int32_t>;

void bench_merge(nvbench::state& state)
{
  cudf::size_type const avg_rows = 1 << 19;  // 512K rows
  int const num_tables           = static_cast<int>(state.get_int64("num_tables"));

  // Content is irrelevant for the benchmark
  auto data_sequence = cuda::make_constant_iterator(0);

  // Using 0 seed to ensure consistent pseudo-numbers on each run
  std::mt19937 rand_gen(0);
  // Gaussian distribution with 98% of elements are in range [0, avg_rows*2]
  std::normal_distribution<> table_size_dist(avg_rows, avg_rows / 2);
  // Used to generate a random monotonic sequence for each table key column
  std::uniform_int_distribution<> key_dist(0, 10);

  std::vector<std::pair<IntColWrap, IntColWrap>> columns;
  size_t total_rows = 0;
  std::vector<cudf::table_view> tables;
  for (int i = 0; i < num_tables; ++i) {
    cudf::size_type const rows = std::round(table_size_dist(rand_gen));
    // Ensure size in range [0, avg_rows*2]
    auto const clamped_rows = std::clamp(rows, 0, avg_rows * 2);

    int32_t prev_key  = 0;
    auto key_sequence = cudf::detail::make_counting_transform_iterator(0, [&](auto row) {
      prev_key += key_dist(rand_gen);
      return prev_key;
    });

    columns.emplace_back(
      std::pair<IntColWrap, IntColWrap>{IntColWrap(key_sequence, key_sequence + clamped_rows),
                                        IntColWrap(data_sequence, data_sequence + clamped_rows)});
    tables.push_back(cudf::table_view{{columns.back().first, columns.back().second}});
    total_rows += clamped_rows;
  }
  std::vector<cudf::size_type> const key_cols{0};
  std::vector<cudf::order> const column_order{cudf::order::ASCENDING};
  std::vector<cudf::null_order> const null_precedence{};

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const data_size = total_rows * 2 * sizeof(int32_t);
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(data_size);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = cudf::merge(tables, key_cols, column_order, null_precedence);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_merge)
  .set_name("merge")
  .add_int64_power_of_two_axis("num_tables", nvbench::range(1, 7, 1));  // 2^1 to 2^7 = 2 to 128

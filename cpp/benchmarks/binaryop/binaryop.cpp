/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/binaryop.hpp>
#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <cstddef>
#include <memory>

// This set of benchmarks is designed to be a comparison for the AST benchmarks

enum class TreeType {
  IMBALANCED_LEFT  // All operator expressions have a left child operator expression and a right
                   // child column reference
};

template <typename key_type, TreeType tree_type, bool reuse_columns>
static void BM_binaryop_transform(nvbench::state& state)
{
  auto const num_rows{static_cast<cudf::size_type>(state.get_int64("num_rows"))};
  auto const tree_levels{static_cast<cudf::size_type>(state.get_int64("tree_levels"))};

  // Create table data
  auto const n_cols       = reuse_columns ? 1 : tree_levels + 1;
  auto const source_table = create_sequence_table(
    cycle_dtypes({cudf::type_to_id<key_type>()}, n_cols), row_count{num_rows});
  cudf::table_view table{*source_table};

  // Use the number of bytes read from global memory
  state.add_global_memory_reads<key_type>(static_cast<std::size_t>(num_rows) * (tree_levels + 1));
  state.add_global_memory_writes<key_type>(num_rows);
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    // Execute tree that chains additions like (((a + b) + c) + d)
    auto const op               = cudf::binary_operator::ADD;
    auto const result_data_type = cudf::data_type(cudf::type_to_id<key_type>());
    if (reuse_columns) {
      auto result = cudf::binary_operation(table.column(0), table.column(0), op, result_data_type);
      for (cudf::size_type i = 0; i < tree_levels - 1; i++) {
        result = cudf::binary_operation(result->view(), table.column(0), op, result_data_type);
      }
    } else {
      auto result = cudf::binary_operation(table.column(0), table.column(1), op, result_data_type);
      std::for_each(std::next(table.begin(), 2), table.end(), [&](auto const& col) {
        result = cudf::binary_operation(result->view(), col, op, result_data_type);
      });
    }
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

template <cudf::binary_operator cmp_op, cudf::binary_operator reduce_op>
static void BM_string_compare_binaryop_transform(nvbench::state& state)
{
  auto const string_width = static_cast<cudf::size_type>(state.get_int64("string_width"));
  auto const num_rows     = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const tree_levels  = static_cast<cudf::size_type>(state.get_int64("tree_levels"));
  auto const hit_rate     = static_cast<cudf::size_type>(state.get_int64("hit_rate"));

  CUDF_EXPECTS(tree_levels > 0, "benchmarks require 1 or more comparisons");

  // Create table data
  auto const num_cols = tree_levels * 2;
  std::vector<std::unique_ptr<cudf::column>> columns;
  std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                cuda::counting_iterator{num_cols},
                [&](cudf::size_type) {
                  columns.emplace_back(create_string_column(num_rows, string_width, hit_rate));
                });

  cudf::table table{std::move(columns)};
  cudf::table_view const table_view = table.view();

  int64_t const chars_size = std::accumulate(
    table_view.begin(), table_view.end(), static_cast<int64_t>(0), [](int64_t size, auto& column) {
      return size + cudf::strings_column_view{column}.chars_size(cudf::get_default_stream());
    });

  // Create column references

  // Use the number of bytes read from global memory
  state.add_element_count(chars_size, "chars_size");
  state.add_global_memory_reads<nvbench::uint8_t>(chars_size);
  state.add_global_memory_writes<nvbench::int32_t>(num_rows);

  // Construct binary operations (a == b && c == d && e == f && ...)
  auto constexpr bool_type    = cudf::data_type{cudf::type_id::BOOL8};
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream{launch.get_stream().get_stream()};
    std::unique_ptr<cudf::column> reduction =
      cudf::binary_operation(table.get_column(0), table.get_column(1), cmp_op, bool_type, stream);
    std::for_each(
      cuda::counting_iterator<cudf::size_type>{1},
      cuda::counting_iterator{tree_levels},
      [&](std::size_t idx) {
        std::unique_ptr<cudf::column> comparison = cudf::binary_operation(
          table.get_column(idx * 2), table.get_column(idx * 2 + 1), cmp_op, bool_type, stream);
        std::unique_ptr<cudf::column> reduced =
          cudf::binary_operation(*comparison, *reduction, reduce_op, bool_type, stream);
        stream.synchronize();
        reduction = std::move(reduced);
      });
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

#define BINARYOP_TRANSFORM_BENCHMARK_DEFINE(name, key_type, tree_type, reuse_columns) \
                                                                                      \
  static void name(::nvbench::state& st)                                              \
  {                                                                                   \
    ::BM_binaryop_transform<key_type, tree_type, reuse_columns>(st);                  \
  }                                                                                   \
  NVBENCH_BENCH(name)                                                                 \
    .add_int64_axis("tree_levels", {1, 2, 5, 10})                                     \
    .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000})

BINARYOP_TRANSFORM_BENCHMARK_DEFINE(binaryop_int32_imbalanced_unique,
                                    int32_t,
                                    TreeType::IMBALANCED_LEFT,
                                    false);
BINARYOP_TRANSFORM_BENCHMARK_DEFINE(binaryop_int32_imbalanced_reuse,
                                    int32_t,
                                    TreeType::IMBALANCED_LEFT,
                                    true);
BINARYOP_TRANSFORM_BENCHMARK_DEFINE(binaryop_double_imbalanced_unique,
                                    double,
                                    TreeType::IMBALANCED_LEFT,
                                    false);

#define STRING_COMPARE_BINARYOP_TRANSFORM_BENCHMARK_DEFINE(name, cmp_op, reduce_op) \
                                                                                    \
  static void name(::nvbench::state& st)                                            \
  {                                                                                 \
    ::BM_string_compare_binaryop_transform<cmp_op, reduce_op>(st);                  \
  }                                                                                 \
  NVBENCH_BENCH(name)                                                               \
    .set_name(#name)                                                                \
    .add_int64_axis("string_width", {32, 64, 128, 256})                             \
    .add_int64_axis("num_rows", {32768, 262144, 2097152})                           \
    .add_int64_axis("tree_levels", {1, 2, 3, 4})                                    \
    .add_int64_axis("hit_rate", {50, 100})

STRING_COMPARE_BINARYOP_TRANSFORM_BENCHMARK_DEFINE(string_compare_binaryop_transform,
                                                   cudf::binary_operator::EQUAL,
                                                   cudf::binary_operator::LOGICAL_AND);

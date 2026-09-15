/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/column/column.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>
#include <nvbench/types.cuh>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

enum class TreeType {
  IMBALANCED_LEFT  // All operator expressions have a left child operator expression and a right
                   // child column reference
};

template <typename key_type, TreeType tree_type, bool reuse_columns, bool Nullable>
static void BM_transform(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const tree_levels = static_cast<cudf::size_type>(state.get_int64("tree_levels"));

  // Create table data
  auto const num_columns = reuse_columns ? 1 : tree_levels + 1;
  auto const source_table =
    create_sequence_table(cycle_dtypes({cudf::type_to_id<key_type>()}, num_columns),
                          row_count{num_rows},
                          Nullable ? std::optional<double>{0.5} : std::nullopt);
  auto table = source_table->view();

  // Construct expression that chains additions like (((a + b) + c) + d)
  std::string const op = "+";
  std::string expression;
  if constexpr (reuse_columns) {
    expression = "c0 " + op + " c0";
    std::for_each(cuda::counting_iterator<cudf::size_type>{1},
                  cuda::counting_iterator{num_columns},
                  [&](cudf::size_type) { expression = "( " + expression + " ) " + op + " c0 "; });
  } else {
    expression = "c0 " + op + " c1";
    std::for_each(cuda::counting_iterator<cudf::size_type>{2},
                  cuda::counting_iterator{num_columns},
                  [&](cudf::size_type col) {
                    expression = "( " + expression + " ) " + op + " c" + std::to_string(col);
                  });
  }

  std::string type_name = cudf::type_to_name(cudf::data_type{cudf::type_to_id<key_type>()});
  std::string params    = type_name + " c0";

  std::for_each(
    cuda::counting_iterator<cudf::size_type>{1},
    cuda::counting_iterator{num_columns},
    [&](cudf::size_type param) { params += ", " + type_name + " c" + std::to_string(param); });

  std::string code =
    "void transform(" + type_name + "* out, " + params + " ) {  *out = " + expression + "; }";

  std::vector<cudf::transform_input> inputs;

  std::transform(cuda::counting_iterator<cudf::size_type>{0},
                 cuda::counting_iterator{source_table->num_columns()},
                 std::back_inserter(inputs),
                 [&source_table](int col) {
                   return cudf::transform_input{source_table->get_column(col).view()};
                 });

  // Use the number of bytes read from global memory
  state.add_global_memory_reads<key_type>(static_cast<std::size_t>(num_rows) * (tree_levels + 1));
  state.add_global_memory_writes<key_type>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::transform(code,
                    cudf::udf_source_type::CUDA,
                    cudf::null_aware::NO,
                    std::nullopt,
                    inputs,
                    std::array{cudf::transform_output{cudf::data_type{cudf::type_to_id<key_type>()},
                                                      cudf::output_nullability::PRESERVE}},
                    {},
                    std::nullopt,
                    launch.get_stream().get_stream());
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

#define AST_TRANSFORM_BENCHMARK_DEFINE(name, key_type, tree_type, reuse_columns, nullable) \
  static void name(::nvbench::state& st)                                                   \
  {                                                                                        \
    ::BM_transform<key_type, tree_type, reuse_columns, nullable>(st);                      \
  }                                                                                        \
  NVBENCH_BENCH(name)                                                                      \
    .set_name(#name)                                                                       \
    .add_int64_axis("tree_levels", {1, 5, 10})                                             \
    .add_int64_axis("num_rows", {100'000, 1'000'000, 10'000'000, 100'000'000})

AST_TRANSFORM_BENCHMARK_DEFINE(
  transform_int32_imbalanced_unique, int32_t, TreeType::IMBALANCED_LEFT, false, false);
AST_TRANSFORM_BENCHMARK_DEFINE(
  transform_int32_imbalanced_reuse, int32_t, TreeType::IMBALANCED_LEFT, true, false);
AST_TRANSFORM_BENCHMARK_DEFINE(
  transform_double_imbalanced_unique, double, TreeType::IMBALANCED_LEFT, false, false);

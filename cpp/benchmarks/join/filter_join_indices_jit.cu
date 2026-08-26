/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "generate_input_tables.hpp"
#include "join_common.hpp"

#include <cudf/ast/expressions.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <nvbench/nvbench.cuh>

template <typename JoinFunc>
void filter_join_indices_benchmark(nvbench::state& state,
                                   JoinFunc join_func,
                                   cudf::join_kind join_kind)
{
  auto const build_size = static_cast<cudf::size_type>(state.get_int64("build_size"));
  auto const probe_size = static_cast<cudf::size_type>(state.get_int64("probe_size"));

  // Generate build (right) and probe (left) tables
  // 1 key column + 2 payload columns = 3 columns total
  std::vector<cudf::type_id> key_types{cudf::type_id::INT32};

  auto [build_table, probe_table] =
    generate_input_tables<false>(key_types, build_size, probe_size, 2, 1, 0.3);

  // Perform hash join on key column (column 0) to get indices
  auto probe_keys = probe_table->view().select({0});
  auto build_keys = build_table->view().select({0});

  cudf::hash_join hash_joiner(build_keys, cudf::null_equality::EQUAL);
  auto [left_indices, right_indices] = join_func(hash_joiner, probe_keys);

  cudf::device_span<cudf::size_type const> left_span{left_indices->data(), left_indices->size()};
  cudf::device_span<cudf::size_type const> right_span{right_indices->data(), right_indices->size()};

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  auto const method = state.get_string("method");

  auto const join_input_size = estimate_size(build_keys) + estimate_size(probe_keys);
  auto const filter_input_size =
    join_input_size + (left_indices->size() + right_indices->size()) * sizeof(cudf::size_type);
  state.add_element_count(filter_input_size, "filter_input_size");  // number of bytes
  state.template add_global_memory_reads<nvbench::int8_t>(filter_input_size);

  if (method == "AST") {
    auto col_ref_left_1  = cudf::ast::column_reference(1, cudf::ast::table_reference::LEFT);
    auto col_ref_right_1 = cudf::ast::column_reference(1, cudf::ast::table_reference::RIGHT);
    auto predicate =
      cudf::ast::operation(cudf::ast::ast_operator::GREATER, col_ref_left_1, col_ref_right_1);

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = cudf::filter_join_indices(
        probe_table->view(), build_table->view(), left_span, right_span, predicate, join_kind);
    });
  } else {
    std::string predicate_code = R"(
      __device__ void predicate(bool* output,
                                int32_t left_col0, int32_t left_col1, int32_t left_col2,
                                int32_t right_col0, int32_t right_col1, int32_t right_col2) {
        *output = left_col1 > right_col1;
      }
    )";

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      auto result = cudf::filter_join_indices_jit(
        probe_table->view(), build_table->view(), left_span, right_span, predicate_code, join_kind);
    });
  }
  set_throughputs(state);
}

void filter_join_indices_inner_join(nvbench::state& state)
{
  filter_join_indices_benchmark(
    state,
    [](cudf::hash_join& joiner, cudf::table_view probe_keys) {
      return joiner.inner_join(probe_keys);
    },
    cudf::join_kind::INNER_JOIN);
}

void filter_join_indices_left_join(nvbench::state& state)
{
  filter_join_indices_benchmark(
    state,
    [](cudf::hash_join& joiner, cudf::table_view probe_keys) {
      return joiner.left_join(probe_keys);
    },
    cudf::join_kind::LEFT_JOIN);
}

NVBENCH_BENCH(filter_join_indices_inner_join)
  .set_name("filter_join_indices_inner")
  .add_string_axis("method", {"AST", "JIT"})
  .add_int64_axis("build_size", JOIN_SIZE_RANGE)
  .add_int64_axis("probe_size", JOIN_SIZE_RANGE);

NVBENCH_BENCH(filter_join_indices_left_join)
  .set_name("filter_join_indices_left")
  .add_string_axis("method", {"AST", "JIT"})
  .add_int64_axis("build_size", JOIN_SIZE_RANGE)
  .add_int64_axis("probe_size", JOIN_SIZE_RANGE);

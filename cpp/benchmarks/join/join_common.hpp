/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "generate_input_tables.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>
#include <benchmarks/common/table_utilities.hpp>
#include <benchmarks/join/nvbench_helpers.hpp>

#include <cudf/ast/expressions.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

auto const JOIN_SIZE_RANGE = std::vector<nvbench::int64_t>{1000, 100'000, 10'000'000};
auto const JOIN_SELECTIVITY_RANGE =
  std::vector<double>{0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9};
using JOIN_NULLABLE_RANGE = nvbench::enum_type_list<false, true>;

using JOIN_ALGORITHM = nvbench::enum_type_list<join_t::HASH, join_t::SORT_MERGE>;
using JOIN_DATATYPES = nvbench::enum_type_list<data_type::INT32,
                                               data_type::INT64,
                                               data_type::FLOAT32,
                                               data_type::FLOAT64,
                                               data_type::STRING,
                                               data_type::LIST,
                                               data_type::STRUCT>;
using JOIN_NULL_EQUALITY =
  nvbench::enum_type_list<cudf::null_equality::EQUAL, cudf::null_equality::UNEQUAL>;

using DEFAULT_JOIN_DATATYPES     = nvbench::enum_type_list<data_type::INT32>;
using SELECTIVITY_JOIN_DATATYPES = nvbench::enum_type_list<data_type::INT32, data_type::STRING>;
using DEFAULT_JOIN_NULL_EQUALITY = nvbench::enum_type_list<cudf::null_equality::UNEQUAL>;

inline void create_complex_ast_expression(cudf::ast::tree& tree, cudf::size_type ast_levels)
{
  CUDF_EXPECTS(ast_levels > 0, "Number of AST levels must be greater than 0");

  tree.push(cudf::ast::column_reference(0));
  tree.push(cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT));

  tree.push(cudf::ast::operation(cudf::ast::ast_operator::EQUAL, tree.at(0), tree.at(1)));

  if (ast_levels == 1) { return; }

  for (cudf::size_type i = 1; i < ast_levels; i++) {
    tree.push(cudf::ast::operation(cudf::ast::ast_operator::EQUAL, tree.at(0), tree.at(1)));

    tree.push(cudf::ast::operation(
      cudf::ast::ast_operator::LOGICAL_AND, tree.at(tree.size() - 2), tree.back()));
  }
}

// Returns true (and marks `state` as skipped) when the `skip_large_sizes` axis is enabled and
// the build side of this benchmark is larger than the probe side. `build_is_right` selects which
// side is the build side: true for hash-style benches that preprocess the right table, false for
// benches like `mark_join` that preprocess the left.
inline bool should_skip_large_sizes(nvbench::state& state, bool build_is_right = true)
{
  if (state.get_int64("skip_large_sizes") == 0) { return false; }
  auto const left_size  = state.get_int64("left_size");
  auto const right_size = state.get_int64("right_size");
  if (build_is_right && right_size > left_size) {
    state.skip("build (right) should be smaller than probe (left)");
    return true;
  }
  if (!build_is_right && left_size > right_size) {
    state.skip("build (left) should be smaller than probe (right)");
    return true;
  }
  return false;
}

template <bool Nullable,
          join_t join_type                  = join_t::HASH,
          cudf::null_equality compare_nulls = cudf::null_equality::UNEQUAL,
          typename state_type,
          typename Join>
void BM_join(state_type& state,
             std::vector<cudf::type_id>& key_types,
             Join JoinFunc,
             int multiplicity    = 1,
             double selectivity  = 0.3,
             bool build_is_right = true)
{
  if (should_skip_large_sizes(state, build_is_right)) { return; }

  auto const right_size = static_cast<size_t>(state.get_int64("right_size"));
  auto const left_size  = static_cast<size_t>(state.get_int64("left_size"));

  auto const num_keys             = key_types.size();
  auto const num_payload_cols     = 2;
  auto [build_table, probe_table] = generate_input_tables<Nullable>(
    key_types, right_size, left_size, num_payload_cols, multiplicity, selectivity);
  auto const probe_view = probe_table->view();
  auto const build_view = build_table->view();

  auto const join_input_size = estimate_size(build_view) + estimate_size(probe_view);

  // Setup join parameters and result table
  std::vector<cudf::size_type> columns_to_join(num_keys);
  std::iota(columns_to_join.begin(), columns_to_join.end(), 0);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  if constexpr (join_type == join_t::HASH || join_type == join_t::SORT_MERGE) {
    state.add_element_count(join_input_size, "join_input_size");  // number of bytes
    state.template add_global_memory_reads<nvbench::int8_t>(join_input_size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto result = JoinFunc(
        probe_view.select(columns_to_join), build_view.select(columns_to_join), compare_nulls);
    });
    set_throughputs(state);
  }
  if constexpr (join_type == join_t::CONDITIONAL) {
    auto const col_ref_left_0  = cudf::ast::column_reference(0);
    auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
    auto left_zero_eq_right_zero =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_left_0, col_ref_right_0);
    state.add_element_count(join_input_size, "join_input_size");  // number of bytes
    state.template add_global_memory_reads<nvbench::int8_t>(join_input_size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto result = JoinFunc(probe_view, build_view, left_zero_eq_right_zero, compare_nulls);
      ;
    });
    set_throughputs(state);
  }
  if constexpr (join_type == join_t::MIXED) {
    auto const col_ref_left_0  = cudf::ast::column_reference(0);
    auto const col_ref_right_0 = cudf::ast::column_reference(0, cudf::ast::table_reference::RIGHT);
    auto left_zero_eq_right_zero =
      cudf::ast::operation(cudf::ast::ast_operator::EQUAL, col_ref_left_0, col_ref_right_0);
    state.add_element_count(join_input_size, "join_input_size");  // number of bytes
    state.template add_global_memory_reads<nvbench::int8_t>(join_input_size);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      auto result = JoinFunc(probe_view.select(std::vector<cudf::size_type>(
                               columns_to_join.begin(), columns_to_join.begin() + num_keys / 2)),
                             build_view.select(std::vector<cudf::size_type>(
                               columns_to_join.begin(), columns_to_join.begin() + num_keys / 2)),
                             probe_view.select(std::vector<cudf::size_type>(
                               columns_to_join.begin() + num_keys / 2, columns_to_join.end())),
                             build_view.select(std::vector<cudf::size_type>(
                               columns_to_join.begin() + num_keys / 2, columns_to_join.end())),
                             left_zero_eq_right_zero,
                             compare_nulls);
    });
    set_throughputs(state);
  }
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/join/join_common.hpp>

#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>

auto const num_keys = 2;

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_filter_join_indices_inner_join(nvbench::state& state,
                                            nvbench::type_list<nvbench::enum_type<Nullable>,
                                                               nvbench::enum_type<NullEquality>,
                                                               nvbench::enum_type<DataType>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls) {
    auto hash_joiner                   = cudf::hash_join(right_equality_input, compare_nulls);
    auto [left_indices, right_indices] = hash_joiner.inner_join(left_equality_input);

    return cudf::filter_join_indices(left_conditional_input,
                                     right_conditional_input,
                                     cudf::device_span<cudf::size_type const>(*left_indices),
                                     cudf::device_span<cudf::size_type const>(*right_indices),
                                     binary_pred,
                                     cudf::join_kind::INNER_JOIN);
  };

  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);
  BM_join<Nullable, join_t::MIXED, NullEquality>(state, dtypes, join);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_filter_join_indices_inner_join_complex_ast(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<Nullable>,
                     nvbench::enum_type<NullEquality>,
                     nvbench::enum_type<DataType>>)
{
  auto const ast_levels = static_cast<cudf::size_type>(state.get_int64("ast_levels"));

  auto join = [ast_levels](cudf::table_view const& left_equality_input,
                           cudf::table_view const& right_equality_input,
                           cudf::table_view const& left_conditional_input,
                           cudf::table_view const& right_conditional_input,
                           cudf::ast::operation binary_pred,
                           cudf::null_equality compare_nulls) {
    cudf::ast::tree tree;
    create_complex_ast_expression(tree, ast_levels);

    auto hash_joiner                   = cudf::hash_join(right_equality_input, compare_nulls);
    auto [left_indices, right_indices] = hash_joiner.inner_join(left_equality_input);

    return cudf::filter_join_indices(left_conditional_input,
                                     right_conditional_input,
                                     cudf::device_span<cudf::size_type const>(*left_indices),
                                     cudf::device_span<cudf::size_type const>(*right_indices),
                                     tree.back(),
                                     cudf::join_kind::INNER_JOIN);
  };

  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);
  BM_join<Nullable, join_t::MIXED, NullEquality>(state, dtypes, join);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_filter_join_indices_left_join(nvbench::state& state,
                                           nvbench::type_list<nvbench::enum_type<Nullable>,
                                                              nvbench::enum_type<NullEquality>,
                                                              nvbench::enum_type<DataType>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls) {
    auto hash_joiner                   = cudf::hash_join(right_equality_input, compare_nulls);
    auto [left_indices, right_indices] = hash_joiner.left_join(left_equality_input);

    return cudf::filter_join_indices(left_conditional_input,
                                     right_conditional_input,
                                     cudf::device_span<cudf::size_type const>(*left_indices),
                                     cudf::device_span<cudf::size_type const>(*right_indices),
                                     binary_pred,
                                     cudf::join_kind::LEFT_JOIN);
  };

  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);
  BM_join<Nullable, join_t::MIXED, NullEquality>(state, dtypes, join);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_filter_join_indices_full_join(nvbench::state& state,
                                           nvbench::type_list<nvbench::enum_type<Nullable>,
                                                              nvbench::enum_type<NullEquality>,
                                                              nvbench::enum_type<DataType>>)
{
  auto join = [](cudf::table_view const& left_equality_input,
                 cudf::table_view const& right_equality_input,
                 cudf::table_view const& left_conditional_input,
                 cudf::table_view const& right_conditional_input,
                 cudf::ast::operation binary_pred,
                 cudf::null_equality compare_nulls) {
    auto hash_joiner                   = cudf::hash_join(right_equality_input, compare_nulls);
    auto [left_indices, right_indices] = hash_joiner.full_join(left_equality_input);

    return cudf::filter_join_indices(left_conditional_input,
                                     right_conditional_input,
                                     cudf::device_span<cudf::size_type const>(*left_indices),
                                     cudf::device_span<cudf::size_type const>(*right_indices),
                                     binary_pred,
                                     cudf::join_kind::FULL_JOIN);
  };

  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);
  BM_join<Nullable, join_t::MIXED, NullEquality>(state, dtypes, join);
}

NVBENCH_BENCH_TYPES(nvbench_filter_join_indices_inner_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("filter_join_indices_inner_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("skip_large_sizes", {1});

NVBENCH_BENCH_TYPES(nvbench_filter_join_indices_inner_join_complex_ast,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("filter_join_indices_inner_join_complex_ast")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("ast_levels", {1, 5, 10})
  .add_int64_axis("skip_large_sizes", {1});

NVBENCH_BENCH_TYPES(nvbench_filter_join_indices_left_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("filter_join_indices_left_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("skip_large_sizes", {1});

NVBENCH_BENCH_TYPES(nvbench_filter_join_indices_full_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("filter_join_indices_full_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("skip_large_sizes", {1});

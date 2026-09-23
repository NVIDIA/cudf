/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join_common.hpp"

#include <cudf/join/distinct_hash_join.hpp>

double constexpr LOAD_FACTOR = 0.5;
auto const num_keys          = 1;

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_distinct_inner_join(nvbench::state& state,
                                 nvbench::type_list<nvbench::enum_type<Nullable>,
                                                    nvbench::enum_type<NullEquality>,
                                                    nvbench::enum_type<DataType>>)
{
  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    auto hj_obj = cudf::distinct_hash_join{right_input, compare_nulls, LOAD_FACTOR};
    return hj_obj.inner_join(left_input);
  };

  BM_join<Nullable, join_t::HASH, NullEquality>(state, dtypes, join);
}

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType>
void nvbench_distinct_left_join(nvbench::state& state,
                                nvbench::type_list<nvbench::enum_type<Nullable>,
                                                   nvbench::enum_type<NullEquality>,
                                                   nvbench::enum_type<DataType>>)
{
  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    auto hj_obj = cudf::distinct_hash_join{right_input, compare_nulls, LOAD_FACTOR};
    return hj_obj.left_join(left_input);
  };

  BM_join<Nullable, join_t::HASH, NullEquality>(state, dtypes, join);
}

NVBENCH_BENCH_TYPES(nvbench_distinct_inner_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("distinct_inner_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("skip_large_sizes", {1});

NVBENCH_BENCH_TYPES(nvbench_distinct_left_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      DEFAULT_JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES))
  .set_name("distinct_left_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType"})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("skip_large_sizes", {1});

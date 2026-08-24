/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/join/join_common.hpp>

#include <cudf/column/column.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include <memory>
#include <numeric>
#include <vector>

namespace {

template <bool Nullable, cudf::null_equality NullEquality, typename Join>
void BM_join_dictionary(nvbench::state& state, Join join_func)
{
  if (should_skip_large_sizes(state)) { return; }

  auto const right_size      = static_cast<cudf::size_type>(state.get_int64("right_size"));
  auto const left_size       = static_cast<cudf::size_type>(state.get_int64("left_size"));
  auto const num_keys        = static_cast<std::size_t>(state.get_int64("num_keys"));
  auto const multiplicity    = static_cast<int>(state.get_int64("multiplicity"));
  auto constexpr selectivity = 0.3;

  auto stream = cudf::get_default_stream();

  // Generate plain INT32 key tables with controlled selectivity/cardinality, then dictionary-encode
  // each key column on both sides. No payload columns are added; all columns participate in the
  // join.
  std::vector<cudf::type_id> const key_types(num_keys, cudf::type_id::INT32);
  auto [build_table, probe_table] =
    generate_input_tables<Nullable>(key_types, right_size, left_size, 0, multiplicity, selectivity);

  auto encode_table = [&stream](cudf::table_view const& input) {
    std::vector<std::unique_ptr<cudf::column>> cols;
    for (auto const& col : input) {
      cols.push_back(cudf::dictionary::encode(col, cudf::data_type{cudf::type_id::INT32}, stream));
    }
    return cols;
  };
  auto const build_cols = encode_table(build_table->view());
  auto const probe_cols = encode_table(probe_table->view());

  auto to_view = [](auto const& cols) {
    std::vector<cudf::column_view> views;
    for (auto const& col : cols) {
      views.push_back(col->view());
    }
    return cudf::table_view{views};
  };
  auto const build_view = to_view(build_cols);
  auto const probe_view = to_view(probe_cols);

  // row_bit_count (used by estimate_size) does not support dictionary columns, so sum the encoded
  // columns' allocation sizes instead.
  auto sum_alloc_size = [](auto const& cols) {
    return std::accumulate(cols.begin(), cols.end(), int64_t{0}, [](int64_t acc, auto const& col) {
      return acc + static_cast<int64_t>(col->alloc_size());
    });
  };
  auto const join_input_size = sum_alloc_size(build_cols) + sum_alloc_size(probe_cols);
  state.add_element_count(join_input_size, "join_input_size");
  state.add_global_memory_reads<nvbench::int8_t>(join_input_size);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = join_func(probe_view, build_view, NullEquality);
  });

  set_throughputs(state);
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

template <bool Nullable, cudf::null_equality NullEquality>
void nvbench_inner_join_dictionary(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<Nullable>, nvbench::enum_type<NullEquality>>)
{
  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    return cudf::inner_join(left_input, right_input, compare_nulls);
  };
  BM_join_dictionary<Nullable, NullEquality>(state, join);
}

template <bool Nullable, cudf::null_equality NullEquality>
void nvbench_left_join_dictionary(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<Nullable>, nvbench::enum_type<NullEquality>>)
{
  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    return cudf::left_join(left_input, right_input, compare_nulls);
  };
  BM_join_dictionary<Nullable, NullEquality>(state, join);
}

template <bool Nullable, cudf::null_equality NullEquality>
void nvbench_full_join_dictionary(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<Nullable>, nvbench::enum_type<NullEquality>>)
{
  auto join = [](cudf::table_view const& left_input,
                 cudf::table_view const& right_input,
                 cudf::null_equality compare_nulls) {
    return cudf::full_join(left_input, right_input, compare_nulls);
  };
  BM_join_dictionary<Nullable, NullEquality>(state, join);
}

NVBENCH_BENCH_TYPES(nvbench_inner_join_dictionary,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE, DEFAULT_JOIN_NULL_EQUALITY))
  .set_name("inner_join_dictionary")
  .set_type_axes_names({"Nullable", "NullEquality"})
  .add_int64_axis("num_keys", {1})
  .add_int64_axis("multiplicity", {1, 100})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("skip_large_sizes", {1});

NVBENCH_BENCH_TYPES(nvbench_left_join_dictionary,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE, DEFAULT_JOIN_NULL_EQUALITY))
  .set_name("left_join_dictionary")
  .set_type_axes_names({"Nullable", "NullEquality"})
  .add_int64_axis("num_keys", {1})
  .add_int64_axis("multiplicity", {1, 100})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("skip_large_sizes", {1});

NVBENCH_BENCH_TYPES(nvbench_full_join_dictionary,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE, DEFAULT_JOIN_NULL_EQUALITY))
  .set_name("full_join_dictionary")
  .set_type_axes_names({"Nullable", "NullEquality"})
  .add_int64_axis("num_keys", {1})
  .add_int64_axis("multiplicity", {1, 100})
  .add_int64_axis("left_size", JOIN_SIZE_RANGE)
  .add_int64_axis("right_size", JOIN_SIZE_RANGE)
  .add_int64_axis("skip_large_sizes", {1});

/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_nested_types.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/lists/lists_column_view.hpp>
#include <cudf/sorting.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

namespace {
constexpr cudf::size_type min_val = 0;
constexpr cudf::size_type max_val = 10;

void sort_multiple_lists(nvbench::state& state)
{
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const input_table = create_lists_data(state, num_columns, min_val, max_val);
  auto const stream      = cudf::get_default_stream();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    cudf::sorted_order(*input_table, {}, {}, stream, cudf::get_current_device_resource_ref());
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void sort_lists_of_structs(nvbench::state& state)
{
  auto const num_columns = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const lists_table = create_lists_data(state, num_columns, min_val, max_val);

  // After having a table of (multiple) lists columns, convert those lists columns into lists of
  // structs columns. The children of these structs columns are also children of the original lists
  // columns.
  // Such resulted lists-of-structs columns are very similar to the original lists-of-integers
  // columns so their benchmarks can be somewhat comparable.
  std::vector<cudf::column_view> lists_of_structs;
  for (auto const& col : lists_table->view()) {
    auto const child = col.child(cudf::lists_column_view::child_column_index);

    // Put the child column under a struct column having the same null mask/null count.
    auto const new_child = cudf::column_view{cudf::data_type{cudf::type_id::STRUCT},
                                             child.size(),
                                             nullptr,
                                             child.null_mask(),
                                             child.null_count(),
                                             child.offset(),
                                             {child}};
    auto const converted_col =
      cudf::column_view{cudf::data_type{cudf::type_id::LIST},
                        col.size(),
                        nullptr,
                        col.null_mask(),
                        col.null_count(),
                        col.offset(),
                        {col.child(cudf::lists_column_view::offsets_column_index), new_child}};
    lists_of_structs.push_back(converted_col);
  }

  auto const input_table = cudf::table_view{lists_of_structs};
  auto const stream      = cudf::get_default_stream();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream_view{launch.get_stream()};
    cudf::sorted_order(input_table, {}, {}, stream, cudf::get_current_device_resource_ref());
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

}  // namespace

void nvbench_sort_lists(nvbench::state& state)
{
  auto const has_lists_of_structs = state.get_int64("lists_of_structs") > 0;
  if (has_lists_of_structs) {
    sort_lists_of_structs(state);
  } else {
    sort_multiple_lists(state);
  }
}

NVBENCH_BENCH(nvbench_sort_lists)
  .set_name("sort_list")
  .add_int64_power_of_two_axis("size_bytes", {18, 24, 28})
  .add_int64_axis("depth", {1, 4})
  .add_int64_axis("num_columns", {1})
  .add_int64_axis("lists_of_structs", {0, 1})
  .add_float64_axis("null_frequency", {0, 0.2});

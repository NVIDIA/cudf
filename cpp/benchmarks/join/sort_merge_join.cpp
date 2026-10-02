/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/join/join_common.hpp>
#include <benchmarks/join/nvbench_helpers.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/join_factorizer.hpp>
#include <cudf/join/sort_merge_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

/**
 * Sort-merge join benchmark with multiple data types.
 * Tests both HASH and SORT_MERGE algorithms to compare performance
 * across different data types including strings and nested types.
 * Optionally applies key remapping before the join.
 */

template <bool Nullable, cudf::null_equality NullEquality, data_type DataType, join_t Algorithm>
void nvbench_sort_merge_inner_join(nvbench::state& state,
                                   nvbench::type_list<nvbench::enum_type<Nullable>,
                                                      nvbench::enum_type<NullEquality>,
                                                      nvbench::enum_type<DataType>,
                                                      nvbench::enum_type<Algorithm>>)
{
  if constexpr (not Nullable && NullEquality == cudf::null_equality::UNEQUAL) {
    state.skip("Non-nullable with NULLS_UNEQUAL is redundant");
    return;
  }

  auto const num_keys     = state.get_int64("num_keys");
  auto const use_remap    = state.get_int64("use_factorizer") != 0;
  auto const left_size    = static_cast<cudf::size_type>(state.get_int64("left_size"));
  auto const right_size   = static_cast<cudf::size_type>(state.get_int64("right_size"));
  auto const multiplicity = 1;
  auto const selectivity  = 0.3;

  if (should_skip_large_sizes(state)) { return; }

  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto constexpr NUM_PAYLOAD_COLS = 2;
  // generate_input_tables returns (build_table, probe_table) = (right, left)
  auto [right_table, left_table] = generate_input_tables<Nullable>(
    dtypes, right_size, left_size, NUM_PAYLOAD_COLS, multiplicity, selectivity);

  auto const left_view  = left_table->view();
  auto const right_view = right_table->view();

  std::vector<cudf::size_type> columns_to_join(num_keys);
  std::iota(columns_to_join.begin(), columns_to_join.end(), 0);

  auto const left_keys  = left_view.select(columns_to_join);
  auto const right_keys = right_view.select(columns_to_join);

  auto const join_input_size = estimate_size(left_view) + estimate_size(right_view);
  state.add_element_count(join_input_size, "join_input_size");
  state.add_global_memory_reads<nvbench::int8_t>(join_input_size);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));

  if (use_remap) {
    // Benchmark with key remapping
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      // Step 1: Create factorizer from right keys (with statistics disabled, the statistics
      //  need to be calculated for the join type selection heuristic, either way)
      cudf::join_factorizer remap(
        right_keys, NullEquality, cudf::join_statistics::SKIP, cudf::get_default_stream());

      // Step 2: Remap left and right keys to integers
      auto remapped_right = remap.factorize_right_keys();
      auto remapped_left  = remap.factorize_left_keys(left_keys);

      // Step 3: Create table views from remapped columns
      cudf::table_view remapped_left_view({remapped_left->view()});
      cudf::table_view remapped_right_view({remapped_right->view()});

      // Step 4: Perform the join on remapped integer keys
      if constexpr (Algorithm == join_t::HASH) {
        [[maybe_unused]] auto result =
          cudf::inner_join(remapped_left_view, remapped_right_view, NullEquality);
      } else if constexpr (Algorithm == join_t::SORT_MERGE) {
        auto smj = cudf::sort_merge_join(remapped_right_view, cudf::sorted::NO, NullEquality);
        [[maybe_unused]] auto result = smj.inner_join(remapped_left_view);
      }
    });
  } else {
    // Benchmark without key remapping (direct join)
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      if constexpr (Algorithm == join_t::HASH) {
        [[maybe_unused]] auto result = cudf::inner_join(left_keys, right_keys, NullEquality);
      } else if constexpr (Algorithm == join_t::SORT_MERGE) {
        auto smj = cudf::sort_merge_join(right_keys, cudf::sorted::NO, NullEquality);
        [[maybe_unused]] auto result = smj.inner_join(left_keys);
      }
    });
  }

  set_throughputs(state);
}

// Sort-merge inner join with multiple data types and optional key remapping
NVBENCH_BENCH_TYPES(nvbench_sort_merge_inner_join,
                    NVBENCH_TYPE_AXES(JOIN_NULLABLE_RANGE,
                                      JOIN_NULL_EQUALITY,
                                      DEFAULT_JOIN_DATATYPES,
                                      JOIN_ALGORITHM))
  .set_name("sort_merge_inner_join")
  .set_type_axes_names({"Nullable", "NullEquality", "DataType", "Algorithm"})
  .add_int64_axis("num_keys", nvbench::range(1, 3, 1))
  .add_int64_axis("left_size", {10'000, 100'000})
  .add_int64_axis("right_size", {10'000, 100'000})
  .add_int64_axis("use_factorizer", {0, 1})
  .add_int64_axis("skip_large_sizes", {1});

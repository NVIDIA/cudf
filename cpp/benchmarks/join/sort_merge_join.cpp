/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/join/join_common.hpp>
#include <benchmarks/join/nvbench_helpers.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/join/join.hpp>
#include <cudf/join/key_remapping.hpp>
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
  auto const use_remap    = state.get_int64("use_key_remap") != 0;
  auto const left_size    = static_cast<cudf::size_type>(state.get_int64("left_size"));
  auto const right_size   = static_cast<cudf::size_type>(state.get_int64("right_size"));
  auto const multiplicity = 1;
  auto const selectivity  = 0.3;

  if (should_skip_large_sizes(state)) { return; }

  auto dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);

  auto constexpr NUM_PAYLOAD_COLS = 2;
  auto [build_table, probe_table] = generate_input_tables<Nullable>(
    dtypes, right_size, left_size, NUM_PAYLOAD_COLS, multiplicity, selectivity);

  auto const build_view = build_table->view();
  auto const probe_view = probe_table->view();

  std::vector<cudf::size_type> columns_to_join(num_keys);
  std::iota(columns_to_join.begin(), columns_to_join.end(), 0);

  auto const build_keys = build_view.select(columns_to_join);
  auto const probe_keys = probe_view.select(columns_to_join);

  auto const join_input_size = estimate_size(build_view) + estimate_size(probe_view);
  state.add_element_count(join_input_size, "join_input_size");
  state.add_global_memory_reads<nvbench::int8_t>(join_input_size);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));

  if (use_remap) {
    // Benchmark with key remapping
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      // Step 1: Build key remapping (with metrics disabled, the metrics need to be calculated
      //  for the join type selection heuristic, either way)
      cudf::key_remapping remap(
        build_keys, NullEquality, cudf::compute_metrics::NO, cudf::get_default_stream());

      // Step 2: Remap build and probe keys to integers
      auto remapped_build = remap.remap_right_keys();
      auto remapped_probe = remap.remap_left_keys(probe_keys);

      // Step 3: Create table views from remapped columns
      cudf::table_view remapped_build_view({remapped_build->view()});
      cudf::table_view remapped_probe_view({remapped_probe->view()});

      // Step 4: Perform the join on remapped integer keys
      if constexpr (Algorithm == join_t::HASH) {
        [[maybe_unused]] auto result =
          cudf::inner_join(remapped_probe_view, remapped_build_view, NullEquality);
      } else if constexpr (Algorithm == join_t::SORT_MERGE) {
        auto smj = cudf::sort_merge_join(remapped_build_view, cudf::sorted::NO, NullEquality);
        [[maybe_unused]] auto result = smj.inner_join(remapped_probe_view);
      }
    });
  } else {
    // Benchmark without key remapping (direct join)
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
      if constexpr (Algorithm == join_t::HASH) {
        [[maybe_unused]] auto result = cudf::inner_join(probe_keys, build_keys, NullEquality);
      } else if constexpr (Algorithm == join_t::SORT_MERGE) {
        auto smj = cudf::sort_merge_join(build_keys, cudf::sorted::NO, NullEquality);
        [[maybe_unused]] auto result = smj.inner_join(probe_keys);
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
  .add_int64_axis("use_key_remap", {0, 1})
  .add_int64_axis("skip_large_sizes", {1});

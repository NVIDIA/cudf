/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/join/join_common.hpp>
#include <benchmarks/join/nvbench_helpers.hpp>

#include <cudf/copying.hpp>
#include <cudf/join/streaming_hash_join.hpp>

#include <numeric>
#include <vector>

// Measures one complete streaming join: construction, inserting every right batch, probing the
// left table, and teardown.
template <bool Nullable, data_type DataType>
void nvbench_streaming_inner_join(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<Nullable>, nvbench::enum_type<DataType>>)
{
  if (should_skip_large_sizes(state)) { return; }

  auto const right_size   = static_cast<cudf::size_type>(state.get_int64("right_size"));
  auto const left_size    = static_cast<cudf::size_type>(state.get_int64("left_size"));
  auto const num_batches  = static_cast<cudf::size_type>(state.get_int64("num_batches"));
  auto const multiplicity = static_cast<cudf::size_type>(state.get_int64("multiplicity"));
  auto const num_keys     = state.get_int64("num_keys");
  if (num_batches > right_size) {
    state.skip("every batch must have at least one row");
    return;
  }

  auto const dtypes = cycle_dtypes(get_type_or_group(static_cast<int32_t>(DataType)), num_keys);
  auto [build_table, probe_table] = generate_input_tables<Nullable>(
    dtypes, right_size, left_size, /*num_payload_cols=*/2, multiplicity, /*selectivity=*/0.3);

  std::vector<cudf::size_type> key_indices(num_keys);
  std::iota(key_indices.begin(), key_indices.end(), 0);
  auto const build_view = build_table->view();
  auto const probe_keys = probe_table->view().select(key_indices);

  std::vector<cudf::size_type> splits;
  for (cudf::size_type batch = 1; batch < num_batches; ++batch) {
    splits.push_back(static_cast<cudf::size_type>(int64_t{right_size} * batch / num_batches));
  }
  auto const batches = cudf::split(build_view, splits);

  auto const join_input_size = estimate_size(build_view) + estimate_size(probe_keys);
  state.add_element_count(join_input_size, "join_input_size");
  state.add_global_memory_reads<nvbench::int8_t>(join_input_size);

  auto const stream = cudf::get_default_stream();
  stream.sync();  // Keep input construction out of both metrics.
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::streaming_hash_join join{build_view,
                                   key_indices,
                                   right_size,
                                   num_batches,
                                   Nullable ? cudf::nullable_join::YES : cudf::nullable_join::NO,
                                   cudf::null_equality::UNEQUAL,
                                   /*load_factor=*/0.5,
                                   stream};
    for (auto const& batch : batches) {
      join.insert(batch, stream);
    }
    [[maybe_unused]] auto const result = join.inner_join(probe_keys, {}, stream);
  });

  set_throughputs(state);
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(
  nvbench_streaming_inner_join,
  NVBENCH_TYPE_AXES(
    JOIN_NULLABLE_RANGE,
    nvbench::
      enum_type_list<data_type::INT32, data_type::INT64, data_type::STRING, data_type::STRUCT>))
  .set_name("streaming_inner_join")
  .set_type_axes_names({"Nullable", "DataType"})
  .add_int64_axis("num_keys", {1})
  .add_int64_axis("left_size", {100'000, 10'000'000})
  .add_int64_axis("right_size", {100'000, 10'000'000})
  .add_int64_axis("num_batches", {1, 16, 128})
  .add_int64_axis("multiplicity", {1, 10, 100})
  .add_int64_axis("skip_large_sizes", {1});

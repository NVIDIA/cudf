/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/copying.hpp>
#include <cudf/detail/batch_gather.hpp>
#include <cudf/filling.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

#include <limits>

// Identical datagen to the `gather` benchmark so the two compare directly in the same
// COPYING_NVBENCH binary: coalesced maps are reversed sequences, random maps are uniform.
static void bench_batch_gather(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const coalesce = static_cast<bool>(state.get_int64("coalesce"));

  if (static_cast<long>(num_rows) * static_cast<long>(num_cols) >
      std::numeric_limits<int32_t>::max()) {  // reasonable limit
    state.skip("input benchmark size too large");
    return;
  }

  auto gather_map = [&] {
    if (coalesce) {
      return cudf::sequence(num_rows,
                            cudf::numeric_scalar<cudf::size_type>(num_rows - 1),
                            cudf::numeric_scalar<cudf::size_type>(-1));
    }

    data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
      cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0, num_rows - 1);
    return create_random_column(cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile);
  }();

  // Every element is valid
  auto source_table = create_sequence_table(cycle_dtypes({cudf::type_to_id<int64_t>()}, num_cols),
                                            row_count{num_rows});

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_reads<int8_t>(source_table->alloc_size());
  state.add_global_memory_writes<int8_t>(source_table->alloc_size());

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    cudf::detail::batch_gather(source_table->view(),
                               gather_map->view(),
                               cudf::out_of_bounds_policy::DONT_CHECK,
                               cudf::negative_index_policy::NOT_ALLOWED,
                               stream,
                               cudf::memory_resources{cudf::get_current_device_resource_ref()});
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_batch_gather)
  .set_name("batch_gather")
  .add_int64_axis("num_rows", {65536, 262144, 1048576})
  .add_int64_axis("num_cols", {8, 100, 256, 512})
  .add_int64_axis("coalesce", {true, false});

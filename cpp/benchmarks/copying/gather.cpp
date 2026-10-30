/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>
#include <cudf_test/iterator_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/filling.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

static void bench_gather(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const map_kind = state.get_string("map_kind");

  if (static_cast<long>(num_rows) * static_cast<long>(num_cols) >
      std::numeric_limits<int32_t>::max()) {  // reasonable limit
    state.skip("input benchmark size too large");
    return;
  }

  // Map generators are expression-identical to the `batch_gather_routed` bench so a runner can
  // pair stock/batch cells by name with byte-identical maps: "reversed" and "random" reproduce
  // the prior `coalesce` bool's generators (the prior cells are map_kind=reversed/random),
  // "sequence" is the ascending coalesced map, and "fanout2"/"periodic9" are reuse-heavy
  // small-spread maps the router must not stream.
  auto const stream = cudf::get_default_stream();
  auto const mr     = cudf::memory_resources{cudf::get_current_device_resource_ref()};
  auto gather_map   = [&]() -> std::unique_ptr<cudf::column> {
    if (map_kind == "sequence") {
      return cudf::sequence(num_rows,
                            cudf::numeric_scalar<cudf::size_type>(0),
                            cudf::numeric_scalar<cudf::size_type>(1));
    }
    if (map_kind == "reversed") {
      return cudf::sequence(num_rows,
                            cudf::numeric_scalar<cudf::size_type>(num_rows - 1),
                            cudf::numeric_scalar<cudf::size_type>(-1));
    }
    if (map_kind == "random") {
      data_profile const profile = data_profile_builder().cardinality(0).no_validity().distribution(
        cudf::type_to_id<cudf::size_type>(), distribution_id::UNIFORM, 0, num_rows - 1);
      return create_random_column(
        cudf::type_to_id<cudf::size_type>(), row_count{num_rows}, profile);
    }
    // Reuse-heavy maps with small per-window spread: fanout2 reads each element of the first
    // half twice; periodic9 pins the whole gather to nine rows.
    if (map_kind == "fanout2") {
      auto const elems = cudf::detail::make_counting_transform_iterator(
        0, [](cudf::size_type e) { return static_cast<cudf::size_type>(e / 2); });
      return cudf::test::fixed_width_column_wrapper<cudf::size_type>(
               elems, elems + num_rows, cudf::test::iterators::no_nulls(), stream, mr)
        .release();
    }
    if (map_kind == "periodic9") {
      auto const elems = cudf::detail::make_counting_transform_iterator(
        0, [](cudf::size_type e) { return static_cast<cudf::size_type>(e % 9); });
      return cudf::test::fixed_width_column_wrapper<cudf::size_type>(
               elems, elems + num_rows, cudf::test::iterators::no_nulls(), stream, mr)
        .release();
    }
    CUDF_FAIL("Unhandled map_kind", std::invalid_argument);
  }();

  // Every element is valid
  auto source_table = create_sequence_table(cycle_dtypes({cudf::type_to_id<int64_t>()}, num_cols),
                                            row_count{num_rows});

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  state.add_global_memory_reads<int8_t>(source_table->alloc_size());
  state.add_global_memory_writes<int8_t>(source_table->alloc_size());

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { cudf::gather(*source_table, gather_map->view()); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_gather)
  .set_name("gather")
  .add_int64_axis("num_rows", {4096, 32768, 65536, 262144, 1048576, 2097152, 16777216})
  .add_int64_axis("num_cols", {1, 8, 100, 256, 512, 1000})
  .add_string_axis("map_kind", {"sequence", "reversed", "random", "fanout2", "periodic9"});

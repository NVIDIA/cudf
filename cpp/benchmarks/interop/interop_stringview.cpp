/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/nanoarrow_utils.hpp>

#include <cudf/interop.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

void BM_from_arrow_host_stringview(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));

  auto stream = cudf::get_default_stream();

  std::string characters('x', max_width);  // actual data is not important
  std::random_device rd;
  std::mt19937 generator(rd());
  std::uniform_int_distribution<> distribution(min_width, max_width);

  ArrowArray input;
  NANOARROW_THROW_NOT_OK(ArrowArrayInitFromType(&input, NANOARROW_TYPE_STRING_VIEW));
  NANOARROW_THROW_NOT_OK(ArrowArrayStartAppending(&input));
  auto total_size = 0L;
  for (auto i = 0; i < num_rows; ++i) {
    auto const size = distribution(generator);
    auto const ptr  = characters.data();
    total_size += size;
    NANOARROW_THROW_NOT_OK(ArrowArrayAppendString(&input, {ptr, size}));
  }
  NANOARROW_THROW_NOT_OK(
    ArrowArrayFinishBuilding(&input, NANOARROW_VALIDATION_LEVEL_NONE, nullptr));

  state.add_element_count(num_rows, "num_rows");
  state.add_global_memory_reads(total_size);
  state.add_global_memory_writes(total_size);
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  ArrowSchema schema;
  NANOARROW_THROW_NOT_OK(ArrowSchemaInitFromType(&schema, NANOARROW_TYPE_STRING_VIEW));

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = cudf::from_arrow_column(&schema, &input);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(BM_from_arrow_host_stringview)
  .set_name("from_arrow_host_stringview")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000})
  .add_int64_axis("min_width", {1})
  .add_int64_axis("max_width", {10, 100, 1000});

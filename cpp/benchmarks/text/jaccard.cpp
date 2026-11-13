/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/strings/combine.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtext/jaccard.hpp>

#include <rmm/device_buffer.hpp>

#include <nvbench/nvbench.cuh>

static void bench_jaccard(nvbench::state& state)
{
  auto const num_rows        = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width       = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width       = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const substring_width = static_cast<cudf::size_type>(state.get_int64("substring_width"));

  if (static_cast<int64_t>(num_rows) * max_width > (1L << 30)) {
    state.skip("Skip benchmarks larger than 1GB");
    return;
  }

  // Each pair of rows shares a common half so the intersection counts are not trivially 0:
  //   input1 = A + B and input2 = A + C
  data_profile const strings_profile =
    data_profile_builder()
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, min_width / 2, max_width / 2)
      .no_validity();
  auto const parts =
    create_random_table({cudf::type_id::STRING, cudf::type_id::STRING, cudf::type_id::STRING},
                        row_count{num_rows},
                        strings_profile);
  auto const col1 = cudf::strings::concatenate(
    cudf::table_view({parts->view().column(0), parts->view().column(1)}));
  auto const col2 = cudf::strings::concatenate(
    cudf::table_view({parts->view().column(0), parts->view().column(2)}));
  cudf::strings_column_view input1(col1->view());
  cudf::strings_column_view input2(col2->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));

  auto chars_size = input1.chars_size(stream) + input2.chars_size(stream);
  state.add_global_memory_reads<nvbench::int8_t>(chars_size);
  state.add_global_memory_writes<nvbench::float32_t>(num_rows);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto result = nvtext::jaccard_index(input1, input2, substring_width);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_jaccard)
  .set_name("jaccard")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {128, 512, 2048, 8192, 32768})
  .add_int64_axis("num_rows", {32768, 131072, 262144})
  .add_int64_axis("substring_width", {5, 10});

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/slice.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

#include <limits>

static void bench_slice(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const row_width = static_cast<cudf::size_type>(state.get_int64("row_width"));
  auto const stype     = state.get_string("type");

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, 0, row_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);
  cudf::strings_column_view input(column->view());
  auto starts_itr = cuda::constant_iterator<cudf::size_type>(row_width / 4);
  auto starts =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(starts_itr, starts_itr + num_rows);
  auto stops_itr = cuda::constant_iterator<cudf::size_type>(row_width / 3);
  auto stops =
    cudf::test::fixed_width_column_wrapper<cudf::size_type>(stops_itr, stops_itr + num_rows);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  // gather some throughput statistics as well
  auto const data_size = column->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);  // all bytes are read
  auto output_size = (row_width / 3 - row_width / 4) * num_rows;
  state.add_global_memory_writes<nvbench::int8_t>(output_size);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  if (stype == "multi") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::slice_strings(input, starts, stops, stream);
    });
  } else {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::slice_strings(input,
                                   std::optional<cudf::size_type>(row_width / 4),
                                   std::optional<cudf::size_type>(row_width / 3),
                                   std::optional<cudf::size_type>(1),
                                   stream);
    });
  }
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  set_throughputs(state);
}

NVBENCH_BENCH(bench_slice)
  .set_name("slice")
  .add_int64_axis("row_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("type", {"position", "multi"});

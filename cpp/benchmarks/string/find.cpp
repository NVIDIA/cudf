/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/generate_skewed_data.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/find.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

static void bench_find_string(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const hit_rate  = static_cast<cudf::size_type>(state.get_int64("hit_rate"));
  auto const api       = state.get_string("api");
  auto const tgt_type  = state.get_string("target");

  auto const stream = cudf::get_default_stream();
  auto const col    = create_string_column(num_rows, max_width, hit_rate);
  auto const input  = cudf::strings_column_view(col->view());

  auto target        = cudf::string_scalar("0987 5W43");
  auto targets_col   = cudf::make_column_from_scalar(target, num_rows);
  auto const targets = cudf::strings_column_view(targets_col->view());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const data_size = col->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  if (api == "find") {
    state.add_global_memory_writes<nvbench::int32_t>(input.size());
  } else {
    state.add_global_memory_writes<nvbench::int8_t>(input.size());
  }

  auto const mem_stats_logger = cudf::memory_stats_logger();
  if (api == "find") {
    if (tgt_type == "scalar") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::find(input, target); });
    } else if (tgt_type == "column") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::find(input, targets); });
    }
  } else if (api == "contains") {
    if (tgt_type == "scalar") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::contains(input, target); });
    } else if (tgt_type == "column") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::contains(input, targets); });
    }
  } else if (api == "starts_with") {
    if (tgt_type == "scalar") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::starts_with(input, target); });
    } else if (tgt_type == "column") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::starts_with(input, targets); });
    }
  } else if (api == "ends_with") {
    if (tgt_type == "scalar") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::ends_with(input, target); });
    } else if (tgt_type == "column") {
      state.exec(nvbench::exec_tag::sync,
                 [&](nvbench::launch& launch) { cudf::strings::ends_with(input, targets); });
    }
  }
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_find_string)
  .set_name("find_string")
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("hit_rate", {20, 80})  // percentage
  .add_string_axis("api", {"find", "contains", "starts_with", "ends_with"})
  .add_string_axis("target", {"scalar", "column"});

static void bench_find_string_skewed(nvbench::state& state)
{
  auto const num_rows         = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const short_length     = static_cast<cudf::size_type>(state.get_int64("short_length"));
  auto const long_tail_length = static_cast<cudf::size_type>(state.get_int64("long_tail_length"));
  auto const short_string_pct = static_cast<int32_t>(state.get_int64("short_string_pct"));
  auto const hit_rate         = static_cast<int32_t>(state.get_int64("hit_rate"));

  auto const stream = cudf::get_default_stream();
  auto const col    = create_skewed_string_column(
    num_rows, short_length, long_tail_length, short_string_pct, hit_rate);
  auto const input = cudf::strings_column_view(col->view());

  auto target = cudf::string_scalar(skewed_string_target_substring);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const data_size = col->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(input.size());

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { cudf::strings::contains(input, target); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_find_string_skewed)
  .set_name("find_string_skewed")
  .add_int64_axis("short_length", {16, 32, 64, 96})
  .add_int64_axis("long_tail_length", {1024, 4096, 16384})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_int64_axis("short_string_pct", {90, 95, 99})
  .add_int64_axis("hit_rate", {20, 80});

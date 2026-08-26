/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf_test/column_wrapper.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

enum replace_type { scalar, slice, multi };

static void bench_replace(nvbench::state& state)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const min_width = static_cast<cudf::size_type>(state.get_int64("min_width"));
  auto const max_width = static_cast<cudf::size_type>(state.get_int64("max_width"));
  auto const api       = state.get_string("api");

  data_profile const profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, min_width, max_width);
  auto const column = create_random_column(cudf::type_id::STRING, row_count{num_rows}, profile);

  cudf::strings_column_view input(column->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const data_size = column->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(data_size);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  if (api == "scalar") {
    cudf::string_scalar target("+");
    cudf::string_scalar repl("-");
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::replace(input, target, repl); });
  } else if (api == "multi") {
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::test::strings_column_wrapper targets({"+", " "});
      cudf::test::strings_column_wrapper repls({"-", "_"});
      cudf::strings::replace_multiple(
        input, cudf::strings_column_view(targets), cudf::strings_column_view(repls));
    });
  } else if (api == "slice") {
    cudf::string_scalar repl("0123456789");
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::replace_slice(input, repl, 1, 10); });
  } else if (api == "column") {
    // Per-row targets and replacements — same length as input
    std::vector<std::string> t_vals(num_rows, "+");
    std::vector<std::string> r_vals(num_rows, "-");
    cudf::test::strings_column_wrapper targets_col(t_vals.begin(), t_vals.end());
    cudf::test::strings_column_wrapper repls_col(r_vals.begin(), r_vals.end());
    cudf::strings_column_view targets(targets_col);
    cudf::strings_column_view repls(repls_col);
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::replace(input, targets, repls); });
  }
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_replace)
  .set_name("replace")
  .add_int64_axis("min_width", {0})
  .add_int64_axis("max_width", {32, 64, 128, 256})
  .add_int64_axis("num_rows", {32768, 262144, 2097152})
  .add_string_axis("api", {"scalar", "multi", "slice", "column"});

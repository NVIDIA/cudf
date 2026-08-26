/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/strings/convert/convert_durations.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/wrappers/durations.hpp>

#include <nvbench/nvbench.cuh>

NVBENCH_DECLARE_TYPE_STRINGS(cudf::duration_D, "cudf::duration_D", "cudf::duration_D");
NVBENCH_DECLARE_TYPE_STRINGS(cudf::duration_s, "cudf::duration_s", "cudf::duration_s");
NVBENCH_DECLARE_TYPE_STRINGS(cudf::duration_ms, "cudf::duration_ms", "cudf::duration_ms");
NVBENCH_DECLARE_TYPE_STRINGS(cudf::duration_us, "cudf::duration_us", "cudf::duration_us");
NVBENCH_DECLARE_TYPE_STRINGS(cudf::duration_ns, "cudf::duration_ns", "cudf::duration_ns");

using Types = nvbench::type_list<cudf::duration_D,
                                 cudf::duration_s,
                                 cudf::duration_ms,
                                 cudf::duration_us,
                                 cudf::duration_ns>;

template <class DataType>
void bench_convert_duration(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const data_type = cudf::data_type(cudf::type_to_id<DataType>());
  auto const from_dur  = state.get_string("dir") == "from";

  auto const ts_col = create_random_column(data_type.id(), row_count{num_rows});
  cudf::column_view input(ts_col->view());

  auto format = std::string{"%D days %H:%M:%S"};
  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  if (from_dur) {
    state.add_global_memory_reads<DataType>(num_rows);
    state.add_global_memory_writes<int8_t>(format.size() * num_rows);
    auto const mem_stats_logger = cudf::memory_stats_logger();
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::from_durations(input, format); });
    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  } else {
    auto source = cudf::strings::from_durations(input, format);
    auto view   = cudf::strings_column_view(source->view());
    state.add_global_memory_reads<int8_t>(source->alloc_size());
    state.add_global_memory_writes<DataType>(num_rows);
    auto const mem_stats_logger = cudf::memory_stats_logger();

    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      cudf::strings::to_durations(view, data_type, format);
    });

    state.add_buffer_size(
      mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  }
}

NVBENCH_BENCH_TYPES(bench_convert_duration, NVBENCH_TYPE_AXES(Types))
  .set_name("duration")
  .set_type_axes_names({"DataType"})
  .add_string_axis("dir", {"to", "from"})
  .add_int64_axis("num_rows", {1 << 10, 1 << 15, 1 << 20, 1 << 25});

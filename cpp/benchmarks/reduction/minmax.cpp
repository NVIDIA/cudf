/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

template <typename DataType>
static void reduction_minmax(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const size = static_cast<cudf::size_type>(state.get_int64("size"));

  auto const input_type = cudf::type_to_id<DataType>();

  data_profile const profile =
    data_profile_builder().no_validity().distribution(input_type, distribution_id::UNIFORM, 0, 100);
  auto const input_column = create_random_column(input_type, row_count{size}, profile);

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_element_count(size);
  state.add_global_memory_reads<DataType>(size);
  state.add_global_memory_writes<DataType>(2);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&input_column](nvbench::launch& launch) { cudf::minmax(*input_column); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  set_throughputs(state);
}

NVBENCH_DECLARE_TYPE_STRINGS(cudf::timestamp_ms, "cudf::timestamp_ms", "cudf::timestamp_ms");

using Types = nvbench::type_list<bool, int8_t, int32_t, float, cudf::timestamp_ms>;

NVBENCH_BENCH_TYPES(reduction_minmax, NVBENCH_TYPE_AXES(Types))
  .set_name("minmax")
  .set_type_axes_names({"DataType"})
  .add_int64_axis("size", {100'000, 1'000'000, 10'000'000, 100'000'000});

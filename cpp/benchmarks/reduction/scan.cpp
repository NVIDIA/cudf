/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

template <typename DataType>
static void reduction_scan(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const size       = static_cast<cudf::size_type>(state.get_int64("size"));
  auto const nulls      = state.get_float64("nulls");
  auto const input_type = cudf::type_to_id<DataType>();

  data_profile const profile = data_profile_builder().null_probability(nulls).distribution(
    input_type, distribution_id::UNIFORM, 0, 100);
  auto const input_column = create_random_column(input_type, row_count{size}, profile);

  auto agg = cudf::make_min_aggregation<cudf::scan_aggregation>();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_element_count(size);
  state.add_global_memory_reads<DataType>(size);
  state.add_global_memory_writes<DataType>(1);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&input_column, &agg](nvbench::launch& launch) {
    cudf::scan(*input_column, *agg, cudf::scan_type::INCLUSIVE);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  set_throughputs(state);
}

using Types = nvbench::type_list<int8_t, int32_t, uint64_t, float, int16_t, uint32_t, double>;

NVBENCH_BENCH_TYPES(reduction_scan, NVBENCH_TYPE_AXES(Types))
  .set_name("scan")
  .set_type_axes_names({"DataType"})
  .add_float64_axis("nulls", {0.0, 0.1})
  .add_int64_axis("size", {100'000, 1'000'000, 10'000'000, 100'000'000});

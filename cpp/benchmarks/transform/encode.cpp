/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/strings/string_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <nvbench/nvbench.cuh>

template <typename DataType>
static void bench_encode(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const num_rows  = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const width     = static_cast<cudf::size_type>(state.get_int64("width"));
  auto const nulls     = state.get_float64("nulls");
  auto const data_type = cudf::type_to_id<DataType>();

  auto range = data_type == cudf::type_id::STRING ? (width / 10) : width;
  data_profile const profile =
    data_profile_builder().cardinality(0).null_probability(nulls).distribution(
      data_type, distribution_id::UNIFORM, 0, range);
  auto input = create_random_column(data_type, row_count{num_rows}, profile);
  auto tv    = cudf::table_view({input->view()});

  auto alloc_size = input->alloc_size();
  state.add_global_memory_reads<uint8_t>(alloc_size);
  state.add_global_memory_writes<cudf::size_type>(num_rows);
  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::encode(tv, stream); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_DECLARE_TYPE_STRINGS(cudf::string_view, "string_view", "string_view");

using Types = nvbench::type_list<int64_t, double, cudf::string_view>;

NVBENCH_BENCH_TYPES(bench_encode, NVBENCH_TYPE_AXES(Types))
  .set_name("encode")
  .add_int64_axis("width", {10, 100})
  .add_int64_axis("num_rows", {262144, 2097152, 16777216, 67108864})
  .add_float64_axis("nulls", {0, 0.1});

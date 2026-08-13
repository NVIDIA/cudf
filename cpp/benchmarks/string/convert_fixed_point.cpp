/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/strings/convert/convert_fixed_point.hpp>
#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

using Types = nvbench::type_list<numeric::decimal32, numeric::decimal64>;

NVBENCH_DECLARE_TYPE_STRINGS(numeric::decimal32, "decimal32", "decimal32");
NVBENCH_DECLARE_TYPE_STRINGS(numeric::decimal64, "decimal64", "decimal64");

template <typename DataType>
void bench_convert_fixed_point(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const from_num = state.get_string("dir") == "from";

  auto const data_type = cudf::data_type{cudf::type_to_id<DataType>(), numeric::scale_type{-2}};
  auto const fp_col    = create_random_column(data_type.id(), row_count{num_rows});

  auto const strings_col = cudf::strings::from_fixed_point(fp_col->view());
  auto const sv          = cudf::strings_column_view(strings_col->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const mem_stats_logger = cudf::memory_stats_logger();

  if (from_num) {
    state.add_global_memory_reads<int8_t>(num_rows * cudf::size_of(data_type));
    state.add_global_memory_writes<int8_t>(strings_col->alloc_size());
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::to_fixed_point(sv, data_type); });
  } else {
    state.add_global_memory_reads<int8_t>(strings_col->alloc_size());
    state.add_global_memory_writes<int8_t>(num_rows * cudf::size_of(data_type));
    state.exec(nvbench::exec_tag::sync,
               [&](nvbench::launch& launch) { cudf::strings::from_fixed_point(fp_col->view()); });
  }

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(bench_convert_fixed_point, NVBENCH_TYPE_AXES(Types))
  .set_name("fixed_point")
  .set_type_axes_names({"DataType"})
  .add_string_axis("dir", {"to", "from"})
  .add_int64_axis("num_rows", {1 << 16, 1 << 18, 1 << 20, 1 << 22});

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/strings/convert/convert_floats.hpp>
#include <cudf/strings/convert/convert_integers.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

namespace {

template <typename NumericType>
std::unique_ptr<cudf::column> get_strings_column(cudf::column_view const& nv)
{
  if constexpr (std::is_floating_point_v<NumericType>) {
    return cudf::strings::from_floats(nv);
  } else {
    return cudf::strings::from_integers(nv);
  }
}
}  // namespace

using Types = nvbench::type_list<float, double, int32_t, int64_t, uint8_t, uint16_t>;

template <typename NumericType>
void bench_convert_number(nvbench::state& state, nvbench::type_list<NumericType>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const from_num = state.get_string("dir") == "from";

  auto const data_type = cudf::data_type(cudf::type_to_id<NumericType>());
  auto const num_col   = create_random_column(data_type.id(), row_count{num_rows});

  auto const strings_col = get_strings_column<NumericType>(num_col->view());
  auto const sv          = cudf::strings_column_view(strings_col->view());

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const mem_stats_logger = cudf::memory_stats_logger();

  if (from_num) {
    state.add_global_memory_reads<NumericType>(num_rows);
    state.add_global_memory_writes<int8_t>(strings_col->alloc_size());
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      if constexpr (std::is_floating_point_v<NumericType>) {
        cudf::strings::to_floats(sv, data_type);
      } else {
        cudf::strings::to_integers(sv, data_type);
      }
    });
  } else {
    state.add_global_memory_reads<int8_t>(strings_col->alloc_size());
    state.add_global_memory_writes<NumericType>(num_rows);
    state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
      if constexpr (std::is_floating_point_v<NumericType>)
        cudf::strings::from_floats(num_col->view());
      else
        cudf::strings::from_integers(num_col->view());
    });
  }

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(bench_convert_number, NVBENCH_TYPE_AXES(Types))
  .set_name("numeric")
  .set_type_axes_names({"NumericType"})
  .add_string_axis("dir", {"to", "from"})
  .add_int64_axis("num_rows", {1 << 16, 1 << 18, 1 << 20, 1 << 22});

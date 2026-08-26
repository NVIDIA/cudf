/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/replace.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

using ClampTypes = nvbench::type_list<int8_t, int16_t, int32_t, uint32_t, uint64_t, float, double>;

template <typename ClampType>
void bench_clamp(nvbench::state& state, nvbench::type_list<ClampType>)
{
  auto const n_rows        = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const include_nulls = state.get_string("nulls") == "include";

  auto const dtype = cudf::type_to_id<ClampType>();
  auto const input = create_random_column(dtype, row_count{n_rows});
  if (!include_nulls) input->set_null_mask(rmm::device_buffer{}, 0);

  auto [low_scalar, high_scalar] = cudf::minmax(*input);

  // set the clamps 2 in from the min and max
  {
    using ScalarType = cudf::scalar_type_t<ClampType>;
    auto lvalue      = static_cast<ScalarType*>(low_scalar.get());
    auto hvalue      = static_cast<ScalarType*>(high_scalar.get());

    // super heavy clamp
    auto mid = lvalue->value() + (hvalue->value() - lvalue->value()) / 2;
    lvalue->set_value(mid - 10);
    hvalue->set_value(mid + 10);
  }

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  auto const data_size = input->alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(data_size);
  state.add_global_memory_writes<nvbench::int8_t>(data_size);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = cudf::clamp(*input, *low_scalar, *high_scalar);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH_TYPES(bench_clamp, NVBENCH_TYPE_AXES(ClampTypes))
  .set_name("clamp")
  .set_type_axes_names({"ClampType"})
  .add_int64_axis("num_rows", {10000, 100000, 1000000, 10000000, 100000000})
  .add_string_axis("nulls", {"include", "exclude"});

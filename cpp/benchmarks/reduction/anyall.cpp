/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>

#include <cudf/column/column_view.hpp>
#include <cudf/reduction.hpp>
#include <cudf/types.hpp>

#include <nvbench/nvbench.cuh>

#include <memory>

template <typename DataType>
static void reduction_anyall(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const size     = static_cast<cudf::size_type>(state.get_int64("size"));
  auto const kind_str = state.get_string("kind");

  auto const input_type = cudf::type_to_id<DataType>();
  auto const agg        = kind_str == "any" ? cudf::make_any_aggregation<cudf::reduce_aggregation>()
                                            : cudf::make_all_aggregation<cudf::reduce_aggregation>();

  data_profile const profile =
    data_profile_builder().no_validity().distribution(input_type,
                                                      distribution_id::UNIFORM,
                                                      (kind_str == "all" ? 1 : 0),
                                                      (kind_str == "any" ? 0 : 100));
  auto const values = create_random_column(input_type, row_count{size}, profile);

  auto const output_type = cudf::data_type{cudf::type_id::BOOL8};
  auto stream            = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_element_count(size);
  state.add_global_memory_reads<DataType>(size);
  state.add_global_memory_writes<nvbench::int8_t>(1);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&values, output_type, &agg](nvbench::launch& launch) {
    cudf::reduce(*values, *agg, output_type);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  set_throughputs(state);
}

using Types = nvbench::type_list<bool, int8_t, int32_t, float>;

NVBENCH_BENCH_TYPES(reduction_anyall, NVBENCH_TYPE_AXES(Types))
  .set_name("anyall")
  .set_type_axes_names({"DataType"})
  .add_string_axis("kind", {"any", "all"})
  .add_int64_axis("size", {100'000, 1'000'000, 10'000'000, 100'000'000});

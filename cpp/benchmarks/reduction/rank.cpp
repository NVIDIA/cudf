/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>
#include <benchmarks/common/table_utilities.hpp>

#include <cudf/detail/scan.hpp>
#include <cudf/filling.hpp>
#include <cudf/lists/list_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

template <typename type>
static void nvbench_reduction_scan(nvbench::state& state, nvbench::type_list<type>)
{
  auto const dtype = cudf::type_to_id<type>();

  double const null_probability = state.get_float64("null_probability");
  size_t const size             = state.get_int64("data_size");

  data_profile const profile = data_profile_builder()
                                 .null_probability(null_probability)
                                 .distribution(dtype, distribution_id::UNIFORM, 0, 5);

  auto const table = create_random_table({dtype}, table_size_bytes{size / 2}, profile);

  auto const new_tbl = cudf::repeat(table->view(), 2);
  cudf::column_view input(new_tbl->view().column(0));

  std::unique_ptr<cudf::column> result = nullptr;
  auto const mem_stats_logger          = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    rmm::cuda_stream_view stream_view{launch.get_stream()};
    result = cudf::detail::inclusive_dense_rank_scan(
      input, stream_view, cudf::get_current_device_resource_ref());
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  state.add_element_count(input.size());
  state.add_global_memory_reads(estimate_size(input));
  state.add_global_memory_writes(estimate_size(result->view()));

  set_throughputs(state);
}

using data_type = nvbench::type_list<int32_t, cudf::list_view>;

NVBENCH_BENCH_TYPES(nvbench_reduction_scan, NVBENCH_TYPE_AXES(data_type))
  .set_name("rank_scan")
  .add_float64_axis("null_probability", {0, 0.1, 0.5, 0.9})
  .add_int64_axis("data_size",
                  {
                    10000,      // 10k
                    100000,     // 100k
                    1000000,    // 1M
                    10000000,   // 10M
                    100000000,  // 100M
                  });

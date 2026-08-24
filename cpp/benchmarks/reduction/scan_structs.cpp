/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/common/nvbench_utilities.hpp>
#include <benchmarks/common/table_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/reduction.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvbench/nvbench.cuh>

static constexpr cudf::size_type num_struct_members = 8;
static constexpr cudf::size_type max_int            = 100;
static constexpr cudf::size_type max_str_length     = 32;

static void nvbench_structs_scan(nvbench::state& state)
{
  auto const null_probability = [&] {
    auto const null_prob_val = state.get_float64("null_probability");
    return null_prob_val > 0 ? std::optional{null_prob_val} : std::nullopt;
  }();
  auto const size    = static_cast<cudf::size_type>(state.get_int64("data_size"));
  auto const profile = static_cast<data_profile>(
    data_profile_builder()
      .null_probability(null_probability)
      .distribution(cudf::type_id::INT32, distribution_id::UNIFORM, 0, max_int)
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length));

  auto data_table = create_random_table(
    cycle_dtypes({cudf::type_id::INT32, cudf::type_id::STRING}, num_struct_members),
    row_count{size},
    profile);
  auto [null_mask, null_count] = create_random_null_mask(size, null_probability);
  auto const input             = cudf::make_structs_column(
    size, std::move(data_table->release()), null_count, std::move(null_mask));
  auto input_view = input->view();

  auto const agg         = cudf::make_min_aggregation<cudf::scan_aggregation>();
  auto const null_policy = static_cast<cudf::null_policy>(state.get_int64("null_policy"));
  auto const stream      = cudf::get_default_stream();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  std::unique_ptr<cudf::column> result = nullptr;
  auto const mem_stats_logger          = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    result = cudf::scan(input_view, *agg, cudf::scan_type::INCLUSIVE, null_policy, stream);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");

  state.add_element_count(input_view.size());
  state.add_global_memory_reads(estimate_size(input_view));
  state.add_global_memory_writes(estimate_size(result->view()));

  set_throughputs(state);
}

NVBENCH_BENCH(nvbench_structs_scan)
  .set_name("structs_scan")
  .add_float64_axis("null_probability", {0, 0.1, 0.5, 0.9})
  .add_int64_axis("null_policy", {0, 1})
  .add_int64_axis("data_size",
                  {
                    10000,     // 10k
                    100000,    // 100k
                    1000000,   // 1M
                    10000000,  // 10M
                  });

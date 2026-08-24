/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/structs/structs_column_view.hpp>

#include <nvbench/nvbench.cuh>

static constexpr cudf::size_type num_struct_members = 8;
static constexpr cudf::size_type max_int            = 100;
static constexpr cudf::size_type max_str_length     = 32;

static auto create_data_table(cudf::size_type n_rows)
{
  data_profile const table_profile =
    data_profile_builder()
      .distribution(cudf::type_id::INT32, distribution_id::UNIFORM, 0, max_int)
      .distribution(cudf::type_id::STRING, distribution_id::NORMAL, 0, max_str_length);

  // The first two struct members are int32 and string.
  // The first column is also used as keys in groupby.
  // The subsequent struct members are int32 and string again.
  return create_random_table(
    cycle_dtypes({cudf::type_id::INT32, cudf::type_id::STRING}, num_struct_members),
    row_count{n_rows},
    table_profile);
}

// Max aggregation/scan technically has the same performance as min.
static void bench_groupby_min_struct(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto data_cols      = create_data_table(num_rows)->release();

  auto const keys_view = data_cols.front()->view();
  auto const values =
    cudf::make_structs_column(keys_view.size(), std::move(data_cols), 0, rmm::device_buffer());

  auto gb_obj   = cudf::groupby::groupby(cudf::table_view({keys_view}));
  auto requests = std::vector<cudf::groupby::aggregation_request>();
  requests.emplace_back(cudf::groupby::aggregation_request());
  requests.front().aggregations.push_back(cudf::make_min_aggregation<cudf::groupby_aggregation>());
  requests.front().values = values->view();

  state.add_global_memory_reads<nvbench::int8_t>(values->alloc_size());
  state.add_global_memory_writes<nvbench::int8_t>(values->alloc_size());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch& launch) { auto result = gb_obj.aggregate(requests); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_groupby_min_struct)
  .set_name("min_struct")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000});

static void bench_groupby_min_struct_scan(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto data_cols      = create_data_table(num_rows)->release();

  auto const keys_view = data_cols.front()->view();
  auto const values =
    cudf::make_structs_column(keys_view.size(), std::move(data_cols), 0, rmm::device_buffer());

  auto requests = std::vector<cudf::groupby::scan_request>();
  requests.emplace_back(cudf::groupby::scan_request());
  requests.front().aggregations.push_back(
    cudf::make_min_aggregation<cudf::groupby_scan_aggregation>());
  requests.front().values = values->view();

  state.add_global_memory_reads<nvbench::int8_t>(values->alloc_size());
  state.add_global_memory_writes<nvbench::int8_t>(values->alloc_size());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    auto gb_obj = cudf::groupby::groupby(cudf::table_view({keys_view}));
    auto result = gb_obj.scan(requests);
  });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_groupby_min_struct_scan)
  .set_name("min_struct_scan")
  .add_int64_axis("num_rows", {10'000, 100'000, 1'000'000, 10'000'000, 100'000'000});

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/contiguous_split.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <nvbench/nvbench.cuh>

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

namespace {

/**
 * @brief Registers the default CUDA stream on `state` and builds the input table from the axis
 * parameters.
 */
std::unique_ptr<cudf::table> setup_bench(nvbench::state& state)
{
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));

  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const nulls    = state.get_float64("nulls");
  return create_sequence_table(
    cycle_dtypes({cudf::type_to_id<int64_t>()}, num_cols), row_count{num_rows}, nulls);
}

/**
 * @brief Annotates `state` with the bytes read and written by `cudf::pack`.
 *
 * Only pack does device-side I/O (deep-copies source columns into a contiguous buffer). Unpack
 * allocates no device memory, so annotating it would misreport bandwidth.
 */
void set_throughput_counters(nvbench::state& state, cudf::table const& table)
{
  auto const bytes = table.alloc_size();
  state.add_global_memory_reads<nvbench::int8_t>(bytes);
  state.add_global_memory_writes<nvbench::int8_t>(bytes);
}

/**
 * @brief Shared body for the pack benchmarks. `host_output` selects whether the packed buffer
 * uses the pinned host MR or the current device MR.
 */
void run_pack(nvbench::state& state, bool host_output)
{
  auto const table      = setup_bench(state);
  auto const table_view = table->view();
  auto stream           = cudf::get_default_stream();
  set_throughput_counters(state, *table);

  rmm::mr::pinned_host_memory_resource phmr;
  auto const mem_stats_logger = cudf::memory_stats_logger();
  auto const output_mr =
    host_output ? rmm::device_async_resource_ref{phmr} : cudf::get_current_device_resource_ref();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { std::ignore = cudf::pack(table_view, stream, output_mr); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

/**
 * @brief Packs a device table into a device-backed buffer.
 */
void bench_device_pack(nvbench::state& state) { run_pack(state, false); }

/**
 * @brief Packs a device table into a pinned-host-backed buffer.
 */
void bench_host_pack(nvbench::state& state) { run_pack(state, true); }

}  // namespace

auto const pack_num_rows_axis = std::vector<nvbench::int64_t>{4096, 32768, 262144};
auto const pack_num_cols_axis = std::vector<nvbench::int64_t>{64, 512, 1024};
auto const pack_nulls_axis    = std::vector<nvbench::float64_t>{0.0, 0.3};

NVBENCH_BENCH(bench_device_pack)
  .set_name("device_pack")
  .add_int64_axis("num_rows", pack_num_rows_axis)
  .add_int64_axis("num_cols", pack_num_cols_axis)
  .add_float64_axis("nulls", pack_nulls_axis);

NVBENCH_BENCH(bench_host_pack)
  .set_name("host_pack")
  .add_int64_axis("num_rows", pack_num_rows_axis)
  .add_int64_axis("num_cols", pack_num_cols_axis)
  .add_float64_axis("nulls", pack_nulls_axis);

/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <benchmarks/common/generate_input.hpp>

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/reduction.hpp>
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
void set_throughput_counters(nvbench::state& state)
{
  auto const num_rows   = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols   = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const data_bytes = static_cast<uint64_t>(num_rows) * num_cols * sizeof(int64_t);
  // Both axis values for `nulls` produce a null mask, so its bytes contribute to pack's I/O.
  auto const null_mask_bytes =
    static_cast<uint64_t>(num_cols) * cudf::bitmask_allocation_size_bytes(num_rows);
  state.add_global_memory_reads<nvbench::int8_t>(data_bytes + null_mask_bytes);
  state.add_global_memory_writes<nvbench::int8_t>(data_bytes + null_mask_bytes);
}

/**
 * @brief Reduces `col_view` with a sum aggregation, forcing the device to touch the column data.
 */
void column_sum(cudf::column_view const& col_view)
{
  auto sum_agg = cudf::make_sum_aggregation<cudf::reduce_aggregation>();
  std::ignore  = cudf::reduce(col_view, *sum_agg, cudf::data_type{cudf::type_id::INT64});
}

/**
 * @brief Shared body for the pack benchmarks. `packed_mr` selects the destination of the packed
 * buffer (device MR for device_pack, pinned host MR for host_pack).
 */
void run_pack(nvbench::state& state, rmm::device_async_resource_ref packed_mr)
{
  auto const table      = setup_bench(state);
  auto const table_view = table->view();
  auto stream           = cudf::get_default_stream();
  set_throughput_counters(state);
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { std::ignore = cudf::pack(table_view, stream, packed_mr); });
}

/**
 * @brief Shared body for the unpack benchmarks. The table is packed once outside the timed
 * region, so measurements only reflect the unpack call (and, when `access_column` is true, a
 * single reduce over the first unpacked column so the device actually touches the data).
 */
void run_unpack(nvbench::state& state, rmm::device_async_resource_ref packed_mr, bool access_column)
{
  auto const table      = setup_bench(state);
  auto const table_view = table->view();
  auto stream           = cudf::get_default_stream();
  auto packed           = cudf::pack(table_view, stream, packed_mr);
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto unpacked = cudf::unpack(packed);
    if (access_column) { column_sum(unpacked.column(0)); }
  });
}

/**
 * @brief Packs a device table into a device-backed buffer.
 */
void bench_device_pack(nvbench::state& state)
{
  run_pack(state, cudf::get_current_device_resource_ref());
}

/**
 * @brief Unpacks a device-backed packed table.
 */
void bench_device_unpack(nvbench::state& state)
{
  auto const access_column = state.get_int64("access_column") != 0;
  run_unpack(state, cudf::get_current_device_resource_ref(), access_column);
}

/**
 * @brief Packs a device table into a pinned-host-backed buffer.
 */
void bench_host_pack(nvbench::state& state)
{
  rmm::mr::pinned_host_memory_resource phmr;
  run_pack(state, phmr);
}

/**
 * @brief Unpacks a pinned-host-backed packed table.
 */
void bench_host_unpack(nvbench::state& state)
{
  auto const access_column = state.get_int64("access_column") != 0;
  rmm::mr::pinned_host_memory_resource phmr;
  run_unpack(state, phmr, access_column);
}

}  // namespace

auto const pack_num_rows_axis      = std::vector<nvbench::int64_t>{4096, 32768, 262144};
auto const pack_num_cols_axis      = std::vector<nvbench::int64_t>{64, 512, 1024};
auto const pack_nulls_axis         = std::vector<nvbench::float64_t>{0.0, 0.3};
auto const pack_access_column_axis = std::vector<nvbench::int64_t>{0, 1};

NVBENCH_BENCH(bench_device_pack)
  .set_name("device_pack")
  .add_int64_axis("num_rows", pack_num_rows_axis)
  .add_int64_axis("num_cols", pack_num_cols_axis)
  .add_float64_axis("nulls", pack_nulls_axis);

NVBENCH_BENCH(bench_device_unpack)
  .set_name("device_unpack")
  .add_int64_axis("num_rows", pack_num_rows_axis)
  .add_int64_axis("num_cols", pack_num_cols_axis)
  .add_float64_axis("nulls", pack_nulls_axis)
  .add_int64_axis("access_column", pack_access_column_axis);

NVBENCH_BENCH(bench_host_pack)
  .set_name("host_pack")
  .add_int64_axis("num_rows", pack_num_rows_axis)
  .add_int64_axis("num_cols", pack_num_cols_axis)
  .add_float64_axis("nulls", pack_nulls_axis);

NVBENCH_BENCH(bench_host_unpack)
  .set_name("host_unpack")
  .add_int64_axis("num_rows", pack_num_rows_axis)
  .add_int64_axis("num_cols", pack_num_cols_axis)
  .add_float64_axis("nulls", pack_nulls_axis)
  .add_int64_axis("access_column", pack_access_column_axis);

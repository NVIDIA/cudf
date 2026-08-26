/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/device_buffer.hpp>

#include <nvbench/nvbench.cuh>

template <typename DataType>
static void bench_copy_if_else(nvbench::state& state, nvbench::type_list<DataType>)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const nulls    = static_cast<bool>(state.get_int64("nulls"));

  auto input_type  = cudf::type_to_id<DataType>();
  auto bool_type   = cudf::type_id::BOOL8;
  auto const input = create_random_table({input_type, input_type, bool_type}, row_count{num_rows});

  if (!nulls) {
    input->get_column(0).set_null_mask(rmm::device_buffer{}, 0);
    input->get_column(1).set_null_mask(rmm::device_buffer{}, 0);
    input->get_column(2).set_null_mask(rmm::device_buffer{}, 0);
  }

  cudf::column_view lhs(input->view().column(0));
  cudf::column_view rhs(input->view().column(1));
  cudf::column_view decision(input->view().column(2));

  auto const bytes_read    = num_rows * (sizeof(DataType) + sizeof(bool));
  auto const bytes_written = num_rows * sizeof(DataType);
  auto const null_bytes    = nulls ? 2 * cudf::bitmask_allocation_size_bytes(num_rows) : 0;

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  state.add_global_memory_reads<int8_t>(bytes_read);
  state.add_global_memory_writes<int8_t>(bytes_written + null_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { cudf::copy_if_else(lhs, rhs, decision); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

using Types = nvbench::type_list<int16_t, uint32_t, double>;

NVBENCH_BENCH_TYPES(bench_copy_if_else, NVBENCH_TYPE_AXES(Types))
  .set_name("copy_if_else")
  .set_type_axes_names({"DataType"})
  .add_int64_axis("nulls", {true, false})
  .add_int64_axis("num_rows", {4096, 32768, 262144, 2097152, 16777216, 134217728});

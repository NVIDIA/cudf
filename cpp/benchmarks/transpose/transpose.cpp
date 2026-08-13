/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/table/table.hpp>
#include <cudf/transpose.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/iterator>
#include <thrust/iterator/transform_iterator.h>

#include <nvbench/nvbench.cuh>

void bench_transpose(nvbench::state& state)
{
  auto const count              = static_cast<cudf::size_type>(state.get_int64("table_size"));
  constexpr auto column_type_id = cudf::type_id::INT32;

  auto int_column_generator =
    thrust::make_transform_iterator(cuda::counting_iterator<std::size_t>{0}, [count](int i) {
      return cudf::make_numeric_column(
        cudf::data_type{column_type_id}, count, cudf::mask_state::ALL_VALID);
    });

  auto input_table = cudf::table(std::vector(int_column_generator, int_column_generator + count));
  auto input       = input_table.view();

  auto stream = cudf::get_default_stream();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));

  // Collect memory statistics.
  auto const bytes_read = static_cast<uint64_t>(input.num_columns()) * input.num_rows() *
                          sizeof(cudf::id_to_type<column_type_id>);
  auto const bytes_written = bytes_read;
  // Account for nullability in input and output.
  auto const null_bytes = 2 * static_cast<uint64_t>(input.num_columns()) *
                          cudf::bitmask_allocation_size_bytes(input.num_rows());

  state.add_global_memory_reads<nvbench::int8_t>(bytes_read + null_bytes);
  state.add_global_memory_writes<nvbench::int8_t>(bytes_written + null_bytes);

  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto output = cudf::transpose(input); });
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_transpose)
  .set_name("transpose")
  .add_int64_axis("table_size", {4, 16, 64, 256, 1024, 4096, 16384, 32768});

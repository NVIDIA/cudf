/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <nvbench/nvbench.cuh>

static void bench_dictionary_sort(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto constexpr width   = 32;  // width does not matter so keep it smallish

  auto input = create_string_column(num_rows, width, cardinality);

  auto stream = cudf::get_default_stream();
  auto encoded =
    cudf::dictionary::encode(input->view(), cudf::data_type{cudf::type_id::INT32}, stream);

  cudf::table_view input_table({encoded->view()});

  state.add_global_memory_reads<uint8_t>(encoded->alloc_size());
  auto result = cudf::sort(input_table, {}, {}, stream);
  state.add_global_memory_writes<uint8_t>(result->alloc_size());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync,
             [&](nvbench::launch&) { auto result = cudf::sort(input_table, {}, {}, stream); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_dictionary_sort)
  .set_name("sort")
  .add_int64_axis("num_rows", {262144, 2097152, 16777216, 67108864})
  .add_int64_axis("cardinality", {10, 50, 80});

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

static void bench_dictionary_concatenate(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols    = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto constexpr width   = 32;  // width does not matter so keep it smallish

  auto stream = cudf::get_default_stream();

  auto columns = std::vector<std::unique_ptr<cudf::column>>{};
  auto views   = std::vector<cudf::column_view>{};
  for (cudf::size_type i = 0; i < num_cols; ++i) {
    auto input = create_string_column(num_rows, width, cardinality);
    columns.emplace_back(
      cudf::dictionary::encode(input->view(), cudf::data_type{cudf::type_id::INT32}, stream));
    views.push_back(columns.back()->view());
  }

  auto input_table = cudf::table(std::move(columns));

  state.add_global_memory_reads<uint8_t>(input_table.alloc_size());
  auto result = cudf::concatenate(views, stream);
  state.add_global_memory_writes<uint8_t>(result->alloc_size());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) { cudf::concatenate(views, stream); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_dictionary_concatenate)
  .set_name("concatenate")
  .add_int64_axis("num_rows", {262144, 2097152, 16777216, 67108864})
  .add_int64_axis("cardinality", {10})
  .add_int64_axis("num_cols", {2, 10, 20});

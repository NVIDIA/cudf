/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/span.hpp>

#include <nvbench/nvbench.cuh>

#include <string>
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

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
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

/**
 * @brief Concatenate dictionary columns with narrow, mixed, or widening indices
 *
 * The `indices` axis selects the input configuration:
 * - `int8`, `int16`, `int32`: every column has the same key set and the given indices type,
 *   so the output keeps that indices type;
 * - `mixed`: alternating INT8 and INT16 indices over the same key set, so the narrower inputs are
 *   cast to INT16 before concatenation;
 * - `widen`: each column has its own 100 keys with INT8 indices, so the concatenated key set no
 *   longer fits INT8 and the output indices are widened to INT16.
 */
static void bench_dictionary_concatenate_indices(nvbench::state& state)
{
  auto const num_rows = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_cols = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const indices  = state.get_string("indices");

  auto stream = cudf::get_default_stream();

  auto const indices_type = [&](cudf::size_type col) {
    if (indices == "int16") { return cudf::type_id::INT16; }
    if (indices == "int32") { return cudf::type_id::INT32; }
    if (indices == "mixed") { return col % 2 == 0 ? cudf::type_id::INT8 : cudf::type_id::INT16; }
    return cudf::type_id::INT8;  // int8, widen
  };
  // 100 distinct keys per column fit INT8 indices; "widen" gives every column its own key range
  auto constexpr keys_per_column = 100;
  auto const key_range_start     = [&](cudf::size_type col) {
    return indices == "widen" ? col * keys_per_column : 0;
  };

  auto columns = std::vector<std::unique_ptr<cudf::column>>{};
  auto views   = std::vector<cudf::column_view>{};
  for (cudf::size_type i = 0; i < num_cols; ++i) {
    auto const lo              = key_range_start(i);
    data_profile const profile = data_profile_builder().distribution(
      cudf::type_id::INT32, distribution_id::UNIFORM, lo, lo + keys_per_column - 1);
    auto input = create_random_column(cudf::type_id::INT32, row_count{num_rows}, profile);
    columns.emplace_back(
      cudf::dictionary::encode(input->view(), cudf::data_type{indices_type(i)}, stream));
    views.push_back(columns.back()->view());
  }

  auto input_table = cudf::table(std::move(columns));

  state.add_global_memory_reads<uint8_t>(input_table.alloc_size());
  auto result = cudf::concatenate(views, stream);
  state.add_global_memory_writes<uint8_t>(result->alloc_size());
  // throughput is per processed index; the resulting key count is a plain summary
  state.add_element_count(static_cast<double>(num_rows) * num_cols);
  auto& keys_summary = state.add_summary("output_keys");
  keys_summary.set_string("description", "Number of keys in the concatenated dictionary");
  keys_summary.set_int64("value", cudf::dictionary_column_view(result->view()).keys_size());

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.get()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) { cudf::concatenate(views, stream); });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_dictionary_concatenate_indices)
  .set_name("concatenate_indices")
  .add_int64_axis("num_rows", {262144, 2097152, 16777216})
  .add_int64_axis("num_cols", {2, 10})
  .add_string_axis("indices", {"int8", "int16", "int32", "mixed", "widen"});

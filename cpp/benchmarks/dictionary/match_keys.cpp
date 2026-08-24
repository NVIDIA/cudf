/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>

#include <cudf/column/column.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/encode.hpp>
#include <cudf/dictionary/update_keys.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <nvbench/nvbench.cuh>

#include <vector>

static void bench_dictionary_match_keys(nvbench::state& state)
{
  auto const num_rows    = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const num_tables  = static_cast<cudf::size_type>(state.get_int64("num_tables"));
  auto constexpr width   = 32;  // width does not matter so keep it smallish

  auto stream = cudf::get_default_stream();

  // Each column is encoded from an independently generated string column of the same cardinality,
  // so the key sets overlap partially -- the common case that forces a real combined-keys rebuild.
  std::vector<std::unique_ptr<cudf::column>> encoded_columns;
  std::vector<cudf::dictionary_column_view> dict_views;
  int64_t input_bytes = 0;
  for (cudf::size_type i = 0; i < num_tables; ++i) {
    auto input = create_string_column(num_rows, width, cardinality);
    auto encoded =
      cudf::dictionary::encode(input->view(), cudf::data_type{cudf::type_id::INT32}, stream);
    input_bytes += encoded->alloc_size();
    encoded_columns.push_back(std::move(encoded));
  }
  for (auto const& col : encoded_columns) {
    dict_views.emplace_back(col->view());
  }

  state.add_global_memory_reads<uint8_t>(input_bytes);

  state.set_cuda_stream(nvbench::make_cuda_stream_view(stream.value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch&) {
    auto result = cudf::dictionary::match_dictionaries(dict_views, stream);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

NVBENCH_BENCH(bench_dictionary_match_keys)
  .set_name("match_keys")
  .add_int64_axis("num_rows", {262144, 2097152, 16777216, 67108864})
  .add_int64_axis("cardinality", {10, 1000, 100000})
  .add_int64_axis("num_tables", {2});

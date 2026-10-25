/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <limits>
#include <optional>

// Time the full read without page indexes. Input generation and writing are excluded.
// Vary pages per chunk and chunk count to expose header-parsing costs.
void BM_parquet_read_page_headers(nvbench::state& state)
{
  auto constexpr d_type = cudf::type_id::INT32;

  auto const source_type     = retrieve_io_type_enum(state.get_string("io_type"));
  auto const num_rows        = state.get_int64("num_rows");
  auto const num_columns     = state.get_int64("num_columns");
  auto const num_row_groups  = state.get_int64("num_row_groups");
  auto const pages_per_chunk = state.get_int64("pages_per_chunk");

  if (num_rows <= 0 || num_columns <= 0 || num_row_groups <= 0 || pages_per_chunk <= 0) {
    state.skip("Row, column, row-group, and page counts must be positive");
    return;
  }
  if (std::max({num_rows, num_columns, num_row_groups, pages_per_chunk}) >
      std::numeric_limits<cudf::size_type>::max()) {
    state.skip("Row, column, row-group, and page counts must fit in cudf::size_type");
    return;
  }

  auto const rows_per_row_group = static_cast<cudf::size_type>(num_rows / num_row_groups);
  auto const rows_per_page = static_cast<cudf::size_type>(rows_per_row_group / pages_per_chunk);

  if (rows_per_page < 1) {
    state.skip("pages_per_chunk exceeds the number of rows available in a row group");
    return;
  }
  auto const num_chunks  = num_columns * num_row_groups;
  auto const total_pages = num_chunks * pages_per_chunk;

  cuio_source_sink_pair source_sink(source_type);

  auto const tbl =
    create_random_table(cycle_dtypes({d_type}, static_cast<cudf::size_type>(num_columns)),
                        row_count{static_cast<cudf::size_type>(num_rows)},
                        data_profile_builder().null_probability(std::nullopt));
  auto const view = tbl->view();

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      // Exclude decompression costs.
      .compression(cudf::io::compression_type::NONE)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER)
      .row_group_size_rows(rows_per_row_group)
      .max_page_size_rows(rows_per_page)
      .max_page_fragment_size(rows_per_page)
      .stats_level(cudf::io::statistics_freq::STATISTICS_ROWGROUP);
  cudf::io::write_parquet(write_opts);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
      drop_page_cache_if_enabled(read_opts.get_source().filepaths());

      timer.start();
      auto const result = cudf::io::read_parquet(read_opts);
      timer.stop();

      CUDF_EXPECTS(result.tbl->num_columns() == num_columns, "Unexpected number of columns");
      CUDF_EXPECTS(result.tbl->num_rows() == num_rows, "Unexpected number of rows");
    });

  state.add_element_count(total_pages, "estimated_pages");
  state.add_element_count(num_chunks, "num_chunks");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

NVBENCH_BENCH(BM_parquet_read_page_headers)
  .set_name("parquet_read_page_headers")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  // Fixed rows per column.
  .add_int64_axis("num_rows", {4'000'000})
  // Header-chain length.
  .add_int64_axis("pages_per_chunk", {10, 100, 1'000, 10'000})
  // Requested chunks = columns * row groups. More columns add data; more row groups split it.
  .add_int64_axis("num_columns", {1, 16})
  .add_int64_axis("num_row_groups", {1, 8});

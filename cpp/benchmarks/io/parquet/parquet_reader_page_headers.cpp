/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

#include <optional>

// End-to-end read benchmark emphasizing page-header parsing. Input generation and writing are
// outside the timed region. Uncompressed, non-nullable INT32 data keeps payload processing cheap.
// Holding row count fixed while varying page size changes the length of each chunk's serial header
// chain. Column and row-group counts vary the number of independently parsed chunks.
//
// No offset index is written, so reads exercise header counting/parsing, including the fused path.
// This measures the complete read, including scratch/output allocation and downstream decoding;
// use a profiler to attribute changes to individual kernels. Page counts below are requested shape
// estimates: writer sizing limits and partial pages can change the actual number of headers.
void BM_parquet_read_page_headers(nvbench::state& state)
{
  auto constexpr d_type = cudf::type_id::INT32;

  auto const source_type     = retrieve_io_type_enum(state.get_string("io_type"));
  auto const num_rows        = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_columns     = static_cast<cudf::size_type>(state.get_int64("num_columns"));
  auto const num_row_groups  = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const pages_per_chunk = static_cast<cudf::size_type>(state.get_int64("pages_per_chunk"));

  if (num_rows <= 0 || num_columns <= 0 || num_row_groups <= 0 || pages_per_chunk <= 0) {
    state.skip("Row, column, row-group, and page counts must be positive");
    return;
  }

  auto const rows_per_row_group = num_rows / num_row_groups;
  auto const rows_per_page      = rows_per_row_group / pages_per_chunk;
  auto const num_chunks         = static_cast<int64_t>(num_columns) * num_row_groups;
  auto const total_pages        = num_chunks * pages_per_chunk;

  // A page needs at least one row, so the sweep cannot ask for more pages than there are rows in a
  // row group. Skipping keeps an over-ambitious axis combination from silently measuring a file
  // with a different shape than the one requested.
  if (rows_per_page < 1) {
    state.skip("pages_per_chunk exceeds the number of rows available in a row group");
    return;
  }

  cuio_source_sink_pair source_sink(source_type);

  auto const tbl  = create_random_table(cycle_dtypes({d_type}, num_columns),
                                       row_count{num_rows},
                                       data_profile_builder().null_probability(std::nullopt));
  auto const view = tbl->view();

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      // Uncompressed on purpose: decompression would otherwise dwarf the header walk.
      .compression(cudf::io::compression_type::NONE)
      .dictionary_policy(cudf::io::dictionary_policy::NEVER)
      .row_group_size_rows(rows_per_row_group)
      .max_page_size_rows(rows_per_page)
      // Pages are assembled out of fragments, so a page cannot be smaller than one. Left at its
      // default of 5000 rows this silently caps the page count: every request for a page shorter
      // than a fragment yields the fragment instead, and the sweep flattens out without saying so.
      .max_page_fragment_size(rows_per_page)
      .stats_level(cudf::io::statistics_freq::STATISTICS_ROWGROUP);
  cudf::io::write_parquet(write_opts);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().get()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
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
  // Hold payload size fixed while varying the requested page count for each column configuration.
  .add_int64_axis("num_rows", {4'000'000})
  // The axis that sets the serial chain length, and the one that matters most here.
  .add_int64_axis("pages_per_chunk", {10, 100, 1000, 10000})
  // Chunk count, i.e. how many chains are walked at once. Together these separate the length of a
  // chain from the number of them, which the row group based benchmarks conflate.
  .add_int64_axis("num_columns", {1, 16})
  .add_int64_axis("num_row_groups", {1, 8});

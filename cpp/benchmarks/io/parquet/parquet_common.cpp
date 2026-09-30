/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <nvbench/nvbench.cuh>

#include <limits>
#include <string>

std::optional<double> null_probability_from_percent(int64_t null_percent)
{
  if (null_percent < 0) { return std::nullopt; }
  CUDF_EXPECTS(null_percent <= 100, "null_percent must be -1 or in [0, 100]");
  return static_cast<double>(null_percent) / 100.0;
}

void parquet_read_common(cudf::size_type num_rows_to_read,
                         cudf::size_type num_cols_to_read,
                         cuio_source_sink_pair& source_sink,
                         nvbench::state& state)
{
  auto const data_size = static_cast<size_t>(state.get_int64("data_size"));
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

      CUDF_EXPECTS(result.tbl->num_columns() == num_cols_to_read, "Unexpected number of columns");
      CUDF_EXPECTS(result.tbl->num_rows() == num_rows_to_read, "Unexpected number of rows");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

cuio_source_sink_pair write_file_shape_parquet_file(cudf::type_id dtype,
                                                    cudf::size_type num_rows,
                                                    cudf::size_type num_row_groups,
                                                    cudf::size_type pages_per_row_group,
                                                    io_type source_type,
                                                    bool write_page_index)
{
  cuio_source_sink_pair source_sink(source_type);

  auto const tbl =
    create_random_table({dtype},
                        row_count{num_rows},
                        data_profile_builder().cardinality(num_rows / 10).avg_run_length(4));
  auto const view = tbl->view();

  auto const rows_per_page = num_rows / (num_row_groups * pages_per_row_group);
  CUDF_EXPECTS(rows_per_page > 0, "num_row_groups * pages_per_row_group must not exceed num_rows");

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(cudf::io::compression_type::NONE)
      .row_group_size_rows(num_rows / num_row_groups)
      .max_page_size_rows(rows_per_page)
      // Pages are assembled out of whole fragments, so without this the default 5000-row
      // fragment is a floor on page size and fewer rows per page cannot be honored
      .max_page_fragment_size(rows_per_page)
      // Use the largest page size to prevent pages from being closed by the byte limit
      .max_page_size_bytes(static_cast<size_t>(std::numeric_limits<int32_t>::max()))
      .stats_level(write_page_index ? cudf::io::statistics_freq::STATISTICS_COLUMN
                                    : cudf::io::statistics_freq::STATISTICS_ROWGROUP);
  cudf::io::write_parquet(write_opts);

  return source_sink;
}

std::vector<std::string> named_resolution_column_names(cudf::size_type num_cols)
{
  std::vector<std::string> names(num_cols);
  for (cudf::size_type i = 0; i < num_cols; ++i) {
    names[i] = "col" + std::to_string(i);
  }
  return names;
}

cuio_source_sink_pair write_named_resolution_parquet_file(cudf::size_type num_cols,
                                                          io_type source_type)
{
  cuio_source_sink_pair source_sink(source_type);

  // Flat, single-row table of INT32 columns with deterministic names col0..col{n-1}. INT32 keeps
  // the filter literal trivially type-correct; name-resolution cost is independent of dtype.
  constexpr cudf::size_type num_rows = 1;
  auto const tbl =
    create_random_table(cycle_dtypes({cudf::type_id::INT32}, num_cols),
                        row_count{num_rows},
                        data_profile_builder().cardinality(0).avg_run_length(1).no_validity());

  auto const column_names = named_resolution_column_names(num_cols);
  cudf::io::table_input_metadata input_meta(tbl->view());
  for (cudf::size_type i = 0; i < num_cols; ++i) {
    input_meta.column_metadata[i].set_name(column_names[i]);
  }

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), tbl->view())
      .metadata(std::move(input_meta))
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(write_opts);

  return source_sink;
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "parquet_common.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <cuda/iterator>

#include <nvbench/nvbench.cuh>

#include <algorithm>
#include <string>

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

std::vector<cudf::type_id> const& mixed_dtypes()
{
  static std::vector<cudf::type_id> const dtypes =
    get_type_or_group({static_cast<int32_t>(data_type::STRING),
                       static_cast<int32_t>(data_type::INTEGRAL),
                       static_cast<int32_t>(data_type::FLOAT),
                       static_cast<int32_t>(data_type::DECIMAL),
                       static_cast<int32_t>(data_type::LIST)});
  return dtypes;
}

cuio_source_sink_pair write_mixed_dtype_parquet_file(cudf::size_type num_cols,
                                                     cudf::size_type num_row_groups,
                                                     io_type source_type,
                                                     bool write_page_index)
{
  cuio_source_sink_pair source_sink(source_type);

  // Minimum row group size that cudf will almost always follow
  constexpr auto rows_per_row_group = 5000;
  constexpr auto min_row_groups     = 10;
  constexpr auto table_rows         = min_row_groups * rows_per_row_group;

  CUDF_EXPECTS(num_row_groups > 0 and num_row_groups % min_row_groups == 0,
               "Number of requested row groups must be non-zero and a multiple of " +
                 std::to_string(min_row_groups));

  // Create a table with the enough rows to cover min_row_groups
  auto const tbl =
    create_random_table(cycle_dtypes(mixed_dtypes(), num_cols),
                        row_count{table_rows},
                        data_profile_builder().cardinality(0).avg_run_length(1).distribution(
                          cudf::type_id::LIST, distribution_id::GEOMETRIC, 0, 4));
  auto const view = tbl->view();

  auto const stats_level = write_page_index ? cudf::io::statistics_freq::STATISTICS_COLUMN
                                            : cudf::io::statistics_freq::STATISTICS_ROWGROUP;
  auto const options =
    cudf::io::chunked_parquet_writer_options::builder(source_sink.make_sink_info())
      .row_group_size_rows(rows_per_row_group)
      .compression(cudf::io::compression_type::NONE)
      .stats_level(stats_level)
      .build();
  auto writer = cudf::io::chunked_parquet_writer(options, cudf::get_default_stream());

  // Compute the number of times the table needs to be written to cover the requested number of row
  // groups
  auto num_writes = cudf::util::div_rounding_up_unsafe(num_row_groups, min_row_groups);
  std::for_each(cuda::counting_iterator<cudf::size_type>{0},
                cuda::counting_iterator{num_writes},
                [&](cudf::size_type) { writer.write(view); });

  std::ignore = writer.close();

  return source_sink;
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

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(cudf::io::compression_type::NONE)
      .row_group_size_rows(num_rows / num_row_groups)
      .max_page_size_rows(rows_per_page)
      // Pages are assembled out of whole fragments, so without this the default 5000-row
      // fragment is a floor on page size and fewer rows per page cannot be honored
      .max_page_fragment_size(rows_per_page)
      // Lift the default 512KB page limit so that it does not close pages before
      // `max_page_size_rows` does
      .max_page_size_bytes(1ul << 30)
      // Write page index by setting stats_level to STATISTICS_COLUMN
      .stats_level(write_page_index ? cudf::io::statistics_freq::STATISTICS_COLUMN
                                    : cudf::io::statistics_freq::STATISTICS_ROWGROUP);
  cudf::io::write_parquet(write_opts);

  return source_sink;
}

cuio_source_sink_pair write_named_resolution_parquet_file(cudf::size_type num_cols,
                                                          io_type source_type)
{
  cuio_source_sink_pair source_sink(source_type);

  // A single INT32 row keeps the file negligible so name resolution dominates the read; resolution
  // cost is independent of dtype, so the cheapest type is used.
  constexpr cudf::size_type num_rows = 1;
  auto const tbl =
    create_random_table(cycle_dtypes({cudf::type_id::INT32}, num_cols),
                        row_count{num_rows},
                        data_profile_builder().cardinality(0).avg_run_length(1).no_validity());

  cudf::io::table_input_metadata input_meta(tbl->view());
  for (cudf::size_type i = 0; i < num_cols; ++i) {
    input_meta.column_metadata[i].set_name("col" + std::to_string(i));
  }

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), tbl->view())
      .metadata(std::move(input_meta))
      .compression(cudf::io::compression_type::NONE);
  cudf::io::write_parquet(write_opts);

  return source_sink;
}

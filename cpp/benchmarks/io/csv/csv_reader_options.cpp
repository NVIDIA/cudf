/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/csv.hpp>

#include <nvbench/nvbench.cuh>

constexpr size_t data_size = 256 << 20;

// Use alphanumeric character range to avoid CSV special characters (comma, quote, hash)
// that can trigger quoting issues.
data_profile const profile = data_profile_builder().string_char_range('0', 'z');  // ASCII 48-122

template <column_selection ColSelection, row_selection RowSelection>
void BM_csv_read_varying_options(
  nvbench::state& state,
  nvbench::type_list<nvbench::enum_type<ColSelection>, nvbench::enum_type<RowSelection>>)
{
  auto const data_types =
    dtypes_for_column_selection(get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                                   static_cast<int32_t>(data_type::FLOAT),
                                                   static_cast<int32_t>(data_type::DECIMAL),
                                                   static_cast<int32_t>(data_type::TIMESTAMP),
                                                   static_cast<int32_t>(data_type::DURATION),
                                                   static_cast<int32_t>(data_type::STRING)}),
                                ColSelection);
  auto const cols_to_read                 = select_column_indexes(data_types.size(), ColSelection);
  cudf::size_type const expected_num_cols = cols_to_read.size();
  size_t const num_chunks                 = state.get_int64("num_chunks");

  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size}, profile);
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  cudf::io::csv_writer_options options =
    cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view)
      .include_header(true)
      .line_terminator("\r\n");
  cudf::io::write_csv(options);

  cudf::io::csv_reader_options read_options =
    cudf::io::csv_reader_options::builder(source_sink.make_source_info())
      .compression(cudf::io::compression_type::NONE)
      .use_cols_indexes(cols_to_read)
      .thousands('\'')
      .windowslinetermination(true)
      .comment('#')
      .prefix("BM_");

  size_t const chunk_size = cudf::util::div_rounding_up_safe(source_sink.size(), num_chunks);
  auto const chunk_row_cnt =
    cudf::util::div_rounding_up_safe(view.num_rows(), static_cast<cudf::size_type>(num_chunks));
  auto const mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               drop_page_cache_if_enabled(read_options.get_source().filepaths());
               cudf::size_type num_rows_read = 0;
               timer.start();
               for (auto chunk = 0u; chunk < num_chunks; ++chunk) {
                 switch (RowSelection) {
                   case row_selection::ALL: break;
                   case row_selection::BYTE_RANGE:
                     // with byte_range, we can't read the header in any chunk but the first
                     read_options.set_header(chunk == 0 ? 0 : -1);
                     read_options.set_byte_range_offset(chunk * chunk_size);
                     read_options.set_byte_range_size(chunk_size);
                     break;
                   case row_selection::NROWS:
                     read_options.set_skiprows(chunk * chunk_row_cnt);
                     read_options.set_nrows(chunk_row_cnt);
                     break;
                   case row_selection::SKIPFOOTER: {
                     read_options.set_skiprows(chunk * chunk_row_cnt);
                     cudf::size_type const next_chunk_start = (chunk + 1) * chunk_row_cnt;
                     auto const skip_footer =
                       view.num_rows() > next_chunk_start ? view.num_rows() - next_chunk_start : 0;
                     read_options.set_skipfooter(skip_footer);
                     break;
                   }
                   default: CUDF_FAIL("Unsupported row selection method");
                 }

                 auto const result = cudf::io::read_csv(read_options);

                 num_rows_read += result.tbl->num_rows();
                 CUDF_EXPECTS(result.tbl->num_columns() == expected_num_cols,
                              "Unexpected number of columns");
               }
               timer.stop();
               CUDF_EXPECTS(num_rows_read == view.num_rows(), "Unexpected number of rows");
             });

  auto const elapsed_time   = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  auto const data_processed = data_size * cols_to_read.size() / view.num_columns();
  state.add_element_count(static_cast<double>(data_processed) / elapsed_time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

using col_selections = nvbench::enum_type_list<column_selection::ALL,
                                               column_selection::ALTERNATE,
                                               column_selection::FIRST_HALF,
                                               column_selection::SECOND_HALF>;

using row_selections = nvbench::
  enum_type_list<row_selection::BYTE_RANGE, row_selection::NROWS, row_selection::SKIPFOOTER>;

NVBENCH_BENCH_TYPES(BM_csv_read_varying_options,
                    NVBENCH_TYPE_AXES(col_selections, nvbench::enum_type_list<row_selection::ALL>))
  .set_name("csv_read_column_selection")
  .set_type_axes_names({"column_selection", "row_selection"})
  .set_min_samples(4)
  .add_int64_axis("num_chunks", {1});

NVBENCH_BENCH_TYPES(BM_csv_read_varying_options,
                    NVBENCH_TYPE_AXES(nvbench::enum_type_list<column_selection::ALL>,
                                      row_selections))
  .set_name("csv_read_row_selection")
  .set_type_axes_names({"column_selection", "row_selection"})
  .set_min_samples(4)
  .add_int64_axis("num_chunks", {1, 8});

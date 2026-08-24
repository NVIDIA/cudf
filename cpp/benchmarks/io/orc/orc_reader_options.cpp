/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/orc_metadata.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr int64_t data_size = 512 << 20;
// The number of separate read calls to use when reading files in multiple chunks
// Each call reads roughly equal amounts of data
constexpr int32_t chunked_read_num_chunks = 4;

std::vector<std::string> get_top_level_col_names(cudf::io::source_info const& source)
{
  auto const top_lvl_cols = cudf::io::read_orc_metadata(source).schema().root().children();
  std::vector<std::string> col_names;
  std::transform(top_lvl_cols.cbegin(),
                 top_lvl_cols.cend(),
                 std::back_inserter(col_names),
                 [](auto const& col_meta) { return col_meta.name(); });
  return col_names;
}

template <column_selection ColSelection,
          row_selection RowSelection,
          uses_index UsesIndex,
          uses_numpy_dtype UsesNumpyDType,
          cudf::type_id Timestamp>
void BM_orc_read_varying_options(nvbench::state& state,
                                 nvbench::type_list<nvbench::enum_type<ColSelection>,
                                                    nvbench::enum_type<RowSelection>,
                                                    nvbench::enum_type<UsesIndex>,
                                                    nvbench::enum_type<UsesNumpyDType>,
                                                    nvbench::enum_type<Timestamp>>)
{
  auto const num_chunks = RowSelection == row_selection::ALL ? 1 : chunked_read_num_chunks;

  auto const use_index     = UsesIndex == uses_index::YES;
  auto const use_np_dtypes = UsesNumpyDType == uses_numpy_dtype::YES;
  auto const ts_type       = cudf::data_type{Timestamp};

  auto const stripe_size_bytes = state.get_int64("stripe_size_bytes");
  auto const stripe_size_rows  = state.get_int64("stripe_size_rows");

  // skip_rows is not supported on nested types
  auto const data_types =
    dtypes_for_column_selection(get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
                                                   static_cast<int32_t>(data_type::FLOAT),
                                                   static_cast<int32_t>(data_type::DECIMAL),
                                                   static_cast<int32_t>(data_type::TIMESTAMP),
                                                   static_cast<int32_t>(data_type::STRING)}),
                                ColSelection);
  auto const tbl  = create_random_table(data_types, table_size_bytes{data_size});
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(io_type::HOST_BUFFER);
  cudf::io::orc_writer_options options =
    cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view);
  // Sentinel 0 == use cuDF default.
  if (stripe_size_bytes > 0) options.set_stripe_size_bytes(stripe_size_bytes);
  if (stripe_size_rows > 0) options.set_stripe_size_rows(stripe_size_rows);
  cudf::io::write_orc(options);

  auto const cols_to_read =
    select_column_names(get_top_level_col_names(source_sink.make_source_info()), ColSelection);
  cudf::size_type const expected_num_cols = cols_to_read.size();
  cudf::io::orc_reader_options read_options =
    cudf::io::orc_reader_options::builder(source_sink.make_source_info())
      .columns(cols_to_read)
      .use_index(use_index)
      .use_np_dtypes(use_np_dtypes)
      .timestamp_type(ts_type);

  auto const num_stripes =
    cudf::io::read_orc_metadata(source_sink.make_source_info()).num_stripes();
  CUDF_EXPECTS(RowSelection != row_selection::STRIPES || num_stripes >= num_chunks,
               "STRIPES option requires at least one stripe per read chunk");

  auto const chunk_row_cnt = cudf::util::div_rounding_up_unsafe(view.num_rows(), num_chunks);

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_options.get_source().filepaths());
      cudf::size_type num_rows_read = 0;
      timer.start();
      for (int32_t chunk = 0; chunk < num_chunks; ++chunk) {
        switch (RowSelection) {
          case row_selection::ALL: break;
          case row_selection::STRIPES:
            read_options.set_stripes({segments_in_chunk(num_stripes, num_chunks, chunk)});
            break;
          case row_selection::NROWS:
            read_options.set_skip_rows(chunk * chunk_row_cnt);
            read_options.set_num_rows(chunk_row_cnt);
            break;
          default: CUDF_FAIL("Unsupported row selection method");
        }

        auto const result = cudf::io::read_orc(read_options);

        num_rows_read += result.tbl->num_rows();
        CUDF_EXPECTS(result.tbl->num_columns() == expected_num_cols,
                     "Unexpected number of columns");
      }

      timer.stop();
      CUDF_EXPECTS(num_rows_read == view.num_rows(), "Benchmark did not read the entire table");
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
NVBENCH_BENCH_TYPES(BM_orc_read_varying_options,
                    NVBENCH_TYPE_AXES(col_selections,
                                      nvbench::enum_type_list<row_selection::ALL>,
                                      nvbench::enum_type_list<uses_index::YES>,
                                      nvbench::enum_type_list<uses_numpy_dtype::YES>,
                                      nvbench::enum_type_list<cudf::type_id::EMPTY>))
  .set_name("orc_read_column_selection")
  .set_type_axes_names(
    {"column_selection", "row_selection", "uses_index", "uses_numpy_dtype", "timestamp_type"})
  .set_min_samples(4)
  .add_int64_axis("stripe_size_bytes", {0})
  .add_int64_axis("stripe_size_rows", {0});

using row_selections =
  nvbench::enum_type_list<row_selection::ALL, row_selection::NROWS, row_selection::STRIPES>;
NVBENCH_BENCH_TYPES(BM_orc_read_varying_options,
                    NVBENCH_TYPE_AXES(nvbench::enum_type_list<column_selection::ALL>,
                                      row_selections,
                                      nvbench::enum_type_list<uses_index::YES>,
                                      nvbench::enum_type_list<uses_numpy_dtype::YES>,
                                      nvbench::enum_type_list<cudf::type_id::EMPTY>))
  .set_name("orc_read_row_selection")
  .set_type_axes_names(
    {"column_selection", "row_selection", "uses_index", "uses_numpy_dtype", "timestamp_type"})
  .set_min_samples(4)
  // NOTE: row_selection::STRIPES reads a fraction of stripes; non-zero
  // stripe sizing values here change what each chunk reads.
  .add_int64_axis("stripe_size_bytes", {0})
  .add_int64_axis("stripe_size_rows", {0});

NVBENCH_BENCH_TYPES(
  BM_orc_read_varying_options,
  NVBENCH_TYPE_AXES(
    nvbench::enum_type_list<column_selection::ALL>,
    nvbench::enum_type_list<row_selection::ALL>,
    nvbench::enum_type_list<uses_index::YES, uses_index::NO>,
    nvbench::enum_type_list<uses_numpy_dtype::YES, uses_numpy_dtype::NO>,
    nvbench::enum_type_list<cudf::type_id::EMPTY, cudf::type_id::TIMESTAMP_NANOSECONDS>))
  .set_name("orc_read_misc_options")
  .set_type_axes_names(
    {"column_selection", "row_selection", "uses_index", "uses_numpy_dtype", "timestamp_type"})
  .set_min_samples(4)
  .add_int64_axis("stripe_size_bytes", {0})
  .add_int64_axis("stripe_size_rows", {0});

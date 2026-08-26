/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/csv.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

constexpr size_t data_size         = 256 << 20;
constexpr cudf::size_type num_cols = 64;

// Use alphanumeric character range to avoid CSV special characters (comma, quote, hash)
// that can trigger quoting issues.
data_profile const profile = data_profile_builder().string_char_range('0', 'z');  // ASCII 48-122

template <typename DataType>
void csv_read_common(DataType const& data_types, io_type const& source_type, nvbench::state& state)
{
  auto const tbl =
    create_random_table(cycle_dtypes(data_types, num_cols), table_size_bytes{data_size}, profile);
  auto const view = tbl->view();

  cuio_source_sink_pair source_sink(source_type);
  cudf::io::csv_writer_options options =
    cudf::io::csv_writer_options::builder(source_sink.make_sink_info(), view).include_header(true);

  cudf::io::write_csv(options);

  // Extract column types from the source table to avoid type inference issues
  // (e.g., timestamp/duration columns being inferred as strings)
  std::vector<cudf::data_type> column_types;
  column_types.reserve(view.num_columns());
  for (cudf::size_type i = 0; i < view.num_columns(); ++i) {
    column_types.push_back(view.column(i).type());
  }

  cudf::io::csv_reader_options const read_options =
    cudf::io::csv_reader_options::builder(source_sink.make_source_info())
      .compression(cudf::io::compression_type::NONE)
      .dtypes(column_types);

  auto const mem_stats_logger = cudf::memory_stats_logger();  // init stats logger
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch& launch, auto& timer) {
      drop_page_cache_if_enabled(read_options.get_source().filepaths());

      timer.start();
      auto const result = cudf::io::read_csv(read_options);
      timer.stop();

      CUDF_EXPECTS(result.tbl->num_columns() == view.num_columns(), "Unexpected number of columns");
      CUDF_EXPECTS(result.tbl->num_rows() == view.num_rows(), "Unexpected number of rows");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

template <data_type DataType, io_type IOType>
void BM_csv_read_input(nvbench::state& state,
                       nvbench::type_list<nvbench::enum_type<DataType>, nvbench::enum_type<IOType>>)
{
  auto const d_type      = get_type_or_group(static_cast<int32_t>(DataType));
  auto const source_type = IOType;

  csv_read_common(d_type, source_type, state);
}

template <io_type IOType>
void BM_csv_read_io(nvbench::state& state, nvbench::type_list<nvbench::enum_type<IOType>>)
{
  auto const d_type      = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL),
                                              static_cast<int32_t>(data_type::FLOAT),
                                              static_cast<int32_t>(data_type::DECIMAL),
                                              static_cast<int32_t>(data_type::TIMESTAMP),
                                              static_cast<int32_t>(data_type::DURATION),
                                              static_cast<int32_t>(data_type::STRING)});
  auto const source_type = IOType;

  csv_read_common(d_type, source_type, state);
}

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL,
                                            data_type::FLOAT,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::DURATION,
                                            data_type::STRING>;

using io_list = nvbench::enum_type_list<io_type::FILEPATH, io_type::HOST_BUFFER>;

NVBENCH_BENCH_TYPES(BM_csv_read_input,
                    NVBENCH_TYPE_AXES(d_type_list, nvbench::enum_type_list<io_type::DEVICE_BUFFER>))
  .set_name("csv_read_data_type")
  .set_type_axes_names({"data_type", "io"})
  .set_min_samples(4);

NVBENCH_BENCH_TYPES(BM_csv_read_io, NVBENCH_TYPE_AXES(io_list))
  .set_name("csv_read_io")
  .set_type_axes_names({"io"})
  .set_min_samples(4);

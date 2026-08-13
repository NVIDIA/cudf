/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/experimental/cudftable.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

constexpr cudf::size_type num_cols = 64;

void cudftable_read_common(cudf::size_type num_rows_to_read,
                           cudf::size_type num_cols_to_read,
                           cuio_source_sink_pair& source_sink,
                           nvbench::state& state)
{
  auto const data_size = static_cast<size_t>(state.get_int64("data_size"));
  auto source_info     = source_sink.make_source_info();

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  auto const mem_stats_logger = cudf::memory_stats_logger();

  state.exec(
    nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
      drop_page_cache_if_enabled(source_info.filepaths());

      timer.start();
      auto result = cudf::io::experimental::read_cudftable(
        cudf::io::experimental::cudftable_reader_options::builder(source_info).build());
      timer.stop();

      CUDF_EXPECTS(result.table.num_columns() == num_cols_to_read, "Unexpected number of columns");
      CUDF_EXPECTS(result.table.num_rows() == num_rows_to_read, "Unexpected number of rows");
    });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
}

void BM_cudftable_read_data_sizes(nvbench::state& state)
{
  auto const d_type      = get_type_or_group(static_cast<int32_t>(data_type::INTEGRAL));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl =
      create_random_table(cycle_dtypes(d_type, num_cols), table_size_bytes{data_size});
    auto const view = tbl->view();

    cudf::io::experimental::write_cudftable(
      cudf::io::experimental::cudftable_writer_options::builder(source_sink.make_sink_info(), view)
        .build());
    return view.num_rows();
  }();

  cudftable_read_common(num_rows_written, num_cols, source_sink, state);
}

template <data_type DataType>
void BM_cudftable_read_data_common(nvbench::state& state,
                                   data_profile const& profile,
                                   nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const d_type      = get_type_or_group(static_cast<int32_t>(DataType));
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl =
      create_random_table(cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile);
    auto const view = tbl->view();

    cudf::io::experimental::write_cudftable(
      cudf::io::experimental::cudftable_writer_options::builder(source_sink.make_sink_info(), view)
        .build());
    return view.num_rows();
  }();

  cudftable_read_common(num_rows_written, num_cols, source_sink, state);
}

template <data_type DataType>
void BM_cudftable_read_data_types(nvbench::state& state,
                                  nvbench::type_list<nvbench::enum_type<DataType>> type_list)
{
  BM_cudftable_read_data_common<DataType>(state, data_profile{}, type_list);
}

template <data_type DataType>
void BM_cudftable_read_num_columns(nvbench::state& state,
                                   nvbench::type_list<nvbench::enum_type<DataType>> type_list)
{
  auto const d_type = get_type_or_group(static_cast<int32_t>(DataType));

  auto const n_col           = static_cast<cudf::size_type>(state.get_int64("num_cols"));
  auto const data_size_bytes = static_cast<size_t>(state.get_int64("data_size"));
  auto const source_type     = retrieve_io_type_enum(state.get_string("io_type"));
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl =
      create_random_table(cycle_dtypes(d_type, n_col), table_size_bytes{data_size_bytes});
    auto const view = tbl->view();

    cudf::io::experimental::write_cudftable(
      cudf::io::experimental::cudftable_writer_options::builder(source_sink.make_sink_info(), view)
        .build());
    return view.num_rows();
  }();

  cudftable_read_common(num_rows_written, n_col, source_sink, state);
}

using d_type_list_reduced = nvbench::enum_type_list<data_type::INTEGRAL,
                                                    data_type::FLOAT,
                                                    data_type::BOOL8,
                                                    data_type::DECIMAL,
                                                    data_type::TIMESTAMP,
                                                    data_type::DURATION,
                                                    data_type::STRING,
                                                    data_type::LIST,
                                                    data_type::STRUCT>;
NVBENCH_BENCH_TYPES(BM_cudftable_read_data_types, NVBENCH_TYPE_AXES(d_type_list_reduced))
  .set_name("cudftable_read_data_types")
  .set_type_axes_names({"data_type"})
  .add_string_axis("io_type", {"FILEPATH", "HOST_BUFFER", "DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("data_size", {128 << 20});

NVBENCH_BENCH(BM_cudftable_read_data_sizes)
  .set_name("cudftable_read_data_sizes")
  .set_min_samples(4)
  .add_string_axis("io_type", {"FILEPATH", "HOST_BUFFER", "DEVICE_BUFFER"})
  .add_int64_power_of_two_axis("data_size", nvbench::range(24, 31, 1));  // 16MB to 2GB

NVBENCH_BENCH_TYPES(BM_cudftable_read_num_columns,
                    NVBENCH_TYPE_AXES(nvbench::enum_type_list<data_type::STRING>))
  .set_name("cudftable_read_num_columns")
  .set_type_axes_names({"data_type"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("data_size", {128 << 20})
  .add_int64_power_of_two_axis("num_cols", nvbench::range(0, 12, 2));  // 1 to 4096

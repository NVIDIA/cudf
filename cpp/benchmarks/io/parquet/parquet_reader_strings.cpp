/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_common.hpp"

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/parquet.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

void BM_parquet_read_long_strings(nvbench::state& state)
{
  auto const cardinality = static_cast<cudf::size_type>(state.get_int64("cardinality"));
  auto const data_size   = static_cast<size_t>(state.get_int64("data_size"));

  auto const d_type        = get_type_or_group(static_cast<int32_t>(data_type::STRING));
  auto const source_type   = retrieve_io_type_enum(state.get_string("io_type"));
  auto const compression   = cudf::io::compression_type::SNAPPY;
  auto const rg_size_bytes = state.get_int64("row_group_size_bytes");
  auto const rg_size_rows  = state.get_int64("row_group_size_rows");
  cuio_source_sink_pair source_sink(source_type);

  auto const avg_string_length = static_cast<cudf::size_type>(state.get_int64("avg_string_length"));
  // corresponds to 3 sigma (full width 6 sigma: 99.7% of range)
  auto const half_width =
    avg_string_length >> 3;  // 32 +/- 4, 128 +/- 16, 1024 +/- 128, 8k +/- 1k, etc.
  auto const length_min = avg_string_length - half_width;
  auto const length_max = avg_string_length + half_width;

  data_profile profile =
    data_profile_builder()
      .cardinality(cardinality)
      .avg_run_length(1)
      .distribution(data_type::STRING, distribution_id::NORMAL, length_min, length_max);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols), table_size_bytes{data_size}, profile);  // THIS
    auto const view = tbl->view();

    // set smaller threshold to reduce file size and execution time
    auto const threshold = 1;
    setenv("LIBCUDF_LARGE_STRINGS_THRESHOLD", std::to_string(threshold).c_str(), 1);

    cudf::io::parquet_writer_options write_opts =
      cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(compression);
    // Sentinel 0 == use cuDF default (parquet bytes default is size_t::max).
    if (rg_size_bytes > 0) write_opts.set_row_group_size_bytes(rg_size_bytes);
    if (rg_size_rows > 0) write_opts.set_row_group_size_rows(rg_size_rows);
    cudf::io::write_parquet(write_opts);
    return view.num_rows();
  }();

  parquet_read_common(num_rows_written, num_cols, source_sink, state);
  unsetenv("LIBCUDF_LARGE_STRINGS_THRESHOLD");
}

void BM_parquet_read_file_shape(nvbench::state& state)
{
  // Currently the parquet reader only reads the page index if there are string columns
  auto constexpr d_type = cudf::type_id::STRING;

  auto const source_type    = retrieve_io_type_enum(state.get_string("io_type"));
  auto const num_rows       = static_cast<cudf::size_type>(state.get_int64("num_rows"));
  auto const num_row_groups = static_cast<cudf::size_type>(state.get_int64("num_row_groups"));
  auto const num_pages_per_row_group =
    static_cast<cudf::size_type>(state.get_int64("pages_per_row_group"));
  auto const has_page_idx = static_cast<bool>(state.get_int64("has_page_idx"));

  cuio_source_sink_pair source_sink(source_type);

  auto const tbl =
    create_random_table({d_type},
                        row_count{num_rows},
                        data_profile_builder().cardinality(num_rows / 10).avg_run_length(4));
  auto const view = tbl->view();

  cudf::io::parquet_writer_options write_opts =
    cudf::io::parquet_writer_options::builder(source_sink.make_sink_info(), view)
      .compression(cudf::io::compression_type::NONE)
      .row_group_size_rows(num_rows / num_row_groups)
      .max_page_size_rows(num_rows / (num_row_groups * num_pages_per_row_group))
      // Write page index by setting stats_level to STATISTICS_COLUMN
      .stats_level(has_page_idx ? cudf::io::statistics_freq::STATISTICS_COLUMN
                                : cudf::io::statistics_freq::STATISTICS_ROWGROUP);
  cudf::io::write_parquet(write_opts);

  cudf::io::parquet_reader_options read_opts =
    cudf::io::parquet_reader_options::builder(source_sink.make_source_info());

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync | nvbench::exec_tag::timer,
             [&](nvbench::launch& launch, auto& timer) {
               drop_page_cache_if_enabled(read_opts.get_source().filepaths());

               timer.start();
               auto const result = cudf::io::read_parquet(read_opts);
               timer.stop();

               CUDF_EXPECTS(result.tbl->num_columns() == 1, "Unexpected number of columns");
               CUDF_EXPECTS(result.tbl->num_rows() == num_rows, "Unexpected number of rows");
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(num_rows) / time, "rows_per_sec");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

NVBENCH_BENCH(BM_parquet_read_long_strings)
  .set_name("parquet_read_long_strings")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("data_size", {512 << 20})
  .add_int64_power_of_two_axis("avg_string_length", nvbench::range(4, 16, 2))  // 16, 64, ... -> 64k
  .add_int64_axis("row_group_size_bytes", {0})
  .add_int64_axis("row_group_size_rows", {0});

NVBENCH_BENCH(BM_parquet_read_file_shape)
  .set_name("parquet_read_file_shape")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("num_rows", {10'000'000, 100'000'000})
  .add_int64_axis("num_row_groups", {1, 10})
  .add_int64_axis("pages_per_row_group", {1'000, 10'000})
  .add_int64_axis("has_page_idx", {true, false});

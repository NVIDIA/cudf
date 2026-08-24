/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/orc.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

namespace {

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr cudf::size_type num_cols = 64;
constexpr std::size_t data_size    = 512 << 20;
constexpr std::size_t Mbytes       = 1024 * 1024;

template <bool is_chunked_read>
void orc_read_common(cudf::size_type num_rows_to_read,
                     cuio_source_sink_pair& source_sink,
                     nvbench::state& state)
{
  auto const read_opts =
    cudf::io::orc_reader_options::builder(source_sink.make_source_info()).build();

  auto mem_stats_logger = cudf::memory_stats_logger();  // init stats logger
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));

  if constexpr (is_chunked_read) {
    state.exec(
      nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
        drop_page_cache_if_enabled(read_opts.get_source().filepaths());
        auto const output_limit_MB =
          static_cast<std::size_t>(state.get_int64("chunk_read_limit_MB"));
        auto const read_limit_MB = static_cast<std::size_t>(state.get_int64("pass_read_limit_MB"));

        auto reader =
          cudf::io::chunked_orc_reader(output_limit_MB * Mbytes, read_limit_MB * Mbytes, read_opts);
        cudf::size_type num_rows{0};

        timer.start();
        do {
          auto chunk = reader.read_chunk();
          num_rows += chunk.tbl->num_rows();
        } while (reader.has_next());
        timer.stop();

        CUDF_EXPECTS(num_rows == num_rows_to_read, "Unexpected number of rows");
      });
  } else {  // not is_chunked_read
    state.exec(
      nvbench::exec_tag::sync | nvbench::exec_tag::timer, [&](nvbench::launch&, auto& timer) {
        drop_page_cache_if_enabled(read_opts.get_source().filepaths());

        timer.start();
        auto const result = cudf::io::read_orc(read_opts);
        timer.stop();

        CUDF_EXPECTS(result.tbl->num_columns() == num_cols, "Unexpected number of columns");
        CUDF_EXPECTS(result.tbl->num_rows() == num_rows_to_read, "Unexpected number of rows");
      });
  }

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(source_sink.size(), "encoded_file_size", "encoded_file_size");
}

}  // namespace

template <data_type DataType>
void BM_orc_read_data(nvbench::state& state, nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const d_type                 = get_type_or_group(static_cast<int32_t>(DataType));
  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  auto const source_type            = retrieve_io_type_enum(state.get_string("io_type"));
  auto const stripe_size_bytes      = state.get_int64("stripe_size_bytes");
  auto const stripe_size_rows       = state.get_int64("stripe_size_rows");
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols),
      table_size_bytes{data_size},
      data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::orc_writer_options opts =
      cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view);
    // Sentinel 0 == use cuDF default.
    if (stripe_size_bytes > 0) opts.set_stripe_size_bytes(stripe_size_bytes);
    if (stripe_size_rows > 0) opts.set_stripe_size_rows(stripe_size_rows);
    cudf::io::write_orc(opts);
    return view.num_rows();
  }();

  orc_read_common<false>(num_rows_written, source_sink, state);
}

template <bool chunked_read>
void orc_read_io_compression(nvbench::state& state)
{
  auto const source_type = retrieve_io_type_enum(state.get_string("io_type"));
  auto const compression = retrieve_compression_type_enum(state.get_string("compression_type"));
  auto const d_type      = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
                                              static_cast<int32_t>(data_type::FLOAT),
                                              static_cast<int32_t>(data_type::DECIMAL),
                                              static_cast<int32_t>(data_type::TIMESTAMP),
                                              static_cast<int32_t>(data_type::STRING),
                                              static_cast<int32_t>(data_type::LIST),
                                              static_cast<int32_t>(data_type::STRUCT)});

  auto const [cardinality, run_length] = [&]() -> std::pair<cudf::size_type, cudf::size_type> {
    if constexpr (chunked_read) {
      return {0, 4};
    } else {
      return {static_cast<cudf::size_type>(state.get_int64("cardinality")),
              static_cast<cudf::size_type>(state.get_int64("run_length"))};
    }
  }();
  auto const stripe_size_bytes = state.get_int64("stripe_size_bytes");
  auto const stripe_size_rows  = state.get_int64("stripe_size_rows");
  cuio_source_sink_pair source_sink(source_type);

  auto const num_rows_written = [&]() {
    auto const tbl = create_random_table(
      cycle_dtypes(d_type, num_cols),
      table_size_bytes{data_size},
      data_profile_builder{}.cardinality(cardinality).avg_run_length(run_length));
    auto const view = tbl->view();

    cudf::io::orc_writer_options opts =
      cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view)
        .compression(compression);
    if (stripe_size_bytes > 0) opts.set_stripe_size_bytes(stripe_size_bytes);
    if (stripe_size_rows > 0) opts.set_stripe_size_rows(stripe_size_rows);
    cudf::io::write_orc(opts);
    return view.num_rows();
  }();

  orc_read_common<chunked_read>(num_rows_written, source_sink, state);
}

void BM_orc_read_io_compression(nvbench::state& state)
{
  return orc_read_io_compression<false>(state);
}

void BM_orc_chunked_read_io_compression(nvbench::state& state)
{
  return orc_read_io_compression<true>(state);
}

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL_SIGNED,
                                            data_type::FLOAT,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::STRING,
                                            data_type::LIST,
                                            data_type::STRUCT>;

NVBENCH_BENCH_TYPES(BM_orc_read_data, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("orc_read_decode")
  .set_type_axes_names({"data_type"})
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("stripe_size_bytes", {0})
  .add_int64_axis("stripe_size_rows", {0});

NVBENCH_BENCH(BM_orc_read_io_compression)
  .set_name("orc_read_io_compression")
  .add_string_axis("io_type", {"FILEPATH", "HOST_BUFFER", "DEVICE_BUFFER"})
  .add_string_axis("compression_type", {"SNAPPY", "ZSTD", "ZLIB", "LZ4", "NONE"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("stripe_size_bytes", {0})
  .add_int64_axis("stripe_size_rows", {0});

// Should have the same parameters as `BM_orc_read_io_compression` for comparison.
NVBENCH_BENCH(BM_orc_chunked_read_io_compression)
  .set_name("orc_chunked_read_io_compression")
  .add_string_axis("io_type", {"DEVICE_BUFFER"})
  .add_string_axis("compression_type", {"SNAPPY", "ZSTD", "ZLIB", "LZ4", "NONE"})
  .set_min_samples(4)
  // The input has approximately 520MB and 127K rows.
  // The limits below are given in MBs.
  .add_int64_axis("chunk_read_limit_MB", {50, 250, 700})
  .add_int64_axis("pass_read_limit_MB", {50, 250, 700})
  .add_int64_axis("stripe_size_bytes", {0})
  .add_int64_axis("stripe_size_rows", {0});

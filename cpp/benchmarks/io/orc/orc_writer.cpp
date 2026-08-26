/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>
#include <benchmarks/io/nvbench_helpers.hpp>

#include <cudf/io/orc.hpp>
#include <cudf/io/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvbench/nvbench.cuh>

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  cudf::io::statistics_freq,
  [](auto value) {
    switch (value) {
      case cudf::io::statistics_freq::STATISTICS_NONE: return "STATISTICS_NONE";
      case cudf::io::statistics_freq::STATISTICS_ROWGROUP: return "ORC_STATISTICS_STRIPE";
      case cudf::io::statistics_freq::STATISTICS_PAGE: return "ORC_STATISTICS_ROW_GROUP";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

// Size of the data in the benchmark dataframe; chosen to be low enough to allow benchmarks to
// run on most GPUs, but large enough to allow highest throughput
constexpr int64_t data_size        = 512 << 20;
constexpr cudf::size_type num_cols = 64;

template <data_type DataType>
void BM_orc_write_encode(nvbench::state& state, nvbench::type_list<nvbench::enum_type<DataType>>)
{
  auto const d_type                 = get_type_or_group(static_cast<int32_t>(DataType));
  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  auto const compression            = cudf::io::compression_type::SNAPPY;
  auto const sink_type              = io_type::VOID;
  auto const stripe_size_bytes      = state.get_int64("stripe_size_bytes");
  auto const stripe_size_rows       = state.get_int64("stripe_size_rows");

  auto const tbl =
    create_random_table(cycle_dtypes(d_type, num_cols),
                        table_size_bytes{data_size},
                        data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
  auto const view = tbl->view();

  std::size_t encoded_file_size = 0;

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               cuio_source_sink_pair source_sink(sink_type);

               timer.start();
               cudf::io::orc_writer_options options =
                 cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view)
                   .compression(compression);
               // Sentinel 0 == use cuDF default.
               if (stripe_size_bytes > 0) options.set_stripe_size_bytes(stripe_size_bytes);
               if (stripe_size_rows > 0) options.set_stripe_size_rows(stripe_size_rows);
               cudf::io::write_orc(options);
               timer.stop();

               encoded_file_size = source_sink.size();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
}

void BM_orc_write_io_compression(nvbench::state& state)
{
  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST),
                                         static_cast<int32_t>(data_type::STRUCT)});

  cudf::size_type const cardinality = state.get_int64("cardinality");
  cudf::size_type const run_length  = state.get_int64("run_length");
  auto const sink_type              = retrieve_io_type_enum(state.get_string("io_type"));
  auto const compression = retrieve_compression_type_enum(state.get_string("compression_type"));
  auto const stripe_size_bytes = state.get_int64("stripe_size_bytes");
  auto const stripe_size_rows  = state.get_int64("stripe_size_rows");

  auto const tbl =
    create_random_table(cycle_dtypes(d_type, num_cols),
                        table_size_bytes{data_size},
                        data_profile_builder().cardinality(cardinality).avg_run_length(run_length));
  auto const view = tbl->view();

  std::size_t encoded_file_size = 0;

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               cuio_source_sink_pair source_sink(sink_type);

               timer.start();
               cudf::io::orc_writer_options options =
                 cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view)
                   .compression(compression);
               if (stripe_size_bytes > 0) options.set_stripe_size_bytes(stripe_size_bytes);
               if (stripe_size_rows > 0) options.set_stripe_size_rows(stripe_size_rows);
               cudf::io::write_orc(options);
               timer.stop();

               encoded_file_size = source_sink.size();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
}

template <cudf::io::statistics_freq Statistics>
void BM_orc_write_statistics(nvbench::state& state,
                             nvbench::type_list<nvbench::enum_type<Statistics>>)
{
  auto const d_type = get_type_or_group({static_cast<int32_t>(data_type::INTEGRAL_SIGNED),
                                         static_cast<int32_t>(data_type::FLOAT),
                                         static_cast<int32_t>(data_type::DECIMAL),
                                         static_cast<int32_t>(data_type::TIMESTAMP),
                                         static_cast<int32_t>(data_type::STRING),
                                         static_cast<int32_t>(data_type::LIST)});

  auto const compression = retrieve_compression_type_enum(state.get_string("compression_type"));
  auto const stats_freq  = Statistics;
  auto const stripe_size_bytes = state.get_int64("stripe_size_bytes");
  auto const stripe_size_rows  = state.get_int64("stripe_size_rows");

  auto const tbl  = create_random_table(d_type, table_size_bytes{data_size});
  auto const view = tbl->view();

  std::size_t encoded_file_size = 0;

  auto mem_stats_logger = cudf::memory_stats_logger();
  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::timer | nvbench::exec_tag::sync,
             [&](nvbench::launch& launch, auto& timer) {
               cuio_source_sink_pair source_sink(io_type::FILEPATH);

               timer.start();
               cudf::io::orc_writer_options options =
                 cudf::io::orc_writer_options::builder(source_sink.make_sink_info(), view)
                   .compression(compression)
                   .enable_statistics(stats_freq);
               if (stripe_size_bytes > 0) options.set_stripe_size_bytes(stripe_size_bytes);
               if (stripe_size_rows > 0) options.set_stripe_size_rows(stripe_size_rows);
               cudf::io::write_orc(options);
               timer.stop();

               encoded_file_size = source_sink.size();
             });

  auto const time = state.get_summary("nv/cold/time/gpu/mean").get_float64("value");
  state.add_element_count(static_cast<double>(data_size) / time, "bytes_per_second");
  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(encoded_file_size, "encoded_file_size", "encoded_file_size");
}

using d_type_list = nvbench::enum_type_list<data_type::INTEGRAL_SIGNED,
                                            data_type::FLOAT,
                                            data_type::DECIMAL,
                                            data_type::TIMESTAMP,
                                            data_type::STRING,
                                            data_type::LIST,
                                            data_type::STRUCT>;

using stats_list = nvbench::enum_type_list<cudf::io::STATISTICS_NONE,
                                           cudf::io::ORC_STATISTICS_STRIPE,
                                           cudf::io::ORC_STATISTICS_ROW_GROUP>;

NVBENCH_BENCH_TYPES(BM_orc_write_encode, NVBENCH_TYPE_AXES(d_type_list))
  .set_name("orc_write_encode")
  .set_type_axes_names({"data_type"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("stripe_size_bytes", {0})
  .add_int64_axis("stripe_size_rows", {0});

NVBENCH_BENCH(BM_orc_write_io_compression)
  .set_name("orc_write_io_compression")
  .add_string_axis("io_type", {"FILEPATH", "HOST_BUFFER", "VOID"})
  .add_string_axis("compression_type", {"SNAPPY", "ZSTD", "ZLIB", "LZ4", "NONE"})
  .set_min_samples(4)
  .add_int64_axis("cardinality", {0, 1000})
  .add_int64_axis("run_length", {1, 32})
  .add_int64_axis("stripe_size_bytes", {0})
  .add_int64_axis("stripe_size_rows", {0});

NVBENCH_BENCH_TYPES(BM_orc_write_statistics, NVBENCH_TYPE_AXES(stats_list))
  .set_name("orc_write_statistics")
  .set_type_axes_names({"statistics"})
  .add_string_axis("compression_type", {"SNAPPY", "ZSTD", "ZLIB", "LZ4", "NONE"})
  .set_min_samples(4)
  .add_int64_axis("stripe_size_bytes", {0})
  .add_int64_axis("stripe_size_rows", {0});

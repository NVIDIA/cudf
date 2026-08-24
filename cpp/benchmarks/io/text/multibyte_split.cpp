/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/common/generate_input.hpp>
#include <benchmarks/common/memory_stats.hpp>
#include <benchmarks/io/cuio_common.hpp>

#include <cudf_test/file_utilities.hpp>

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/io/text/data_chunk_source_factories.hpp>
#include <cudf/io/text/detail/bgzip_utils.hpp>
#include <cudf/io/text/multibyte_split.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <thrust/transform.h>

#include <nvbench/nvbench.cuh>

#include <cstdio>
#include <fstream>
#include <memory>
#include <numeric>
#include <random>

temp_directory const temp_dir("cudf_nvbench");

enum class data_chunk_source_type { device, file, file_datasource, host, host_pinned, file_bgzip };

NVBENCH_DECLARE_ENUM_TYPE_STRINGS(
  data_chunk_source_type,
  [](auto value) {
    switch (value) {
      case data_chunk_source_type::device: return "device";
      case data_chunk_source_type::file: return "file";
      case data_chunk_source_type::file_datasource: return "file_datasource";
      case data_chunk_source_type::host: return "host";
      case data_chunk_source_type::host_pinned: return "host_pinned";
      case data_chunk_source_type::file_bgzip: return "file_bgzip";
      default: return "Unknown";
    }
  },
  [](auto) { return std::string{}; })

static cudf::string_scalar create_random_input(int32_t num_chars,
                                               double delim_factor,
                                               double deviation,
                                               std::string delim)
{
  auto const num_delims      = static_cast<int32_t>((num_chars * delim_factor) / delim.size());
  auto const num_delim_chars = num_delims * delim.size();
  auto const num_value_chars = num_chars - num_delim_chars;
  auto const num_rows        = num_delims;
  auto const value_size_avg  = static_cast<int32_t>(num_value_chars / num_rows);
  auto const value_size_min  = static_cast<int32_t>(value_size_avg * (1 - deviation));
  auto const value_size_max  = static_cast<int32_t>(value_size_avg * (1 + deviation));

  data_profile const table_profile = data_profile_builder().distribution(
    cudf::type_id::STRING, distribution_id::NORMAL, value_size_min, value_size_max);

  auto const values =
    create_random_column(cudf::type_id::STRING, row_count{num_rows}, table_profile);

  auto delim_scalar  = cudf::make_string_scalar(delim);
  auto delims_column = cudf::make_column_from_scalar(*delim_scalar, num_rows);
  auto input_table   = cudf::table_view({values->view(), delims_column->view()});
  auto input_column  = cudf::strings::concatenate(input_table);

  // extract the chars from the returned strings column.
  auto input_column_contents = input_column->release();
  auto chars_buffer          = input_column_contents.data.release();

  // turn the chars in to a string scalar.
  return cudf::string_scalar(std::move(*chars_buffer));
}

static void write_bgzip_file(cudf::host_span<char const> host_data, std::ostream& output_stream)
{
  // a bit of variability with a decent amount of padding so we don't overflow 16 bit block sizes
  std::uniform_int_distribution<std::size_t> chunk_size_dist{64000, 65000};
  std::default_random_engine rng{};
  std::size_t pos = 0;
  while (pos < host_data.size()) {
    auto const remainder  = host_data.size() - pos;
    auto const chunk_size = std::min(remainder, chunk_size_dist(rng));
    cudf::io::text::detail::bgzip::write_compressed_block(output_stream,
                                                          {host_data.data() + pos, chunk_size});
    pos += chunk_size;
  }
  // empty block denotes EOF
  cudf::io::text::detail::bgzip::write_uncompressed_block(output_stream, {});
}

template <data_chunk_source_type source_type>
static void bench_multibyte_split(nvbench::state& state,
                                  nvbench::type_list<nvbench::enum_type<source_type>>)
{
  auto const delim_size         = state.get_int64("delim_size");
  auto const delim_percent      = state.get_int64("delim_percent");
  auto const file_size_approx   = state.get_int64("size_approx");
  auto const byte_range_percent = state.get_int64("byte_range_percent");
  auto const strip_delimiters   = bool(state.get_int64("strip_delimiters"));

  auto const byte_range_factor = static_cast<double>(byte_range_percent) / 100;
  CUDF_EXPECTS(delim_percent >= 1, "delimiter percent must be at least 1");
  CUDF_EXPECTS(delim_percent <= 50, "delimiter percent must be at most 50");
  CUDF_EXPECTS(byte_range_percent >= 1, "byte range percent must be at least 1");
  CUDF_EXPECTS(byte_range_percent <= 100, "byte range percent must be at most 100");

  auto delim = std::string(delim_size, '0');
  // the algorithm can only support 7 equal characters, so use different chars in the delimiter
  std::iota(delim.begin(), delim.end(), '1');

  auto const delim_factor = static_cast<double>(delim_percent) / 100;
  std::unique_ptr<cudf::io::datasource> datasource;
  auto device_input      = create_random_input(file_size_approx, delim_factor, 0.05, delim);
  auto host_input        = std::vector<char>{};
  auto host_pinned_input = cudf::detail::make_host_vector<char>(0, cudf::get_default_stream());

  if (source_type != data_chunk_source_type::device &&
      source_type != data_chunk_source_type::host_pinned) {
    host_input = cudf::detail::make_std_vector<char>(
      {device_input.data(), static_cast<std::size_t>(device_input.size())},
      cudf::get_default_stream());
  }
  if (source_type == data_chunk_source_type::host_pinned) {
    host_pinned_input = cudf::detail::make_pinned_vector(
      cudf::device_span<char const>{device_input.data(),
                                    static_cast<std::size_t>(device_input.size())},
      cudf::get_default_stream());
  }

  std::string file_name;
  auto source = [&] {
    switch (source_type) {
      case data_chunk_source_type::file:
      case data_chunk_source_type::file_datasource: {
        file_name = random_file_in_dir(temp_dir.path());
        std::ofstream(file_name, std::ofstream::out).write(host_input.data(), host_input.size());
        if (source_type == data_chunk_source_type::file) {
          return cudf::io::text::make_source_from_file(file_name);
        } else {
          datasource = cudf::io::datasource::create(file_name);
          return cudf::io::text::make_source(*datasource);
        }
      }
      case data_chunk_source_type::host:  //
        return cudf::io::text::make_source(host_input);
      case data_chunk_source_type::host_pinned:
        return cudf::io::text::make_source(host_pinned_input);
      case data_chunk_source_type::device:  //
        return cudf::io::text::make_source(device_input);
      case data_chunk_source_type::file_bgzip: {
        auto const temp_file_name = random_file_in_dir(temp_dir.path());
        {
          std::ofstream output_stream(temp_file_name, std::ofstream::out);
          write_bgzip_file(host_input, output_stream);
        }
        return cudf::io::text::make_source_from_bgzip_file(temp_file_name);
      }
      default: CUDF_FAIL("Unexpected case");
    }
  }();

  auto mem_stats_logger   = cudf::memory_stats_logger();
  auto const range_size   = static_cast<int64_t>(device_input.size() * byte_range_factor);
  auto const range_offset = (device_input.size() - range_size) / 2;
  cudf::io::text::byte_range_info range{range_offset, range_size};
  cudf::io::text::parse_options options{range, strip_delimiters};
  std::unique_ptr<cudf::column> output;

  state.set_cuda_stream(nvbench::make_cuda_stream_view(cudf::get_default_stream().value()));
  state.exec(nvbench::exec_tag::sync, [&](nvbench::launch& launch) {
    drop_page_cache_if_enabled({file_name});
    output = cudf::io::text::multibyte_split(*source, delim, options);
  });

  state.add_buffer_size(
    mem_stats_logger.peak_memory_usage(), "peak_memory_usage", "peak_memory_usage");
  state.add_buffer_size(range_size, "encoded_file_size", "encoded_file_size");
}

using source_type_list = nvbench::enum_type_list<data_chunk_source_type::device,
                                                 data_chunk_source_type::file,
                                                 data_chunk_source_type::file_datasource,
                                                 data_chunk_source_type::host,
                                                 data_chunk_source_type::host_pinned,
                                                 data_chunk_source_type::file_bgzip>;

NVBENCH_BENCH_TYPES(bench_multibyte_split,
                    NVBENCH_TYPE_AXES(nvbench::enum_type_list<data_chunk_source_type::file>))
  .set_name("multibyte_split_delimiters")
  .set_min_samples(4)
  .add_int64_axis("strip_delimiters", {0, 1})
  .add_int64_axis("delim_size", {1, 4, 7})
  .add_int64_axis("delim_percent", {1, 25})
  .add_int64_power_of_two_axis("size_approx", {15})
  .add_int64_axis("byte_range_percent", {50});

NVBENCH_BENCH_TYPES(bench_multibyte_split, NVBENCH_TYPE_AXES(source_type_list))
  .set_name("multibyte_split_source")
  .set_min_samples(4)
  .add_int64_axis("strip_delimiters", {0, 1})
  .add_int64_axis("delim_size", {1})
  .add_int64_axis("delim_percent", {1})
  .add_int64_power_of_two_axis("size_approx", {15, 30})
  .add_int64_axis("byte_range_percent", {10, 100});

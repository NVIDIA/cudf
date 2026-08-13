/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <benchmarks/io/cuio_common.hpp>

#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <kvikio/file_utils.hpp>

#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <unistd.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <regex>
#include <string>
#include <utility>

temp_directory const cuio_source_sink_pair::tmpdir{"cudf_bench"};

// Don't use cudf's pinned pool for the source data
rmm::host_async_resource_ref pinned_memory_resource()
{
  static auto mr = rmm::mr::pinned_host_memory_resource{};

  return mr;
}

std::string random_file_in_dir(std::string const& dir_path)
{
  // `mkstemp` modifies the template in place
  std::string filename = dir_path + "io.XXXXXX";

  // `mkstemp` opens the file; closing immediately, only need the name
  close(mkstemp(const_cast<char*>(filename.data())));

  return filename;
}

cuio_source_sink_pair::cuio_source_sink_pair(io_type type_param)
  : type{type_param},
    pinned_buffer({pinned_memory_resource(), cudf::get_default_stream()}),
    d_buffer{0, cudf::get_default_stream()},
    file_name{random_file_in_dir(tmpdir.path())},
    void_sink{cudf::io::data_sink::create()},
    owns_file{true}
{
}

cuio_source_sink_pair::~cuio_source_sink_pair()
{
  if (owns_file) { cleanup(); }
}

cuio_source_sink_pair::cuio_source_sink_pair(cuio_source_sink_pair&& ss) noexcept
  : type{std::exchange(ss.type, io_type::VOID)},
    h_buffer{std::move(ss.h_buffer)},
    pinned_buffer{std::move(ss.pinned_buffer)},
    d_buffer{std::move(ss.d_buffer)},
    file_name{std::move(ss.file_name)},
    void_sink{std::move(ss.void_sink)},
    owns_file{std::exchange(ss.owns_file, false)}
{
}

cuio_source_sink_pair& cuio_source_sink_pair::operator=(cuio_source_sink_pair&& ss) noexcept
{
  if (this != &ss) {
    if (owns_file) {
      // Clean up current resource. This needs to happen before file_name is reassigned.
      cleanup();
    }

    type          = std::exchange(ss.type, io_type::VOID);
    h_buffer      = std::move(ss.h_buffer);
    pinned_buffer = std::move(ss.pinned_buffer);
    d_buffer      = std::move(ss.d_buffer);
    file_name     = std::move(ss.file_name);
    void_sink     = std::move(ss.void_sink);
    owns_file     = std::exchange(ss.owns_file, false);
  }
  return *this;
}

void cuio_source_sink_pair::cleanup() { std::remove(file_name.c_str()); }

cudf::io::source_info cuio_source_sink_pair::make_source_info()
{
  switch (type) {
    case io_type::FILEPATH: return cudf::io::source_info(file_name);
    case io_type::HOST_BUFFER:
      return cudf::io::source_info(cudf::host_span<std::byte const>(
        reinterpret_cast<std::byte const*>(h_buffer.data()), h_buffer.size()));
    case io_type::PINNED_BUFFER: {
      pinned_buffer.resize(h_buffer.size());
      std::copy(h_buffer.begin(), h_buffer.end(), pinned_buffer.begin());
      return cudf::io::source_info(cudf::host_span<std::byte const>(
        reinterpret_cast<std::byte const*>(pinned_buffer.data()), pinned_buffer.size()));
    }
    case io_type::DEVICE_BUFFER: {
      // TODO: make cuio_source_sink_pair stream-friendly and avoid implicit use of the default
      // stream
      auto const stream = cudf::get_default_stream();
      d_buffer.resize(h_buffer.size(), stream);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
        d_buffer.data(), h_buffer.data(), h_buffer.size(), cudaMemcpyDefault, stream.value()));

      return cudf::io::source_info(d_buffer);
    }
    default: CUDF_FAIL("invalid input type");
  }
}

cudf::io::sink_info cuio_source_sink_pair::make_sink_info()
{
  switch (type) {
    case io_type::VOID: return cudf::io::sink_info(void_sink.get());
    case io_type::FILEPATH: return cudf::io::sink_info(file_name);
    case io_type::HOST_BUFFER:
    case io_type::PINNED_BUFFER:
    case io_type::DEVICE_BUFFER: return cudf::io::sink_info(&h_buffer);
    default: CUDF_FAIL("invalid output type");
  }
}

size_t cuio_source_sink_pair::size()
{
  switch (type) {
    case io_type::VOID: return void_sink->bytes_written();
    case io_type::FILEPATH:
      return static_cast<size_t>(
        std::ifstream(file_name, std::ifstream::ate | std::ifstream::binary).tellg());
    case io_type::HOST_BUFFER:
    case io_type::PINNED_BUFFER:
    case io_type::DEVICE_BUFFER: return h_buffer.size();
    default: CUDF_FAIL("invalid output type");
  }
}

std::vector<cudf::type_id> dtypes_for_column_selection(std::vector<cudf::type_id> const& data_types,
                                                       column_selection col_sel)
{
  std::vector<cudf::type_id> out_dtypes;
  out_dtypes.reserve(2 * data_types.size());
  switch (col_sel) {
    case column_selection::ALL:
    case column_selection::FIRST_HALF:
    case column_selection::SECOND_HALF:
      std::copy(data_types.begin(), data_types.end(), std::back_inserter(out_dtypes));
      std::copy(data_types.begin(), data_types.end(), std::back_inserter(out_dtypes));
      break;
    case column_selection::ALTERNATE:
      for (auto const& type : data_types) {
        out_dtypes.push_back(type);
        out_dtypes.push_back(type);
      }
      break;
  }
  return out_dtypes;
}

std::vector<int> select_column_indexes(int num_cols, column_selection col_sel)
{
  std::vector<int> col_idxs(num_cols / 2);
  switch (col_sel) {
    case column_selection::ALL: col_idxs.resize(num_cols);
    case column_selection::FIRST_HALF:
    case column_selection::SECOND_HALF:
      std::iota(std::begin(col_idxs),
                std::end(col_idxs),
                (col_sel == column_selection::SECOND_HALF) ? num_cols / 2 : 0);
      break;
    case column_selection::ALTERNATE:
      for (size_t i = 0; i < col_idxs.size(); ++i)
        col_idxs[i] = 2 * i;
      break;
  }
  return col_idxs;
}

std::vector<std::string> select_column_names(std::vector<std::string> const& col_names,
                                             column_selection col_sel)
{
  auto const col_idxs_to_read = select_column_indexes(col_names.size(), col_sel);

  std::vector<std::string> col_names_to_read;
  std::transform(col_idxs_to_read.begin(),
                 col_idxs_to_read.end(),
                 std::back_inserter(col_names_to_read),
                 [&](auto& idx) { return col_names[idx]; });
  return col_names_to_read;
}

std::vector<cudf::size_type> segments_in_chunk(int num_segments, int num_chunks, int chunk_idx)
{
  CUDF_EXPECTS(num_segments >= num_chunks,
               "Number of chunks cannot be greater than the number of segments in the file");
  CUDF_EXPECTS(chunk_idx < num_chunks,
               "Chunk index must be smaller than the number of chunks in the file");

  auto const segments_in_chunk = cudf::util::div_rounding_up_unsafe(num_segments, num_chunks);
  auto const begin_segment     = std::min(chunk_idx * segments_in_chunk, num_segments);
  auto const end_segment       = std::min(begin_segment + segments_in_chunk, num_segments);
  std::vector<cudf::size_type> selected_segments(end_segment - begin_segment);
  std::iota(selected_segments.begin(), selected_segments.end(), begin_segment);

  return selected_segments;
}

namespace {
void log_page_cache_warning_once()
{
  static bool is_logged = false;
  if (not is_logged) {
    CUDF_LOG_WARN(
      "Running benchmarks without dropping the page cache; results may not reflect file IO "
      "throughput");
    is_logged = true;
  }
}

std::pair<bool, bool> parse_cache_dropping_env()
{
  bool is_drop_cache_enabled{false};
  bool is_file_scope{true};

  auto const* env = std::getenv("CUDF_BENCHMARK_DROP_CACHE");
  if (env == nullptr) { return {is_drop_cache_enabled, is_file_scope}; }

  // Trim leading/trailing whitespace
  std::regex static const pattern{R"(^\s+|\s+$)"};
  auto env_sanitized = std::regex_replace(env, pattern, "");

  // Convert to lowercase
  std::transform(
    env_sanitized.begin(), env_sanitized.end(), env_sanitized.begin(), [](unsigned char c) {
      return std::tolower(c);
    });

  // Interpret value
  if (env_sanitized == "true" or env_sanitized == "on" or env_sanitized == "yes" or
      env_sanitized == "1" or env_sanitized == "system" or env_sanitized == "file") {
    is_drop_cache_enabled = true;
    is_file_scope         = (env_sanitized != "system");
    return {is_drop_cache_enabled, is_file_scope};
  }

  if (env_sanitized == "false" or env_sanitized == "off" or env_sanitized == "no" or
      env_sanitized == "0") {
    return {is_drop_cache_enabled, is_file_scope};
  }

  CUDF_FAIL(
    "Environment variable CUDF_BENCHMARK_DROP_CACHE has an unknown value: " + std::string{env},
    std::invalid_argument);
}
}  // namespace

void drop_page_cache_if_enabled(std::vector<std::string> const& file_paths)
{
  static auto const parsed_env                = parse_cache_dropping_env();
  auto [is_drop_cache_enabled, is_file_scope] = parsed_env;
  if (not is_drop_cache_enabled) {
    log_page_cache_warning_once();
    return;
  }

  if (not is_file_scope) {
    auto const is_cache_dropped = kvikio::drop_system_page_cache();
    CUDF_EXPECTS(is_cache_dropped, "Failed to execute the drop cache command");
    return;
  }

  if (file_paths.empty()) {
    CUDF_LOG_WARN("No file is specified for page cache dropping");
    return;
  }

  for (auto const& path : file_paths) {
    if (path.empty()) { continue; }
    kvikio::drop_file_page_cache(path);
  }
}

io_type retrieve_io_type_enum(std::string_view io_string)
{
  if (io_string == "FILEPATH") { return io_type::FILEPATH; }
  if (io_string == "HOST_BUFFER") { return io_type::HOST_BUFFER; }
  if (io_string == "PINNED_BUFFER") { return io_type::PINNED_BUFFER; }
  if (io_string == "DEVICE_BUFFER") { return io_type::DEVICE_BUFFER; }
  if (io_string == "VOID") { return io_type::VOID; }
  CUDF_FAIL("Unsupported io_type.");
}

cudf::io::compression_type retrieve_compression_type_enum(std::string_view compression_string)
{
  if (compression_string == "NONE") { return cudf::io::compression_type::NONE; }
  if (compression_string == "AUTO") { return cudf::io::compression_type::AUTO; }
  if (compression_string == "SNAPPY") { return cudf::io::compression_type::SNAPPY; }
  if (compression_string == "GZIP") { return cudf::io::compression_type::GZIP; }
  if (compression_string == "BZIP2") { return cudf::io::compression_type::BZIP2; }
  if (compression_string == "BROTLI") { return cudf::io::compression_type::BROTLI; }
  if (compression_string == "ZIP") { return cudf::io::compression_type::ZIP; }
  if (compression_string == "XZ") { return cudf::io::compression_type::XZ; }
  if (compression_string == "ZLIB") { return cudf::io::compression_type::ZLIB; }
  if (compression_string == "LZ4") { return cudf::io::compression_type::LZ4; }
  if (compression_string == "LZO") { return cudf::io::compression_type::LZO; }
  if (compression_string == "ZSTD") { return cudf::io::compression_type::ZSTD; }
  CUDF_FAIL("Unsupported compression_type.");
}

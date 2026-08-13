/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/orc/orc.hpp"
#include "io/parquet/reader_impl_helpers.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/io/avro.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/data_sink.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/avro.hpp>
#include <cudf/io/detail/codec.hpp>
#include <cudf/io/detail/csv.hpp>
#include <cudf/io/detail/json.hpp>
#include <cudf/io/detail/orc.hpp>
#include <cudf/io/detail/parquet.hpp>
#include <cudf/io/detail/utils.hpp>
#include <cudf/io/experimental/cudftable.hpp>
#include <cudf/io/json.hpp>
#include <cudf/io/orc.hpp>
#include <cudf/io/orc_metadata.hpp>
#include <cudf/io/parquet.hpp>
#include <cudf/io/parquet_metadata.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <algorithm>
#include <functional>
#include <utility>

namespace cudf::io {
namespace {

compression_type infer_compression_type(compression_type compression, source_info const& info)
{
  if (compression != compression_type::AUTO) { return compression; }

  if (info.type() != io_type::FILEPATH) {
    CUDF_LOG_WARN(
      "Auto detection of compression type is supported only for file type buffers. For other "
      "buffer types, AUTO compression type assumes uncompressed input.");
    return compression_type::NONE;
  }

  auto filepath = info.filepaths()[0];

  // Attempt to infer from the file extension
  auto const pos = filepath.find_last_of('.');

  if (pos == std::string::npos) { return {}; }

  auto str_tolower = [](auto const& begin, auto const& end) {
    std::string out;
    std::transform(begin, end, std::back_inserter(out), ::tolower);
    return out;
  };

  auto const ext = str_tolower(filepath.begin() + pos + 1, filepath.end());

  if (ext == "gz") { return compression_type::GZIP; }
  if (ext == "zip") { return compression_type::ZIP; }
  if (ext == "bz2") { return compression_type::BZIP2; }
  if (ext == "zstd") { return compression_type::ZSTD; }
  if (ext == "sz") { return compression_type::SNAPPY; }
  if (ext == "xz") { return compression_type::XZ; }

  return compression_type::NONE;
}

}  // namespace

// Returns builder for csv_reader_options
csv_reader_options_builder csv_reader_options::builder(source_info src)
{
  return csv_reader_options_builder{std::move(src)};
}

// Returns builder for csv_writer_options
csv_writer_options_builder csv_writer_options::builder(sink_info const& sink,
                                                       table_view const& table)
{
  return csv_writer_options_builder{sink, table};
}

// Returns builder for orc_reader_options
orc_reader_options_builder orc_reader_options::builder(source_info src)
{
  return orc_reader_options_builder{std::move(src)};
}

// Returns builder for orc_writer_options
orc_writer_options_builder orc_writer_options::builder(sink_info const& sink,
                                                       table_view const& table)
{
  return orc_writer_options_builder{sink, table};
}

// Returns builder for chunked_orc_writer_options
chunked_orc_writer_options_builder chunked_orc_writer_options::builder(sink_info const& sink)
{
  return chunked_orc_writer_options_builder{sink};
}

// Returns builder for avro_reader_options
avro_reader_options_builder avro_reader_options::builder(source_info src)
{
  return avro_reader_options_builder(std::move(src));
}

// Returns builder for json_reader_options
json_reader_options_builder json_reader_options::builder(source_info src)
{
  return json_reader_options_builder(std::move(src));
}

// Returns builder for orc_writer_options
json_writer_options_builder json_writer_options::builder(sink_info const& sink,
                                                         table_view const& table)
{
  return json_writer_options_builder{sink, table};
}

// Returns builder for parquet_reader_options
parquet_reader_options_builder parquet_reader_options::builder(source_info src)
{
  return parquet_reader_options_builder{std::move(src)};
}

// Returns builder for parquet_writer_options
parquet_writer_options_builder parquet_writer_options::builder(sink_info const& sink,
                                                               table_view const& table)
{
  return parquet_writer_options_builder{sink, table};
}

// Returns builder for parquet_writer_options
parquet_writer_options_builder parquet_writer_options::builder()
{
  return parquet_writer_options_builder();
}

// Returns builder for chunked_parquet_writer_options
chunked_parquet_writer_options_builder chunked_parquet_writer_options::builder(
  sink_info const& sink)
{
  return chunked_parquet_writer_options_builder{sink};
}

/**
 * @copydoc cudf::io::make_datasources
 */
std::vector<std::unique_ptr<cudf::io::datasource>> make_datasources(source_info const& info,
                                                                    size_t offset,
                                                                    size_t max_size_estimate)
{
  switch (info.type()) {
    case io_type::FILEPATH: {
      std::vector<std::unique_ptr<cudf::io::datasource>> sources;
      sources.reserve(info.filepath_sources().size());
      // Creating sources in a single thread is faster for a small number of sources
      auto const pool_use_threshold =
        cudf::detail::getenv_or("LIBCUDF_DATASOURCE_PARALLEL_CREATION_THRESHOLD", 8ul);
      if (info.filepath_sources().size() >= pool_use_threshold) {
        std::vector<std::future<std::unique_ptr<cudf::io::datasource>>> source_tasks;
        source_tasks.reserve(info.filepath_sources().size());
        for (auto const& fs : info.filepath_sources()) {
          source_tasks.emplace_back(cudf::detail::host_worker_pool().submit_task([=] {
            return cudf::io::datasource::create(fs.path, offset, max_size_estimate, fs.size);
          }));
        }
        std::transform(
          source_tasks.begin(), source_tasks.end(), std::back_inserter(sources), [](auto& task) {
            return task.get();
          });
      } else {
        for (auto const& fs : info.filepath_sources()) {
          sources.emplace_back(
            cudf::io::datasource::create(fs.path, offset, max_size_estimate, fs.size));
        }
      }
      return sources;
    }
    case io_type::HOST_BUFFER: return cudf::io::datasource::create(info.host_buffers());
    case io_type::DEVICE_BUFFER: return cudf::io::datasource::create(info.device_buffers());
    case io_type::USER_IMPLEMENTED: return cudf::io::datasource::create(info.user_sources());
    default: CUDF_FAIL("Unsupported source type");
  }
}

namespace {

std::vector<std::unique_ptr<data_sink>> make_datasinks(sink_info const& info)
{
  switch (info.type()) {
    case io_type::FILEPATH: return cudf::io::data_sink::create(info.filepaths());
    case io_type::HOST_BUFFER: return cudf::io::data_sink::create(info.buffers());
    case io_type::VOID: {
      std::vector<std::unique_ptr<data_sink>> sinks;
      for (size_t i = 0; i < info.num_sinks(); ++i) {
        sinks.push_back(cudf::io::data_sink::create());
      }
      return sinks;
    }
    case io_type::USER_IMPLEMENTED: return cudf::io::data_sink::create(info.user_sinks());
    default: CUDF_FAIL("Unsupported sink type");
  }
}

}  // namespace

table_with_metadata read_avro(avro_reader_options const& options,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  namespace avro = cudf::io::detail::avro;

  CUDF_FUNC_RANGE();

  auto datasources = make_datasources(options.get_source());

  CUDF_EXPECTS(datasources.size() == 1, "Only a single source is currently supported.");

  return avro::read_avro(std::move(datasources[0]), options, stream, mr);
}

table_with_metadata read_json(json_reader_options options,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  options.set_compression(infer_compression_type(options.get_compression(), options.get_source()));

  auto datasources = make_datasources(options.get_source(),
                                      options.get_byte_range_offset(),
                                      options.get_byte_range_size_with_padding());

  return json::detail::read_json(datasources, options, stream, mr);
}

json_reader_result read_json_with_diagnostics(json_reader_options options,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  options.set_compression(infer_compression_type(options.get_compression(), options.get_source()));

  auto datasources = make_datasources(options.get_source(),
                                      options.get_byte_range_offset(),
                                      options.get_byte_range_size_with_padding());

  return json::detail::read_json_with_diagnostics(datasources, options, stream, mr);
}

json_reader_result_with_row_diagnostics read_json_with_row_diagnostics(
  json_reader_options options, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  options.set_compression(infer_compression_type(options.get_compression(), options.get_source()));

  auto datasources = make_datasources(options.get_source(),
                                      options.get_byte_range_offset(),
                                      options.get_byte_range_size_with_padding());

  return json::detail::read_json_with_row_diagnostics(datasources, options, stream, mr);
}

void write_json(json_writer_options const& options, rmm::cuda_stream_view stream)
{
  auto sinks = make_datasinks(options.get_sink());
  CUDF_EXPECTS(sinks.size() == 1, "Multiple sinks not supported for JSON writing");

  return json::detail::write_json(  //
    sinks[0].get(),
    options.get_table(),
    options,
    stream);
}

table_with_metadata read_csv(csv_reader_options options,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  options.set_compression(infer_compression_type(options.get_compression(), options.get_source()));

  auto datasources = make_datasources(options.get_source(),
                                      options.get_byte_range_offset(),
                                      options.get_byte_range_size_with_padding());

  CUDF_EXPECTS(datasources.size() == 1, "Only a single source is currently supported.");

  return cudf::io::detail::csv::read_csv(  //
    std::move(datasources[0]),
    options,
    stream,
    mr);
}

// Freeform API wraps the detail writer class API
void write_csv(csv_writer_options const& options, rmm::cuda_stream_view stream)
{
  using namespace cudf::io::detail;

  auto sinks = make_datasinks(options.get_sink());
  CUDF_EXPECTS(sinks.size() == 1, "Multiple sinks not supported for CSV writing");

  return csv::write_csv(  //
    sinks[0].get(),
    options.get_table(),
    options.get_names(),
    options,
    stream);
}

bool is_supported_read_orc(compression_type compression)
{
  if (compression == compression_type::AUTO or compression == compression_type::NONE) {
    return true;
  }

  return ((compression == compression_type::ZLIB or compression == compression_type::SNAPPY or
           compression == compression_type::ZSTD or compression == compression_type::LZ4) and
          detail::is_decompression_supported(compression));
}

bool is_supported_write_orc(compression_type compression)
{
  if (compression == compression_type::AUTO or compression == compression_type::NONE) {
    return true;
  }

  return ((compression == compression_type::ZLIB or compression == compression_type::SNAPPY or
           compression == compression_type::ZSTD or compression == compression_type::LZ4) and
          detail::is_compression_supported(compression));
}

raw_orc_statistics read_raw_orc_statistics(source_info const& src_info,
                                           rmm::cuda_stream_view stream)
{
  // Get source to read statistics from
  std::unique_ptr<datasource> source;
  if (src_info.type() == io_type::FILEPATH) {
    CUDF_EXPECTS(src_info.filepaths().size() == 1, "Only a single source is currently supported.");
    source = cudf::io::datasource::create(src_info.filepaths()[0]);
  } else if (src_info.type() == io_type::HOST_BUFFER) {
    CUDF_EXPECTS(src_info.host_buffers().size() == 1,
                 "Only a single source is currently supported.");
    source = cudf::io::datasource::create(src_info.host_buffers()[0]);
  } else if (src_info.type() == io_type::DEVICE_BUFFER) {
    CUDF_EXPECTS(src_info.device_buffers().size() == 1,
                 "Only a single source is currently supported.");
    source = cudf::io::datasource::create(src_info.device_buffers()[0]);
  } else if (src_info.type() == io_type::USER_IMPLEMENTED) {
    CUDF_EXPECTS(src_info.user_sources().size() == 1,
                 "Only a single source is currently supported.");
    source = cudf::io::datasource::create(src_info.user_sources()[0]);
  } else {
    CUDF_FAIL("Unsupported source type");
  }

  orc::detail::metadata const metadata(source.get(), stream);

  // Initialize statistics to return
  raw_orc_statistics result;

  // Get column names
  for (auto i = 0; i < metadata.get_num_columns(); i++) {
    result.column_names.push_back(metadata.column_name(i));
  }

  // Get file-level statistics, statistics of each column of file
  for (auto const& stats : metadata.ff.statistics) {
    result.file_stats.emplace_back(stats.cbegin(), stats.cend());
  }

  // Get stripe-level statistics
  for (auto const& stripes_stats : metadata.md.stripeStats) {
    result.stripes_stats.emplace_back();
    for (auto const& stats : stripes_stats.colStats) {
      result.stripes_stats.back().emplace_back(stats.cbegin(), stats.cend());
    }
  }

  return result;
}

column_statistics::column_statistics(orc::detail::column_statistics&& cs)
{
  number_of_values = cs.number_of_values;
  has_null         = cs.has_null;
  if (cs.int_stats) {
    type_specific_stats = *cs.int_stats;
  } else if (cs.double_stats) {
    type_specific_stats = *cs.double_stats;
  } else if (cs.string_stats) {
    type_specific_stats = *cs.string_stats;
  } else if (cs.bucket_stats) {
    type_specific_stats = *cs.bucket_stats;
  } else if (cs.decimal_stats) {
    type_specific_stats = *cs.decimal_stats;
  } else if (cs.date_stats) {
    type_specific_stats = *cs.date_stats;
  } else if (cs.binary_stats) {
    type_specific_stats = *cs.binary_stats;
  } else if (cs.timestamp_stats) {
    type_specific_stats = *cs.timestamp_stats;
  }
}

parsed_orc_statistics read_parsed_orc_statistics(source_info const& src_info,
                                                 rmm::cuda_stream_view stream)
{
  auto const raw_stats = read_raw_orc_statistics(src_info, stream);

  parsed_orc_statistics result;
  result.column_names = raw_stats.column_names;

  auto parse_column_statistics = [](auto const& raw_col_stats) {
    orc::detail::column_statistics stats_internal;
    orc::detail::protobuf_reader(reinterpret_cast<uint8_t const*>(raw_col_stats.c_str()),
                                 raw_col_stats.size())
      .read(stats_internal);
    return column_statistics(std::move(stats_internal));
  };

  std::transform(raw_stats.file_stats.cbegin(),
                 raw_stats.file_stats.cend(),
                 std::back_inserter(result.file_stats),
                 parse_column_statistics);

  for (auto const& raw_stripe_stats : raw_stats.stripes_stats) {
    result.stripes_stats.emplace_back();
    std::transform(raw_stripe_stats.cbegin(),
                   raw_stripe_stats.cend(),
                   std::back_inserter(result.stripes_stats.back()),
                   parse_column_statistics);
  }

  return result;
}
namespace {
orc_column_schema make_orc_column_schema(host_span<orc::detail::SchemaType const> orc_schema,
                                         uint32_t column_id,
                                         std::string column_name)
{
  auto const& orc_col_schema = orc_schema[column_id];
  std::vector<orc_column_schema> children;
  children.reserve(orc_col_schema.subtypes.size());
  std::transform(
    orc_col_schema.subtypes.cbegin(),
    orc_col_schema.subtypes.cend(),
    cudf::detail::make_counting_transform_iterator(0,
                                                   [&names = orc_col_schema.fieldNames](size_t i) {
                                                     return i < names.size() ? names[i]
                                                                             : std::string{};
                                                   }),
    std::back_inserter(children),
    [&](auto& type, auto name) { return make_orc_column_schema(orc_schema, type, name); });

  return {std::move(column_name), orc_schema[column_id].kind, std::move(children)};
}
};  // namespace

orc_metadata read_orc_metadata(source_info const& src_info, rmm::cuda_stream_view stream)
{
  auto sources = make_datasources(src_info);

  CUDF_EXPECTS(sources.size() == 1, "Only a single source is currently supported.");
  auto const footer = orc::detail::metadata(sources.front().get(), stream).ff;

  return {{make_orc_column_schema(footer.types, 0, "")},
          footer.numberOfRows,
          static_cast<size_type>(footer.stripes.size())};
}

/**
 * @copydoc cudf::io::read_orc
 */
table_with_metadata read_orc(orc_reader_options const& options,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto datasources = make_datasources(options.get_source());
  auto reader = std::make_unique<orc::detail::reader>(std::move(datasources), options, stream, mr);
  return reader->read();
}

/**
 * @copydoc cudf::io::write_orc
 */
void write_orc(orc_writer_options const& options, rmm::cuda_stream_view stream)
{
  namespace io_detail = cudf::io::detail;

  CUDF_FUNC_RANGE();

  auto sinks = make_datasinks(options.get_sink());
  CUDF_EXPECTS(sinks.size() == 1, "Multiple sinks not supported for ORC writing");

  auto writer = std::make_unique<orc::detail::writer>(
    std::move(sinks[0]), options, io_detail::single_write_mode::YES, stream);
  writer->write(options.get_table());
}

chunked_orc_reader::chunked_orc_reader(std::size_t chunk_read_limit,
                                       std::size_t pass_read_limit,
                                       size_type output_row_granularity,
                                       orc_reader_options const& options,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
  : reader{std::make_unique<orc::detail::chunked_reader>(chunk_read_limit,
                                                         pass_read_limit,
                                                         output_row_granularity,
                                                         make_datasources(options.get_source()),
                                                         options,
                                                         stream,
                                                         mr)}
{
}

chunked_orc_reader::chunked_orc_reader(std::size_t chunk_read_limit,
                                       std::size_t pass_read_limit,
                                       orc_reader_options const& options,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
  : reader{std::make_unique<orc::detail::chunked_reader>(chunk_read_limit,
                                                         pass_read_limit,
                                                         make_datasources(options.get_source()),
                                                         options,
                                                         stream,
                                                         mr)}
{
}

chunked_orc_reader::chunked_orc_reader(std::size_t chunk_read_limit,
                                       orc_reader_options const& options,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
  : chunked_orc_reader(chunk_read_limit, 0UL, options, stream, mr)
{
}

chunked_orc_reader::chunked_orc_reader() = default;

// This destructor destroys the internal reader instance.
// Since the declaration of the internal `reader` object does not exist in the header, this
// destructor needs to be defined in a separate source file which can access to that object's
// declaration.
chunked_orc_reader::~chunked_orc_reader() = default;

bool chunked_orc_reader::has_next() const
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(reader != nullptr, "Reader has not been constructed properly.");
  return reader->has_next();
}

table_with_metadata chunked_orc_reader::read_chunk() const
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(reader != nullptr, "Reader has not been constructed properly.");
  return reader->read_chunk();
}

orc_chunked_writer::orc_chunked_writer() = default;

orc_chunked_writer::~orc_chunked_writer() = default;

/**
 * @copydoc cudf::io::orc_chunked_writer::orc_chunked_writer
 */
orc_chunked_writer::orc_chunked_writer(chunked_orc_writer_options const& options,
                                       rmm::cuda_stream_view stream)
{
  namespace io_detail = cudf::io::detail;

  auto sinks = make_datasinks(options.get_sink());
  CUDF_EXPECTS(sinks.size() == 1, "Multiple sinks not supported for ORC writing");

  writer = std::make_unique<orc::detail::writer>(
    std::move(sinks[0]), options, io_detail::single_write_mode::NO, stream);
}

/**
 * @copydoc cudf::io::orc_chunked_writer::write
 */
orc_chunked_writer& orc_chunked_writer::write(table_view const& table)
{
  CUDF_FUNC_RANGE();

  writer->write(table);

  return *this;
}

/**
 * @copydoc cudf::io::orc_chunked_writer::close
 */
void orc_chunked_writer::close()
{
  CUDF_FUNC_RANGE();

  writer->close();
}

using namespace cudf::io::parquet::detail;
namespace detail_parquet = cudf::io::parquet::detail;

bool is_supported_read_parquet(compression_type compression)
{
  if (compression == compression_type::AUTO or compression == compression_type::NONE) {
    return true;
  }

  return ((compression == compression_type::BROTLI or compression == compression_type::GZIP or
           compression == compression_type::LZ4 or compression == compression_type::SNAPPY or
           compression == compression_type::ZSTD) and
          detail::is_decompression_supported(compression));
}

bool is_supported_write_parquet(compression_type compression)
{
  if (compression == compression_type::AUTO or compression == compression_type::NONE) {
    return true;
  }

  return ((compression == compression_type::LZ4 or compression == compression_type::SNAPPY or
           compression == compression_type::ZSTD) and
          detail::is_compression_supported(compression));
}

table_with_metadata read_parquet(parquet_reader_options const& options,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto datasources = make_datasources(options.get_source());
  auto reader      = std::make_unique<detail_parquet::reader>(
    std::move(datasources), std::vector<parquet::FileMetaData>{}, options, stream, mr);

  return reader->read();
}

table_with_metadata read_parquet(std::vector<std::unique_ptr<cudf::io::datasource>>&& datasources,
                                 std::vector<parquet::FileMetaData>&& parquet_metadatas,
                                 parquet_reader_options const& options,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto reader = std::make_unique<detail_parquet::reader>(
    std::move(datasources), std::move(parquet_metadatas), options, stream, mr);

  return reader->read();
}

parquet_metadata read_parquet_metadata(source_info const& src_info)
{
  CUDF_FUNC_RANGE();

  auto datasources = make_datasources(src_info);
  return detail_parquet::read_parquet_metadata(datasources);
}

std::vector<parquet::FileMetaData> read_parquet_footers(
  std::span<std::unique_ptr<cudf::io::datasource> const> sources)
{
  CUDF_FUNC_RANGE();
  return detail_parquet::read_parquet_footers(sources);
}

/**
 * @copydoc cudf::io::merge_row_group_metadata
 */
std::unique_ptr<std::vector<uint8_t>> merge_row_group_metadata(
  std::vector<std::unique_ptr<std::vector<uint8_t>>> const& metadata_list)
{
  CUDF_FUNC_RANGE();
  return detail_parquet::writer::merge_row_group_metadata(metadata_list);
}

table_input_metadata::table_input_metadata(table_view const& table)
{
  // Create a metadata hierarchy using `table`
  std::function<column_in_metadata(column_view const&)> get_children = [&](column_view const& col) {
    auto col_meta = column_in_metadata{};
    std::transform(
      col.child_begin(), col.child_end(), std::back_inserter(col_meta.children), get_children);
    return col_meta;
  };

  std::transform(
    table.begin(), table.end(), std::back_inserter(this->column_metadata), get_children);
}

table_input_metadata::table_input_metadata(table_metadata const& metadata)
{
  auto const& names = metadata.schema_info;

  // Create a metadata hierarchy with naming and nullability using `table_metadata`
  std::function<column_in_metadata(column_name_info const&)> process_node =
    [&](column_name_info const& name) {
      auto col_meta = column_in_metadata{name.name};
      if (name.is_nullable.has_value()) { col_meta.set_nullability(name.is_nullable.value()); }
      if (name.is_binary.value_or(false)) { col_meta.set_output_as_binary(true); }
      if (name.type_length.has_value()) { col_meta.set_type_length(name.type_length.value()); }
      std::transform(name.children.begin(),
                     name.children.end(),
                     std::back_inserter(col_meta.children),
                     process_node);
      return col_meta;
    };

  std::transform(
    names.begin(), names.end(), std::back_inserter(this->column_metadata), process_node);
}

/**
 * @copydoc cudf::io::write_parquet
 */
std::unique_ptr<std::vector<uint8_t>> write_parquet(parquet_writer_options const& options,
                                                    rmm::cuda_stream_view stream)
{
  namespace io_detail = cudf::io::detail;

  CUDF_FUNC_RANGE();

  auto sinks  = make_datasinks(options.get_sink());
  auto writer = std::make_unique<detail_parquet::writer>(
    std::move(sinks), options, io_detail::single_write_mode::YES, stream);

  writer->write(options.get_table(), options.get_partitions());

  return writer->close(options.get_column_chunks_file_paths());
}

chunked_parquet_reader::chunked_parquet_reader() = default;

/**
 * @copydoc cudf::io::chunked_parquet_reader::chunked_parquet_reader
 */
chunked_parquet_reader::chunked_parquet_reader(std::size_t chunk_read_limit,
                                               parquet_reader_options const& options,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
  : reader{std::make_unique<detail_parquet::chunked_reader>(
      chunk_read_limit,
      detail_parquet::derive_pass_read_limit(chunk_read_limit),
      make_datasources(options.get_source()),
      std::vector<parquet::FileMetaData>{},
      options,
      stream,
      mr)}
{
}

/**
 * @copydoc cudf::io::chunked_parquet_reader::chunked_parquet_reader
 */
chunked_parquet_reader::chunked_parquet_reader(
  std::size_t chunk_read_limit,
  std::vector<std::unique_ptr<cudf::io::datasource>>&& datasources,
  std::vector<parquet::FileMetaData>&& parquet_metadatas,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
  : reader{std::make_unique<detail_parquet::chunked_reader>(
      chunk_read_limit,
      detail_parquet::derive_pass_read_limit(chunk_read_limit),
      std::move(datasources),
      std::move(parquet_metadatas),
      options,
      stream,
      mr)}
{
}

/**
 * @copydoc cudf::io::chunked_parquet_reader::chunked_parquet_reader
 */
chunked_parquet_reader::chunked_parquet_reader(std::size_t chunk_read_limit,
                                               std::size_t pass_read_limit,
                                               parquet_reader_options const& options,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
  : reader{std::make_unique<detail_parquet::chunked_reader>(chunk_read_limit,
                                                            pass_read_limit,
                                                            make_datasources(options.get_source()),
                                                            std::vector<parquet::FileMetaData>{},
                                                            options,
                                                            stream,
                                                            mr)}
{
}

/**
 * @copydoc cudf::io::chunked_parquet_reader::chunked_parquet_reader
 */
chunked_parquet_reader::chunked_parquet_reader(
  std::size_t chunk_read_limit,
  std::size_t pass_read_limit,
  std::vector<std::unique_ptr<cudf::io::datasource>>&& datasources,
  std::vector<parquet::FileMetaData>&& parquet_metadatas,
  parquet_reader_options const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
  : reader{std::make_unique<detail_parquet::chunked_reader>(chunk_read_limit,
                                                            pass_read_limit,
                                                            std::move(datasources),
                                                            std::move(parquet_metadatas),
                                                            options,
                                                            stream,
                                                            mr)}
{
}

/**
 * @copydoc cudf::io::chunked_parquet_reader::~chunked_parquet_reader
 */
chunked_parquet_reader::~chunked_parquet_reader() = default;

/**
 * @copydoc cudf::io::chunked_parquet_reader::has_next
 */
bool chunked_parquet_reader::has_next() const
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(reader != nullptr, "Reader has not been constructed properly.");
  return reader->has_next();
}

/**
 * @copydoc cudf::io::chunked_parquet_reader::read_chunk
 */
table_with_metadata chunked_parquet_reader::read_chunk() const
{
  CUDF_FUNC_RANGE();
  CUDF_EXPECTS(reader != nullptr, "Reader has not been constructed properly.");
  return reader->read_chunk();
}

chunked_parquet_writer::chunked_parquet_writer() = default;

/**
 * @copydoc cudf::io::chunked_parquet_writer::chunked_parquet_writer
 */
chunked_parquet_writer::chunked_parquet_writer(chunked_parquet_writer_options const& options,
                                               rmm::cuda_stream_view stream)
{
  namespace io_detail = cudf::io::detail;

  auto sinks = make_datasinks(options.get_sink());

  writer = std::make_unique<detail_parquet::writer>(
    std::move(sinks), options, io_detail::single_write_mode::NO, stream);
}

chunked_parquet_writer::~chunked_parquet_writer() = default;

/**
 * @copydoc cudf::io::chunked_parquet_writer::write
 */
chunked_parquet_writer& chunked_parquet_writer::write(table_view const& table,
                                                      std::vector<partition_info> const& partitions)
{
  CUDF_FUNC_RANGE();

  writer->write(table, partitions);

  return *this;
}

/**
 * @copydoc cudf::io::chunked_parquet_writer::close
 */
std::unique_ptr<std::vector<uint8_t>> chunked_parquet_writer::close(
  std::vector<std::string> const& column_chunks_file_path)
{
  CUDF_FUNC_RANGE();
  return writer->close(column_chunks_file_path);
}

void parquet_reader_options::set_row_groups(std::vector<std::vector<size_type>> row_groups)
{
  if ((!row_groups.empty()) and
      (_skip_rows != 0 or _num_rows.has_value() or _skip_bytes != 0 or _num_bytes.has_value())) {
    CUDF_FAIL(
      "row_groups can't be set along with skip_rows and num_rows or skip_bytes and num_bytes");
  }

  _row_groups = std::move(row_groups);
}

void parquet_reader_options::set_skip_rows(int64_t val)
{
  CUDF_EXPECTS(val >= 0, "skip_rows cannot be negative");
  CUDF_EXPECTS(_row_groups.empty(), "skip_rows can't be set along with a non-empty row_groups");
  CUDF_EXPECTS(_skip_bytes == 0 and not _num_bytes.has_value(),
               "skip_rows can't be set along with skip_bytes or num_bytes");

  _skip_rows = val;
}

void parquet_reader_options::set_num_rows(int64_t val)
{
  CUDF_EXPECTS(val >= 0, "num_rows cannot be negative");
  CUDF_EXPECTS(_row_groups.empty(), "num_rows can't be set along with a non-empty row_groups");
  CUDF_EXPECTS(_skip_bytes == 0 and not _num_bytes.has_value(),
               "num_rows can't be set along with skip_bytes or num_bytes");

  _num_rows = val;
}

void parquet_reader_options::set_skip_bytes(size_t val)
{
  // Hybrid scan reader does not contain a source so relaxing this check to zero or one source
  CUDF_EXPECTS(val == 0 or _source.num_sources() == 1 or _source.num_sources() == 0,
               "skip_bytes can only be set for single parquet source case");
  CUDF_EXPECTS(val == 0 or (not _num_rows.has_value() and _skip_rows == 0),
               "skip_bytes cannot be set along with skip_rows and num_rows");
  CUDF_EXPECTS(val == 0 or _row_groups.empty(),
               "skip_bytes can't be set along with a non-empty row_groups");

  _skip_bytes = val;
}

void parquet_reader_options::set_num_bytes(size_t val)
{
  // Hybrid scan reader does not contain a source so relaxing this check to zero or one source
  CUDF_EXPECTS(val == std::numeric_limits<size_t>::max() or _source.num_sources() == 1 or
                 _source.num_sources() == 0,
               "num_bytes can only be set for single parquet source case");
  CUDF_EXPECTS(not _num_rows.has_value() and _skip_rows == 0,
               "num_bytes cannot be set along with skip_rows and num_rows");
  CUDF_EXPECTS(_row_groups.empty(), "num_bytes can't be set along with a non-empty row_groups");

  _num_bytes = val;
}

void parquet_writer_options_base::set_metadata(table_input_metadata metadata)
{
  _metadata = std::move(metadata);
}

void parquet_writer_options_base::set_key_value_metadata(
  std::vector<std::map<std::string, std::string>> metadata)
{
  CUDF_EXPECTS(metadata.size() == get_sink().num_sinks(),
               "Mismatch between number of sinks and number of metadata maps");
  _user_data = std::move(metadata);
}

void parquet_writer_options_base::set_stats_level(statistics_freq sf) { _stats_level = sf; }

void parquet_writer_options_base::set_compression(compression_type compression)
{
  _compression = compression;
  if (compression == compression_type::AUTO) { _compression = compression_type::SNAPPY; }
}

void parquet_writer_options_base::enable_int96_timestamps(bool req)
{
  CUDF_EXPECTS(not req or not is_enabled_write_arrow_schema(),
               "INT96 timestamps and arrow schema cannot be simultaneously "
               "enabled as INT96 timestamps are deprecated in Arrow.");
  _write_timestamps_as_int96 = req;
}

void parquet_writer_options_base::enable_utc_timestamps(bool val)
{
  _write_timestamps_as_UTC = val;
}

void parquet_writer_options_base::enable_write_arrow_schema(bool val)
{
  CUDF_EXPECTS(not val or not is_enabled_int96_timestamps(),
               "arrow schema and INT96 timestamps cannot be simultaneously "
               "enabled as INT96 timestamps are deprecated in Arrow.");
  _write_arrow_schema = val;
}

void parquet_writer_options_base::set_row_group_size_bytes(size_t size_bytes)
{
  CUDF_EXPECTS(
    size_bytes >= 1024,
    "The maximum row group size cannot be smaller than the minimum page size, which is 1KB.");
  _row_group_size_bytes = size_bytes;
}

void parquet_writer_options_base::set_row_group_size_rows(size_type size_rows)
{
  CUDF_EXPECTS(size_rows > 0, "The maximum row group row count must be a positive integer.");
  _row_group_size_rows = size_rows;
}

void parquet_writer_options_base::set_max_page_size_bytes(size_t size_bytes)
{
  CUDF_EXPECTS(size_bytes >= 1024, "The maximum page size cannot be smaller than 1KB.");
  CUDF_EXPECTS(size_bytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
               "The maximum page size cannot exceed 2GB.");
  _max_page_size_bytes = size_bytes;
}

void parquet_writer_options_base::set_max_page_size_rows(size_type size_rows)
{
  CUDF_EXPECTS(size_rows > 0, "The maximum page row count must be a positive integer.");
  _max_page_size_rows = size_rows;
}

void parquet_writer_options_base::set_column_index_truncate_length(int32_t size_bytes)
{
  CUDF_EXPECTS(size_bytes >= 0, "Column index truncate length cannot be negative.");
  _column_index_truncate_length = size_bytes;
}

void parquet_writer_options_base::set_dictionary_policy(dictionary_policy policy)
{
  _dictionary_policy = policy;
}

void parquet_writer_options_base::set_max_dictionary_size(size_t size_bytes)
{
  CUDF_EXPECTS(size_bytes <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
               "The maximum dictionary size cannot exceed 2GB.");
  _max_dictionary_size = size_bytes;
}

void parquet_writer_options_base::set_max_page_fragment_size(size_type size_rows)
{
  CUDF_EXPECTS(size_rows > 0, "Page fragment size must be a positive integer.");
  _max_page_fragment_size = size_rows;
}

void parquet_writer_options_base::set_compression_statistics(
  std::shared_ptr<writer_compression_statistics> comp_stats)
{
  _compression_stats = std::move(comp_stats);
}

void parquet_writer_options_base::enable_write_v2_headers(bool val) { _v2_page_headers = val; }

void parquet_writer_options_base::enable_page_level_compression(bool val)
{
  _page_level_compression = val;
}

void parquet_writer_options_base::set_sorting_columns(std::vector<sorting_column> sorting_columns)
{
  _sorting_columns = std::move(sorting_columns);
}

parquet_writer_options::parquet_writer_options(sink_info const& sink, table_view table)
  : parquet_writer_options_base(sink), _table(std::move(table))
{
}

void parquet_writer_options::set_partitions(std::vector<partition_info> partitions)
{
  CUDF_EXPECTS(partitions.size() == get_sink().num_sinks(),
               "Mismatch between number of sinks and number of partitions");
  _partitions = std::move(partitions);
}

void parquet_writer_options::set_column_chunks_file_paths(std::vector<std::string> file_paths)
{
  CUDF_EXPECTS(file_paths.size() == get_sink().num_sinks(),
               "Mismatch between number of sinks and number of chunk paths to set");
  _column_chunks_file_paths = std::move(file_paths);
}

template <class BuilderT, class OptionsT>
parquet_writer_options_builder_base<BuilderT, OptionsT>::parquet_writer_options_builder_base(
  OptionsT options)
  : _options(std::move(options))
{
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::metadata(
  table_input_metadata metadata)
{
  _options.set_metadata(std::move(metadata));
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::key_value_metadata(
  std::vector<std::map<std::string, std::string>> metadata)
{
  _options.set_key_value_metadata(std::move(metadata));
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::stats_level(statistics_freq sf)
{
  _options.set_stats_level(sf);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::compression(
  compression_type compression)
{
  _options.set_compression(compression);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::row_group_size_bytes(size_t val)
{
  _options.set_row_group_size_bytes(val);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::row_group_size_rows(
  size_type val)
{
  _options.set_row_group_size_rows(val);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::max_page_size_bytes(size_t val)
{
  _options.set_max_page_size_bytes(val);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::max_page_size_rows(size_type val)
{
  _options.set_max_page_size_rows(val);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::column_index_truncate_length(
  int32_t val)
{
  _options.set_column_index_truncate_length(val);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::dictionary_policy(
  enum dictionary_policy val)
{
  _options.set_dictionary_policy(val);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::max_dictionary_size(size_t val)
{
  _options.set_max_dictionary_size(val);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::max_page_fragment_size(
  size_type val)
{
  _options.set_max_page_fragment_size(val);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::compression_statistics(
  std::shared_ptr<writer_compression_statistics> const& comp_stats)
{
  _options.set_compression_statistics(comp_stats);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::int96_timestamps(bool enabled)
{
  _options.enable_int96_timestamps(enabled);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::utc_timestamps(bool enabled)
{
  _options.enable_utc_timestamps(enabled);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::write_arrow_schema(bool enabled)
{
  _options.enable_write_arrow_schema(enabled);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::write_v2_headers(bool enabled)
{
  _options.enable_write_v2_headers(enabled);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::page_level_compression(
  bool enabled)
{
  _options.enable_page_level_compression(enabled);
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
BuilderT& parquet_writer_options_builder_base<BuilderT, OptionsT>::sorting_columns(
  std::vector<sorting_column> sorting_columns)
{
  _options.set_sorting_columns(std::move(sorting_columns));
  return static_cast<BuilderT&>(*this);
}

template <class BuilderT, class OptionsT>
parquet_writer_options_builder_base<BuilderT, OptionsT>::operator OptionsT&&()
{
  return std::move(_options);
}

template <class BuilderT, class OptionsT>
OptionsT&& parquet_writer_options_builder_base<BuilderT, OptionsT>::build()
{
  return std::move(_options);
}

template class parquet_writer_options_builder_base<parquet_writer_options_builder,
                                                   parquet_writer_options>;
template class parquet_writer_options_builder_base<chunked_parquet_writer_options_builder,
                                                   chunked_parquet_writer_options>;

parquet_writer_options_builder::parquet_writer_options_builder(sink_info const& sink,
                                                               table_view const& table)
  : parquet_writer_options_builder_base(parquet_writer_options{sink, table})
{
}

parquet_writer_options_builder& parquet_writer_options_builder::partitions(
  std::vector<partition_info> partitions)
{
  get_options().set_partitions(std::move(partitions));
  return *this;
}

parquet_writer_options_builder& parquet_writer_options_builder::column_chunks_file_paths(
  std::vector<std::string> file_paths)
{
  get_options().set_column_chunks_file_paths(std::move(file_paths));
  return *this;
}

chunked_parquet_writer_options::chunked_parquet_writer_options(sink_info sink)
  : parquet_writer_options_base(std::move(sink))
{
}

chunked_parquet_writer_options_builder::chunked_parquet_writer_options_builder(
  sink_info const& sink)
  : parquet_writer_options_builder_base(chunked_parquet_writer_options{sink})
{
}

namespace experimental {

// Forward declarations for detail functions
namespace detail {
void write_cudftable(data_sink* sink, table_view const& input, rmm::cuda_stream_view stream);
packed_table read_cudftable(datasource* source,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr);
}  // namespace detail

/**
 * @copydoc cudf::io::experimental::write_cudftable
 */
void write_cudftable(cudftable_writer_options const& options, rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();

  auto sinks = make_datasinks(options.get_sink());
  CUDF_EXPECTS(sinks.size() == 1, "CudfTable format only supports single sink");

  detail::write_cudftable(sinks[0].get(), options.get_table(), stream);
}

/**
 * @copydoc cudf::io::experimental::read_cudftable
 */
packed_table read_cudftable(cudftable_reader_options const& options,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  auto datasources = make_datasources(options.get_source());
  CUDF_EXPECTS(datasources.size() == 1, "CudfTable format only supports single source");

  return detail::read_cudftable(datasources[0].get(), stream, mr);
}

}  // namespace experimental
}  // namespace cudf::io

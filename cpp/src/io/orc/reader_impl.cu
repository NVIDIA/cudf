/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/orc/reader_impl.hpp"
#include "io/orc/reader_impl_chunking.hpp"
#include "io/orc/reader_impl_helpers.hpp"

#include <cudf/detail/copy.hpp>

#include <algorithm>

namespace cudf::io::orc::detail {

// This is just the proxy to call all other data preprocessing functions.
void reader_impl::prepare_data(read_mode mode)
{
  // There are no columns in the table.
  if (_selected_columns.num_levels() == 0) { return; }

  // This will be no-op if it was called before.
  preprocess_file(mode);

  if (!_chunk_read_data.more_table_chunks_to_output()) {
    if (!_chunk_read_data.more_stripes_to_decode() && _chunk_read_data.more_stripes_to_load()) {
      // Only load stripe data if:
      //  - There is more stripe to load, and
      //  - All loaded stripes were decoded, and
      //  - All the decoded results were output.
      load_next_stripe_data(mode);
    }
    if (_chunk_read_data.more_stripes_to_decode()) {
      // Only decompress/decode the loaded stripes if:
      //  - There are loaded stripes that were not decoded yet, and
      //  - All the decoded results were output.
      decompress_and_decode_stripes(mode);
    }
  }
}

table_with_metadata reader_impl::make_output_chunk()
{
  // There are no columns in the table.
  if (_selected_columns.num_levels() == 0) { return {std::make_unique<table>(), table_metadata{}}; }

  // If no rows or stripes to read, return empty columns.
  if (!_chunk_read_data.more_table_chunks_to_output()) {
    std::vector<std::unique_ptr<column>> out_columns;
    auto out_metadata = get_meta_with_user_data();
    std::transform(_selected_columns.levels[0].begin(),
                   _selected_columns.levels[0].end(),
                   std::back_inserter(out_columns),
                   [&](auto const& col_meta) {
                     out_metadata.schema_info.emplace_back("");
                     return create_empty_column(col_meta.id,
                                                _metadata,
                                                _options.decimal128_columns,
                                                _options.use_np_dtypes,
                                                _options.timestamp_type,
                                                out_metadata.schema_info.back(),
                                                _stream);
                   });
    return {std::make_unique<table>(std::move(out_columns)), std::move(out_metadata)};
  }

  auto const make_output_table = [&] {
    if (_chunk_read_data.output_table_ranges.size() == 1) {
      // Must change the index of the current output range such that calling `has_next()` after
      // this will return the correct answer (`false`, since there is only one range).
      _chunk_read_data.curr_output_table_range++;

      // Just hand over the decoded table without slicing.
      return std::move(_chunk_read_data.decoded_table);
    }

    // The range of rows in the decoded table to output.
    auto const out_range =
      _chunk_read_data.output_table_ranges[_chunk_read_data.curr_output_table_range++];
    auto const out_tview = cudf::detail::slice(
      _chunk_read_data.decoded_table->view(),
      {static_cast<size_type>(out_range.begin), static_cast<size_type>(out_range.end)},
      _stream)[0];
    auto output = std::make_unique<table>(out_tview, _stream, _mr);

    // If this is the last slice, we also delete the decoded table to free up memory.
    if (!_chunk_read_data.more_table_chunks_to_output()) {
      _chunk_read_data.decoded_table.reset(nullptr);
    }

    return output;
  };

  return {make_output_table(), table_metadata{_out_metadata} /*copy cached metadata*/};
}

table_metadata reader_impl::get_meta_with_user_data()
{
  if (_meta_with_user_data) { return table_metadata{*_meta_with_user_data}; }

  // Copy user data to the output metadata.
  table_metadata out_metadata;
  out_metadata.per_file_user_data.reserve(_metadata.per_file_metadata.size());
  std::transform(_metadata.per_file_metadata.cbegin(),
                 _metadata.per_file_metadata.cend(),
                 std::back_inserter(out_metadata.per_file_user_data),
                 [](auto const& meta) {
                   std::unordered_map<std::string, std::string> kv_map;
                   std::transform(meta.ff.metadata.cbegin(),
                                  meta.ff.metadata.cend(),
                                  std::inserter(kv_map, kv_map.end()),
                                  [](auto const& kv) { return std::pair{kv.name, kv.value}; });
                   return kv_map;
                 });
  out_metadata.user_data = {out_metadata.per_file_user_data[0].begin(),
                            out_metadata.per_file_user_data[0].end()};

  // Save the output table metadata into `_meta_with_user_data` for reuse next time.
  _meta_with_user_data = std::make_unique<table_metadata>(out_metadata);

  return out_metadata;
}

reader_impl::reader_impl(std::vector<std::unique_ptr<datasource>>&& sources,
                         orc_reader_options const& options,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
  : reader_impl::reader_impl(0UL, 0UL, std::move(sources), options, stream, mr)
{
}

reader_impl::reader_impl(std::size_t chunk_read_limit,
                         std::size_t pass_read_limit,
                         std::vector<std::unique_ptr<datasource>>&& sources,
                         orc_reader_options const& options,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
  : reader_impl::reader_impl(chunk_read_limit,
                             pass_read_limit,
                             DEFAULT_OUTPUT_ROW_GRANULARITY,
                             std::move(sources),
                             options,
                             stream,
                             mr)
{
}

reader_impl::reader_impl(std::size_t chunk_read_limit,
                         std::size_t pass_read_limit,
                         size_type output_row_granularity,
                         std::vector<std::unique_ptr<datasource>>&& sources,
                         orc_reader_options const& options,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
  : _stream(stream),
    _mr(mr),
    _options{options.get_timestamp_type(),
             options.is_enabled_use_index(),
             options.is_enabled_use_np_dtypes(),
             options.get_decimal128_columns(),
             options.get_ignore_timezone_in_stripe_footer(),
             options.get_skip_rows(),
             options.get_num_rows(),
             options.get_stripes()},
    _col_meta{std::make_unique<reader_column_meta>()},
    _sources(std::move(sources)),
    _metadata{_sources, stream},
    _selected_columns{_metadata.select_columns(options.get_columns())},
    _chunk_read_data{chunk_read_limit, pass_read_limit, output_row_granularity}
{
  // Selected columns at different levels of nesting are stored in different elements
  // of `selected_columns`; thus, size == 1 means no nested columns.
  CUDF_EXPECTS(_options.skip_rows == 0 or _selected_columns.num_levels() == 1,
               "skip_rows is not supported by nested column");
}

table_with_metadata reader_impl::read()
{
  prepare_data(read_mode::READ_ALL);
  return make_output_chunk();
}

bool reader_impl::has_next()
{
  prepare_data(read_mode::CHUNKED_READ);
  return _chunk_read_data.has_next();
}

table_with_metadata reader_impl::read_chunk()
{
  prepare_data(read_mode::CHUNKED_READ);
  return make_output_chunk();
}

chunked_reader::chunked_reader(std::size_t chunk_read_limit,
                               std::size_t pass_read_limit,
                               std::vector<std::unique_ptr<datasource>>&& sources,
                               orc_reader_options const& options,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
  : _impl{std::make_unique<reader_impl>(
      chunk_read_limit, pass_read_limit, std::move(sources), options, stream, mr)}
{
}

chunked_reader::chunked_reader(std::size_t chunk_read_limit,
                               std::size_t pass_read_limit,
                               size_type output_row_granularity,
                               std::vector<std::unique_ptr<datasource>>&& sources,
                               orc_reader_options const& options,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
  : _impl{std::make_unique<reader_impl>(chunk_read_limit,
                                        pass_read_limit,
                                        output_row_granularity,
                                        std::move(sources),
                                        options,
                                        stream,
                                        mr)}
{
}

chunked_reader::~chunked_reader() = default;

bool chunked_reader::has_next() const { return _impl->has_next(); }

table_with_metadata chunked_reader::read_chunk() const { return _impl->read_chunk(); }

reader::reader(std::vector<std::unique_ptr<cudf::io::datasource>>&& sources,
               orc_reader_options const& options,
               cuda::stream_ref stream,
               rmm::device_async_resource_ref mr)
  : _impl{std::make_unique<reader_impl>(std::move(sources), options, stream, mr)}
{
}

reader::~reader() = default;

table_with_metadata reader::read() { return _impl->read(); }

}  // namespace cudf::io::orc::detail

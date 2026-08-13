/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "io/comp/common.hpp"
#include "io/orc/reader_impl.hpp"
#include "io/orc/reader_impl_chunking.hpp"
#include "io/orc/reader_impl_helpers.hpp"
#include "io/utilities/hostdevice_span.hpp"

#include <cudf/detail/timezone.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/logger.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/iterator>
#include <thrust/binary_search.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>

#include <algorithm>
#include <numeric>
#include <tuple>

namespace cudf::io::orc::detail {

std::size_t gather_stream_info_and_column_desc(
  std::size_t stripe_id,
  std::size_t level,
  StripeInformation const* stripeinfo,
  StripeFooter const* stripefooter,
  host_span<int const> orc2gdf,
  host_span<SchemaType const> types,
  bool use_index,
  bool apply_struct_map,
  int64_t* num_dictionary_entries,
  std::size_t* local_stream_order,
  std::vector<orc_stream_info>* stream_info,
  cudf::detail::hostdevice_2dvector<column_desc>* chunks)
{
  CUDF_EXPECTS((stream_info == nullptr) ^ (chunks == nullptr),
               "Either stream_info or chunks must be provided, but not both.");

  std::size_t src_offset = 0;
  std::size_t dst_offset = 0;

  auto const get_stream_index_type = [](StreamKind kind) {
    switch (kind) {
      case DATA: return CI_DATA;
      case LENGTH:
      case SECONDARY: return CI_DATA2;
      case DICTIONARY_DATA: return CI_DICTIONARY;
      case PRESENT: return CI_PRESENT;
      case ROW_INDEX: return CI_INDEX;
      default:
        // Skip this stream as it's not strictly required
        return CI_NUM_STREAMS;
    }
  };

  for (auto const& stream : stripefooter->streams) {
    if (!stream.column_id || *stream.column_id >= orc2gdf.size()) {
      // Ignore reading this stream from source.
      CUDF_LOG_WARN("Unexpected stream in the input ORC source. The stream will be ignored.");
      src_offset += stream.length;
      continue;
    }

    auto const column_id = *stream.column_id;
    auto col             = orc2gdf[column_id];

    if (col == -1 and apply_struct_map) {
      // A struct-type column has no data itself, but rather child columns
      // for each of its fields. There is only a PRESENT stream, which
      // needs to be included for the reader.
      auto const schema_type = types[column_id];
      if (!schema_type.subtypes.empty() && schema_type.kind == STRUCT && stream.kind == PRESENT) {
        for (auto const& idx : schema_type.subtypes) {
          auto const child_idx = (idx < orc2gdf.size()) ? orc2gdf[idx] : -1;
          if (child_idx >= 0) {
            col = child_idx;
            if (chunks) {
              auto& chunk                = (*chunks)[stripe_id][col];
              chunk.strm_id[CI_PRESENT]  = *local_stream_order;
              chunk.strm_len[CI_PRESENT] = stream.length;
            }
          }
        }
      }
    } else if (col != -1) {
      if (chunks) {
        if (src_offset >= stripeinfo->indexLength || use_index) {
          auto const index_type = get_stream_index_type(stream.kind);
          if (index_type < CI_NUM_STREAMS) {
            auto& chunk                = (*chunks)[stripe_id][col];
            chunk.strm_id[index_type]  = *local_stream_order;
            chunk.strm_len[index_type] = stream.length;
            // NOTE: skip_count field is temporarily used to track the presence of index streams
            chunk.skip_count |= 1 << index_type;

            if (index_type == CI_DICTIONARY) {
              chunk.dictionary_start = *num_dictionary_entries;
              chunk.dict_len         = stripefooter->columns[column_id].dictionarySize;
              *num_dictionary_entries +=
                static_cast<int64_t>(stripefooter->columns[column_id].dictionarySize);
            }
          }
        }

        (*local_stream_order)++;
      } else {  // chunks == nullptr
        stream_info->emplace_back(
          orc_stream_info{stripeinfo->offset + src_offset,
                          dst_offset,
                          stream.length,
                          stream_source_info{stripe_id, level, column_id, stream.kind}});
      }

      dst_offset += stream.length;
    }
    src_offset += stream.length;
  }

  return dst_offset;
}

template <typename T>
std::vector<range> find_splits(host_span<T const> cumulative_sizes,
                               std::size_t total_count,
                               std::size_t size_limit)
{
  CUDF_EXPECTS(size_limit > 0, "Invalid size limit", std::invalid_argument);

  std::vector<range> splits;
  std::size_t cur_count{0};
  int64_t cur_pos{0};
  std::size_t cur_cumulative_size{0};

  [[maybe_unused]] std::size_t cur_cumulative_rows{0};

  auto const start = thrust::make_transform_iterator(
    cumulative_sizes.begin(),
    [&](auto const& size) { return size.size_bytes - cur_cumulative_size; });
  auto const end = start + cumulative_sizes.size();

  while (cur_count < total_count) {
    int64_t split_pos = static_cast<int64_t>(cuda::std::distance(
      start, thrust::lower_bound(thrust::seq, start + cur_pos, end, size_limit)));

    // If we're past the end, or if the returned range has size exceeds the given size limit,
    // move back one position.
    if (split_pos >= static_cast<int64_t>(cumulative_sizes.size()) ||
        (cumulative_sizes[split_pos].size_bytes > cur_cumulative_size + size_limit)) {
      split_pos--;
    }

    if constexpr (std::is_same_v<T, cumulative_size_and_row>) {
      // Similarly, while the returned range has total number of rows exceeds column size limit,
      // move back one position.
      while (split_pos > 0 && cumulative_sizes[split_pos].num_rows >
                                cur_cumulative_rows +
                                  static_cast<std::size_t>(std::numeric_limits<size_type>::max())) {
        split_pos--;
      }
    }

    // In case we have moved back too much in the steps above, far beyond the last split point, that
    // means we could not find any range that has size fits within the given size limit.
    // In such situations, we need to move forward until we move pass the last output range.
    while (split_pos < (static_cast<int64_t>(cumulative_sizes.size()) - 1) &&
           (split_pos < 0 || cumulative_sizes[split_pos].count <= cur_count)) {
      split_pos++;
    }

    auto const start_count = cur_count;
    cur_count              = cumulative_sizes[split_pos].count;
    splits.emplace_back(range{start_count, cur_count});
    cur_pos             = split_pos;
    cur_cumulative_size = cumulative_sizes[split_pos].size_bytes;

    if constexpr (std::is_same_v<T, cumulative_size_and_row>) {
      cur_cumulative_rows = cumulative_sizes[split_pos].num_rows;
    }
  }

  // If the last range has size smaller than `merge_threshold` the size of the second last one,
  // merge it with the second last one.
  // This is to prevent having the last range too small.
  if (splits.size() > 1) {
    double constexpr merge_threshold = 0.15;
    if (auto const last = splits.back(), second_last = splits[splits.size() - 2];
        last.size() <= static_cast<std::size_t>(merge_threshold * second_last.size())) {
      splits.pop_back();
      splits.back().end = last.end;
    }
  }

  return splits;
}

// Since `find_splits` is a template function, we need to explicitly instantiate it so it can be
// used outside of this TU.
template std::vector<range> find_splits<cumulative_size>(host_span<cumulative_size const> sizes,
                                                         std::size_t total_count,
                                                         std::size_t size_limit);
template std::vector<range> find_splits<cumulative_size_and_row>(
  host_span<cumulative_size_and_row const> sizes, std::size_t total_count, std::size_t size_limit);

// In this step, the metadata of all stripes in the data sources is parsed, and information about
// data streams of the selected columns in all stripes are generated. If the reader has a data
// read limit, sizes of these streams are used to split the list of all stripes into multiple
// subsets, each of which will be loaded into memory in the `load_next_stripe_data()` step. These
// subsets are computed such that memory usage will be kept to be around a fixed size limit.
void reader_impl::preprocess_file(read_mode mode)
{
  if (_file_itm_data.global_preprocessed) { return; }
  _file_itm_data.global_preprocessed = true;

  //
  // Load stripes' metadata:
  //
  std::tie(
    _file_itm_data.rows_to_skip, _file_itm_data.rows_to_read, _file_itm_data.selected_stripes) =
    _metadata.select_stripes(
      _options.selected_stripes, _options.skip_rows, _options.num_read_rows, _stream);
  if (!_file_itm_data.has_data()) { return; }

  CUDF_EXPECTS(
    mode == read_mode::CHUNKED_READ ||
      _file_itm_data.rows_to_read <= static_cast<int64_t>(std::numeric_limits<size_type>::max()),
    "READ_ALL mode does not support reading number of rows more than cudf's column size limit. "
    "For reading large number of rows, please use chunked_reader.",
    std::overflow_error);

  auto const& selected_stripes = _file_itm_data.selected_stripes;
  auto const num_total_stripes = selected_stripes.size();
  auto const num_levels        = _selected_columns.num_levels();

  // Set up table for converting timestamp columns from local to UTC time
  _file_itm_data.tz_table = [&] {
    auto const has_timestamp_column = std::any_of(
      _selected_columns.levels.cbegin(), _selected_columns.levels.cend(), [&](auto const& col_lvl) {
        return std::any_of(col_lvl.cbegin(), col_lvl.cend(), [&](auto const& col_meta) {
          return _metadata.get_col_type(col_meta.id).kind == TypeKind::TIMESTAMP;
        });
      });

    return (has_timestamp_column && !_options.ignore_timezone_in_stripe_footer)
             ? cudf::detail::make_timezone_transition_table(
                 {},
                 selected_stripes[0].stripe_footer->writerTimezone,
                 _stream,
                 cudf::get_current_device_resource_ref())
             : std::make_unique<cudf::table>();
  }();

  //
  // Pre allocate necessary memory for data processed in the other reading steps:
  //
  auto& stripe_data_read_ranges = _file_itm_data.stripe_data_read_ranges;
  stripe_data_read_ranges.resize(num_total_stripes);

  auto& lvl_stripe_data          = _file_itm_data.lvl_stripe_data;
  auto& lvl_stripe_sizes         = _file_itm_data.lvl_stripe_sizes;
  auto& lvl_stream_info          = _file_itm_data.lvl_stream_info;
  auto& lvl_stripe_stream_ranges = _file_itm_data.lvl_stripe_stream_ranges;
  auto& lvl_column_types         = _file_itm_data.lvl_column_types;
  auto& lvl_nested_cols          = _file_itm_data.lvl_nested_cols;

  lvl_stripe_data.resize(num_levels);
  lvl_stripe_sizes.resize(num_levels);
  lvl_stream_info.resize(num_levels);
  lvl_stripe_stream_ranges.resize(num_levels);
  lvl_column_types.resize(num_levels);
  lvl_nested_cols.resize(num_levels);
  _out_buffers.resize(num_levels);

  auto& read_info = _file_itm_data.data_read_info;
  auto& col_meta  = *_col_meta;

  //
  // Collect columns' types:
  //
  for (std::size_t level = 0; level < num_levels; ++level) {
    lvl_stripe_sizes[level].resize(num_total_stripes);
    lvl_stripe_stream_ranges[level].resize(num_total_stripes);

    // Association between each ORC column and its cudf::column
    col_meta.orc_col_map.emplace_back(_metadata.get_num_cols(), -1);

    auto const& columns_level = _selected_columns.levels[level];
    size_type col_id{0};

    for (auto const& col : columns_level) {
      // Map each ORC column to its column
      col_meta.orc_col_map[level][col.id] = col_id++;

      auto const col_type =
        to_cudf_type(_metadata.get_col_type(col.id).kind,
                     _options.use_np_dtypes,
                     _options.timestamp_type.id(),
                     to_cudf_decimal_type(_options.decimal128_columns, _metadata, col.id));
      CUDF_EXPECTS(col_type != type_id::EMPTY, "Unknown type");

      auto& column_types = lvl_column_types[level];
      auto& nested_cols  = lvl_nested_cols[level];

      if (col_type == type_id::DECIMAL32 or col_type == type_id::DECIMAL64 or
          col_type == type_id::DECIMAL128) {
        // sign of the scale is changed since cuDF follows c++ libraries like CNL
        // which uses negative scaling, but liborc and other libraries
        // follow positive scaling.
        auto const scale =
          -static_cast<size_type>(_metadata.get_col_type(col.id).scale.value_or(0));
        column_types.emplace_back(col_type, scale);
      } else {
        column_types.emplace_back(col_type);
      }

      // Only nested columns with selected children require child metadata aggregation.
      if (col.num_children > 0 and (col_type == type_id::LIST or col_type == type_id::STRUCT)) {
        nested_cols.emplace_back(col);
      }
    }

    // Try to reserve some memory, but the final size is unknown,
    // since each column may have more than one stream.
    auto const num_columns = columns_level.size();
    lvl_stream_info[level].reserve(num_total_stripes * num_columns);
    if (read_info.capacity() < num_total_stripes * num_columns) {
      read_info.reserve(num_total_stripes * num_columns);
    }
  }

  //
  // Collect all data streams' information:
  //

  // Load all stripes if we are in READ_ALL mode or there is no read limit.
  auto const load_all_stripes =
    mode == read_mode::READ_ALL || _chunk_read_data.pass_read_limit == 0;

  // Accumulate data size for data streams in each stripe, used for chunking.
  // This will be used only for CHUNKED_READ mode when there is a read limit.
  // Otherwise, we do not need this since we just load all stripes.
  cudf::detail::hostdevice_vector<cumulative_size> total_stripe_sizes(
    load_all_stripes ? std::size_t{0} : num_total_stripes, _stream);

  for (std::size_t stripe_global_idx = 0; stripe_global_idx < num_total_stripes;
       ++stripe_global_idx) {
    auto const& stripe       = selected_stripes[stripe_global_idx];
    auto const stripe_info   = stripe.stripe_info;
    auto const stripe_footer = stripe.stripe_footer;

    std::size_t this_stripe_size{0};
    auto const last_read_size = read_info.size();
    for (std::size_t level = 0; level < num_levels; ++level) {
      auto& stream_info = _file_itm_data.lvl_stream_info[level];

      auto stream_level_count = stream_info.size();
      auto const stripe_level_size =
        gather_stream_info_and_column_desc(stripe_global_idx,
                                           level,
                                           stripe_info,
                                           stripe_footer,
                                           col_meta.orc_col_map[level],
                                           _metadata.get_types(),
                                           false,  // use_index,
                                           level == 0,
                                           nullptr,  // num_dictionary_entries
                                           nullptr,  // local_stream_order
                                           &stream_info,
                                           nullptr  // chunks
        );

      auto const is_stripe_data_empty = stripe_level_size == 0;
      CUDF_EXPECTS(not is_stripe_data_empty or stripe_info->indexLength == 0,
                   "Invalid index rowgroup stream data");

      lvl_stripe_sizes[level][stripe_global_idx] = stripe_level_size;
      this_stripe_size += stripe_level_size;

      // Range of the streams in `stream_info` corresponding to this stripe at the current level.
      lvl_stripe_stream_ranges[level][stripe_global_idx] =
        range{stream_level_count, stream_info.size()};

      // Coalesce consecutive streams into one read.
      while (not is_stripe_data_empty and stream_level_count < stream_info.size()) {
        auto const d_dst  = stream_info[stream_level_count].dst_pos;
        auto const offset = stream_info[stream_level_count].offset;
        auto len          = stream_info[stream_level_count].length;
        stream_level_count++;

        while (stream_level_count < stream_info.size() &&
               stream_info[stream_level_count].offset == offset + len) {
          len += stream_info[stream_level_count].length;
          stream_level_count++;
        }
        read_info.emplace_back(stream_data_read_info{offset,
                                                     d_dst,
                                                     len,
                                                     static_cast<std::size_t>(stripe.source_idx),
                                                     stripe_global_idx,
                                                     level});
      }
    }  // end loop level

    if (!load_all_stripes) { total_stripe_sizes[stripe_global_idx] = {1, this_stripe_size}; }

    // Range of all stream reads in `read_info` corresponding to this stripe, in all levels.
    stripe_data_read_ranges[stripe_global_idx] = range{last_read_size, read_info.size()};
  }

  //
  // Split range of all stripes into subranges that can be loaded separately while maintaining
  // the memory usage under the given pass limit:
  //

  // Load range is reset to start from the first position in `load_stripe_ranges`.
  _chunk_read_data.curr_load_stripe_range = 0;

  if (load_all_stripes) {
    _chunk_read_data.load_stripe_ranges = {range{0UL, num_total_stripes}};
    return;
  }

  // Compute the prefix sum of stripes' data sizes.
  total_stripe_sizes.host_to_device_async(_stream);
  thrust::inclusive_scan(rmm::exec_policy_nosync(_stream, cudf::get_current_device_resource_ref()),
                         total_stripe_sizes.d_begin(),
                         total_stripe_sizes.d_end(),
                         total_stripe_sizes.d_begin(),
                         cumulative_size_plus{});
  total_stripe_sizes.device_to_host(_stream);

  auto const load_limit = [&] {
    auto const tmp = static_cast<std::size_t>(_chunk_read_data.pass_read_limit *
                                              chunk_read_data::load_limit_ratio);
    // Make sure not to pass 0 byte limit (due to round-off) to `find_splits`.
    return std::max(tmp, 1UL);
  }();

  _chunk_read_data.load_stripe_ranges =
    find_splits<cumulative_size>(total_stripe_sizes, num_total_stripes, load_limit);
}

// If there is a data read limit, only a subset of stripes are read at a time such that
// their total data size does not exceed a fixed size limit. Then, the data is probed to
// estimate its uncompressed sizes, which are in turn used to split that stripe subset into
// smaller subsets, each of which to be decompressed and decoded in the next step
// `decompress_and_decode_stripes()`. This is to ensure that loading data from data sources
// together with decompression and decoding will be capped around the given data read limit.
void reader_impl::load_next_stripe_data(read_mode mode)
{
  if (!_file_itm_data.has_data()) { return; }

  auto const load_stripe_range =
    _chunk_read_data.load_stripe_ranges[_chunk_read_data.curr_load_stripe_range++];
  auto const stripe_start = load_stripe_range.begin;
  auto const stripe_count = load_stripe_range.size();

  auto& lvl_stripe_data = _file_itm_data.lvl_stripe_data;
  auto const num_levels = _selected_columns.num_levels();

  // Prepare the buffer to read raw data onto.
  for (std::size_t level = 0; level < num_levels; ++level) {
    auto& stripe_data = lvl_stripe_data[level];
    stripe_data.resize(stripe_count);

    for (std::size_t idx = 0; idx < stripe_count; ++idx) {
      auto const stripe_size = _file_itm_data.lvl_stripe_sizes[level][idx + stripe_start];
      stripe_data[idx]       = rmm::device_buffer(
        cudf::util::round_up_safe(stripe_size, BUFFER_PADDING_MULTIPLE), _stream);
    }
  }

  //
  // Load stripe data into memory:
  //

  // Storing the future and the expected size of the read data
  std::vector<std::pair<std::future<std::size_t>, std::size_t>> device_read_tasks;
  // Storing the future, the expected size of the read data and the device destination pointer
  std::vector<std::tuple<std::future<std::unique_ptr<datasource::buffer>>, std::size_t, uint8_t*>>
    host_read_tasks;

  // Range of the read info (offset, length) to read for the current being loaded stripes.
  auto const [read_begin, read_end] =
    merge_selected_ranges(_file_itm_data.stripe_data_read_ranges, load_stripe_range);

  bool stream_synchronized{false};

  for (auto read_idx = read_begin; read_idx < read_end; ++read_idx) {
    auto const& read_info = _file_itm_data.data_read_info[read_idx];
    auto const source_ptr = _metadata.per_file_metadata[read_info.source_idx].source;
    auto const dst_base   = static_cast<uint8_t*>(
      lvl_stripe_data[read_info.level][read_info.stripe_idx - stripe_start].data());

    if (source_ptr->is_device_read_preferred(read_info.length)) {
      // `device_read_async` may not use _stream at all.
      // Instead, it may use some other stream(s) to sync the H->D memcpy.
      // As such, we need to make sure the device buffers in `lvl_stripe_data` are ready first.
      if (!stream_synchronized) {
        _stream.synchronize();
        stream_synchronized = true;
      }
      device_read_tasks.emplace_back(
        source_ptr->device_read_async(
          read_info.offset, read_info.length, dst_base + read_info.dst_pos, _stream),
        read_info.length);
    } else {
      host_read_tasks.emplace_back(source_ptr->host_read_async(read_info.offset, read_info.length),
                                   read_info.length,
                                   dst_base + read_info.dst_pos);
    }
  }
  std::vector<std::unique_ptr<cudf::io::datasource::buffer>> host_read_buffers;
  for (auto& [fut, expected_size, dev_dst] : host_read_tasks) {  // if there were host reads
    host_read_buffers.emplace_back(fut.get());
    auto* host_buffer = host_read_buffers.back().get();
    CUDF_EXPECTS(host_buffer->size() == expected_size, "Unexpected discrepancy in bytes read.");
    CUDF_CUDA_TRY(
      cudf::detail::memcpy_async(dev_dst, host_buffer->data(), host_buffer->size(), _stream));
  }
  _stream.synchronize();

  for (auto& task : device_read_tasks) {  // if there were device reads
    CUDF_EXPECTS(task.first.get() == task.second, "Unexpected discrepancy in bytes read.");
  }

  // Compute number of rows in the loading stripes.
  auto const num_loading_rows = std::accumulate(
    _file_itm_data.selected_stripes.begin() + stripe_start,
    _file_itm_data.selected_stripes.begin() + stripe_start + stripe_count,
    std::size_t{0},
    [](std::size_t count, auto const& stripe) { return count + stripe.stripe_info->numberOfRows; });

  // Decoding range needs to be reset to start from the first position in `decode_stripe_ranges`.
  _chunk_read_data.curr_decode_stripe_range = 0;

  // The cudf's column size limit.
  auto constexpr column_size_limit =
    static_cast<std::size_t>(std::numeric_limits<size_type>::max());

  // Decode all loaded stripes if there is no read limit, or if we are in READ_ALL mode,
  // and the number of loading rows is less than the column size limit.
  // In theory, we should just decode 'enough' stripes for output one table chunk, instead of
  // decoding all stripes like this, for better load-balancing and reduce memory usage.
  // However, we do not have any good way to know how many stripes are 'enough'.
  if ((mode == read_mode::READ_ALL || _chunk_read_data.pass_read_limit == 0) &&
      // In addition to read limit, we also need to check if the total number of
      // rows in the loaded stripes exceeds the column size limit.
      // If that is the case, we cannot decode all stripes at once into a cudf table.
      num_loading_rows <= column_size_limit) {
    _chunk_read_data.decode_stripe_ranges = {load_stripe_range};
    return;
  }

  // From here, we have reading mode that is either:
  // - CHUNKED_READ without read limit but the number of reading rows exceeds column size limit, or
  // - CHUNKED_READ with a pass read limit.
  // READ_ALL mode with number of rows more than cudf's column size limit should be handled early in
  // `preprocess_file`. We just check again to make sure such situations never happen here.
  CUDF_EXPECTS(
    mode != read_mode::READ_ALL,
    "READ_ALL mode does not support reading number of rows more than cudf's column size limit.");

  // This is the post-processing step after we've done with splitting `load_stripe_range` into
  // `decode_stripe_ranges`.
  auto const add_range_offset = [stripe_start](std::vector<range>& new_ranges) {
    // The split ranges always start from zero.
    // We need to change these ranges to start from `stripe_start` which are the correct subranges
    // of the current loaded stripe range.
    for (auto& range : new_ranges) {
      range.begin += stripe_start;
      range.end += stripe_start;
    }
  };

  // Optimized code path when we do not have any read limit but the number of rows in the
  // loaded stripes exceeds column size limit.
  // Note that the values `max_uncompressed_size` for each stripe are not computed here.
  // Instead, they will be computed on the fly during decoding to avoid the overhead of
  // storing and retrieving from memory.
  if (_chunk_read_data.pass_read_limit == 0 && num_loading_rows > column_size_limit) {
    std::vector<cumulative_size_and_row> cumulative_stripe_rows(stripe_count);
    std::size_t rows{0};

    for (std::size_t idx = 0; idx < stripe_count; ++idx) {
      auto const& stripe     = _file_itm_data.selected_stripes[idx + stripe_start];
      auto const stripe_info = stripe.stripe_info;
      rows += stripe_info->numberOfRows;

      // We will split stripe ranges based only on stripes' number of rows, not data size.
      // Thus, we override the cumulative `size_bytes` using the prefix sum of rows in stripes and
      // will use the column size limit as the split size limit.
      cumulative_stripe_rows[idx] =
        cumulative_size_and_row{idx + 1UL /*count*/, rows /*size_bytes*/, rows};
    }

    _chunk_read_data.decode_stripe_ranges =
      find_splits<cumulative_size_and_row>(cumulative_stripe_rows, stripe_count, column_size_limit);
    add_range_offset(_chunk_read_data.decode_stripe_ranges);
    return;
  }

  //
  // Split range of loaded stripes into subranges that can be decoded separately such that the
  // memory usage is maintained around the given limit:
  //

  // This is for estimating the decompressed sizes of the loaded stripes.
  cudf::detail::hostdevice_vector<cumulative_size_and_row> stripe_decomp_sizes(stripe_count,
                                                                               _stream);

  // Fill up the `cumulative_size_and_row` array with initial values.
  // Note: `hostdevice_vector::begin()` mirrors `std::vector::data()` using incorrect API name.
  for (std::size_t idx = 0; idx < stripe_count; ++idx) {
    auto const& stripe     = _file_itm_data.selected_stripes[idx + stripe_start];
    auto const stripe_info = stripe.stripe_info;
    stripe_decomp_sizes[idx] =
      cumulative_size_and_row{1UL /*count*/, 0UL /*size_bytes*/, stripe_info->numberOfRows};
  }

  auto& compinfo_map = _file_itm_data.compinfo_map;
  compinfo_map.clear();  // clear cache of the last load

  // For parsing decompression data.
  // We create an array that is large enough to use for all levels, thus only need to allocate
  // memory once.
  auto hd_compinfo = [&] {
    std::size_t max_num_streams{0};
    if (_metadata.per_file_metadata[0].ps.compression != NONE) {
      // Find the maximum number of streams in all levels of the loaded stripes.
      for (std::size_t level = 0; level < num_levels; ++level) {
        auto const stream_range =
          merge_selected_ranges(_file_itm_data.lvl_stripe_stream_ranges[level], load_stripe_range);
        max_num_streams = std::max(max_num_streams, stream_range.size());
      }
    }
    return cudf::detail::hostdevice_vector<compressed_stream_info>(max_num_streams, _stream);
  }();

  for (std::size_t level = 0; level < num_levels; ++level) {
    auto const& stream_info = _file_itm_data.lvl_stream_info[level];
    auto const num_columns  = _selected_columns.levels[level].size();

    auto& stripe_data = lvl_stripe_data[level];
    if (stripe_data.empty()) { continue; }

    // Range of all streams in the loaded stripes.
    auto const stream_range =
      merge_selected_ranges(_file_itm_data.lvl_stripe_stream_ranges[level], load_stripe_range);

    if (_metadata.per_file_metadata[0].ps.compression != NONE) {
      auto const& decompressor = *_metadata.per_file_metadata[0].decompressor;

      auto compinfo = cudf::detail::hostdevice_span<compressed_stream_info>{hd_compinfo}.subspan(
        0, stream_range.size());
      for (auto stream_idx = stream_range.begin; stream_idx < stream_range.end; ++stream_idx) {
        auto const& info = stream_info[stream_idx];
        auto const dst_base =
          static_cast<uint8_t const*>(stripe_data[info.source.stripe_idx - stripe_start].data());
        compinfo[stream_idx - stream_range.begin] =
          compressed_stream_info(dst_base + info.dst_pos, info.length);
      }

      // Estimate the uncompressed data.
      compinfo.host_to_device_async(_stream);
      parse_compressed_stripe_data(compinfo.device_ptr(),
                                   compinfo.size(),
                                   decompressor.GetBlockSize(),
                                   decompressor.GetLog2MaxCompressionRatio(),
                                   _stream);
      compinfo.device_to_host(_stream);

      for (auto stream_idx = stream_range.begin; stream_idx < stream_range.end; ++stream_idx) {
        auto const& info           = stream_info[stream_idx];
        auto const stream_compinfo = compinfo[stream_idx - stream_range.begin];

        // Cache these parsed numbers so they can be reused in the decompression/decoding step.
        compinfo_map[info.source] = {stream_compinfo.num_compressed_blocks,
                                     stream_compinfo.num_uncompressed_blocks,
                                     stream_compinfo.max_uncompressed_size,
                                     stream_compinfo.max_uncompressed_block_size};
        stripe_decomp_sizes[info.source.stripe_idx - stripe_start].size_bytes +=
          stream_compinfo.max_uncompressed_size;
      }

    } else {  // no decompression
      // Set decompression sizes equal to the input sizes.
      for (auto stream_idx = stream_range.begin; stream_idx < stream_range.end; ++stream_idx) {
        auto const& info = stream_info[stream_idx];
        stripe_decomp_sizes[info.source.stripe_idx - stripe_start].size_bytes += info.length;
      }
    }
  }  // end loop level

  // Compute the prefix sum of stripe data sizes and rows.
  stripe_decomp_sizes.host_to_device_async(_stream);
  thrust::inclusive_scan(rmm::exec_policy_nosync(_stream, cudf::get_current_device_resource_ref()),
                         stripe_decomp_sizes.d_begin(),
                         stripe_decomp_sizes.d_end(),
                         stripe_decomp_sizes.d_begin(),
                         cumulative_size_plus{});
  stripe_decomp_sizes.device_to_host(_stream);

  auto const decode_limit = [&] {
    auto const tmp = static_cast<std::size_t>(_chunk_read_data.pass_read_limit *
                                              chunk_read_data::decompress_and_decode_limit_ratio);
    // Make sure not to pass 0 byte limit to `find_splits`.
    return std::max(tmp, 1UL);
  }();

  _chunk_read_data.decode_stripe_ranges =
    find_splits<cumulative_size_and_row>(stripe_decomp_sizes, stripe_count, decode_limit);

  add_range_offset(_chunk_read_data.decode_stripe_ranges);
}

}  // namespace cudf::io::orc::detail

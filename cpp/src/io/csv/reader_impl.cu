/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file reader_impl.cu
 * @brief cuDF-IO CSV reader class implementation
 */

#include "csv_common.hpp"
#include "csv_gpu.hpp"
#include "io/utilities/column_buffer.hpp"
#include "io/utilities/hostdevice_vector.hpp"
#include "io/utilities/parsing_utils.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_worker_pool.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/detail/utilities/visitor_overload.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/io/datasource.hpp>
#include <cudf/io/detail/codec.hpp>
#include <cudf/io/detail/csv.hpp>
#include <cudf/io/types.hpp>
#include <cudf/logger.hpp>
#include <cudf/strings/detail/copy_if_else.cuh>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/host_vector.h>

#include <algorithm>
#include <future>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using std::string;
using std::vector;

using cudf::device_span;
using cudf::host_span;
using cudf::detail::make_device_uvector_async;

namespace cudf {
namespace io {
namespace detail {
namespace csv {
using namespace cudf::io::csv;
using namespace cudf::io;

namespace {

/**
 * @brief Offsets of CSV rows in device memory, accessed through a shrinkable span.
 *
 * Row offsets are stored this way to avoid reallocation/copies when discarding front or back
 * elements.
 */
class selected_rows_offsets {
  rmm::device_uvector<uint64_t> all;
  device_span<uint64_t const> selected;

 public:
  selected_rows_offsets(rmm::device_uvector<uint64_t>&& data,
                        device_span<uint64_t const> selected_span)
    : all{std::move(data)}, selected{selected_span}
  {
  }
  explicit selected_rows_offsets(rmm::cuda_stream_view stream) : all{0, stream}, selected{all} {}

  operator device_span<uint64_t const>() const { return selected; }
  void shrink(size_t size)
  {
    CUDF_EXPECTS(size <= selected.size(), "New size must be smaller");
    selected = selected.subspan(0, size);
  }
  void erase_first_n(size_t n)
  {
    CUDF_EXPECTS(n <= selected.size(), "Too many elements to remove");
    selected = selected.subspan(n, selected.size() - n);
  }
  auto size() const { return selected.size(); }
  auto data() const { return selected.data(); }
};

/**
 * @brief Discards any other characters found before the first quotechar and after the last
 * quotechar in the string (if any quotechar exists)
 *
 * ```
 * Example:
 * "column" => column
 * \t"column"\t => column
 *     "column"     => column
 * ```
 *
 */
std::string_view remove_quotes(std::string_view str, char quotechar)
{
  // Exclude first and last quotation char
  auto const first_quote = str.find(quotechar);

  if (first_quote == string::npos) { return str; }

  str = str.substr(first_quote + 1);

  auto const last_quote = str.rfind(quotechar);

  if (last_quote == string::npos) { return str; }

  return str.substr(0, last_quote);
}

/**
 * @brief Parse a row of input to get the column names. The row can either be the header, or the
 * first data row. If the header is not used, column names are generated automatically.
 */
std::vector<std::string> get_column_names(std::vector<char> const& row,
                                          parse_options_view const& parse_opts,
                                          int header_row,
                                          std::string prefix)
{
  // Empty row, return empty column names vector
  if (row.empty()) { return {}; }

  std::vector<std::string> col_names;
  bool quotation = false;
  for (size_t pos = 0, prev = 0; pos < row.size(); ++pos) {
    // Flip the quotation flag if current character is a quotechar
    if (row[pos] == parse_opts.quotechar) { quotation = !quotation; }
    // Check if end of a column/row
    if (pos == row.size() - 1 || (!quotation && row[pos] == parse_opts.terminator) ||
        (!quotation && row[pos] == parse_opts.delimiter)) {
      // Include the current character, in case the line is not terminated
      int col_name_len = pos - prev + 1;
      // Exclude the delimiter/terminator if present
      if (row[pos] == parse_opts.delimiter || row[pos] == parse_opts.terminator) { --col_name_len; }
      // Also exclude '\r' character at the end of the column name if it's
      // part of the terminator
      if (col_name_len > 0 && parse_opts.terminator == '\n' && row[pos] == '\n' &&
          row[pos - 1] == '\r') {
        --col_name_len;
      }

      // In delim_whitespace mode, leading/trailing/internal whitespace runs must not produce
      // empty fields (pandas behavior). Internal runs are collapsed by the adjacent-delimiter
      // skip loop below; this flag handles the leading/trailing empty-field cases here.
      bool const skip_empty_field = parse_opts.multi_delimiter && col_name_len == 0;

      if (!skip_empty_field) {
        if (header_row >= 0) {
          // This is the header, add the column name
          col_names.emplace_back(remove_quotes(
            std::string_view{row.data() + prev, static_cast<std::size_t>(col_name_len)},
            parse_opts.quotechar));
        } else {
          // This is the first data row, add the automatically generated name
          col_names.push_back(prefix + std::to_string(col_names.size()));
        }
      }

      // Stop parsing when we hit the line terminator; relevant when there is
      // a blank line following the header. In this case, row includes
      // multiple line terminators at the end, as the new recStart belongs to
      // a line that comes after the blank line(s)
      if (!quotation && row[pos] == parse_opts.terminator) { break; }

      // Skip adjacent delimiters if delim_whitespace is set
      while (parse_opts.multi_delimiter && pos + 1 < row.size() &&
             row[pos] == parse_opts.delimiter && row[pos + 1] == parse_opts.delimiter) {
        ++pos;
      }
      prev = pos + 1;
    }
  }

  return col_names;
}

template <typename C>
void erase_except_last(C& container, rmm::cuda_stream_view stream)
{
  cudf::detail::device_single_thread(
    [span = device_span<typename C::value_type>{container}] __device__() mutable -> void {
      span.front() = span.back();
    },
    stream);
  container.resize(1, stream);
}

constexpr std::array<uint8_t, 3> UTF8_BOM = {0xEF, 0xBB, 0xBF};
[[nodiscard]] bool has_utf8_bom(host_span<char const> data)
{
  return data.size() >= UTF8_BOM.size() &&
         memcmp(data.data(), UTF8_BOM.data(), UTF8_BOM.size()) == 0;
}

/**
 * @brief Finds row positions in the specified input data, and loads the selected data onto GPU.
 *
 * This function scans the input data to record the row offsets (relative to the start of the
 * input data). A row is actually the data/offset between two termination symbols.
 *
 *  @param[in] source The source data (may be compressed)
 *  @param[in] reader_opts Settings for controlling reading behavior
 *  @param[in] parse_opts Settings for controlling parsing behavior
 *  @param[out] header The header row, if any
 *  @param[in] data Host buffer containing uncompressed data, if input is compressed
 *  @param[in] byte_range_offset Offset of the byte range
 *  @param[in] range_begin Start of the first row, relative to the byte range start
 *  @param[in] range_end End of the data to read, relative to the byte range start; equal to the
 * data size if all data after byte_range_offset needs to be read
 *  @param[in] skip_rows Number of rows to skip from the start
 *  @param[in] num_rows Number of rows to read; -1 means all
 *  @param[in] load_whole_file Indicates if the whole file should be read
 *  @param[in] stream CUDA stream used for device memory operations and kernel launches
 *  @return Input data and row offsets in the device memory
 */
std::pair<rmm::device_uvector<char>, selected_rows_offsets> load_data_and_gather_row_offsets(
  cudf::io::datasource* source,
  csv_reader_options const& reader_opts,
  parse_options const& parse_opts,
  std::vector<char>& header,
  std::optional<host_span<char const>> data,
  size_t byte_range_offset,
  size_t range_begin,
  size_t range_end,
  size_t skip_rows,
  int64_t num_rows,
  bool load_whole_file,
  rmm::cuda_stream_view stream)
{
  constexpr size_t max_chunk_bytes = 64 * 1024 * 1024;  // 64MB

  auto const data_size      = data.has_value() ? data->size() : source->size();
  auto const buffer_size    = std::min(max_chunk_bytes, data_size);
  auto const max_input_size = [&] {
    if (range_end == data_size) {
      return data_size - byte_range_offset;
    } else {
      return std::min<size_t>(reader_opts.get_byte_range_size_with_padding(),
                              data_size - byte_range_offset);
    }
  }();
  auto const header_rows = (reader_opts.get_header() >= 0) ? reader_opts.get_header() + 1 : 0;

  // For compatibility with the previous parser, a row is considered in-range if the
  // previous row terminator is within the given range
  range_end += (range_end < data_size);

  auto pos = range_begin;
  // When using byte range, need the line terminator of last line before the range
  auto input_pos = byte_range_offset == 0 ? pos : pos - 1;
  uint64_t ctx   = 0;

  rmm::device_uvector<char> d_data{0, stream};
  d_data.reserve((load_whole_file) ? data_size : std::min(buffer_size * 2, max_input_size), stream);
  rmm::device_uvector<uint64_t> all_row_offsets{0, stream};

  auto const max_blocks =
    std::max<size_t>((buffer_size / cudf::io::csv::gpu::rowofs_block_bytes) + 1, 2);
  cudf::detail::hostdevice_vector<uint64_t> row_ctx(max_blocks, stream);
  do {
    auto const target_pos = std::min(pos + max_chunk_bytes, max_input_size);
    auto const chunk_size = target_pos - pos;

    auto const previous_data_size = d_data.size();
    d_data.resize(target_pos - input_pos, stream);

    auto const read_offset = byte_range_offset + input_pos + previous_data_size;
    auto const read_size   = target_pos - input_pos - previous_data_size;
    if (data.has_value()) {
      cudf::detail::cuda_memcpy_async(
        device_span<char>{d_data.data() + previous_data_size, read_size},
        data->subspan(read_offset, read_size),
        stream);
    } else {
      if (source->is_device_read_preferred(read_size)) {
        source->device_read(read_offset,
                            read_size,
                            reinterpret_cast<uint8_t*>(d_data.data() + previous_data_size),
                            stream);
      } else {
        auto const buffer = source->host_read(read_offset, read_size);
        // Use sync version to prevent buffer going out of scope before we copy the data.
        cudf::detail::cuda_memcpy(
          device_span<char>{d_data.data() + previous_data_size, read_size},
          host_span<char const>{reinterpret_cast<char const*>(buffer->data()), buffer->size()},
          stream);
      }
    }

    // Pass 1: Count the potential number of rows in each character block for each
    // possible parser state at the beginning of the block.
    auto const num_blocks = cudf::io::csv::gpu::gather_row_offsets(parse_opts.view(),
                                                                   row_ctx.device_ptr(),
                                                                   device_span<uint64_t>(),
                                                                   d_data,
                                                                   chunk_size,
                                                                   pos,
                                                                   input_pos,
                                                                   max_input_size,
                                                                   range_begin,
                                                                   range_end,
                                                                   skip_rows,
                                                                   stream);

    cudf::detail::cuda_memcpy(
      host_span<uint64_t>(row_ctx.host_ptr(), row_ctx.size(), true).subspan(0, num_blocks),
      device_span<uint64_t const>(row_ctx.device_ptr(), row_ctx.size()).subspan(0, num_blocks),
      stream);

    // Sum up the rows in each character block, selecting the row count that
    // corresponds to the current input context. Also stores the now known input
    // context per character block that will be needed by the second pass.
    for (uint32_t i = 0; i < num_blocks; i++) {
      uint64_t ctx_next = cudf::io::csv::gpu::select_row_context(ctx, row_ctx[i]);
      row_ctx[i]        = ctx;
      ctx               = ctx_next;
    }
    size_t total_rows = ctx >> 2;
    if (total_rows > skip_rows) {
      // At least one row in range in this batch
      all_row_offsets.resize(total_rows - skip_rows, stream);

      cudf::detail::cuda_memcpy_async(
        device_span<uint64_t>(row_ctx.device_ptr(), row_ctx.size()).subspan(0, num_blocks),
        host_span<uint64_t const>(row_ctx.host_ptr(), row_ctx.size(), true).subspan(0, num_blocks),
        stream);

      // Pass 2: Output row offsets
      cudf::io::csv::gpu::gather_row_offsets(parse_opts.view(),
                                             row_ctx.device_ptr(),
                                             all_row_offsets,
                                             d_data,
                                             chunk_size,
                                             pos,
                                             input_pos,
                                             max_input_size,
                                             range_begin,
                                             range_end,
                                             skip_rows,
                                             stream);
      // With byte range, we want to keep only one row out of the specified range
      if (range_end < data_size) {
        cudf::detail::cuda_memcpy(
          host_span<uint64_t>(row_ctx.host_ptr(), row_ctx.size(), true).subspan(0, num_blocks),
          device_span<uint64_t const>(row_ctx.device_ptr(), row_ctx.size()).subspan(0, num_blocks),
          stream);

        size_t rows_out_of_range = 0;
        for (uint32_t i = 0; i < num_blocks; i++) {
          rows_out_of_range += row_ctx[i];
        }
        if (rows_out_of_range != 0) {
          // Keep one row out of range (used to infer length of previous row)
          auto new_row_offsets_size =
            all_row_offsets.size() - std::min(rows_out_of_range - 1, all_row_offsets.size());
          all_row_offsets.resize(new_row_offsets_size, stream);
          // Implies we reached the end of the range
          break;
        }
      }
      // num_rows does not include blank rows
      if (num_rows >= 0) {
        if (all_row_offsets.size() > header_rows + static_cast<size_t>(num_rows)) {
          size_t num_blanks = cudf::io::csv::gpu::count_blank_rows(
            parse_opts.view(), d_data, all_row_offsets, stream);
          if (all_row_offsets.size() - num_blanks > header_rows + static_cast<size_t>(num_rows)) {
            // Got the desired number of rows
            break;
          }
        }
      }
    } else {
      // Discard data (all rows below skip_rows), keeping one character for history
      size_t discard_bytes = std::max(d_data.size(), sizeof(char)) - sizeof(char);
      if (discard_bytes != 0) {
        erase_except_last(d_data, stream);
        input_pos += discard_bytes;
      }
    }
    pos = target_pos;
  } while (pos < max_input_size);

  auto const non_blank_row_offsets =
    io::csv::gpu::remove_blank_rows(parse_opts.view(), d_data, all_row_offsets, stream);
  auto row_offsets = selected_rows_offsets{std::move(all_row_offsets), non_blank_row_offsets};

  // Remove header rows and extract header
  auto const header_row_index = std::max<size_t>(header_rows, 1) - 1;
  if (header_row_index + 1 < row_offsets.size()) {
    cudf::detail::cuda_memcpy(host_span<uint64_t>{row_ctx}.subspan(0, 2),
                              device_span<uint64_t const>{row_offsets.data() + header_row_index, 2},
                              stream);

    auto const header_start = input_pos + row_ctx[0];
    auto const header_end   = input_pos + row_ctx[1];
    CUDF_EXPECTS(header_start <= header_end && header_end <= max_input_size,
                 "Invalid csv header location");
    header.resize(header_end - header_start);
    if (data.has_value()) {
      std::copy(data->begin() + byte_range_offset + header_start,
                data->begin() + byte_range_offset + header_end,
                header.begin());
    } else {
      source->host_read(header_start + byte_range_offset,
                        header_end - header_start,
                        reinterpret_cast<uint8_t*>(header.data()));
    }
    if (header_rows > 0) { row_offsets.erase_first_n(header_rows); }
  }
  // Apply num_rows limit
  if (num_rows >= 0 && static_cast<size_t>(num_rows) < row_offsets.size() - 1) {
    row_offsets.shrink(num_rows + 1);
  }
  return {std::move(d_data), std::move(row_offsets)};
}

std::pair<rmm::device_uvector<char>, selected_rows_offsets> select_data_and_row_offsets(
  cudf::io::datasource* source,
  csv_reader_options const& reader_opts,
  std::vector<char>& header,
  parse_options const& parse_opts,
  rmm::cuda_stream_view stream)
{
  auto range_offset  = reader_opts.get_byte_range_offset();
  auto range_size    = reader_opts.get_byte_range_size();
  auto skip_rows     = reader_opts.get_skiprows();
  auto skip_end_rows = reader_opts.get_skipfooter();
  auto num_rows      = reader_opts.get_nrows();

  if (range_offset > 0 || range_size > 0) {
    CUDF_EXPECTS(reader_opts.get_compression() == compression_type::NONE,
                 "Reading compressed data using `byte range` is unsupported");
  }

  CUDF_EXPECTS(range_offset <= source->size(), "Invalid byte range offset", std::invalid_argument);

  // TODO: Allow parsing the header outside the mapped range
  CUDF_EXPECTS((range_offset == 0 || reader_opts.get_header() < 0),
               "byte_range offset with header not supported");

  if (source->is_empty()) {
    return {rmm::device_uvector<char>{0, stream}, selected_rows_offsets{stream}};
  }

  std::optional<host_span<char const>> h_data;
  std::vector<uint8_t> h_uncomp_data_owner;
  if (reader_opts.get_compression() != compression_type::NONE) {
    auto const h_comp_data = source->host_read(0, source->size());
    h_uncomp_data_owner =
      decompress(reader_opts.get_compression(), {h_comp_data->data(), h_comp_data->size()});
    h_data = host_span<char const>{reinterpret_cast<char const*>(h_uncomp_data_owner.data()),
                                   h_uncomp_data_owner.size()};
  }

  size_t data_start_offset = range_offset;
  if (h_data.has_value()) {
    if (has_utf8_bom(*h_data)) { data_start_offset += sizeof(UTF8_BOM); }
  } else {
    if (range_offset == 0) {
      auto bom_buffer = source->host_read(0, std::min<size_t>(source->size(), sizeof(UTF8_BOM)));
      auto bom_chars  = host_span<char const>{reinterpret_cast<char const*>(bom_buffer->data()),
                                              bom_buffer->size()};
      if (has_utf8_bom(bom_chars)) { data_start_offset += sizeof(UTF8_BOM); }
    } else {
      auto find_data_start_chunk_size = 1024ul;
      while (data_start_offset < source->size()) {
        auto const read_size =
          std::min(find_data_start_chunk_size, source->size() - data_start_offset);
        auto buffer = source->host_read(data_start_offset, read_size);
        auto buffer_chars =
          host_span<char const>{reinterpret_cast<char const*>(buffer->data()), buffer->size()};

        if (auto first_row_start =
              std::find(buffer_chars.begin(), buffer_chars.end(), parse_opts.terminator);
            first_row_start != buffer_chars.end()) {
          data_start_offset += std::distance(buffer_chars.begin(), first_row_start) + 1;
          break;
        }
        data_start_offset += read_size;
        find_data_start_chunk_size *= 2;
      }
    }
  }

  // None of the parameters for row selection is used, we are parsing the entire file
  bool const load_whole_file =
    range_offset == 0 && range_size == 0 && skip_rows <= 0 && skip_end_rows <= 0 && num_rows == -1;

  // Transfer source data to GPU and gather row offsets
  auto const uncomp_size = h_data.has_value() ? h_data->size() : source->size();
  auto data_row_offsets  = load_data_and_gather_row_offsets(source,
                                                           reader_opts,
                                                           parse_opts,
                                                           header,
                                                           h_data,
                                                           range_offset,
                                                           data_start_offset - range_offset,
                                                           (range_size) ? range_size : uncomp_size,
                                                           (skip_rows > 0) ? skip_rows : 0,
                                                           num_rows,
                                                           load_whole_file,
                                                           stream);
  auto& row_offsets      = data_row_offsets.second;
  // Exclude the rows that are to be skipped from the end
  if (skip_end_rows > 0 && static_cast<size_t>(skip_end_rows) < row_offsets.size()) {
    row_offsets.shrink(row_offsets.size() - skip_end_rows);
  }

  return data_row_offsets;
}

void select_data_types(host_span<data_type const> user_dtypes,
                       host_span<column_parse::flags> column_flags,
                       host_span<data_type> column_types)
{
  if (user_dtypes.empty()) { return; }

  CUDF_EXPECTS(user_dtypes.size() == 1 || user_dtypes.size() == column_flags.size(),
               "Specify data types for all columns in file, or use a dictionary/map");

  for (auto col_idx = 0u; col_idx < column_flags.size(); ++col_idx) {
    if (column_flags[col_idx] & column_parse::enabled) {
      // If it's a single dtype, assign that dtype to all active columns
      auto const& dtype     = user_dtypes.size() == 1 ? user_dtypes[0] : user_dtypes[col_idx];
      column_types[col_idx] = dtype;
      // Reset the inferred flag, no need to infer the types from the data
      column_flags[col_idx] &= ~column_parse::inferred;
    }
  }
}

void get_data_types_from_column_names(std::map<std::string, data_type> const& user_dtypes,
                                      host_span<std::string const> column_names,
                                      host_span<column_parse::flags> column_flags,
                                      host_span<data_type> column_types)
{
  if (user_dtypes.empty()) { return; }
  for (auto col_idx = 0u; col_idx < column_flags.size(); ++col_idx) {
    if (column_flags[col_idx] & column_parse::enabled) {
      auto const col_type_it = user_dtypes.find(column_names[col_idx]);
      if (col_type_it != user_dtypes.end()) {
        // Assign the type from the map
        column_types[col_idx] = col_type_it->second;
        // Reset the inferred flag, no need to infer the types from the data
        column_flags[col_idx] &= ~column_parse::inferred;
      }
    }
  }
}

void infer_column_types(parse_options const& parse_opts,
                        host_span<column_parse::flags const> column_flags,
                        device_span<char const> data,
                        device_span<uint64_t const> row_offsets,
                        int32_t num_records,
                        data_type timestamp_type,
                        host_span<data_type> column_types,
                        rmm::cuda_stream_view stream)
{
  if (num_records == 0) {
    for (auto col_idx = 0u; col_idx < column_flags.size(); ++col_idx) {
      if (column_flags[col_idx] & column_parse::inferred) {
        column_types[col_idx] = data_type(cudf::type_id::STRING);
      }
    }
    return;
  }

  auto const num_inferred_columns =
    std::count_if(column_flags.begin(), column_flags.end(), [](auto& flags) {
      return flags & column_parse::inferred;
    });
  if (num_inferred_columns == 0) { return; }

  auto const column_stats = cudf::io::csv::gpu::detect_column_types(
    parse_opts.view(),
    data,
    make_device_uvector_async(column_flags, stream, cudf::get_current_device_resource_ref()),
    row_offsets,
    num_inferred_columns,
    stream);
  stream.synchronize();

  auto inf_col_idx = 0;
  for (auto col_idx = 0u; col_idx < column_flags.size(); ++col_idx) {
    if (not(column_flags[col_idx] & column_parse::inferred)) { continue; }
    auto const& stats = column_stats[inf_col_idx++];
    if (stats.null_count == num_records or stats.total_count() == 0) {
      // Entire column is NULL; allocate the smallest amount of memory
      column_types[col_idx] = data_type(cudf::type_id::INT8);
    } else if (stats.string_count > 0L) {
      column_types[col_idx] = data_type(cudf::type_id::STRING);
    } else if (stats.datetime_count > 0L) {
      column_types[col_idx] = timestamp_type.id() == cudf::type_id::EMPTY
                                ? data_type(cudf::type_id::TIMESTAMP_NANOSECONDS)
                                : timestamp_type;
    } else if (stats.bool_count > 0L) {
      column_types[col_idx] = data_type(cudf::type_id::BOOL8);
    } else if (stats.float_count > 0L) {
      column_types[col_idx] = data_type(cudf::type_id::FLOAT64);
    } else if (stats.big_int_count == 0) {
      column_types[col_idx] = data_type(cudf::type_id::INT64);
    } else if (stats.big_int_count != 0 && stats.negative_small_int_count != 0) {
      column_types[col_idx] = data_type(cudf::type_id::STRING);
    } else {
      // Integers are stored as 64-bit to conform to PANDAS
      column_types[col_idx] = data_type(cudf::type_id::UINT64);
    }
  }
}

/**
 * @brief Result of decode_data containing column buffers and quoted field tracking
 */
struct decode_result {
  std::vector<column_buffer> buffers;
  std::vector<rmm::device_uvector<bool>> is_quoted_flags;
};

decode_result decode_data(parse_options const& parse_opts,
                          host_span<column_parse::flags const> column_flags,
                          std::vector<std::string> const& column_names,
                          device_span<char const> data,
                          device_span<uint64_t const> row_offsets,
                          host_span<data_type const> column_types,
                          int32_t num_records,
                          int32_t num_actual_columns,
                          int32_t num_active_columns,
                          rmm::cuda_stream_view stream,
                          rmm::device_async_resource_ref mr)
{
  // Alloc output; columns' data memory is still expected for empty dataframe
  std::vector<column_buffer> out_buffers;
  out_buffers.reserve(column_types.size());

  for (int col = 0, active_col = 0; col < num_actual_columns; ++col) {
    if (column_flags[col] & column_parse::enabled) {
      auto out_buffer = column_buffer(column_types[active_col], num_records, true, stream, mr);

      out_buffer.name = column_names[col];
      out_buffers.emplace_back(std::move(out_buffer));
      active_col++;
    }
  }

  auto h_data  = cudf::detail::make_host_vector<void*>(num_active_columns, stream);
  auto h_valid = cudf::detail::make_host_vector<bitmask_type*>(num_active_columns, stream);

  for (int i = 0; i < num_active_columns; ++i) {
    h_data[i]  = out_buffers[i].data();
    h_valid[i] = out_buffers[i].null_mask();
  }

  // Allocate is_quoted_flags arrays for string columns to track which fields were quoted
  std::vector<rmm::device_uvector<bool>> is_quoted_flags_storage;
  auto h_is_quoted_flags = cudf::detail::make_host_vector<bool*>(num_active_columns, stream);
  for (int i = 0; i < num_active_columns; ++i) {
    if (column_types[i].id() == type_id::STRING) {
      is_quoted_flags_storage.emplace_back(cudf::detail::make_zeroed_device_uvector_async<bool>(
        num_records, stream, cudf::get_current_device_resource_ref()));
      h_is_quoted_flags[i] = is_quoted_flags_storage.back().data();
    }
  }

  auto d_valid_counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    num_active_columns, stream, cudf::get_current_device_resource_ref());

  cudf::io::csv::gpu::decode_row_column_data(
    parse_opts.view(),
    data,
    make_device_uvector_async(column_flags, stream, cudf::get_current_device_resource_ref()),
    row_offsets,
    make_device_uvector_async(column_types, stream, cudf::get_current_device_resource_ref()),
    make_device_uvector_async(h_data, stream, cudf::get_current_device_resource_ref()),
    make_device_uvector_async(h_valid, stream, cudf::get_current_device_resource_ref()),
    d_valid_counts,
    make_device_uvector_async(h_is_quoted_flags, stream, cudf::get_current_device_resource_ref()),
    stream);

  auto const h_valid_counts = cudf::detail::make_host_vector(d_valid_counts, stream);
  for (int i = 0; i < num_active_columns; ++i) {
    out_buffers[i].null_count() = num_records - h_valid_counts[i];
  }

  return {std::move(out_buffers), std::move(is_quoted_flags_storage)};
}

cudf::detail::host_vector<data_type> determine_column_types(
  csv_reader_options const& reader_opts,
  parse_options const& parse_opts,
  host_span<std::string const> column_names,
  device_span<char const> data,
  device_span<uint64_t const> row_offsets,
  int32_t num_records,
  host_span<column_parse::flags> column_flags,
  cudf::size_type num_active_columns,
  rmm::cuda_stream_view stream)
{
  std::vector<data_type> column_types(column_flags.size());

  std::visit(cudf::detail::visitor_overload{
               [&](std::vector<data_type> const& user_dtypes) {
                 return select_data_types(user_dtypes, column_flags, column_types);
               },
               [&](std::map<std::string, data_type> const& user_dtypes) {
                 return get_data_types_from_column_names(
                   user_dtypes, column_names, column_flags, column_types);
               }},
             reader_opts.get_dtypes());

  infer_column_types(parse_opts,
                     column_flags,
                     data,
                     row_offsets,
                     num_records,
                     reader_opts.get_timestamp_type(),
                     column_types,
                     stream);

  // compact column_types to only include active columns
  auto active_col_types =
    cudf::detail::make_empty_host_vector<data_type>(num_active_columns, stream);
  std::copy_if(column_types.cbegin(),
               column_types.cend(),
               std::back_inserter(active_col_types),
               [&column_flags, &types = std::as_const(column_types)](auto& dtype) {
                 auto const idx = std::distance(types.data(), &dtype);
                 return column_flags[idx] & column_parse::enabled;
               });

  return active_col_types;
}

table_with_metadata read_csv(cudf::io::datasource* source,
                             csv_reader_options const& reader_opts,
                             parse_options const& parse_opts,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  std::vector<char> header;

  auto const data_row_offsets =
    select_data_and_row_offsets(source, reader_opts, header, parse_opts, stream);

  auto const& data        = data_row_offsets.first;
  auto const& row_offsets = data_row_offsets.second;

  auto const unique_use_cols_indexes = std::set(reader_opts.get_use_cols_indexes().cbegin(),
                                                reader_opts.get_use_cols_indexes().cend());

  auto const detected_column_names =
    get_column_names(header, parse_opts.view(), reader_opts.get_header(), reader_opts.get_prefix());
  auto const opts_have_all_col_names =
    not reader_opts.get_names().empty() and
    (
      // no data to detect (the number of) columns
      detected_column_names.empty() or
      // number of user specified names matches what is detected
      reader_opts.get_names().size() == detected_column_names.size() or
      // Columns are not selected by indices; read first reader_opts.get_names().size() columns
      unique_use_cols_indexes.empty());
  auto column_names = opts_have_all_col_names ? reader_opts.get_names() : detected_column_names;

  auto const num_actual_columns = static_cast<int32_t>(column_names.size());
  auto num_active_columns       = num_actual_columns;
  auto column_flags =
    cudf::detail::make_host_vector<column_parse::flags>(num_actual_columns, stream);
  std::fill(
    column_flags.begin(), column_flags.end(), column_parse::enabled | column_parse::inferred);

  // User did not pass column names to override names in the file
  // Process names from the file to remove empty and duplicated strings
  if (not opts_have_all_col_names) {
    std::vector<size_t> col_loop_order(column_names.size());
    auto unnamed_it = std::copy_if(
      cuda::counting_iterator<size_t>{0},
      cuda::counting_iterator<size_t>{column_names.size()},
      col_loop_order.begin(),
      [&column_names](auto col_idx) -> bool { return not column_names[col_idx].empty(); });

    // Rename empty column names to "Unnamed: col_index"
    std::copy_if(cuda::counting_iterator<size_t>{0},
                 cuda::counting_iterator<size_t>{column_names.size()},
                 unnamed_it,
                 [&column_names](auto col_idx) -> bool {
                   auto is_empty = column_names[col_idx].empty();
                   if (is_empty)
                     column_names[col_idx] = string("Unnamed: ") + std::to_string(col_idx);
                   return is_empty;
                 });

    // Looking for duplicates
    std::unordered_map<string, int> col_names_counts;
    if (!reader_opts.is_enabled_mangle_dupe_cols()) {
      for (auto& col_name : column_names) {
        if (++col_names_counts[col_name] > 1) {
          CUDF_LOG_WARN("Multiple columns with name %s; only the first appearance is parsed",
                        col_name);

          auto const idx    = &col_name - column_names.data();
          column_flags[idx] = column_parse::disabled;
        }
      }
    } else {
      // For constant/linear search.
      std::unordered_multiset<std::string> header(column_names.begin(), column_names.end());
      for (auto const col_idx : col_loop_order) {
        auto col       = column_names[col_idx];
        auto cur_count = col_names_counts[col];
        if (cur_count > 0) {
          auto const old_col = col;
          // Rename duplicates of column X as X.1, X.2, ...; First appearance stays as X
          while (cur_count > 0) {
            col_names_counts[old_col] = cur_count + 1;
            col                       = old_col + "." + std::to_string(cur_count);
            if (header.find(col) != header.end()) {
              cur_count++;
            } else {
              cur_count = col_names_counts[col];
            }
          }
          if (auto pos = header.find(old_col); pos != header.end()) { header.erase(pos); }
          header.insert(col);
          column_names[col_idx] = col;
        }
        col_names_counts[col] = cur_count + 1;
      }
    }

    // Update the number of columns to be processed, if some might have been removed
    if (!reader_opts.is_enabled_mangle_dupe_cols()) {
      num_active_columns = col_names_counts.size();
    }
  }

  // User can specify which columns should be parsed
  auto const unique_use_cols_names = std::unordered_set(reader_opts.get_use_cols_names().cbegin(),
                                                        reader_opts.get_use_cols_names().cend());
  auto const is_column_selection_used =
    not unique_use_cols_names.empty() or not unique_use_cols_indexes.empty();

  // Reset flags and output column count; columns will be reactivated based on the selection options
  if (is_column_selection_used) {
    std::fill(column_flags.begin(), column_flags.end(), column_parse::disabled);
    num_active_columns = 0;
  }

  // Column selection via column indexes
  if (not unique_use_cols_indexes.empty()) {
    // Users can pass names for the selected columns only, if selecting column by their indices
    auto const are_opts_col_names_used =
      not reader_opts.get_names().empty() and not opts_have_all_col_names;
    CUDF_EXPECTS(not are_opts_col_names_used or
                   reader_opts.get_names().size() == unique_use_cols_indexes.size(),
                 "Specify names of all columns in the file, or names of all selected columns");

    for (auto const index : unique_use_cols_indexes) {
      column_flags[index] = column_parse::enabled | column_parse::inferred;
      if (are_opts_col_names_used) {
        column_names[index] = reader_opts.get_names()[num_active_columns];
      }
      ++num_active_columns;
    }
  }

  // Column selection via column names
  if (not unique_use_cols_names.empty()) {
    for (auto const& name : unique_use_cols_names) {
      auto const it = std::find(column_names.cbegin(), column_names.cend(), name);
      CUDF_EXPECTS(it != column_names.end(), "Nonexistent column selected");
      auto const col_idx = std::distance(column_names.cbegin(), it);
      if (column_flags[col_idx] == column_parse::disabled) {
        column_flags[col_idx] = column_parse::enabled | column_parse::inferred;
        ++num_active_columns;
      }
    }
  }

  // User can specify which columns should be read as datetime
  if (!reader_opts.get_parse_dates_indexes().empty() ||
      !reader_opts.get_parse_dates_names().empty()) {
    for (auto const index : reader_opts.get_parse_dates_indexes()) {
      column_flags[index] |= column_parse::as_datetime;
    }

    for (auto const& name : reader_opts.get_parse_dates_names()) {
      auto it = std::find(column_names.begin(), column_names.end(), name);
      if (it != column_names.end()) {
        column_flags[it - column_names.begin()] |= column_parse::as_datetime;
      }
    }
  }

  // User can specify which columns should be parsed as hexadecimal
  if (!reader_opts.get_parse_hex_indexes().empty() || !reader_opts.get_parse_hex_names().empty()) {
    for (auto const index : reader_opts.get_parse_hex_indexes()) {
      column_flags[index] |= column_parse::as_hexadecimal;
    }

    for (auto const& name : reader_opts.get_parse_hex_names()) {
      auto it = std::find(column_names.begin(), column_names.end(), name);
      if (it != column_names.end()) {
        column_flags[it - column_names.begin()] |= column_parse::as_hexadecimal;
      }
    }
  }

  // Return empty table rather than exception if nothing to load
  if (num_active_columns == 0) { return {std::make_unique<table>(), {}}; }

  // Exclude the end-of-data row from number of rows with actual data
  auto const num_records  = std::max(row_offsets.size(), 1ul) - 1;
  auto const column_types = determine_column_types(reader_opts,
                                                   parse_opts,
                                                   column_names,
                                                   data,
                                                   row_offsets,
                                                   num_records,
                                                   column_flags,
                                                   num_active_columns,
                                                   stream);

  auto metadata    = table_metadata{};
  auto out_columns = std::vector<std::unique_ptr<cudf::column>>();
  out_columns.reserve(column_types.size());
  if (num_records != 0) {
    auto decode_result = decode_data(  //
      parse_opts,
      column_flags,
      column_names,
      data,
      row_offsets,
      column_types,
      num_records,
      num_actual_columns,
      num_active_columns,
      stream,
      mr);

    out_columns.resize(column_types.size());

    auto& out_buffers     = decode_result.buffers;
    auto& is_quoted_flags = decode_result.is_quoted_flags;

    bool const doublequote_enabled = (parse_opts.quotechar != '\0' && parse_opts.doublequote);

    // Identify string columns that need doublequote processing
    std::vector<size_t> string_col_indices;
    if (doublequote_enabled) {
      for (size_t i = 0; i < column_types.size(); ++i) {
        if (column_types[i].id() == type_id::STRING) { string_col_indices.push_back(i); }
      }
    }

    // Process string columns with doublequote handling in parallel using thread pool
    auto const num_string_cols = string_col_indices.size();
    if (num_string_cols > 0) {
      auto const quotechar = parse_opts.quotechar;
      cudf::string_scalar quotechar_scalar(
        std::string(1, quotechar), true, stream, cudf::get_current_device_resource_ref());
      cudf::string_scalar dblquotechar_scalar(
        std::string(2, quotechar), true, stream, cudf::get_current_device_resource_ref());
      constexpr size_t max_tasks = 4;
      auto const cols_per_task   = cudf::util::div_rounding_up_safe(num_string_cols, max_tasks);
      auto const num_tasks       = cudf::util::div_rounding_up_safe(num_string_cols, cols_per_task);
      auto streams               = cudf::detail::fork_streams(stream, num_tasks);

      auto process_string_column = [&](size_t str_col_idx, rmm::cuda_stream_view col_stream) {
        auto const col_idx   = string_col_indices[str_col_idx];
        auto const is_quoted = device_span<bool>(is_quoted_flags[str_col_idx]);
        auto* buffer         = &out_buffers[col_idx];

        auto const is_quoted_row = [] __device__(auto const flag) -> bool { return flag; };
        if (cudf::detail::none_of(is_quoted.begin(), is_quoted.end(), is_quoted_row, col_stream)) {
          // Fast path: no rows were quoted, skip replacement entirely
          out_columns[col_idx] = make_column(*buffer, nullptr, std::nullopt, col_stream);
        } else {
          auto replaced_all_col = cudf::strings::detail::replace(
            cudf::make_strings_column(
              *buffer->_strings, col_stream, cudf::get_current_device_resource_ref())
              ->view(),
            dblquotechar_scalar,
            quotechar_scalar,
            -1,
            col_stream,
            mr);
          if (cudf::detail::all_of(is_quoted.begin(), is_quoted.end(), is_quoted_row, col_stream)) {
            // Fast path: all rows were quoted, apply replacement to all
            out_columns[col_idx] = std::move(replaced_all_col);
          } else {
            // Need to replace only the quoted rows
            auto const replaced_all_view =
              cudf::column_device_view::create(replaced_all_col->view(), col_stream);
            auto const replaced_all_iter = cudf::detail::make_optional_iterator<cudf::string_view>(
              *replaced_all_view, cudf::nullate::DYNAMIC{replaced_all_col->nullable()});

            auto const* original_pairs = buffer->_strings->data();
            auto const original_iter   = thrust::make_transform_iterator(
              cuda::counting_iterator<size_type>{0},
              cuda::proclaim_return_type<cuda::std::optional<cudf::string_view>>(
                [original_pairs] __device__(
                  size_type idx) -> cuda::std::optional<cudf::string_view> {
                  auto const& p = original_pairs[idx];
                  return p.first != nullptr
                             ? cuda::std::optional<cudf::string_view>{cudf::string_view{p.first,
                                                                                      p.second}}
                             : cuda::std::nullopt;
                }));

            out_columns[col_idx] = cudf::strings::detail::copy_if_else(
              replaced_all_iter,
              replaced_all_iter + num_records,
              original_iter,
              [is_quoted] __device__(size_type idx) -> bool { return is_quoted[idx]; },
              col_stream,
              mr);
          }
        }
      };

      std::vector<std::future<void>> tasks;
      tasks.reserve(num_tasks);

      for (size_t task_id = 0; task_id < num_tasks; ++task_id) {
        auto const start_col  = task_id * cols_per_task;
        auto const end_col    = std::min(start_col + cols_per_task, num_string_cols);
        auto const col_stream = streams[task_id];
        tasks.emplace_back(
          cudf::detail::host_worker_pool().submit_task([&, start_col, end_col, col_stream]() {
            for (size_t str_col_idx = start_col; str_col_idx < end_col; ++str_col_idx) {
              process_string_column(str_col_idx, col_stream);
            }
          }));
      }

      for (auto& task : tasks) {
        task.get();
      }

      cudf::detail::join_streams(streams, stream);

      for (auto const col_idx : string_col_indices) {
        if (out_columns[col_idx]) {
          out_columns[col_idx] = cudf::rebind_stream(std::move(*out_columns[col_idx]), stream);
        }
      }
    }

    // Create output columns for the columns that were not processed in the parallel loop
    for (size_t i = 0; i < column_types.size(); ++i) {
      if (!out_columns[i]) {
        out_columns[i] = make_column(out_buffers[i], nullptr, std::nullopt, stream);
      }
    }

    for (size_t i = 0; i < column_types.size(); ++i) {
      metadata.schema_info.emplace_back(out_buffers[i].name);
    }
  } else {
    // Create empty columns
    for (auto column_type : column_types) {
      out_columns.emplace_back(make_empty_column(column_type));
    }
    // Handle empty metadata
    for (int col = 0; col < num_actual_columns; ++col) {
      if (column_flags[col] & column_parse::enabled) {
        metadata.schema_info.emplace_back(column_names[col]);
      }
    }
  }
  return {std::make_unique<table>(std::move(out_columns)), std::move(metadata)};
}

/**
 * @brief Create a serialized trie for N/A value matching, based on the options.
 */
cudf::detail::trie create_na_trie(char quotechar,
                                  csv_reader_options const& reader_opts,
                                  rmm::cuda_stream_view stream)
{
  // Default values to recognize as null values
  static std::vector<std::string> const default_na_values{"",
                                                          "#N/A",
                                                          "#N/A N/A",
                                                          "#NA",
                                                          "-1.#IND",
                                                          "-1.#QNAN",
                                                          "-NaN",
                                                          "-nan",
                                                          "1.#IND",
                                                          "1.#QNAN",
                                                          "<NA>",
                                                          "N/A",
                                                          "NA",
                                                          "NULL",
                                                          "NaN",
                                                          "n/a",
                                                          "nan",
                                                          "null"};

  if (!reader_opts.is_enabled_na_filter()) { return cudf::detail::trie(0, stream); }

  std::vector<std::string> na_values = reader_opts.get_na_values();
  if (reader_opts.is_enabled_keep_default_na()) {
    na_values.insert(na_values.end(), default_na_values.begin(), default_na_values.end());
  }

  // Pandas treats empty strings as N/A if empty fields are treated as N/A
  if (std::find(na_values.begin(), na_values.end(), "") != na_values.end()) {
    na_values.emplace_back(2, quotechar);
  }

  return cudf::detail::create_serialized_trie(na_values, stream);
}

parse_options make_parse_options(csv_reader_options const& reader_opts,
                                 rmm::cuda_stream_view stream)
{
  auto parse_opts = parse_options{};

  if (reader_opts.is_enabled_delim_whitespace()) {
    parse_opts.delimiter       = ' ';
    parse_opts.multi_delimiter = true;
  } else {
    parse_opts.delimiter       = reader_opts.get_delimiter();
    parse_opts.multi_delimiter = false;
  }

  parse_opts.terminator = reader_opts.get_lineterminator();

  if (reader_opts.get_quotechar() != '\0' && reader_opts.get_quoting() != quote_style::NONE) {
    parse_opts.quotechar  = reader_opts.get_quotechar();
    parse_opts.keepquotes = false;
    parse_opts.detect_whitespace_around_quotes =
      reader_opts.is_enabled_detect_whitespace_around_quotes();
    parse_opts.doublequote = reader_opts.is_enabled_doublequote();
  } else {
    parse_opts.quotechar   = '\0';
    parse_opts.keepquotes  = true;
    parse_opts.doublequote = false;
  }

  parse_opts.skipblanklines = reader_opts.is_enabled_skip_blank_lines();
  parse_opts.comment        = reader_opts.get_comment();
  parse_opts.dayfirst       = reader_opts.is_enabled_dayfirst();
  parse_opts.decimal        = reader_opts.get_decimal();
  parse_opts.thousands      = reader_opts.get_thousands();

  CUDF_EXPECTS(parse_opts.decimal != parse_opts.delimiter,
               "Decimal point cannot be the same as the delimiter");
  CUDF_EXPECTS(parse_opts.thousands != parse_opts.delimiter,
               "Thousands separator cannot be the same as the delimiter");

  // Handle user-defined true values, whereby field data is substituted with a
  // boolean true or numeric `1` value
  if (not reader_opts.get_true_values().empty()) {
    parse_opts.trie_true =
      cudf::detail::create_serialized_trie(reader_opts.get_true_values(), stream);
  }

  // Handle user-defined false values, whereby field data is substituted with a
  // boolean false or numeric `0` value
  if (not reader_opts.get_false_values().empty()) {
    parse_opts.trie_false =
      cudf::detail::create_serialized_trie(reader_opts.get_false_values(), stream);
  }

  // Handle user-defined N/A values, whereby field data is treated as null
  parse_opts.trie_na = create_na_trie(parse_opts.quotechar, reader_opts, stream);

  return parse_opts;
}

}  // namespace

table_with_metadata read_csv(std::unique_ptr<cudf::io::datasource>&& source,
                             csv_reader_options const& options,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto parse_options = make_parse_options(options, stream);

  return read_csv(source.get(), options, parse_options, stream, mr);
}

}  // namespace csv
}  // namespace detail
}  // namespace io
}  // namespace cudf

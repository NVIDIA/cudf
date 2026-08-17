/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl.hpp"

#include "column_path_helpers.hpp"
#include "error.hpp"
#include "runtime/context.hpp"

#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/stream_compaction.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/io/parquet_schema.hpp>
#include <cudf/logger.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

#include <bitset>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace cudf::io::parquet::detail {

void reader_impl::decode_page_data(read_mode mode, size_t skip_rows, size_t num_rows)
{
  CUDF_FUNC_RANGE();

  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  auto& page_nesting        = subpass.page_nesting_info;
  auto& page_nesting_decode = subpass.page_nesting_decode_info;

  auto const level_type_size = pass.level_type_size;

  // Should not reach here if there is no page data.
  CUDF_EXPECTS(subpass.pages.size() > 0, "There are no pages to decode");

  size_t const sum_max_depths = std::accumulate(
    pass.chunks.begin(), pass.chunks.end(), 0, [&](size_t cursum, ColumnChunkDesc const& chunk) {
      return cursum + _metadata->get_output_nesting_depth(chunk.src_col_schema);
    });

  auto const kernel_mask = subpass.kernel_mask;
  auto const has_strings = (kernel_mask & STRINGS_MASK) != 0;

  // Check to see if there are any string columns present. If so, then we need to get size info
  // for each string page. This size info will be used to pre-allocate memory for the column,
  // allowing the page decoder to write string data directly to the column buffer, rather than
  // doing a gather operation later on.
  // TODO: This step is somewhat redundant if size info has already been calculated (nested schema,
  // chunked reader).
  auto col_string_sizes = cudf::detail::make_host_vector<size_t>(_input_columns.size(), _stream);
  if (has_strings) {
    // Compute column string sizes (using page string offsets) for this output table chunk
    col_string_sizes = calculate_page_string_offsets();

    // Check for overflow in cumulative column string sizes of this pass so that the page string
    // offsets of overflowing (large) string columns are treated as 64-bit.
    auto const threshold         = static_cast<size_t>(strings::detail::get_offset64_threshold());
    auto const has_large_strings = std::any_of(col_string_sizes.cbegin(),
                                               col_string_sizes.cend(),
                                               [=](std::size_t sz) { return sz > threshold; });
    if (has_large_strings and not strings::detail::is_large_strings_enabled()) {
      CUDF_FAIL("String column exceeds the column size limit", std::overflow_error);
    }

    // Mark/unmark column-chunk descriptors depending on the string sizes of corresponding output
    // column chunks and the large strings threshold.
    for (auto& chunk : pass.chunks) {
      auto const idx            = chunk.src_col_index;
      chunk.is_large_string_col = (col_string_sizes[idx] > threshold);
    }
  }

  // In order to reduce the number of allocations of hostdevice_vector, we allocate a single vector
  // to store all per-chunk pointers to nested data/nullmask. `chunk_offsets[i]` will store the
  // offset into `chunk_nested_data`/`chunk_nested_valids` for the array of pointers for chunk `i`
  auto chunk_nested_valids =
    cudf::detail::hostdevice_vector<bitmask_type*>(sum_max_depths, _stream);
  auto chunk_nested_data = cudf::detail::hostdevice_vector<void*>(sum_max_depths, _stream);
  auto chunk_offsets     = std::vector<size_t>();
  auto chunk_nested_str_data =
    cudf::detail::hostdevice_vector<void*>(has_strings ? sum_max_depths : 0, _stream);

  // Update chunks with pointers to column data.
  for (size_t c = 0, chunk_off = 0; c < pass.chunks.size(); c++) {
    input_column_info const& input_col = _input_columns[pass.chunks[c].src_col_index];
    CUDF_EXPECTS(input_col.schema_idx == pass.chunks[c].src_col_schema,
                 "Column/page schema index mismatch");

    size_t const max_depth = _metadata->get_output_nesting_depth(pass.chunks[c].src_col_schema);
    chunk_offsets.push_back(chunk_off);

    // get a slice of size `nesting depth` from `chunk_nested_valids` to store an array of pointers
    // to validity data
    auto valids                   = chunk_nested_valids.host_ptr(chunk_off);
    pass.chunks[c].valid_map_base = chunk_nested_valids.device_ptr(chunk_off);

    // get a slice of size `nesting depth` from `chunk_nested_data` to store an array of pointers to
    // out data
    auto data                       = chunk_nested_data.host_ptr(chunk_off);
    pass.chunks[c].column_data_base = chunk_nested_data.device_ptr(chunk_off);

    auto str_data = has_strings ? chunk_nested_str_data.host_ptr(chunk_off) : nullptr;
    pass.chunks[c].column_string_base =
      has_strings ? chunk_nested_str_data.device_ptr(chunk_off) : nullptr;

    chunk_off += max_depth;

    // fill in the arrays on the host.  there are some important considerations to
    // take into account here for nested columns.  specifically, with structs
    // there is sharing of output buffers between input columns.  consider this schema
    //
    //  required group field_id=1 name {
    //    required binary field_id=2 firstname (String);
    //    required binary field_id=3 middlename (String);
    //    required binary field_id=4 lastname (String);
    // }
    //
    // there are 3 input columns of data here (firstname, middlename, lastname), but
    // only 1 output column (name).  The structure of the output column buffers looks like
    // the schema itself
    //
    // struct      (name)
    //     string  (firstname)
    //     string  (middlename)
    //     string  (lastname)
    //
    // The struct column can contain validity information. the problem is, the decode
    // step for the input columns will all attempt to decode this validity information
    // because each one has it's own copy of the repetition/definition levels. but
    // since this is all happening in parallel it would mean multiple blocks would
    // be stomping all over the same memory randomly.  to work around this, we set
    // things up so that only 1 child of any given nesting level fills in the
    // data (offsets in the case of lists) or validity information for the higher
    // levels of the hierarchy that are shared.  In this case, it would mean we
    // would just choose firstname to be the one that decodes the validity for name.
    //
    // we do this by only handing out the pointers to the first child we come across.
    //
    auto* cols = &_output_buffers;
    for (size_t idx = 0; idx < max_depth; idx++) {
      auto& out_buf = (*cols)[input_col.nesting[idx]];
      cols          = &out_buf.children;

      int const owning_schema = out_buf.user_data & PARQUET_COLUMN_BUFFER_SCHEMA_MASK;
      if (owning_schema == 0 || owning_schema == input_col.schema_idx) {
        valids[idx] = out_buf.null_mask();
        data[idx]   = out_buf.data();
        // only do string buffer for leaf
        if (idx == max_depth - 1 and out_buf.string_size() == 0 and
            col_string_sizes[pass.chunks[c].src_col_index] > 0) {
          out_buf.create_string_data(col_string_sizes[pass.chunks[c].src_col_index],
                                     pass.chunks[c].is_large_string_col,
                                     _stream);
        }
        if (has_strings) { str_data[idx] = out_buf.string_data(); }
        out_buf.user_data |=
          static_cast<uint32_t>(input_col.schema_idx) & PARQUET_COLUMN_BUFFER_SCHEMA_MASK;
      } else {
        valids[idx] = nullptr;
        data[idx]   = nullptr;
      }
    }
  }

  // Create an empty device vector to store the initial str offset for large string columns from for
  // string decoders.
  auto initial_str_offsets = rmm::device_uvector<size_t>{0, _stream, _mr};

  pass.chunks.host_to_device_async(_stream);
  chunk_nested_valids.host_to_device_async(_stream);
  chunk_nested_data.host_to_device_async(_stream);
  if (has_strings) {
    // Host vector to initialize the initial string offsets
    auto host_offsets_vector =
      cudf::detail::make_pinned_vector_async<size_t>(_input_columns.size(), _stream);
    std::fill(
      host_offsets_vector.begin(), host_offsets_vector.end(), std::numeric_limits<size_t>::max());
    // Initialize the initial string offsets vector from the host vector
    initial_str_offsets = cudf::detail::make_device_uvector(host_offsets_vector, _stream, _mr);
    chunk_nested_str_data.host_to_device_async(_stream);
  }

  // create this before we fork streams
  kernel_error error_code(_stream);

  // get the number of streams we need from the pool and tell them to wait on the H2D copies
  int const nkernels = std::bitset<32>(kernel_mask).count();
  auto streams       = cudf::detail::fork_streams(_stream, nkernels);

  int s_idx = 0;

  auto decode_data = [&](decode_kernel_mask decoder_mask) {
    detail::decode_page_data(subpass.pages,
                             pass.chunks,
                             num_rows,
                             skip_rows,
                             level_type_size,
                             decoder_mask,
                             subpass_page_mask_span(),
                             initial_str_offsets,
                             subpass.page_string_offset_indices,
                             error_code.data(),
                             streams[s_idx++]);
  };

  // launch string decoder for plain encoded flat columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING) != 0) {
    decode_data(decode_kernel_mask::STRING);
  }

  // launch string decoder for plain encoded nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_NESTED) != 0) {
    decode_data(decode_kernel_mask::STRING_NESTED);
  }

  // launch string decoder for plain encoded list columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_LIST) != 0) {
    decode_data(decode_kernel_mask::STRING_LIST);
  }

  // launch string decoder for dictionary encoded flat columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_DICT) != 0) {
    decode_data(decode_kernel_mask::STRING_DICT);
  }

  // launch string decoder for dictionary encoded nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_DICT_NESTED) != 0) {
    decode_data(decode_kernel_mask::STRING_DICT_NESTED);
  }

  // launch string decoder for dictionary encoded list columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_DICT_LIST) != 0) {
    decode_data(decode_kernel_mask::STRING_DICT_LIST);
  }

  // launch string decoder for byte-stream-split encoded flat columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_STREAM_SPLIT) != 0) {
    decode_data(decode_kernel_mask::STRING_STREAM_SPLIT);
  }

  // launch string decoder for byte-stream-split encoded nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_STREAM_SPLIT_NESTED) != 0) {
    decode_data(decode_kernel_mask::STRING_STREAM_SPLIT_NESTED);
  }

  // launch string decoder for byte-stream-split encoded list columns
  if (BitAnd(kernel_mask, decode_kernel_mask::STRING_STREAM_SPLIT_LIST) != 0) {
    decode_data(decode_kernel_mask::STRING_STREAM_SPLIT_LIST);
  }

  // launch dict-index-as-int32 decoder for flat columns
  if (BitAnd(kernel_mask, decode_kernel_mask::DICT_INT32) != 0) {
    decode_data(decode_kernel_mask::DICT_INT32);
  }

  // launch delta byte array decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::DELTA_BYTE_ARRAY) != 0) {
    decode_delta_byte_array(subpass.pages,
                            pass.chunks,
                            num_rows,
                            skip_rows,
                            level_type_size,
                            subpass_page_mask_span(),
                            initial_str_offsets,
                            error_code.data(),
                            streams[s_idx++]);
  }

  // launch delta length byte array decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::DELTA_LENGTH_BA) != 0) {
    decode_delta_length_byte_array(subpass.pages,
                                   pass.chunks,
                                   num_rows,
                                   skip_rows,
                                   level_type_size,
                                   subpass_page_mask_span(),
                                   initial_str_offsets,
                                   error_code.data(),
                                   streams[s_idx++]);
  }

  // launch delta binary decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::DELTA_BINARY) != 0) {
    decode_delta_binary(subpass.pages,
                        pass.chunks,
                        num_rows,
                        skip_rows,
                        level_type_size,
                        subpass_page_mask_span(),
                        error_code.data(),
                        streams[s_idx++]);
  }

  // launch byte stream split decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_FLAT) != 0) {
    decode_data(decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_FLAT);
  }

  // launch byte stream split decoder, for nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED) != 0) {
    decode_data(decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_NESTED);
  }

  // launch byte stream split decoder, for list columns
  if (BitAnd(kernel_mask, decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST) != 0) {
    decode_data(decode_kernel_mask::BYTE_STREAM_SPLIT_FIXED_WIDTH_LIST);
  }

  // launch byte stream split decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::BYTE_STREAM_SPLIT) != 0) {
    decode_split_page_data(subpass.pages,
                           pass.chunks,
                           num_rows,
                           skip_rows,
                           level_type_size,
                           subpass_page_mask_span(),
                           error_code.data(),
                           streams[s_idx++]);
  }

  // launch fixed width type decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_NO_DICT) != 0) {
    decode_data(decode_kernel_mask::FIXED_WIDTH_NO_DICT);
  }

  // launch fixed width type decoder for lists
  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_NO_DICT_LIST) != 0) {
    decode_data(decode_kernel_mask::FIXED_WIDTH_NO_DICT_LIST);
  }

  // launch fixed width type decoder, for nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_NO_DICT_NESTED) != 0) {
    decode_data(decode_kernel_mask::FIXED_WIDTH_NO_DICT_NESTED);
  }

  // launch boolean type decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::BOOLEAN) != 0) {
    decode_data(decode_kernel_mask::BOOLEAN);
  }

  // launch boolean type decoder, for nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::BOOLEAN_NESTED) != 0) {
    decode_data(decode_kernel_mask::BOOLEAN_NESTED);
  }

  // launch boolean type decoder, for nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::BOOLEAN_LIST) != 0) {
    decode_data(decode_kernel_mask::BOOLEAN_LIST);
  }

  // launch fixed width type decoder with dictionaries
  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_DICT) != 0) {
    decode_data(decode_kernel_mask::FIXED_WIDTH_DICT);
  }

  // launch fixed width type decoder with dictionaries for lists
  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_DICT_LIST) != 0) {
    decode_data(decode_kernel_mask::FIXED_WIDTH_DICT_LIST);
  }

  // launch fixed width type decoder with dictionaries, for nested columns
  if (BitAnd(kernel_mask, decode_kernel_mask::FIXED_WIDTH_DICT_NESTED) != 0) {
    decode_data(decode_kernel_mask::FIXED_WIDTH_DICT_NESTED);
  }

  // launch the catch-all page decoder
  if (BitAnd(kernel_mask, decode_kernel_mask::GENERAL) != 0) {
    detail::decode_page_data(subpass.pages,
                             pass.chunks,
                             num_rows,
                             skip_rows,
                             level_type_size,
                             subpass_page_mask_span(),
                             error_code.data(),
                             streams[s_idx++]);
  }

  // synchronize the streams
  cudf::detail::join_streams(streams, _stream);

  // the delta_temp_buf in the subpass struct can be freed now
  subpass.delta_temp_buf.release();

  subpass.pages.device_to_host_async(_stream);
  page_nesting.device_to_host_async(_stream);
  page_nesting_decode.device_to_host_async(_stream);

  // Invalidate output buffer nullmasks at row indices spanned by pruned pages
  update_output_nullmasks_for_pruned_pages(subpass_page_mask_span(), skip_rows, num_rows);

  // Fill output offsets for pruned pages before retrieving large string initial offsets.
  fill_pruned_offsets(skip_rows, num_rows, initial_str_offsets);

  // Copy over initial string offsets from device
  auto h_initial_str_offsets = cudf::detail::make_pinned_vector_async(initial_str_offsets, _stream);

  if (auto const error = error_code.value_sync(_stream); error != 0) {
    CUDF_FAIL("Parquet data decode failed with code(s) " + kernel_error::to_string(error));
  }

  // For list and string columns, add the final offset to every offset buffer.
  // Note : the reason we are doing this here instead of in the decode kernel is
  // that it is difficult/impossible for a given page to know that it is writing the very
  // last value that should then be followed by a terminator (because rows can span
  // page boundaries).
  std::vector<size_type*> out_buffers;
  std::vector<size_type> final_offsets;
  out_buffers.reserve(_input_columns.size());
  final_offsets.reserve(_input_columns.size());
  for (size_t idx = 0; idx < _input_columns.size(); idx++) {
    input_column_info const& input_col = _input_columns[idx];

    auto* cols = &_output_buffers;
    for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      if (out_buf.type.id() == type_id::LIST &&
          (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED) == 0) {
        CUDF_EXPECTS(l_idx < input_col.nesting_depth() - 1, "Encountered a leaf list column");
        auto const& child = (*cols)[input_col.nesting[l_idx + 1]];

        // the final offset for a list at level N is the size of it's child
        size_type const offset = child.type.id() == type_id::LIST ? child.size - 1 : child.size;
        out_buffers.emplace_back(static_cast<size_type*>(out_buf.data()) + (out_buf.size - 1));
        final_offsets.emplace_back(offset);
        out_buf.user_data |= PARQUET_COLUMN_BUFFER_FLAG_LIST_TERMINATED;
      } else if (out_buf.type.id() == type_id::STRING) {
        // only if it is not a large strings column
        if (std::cmp_less_equal(col_string_sizes[idx], strings::detail::get_offset64_threshold())) {
          out_buffers.emplace_back(static_cast<size_type*>(out_buf.data()) + out_buf.size);
          final_offsets.emplace_back(static_cast<size_type>(col_string_sizes[idx]));
        }
        // Nested large strings column
        else if (input_col.nesting_depth() > 0) {
          // A fully pruned list may have no string child values and therefore no page from which
          // to record an initial offset.
          if (out_buf.size == 0) {
            out_buf.set_initial_string_offset(0);
          } else {
            CUDF_EXPECTS(h_initial_str_offsets[idx] != std::numeric_limits<size_t>::max(),
                         "Encountered invalid initial offset for large string column");
            out_buf.set_initial_string_offset(h_initial_str_offsets[idx]);
          }
        }
      }
    }
  }
  // Write the final offsets for list and string columns in a batched manner
  auto pinned_final_offsets = cudf::detail::make_pinned_vector(
    cudf::host_span<cudf::size_type const>{final_offsets}, _stream);
  auto pinned_out_buffers =
    cudf::detail::make_pinned_vector(cudf::host_span<cudf::size_type* const>{out_buffers}, _stream);
  write_final_offsets(pinned_final_offsets, pinned_out_buffers, _stream);

  // update null counts in the final column buffers
  for (size_t idx = 0; idx < subpass.pages.size(); idx++) {
    PageInfo* pi = &subpass.pages[idx];
    if (pi->flags & PAGEINFO_FLAGS_DICTIONARY) { continue; }
    ColumnChunkDesc* col               = &pass.chunks[pi->chunk_idx];
    input_column_info const& input_col = _input_columns[col->src_col_index];

    int const index             = pi->nesting_decode - page_nesting_decode.device_ptr();
    PageNestingDecodeInfo* pndi = &page_nesting_decode[index];

    auto* cols = &_output_buffers;
    for (size_t l_idx = 0; l_idx < input_col.nesting_depth(); l_idx++) {
      auto& out_buf = (*cols)[input_col.nesting[l_idx]];
      cols          = &out_buf.children;

      // if I wasn't the one who wrote out the validity bits, skip it
      if (chunk_nested_valids.host_ptr(chunk_offsets[pi->chunk_idx])[l_idx] == nullptr) {
        continue;
      }
      out_buf.null_count() += pndi[l_idx].null_count;
    }
  }

  _stream.sync();
}

reader_impl::reader_impl() : _stream{cudaStream_t{nullptr}}, _options{} {}

reader_impl::reader_impl(std::vector<std::unique_ptr<datasource>>&& sources,
                         std::vector<FileMetaData>&& parquet_metadatas,
                         parquet_reader_options const& options,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
  : reader_impl(0 /*chunk_read_limit*/,
                0 /*input_pass_read_limit*/,
                std::forward<std::vector<std::unique_ptr<cudf::io::datasource>>>(sources),
                std::forward<std::vector<FileMetaData>>(parquet_metadatas),
                options,
                stream,
                mr)
{
}

reader_impl::reader_impl(std::size_t chunk_read_limit,
                         std::size_t pass_read_limit,
                         std::vector<std::unique_ptr<datasource>>&& sources,
                         std::vector<FileMetaData>&& file_metadatas,
                         parquet_reader_options const& options,
                         cuda::stream_ref stream,
                         rmm::device_async_resource_ref mr)
  : _stream{std::move(stream)},
    _mr{std::move(mr)},
    _options{options.get_timestamp_type(),
             options.get_decimal_width(),
             options.get_skip_rows(),
             options.get_num_rows(),
             options.get_skip_bytes(),
             options.get_num_bytes(),
             options.get_row_groups(),
             options.is_enabled_use_jit_filter(),
             options.is_enabled_case_sensitive_names(),
             options.is_enabled_prepend_source_index_column(),
             options.is_enabled_prepend_row_index_column(),
             options.is_enabled_output_dict_columns()},
    _sources{std::move(sources)},
    _output_chunk_read_limit{chunk_read_limit},
    _input_pass_read_limit{pass_read_limit}
{
  // The direct parquet-dict → DICTIONARY32 transcode fast path only supports single-pass,
  // non-chunked reads.
  if (_options.output_dict_columns and (chunk_read_limit != 0 or pass_read_limit != 0)) {
    CUDF_LOG_WARN(
      "output_dict_columns: the direct parquet-dict transcode fast path is disabled for chunked / "
      "multi-pass reads (non-zero chunk_read_limit or pass_read_limit); falling back to encoding "
      "DICTIONARY32 columns at the output.");
  }

  // Open and parse the source dataset metadata
  CUDF_EXPECTS(file_metadatas.empty() or file_metadatas.size() == _sources.size(),
               "Encountered a mismatch in the number of provided data sources and metadatas");

  _metadata = file_metadatas.empty() ? std::make_unique<aggregate_reader_metadata>(
                                         _sources,
                                         options.is_enabled_use_arrow_schema(),
                                         has_cols_from_mismatched_sources(options))
                                     : std::make_unique<aggregate_reader_metadata>(
                                         std::forward<std::vector<FileMetaData>>(file_metadatas),
                                         options.is_enabled_use_arrow_schema(),
                                         has_cols_from_mismatched_sources(options));

  // Number of input sources
  _num_sources = _sources.size();

  // Strings may be returned as either string or categorical columns
  _strings_to_categorical = options.is_enabled_convert_strings_to_categories();

  // Binary columns can be read as binary or strings
  _reader_column_schema = options.get_column_schema();

  // Select only columns required by the options and filter
  auto select_column_names = get_column_projection(options);

  std::optional<std::vector<std::string>> filter_only_columns_names;
  auto const has_column_selection = options.get_column_names().has_value() or
                                    options.get_column_indices().has_value() or
                                    options.get_column_field_ids().has_value();
  if (options.get_filter().has_value() and has_column_selection) {
    // list, struct, dictionary are not supported by AST filter yet.
    // extract columns not present in get_column_names() & keep count to remove at end.
    filter_only_columns_names = get_column_names_in_expression(
      options.get_filter(), *select_column_names, options, _metadata->get_schema_tree());
    _num_filter_only_columns = filter_only_columns_names->size();
  }
  std::tie(_input_columns, _output_buffers, _output_column_schemas) = _metadata->select_columns(
    select_column_names, filter_only_columns_names, make_column_selection_options(options));

  // Save the states of the output buffers for reuse in `chunk_read()`.
  std::transform(
    _output_buffers.begin(),
    _output_buffers.end(),
    std::back_inserter(_output_buffers_template),
    [](auto const& buff) { return cudf::io::detail::inline_column_buffer::empty_like(buff); });

  // Save the name to reference converter to extract output filter AST in
  // `preprocess_file()` and `finalize_output()`
  table_metadata metadata;
  populate_metadata(metadata);
  _expr_conv =
    named_to_reference_converter(options.get_filter(), metadata, _options.case_sensitive_names);
}

void reader_impl::prepare_data(read_mode mode)
{
  // if we have not preprocessed at the whole-file level, do that now
  if (!_file_preprocessed) {
    // setup file level information
    // - read row group information
    // - setup information on (parquet) chunks
    // - compute schedule of input passes
    preprocess_file(mode);
  }

  // handle any chunking work (ratcheting through the subpasses and chunks within
  // our current pass) if in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) { handle_chunking(mode); }
}

void reader_impl::populate_metadata(table_metadata& out_metadata)
{
  // Return column names
  out_metadata.schema_info.resize(_output_buffers.size());
  for (size_t i = 0; i < _output_column_schemas.size(); i++) {
    auto const& schema               = _metadata->get_schema(_output_column_schemas[i]);
    out_metadata.schema_info[i].name = schema.name;
    out_metadata.schema_info[i].is_nullable =
      schema.repetition_type != FieldRepetitionType::REQUIRED;
  }

  // Return user metadata
  out_metadata.per_file_user_data = _metadata->get_key_value_metadata();
  out_metadata.user_data          = {out_metadata.per_file_user_data[0].begin(),
                                     out_metadata.per_file_user_data[0].end()};
}

void reader_impl::preprocess_chunk_strings(read_mode mode, row_range const& read_info)
{
  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  if (!(subpass.kernel_mask & STRINGS_MASK)) { return; }

  // we may need to (re)compute string sizes.
  // string sizes were computed in the page preprocess step for the subpass if we are doing an
  // output chunked read, but contains the size for all strings, not necessarily just the rows that
  // may have been selected for an output chunk. so, if we didn't compute the sizes at all, or
  // if we are actually doing a chunked subpass (reading anything other than all the rows in the
  // entire pass) we need to recompute
  bool full_string_sizes_computed = _output_chunk_read_limit > 0;
  bool const need_string_size_recompute =
    // if we haven't computed string sizes at all because we're not in the chunked read path
    (!full_string_sizes_computed) ||
    // if we are in the chunked path and we have chunked to something other than the initial row
    // bounds
    ((pass.skip_rows != read_info.skip_rows) || (pass.num_rows != read_info.num_rows)) ||
    // if the user has specified skip_rows / num_rows at all. i am explicitly not calling
    // uses_custom_row_bounds() here because it will return true at all times in the chunked
    // read case, which we don't want. if we are doing a chunked read, but no chunking has occurred,
    // the full size compute step is all we need.
    ((_options.num_rows.has_value() or _options.skip_rows != 0));

  if (need_string_size_recompute) {
    constexpr bool compute_all_string_sizes = false;
    compute_page_string_sizes_pass1(subpass.pages,
                                    pass.chunks,
                                    subpass_page_mask_span(),
                                    subpass.page_string_offset_indices,
                                    read_info.skip_rows,
                                    read_info.num_rows,
                                    subpass.kernel_mask,
                                    compute_all_string_sizes,
                                    _pass_itm_data->level_type_size,
                                    _stream);
  }
  compute_page_string_sizes_pass2(subpass.pages, pass.chunks, subpass.delta_temp_buf, _stream);
}

table_with_metadata reader_impl::read_chunk_internal(read_mode mode)
{
  CUDF_FUNC_RANGE();

  // If `_output_metadata` has been constructed, just copy it over.
  auto out_metadata = _output_metadata ? table_metadata{*_output_metadata} : table_metadata{};
  out_metadata.schema_info.resize(_output_buffers.size());

  // output cudf columns as determined by the top level schema
  auto out_columns = std::vector<std::unique_ptr<column>>{};
  out_columns.reserve(_output_buffers.size());

  // Copy number of total input row groups and number of surviving row groups from predicate
  // pushdown.
  out_metadata.num_input_row_groups = _file_itm_data.num_input_row_groups;
  // Copy the number surviving row groups from each predicate pushdown only if the filter has value
  if (_expr_conv.get_converted_expr().has_value()) {
    out_metadata.num_row_groups_after_stats_filter =
      _file_itm_data.surviving_row_groups.after_stats_filter;
    out_metadata.num_row_groups_after_bloom_filter =
      _file_itm_data.surviving_row_groups.after_bloom_filter;
  }

  // no work to do (this can happen on the first pass if we have no rows to read)
  if (!has_more_work()) {
    // Empty dataframe case: Simply initialize to a list of zeros
    out_metadata.num_rows_per_source =
      std::vector<size_t>(_file_itm_data.num_rows_per_source.size(), 0);

    // Finalize output
    return finalize_output(mode, out_metadata, out_columns);
  }

  auto& pass            = *_pass_itm_data;
  auto& subpass         = *pass.subpass;
  auto const& read_info = subpass.output_chunk_read_info[subpass.current_output_chunk];

  // If the caller asked for direct parquet-dict → DICTIONARY32 transcode, detect per-column
  // eligibility and mutate `_output_buffers` / `subpass.pages` before we allocate column buffers
  // or dispatch decode kernels. This has to happen before `preprocess_chunk_strings` /
  // `allocate_columns` because those branch on `subpass.kernel_mask` and on `out_buf.type`.
  prepare_dict_transcode(mode);

  // computes:
  // PageNestingInfo::batch_size for each level of nesting, for each page, taking row bounds into
  // account. PageInfo::skipped_values, which tells us where to start decoding in the input to
  // respect the user bounds. It is only necessary to do this second pass if uses_custom_row_bounds
  // is set (if the user has specified artificial bounds).
  if (uses_custom_row_bounds(mode)) {
    compute_page_sizes(subpass.pages,
                       pass.chunks,
                       subpass_page_mask_span(),
                       read_info.skip_rows,
                       read_info.num_rows,
                       false,  // num_rows is already computed
                       pass.level_type_size,
                       _stream);
  }

  // preprocess strings
  preprocess_chunk_strings(mode, read_info);

  // Allocate memory buffers for the output columns.
  allocate_columns(mode, read_info.skip_rows, read_info.num_rows);

  // Parse data into the output buffers.
  decode_page_data(mode, read_info.skip_rows, read_info.num_rows);

  // Create the final output cudf columns.
  for (size_t i = 0; i < _output_buffers.size(); ++i) {
    auto metadata           = _reader_column_schema.has_value()
                                ? std::make_optional<reader_column_schema>((*_reader_column_schema)[i])
                                : std::nullopt;
    auto const& schema      = _metadata->get_schema(_output_column_schemas[i]);
    auto const logical_type = schema.logical_type.value_or(LogicalType{});
    // FIXED_LEN_BYTE_ARRAY never read as string.
    // TODO: if we ever decide that the default reader behavior is to treat unannotated BINARY as
    // binary and not strings, this test needs to change.
    if (schema.type == Type::FIXED_LEN_BYTE_ARRAY and logical_type.type != LogicalType::DECIMAL) {
      metadata = std::make_optional<reader_column_schema>();
      metadata->set_convert_binary_to_strings(false);
      metadata->set_type_length(schema.type_length);
    }
    // Only construct `out_metadata` if `_output_metadata` has not been cached.
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info[i];
      out_columns.emplace_back(make_column(_output_buffers[i], &col_name, metadata, _stream));
    } else {
      out_columns.emplace_back(make_column(_output_buffers[i], nullptr, metadata, _stream));
    }
  }

  // For any columns that were selected for direct parquet-dict → DICTIONARY32 transcode in
  // `prepare_dict_transcode`, the entries in `out_columns` are currently INT32 indices columns.
  // Assemble them into DICTIONARY32 columns here by attaching per-chunk keys; concatenate
  // remaps indices to the unified keys child.
  assemble_dict_transcoded_columns(out_columns);

  out_columns =
    cudf::structs::detail::enforce_null_consistency(std::move(out_columns), _stream, _mr);

  // Compute the output number of rows per source
  if (mode == read_mode::CHUNKED_READ) {
    out_metadata.num_rows_per_source =
      calculate_output_num_rows_per_source(read_info.skip_rows, read_info.num_rows);
  } else {
    // Move is okay here as we are reading in one go.
    out_metadata.num_rows_per_source = std::move(_file_itm_data.num_rows_per_source);
  }

  // Add empty columns if needed. Filter output columns based on filter.
  return finalize_output(mode, out_metadata, out_columns);
}

std::vector<size_t> reader_impl::calculate_output_num_rows_per_source(size_t const chunk_start_row,
                                                                      size_t const chunk_num_rows)
{
  // Handle base cases.
  if (_file_itm_data.num_rows_per_source.size() == 0) {
    return {};
  } else if (_file_itm_data.num_rows_per_source.size() == 1) {
    return {chunk_num_rows};
  }

  std::vector<size_t> num_rows_per_source(_file_itm_data.num_rows_per_source.size(), 0);

  // Subtract global skip rows from the start_row as we took care of that when computing
  // _file_itm_data.num_rows_per_source
  auto const start_row = chunk_start_row - _file_itm_data.global_skip_rows;
  auto const end_row   = start_row + chunk_num_rows;
  CUDF_EXPECTS(start_row <= end_row and end_row <= _file_itm_data.global_num_rows,
               "Encountered invalid output chunk row bounds.");

  // Copy reference to a const local variable for better readability
  auto const& partial_sum_nrows_source = _file_itm_data.exclusive_sum_num_rows_per_source;

  // Binary search start_row and end_row in exclusive_sum_num_rows_per_source vector
  auto const start_iter =
    std::upper_bound(partial_sum_nrows_source.cbegin(), partial_sum_nrows_source.cend(), start_row);
  auto const end_iter = std::lower_bound(start_iter, partial_sum_nrows_source.cend(), end_row);

  // Compute the array offset index for both iterators
  auto const start_idx   = std::distance(partial_sum_nrows_source.cbegin(), start_iter);
  auto const end_idx     = std::distance(partial_sum_nrows_source.cbegin(), end_iter);
  auto const num_sources = static_cast<cudf::size_type>(partial_sum_nrows_source.size());
  CUDF_EXPECTS(start_idx <= end_idx and start_idx < num_sources and end_idx < num_sources,
               "Encountered out of range output table chunk row bounds");

  // If the entire chunk is from the same source file, then the count is simply num_rows
  if (start_idx == end_idx) {
    num_rows_per_source[start_idx] = chunk_num_rows;
  } else {
    // Compute the number of rows from the first source file
    num_rows_per_source[start_idx] = partial_sum_nrows_source[start_idx] - start_row;
    // Compute the number of rows from the last source file
    num_rows_per_source[end_idx] = end_row - partial_sum_nrows_source[end_idx - 1];
    // Simply copy the number of rows for each source in range: (start_idx, end_idx)
    std::copy(_file_itm_data.num_rows_per_source.cbegin() + start_idx + 1,
              _file_itm_data.num_rows_per_source.cbegin() + end_idx,
              num_rows_per_source.begin() + start_idx + 1);
  }

  return num_rows_per_source;
}

std::optional<std::vector<std::string>> reader_impl::get_column_projection(
  parquet_reader_options const& options) const
{
  auto const ignore_missing_columns = ignore_missing_columns_policy(options);

  auto const has_column_names     = options.get_column_names().has_value();
  auto const has_column_indices   = options.get_column_indices().has_value();
  auto const has_column_field_ids = options.get_column_field_ids().has_value();

  if (has_column_names) { return options.get_column_names(); }

  if (has_column_indices) {
    std::vector<std::string> col_names;
    auto const& top_level_schema_indices = _metadata->get_schema(0).children_idx;
    for (auto const index : options.get_column_indices().value()) {
      auto const is_valid_index =
        std::cmp_greater_equal(index, 0) and std::cmp_less(index, top_level_schema_indices.size());
      CUDF_EXPECTS(ignore_missing_columns or is_valid_index,
                   "Encountered an invalid col index in the top-level column selection");
      if (is_valid_index) {
        col_names.emplace_back(_metadata->get_schema(top_level_schema_indices[index]).name);
      }
    }
    return std::make_optional(std::move(col_names));
  }

  if (has_column_field_ids) {
    std::vector<std::string> col_names;
    auto const& schema_tree = _metadata->get_schema_tree();
    for (auto const field_id : options.get_column_field_ids().value()) {
      auto const schema_iter =
        std::find_if(schema_tree.cbegin() + 1, schema_tree.cend(), [field_id](auto const& schema) {
          return schema.field_id.has_value() and schema.field_id.value() == field_id;
        });
      CUDF_EXPECTS(ignore_missing_columns or schema_iter != schema_tree.end(),
                   "Encountered a non-existent Parquet field ID in selected columns",
                   std::invalid_argument);
      if (schema_iter != schema_tree.end()) {
        auto const schema_idx = static_cast<int>(std::distance(schema_tree.cbegin(), schema_iter));
        col_names.emplace_back(column_path_from_index(schema_tree, schema_idx));
      }
    }
    return std::make_optional(std::move(col_names));
  }

  // No column selection specified. Return nullopt indicating all columns to be selected.
  return std::nullopt;
}

void reader_impl::apply_decimal_width_cast(std::vector<std::unique_ptr<column>>& out_columns)
{
  // TODO: Instead of casting the columns after the read is done, we should
  // be able to decode data directly into the target decimal type buffer.
  if (_options.decimal_width == type_id::EMPTY) { return; }
  for (auto& col : out_columns) {
    auto const col_type = col->type();
    if (cudf::is_fixed_point(col_type) && col_type.id() != _options.decimal_width) {
      col =
        cudf::cast(col->view(), data_type{_options.decimal_width, col_type.scale()}, _stream, _mr);
    }
  }
}

column_selection_options reader_impl::make_column_selection_options(
  parquet_reader_options const& options) const
{
  auto const selection_mode = [&]() {
    if (options.get_column_names().has_value()) {
      return column_selection_mode::BY_NAME;
    } else if (options.get_column_indices().has_value()) {
      return column_selection_mode::BY_INDEX;
    } else if (options.get_column_field_ids().has_value()) {
      return column_selection_mode::BY_FIELD_ID;
    } else {
      return column_selection_mode::NONE;
    }
  }();

  return column_selection_options{
    .selection_mode         = selection_mode,
    .include_index          = options.is_enabled_use_pandas_metadata(),
    .strings_to_categorical = options.is_enabled_convert_strings_to_categories(),
    .ignore_missing_columns = ignore_missing_columns_policy(options),
    .timestamp_type_id      = options.get_timestamp_type().id(),
    .decimal_type_id        = options.get_decimal_width(),
    .case_sensitive_names   = options.is_enabled_case_sensitive_names()};
}

table_with_metadata reader_impl::finalize_output(read_mode mode,
                                                 table_metadata& out_metadata,
                                                 std::vector<std::unique_ptr<column>>& out_columns)
{
  CUDF_FUNC_RANGE();

  // Create empty columns as needed (this can happen if we've ended up with no actual data to read)
  for (size_t i = out_columns.size(); i < _output_buffers.size(); ++i) {
    if (!_output_metadata) {
      column_name_info& col_name = out_metadata.schema_info[i];
      out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], &col_name, _stream, _mr));
    } else {
      out_columns.emplace_back(io::detail::empty_like(_output_buffers[i], nullptr, _stream, _mr));
    }
  }

  apply_decimal_width_cast(out_columns);

  // Encode any remaining flat STRING columns to DICTIONARY32 via a post-hoc
  // `dictionary::detail::encode`: columns not produced by the direct transcode fast path, or all of
  // them when the fast path was disabled (e.g. under a filter). This is applied to the FINAL output
  // table below -- after any filter is evaluated -- so the filter still operates on STRING columns
  // and the dictionary is built over only the surviving rows.
  auto const encode_output_dict_columns =
    [&](std::unique_ptr<table> tbl) -> std::unique_ptr<table> {
    if (not _options.output_dict_columns) { return tbl; }
    auto columns = tbl->release();
    for (auto& col : columns) {
      if (col and col->type().id() == type_id::STRING) {
        col =
          cudf::dictionary::detail::encode(col->view(), data_type{type_id::INT32}, _stream, _mr);
      }
    }
    return std::make_unique<table>(std::move(columns));
  };

  if (!_output_metadata) {
    populate_metadata(out_metadata);
    // Finally, save the output table metadata into `_output_metadata` for reuse next time.
    _output_metadata = std::make_unique<table_metadata>(out_metadata);
  }

  // Row-range of the current output chunk relative to the current row group selection.
  auto const read_info =
    (_file_itm_data._current_input_pass < _file_itm_data.num_passes())
      ? _pass_itm_data->subpass
          ->output_chunk_read_info[_pass_itm_data->subpass->current_output_chunk]
      : row_range{0, 0};

  // advance output chunk/subpass/pass info for non-empty tables if and only if we are in bounds
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes()) {
    auto& pass    = *_pass_itm_data;
    auto& subpass = *pass.subpass;
    subpass.current_output_chunk++;
  }

  // increment the output chunk count
  _file_itm_data._output_chunk_count++;

  // Prepend the source and row index columns if requested
  {
    if (_options.prepend_row_index_column) {
      out_columns.emplace(out_columns.begin(),
                          synthesize_row_index_column(read_info, _stream, _mr));
      out_metadata.schema_info.emplace(out_metadata.schema_info.begin(),
                                       column_name_info{.name = "row_index", .is_nullable = false});
    }
    if (_options.prepend_source_index_column) {
      out_columns.emplace(
        out_columns.begin(),
        synthesize_source_index_column(out_metadata.num_rows_per_source, _stream, _mr));
      out_metadata.schema_info.emplace(
        out_metadata.schema_info.begin(),
        column_name_info{.name = "source_index", .is_nullable = false});
    }
  }

  // Offset column references in `_expr_conv` by the number of prepended columns
  auto const final_filter      = compute_offset_filter();
  auto const final_filter_expr = final_filter.get_converted_expr();

  // check if the output filter AST expression (= _expr_conv.get_converted_expr()) exists
  if (final_filter_expr.has_value()) {
    auto read_table         = std::make_unique<table>(std::move(out_columns));
    auto counting_it        = cuda::counting_iterator<std::size_t>{0};
    auto const output_count = read_table->num_columns() - _num_filter_only_columns;
    auto only_output        = read_table->select(counting_it, counting_it + output_count);

    if (_num_filter_only_columns > 0) { out_metadata.schema_info.resize(output_count); }

    // Clear the number of rows per source as it is not valid after filtering
    out_metadata.num_rows_per_source.clear();

    bool use_jit = cudf::get_context().use_jit() || _options.use_jit_filter;

    if (!use_jit) {
      auto predicate =
        cudf::detail::compute_column(*read_table, final_filter_expr.value().get(), _stream, _mr);
      CUDF_EXPECTS(predicate->view().type().id() == type_id::BOOL8,
                   "Predicate filter should return a boolean");
      // Exclude columns present in filter only in output
      auto output_table = cudf::detail::apply_mask(
        only_output, *predicate, cudf::detail::mask_type::RETENTION, _stream, _mr);
      return {encode_output_dict_columns(std::move(output_table)), std::move(out_metadata)};
    } else {
      auto output_table = cudf::filter(
        read_table->view(), final_filter_expr.value().get(), only_output, _stream, _mr);

      return {encode_output_dict_columns(std::move(output_table)), std::move(out_metadata)};
    }
  }
  return {encode_output_dict_columns(std::make_unique<table>(std::move(out_columns))),
          std::move(out_metadata)};
}

table_with_metadata reader_impl::read()
{
  CUDF_EXPECTS(_output_chunk_read_limit == 0,
               "Reading the whole file must not have non-zero byte_limit.");
  CUDF_EXPECTS(!_options.num_rows.has_value() ||
                 _options.num_rows.value() <= std::numeric_limits<size_type>::max(),
               "Requested number of rows to read exceeds column size limit",
               std::overflow_error);
  prepare_data(read_mode::READ_ALL);
  return read_chunk_internal(read_mode::READ_ALL);
}

table_with_metadata reader_impl::read_chunk()
{
  // Reset the output buffers to their original states (right after reader construction).
  // Don't need to do it if we read the file all at once.
  if (_file_itm_data._current_input_pass < _file_itm_data.num_passes() and
      not is_first_output_chunk()) {
    _output_buffers.resize(0);
    for (auto const& buff : _output_buffers_template) {
      _output_buffers.emplace_back(cudf::io::detail::inline_column_buffer::empty_like(buff));
    }
  }

  prepare_data(read_mode::CHUNKED_READ);
  return read_chunk_internal(read_mode::CHUNKED_READ);
}

bool reader_impl::has_next()
{
  prepare_data(read_mode::CHUNKED_READ);

  // current_input_pass will only be incremented to be == num_passes after
  // the last chunk in the last subpass in the last pass has been returned
  // if not has_more_work then check if this is the first pass in an empty
  // table and return true so it could be read once.
  return has_more_work() or is_first_output_chunk();
}

void reader_impl::update_output_nullmasks_for_pruned_pages(cudf::host_span<bool const> page_mask,
                                                           size_t skip_rows,
                                                           size_t num_rows)
{
  auto const& subpass    = _pass_itm_data->subpass;
  auto const& pages      = subpass->pages;
  auto const& chunks     = _pass_itm_data->chunks;
  auto const num_columns = _input_columns.size();

  CUDF_EXPECTS(pages.size() == page_mask.size(), "Page mask size mismatch");

  // Return early if page mask is empty or all pages are required
  if (page_mask.empty() or std::all_of(page_mask.begin(), page_mask.end(), std::identity{})) {
    return;
  }

  auto page_and_mask_begin =
    cuda::make_zip_iterator(cuda::std::make_tuple(pages.host_begin(), page_mask.begin()));

  auto null_masks = std::vector<bitmask_type*>{};
  auto begin_bits = std::vector<cudf::size_type>{};
  auto end_bits   = std::vector<cudf::size_type>{};

  std::for_each(
    page_and_mask_begin, page_and_mask_begin + pages.size(), [&](auto const& page_and_mask_pair) {
      // Return early if the page is valid - Note: dictionary pages are always valid
      if (cuda::std::get<1>(page_and_mask_pair)) { return; }

      // Get the current page
      auto const& page = cuda::std::get<0>(page_and_mask_pair);

      // Table row bounds
      auto const table_start_row = skip_rows;
      auto const table_end_row   = skip_rows + num_rows;

      // Current page row bounds
      auto const chunk_idx      = page.chunk_idx;
      auto const page_start_row = chunks[chunk_idx].start_row + page.chunk_row;
      auto const page_end_row   = page_start_row + page.num_rows;

      auto const is_page_out_of_bounds =
        (page_start_row < table_start_row and page_end_row < table_start_row) or
        (page_start_row > table_end_row and page_end_row > table_end_row);
      if (is_page_out_of_bounds) { return; }

      // The page is either completely or partially within the table row bounds
      auto const [start_row, end_row] = [&]() -> std::pair<size_type, size_type> {
        // Page is completely within the table row bounds
        if (page_start_row >= table_start_row and page_end_row <= table_end_row) {
          return {page_start_row - table_start_row, page_end_row - table_start_row};
        }
        // Page starts earlier but ends within the table row bounds
        else if (page_start_row < table_start_row) {
          return {0, page_end_row - table_start_row};
        }
        // Page starts within the table row bounds but ends after
        else {
          return {page_start_row - table_start_row, num_rows};
        }
      }();

      // Make sure the row bounds are valid
      CUDF_EXPECTS(std::cmp_greater_equal(start_row, 0) and std::cmp_less_equal(end_row, num_rows),
                   "Invalid row bounds");

      auto& input_col      = _input_columns[chunk_idx % num_columns];
      auto const max_depth = input_col.nesting_depth();
      auto* cols           = &_output_buffers;

      for (size_t l_idx = 0; l_idx < max_depth; l_idx++) {
        auto& out_buf = (*cols)[input_col.nesting[l_idx]];
        cols          = &out_buf.children;
        if (out_buf.user_data & PARQUET_COLUMN_BUFFER_FLAG_HAS_LIST_PARENT) { continue; }
        // Add the nullmask and bit bounds to corresponding lists
        null_masks.emplace_back(out_buf.null_mask());
        begin_bits.emplace_back(start_row);
        end_bits.emplace_back(end_row);

        // Increment the null count by the number of rows in this page
        out_buf.null_count() += page.num_rows;
      }
    });

  // Min number of nullmasks to use bulk update optimally
  constexpr auto min_nullmasks_for_bulk_update = 32;

  // Use a bounce buffer to avoid pageable copies
  auto pinned_null_masks =
    cudf::detail::make_pinned_vector(cudf::host_span<bitmask_type* const>{null_masks}, _stream);
  auto const pinned_begin_bits =
    cudf::detail::make_pinned_vector(cudf::host_span<cudf::size_type const>{begin_bits}, _stream);
  auto const pinned_end_bits =
    cudf::detail::make_pinned_vector(cudf::host_span<cudf::size_type const>{end_bits}, _stream);

  // Bulk update the nullmasks if the number of pages is above the threshold
  if (pinned_null_masks.size() >= min_nullmasks_for_bulk_update) {
    auto pinned_valids = cudf::detail::make_pinned_vector<bool>(pinned_null_masks.size(), _stream);
    std::fill(pinned_valids.begin(), pinned_valids.end(), false);
    cudf::set_null_masks_safe(
      pinned_null_masks, pinned_begin_bits, pinned_end_bits, pinned_valids, _stream);
  }
  // Otherwise, update the nullmasks in a loop
  else {
    auto nullmask_iter = cuda::make_zip_iterator(cuda::std::make_tuple(
      pinned_null_masks.begin(), pinned_begin_bits.begin(), pinned_end_bits.begin()));
    std::for_each(
      nullmask_iter, nullmask_iter + pinned_null_masks.size(), [&](auto const& nullmask_tuple) {
        cudf::set_null_mask(cuda::std::get<0>(nullmask_tuple),
                            cuda::std::get<1>(nullmask_tuple),
                            cuda::std::get<2>(nullmask_tuple),
                            false,
                            _stream);
      });
  }
}

namespace {
parquet_column_schema walk_schema(aggregate_reader_metadata const* mt, int idx)
{
  SchemaElement const& sch = mt->get_schema(idx);
  std::vector<parquet_column_schema> children;
  for (auto const& child_idx : sch.children_idx) {
    children.push_back(walk_schema(mt, child_idx));
  }

  auto const type_id   = to_type_id(sch, false, type_id::EMPTY, type_id::EMPTY);
  auto const cudf_type = to_data_type(type_id, sch);

  return parquet_column_schema{
    sch.name, static_cast<parquet::Type>(sch.type), std::move(children), cudf_type};
}
}  // namespace

parquet_metadata read_parquet_metadata(host_span<std::unique_ptr<datasource> const> sources)
{
  // Do not use arrow schema when only reading the parquet footer metadata.
  constexpr auto use_arrow_schema = false;

  // Do not select any columns when only reading the parquet metadata.
  constexpr auto has_column_projection = false;

  // Do not read page indexes as it's not used to construct `parquet_metadata`.
  constexpr auto read_page_indexes = false;

  // Open and parse the source dataset metadata
  auto metadata =
    aggregate_reader_metadata(sources, use_arrow_schema, has_column_projection, read_page_indexes);

  return parquet_metadata{parquet_schema{walk_schema(&metadata, 0)},
                          metadata.get_num_rows(),
                          metadata.get_num_row_groups(),
                          metadata.get_num_row_groups_per_file(),
                          metadata.get_key_value_metadata()[0],
                          metadata.get_rowgroup_metadata(),
                          metadata.get_column_chunk_metadata()};
}

std::vector<parquet::FileMetaData> read_parquet_footers(
  std::span<std::unique_ptr<datasource> const> sources)
{
  // Do not use arrow schema when only reading the parquet metadata.
  constexpr auto use_arrow_schema = false;

  // Do not select any columns when only reading the parquet metadata.
  constexpr auto has_column_projection = false;

  // Read page indexes if available here since we will want to reuse the raw metadata for later use.
  constexpr auto read_page_indexes = true;

  // Parse the source dataset metadata
  return aggregate_reader_metadata(
           host_span<std::unique_ptr<datasource> const>{sources.data(), sources.size()},
           use_arrow_schema,
           has_column_projection,
           read_page_indexes)
    .get_parquet_metadatas();
}

}  // namespace cudf::io::parquet::detail

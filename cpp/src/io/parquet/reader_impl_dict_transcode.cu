/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/reduction/detail/distinct_count.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>

namespace cudf::io::parquet::detail {

namespace {

/**
 * @brief Whether a column chunk is a plain BYTE_ARRAY string chunk.
 *
 * Narrows `is_string_col` (parquet_gpu.hpp) to BYTE_ARRAY only: `is_string_col` also accepts
 * FIXED_LEN_BYTE_ARRAY, which is typically a binary payload and is excluded from transcode. This is
 * a string-type classifier -- one of several inputs to eligibility, not the eligibility decision.
 *
 * @param chunk The column chunk descriptor to classify
 * @return True if the chunk is a plain (non-categorical, non-decimal) BYTE_ARRAY string chunk
 */
[[nodiscard]] bool is_byte_array_string_chunk(ColumnChunkDesc const& chunk)
{
  return is_string_col(chunk) and chunk.physical_type == Type::BYTE_ARRAY;
}

/**
 * @brief Per-input-column eligibility flags for Parquet-dict → DICTIONARY32 transcode.
 *
 * Each column must satisfy all of these conditions to be eligible for direct transcode.
 */
struct column_eligibility {
  bool has_string_buffer = false;  ///< Output buffer is currently typed as STRING
  bool has_any_chunk     = false;  ///< At least one chunk was seen for this column
  bool all_chunks_string = true;   ///< Every chunk is a flat BYTE_ARRAY string chunk with a dict
  bool all_pages_dict    = true;   ///< Every data page uses a dictionary encoding

  /**
   * @brief Whether the column satisfies every transcode-eligibility condition.
   *
   * @return True if the column is eligible for direct DICTIONARY32 transcode
   */
  [[nodiscard]] bool is_eligible() const
  {
    return has_string_buffer and has_any_chunk and all_chunks_string and all_pages_dict;
  }
};

/**
 * @brief Fold a single chunk's properties into its column's eligibility state.
 *
 * @param e The per-column eligibility state to update in place
 * @param chunk The column chunk descriptor to classify
 */
void update_from_chunk(column_eligibility& e, ColumnChunkDesc const& chunk)
{
  e.has_any_chunk = true;
  if (chunk.max_nesting_depth != 1 or chunk.max_level[level_type::REPETITION] != 0 or
      not is_byte_array_string_chunk(chunk) or chunk.num_dict_pages < 1) {
    e.all_chunks_string = false;
  }
}

/**
 * @brief Compute per-input-column eligibility for Parquet-dict → DICTIONARY32 transcode.
 *
 * A column is eligible iff
 *  - the corresponding output buffer is currently typed as STRING (i.e. a flat string column),
 *  - every chunk of that column is a BYTE_ARRAY string chunk with a dictionary page,
 *  - every data page of every chunk of that column uses DICTIONARY encoding,
 *  - the chunk has a flat (non-list, non-nested) schema.
 *
 * @param pass The pass intermediate data holding host-side chunks and pages
 * @param input_columns The reader's input column descriptors
 * @param output_buffers The output column buffers (used to detect flat STRING columns)
 * @return A vector of per-input-column eligibility records, indexed by input column
 */
[[nodiscard]] std::vector<column_eligibility> compute_dict_transcode_eligibility(
  pass_intermediate_data const& pass,
  std::vector<input_column_info> const& input_columns,
  std::vector<cudf::io::detail::inline_column_buffer> const& output_buffers)
{
  auto const num_input_cols = input_columns.size();
  std::vector<column_eligibility> elig(num_input_cols);

  // Mark columns whose output buffer is a flat string column.
  std::transform(
    input_columns.begin(), input_columns.end(), elig.begin(), [&](input_column_info const& col) {
      column_eligibility e{};
      e.has_string_buffer =
        col.nesting_depth() == 1 and output_buffers[col.nesting[0]].type.id() == type_id::STRING;
      return e;
    });

  // Fold per-chunk info into the per-column eligibility flags.
  for (auto const& chunk : pass.chunks) {
    auto const col_idx = chunk.src_col_index;
    update_from_chunk(elig[col_idx], chunk);
  }

  // Any non-dictionary data-page encoding disqualifies the whole column. Dictionary pages
  // themselves (PAGEINFO_FLAGS_DICTIONARY) are skipped since they are not data pages.
  for (auto const& page : pass.pages) {
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) { continue; }
    auto const chunk_idx = page.chunk_idx;
    auto const col_idx   = pass.chunks[chunk_idx].src_col_index;
    if (not is_dictionary_encoding(page.encoding)) { elig[col_idx].all_pages_dict = false; }
  }

  return elig;
}

/**
 * @brief Build a STRING keys column from a chunk's dictionary entries.
 *
 * @param begin Pointer to the first `string_index_pair` entry for this chunk's dictionary
 * @param entry_count Number of dictionary entries (keys) for this chunk
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's memory
 * @return A STRING column holding this chunk's dictionary keys (empty if `entry_count <= 0`)
 */
[[nodiscard]] std::unique_ptr<column> make_keys_column_from_index_pairs(
  string_index_pair const* begin,
  size_type entry_count,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  if (entry_count <= 0) { return cudf::make_empty_column(data_type{type_id::STRING}); }
  return cudf::strings::detail::make_strings_column(begin, begin + entry_count, stream, mr);
}

/**
 * @brief Gathers one column's stacked keys out of the shared `pass.str_dict_index` buffer.
 *
 * Maps a stacked-key position in `[0, total_keys)` to its `string_index_pair`.
 * Fed through a counting-transform iterator.
 */
struct stacked_key_gather_fn {
  string_index_pair const* str_dict_index;
  cudf::device_span<size_t const> key_counts_prefix;  ///< size num_chunks + 1
  cudf::device_span<size_t const> key_base_offsets;   ///< size num_chunks

  __device__ string_index_pair operator()(size_type stacked_pos) const
  {
    auto const it = thrust::upper_bound(
      thrust::seq, key_counts_prefix.begin(), key_counts_prefix.end(), stacked_pos);
    auto const k     = static_cast<size_type>(it - key_counts_prefix.begin() - 1);
    auto const local = stacked_pos - key_counts_prefix[k];
    return str_dict_index[key_base_offsets[k] + local];
  }
};

}  // namespace

reader_impl::dict_transcode_plan reader_impl::prepare_dict_transcode(read_mode mode)
{
  CUDF_FUNC_RANGE();

  // Direct transcode requires a whole, unfiltered column in one subpass. Other reads keep the
  // materialize-then-encode fallback in finalize_output.
  if (not _options.output_dict_columns or _output_chunk_read_limit != 0 or
      _input_pass_read_limit != 0 or uses_custom_row_bounds(mode) or
      _expr_conv.get_converted_expr().has_value()) {
    return {};
  }

  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;
  if (pass.chunks.empty() or subpass.pages.size() == 0) { return {}; }

  auto const eligibility =
    compute_dict_transcode_eligibility(pass, _input_columns, _output_buffers);
  std::vector<bool> selected(_input_columns.size(), false);
  for (size_t page_idx = 0; page_idx < subpass.pages.size(); ++page_idx) {
    auto const& page = subpass.pages[page_idx];
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0 or
        page.kernel_mask != decode_kernel_mask::STRING_DICT) {
      continue;
    }
    auto const col = pass.chunks[page.chunk_idx].src_col_index;
    selected[col]  = eligibility[col].is_eligible();
  }
  if (std::none_of(selected.begin(), selected.end(), [](bool value) { return value; })) {
    return {};
  }

  // Gather host layout information once, before publishing any output-type or page-mask changes.
  std::vector<size_type> chunk_key_counts(pass.chunks.size(), 0);
  for (auto const& page : pass.pages) {
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) == 0) { continue; }
    auto const chunk_idx = page.chunk_idx;
    if (chunk_idx < 0 or static_cast<size_t>(chunk_idx) >= pass.chunks.size()) { continue; }
    if (pass.chunks[chunk_idx].dict_page == nullptr) { continue; }
    chunk_key_counts[chunk_idx] = page.num_input_values;
  }
  std::vector<std::vector<size_t>> chunks_by_column(_input_columns.size());
  for (size_t chunk = 0; chunk < pass.chunks.size(); ++chunk) {
    auto const col = pass.chunks[chunk].src_col_index;
    if (selected[col]) { chunks_by_column[col].push_back(chunk); }
  }

  dict_transcode_plan plan;
  std::vector<size_t> gather_metadata;
  for (size_t col = 0; col < selected.size(); ++col) {
    if (not selected[col]) { continue; }
    dict_transcode_column column_plan{};
    column_plan.output_column = static_cast<size_t>(_input_columns[col].nesting[0]);
    column_plan.chunks        = std::move(chunks_by_column[col]);
    column_plan.contiguous    = true;
    column_plan.key_counts_prefix.reserve(column_plan.chunks.size() + 1);
    column_plan.key_counts_prefix.push_back(0);
    std::vector<size_t> key_base_offsets(column_plan.chunks.size(), 0);
    size_type total_keys = 0;
    for (size_t k = 0; k < column_plan.chunks.size(); ++k) {
      auto const chunk_idx = column_plan.chunks[k];
      auto const count     = chunk_key_counts[chunk_idx];
      CUDF_EXPECTS(count >= 0 and count <= std::numeric_limits<size_type>::max() - total_keys,
                   "Dictionary keys exceed the column size limit");
      if (count > 0) {
        auto const offset =
          static_cast<size_t>(pass.chunks[chunk_idx].str_dict_index - pass.str_dict_index.data());
        key_base_offsets[k] = offset;
        if (total_keys == 0) { column_plan.key_base_offset = offset; }
        column_plan.contiguous =
          column_plan.contiguous and offset == column_plan.key_base_offset + total_keys;
      }
      total_keys += count;
      column_plan.key_counts_prefix.push_back(total_keys);
    }
    // Empty chunks need no descriptor pointer. Repeated prefixes skip them in the strided gather.
    // Contiguous columns do not need any device gather metadata.
    if (not column_plan.contiguous) {
      column_plan.gather_offset = gather_metadata.size();
      gather_metadata.insert(gather_metadata.end(),
                             column_plan.key_counts_prefix.begin(),
                             column_plan.key_counts_prefix.end());
      gather_metadata.insert(
        gather_metadata.end(), key_base_offsets.begin(), key_base_offsets.end());
    }
    plan.columns.push_back(std::move(column_plan));
  }

  // Allocate both mirrors up front. They remain owned by the plan through the decode-stream join.
  // Pack prefixes and descriptor offsets for every strided column into one metadata upload.
  plan.gather_metadata = cudf::detail::hostdevice_vector<size_t>(gather_metadata.size(), _stream);
  std::copy(gather_metadata.begin(), gather_metadata.end(), plan.gather_metadata.begin());
  plan.chunk_index_maps =
    cudf::detail::hostdevice_vector<int32_t const*>(pass.chunks.size(), _stream);
  std::fill(plan.chunk_index_maps.begin(), plan.chunk_index_maps.end(), nullptr);
  if (not plan.gather_metadata.empty()) { plan.gather_metadata.host_to_device_async(_stream); }

  for (auto const& column_plan : plan.columns) {
    _output_buffers[column_plan.output_column].type = data_type{type_id::INT32};
  }
  for (size_t page_idx = 0; page_idx < subpass.pages.size(); ++page_idx) {
    auto& page = subpass.pages[page_idx];
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) == 0 and
        page.kernel_mask == decode_kernel_mask::STRING_DICT and
        selected[pass.chunks[page.chunk_idx].src_col_index]) {
      page.kernel_mask = decode_kernel_mask::DICT_INT32;
    }
  }
  subpass.pages.host_to_device_async(_stream);
  subpass.kernel_mask = std::transform_reduce(
    subpass.pages.host_begin(),
    subpass.pages.host_end(),
    uint32_t{0},
    std::bit_or<>{},
    [](PageInfo const& page) { return static_cast<uint32_t>(page.kernel_mask); });
  return plan;
}

void reader_impl::prepare_dict_transcode_keys(dict_transcode_plan& plan)
{
  CUDF_FUNC_RANGE();
  if (plan.columns.empty()) { return; }

  auto const& pass = *_pass_itm_data;
  for (auto& column_plan : plan.columns) {
    auto const total_keys = column_plan.key_counts_prefix.back();
    auto const num_chunks = column_plan.chunks.size();
    std::unique_ptr<column> stacked_keys;
    if (num_chunks == 1) {
      stacked_keys = make_keys_column_from_index_pairs(
        pass.chunks[column_plan.chunks.front()].str_dict_index, total_keys, _stream, _mr);
      auto const num_distinct_keys = cudf::detail::distinct_count(
        stacked_keys->view(), null_policy::INCLUDE, nan_policy::NAN_IS_VALID, _stream);
      if (num_distinct_keys == total_keys) {
        column_plan.keys = std::move(stacked_keys);
        continue;  // A null map pointer preserves this chunk's local IDs.
      }
      // Reuse the already materialized keys when a single dictionary contains duplicates.
    } else if (column_plan.contiguous) {
      // Avoid pointer arithmetic on an empty descriptor allocation.
      auto const* begin =
        total_keys == 0 ? nullptr : pass.str_dict_index.data() + column_plan.key_base_offset;
      stacked_keys = make_keys_column_from_index_pairs(
        begin, total_keys, _stream, get_current_device_resource_ref());
    } else {
      auto const metadata   = cudf::device_span<size_t const>{plan.gather_metadata.device_ptr(),
                                                              plan.gather_metadata.size()};
      auto const keys_begin = cudf::detail::make_counting_transform_iterator(
        size_type{0},
        stacked_key_gather_fn{
          pass.str_dict_index.data(),
          metadata.subspan(column_plan.gather_offset, num_chunks + 1),
          metadata.subspan(column_plan.gather_offset + num_chunks + 1, num_chunks)});
      stacked_keys = cudf::strings::detail::make_strings_column(
        keys_begin, keys_begin + total_keys, _stream, get_current_device_resource_ref());
    }

    auto encoded = cudf::dictionary::detail::encode(
      stacked_keys->view(), data_type{type_id::INT32}, _stream, _mr);
    auto contents = encoded->release();
    column_plan.index_map =
      std::move(contents.children[dictionary_column_view::indices_column_index]);
    column_plan.keys = std::move(contents.children[dictionary_column_view::keys_column_index]);

    auto const* map = column_plan.index_map->view().data<int32_t>();
    for (size_t k = 0; k < num_chunks; ++k) {
      auto const first = column_plan.key_counts_prefix[k];
      if (first != column_plan.key_counts_prefix[k + 1]) {
        plan.chunk_index_maps[column_plan.chunks[k]] = map + first;
      }
    }
  }
  // Publish the complete pointer table once. Decode forks from this stream after the upload,
  // and the plan keeps the host mirror, device table, and map owners alive until all streams join.
  plan.chunk_index_maps.host_to_device_async(_stream);
}

void reader_impl::assemble_dict_transcoded_columns(
  std::vector<std::unique_ptr<column>>& out_columns, dict_transcode_plan& plan)
{
  CUDF_FUNC_RANGE();
  for (auto& column_plan : plan.columns) {
    auto& indices = out_columns[column_plan.output_column];
    CUDF_EXPECTS(
      column_plan.keys != nullptr and indices != nullptr and indices->type().id() == type_id::INT32,
      "Expected prepared keys and INT32 indices for dictionary transcode");
    indices =
      cudf::make_dictionary_column(std::move(column_plan.keys), std::move(indices), _stream, _mr);
  }
}

}  // namespace cudf::io::parquet::detail

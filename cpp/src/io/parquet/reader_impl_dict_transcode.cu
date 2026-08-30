/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/concatenate.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/batched_memset.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/dictionary/detail/encode.hpp>
#include <cudf/dictionary/dictionary_factories.hpp>
#include <cudf/reduction/detail/distinct_count.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda/iterator>

#include <algorithm>
#include <functional>
#include <iterator>
#include <numeric>
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
 * @brief Whether a column chunk is a flat fixed-width chunk whose decode is a plain copy of the
 * physical values (no decimal, timestamp-unit, or width conversion), so its dictionary page holds
 * the output values verbatim.
 *
 * @param chunk The column chunk descriptor to classify
 * @param out_type The current output buffer type of the chunk's column
 * @return True if the chunk's dictionary page entries are `out_type` values as stored
 */
[[nodiscard]] bool is_fixed_width_dict_chunk(ColumnChunkDesc const& chunk, data_type out_type)
{
  if (chunk.physical_type != Type::INT32 and chunk.physical_type != Type::INT64) { return false; }
  if (chunk.logical_type.has_value() and chunk.logical_type->type == LogicalType::DECIMAL) {
    return false;
  }
  // a non-zero clock rate means the decode rescales timestamp values
  if (chunk.ts_clock_rate != 0) { return false; }
  if (not cudf::is_fixed_width(out_type)) { return false; }
  auto const physical_size = chunk.physical_type == Type::INT32 ? std::size_t{4} : std::size_t{8};
  return cudf::size_of(out_type) == physical_size;
}

/**
 * @brief Per-input-column eligibility flags for Parquet-dict → DICTIONARY32 transcode.
 *
 * Each column must satisfy all of these conditions to be eligible for direct transcode.
 */
struct column_eligibility {
  data_type key_type{type_id::EMPTY};    ///< Logical output type; the DICTIONARY32 keys type
  bool has_transcodable_buffer = false;  ///< Output buffer is a flat STRING or fixed-width column
  bool has_any_chunk           = false;  ///< At least one chunk was seen for this column
  bool all_chunks_eligible = true;  ///< Every chunk is a flat dictionary chunk of the buffer type
  bool all_pages_dict      = true;  ///< Every data page uses a dictionary encoding

  /**
   * @brief Whether the column satisfies every transcode-eligibility condition.
   *
   * @return True if the column is eligible for direct DICTIONARY32 transcode
   */
  [[nodiscard]] bool is_eligible() const
  {
    return has_transcodable_buffer and has_any_chunk and all_chunks_eligible and all_pages_dict;
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
  e.has_any_chunk                 = true;
  bool const chunk_matches_buffer = e.key_type.id() == type_id::STRING
                                      ? is_byte_array_string_chunk(chunk)
                                      : is_fixed_width_dict_chunk(chunk, e.key_type);
  if (chunk.max_nesting_depth != 1 or chunk.max_level[level_type::REPETITION] != 0 or
      not chunk_matches_buffer or chunk.num_dict_pages < 1) {
    e.all_chunks_eligible = false;
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

  // Mark columns whose output buffer is a flat string or fixed-width column, and record the
  // buffer's logical type: it becomes the DICTIONARY32 keys type.
  std::transform(
    input_columns.begin(), input_columns.end(), elig.begin(), [&](input_column_info const& col) {
      column_eligibility e{};
      if (col.nesting_depth() != 1) { return e; }
      auto const out_type = output_buffers[col.nesting[0]].type;
      e.has_transcodable_buffer =
        out_type.id() == type_id::STRING or
        (cudf::is_fixed_width(out_type) and out_type.id() != type_id::BOOL8);
      if (e.has_transcodable_buffer) { e.key_type = out_type; }
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
 * @brief Build a keys column of `key_type` from a chunk's dictionary entries.
 *
 * String chunks use the reader's `str_dict_index` (pointer/length pairs). Fixed-width chunks copy
 * the PLAIN-encoded dictionary page, whose entries are the output values verbatim (guaranteed by
 * `is_fixed_width_dict_chunk`).
 *
 * @param chunk The column chunk whose dictionary becomes the keys
 * @param dict_page_data Device pointer to the chunk's (decompressed) dictionary page payload
 * @param key_type The logical output type of the column
 * @param entry_count Number of dictionary entries (keys) for this chunk
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's memory
 * @return A `key_type` column holding this chunk's dictionary keys
 */
[[nodiscard]] std::unique_ptr<column> make_keys_column(ColumnChunkDesc const& chunk,
                                                       uint8_t const* dict_page_data,
                                                       data_type key_type,
                                                       size_type entry_count,
                                                       rmm::cuda_stream_view stream,
                                                       rmm::device_async_resource_ref mr)
{
  if (key_type.id() == type_id::STRING) {
    return make_keys_column_from_index_pairs(chunk.str_dict_index, entry_count, stream, mr);
  }
  if (entry_count <= 0) { return cudf::make_empty_column(key_type); }
  auto const keys_bytes = static_cast<std::size_t>(entry_count) * cudf::size_of(key_type);
  return std::make_unique<column>(key_type,
                                  entry_count,
                                  rmm::device_buffer{dict_page_data, keys_bytes, stream, mr},
                                  rmm::device_buffer{},
                                  0);
}

}  // namespace

void reader_impl::prepare_dict_transcode(read_mode mode)
{
  CUDF_FUNC_RANGE();

  _dict_transcode_eligible.assign(_input_columns.size(), false);
  _dict_transcode_key_types.assign(_input_columns.size(), data_type{type_id::EMPTY});

  if (not _options.output_dict_columns) { return; }

  // The fast path requires the whole column to live in a single subpass. For chunked / multi-pass
  // reads (non-zero chunk or pass read limit) we skip it and let `finalize_output` produce the
  // DICTIONARY32 columns via a post-hoc `dictionary::detail::encode` instead.
  if (_output_chunk_read_limit != 0 or _input_pass_read_limit != 0) { return; }

  // Skip the fast path if custom row bounds are in effect.
  if (uses_custom_row_bounds(mode)) { return; }

  // AST/JIT filters evaluate predicates on materialized STRING columns, so the direct transcode
  // fast path cannot run under a filter. Skip it and let `finalize_output` encode the filtered
  // STRING result to DICTIONARY32 via the post-hoc `dictionary::detail::encode` fallback.
  if (_expr_conv.get_converted_expr().has_value()) { return; }

  auto& pass    = *_pass_itm_data;
  auto& subpass = *pass.subpass;

  if (pass.chunks.empty() or subpass.pages.size() == 0) { return; }

  auto const elig = compute_dict_transcode_eligibility(pass, _input_columns, _output_buffers);
  std::transform(
    elig.begin(), elig.end(), _dict_transcode_eligible.begin(), [](column_eligibility const& e) {
      return e.is_eligible();
    });
  std::transform(
    elig.begin(), elig.end(), _dict_transcode_key_types.begin(), [](column_eligibility const& e) {
      return e.is_eligible() ? e.key_type : data_type{type_id::EMPTY};
    });

  auto const num_eligible =
    std::count(_dict_transcode_eligible.begin(), _dict_transcode_eligible.end(), true);
  if (num_eligible == 0) { return; }

  auto const num_input_cols = _input_columns.size();

  // Per-column index type from the largest chunk dictionary, following the same width rule as
  // `dictionary::encode` (`get_indices_type_for_size`), so that concatenating batches keeps the
  // width unless the merged keys overflow it.
  std::vector<data_type> index_types(num_input_cols, data_type{type_id::INT32});
  {
    std::vector<size_type> max_keys(num_input_cols, 0);
    for (auto const& page : pass.pages) {
      if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) == 0) { continue; }
      auto const col_idx = pass.chunks[page.chunk_idx].src_col_index;
      max_keys[col_idx]  = std::max(max_keys[col_idx], size_type{page.num_input_values});
    }
    std::transform(max_keys.begin(), max_keys.end(), index_types.begin(), [](size_type keys) {
      return cudf::dictionary::detail::get_indices_type_for_size(keys);
    });
  }

  // Change the output buffer type for eligible columns to the index type: the decode writes the
  // dictionary indices instead of the logical values.
  std::for_each(
    cuda::counting_iterator<size_t>{0}, cuda::counting_iterator{num_input_cols}, [&](size_t i) {
      if (not _dict_transcode_eligible[i]) { return; }
      auto& out_buf = _output_buffers[_input_columns[i].nesting[0]];
      out_buf.type  = index_types[i];
    });

  // Rewrite per-page `kernel_mask` for eligible columns on the host subpass pages from
  // STRING_DICT → DICT_INT32, then H2D so the device pages agree.
  bool any_rewritten = false;
  std::for_each(subpass.pages.host_begin(), subpass.pages.host_end(), [&](PageInfo& page) {
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) { return; }
    auto const chunk_idx = page.chunk_idx;
    auto const col_idx   = pass.chunks[chunk_idx].src_col_index;
    if (not _dict_transcode_eligible[col_idx]) { return; }
    if (page.kernel_mask == decode_kernel_mask::STRING_DICT or
        page.kernel_mask == decode_kernel_mask::FIXED_WIDTH_DICT) {
      page.kernel_mask = decode_kernel_mask::DICT_INT32;
      pass.chunks[chunk_idx].dict_index_bytes =
        static_cast<uint8_t>(cudf::size_of(index_types[col_idx]));
      any_rewritten = true;
    }
  });

  // No page was actually rewritten. Clear the eligibility flags so the member reflects the true
  // "inactive" state.
  if (not any_rewritten) {
    _dict_transcode_eligible.assign(_input_columns.size(), false);
    return;
  }

  // Push the rewritten `kernel_mask`s and chunk index widths back to device so subsequent decode
  // kernels dispatch and write correctly.
  subpass.pages.host_to_device_async(_stream);
  pass.chunks.host_to_device_async(_stream);
  subpass.kernel_mask = std::transform_reduce(
    subpass.pages.host_begin(),
    subpass.pages.host_end(),
    uint32_t{0},
    std::bit_or<>{},
    [](PageInfo const& page) { return static_cast<uint32_t>(page.kernel_mask); });
}

void reader_impl::assemble_dict_transcoded_columns(
  std::vector<std::unique_ptr<column>>& out_columns)
{
  CUDF_FUNC_RANGE();

  if (_pass_itm_data == nullptr) { return; }

  // Nothing to assemble unless `prepare_dict_transcode` marked at least one column eligible.
  if (std::none_of(_dict_transcode_eligible.begin(),
                   _dict_transcode_eligible.end(),
                   [](bool eligible) { return eligible; })) {
    return;
  }

  auto const& pass = *_pass_itm_data;

  // For each eligible input column, collect its chunks in row-group order, build a per-chunk
  // DICTIONARY32 segment (local 0-based indices + per-chunk keys column), and concatenate.
  //
  // IMPORTANT: Each segment carries row-group-local indices into its own keys column. We do NOT
  // pre-shift indices into a global keyspace, because `cudf::dictionary::detail::concatenate`
  // already re-maps the indices using `compute_children_offsets_fn`. Pre-shifting would cause
  // double-offsetting and out-of-bounds reads in the `dispatch_compute_indices` kernel.
  std::for_each(
    cuda::counting_iterator<size_t>{0},
    cuda::counting_iterator{_input_columns.size()},
    [&](size_t i) {
      if (not _dict_transcode_eligible[i]) { return; }

      // Gather chunk indices for this input column in row-group order.
      std::vector<size_t> chunk_indices;
      chunk_indices.reserve(pass.chunks.size() / std::max<size_t>(_input_columns.size(), 1));
      std::copy_if(cuda::counting_iterator<size_t>{0},
                   cuda::counting_iterator{pass.chunks.size()},
                   std::back_inserter(chunk_indices),
                   [&](size_t c) { return pass.chunks[c].src_col_index == static_cast<int>(i); });
      if (chunk_indices.empty()) { return; }

      // `out_columns` is indexed by output-buffer (root column) ordinal, not input-column
      // ordinal: a nested struct/list column contributes one entry to `_output_buffers` but one
      // entry per leaf to `_input_columns`, so `i` and the corresponding root index can diverge
      // as soon as any nested column precedes this one. Eligibility requires a flat (depth-1)
      // column, so `nesting[0]` is the correct, and only, output-buffer index to use here.
      auto const out_idx = static_cast<size_t>(_input_columns[i].nesting[0]);

      auto const key_type = _dict_transcode_key_types[i];
      CUDF_EXPECTS(key_type.id() != type_id::EMPTY,
                   "Missing keys type for a dict-transcoded column");

      // Per-chunk key counts and dictionary page payloads from the dictionary page, mirrored
      // back to host when `pass.pages` was copied by `decode_page_headers`.
      std::vector<size_type> chunk_key_counts(chunk_indices.size(), 0);
      std::vector<uint8_t const*> chunk_dict_data(chunk_indices.size(), nullptr);
      for (size_t k = 0; k < chunk_indices.size(); ++k) {
        auto const chunk_idx = chunk_indices[k];
        if (pass.chunks[chunk_idx].dict_page == nullptr) { continue; }
        for (auto const& page : pass.pages) {
          if (page.chunk_idx == static_cast<int32_t>(chunk_idx) and
              (page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) {
            chunk_dict_data[k]  = page.page_data;
            chunk_key_counts[k] = static_cast<size_type>(page.num_input_values);
            break;
          }
        }
      }

      auto& indices_col = out_columns[out_idx];
      CUDF_EXPECTS(indices_col != nullptr and (indices_col->type().id() == type_id::INT8 or
                                               indices_col->type().id() == type_id::INT16 or
                                               indices_col->type().id() == type_id::INT32),
                   "Expected a signed integer indices column for a dict-transcoded flat column");
      auto indices_owner = std::move(indices_col);

      // Single row group fast path: the Parquet dictionary page's entries become the keys as-is,
      // in page order. If a file carries duplicate dictionary entries, the
      // shortcut is skipped and the general multi-chunk path below runs instead:
      // `emit_single_row_group_column` returns true when it fell back (keys not distinct), leaving
      // `indices_owner` intact for the path below.
      auto const emit_single_row_group_column = [&]() -> bool {
        auto const& chunk = pass.chunks[chunk_indices[0]];
        auto keys =
          make_keys_column(chunk, chunk_dict_data[0], key_type, chunk_key_counts[0], _stream, _mr);
        auto const num_distinct_keys = cudf::detail::distinct_count(
          keys->view(), null_policy::INCLUDE, nan_policy::NAN_IS_VALID, _stream);
        if (num_distinct_keys != keys->size()) { return true; }  // fall back: dedup below
        out_columns[out_idx] =
          cudf::make_dictionary_column(std::move(keys), std::move(indices_owner), _stream, _mr);
        return false;
      };

      if (chunk_indices.size() == 1) {
        bool const fallback_used = emit_single_row_group_column();
        if (not fallback_used) { return; }
        // Keys were not distinct: fall through to the multi-row-group path, which deduplicates.
      }

      // Multi-row-group path: the indices buffer is shared (aliased) by per-chunk DICTIONARY32
      // views below via the parent's offset/size, so it must stay alive until concatenate
      // completes.
      column_view const indices_view{indices_owner->view()};

      // Per-chunk boundaries along the row axis: chunk k occupies rows
      // [chunk_row_offsets[k], chunk_row_offsets[k+1]).
      std::vector<size_type> chunk_row_offsets(chunk_indices.size() + 1, 0);
      std::transform(
        chunk_indices.begin(),
        chunk_indices.end(),
        chunk_row_offsets.begin() + 1,
        [&](size_t chunk_idx) { return static_cast<size_type>(pass.chunks[chunk_idx].num_rows); });
      std::inclusive_scan(
        chunk_row_offsets.begin() + 1, chunk_row_offsets.end(), chunk_row_offsets.begin() + 1);
      CUDF_EXPECTS(chunk_row_offsets.back() == indices_view.size(),
                   "Row counts on pass chunks must sum to the indices column size");

      // Pre-compute null counts for all segments in a single kernel launch. Building the
      // column_views below requires a per-segment null count, and calling null_count(begin, end)
      // inside the loop would launch one kernel per chunk. Batch them here instead.
      std::vector<size_type> seg_null_counts(chunk_indices.size(), 0);
      if (indices_view.nullable()) {
        std::vector<size_type> indices_pairs;
        indices_pairs.reserve(chunk_indices.size() * 2);
        for (size_t k = 0; k < chunk_indices.size(); ++k) {
          indices_pairs.push_back(chunk_row_offsets[k]);
          indices_pairs.push_back(chunk_row_offsets[k + 1]);
        }
        seg_null_counts =
          cudf::detail::segmented_null_count(indices_view.null_mask(), indices_pairs, _stream);
      }

      // Build a per-chunk DICTIONARY32 *view* that aliases the shared decoded INT32 buffer (no
      // copy): keys = this chunk's STRING column, indices = `indices_view`. The row range, null
      // mask, and null count must all live on the *parent* view (via offset/size), not the indices
      // child, because `get_indices_annotated()` rebuilds the indices from the child's `head()`
      // plus the parent's offset/size/null_mask -- anything set on the child is ignored. A wrong
      // null count (e.g. a hardcoded 0) would silently turn nulls into a valid index once
      // `cudf::detail::concatenate` remaps the indices against the unified keys.
      std::vector<std::unique_ptr<column>> seg_keys_owners(chunk_indices.size());
      std::vector<column_view> dict_segment_views(chunk_indices.size());
      std::transform(cuda::counting_iterator<size_t>{0},
                     cuda::counting_iterator{chunk_indices.size()},
                     dict_segment_views.begin(),
                     [&](size_t k) {
                       auto const chunk_idx = chunk_indices[k];
                       auto const& chunk    = pass.chunks[chunk_idx];

                       seg_keys_owners[k] = make_keys_column(chunk,
                                                             chunk_dict_data[k],
                                                             key_type,
                                                             chunk_key_counts[k],
                                                             _stream,
                                                             get_current_device_resource_ref());

                       auto const seg_begin = chunk_row_offsets[k];
                       auto const seg_end   = chunk_row_offsets[k + 1];
                       auto const seg_rows  = seg_end - seg_begin;
                       return column_view{data_type{type_id::DICTIONARY32},
                                          seg_rows,
                                          nullptr,  // dictionary parent holds no data
                                          indices_view.null_mask(),  // shared with indices_view
                                          seg_null_counts[k],
                                          seg_begin,  // reslices shared indices child + null mask
                                          {indices_view, seg_keys_owners[k]->view()}};
                     });

      // `cudf::detail::concatenate` deduplicates + sorts keys and recomputes indices.
      out_columns[out_idx] = cudf::detail::concatenate(dict_segment_views, _stream, _mr);
    });
}

}  // namespace cudf::io::parquet::detail

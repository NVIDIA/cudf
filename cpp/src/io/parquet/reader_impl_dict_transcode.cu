/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reader_impl.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/unary.hpp>
#include <cudf/detail/utilities/batched_memset.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/host_vector.hpp>
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
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>

#include <algorithm>
#include <functional>
#include <numeric>
#include <type_traits>
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
 *  - the corresponding output buffer is a flat STRING column, or a flat fixed-width column whose
 *    decode is a plain copy of INT32/INT64 physical values (no decimal, timestamp-unit, or width
 *    conversion) -- the buffer's logical type becomes the DICTIONARY32 keys type,
 *  - every chunk of that column is a matching chunk of that type with a dictionary page,
 *  - every data page of every chunk of that column uses DICTIONARY encoding,
 *  - the chunk has a flat (non-list, non-nested) schema.
 *
 * @param pass The pass intermediate data holding host-side chunks and pages
 * @param input_columns The reader's input column descriptors
 * @param output_buffers The output column buffers (used to detect eligible flat columns and
 * their logical key types)
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
 * @brief Gathers one column's stacked keys out of the shared `pass.str_dict_index` buffer.
 *
 * Maps a stacked-key position in `[0, total_keys)` to its `string_index_pair`.
 * Fed through a counting-transform iterator.
 */
struct stacked_key_gather_fn {
  string_index_pair const* str_dict_index;
  cudf::device_span<size_type const> key_counts_prefix;  ///< size num_chunks + 1
  cudf::device_span<size_type const> key_base_offsets;   ///< size num_chunks

  __device__ string_index_pair operator()(size_type stacked_pos) const
  {
    auto const it = thrust::upper_bound(
      thrust::seq, key_counts_prefix.begin(), key_counts_prefix.end(), stacked_pos);
    auto const k     = static_cast<size_type>(it - key_counts_prefix.begin() - 1);
    auto const local = stacked_pos - key_counts_prefix[k];
    return str_dict_index[key_base_offsets[k] + local];
  }
};

/**
 * @brief Remap each row's dictionary index onto the deduplicated key space (in place).
 *
 * Each input row's decoded index is local to its own row group's dictionary. This function shifts
 * that index to point at the correct entry in the compact, unique keys column. Done in place, in
 * one pass over the rows, in lieu of `cudf::dictionary::detail::concatenate`.
 *
 * @tparam Index Signed integer index type
 * @param indices Signed integer index buffer (one entry per row), mutated in place
 * @param row_offsets Per-chunk row boundaries `[offsets[k], offsets[k+1])`, size num_chunks+1
 * @param key_counts_prefix Per-chunk key-prefix offsets into the stacked key space, size
 * num_chunks+1
 * @param stacked_to_unique Map from stacked-key position to compact unique-key index
 * @param stream CUDA stream used for the kernel launch
 */
template <typename Index>
void remap_dict_indices_by_chunk(cudf::device_span<Index> indices,
                                 cudf::device_span<size_type const> row_offsets,
                                 cudf::device_span<size_type const> key_counts_prefix,
                                 cudf::device_span<int32_t const> stacked_to_unique,
                                 cuda::stream_ref stream)
{
  thrust::for_each(
    rmm::exec_policy_nosync(stream, get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator{static_cast<size_type>(indices.size())},
    [row_offsets, key_counts_prefix, stacked_to_unique, indices] __device__(size_type row) -> void {
      // Chunk owning `row` is the last offset <= row.
      auto const it = thrust::upper_bound(thrust::seq, row_offsets.begin(), row_offsets.end(), row);
      auto const k  = static_cast<size_type>(it - row_offsets.begin() - 1);
      // Guard for all-null case.
      if (key_counts_prefix[k] == key_counts_prefix[k + 1]) {
        indices[row] = 0;
        return;
      }
      auto const stacked_pos = key_counts_prefix[k] + indices[row];
      indices[row]           = static_cast<Index>(stacked_to_unique[stacked_pos]);
    });
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
                                                       cuda::stream_ref stream,
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

  // Keys are materialized per eligible column.

  // Pre-pass 1: Map each chunk to its dictionary page's key count.
  // Chunks without a dictionary page keep a count of 0.
  std::vector<size_type> chunk_dict_key_counts(pass.chunks.size(), 0);
  std::vector<uint8_t const*> chunk_dict_payloads(pass.chunks.size(), nullptr);
  for (auto const& page : pass.pages) {
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) == 0) { continue; }
    auto const chunk_idx = page.chunk_idx;
    if (chunk_idx < 0 or static_cast<size_t>(chunk_idx) >= pass.chunks.size()) { continue; }
    if (pass.chunks[chunk_idx].dict_page == nullptr) { continue; }
    chunk_dict_key_counts[chunk_idx] = static_cast<size_type>(page.num_input_values);
    chunk_dict_payloads[chunk_idx]   = page.page_data;
  }

  // Pre-pass 2: Bucket chunk indices by their source input-column ordinal. Because
  // `pass.chunks` is laid out row-group-major, appending in index order yields each column's chunks
  // already in row-group order.
  std::vector<std::vector<size_t>> chunks_by_input_col(_input_columns.size());
  for (size_t c = 0; c < pass.chunks.size(); ++c) {
    auto const col = pass.chunks[c].src_col_index;
    if (col >= 0 and static_cast<size_t>(col) < _input_columns.size()) {
      chunks_by_input_col[col].push_back(c);
    }
  }

  // For each eligible input column, collect its chunks in row-group order and assemble a
  // DICTIONARY32 output.
  //
  // A single-row-group column takes a zero-copy fast path (keys + decoded indices stapled
  // together). A multi-row-group column stacks the per-chunk keys, deduplicates them, and remaps
  // the decoded indices onto the compact key space in place -- avoiding
  // `cudf::dictionary::detail::concatenate` and its redundant per-chunk index copy.
  std::for_each(
    cuda::counting_iterator<size_t>{0},
    cuda::counting_iterator{_input_columns.size()},
    [&](size_t i) {
      if (not _dict_transcode_eligible[i]) { return; }

      // This column's chunks, in row-group order (bucketed in pre-pass 2 above).
      auto const& chunk_indices = chunks_by_input_col[i];
      if (chunk_indices.empty()) { return; }

      // `out_columns` is indexed by output-buffer (root column) ordinal, not input-column
      // ordinal: a nested struct/list column contributes one entry to `_output_buffers` but one
      // entry per leaf to `_input_columns`, so `i` and the corresponding root index can diverge
      // as soon as any nested column precedes this one. Eligibility requires a flat (depth-1)
      // column, so `nesting[0]` is the correct, and only, output-buffer index to use.
      auto const out_idx = static_cast<size_t>(_input_columns[i].nesting[0]);

      auto const key_type = _dict_transcode_key_types[i];
      CUDF_EXPECTS(key_type.id() != type_id::EMPTY,
                   "Missing keys type for a dict-transcoded column");

      // Per-chunk key counts and payloads from pre-pass 1.
      std::vector<size_type> chunk_key_counts(chunk_indices.size());
      std::vector<uint8_t const*> chunk_dict_data(chunk_indices.size());
      for (size_t k = 0; k < chunk_indices.size(); ++k) {
        chunk_key_counts[k] = chunk_dict_key_counts[chunk_indices[k]];
        chunk_dict_data[k]  = chunk_dict_payloads[chunk_indices[k]];
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

      // Multi-row-group path (dedup-and-shift): stack every chunk's keys into a single column,
      // deduplicate the key set once, then remap each row's index onto the compact key space in
      // place. This avoids `cudf::dictionary::detail::concatenate`, which would re-copy the
      // already-contiguous per-chunk indices (`indices_owner`) into a fresh buffer.
      auto const num_row_vals = static_cast<size_type>(indices_owner->size());

      // Per-chunk row boundaries: chunk k occupies rows [chunk_row_offsets[k],
      // chunk_row_offsets[k+1]).
      auto chunk_row_offsets =
        cudf::detail::make_pinned_vector_async<size_type>(chunk_indices.size() + 1, _stream);
      chunk_row_offsets[0] = 0;
      std::transform(
        chunk_indices.begin(),
        chunk_indices.end(),
        chunk_row_offsets.begin() + 1,
        [&](size_t chunk_idx) { return static_cast<size_type>(pass.chunks[chunk_idx].num_rows); });
      std::inclusive_scan(
        chunk_row_offsets.begin() + 1, chunk_row_offsets.end(), chunk_row_offsets.begin() + 1);
      CUDF_EXPECTS(chunk_row_offsets.back() == num_row_vals,
                   "Row counts on pass chunks must sum to the indices column size");

      // Per-chunk key prefix offsets into the stacked key space: chunk k's keys occupy
      // [key_counts_prefix[k], key_counts_prefix[k+1]).
      auto key_counts_prefix =
        cudf::detail::make_pinned_vector_async<size_type>(chunk_indices.size() + 1, _stream);
      key_counts_prefix[0] = 0;
      std::inclusive_scan(
        chunk_key_counts.begin(), chunk_key_counts.end(), key_counts_prefix.begin() + 1);
      auto const total_keys = key_counts_prefix.back();

      // Per-chunk base offsets: where chunk k's keys begin in the shared `pass.str_dict_index`.
      auto const key_offset_of = [&](size_t k) {
        return static_cast<size_type>(pass.chunks[chunk_indices[k]].str_dict_index -
                                      pass.str_dict_index.data());
      };

      // The per-chunk key ranges are contiguous in `pass.str_dict_index` iff this is the only
      // eligible string column: chunks are laid out row-group-major, so a second string column
      // interleaves its chunks between this one's.
      auto const contiguous =
        key_type.id() == type_id::STRING and
        std::all_of(
          cuda::counting_iterator<size_t>{0},
          cuda::counting_iterator{chunk_indices.size() - 1},
          [&](size_t k) { return key_offset_of(k + 1) == key_offset_of(k) + chunk_key_counts[k]; });

      // Device copies of the per-chunk row/key boundaries, reused by the strided key gather below
      // and by `remap_dict_indices_by_chunk`.
      auto const d_row_offsets = cudf::detail::make_device_uvector_async(
        chunk_row_offsets, _stream, get_current_device_resource_ref());
      auto const d_key_counts_prefix = cudf::detail::make_device_uvector_async(
        key_counts_prefix, _stream, get_current_device_resource_ref());

      // Host source for the strided gather's per-chunk base offsets. Declared at iteration scope
      // (populated only in the strided branch) so it outlives its async copy until the synchronize.
      auto key_base_offsets = cudf::detail::make_pinned_vector_async<size_type>(0, _stream);

      // Stack this column's per-chunk keys into one STRING column, materialized per column so each
      // `make_strings_column` stays within the 2 GiB string-offset limit.
      //
      // Contiguous (single eligible string column): one `make_strings_column` over the contiguous
      // entry range.
      //
      // Strided (multiple string columns): a single gather `make_strings_column` that pulls each
      // stacked position from the correct place in `pass.str_dict_index` via a counting-transform
      // iterator.
      std::unique_ptr<column> stacked_keys_owner;
      if (key_type.id() != type_id::STRING) {
        // Dictionary payloads follow variable-length page headers and need not be aligned
        // for key_type. Copy bytes into an aligned allocation before exposing typed keys.
        auto const key_width = cudf::size_of(key_type);
        rmm::device_buffer stacked_data{static_cast<std::size_t>(total_keys) * key_width,
                                        _stream,
                                        get_current_device_resource_ref()};
        auto* const dst = static_cast<uint8_t*>(stacked_data.data());
        for (size_t k = 0; k < chunk_indices.size(); ++k) {
          if (chunk_key_counts[k] == 0) { continue; }
          CUDF_CUDA_TRY(cudf::detail::memcpy_async(
            dst + static_cast<std::size_t>(key_counts_prefix[k]) * key_width,
            chunk_dict_data[k],
            static_cast<std::size_t>(chunk_key_counts[k]) * key_width,
            _stream));
        }
        stacked_keys_owner = std::make_unique<column>(
          key_type, total_keys, std::move(stacked_data), rmm::device_buffer{}, 0);
      } else if (contiguous) {
        stacked_keys_owner =
          make_keys_column_from_index_pairs(pass.chunks[chunk_indices[0]].str_dict_index,
                                            total_keys,
                                            _stream,
                                            get_current_device_resource_ref());
      } else {
        key_base_offsets.resize(chunk_indices.size());
        std::transform(cuda::counting_iterator<size_t>{0},
                       cuda::counting_iterator{chunk_indices.size()},
                       key_base_offsets.begin(),
                       [&](size_t k) { return key_offset_of(k); });
        auto const d_key_base_offsets = cudf::detail::make_device_uvector_async(
          key_base_offsets, _stream, get_current_device_resource_ref());

        auto const keys_begin = cudf::detail::make_counting_transform_iterator(
          size_type{0},
          stacked_key_gather_fn{pass.str_dict_index.data(),
                                cudf::device_span<size_type const>{d_key_counts_prefix.data(),
                                                                   d_key_counts_prefix.size()},
                                cudf::device_span<size_type const>{d_key_base_offsets.data(),
                                                                   d_key_base_offsets.size()}});
        stacked_keys_owner = cudf::strings::detail::make_strings_column(
          keys_begin, keys_begin + total_keys, _stream, get_current_device_resource_ref());
      }
      column_view const stacked_keys = stacked_keys_owner->view();

      // Deduplicate the stacked keys. `encode` yields the compact unique keys (on `_mr`, the output
      // keys child) plus an INT32 map from each stacked-key position to its compact index.
      auto encoded =
        cudf::dictionary::detail::encode(stacked_keys, data_type{type_id::INT32}, _stream, _mr);
      auto encoded_contents  = encoded->release();
      auto stacked_to_unique = std::move(
        encoded_contents.children[dictionary_column_view::indices_column_index]);  // INT32 map
      auto unique_keys = std::move(
        encoded_contents
          .children[dictionary_column_view::keys_column_index]);  // compact keys, owned on _mr

      // Remap every row's index onto the compact key space in place. Null rows carry a zero index
      // (fill_pruned_offsets); the shift keeps them in range and the null mask (carried by
      // `indices_owner`) still nullifies them in `decode`.
      // The union of row-group keys may require wider indices than any individual dictionary.
      auto const required_type =
        cudf::dictionary::detail::get_indices_type_for_size(unique_keys->size());
      if (cudf::size_of(required_type) > cudf::size_of(indices_owner->type())) {
        indices_owner = cudf::detail::cast(indices_owner->view(), required_type, _stream, _mr);
      }
      cudf::type_dispatcher(indices_owner->type(), [&]<typename Index>() {
        if constexpr (std::is_same_v<Index, int8_t> or std::is_same_v<Index, int16_t> or
                      std::is_same_v<Index, int32_t>) {
          remap_dict_indices_by_chunk(
            cudf::device_span<Index>{indices_owner->mutable_view().data<Index>(),
                                     static_cast<std::size_t>(num_row_vals)},
            cudf::device_span<size_type const>{d_row_offsets.data(), d_row_offsets.size()},
            cudf::device_span<size_type const>{d_key_counts_prefix.data(),
                                               d_key_counts_prefix.size()},
            cudf::device_span<int32_t const>{stacked_to_unique->view().data<int32_t>(),
                                             static_cast<std::size_t>(stacked_to_unique->size())},
            _stream);
        } else {
          CUDF_FAIL("Expected signed INT8, INT16 or INT32 dictionary indices");
        }
      });

      _stream.sync();

      out_columns[out_idx] = cudf::make_dictionary_column(
        std::move(unique_keys), std::move(indices_owner), _stream, _mr);
    });
}

}  // namespace cudf::io::parquet::detail

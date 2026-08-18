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

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

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
 * @brief Remap each row's dictionary index onto the deduplicated key space (in place).
 *
 * If output_dict_columns is set, the transcode fast path copies the dictionary indices as is on
 * Parquet, into a new INT32 column. These indices (d_indices) were indexing the keys for the local
 * row group. This function shifts the indices into the chunk's region of the stacked
 * (non-deduplicated) key space, and remaps them onto the deduplicated key space spanning all row
 * groups. Remapping is done in place. Used in-lieu of `cudf::dictionary::detail::concatenate`.
 *
 * @param d_indices Device pointer to the INT32 index buffer, mutated in place
 * @param num_rows Number of index values
 * @param row_offsets Per-chunk row boundaries `[offsets[k], offsets[k+1])`, size num_chunks+1
 * @param key_counts_prefix Per-chunk key-prefix offsets into the stacked key space, size
 * num_chunks+1
 * @param stacked_to_unique Map from stacked-key position to compact unique-key index
 * @param stream CUDA stream used for the kernel launch
 */
void remap_dict_indices_by_chunk(int32_t* d_indices,
                                 size_type num_rows,
                                 cudf::device_span<size_type const> row_offsets,
                                 cudf::device_span<size_type const> key_counts_prefix,
                                 cudf::device_span<int32_t const> stacked_to_unique,
                                 rmm::cuda_stream_view stream)
{
  thrust::for_each(
    rmm::exec_policy_nosync(stream),
    thrust::make_counting_iterator(size_type{0}),
    thrust::make_counting_iterator(num_rows),
    [row_offsets, key_counts_prefix, stacked_to_unique, d_indices] __device__(size_type row) {
      // Chunk owning `row` is the last offset <= row.
      auto const it = thrust::upper_bound(thrust::seq, row_offsets.begin(), row_offsets.end(), row);
      auto const k  = static_cast<size_type>(it - row_offsets.begin() - 1);
      auto const stacked_pos = key_counts_prefix[k] + d_indices[row];
      d_indices[row]         = stacked_to_unique[stacked_pos];
    });
}

}  // namespace

void reader_impl::prepare_dict_transcode(read_mode mode)
{
  CUDF_FUNC_RANGE();

  _dict_transcode_eligible.assign(_input_columns.size(), false);

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

  auto const num_eligible =
    std::count(_dict_transcode_eligible.begin(), _dict_transcode_eligible.end(), true);
  if (num_eligible == 0) { return; }

  auto const num_input_cols = _input_columns.size();

  // Change the output buffer type for eligible columns from STRING → INT32.
  std::for_each(
    cuda::counting_iterator<size_t>{0}, cuda::counting_iterator{num_input_cols}, [&](size_t i) {
      if (not _dict_transcode_eligible[i]) { return; }
      auto& out_buf = _output_buffers[_input_columns[i].nesting[0]];
      out_buf.type  = data_type{type_id::INT32};
    });

  // Rewrite per-page `kernel_mask` for eligible columns on the host subpass pages from
  // STRING_DICT → DICT_INT32, then H2D so the device pages agree.
  bool any_rewritten = false;
  std::for_each(subpass.pages.host_begin(), subpass.pages.host_end(), [&](PageInfo& page) {
    if ((page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) { return; }
    auto const chunk_idx = page.chunk_idx;
    auto const col_idx   = pass.chunks[chunk_idx].src_col_index;
    if (not _dict_transcode_eligible[col_idx]) { return; }
    if (page.kernel_mask == decode_kernel_mask::STRING_DICT) {
      page.kernel_mask = decode_kernel_mask::DICT_INT32;
      any_rewritten    = true;
    }
  });

  // No page was actually rewritten. Clear the eligibility flags so the member reflects the true
  // "inactive" state.
  if (not any_rewritten) {
    _dict_transcode_eligible.assign(_input_columns.size(), false);
    return;
  }

  // Push the rewritten `kernel_mask`s back to device so subsequent decode kernels dispatch
  // correctly.
  subpass.pages.host_to_device_async(_stream);
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

  // Every string chunk's dictionary entries live contiguously in one buffer in
  // `pass.str_dict_index` (each `chunk.str_dict_index` is a pointer into that one buffer).
  // Materialize all keys into a single column using `make_strings_column` (contains duplicates).
  // All keys stores keys of all columns, and not just column i.
  std::unique_ptr<column> all_keys;
  auto ensure_all_keys = [&]() -> column_view {
    if (all_keys == nullptr) {
      all_keys =
        make_keys_column_from_index_pairs(pass.str_dict_index.data(),
                                          static_cast<size_type>(pass.str_dict_index.size()),
                                          _stream,
                                          get_current_device_resource_ref());
    }
    return all_keys->view();
  };

  // For each eligible input column, collect its chunks in row-group order, build a per-chunk
  // DICTIONARY32 segment (local 0-based indices + per-chunk keys column), and concatenate.
  //
  // A single-row-group column takes a zero-copy fast path (keys + decoded indices stapled
  // together). A multi-row-group column stacks the per-chunk keys, deduplicates them, and remaps
  // the decoded indices onto the compact key space in place,  avoiding
  // `cudf::dictionary::detail::concatenate` and its redundant per-chunk index copy.
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
      // column, so `nesting[0]` is the correct, and only, output-buffer index to use.
      auto const out_idx = static_cast<size_t>(_input_columns[i].nesting[0]);

      // Per-chunk key counts from the dictionary page's `num_input_values`, mirrored back to
      // host when `pass.pages` was copied by `decode_page_headers`.
      std::vector<size_type> chunk_key_counts(chunk_indices.size(), 0);
      std::transform(chunk_indices.begin(),
                     chunk_indices.end(),
                     chunk_key_counts.begin(),
                     [&](size_t chunk_idx) -> size_type {
                       if (pass.chunks[chunk_idx].dict_page == nullptr) { return 0; }
                       for (auto const& page : pass.pages) {
                         if (page.chunk_idx == static_cast<int32_t>(chunk_idx) and
                             (page.flags & PAGEINFO_FLAGS_DICTIONARY) != 0) {
                           return static_cast<size_type>(page.num_input_values);
                         }
                       }
                       return size_type{0};
                     });

      auto& indices_col = out_columns[out_idx];
      CUDF_EXPECTS(indices_col != nullptr and indices_col->type().id() == type_id::INT32,
                   "Expected INT32 indices column for dict-transcoded flat string column");
      // Claim ownership of the indices column. (output_columns vectoris now empty.)
      auto indices_owner = std::move(indices_col);

      // Single row group fast path: the Parquet dictionary page's entries become the keys as-is,
      // in page order. If a file carries duplicate dictionary entries, the
      // shortcut is skipped and the general multi-chunk path below runs instead:
      // `emit_single_row_group_column` returns true when it fell back (keys not distinct), leaving
      // `indices_owner` intact for the path below.
      auto const emit_single_row_group_column = [&]() -> bool {
        auto const& chunk = pass.chunks[chunk_indices[0]];
        auto keys         = make_keys_column_from_index_pairs(
          chunk.str_dict_index, chunk_key_counts[0], _stream, _mr);
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
      // deduplicate the ( key set once, then remap each row's index onto the compact key space in
      // place. This avoids `cudf::dictionary::detail::concatenate`, which would re-copy the
      // already-contiguous per-chunk indices (`indices_owner`) into a fresh buffer.
      auto const num_row_vals = static_cast<size_type>(indices_owner->size());

      // Per-chunk row boundaries: chunk k occupies rows [chunk_row_offsets[k],
      // chunk_row_offsets[k+1]).
      std::vector<size_type> chunk_row_offsets(chunk_indices.size() + 1, 0);
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
      std::vector<size_type> key_counts_prefix(chunk_indices.size() + 1, 0);
      std::inclusive_scan(
        chunk_key_counts.begin(), chunk_key_counts.end(), key_counts_prefix.begin() + 1);

      // Stack keys: concatenate every chunk's key slice from the batched `all_keys` (not yet
      // deduplicated). `all_keys` owns the data and outlives this concatenate, so the slices stay
      // valid. Chunk `k`'s entries occupy `[key_offset, key_offset + chunk_key_counts[k])` in
      // `pass.str_dict_index`, where `key_offset` is recovered from the chunk's stored pointer.
      auto const all_keys_column = ensure_all_keys();
      std::vector<column_view> key_slices(chunk_indices.size());
      std::transform(cuda::counting_iterator<size_t>{0},
                     cuda::counting_iterator{chunk_indices.size()},
                     key_slices.begin(),
                     [&](size_t k) {
                       auto const& chunk = pass.chunks[chunk_indices[k]];
                       auto const key_offset =
                         static_cast<size_type>(chunk.str_dict_index - pass.str_dict_index.data());
                       return cudf::detail::slice(
                         all_keys_column, key_offset, key_offset + chunk_key_counts[k], _stream);
                     });
      auto const stacked_keys =
        cudf::detail::concatenate(key_slices, _stream, get_current_device_resource_ref());

      // Deduplicate the stacked keys. `encode` yields the compact unique keys (on `_mr`, the output
      // keys child) plus an INT32 map from each stacked-key position to its compact index.
      auto encoded = cudf::dictionary::detail::encode(
        stacked_keys->view(), data_type{type_id::INT32}, _stream, _mr);
      auto encoded_contents = encoded->release();
      auto stacked_to_unique =
        std::move(encoded_contents.children[0]);                   // INT32 map (keep for kernel)
      auto unique_keys = std::move(encoded_contents.children[1]);  // compact keys, owned on _mr

      // Remap every row's index onto the compact key space in place. Null rows carry a zero index
      // (fill_pruned_offsets); the shift keeps them in range and the null mask (carried by
      // `indices_owner`) still nullifies them in `decode`.
      auto const d_row_offsets = cudf::detail::make_device_uvector_async(
        chunk_row_offsets, _stream, get_current_device_resource_ref());
      auto const d_key_counts_prefix = cudf::detail::make_device_uvector_async(
        key_counts_prefix, _stream, get_current_device_resource_ref());
      remap_dict_indices_by_chunk(
        indices_owner->mutable_view().data<int32_t>(),
        num_row_vals,
        cudf::device_span<size_type const>{d_row_offsets.data(), d_row_offsets.size()},
        cudf::device_span<size_type const>{d_key_counts_prefix.data(), d_key_counts_prefix.size()},
        cudf::device_span<int32_t const>{stacked_to_unique->view().data<int32_t>(),
                                         static_cast<std::size_t>(stacked_to_unique->size())},
        _stream);

      out_columns[out_idx] = cudf::make_dictionary_column(
        std::move(unique_keys), std::move(indices_owner), _stream, _mr);
    });
}

}  // namespace cudf::io::parquet::detail

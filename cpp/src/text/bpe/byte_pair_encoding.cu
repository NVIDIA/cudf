/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/bpe/byte_pair_encoding.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/byte_pair_encoding.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/unique.h>

namespace nvtext {

/**
 * @brief Access the bpe_merge_pairs impl member
 *
 * This is used by the encoder to access the impl member functions.
 *
 * @param bpe The merge pairs struct
 * @return The impl object with detailed, internal member data
 */
bpe_merge_pairs::bpe_merge_pairs_impl const* get_bpe_merge_pairs_impl(bpe_merge_pairs const& bpe)
{
  return bpe.impl;
}

namespace detail {
namespace {

constexpr int block_size = 512;

/**
 * @brief Produces offsets to unpairable locations in the given chars array
 *
 * Launched as a thread per byte of the chars array.
 * The output is non-zero offsets to locations of unpairable substrings.
 * An unpairable substring does not exist in the given map and so will
 * never be paired. Fortunately, this can be used as an artificial
 * boundary providing increased parallelism in the BPE kernel.
 *
 * @tparam MapRefType The type of the map finder object
 */
template <typename MapRefType>
struct bpe_unpairable_offsets_fn {
  cudf::device_span<char const> d_chars;
  int64_t offset;
  MapRefType const d_map;
  __device__ int64_t operator()(int64_t idx)
  {
    if (!cudf::strings::detail::is_begin_utf8_char(d_chars[idx])) { return 0; }

    auto const itr  = d_chars.data() + idx;
    auto const end  = d_chars.data() + d_chars.size();
    auto const lhs  = cudf::string_view(itr, cudf::strings::detail::bytes_in_utf8_byte(*itr));
    auto const next = itr + lhs.size_bytes();
    auto output     = 0L;
    if (next < end) {
      auto const rhs = cudf::string_view(next, cudf::strings::detail::bytes_in_utf8_byte(*next));
      // see if both halves exist anywhere in the table, if not these are unpairable
      if (d_map.find(lhs) == d_map.end() && d_map.find(rhs) == d_map.end()) {
        output = idx + lhs.size_bytes() + offset;  // offset for artificial boundary
      }
    }
    return output;
  }
};

/**
 * @brief Performs byte-pair-encoding
 *
 * Computes the locations where the separator will be inserted in `d_spaces_data`.
 * This is launched as a string per block.
 *
 * The process first initializes all characters to 1 per position in `d_spaces_data`.
 * All pairs are realized and their ranks stored in `d_ranks_data`.
 *
 * Iteratively, the minimum rank is located, the corresponding `d_spaces_data` location
 * is set to 0 resulting in new potential pairs. The process repeats accounting for
 * the rank of the newly formed pairs.
 *
 * Once there are no more rankable pairs, the process finishes and the `d_spaces_data`
 * values identify the location to insert the separator.
 *
 * @tparam MapRefType The type of the map finder object
 * @param d_strings Input data
 * @param d_input_chars Pointer to the input character data
 * @param d_map For looking up individual string candidates
 * @param d_spaces_data Output the location where separator will be inserted
 * @param d_ranks_data Working memory to hold pair ranks
 * @param d_rerank_data Working memory to hold locations where reranking is required
 */
template <typename MapRefType>
CUDF_KERNEL void bpe_parallel_fn(cudf::column_device_view const d_strings,
                                 char const* d_input_chars,
                                 MapRefType const d_map,
                                 int8_t* d_spaces_data,          // working memory
                                 cudf::size_type* d_ranks_data,  // more working memory
                                 int8_t* d_rerank_data           // and one more working memory
)
{
  // string per block
  auto const str_idx =
    static_cast<cudf::size_type>(cudf::detail::grid_1d::global_thread_id() / block_size);
  auto const lane_idx = static_cast<cudf::size_type>(threadIdx.x);

  auto const d_str  = d_strings.element<cudf::string_view>(str_idx);
  auto const offset = cuda::std::distance(d_input_chars, d_str.data());

  auto const d_spaces   = d_spaces_data + offset;
  auto const end_spaces = d_spaces + d_str.size_bytes();
  auto const d_ranks    = d_ranks_data + offset;
  auto const end_ranks  = d_ranks + d_str.size_bytes();
  auto const d_rerank   = d_rerank_data + offset;
  auto const end_rerank = d_rerank + d_str.size_bytes();

  auto constexpr max_rank = cuda::std::numeric_limits<cudf::size_type>::max();

  __shared__ cudf::size_type block_min_rank;
  using block_reduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  auto const num_valid = block_size < d_str.size_bytes() ? block_size : d_str.size_bytes();

  // init all the re-rank identifiers to zero
  for (auto itr = d_rerank + lane_idx; itr < end_rerank; itr += block_size) {
    *itr = 0;
  }
  // init all ranks to max
  for (auto itr = d_ranks + lane_idx; itr < end_ranks; itr += block_size) {
    *itr = max_rank;
  }
  // init all spaces to 1 as appropriate
  for (auto itr = d_spaces + lane_idx; itr < end_spaces; itr += block_size) {
    auto const index = cuda::std::distance(d_spaces, itr);
    *itr = static_cast<int8_t>(cudf::strings::detail::is_begin_utf8_char(d_str.data()[index]));
  }
  __syncthreads();

  // for finding the next half of a pair
  auto next_substr = [d_str, d_spaces, end = end_spaces](int8_t* begin) {
    auto const next = thrust::find(thrust::seq, begin + 1, end, 1);
    auto const size = static_cast<cudf::size_type>(cuda::std::distance(begin, next));
    return cudf::string_view(d_str.data() + cuda::std::distance(d_spaces, begin), size);
  };
  // for locating adjacent pairs after merging a pair
  auto find_prev = [begin = d_spaces](int8_t* ptr) {
    while (ptr > begin && *ptr == 0) {
      --ptr;
    }
    return ptr;
  };

  auto min_rank = max_rank;

  // store all the initial ranks for each pair
  // every character but the first one will have a initial rank
  //
  // Example:
  // string:   abcdefghij
  // spaces:   1111111111
  // ranks:    *948516327
  for (auto itr = d_spaces + lane_idx; itr < end_spaces; itr += block_size) {
    if (*itr == 0) { continue; }  // skips any UTF-8 continuation bytes
    // resolve pair and lookup its rank
    auto const lhs      = next_substr(itr);  // retrieve lhs of the pair
    auto const next_itr = itr + lhs.size_bytes();
    if (next_itr < end_spaces) {
      auto const rhs = next_substr(next_itr);  // retrieve rhs of the pair
      if (!rhs.empty()) {
        auto rank          = max_rank;
        auto const mp      = merge_pair_type{lhs, rhs};
        auto const map_itr = d_map.find(mp);                      // lookup pair in merges table;
        if (map_itr != d_map.end()) { rank = map_itr->second; }   // found a match;
        d_ranks[cuda::std::distance(d_spaces, next_itr)] = rank;  // store the rank
        if (rank < min_rank) { min_rank = rank; }
      }
    }
  }
  // compute the min rank across the block
  auto const reduce_rank = block_reduce(temp_storage).Reduce(min_rank, cuda::minimum{}, num_valid);
  if (lane_idx == 0) { block_min_rank = reduce_rank; }
  __syncthreads();

  // loop through the ranks processing the current minimum until there are no more
  while (block_min_rank < max_rank) {
    // search the d_ranks for matches to block_min_rank
    for (auto itr = d_ranks + lane_idx; itr < end_ranks; itr += block_size) {
      if (*itr == block_min_rank) {
        auto ptr = itr - 1;  // check for adjacent min-rank (edge-case)
        while (ptr > d_ranks && *ptr == max_rank) {
          --ptr;
        }
        // set the output value to 0 at this position (erases separator, merges pair)
        // using example string above, the min-rank is 1 at position 5
        // string: abcdefghij
        // spaces: 1111101111  (set position 5 to 0)
        if (*ptr != block_min_rank) { d_spaces[cuda::std::distance(d_ranks, itr)] = 0; }
      }
    }
    __syncthreads();

    // identify all the re-rank locations (logic above invalidated adjacent pairs)
    // using example string above, the adjacent pairs have to be re-ranked
    // string: abcdefghij
    // spaces: 1111101111 (pair 'e,f' is now merged)
    // rerank: 0000101000 ('ef' and 'fg' need re-ranking as 'd,ef' and 'ef,g'
    for (auto itr = d_ranks + lane_idx; itr < end_ranks; itr += block_size) {
      auto const index = cuda::std::distance(d_ranks, itr);
      if (*itr == block_min_rank && d_spaces[index] == 0) {
        // find previous pair mid-point
        auto ptr = find_prev(d_spaces + index - 1);
        if (ptr > d_spaces) { d_rerank[cuda::std::distance(d_spaces, ptr)] = 1; }
        // find next pair mid-point
        ptr = thrust::find(thrust::seq, d_spaces + index + 1, end_spaces, 1);
        if (ptr < end_spaces) { d_rerank[cuda::std::distance(d_spaces, ptr)] = 1; }
        *itr = max_rank;  // reset this rank
      }
    }
    __syncthreads();

    // compute the ranks for the newly created pairs
    min_rank = max_rank;  // and record the new minimum along the way
    for (auto itr = d_rerank + lane_idx; itr < end_rerank; itr += block_size) {
      auto const index = cuda::std::distance(d_rerank, itr);
      auto rank        = d_ranks[index];
      if (*itr) {
        *itr = 0;  // reset re-rank
        // build lhs of pair
        auto const ptr  = find_prev(d_spaces + index - 1);
        auto const size = static_cast<cudf::size_type>(cuda::std::distance(ptr, d_spaces + index));
        auto const lhs = cudf::string_view(d_str.data() + cuda::std::distance(d_spaces, ptr), size);
        auto const rhs = next_substr(d_spaces + index);  // retrieve rhs of pair
        rank           = max_rank;
        if (!rhs.empty()) {
          auto const mp      = merge_pair_type{lhs, rhs};
          auto const map_itr = d_map.find(mp);                     // lookup rank for this pair;
          if (map_itr != d_map.end()) { rank = map_itr->second; }  // found a match
        }
        d_ranks[index] = rank;  // store new rank
      }
      if (rank < min_rank) { min_rank = rank; }
    }

    // re-compute the minimum rank across the block (since new pairs are created above)
    auto const reduce_rank =
      block_reduce(temp_storage).Reduce(min_rank, cuda::minimum{}, num_valid);
    if (lane_idx == 0) { block_min_rank = reduce_rank; }
    __syncthreads();
  }  // if no min ranks are found we are done, otherwise start again
}

/**
 * @brief Computes the output size of each strings row
 *
 * This launches as a string per block.
 * The non-zero values in `d_spaces_data` for each string is added to
 * the current string size to produce the total output bytes.
 *
 * @param d_strings Input data
 * @param d_input_chars Pointer to the input character data
 * @param d_spaces_data Output the location where separator will be inserted
 * @param d_sizes Output sizes of each row
 */
CUDF_KERNEL void bpe_finalize(cudf::column_device_view const d_strings,
                              char const* d_input_chars,
                              int8_t* d_spaces_data,    // where separators are inserted
                              cudf::size_type* d_sizes  // output sizes of encoded strings
)
{
  // string per block
  auto const str_idx =
    static_cast<cudf::size_type>(cudf::detail::grid_1d::global_thread_id() / block_size);
  auto const lane_idx = static_cast<cudf::size_type>(threadIdx.x);

  if (d_strings.is_null(str_idx)) {
    d_sizes[str_idx] = 0;
    return;
  }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) {
    d_sizes[str_idx] = 0;
    return;
  }

  auto const offset = cuda::std::distance(d_input_chars, d_str.data());

  auto const d_spaces   = d_spaces_data + offset;
  auto const end_spaces = d_spaces + d_str.size_bytes();
  auto const num_valid  = block_size < d_str.size_bytes() ? block_size : d_str.size_bytes();

  using block_reduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;

  // reset the first position -- no separator to be added here
  if (lane_idx == 0) { *d_spaces = 0; }

  // compute the output size for this string by counting the resulting separator positions
  auto bytes = 0;
  for (auto itr = d_spaces + lane_idx; itr < end_spaces; itr += block_size) {
    bytes += (*itr > 0);
  }
  auto const total_bytes = block_reduce(temp_storage).Sum(bytes, num_valid);
  if (lane_idx == 0) { d_sizes[str_idx] = total_bytes + d_str.size_bytes(); }
}

}  // namespace

std::unique_ptr<cudf::column> byte_pair_encoding(cudf::strings_column_view const& input,
                                                 bpe_merge_pairs const& merge_pairs,
                                                 cudf::string_scalar const& separator,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  if (input.is_empty() || input.chars_size(stream) == 0) {
    return cudf::make_empty_column(cudf::type_id::STRING);
  }

  CUDF_EXPECTS(separator.is_valid(stream), "separator parameter must be valid");
  auto const d_separator = separator.value(stream);
  CUDF_EXPECTS(d_separator.size_bytes() == 1, "for now, separator must be a single-byte character");

  auto const d_strings = cudf::column_device_view::create(input.parent(), stream);

  auto const first_offset  = (input.offset() == 0) ? 0L
                                                   : cudf::strings::detail::get_offset_value(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset   = (input.offset() == 0 && input.size() == input.offsets().size() - 1)
                               ? static_cast<int64_t>(input.chars_size(stream))
                               : cudf::strings::detail::get_offset_value(
                                 input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size    = last_offset - first_offset;
  auto const d_input_chars = input.chars_begin(stream) + first_offset;

  rmm::device_uvector<int8_t> d_spaces(chars_size, stream);  // identifies non-merged pairs
  // used for various purposes below: unpairable-offsets, pair ranks, separator insert positions
  rmm::device_uvector<int64_t> d_working(chars_size, stream);

  auto const chars_begin = cuda::counting_iterator<int64_t>{0};
  auto const chars_end   = cuda::counting_iterator<int64_t>{chars_size};

  {
    // this kernel locates unpairable sections of strings to create artificial string row
    // boundaries; the boundary values are recorded as offsets in d_up_offsets
    auto const d_up_offsets = d_working.data();  // store unpairable offsets here
    auto const mp_map = get_bpe_merge_pairs_impl(merge_pairs)->get_mp_table_ref();  // lookup table
    auto const d_chars_span = cudf::device_span<char const>(d_input_chars, chars_size);
    auto up_fn = bpe_unpairable_offsets_fn<decltype(mp_map)>{d_chars_span, first_offset, mp_map};
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      chars_begin,
                      chars_end,
                      d_up_offsets,
                      up_fn);
    auto const up_end =  // remove all but the unpairable offsets
      thrust::remove(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     d_up_offsets,
                     d_up_offsets + chars_size,
                     0L);
    auto const unpairables = cuda::std::distance(d_up_offsets, up_end);  // number of unpairables

    // new string boundaries created by combining unpairable offsets with the existing offsets
    auto tmp_offsets = rmm::device_uvector<int64_t>(unpairables + input.size() + 1, stream);
    auto input_offsets =
      cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());
    thrust::merge(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  input_offsets,
                  input_offsets + input.size() + 1,
                  d_up_offsets,
                  up_end,
                  tmp_offsets.begin());
    // remove any adjacent duplicate offsets (i.e. empty or null rows)
    auto const offsets_end =
      thrust::unique(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     tmp_offsets.begin(),
                     tmp_offsets.end());
    auto const offsets_total =
      static_cast<cudf::size_type>(cuda::std::distance(tmp_offsets.begin(), offsets_end));
    tmp_offsets.resize(offsets_total, stream);

    // temp column created with the merged offsets and the original chars data
    auto const col_offsets = cudf::column_view(cudf::device_span<int64_t const>(tmp_offsets));
    auto const tmp_size    = offsets_total - 1;
    auto const tmp_input   = cudf::column_view(
      input.parent().type(), tmp_size, input.chars_begin(stream), nullptr, 0, 0, {col_offsets});
    auto const d_tmp_strings = cudf::column_device_view::create(tmp_input, stream);

    // launch the byte-pair-encoding kernel on the temp column
    rmm::device_uvector<int8_t> d_rerank(chars_size, stream);  // more working memory;
    rmm::device_uvector<cudf::size_type> d_ranks(chars_size, stream);
    auto const pair_map = get_bpe_merge_pairs_impl(merge_pairs)->get_merge_pairs_ref();
    bpe_parallel_fn<decltype(pair_map)><<<tmp_size, block_size, 0, stream.value()>>>(
      *d_tmp_strings, d_input_chars, pair_map, d_spaces.data(), d_ranks.data(), d_rerank.data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // compute the output sizes
  auto output_sizes = rmm::device_uvector<cudf::size_type>(input.size(), stream);
  bpe_finalize<<<input.size(), block_size, 0, stream.value()>>>(
    *d_strings, d_input_chars, d_spaces.data(), output_sizes.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  // convert sizes to offsets in-place
  auto [offsets, bytes] = cudf::strings::detail::make_offsets_child_column(
    output_sizes.begin(), output_sizes.end(), stream, mr);

  // build the output: inserting separators to the input character data
  rmm::device_uvector<char> chars(bytes, stream, mr);
  auto d_chars = chars.data();

  auto const d_inserts     = d_working.data();  // stores the insert positions
  auto offsets_at_non_zero = [d_spaces = d_spaces.data()] __device__(auto idx) -> bool {
    return d_spaces[idx] > 0;  // separator to be inserted here
  };
  auto const copy_end =
    cudf::detail::copy_if(chars_begin + 1, chars_end, d_inserts, offsets_at_non_zero, stream);

  // this will insert the single-byte separator into positions specified in d_inserts
  auto const sep_char = cuda::constant_iterator<char>(separator.to_string(stream)[0]);
  thrust::merge_by_key(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       d_inserts,      // where to insert separator byte
                       copy_end,       //
                       chars_begin,    // all indices
                       chars_end,      //
                       sep_char,       // byte to insert
                       d_input_chars,  // original data
                       cuda::make_discard_iterator(),
                       d_chars);  // result

  return cudf::make_strings_column(input.size(),
                                   std::move(offsets),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<cudf::column> byte_pair_encoding(cudf::strings_column_view const& input,
                                                 bpe_merge_pairs const& merges_table,
                                                 cudf::string_scalar const& separator,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::byte_pair_encoding(input, merges_table, separator, stream, mr);
}

}  // namespace nvtext

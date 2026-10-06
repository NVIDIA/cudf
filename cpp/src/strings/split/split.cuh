/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "strings/positions.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/strings/detail/split_utils.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>

#include <type_traits>

namespace cudf::strings::detail {

/**
 * @brief Returns `true` if the byte at `idx` is the start of the delimiter
 */
struct string_delimiter_fn {
  __device__ bool operator()(int64_t idx) const
  {
    if (idx + d_delimiter.size_bytes() > chars_bytes) { return false; }
    return d_delimiter.compare(d_chars + idx, d_delimiter.size_bytes()) == 0;
  }
  cudf::string_view d_delimiter;
  int64_t chars_bytes{};
  char const* d_chars{};
};

// Keep one host entry point per fixed predicate so split_part.cu owns the count/select kernels
// instead of emitting duplicate instantiations in each caller TU.
rmm::device_uvector<int64_t> find_string_delimiter_positions(strings_column_view const& input,
                                                             cudf::string_view delimiter,
                                                             cuda::stream_ref stream);

rmm::device_uvector<int64_t> find_whitespace_delimiter_positions(strings_column_view const& input,
                                                                 cuda::stream_ref stream);

/**
 * @brief Returns `true` if the byte at `idx` is a whitespace character
 */
struct whitespace_delimiter_fn {
  __device__ bool operator()(int64_t idx) const
  {
    return idx < chars_bytes && static_cast<u_char>(d_chars[idx]) <= ' ';
  }
  int64_t chars_bytes{};
  char const* d_chars{};
};

/**
 * @brief Get the offsets for a string in the strings column
 *
 * @param d_strings The strings column
 * @param idx The index of the string to retrieve
 * @return The start and end offsets for the string
 */
__device__ __inline__ auto get_string_offsets(column_device_view const& d_strings, size_type idx)
{
  auto const offsets = d_strings.child(d_strings.offsets_column_index);
  auto const itr     = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
  auto const index   = d_strings.offset() + idx;
  return cuda::std::make_pair(itr[index], itr[index + 1]);
}

/**
 * @brief Base class for delimiter-based tokenizers
 *
 * These are common methods used by both split and rsplit tokenizer functors.
 *
 * The Derived class is required to implement the `process_tokens` function.
 */
template <typename Derived>
struct base_split_tokenizer {
  /**
   * @brief This counts the tokens for strings that contain delimiters
   *
   * Counting tokens is the same regardless if counting from the left
   * or from the right. This logic counts from the left which is simpler.
   * The count will be truncated appropriately to the max_tokens value.
   *
   * @param idx Index of input string
   * @param d_positions Start positions of all the delimiters
   * @param d_delimiter_offsets Offsets per string to delimiters in d_positions
   */
  __device__ size_type count_tokens(size_type idx,
                                    int64_t const* d_positions,
                                    cudf::detail::input_offsetalator d_delimiter_offsets) const
  {
    if (d_strings.is_null(idx)) { return 0; }

    auto const d_str_offsets = get_string_offsets(d_strings, idx);
    auto const delimiters =
      cudf::device_span<int64_t const>(d_positions + d_delimiter_offsets[idx],
                                       d_delimiter_offsets[idx + 1] - d_delimiter_offsets[idx]);

    size_type token_count = 1;  // all strings will have at least one token
    auto last_pos         = d_str_offsets.first - delimiter_size;
    for (auto d_pos : delimiters) {
      // delimiter must fit within the string and overlapping delimiters are ignored
      if (((d_pos + delimiter_size) <= d_str_offsets.second) &&
          ((d_pos - last_pos) >= delimiter_size)) {
        ++token_count;
        last_pos = d_pos;
      }
    }
    // number of tokens is capped to max_tokens
    return ((max_tokens > 0) && (token_count > max_tokens)) ? max_tokens : token_count;
  }

  /**
   * @brief This will create tokens around each delimiter honoring the string boundaries
   * in which the delimiter resides
   *
   * Each token is placed in `d_all_tokens` so they align consecutively
   * with other tokens for the same output column.
   *
   * The actual token extraction is performed in the subclass process_tokens() function.
   *
   * @param idx Index of the string to tokenize
   * @param d_tokens_offsets Token offsets for each string
   * @param d_positions The beginning byte position of each delimiter
   * @param d_delimiter_offsets Offsets to d_positions to each delimiter set per string
   * @param d_all_tokens All output tokens for the strings column
   */
  __device__ void get_tokens(size_type idx,
                             cudf::detail::input_offsetalator const d_tokens_offsets,
                             int64_t const* d_positions,
                             cudf::detail::input_offsetalator d_delimiter_offsets,
                             string_index_pair* d_all_tokens) const
  {
    if (d_strings.is_null(idx)) { return; }

    auto const d_tokens =  // this string's tokens output
      cudf::device_span<string_index_pair>(d_all_tokens + d_tokens_offsets[idx],
                                           d_tokens_offsets[idx + 1] - d_tokens_offsets[idx]);

    auto const d_str_offsets = get_string_offsets(d_strings, idx);

    auto const delimiters =
      cudf::device_span<int64_t const>(d_positions + d_delimiter_offsets[idx],
                                       d_delimiter_offsets[idx + 1] - d_delimiter_offsets[idx]);

    auto& derived = static_cast<Derived const&>(*this);
    derived.process_tokens(d_str_offsets.first, d_str_offsets.second, delimiters, d_tokens);
  }

  base_split_tokenizer(column_device_view const& d_strings,
                       size_type delimiter_size,
                       size_type max_tokens)
    : d_strings(d_strings), delimiter_size(delimiter_size), max_tokens(max_tokens)
  {
  }

 protected:
  column_device_view const d_strings;  // strings to split
  size_type delimiter_size;            // size of the delimiter
  size_type max_tokens;                // maximum number of tokens to identify
};

/**
 * @brief The tokenizer functions for forward splitting
 */
struct split_tokenizer_fn : base_split_tokenizer<split_tokenizer_fn> {
  /**
   * @brief This will create tokens around each delimiter honoring the string boundaries
   *
   * The tokens are processed from the beginning of each string ignoring overlapping
   * delimiters and honoring the `max_tokens` value.
   *
   * @param pos_begin Start position of the string in the character data
   * @param pos_end End position of the string in the character data
   * @param d_delimiters Positions of delimiters for this string
   * @param d_tokens Output vector to store tokens for this string
   */
  __device__ void process_tokens(int64_t pos_begin,
                                 int64_t pos_end,
                                 device_span<int64_t const> d_delimiters,
                                 device_span<string_index_pair> d_tokens) const
  {
    auto const base_ptr = d_strings.head<char>();  // d_delimiters, pos_begin/end based on this ptr
    auto const token_count = static_cast<size_type>(d_tokens.size());
    auto const delim_count = static_cast<size_type>(d_delimiters.size());

    // build the index-pair of each token for this string
    size_type token_idx = 0;
    auto last_pos       = pos_begin - delimiter_size;
    for (auto di = 0; di < delim_count && token_idx < token_count; ++di) {
      auto const d_pos = d_delimiters[di];
      if (((d_pos + delimiter_size) > pos_end) || ((d_pos - last_pos) < delimiter_size)) {
        continue;
      }
      auto const end_pos = (token_idx + 1 < token_count) ? d_pos : pos_end;

      // store the token into the output vector
      last_pos += delimiter_size;
      d_tokens[token_idx] = string_index_pair{base_ptr + last_pos, end_pos - last_pos};

      last_pos = d_pos;
      ++token_idx;
    }
    // include anything leftover
    if (token_idx < token_count) {
      last_pos += delimiter_size;
      d_tokens[token_idx] = base_ptr ? string_index_pair{base_ptr + last_pos, pos_end - last_pos}
                                     : string_index_pair{"", 0};
    }
  }

  split_tokenizer_fn(column_device_view const& d_strings,
                     size_type delimiter_size,
                     size_type max_tokens)
    : base_split_tokenizer(d_strings, delimiter_size, max_tokens)
  {
  }
};

/**
 * @brief The tokenizer functions for backwards splitting
 *
 * Same as split_tokenizer_fn except delimiters are searched from the end of each string.
 */
struct rsplit_tokenizer_fn : base_split_tokenizer<rsplit_tokenizer_fn> {
  /**
   * @brief This will create tokens around each delimiter honoring the string boundaries
   *
   * The tokens are processed from the end of each string ignoring overlapping
   * delimiters and honoring the `max_tokens` value.
   *
   * @param pos_begin Start position of the string in the character data
   * @param pos_end End position of the string in the character data
   * @param d_delimiters Positions of delimiters for this string
   * @param d_tokens Output vector to store tokens for this string
   */
  __device__ void process_tokens(int64_t pos_begin,
                                 int64_t pos_end,
                                 device_span<int64_t const> d_delimiters,
                                 device_span<string_index_pair> d_tokens) const
  {
    auto const base_ptr = d_strings.head<char>();  // d_delimiters, pos_begin/end based on this ptr
    auto const token_count = static_cast<size_type>(d_tokens.size());
    auto const delim_count = static_cast<size_type>(d_delimiters.size());

    // build the index-pair of each token for this string
    auto last_pos       = pos_end;
    size_type token_idx = 0;
    for (auto d = delim_count - 1; d >= 0 && token_idx < token_count; --d) {  // read right-to-left
      auto const d_pos = d_delimiters[d];
      if (((d_pos + delimiter_size) > pos_end) || ((last_pos - d_pos) < delimiter_size)) {
        continue;
      }
      auto const start_pos = (token_idx + 1 < token_count) ? d_pos + delimiter_size : pos_begin;

      // store the token into the output vector right-to-left
      d_tokens[token_count - token_idx - 1] =
        string_index_pair{base_ptr + start_pos, last_pos - start_pos};

      last_pos = d_pos;
      ++token_idx;
    }
    // include anything leftover (rightover?)
    if (token_idx < token_count) {
      d_tokens[0] = base_ptr ? string_index_pair{base_ptr + pos_begin, last_pos - pos_begin}
                             : string_index_pair{"", 0};
    }
  }

  rsplit_tokenizer_fn(column_device_view const& d_strings,
                      size_type delimiter_size,
                      size_type max_tokens)
    : base_split_tokenizer(d_strings, delimiter_size, max_tokens)
  {
  }
};

/**
 * @brief Base class for whitespace-delimited tokenizing
 *
 * Generally, consecutive whitespace delimiters are treated as a single delimiter
 * and so empty strings are not produced.
 */
template <typename Derived>
struct base_ws_split_tokenizer {
  __device__ size_type count_tokens(size_type idx,
                                    int64_t const* d_positions,
                                    cudf::detail::input_offsetalator d_delimiter_offsets) const
  {
    if (d_strings.is_null(idx)) { return 0; }

    auto const d_str_offsets = get_string_offsets(d_strings, idx);

    auto const delimiters =
      cudf::device_span<int64_t const>(d_positions + d_delimiter_offsets[idx],
                                       d_delimiter_offsets[idx + 1] - d_delimiter_offsets[idx]);
    if (delimiters.size() == (d_str_offsets.second - d_str_offsets.first)) { return 0; }
    if (delimiters.empty()) { return 1; }

    size_type token_count = (delimiters.front() != d_str_offsets.first);
    for (std::size_t i = 0; i < delimiters.size(); ++i) {
      auto const d_pos  = delimiters[i];
      auto const d_next = (i + 1 < delimiters.size()) ? delimiters[i + 1] : d_str_offsets.second;
      token_count += (d_next - d_pos > 1);
    }
    // number of tokens is capped to max_tokens
    return ((max_tokens > 0) && (token_count > max_tokens)) ? max_tokens : token_count;
  }

  __device__ void get_tokens(size_type idx,
                             cudf::detail::input_offsetalator const d_tokens_offsets,
                             int64_t const* d_positions,
                             cudf::detail::input_offsetalator const d_delimiter_offsets,
                             string_index_pair* d_all_tokens) const
  {
    if (d_strings.is_null(idx)) { return; }

    auto const d_tokens =  // this string's tokens output
      cudf::device_span<string_index_pair>(d_all_tokens + d_tokens_offsets[idx],
                                           d_tokens_offsets[idx + 1] - d_tokens_offsets[idx]);
    if (d_tokens.empty()) { return; }

    auto const d_str_offsets = get_string_offsets(d_strings, idx);

    auto const delimiters =
      cudf::device_span<int64_t const>(d_positions + d_delimiter_offsets[idx],
                                       d_delimiter_offsets[idx + 1] - d_delimiter_offsets[idx]);

    auto const str_bytes = d_str_offsets.second - d_str_offsets.first;
    if (delimiters.size() == str_bytes) {
      d_tokens[0] = string_index_pair{nullptr, 0};
      return;
    }
    if (delimiters.empty()) {
      auto const base_ptr = d_strings.head<char>();
      d_tokens[0]         = string_index_pair{base_ptr + d_str_offsets.first, str_bytes};
      return;
    }

    auto& derived = static_cast<Derived const&>(*this);
    derived.process_tokens(d_str_offsets.first, d_str_offsets.second, delimiters, d_tokens);
  }

  base_ws_split_tokenizer(column_device_view const& d_strings, size_type max_tokens)
    : d_strings(d_strings), max_tokens(max_tokens)
  {
  }

 protected:
  column_device_view d_strings;  // strings to split
  size_type max_tokens;          // maximum number of tokens to identify
};

/**
 * @brief The forward splitting whitespace tokenizer
 *
 * Note that this processes only the delimiter positions already identified
 * and so could technically handle identifying tokens between non-whitespace
 * consecutive delimiters as well.
 */
struct split_ws_tokenizer_fn : base_ws_split_tokenizer<split_ws_tokenizer_fn> {
  __device__ void process_tokens(int64_t pos_begin,
                                 int64_t pos_end,
                                 device_span<int64_t const> delimiters,
                                 device_span<string_index_pair> d_tokens) const
  {
    auto const base_ptr = d_strings.head<char>();  // d_delimiters, pos_begin/end based on this ptr
    auto const token_count = static_cast<size_type>(d_tokens.size());

    // build the index-pair of each token for this string
    size_type token_idx = 0;
    auto last_pos       = pos_begin;
    for (size_t di = 0; di < delimiters.size() && token_idx < token_count; ++di) {
      auto const d_pos = delimiters[di];
      if (last_pos == d_pos) {
        ++last_pos;
        continue;
      }
      auto const end_pos =
        (token_count < max_tokens) || (token_idx + 1 < token_count) ? d_pos : pos_end;
      d_tokens[token_idx++] = string_index_pair{base_ptr + last_pos, end_pos - last_pos};

      last_pos = d_pos + 1;
    }
    // include anything leftover
    if (token_idx < token_count) {
      d_tokens[token_idx] = string_index_pair{base_ptr + last_pos, pos_end - last_pos};
    }
  }

  split_ws_tokenizer_fn(column_device_view const& d_strings, size_type max_tokens)
    : base_ws_split_tokenizer(d_strings, max_tokens)
  {
  }
};

/**
 * @brief The backward splitting whitespace tokenizer
 *
 * Same as split_ws_tokenizer_fn except delimiters are searched from the end of each string.
 */
struct rsplit_ws_tokenizer_fn : base_ws_split_tokenizer<rsplit_ws_tokenizer_fn> {
  __device__ void process_tokens(int64_t pos_begin,
                                 int64_t pos_end,
                                 device_span<int64_t const> delimiters,
                                 device_span<string_index_pair> d_tokens) const
  {
    auto const base_ptr = d_strings.head<char>();  // d_delimiters, pos_begin/end based on this ptr
    auto const token_count = static_cast<size_type>(d_tokens.size());
    auto const delim_count = static_cast<size_type>(delimiters.size());

    // build the index-pair of each token for this string
    auto last_pos       = pos_end;
    size_type token_idx = 0;
    for (auto di = delim_count - 1; di >= 0 && token_idx < token_count; --di) {
      auto const d_pos = delimiters[di];
      if (last_pos == d_pos + 1) {
        --last_pos;
        continue;
      }
      // store the token into the output vector right-to-left
      auto const start_pos =
        (token_count < max_tokens) || (token_idx + 1 < token_count) ? d_pos + 1 : pos_begin;
      d_tokens[token_count - token_idx - 1] =
        string_index_pair{base_ptr + start_pos, last_pos - start_pos};

      last_pos = d_pos;
      ++token_idx;
    }
    // include anything leftover (rightover?)
    if (token_idx < token_count) {
      d_tokens[0] = string_index_pair{base_ptr + pos_begin, last_pos - pos_begin};
    }
  }

  rsplit_ws_tokenizer_fn(column_device_view const& d_strings, size_type max_tokens)
    : base_ws_split_tokenizer(d_strings, max_tokens)
  {
  }
};

// Per-string kernel outperforms the global-scan split_helper for short strings;
// fall back to split_helper above this average byte length per non-null string.
// This value was determined using the split benchmarks by varying the row width
// and the row count using an RTX A6000.
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 120;

/**
 * @brief Count the number of delimiters in a strings column
 *
 * @tparam DelimiterFn Functor for locating delimiters
 * @tparam block_size Number of threads per block
 * @tparam bytes_per_thread Number of bytes processed per thread
 *
 * @param delimiter_fn Functor called on each byte to check for delimiters
 * @param chars_bytes Number of bytes in the strings column
 * @param d_output Result of the count
 */
template <typename DelimiterFn, int64_t block_size, size_type bytes_per_thread>
CUDF_KERNEL void count_delimiters_kernel(DelimiterFn delimiter_fn,
                                         int64_t chars_bytes,
                                         int64_t* d_output)
{
  auto const idx      = cudf::detail::grid_1d::global_thread_id();
  auto const byte_idx = static_cast<int64_t>(idx) * bytes_per_thread;
  auto const lane_idx = static_cast<cudf::size_type>(threadIdx.x);

  using block_reduce = cub::BlockReduce<int64_t, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;

  int64_t count = 0;
  // each thread processes multiple bytes
  for (auto i = byte_idx; (i < (byte_idx + bytes_per_thread)) && (i < chars_bytes); ++i) {
    count += delimiter_fn(i);
  }
  auto const total = block_reduce(temp_storage).Reduce(count, cuda::std::plus());

  if ((lane_idx == 0) && (total > 0)) {
    cuda::atomic_ref<int64_t, cuda::thread_scope_device> ref{*d_output};
    ref.fetch_add(total, cuda::std::memory_order_relaxed);
  }
}

// Declaration-only owners avoid instantiating split_per_row_impl for each caller's
// extractor-building lambda. Both helpers are explicitly instantiated in split_record.cu,
// so table and record splits share count/scan/extract kernels while direction stays compile-time.
template <bool Forward>
std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_per_row(
  column_device_view const& d_strings,
  string_view delimiter,
  size_type max_tokens,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

template <bool Forward>
std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_ws_per_row(
  column_device_view const& d_strings,
  size_type max_tokens,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

// These non-template overloads prevent callers from emitting the same fixed helper kernels in
// multiple TUs. split.cu owns the forward explicit-delimiter variant; split_record.cu owns the
// reverse and whitespace variants. Keep definitions out of this header to preserve that ownership.
std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_helper(
  strings_column_view const& input,
  rsplit_tokenizer_fn tokenizer,
  string_delimiter_fn delimiter_fn,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_helper(
  strings_column_view const& input,
  split_ws_tokenizer_fn tokenizer,
  whitespace_delimiter_fn delimiter_fn,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_helper(
  strings_column_view const& input,
  rsplit_ws_tokenizer_fn tokenizer,
  whitespace_delimiter_fn delimiter_fn,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_helper(
  strings_column_view const& input,
  split_tokenizer_fn tokenizer,
  string_delimiter_fn delimiter_fn,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Helper function used by split/rsplit and split_record/rsplit_record
 *
 * This function returns all the token/split positions within the input column as processed by
 * the given tokenizer. It also returns the offsets for each set of tokens identified per string.
 *
 * @tparam Tokenizer Type of the tokenizer object
 * @tparam DelimiterFn Functor for locating delimiters
 *
 * @param input The input column of strings to split
 * @param tokenizer Object used for counting and identifying delimiters and tokens
 * @param delimiter_fn Functor called on each byte to check for delimiters
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned objects' device memory
 * @return Token offsets and a vector of string indices
 */
template <typename Tokenizer, typename DelimiterFn>
std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_helper(
  strings_column_view const& input,
  Tokenizer tokenizer,
  DelimiterFn delimiter_fn,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto delimiter_positions = [&] {
    if constexpr (std::is_same_v<DelimiterFn, string_delimiter_fn>) {
      return find_string_delimiter_positions(input, delimiter_fn.d_delimiter, stream);
    } else if constexpr (std::is_same_v<std::remove_cvref_t<DelimiterFn>,
                                        whitespace_delimiter_fn>) {
      return find_whitespace_delimiter_positions(input, stream);
    } else {
      auto [first_offset, last_offset] = get_first_and_last_offset(input, stream);
      auto const chars_bytes           = last_offset - first_offset;
      delimiter_fn.d_chars             = input.chars_begin(stream) + first_offset;
      delimiter_fn.chars_bytes         = chars_bytes;

      cudf::detail::device_scalar<int64_t> d_count(
        0, stream, cudf::get_current_device_resource_ref());
      if (chars_bytes > 0) {
        constexpr int64_t block_size         = 512;
        constexpr size_type bytes_per_thread = 4;
        auto const num_blocks                = util::div_rounding_up_safe(
          util::div_rounding_up_safe(chars_bytes, static_cast<int64_t>(bytes_per_thread)),
          block_size);
        count_delimiters_kernel<DelimiterFn, block_size, bytes_per_thread>
          <<<num_blocks, block_size, 0, stream.get()>>>(delimiter_fn, chars_bytes, d_count.data());
        CUDF_CUDA_TRY(cudaGetLastError());
      }
      auto positions = rmm::device_uvector<int64_t>(d_count.value(stream), stream);
      cudf::detail::copy_if_async(cuda::counting_iterator<int64_t>{0},
                                  cuda::counting_iterator<int64_t>{chars_bytes},
                                  positions.begin(),
                                  delimiter_fn,
                                  stream);
      return positions;
    }
  }();

  // create a vector of offsets to each string's delimiter set within delimiter_positions
  auto const delimiter_offsets =
    create_offsets_from_positions(input, delimiter_positions, stream, mr);
  auto const d_delimiter_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(delimiter_offsets->view());

  // compute the number of tokens per string
  auto token_counts    = rmm::device_uvector<size_type>(input.size(), stream);
  auto d_positions     = delimiter_positions.data();
  auto const zero_iter = cuda::counting_iterator<size_type>{0};
  thrust::transform(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    zero_iter,
    zero_iter + input.size(),
    token_counts.begin(),
    [tokenizer, d_positions, d_delimiter_offsets] __device__(size_type idx) -> size_type {
      return tokenizer.count_tokens(idx, d_positions, d_delimiter_offsets);
    });

  // create offsets from the counts for return to the caller
  auto [offsets, total_tokens] =
    cudf::detail::make_offsets_child_column(token_counts.begin(), token_counts.end(), stream, mr);
  auto const d_tokens_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // build a vector of all the token positions for all the strings
  auto tokens   = rmm::device_uvector<string_index_pair>(total_tokens, stream);
  auto d_tokens = tokens.data();
  auto get_tokens_fn =
    [tokenizer, d_tokens_offsets, d_positions, d_delimiter_offsets, d_tokens] __device__(
      size_type idx) {
      tokenizer.get_tokens(idx, d_tokens_offsets, d_positions, d_delimiter_offsets, d_tokens);
    };
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     zero_iter,
                     input.size(),
                     get_tokens_fn);

  return std::make_pair(std::move(offsets), std::move(tokens));
}

}  // namespace cudf::strings::detail

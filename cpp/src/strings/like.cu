/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/strings/contains.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

/**
 * @brief Threshold to decide on using string or warp parallel functions.
 *
 * If the average byte length of a string in a column exceeds this value then
 * the warp-parallel function is used to compute the output sizes.
 * Otherwise, a regular row-per-thread function is used.
 *
 * This value was found using benchmark results.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 72;

constexpr char multi_wildcard  = '%';
constexpr char single_wildcard = '_';

/**
 * @brief Like functor for matching a pattern to a string
 *
 * This function performs the like pattern on a single string
 * per thread. The patterns_itr may provide a unique pattern
 * per string as well.
 */
template <typename PatternIterator>
struct like_fn {
  column_device_view const d_strings;
  PatternIterator const patterns_itr;
  string_view const d_escape;

  like_fn(column_device_view d_strings, PatternIterator patterns_itr, string_view d_escape)
    : d_strings{d_strings}, patterns_itr{patterns_itr}, d_escape{d_escape}
  {
  }

  __device__ bool operator()(size_type const idx)
  {
    if (d_strings.is_null(idx)) return false;
    auto const d_str     = d_strings.element<string_view>(idx);
    auto const d_pattern = patterns_itr[idx];

    // incrementing by bytes instead of character improves performance 10-20%
    auto target_itr  = d_str.data();
    auto pattern_itr = d_pattern.begin();

    auto const target_end  = target_itr + d_str.size_bytes();
    auto const pattern_end = d_pattern.end();
    auto const esc_char    = d_escape.empty() ? 0 : d_escape.data()[0];

    auto last_target_itr  = target_end;
    auto last_pattern_itr = pattern_end;

    bool result = true;
    while (true) {
      // walk through the pattern and check against the current character
      while (pattern_itr < pattern_end) {
        auto const escaped = *pattern_itr == esc_char;
        auto const pattern_char =
          escaped && (pattern_itr + 1 < pattern_end) ? *(++pattern_itr) : *pattern_itr;

        if (escaped || (pattern_char != multi_wildcard)) {
          size_type char_width = 0;
          // check match with the current character
          result = (target_itr != target_end);
          if (result) {
            if (escaped || pattern_char != single_wildcard) {
              char_utf8 target_char = 0;
              // retrieve the target character to compare with the current pattern_char
              char_width = to_char_utf8(target_itr, target_char);
              result     = (pattern_char == target_char);
            }
          }
          if (!result) { break; }
          ++pattern_itr;
          target_itr += char_width ? char_width : bytes_in_utf8_byte(*target_itr);
        } else {
          // process wildcard '%'
          result = true;
          ++pattern_itr;
          if (pattern_itr == pattern_end) {  // pattern ends with '%' so we are done
            target_itr = target_end;
            break;
          }
          // save positions
          last_pattern_itr = pattern_itr;
          last_target_itr  = target_itr;
        }  // next pattern character
      }

      if (result && (target_itr == target_end)) { break; }  // success

      result = false;
      // check if exhausted either the pattern or the target string
      if (last_pattern_itr == pattern_end || last_target_itr == target_end) { break; }

      // restore saved positions
      pattern_itr = last_pattern_itr;
      last_target_itr += bytes_in_utf8_byte(*last_target_itr);
      target_itr = last_target_itr;
    }
    return result;
  }
};

/**
 * @brief Compare a literal part of the pattern against a target string
 *
 * The literal part of a pattern is between two multi-wildcards.
 * This handles the escape character and the single-wildcard character.
 */
__device__ cuda::std::pair<bool, size_type> compare_literal(char const* target_itr,
                                                            char const* target_end,
                                                            char const esc_char,
                                                            char const* pattern_itr,
                                                            char const* pattern_end)
{
  bool result           = true;
  auto target_itr_start = target_itr;
  while (result && pattern_itr < pattern_end) {
    auto const escaped = *pattern_itr == esc_char;
    auto const pattern_char =
      escaped && (pattern_itr + 1 < pattern_end) ? *(++pattern_itr) : *pattern_itr;
    ++pattern_itr;
    if (!escaped && pattern_char == single_wildcard && target_itr < target_end) {
      target_itr += bytes_in_utf8_byte(*target_itr);
      continue;
    }
    auto const target_char = target_itr < target_end ? *target_itr++ : 0;

    result = (pattern_char == target_char);
  }
  return {result, cuda::std::distance(target_itr_start, target_itr)};
}

/**
 * @brief Like function for a single string and pattern
 *
 * This is the warp-parallel version of the like function.
 * It is only used for longer strings.
 *
 * @param d_strings The input strings column
 * @param pattern_itr The pattern to match
 * @param d_escape The escape character
 * @param results The output of boolean values
 */
template <typename PatternIterator>
CUDF_KERNEL void like_kernel(column_device_view d_strings,
                             PatternIterator pattern_itr,
                             string_view d_escape,
                             bool* results)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();

  // Each warp handles one string
  auto const str_idx = tid / cudf::detail::warp_size;
  if (str_idx >= d_strings.size()) { return; }
  if (d_strings.is_null(str_idx)) { return; }

  namespace cg    = cooperative_groups;
  auto const warp = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());

  auto const d_str      = d_strings.element<string_view>(str_idx);
  auto target_itr       = d_str.data();
  auto const target_end = target_itr + d_str.size_bytes();

  auto const d_pattern     = pattern_itr[str_idx];
  auto const pattern_begin = d_pattern.data();
  auto const pattern_end   = pattern_begin + d_pattern.size_bytes();
  auto const esc_char      = d_escape.empty() ? 0 : d_escape.data()[0];
  auto const lane_idx      = warp.thread_rank();

  auto count_wc_fn = [p = pattern_begin, esc_char](auto const idx) {
    return (p[idx] == multi_wildcard) && (idx == 0 || p[idx - 1] != esc_char);
  };
  auto find_next_wc = [pattern_begin, pattern_end, esc_char](size_type idx) {
    auto itr = pattern_begin + idx;
    while (itr < pattern_end) {
      if (*itr == multi_wildcard && (*(itr - 1) != esc_char)) { return idx; }
      ++itr;
      ++idx;
    }
    return idx;
  };

  auto const itr_zero = cuda::counting_iterator<size_type>{0};
  auto const itr_size = cuda::counting_iterator<size_type>{d_pattern.size_bytes()};
  auto const wcs_size =
    esc_char == multi_wildcard ? 0 : thrust::count_if(thrust::seq, itr_zero, itr_size, count_wc_fn);

  auto wcs_idx = 0;
  auto next_wc = 0;
  bool result  = true;
  if (wcs_size == 0 || *pattern_begin != multi_wildcard) {
    // pattern does not start with a wildcard; check for literal match
    next_wc       = wcs_size > 0 ? find_next_wc(0) : d_pattern.size_bytes();
    auto out_size = 0;
    if (lane_idx == 0) {
      auto const end = next_wc + pattern_begin;
      cuda::std::tie(result, out_size) =
        compare_literal(target_itr, target_end, esc_char, pattern_begin, end);
    }
    out_size = warp.shfl(out_size, 0);  // copy out_size to all threads in the warp
    target_itr += out_size;
  }

  result = warp.shfl(result, 0);  // copy result to all threads in the warp

  // process literals between wildcards
  while (result && ((wcs_idx + 1) < wcs_size)) {
    auto const begin = next_wc + pattern_begin + 1;
    next_wc          = find_next_wc(next_wc + 1);
    auto const end   = next_wc + pattern_begin;
    ++wcs_idx;
    if (cuda::std::distance(begin, end) == 0) { continue; }

    auto const not_found = d_str.size_bytes();
    auto find_idx        = not_found;
    while (find_idx == not_found && target_itr < target_end) {
      size_type tid_find = find_idx;
      size_type out_size = 0;
      auto itr           = target_itr + lane_idx;  // each thread starts at a different position
      if ((itr < target_end) && is_begin_utf8_char(*itr)) {
        bool cmp                      = false;
        cuda::std::tie(cmp, out_size) = compare_literal(itr, target_end, esc_char, begin, end);
        if (cmp) { tid_find = lane_idx; }
      }
      warp.sync();
      find_idx = cg::reduce(warp, tid_find, cg::less<size_type>{});
      out_size = warp.shfl(out_size, find_idx);
      target_itr += (find_idx == not_found) ? cudf::detail::warp_size : find_idx + out_size;
    }

    // all threads have the same find_idx here
    result = (find_idx != not_found);
    warp.sync();
  }

  // check for last literal match
  if (result && wcs_size > 0) {
    auto const begin = pattern_begin + next_wc + 1;
    if (begin < pattern_end) {
      auto itr  = begin;
      auto size = size_type{0};
      while (itr < pattern_end) {  // count remaining accounting for any escape characters
        auto const chr   = (*itr == esc_char) && (itr + 1 < pattern_end) ? *(++itr) : *itr;
        auto const bytes = bytes_in_utf8_byte(chr);
        itr += bytes;
        size += (bytes > 0);  // counting whole characters and not bytes
      }
      if (target_itr + size <= target_end) {  // quick max estimate
        if (lane_idx == 0) {
          target_itr = target_end;
          while (size > 0 && target_itr > d_str.data()) {
            size -= is_begin_utf8_char(*(--target_itr));
          }
          cuda::std::tie(result, size) =
            compare_literal(target_itr, target_end, esc_char, begin, pattern_end);
          target_itr += size;
        }
      } else {
        result = false;  // literal will not match the remaining string
      }
    } else {
      target_itr = target_end;  // pattern ends with a multi-wildcard
    }
  }

  // result is good if everything matches and we have exhausted the target string
  if (lane_idx == 0) { results[str_idx] = result && target_itr == target_end; }
}

template <typename PatternIterator>
std::unique_ptr<column> like(strings_column_view const& input,
                             PatternIterator const patterns_itr,
                             string_view const& d_escape,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto results = make_numeric_column(data_type{type_id::BOOL8},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  if (input.is_empty()) { return results; }

  auto const d_strings = column_device_view::create(input.parent(), stream);

  // string per thread for smaller strings
  auto [first_offset, last_offset] = get_first_and_last_offset(input, stream);
  if ((input.size() == input.null_count()) ||
      ((last_offset - first_offset) / (input.size() - input.null_count())) <
        AVG_CHAR_BYTES_THRESHOLD) {
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{input.size()},
                      results->mutable_view().data<bool>(),
                      like_fn{*d_strings, patterns_itr, d_escape});
  } else {
    // warp-parallel for longer strings
    constexpr thread_index_type block_size = 512;
    constexpr thread_index_type warp_size  = cudf::detail::warp_size;
    auto const grid = cudf::detail::grid_1d(input.size() * warp_size, block_size);
    like_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream>>>(
      *d_strings, patterns_itr, d_escape, results->mutable_view().data<bool>());
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  results->set_null_count(input.null_count());
  return results;
}

}  // namespace

std::unique_ptr<column> like(strings_column_view const& input,
                             string_scalar const& pattern,
                             string_scalar const& escape_character,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(pattern.is_valid(stream), "Parameter pattern must be valid", std::invalid_argument);
  CUDF_EXPECTS(escape_character.is_valid(stream),
               "Parameter escape_character must be valid",
               std::invalid_argument);

  auto const d_escape = escape_character.value(stream);
  CUDF_EXPECTS(d_escape.size_bytes() <= 1,
               "Parameter escape_character must be a single character",
               std::invalid_argument);

  auto const d_pattern    = pattern.value(stream);
  auto const patterns_itr = cuda::make_constant_iterator(d_pattern);
  return like(input, patterns_itr, d_escape, stream, mr);
}

std::unique_ptr<column> like(strings_column_view const& input,
                             std::string_view const& pattern,
                             std::string_view const& escape_character,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  auto const ptn = string_scalar(pattern, true, stream, cudf::get_current_device_resource_ref());
  auto const esc =
    string_scalar(escape_character, true, stream, cudf::get_current_device_resource_ref());
  return like(input, ptn, esc, stream, mr);
}

std::unique_ptr<column> like(strings_column_view const& input,
                             strings_column_view const& patterns,
                             string_scalar const& escape_character,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(patterns.size() == input.size(),
               "Number of patterns must match the input size",
               std::invalid_argument);
  CUDF_EXPECTS(patterns.has_nulls() == false,
               "Parameter patterns must not contain nulls",
               std::invalid_argument);
  CUDF_EXPECTS(escape_character.is_valid(stream),
               "Parameter escape_character must be valid",
               std::invalid_argument);

  auto const d_escape = escape_character.value(stream);
  CUDF_EXPECTS(d_escape.size_bytes() <= 1,
               "Parameter escape_character must be a single character",
               std::invalid_argument);

  auto const d_patterns   = column_device_view::create(patterns.parent(), stream);
  auto const patterns_itr = d_patterns->begin<string_view>();
  return like(input, patterns_itr, d_escape, stream, mr);
}

}  // namespace detail

// external API

std::unique_ptr<column> like(strings_column_view const& input,
                             std::string_view const& pattern,
                             std::string_view const& escape_character,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::like(input, pattern, escape_character, stream, mr);
}

std::unique_ptr<column> like(strings_column_view const& input,
                             strings_column_view const& patterns,
                             string_scalar const& escape_character,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::like(input, patterns, escape_character, stream, mr);
}

}  // namespace strings
}  // namespace cudf

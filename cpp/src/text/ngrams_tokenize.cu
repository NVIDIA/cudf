/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/utilities/tokenize_ops.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/detail/tokenize.hpp>
#include <nvtext/ngrams_tokenize.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/for_each.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {
/**
 * @brief This records the byte positions of each token within each string.
 *
 * The position values are recorded since we need to reference tokens
 * within a string multiple times to generate the ngrams. For example,
 * to generate tri-grams for string "aa b ccc dd" requires creating
 * the following two strings ["aa_b_ccc","b_ccc_dd"]. Notice the
 * tokens "b" and "ccc" needed to be copied twice for this string.
 *
 * Most of the work is done in the characters_tokenizer locating the tokens.
 * This functor simply records the byte positions in the d_token_positions
 * member.
 */
struct string_tokens_positions_fn {
  cudf::column_device_view const d_strings;          // strings to tokenize
  cudf::string_view const d_delimiter;               // delimiter to tokenize around
  cudf::detail::input_offsetalator d_token_offsets;  // offsets of d_token_positions for each string
  position_pair* d_token_positions;                  // token positions in each string

  __device__ void operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) return;
    cudf::string_view d_str = d_strings.element<cudf::string_view>(idx);
    // create tokenizer for this string
    characters_tokenizer tokenizer(d_str, d_delimiter);
    // record the token positions for this string
    cudf::size_type token_index = 0;
    auto token_positions        = d_token_positions + d_token_offsets[idx];
    while (tokenizer.next_token())
      token_positions[token_index++] = tokenizer.token_byte_positions();
  }
};

/**
 * @brief Generate the ngrams for each string.
 *
 * The ngrams for each string are placed contiguously within the section of memory
 * assigned for the input string. At the same time, the size of each ngram is recorded
 * in order to build the output offsets column.
 *
 * This functor can be called to compute the size of memory needed to write out
 * each set of ngrams per string. Once the memory offsets (d_chars_offsets) are
 * set and the output memory is allocated (d_chars), the ngrams for each string
 * can be generated into the output buffer.
 */
struct ngram_builder_fn {
  cudf::column_device_view const d_strings;  // strings to generate ngrams from
  cudf::string_view const d_separator;       // separator to place between them 'grams
  cudf::size_type const ngrams;              // ngram number to generate (2=bi-gram, 3=tri-gram)
  cudf::detail::input_offsetalator d_token_offsets;    // offsets for token position for each string
  position_pair const* d_token_positions;              // token positions for each string
  cudf::detail::input_offsetalator d_chars_offsets{};  // offsets for each string's ngrams
  char* d_chars{};                                     // write ngram strings to here
  cudf::size_type const* d_ngram_offsets{};            // offsets for sizes of each string's ngrams
  cudf::size_type* d_ngram_sizes{};                    // write ngram sizes to here

  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    if (d_strings.is_null(idx)) { return 0; }
    auto const d_str            = d_strings.element<cudf::string_view>(idx);
    auto const token_positions  = d_token_positions + d_token_offsets[idx];
    auto const token_count      = d_token_offsets[idx + 1] - d_token_offsets[idx];
    cudf::size_type nbytes      = 0;  // total number of output bytes needed for this string
    cudf::size_type ngram_index = 0;
    auto out_ptr                = d_chars ? d_chars + d_chars_offsets[idx] : nullptr;
    auto d_sizes                = d_ngram_sizes ? d_ngram_sizes + d_ngram_offsets[idx] : nullptr;
    // for ngrams=2, this will turn string "a b c d e" into "a_bb_cc_dd_e"
    for (cudf::size_type token_index = (ngrams - 1); token_index < token_count; ++token_index) {
      cudf::size_type length = 0;                          // calculate size of each ngram in bytes
      for (cudf::size_type n = (ngrams - 1); n >= 0; --n)  // sliding window of tokens
      {
        auto const item = token_positions[token_index - n];
        length += item.second - item.first;
        if (out_ptr) {
          out_ptr = cudf::strings::detail::copy_and_increment(
            out_ptr, d_str.data() + item.first, item.second - item.first);
        }
        if (n > 0) {  // include the separator (except for the last one)
          if (out_ptr) { out_ptr = cudf::strings::detail::copy_string(out_ptr, d_separator); }
          length += d_separator.size_bytes();
        }
      }
      if (d_sizes) { d_sizes[ngram_index++] = length; }
      nbytes += length;
    }
    return nbytes;
  }
};

}  // namespace

// detail APIs

std::unique_ptr<cudf::column> ngrams_tokenize(cudf::strings_column_view const& strings,
                                              cudf::size_type ngrams,
                                              cudf::string_scalar const& delimiter,
                                              cudf::string_scalar const& separator,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");
  cudf::string_view d_delimiter(delimiter.data(), delimiter.size());
  CUDF_EXPECTS(separator.is_valid(stream), "Parameter separator must be valid");
  cudf::string_view d_separator(separator.data(), separator.size());

  CUDF_EXPECTS(ngrams >= 1, "Parameter ngrams should be an integer value of 1 or greater");
  if (ngrams == 1)  // this is just a straight tokenize
    return tokenize(strings, delimiter, stream, mr);
  auto strings_count = strings.size();
  if (strings.is_empty()) return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  auto d_strings      = *strings_column;

  // Example for comments with ngrams=2
  // ["a bb ccc","dd e"] => ["a_bb", "bb_ccc", "dd_e"]

  // first, get the number of tokens per string to get the token-offsets
  // Ex. token-counts = [3,2]; token-offsets = [0,3,5]
  auto const count_itr =
    cudf::detail::make_counting_transform_iterator(0, strings_tokenizer{d_strings, d_delimiter});
  auto [token_offsets, total_tokens] = cudf::strings::detail::make_offsets_child_column(
    count_itr, count_itr + strings_count, stream, cudf::get_current_device_resource_ref());
  auto d_token_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(token_offsets->view());

  // get the token positions (in bytes) per string
  // Ex. start/end pairs: [(0,1),(2,4),(5,8), (0,2),(3,4)]
  rmm::device_uvector<position_pair> token_positions(total_tokens, stream);
  auto d_token_positions = token_positions.data();
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<cudf::size_type>{0},
    strings_count,
    string_tokens_positions_fn{d_strings, d_delimiter, d_token_offsets, d_token_positions});

  // compute the number of ngrams per string to get the total number of ngrams to generate
  // Ex. ngram-counts = [2,1]; ngram-offsets = [0,2,3]; total = 3 bigrams
  auto const ngram_counts = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cudf::size_type>(
      [d_token_offsets, ngrams] __device__(cudf::size_type idx) {
        auto token_count =
          static_cast<cudf::size_type>(d_token_offsets[idx + 1] - d_token_offsets[idx]);
        return (token_count >= ngrams) ? token_count - ngrams + 1 : 0;
      }));
  auto [ngram_offsets, total_ngrams] = cudf::detail::make_offsets_child_column(
    ngram_counts, ngram_counts + strings_count, stream, cudf::get_current_device_resource_ref());
  auto d_ngram_offsets = ngram_offsets->view().begin<cudf::size_type>();

  // Compute the total size of the ngrams for each string (not for each ngram)
  // Ex. 2 bigrams in 1st string total to 10 bytes; 1 bigram in 2nd string is 4 bytes
  //     => sizes = [10,4]; offsets = [0,10,14]
  //
  // This produces a set of offsets for the output memory where we can build adjacent
  // ngrams for each string.
  // Ex. bigram for first string produces 2 bigrams ("a_bb","bb_ccc") which
  //     is built in memory like this: "a_bbbb_ccc"

  //  First compute the output sizes for each string (this not the final output result)
  auto const sizes_itr = cudf::detail::make_counting_transform_iterator(
    0, ngram_builder_fn{d_strings, d_separator, ngrams, d_token_offsets, d_token_positions});
  auto [chars_offsets, output_chars_size] = cudf::strings::detail::make_offsets_child_column(
    sizes_itr, sizes_itr + strings_count, stream, cudf::get_current_device_resource_ref());
  auto d_chars_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(chars_offsets->view());

  // This will contain the size in bytes of each ngram to generate
  rmm::device_uvector<cudf::size_type> ngram_sizes(total_ngrams, stream);

  // build output chars column
  rmm::device_uvector<char> chars(output_chars_size, stream, mr);
  auto d_chars = chars.data();
  // Generate the ngrams into the chars column data buffer.
  // The ngram_builder_fn functor also fills the ngram_sizes vector with the
  // size of each ngram.
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<cudf::size_type>{0},
                     strings_count,
                     ngram_builder_fn{d_strings,
                                      d_separator,
                                      ngrams,
                                      d_token_offsets,
                                      d_token_positions,
                                      d_chars_offsets,
                                      d_chars,
                                      d_ngram_offsets,
                                      ngram_sizes.data()});
  // build the offsets column -- converting the ngram sizes into offsets
  auto offsets_column = std::get<0>(cudf::strings::detail::make_offsets_child_column(
    ngram_sizes.begin(), ngram_sizes.end(), stream, mr));
  // create the output strings column
  return make_strings_column(
    total_ngrams, std::move(offsets_column), chars.release(), 0, rmm::device_buffer{});
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> ngrams_tokenize(cudf::strings_column_view const& strings,
                                              cudf::size_type ngrams,
                                              cudf::string_scalar const& delimiter,
                                              cudf::string_scalar const& separator,
                                              rmm::cuda_stream_view stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::ngrams_tokenize(strings, ngrams, delimiter, separator, stream, mr);
}

}  // namespace nvtext

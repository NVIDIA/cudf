/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/utilities/tokenize_ops.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/replace.hpp>

#include <cuda/atomic>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/remove.h>

namespace nvtext {
namespace detail {
namespace {

using replace_result = cuda::std::pair<bool, cudf::string_view>;

struct base_token_replacer_fn {
  cudf::column_device_view d_strings;          ///< strings to tokenize
  cudf::string_view const d_delimiter;         ///< delimiter characters for tokenizing
  cudf::size_type* d_sizes{};                  ///< for output string size
  char* d_chars{};                             ///< output buffer
  cudf::detail::input_offsetalator d_offsets;  ///< offsets for output buffer
  cudf::size_type const* d_indices{};          ///< indices for long strings
  cudf::size_type* d_output_sizes{};           ///< output sizes for long strings

  /**
   * @brief Tokenizes each string and calls the provided `replacer` function
   * for each token.
   *
   * @tparam ReplaceFn Should accept a `string_view` and return a `replace_result`
   * @param idx Index of the current string to process
   * @param replacer Function to call for each token to determined its replacement
   */
  template <typename ReplaceFn>
  __device__ void process_string(cudf::size_type idx, ReplaceFn replacer) const
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    auto const d_str  = d_strings.element<cudf::string_view>(idx);
    auto const in_ptr = d_str.data();
    auto out_ptr      = d_chars ? d_chars + d_offsets[idx] : nullptr;
    auto nbytes       = d_str.size_bytes();  // count the output bytes
    auto last_pos     = cudf::size_type{0};
    auto tokenizer    = characters_tokenizer{d_str, d_delimiter};
    // process each token
    while (tokenizer.next_token()) {
      auto const token_pos = tokenizer.token_byte_positions();
      auto const token =
        cudf::string_view{d_str.data() + token_pos.first, token_pos.second - token_pos.first};
      // ask replacer if this token should be replaced
      auto const result = replacer(token);
      if (result.first) {  // first == replace indicator, second == new string
        auto d_replacement = result.second;
        nbytes += d_replacement.size_bytes() - token.size_bytes();
        if (out_ptr) {
          // copy over string up to the token location
          out_ptr = cudf::strings::detail::copy_and_increment(
            out_ptr, in_ptr + last_pos, token_pos.first - last_pos);
          // copy over replacement string
          out_ptr  = cudf::strings::detail::copy_string(out_ptr, d_replacement);
          last_pos = token_pos.second;  // update last byte position for this string
        }
      }
    }

    // copy the remainder of the string's bytes to the output buffer
    if (out_ptr) {
      memcpy(out_ptr, in_ptr + last_pos, d_str.size_bytes() - last_pos);
    } else {
      d_sizes[idx] = nbytes;
      // handles output size calculation for long strings
      if (nbytes > 0 && d_indices) {
        auto out_idx = d_indices[idx] - 1;  // adjust for upper_bound
        cuda::atomic_ref<cudf::size_type, cuda::thread_scope_block> ref{
          *(d_output_sizes + out_idx)};
        ref.fetch_add(nbytes, cuda::std::memory_order_relaxed);
      }
    }
  }
};

using strings_iterator = cudf::column_device_view::const_iterator<cudf::string_view>;

/**
 * @brief Functor to replace tokens in each string.
 *
 * This tokenizes a string using the given d_delimiter and replaces any tokens that match
 * a string in d_targets_begin/end with those from the d_replacements column.
 * Strings with no matching tokens are left unchanged.
 *
 * This should be called first to compute the size of each output string and then a second
 * time to fill in the allocated output buffer for each string.
 */
struct replace_tokens_fn : base_token_replacer_fn {
  strings_iterator d_targets_begin;  ///< strings to search for
  strings_iterator d_targets_end;
  cudf::column_device_view const d_replacements;

  replace_tokens_fn(cudf::column_device_view const& d_strings,
                    cudf::string_view const& d_delimiter,
                    strings_iterator d_targets_begin,
                    strings_iterator d_targets_end,
                    cudf::column_device_view const& d_replacements)
    : base_token_replacer_fn{d_strings, d_delimiter},
      d_targets_begin{d_targets_begin},
      d_targets_end{d_targets_end},
      d_replacements{d_replacements}
  {
  }

  /**
   * @brief Return replacement string for the given token.
   *
   * @param token Token candidate to be replaced.
   * @return result pair specifies replacement condition and new string
   */
  __device__ replace_result token_replacement(cudf::string_view const& token) const
  {
    // check if the token matches any of the targets
    auto const found_itr = thrust::find(thrust::seq, d_targets_begin, d_targets_end, token);
    if (found_itr != d_targets_end) {  // match found
      // retrieve the corresponding replacement string or
      // if only one repl string, use that one for all targets
      auto const d_repl = [&] {
        auto const repl_idx = cuda::std::distance(d_targets_begin, found_itr);
        return d_replacements.size() == 1 ? d_replacements.element<cudf::string_view>(0)
                                          : d_replacements.element<cudf::string_view>(repl_idx);
      }();
      return replace_result{true, d_repl};
    }
    // otherwise, do not replace this token
    return replace_result{false, cudf::string_view()};
  }

  __device__ void operator()(cudf::size_type idx) const
  {
    process_string(
      idx, [this] __device__(cudf::string_view const& token) { return token_replacement(token); });
  }
};

// For determining long strings processing
constexpr cudf::size_type AVG_CHAR_BYTES_THRESHOLD = 64;
// For computing sub-block sizes of long strings
constexpr cudf::size_type LS_SUB_BLOCK_SIZE = 64;

/**
 * @brief Locate delimiters to produce sub-offsets in the input device array
 *
 * The sub-offsets provide additional tokenize boundaries within longer strings.
 */
struct sub_offset_fn {
  char const* d_input_chars;
  int64_t first_offset;
  int64_t last_offset;
  cudf::string_view const d_delimiter;

  __device__ int64_t operator()(int64_t idx) const
  {
    // keep delimiter search within this sub-block
    auto const end =
      d_input_chars + cuda::std::min(last_offset, ((idx + 2) * LS_SUB_BLOCK_SIZE) + first_offset);
    // starting point of this sub-block
    auto itr = d_input_chars + first_offset + ((idx + 1) * LS_SUB_BLOCK_SIZE);
    while ((itr < end) &&
           cudf::strings::detail::is_utf8_continuation_char(static_cast<u_char>(*itr))) {
      ++itr;
    }
    if (itr >= end) { return 0; }  // 0s will be filtered out
    // now check for a delimiter in this block
    auto tokenizer = characters_tokenizer(cudf::string_view{}, d_delimiter);
    while (itr < end) {
      auto chr      = cudf::char_utf8{};
      auto chr_size = cudf::strings::detail::to_char_utf8(itr, chr);
      if (tokenizer.is_delimiter(chr)) { break; }
      itr += chr_size;
    }
    return (itr < end) ? cuda::std::distance(d_input_chars, itr) : 0L;
  }
};

/**
 * @brief Functor to filter tokens in each string.
 *
 * This tokenizes a string using the given d_delimiter and replaces any tokens
 * that are shorter than min_token_length with a replacement string.
 *
 * This should be called first to compute the size of each output string and then
 * a second time to fill in the allocated output buffer for each string.
 */
struct remove_small_tokens_fn : base_token_replacer_fn {
  cudf::size_type min_token_length;       ///< minimum size for found tokens
  cudf::string_view const d_replacement;  ///< replacement string

  remove_small_tokens_fn(cudf::column_device_view const& d_strings,
                         cudf::string_view const& d_delimiter,
                         cudf::size_type min_token_length,
                         cudf::string_view const& d_replacement)
    : base_token_replacer_fn{d_strings, d_delimiter},
      min_token_length{min_token_length},
      d_replacement{d_replacement}
  {
  }

  __device__ replace_result token_replacement(cudf::string_view token) const
  {
    return replace_result{token.length() < min_token_length, d_replacement};
  }

  __device__ void operator()(cudf::size_type idx) const
  {
    process_string(
      idx, [this] __device__(cudf::string_view const& token) { return token_replacement(token); });
  }
};

/**
 * @brief Common code for replace and filter
 *
 * Builds the output strings column using the given replace functor.
 *
 * @tparam ReplaceFn Functor called for replacing tokens
 *
 * @param replacer Functor for determining matching token and its replacement
 * @param input Strings column to tokenize and replace
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings columns of with replaced strings
 */
template <typename ReplacerFn>
std::unique_ptr<cudf::column> replace_helper(ReplacerFn replacer,
                                             cudf::strings_column_view const& input,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  auto const first_offset = (input.offset() == 0) ? 0L
                                                  : cudf::strings::detail::get_offset_value(
                                                      input.offsets(), input.offset(), stream);
  auto const last_offset =
    cudf::strings::detail::get_offset_value(input.offsets(), input.size() + input.offset(), stream);
  auto const chars_size = last_offset - first_offset;

  if ((chars_size / (input.size() - input.null_count())) < AVG_CHAR_BYTES_THRESHOLD) {
    // this utility calls replacer to build the offsets and chars columns
    auto [offsets_column, chars] =
      cudf::strings::detail::make_strings_children(replacer, input.size(), stream, mr);
    // return new strings column
    return cudf::make_strings_column(input.size(),
                                     std::move(offsets_column),
                                     chars.release(),
                                     input.null_count(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr));
  }

  // Long strings logic builds a new fake strings column with the same data but additional offsets
  // thus converting the input to a larger column of smaller strings.
  // This can be processed in parallel more efficiently than long strings in general.

  auto const input_chars = input.chars_begin(stream);
  auto const input_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(input.offsets(), input.offset());

  // divide up long strings into shorter strings by finding new sub-offsets at delimiters
  auto sub_count   = chars_size / LS_SUB_BLOCK_SIZE;
  auto tmp_offsets = rmm::device_uvector<int64_t>(sub_count + input.size() + 1, stream);
  {
    rmm::device_uvector<int64_t> sub_offsets(sub_count, stream);
    auto const count_itr = cuda::counting_iterator<int64_t>{0};
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      count_itr,
                      count_itr + sub_count,
                      sub_offsets.data(),
                      sub_offset_fn{input_chars, first_offset, last_offset});
    // remove 0s -- where sub-offset could not be computed
    auto const remove_end =
      thrust::remove(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     sub_offsets.begin(),
                     sub_offsets.end(),
                     0L);
    sub_count = cuda::std::distance(sub_offsets.begin(), remove_end);

    // merge them with input offsets
    thrust::merge(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  input_offsets,
                  input_offsets + input.size() + 1,
                  sub_offsets.begin(),
                  sub_offsets.begin() + sub_count,
                  tmp_offsets.begin());
    tmp_offsets.resize(sub_count + input.size() + 1, stream);
    stream.sync();  // protect against destruction of sub_offsets
  }

  // cobble together a column_view of type STRING using the original data and the tmp offsets
  auto const tmp_size    = static_cast<cudf::size_type>(tmp_offsets.size()) - 1;
  auto const children    = std::vector<cudf::column_view>({cudf::column_view(
    cudf::data_type{cudf::type_id::INT64}, tmp_size + 1, tmp_offsets.data(), nullptr, 0)});
  auto const tmp_strings = cudf::column_view(
    cudf::data_type{cudf::type_id::STRING}, tmp_size, input_chars, nullptr, 0, 0, children);
  auto const d_tmp_strings = cudf::column_device_view::create(tmp_strings, stream);

  // compute indices to the actual output rows
  auto indices = rmm::device_uvector<cudf::size_type>(tmp_offsets.size(), stream);
  thrust::upper_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      input_offsets,
                      input_offsets + input.size() + 1,
                      tmp_offsets.begin(),
                      tmp_offsets.end(),
                      indices.begin());

  // initialize the output row sizes
  auto d_sizes = rmm::device_uvector<cudf::size_type>(input.size(), stream);
  thrust::fill(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               d_sizes.begin(),
               d_sizes.end(),
               0);

  replacer.d_strings      = *d_tmp_strings;
  replacer.d_indices      = indices.data();
  replacer.d_output_sizes = d_sizes.data();

  auto chars = std::get<1>(
    cudf::strings::detail::make_strings_children(replacer, tmp_strings.size(), stream, mr));
  auto offsets_column =
    std::get<0>(cudf::strings::detail::make_offsets_child_column(d_sizes, stream, mr));
  return cudf::make_strings_column(input.size(),
                                   std::move(offsets_column),
                                   chars.release(),
                                   input.null_count(),
                                   cudf::detail::copy_bitmask(input.parent(), stream, mr));
}
}  // namespace

// detail APIs

std::unique_ptr<cudf::column> replace_tokens(cudf::strings_column_view const& input,
                                             cudf::strings_column_view const& targets,
                                             cudf::strings_column_view const& replacements,
                                             cudf::string_scalar const& delimiter,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!targets.has_nulls(), "Parameter targets must not have nulls");
  CUDF_EXPECTS(!replacements.has_nulls(), "Parameter replacements must not have nulls");
  if (replacements.size() != 1)
    CUDF_EXPECTS(replacements.size() == targets.size(),
                 "Parameter targets and replacements must be the same size");
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  if (input.is_empty()) { return cudf::make_empty_column(cudf::type_id::STRING); }

  auto const d_strings      = cudf::column_device_view::create(input.parent(), stream);
  auto const d_targets      = cudf::column_device_view::create(targets.parent(), stream);
  auto const d_replacements = cudf::column_device_view::create(replacements.parent(), stream);
  auto const d_delimiter    = cudf::string_view(delimiter.data(), delimiter.size());

  replace_tokens_fn replacer{*d_strings,
                             d_delimiter,
                             d_targets->begin<cudf::string_view>(),
                             d_targets->end<cudf::string_view>(),
                             *d_replacements};

  return replace_helper(replacer, input, stream, mr);
}

std::unique_ptr<cudf::column> filter_tokens(cudf::strings_column_view const& input,
                                            cudf::size_type min_token_length,
                                            cudf::string_scalar const& replacement,
                                            cudf::string_scalar const& delimiter,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(replacement.is_valid(stream), "Parameter replacement must be valid");
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  if (input.is_empty()) { return cudf::make_empty_column(cudf::type_id::STRING); }

  auto const d_strings     = cudf::column_device_view::create(input.parent(), stream);
  auto const d_replacement = cudf::string_view(replacement.data(), replacement.size());
  auto const d_delimiter   = cudf::string_view(delimiter.data(), delimiter.size());

  remove_small_tokens_fn filterer{*d_strings, d_delimiter, min_token_length, d_replacement};

  return replace_helper(filterer, input, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> replace_tokens(cudf::strings_column_view const& input,
                                             cudf::strings_column_view const& targets,
                                             cudf::strings_column_view const& replacements,
                                             cudf::string_scalar const& delimiter,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_tokens(input, targets, replacements, delimiter, stream, mr);
}

std::unique_ptr<cudf::column> filter_tokens(cudf::strings_column_view const& input,
                                            cudf::size_type min_token_length,
                                            cudf::string_scalar const& replacement,
                                            cudf::string_scalar const& delimiter,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_tokens(input, min_token_length, replacement, delimiter, stream, mr);
}

}  // namespace nvtext

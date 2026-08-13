/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/utilities/tokenize_ops.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/attributes.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/detail/tokenize.hpp>
#include <nvtext/tokenize.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/copy.h>
#include <thrust/for_each.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {
// common pattern for token_count functions
template <typename TokenCounter>
std::unique_ptr<cudf::column> token_count_fn(cudf::size_type strings_count,
                                             TokenCounter tokenizer,
                                             rmm::cuda_stream_view stream,
                                             rmm::device_async_resource_ref mr)
{
  // create output column
  auto token_counts =
    cudf::make_numeric_column(cudf::data_type{cudf::type_to_id<cudf::size_type>()},
                              strings_count,
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr);
  auto d_token_counts = token_counts->mutable_view().data<cudf::size_type>();
  // add the counts to the column
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{strings_count},
                    d_token_counts,
                    tokenizer);
  return token_counts;
}

// common pattern for tokenize functions
template <typename Tokenizer>
std::unique_ptr<cudf::column> tokenize_fn(cudf::size_type strings_count,
                                          Tokenizer tokenizer,
                                          rmm::cuda_stream_view stream,
                                          rmm::device_async_resource_ref mr)
{
  // get the number of tokens in each string
  auto const token_counts =
    token_count_fn(strings_count, tokenizer, stream, cudf::get_current_device_resource_ref());
  auto d_token_counts = token_counts->view();
  // create token-index offsets from the counts
  auto [token_offsets, total_tokens] =
    cudf::detail::make_offsets_child_column(d_token_counts.template begin<cudf::size_type>(),
                                            d_token_counts.template end<cudf::size_type>(),
                                            stream,
                                            cudf::get_current_device_resource_ref());
  //  build a list of pointers to each token
  rmm::device_uvector<string_index_pair> tokens(total_tokens, stream);
  // now go get the tokens
  tokenizer.d_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(token_offsets->view());
  tokenizer.d_tokens = tokens.data();
  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<cudf::size_type>{0},
                     strings_count,
                     tokenizer);
  // create the strings column using the tokens pointers
  return cudf::strings::detail::make_strings_column(tokens.begin(), tokens.end(), stream, mr);
}

}  // namespace

// detail APIs

// zero or more character tokenizer
std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& strings,
                                       cudf::string_scalar const& delimiter,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");
  cudf::string_view d_delimiter(delimiter.data(), delimiter.size());
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  return tokenize_fn(strings.size(), strings_tokenizer{*strings_column, d_delimiter}, stream, mr);
}

// zero or more character token counter
std::unique_ptr<cudf::column> count_tokens(cudf::strings_column_view const& strings,
                                           cudf::string_scalar const& delimiter,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");
  cudf::string_view d_delimiter(delimiter.data(), delimiter.size());
  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  return token_count_fn(
    strings.size(), strings_tokenizer{*strings_column, d_delimiter}, stream, mr);
}

// one or more string delimiter tokenizer
std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& strings,
                                       cudf::strings_column_view const& delimiters,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiters.size() > 0, "Parameter delimiters must not be empty");
  CUDF_EXPECTS(!delimiters.has_nulls(), "Parameter delimiters must not have nulls");
  auto strings_column    = cudf::column_device_view::create(strings.parent(), stream);
  auto delimiters_column = cudf::column_device_view::create(delimiters.parent(), stream);
  return tokenize_fn(
    strings.size(),
    multi_delimiter_strings_tokenizer{*strings_column,
                                      delimiters_column->begin<cudf::string_view>(),
                                      delimiters_column->end<cudf::string_view>()},
    stream,
    mr);
}

// one or more string delimiter token counter
std::unique_ptr<cudf::column> count_tokens(cudf::strings_column_view const& strings,
                                           cudf::strings_column_view const& delimiters,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiters.size() > 0, "Parameter delimiters must not be empty");
  CUDF_EXPECTS(!delimiters.has_nulls(), "Parameter delimiters must not have nulls");
  auto strings_column    = cudf::column_device_view::create(strings.parent(), stream);
  auto delimiters_column = cudf::column_device_view::create(delimiters.parent(), stream);
  return token_count_fn(
    strings.size(),
    multi_delimiter_strings_tokenizer{*strings_column,
                                      delimiters_column->begin<cudf::string_view>(),
                                      delimiters_column->end<cudf::string_view>()},
    stream,
    mr);
}

// tokenize on every character
std::unique_ptr<cudf::column> character_tokenize(cudf::strings_column_view const& strings_column,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  auto strings_count = strings_column.size();
  if (strings_count == 0) {
    return cudf::make_empty_lists_column(cudf::data_type{cudf::type_id::STRING});
  }

  CUDF_EXPECTS(
    strings_column.null_count() == 0, "input must not contain nulls", std::invalid_argument);

  auto const offsets = strings_column.offsets();
  auto const offset =
    cudf::strings::detail::get_offset_value(offsets, strings_column.offset(), stream);
  auto const chars_bytes = cudf::strings::detail::get_offset_value(
                             offsets, strings_column.offset() + strings_count, stream) -
                           offset;
  // no bytes -- this could happen in an all-empty column
  if (chars_bytes == 0) {
    return cudf::make_empty_lists_column(cudf::data_type{cudf::type_id::STRING});
  }
  auto d_chars =
    strings_column.parent().data<uint8_t>();  // unsigned is necessary for checking bits
  d_chars += offset;

  auto const character_counts = cudf::strings::detail::count_characters(
    strings_column, stream, cudf::get_current_device_resource_ref());
  auto [list_offsets, num_characters] =
    cudf::detail::make_offsets_child_column(character_counts->view().begin<cudf::size_type>(),
                                            character_counts->view().end<cudf::size_type>(),
                                            stream,
                                            mr);

  // number of characters becomes the number of rows so need to check the row limit
  CUDF_EXPECTS(
    num_characters + 1 < static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
    "output exceeds the column size limit",
    std::overflow_error);

  // create output offsets column
  auto offsets_column = cudf::make_numeric_column(
    offsets.type(), num_characters + 1, cudf::mask_state::UNALLOCATED, stream, mr);
  auto d_new_offsets =
    cudf::detail::offsetalator_factory::make_output_iterator(offsets_column->mutable_view());
  // offsets are at the beginning byte of each character
  cudf::detail::copy_if_async(
    cuda::counting_iterator<int64_t>{0},
    cuda::counting_iterator<int64_t>{chars_bytes + 1},
    d_new_offsets,
    [d_chars, chars_bytes] __device__(auto idx) -> bool {
      // this will also set the final value to the size chars_bytes
      return idx < chars_bytes ? cudf::strings::detail::is_begin_utf8_char(d_chars[idx]) : true;
    },
    stream);

  // create the output chars buffer -- just a copy of the input's chars
  rmm::device_uvector<char> output_chars(chars_bytes, stream, mr);
  thrust::copy(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
               d_chars,
               d_chars + chars_bytes,
               output_chars.data());

  auto output_strings = cudf::make_strings_column(
    num_characters, std::move(offsets_column), output_chars.release(), 0, rmm::device_buffer{});
  return cudf::make_lists_column(strings_count,
                                 std::move(list_offsets),
                                 std::move(output_strings),
                                 strings_column.null_count(),
                                 cudf::detail::copy_bitmask(strings_column.parent(), stream, mr));
}

}  // namespace detail

// external APIs

std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& input,
                                       cudf::string_scalar const& delimiter,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::tokenize(input, delimiter, stream, mr);
}

std::unique_ptr<cudf::column> tokenize(cudf::strings_column_view const& input,
                                       cudf::strings_column_view const& delimiters,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::tokenize(input, delimiters, stream, mr);
}

std::unique_ptr<cudf::column> count_tokens(cudf::strings_column_view const& input,
                                           cudf::string_scalar const& delimiter,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_tokens(input, delimiter, stream, mr);
}

std::unique_ptr<cudf::column> count_tokens(cudf::strings_column_view const& input,
                                           cudf::strings_column_view const& delimiters,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_tokens(input, delimiters, stream, mr);
}

std::unique_ptr<cudf::column> character_tokenize(cudf::strings_column_view const& input,
                                                 rmm::cuda_stream_view stream,
                                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::character_tokenize(input, stream, mr);
}

}  // namespace nvtext

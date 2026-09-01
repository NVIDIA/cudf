/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "split.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/strings/detail/split_utils.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/functional>
#include <cuda/stream>
#include <thrust/for_each.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

template <typename Tokenizer, typename DelimiterFn>
std::unique_ptr<column> split_record_fn(strings_column_view const& input,
                                        Tokenizer tokenizer,
                                        DelimiterFn delimiter_fn,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) {
    return cudf::lists::detail::make_empty_lists_column(data_type{type_id::STRING});
  }
  if (input.size() == input.null_count()) {
    auto offsets = std::make_unique<column>(input.offsets(), stream, mr);
    auto results = make_empty_column(type_id::STRING);
    return make_lists_column(input.size(),
                             std::move(offsets),
                             std::move(results),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
  }

  auto [offsets, tokens] = split_helper(input, tokenizer, delimiter_fn, stream, mr);
  CUDF_EXPECTS(tokens.size() < static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "Size of output exceeds the column size limit",
               std::overflow_error);

  auto strings_child = cudf::make_strings_column(tokens, stream, mr);
  return make_lists_column(input.size(),
                           std::move(offsets),
                           std::move(strings_child),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

// Build a lists column from the per-row split of a non-whitespace delimiter.
template <bool Forward>
std::unique_ptr<column> split_record_per_row_fn(strings_column_view const& input,
                                                string_view const d_delimiter,
                                                size_type const max_tokens,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) {
    return cudf::lists::detail::make_empty_lists_column(data_type{type_id::STRING});
  }
  if (input.size() == input.null_count()) {
    auto offsets = std::make_unique<column>(input.offsets(), stream, mr);
    auto results = make_empty_column(type_id::STRING);
    return make_lists_column(input.size(),
                             std::move(offsets),
                             std::move(results),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
  }

  auto d_strings         = column_device_view::create(input.parent(), stream);
  auto [offsets, tokens] = split_per_row_impl(
    *d_strings,
    token_count_fn{*d_strings, d_delimiter, max_tokens},
    [d_str = *d_strings, d_delimiter](auto d_offsets, auto* d_tokens) {
      using fn_t = std::conditional_t<Forward, split_extract_fn, rsplit_extract_fn>;
      return fn_t{d_str, d_delimiter, d_offsets, d_tokens};
    },
    stream,
    mr);
  CUDF_EXPECTS(tokens.size() < static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "Size of output exceeds the column size limit",
               std::overflow_error);

  auto strings_child = cudf::make_strings_column(tokens, stream, mr);
  return make_lists_column(input.size(),
                           std::move(offsets),
                           std::move(strings_child),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

// Build a lists column from the per-row whitespace split.
template <bool Forward>
std::unique_ptr<column> split_record_ws_per_row_fn(strings_column_view const& input,
                                                   size_type const max_tokens,
                                                   cuda::stream_ref stream,
                                                   rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) {
    return cudf::lists::detail::make_empty_lists_column(data_type{type_id::STRING});
  }
  if (input.size() == input.null_count()) {
    auto offsets = std::make_unique<column>(input.offsets(), stream, mr);
    auto results = make_empty_column(type_id::STRING);
    return make_lists_column(input.size(),
                             std::move(offsets),
                             std::move(results),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
  }

  auto d_strings         = column_device_view::create(input.parent(), stream);
  auto [offsets, tokens] = split_per_row_impl(
    *d_strings,
    ws_token_count_fn{*d_strings, max_tokens},
    [d_str = *d_strings, max_tokens](auto d_offsets, auto* d_tokens) {
      using fn_t = std::conditional_t<Forward, split_ws_extract_fn, rsplit_ws_extract_fn>;
      return fn_t{d_str, d_offsets, d_tokens, max_tokens};
    },
    stream,
    mr);
  CUDF_EXPECTS(tokens.size() < static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "Size of output exceeds the column size limit",
               std::overflow_error);

  auto strings_child = make_strings_column(tokens.begin(), tokens.end(), stream, mr);
  return make_lists_column(input.size(),
                           std::move(offsets),
                           std::move(strings_child),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

template <bool Forward>
std::unique_ptr<column> split_record_impl(strings_column_view const& input,
                                          string_scalar const& delimiter,
                                          size_type maxsplit,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  // makes consistent with Pandas
  size_type const max_tokens = maxsplit > 0 ? maxsplit + 1 : std::numeric_limits<size_type>::max();

  auto const non_null_count = input.size() - input.null_count();
  if (delimiter.size() == 0) {
    if (non_null_count == 0 ||
        (input.chars_size(stream) / non_null_count) < AVG_CHAR_BYTES_THRESHOLD) {
      return split_record_ws_per_row_fn<Forward>(input, max_tokens, stream, mr);
    }
    auto d_strings    = column_device_view::create(input.parent(), stream);
    using ws_tok_t    = std::conditional_t<Forward, split_ws_tokenizer_fn, rsplit_ws_tokenizer_fn>;
    auto tokenizer    = ws_tok_t{*d_strings, max_tokens};
    auto delimiter_fn = whitespace_delimiter_fn{};
    return split_record_fn(input, tokenizer, delimiter_fn, stream, mr);
  }

  if (non_null_count == 0 ||
      (input.chars_size(stream) / non_null_count) < AVG_CHAR_BYTES_THRESHOLD) {
    return split_record_per_row_fn<Forward>(input, delimiter.value(stream), max_tokens, stream, mr);
  }

  auto d_strings    = column_device_view::create(input.parent(), stream);
  using tok_t       = std::conditional_t<Forward, split_tokenizer_fn, rsplit_tokenizer_fn>;
  auto tokenizer    = tok_t{*d_strings, delimiter.size(), max_tokens};
  auto delimiter_fn = string_delimiter_fn{delimiter.value(stream)};
  return split_record_fn(input, tokenizer, delimiter_fn, stream, mr);
}

}  // namespace

std::unique_ptr<column> split_record(strings_column_view const& input,
                                     string_scalar const& delimiter,
                                     size_type maxsplit,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  return split_record_impl<true>(input, delimiter, maxsplit, stream, mr);
}

std::unique_ptr<column> rsplit_record(strings_column_view const& input,
                                      string_scalar const& delimiter,
                                      size_type maxsplit,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  return split_record_impl<false>(input, delimiter, maxsplit, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<column> split_record(strings_column_view const& input,
                                     string_scalar const& delimiter,
                                     size_type maxsplit,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::split_record(input, delimiter, maxsplit, stream, mr);
}

std::unique_ptr<column> rsplit_record(strings_column_view const& input,
                                      string_scalar const& delimiter,
                                      size_type maxsplit,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rsplit_record(input, delimiter, maxsplit, stream, mr);
}

}  // namespace strings
}  // namespace cudf

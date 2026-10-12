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

namespace cudf {
namespace strings {
namespace detail {

// Per-string token count — returns number of tokens (delimiters found + 1), capped at max_tokens.
struct token_count_fn {
  column_device_view const d_strings;
  string_view const d_delimiter;
  size_type const max_tokens;

  __device__ size_type operator()(size_type const idx) const
  {
    if (d_strings.is_null(idx)) { return 0; }
    auto const d_str    = d_strings.element<string_view>(idx);
    auto const del_size = d_delimiter.size_bytes();
    size_type count     = 1;
    size_type pos       = 0;
    while (pos + del_size <= d_str.size_bytes()) {
      if (d_delimiter.compare(d_str.data() + pos, del_size) == 0) {
        if (++count == max_tokens) { break; }
        pos += del_size;
      } else {
        ++pos;
      }
    }
    return count;
  }
};

// Per-string token count for whitespace split — returns number of non-whitespace runs, capped at
// max_tokens. Both forward and backward splits produce the same count.
struct ws_token_count_fn {
  column_device_view const d_strings;
  size_type const max_tokens;

  __device__ size_type operator()(size_type const idx) const
  {
    if (d_strings.is_null(idx)) { return 0; }
    auto const d_str = d_strings.element<string_view>(idx);
    auto const size  = d_str.size_bytes();
    auto const base  = d_str.data();
    size_type count  = 0;
    bool in_token    = false;
    for (size_type i = 0; i < size && count < max_tokens; ++i) {
      bool const is_ws = is_whitespace(static_cast<char_utf8>(base[i]));
      if (!is_ws && !in_token) {
        ++count;
        in_token = true;
      } else if (is_ws) {
        in_token = false;
      }
    }
    return count;
  }
};

// Extract tokens left-to-right for non-whitespace split.
struct split_extract_fn {
  column_device_view const d_strings;
  string_view const d_delimiter;
  cudf::detail::input_offsetalator const d_token_offsets;
  string_index_pair* const d_tokens;

  __device__ void operator()(size_type const idx) const
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str        = d_strings.element<string_view>(idx);
    auto const token_offset = d_token_offsets[idx];
    auto const token_count  = static_cast<size_type>(d_token_offsets[idx + 1] - token_offset);
    auto* const d_result    = d_tokens + token_offset;
    auto const size         = d_str.size_bytes();
    auto const del_size     = d_delimiter.size_bytes();
    auto const base         = d_str.data();

    if (size == 0) {
      d_result[0] = string_index_pair{"", 0};
      return;
    }
    size_type token_idx = 0;
    size_type last_pos  = 0;
    size_type pos       = 0;
    while (pos + del_size <= size && token_idx < token_count - 1) {
      if (d_delimiter.compare(base + pos, del_size) == 0) {
        d_result[token_idx++] = string_index_pair{base + last_pos, pos - last_pos};
        last_pos              = pos + del_size;
        pos                   = last_pos;
      } else {
        ++pos;
      }
    }
    d_result[token_idx] = string_index_pair{base + last_pos, size - last_pos};
  }
};

// Extract tokens right-to-left for non-whitespace split.
struct rsplit_extract_fn {
  column_device_view const d_strings;
  string_view const d_delimiter;
  cudf::detail::input_offsetalator const d_token_offsets;
  string_index_pair* const d_tokens;

  __device__ void operator()(size_type const idx) const
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str        = d_strings.element<string_view>(idx);
    auto const token_offset = d_token_offsets[idx];
    auto const token_count  = static_cast<size_type>(d_token_offsets[idx + 1] - token_offset);
    auto* const d_result    = d_tokens + token_offset;
    auto const size         = d_str.size_bytes();
    auto const del_size     = d_delimiter.size_bytes();
    auto const base         = d_str.data();

    if (size == 0) {
      d_result[0] = string_index_pair{"", 0};
      return;
    }
    size_type token_idx = 0;
    size_type last_end  = size;
    size_type pos       = size - del_size;
    while (pos >= 0 && token_idx < token_count - 1) {
      if (d_delimiter.compare(base + pos, del_size) == 0) {
        auto const start                      = pos + del_size;
        d_result[token_count - 1 - token_idx] = string_index_pair{base + start, last_end - start};
        last_end                              = pos;
        pos -= del_size;
        ++token_idx;
      } else {
        --pos;
      }
    }
    d_result[0] = string_index_pair{base, last_end};
  }
};

// Extract whitespace tokens left-to-right. Leading/trailing whitespace is skipped; consecutive
// whitespace counts as one delimiter. The last slot retains trailing whitespace when max_tokens
// is reached.
struct split_ws_extract_fn {
  column_device_view const d_strings;
  cudf::detail::input_offsetalator const d_token_offsets;
  string_index_pair* const d_tokens;
  size_type const max_tokens;

  __device__ void operator()(size_type const idx) const
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str        = d_strings.element<string_view>(idx);
    auto const token_offset = d_token_offsets[idx];
    auto const token_count  = static_cast<size_type>(d_token_offsets[idx + 1] - token_offset);
    if (token_count == 0) { return; }
    auto* const d_result = d_tokens + token_offset;
    auto const size      = d_str.size_bytes();
    auto const base      = d_str.data();
    size_type token_idx  = 0;
    size_type i          = 0;
    while (i < size && is_whitespace(static_cast<char_utf8>(base[i]))) {
      ++i;
    }
    while (i < size && token_idx < token_count) {
      auto const tok_start = i;
      if ((token_count < max_tokens) || (token_idx + 1 < token_count)) {
        while (i < size && !is_whitespace(static_cast<char_utf8>(base[i]))) {
          ++i;
        }
        d_result[token_idx++] = string_index_pair{base + tok_start, i - tok_start};
        while (i < size && is_whitespace(static_cast<char_utf8>(base[i]))) {
          ++i;
        }
      } else {
        // cap reached at last slot: preserve rest of string including trailing whitespace
        d_result[token_idx++] = string_index_pair{base + tok_start, size - tok_start};
      }
    }
  }
};

// Extract whitespace tokens right-to-left. Trailing/leading whitespace is skipped; the first
// output slot retains leading whitespace when max_tokens is reached.
struct rsplit_ws_extract_fn {
  column_device_view const d_strings;
  cudf::detail::input_offsetalator const d_token_offsets;
  string_index_pair* const d_tokens;
  size_type const max_tokens;

  __device__ void operator()(size_type const idx) const
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str        = d_strings.element<string_view>(idx);
    auto const token_offset = d_token_offsets[idx];
    auto const token_count  = static_cast<size_type>(d_token_offsets[idx + 1] - token_offset);
    if (token_count == 0) { return; }
    auto* const d_result = d_tokens + token_offset;
    auto const size      = d_str.size_bytes();
    auto const base      = d_str.data();
    size_type token_idx  = 0;
    size_type i          = size - 1;
    while (i >= 0 && is_whitespace(static_cast<char_utf8>(base[i]))) {
      --i;
    }
    while (i >= 0 && token_idx < token_count) {
      auto const tok_end = i + 1;
      if ((token_count < max_tokens) || (token_idx + 1 < token_count)) {
        while (i >= 0 && !is_whitespace(static_cast<char_utf8>(base[i]))) {
          --i;
        }
        auto const tok_start = i + 1;
        d_result[token_count - 1 - token_idx] =
          string_index_pair{base + tok_start, tok_end - tok_start};
        ++token_idx;
        while (i >= 0 && is_whitespace(static_cast<char_utf8>(base[i]))) {
          --i;
        }
      } else {
        // cap reached at first output slot: preserve rest from beginning including leading ws
        d_result[0] = string_index_pair{base, tok_end};
        break;
      }
    }
  }
};

/**
 * @brief Common implementation for per-row split helpers
 *
 * Three kernel launches: count tokens per string, prefix-sum into offsets, extract tokens.
 * The extract functor is constructed via make_extract once d_offsets and d_tokens are known.
 * Returns the same (offsets, tokens) pair as split_helper so callers are interchangeable.
 */
template <typename CountFn, typename MakeExtractFn>
std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_per_row_impl(
  column_device_view const& d_strings,
  CountFn count_fn,
  MakeExtractFn make_extract,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const strings_count = d_strings.size();
  auto const temp_mr       = cudf::get_current_device_resource_ref();
  auto const iota_itr      = cuda::counting_iterator<size_type>{0};
  auto const policy        = rmm::exec_policy_nosync(stream, temp_mr);

  auto token_counts = rmm::device_uvector<size_type>(strings_count, stream, temp_mr);
  thrust::transform(policy, iota_itr, iota_itr + strings_count, token_counts.begin(), count_fn);

  auto [offsets, total_tokens] =
    cudf::detail::make_offsets_child_column(token_counts.begin(), token_counts.end(), stream, mr);
  auto const d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  auto tokens = rmm::device_uvector<string_index_pair>(total_tokens, stream, mr);
  if (total_tokens > 0) {
    thrust::for_each_n(policy, iota_itr, strings_count, make_extract(d_offsets, tokens.data()));
  }
  return {std::move(offsets), std::move(tokens)};
}

template <bool Forward>
std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_per_row(
  column_device_view const& d_strings,
  string_view delimiter,
  size_type max_tokens,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return split_per_row_impl(
    d_strings,
    token_count_fn{d_strings, delimiter, max_tokens},
    [d_strings, delimiter](auto d_offsets, auto* d_tokens) {
      using fn_t = std::conditional_t<Forward, split_extract_fn, rsplit_extract_fn>;
      return fn_t{d_strings, delimiter, d_offsets, d_tokens};
    },
    stream,
    mr);
}

template <bool Forward>
std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_ws_per_row(
  column_device_view const& d_strings,
  size_type max_tokens,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return split_per_row_impl(
    d_strings,
    ws_token_count_fn{d_strings, max_tokens},
    [d_strings, max_tokens](auto d_offsets, auto* d_tokens) {
      using fn_t = std::conditional_t<Forward, split_ws_extract_fn, rsplit_ws_extract_fn>;
      return fn_t{d_strings, d_offsets, d_tokens, max_tokens};
    },
    stream,
    mr);
}

template std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>>
split_per_row<true>(column_device_view const&,
                    string_view,
                    size_type,
                    cuda::stream_ref,
                    rmm::device_async_resource_ref);

template std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>>
split_per_row<false>(column_device_view const&,
                     string_view,
                     size_type,
                     cuda::stream_ref,
                     rmm::device_async_resource_ref);

template std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>>
split_ws_per_row<true>(column_device_view const&,
                       size_type,
                       cuda::stream_ref,
                       rmm::device_async_resource_ref);

template std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>>
split_ws_per_row<false>(column_device_view const&,
                        size_type,
                        cuda::stream_ref,
                        rmm::device_async_resource_ref);

std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_helper(
  strings_column_view const& input,
  rsplit_tokenizer_fn tokenizer,
  string_delimiter_fn delimiter_fn,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return split_helper<rsplit_tokenizer_fn, string_delimiter_fn>(
    input, tokenizer, delimiter_fn, stream, mr);
}

std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_helper(
  strings_column_view const& input,
  split_ws_tokenizer_fn tokenizer,
  whitespace_delimiter_fn delimiter_fn,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return split_helper<split_ws_tokenizer_fn, whitespace_delimiter_fn>(
    input, tokenizer, delimiter_fn, stream, mr);
}

std::pair<std::unique_ptr<column>, rmm::device_uvector<string_index_pair>> split_helper(
  strings_column_view const& input,
  rsplit_ws_tokenizer_fn tokenizer,
  whitespace_delimiter_fn delimiter_fn,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  return split_helper<rsplit_ws_tokenizer_fn, whitespace_delimiter_fn>(
    input, tokenizer, delimiter_fn, stream, mr);
}

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

  using delimiter_type   = std::remove_cvref_t<DelimiterFn>;
  auto [offsets, tokens] = split_helper(input, tokenizer, delimiter_type{delimiter_fn}, stream, mr);
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
  auto [offsets, tokens] = split_per_row<Forward>(*d_strings, d_delimiter, max_tokens, stream, mr);
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
  auto [offsets, tokens] = split_ws_per_row<Forward>(*d_strings, max_tokens, stream, mr);
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

/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "io/utilities/trie.hpp"

#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>

namespace cudf::io {
/**
 * @brief Non-owning view for json type inference options
 */
struct json_inference_options_view {
  char quote_char;
  cudf::detail::trie_view trie_true;
  cudf::detail::trie_view trie_false;
  cudf::detail::trie_view trie_na;
};

/**
 * @brief Structure for holding various options used when parsing and
 * converting CSV/json data to cuDF data type values.
 */
struct parse_options_view {
  char delimiter;
  char terminator;
  char quotechar;
  char decimal;
  char thousands;
  char comment;
  bool keepquotes;
  bool detect_whitespace_around_quotes;
  bool doublequote;
  bool dayfirst;
  bool skipblanklines;
  bool normalize_whitespace;
  bool mixed_types_as_string;
  cudf::detail::trie_view trie_true;
  cudf::detail::trie_view trie_false;
  cudf::detail::trie_view trie_na;
  bool multi_delimiter;
};

struct parse_options {
  char delimiter;
  char terminator;
  char quotechar;
  char decimal;
  char thousands;
  char comment;
  bool keepquotes;
  bool detect_whitespace_around_quotes;
  bool doublequote;
  bool dayfirst;
  bool skipblanklines;
  bool normalize_whitespace;
  bool mixed_types_as_string;
  cudf::detail::optional_trie trie_true;
  cudf::detail::optional_trie trie_false;
  cudf::detail::optional_trie trie_na;
  bool multi_delimiter;

  [[nodiscard]] json_inference_options_view json_view() const
  {
    return {quotechar,
            cudf::detail::make_trie_view(trie_true),
            cudf::detail::make_trie_view(trie_false),
            cudf::detail::make_trie_view(trie_na)};
  }

  [[nodiscard]] parse_options_view view() const
  {
    return {delimiter,
            terminator,
            quotechar,
            decimal,
            thousands,
            comment,
            keepquotes,
            detect_whitespace_around_quotes,
            doublequote,
            dayfirst,
            skipblanklines,
            normalize_whitespace,
            mixed_types_as_string,
            cudf::detail::make_trie_view(trie_true),
            cudf::detail::make_trie_view(trie_false),
            cudf::detail::make_trie_view(trie_na),
            multi_delimiter};
  }
};

namespace detail {

/**
 * @brief Infers data type for a given JSON string input `data`.
 *
 * @throw cudf::logic_error if input size is 0
 * @throw cudf::logic_error if date time is not inferred as string
 * @throw cudf::logic_error if data type inference failed
 *
 * @param options View of inference options
 * @param data JSON string input
 * @param offset_length_begin The beginning of an offset-length tuple sequence
 * @param size Size of the string input
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @return The inferred data type
 */
CUDF_EXPORT cudf::data_type infer_data_type(
  cudf::io::json_inference_options_view const& options,
  device_span<char const> data,
  cuda::zip_iterator<size_type const*, size_type const*> offset_length_begin,
  std::size_t const size,
  rmm::cuda_stream_view stream);
}  // namespace detail

namespace json::detail {

/**
 * @brief Parses the data from an iterator of string views, casting it to the given target data type
 *
 * @param data string input base pointer
 * @param offset_length_begin The beginning of an offset-length tuple sequence
 * @param col_size The total number of items of this column
 * @param col_type The column's target data type
 * @param null_mask A null mask that renders certain items from the input invalid
 * @param options Settings for controlling the processing behavior
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr The resource to be used for device memory allocation
 * @return The column that contains the parsed data
 */
CUDF_EXPORT std::unique_ptr<column> parse_data(
  char const* data,
  cuda::zip_iterator<size_type const*, size_type const*> offset_length_begin,
  size_type col_size,
  data_type col_type,
  rmm::device_buffer&& null_mask,
  size_type null_count,
  cudf::io::parse_options_view const& options,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
}  // namespace json::detail
}  // namespace cudf::io

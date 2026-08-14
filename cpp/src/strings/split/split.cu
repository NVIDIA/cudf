/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "split.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/split_utils.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/binary_search.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

/**
 * @brief Generic split function called by split() and rsplit().
 *
 * This function will first count the number of delimiters in the entire strings
 * column. Next it records the position of all the delimiters. These positions
 * are used for the remainder of the code to build string_index_pair elements
 * for each output column.
 *
 * The number of tokens for each string is computed by analyzing the delimiter
 * position values and mapping them to each string.
 * The number of output columns is determined by the string with the most tokens.
 * Next the `string_index_pairs` for the entire column are created using the
 * delimiter positions and their string indices vector.
 *
 * Finally, each column is built by creating a vector of tokens (`string_index_pairs`)
 * according to their position in each string. The first token from each string goes
 * into the first output column, the 2nd token from each string goes into the 2nd
 * output column, etc.
 *
 * Output should be comparable to Pandas `split()` with `expand=True` but the
 * rows/columns are transposed.
 *
 * ```
 *   import pandas as pd
 *   pd_series = pd.Series(['', None, 'a_b', '_a_b_', '__aa__bb__', '_a__bbb___c', '_aa_b__ccc__'])
 *   print(pd_series.str.split(pat='_', expand=True))
 *            0     1     2     3     4     5     6
 *      0    ''  None  None  None  None  None  None
 *      1  None  None  None  None  None  None  None
 *      2     a     b  None  None  None  None  None
 *      3    ''     a     b    ''  None  None  None
 *      4    ''    ''    aa    ''    bb    ''    ''
 *      5    ''     a    ''   bbb    ''    ''     c
 *      6    ''    aa     b    ''   ccc    ''    ''
 *
 *   print(pd_series.str.split(pat='_', n=1, expand=True))
 *            0            1
 *      0    ''         None
 *      1  None         None
 *      2     a            b
 *      3    ''         a_b_
 *      4    ''    _aa__bb__
 *      5    ''   a__bbb___c
 *      6    ''  aa_b__ccc__
 *
 *   print(pd_series.str.split(pat='_', n=2, expand=True))
 *            0     1         2
 *      0    ''  None      None
 *      1  None  None      None
 *      2     a     b      None
 *      3    ''     a        b_
 *      4    ''        aa__bb__
 *      5    ''     a  _bbb___c
 *      6    ''    aa  b__ccc__
 * ```
 *
 * The whitespace split function works differently in that consecutive whitespace is
 * treated as a single delimiter.
 *
 * ```
 *  import pandas as pd
 *  pd_series = pd.Series(['', None, 'a b', ' a b ', '  aa  bb  ', ' a  bbb   c', ' aa b  ccc  '])
 *  print(pd_series.str.split(pat=None, expand=True))
 *            0     1     2
 *      0  None  None  None
 *      1  None  None  None
 *      2     a     b  None
 *      3     a     b  None
 *      4    aa    bb  None
 *      5     a   bbb     c
 *      6    aa     b   ccc
 *
 *  print(pd_series.str.split(pat=None, n=1, expand=True))
 *            0         1
 *      0  None      None
 *      1  None      None
 *      2     a         b
 *      3     a        b
 *      4    aa      bb
 *      5     a   bbb   c
 *      6    aa  b  ccc
 *
 *  print(pd_series.str.split(pat=None, n=2, expand=True))
 *            0     1      2
 *      0  None  None   None
 *      1  None  None   None
 *      2     a     b   None
 *      3     a     b   None
 *      4    aa    bb   None
 *      5     a   bbb      c
 *      6    aa     b  ccc *
 * ```
 *
 * @tparam Tokenizer provides unique functions for split/rsplit and whitespace split/rsplit
 * @tparam DelimiterFn Functor for locating delimiters
 * @param input The strings to split
 * @param tokenizer Tokenizer for counting and producing tokens
 * @param delimiter_fn Functor called on each byte to check for delimiters
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned objects' device memory
 * @return table of columns for the output of the split
 */
template <typename Tokenizer, typename DelimiterFn>
std::unique_ptr<table> split_fn(strings_column_view const& input,
                                Tokenizer tokenizer,
                                DelimiterFn delimiter_fn,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<column>> results;
  if (input.size() == input.null_count()) {
    results.push_back(std::make_unique<column>(input.parent(), stream, mr));
    return std::make_unique<table>(std::move(results));
  }

  // builds the offsets and the vector of all tokens
  auto [offsets, tokens] = split_helper(input, tokenizer, delimiter_fn, stream, mr);
  auto const d_offsets   = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());
  auto const d_tokens    = tokens.data();

  // compute the maximum number of tokens for any string
  auto const columns_count = thrust::transform_reduce(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{input.size()},
    cuda::proclaim_return_type<size_type>([d_offsets] __device__(auto idx) -> size_type {
      return static_cast<size_type>(d_offsets[idx + 1] - d_offsets[idx]);
    }),
    0,
    cuda::maximum{});

  // build strings columns for each token position
  for (size_type col = 0; col < columns_count; ++col) {
    auto itr = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<string_index_pair>(
        [d_tokens, d_offsets, col] __device__(size_type idx) {
          auto const offset      = d_offsets[idx];
          auto const token_count = static_cast<size_type>(d_offsets[idx + 1] - offset);
          return (col < token_count) ? d_tokens[offset + col] : string_index_pair{nullptr, 0};
        }));
    results.emplace_back(make_strings_column(itr, itr + input.size(), stream, mr));
  }

  return std::make_unique<table>(std::move(results));
}

// Build an output table from a (offsets, tokens) pair produced by split_per_row_helper.
std::unique_ptr<table> build_table_from_tokens(strings_column_view const& input,
                                               column_view offsets,
                                               rmm::device_uvector<string_index_pair> const& tokens,
                                               rmm::cuda_stream_view stream,
                                               rmm::device_async_resource_ref mr)
{
  auto const d_offsets     = cudf::detail::offsetalator_factory::make_input_iterator(offsets);
  auto const d_tokens      = tokens.data();
  auto const columns_count = thrust::transform_reduce(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{input.size()},
    cuda::proclaim_return_type<size_type>([d_offsets] __device__(auto idx) -> size_type {
      return static_cast<size_type>(d_offsets[idx + 1] - d_offsets[idx]);
    }),
    0,
    cuda::maximum{});
  std::vector<std::unique_ptr<column>> results;
  for (size_type col = 0; col < columns_count; ++col) {
    auto itr = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<string_index_pair>(
        [d_tokens, d_offsets, col] __device__(size_type idx) {
          auto const offset      = d_offsets[idx];
          auto const token_count = static_cast<size_type>(d_offsets[idx + 1] - offset);
          return (col < token_count) ? d_tokens[offset + col] : string_index_pair{nullptr, 0};
        }));
    results.emplace_back(make_strings_column(itr, itr + input.size(), stream, mr));
  }
  return std::make_unique<table>(std::move(results));
}

// Create a table with a single strings column with all nulls
std::unique_ptr<table> make_all_null_table(size_type size,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  std::vector<std::unique_ptr<column>> results;
  auto mask = cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr);
  results.push_back(std::make_unique<column>(
    data_type{type_id::STRING}, size, rmm::device_buffer{}, std::move(mask), size));
  return std::make_unique<table>(std::move(results));
}

}  // namespace

std::unique_ptr<table> split(strings_column_view const& input,
                             string_scalar const& delimiter,
                             size_type maxsplit,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  size_type max_tokens = maxsplit > 0 ? maxsplit + 1 : std::numeric_limits<size_type>::max();

  auto d_strings = column_device_view::create(input.parent(), stream);
  if (delimiter.size() == 0) {
    auto tokenizer    = split_ws_tokenizer_fn{*d_strings, max_tokens};
    auto delimiter_fn = whitespace_delimiter_fn{};
    auto results      = split_fn(input, tokenizer, delimiter_fn, stream, mr);
    // boundary case: if no columns, return one null column (issue #119)
    return (results->num_columns() == 0) ? make_all_null_table(input.size(), stream, mr)
                                         : std::move(results);
  }

  auto const non_null_count = input.size() - input.null_count();
  if (non_null_count > 0 &&
      (input.chars_size(stream) / non_null_count) < AVG_CHAR_BYTES_THRESHOLD) {
    auto tmp_mr = cudf::get_current_device_resource_ref();
    auto [offsets, tokens] =
      split_per_row_helper<true>(*d_strings, delimiter.value(stream), max_tokens, stream, tmp_mr);
    return build_table_from_tokens(input, offsets->view(), tokens, stream, mr);
  }

  auto tokenizer    = split_tokenizer_fn{*d_strings, delimiter.size(), max_tokens};
  auto delimiter_fn = string_delimiter_fn{delimiter.value(stream)};
  return split_fn(input, tokenizer, delimiter_fn, stream, mr);
}

std::unique_ptr<table> rsplit(strings_column_view const& input,
                              string_scalar const& delimiter,
                              size_type maxsplit,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(delimiter.is_valid(stream), "Parameter delimiter must be valid");

  size_type max_tokens = maxsplit > 0 ? maxsplit + 1 : std::numeric_limits<size_type>::max();

  auto d_strings = column_device_view::create(input.parent(), stream);
  if (delimiter.size() == 0) {
    auto tokenizer    = rsplit_ws_tokenizer_fn{*d_strings, max_tokens};
    auto delimiter_fn = whitespace_delimiter_fn{};
    auto results      = split_fn(input, tokenizer, delimiter_fn, stream, mr);
    // boundary case: if no columns, return one null column (issue #119)
    return (results->num_columns() == 0) ? make_all_null_table(input.size(), stream, mr)
                                         : std::move(results);
  }

  auto const non_null_count = input.size() - input.null_count();
  if (non_null_count > 0 &&
      (input.chars_size(stream) / non_null_count) < AVG_CHAR_BYTES_THRESHOLD) {
    auto tmp_mr = cudf::get_current_device_resource_ref();
    auto [offsets, tokens] =
      split_per_row_helper<false>(*d_strings, delimiter.value(stream), max_tokens, stream, tmp_mr);
    return build_table_from_tokens(input, offsets->view(), tokens, stream, mr);
  }

  auto tokenizer    = rsplit_tokenizer_fn{*d_strings, delimiter.size(), max_tokens};
  auto delimiter_fn = string_delimiter_fn{delimiter.value(stream)};
  return split_fn(input, tokenizer, delimiter_fn, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<table> split(strings_column_view const& input,
                             string_scalar const& delimiter,
                             size_type maxsplit,
                             rmm::cuda_stream_view stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::split(input, delimiter, maxsplit, stream, mr);
}

std::unique_ptr<table> rsplit(strings_column_view const& input,
                              string_scalar const& delimiter,
                              size_type maxsplit,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rsplit(input, delimiter, maxsplit, stream, mr);
}

}  // namespace strings
}  // namespace cudf

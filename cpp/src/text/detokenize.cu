/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/copy_if.cuh>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/tokenize.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <cuda/stream>

namespace nvtext {
namespace detail {
namespace {
/**
 * @brief Generate strings from tokens.
 *
 * Each string is created by appending all the tokens assigned to
 * the same row. The `d_separator` is appended between each token.
 */
struct detokenizer_fn {
  cudf::column_device_view const d_strings;    // these are the tokens
  cudf::size_type const* d_row_map;            // indices sorted by output row
  cudf::size_type const* d_token_offsets;      // to each input token array
  cudf::string_view const d_separator;         // append after each token
  cudf::size_type* d_sizes{};                  // output sizes
  char* d_chars{};                             // output buffer for characters
  cudf::detail::input_offsetalator d_offsets;  // for addressing output row data in d_chars

  __device__ void operator()(cudf::size_type idx)
  {
    auto const offset      = d_token_offsets[idx];
    auto d_tokens          = d_row_map + offset;
    auto const token_count = d_token_offsets[idx + 1] - offset;
    auto out_ptr           = d_chars ? d_chars + d_offsets[idx] : nullptr;
    cudf::size_type nbytes = 0;
    for (cudf::size_type jdx = 0; jdx < token_count; ++jdx) {
      auto const str_index = d_tokens[jdx];
      if (d_strings.is_null(str_index)) continue;
      auto const d_str = d_strings.element<cudf::string_view>(str_index);
      if (out_ptr) {
        out_ptr = cudf::strings::detail::copy_string(out_ptr, d_str);
        if (jdx + 1 < token_count)
          out_ptr = cudf::strings::detail::copy_string(out_ptr, d_separator);
      } else {
        nbytes += d_str.size_bytes();
        nbytes += d_separator.size_bytes();
      }
    }
    if (!d_chars) { d_sizes[idx] = (nbytes > 0) ? (nbytes - d_separator.size_bytes()) : 0; }
  }
};

struct index_changed_fn {
  cudf::detail::input_indexalator const d_rows;
  cudf::size_type const* d_row_map;
  __device__ bool operator()(cudf::size_type idx) const
  {
    return (idx == 0) || (d_rows[d_row_map[idx]] != d_rows[d_row_map[idx - 1]]);
  }
};

/**
 * @brief Convert the row indices into token offsets
 *
 * @param row_indices Indices where each token should land
 * @param sorted_indices Map of row_indices sorted
 * @param tokens_counts Token counts for each row
 * @param stream CUDA stream used for kernel launches
 */
rmm::device_uvector<cudf::size_type> create_token_row_offsets(
  cudf::column_view const& row_indices,
  cudf::column_view const& sorted_indices,
  cudf::size_type tokens_counts,
  cuda::stream_ref stream)
{
  index_changed_fn fn{cudf::detail::indexalator_factory::make_input_iterator(row_indices),
                      sorted_indices.data<cudf::size_type>()};

  auto const output_count =
    cudf::detail::count_if(cuda::counting_iterator<cudf::size_type>{0},
                           cuda::counting_iterator<cudf::size_type>{tokens_counts},
                           fn,
                           stream);

  auto tokens_offsets = rmm::device_uvector<cudf::size_type>(output_count + 1, stream);

  cudf::detail::copy_if_async(cuda::counting_iterator<cudf::size_type>{0},
                              cuda::counting_iterator<cudf::size_type>{tokens_counts},
                              tokens_offsets.begin(),
                              fn,
                              stream);

  // set the last element to the total number of tokens
  tokens_offsets.set_element(output_count, tokens_counts, stream);
  return tokens_offsets;
}

}  // namespace

/**
 * @copydoc nvtext::detokenize
 */
std::unique_ptr<cudf::column> detokenize(cudf::strings_column_view const& strings,
                                         cudf::column_view const& row_indices,
                                         cudf::string_scalar const& separator,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(separator.is_valid(stream), "Parameter separator must be valid");
  CUDF_EXPECTS(row_indices.size() == strings.size(),
               "Parameter row_indices must be the same size as the input column");
  CUDF_EXPECTS(not row_indices.has_nulls(), "Parameter row_indices must not have nulls");

  auto tokens_counts = strings.size();
  if (tokens_counts == 0)  // if no input strings, return an empty column
    return cudf::make_empty_column(cudf::data_type{cudf::type_id::STRING});

  auto strings_column = cudf::column_device_view::create(strings.parent(), stream);
  // the indices may not be in order so we need to build a sorted map
  auto sorted_rows = cudf::detail::stable_sorted_order(
    cudf::table_view({row_indices}), {}, {}, stream, cudf::get_current_device_resource_ref());
  auto const d_row_map = sorted_rows->view().data<cudf::size_type>();

  // create offsets for the tokens for each output string
  auto tokens_offsets =
    create_token_row_offsets(row_indices, sorted_rows->view(), tokens_counts, stream);
  auto const output_count = tokens_offsets.size() - 1;  // number of output strings

  cudf::string_view const d_separator(separator.data(), separator.size());

  auto [offsets_column, chars] = cudf::strings::detail::make_strings_children(
    detokenizer_fn{*strings_column, d_row_map, tokens_offsets.data(), d_separator},
    output_count,
    stream,
    mr);

  // make the output strings column from the offsets and chars column
  return cudf::make_strings_column(
    output_count, std::move(offsets_column), chars.release(), 0, rmm::device_buffer{});
}

}  // namespace detail

std::unique_ptr<cudf::column> detokenize(cudf::strings_column_view const& input,
                                         cudf::column_view const& row_indices,
                                         cudf::string_scalar const& separator,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::detokenize(input, row_indices, separator, stream, mr);
}

}  // namespace nvtext

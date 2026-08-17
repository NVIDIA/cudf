/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "split.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/split_utils.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/split/split.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

struct split_part_ws_tokenizer_fn : base_ws_split_tokenizer<split_part_ws_tokenizer_fn> {
  __device__ void process_tokens(int64_t pos_begin,
                                 int64_t pos_end,
                                 device_span<int64_t const> delimiters,
                                 device_span<string_index_pair> d_tokens) const
  {
    auto const base_ptr    = d_strings.head<char>();
    auto const token_count = static_cast<size_type>(d_tokens.size());

    auto token_idx = size_type{0};
    auto last_pos  = pos_begin;
    for (size_t di = 0; di < delimiters.size() && token_idx < token_count; ++di) {
      auto const d_pos = delimiters[di];
      if (last_pos == d_pos) {
        ++last_pos;
        continue;
      }
      d_tokens[token_idx++] = string_index_pair{base_ptr + last_pos, d_pos - last_pos};
      last_pos              = d_pos + 1;
    }
    // include anything leftover
    if (token_idx < token_count) {
      d_tokens[token_idx] = string_index_pair{base_ptr + last_pos, pos_end - last_pos};
    }
  }

  split_part_ws_tokenizer_fn(column_device_view const& d_strings, size_type index)
    : base_ws_split_tokenizer(d_strings, index + 2)
  {
  }
};

template <typename Tokenizer, typename DelimiterFn>
std::unique_ptr<column> split_part_fn(strings_column_view const& input,
                                      size_type index,
                                      Tokenizer tokenizer,
                                      DelimiterFn delimiter_fn,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  if (input.size() == input.null_count()) {
    return std::make_unique<column>(input.parent(), stream, mr);
  }

  // builds the offsets and the vector of all tokens
  auto [offsets, tokens] = split_helper(input, tokenizer, delimiter_fn, stream, mr);
  auto const d_offsets   = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());
  auto const d_tokens    = tokens.data();

  // get just the indexed value of each element
  auto d_indices = rmm::device_uvector<string_index_pair>(input.size(), stream);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator<size_type>{input.size()},
                    d_indices.begin(),
                    [d_offsets, d_tokens, index] __device__(size_type idx) {
                      auto const offset      = d_offsets[idx];
                      auto const token_count = static_cast<size_type>(d_offsets[idx + 1] - offset);
                      return (index < token_count) ? d_tokens[offset + index]
                                                   : string_index_pair{nullptr, 0};
                    });

  return make_strings_column(d_indices.begin(), d_indices.end(), stream, mr);
}

}  // namespace

std::unique_ptr<column> split_part(strings_column_view const& input,
                                   string_scalar const& delimiter,
                                   size_type index,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(
    delimiter.is_valid(stream), "Parameter delimiter must be valid", std::invalid_argument);
  CUDF_EXPECTS(
    index >= 0, "Parameter index must be greater than or equal to 0", std::invalid_argument);

  auto d_strings = column_device_view::create(input.parent(), stream);
  if (delimiter.size() == 0) {
    auto tokenizer    = split_part_ws_tokenizer_fn{*d_strings, index};
    auto delimiter_fn = whitespace_delimiter_fn{};
    return split_part_fn(input, index, tokenizer, delimiter_fn, stream, mr);
  }

  // Set the max_tokens to stop splitting once index is achieved.
  // The max_tokens is set to index+2 to ensure a complete split occurs at index.
  auto tokenizer    = split_tokenizer_fn{*d_strings, delimiter.size(), index + 2};
  auto delimiter_fn = string_delimiter_fn{delimiter.value(stream)};
  return split_part_fn(input, index, tokenizer, delimiter_fn, stream, mr);
}
}  // namespace detail

std::unique_ptr<column> split_part(strings_column_view const& input,
                                   string_scalar const& delimiter,
                                   size_type index,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::split_part(input, delimiter, index, stream, mr);
}

}  // namespace strings
}  // namespace cudf

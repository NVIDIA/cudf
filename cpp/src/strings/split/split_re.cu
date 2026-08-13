/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/count_matches.hpp"
#include "strings/regex/regex_program_impl.h"
#include "strings/regex/utilities.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/split/split_re.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/iterator>
#include <cuda/std/tuple>
#include <thrust/transform_reduce.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

enum class split_direction {
  FORWARD,  ///< for split logic
  BACKWARD  ///< for rsplit logic
};

/**
 * @brief Identify the tokens from the `idx'th` string element of `d_strings`.
 *
 * Each string's tokens are stored in the `d_tokens` vector.
 * The `d_token_offsets` specifies the output position within `d_tokens`
 * for each string.
 */
struct token_reader_fn {
  column_device_view const d_strings;
  split_direction const direction;
  cudf::detail::input_offsetalator const d_token_offsets;
  string_index_pair* d_tokens;

  template <typename ProgDevice>
  __device__ void operator()(size_type const idx, ProgDevice const prog, int32_t const prog_idx)
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();

    auto const token_offset = d_token_offsets[idx];
    auto const token_count  = d_token_offsets[idx + 1] - token_offset;
    auto const d_result     = d_tokens + token_offset;  // store tokens here
    if (nchars == 0) {
      d_result[0] = string_index_pair{"", 0};
      return;
    }

    int64_t token_idx = 0;
    auto itr          = d_str.begin();
    auto last_pos     = itr;
    while (itr.position() <= nchars) {
      auto const match = prog.template find(prog_idx, d_str, itr);
      if (!match) { break; }

      auto const start_pos = cuda::std::get<0>(match_positions_to_bytes(*match, d_str, last_pos));

      // get the token (characters just before this match)
      auto const token = string_index_pair{d_str.data() + last_pos.byte_offset(),
                                           start_pos - last_pos.byte_offset()};
      // store it if we have space
      if (token_idx < token_count - 1) {
        d_result[token_idx++] = token;
      } else {
        if (direction == split_direction::FORWARD) { break; }  // we are done
        for (auto l = 0L; l < token_idx - 1; ++l) {
          d_result[l] = d_result[l + 1];  // shift left
        }
        d_result[token_idx - 1] = token;
      }
      // setup for next match
      last_pos += (match->second - last_pos.position());
      itr = last_pos + (match->first == match->second);
    }

    // set the last token to the remainder of the string
    d_result[token_idx] = string_index_pair{d_str.data() + last_pos.byte_offset(),
                                            d_str.size_bytes() - last_pos.byte_offset()};

    if (direction == split_direction::BACKWARD) {
      // update first entry -- this happens when max_tokens is hit before the end of the string
      auto const first_offset =
        d_result[0].first
          ? static_cast<size_type>(cuda::std::distance(d_str.data(), d_result[0].first))
          : 0;
      if (first_offset) {
        d_result[0] = string_index_pair{d_str.data(), first_offset + d_result[0].second};
      }
    }
  }
};

/**
 * @brief Call regex to split each input string into tokens.
 *
 * Also computes the match counts (token offsets) against the given, already-built
 * device regex program so a second device program build is not needed for this step.
 *
 * @param d_strings Strings to split
 * @param d_prog Device regex program to evaluate against each string
 * @param direction Whether tokens are generated forwards or backwards.
 * @param maxsplit The maximum number of tokens for each split.
 * @param stream CUDA stream used for kernel launches.
 */
template <typename ProgDevice>
std::pair<rmm::device_uvector<string_index_pair>, std::unique_ptr<column>> generate_tokens(
  column_device_view const& d_strings,
  ProgDevice& d_prog,
  split_direction direction,
  size_type maxsplit,
  rmm::cuda_stream_view stream)
{
  auto const strings_count = d_strings.size();
  auto const max_tokens    = maxsplit > 0 ? maxsplit : std::numeric_limits<size_type>::max();

  auto const counts = count_matches(
    d_strings, d_prog, strings_count, stream, cudf::get_current_device_resource_ref());
  auto const d_counts = counts->view().template data<size_type>();

  // convert match counts to token offsets
  auto map_fn = cuda::proclaim_return_type<size_type>(
    [d_strings, d_counts, max_tokens] __device__(auto idx) -> size_type {
      return d_strings.is_null(idx) ? 0 : cuda::std::min(d_counts[idx], max_tokens) + 1;
    });

  auto const begin = cudf::detail::make_counting_transform_iterator(0, map_fn);
  auto const end   = begin + strings_count;

  auto [offsets, total_tokens] = cudf::detail::make_offsets_child_column(
    begin, end, stream, cudf::get_current_device_resource_ref());
  auto const d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // build a vector of tokens
  rmm::device_uvector<string_index_pair> tokens(total_tokens, stream);
  if (total_tokens > 0) {
    auto tr_fn = token_reader_fn{d_strings, direction, d_offsets, tokens.data()};
    launch_for_each_kernel(tr_fn, d_prog, d_strings.size(), stream);
  }
  return std::pair(std::move(tokens), std::move(offsets));
}

/**
 * @brief Builds the device regex program once and generates the split tokens with it.
 */
std::pair<rmm::device_uvector<string_index_pair>, std::unique_ptr<column>> generate_tokens(
  column_device_view const& d_strings,
  regex_program const& prog,
  split_direction direction,
  size_type maxsplit,
  rmm::cuda_stream_view stream)
{
  if (regex_device_builder::glushkov_fast_path_supported(prog)) {
    auto d_prog = regex_device_builder::create_gkprog_device(prog, stream);
    return generate_tokens(d_strings, *d_prog, direction, maxsplit, stream);
  }
  auto d_prog = regex_device_builder::create_prog_device(prog, stream);
  return generate_tokens(d_strings, *d_prog, direction, maxsplit, stream);
}

/**
 * @brief Returns string pair for the specified column for each string in `d_strings`
 *
 * This is used to build the table result of a split.
 * Null is returned if the row is null or if the `column_index` is larger
 * than the token count for that string.
 */
struct tokens_transform_fn {
  column_device_view const d_strings;
  string_index_pair const* d_tokens;
  cudf::detail::input_offsetalator const d_token_offsets;
  size_type const column_index;

  __device__ string_index_pair operator()(size_type idx) const
  {
    auto const offset      = d_token_offsets[idx];
    auto const token_count = static_cast<size_type>(d_token_offsets[idx + 1] - offset);
    return (column_index >= token_count) || d_strings.is_null(idx)
             ? string_index_pair{nullptr, 0}
             : d_tokens[offset + column_index];
  }
};

std::unique_ptr<table> split_re(strings_column_view const& input,
                                regex_program const& prog,
                                split_direction direction,
                                size_type maxsplit,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!prog.pattern().empty(), "Parameter pattern must not be empty");

  auto const strings_count = input.size();

  std::vector<std::unique_ptr<column>> results;
  if (strings_count == 0) {
    results.push_back(make_empty_column(type_id::STRING));
    return std::make_unique<table>(std::move(results));
  }

  auto d_strings = column_device_view::create(input.parent(), stream);

  // get the split tokens from the input column; this also computes the match counts
  // and converts them into offsets, using a single device regex program build
  auto [tokens, offsets] = generate_tokens(*d_strings, prog, direction, maxsplit, stream);
  auto const d_offsets   = cudf::detail::offsetalator_factory::make_input_iterator(offsets->view());

  // the output column count is the maximum number of tokens generated for any input string
  auto const columns_count = thrust::transform_reduce(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{strings_count},
    cuda::proclaim_return_type<size_type>([d_offsets] __device__(auto const idx) -> size_type {
      return static_cast<size_type>(d_offsets[idx + 1] - d_offsets[idx]);
    }),
    0,
    cuda::maximum<size_type>{});

  // boundary case: if no columns, return one all-null column (custrings issue #119)
  if (columns_count == 0) {
    results.push_back(std::make_unique<column>(
      data_type{type_id::STRING},
      strings_count,
      rmm::device_buffer{0, stream, mr},  // no data
      cudf::detail::create_null_mask(strings_count, mask_state::ALL_NULL, stream, mr),
      strings_count));
    return std::make_unique<table>(std::move(results));
  }

  // convert the tokens into multiple strings columns
  auto d_tokens            = tokens.data();
  auto make_strings_lambda = [&](size_type column_index) {
    // returns appropriate token for each row/column
    auto indices_itr = cudf::detail::make_counting_transform_iterator(
      0, tokens_transform_fn{*d_strings, d_tokens, d_offsets, column_index});
    return make_strings_column(indices_itr, indices_itr + strings_count, stream, mr);
  };
  // build a vector of columns
  results.resize(columns_count);
  std::transform(cuda::counting_iterator<size_type>{0},
                 cuda::counting_iterator<size_type>{columns_count},
                 results.begin(),
                 make_strings_lambda);

  return std::make_unique<table>(std::move(results));
}

std::unique_ptr<column> split_record_re(strings_column_view const& input,
                                        regex_program const& prog,
                                        split_direction direction,
                                        size_type maxsplit,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!prog.pattern().empty(), "Parameter pattern must not be empty");

  auto const strings_count = input.size();
  auto d_strings           = column_device_view::create(input.parent(), stream);

  // get the split tokens from the input column; this also computes the match counts
  // and converts them into offsets, using a single device regex program build
  auto [tokens, offsets] = generate_tokens(*d_strings, prog, direction, maxsplit, stream);

  CUDF_EXPECTS(tokens.size() < static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "Size of output exceeds the column size limit",
               std::overflow_error);

  // convert the tokens into one big strings column
  auto strings_output = make_strings_column(tokens.begin(), tokens.end(), stream, mr);

  // create a lists column using the offsets and the strings columns
  return make_lists_column(strings_count,
                           std::move(offsets),
                           std::move(strings_output),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace

std::unique_ptr<table> split_re(strings_column_view const& input,
                                regex_program const& prog,
                                size_type maxsplit,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  return split_re(input, prog, split_direction::FORWARD, maxsplit, stream, mr);
}

std::unique_ptr<column> split_record_re(strings_column_view const& input,
                                        regex_program const& prog,
                                        size_type maxsplit,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  return split_record_re(input, prog, split_direction::FORWARD, maxsplit, stream, mr);
}

std::unique_ptr<table> rsplit_re(strings_column_view const& input,
                                 regex_program const& prog,
                                 size_type maxsplit,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  return split_re(input, prog, split_direction::BACKWARD, maxsplit, stream, mr);
}

std::unique_ptr<column> rsplit_record_re(strings_column_view const& input,
                                         regex_program const& prog,
                                         size_type maxsplit,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  return split_record_re(input, prog, split_direction::BACKWARD, maxsplit, stream, mr);
}

}  // namespace detail

// external APIs

std::unique_ptr<table> split_re(strings_column_view const& input,
                                regex_program const& prog,
                                size_type maxsplit,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::split_re(input, prog, maxsplit, stream, mr);
}

std::unique_ptr<column> split_record_re(strings_column_view const& input,
                                        regex_program const& prog,
                                        size_type maxsplit,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::split_record_re(input, prog, maxsplit, stream, mr);
}

std::unique_ptr<table> rsplit_re(strings_column_view const& input,
                                 regex_program const& prog,
                                 size_type maxsplit,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rsplit_re(input, prog, maxsplit, stream, mr);
}

std::unique_ptr<column> rsplit_record_re(strings_column_view const& input,
                                         regex_program const& prog,
                                         size_type maxsplit,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::rsplit_record_re(input, prog, maxsplit, stream, mr);
}

}  // namespace strings
}  // namespace cudf

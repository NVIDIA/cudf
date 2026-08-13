/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/count_matches.hpp"
#include "strings/regex/regex_program_impl.h"
#include "strings/regex/utilities.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/extract.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <thrust/transform_scan.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

/**
 * @brief Functor extracts matched string pointers for each input string.
 *
 * For regex match within a string, the specified groups are extracted into
 * the `d_indices` output vector.
 * The `d_offsets` are pre-computed to identify the location of where each
 * string's output groups are to be written.
 */
struct extract_fn {
  column_device_view const d_strings;
  size_type const* d_offsets;
  string_index_pair* d_indices;

  __device__ void operator()(size_type const idx,
                             reprog_device const d_prog,
                             int32_t const prog_idx)
  {
    if (d_strings.is_null(idx)) { return; }

    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();

    auto const groups    = d_prog.group_counts();
    auto d_output        = d_indices + d_offsets[idx];
    size_type output_idx = 0;

    auto itr = d_str.begin();

    while (itr.position() < nchars) {
      // first, match the regex
      auto const match = d_prog.find(prog_idx, d_str, itr);
      if (!match) { break; }
      itr += (match->first - itr.position());  // position to beginning of the match
      auto last_pos = itr;
      // extract each group into the output
      for (auto group_idx = 0; group_idx < groups; ++group_idx) {
        // result is an optional containing the bounds of the extracted string at group_idx
        auto const extracted = d_prog.extract(prog_idx, d_str, itr, match->second, group_idx);
        if (extracted) {
          auto const d_result = string_from_match(*extracted, d_str, last_pos);
          d_output[group_idx + output_idx] =
            string_index_pair{d_result.data(), d_result.size_bytes()};
        } else {
          d_output[group_idx + output_idx] = string_index_pair{nullptr, 0};
        }
        last_pos += (extracted->second - last_pos.position());
      }
      // point to the end of this match to start the next match
      itr += (match->second - itr.position());
      output_idx += groups;
    }
  }
};

}  // namespace

/**
 * @copydoc cudf::strings::extract_all_record
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> extract_all_record(strings_column_view const& input,
                                           regex_program const& prog,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto const strings_count = input.size();
  auto const d_strings     = column_device_view::create(input.parent(), stream);

  // create device object from regex_program
  auto d_prog = regex_device_builder::create_prog_device(prog, stream);

  // The extract pattern should always include groups.
  auto const groups = d_prog->group_counts();
  CUDF_EXPECTS(groups > 0, "extract_all requires group indicators in the regex pattern.");

  // Get the match counts for each string, reusing the device program built above
  // instead of building a second one.
  // This column will become the output lists child offsets column.
  auto counts   = count_matches(*d_strings, *d_prog, strings_count, stream, mr);
  auto d_counts = counts->mutable_view().data<size_type>();

  // Compute null output rows
  auto [null_mask, null_count] = cudf::detail::valid_if(
    d_counts, d_counts + strings_count, [] __device__(auto v) { return v > 0; }, stream, mr);

  // Return an empty lists column if there are no valid rows
  if (strings_count == null_count) {
    return cudf::lists::detail::make_empty_lists_column(data_type{type_id::STRING});
  }

  // Convert counts into offsets.
  // Multiply each count by the number of groups.
  auto sizes_itr = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<size_type>([d_counts, groups] __device__(auto idx) {
      return d_counts[idx] * groups;
    }));
  auto [offsets, total_strings] =
    cudf::detail::make_offsets_child_column(sizes_itr, sizes_itr + strings_count, stream, mr);
  auto d_offsets = offsets->view().data<size_type>();

  rmm::device_uvector<string_index_pair> indices(total_strings, stream);

  launch_for_each_kernel(
    extract_fn{*d_strings, d_offsets, indices.data()}, *d_prog, strings_count, stream);

  auto strings_output = make_strings_column(indices.begin(), indices.end(), stream, mr);

  // Build the lists column from the offsets and the strings.
  return make_lists_column(
    strings_count, std::move(offsets), std::move(strings_output), null_count, std::move(null_mask));
}

}  // namespace detail

// external API

std::unique_ptr<column> extract_all_record(strings_column_view const& input,
                                           regex_program const& prog,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::extract_all_record(input, prog, stream, mr);
}

}  // namespace strings
}  // namespace cudf

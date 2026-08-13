/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/count_matches.hpp"
#include "strings/regex/regex_program_impl.h"
#include "strings/regex/utilities.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/lists/detail/lists_column_factories.hpp>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/findall.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

namespace cudf {
namespace strings {
namespace detail {

namespace {

/**
 * @brief This functor handles extracting matched strings by applying the compiled regex pattern
 * and creating string_index_pairs for all the substrings.
 */
struct findall_fn {
  column_device_view const d_strings;
  size_type const* d_offsets;
  string_index_pair* d_indices;

  template <typename ProgDevice>
  __device__ void operator()(size_type const idx, ProgDevice const prog, int32_t const prog_idx)
  {
    if (d_strings.is_null(idx)) { return; }
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();

    auto d_output        = d_indices + d_offsets[idx];
    size_type output_idx = 0;

    auto itr = d_str.begin();
    while (itr.position() <= nchars) {
      auto const match = prog.template find(prog_idx, d_str, itr);
      if (!match) { break; }

      auto const d_result    = string_from_match(*match, d_str, itr);
      d_output[output_idx++] = string_index_pair{d_result.data(), d_result.size_bytes()};

      itr += (match->second - itr.position()) + (match->first == match->second);
    }
  }
};

/**
 * @brief Extracts the first capture group instead of the whole match
 */
struct one_capture_fn {
  column_device_view const d_strings;
  size_type const* d_offsets;
  string_index_pair* d_indices;

  __device__ void operator()(size_type const idx,
                             reprog_device const d_prog,
                             int32_t const prog_idx)
  {
    if (d_strings.is_null(idx)) { return; }

    auto const d_str = d_strings.element<string_view>(idx);
    auto const bytes = d_str.size_bytes();

    auto d_output   = d_indices + d_offsets[idx];
    auto output_idx = size_type{0};

    auto itr = d_str.begin();
    while (itr.byte_offset() <= bytes) {
      auto const match = d_prog.find(prog_idx, d_str, itr);
      if (!match) { break; }
      itr += (match->first - itr.position());  // position to beginning of the match
      auto const result = d_prog.extract(prog_idx, d_str, itr, match->second, 0);
      if (result) {
        auto const ext_str   = string_from_match(*result, d_str, itr);
        d_output[output_idx] = string_index_pair{ext_str.data(), ext_str.size_bytes()};
      } else {
        d_output[output_idx] = string_index_pair{"", 0};  // empty string
      }
      if (itr.byte_offset() >= bytes) { break; }
      itr += (match->second - itr.position()) + (match->first == match->second);
      ++output_idx;
    }
  }
};

}  // namespace

//
std::unique_ptr<column> findall(strings_column_view const& input,
                                regex_program const& prog,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  auto const groups = prog.groups_count();
  CUDF_EXPECTS(groups <= 1, "findall does not support more than 1 capture group");

  if (input.is_empty()) {
    return cudf::lists::detail::make_empty_lists_column(input.parent().type());
  }

  auto const d_strings = column_device_view::create(input.parent(), stream);

  // Builds the lists offsets column and the matches from a single, already-built
  // device regex program (avoids a redundant device program build for the count pass).
  auto const build_matches = [&](auto& d_prog, auto make_functor) {
    auto const sizes = count_matches(*d_strings, d_prog, input.size(), stream, mr);
    auto [offsets, total_matches] =
      cudf::detail::make_offsets_child_column(sizes->view().template begin<size_type>(),
                                              sizes->view().template end<size_type>(),
                                              stream,
                                              mr);
    auto const d_offsets = offsets->view().template data<size_type>();

    rmm::device_uvector<string_index_pair> indices(total_matches, stream);
    launch_for_each_kernel(make_functor(d_offsets, indices.data()), d_prog, input.size(), stream);
    return std::pair(std::move(offsets), std::move(indices));
  };

  auto [offsets, indices] = [&]() {
    if (groups == 1) {
      auto d_prog = regex_device_builder::create_prog_device(prog, stream);
      return build_matches(*d_prog, [&](size_type const* d_offsets, string_index_pair* d_indices) {
        return one_capture_fn{*d_strings, d_offsets, d_indices};
      });
    }
    if (regex_device_builder::glushkov_fast_path_supported(prog)) {
      auto d_prog = regex_device_builder::create_gkprog_device(prog, stream);
      return build_matches(*d_prog, [&](size_type const* d_offsets, string_index_pair* d_indices) {
        return findall_fn{*d_strings, d_offsets, d_indices};
      });
    }
    auto d_prog = regex_device_builder::create_prog_device(prog, stream);
    return build_matches(*d_prog, [&](size_type const* d_offsets, string_index_pair* d_indices) {
      return findall_fn{*d_strings, d_offsets, d_indices};
    });
  }();

  auto strings_output = make_strings_column(indices.begin(), indices.end(), stream, mr);

  // Build the lists column from the offsets and the strings
  return make_lists_column(input.size(),
                           std::move(offsets),
                           std::move(strings_output),
                           input.null_count(),
                           cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

namespace {
struct find_re_fn {
  column_device_view d_strings;

  template <typename ProgDevice>
  __device__ size_type operator()(size_type const idx,
                                  ProgDevice const prog,
                                  int32_t const thread_idx) const
  {
    if (d_strings.is_null(idx)) { return 0; }
    auto const d_str = d_strings.element<string_view>(idx);

    auto const result = prog.template find(thread_idx, d_str, d_str.begin());
    return result.has_value() ? result.value().first : -1;
  }
};
}  // namespace

std::unique_ptr<column> find_re(strings_column_view const& input,
                                regex_program const& prog,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  auto results = make_numeric_column(data_type{type_to_id<size_type>()},
                                     input.size(),
                                     cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                     input.null_count(),
                                     stream,
                                     mr);
  if (input.is_empty()) { return results; }

  auto d_results       = results->mutable_view().data<size_type>();
  auto const d_strings = column_device_view::create(input.parent(), stream);

  if (regex_device_builder::glushkov_fast_path_supported(prog)) {
    auto d_prog = regex_device_builder::create_gkprog_device(prog, stream);
    launch_transform_kernel(find_re_fn{*d_strings}, *d_prog, d_results, input.size(), stream);
  } else {
    auto d_prog = regex_device_builder::create_prog_device(prog, stream);
    launch_transform_kernel(find_re_fn{*d_strings}, *d_prog, d_results, input.size(), stream);
  }

  return results;
}
}  // namespace detail

// external API

std::unique_ptr<column> findall(strings_column_view const& input,
                                regex_program const& prog,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::findall(input, prog, stream, mr);
}

std::unique_ptr<column> find_re(strings_column_view const& input,
                                regex_program const& prog,
                                rmm::cuda_stream_view stream,
                                rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::find_re(input, prog, stream, mr);
}

}  // namespace strings
}  // namespace cudf

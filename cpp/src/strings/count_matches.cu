/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "strings/count_matches.hpp"
#include "strings/regex/regex_program_impl.h"
#include "strings/regex/utilities.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/regex/regex_program.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <type_traits>

namespace cudf {
namespace strings {
namespace detail {

namespace {
/**
 * @brief Kernel counts the total matches for the given regex in each string.
 */
template <positional P>
struct count_fn {
  column_device_view const d_strings;

  template <typename ProgDevice>
  __device__ int32_t operator()(size_type const idx,
                                ProgDevice const prog,
                                int32_t const thread_idx)
  {
    if (d_strings.is_null(idx)) return 0;
    auto const d_str  = d_strings.element<string_view>(idx);
    auto const nchars = d_str.length();
    int32_t count     = 0;

    auto itr = d_str.begin();
    while (itr.position() <= nchars) {
      auto result = prog.template find<P>(thread_idx, d_str, itr);
      if (!result) { break; }
      ++count;
      // increment the iterator is faster than creating a new one
      // +1 if the match was on a virtual position (e.g. word boundary)
      itr += (result->second - itr.position()) + (result->first == result->second);
    }
    return count;
  }
};

}  // namespace

template <typename ProgDevice>
std::unique_ptr<column> count_matches(column_device_view const& d_strings,
                                      ProgDevice& d_prog,
                                      size_type strings_count,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  auto results = make_numeric_column(
    data_type{type_to_id<size_type>()}, strings_count, mask_state::UNALLOCATED, stream, mr);

  if (strings_count == 0) { return results; }

  auto d_results = results->mutable_view().data<cudf::size_type>();

  // Glushkov's engine always requires the begin/end positional check; the Thompson
  // engine can skip it (cheaper) when an empty match is not possible for this pattern.
  if constexpr (std::is_same_v<ProgDevice, gkprog_device>) {
    launch_transform_kernel(
      count_fn<positional::BEGIN_END>{d_strings}, d_prog, d_results, strings_count, stream);
  } else {
    if (d_prog.is_empty_match_possible()) {
      launch_transform_kernel(
        count_fn<positional::BEGIN_END>{d_strings}, d_prog, d_results, strings_count, stream);
    } else {
      launch_transform_kernel(
        count_fn<positional::END_ONLY>{d_strings}, d_prog, d_results, strings_count, stream);
    }
  }

  return results;
}

template std::unique_ptr<column> count_matches<reprog_device>(column_device_view const&,
                                                              reprog_device&,
                                                              size_type,
                                                              rmm::cuda_stream_view,
                                                              rmm::device_async_resource_ref);

template std::unique_ptr<column> count_matches<gkprog_device>(column_device_view const&,
                                                              gkprog_device&,
                                                              size_type,
                                                              rmm::cuda_stream_view,
                                                              rmm::device_async_resource_ref);

std::unique_ptr<column> count_matches(column_device_view const& d_strings,
                                      regex_program const& prog,
                                      rmm::cuda_stream_view stream,
                                      rmm::device_async_resource_ref mr)
{
  auto const strings_count = d_strings.size();
  if (regex_device_builder::glushkov_fast_path_supported(prog)) {
    auto d_prog = regex_device_builder::create_gkprog_device(prog, stream);
    return count_matches(d_strings, *d_prog, strings_count, stream, mr);
  }
  auto d_prog = regex_device_builder::create_prog_device(prog, stream);
  return count_matches(d_strings, *d_prog, strings_count, stream, mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf

/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/detail/combine.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/utility>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>

namespace cudf {
namespace strings {
namespace detail {

namespace {

/**
 * @brief Threshold to decide on using string-per-thread vs the string-gather
 * approaches.
 *
 * If the average byte length of a string in a column exceeds this value then
 * the string-gather function is used.
 * Otherwise, a regular string-parallel function is used.
 *
 * This value was found using the strings_join benchmark results.
 */
constexpr size_type AVG_CHAR_BYTES_THRESHOLD = 32;

struct join_base_fn {
  column_device_view const d_strings;
  string_view d_separator;
  string_scalar_device_view d_narep;

  __device__ cuda::std::pair<string_view, string_view> process_string(size_type idx) const
  {
    string_view d_str{};
    string_view d_sep = (idx + 1 < d_strings.size()) ? d_separator : d_str;
    if (d_strings.is_null(idx)) {
      if (d_narep.is_valid()) {
        d_str = d_narep.value();
      } else {
        // if null and no narep, don't output a separator either
        d_sep = d_str;
      }
    } else {
      d_str = d_strings.element<string_view>(idx);
    }
    return {d_str, d_sep};
  }
};

/**
 * @brief Compute output sizes and write output bytes
 *
 * This functor is suitable for make_strings_children
 */
struct join_fn : public join_base_fn {
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  join_fn(column_device_view const d_strings,
          string_view d_separator,
          string_scalar_device_view d_narep)
    : join_base_fn{d_strings, d_separator, d_narep}
  {
  }

  __device__ void operator()(size_type idx) const
  {
    auto const [d_str, d_sep] = process_string(idx);

    char* d_buffer  = d_chars ? d_chars + d_offsets[idx] : nullptr;
    size_type bytes = 0;
    if (d_buffer) {
      d_buffer = detail::copy_string(d_buffer, d_str);
      d_buffer = detail::copy_string(d_buffer, d_sep);
    } else {
      bytes += d_str.size_bytes() + d_sep.size_bytes();
    }
    if (!d_chars) { d_sizes[idx] = bytes; }
  }
};

struct join_gather_fn : public join_base_fn {
  join_gather_fn(column_device_view const d_strings,
                 string_view d_separator,
                 string_scalar_device_view d_narep)
    : join_base_fn{d_strings, d_separator, d_narep}
  {
  }

  __device__ string_index_pair operator()(size_type idx) const
  {
    auto const [d_str, d_sep] = process_string(idx / 2);
    // every other string is the separator
    return idx % 2 ? string_index_pair{d_sep.data(), d_sep.size_bytes()}
                   : string_index_pair{d_str.data(), d_str.size_bytes()};
  }
};
}  // namespace

std::unique_ptr<column> join_strings(strings_column_view const& input,
                                     string_scalar const& separator,
                                     string_scalar const& narep,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }

  CUDF_EXPECTS(separator.is_valid(stream), "Parameter separator must be a valid string_scalar");

  string_view d_separator(separator.data(), separator.size());
  auto d_narep = get_scalar_device_view(const_cast<string_scalar&>(narep));

  auto d_strings = column_device_view::create(input.parent(), stream);

  auto chars = [&] {
    // build the strings column and commandeer the chars column
    if ((input.size() == input.null_count()) ||
        ((input.chars_size(stream) / (input.size() - input.null_count())) <=
         AVG_CHAR_BYTES_THRESHOLD)) {
      return std::get<1>(make_strings_children(
                           join_fn{*d_strings, d_separator, d_narep}, input.size(), stream, mr))
        .release();
    }
    // dynamically feeds index pairs to build the output
    auto indices = cudf::detail::make_counting_transform_iterator(
      0, join_gather_fn{*d_strings, d_separator, d_narep});
    auto joined_col = make_strings_column(indices, indices + (input.size() * 2), stream, mr);
    auto chars_data = joined_col->release().data;
    return std::move(*chars_data);
  }();

  // API returns a single output row which cannot exceed row limit(max of size_type).
  CUDF_EXPECTS(chars.size() < static_cast<std::size_t>(std::numeric_limits<size_type>::max()),
               "The output exceeds the row size limit",
               std::overflow_error);

  // build the offsets: single string output has offsets [0,chars-size]
  auto sizes_itr      = thrust::constant_iterator(static_cast<size_type>(chars.size()));
  auto offsets_column = std::get<0>(
    cudf::strings::detail::make_offsets_child_column(sizes_itr, sizes_itr + 1, stream, mr));

  // build the null mask: only one output row so it is either all-valid or all-null
  auto const null_count =
    static_cast<size_type>(input.null_count() == input.size() && !narep.is_valid(stream));
  auto null_mask = null_count
                     ? cudf::detail::create_null_mask(1, cudf::mask_state::ALL_NULL, stream, mr)
                     : rmm::device_buffer{0, stream, mr};

  // perhaps this return a string_scalar instead of a single-row column
  return make_strings_column(
    1, std::move(offsets_column), std::move(chars), null_count, std::move(null_mask));
}

}  // namespace detail

// external API

std::unique_ptr<column> join_strings(strings_column_view const& strings,
                                     string_scalar const& separator,
                                     string_scalar const& narep,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::join_strings(strings, separator, narep, stream, mr);
}

}  // namespace strings
}  // namespace cudf

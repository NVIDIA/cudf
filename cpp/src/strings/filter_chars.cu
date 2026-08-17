/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/translate.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/host_vector.h>

#include <algorithm>

namespace cudf {
namespace strings {
namespace detail {

using char_range = cuda::std::pair<char_utf8, char_utf8>;

namespace {
/**
 * @brief This is the filter functor for replacing characters
 * in each string given a vector of char_range values.
 */
struct filter_fn {
  column_device_view const d_strings;
  filter_type keep_characters;
  rmm::device_uvector<char_range>::iterator table_begin;
  rmm::device_uvector<char_range>::iterator table_end;
  string_view const d_replacement;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  /**
   * @brief Return true if this character should be removed.
   *
   * @param ch Character to check
   * @return True if character should be removed.
   */
  __device__ bool remove_char(char_utf8 ch)
  {
    auto const entry =
      thrust::find_if(thrust::seq, table_begin, table_end, [ch] __device__(auto const& range) {
        return (range.first <= ch) && (ch <= range.second);
      });
    // if keep==true and entry-not-found OR
    // if keep==false and entry-found
    return (keep_characters == filter_type::KEEP) == (entry == table_end);
  }

  /**
   * @brief Execute the filter operation on each string.
   *
   * This is also used to calculate the size of the output.
   *
   * @param idx Index of the current string to process.
   */
  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const d_str = d_strings.element<string_view>(idx);

    auto nbytes  = d_str.size_bytes();
    auto out_ptr = d_chars ? d_chars + d_offsets[idx] : nullptr;
    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      auto const char_size        = bytes_in_char_utf8(*itr);
      string_view const d_newchar = remove_char(*itr)
                                      ? d_replacement
                                      : string_view(d_str.data() + itr.byte_offset(), char_size);
      if (out_ptr)
        out_ptr = cudf::strings::detail::copy_string(out_ptr, d_newchar);
      else
        nbytes += d_newchar.size_bytes() - char_size;
    }
    if (!out_ptr) { d_sizes[idx] = nbytes; }
  }
};

}  // namespace

/**
 * @copydoc cudf::strings::filter_characters
 */
std::unique_ptr<column> filter_characters(
  strings_column_view const& strings,
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> characters_to_filter,
  filter_type keep_characters,
  string_scalar const& replacement,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  size_type strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);
  CUDF_EXPECTS(replacement.is_valid(stream), "Parameter replacement must be valid");
  cudf::string_view d_replacement(replacement.data(), replacement.size());

  // convert input table for copy to device memory
  size_type table_size = static_cast<size_type>(characters_to_filter.size());
  auto htable          = cudf::detail::make_host_vector<char_range>(table_size, stream);
  std::transform(
    characters_to_filter.begin(), characters_to_filter.end(), htable.begin(), [](auto entry) {
      return char_range{entry.first, entry.second};
    });
  rmm::device_uvector<char_range> table =
    cudf::detail::make_device_uvector(htable, stream, cudf::get_current_device_resource_ref());

  auto d_strings = column_device_view::create(strings.parent(), stream);

  // this utility calls the strip_fn to build the offsets and chars columns
  filter_fn ffn{*d_strings, keep_characters, table.begin(), table.end(), d_replacement};
  auto [offsets_column, chars] = make_strings_children(ffn, strings.size(), stream, mr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             chars.release(),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr));
}

}  // namespace detail

/**
 * @copydoc cudf::strings::filter_characters
 */
std::unique_ptr<column> filter_characters(
  strings_column_view const& input,
  std::vector<std::pair<cudf::char_utf8, cudf::char_utf8>> characters_to_filter,
  filter_type keep_characters,
  string_scalar const& replacement,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::filter_characters(
    input, characters_to_filter, keep_characters, replacement, stream, mr);
}

}  // namespace strings
}  // namespace cudf

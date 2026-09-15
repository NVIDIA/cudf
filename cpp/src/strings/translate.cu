/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/translate.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <algorithm>

namespace cudf {
namespace strings {
namespace detail {
using translate_table = cuda::std::pair<char_utf8, char_utf8>;

namespace {
/**
 * @brief This is the translate functor for replacing individual characters
 * in each string.
 */
struct translate_fn {
  column_device_view const d_strings;
  rmm::device_uvector<translate_table>::iterator table_begin;
  rmm::device_uvector<translate_table>::iterator table_end;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    string_view const d_str = d_strings.element<string_view>(idx);

    size_type bytes = d_str.size_bytes();
    char* out_ptr   = d_chars ? d_chars + d_offsets[idx] : nullptr;
    for (auto chr : d_str) {
      auto const entry =
        thrust::lower_bound(thrust::seq,
                            table_begin,
                            table_end,
                            translate_table{chr, 0},
                            [](auto const& lhs, auto const& rhs) { return lhs.first < rhs.first; });
      if (entry != table_end && entry->first == chr) {
        bytes -= bytes_in_char_utf8(chr);
        chr = entry->second;
        if (chr)  // if null, skip the character
          bytes += bytes_in_char_utf8(chr);
      }
      if (chr && out_ptr) out_ptr += from_char_utf8(chr, out_ptr);
    }
    if (!d_chars) { d_sizes[idx] = bytes; }
  }
};

}  // namespace

//
std::unique_ptr<column> translate(strings_column_view const& strings,
                                  std::vector<std::pair<char_utf8, char_utf8>> const& chars_table,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  if (strings.is_empty()) return make_empty_column(type_id::STRING);

  size_type table_size = static_cast<size_type>(chars_table.size());
  // convert input table
  auto htable = cudf::detail::make_host_vector<translate_table>(table_size, stream);
  std::transform(chars_table.begin(), chars_table.end(), htable.begin(), [](auto entry) {
    return translate_table{entry.first, entry.second};
  });
  // The size of this table is usually much less than 100 so it is was
  // found to be more efficient to sort on the CPU than the GPU.
  thrust::sort(htable.begin(), htable.end(), [](auto const& lhs, auto const& rhs) {
    return lhs.first < rhs.first;
  });
  // copy translate table to device memory
  rmm::device_uvector<translate_table> table =
    cudf::detail::make_device_uvector(htable, stream, cudf::get_current_device_resource_ref());

  auto d_strings = column_device_view::create(strings.parent(), stream);

  auto [offsets_column, chars] = make_strings_children(
    translate_fn{*d_strings, table.begin(), table.end()}, strings.size(), stream, mr);

  return make_strings_column(strings.size(),
                             std::move(offsets_column),
                             chars.release(),
                             strings.null_count(),
                             cudf::detail::copy_bitmask(strings.parent(), stream, mr));
}

}  // namespace detail

// external APIs

std::unique_ptr<column> translate(strings_column_view const& input,
                                  std::vector<std::pair<uint32_t, uint32_t>> const& chars_table,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::translate(input, chars_table, stream, mr);
}

}  // namespace strings
}  // namespace cudf

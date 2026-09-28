/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/replace.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.cuh>
#include <cudf/strings/replace.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/functional>
#include <cuda/stream>

namespace cudf {
namespace strings {
namespace detail {
namespace {
/**
 * @brief Function logic for the replace_slice API.
 *
 * This will perform a replace_slice operation on each string.
 */
struct replace_slice_fn {
  column_device_view const d_strings;
  string_view const d_repl;
  size_type const start;
  size_type const stop;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type idx)
  {
    if (d_strings.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const d_str   = d_strings.element<string_view>(idx);
    auto const length  = d_str.length();
    char const* in_ptr = d_str.data();
    auto const begin   = d_str.byte_offset(((start < 0) || (start > length) ? length : start));
    auto const end     = d_str.byte_offset(((stop < 0) || (stop > length) ? length : stop));

    if (d_chars) {
      char* out_ptr = d_chars + d_offsets[idx];

      out_ptr = copy_and_increment(out_ptr, in_ptr, begin);  // copy beginning
      out_ptr = copy_string(out_ptr, d_repl);                // insert replacement
      out_ptr = copy_and_increment(out_ptr,                  // copy end
                                   in_ptr + end,
                                   d_str.size_bytes() - end);
    } else {
      d_sizes[idx] = d_str.size_bytes() + d_repl.size_bytes() - (end - begin);
    }
  }
};

}  // namespace

std::unique_ptr<column> replace_slice(strings_column_view const& input,
                                      string_scalar const& repl,
                                      size_type start,
                                      size_type stop,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  CUDF_EXPECTS(repl.is_valid(stream), "Parameter repl must be valid.");
  if (stop > 0) {
    CUDF_EXPECTS(start <= stop, "Parameter start must be less than or equal to stop.");
  }

  string_view d_repl(repl.data(), repl.size());

  auto d_strings = column_device_view::create(input.parent(), stream);

  // this utility calls the given functor to build the offsets and chars columns
  auto [offsets_column, chars] = make_strings_children(
    replace_slice_fn{*d_strings, d_repl, start, stop}, input.size(), stream, mr);

  return make_strings_column(input.size(),
                             std::move(offsets_column),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

}  // namespace detail

std::unique_ptr<column> replace_slice(strings_column_view const& input,
                                      string_scalar const& repl,
                                      size_type start,
                                      size_type stop,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::replace_slice(input, repl, start, stop, stream, mr);
}

}  // namespace strings
}  // namespace cudf

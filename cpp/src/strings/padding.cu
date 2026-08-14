/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/strings/detail/pad_impl.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/padding.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/iterator>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Base class for pad_fn and zfill_fn functors
 *
 * This handles the output size calculation while delegating the
 * pad operation to Derived.
 *
 * @tparam Derived class uses the CRTP pattern to reuse code logic
 *         and must include a `pad(string_view,char*)` member function.
 */
template <typename Derived>
struct base_fn {
  column_device_view const d_column;
  size_type const width;
  size_type const fill_char_size;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  base_fn(column_device_view const& d_column, size_type width, size_type fill_char_size)
    : d_column(d_column), width(width), fill_char_size(fill_char_size)
  {
  }

  __device__ void operator()(size_type idx) const
  {
    if (d_column.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }

    auto const d_str    = d_column.element<string_view>(idx);
    auto const& derived = static_cast<Derived const&>(*this);
    if (d_chars) {
      derived.pad(d_str, d_chars + d_offsets[idx]);
    } else {
      d_sizes[idx] = compute_padded_size(d_str, width, fill_char_size);
    }
  };
};

/**
 * @brief Pads each string to specified width
 *
 * @tparam side Side of the string to pad
 */
template <side_type side>
struct pad_fn : base_fn<pad_fn<side>> {
  using Base = base_fn<pad_fn<side>>;

  cudf::char_utf8 const d_fill_char;

  pad_fn(column_device_view const& d_column,
         size_type width,
         size_type fill_char_size,
         char_utf8 fill_char)
    : Base(d_column, width, fill_char_size), d_fill_char(fill_char)
  {
  }

  __device__ void pad(string_view d_str, char* output) const
  {
    pad_impl<side>(d_str, Base::width, d_fill_char, output);
  }
};

}  // namespace

std::unique_ptr<column> pad(strings_column_view const& input,
                            size_type width,
                            side_type side,
                            std::string_view fill_char,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);
  CUDF_EXPECTS(!fill_char.empty(), "fill_char parameter must not be empty");
  auto d_fill_char          = char_utf8{0};
  auto const fill_char_size = to_char_utf8(fill_char.data(), d_fill_char);

  auto d_strings = column_device_view::create(input.parent(), stream);

  auto [offsets_column, chars] = [&] {
    if (side == side_type::LEFT) {
      auto fn = pad_fn<side_type::LEFT>{*d_strings, width, fill_char_size, d_fill_char};
      return make_strings_children(fn, input.size(), stream, mr);
    } else if (side == side_type::RIGHT) {
      auto fn = pad_fn<side_type::RIGHT>{*d_strings, width, fill_char_size, d_fill_char};
      return make_strings_children(fn, input.size(), stream, mr);
    }
    auto fn = pad_fn<side_type::BOTH>{*d_strings, width, fill_char_size, d_fill_char};
    return make_strings_children(fn, input.size(), stream, mr);
  }();

  return make_strings_column(input.size(),
                             std::move(offsets_column),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

namespace {

/// Simple scalar per-row width for zfill_fn
struct constant_width {
  size_type const width;
  __device__ size_type operator[](size_type) const { return width; }
  __device__ bool is_null(size_type) const { return false; }  // compiler will optimize this away
};

/// Wrapper for a column of widths and matches the signature of constant_width above
struct column_width {
  column_device_view const d_widths;
  __device__ size_type operator[](size_type idx) const
  {
    auto const itr =
      cudf::detail::input_indexalator(d_widths.head(), d_widths.type(), d_widths.offset());
    return itr[idx];
  }
  __device__ bool is_null(size_type idx) const { return d_widths.is_null(idx); }
};

/**
 * @brief Zero-fill each string to a per-row width
 *
 * @tparam Widths Provides the width and its validity for each row
 */
template <typename Widths>
struct zfill_fn {
  column_device_view const d_strings;
  Widths const d_widths;
  size_type* d_sizes{};
  char* d_chars{};
  cudf::detail::input_offsetalator d_offsets;

  __device__ void operator()(size_type idx) const
  {
    if (d_strings.is_null(idx) || d_widths.is_null(idx)) {
      if (!d_chars) { d_sizes[idx] = 0; }
      return;
    }
    auto const d_str = d_strings.element<string_view>(idx);
    auto const width = d_widths[idx];
    if (d_chars) {
      zfill_impl(d_str, width, d_chars + d_offsets[idx]);
    } else {
      d_sizes[idx] = compute_padded_size(d_str, width, 1);
    }
  }
};
}  // namespace

std::unique_ptr<column> zfill(strings_column_view const& input,
                              size_type width,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) return make_empty_column(type_id::STRING);

  auto d_strings               = column_device_view::create(input.parent(), stream);
  auto [offsets_column, chars] = make_strings_children(
    zfill_fn<constant_width>{*d_strings, constant_width{width}}, input.size(), stream, mr);

  return make_strings_column(input.size(),
                             std::move(offsets_column),
                             chars.release(),
                             input.null_count(),
                             cudf::detail::copy_bitmask(input.parent(), stream, mr));
}

std::unique_ptr<column> zfill_by_widths(strings_column_view const& input,
                                        column_view const& widths,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  if (input.is_empty()) { return make_empty_column(type_id::STRING); }
  CUDF_EXPECTS(widths.size() == input.size(),
               "widths column must be the same size as the input column",
               std::invalid_argument);

  auto d_strings = column_device_view::create(input.parent(), stream);
  auto d_widths  = column_device_view::create(widths, stream);

  auto [offsets_column, chars] = make_strings_children(
    zfill_fn<column_width>{*d_strings, column_width{*d_widths}}, input.size(), stream, mr);

  auto [null_mask, null_count] =
    cudf::detail::bitmask_and(table_view({input.parent(), widths}), stream, mr);

  return make_strings_column(
    input.size(), std::move(offsets_column), chars.release(), null_count, std::move(null_mask));
}

}  // namespace detail

// Public APIs

std::unique_ptr<column> pad(strings_column_view const& input,
                            size_type width,
                            side_type side,
                            std::string_view fill_char,
                            rmm::cuda_stream_view stream,
                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::pad(input, width, side, fill_char, stream, mr);
}

std::unique_ptr<column> zfill(strings_column_view const& input,
                              size_type width,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::zfill(input, width, stream, mr);
}

std::unique_ptr<column> zfill_by_widths(strings_column_view const& input,
                                        column_view const& widths,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::zfill_by_widths(input, widths, stream, mr);
}

}  // namespace strings
}  // namespace cudf

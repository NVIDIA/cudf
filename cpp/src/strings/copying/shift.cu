/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/strings/detail/copying.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf::strings::detail {

namespace {

struct output_sizes_fn {
  column_device_view const d_column;  // input strings column
  string_view const d_filler;
  size_type const offset;

  __device__ size_type get_string_size_at(size_type idx)
  {
    return d_column.is_null(idx) ? 0 : d_column.element<string_view>(idx).size_bytes();
  }

  __device__ size_type operator()(size_type idx)
  {
    auto const last_index = offset < 0 ? d_column.size() + offset : offset;
    if (offset < 0) {
      // shift left:  a,b,c,d,e,f -> b,c,d,e,f,x
      return (idx < last_index) ? get_string_size_at(idx - offset) : d_filler.size_bytes();
    } else {
      // shift right:  a,b,c,d,e,f -> x,a,b,c,d,e
      return (idx < last_index) ? d_filler.size_bytes() : get_string_size_at(idx - offset);
    }
  }
};

struct shift_chars_fn {
  column_device_view const d_column;  // input strings column
  string_view const d_filler;
  int64_t const offset;

  __device__ char operator()(int64_t idx)
  {
    if (offset < 0) {
      auto const last_index = -offset;
      if (idx < last_index) {
        auto const offsets     = d_column.child(strings_column_view::offsets_column_index);
        auto const off_itr     = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
        auto const first_index = offset + off_itr[d_column.offset() + d_column.size()];
        return d_column.head<char>()[idx + first_index];
      } else {
        auto const char_index = idx - last_index;
        return d_filler.data()[char_index % d_filler.size_bytes()];
      }
    } else {
      if (idx < offset) {
        return d_filler.data()[idx % d_filler.size_bytes()];
      } else {
        auto const offsets = d_column.child(strings_column_view::offsets_column_index);
        auto const off_itr = cudf::detail::input_offsetalator(offsets.head(), offsets.type());
        return d_column.head<char>()[idx - offset + off_itr[d_column.offset()]];
      }
    }
  }
};

}  // namespace

std::unique_ptr<column> shift(strings_column_view const& input,
                              size_type offset,
                              scalar const& fill_value,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  auto d_fill_str = static_cast<string_scalar const&>(fill_value).value(stream);

  // adjust offset when greater than the size of the input
  if (std::abs(offset) > input.size()) { offset = input.size(); }

  // build the output offsets by computing the sizes of each output row
  auto const d_input = column_device_view::create(input.parent(), stream);
  auto sizes_itr     = cudf::detail::make_counting_transform_iterator(
    0, output_sizes_fn{*d_input, d_fill_str, offset});
  auto [offsets_column, total_bytes] = cudf::strings::detail::make_offsets_child_column(
    sizes_itr, sizes_itr + input.size(), stream, mr);
  auto offsets_view = offsets_column->view();

  // compute the shift-offset for the output characters child column
  auto const shift_offset = [&] {
    auto const index = (offset < 0) ? input.size() + offset : offset;
    return (offset < 0 ? -1 : 1) * get_offset_value(offsets_view, index, stream);
  }();

  // create output chars child column
  rmm::device_uvector<char> chars(total_bytes, stream, mr);
  auto d_chars = chars.data();

  // run kernel to shift all the characters
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<int64_t>{0},
                    cuda::counting_iterator<int64_t>{total_bytes},
                    d_chars,
                    shift_chars_fn{*d_input, d_fill_str, shift_offset});

  // caller sets the null-mask
  return make_strings_column(
    input.size(), std::move(offsets_column), chars.release(), 0, rmm::device_buffer{});
}

}  // namespace cudf::strings::detail

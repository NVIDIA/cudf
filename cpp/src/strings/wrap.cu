/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/strings/wrap.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {  // anonym.

// execute string wrap:
//
struct execute_wrap {
  execute_wrap(column_device_view const d_column,
               cudf::detail::input_offsetalator d_offsets,
               char* d_chars,
               size_type width)
    : d_column_(d_column), d_offsets_(d_offsets), d_chars_(d_chars), width_(width)
  {
  }

  __device__ int32_t operator()(size_type idx)
  {
    if (d_column_.is_null(idx)) return 0;  // null string

    string_view d_str = d_column_.template element<string_view>(idx);
    char* d_buffer    = d_chars_ + d_offsets_[idx];

    int charOffsetToLastSpace = -1;
    int byteOffsetToLastSpace = -1;
    int spos                  = 0;
    int bidx                  = 0;

    for (auto itr = d_str.begin(); itr != d_str.end(); ++itr) {
      auto const the_chr = *itr;
      auto const pos     = itr.position();

      // execute conditions:
      if (the_chr <= ' ') {  // convert all whitespace to space
        d_buffer[bidx]        = ' ';
        byteOffsetToLastSpace = bidx;
        charOffsetToLastSpace = pos;
      }
      if (pos - spos >= width_ && byteOffsetToLastSpace >= 0) {
        d_buffer[byteOffsetToLastSpace] = '\n';
        spos                            = charOffsetToLastSpace;
        byteOffsetToLastSpace           = -1;
        charOffsetToLastSpace           = -1;
      }
      bidx += detail::bytes_in_char_utf8(the_chr);
    }
    return 0;
  }

 private:
  column_device_view const d_column_;
  cudf::detail::input_offsetalator d_offsets_;
  char* d_chars_;
  size_type width_;
};

}  // namespace

template <typename device_execute_functor>
std::unique_ptr<column> wrap(strings_column_view const& strings,
                             size_type width,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(width > 0, "Positive wrap width required");

  auto strings_count = strings.size();
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  auto strings_column  = column_device_view::create(strings.parent(), stream);
  auto d_column        = *strings_column;
  size_type null_count = strings.null_count();

  // copy null mask
  rmm::device_buffer null_mask = cudf::detail::copy_bitmask(strings.parent(), stream, mr);

  // build offsets column
  auto offsets_column = std::make_unique<column>(strings.offsets(), stream, mr);  // makes a copy
  auto d_new_offsets =
    cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());

  auto chars_buffer = rmm::device_buffer{strings.chars_begin(stream),
                                         static_cast<std::size_t>(strings.chars_size(stream)),
                                         stream,
                                         mr};  // makes a copy
  auto d_chars      = static_cast<char*>(chars_buffer.data());

  device_execute_functor d_execute_fctr{d_column, d_new_offsets, d_chars, width};

  thrust::for_each_n(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     cuda::counting_iterator<size_type>{0},
                     strings_count,
                     d_execute_fctr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             std::move(chars_buffer),
                             null_count,
                             std::move(null_mask));
}

}  // namespace detail

std::unique_ptr<column> wrap(strings_column_view const& strings,
                             size_type width,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::wrap<detail::execute_wrap>(strings, width, stream, mr);
}

}  // namespace strings
}  // namespace cudf

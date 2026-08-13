/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/copy_range.hpp>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/for_each.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {
struct compute_element_size {
  column_device_view d_source;
  column_device_view d_target;
  size_type source_begin;
  size_type target_begin;
  size_type target_end;

  __device__ cudf::size_type operator()(cudf::size_type idx)
  {
    if (idx >= target_begin && idx < target_end) {
      auto const str_idx = source_begin + (idx - target_begin);
      return d_source.is_null(str_idx) ? 0 : d_source.element<string_view>(str_idx).size_bytes();
    } else {
      return d_target.is_null(idx) ? 0 : d_target.element<string_view>(idx).size_bytes();
    }
  }
};

}  // namespace

std::unique_ptr<column> copy_range(strings_column_view const& source,
                                   strings_column_view const& target,
                                   size_type source_begin,
                                   size_type source_end,
                                   size_type target_begin,
                                   rmm::cuda_stream_view stream,
                                   rmm::device_async_resource_ref mr)
{
  auto target_end = target_begin + (source_end - source_begin);
  CUDF_EXPECTS(
    (target_begin >= 0) && (target_begin < target.size()) && (target_end <= target.size()),
    "Range is out of bounds.",
    std::invalid_argument);

  if (target_end == target_begin) { return std::make_unique<column>(target.parent(), stream, mr); }
  auto source_device_view = column_device_view::create(source.parent(), stream);
  auto d_source           = *source_device_view;
  auto target_device_view = column_device_view::create(target.parent(), stream);
  auto d_target           = *target_device_view;

  // create null mask
  auto [null_mask, null_count] = [&] {
    if (!target.parent().nullable() && !source.parent().nullable()) {
      return std::pair(rmm::device_buffer{}, 0);
    }
    return cudf::detail::valid_if(
      cuda::counting_iterator<size_type>{0},
      cuda::counting_iterator<size_type>{target.size()},
      [d_source, d_target, source_begin, target_begin, target_end] __device__(size_type idx) {
        return (idx >= target_begin && idx < target_end)
                 ? d_source.is_valid(source_begin + (idx - target_begin))
                 : d_target.is_valid(idx);
      },
      stream,
      mr);
  }();

  // create offsets
  auto sizes_begin = cudf::detail::make_counting_transform_iterator(
    0, compute_element_size{d_source, d_target, source_begin, target_begin, target_end});
  auto [offsets_column, chars_bytes] = cudf::strings::detail::make_offsets_child_column(
    sizes_begin, sizes_begin + target.size(), stream, mr);
  auto d_offsets = cudf::detail::offsetalator_factory::make_input_iterator(offsets_column->view());

  // create chars
  auto chars_data = rmm::device_uvector<char>(chars_bytes, stream, mr);
  auto d_chars    = chars_data.data();
  thrust::for_each(
    rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator{target.size()},
    [d_source, d_target, source_begin, target_begin, target_end, d_offsets, d_chars] __device__(
      size_type idx) {
      if (d_offsets[idx + 1] - d_offsets[idx] > 0) {
        auto const source = (idx >= target_begin && idx < target_end)
                              ? d_source.element<string_view>(source_begin + (idx - target_begin))
                              : d_target.element<string_view>(idx);
        memcpy(d_chars + d_offsets[idx], source.data(), source.size_bytes());
      }
    });

  return make_strings_column(target.size(),
                             std::move(offsets_column),
                             chars_data.release(),
                             null_count,
                             std::move(null_mask));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf

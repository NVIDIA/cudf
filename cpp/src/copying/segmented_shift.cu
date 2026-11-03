/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/copy.hpp>
#include <cudf/detail/copy_if_else.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/detail/copy_if_else.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

namespace cudf {
namespace detail {

namespace {

/**
 * @brief Common filter function to convert index values into copy-if-else left/right result.
 *
 * The offset position is used to identify which segment to copy from.
 */
struct segmented_shift_filter {
  device_span<size_type const> const segment_offsets;
  size_type const offset;

  __device__ bool operator()(size_type const i) const
  {
    auto const segment_bound_idx =
      thrust::upper_bound(thrust::seq, segment_offsets.begin(), segment_offsets.end(), i) -
      (offset > 0);
    auto const left_idx  = *segment_bound_idx + (offset < 0 ? offset : 0);
    auto const right_idx = *segment_bound_idx + (offset > 0 ? offset : 0);
    return not(left_idx <= i and i < right_idx);
  };
};

template <typename T, typename Enable = void>
struct segmented_shift_functor {
  template <typename... Args>
  std::unique_ptr<column> operator()(Args&&...)
  {
    CUDF_FAIL("Unsupported type for segmented_shift.");
  }
};

/**
 * @brief Segmented shift specialization for representation layout compatible types.
 */
template <typename T>
struct segmented_shift_functor<T, std::enable_if_t<is_rep_layout_compatible<T>()>> {
  std::unique_ptr<column> operator()(column_view const& segmented_values,
                                     device_span<size_type const> segment_offsets,
                                     size_type offset,
                                     scalar const& fill_value,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
  {
    auto values_device_view = column_device_view::create(segmented_values, stream);
    bool nullable           = not fill_value.is_valid(stream) or segmented_values.nullable();
    auto input_iterator     = cudf::detail::make_optional_iterator<T>(
                            *values_device_view, nullate::DYNAMIC{segmented_values.has_nulls()}) -
                          offset;
    auto fill_iterator = cudf::detail::make_optional_iterator<T>(fill_value, nullate::YES{});
    return copy_if_else(nullable,
                        input_iterator,
                        input_iterator + segmented_values.size(),
                        fill_iterator,
                        segmented_shift_filter{segment_offsets, offset},
                        segmented_values.type(),
                        stream,
                        mr);
  }
};

/**
 * @brief Segmented shift specialization for `string_view`.
 */
template <>
struct segmented_shift_functor<string_view> {
  std::unique_ptr<column> operator()(column_view const& segmented_values,
                                     device_span<size_type const> segment_offsets,
                                     size_type offset,
                                     scalar const& fill_value,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
  {
    auto values_device_view = column_device_view::create(segmented_values, stream);
    auto input_iterator     = make_optional_iterator<cudf::string_view>(
                            *values_device_view, nullate::DYNAMIC{segmented_values.has_nulls()}) -
                          offset;
    auto fill_iterator = make_optional_iterator<cudf::string_view>(fill_value, nullate::YES{});
    return strings::detail::copy_if_else(input_iterator,
                                         input_iterator + segmented_values.size(),
                                         fill_iterator,
                                         segmented_shift_filter{segment_offsets, offset},
                                         stream,
                                         mr);
  }
};

/**
 * @brief Functor to instantiate the specializations for segmented shift and
 * forward arguments.
 */
struct segmented_shift_functor_forwarder {
  template <typename T>
  std::unique_ptr<column> operator()(column_view const& segmented_values,
                                     device_span<size_type const> segment_offsets,
                                     size_type offset,
                                     scalar const& fill_value,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr)
  {
    segmented_shift_functor<T> shifter;
    return shifter(segmented_values, segment_offsets, offset, fill_value, stream, mr);
  }
};

}  // namespace

std::unique_ptr<column> segmented_shift(column_view const& segmented_values,
                                        device_span<size_type const> segment_offsets,
                                        size_type offset,
                                        scalar const& fill_value,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  if (segmented_values.is_empty()) { return empty_like(segmented_values); }
  if (offset == 0) { return std::make_unique<column>(segmented_values, stream, mr); };

  return type_dispatcher<dispatch_storage_type>(segmented_values.type(),
                                                segmented_shift_functor_forwarder{},
                                                segmented_values,
                                                segment_offsets,
                                                offset,
                                                fill_value,
                                                stream,
                                                mr);
}

}  // namespace detail
}  // namespace cudf

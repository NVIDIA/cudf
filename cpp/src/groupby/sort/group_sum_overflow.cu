/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_reductions.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/valid_if.cuh>
#include <cudf/reduction/detail/sum_overflow.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <thrust/reduce.h>

#include <memory>
#include <utility>
#include <vector>

namespace cudf::groupby::detail {
namespace {

// Splits a reduced {sum, wraps} accumulator into the (sum, overflow) pair of the output struct.
template <typename DeviceType, typename Overflow>
struct split_accumulator {
  using accumulator = cudf::reduction::detail::
    sum_overflow_result<DeviceType, cudf::reduction::detail::sum_overflow_wraps_t<Overflow>>;

  __device__ cuda::std::tuple<DeviceType, Overflow> operator()(accumulator const& acc) const
  {
    return {acc.sum, cudf::reduction::detail::wraps_to_overflow<Overflow>(acc.wraps)};
  }
};

template <typename Source, typename Overflow>
std::unique_ptr<column> group_sum_overflow_impl(column_view const& values,
                                                size_type num_groups,
                                                cudf::device_span<size_type const> group_labels,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  using DeviceType = cudf::device_storage_type_t<Source>;
  using Wraps      = cudf::reduction::detail::sum_overflow_wraps_t<Overflow>;

  auto const dcol = cudf::column_device_view::create(values, stream);

  auto sum_child =
    cudf::make_fixed_width_column(values.type(), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto overflow_child = cudf::make_fixed_width_column(
    cudf::data_type{type_to_id<Overflow>()}, num_groups, mask_state::UNALLOCATED, stream, mr);

  // Segmented reduction per group, written straight into the two struct children.
  auto const values_in = cudf::detail::make_counting_transform_iterator(
    0, cudf::reduction::detail::null_aware_to_sum_overflow<DeviceType, Wraps>{*dcol});
  auto const children_out = cuda::transform_output_iterator{
    cuda::make_zip_iterator(sum_child->mutable_view().begin<DeviceType>(),
                            overflow_child->mutable_view().begin<Overflow>()),
    split_accumulator<DeviceType, Overflow>{}};

  thrust::reduce_by_key(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        group_labels.begin(),
                        group_labels.end(),
                        values_in,
                        cuda::make_discard_iterator(),
                        children_out,
                        cuda::std::equal_to<size_type>{},
                        cudf::reduction::detail::overflow_sum_op<DeviceType, Wraps>{});

  // A group's struct entry is null only when every row in the group is null (mirrors group_sum):
  // reduce per-row validity with logical-or, then build the mask from the per-group result.
  auto [null_mask,
        null_count] = [&]() -> std::pair<cuda::device_buffer<cudf::bitmask_type>, size_type> {
    if (!values.has_nulls()) {
      return {cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), size_type{0}};
    }
    cuda::device_buffer<bool> group_valid(
      stream, cudf::get_current_device_resource_ref(), num_groups, cuda::no_init);
    thrust::reduce_by_key(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                          group_labels.begin(),
                          group_labels.end(),
                          cudf::detail::make_validity_iterator(*dcol),
                          cuda::make_discard_iterator(),
                          group_valid.begin(),
                          cuda::std::equal_to<size_type>{},
                          cuda::std::logical_or<bool>{});
    return cudf::detail::valid_if(
      group_valid.begin(), group_valid.end(), cuda::std::identity{}, stream, mr);
  }();

  std::vector<std::unique_ptr<column>> children;
  children.push_back(std::move(sum_child));
  children.push_back(std::move(overflow_child));
  return cudf::create_structs_hierarchy(
    num_groups, std::move(children), null_count, std::move(null_mask), stream, mr);
}

struct group_sum_overflow_fn {
  template <cudf::detail::sum_overflow_supported Source>
  std::unique_ptr<column> operator()(column_view const& values,
                                     size_type num_groups,
                                     cudf::device_span<size_type const> group_labels,
                                     sum_overflow_output output,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
  {
    return output == sum_overflow_output::COUNT
             ? group_sum_overflow_impl<Source, int64_t>(
                 values, num_groups, group_labels, stream, mr)
             : group_sum_overflow_impl<Source, bool>(values, num_groups, group_labels, stream, mr);
  }

  template <typename Source, typename... Args>
    requires(!cudf::detail::sum_overflow_supported<Source>)
  std::unique_ptr<column> operator()(Args&&...) const
  {
    CUDF_FAIL("SUM_OVERFLOW is only supported for signed integral and fixed-point types");
  }
};

}  // namespace

std::unique_ptr<column> group_sum_overflow(column_view const& values,
                                           size_type num_groups,
                                           cudf::device_span<size_type const> group_labels,
                                           sum_overflow_output output,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  return cudf::type_dispatcher(
    values.type(), group_sum_overflow_fn{}, values, num_groups, group_labels, output, stream, mr);
}

}  // namespace cudf::groupby::detail

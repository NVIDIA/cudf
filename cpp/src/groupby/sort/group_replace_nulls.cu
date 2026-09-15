/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/groupby/group_replace_nulls.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/replace/nulls.cuh>
#include <cudf/replace.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/scan.h>

#include <utility>

namespace cudf {
namespace groupby {
namespace detail {

std::unique_ptr<column> group_replace_nulls(cudf::column_view const& grouped_value,
                                            device_span<size_type const> group_labels,
                                            cudf::replace_policy replace_policy,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  cudf::size_type size = grouped_value.size();

  auto device_in = cudf::column_device_view::create(grouped_value, stream);
  auto index     = cuda::counting_iterator<cudf::size_type>{0};
  auto valid_it  = cudf::detail::make_validity_iterator(*device_in);
  auto in_begin  = cuda::make_zip_iterator(cuda::std::make_tuple(index, valid_it));

  rmm::device_uvector<cudf::size_type> gather_map(size, stream);
  auto gm_begin = cuda::make_zip_iterator(
    cuda::std::make_tuple(gather_map.begin(), cuda::make_discard_iterator()));

  auto func = cudf::detail::replace_policy_functor();
  cuda::std::equal_to<cudf::size_type> eq;
  if (replace_policy == cudf::replace_policy::PRECEDING) {
    thrust::inclusive_scan_by_key(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      group_labels.begin(),
      group_labels.begin() + size,
      in_begin,
      gm_begin,
      eq,
      func);
  } else {
    auto gl_rbegin = cuda::std::make_reverse_iterator(group_labels.begin() + size);
    auto in_rbegin = cuda::std::make_reverse_iterator(in_begin + size);
    auto gm_rbegin = cuda::std::make_reverse_iterator(gm_begin + size);
    thrust::inclusive_scan_by_key(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      gl_rbegin,
      gl_rbegin + size,
      in_rbegin,
      gm_rbegin,
      eq,
      func);
  }

  auto output = cudf::detail::gather(cudf::table_view({grouped_value}),
                                     gather_map,
                                     cudf::out_of_bounds_policy::DONT_CHECK,
                                     cudf::negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);

  return std::move(output->release()[0]);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf

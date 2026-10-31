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

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/functional>
#include <cuda/std/tuple>
#include <thrust/scan.h>

#include <utility>

namespace cudf {
namespace groupby {
namespace detail {

std::unique_ptr<column> group_replace_nulls(cudf::column_view const& value,
                                            device_span<size_type const> grouped_order,
                                            device_span<size_type const> group_labels,
                                            cudf::replace_policy replace_policy,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  auto const size = static_cast<cudf::size_type>(grouped_order.size());

  // Scanning the input rows of the grouped positions yields a gather map into the ungrouped
  // column, so the values are gathered once instead of once into grouped order and once more.
  auto device_in = cudf::column_device_view::create(value, stream, mr.get_temporary_mr());
  auto index     = grouped_order.begin();
  auto valid_it =
    cuda::make_permutation_iterator(cudf::detail::make_validity_iterator(*device_in), index);
  auto in_begin = cuda::make_zip_iterator(cuda::std::make_tuple(index, valid_it));

  cuda::device_buffer<cudf::size_type> gather_map(
    stream, mr.get_temporary_mr(), size, cuda::no_init);
  auto gm_begin = cuda::make_zip_iterator(
    cuda::std::make_tuple(gather_map.begin(), cuda::make_discard_iterator()));

  auto func = cudf::detail::replace_policy_functor();
  cuda::std::equal_to<cudf::size_type> eq;
  if (replace_policy == cudf::replace_policy::PRECEDING) {
    thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
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
    thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                                  gl_rbegin,
                                  gl_rbegin + size,
                                  in_rbegin,
                                  gm_rbegin,
                                  eq,
                                  func);
  }

  auto output = cudf::detail::gather(cudf::table_view({value}),
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

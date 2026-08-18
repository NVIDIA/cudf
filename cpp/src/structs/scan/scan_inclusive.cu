/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "reductions/nested_types_extrema_utils.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/scan.h>

#include <vector>

namespace cudf {
namespace structs {
namespace detail {
namespace {

}  // namespace

template <typename Op>
std::unique_ptr<column> scan_inclusive(column_view const& input,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  // Create a gather map containing indices of the prefix min/max elements.
  auto gather_map = rmm::device_uvector<size_type>(input.size(), stream);
  auto const binop_generator =
    cudf::reduction::detail::arg_minmax_binop_generator::create<Op>(input, stream);
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         cuda::counting_iterator<size_type>{0},
                         cuda::counting_iterator<size_type>{input.size()},
                         gather_map.begin(),
                         binop_generator.binop());

  // Gather the children columns of the input column. Must use `get_sliced_child` to properly
  // handle input in case it is a sliced view.
  auto const structs_view   = structs_column_view{input};
  auto const input_children = [&] {
    auto const it = cudf::detail::make_counting_transform_iterator(
      0, [&structs_view, &stream](auto const child_idx) {
        return structs_view.get_sliced_child(child_idx, stream);
      });
    return std::vector<column_view>(it, it + input.num_children());
  }();

  // Gather the children elements of the prefix min/max struct elements for the output.
  auto scanned_children = cudf::detail::gather(table_view{input_children},
                                               gather_map,
                                               cudf::out_of_bounds_policy::DONT_CHECK,
                                               cudf::negative_index_policy::NOT_ALLOWED,
                                               stream,
                                               mr)
                            ->release();

  // Don't need to set a null mask because that will be handled at the caller.
  return make_structs_column(
    input.size(), std::move(scanned_children), 0, rmm::device_buffer{0, stream, mr}, stream, mr);
}

template std::unique_ptr<column> scan_inclusive<DeviceMin>(column_view const& input_view,
                                                           cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr);

template std::unique_ptr<column> scan_inclusive<DeviceMax>(column_view const& input_view,
                                                           cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace structs
}  // namespace cudf

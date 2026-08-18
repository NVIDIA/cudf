/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace detail {
std::unique_ptr<column> mask_to_bools(bitmask_type const* bitmask,
                                      size_type begin_bit,
                                      size_type end_bit,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  auto const length = end_bit - begin_bit;
  CUDF_EXPECTS(length >= 0, "begin_bit should be less than or equal to end_bit");
  CUDF_EXPECTS((bitmask != nullptr) or (length == 0), "nullmask is null");

  auto out_col =
    make_fixed_width_column(data_type(type_id::BOOL8), length, mask_state::UNALLOCATED, stream, mr);

  if (length > 0) {
    auto mutable_view = out_col->mutable_view();

    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<cudf::size_type>{begin_bit},
                      cuda::counting_iterator<cudf::size_type>{end_bit},
                      mutable_view.begin<bool>(),
                      [bitmask] __device__(auto index) { return bit_is_set(bitmask, index); });
  }

  return out_col;
}
}  // namespace detail

std::unique_ptr<column> mask_to_bools(bitmask_type const* bitmask,
                                      size_type begin_bit,
                                      size_type end_bit,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::mask_to_bools(bitmask, begin_bit, end_bit, stream, mr);
}
}  // namespace cudf

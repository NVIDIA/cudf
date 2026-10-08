/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "nested_types_extrema_utils.cuh"

#include <cudf/detail/copy.hpp>
#include <cudf/detail/utilities/cast_functor.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <thrust/reduce.h>

namespace cudf::reduction::detail {

std::unique_ptr<scalar> nested_minmax(column_view const& input,
                                      bool is_min_op,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  // Reduce to the ARGMIN/ARGMAX index, then return the nested element. The
  // generated operation carries the MIN/MAX choice in its state, so one instantiation
  // serves both public operations without changing comparator specialization.
  auto const binop_generator = arg_minmax_binop_generator::create(input, is_min_op, stream);
  auto const binary_op       = cudf::detail::cast_functor<size_type>(binop_generator.binop());
  auto const minmax_idx =
    thrust::reduce(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   cuda::counting_iterator<cudf::size_type>{0},
                   cuda::counting_iterator{input.size()},
                   size_type{0},
                   binary_op);

  return cudf::detail::get_element(input, minmax_idx, stream, mr);
}

}  // namespace cudf::reduction::detail

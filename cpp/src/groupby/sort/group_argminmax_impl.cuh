/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "group_argminmax.hpp"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/reduce.h>

namespace cudf::groupby::detail {

template <typename BinOp>
void launch_argminmax_reduction(cudf::device_span<cudf::size_type const> group_labels,
                                BinOp const& binop,
                                cudf::size_type* output,
                                cuda::stream_ref stream)
{
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        group_labels.data(),
                        group_labels.data() + group_labels.size(),
                        cuda::counting_iterator<cudf::size_type>{0},
                        cuda::make_discard_iterator(),
                        output,
                        cuda::std::equal_to{},
                        binop);
}

}  // namespace cudf::groupby::detail

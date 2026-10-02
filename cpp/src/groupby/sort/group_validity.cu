/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_validity.cuh"

#include <cudf/detail/iterator.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/reduce.h>

namespace cudf::groupby::detail {

void reduce_group_validity(device_span<size_type const> group_labels,
                           column_device_view const& values,
                           bool* validity,
                           cuda::stream_ref stream)
{
  thrust::reduce_by_key(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                        group_labels.data(),
                        group_labels.data() + group_labels.size(),
                        cudf::detail::make_validity_iterator(values),
                        cuda::make_discard_iterator(),
                        validity,
                        cuda::std::equal_to{},
                        cuda::std::logical_or{});
}

}  // namespace cudf::groupby::detail

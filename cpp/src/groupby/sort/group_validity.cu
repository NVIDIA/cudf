/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/sort/group_validity.cuh"

#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/utilities/error.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>

namespace cudf::groupby::detail {

void reduce_group_validity(device_span<size_type const> group_labels,
                           column_device_view const& values,
                           device_span<bool> validity,
                           cuda::stream_ref stream)
{
  CUDF_EXPECTS(group_labels.size() == static_cast<std::size_t>(values.size()),
               "Group labels must match the values' row count.");
  CUDF_EXPECTS(
    validity.size() <= group_labels.size() && (group_labels.empty() || !validity.empty()),
    "Validity output must contain one entry per group.");
  // Callers already know the group count; returning it to the host would synchronize the stream.
  cudf::detail::reduce_by_key_async(group_labels.data(),
                                    group_labels.data() + group_labels.size(),
                                    cudf::detail::make_validity_iterator(values),
                                    cuda::make_discard_iterator(),
                                    validity.begin(),
                                    cuda::std::logical_or{},
                                    stream);
}

}  // namespace cudf::groupby::detail

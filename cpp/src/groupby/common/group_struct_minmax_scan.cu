/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_struct_minmax_scan.hpp"
#include "reductions/nested_types_extrema_utils.cuh"

#include <cudf/aggregation.hpp>
#include <cudf/column/column_view.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/functional>
#include <thrust/scan.h>

namespace cudf::groupby::detail {

void scan_struct_argminmax(column_view const& values,
                           device_span<size_type const> group_labels,
                           size_type* output,
                           bool is_min,
                           cuda::stream_ref stream,
                           cudf::memory_resources mr)
{
  using generator       = cudf::reduction::detail::arg_minmax_binop_generator;
  auto const comparator = is_min ? generator::create<aggregation::MIN>(values, stream, mr)
                                 : generator::create<aggregation::MAX>(values, stream, mr);
  thrust::inclusive_scan_by_key(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                                group_labels.begin(),
                                group_labels.end(),
                                cuda::counting_iterator<size_type>{0},
                                output,
                                cuda::std::equal_to{},
                                comparator.binop());
}

}  // namespace cudf::groupby::detail

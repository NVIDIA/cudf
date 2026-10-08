/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_argminmax_impl.cuh"
#include "reductions/nested_types_extrema_utils.cuh"

#include <utility>

namespace cudf::groupby::detail {

// The nested comparator is expensive to compile; keep its owner independent of primitive types.
using nested_binop =
  decltype(std::declval<cudf::reduction::detail::arg_minmax_binop_generator>().binop());

template void launch_argminmax_reduction(cudf::device_span<cudf::size_type const>,
                                         nested_binop const&,
                                         cudf::size_type*,
                                         cuda::stream_ref);

}  // namespace cudf::groupby::detail

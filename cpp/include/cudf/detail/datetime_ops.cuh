/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/datetime.cuh>

namespace cudf {
namespace datetime {
namespace detail {
using namespace cuda::std::chrono;

template <typename Timestamp>
__device__ Timestamp add_calendrical_months_with_scale_back(Timestamp time_val, months months_val)
{
  return cudf::detail::ops::add_calendrical_months(time_val, months_val);
}

}  // namespace detail
}  // namespace datetime
}  // namespace cudf

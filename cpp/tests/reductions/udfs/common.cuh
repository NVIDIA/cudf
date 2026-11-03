/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/errc.hpp>
#include <cudf/types.hpp>

#include <cuda/std/cstdint>
#include <cuda/std/optional>

using cuda::std::int32_t;
using cuda::std::int64_t;
using cudf::errc;
using cudf::size_type;
using optional_int64 = cuda::std::optional<int64_t>;
using optional_int32 = cuda::std::optional<int32_t>;

extern "C" {
__device__ errc sum_update(size_type, optional_int64*, optional_int64*, optional_int32);
__device__ errc raw_update(size_type, int64_t*, int32_t);
__device__ errc tpch_q6_sales_finalize(
  optional_int64*, optional_int64*, cuda::std::optional<double>*, optional_int64, optional_int64);
}

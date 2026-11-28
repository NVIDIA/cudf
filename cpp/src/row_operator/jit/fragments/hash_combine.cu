/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "row_operator/jit/row_functions.cuh"

#include <cudf/hashing/detail/hashing.hpp>

namespace cudf_abi {

__device__ uint32_t hash_combine(uint32_t a, uint32_t b)
{
  return ::cudf::hashing::detail::hash_combine(a, b);
}

__device__ uint64_t hash_combine(uint64_t a, uint64_t b)
{
  return ::cudf::hashing::detail::hash_combine(static_cast<::uint64_t>(a),
                                               static_cast<::uint64_t>(b));
}

__device__ uint128_t hash_combine(uint128_t a, uint128_t b)
{
  auto const low  = hash_combine(static_cast<uint64_t>(a), static_cast<uint64_t>(b));
  auto const high = hash_combine(static_cast<uint64_t>(a >> 64), static_cast<uint64_t>(b >> 64));
  return (static_cast<uint128_t>(high) << 64) | low;
}

}  // namespace cudf_abi

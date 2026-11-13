/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>

#include <cuda/functional>
#include <cuda/std/cstddef>
#include <cuda/std/span>

namespace cudf::hashing::detail {

template <typename Key>
struct XXHash_64 {
  using result_type = std::uint64_t;

  CUDF_HOST_DEVICE constexpr XXHash_64(uint64_t seed = cudf::DEFAULT_HASH_SEED) : _seed{seed} {}

  __device__ constexpr result_type operator()(Key const& key) const
  {
    return cuda::hash<Key, cuda::hash_algorithm::xxhash_64>{_seed}(key);
  }

  __device__ constexpr result_type compute_bytes(cuda::std::byte const* bytes,
                                                 std::uint64_t size) const
  {
    return cuda::hash<cuda::std::byte const, cuda::hash_algorithm::xxhash_64>{_seed}(
      cuda::std::span<cuda::std::byte const>{bytes, size});
  }

 private:
  template <typename T>
  __device__ constexpr result_type compute(T const& key) const
  {
    return this->compute_bytes(reinterpret_cast<cuda::std::byte const*>(&key), sizeof(T));
  }

  uint64_t _seed;
};

template <>
XXHash_64<bool>::result_type __device__ inline XXHash_64<bool>::operator()(bool const& key) const
{
  return this->compute(static_cast<uint8_t>(key));
}

template <>
XXHash_64<float>::result_type __device__ inline XXHash_64<float>::operator()(float const& key) const
{
  return this->compute(normalize_nans(key));
}

template <>
XXHash_64<double>::result_type __device__ inline XXHash_64<double>::operator()(
  double const& key) const
{
  return this->compute(normalize_nans(key));
}

template <>
XXHash_64<cudf::string_view>::result_type __device__ inline XXHash_64<cudf::string_view>::
operator()(cudf::string_view const& key) const
{
  return this->compute_bytes(reinterpret_cast<cuda::std::byte const*>(key.data()),
                             key.size_bytes());
}

template <>
XXHash_64<numeric::decimal32>::result_type __device__ inline XXHash_64<numeric::decimal32>::
operator()(numeric::decimal32 const& key) const
{
  return this->compute(key.value());
}

template <>
XXHash_64<numeric::decimal64>::result_type __device__ inline XXHash_64<numeric::decimal64>::
operator()(numeric::decimal64 const& key) const
{
  return this->compute(key.value());
}

template <>
XXHash_64<numeric::decimal128>::result_type __device__ inline XXHash_64<numeric::decimal128>::
operator()(numeric::decimal128 const& key) const
{
  return this->compute(key.value());
}

}  // namespace cudf::hashing::detail

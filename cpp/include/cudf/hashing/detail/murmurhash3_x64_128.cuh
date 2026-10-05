/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/strings/string_view.cuh>

#include <cuco/hash_functions.cuh>
#include <cuda/std/array>
#include <cuda/std/cstddef>

namespace cudf::hashing::detail {

template <typename Key>
struct MurmurHash3_x64_128 {
  using result_type = cuda::std::array<uint64_t, 2>;

  CUDF_HOST_DEVICE constexpr MurmurHash3_x64_128(uint64_t seed = cudf::DEFAULT_HASH_SEED)
    : _impl{seed}
  {
  }

  __device__ constexpr result_type operator()(Key const& key) const { return this->_impl(key); }

  __device__ constexpr result_type compute_bytes(cuda::std::byte const* bytes,
                                                 std::uint64_t size) const
  { return this->_impl.compute_hash(bytes, size); }

 private:
  template <typename T>
  __device__ constexpr result_type compute(T const& key) const
  { return this->compute_bytes(reinterpret_cast<cuda::std::byte const*>(&key), sizeof(T)); }

  cuco::murmurhash3_x64_128<Key> _impl;
};

template <>
MurmurHash3_x64_128<bool>::result_type __device__ inline MurmurHash3_x64_128<bool>::operator()(
  bool const& key) const
{ return this->compute<uint8_t>(key); }

template <>
MurmurHash3_x64_128<float>::result_type __device__ inline MurmurHash3_x64_128<float>::operator()(
  float const& key) const
{ return this->compute(normalize_nans(key)); }

template <>
MurmurHash3_x64_128<double>::result_type __device__ inline MurmurHash3_x64_128<double>::operator()(
  double const& key) const
{ return this->compute(normalize_nans(key)); }

template <>
MurmurHash3_x64_128<cudf::string_view>::result_type
  __device__ inline MurmurHash3_x64_128<cudf::string_view>::operator()(
    cudf::string_view const& key) const
{
  return this->compute_bytes(reinterpret_cast<cuda::std::byte const*>(key.data()),
                             key.size_bytes());
}

template <>
MurmurHash3_x64_128<numeric::decimal32>::result_type
  __device__ inline MurmurHash3_x64_128<numeric::decimal32>::operator()(
    numeric::decimal32 const& key) const
{ return this->compute(key.value()); }

template <>
MurmurHash3_x64_128<numeric::decimal64>::result_type
  __device__ inline MurmurHash3_x64_128<numeric::decimal64>::operator()(
    numeric::decimal64 const& key) const
{ return this->compute(key.value()); }

template <>
MurmurHash3_x64_128<numeric::decimal128>::result_type
  __device__ inline MurmurHash3_x64_128<numeric::decimal128>::operator()(
    numeric::decimal128 const& key) const
{ return this->compute(key.value()); }

}  // namespace cudf::hashing::detail

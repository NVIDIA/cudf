/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/hash_functions.cuh>
#include <cudf/lists/list_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/structs/struct_view.hpp>
#include <cudf/types.hpp>

#include <cuco/hash_functions.cuh>
#include <cuda/std/cstddef>

namespace cudf::hashing::detail {

template <typename Key>
struct XXHash_32 {
  using result_type = std::uint32_t;

  CUDF_HOST_DEVICE constexpr XXHash_32(uint32_t seed = cudf::DEFAULT_HASH_SEED) : _impl{seed} {}

  __device__ constexpr result_type operator()(Key const& key) const { return this->_impl(key); }

  __device__ constexpr result_type compute_bytes(cuda::std::byte const* bytes,
                                                 std::uint64_t size) const
  { return this->_impl.compute_hash(bytes, size); }

 private:
  template <typename T>
  __device__ constexpr result_type compute(T const& key) const
  { return this->compute_bytes(reinterpret_cast<cuda::std::byte const*>(&key), sizeof(T)); }

  cuco::xxhash_32<Key> _impl;
};

template <>
XXHash_32<bool>::result_type __device__ inline XXHash_32<bool>::operator()(bool const& key) const
{ return this->compute(static_cast<uint8_t>(key)); }

template <>
XXHash_32<float>::result_type __device__ inline XXHash_32<float>::operator()(float const& key) const
{ return this->compute(normalize_nans_and_zeros(key)); }

template <>
XXHash_32<double>::result_type __device__ inline XXHash_32<double>::operator()(
  double const& key) const
{ return this->compute(normalize_nans_and_zeros(key)); }

template <>
XXHash_32<cudf::string_view>::result_type
  __device__ inline XXHash_32<cudf::string_view>::operator()(cudf::string_view const& key) const
{
  return this->compute_bytes(reinterpret_cast<cuda::std::byte const*>(key.data()),
                             key.size_bytes());
}

template <>
XXHash_32<numeric::decimal32>::result_type
  __device__ inline XXHash_32<numeric::decimal32>::operator()(numeric::decimal32 const& key) const
{ return this->compute(key.value()); }

template <>
XXHash_32<numeric::decimal64>::result_type
  __device__ inline XXHash_32<numeric::decimal64>::operator()(numeric::decimal64 const& key) const
{ return this->compute(key.value()); }

template <>
XXHash_32<numeric::decimal128>::result_type
  __device__ inline XXHash_32<numeric::decimal128>::operator()(numeric::decimal128 const& key) const
{ return this->compute(key.value()); }

template <>
XXHash_32<cudf::list_view>::result_type __device__ inline XXHash_32<cudf::list_view>::operator()(
  cudf::list_view const& key) const
{ CUDF_UNREACHABLE("List column hashing is not supported"); }

template <>
XXHash_32<cudf::struct_view>::result_type
  __device__ inline XXHash_32<cudf::struct_view>::operator()(cudf::struct_view const& key) const
{ CUDF_UNREACHABLE("Direct hashing of struct_view is not supported"); }

}  // namespace cudf::hashing::detail

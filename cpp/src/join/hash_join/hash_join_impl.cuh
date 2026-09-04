/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/join/hash_join.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_multiset.cuh>
#include <cuda/std/functional>

#include <join/join_common_utils.hpp>

namespace cudf::detail {

template <typename Hasher>
struct hash_join<Hasher>::impl {
  struct always_not_equal {
    __device__ constexpr bool operator()(
      cuco::pair<hash_value_type, size_type> const&,
      cuco::pair<hash_value_type, size_type> const&) const noexcept
    {
      // multiset always insert
      return false;
    }
  };

  struct hasher1 {
    __device__ constexpr hash_value_type operator()(
      cuco::pair<hash_value_type, size_type> const& key) const noexcept
    {
      return key.first;
    }
  };

  struct hasher2 {
    hasher2(hash_value_type seed) : _hash{seed} {}

    __device__ constexpr hash_value_type operator()(
      cuco::pair<hash_value_type, size_type> const& key) const noexcept
    {
      return _hash(key.first);
    }

   private:
    Hasher _hash;
  };

  using hash_table_t =
    cuco::static_multiset<cuco::pair<cudf::hash_value_type, cudf::size_type>,
                          cuco::extent<std::size_t>,
                          cuda::thread_scope_device,
                          always_not_equal,
                          cuco::double_hashing<DEFAULT_JOIN_CG_SIZE, hasher1, hasher2>,
                          rmm::mr::polymorphic_allocator<char>,
                          cuco::storage<2>>;

  hash_table_t _hash_table;
};

}  // namespace cudf::detail

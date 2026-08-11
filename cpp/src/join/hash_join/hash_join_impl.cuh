/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "hash_csr.cuh"

#include <cudf/detail/join/hash_join.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

#include <cstdint>

namespace cudf::detail {

template <typename Hasher>
struct hash_join<Hasher>::impl {
  impl(std::uint32_t capacity,
       size_type rows,
       rmm::cuda_stream_view stream,
       cuda::mr::any_resource<cuda::mr::device_accessible> const& mr)
    : entries(capacity, stream, mr),
      cumulative_ends(capacity, stream, mr),
      values(rows, stream, mr),
      capacity{capacity}
  {
  }

  hash_csr_map_view map_view() const
  {
    return {const_cast<std::uint64_t*>(entries.data()), capacity, capacity - 1};
  }

  hash_csr_view csr_view() const { return {cumulative_ends.data(), values.data()}; }

  rmm::device_uvector<std::uint64_t> entries;
  rmm::device_uvector<size_type> cumulative_ends;
  rmm::device_uvector<size_type> values;
  std::uint32_t capacity;
};

}  // namespace cudf::detail

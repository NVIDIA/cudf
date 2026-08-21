/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "hash_csr.cuh"

#include <cudf/detail/join/hash_join.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/cstdint>

#include <cstdint>
#include <utility>

namespace cudf::detail {

template <typename Hasher>
struct hash_join<Hasher>::impl {
  impl(cuda::std::uint32_t capacity,
       size_type rows,
       cuda::stream_ref stream,
       cuda::mr::any_resource<cuda::mr::device_accessible> mr)
    : _mr{std::move(mr)},
      entries(capacity, stream, _mr),
      cumulative_ends(capacity, stream, _mr),
      values(rows, stream, _mr),
      capacity{capacity}
  {
  }

  hash_table_ref hash_table() const
  {
    return {const_cast<hash_table_entry_type*>(entries.data()), capacity};
  }

  csr_ref csr() const { return {cumulative_ends.data(), values.data()}; }

  cuda::mr::any_resource<cuda::mr::device_accessible> _mr;
  rmm::device_uvector<hash_table_entry_type> entries;
  rmm::device_uvector<size_type> cumulative_ends;
  rmm::device_uvector<size_type> values;
  cuda::std::uint32_t capacity;
};

}  // namespace cudf::detail

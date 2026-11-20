/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "hash_csr.cuh"

#include <cudf/detail/join/hash_join.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/buffer>
#include <cuda/std/bit>
#include <cuda/std/cstdint>

#include <cstddef>
#include <cstdint>
#include <utility>

namespace cudf::detail {

template <typename Hasher>
struct hash_join<Hasher>::impl {
  impl(cuda::std::uint32_t capacity,
       size_type rows,
       cuda::stream_ref stream,
       cuda::mr::any_resource<cuda::mr::device_accessible> mr)
    : _mr(std::move(mr)),
      _slots(stream, _mr, capacity, cuda::no_init),
      _offsets(stream, _mr, static_cast<std::size_t>(rows) + 1, cuda::no_init),
      _values(stream, _mr, 0, cuda::no_init),
      _capacity(capacity),
      _row_mask(
        (cuda::std::uint32_t{1} << cuda::std::bit_width(static_cast<cuda::std::uint32_t>(rows))) -
        1)
  {
  }

  hash_table_ref hash_table() const
  {
    return {const_cast<hash_table_slot_type*>(_slots.data()), _capacity, _row_mask};
  }

  csr_ref csr() const { return {_offsets.data(), _values.data()}; }

  cuda::mr::any_resource<cuda::mr::device_accessible> _mr;
  cuda::device_buffer<hash_table_slot_type> _slots;
  cuda::device_buffer<size_type> _offsets;
  cuda::device_buffer<size_type> _values;
  cuda::std::uint32_t _capacity;
  cuda::std::uint32_t _row_mask;
};

}  // namespace cudf::detail

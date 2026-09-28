/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "hash_csr.cuh"

#include <cudf/detail/join/hash_join.hpp>
#include <cudf/types.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/cmath>
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
      _slots(capacity, stream, _mr),
      _offsets(static_cast<std::size_t>(rows) + 1, stream, _mr),
      _values(0, stream, _mr),
      _capacity(capacity),
      _row_mask(
        (cuda::std::uint32_t{1} << cuda::std::bit_width(static_cast<cuda::std::uint32_t>(rows))) -
        1),
      _modulo(capacity)
  {
  }

  hash_table_ref hash_table() const
  {
    return {const_cast<hash_table_slot_type*>(_slots.data()), _capacity, _row_mask, _modulo};
  }

  csr_ref csr() const { return {_offsets.data(), _values.data()}; }

  cuda::mr::any_resource<cuda::mr::device_accessible> _mr;
  rmm::device_uvector<hash_table_slot_type> _slots;
  rmm::device_uvector<size_type> _offsets;
  rmm::device_uvector<size_type> _values;
  cuda::std::uint32_t _capacity;
  cuda::std::uint32_t _row_mask;
  cuda::fast_mod_div<cuda::std::uint32_t> _modulo;
};

}  // namespace cudf::detail

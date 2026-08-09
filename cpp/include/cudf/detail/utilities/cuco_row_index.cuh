/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// How cudf stores a row index inside a cuco hash table slot.
//
// cuco claims a slot with a single atomic, so tables containing two 64-bit row indices require
// 16-byte key support. The experimental 64-bit size_type build deliberately requires that support
// rather than silently narrowing row indices and retaining a 2^31-row limit.

#pragma once

#include <cudf/types.hpp>

#include <cuco/constraints.cuh>
#include <cuda/std/limits>

#include <cstdint>
namespace cudf::detail {

/// True where a cuco slot can hold 16 bytes, i.e. where 128-bit atomics are available.
inline constexpr bool cuco_wide_slot_available = cuco::open_addressing_max_key_size >= 16;

static_assert(CUDF_SIZE_TYPE_BITS != 64 || cuco_wide_slot_available,
              "A 64-bit cudf::size_type requires 16-byte cuco key support");

/// A row index as stored inside a cuco slot.
using cuco_row_type = size_type;

/// Largest row index a cuco-backed table can store.
inline constexpr size_type cuco_max_rows =
  static_cast<size_type>(cuda::std::numeric_limits<cuco_row_type>::max());

/// Converts a row index for storage in a cuco slot.
template <typename StoredIndex = cuco_row_type>
CUDF_HOST_DEVICE constexpr StoredIndex to_cuco_index(size_type row) noexcept
{
  return static_cast<StoredIndex>(static_cast<cuco_row_type>(row));
}

/// Widens a row index read back out of a cuco slot.
template <typename StoredIndex>
CUDF_HOST_DEVICE constexpr size_type from_cuco_index(StoredIndex stored) noexcept
{
  return static_cast<size_type>(static_cast<cuco_row_type>(stored));
}

}  // namespace cudf::detail

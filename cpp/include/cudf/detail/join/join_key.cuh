/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// The key type stored in cudf's cuco-backed join hash tables.
//
// Joins mix build-side and probe-side rows in a single table, so a bare row index would be
// ambiguous; cudf therefore stores each row's index next to its precomputed hash. That pairing is
// what makes the slot size limit described in cuco_row_index.cuh bite here.
//
// A 32-bit size_type pairs with a 32-bit hash in exactly 8 bytes, which is why the shipping build
// has never had to think about this. A 64-bit size_type paired with a 32-bit hash would have four
// bytes of tail padding and therefore lack unique object representations. The experimental 64-bit
// build widens the cached hash to 64 bits as well, producing a padding-free 16-byte key. This
// requires the 16-byte cuco key support enforced by cuco_row_index.cuh.

#pragma once

#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/detail/utilities/cuco_row_index.cuh>
#include <cudf/hashing.hpp>
#include <cudf/join/join.hpp>
#include <cudf/types.hpp>

#include <cuco/constraints.cuh>
#include <cuco/pair.cuh>
#include <cuda/std/limits>

#include <cstdint>
#include <type_traits>

namespace cudf::detail {

/// Hash cached alongside the row index inside a join key.
using join_hash_type =
  std::conditional_t<CUDF_SIZE_TYPE_BITS == 64, uint64_t, hash_value_type>;

/// Sentinel stored in an empty slot's row index, mirroring `cudf::JoinNoMatch`.
inline constexpr cuco_row_type join_no_match_key = cuda::std::numeric_limits<cuco_row_type>::min();

/// `join_no_match_key` as a particular stored index type.
template <typename StoredIndex>
inline constexpr StoredIndex join_no_match_index = static_cast<StoredIndex>(join_no_match_key);

/// Left- and right-side row indices as stored inside a join key.
///
/// These stay distinct types so the comparators that dispatch on which table a key came from keep
/// working.
using join_lhs_index_type = row::lhs_index_type;
using join_rhs_index_type = row::rhs_index_type;

/// A join hash table key: a row index next to that row's precomputed hash.
///
/// @tparam StoredIndex How the row index is stored, which is `cuco_row_type` unless the table needs
/// to distinguish the two input tables at compile time
template <typename StoredIndex = cuco_row_type>
using join_key = cuco::pair<join_hash_type, StoredIndex>;

/// Caches a row's hash in the width the join key stores it in.
///
/// The hash itself is only ever 32 bits wide; the wide key zero-extends it purely so that the pair
/// has no padding.
CUDF_HOST_DEVICE constexpr join_hash_type to_join_hash(hash_value_type hash) noexcept
{
  return static_cast<join_hash_type>(hash);
}

/// Recovers the 32-bit hash cached in a join key.
CUDF_HOST_DEVICE constexpr hash_value_type from_join_hash(join_hash_type hash) noexcept
{
  return static_cast<hash_value_type>(hash);
}

/// Recovers the row operator's left-side index from the one stored in a join key.
CUDF_HOST_DEVICE constexpr row::lhs_index_type as_lhs_index(join_lhs_index_type stored) noexcept
{
  return static_cast<row::lhs_index_type>(from_cuco_index(stored));
}

/// Recovers the row operator's right-side index from the one stored in a join key.
CUDF_HOST_DEVICE constexpr row::rhs_index_type as_rhs_index(join_rhs_index_type stored) noexcept
{
  return static_cast<row::rhs_index_type>(from_cuco_index(stored));
}

}  // namespace cudf::detail

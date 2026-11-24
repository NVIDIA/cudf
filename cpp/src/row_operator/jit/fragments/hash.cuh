/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "jit/abi_map.cuh"
#include "row_operator/jit/row_functions.cuh"

#include <cudf/column/column_device_view.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/lists/lists_column_device_view.cuh>

namespace cudf::row_operators {

using cudf_abi::dictionary_layout;
using cudf_abi::dictionary_list_layout;
using cudf_abi::hash_combine;
using cudf_abi::list_layout;
using cudf_abi::plain_layout;

/** @brief Hashes a scalar value, preserving the seed for nulls in seed-chaining mode. */
template <bool SeedChaining, typename Hash, template <typename> class Hasher, typename Element>
__device__ Hash hash(
  column_device_view const& column, size_type row, auto seed, bool has_nulls, plain_layout<Element>)
{
  if (has_nulls && column.is_null(row)) {
    return SeedChaining ? static_cast<Hash>(seed) : ~Hash{0};
  }
  using Native = jit::element_t<Element>;
  return Hasher<Native>{seed}(column.element<Native>(row));
}

template <bool SeedChaining, typename Hash, template <typename> class Hasher, typename Element>
__device__ Hash hash(
  column_device_view const& column, size_type row, auto seed, bool has_nulls, list_layout<Element>)
{
  auto const lists = lists_column_device_view(column);
  auto result      = SeedChaining ? static_cast<Hash>(seed) : Hash{0};
  if constexpr (!SeedChaining) {
    if (has_nulls) { result = hash_combine(result, column.is_null(row) ? ~Hash{0} : Hash{0}); }
    auto const size =
      lists.is_null(row) ? size_type{0} : lists.offset_at(row + 1) - lists.offset_at(row);
    result = hash_combine(result, Hasher<size_type>{DEFAULT_HASH_SEED}(size));
  }
  auto const begin  = lists.offset_at(row);
  auto const end    = lists.offset_at(row + 1);
  auto const values = lists.child();
  for (auto i = begin; i < end; ++i) {
    if constexpr (SeedChaining) {
      result = hash<true, Hash, Hasher>(
        values, i, static_cast<decltype(seed)>(result), has_nulls, plain_layout<Element>{});
    } else {
      result = hash_combine(
        result, hash<false, Hash, Hasher>(values, i, seed, has_nulls, plain_layout<Element>{}));
    }
  }
  return result;
}

template <bool SeedChaining, typename Hash, template <typename> class Hasher, typename Element>
__device__ Hash hash(column_device_view const& column,
                     size_type row,
                     auto seed,
                     bool has_nulls,
                     dictionary_layout<Element>)
{
  if (has_nulls && column.is_null(row)) {
    return SeedChaining ? static_cast<Hash>(seed) : ~Hash{0};
  }
  auto const keys = column.child(dictionary_column_view::keys_column_index);
  return hash<SeedChaining, Hash, Hasher>(
    keys, column.element<dictionary32>(row).value(), seed, has_nulls, plain_layout<Element>{});
}

template <bool SeedChaining, typename Hash, template <typename> class Hasher, typename Element>
__device__ Hash hash(column_device_view const& column,
                     size_type row,
                     auto seed,
                     bool has_nulls,
                     dictionary_list_layout<Element>)
{
  if (has_nulls && column.is_null(row)) {
    return SeedChaining ? static_cast<Hash>(seed) : ~Hash{0};
  }
  auto const keys = column.child(dictionary_column_view::keys_column_index);
  return hash<SeedChaining, Hash, Hasher>(
    keys, column.element<dictionary32>(row).value(), seed, has_nulls, list_layout<Element>{});
}

}  // namespace cudf::row_operators

#define CUDF_DEFINE_ROW_HASH(Name, SeedChaining, Hash, Seed, Hasher, Layout)                    \
  __device__ Hash Name(void const* column, size_type row, Seed seed, bool has_nulls, Layout)    \
  {                                                                                             \
    return ::cudf::row_operators::hash<SeedChaining, Hash, Hasher>(                             \
      *static_cast<::cudf::column_device_view const*>(column), row, seed, has_nulls, Layout{}); \
  }

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
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/algorithm.min.h>

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
  using type = abi_to_cudf_type<Element>;
  return Hasher<type>{seed}(column.element<type>(row));
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
  using type        = abi_to_cudf_type<Element>;
  if constexpr (SeedChaining && sizeof(device_storage_type_t<type>) >= sizeof(int64_t)) {
    // Overlap later value loads with dependent seed updates. Narrow scalars do not benefit from
    // batching; cap the cached element storage to limit register pressure for larger value types.
    constexpr int max_batch_size = 16;
    constexpr int prefetch_bytes = 128;
    constexpr int batch_size = cuda::std::min<int>(max_batch_size, prefetch_bytes / sizeof(type));
    for (auto i = begin; i < end;) {
      auto const count = cuda::std::min(batch_size, end - i);
      type cached[batch_size];
      bool valid[batch_size];
#pragma unroll
      for (int j = 0; j < batch_size; ++j) {
        valid[j] = j < count && (!has_nulls || values.is_valid(i + j));
        if (valid[j]) { cached[j] = values.element<type>(i + j); }
      }
#pragma unroll
      for (int j = 0; j < batch_size; ++j) {
        if (valid[j]) { result = Hasher<type>{static_cast<decltype(seed)>(result)}(cached[j]); }
      }
      i += count;
    }
  } else {
    for (auto i = begin; i < end; ++i) {
      if constexpr (SeedChaining) {
        result = hash<true, Hash, Hasher>(
          values, i, static_cast<decltype(seed)>(result), has_nulls, plain_layout<Element>{});
      } else {
        result = hash_combine(
          result, hash<false, Hash, Hasher>(values, i, seed, has_nulls, plain_layout<Element>{}));
      }
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
  // Dictionary keys cannot contain nulls.
  return hash<SeedChaining, Hash, Hasher>(
    keys, column.element<dictionary32>(row).value(), seed, false, plain_layout<Element>{});
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
  __device__ Hash Name(                                                                         \
    void const* __restrict__ column, size_type row, Seed seed, bool has_nulls, Layout)          \
  {                                                                                             \
    return ::cudf::row_operators::hash<SeedChaining, Hash, Hasher>(                             \
      *static_cast<::cudf::column_device_view const*>(column), row, seed, has_nulls, Layout{}); \
  }

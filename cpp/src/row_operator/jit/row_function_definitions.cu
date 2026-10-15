/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "row_function_declarations.cuh"

#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/bit>

#include <type_traits>

namespace cudf::row_operators::jit {

using cudf::hashing::detail::MurmurHash3_x86_32;

extern "C" __device__ hash_value_type cudf_hash_combine(hash_value_type lhs, hash_value_type rhs)
{
  return cudf::hashing::detail::hash_combine(lhs, rhs);
}

template <typename IndexT>
__device__ size_type dictionary_index(column_device_view const& dict, size_type row)
{
  if constexpr (std::is_void_v<IndexT>) {
    return dict.element<dictionary32>(row).value();
  } else {
    // element() also accounts for the indices child's own offset.
    return static_cast<size_type>(dict.child(dictionary_column_view::indices_column_index)
                                    .template element<IndexT>(row + dict.offset()));
  }
}

template <typename T>
__device__ hash_value_type hash_dictionary_row(column_device_view const* columns,
                                               size_type column,
                                               size_type row,
                                               hash_value_type seed,
                                               bool nulls)
{
  auto const& dict = columns[column];
  if (nulls && dict.is_null(row)) { return cuda::std::numeric_limits<hash_value_type>::max(); }
  auto const keys = dict.child(dictionary_column_view::keys_column_index);
  return cudf::hashing::detail::MurmurHash3_x86_32<T>{seed}(
    keys.element<T>(dict.element<dictionary32>(row).value()));
}

struct murmurhash_double_bits {
  hash_value_type seed;
  __device__ hash_value_type operator()(double value) const
  {
    auto bits            = cuda::std::bit_cast<uint64_t>(value);
    auto const magnitude = bits & 0x7fffffffffffffffULL;
    // Match MurmurHash's canonical NaN payload and signed-zero semantics
    // without using the GPU's double-precision comparison pipeline.
    if (magnitude == 0) { bits = 0; }
    if (magnitude > 0x7ff0000000000000ULL) {
      bits = cuda::std::bit_cast<uint64_t>(cuda::std::numeric_limits<double>::quiet_NaN());
    }
    return MurmurHash3_x86_32<uint64_t>{seed}(bits);
  }
};

template <typename T, typename Hash = MurmurHash3_x86_32<T>>
__device__ hash_value_type hash_list_row(column_device_view const& list_column,
                                         size_type row,
                                         hash_value_type seed,
                                         bool nulls)
{
  auto const lists = lists_column_device_view(list_column);
  auto hash        = hash_value_type{0};
  if (nulls) {
    hash = cudf::hashing::detail::hash_combine(
      hash, lists.is_null(row) ? cuda::std::numeric_limits<hash_value_type>::max() : 0);
  }
  auto const begin     = lists.offset_at(row);
  auto const end       = lists.offset_at(row + 1);
  auto const list_size = lists.is_null(row) ? size_type{0} : end - begin;
  hash                 = cudf::hashing::detail::hash_combine(
    hash, cudf::hashing::detail::MurmurHash3_x86_32<size_type>{}(list_size));
  auto const child = lists.child().slice(begin, list_size);
  for (size_type index = 0; index < child.size(); ++index) {
    auto const value = nulls && child.is_null(index)
                         ? cuda::std::numeric_limits<hash_value_type>::max()
                         : Hash{seed}(child.template element<T>(index));
    hash             = cudf::hashing::detail::hash_combine(hash, value);
  }
  return hash;
}

template <typename T, typename IndexT = void>
__device__ hash_value_type hash_dictionary_list_row(column_device_view const* columns,
                                                    size_type column,
                                                    size_type row,
                                                    hash_value_type seed,
                                                    bool nulls)
{
  auto const& dict = columns[column];
  if (nulls && dict.is_null(row)) { return cuda::std::numeric_limits<hash_value_type>::max(); }
  auto const key_lists = dict.child(dictionary_column_view::keys_column_index);
  return hash_list_row<T>(key_lists, dictionary_index<IndexT>(dict, row), seed, nulls);
}

template <typename T, typename Hash>
__device__ hash_value_type
hash_value(column_device_view const& col, size_type row, hash_value_type seed, bool nulls)
{
  if (nulls && col.is_null(row)) { return cuda::std::numeric_limits<hash_value_type>::max(); }
  if constexpr (std::is_same_v<T, cudf::string_view>) {
    return Hash{seed}(col.element<cudf::string_view>(row));
  } else {
    return Hash{seed}(col.element<T>(row));
  }
}

}  // namespace cudf::row_operators::jit

namespace cudf::row_operators::jit::functions {

__device__ u32
hash_list_bits(void const* columns, i32 column, i32 row, u32 seed, bool nulls, type_tag<f64>)
{
  auto const* c = static_cast<column_device_view const*>(columns);
  return hash_list_row<double, murmurhash_double_bits>(c[column], row, seed, nulls);
}

__device__ u32 hash_dictionary_list_bits(
  void const* columns, i32 column, i32 row, u32 seed, bool nulls, type_tag<f64>)
{
  auto const& dict = static_cast<column_device_view const*>(columns)[column];
  if (nulls && dict.is_null(row)) { return cuda::std::numeric_limits<hash_value_type>::max(); }
  return hash_list_row<double, murmurhash_double_bits>(
    dict.child(dictionary_column_view::keys_column_index),
    dictionary_index<void>(dict, row),
    seed,
    nulls);
}

#define CUDF_DEFINE_ROW_FUNCTIONS(element, physical_type)                              \
  __device__ u32 hash(                                                                 \
    void const* columns, i32 column, i32 row, u32 seed, bool nulls, type_tag<element>) \
  {                                                                                    \
    auto const* c = static_cast<column_device_view const*>(columns);                   \
    return hash_value<physical_type, MurmurHash3_x86_32<physical_type>>(               \
      c[column], row, seed, nulls);                                                    \
  }                                                                                    \
  __device__ u32 hash_dictionary(                                                      \
    void const* columns, i32 column, i32 row, u32 seed, bool nulls, type_tag<element>) \
  {                                                                                    \
    return hash_dictionary_row<physical_type>(                                         \
      static_cast<column_device_view const*>(columns), column, row, seed, nulls);      \
  }                                                                                    \
  __device__ u32 hash_list(                                                            \
    void const* columns, i32 column, i32 row, u32 seed, bool nulls, type_tag<element>) \
  {                                                                                    \
    return hash_list_row<physical_type>(                                               \
      static_cast<column_device_view const*>(columns)[column], row, seed, nulls);      \
  }                                                                                    \
  __device__ u32 hash_dictionary_list(                                                 \
    void const* columns, i32 column, i32 row, u32 seed, bool nulls, type_tag<element>) \
  {                                                                                    \
    return hash_dictionary_list_row<physical_type>(                                    \
      static_cast<column_device_view const*>(columns), column, row, seed, nulls);      \
  }

#define CUDF_DEFINE_INDEX_FUNCTIONS(element, physical_index)                      \
  __device__ u32 hash_dictionary_list(void const* columns,                        \
                                      i32 column,                                 \
                                      i32 row,                                    \
                                      u32 seed,                                   \
                                      bool nulls,                                 \
                                      type_tag<i32>,                              \
                                      index_type_tag<element>)                    \
  {                                                                               \
    return hash_dictionary_list_row<int32_t, physical_index>(                     \
      static_cast<column_device_view const*>(columns), column, row, seed, nulls); \
  }

CUDF_DEFINE_ROW_FUNCTIONS(i8, int8_t)
CUDF_DEFINE_ROW_FUNCTIONS(i16, int16_t)
CUDF_DEFINE_ROW_FUNCTIONS(i32, int32_t)
CUDF_DEFINE_ROW_FUNCTIONS(i64, int64_t)
CUDF_DEFINE_ROW_FUNCTIONS(i128, __int128_t)
CUDF_DEFINE_ROW_FUNCTIONS(u8, uint8_t)
CUDF_DEFINE_ROW_FUNCTIONS(u16, uint16_t)
CUDF_DEFINE_ROW_FUNCTIONS(u32, uint32_t)
CUDF_DEFINE_ROW_FUNCTIONS(u64, uint64_t)
CUDF_DEFINE_ROW_FUNCTIONS(f32, float)
CUDF_DEFINE_ROW_FUNCTIONS(f64, double)
CUDF_DEFINE_ROW_FUNCTIONS(bool8, bool)
CUDF_DEFINE_ROW_FUNCTIONS(cudf::abi_types::string_view, cudf::string_view)
CUDF_DEFINE_ROW_FUNCTIONS(decimal32, numeric::decimal32)
CUDF_DEFINE_ROW_FUNCTIONS(decimal64, numeric::decimal64)
CUDF_DEFINE_ROW_FUNCTIONS(decimal128, numeric::decimal128)
CUDF_DEFINE_ROW_FUNCTIONS(timestamp_days, cudf::timestamp_D)
CUDF_DEFINE_ROW_FUNCTIONS(timestamp_seconds, cudf::timestamp_s)
CUDF_DEFINE_ROW_FUNCTIONS(timestamp_milliseconds, cudf::timestamp_ms)
CUDF_DEFINE_ROW_FUNCTIONS(timestamp_microseconds, cudf::timestamp_us)
CUDF_DEFINE_ROW_FUNCTIONS(timestamp_nanoseconds, cudf::timestamp_ns)
CUDF_DEFINE_ROW_FUNCTIONS(duration_days, cudf::duration_D)
CUDF_DEFINE_ROW_FUNCTIONS(duration_seconds, cudf::duration_s)
CUDF_DEFINE_ROW_FUNCTIONS(duration_milliseconds, cudf::duration_ms)
CUDF_DEFINE_ROW_FUNCTIONS(duration_microseconds, cudf::duration_us)
CUDF_DEFINE_ROW_FUNCTIONS(duration_nanoseconds, cudf::duration_ns)

CUDF_DEFINE_INDEX_FUNCTIONS(i8, int8_t)
CUDF_DEFINE_INDEX_FUNCTIONS(i16, int16_t)
CUDF_DEFINE_INDEX_FUNCTIONS(i32, int32_t)

#undef CUDF_DEFINE_ROW_FUNCTIONS
#undef CUDF_DEFINE_INDEX_FUNCTIONS

}  // namespace cudf::row_operators::jit::functions

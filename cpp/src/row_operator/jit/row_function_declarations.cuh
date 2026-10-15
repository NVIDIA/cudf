/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/abi_types.cuh>

extern "C" __device__ cudf::abi_types::u32 cudf_hash_combine(cudf::abi_types::u32 lhs,
                                                             cudf::abi_types::u32 rhs);

namespace cudf::row_operators::jit::functions {
using namespace cudf::abi_types;

#define CUDF_DECLARE_ROW_HASH(name, ...) \
  __device__ u32 name(void const* columns, i32 column, i32 row, u32 seed, bool nulls, __VA_ARGS__)

#define CUDF_DECLARE_ROW_OVERLOADS(declare, name)  \
  declare(name, type_tag<i8>);                     \
  declare(name, type_tag<i16>);                    \
  declare(name, type_tag<i32>);                    \
  declare(name, type_tag<i64>);                    \
  declare(name, type_tag<i128>);                   \
  declare(name, type_tag<u8>);                     \
  declare(name, type_tag<u16>);                    \
  declare(name, type_tag<u32>);                    \
  declare(name, type_tag<u64>);                    \
  declare(name, type_tag<f32>);                    \
  declare(name, type_tag<f64>);                    \
  declare(name, type_tag<bool8>);                  \
  declare(name, type_tag<string_view>);            \
  declare(name, type_tag<decimal32>);              \
  declare(name, type_tag<decimal64>);              \
  declare(name, type_tag<decimal128>);             \
  declare(name, type_tag<timestamp_days>);         \
  declare(name, type_tag<timestamp_seconds>);      \
  declare(name, type_tag<timestamp_milliseconds>); \
  declare(name, type_tag<timestamp_microseconds>); \
  declare(name, type_tag<timestamp_nanoseconds>);  \
  declare(name, type_tag<duration_days>);          \
  declare(name, type_tag<duration_seconds>);       \
  declare(name, type_tag<duration_milliseconds>);  \
  declare(name, type_tag<duration_microseconds>);  \
  declare(name, type_tag<duration_nanoseconds>)

// Hash functions for non-nested types.
CUDF_DECLARE_ROW_OVERLOADS(CUDF_DECLARE_ROW_HASH, hash);

// Hash functions for dictionary<non-nested type>.
CUDF_DECLARE_ROW_OVERLOADS(CUDF_DECLARE_ROW_HASH, hash_dictionary);

// Hash functions for list<non-nested type>.
CUDF_DECLARE_ROW_OVERLOADS(CUDF_DECLARE_ROW_HASH, hash_list);
CUDF_DECLARE_ROW_HASH(hash_list_bits, type_tag<f64>);

// Hash functions for dictionary<list<non-nested type>>.
CUDF_DECLARE_ROW_OVERLOADS(CUDF_DECLARE_ROW_HASH, hash_dictionary_list);
CUDF_DECLARE_ROW_HASH(hash_dictionary_list_bits, type_tag<f64>);
CUDF_DECLARE_ROW_HASH(hash_dictionary_list, type_tag<i32>, index_type_tag<i8>);
CUDF_DECLARE_ROW_HASH(hash_dictionary_list, type_tag<i32>, index_type_tag<i16>);
CUDF_DECLARE_ROW_HASH(hash_dictionary_list, type_tag<i32>, index_type_tag<i32>);

#undef CUDF_DECLARE_ROW_OVERLOADS
#undef CUDF_DECLARE_ROW_HASH

}  // namespace cudf::row_operators::jit::functions

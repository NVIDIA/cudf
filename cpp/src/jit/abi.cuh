/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// This header is included by NVRTC-compiled adapters.
// It is intended to be a minimal fast to compile header for forwarding UDF arguments across ABIs.

namespace cudf_abi {

using int8_t       = signed char;
using int16_t      = short;
using int32_t      = int;
using int64_t      = long long;
using uint8_t      = unsigned char;
using uint16_t     = unsigned short;
using uint32_t     = unsigned int;
using uint64_t     = unsigned long long;
using uint128_t    = unsigned __int128;
using float32_t    = float;
using float64_t    = double;
using bool8_t      = bool;
using size_type    = int32_t;
using voidptr_t    = void*;
using error_int    = int32_t;
using bitmask_type = uint32_t;

enum class errc : error_int;

struct string_view {
  char const* data;
  size_type bytes;
  size_type length;
};

struct mutable_string_view {
  char* data;
  size_type bytes;
  size_type length;
};

// Each unit has its own type identity while retaining the same representation layout.
struct duration_D {
  int32_t count;
};

struct duration_h {
  int32_t count;
};

struct duration_m {
  int32_t count;
};

struct duration_s {
  int64_t count;
};

struct duration_ms {
  int64_t count;
};

struct duration_us {
  int64_t count;
};

struct duration_ns {
  int64_t count;
};

template <typename Duration>
struct timestamp {
  Duration duration;
};

using timestamp_D  = timestamp<duration_D>;
using timestamp_h  = timestamp<duration_h>;
using timestamp_m  = timestamp<duration_m>;
using timestamp_s  = timestamp<duration_s>;
using timestamp_ms = timestamp<duration_ms>;
using timestamp_us = timestamp<duration_us>;
using timestamp_ns = timestamp<duration_ns>;

template <typename Rep>
struct decimal {
  Rep value;
  int32_t scale;
};

using decimal32  = decimal<int32_t>;
using decimal64  = decimal<int64_t>;
using decimal128 = decimal<__int128>;

template <typename T>
struct optional {
  union {
    char empty;
    T value;
  };
  bool engaged;
};

template <typename T>
struct span {
  T* data;
  ::size_t size;
};

enum class type_id : int32_t;

struct data_type {
  type_id _id;
  int32_t _fixed_point_scale;
};

struct alignas(16) column_device_view {
  data_type _type;
  size_type _size;
  void const* _data;
  size_type _null_count;
  bitmask_type const* _null_mask;
  size_type _offset;
  void* _children;
  size_type _num_children;
};

struct alignas(16) mutable_column_device_view {
  data_type _type;
  size_type _size;
  void const* _data;
  size_type _null_count;
  bitmask_type const* _null_mask;
  size_type _offset;
  void* _children;
  size_type _num_children;
};

}  // namespace cudf_abi

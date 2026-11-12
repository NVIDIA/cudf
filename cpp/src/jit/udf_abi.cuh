/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// This header is included by NVRTC-compiled adapters.
// It is intended to be a minimal fast to compile header for forwarding UDF arguments across ABIs.

namespace cudf_udf_abi {

using int8_t   = signed char;
using int16_t  = short;
using int32_t  = int;
using int64_t  = long long;
using uint8_t  = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;

namespace cudf {

using size_type = int32_t;
using voidptr_t = void*;
using error_int = int32_t;

enum class errc : error_int;

struct string_view {
  char const* data;
  size_type bytes;
  size_type length;
};

template <typename Rep>
struct duration {
  Rep count;
};

template <typename Duration>
struct timestamp {
  Duration duration;
};

using duration_D  = duration<int32_t>;
using duration_h  = duration<int32_t>;
using duration_m  = duration<int32_t>;
using duration_s  = duration<int64_t>;
using duration_ms = duration<int64_t>;
using duration_us = duration<int64_t>;
using duration_ns = duration<int64_t>;

using timestamp_D  = timestamp<duration_D>;
using timestamp_h  = timestamp<duration_h>;
using timestamp_m  = timestamp<duration_m>;
using timestamp_s  = timestamp<duration_s>;
using timestamp_ms = timestamp<duration_ms>;
using timestamp_us = timestamp<duration_us>;
using timestamp_ns = timestamp<duration_ns>;
}  // namespace cudf

namespace numeric {

template <typename Rep>
struct decimal {
  Rep value;
  int32_t scale;
};

using decimal32  = decimal<int32_t>;
using decimal64  = decimal<int64_t>;
using decimal128 = decimal<__int128>;

}  // namespace numeric

namespace cuda::std {

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

}  // namespace cuda::std

}  // namespace cudf_udf_abi

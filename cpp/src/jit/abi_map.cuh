/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "abi.cuh"

#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

namespace cudf {

template <typename T>
struct abi_to_cudf_type_impl {
  using type = T;
};

template <typename T>
struct cudf_to_abi_type_impl {
  using type = T;
};

#define CUDF_MAP_ELEMENT(Abi, Cudf)    \
  template <>                          \
  struct abi_to_cudf_type_impl<Abi> {  \
    using type = Cudf;                 \
  };                                   \
                                       \
  template <>                          \
  struct cudf_to_abi_type_impl<Cudf> { \
    using type = Abi;                  \
  };

CUDF_MAP_ELEMENT(cudf_abi::int8_t, int8_t);
CUDF_MAP_ELEMENT(cudf_abi::int16_t, int16_t);
CUDF_MAP_ELEMENT(cudf_abi::int32_t, int32_t);
CUDF_MAP_ELEMENT(cudf_abi::int64_t, int64_t);
CUDF_MAP_ELEMENT(cudf_abi::uint8_t, uint8_t);
CUDF_MAP_ELEMENT(cudf_abi::uint16_t, uint16_t);
CUDF_MAP_ELEMENT(cudf_abi::uint32_t, uint32_t);
CUDF_MAP_ELEMENT(cudf_abi::uint64_t, uint64_t);
CUDF_MAP_ELEMENT(cudf_abi::float32_t, float);
CUDF_MAP_ELEMENT(cudf_abi::float64_t, double);
CUDF_MAP_ELEMENT(cudf_abi::string_view, string_view);
CUDF_MAP_ELEMENT(cudf_abi::decimal32, numeric::decimal32);
CUDF_MAP_ELEMENT(cudf_abi::decimal64, numeric::decimal64);
CUDF_MAP_ELEMENT(cudf_abi::decimal128, numeric::decimal128);
CUDF_MAP_ELEMENT(cudf_abi::timestamp_D, timestamp_D);
CUDF_MAP_ELEMENT(cudf_abi::timestamp_s, timestamp_s);
CUDF_MAP_ELEMENT(cudf_abi::timestamp_ms, timestamp_ms);
CUDF_MAP_ELEMENT(cudf_abi::timestamp_us, timestamp_us);
CUDF_MAP_ELEMENT(cudf_abi::timestamp_ns, timestamp_ns);
CUDF_MAP_ELEMENT(cudf_abi::duration_D, duration_D);
CUDF_MAP_ELEMENT(cudf_abi::duration_s, duration_s);
CUDF_MAP_ELEMENT(cudf_abi::duration_ms, duration_ms);
CUDF_MAP_ELEMENT(cudf_abi::duration_us, duration_us);
CUDF_MAP_ELEMENT(cudf_abi::duration_ns, duration_ns);

#undef CUDF_MAP_ELEMENT

template <typename T>
using abi_to_cudf_type = typename abi_to_cudf_type_impl<T>::type;

template <typename T>
using cudf_to_abi_type = typename cudf_to_abi_type_impl<T>::type;

}  // namespace cudf

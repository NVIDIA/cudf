/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "abi.cuh"

#include <cudf/errc.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/mutable_string_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/optional>
#include <cuda/std/span>

namespace cudf {

class column_device_view;
class mutable_column_device_view;

template <typename T>
struct abi_to_cudf_type_impl {
  using type = T;
};

template <typename T>
struct cudf_to_abi_type_impl {
  using type = T;
};

#define CUDF_MAP_ABI(Abi, Cudf)        \
  template <>                          \
  struct abi_to_cudf_type_impl<Abi> {  \
    using type = Cudf;                 \
  };                                   \
                                       \
  template <>                          \
  struct cudf_to_abi_type_impl<Cudf> { \
    using type = Abi;                  \
  };

CUDF_MAP_ABI(cudf_abi::int8_t, int8_t);
CUDF_MAP_ABI(cudf_abi::int16_t, int16_t);
CUDF_MAP_ABI(cudf_abi::int32_t, int32_t);
CUDF_MAP_ABI(cudf_abi::int64_t, int64_t);
CUDF_MAP_ABI(cudf_abi::uint8_t, uint8_t);
CUDF_MAP_ABI(cudf_abi::uint16_t, uint16_t);
CUDF_MAP_ABI(cudf_abi::uint32_t, uint32_t);
CUDF_MAP_ABI(cudf_abi::uint64_t, uint64_t);
CUDF_MAP_ABI(cudf_abi::uint128_t, __uint128_t);
CUDF_MAP_ABI(cudf_abi::float32_t, float);
CUDF_MAP_ABI(cudf_abi::float64_t, double);
CUDF_MAP_ABI(cudf_abi::bool8_t, bool);
CUDF_MAP_ABI(cudf_abi::voidptr_t, void*);
CUDF_MAP_ABI(cudf_abi::errc, errc);
CUDF_MAP_ABI(cudf_abi::string_view, string_view);
CUDF_MAP_ABI(cudf_abi::mutable_string_view, mutable_string_view);
CUDF_MAP_ABI(cudf_abi::decimal32, numeric::decimal32);
CUDF_MAP_ABI(cudf_abi::decimal64, numeric::decimal64);
CUDF_MAP_ABI(cudf_abi::decimal128, numeric::decimal128);
CUDF_MAP_ABI(cudf_abi::timestamp_D, timestamp_D);
CUDF_MAP_ABI(cudf_abi::timestamp_h, timestamp_h);
CUDF_MAP_ABI(cudf_abi::timestamp_m, timestamp_m);
CUDF_MAP_ABI(cudf_abi::timestamp_s, timestamp_s);
CUDF_MAP_ABI(cudf_abi::timestamp_ms, timestamp_ms);
CUDF_MAP_ABI(cudf_abi::timestamp_us, timestamp_us);
CUDF_MAP_ABI(cudf_abi::timestamp_ns, timestamp_ns);
CUDF_MAP_ABI(cudf_abi::duration_D, duration_D);
CUDF_MAP_ABI(cudf_abi::duration_h, duration_h);
CUDF_MAP_ABI(cudf_abi::duration_m, duration_m);
CUDF_MAP_ABI(cudf_abi::duration_s, duration_s);
CUDF_MAP_ABI(cudf_abi::duration_ms, duration_ms);
CUDF_MAP_ABI(cudf_abi::duration_us, duration_us);
CUDF_MAP_ABI(cudf_abi::duration_ns, duration_ns);
CUDF_MAP_ABI(cudf_abi::type_id, type_id);
CUDF_MAP_ABI(cudf_abi::data_type, data_type);
CUDF_MAP_ABI(cudf_abi::column_device_view, column_device_view);
CUDF_MAP_ABI(cudf_abi::mutable_column_device_view, mutable_column_device_view);

#undef CUDF_MAP_ABI

template <typename T>
using abi_to_cudf_type = typename abi_to_cudf_type_impl<T>::type;

template <typename T>
using cudf_to_abi_type = typename cudf_to_abi_type_impl<T>::type;

template <typename T>
struct abi_to_cudf_type_impl<T const> {
  using type = abi_to_cudf_type<T> const;
};

template <typename T>
struct cudf_to_abi_type_impl<T const> {
  using type = cudf_to_abi_type<T> const;
};

template <typename T>
struct abi_to_cudf_type_impl<T volatile> {
  using type = abi_to_cudf_type<T> volatile;
};

template <typename T>
struct cudf_to_abi_type_impl<T volatile> {
  using type = cudf_to_abi_type<T> volatile;
};

template <typename T>
struct abi_to_cudf_type_impl<T const volatile> {
  using type = abi_to_cudf_type<T> const volatile;
};

template <typename T>
struct cudf_to_abi_type_impl<T const volatile> {
  using type = cudf_to_abi_type<T> const volatile;
};

template <typename T>
struct abi_to_cudf_type_impl<cudf_abi::optional<T>> {
  using type = cuda::std::optional<abi_to_cudf_type<T>>;
};

template <typename T>
struct cudf_to_abi_type_impl<cuda::std::optional<T>> {
  using type = cudf_abi::optional<cudf_to_abi_type<T>>;
};

template <typename T>
struct abi_to_cudf_type_impl<cudf_abi::span<T>> {
  using type = cuda::std::span<abi_to_cudf_type<T>>;
};

template <typename T>
struct cudf_to_abi_type_impl<cuda::std::span<T>> {
  using type = cudf_abi::span<cudf_to_abi_type<T>>;
};

}  // namespace cudf

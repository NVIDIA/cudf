/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/fixed_point/conv.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Casts input values to bool.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ bool cast_to_bool8(T a)
  requires(!nullable<T> && requires { static_cast<bool>(a); })
{ return static_cast<bool>(a); }

/**
 * @brief Casts input values to int8_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ int8_t cast_to_int8(T a)
  requires(!nullable<T> && requires { static_cast<int8_t>(a); })
{ return static_cast<int8_t>(a); }

/**
 * @brief Casts input values to int16_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ int16_t cast_to_int16(T a)
  requires(!nullable<T> && requires { static_cast<int16_t>(a); })
{ return static_cast<int16_t>(a); }

/**
 * @brief Casts input values to int32_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ int32_t cast_to_int32(T a)
  requires(!nullable<T> && requires { static_cast<int32_t>(a); })
{ return static_cast<int32_t>(a); }

/**
 * @brief Casts input values to int64_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ int64_t cast_to_int64(T a)
  requires(!nullable<T> && requires { static_cast<int64_t>(a); })
{ return static_cast<int64_t>(a); }

/**
 * @brief Casts input values to uint8_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ uint8_t cast_to_uint8(T a)
  requires(!nullable<T> && requires { static_cast<uint8_t>(a); })
{ return static_cast<uint8_t>(a); }

/**
 * @brief Casts input values to uint16_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ uint16_t cast_to_uint16(T a)
  requires(!nullable<T> && requires { static_cast<uint16_t>(a); })
{ return static_cast<uint16_t>(a); }

/**
 * @brief Casts input values to uint32_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ uint32_t cast_to_uint32(T a)
  requires(!nullable<T> && requires { static_cast<uint32_t>(a); })
{ return static_cast<uint32_t>(a); }

/**
 * @brief Casts input values to uint64_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ uint64_t cast_to_uint64(T a)
  requires(!nullable<T> && requires { static_cast<uint64_t>(a); })
{ return static_cast<uint64_t>(a); }

/**
 * @brief Casts input values to float.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ float cast_to_float32(T a)
  requires(cuda::std::is_integral_v<T> || cuda::std::is_floating_point_v<T>)
{ return static_cast<float>(a); }

template <typename R>
__device__ float cast_to_float32(numeric::decimal<R> a)
{ return convert_fixed_to_floating<float>(a); }

/**
 * @brief Casts input values to double.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ double cast_to_float64(T a)
  requires(cuda::std::is_integral_v<T> || floating_point<T> || fixed_point<T>)
{ return static_cast<double>(a); }

template <typename R>
__device__ double cast_to_float64(numeric::decimal<R> a)
{ return convert_fixed_to_floating<double>(a); }

namespace detail {

/**
 * @brief Casts one fixed-point decimal representation to another.
 *
 * @tparam To Destination representation type.
 * @tparam From Source representation type.
 * @param a Input value.
 */
template <typename To, typename From>
__device__ numeric::decimal<To> decimal_cast(numeric::decimal<From> a)
{
  auto rep = static_cast<To>(a.value());
  return numeric::decimal<To>{numeric::scaled_integer<To>{rep, a.scale()}};
}

}  // namespace detail

/**
 * @brief Casts decimal input values to decimal32.
 *
 * @tparam R Source decimal representation type.
 * @param a Input value.
 */
template <typename R>
__device__ numeric::decimal32 cast_to_decimal32(numeric::decimal<R> a)
{ return detail::decimal_cast<int32_t>(a); }

/**
 * @brief Casts decimal input values to decimal64.
 *
 * @tparam R Source decimal representation type.
 * @param a Input value.
 */
template <typename R>
__device__ numeric::decimal64 cast_to_decimal64(numeric::decimal<R> a)
{ return detail::decimal_cast<int64_t>(a); }

/**
 * @brief Casts decimal input values to decimal128.
 *
 * @tparam R Source decimal representation type.
 * @param a Input value.
 */
template <typename R>
__device__ numeric::decimal128 cast_to_decimal128(numeric::decimal<R> a)
{ return detail::decimal_cast<__int128_t>(a); }

/**
 * @brief Rescales decimal input values to a target scale.
 *
 * @tparam R Decimal representation type.
 * @param a Input value.
 * @param new_scale Target decimal scale.
 */
template <typename R, signed_integer Scale>
__device__ numeric::decimal<R> rescale(numeric::decimal<R> a, Scale new_scale)
  requires(sizeof(Scale) <= sizeof(int32_t))
{ return a.rescaled(numeric::scale_type{static_cast<int32_t>(new_scale)}); }

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf

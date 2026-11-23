/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/operators/concepts.cuh>
#include <cudf/fixed_point/conv.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace detail {
namespace ops {

/**
 * @brief Casts a value using its conversion operator or the destination constructor.
 *
 * @tparam To Destination type.
 * @tparam From Source type.
 * @param a Input value.
 */
template <typename To, typename From>
__device__ To cast(From a)
  requires(!(concepts::chrono<From> && concepts::chrono<To>) &&
           !(concepts::duration<To> && concepts::arithmetic<From>) &&
           !(concepts::duration<From> && concepts::arithmetic<To>) && !concepts::fixed_point<To> &&
           !(concepts::fixed_point<From> && cuda::std::floating_point<To>) &&
           requires { static_cast<To>(a); })
{
  return static_cast<To>(a);
}

template <cuda::std::floating_point To, concepts::fixed_point From>
__device__ To cast(From a)
{
  return convert_fixed_to_floating<To>(a);
}

/**
 * @brief Casts a decimal's representation, preserving its scale.
 *
 * @tparam To Destination decimal type.
 * @tparam R Source decimal representation type.
 * @param a Input value.
 */
template <concepts::fixed_point To, typename R>
__device__ To cast(numeric::decimal<R> a)
{
  using rep = typename To::rep;
  return To{numeric::scaled_integer<rep>{static_cast<rep>(a.value()), a.scale()}};
}

/**
 * @brief Casts timestamps or durations, rounding down when reducing time resolution.
 *
 * @tparam To Destination timestamp or duration type.
 * @tparam From Source timestamp or duration type.
 * @param a Input value.
 */
template <concepts::chrono To, concepts::chrono From>
__device__ To cast(From a)
{
  if constexpr (concepts::timestamp<From>) {
    if constexpr (concepts::timestamp<To>) {
      return To{cuda::std::chrono::floor<typename To::duration>(a.time_since_epoch())};
    } else {
      return cuda::std::chrono::floor<To>(a.time_since_epoch());
    }
  } else if constexpr (concepts::timestamp<To>) {
    return To{cuda::std::chrono::floor<typename To::duration>(a)};
  } else {
    return cuda::std::chrono::floor<To>(a);
  }
}

/**
 * @brief Casts a numeric value to a duration's tick count.
 *
 * @tparam To Destination duration type.
 * @tparam From Source numeric type.
 * @param a Input tick count.
 */
template <concepts::duration To, concepts::arithmetic From>
__device__ To cast(From a)
{
  return To{static_cast<typename To::rep>(a)};
}

/**
 * @brief Casts a duration's tick count to a numeric value.
 *
 * @tparam To Destination numeric type.
 * @tparam From Source duration type.
 * @param a Input duration.
 */
template <concepts::arithmetic To, concepts::duration From>
__device__ To cast(From a)
{
  return static_cast<To>(a.count());
}

/**
 * @brief Casts a numeric value to a decimal with the supplied scale.
 *
 * @tparam To Destination decimal type.
 * @tparam From Source numeric type.
 * @param a Input value.
 * @param scale Destination scale.
 */
template <concepts::fixed_point To, concepts::arithmetic From>
__device__ To cast(From a, numeric::scale_type scale)
{
  if constexpr (cuda::std::floating_point<From>) {
    return convert_floating_to_fixed<To>(a, scale);
  } else {
    return To{a, scale};
  }
}

/**
 * @brief Casts a numeric value to a scale-zero decimal.
 *
 * @tparam To Destination decimal type.
 * @tparam From Source numeric type.
 * @param a Input value.
 */
template <concepts::fixed_point To, concepts::arithmetic From>
__device__ To cast(From a)
{
  return cast<To>(a, numeric::scale_type{0});
}

/**
 * @brief Casts input values to bool.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ bool cast_to_bool8(T a)
  requires(!nullable<T> && requires { static_cast<bool>(a); })
{
  return cast<bool>(a);
}

/**
 * @brief Casts input values to int8_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ int8_t cast_to_int8(T a)
  requires(!nullable<T> && requires { static_cast<int8_t>(a); })
{
  return cast<int8_t>(a);
}

/**
 * @brief Casts input values to int16_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ int16_t cast_to_int16(T a)
  requires(!nullable<T> && requires { static_cast<int16_t>(a); })
{
  return cast<int16_t>(a);
}

/**
 * @brief Casts input values to int32_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ int32_t cast_to_int32(T a)
  requires(!nullable<T> && requires { static_cast<int32_t>(a); })
{
  return cast<int32_t>(a);
}

/**
 * @brief Casts input values to int64_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ int64_t cast_to_int64(T a)
  requires(!nullable<T> && requires { static_cast<int64_t>(a); })
{
  return cast<int64_t>(a);
}

/**
 * @brief Casts input values to uint8_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ uint8_t cast_to_uint8(T a)
  requires(!nullable<T> && requires { static_cast<uint8_t>(a); })
{
  return cast<uint8_t>(a);
}

/**
 * @brief Casts input values to uint16_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ uint16_t cast_to_uint16(T a)
  requires(!nullable<T> && requires { static_cast<uint16_t>(a); })
{
  return cast<uint16_t>(a);
}

/**
 * @brief Casts input values to uint32_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ uint32_t cast_to_uint32(T a)
  requires(!nullable<T> && requires { static_cast<uint32_t>(a); })
{
  return cast<uint32_t>(a);
}

/**
 * @brief Casts input values to uint64_t.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ uint64_t cast_to_uint64(T a)
  requires(!nullable<T> && requires { static_cast<uint64_t>(a); })
{
  return cast<uint64_t>(a);
}

/**
 * @brief Casts input values to float.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ float cast_to_float32(T a)
  requires(cuda::std::integral<T> || cuda::std::floating_point<T>)
{
  return cast<float>(a);
}

template <typename R>
__device__ float cast_to_float32(numeric::decimal<R> a)
{
  return cast<float>(a);
}

/**
 * @brief Casts input values to double.
 *
 * @tparam T Source type.
 * @param a Input value.
 */
template <typename T>
__device__ double cast_to_float64(T a)
  requires(cuda::std::integral<T> || cuda::std::floating_point<T>)
{
  return cast<double>(a);
}

template <typename R>
__device__ double cast_to_float64(numeric::decimal<R> a)
{
  return cast<double>(a);
}

/**
 * @brief Casts numeric input values to decimal32, preserving decimal scales or using scale zero.
 *
 * @tparam T Source numeric type.
 * @param a Input value.
 */
template <typename T>
__device__ numeric::decimal32 cast_to_decimal32(T a)
  requires requires { cast<numeric::decimal32>(a); }
{
  return cast<numeric::decimal32>(a);
}

/**
 * @brief Casts numeric input values to decimal64, preserving decimal scales or using scale zero.
 *
 * @tparam T Source numeric type.
 * @param a Input value.
 */
template <typename T>
__device__ numeric::decimal64 cast_to_decimal64(T a)
  requires requires { cast<numeric::decimal64>(a); }
{
  return cast<numeric::decimal64>(a);
}

/**
 * @brief Casts decimal input values to decimal128.
 *
 * @tparam R Source decimal representation type.
 * @param a Input value.
 */
template <typename R>
__device__ numeric::decimal128 cast_to_decimal128(numeric::decimal<R> a)
{
  return cast<numeric::decimal128>(a);
}

/**
 * @brief Rescales decimal input values to a target scale.
 *
 * @tparam R Decimal representation type.
 * @param a Input value.
 * @param new_scale Target decimal scale.
 */
template <typename R, cuda::std::signed_integral Scale>
__device__ numeric::decimal<R> rescale(numeric::decimal<R> a, Scale new_scale)
  requires(sizeof(Scale) <= sizeof(int32_t))
{
  return a.rescaled(numeric::scale_type{static_cast<int32_t>(new_scale)});
}

}  // namespace ops
}  // namespace detail
}  // namespace CUDF_EXPORT cudf

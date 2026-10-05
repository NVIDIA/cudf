/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/utilities/assert.cuh>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/type_traits>

namespace cudf {
namespace detail {
template <typename T>
CUDF_HOST_DEVICE constexpr bool is_product_supported()
{ return is_numeric<T>(); }

/**
 * @brief Maps an `aggregation::Kind` value to its corresponding binary operator
 *
 * @note Not all values of `aggregation::Kind` have a valid corresponding binary
 * operator. For these values `E`,
 * `std::is_same_v<corresponding_operator<E>::type, void>`.
 *
 * @tparam k The `aggregation::Kind` value to map to its corresponding operator
 */
template <aggregation::Kind k>
struct corresponding_operator {
  using type = void;
};

template <>
struct corresponding_operator<aggregation::MIN> {
  using type = DeviceMin;
};
template <>
struct corresponding_operator<aggregation::MAX> {
  using type = DeviceMax;
};
template <>
struct corresponding_operator<aggregation::ARGMIN> {
  using type = DeviceMin;
};
template <>
struct corresponding_operator<aggregation::ARGMAX> {
  using type = DeviceMax;
};
template <>
struct corresponding_operator<aggregation::ANY> {
  using type = DeviceMax;
};
template <>
struct corresponding_operator<aggregation::ALL> {
  using type = DeviceMin;
};
template <>
struct corresponding_operator<aggregation::SUM> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::SUM_OVERFLOW> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::PRODUCT> {
  using type = DeviceProduct;
};
template <>
struct corresponding_operator<aggregation::SUM_OF_SQUARES> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::M2> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::STD> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::VARIANCE> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::MEAN> {
  using type = DeviceSum;
};
template <>
struct corresponding_operator<aggregation::COUNT_VALID> {
  using type = DeviceCount;
};
template <>
struct corresponding_operator<aggregation::COUNT_ALL> {
  using type = DeviceCount;
};

template <aggregation::Kind k>
using corresponding_operator_t = typename corresponding_operator<k>::type;

template <aggregation::Kind k>
constexpr bool has_corresponding_operator()
{ return !cuda::std::is_same_v<typename corresponding_operator<k>::type, void>; }

/**
 * @brief Checks if the given type and aggregation kind combination is supported
 * for identity initialization.
 *
 * @tparam T The data type
 * @tparam k The aggregation kind
 * @return true if identity initialization is supported for this type/aggregation combo
 */
template <typename T, aggregation::Kind k>
CUDF_HOST_DEVICE constexpr bool is_identity_supported()
{
  return cudf::is_fixed_width<T>() and
         ((k == aggregation::SUM) or (k == aggregation::SUM_OF_SQUARES) or
          (k == aggregation::MIN) or (k == aggregation::MAX) or (k == aggregation::COUNT_VALID) or
          (k == aggregation::COUNT_ALL) or (k == aggregation::ARGMIN) or
          (k == aggregation::ARGMAX) or (k == aggregation::STD) or (k == aggregation::VARIANCE) or
          (k == aggregation::M2) or
          (k == aggregation::PRODUCT) and cudf::detail::is_product_supported<T>());
}

/**
 * @brief Gets the identity value from the corresponding binary operator.
 *
 * @tparam T The data type
 * @tparam k The aggregation kind
 * @return The identity value
 */
template <typename T, aggregation::Kind k>
CUDF_HOST_DEVICE T identity_from_operator()
{
  static_assert(not cuda::std::is_same_v<corresponding_operator_t<k>, void>,
                "Unable to get identity/sentinel from device operator");
  using DeviceType = device_storage_type_t<T>;
  return corresponding_operator_t<k>::template identity<DeviceType>();
}

/**
 * @brief Gets the identity value for the given type and aggregation kind.
 *
 * For ARGMAX/ARGMIN, returns the appropriate sentinel value.
 * For other aggregations, delegates to identity_from_operator.
 *
 * @tparam T The data type
 * @tparam k The aggregation kind
 * @return The identity value
 */
template <typename T, aggregation::Kind k>
CUDF_HOST_DEVICE T get_identity()
{
  if constexpr (k == aggregation::ARGMAX or k == aggregation::ARGMIN) {
    if constexpr (cudf::is_timestamp<T>()) {
      return k == aggregation::ARGMAX ? T{typename T::duration(ARGMAX_SENTINEL)}
                                      : T{typename T::duration(ARGMIN_SENTINEL)};
    } else {
      using DeviceType = device_storage_type_t<T>;
      return k == aggregation::ARGMAX ? static_cast<DeviceType>(ARGMAX_SENTINEL)
                                      : static_cast<DeviceType>(ARGMIN_SENTINEL);
    }
  } else {
    return identity_from_operator<T, k>();
  }
}

}  // namespace detail
}  // namespace cudf

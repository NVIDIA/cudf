/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cudf_abi.cuh"

#include <cudf/errc.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/strings/mutable_string_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/optional>

#include <stdexcept>

namespace cudf {

template <typename T>
struct abi_to_cudf_type_impl;

template <typename T>
struct cudf_to_abi_type_impl;

#define CUDF_MAP_ABI(Abi, Cudf)                                     \
  template <>                                                       \
  struct abi_to_cudf_type_impl<Abi> {                               \
    using type                             = Cudf;                  \
    static constexpr char const* abi_name  = #Abi;                  \
    static constexpr char const* cudf_name = #Cudf;                 \
  };                                                                \
                                                                    \
  template <>                                                       \
  struct cudf_to_abi_type_impl<Cudf> : abi_to_cudf_type_impl<Abi> { \
    using type = Abi;                                               \
  };

CUDF_MAP_ABI(cudf_abi::int8_t, int8_t);
CUDF_MAP_ABI(cudf_abi::int16_t, int16_t);
CUDF_MAP_ABI(cudf_abi::int32_t, int32_t);
CUDF_MAP_ABI(cudf_abi::int64_t, int64_t);
CUDF_MAP_ABI(cudf_abi::uint8_t, uint8_t);
CUDF_MAP_ABI(cudf_abi::uint16_t, uint16_t);
CUDF_MAP_ABI(cudf_abi::uint32_t, uint32_t);
CUDF_MAP_ABI(cudf_abi::uint64_t, uint64_t);
CUDF_MAP_ABI(cudf_abi::float32_t, float);
CUDF_MAP_ABI(cudf_abi::float64_t, double);
CUDF_MAP_ABI(cudf_abi::bool8_t, bool);
CUDF_MAP_ABI(cudf_abi::voidptr_t, cudf::voidptr_t);
CUDF_MAP_ABI(cudf_abi::errc, cudf::errc);
CUDF_MAP_ABI(cudf_abi::string_view, cudf::string_view);
CUDF_MAP_ABI(cudf_abi::mutable_string_view, cudf::mutable_string_view);
CUDF_MAP_ABI(cudf_abi::decimal32, numeric::decimal32);
CUDF_MAP_ABI(cudf_abi::decimal64, numeric::decimal64);
CUDF_MAP_ABI(cudf_abi::decimal128, numeric::decimal128);
CUDF_MAP_ABI(cudf_abi::timestamp_D, cudf::timestamp_D);
CUDF_MAP_ABI(cudf_abi::timestamp_h, cudf::timestamp_h);
CUDF_MAP_ABI(cudf_abi::timestamp_m, cudf::timestamp_m);
CUDF_MAP_ABI(cudf_abi::timestamp_s, cudf::timestamp_s);
CUDF_MAP_ABI(cudf_abi::timestamp_ms, cudf::timestamp_ms);
CUDF_MAP_ABI(cudf_abi::timestamp_us, cudf::timestamp_us);
CUDF_MAP_ABI(cudf_abi::timestamp_ns, cudf::timestamp_ns);
CUDF_MAP_ABI(cudf_abi::duration_D, cudf::duration_D);
CUDF_MAP_ABI(cudf_abi::duration_h, cudf::duration_h);
CUDF_MAP_ABI(cudf_abi::duration_m, cudf::duration_m);
CUDF_MAP_ABI(cudf_abi::duration_s, cudf::duration_s);
CUDF_MAP_ABI(cudf_abi::duration_ms, cudf::duration_ms);
CUDF_MAP_ABI(cudf_abi::duration_us, cudf::duration_us);
CUDF_MAP_ABI(cudf_abi::duration_ns, cudf::duration_ns);

#undef CUDF_MAP_ABI

template <typename T>
using abi_to_cudf_type = typename abi_to_cudf_type_impl<T>::type;

template <typename T>
using cudf_to_abi_type = typename cudf_to_abi_type_impl<T>::type;

template <typename T>
struct abi_to_cudf_type_impl<cudf_abi::optional<T>> {
  using type = cuda::std::optional<abi_to_cudf_type<T>>;
};

template <typename T>
struct cudf_to_abi_type_impl<cuda::std::optional<T>> {
  using type = cudf_abi::optional<cudf_to_abi_type<T>>;
};

/** @brief Canonical names of an unqualified cuDF value type and its ABI equivalent. */
struct abi_type_names {
  char const* abi_name  = "";  ///< Type spelling provided by cudf_abi.cuh
  char const* cudf_name = "";  ///< Type spelling provided by cuDF headers
};

namespace detail {

struct abi_type_names_fn {
  template <typename T>
  abi_type_names operator()() const
  {
    if constexpr (requires { cudf_to_abi_type_impl<T>::abi_name; }) {
      return {cudf_to_abi_type_impl<T>::abi_name, cudf_to_abi_type_impl<T>::cudf_name};
    } else {
      CUDF_FAIL("Unsupported UDF ABI value type", std::invalid_argument);
    }
  }
};

}  // namespace detail

/**
 * @brief Returns canonical cuDF and ABI value-type names for a type identifier.
 *
 * @param id Value type identifier
 * @return Names of the cuDF value type and its ABI equivalent
 * @throws std::invalid_argument If the type has no supported UDF ABI value mapping
 */
inline abi_type_names get_abi_type_names(type_id id)
{
  return type_dispatcher(data_type{id}, detail::abi_type_names_fn{});
}

}  // namespace cudf

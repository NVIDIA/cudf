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

namespace cudf::jit {

// ABI tags select overloads; column loads and operators use the actual cuDF element types.
template <typename T>
struct element_type {
  using type = T;
};

#define CUDF_MAP_ELEMENT(Abi, Native) \
  template <>                         \
  struct element_type<Abi> {          \
    using type = Native;              \
  }

CUDF_MAP_ELEMENT(cudf_abi::int64_t, int64_t);
CUDF_MAP_ELEMENT(cudf_abi::uint64_t, uint64_t);
CUDF_MAP_ELEMENT(cudf_abi::string_view, cudf::string_view);
CUDF_MAP_ELEMENT(cudf_abi::decimal32, numeric::decimal32);
CUDF_MAP_ELEMENT(cudf_abi::decimal64, numeric::decimal64);
CUDF_MAP_ELEMENT(cudf_abi::decimal128, numeric::decimal128);
CUDF_MAP_ELEMENT(cudf_abi::timestamp_D, cudf::timestamp_D);
CUDF_MAP_ELEMENT(cudf_abi::timestamp_s, cudf::timestamp_s);
CUDF_MAP_ELEMENT(cudf_abi::timestamp_ms, cudf::timestamp_ms);
CUDF_MAP_ELEMENT(cudf_abi::timestamp_us, cudf::timestamp_us);
CUDF_MAP_ELEMENT(cudf_abi::timestamp_ns, cudf::timestamp_ns);
CUDF_MAP_ELEMENT(cudf_abi::duration_D, cudf::duration_D);
CUDF_MAP_ELEMENT(cudf_abi::duration_s, cudf::duration_s);
CUDF_MAP_ELEMENT(cudf_abi::duration_ms, cudf::duration_ms);
CUDF_MAP_ELEMENT(cudf_abi::duration_us, cudf::duration_us);
CUDF_MAP_ELEMENT(cudf_abi::duration_ns, cudf::duration_ns);

#undef CUDF_MAP_ELEMENT

template <typename T>
using element_t = typename element_type<T>::type;

}  // namespace cudf::jit

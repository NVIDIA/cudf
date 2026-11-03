/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/io/parquet_schema.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>
#include <cudf/wrappers/timestamps.hpp>

namespace cudf {
namespace io::detail {

struct get_period {
  template <typename T>
  int32_t operator()()
  {
    if constexpr (is_chrono<T>()) { return T::period::den; }
    CUDF_FAIL("Invalid, non chrono type");
  }
};

/**
 * @brief Function that translates cuDF time unit to clock frequency
 */
inline int32_t to_clockrate(type_id timestamp_type_id)
{
  return timestamp_type_id == type_id::EMPTY
           ? 0
           : type_dispatcher(data_type{timestamp_type_id}, get_period{});
}

}  // namespace io::detail
}  // namespace cudf

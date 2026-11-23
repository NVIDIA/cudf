/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <cuda/std/cstdint>

namespace CUDF_EXPORT cudf {
namespace datetime {

/**
 * @brief Types of datetime components that may be extracted.
 * @ingroup datetime_extract
 */
enum class datetime_component : int8_t {
  YEAR        = 0,
  MONTH       = 1,
  DAY         = 2,
  WEEKDAY     = 3,
  HOUR        = 4,
  MINUTE      = 5,
  SECOND      = 6,
  MILLISECOND = 7,
  MICROSECOND = 8,
  NANOSECOND  = 9
};

/**
 * @brief Fixed frequencies supported by datetime rounding functions ceil, floor, round.
 * @ingroup datetime_compute
 */
enum class rounding_frequency : int8_t {
  DAY         = 0,
  HOUR        = 1,
  MINUTE      = 2,
  SECOND      = 3,
  MILLISECOND = 4,
  MICROSECOND = 5,
  NANOSECOND  = 6
};

}  // namespace datetime
}  // namespace CUDF_EXPORT cudf

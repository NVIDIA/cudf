/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <cstdint>

namespace cudf::io::orc::detail {

// Epoch days of 1582-10-15. Earlier hybrid dates use the Julian calendar.
constexpr int32_t calendar_cutover = -141'427;

CUDF_HOST_DEVICE constexpr int64_t floor_div(int64_t value, int64_t divisor)
{
  return value / divisor - (value % divisor < 0);
}

/**
 * @brief Convert epoch days while preserving the civil year, month, and day.
 *
 * March-based 400-year Gregorian and 4-year Julian cycles keep intermediate values positive
 * within each cycle, including for astronomical years zero and below. All arithmetic is 64 bit
 * so that the complete int32 DATE domain can be checked before narrowing the result.
 */
template <bool to_hybrid>
CUDF_HOST_DEVICE constexpr int64_t rebase_days(int64_t days)
{
  if (days >= calendar_cutover) { return days; }

  constexpr int64_t gregorian_epoch_offset = 719'468;
  constexpr int64_t julian_epoch_offset    = 719'470;
  constexpr int64_t gregorian_era_days     = 146'097;
  constexpr int64_t julian_era_days        = 1'461;
  constexpr int64_t gregorian_century_days = gregorian_era_days / 4;
  constexpr int64_t five_month_days        = 153;
  constexpr auto source_era_days           = to_hybrid ? gregorian_era_days : julian_era_days;
  auto const z   = days + (to_hybrid ? gregorian_epoch_offset : julian_epoch_offset);
  auto const era = floor_div(z, source_era_days);
  auto const doe = z - era * source_era_days;
  // Discount leap days to recover the year within the source calendar's era.
  auto const yoe  = to_hybrid ? (doe - doe / (julian_era_days - 1) + doe / gregorian_century_days -
                                doe / (gregorian_era_days - 1)) /
                                 365
                              : (doe == julian_era_days - 1 ? 3 : doe / 365);
  auto const year = yoe + era * (to_hybrid ? 400 : 4);
  auto const doy  = doe - (365 * yoe + (to_hybrid ? yoe / 4 - yoe / 100 : 0));
  // March starts the year so the leap day is last. Five-month blocks contain 153 days;
  // mp maps March through February to 0 through 11, with doy measured from March 1.
  auto const mp  = (5 * doy + 2) / five_month_days;
  auto const day = doy - (five_month_days * mp + 2) / 5 + 1;

  if constexpr (to_hybrid) {
    return 365 * year + floor_div(year, 4) + (five_month_days * mp + 2) / 5 + day - 1 -
           julian_epoch_offset;
  } else {
    return 365 * year + floor_div(year, 4) - floor_div(year, 100) + floor_div(year, 400) +
           (five_month_days * mp + 2) / 5 + day - 1 - gregorian_epoch_offset;
  }
}

template <bool to_hybrid>
CUDF_HOST_DEVICE constexpr bool is_representable_date(int64_t original, int64_t rebased)
{
  // The hybrid cutover gap and Julian-only leap days have no lossless counterpart.
  return rebased >= INT32_MIN and rebased <= INT32_MAX and
         rebase_days<!to_hybrid>(rebased) == original;
}

}  // namespace cudf::io::orc::detail

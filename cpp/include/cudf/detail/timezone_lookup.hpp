/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/timezone.hpp>
#include <cudf/types.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <cuda/std/algorithm>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Returns the UT offset for a timestamp, from a timezone transition table.
 *
 * The table holds the entries from the TZif file, followed by `solar_cycle_entry_count` entries
 * that repeat every solar cycle; a timestamp past the file entries is projected into that cycle.
 *
 * @param times Transition times, ascending within each part of the table
 * @param offsets Offset that takes effect at each transition time
 * @param num_entries Number of entries in the table
 * @param ts Point in time to get the offset for
 *
 * @return Offset from UT, in seconds
 */
CUDF_HOST_DEVICE inline duration_s get_ut_offset(timestamp_s const* times,
                                                 duration_s const* offsets,
                                                 size_type num_entries,
                                                 timestamp_s ts)
{
  if (num_entries == 0) { return duration_s{0}; }

  auto const last_less_equal = [](auto begin, auto end, auto value) {
    auto const first_larger = cuda::std::upper_bound(begin, end, value);
    // Return start of the range if all elements are larger than the value
    if (first_larger == begin) { return begin; }
    // Element before the first larger element is the last one less or equal
    return first_larger - 1;
  };

  auto const file_entry_end = times + (num_entries - solar_cycle_entry_count);

  auto const entry =
    (ts <= *(file_entry_end - 1))
      // Search the file entries if the timestamp is in range
      ? last_less_equal(times, file_entry_end, ts)
      // Search the solar cycle if outside of the file entries range
      : last_less_equal(
          file_entry_end,
          times + num_entries,
          timestamp_s{(ts.time_since_epoch() + solar_cycle_duration()) % solar_cycle_duration()});

  return offsets[entry - times];
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf

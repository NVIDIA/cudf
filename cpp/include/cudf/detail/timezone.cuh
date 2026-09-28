/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/timezone_lookup.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/timezone.hpp>
#include <cudf/types.hpp>

namespace cudf::detail {

/**
 * @brief Returns the UT offset for a given date and given timezone table.
 *
 * @param tz_table Timezone conversion table
 * @param ts ORC timestamp
 *
 * @return offset from UT, in seconds
 */
inline __device__ duration_s get_ut_offset(table_device_view tz_table, timestamp_s ts)
{
  return get_ut_offset(tz_table.column(0).data<timestamp_s>(),
                       tz_table.column(1).data<duration_s>(),
                       tz_table.num_rows(),
                       ts);
}

}  // namespace cudf::detail

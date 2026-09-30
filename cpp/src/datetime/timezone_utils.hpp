/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <optional>
#include <string_view>

namespace cudf {
namespace detail {

/**
 * @brief Returns the UT offset of a timezone at a given point in time.
 *
 * Host-side counterpart of `cudf::detail::get_ut_offset(table_device_view, timestamp_s)`, for when
 * a single offset is needed and building a device table would be wasteful.
 *
 * @throw cudf::logic_error if `timezone_name` does not resolve to a TZif file in `tzif_dir`
 *
 * @param tzif_dir The directory where the TZif files are located
 * @param timezone_name standard timezone name (for example, "America/Los_Angeles")
 * @param ts Point in time to get the offset for
 *
 * @return Offset from UT, in seconds
 */
[[nodiscard]] duration_s get_ut_offset(std::optional<std::string_view> tzif_dir,
                                       std::string_view timezone_name,
                                       timestamp_s ts);

}  // namespace detail
}  // namespace cudf

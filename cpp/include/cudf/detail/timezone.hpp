/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/timezone.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/wrappers/durations.hpp>
#include <cudf/wrappers/timestamps.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::make_timezone_transition_table(std::optional<std::string_view>, std::string_view,
 * rmm::cuda_stream_view, rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<table> make_timezone_transition_table(
  std::optional<std::string_view> tzif_dir,
  std::string_view timezone_name,
  rmm::cuda_stream_view stream      = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Returns the UT offset of a timezone at a given point in time.
 *
 * The offset is the number of seconds to add to UT to get the local time in `timezone_name`, as in
 * the transition table produced by `make_timezone_transition_table`. Host-side counterpart of
 * `cudf::detail::get_ut_offset(table_device_view, timestamp_s)`, for the cases where a single
 * offset is needed and building a device table would be wasteful.
 *
 * @throw cudf::logic_error if `timezone_name` does not resolve to a TZif file in `tzif_dir`
 *
 * @param tzif_dir The directory where the TZif files are located
 * @param timezone_name standard timezone name (for example, "America/Los_Angeles")
 * @param ts Point in time to get the offset for
 *
 * @return Offset from UT, in seconds
 */
duration_s get_ut_offset(std::optional<std::string_view> tzif_dir,
                         std::string_view timezone_name,
                         timestamp_s ts);

}  // namespace detail
}  // namespace cudf

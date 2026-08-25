/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>
#include <optional>
#include <string>

namespace CUDF_EXPORT cudf {
class table;

// Cycle in which the time offsets repeat in Gregorian calendar
static constexpr int32_t solar_cycle_years = 400;
// Number of future entries in the timezone transition table:
// Two entries per year, over the length of the Gregorian calendar's solar cycle
static constexpr uint32_t solar_cycle_entry_count = 2 * solar_cycle_years;

/**
 * @brief Creates a transition table to convert ORC timestamps to UTC.
 *
 * Uses system's TZif files. Assumes little-endian platform when parsing these files.
 * The transition table starts with the entries from the TZif file. For timestamps after the file's
 * last transition, the table includes entries that form a `solar_cycle_years`-year cycle (future
 * entries). This portion of the table has `solar_cycle_entry_count` elements, as it assumes two
 * transitions per year from Daylight Saving Time. If the timezone does not have DST, the table will
 * still include the future entries, which will all have the same offset.
 *
 * If `timezone_name` does not resolve to a TZif file in `tzif_dir`, the directory's `tzdata.zi`
 * index file is consulted (if present) and `Link` entries are followed to a canonical zone name
 * whose TZif file exists on disk.
 *
 * @param tzif_dir The directory where the TZif files are located
 * @param timezone_name standard timezone name (for example, "America/Los_Angeles")
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned table's device memory.
 *
 * @return The transition table for the given timezone
 */
std::unique_ptr<table> make_timezone_transition_table(
  std::optional<std::string_view> tzif_dir,
  std::string_view timezone_name,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

}  // namespace CUDF_EXPORT cudf

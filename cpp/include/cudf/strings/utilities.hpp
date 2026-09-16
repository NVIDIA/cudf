/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace CUDF_EXPORT cudf {
namespace strings {

/**
 * @brief Creates a string_view vector from a strings column.
 *
 * @param strings Strings column instance.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned vector's device memory.
 * @return Device vector of string_views
 */
rmm::device_uvector<string_view> create_string_vector_from_column(
  cudf::strings_column_view const strings,
  cuda::stream_ref stream           = cudf::get_default_stream(),
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

/**
 * @brief Return the threshold size for a strings column to use int64 offsets
 *
 * A computed size above this threshold should using int64 offsets, otherwise
 * int32 offsets. By default this function will return std::numeric_limits<int32_t>::max().
 * This value can be overridden at runtime using the environment variable
 * LIBCUDF_LARGE_STRINGS_THRESHOLD.
 *
 * @return size in bytes
 */
int64_t get_offset64_threshold();

/**
 * @brief Checks if large strings is enabled
 *
 * This checks the setting in the environment variable LIBCUDF_LARGE_STRINGS_ENABLED.
 *
 * @return true if large strings are supported
 */
bool is_large_strings_enabled();

}  // namespace strings
}  // namespace CUDF_EXPORT cudf

/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <rmm/cuda_stream_view.hpp>

/**
 * @file
 * @brief APIs for querying the default CUDA stream and per-thread default stream status.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup default_stream
 * @{
 */

/**
 * @brief Get the current default stream
 *
 * @return The current default stream.
 */
rmm::cuda_stream_view const get_default_stream();

/**
 * @brief Check if per-thread default stream is enabled.
 *
 * @return true if PTDS is enabled, false otherwise.
 */
bool is_ptds_enabled();

/** @} */  // end of group
}  // namespace CUDF_EXPORT cudf

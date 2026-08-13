/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

/**
 * @file
 * @brief APIs for configuring KvikIO-based I/O behavior.
 */

namespace CUDF_EXPORT cudf {
namespace io {
//! `KvikIO`
namespace kvikio_integration {

/**
 * @addtogroup io_configuration
 * @{
 */

/**
 * @brief Set KvikIO parameters
 *
 * Parameters include:
 * - Compatibility mode, according to the environment variable KVIKIO_COMPAT_MODE. If
 *   KVIKIO_COMPAT_MODE is not set, enable it by default, which enforces the use of POSIX I/O.
 * - Thread pool size, according to the environment variable KVIKIO_NTHREADS. If KVIKIO_NTHREADS is
 *   not set, use 4 threads by default.
 */
void set_up_kvikio();

}  // namespace kvikio_integration

//! `nvCOMP`
namespace nvcomp_integration {

/**
 * @brief Returns true if all nvCOMP uses are enabled
 *
 * @return true if all nvCOMP uses are enabled
 */
[[nodiscard]] bool is_all_enabled();

/**
 * @brief Returns true if stable nvCOMP use is enabled
 *
 * @return true true if stable nvCOMP use is enabled
 */
[[nodiscard]] bool is_stable_enabled();

}  // namespace nvcomp_integration

//! IO Integrated Memory Optimization
namespace integrated_memory_optimization {

/**
 * @brief Returns true if integrated memory optimizations are enabled
 *
 * Controlled by the LIBCUDF_INTEGRATED_MEMORY_OPTIMIZATION environment variable.
 * Valid values: "AUTO" (default), "ON", "OFF"
 * - AUTO: Use hardware detection (cudaDevAttrIntegrated)
 * - ON: Always enable optimization
 * - OFF: Always disable optimization
 *
 * @return true if integrated memory optimizations are enabled
 */
[[nodiscard]] bool is_enabled();

/** @} */  // end of group
}  // namespace integrated_memory_optimization
}  // namespace io
}  // namespace CUDF_EXPORT cudf

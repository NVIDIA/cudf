/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <optional>

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
 * @brief KvikIO's compatibility mode, which decides whether file reads use GPUDirect Storage.
 *
 * Even for a file that uses GPUDirect Storage, a read smaller than KvikIO's GDS threshold uses
 * POSIX I/O. The threshold is set with the KVIKIO_GDS_THRESHOLD environment variable; its
 * default is documented at `kvikio::defaults::gds_threshold()`.
 */
enum class compat_mode : int32_t {
  ON,   ///< Always use POSIX I/O
  OFF,  ///< Use GPUDirect Storage; opening a file it cannot read fails
  AUTO  ///< Use GPUDirect Storage where it works for a file, else POSIX I/O
};

/**
 * @brief Set KvikIO parameters
 *
 * Parameters include:
 * - Compatibility mode. If @p mode is provided, it is used directly. Otherwise, the value is read
 *   from the environment variable KVIKIO_COMPAT_MODE. If KVIKIO_COMPAT_MODE is not set, it
 *   defaults to `ON`, which enforces the use of POSIX I/O.
 * - Thread pool size. If @p nthreads is provided, it is used directly. Otherwise, the value is
 *   read from the environment variable KVIKIO_NTHREADS, defaulting to 4 if unset.
 *
 * @param nthreads Thread pool size. Defaults to KVIKIO_NTHREADS, or 4 if it is unset.
 * @param mode Compatibility mode. Defaults to KVIKIO_COMPAT_MODE, or `ON` if it is unset.
 *
 * @note Only the first call takes effect, including the implicit call made when cuDF first opens a
 * file source or sink. To apply @p nthreads or @p mode, call this before any cuDF I/O; later calls
 * are ignored.
 */
void set_up_kvikio(std::optional<uint32_t> nthreads = std::nullopt,
                   std::optional<compat_mode> mode  = std::nullopt);
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

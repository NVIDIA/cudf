/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <span>

namespace CUDF_EXPORT cudf {

/**
 * @addtogroup udfs
 * @{
 */

/**
 * @brief The type of LTO Binary
 */
enum class lto_binary_type : uint8_t {
  LTO_IR,  //< LTO-IR binary
  FATBIN   //< FATBIN binary
};

/**
 * @brief A compiled device module to link into a JIT kernel.
 *
 * A module may define multiple functions and reference functions in other modules. Keep its bytes
 * alive until evaluation completes or a reusable transform program has been constructed.
 */
struct lto_fragment {
  lto_binary_type type          = lto_binary_type::LTO_IR;  ///< Binary format of the module
  std::span<uint8_t const> data = {};                       ///< The module's bytes
};

/// @brief Device modules linked into a JIT kernel.
using lto_fragments = std::span<lto_fragment const>;

/** @} */  // end of group

}  // namespace CUDF_EXPORT cudf

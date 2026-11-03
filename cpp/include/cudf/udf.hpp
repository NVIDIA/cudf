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
  std::span<uint8_t const> data = {};                       ///< The module's bytes
  lto_binary_type type          = lto_binary_type::LTO_IR;  ///< Binary format of the module
};

/// @brief Device modules linked into a JIT kernel.
using lto_fragments = std::span<lto_fragment const>;

}  // namespace CUDF_EXPORT cudf

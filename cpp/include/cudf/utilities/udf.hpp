/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

#include <cstdint>

namespace CUDF_EXPORT cudf {

/** @brief Format of a device UDF binary supplied to an LTO operation. */
enum class lto_binary_type : std::uint8_t {
  LTO_IR,  ///< LTO intermediate representation
  FATBIN   ///< Fatbinary containing LTO intermediate representation
};

}  // namespace CUDF_EXPORT cudf

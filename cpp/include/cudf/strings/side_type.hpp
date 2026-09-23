/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/utilities/export.hpp>

/**
 * @file
 * @brief Enum defining the side of a string to apply strip or pad operations.
 */

namespace CUDF_EXPORT cudf {
namespace strings {
/**
 * @addtogroup strings_modify
 * @{
 */

/**
 * @brief Direction identifier for cudf::strings::strip and cudf::strings::pad functions.
 */
enum class side_type {
  LEFT,   ///< strip/pad characters from the beginning of the string
  RIGHT,  ///< strip/pad characters from the end of the string
  BOTH    ///< strip/pad characters from the beginning and end of the string
};

/** @} */  // end of doxygen group
}  // namespace strings
}  // namespace CUDF_EXPORT cudf

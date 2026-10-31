/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "row_hash_schema.hpp"

#include <cudf/types.hpp>

#include <rtcx/rtcx.hpp>

#include <string>
#include <string_view>

namespace cudf::hashing::detail::jit {

/** @brief Options for schema-specialized row hashing. */
struct row_function_options {
  bool float64_bits{};  ///< Use equivalent integer-bit hashing for FLOAT64 list elements.
  bool check_nulls{};   ///< Include parent and element null checks in generated hash calls.
};

/** @brief Compiles and caches an LTO IR fragment for row hashing. */
rtcx::blob get_row_function_dispatch_fragment(std::string const& name,
                                              row_hash_schema const& schema,
                                              row_function_options options);

}  // namespace cudf::hashing::detail::jit

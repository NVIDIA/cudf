/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/export.hpp>

#include <string>
#include <string_view>

namespace CUDF_EXPORT cudf {
namespace jit {

/**
 * @brief Rename the first PTX device function and references to its symbol.
 *
 * The module, function body, and ABI are preserved for PTX linking. Comments, strings,
 * and identifiers that merely contain the original symbol are left unchanged.
 *
 * @param src The input PTX module
 * @param function_name The target symbol name
 * @return The PTX module with the function symbol renamed
 */
std::string parse_single_function_ptx(std::string_view src,
                                      std::string const& function_name = "transform");

/**
 * @brief In a piece of CUDA code that contains the implementation
 * of a device function, locate the function and replace its function name
 * with the specified one.
 *
 * @param src The input CUDA code.
 * @param function_name The User defined function that the output CUDA function
 * will have.
 * @return The output CUDA device function
 */
std::string parse_single_function_cuda(std::string const& src, std::string const& function_name);

}  // namespace jit
}  // namespace CUDF_EXPORT cudf

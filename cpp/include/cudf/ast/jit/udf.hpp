/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <vector>

/**
 * @file
 * @brief Calls of consumer-supplied device functions from JIT expressions.
 */

namespace CUDF_EXPORT cudf {
namespace ast::jit {
/**
 * @addtogroup expressions
 * @{
 */

/**
 * @brief A device function compiled ahead of time to LTO-IR, which a JIT expression can call.
 *
 * `fragment` holds the function's LTO-IR, or a fatbin carrying it. `symbol` is the function's
 * unmangled name. The kernel of an expression that calls the function is compiled to LTO-IR and
 * linked with `fragment`, so the function's body is never compiled at runtime.
 *
 * The function must have the signature
 *
 * ```cpp
 * extern "C" __device__ cudf::errc symbol(R* out, A... in);
 * ```
 *
 * where `R` is the device type of the call's output type and `A...` are the device types of its
 * arguments, passed by value (`int32_t`, `double`, `cudf::timestamp_D`, ...). It returns
 * `cudf::errc::SUCCESS`, or an error code that fails the row, such as one of the codes
 * `cudf::errc::USER_ERROR_0` to `USER_ERROR_31` that libcudf reserves for user-defined functions.
 * The signature is provisional: it will follow the row ABI that libcudf's JIT is moving to.
 *
 * Rules for the function and its fragment:
 * - The signature is not type-checked: a fragment whose function differs from the declaration
 *   cuDF derives from the call has undefined behavior.
 * - cuDF assumes the function is pure: calls with equal arguments return equal results, and no
 *   side effect has to happen on every call. Reading constant data, such as a table compiled into
 *   the fragment, is allowed. Equal calls may then share one evaluation. The caller must set
 *   `is_pure` to `false` for a function that does not meet this, for example one that reads a
 *   clock, a random state, or memory that changes while the kernel runs. cuDF cannot check this
 *   setting, and a wrong one changes results.
 * - A call of a function that is not pure is evaluated at most once per row, however many
 *   expressions refer to it, and is never merged with another call.
 * - The function must be row-local: no warp-collective operations, `__syncthreads` or shared
 *   memory.
 * - The fragment must target an architecture no newer than the device's, such as the one libcudf
 *   compiles its own LTO-IR for.
 * - The fragment must stay valid while expressions that call it are evaluated, and until a
 *   `cudf::transform_program` compiled from them is constructed.
 * - A fragment is identified by its address and size, so every call of one symbol must pass the
 *   same span, and must agree on purity, argument types and output type.
 */
struct device_binary {
  std::span<uint8_t const> fragment;                      ///< The function's LTO-IR or fatbin
  lto_binary_type binary_type = lto_binary_type::LTO_IR;  ///< What `fragment` holds
  std::string symbol;                                     ///< The function's unmangled name
  bool is_pure = true;  ///< Whether equal calls may share one evaluation
};

/**
 * @brief Creates a JIT expression that calls a consumer-supplied device function.
 *
 * The call is part of the generated JIT kernel, so it fuses with the surrounding expression. It
 * can only be evaluated by the JIT evaluators, such as `cudf::compute_column_jit`.
 *
 * A row with a null argument skips the call and produces null. A failing row raises
 * `cudf::evaluation_error` under `error_policy::PROPAGATE`, and produces null under
 * `error_policy::NULLIFY`.
 *
 * @throws std::invalid_argument if `output_type` is not fixed-width, `args` is empty, `function`
 * has no fragment, or `function.symbol` is not a C identifier
 *
 * @param tree The expression tree to which this expression will be added
 * @param function The device function to call
 * @param output_type The type of the value the function writes
 * @param args The call's arguments
 * @param error_policy How a row whose call fails is handled
 * @return A reference to the created call expression
 */
expression const& call(ast::tree& tree,
                       device_binary const& function,
                       data_type output_type,
                       std::vector<std::reference_wrapper<expression const>> const& args,
                       cudf::error_policy error_policy = cudf::error_policy::PROPAGATE);

/** @} */  // end of group
}  // namespace ast::jit
}  // namespace CUDF_EXPORT cudf

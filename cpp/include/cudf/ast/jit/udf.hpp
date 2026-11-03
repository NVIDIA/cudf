/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/ast/expressions.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <functional>
#include <string>
#include <vector>

/** @file
 * @brief Calls of consumer-supplied device functions from JIT expressions.
 */
namespace CUDF_EXPORT cudf {
namespace ast {
/**
 * @addtogroup expressions
 * @{
 */
namespace jit {

/**
 * @brief Declaration of a consumer-supplied device function.
 *
 * The function must have the following signature:
 * @code{.cpp}
 * extern "C" __device__ cudf::errc symbol(R* out, A... in);
 * @endcode
 *
 * Arguments are passed by value, including an empty argument list. When `is_null_aware` is YES,
 * both R and each A are wrapped in `cuda::std::optional`: the function receives nulls and may
 * produce nulls. Otherwise a null argument skips the call and produces null.
 *
 * The symbol must not conflict with a C-linkage function already declared by the generated
 * kernel or its headers, such as `transform` or `round`. Such conflicts cause JIT compilation
 * to fail with a conflicting declaration. Use a distinct symbol name, for example with a
 * consumer-specific prefix.
 *
 * STRING inputs are passed as `cudf::string_view`.
 *
 * Decimal inputs retain their scales. For decimal outputs, only the representation is used;
 * the scale of the decimal value written by the function is ignored. Users must pass the
 * required output scale as an explicit argument and rescale the output representation as needed
 * to match the scale declared in `output_type`.
 *
 * Supply the modules defining the function when evaluating the expression or constructing a
 * transform program. Modules may define any number of functions and reference other modules.
 * Their signatures must match this declaration; a mismatch has undefined behavior. All calls to
 * a given symbol in a JIT kernel must use a consistent ABI: the output pointer type and the
 * number, order, and device types of input arguments must match, including the optional wrapping
 * selected by `is_null_aware`.
 *
 * Functions must be row-local, without warp collectives, shared memory, or synchronization.
 *
 * Pure functions return equal results for equal arguments and have no required side effects.
 * Equal pure calls may share one evaluation. Set `is_pure` to false for functions that read a
 * clock, mutable state, or otherwise require separate evaluations. Calls of impure functions are
 * never merged, including repeated references to a call expression.
 *
 * Functions return SUCCESS or an error code. Error codes below zero are ignored for error
 * reporting. Codes at or above `errc::USER_RESERVED` belong to the consumer and are preserved
 * by `evaluation_error`.
 */
struct udf {
  std::string symbol;                         ///< Unmangled device function name
  bool is_pure                       = true;  ///< Whether equal calls may share an evaluation
  data_type output_type              = {};    ///< Type written through the output pointer
  std::vector<data_type> input_types = {};    ///< Argument types, in order
  null_aware is_null_aware           = null_aware::NO;  ///< Whether the ABI uses optional values

  bool operator==(udf const&) const = default;
};

/**
 * @brief Add a call of a consumer device function to a JIT expression tree.
 *
 * Calls fuse with surrounding operators. Under PROPAGATE, failures raise `evaluation_error`;
 * under NULLIFY, failing rows produce null.
 *
 * @param tree Expression tree receiving the call
 * @param function Declaration of the function to call
 * @param args Arguments matching the declared input types
 * @param error_policy How failing rows are handled
 * @return The created expression
 */
// The returned expression belongs to tree, independently of argument temporaries.
#if __has_cpp_attribute(gnu::no_dangling)
[[gnu::no_dangling]]
#endif
expression const& call(ast::tree& tree,
                       ast::jit::udf function,
                       std::vector<std::reference_wrapper<expression const>> args,
                       cudf::error_policy error_policy = cudf::error_policy::PROPAGATE);
}  // namespace jit
/** @} */
}  // namespace ast
}  // namespace CUDF_EXPORT cudf

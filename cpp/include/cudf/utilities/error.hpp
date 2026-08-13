/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/errc.hpp>
#include <cudf/utilities/export.hpp>

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>
#include <type_traits>

/**
 * @file
 * @brief Exception types and error-checking macros used throughout libcudf.
 */

namespace CUDF_EXPORT cudf {
/**
 * @addtogroup utility_error
 * @{
 */

/**
 * @brief Exception thrown when logical precondition is violated.
 *
 * This exception should not be thrown directly and is instead thrown by the
 * CUDF_EXPECTS macro.
 */
struct logic_error : std::logic_error {
  /**
   * @brief Constructs a logic_error with the error message.
   *
   * @param message Message to be associated with the exception
   */
  explicit logic_error(char const* const message) : std::logic_error(message) {}

  /**
   * @brief Construct a new logic error object with error message
   *
   * @param message Message to be associated with the exception
   */
  explicit logic_error(std::string const& message) : std::logic_error(message) {}

  // TODO Add an error code member? This would be useful for translating an
  // exception to an error code in a pure-C API

  ~logic_error() override
  {
    // Needed so that the first instance of the implicit destructor for any TU isn't 'constructed'
    // from a host+device function marking the implicit version also as host+device
  }
};
/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 */
struct cuda_error : std::runtime_error {
  /**
   * @brief Construct a new cuda error object with error message and code.
   *
   * @param message Error message
   * @param error CUDA error code
   */
  explicit cuda_error(std::string const& message, cudaError_t const& error)
    : std::runtime_error(message), _cudaError(error)
  {
  }

  /**
   * @brief Returns the CUDA error code associated with the exception.
   *
   * @return CUDA error code
   */
  [[nodiscard]] cudaError_t error_code() const { return _cudaError; }

 protected:
  cudaError_t _cudaError;  //!< CUDA error code
};

struct fatal_cuda_error : cuda_error {
  using cuda_error::cuda_error;  // Inherit constructors
};

/**
 * @brief Exception thrown when an operation is attempted on an unsupported dtype.
 *
 * This exception should be thrown when an operation is attempted on an
 * unsupported data_type. This exception should not be thrown directly and is
 * instead thrown by the CUDF_EXPECTS or CUDF_FAIL macros.
 */
struct data_type_error : std::invalid_argument {
  /**
   * @brief Constructs a data_type_error with the error message.
   *
   * @param message Message to be associated with the exception
   */
  explicit data_type_error(char const* const message) : std::invalid_argument(message) {}

  /**
   * @brief Construct a new data_type_error object with error message
   *
   * @param message Message to be associated with the exception
   */
  explicit data_type_error(std::string const& message) : std::invalid_argument(message) {}
};

/**
 * @brief Exception type thrown when evaluating an operator function results in an error (e.g.
 * overflow, division by zero, etc.)
 *
 */
struct evaluation_error : public std::exception {
  /**
   * @brief Construct a new evaluation error object.
   *
   * @param error The error code that occurred during evaluation
   * @param message An error message describing the evaluation error
   */
  evaluation_error(errc error, std::string message) : error_(error), message_(std::move(message)) {}

  /**
   * @brief Get the error message.
   * @return A C-string describing the error
   */
  [[nodiscard]] char const* what() const noexcept override { return message_.c_str(); }

  /**
   * @brief Get the error code that occurred during evaluation.
   * @return The error code
   */
  [[nodiscard]] errc error_code() const { return error_; }

 private:
  errc error_;           //< The error code that occurred during evaluation
  std::string message_;  //< An error message describing the evaluation error
};

/** @} */

}  // namespace CUDF_EXPORT cudf

#define STRINGIFY_DETAIL(x) #x                   ///< Stringify a macro argument
#define CUDF_STRINGIFY(x)   STRINGIFY_DETAIL(x)  ///< Stringify a macro argument

/**
 * @addtogroup utility_error
 * @{
 */

/**
 * @brief Macro for checking (pre-)conditions that throws an exception when
 * a condition is violated.
 *
 * Defaults to throwing `cudf::logic_error`, but a custom exception may also be
 * specified.
 *
 * Example usage:
 * ```
 * // throws cudf::logic_error
 * CUDF_EXPECTS(p != nullptr, "Unexpected null pointer");
 *
 * // throws std::runtime_error
 * CUDF_EXPECTS(p != nullptr, "Unexpected nullptr", std::runtime_error);
 * ```
 * @param ... This macro accepts either two or three arguments:
 *   - The first argument must be an expression that evaluates to true or
 *     false, and is the condition being checked.
 *   - The second argument is an expression that produces the error message used to construct the
 *     `what` of the exception. This can be a string literal or a dynamic expression (e.g.,
 *     `std::to_string(x) + " is invalid"`). The expression is only evaluated when the condition
 *     fails.
 *   - When given, the third argument is the exception to be thrown. When not
 *     specified, defaults to `cudf::logic_error`.
 * @throw `_exception_type` if the condition evaluates to 0 (false).
 */
#define CUDF_EXPECTS(...)                                             \
  GET_CUDF_EXPECTS_MACRO(__VA_ARGS__, CUDF_EXPECTS_3, CUDF_EXPECTS_2) \
  (__VA_ARGS__)

/// @cond

#define GET_CUDF_EXPECTS_MACRO(_1, _2, _3, NAME, ...) NAME

#define CUDF_EXPECTS_3(_condition, _reason, _exception_type)           \
  do {                                                                 \
    static_assert(std::is_base_of_v<std::exception, _exception_type>); \
    if (!(_condition)) {                                               \
      cudf::detail::cudf_fail<_exception_type>(                        \
        [&]() -> std::string { return _reason; }, __LINE__, __FILE__); \
    }                                                                  \
  } while (0)

#define CUDF_EXPECTS_2(_condition, _reason) CUDF_EXPECTS_3(_condition, _reason, cudf::logic_error)

/// @endcond

/**
 * @brief Indicates that an erroneous code path has been taken.
 *
 * Example usage:
 * ```c++
 * // Throws `cudf::logic_error`
 * CUDF_FAIL("Unsupported code path");
 *
 * // Throws `std::runtime_error`
 * CUDF_FAIL("Unsupported code path", std::runtime_error);
 * ```
 *
 * @param ... This macro accepts either one or two arguments:
 *   - The first argument is an expression that produces the error message used to construct the
 *     `what` of the exception. This can be a string literal or a dynamic expression (e.g.,
 *     `std::to_string(x) + " is invalid"`).
 *   - When given, the second argument is the exception to be thrown. When not
 *     specified, defaults to `cudf::logic_error`.
 * @throw `_exception_type` if the condition evaluates to 0 (false).
 */
#define CUDF_FAIL(...)                                       \
  GET_CUDF_FAIL_MACRO(__VA_ARGS__, CUDF_FAIL_2, CUDF_FAIL_1) \
  (__VA_ARGS__)

/// @cond

#define GET_CUDF_FAIL_MACRO(_1, _2, NAME, ...) NAME

#define CUDF_FAIL_2(_what, _exception_type) \
  cudf::detail::cudf_fail<_exception_type>( \
    [&]() -> std::string { return _what; }, __LINE__, __FILE__)

#define CUDF_FAIL_1(_what) CUDF_FAIL_2(_what, cudf::logic_error)

/// @endcond

namespace CUDF_EXPORT cudf {
namespace detail {
// @cond
inline void throw_cuda_error(cudaError_t error, char const* file, unsigned int line)
{
  // Calls cudaGetLastError to clear the error status. It is nearly certain that a fatal error
  // occurred if it still returns the same error after a cleanup.
  cudaGetLastError();
  auto const last = cudaFree(nullptr);
  auto const msg  = std::string{"CUDA error encountered at: " + std::string{file} + ":" +
                               std::to_string(line) + ": " + std::to_string(error) + " " +
                               cudaGetErrorName(error) + " " + cudaGetErrorString(error)};
  // Call cudaDeviceSynchronize to ensure `last` did not result from an asynchronous error.
  // between two calls.
  if (error == last && last == cudaDeviceSynchronize()) {
    throw fatal_cuda_error{"Fatal " + msg, error};
  } else {
    throw cuda_error{msg, error};
  }
}
// @endcond

// @cond
template <typename Exception, typename MsgFunc>
[[noreturn]] void cudf_fail(MsgFunc&& msg_func, int line_number, char const* filename)
{
  std::string const msg = std::forward<MsgFunc>(msg_func)();
  throw Exception{std::string{"CUDF failure at: "} + filename + ":" + std::to_string(line_number) +
                  ": " + (msg.empty() ? "(no message)" : msg)};
}
// @endcond
}  // namespace detail
}  // namespace CUDF_EXPORT cudf

/**
 * @brief Error checking macro for CUDA runtime API functions.
 *
 * Invokes a CUDA runtime API function call, if the call does not return
 * cudaSuccess, invokes cudaGetLastError() to clear the error and throws an
 * exception detailing the CUDA error that occurred
 */
#define CUDF_CUDA_TRY(call)                                                                    \
  do {                                                                                         \
    cudaError_t const status = (call);                                                         \
    if (cudaSuccess != status) { cudf::detail::throw_cuda_error(status, __FILE__, __LINE__); } \
  } while (0);

/**
 * @brief Debug macro to check for CUDA errors
 *
 * In a non-release build, this macro will synchronize the specified stream
 * before error checking. In both release and non-release builds, this macro
 * checks for any pending CUDA errors from previous calls. If an error is
 * reported, an exception is thrown detailing the CUDA error that occurred.
 *
 * The intent of this macro is to provide a mechanism for synchronous and
 * deterministic execution for debugging asynchronous CUDA execution. It should
 * be used after any asynchronous CUDA call, e.g., cudaMemcpyAsync, or an
 * asynchronous kernel launch.
 */
#ifndef NDEBUG
#define CUDF_CHECK_CUDA(stream)                   \
  do {                                            \
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream)); \
    CUDF_CUDA_TRY(cudaPeekAtLastError());         \
  } while (0);
#else
#define CUDF_CHECK_CUDA(stream) CUDF_CUDA_TRY(cudaPeekAtLastError());
#endif
/** @} */

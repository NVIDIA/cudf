/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/utilities/error.hpp>

#include <stdexcept>
#include <type_traits>

/**
 * @file
 * @brief Defines standard dispatching utilities to simplify
 * runtime-to-compile-time parameter dispatch.
 */
namespace cudf {
namespace detail {

/**
 * @brief Helper to dispatch a single enum value for a function with a non-void return type.
 *
 * For a function that returns a non-void value, we need to pass back two bits
 * of information: whether the current enum value matched the runtime value,
 * and if so, the result of invoking the callable. This helper function accomplishes
 * that by taking a reference to store the actual result and returning a bool flag.
 *
 * @tparam Value The compile-time enum value to compare against.
 * @tparam Func The type of the callable object.
 * @param runtime_value The runtime enum value to compare.
 * @param func The callable object to invoke if the values match.
 * @param result Reference to store the result of the callable invocation.
 * @return `true` if the dispatch was successful (values matched), `false` otherwise.
 */
template <auto Value, typename Func>
bool try_dispatch_enum(auto runtime_value, Func&& func, auto& result)
{
  if (runtime_value == Value) {
    result = func(std::integral_constant<decltype(runtime_value), Value>{});
    return true;
  }
  return false;
}

// Helper to extract first value from parameter pack
template <auto First, auto...>
inline constexpr auto first_value = First;

/**
 * @addtogroup utility_dispatcher
 * @{
 * @file
 */

/**
 * @brief Dispatches a boolean runtime value to a compile-time constant.
 *
 * This function takes a boolean value and a callable object (e.g., lambda or function)
 * and invokes the callable with a `std::bool_constant` corresponding to the boolean value.
 *
 * @tparam Func The type of the callable object.
 * @param value The boolean runtime value to dispatch.
 * @param func The callable object to invoke with the dispatched boolean constant.
 * @return The result of invoking the callable object.
 */
template <typename Func>
auto dispatch_bool(bool value, Func&& func)
{
  // Named aliases avoid an NVCC bug that rewrites bool_constant template arguments
  // as expressions referring to unrelated private types.
  if (value) {
    return func(std::true_type{});
  } else {
    return func(std::false_type{});
  }
}

/**
 * @brief Dispatches a runtime enum value to a compile-time constant.
 *
 * This function takes a runtime enum value and a callable object (e.g., lambda or function)
 * and invokes the callable with a `std::integral_constant` corresponding to the enum value.
 * The set of possible enum values must be provided as template parameters.
 *
 * @throws std::logic_error if the runtime value does not match any of the provided enum values.
 *
 * @tparam Values The set of possible enum values to dispatch.
 * @tparam Func The type of the callable object.
 * @param runtime_value The runtime enum value to dispatch.
 * @param func The callable object to invoke with the dispatched enum constant.
 * @return The result of invoking the callable object.
 */
template <auto... Values, typename Func>
auto dispatch_enum(auto runtime_value, Func&& func)
{
  using EnumType = decltype(runtime_value);
  using RetType  = decltype(func(std::integral_constant<EnumType, first_value<Values...>>{}));

  if constexpr (std::is_void_v<RetType>) {
    // Handle void return type
    bool found = false;
    ((!found && runtime_value == Values
        ? (func(std::integral_constant<EnumType, Values>{}), found = true)
        : false),
     ...);

    // Reaching this point indicates developer error
    CUDF_EXPECTS(found, "Invalid enum value for dispatch_enum");
  } else {
    // Handle non-void return type
    RetType result{};
    bool found = (try_dispatch_enum<Values>(runtime_value, func, result) || ...);

    // Reaching this point indicates developer error
    CUDF_EXPECTS(found, "Invalid enum value for dispatch_enum");
    return result;
  }
}

/** @} */  // end of group
}  // namespace detail
}  // namespace cudf

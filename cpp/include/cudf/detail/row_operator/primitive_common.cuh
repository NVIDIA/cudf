/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/type_traits>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Checks if a table is compatible with primitive row operations
 *
 * A table is compatible with primitive row operations if all its columns have numeric data types.
 *
 * @param table The table to check for compatibility
 * @return Boolean indicating if the table is compatible with primitive row operations
 */
bool is_primitive_row_op_compatible(cudf::table_view const& table);

namespace row::primitive {

/**
 * @brief Returns `void` if it's not a primitive type
 */
template <typename T>
using primitive_type_t = cuda::std::conditional_t<cudf::is_numeric<T>(), T, void>;

/**
 * @brief Custom dispatcher for primitive types
 */
template <cudf::type_id Id>
struct dispatch_primitive_type {
  using type = primitive_type_t<id_to_type<Id>>;  ///< The underlying type
};

}  // namespace row::primitive
}  // namespace detail
}  // namespace CUDF_EXPORT cudf

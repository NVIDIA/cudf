/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/scalar/scalar.hpp>

#include <string>

namespace cudf {
namespace jit {
/**
 * @brief Get the raw pointer to data in a (mutable_)column_view
 */
void const* get_data_ptr(column_view const& view);

/**
 * @brief Get the raw pointer to data in a scalar
 */
void const* get_data_ptr(scalar const& s);

/**
 * @brief Determine the physical type of a given element type.
 * The Physical type is the type that can be used to alias the element type through a
 * `cudf::column_device_view::element`. For example, the physical type of `int32`, `uint32`,
 * `float`, `duration_D` is `uint32`. This is because `uint32` can be used to alias all 5 types
 * through a `cudf::column_device_view::element`.
 *
 * It also means these types can safely alias each other across an ABI boundary as they have the
 * same register storage type (PTX `b32`).
 *
 * e.g.
 *
 * ```cpp
 * // PTX: .extern .func (.param .b32 func_retval0) my_udf(.param .b32 a);
 *
 * extern void my_udf_uint32(uint32_t a);
 * extern void my_udf_int32(int32_t a);
 * extern void my_udf_float(float a);
 * extern void my_udf_duration_D(duration_D a);
 * ```
 *
 * `my_udf_int32` and `my_udf_float`, `my_udf_duration_D` can safely alias `my_udf` because
 * they have the same register storage type (PTX `b32`).
 *
 *
 * This means that some CUDA functions/kernels that are template-specialized on physical types can
 * be re-used for other types that have the same physical type, thus reducing the compilation cost
 * of code specialization.
 *
 */
data_type physical_type_of(data_type type);

}  // namespace jit
}  // namespace cudf

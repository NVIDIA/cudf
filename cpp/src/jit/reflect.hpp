/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/transform.hpp>

#include <jit/column_views.hpp>

#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace cudf::jit {

using transform_reflection =
  std::tuple<std::string, std::string, std::vector<std::string>, std::vector<std::string>>;

/** @brief Reflects input/output accessors and PTX value types. */
transform_reflection reflect(bool is_ptx,
                             std::span<transform_input_spec const> inputs,
                             std::span<transform_output_spec const> outputs,
                             bool use_physical_types = false);

/** @brief Reflects accessors and PTX value types from column storage. */
transform_reflection reflect(bool is_ptx,
                             std::span<transform_input const> inputs,
                             std::span<output_column const> outputs,
                             bool use_physical_types = false);

/** @brief Derives input type specifications from columns or scalars. */
std::vector<transform_input_spec> make_input_specs(std::span<transform_input const> inputs);

/** @brief Derives output type specifications from column storage. */
std::vector<transform_output_spec> make_output_specs(std::span<output_column const> outputs);

/** @brief Derives output type specifications from requested types and offsets. */
std::vector<transform_output_spec> make_output_specs(
  std::span<transform_output const> outputs,
  std::span<std::unique_ptr<column> const> output_offsets);

/** @brief Reflects the UDF function signature from type specifications. */
std::string reflect_udf_signature(bool is_null_aware,
                                  bool has_user_data,
                                  std::span<transform_input_spec const> inputs,
                                  std::span<transform_output_spec const> outputs,
                                  bool use_physical_types);

/** @brief Reflects the UDF function signature from column storage. */
std::string reflect_udf_signature(bool is_null_aware,
                                  bool has_user_data,
                                  std::span<transform_input const> inputs,
                                  std::span<output_column const> outputs,
                                  bool use_physical_types);

}  // namespace cudf::jit

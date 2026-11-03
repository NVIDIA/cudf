/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/transform.hpp>

#include <jit/output_columns.hpp>

#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace cudf::jit {

struct transform_reflection {
  std::string input_accessors;
  std::string output_accessors;
  std::vector<std::string> ptx_input_value_types;
  std::vector<std::string> ptx_output_value_types;

  bool operator==(transform_reflection const& other) const = default;
};

/**
 * @brief Reflects input/output accessors and PTX value types.
 *
 * @param source_type The type of the UDF source (either CUDA source or LTO binary).
 * @param inputs The input type specifications.
 * @param outputs The output type specifications.
 * @param input_table_sources Optional mapping of input table sources. If empty, the default table
 * source `0` is used.
 * @param input_table_indices Optional mapping of input table indices. If empty, the index at
 * position `i` is used.
 * @param output_table_sources Optional mapping of output table sources. If empty, the default table
 * source `0` is used.
 * @param output_table_indices Optional mapping of output table indices. If empty, the index at
 * position `i` is used.
 *
 * @return A transform_reflection containing the reflected accessors and PTX value types.
 */
transform_reflection reflect(std::variant<udf_source_type, lto_binary_type> source_type,
                             std::span<transform_input_spec const> inputs,
                             std::span<transform_output_spec const> outputs,
                             std::span<int32_t const> input_table_sources  = {},
                             std::span<int32_t const> input_table_indices  = {},
                             std::span<int32_t const> output_table_sources = {},
                             std::span<int32_t const> output_table_indices = {});

/** @brief Reflects accessors and PTX value types from column storage.
 *
 * @param source_type The type of the UDF source (either CUDA source or LTO binary).
 * @param inputs The input columns or scalars.
 * @param outputs The output columns.
 * @param input_table_sources Optional mapping of input table sources. If empty, the default table
 * source `0` is used.
 * @param input_table_indices Optional mapping of input table indices. If empty, the index at
 * position `i` is used.
 * @param output_table_sources Optional mapping of output table sources. If empty, the default table
 * source `0` is used.
 * @param output_table_indices Optional mapping of output table indices. If empty, the index at
 * position `i` is used.
 *
 * @return A transform_reflection containing the reflected accessors and PTX value types.
 */
transform_reflection reflect(std::variant<udf_source_type, lto_binary_type> source_type,
                             std::span<transform_input const> inputs,
                             std::span<output_column const> outputs,
                             std::span<int32_t const> input_table_sources  = {},
                             std::span<int32_t const> input_table_indices  = {},
                             std::span<int32_t const> output_table_sources = {},
                             std::span<int32_t const> output_table_indices = {});

/** @brief Derives input type specifications from columns or scalars.
 *
 * @param inputs The input columns or scalars.
 *
 * @return A vector of transform_input_spec representing the input type specifications.
 */
std::vector<transform_input_spec> make_input_specs(std::span<transform_input const> inputs);

/** @brief Derives output type specifications from column storage.
 *
 * @param outputs The output columns.
 *
 * @return A vector of transform_output_spec representing the output type specifications.
 */
std::vector<transform_output_spec> make_output_specs(std::span<output_column const> outputs);

/** @brief Derives output type specifications from requested types and offsets.
 *
 * @param outputs The requested output types.
 * @param string_offsets The string offset columns corresponding to the outputs.
 *
 * @return A vector of transform_output_spec representing the output type specifications.
 */
std::vector<transform_output_spec> make_output_specs(
  std::span<transform_output const> outputs,
  std::span<std::unique_ptr<column> const> string_offsets);

/** @brief Reflects the UDF function signature from type specifications.
 *
 * @param is_null_aware Whether the UDF is null-aware.
 * @param has_user_data Whether the UDF has user data.
 * @param inputs The input type specifications.
 * @param outputs The output type specifications.
 * @param use_physical_types Whether to use physical types in the signature.
 *
 * @return A string representing the UDF function signature.
 */
std::string reflect_udf_signature(bool is_null_aware,
                                  bool has_user_data,
                                  std::span<transform_input_spec const> inputs,
                                  std::span<transform_output_spec const> outputs,
                                  bool use_physical_types);

/** @brief Reflects the UDF function signature from column storage.
 *
 * @param is_null_aware Whether the UDF is null-aware.
 * @param has_user_data Whether the UDF has user data.
 * @param inputs The input columns or scalars.
 * @param outputs The output columns.
 * @param use_physical_types Whether to use physical types in the signature.
 *
 * @return A string representing the UDF function signature.
 */
std::string reflect_udf_signature(bool is_null_aware,
                                  bool has_user_data,
                                  std::span<transform_input const> inputs,
                                  std::span<output_column const> outputs,
                                  bool use_physical_types);

}  // namespace cudf::jit

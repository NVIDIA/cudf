/*
 * SPDX-FileCopyrightText: Copyright 2018-2019 BlazingDB, Inc.
 * SPDX-FileCopyrightText: Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
/*
 * Copyright 2018-2019 BlazingDB, Inc.
 *     Copyright 2018 Christian Noboa Mardini <christian@blazingdb.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "compiled/binary_ops.hpp"
#include "jit/cache.hpp"
#include "jit/helpers.hpp"
#include "jit/parser.hpp"
#include "jit/util.hpp"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/binaryop.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/std/optional>

#include <string>

namespace cudf {
namespace binops {

bool is_supported_operation(data_type out, data_type lhs, data_type rhs, binary_operator op)
{
  return cudf::binops::compiled::is_supported_operation(out, lhs, rhs, op);
}

/**
 * @brief Computes output valid mask for op between a column and a scalar
 */
std::pair<rmm::device_buffer, size_type> scalar_col_valid_mask_and(
  column_view const& col,
  scalar const& s,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  if (col.is_empty()) return std::pair(rmm::device_buffer{0, stream, mr}, 0);

  if (not s.is_valid(stream)) {
    return std::pair(cudf::detail::create_null_mask(col.size(), mask_state::ALL_NULL, stream, mr),
                     col.size());
  } else if (s.is_valid(stream) and col.nullable()) {
    return std::pair(cudf::detail::copy_bitmask(col, stream, mr), col.null_count());
  } else {
    return std::pair(rmm::device_buffer{0, stream, mr}, 0);
  }
}

/**
 * @brief Does the binop need to know if an operand is null/invalid to perform special
 * processing?
 */
inline bool is_null_dependent(binary_operator op)
{
  return op == binary_operator::NULL_EQUALS || op == binary_operator::NULL_NOT_EQUALS ||
         op == binary_operator::NULL_MIN || op == binary_operator::NULL_MAX ||
         op == binary_operator::NULL_LOGICAL_AND || op == binary_operator::NULL_LOGICAL_OR;
}

/**
 * @brief Returns `true` if `binary_operator` `op` is a basic arithmetic binary operation
 */
bool is_basic_arithmetic_binop(binary_operator op)
{
  return op == binary_operator::ADD or       // operator +
         op == binary_operator::SUB or       // operator -
         op == binary_operator::MUL or       // operator *
         op == binary_operator::DIV or       // operator / using common type of lhs and rhs
         op == binary_operator::NULL_MIN or  // 2 null = null, 1 null = value, else min
         op == binary_operator::NULL_MAX or  // 2 null = null, 1 null = value, else max
         op == binary_operator::MOD or       // operator %
         op == binary_operator::PMOD or      // positive modulo operator
         op == binary_operator::PYMOD;  // operator % but following Python's negative sign rules
}

/**
 * @brief Returns `true` if `binary_operator` `op` is a comparison binary operation
 */
bool is_comparison_binop(binary_operator op)
{
  return op == binary_operator::EQUAL or          // operator ==
         op == binary_operator::NOT_EQUAL or      // operator !=
         op == binary_operator::LESS or           // operator <
         op == binary_operator::GREATER or        // operator >
         op == binary_operator::LESS_EQUAL or     // operator <=
         op == binary_operator::GREATER_EQUAL or  // operator >=
         op == binary_operator::NULL_EQUALS or    // 2 null = true; 1 null = false; else ==
         op == binary_operator::NULL_NOT_EQUALS;  // 2 null = false; 1 null = true; else !=
}

/**
 * @brief Returns `true` if `binary_operator` `op` is supported by `fixed_point`
 */
bool is_supported_fixed_point_binop(binary_operator op)
{
  return is_basic_arithmetic_binop(op) or is_comparison_binop(op);
}

/**
 * @brief Helper predicate function that identifies if `op` requires scales to be the same
 *
 * @param op `binary_operator`
 * @return true `op` requires scales of lhs and rhs to be the same
 * @return false `op` does not require scales of lhs and rhs to be the same
 */
bool is_same_scale_necessary(binary_operator op)
{
  return op != binary_operator::MUL && op != binary_operator::DIV;
}

namespace jit {
void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      column_view const& rhs,
                      std::string const& ptx,
                      rmm::cuda_stream_view stream)
{
  std::string const output_type_name = cudf::type_to_name(out.type());

  std::string cuda_source =
    cudf::jit::parse_single_function_ptx(ptx,
                                         "GENERIC_BINARY_OP",
                                         {
                                           {0, output_type_name + " *"},
                                           {1, cudf::type_to_name(lhs.type())},
                                           {2, cudf::type_to_name(rhs.type())},
                                         });

  std::string kernel_reflection = rtcx::reflect_template("cudf::binops::jit::binary_op_kernel",
                                                         output_type_name,
                                                         cudf::type_to_name(lhs.type()),
                                                         cudf::type_to_name(rhs.type()),
                                                         "cudf::binops::jit::UserDefinedOp");

  auto kernel = cudf::jit::get_udf_kernel(
    "cudf/cpp/src/binaryop/jit/kernel.cu", kernel_reflection, cuda_source);
  auto cfg = kernel.max_occupancy_config(0, 0);

  kernel.launch_with({cfg.min_grid_size},
                     {cfg.block_size},
                     0,
                     stream,
                     static_cast<size_type>(out.size()),
                     cudf::jit::get_data_ptr(out),
                     cudf::jit::get_data_ptr(lhs),
                     cudf::jit::get_data_ptr(rhs));
}
}  // namespace jit

// Compiled Binary operation
namespace compiled {

template <typename Lhs, typename Rhs>
void fixed_point_binary_operation_validation(binary_operator op,
                                             Lhs lhs,
                                             Rhs rhs,
                                             cuda::std::optional<cudf::data_type> output_type = {})
{
  CUDF_EXPECTS((is_fixed_point(lhs) or is_fixed_point(rhs)),
               "One of the inputs must have fixed_point data_type.");
  CUDF_EXPECTS(binops::is_supported_fixed_point_binop(op),
               "Unsupported fixed_point binary operation");
  if (output_type.has_value() and binops::is_comparison_binop(op))
    CUDF_EXPECTS(output_type == cudf::data_type{type_id::BOOL8},
                 "Comparison operations require boolean output type.");
}

/**
 * @copydoc cudf::binary_operation(column_view const&, column_view const&,
 * binary_operator, data_type, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
template <typename LhsType, typename RhsType>
std::unique_ptr<column> binary_operation(LhsType const& lhs,
                                         RhsType const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  if constexpr (std::is_same_v<LhsType, column_view> and std::is_same_v<RhsType, column_view>)
    CUDF_EXPECTS(lhs.size() == rhs.size(), "Column sizes don't match", std::invalid_argument);

  if (lhs.type().id() == type_id::STRING and rhs.type().id() == type_id::STRING and
      output_type.id() == type_id::STRING and
      (op == binary_operator::NULL_MAX or op == binary_operator::NULL_MIN))
    return cudf::binops::compiled::string_null_min_max(lhs, rhs, op, output_type, stream, mr);

  if (not cudf::binops::compiled::is_supported_operation(output_type, lhs.type(), rhs.type(), op))
    CUDF_FAIL("Unsupported operator for these types", cudf::data_type_error);

  if (cudf::is_fixed_point(lhs.type()) or cudf::is_fixed_point(rhs.type())) {
    cudf::binops::compiled::fixed_point_binary_operation_validation(
      op, lhs.type(), rhs.type(), output_type);
  }

  auto out = make_fixed_width_column_for_output(lhs, rhs, op, output_type, stream, mr);

  if constexpr (std::is_same_v<LhsType, column_view>)
    if (lhs.is_empty()) return out;
  if constexpr (std::is_same_v<RhsType, column_view>)
    if (rhs.is_empty()) return out;

  auto out_view = out->mutable_view();
  cudf::binops::compiled::binary_operation(out_view, lhs, rhs, op, stream);
  // TODO: consider having the binary_operation count nulls instead
  out->set_null_count(cudf::detail::null_count(out_view.null_mask(), 0, out->size(), stream));
  return out;
}
}  // namespace compiled
}  // namespace binops

namespace detail {

// There are 3 overloads of each of the following functions:
// - `make_fixed_width_column_for_output`
// - `binary_operation`

// The overloads are overloaded on the first two parameters of each function:
// - scalar      const& lhs, column_view const& rhs,
// - column_view const& lhs, scalar      const& rhs
// - column_view const& lhs, column_view const& rhs,

/**
 * @brief Helper function for making output column for binary operation
 *
 * @param lhs Left-hand side `scalar` used in the binary operation
 * @param rhs Right-hand side `column_view` used in the binary operation
 * @param op `binary_operator` to be used to combine `lhs` and `rhs`
 * @param output_type `data_type` of the output column
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource to use for device memory allocation
 * @return std::unique_ptr<column> Output column used for binary operation
 */
std::unique_ptr<column> make_fixed_width_column_for_output(scalar const& lhs,
                                                           column_view const& rhs,
                                                           binary_operator op,
                                                           data_type output_type,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  if (binops::is_null_dependent(op)) {
    return make_fixed_width_column(output_type, rhs.size(), mask_state::ALL_VALID, stream, mr);
  } else {
    auto [new_mask, new_null_count] = binops::scalar_col_valid_mask_and(rhs, lhs, stream, mr);
    return make_fixed_width_column(
      output_type, rhs.size(), std::move(new_mask), new_null_count, stream, mr);
  }
};

/**
 * @brief Helper function for making output column for binary operation
 *
 * @param lhs Left-hand side `column_view` used in the binary operation
 * @param rhs Right-hand side `scalar` used in the binary operation
 * @param op `binary_operator` to be used to combine `lhs` and `rhs`
 * @param output_type `data_type` of the output column
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource to use for device memory allocation
 * @return std::unique_ptr<column> Output column used for binary operation
 */
std::unique_ptr<column> make_fixed_width_column_for_output(column_view const& lhs,
                                                           scalar const& rhs,
                                                           binary_operator op,
                                                           data_type output_type,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  if (binops::is_null_dependent(op)) {
    return make_fixed_width_column(output_type, lhs.size(), mask_state::ALL_VALID, stream, mr);
  } else {
    auto [new_mask, new_null_count] = binops::scalar_col_valid_mask_and(lhs, rhs, stream, mr);
    return make_fixed_width_column(
      output_type, lhs.size(), std::move(new_mask), new_null_count, stream, mr);
  }
};

/**
 * @brief Helper function for making output column for binary operation
 *
 * @param lhs Left-hand side `column_view` used in the binary operation
 * @param rhs Right-hand side `column_view` used in the binary operation
 * @param op `binary_operator` to be used to combine `lhs` and `rhs`
 * @param output_type `data_type` of the output column
 * @param stream CUDA stream used for device memory operations
 * @param mr Device memory resource to use for device memory allocation
 * @return std::unique_ptr<column> Output column used for binary operation
 */
std::unique_ptr<column> make_fixed_width_column_for_output(column_view const& lhs,
                                                           column_view const& rhs,
                                                           binary_operator op,
                                                           data_type output_type,
                                                           rmm::cuda_stream_view stream,
                                                           rmm::device_async_resource_ref mr)
{
  if (binops::is_null_dependent(op)) {
    return make_fixed_width_column(output_type, rhs.size(), mask_state::ALL_VALID, stream, mr);
  } else {
    auto [new_mask, null_count] = cudf::detail::bitmask_and(table_view({lhs, rhs}), stream, mr);
    return make_fixed_width_column(
      output_type, lhs.size(), std::move(new_mask), null_count, stream, mr);
  }
};

std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  return binops::compiled::binary_operation<scalar, column_view>(
    lhs, rhs, op, output_type, stream, mr);
}
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  return binops::compiled::binary_operation<column_view, scalar>(
    lhs, rhs, op, output_type, stream, mr);
}
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  return binops::compiled::binary_operation<column_view, column_view>(
    lhs, rhs, op, output_type, stream, mr);
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         std::string const& ptx,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  // Check for datatype
  auto is_type_supported_ptx = [](data_type type) -> bool {
    return is_fixed_width(type) and not is_fixed_point(type) and
           type.id() != type_id::INT8;  // Numba PTX doesn't support int8
  };

  CUDF_EXPECTS(is_type_supported_ptx(lhs.type()), "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(is_type_supported_ptx(rhs.type()), "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(is_type_supported_ptx(output_type), "Invalid/Unsupported output datatype");

  CUDF_EXPECTS((lhs.size() == rhs.size()), "Column sizes don't match");

  auto [new_mask, null_count] = cudf::detail::bitmask_and(table_view({lhs, rhs}), stream, mr);
  auto out =
    make_fixed_width_column(output_type, lhs.size(), std::move(new_mask), null_count, stream, mr);

  // Check for 0 sized data
  if (lhs.is_empty() or rhs.is_empty()) return out;

  auto out_view = out->mutable_view();
  binops::jit::binary_operation(out_view, lhs, rhs, ptx, stream);
  out->set_null_count(cudf::detail::null_count(out_view.null_mask(), 0, out->size(), stream));
  return out;
}
}  // namespace detail

int32_t binary_operation_fixed_point_scale(binary_operator op,
                                           int32_t left_scale,
                                           int32_t right_scale)
{
  CUDF_EXPECTS(binops::is_supported_fixed_point_binop(op),
               "Unsupported fixed_point binary operation.");
  if (op == binary_operator::MUL) return left_scale + right_scale;
  if (op == binary_operator::DIV) return left_scale - right_scale;
  return std::min(left_scale, right_scale);
}

cudf::data_type binary_operation_fixed_point_output_type(binary_operator op,
                                                         cudf::data_type const& lhs,
                                                         cudf::data_type const& rhs)
{
  cudf::binops::compiled::fixed_point_binary_operation_validation(op, lhs, rhs);

  auto const scale = binary_operation_fixed_point_scale(op, lhs.scale(), rhs.scale());
  return cudf::data_type{lhs.id(), scale};
}

std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::binary_operation(lhs, rhs, op, output_type, stream, mr);
}
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::binary_operation(lhs, rhs, op, output_type, stream, mr);
}
std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::binary_operation(lhs, rhs, op, output_type, stream, mr);
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         std::string const& ptx,
                                         data_type output_type,
                                         rmm::cuda_stream_view stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::binary_operation(lhs, rhs, ptx, output_type, stream, mr);
}

}  // namespace cudf

/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "binary_ops.hpp"
#include "operation.cuh"
#include "struct_binary_ops.cuh"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace binops {
namespace compiled {

namespace {
// This functor compares string inputs represented by column device views. Scalar inputs use row 0.
template <typename OutT, typename CompareFunc>
struct compare_functor {
  column_device_view const lhs_dev_view_;
  column_device_view const rhs_dev_view_;
  CompareFunc const cfunc_;            // Comparison function
  bool const is_lhs_scalar_;
  bool const is_rhs_scalar_;

  compare_functor(column_device_view const& lhs_dev_view,
                  column_device_view const& rhs_dev_view,
                  CompareFunc cf,
                  bool is_lhs_scalar,
                  bool is_rhs_scalar)
    : lhs_dev_view_(lhs_dev_view),
      rhs_dev_view_(rhs_dev_view),
      cfunc_(cf),
      is_lhs_scalar_(is_lhs_scalar),
      is_rhs_scalar_(is_rhs_scalar)
  {
  }

  __device__ inline OutT operator()(cudf::size_type i) const
  {
    auto const lhs_index = is_lhs_scalar_ ? 0 : i;
    auto const rhs_index = is_rhs_scalar_ ? 0 : i;
    return cfunc_(
      lhs_dev_view_.is_valid(lhs_index),
      rhs_dev_view_.is_valid(rhs_index),
      lhs_dev_view_.is_valid(lhs_index)
        ? lhs_dev_view_.template element<cudf::string_view>(lhs_index)
        : cudf::string_view{},
      rhs_dev_view_.is_valid(rhs_index)
        ? rhs_dev_view_.template element<cudf::string_view>(rhs_index)
        : cudf::string_view{});
  }
};

// This functor performs null aware binop between two columns or a column and a scalar by
// iterating over them on the device
struct null_considering_binop {
  template <typename OutT, typename CompareFunc>
  void populate_out_col(column_device_view const& lhsv,
                        column_device_view const& rhsv,
                        cudf::size_type col_size,
                        cuda::stream_ref stream,
                        CompareFunc cfunc,
                        bool is_lhs_scalar,
                        bool is_rhs_scalar,
                        OutT* out_col) const
  {
    // Create binop functor instance
    compare_functor<OutT, CompareFunc> binop_func{
      lhsv, rhsv, cfunc, is_lhs_scalar, is_rhs_scalar};

    // Execute it on every element
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{col_size},
                      out_col,
                      binop_func);
  }

  // This is invoked to perform comparison between cudf string types
  std::unique_ptr<column> operator()(column_device_view const& lhs,
                                     column_device_view const& rhs,
                                     binary_operator op,
                                     data_type output_type,
                                     cudf::size_type col_size,
                                     bool is_lhs_scalar,
                                     bool is_rhs_scalar,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
  {
    // Validate input
    CUDF_EXPECTS(output_type.id() == lhs.type().id(),
                 "Output column type should match input column type");

    // Shallow copy of the resultant strings
    rmm::device_uvector<cudf::string_view> out_col_strings(col_size, stream);

    // Invalid output column strings - null rows
    cudf::string_view const invalid_str{nullptr, 0};

    // Create a compare function lambda
    auto minmax_func = cuda::proclaim_return_type<cudf::string_view>(
      [op, invalid_str] __device__(
        bool lhs_valid, bool rhs_valid, cudf::string_view lhs_value, cudf::string_view rhs_value) {
        if (!lhs_valid && !rhs_valid)
          return invalid_str;
        else if (lhs_valid && rhs_valid) {
          return (op == binary_operator::NULL_MAX)
                   ? cuda::maximum<cudf::string_view>()(lhs_value, rhs_value)
                   : cuda::minimum<cudf::string_view>()(lhs_value, rhs_value);
        } else if (lhs_valid)
          return lhs_value;
        else
          return rhs_value;
      });

    // Populate output column
    populate_out_col(lhs,
                     rhs,
                     col_size,
                     stream,
                     minmax_func,
                     is_lhs_scalar,
                     is_rhs_scalar,
                     out_col_strings.data());

    // Create an output column with the resultant strings
    return cudf::make_strings_column(out_col_strings, invalid_str, stream, mr);
  }
};

}  // namespace

std::unique_ptr<column> string_null_min_max(scalar const& lhs,
                                            column_view const& rhs,
                                            binary_operator op,
                                            data_type output_type,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(op == binary_operator::NULL_MAX or op == binary_operator::NULL_MIN,
               "Unsupported binary operation");
  if (rhs.is_empty()) return cudf::make_empty_column(output_type);
  auto lhs_view        = lhs.as_column_view();
  auto lhs_device_view = cudf::column_device_view::create(lhs_view.as_column_view(), stream);
  auto rhs_device_view = cudf::column_device_view::create(rhs, stream);
  return null_considering_binop{}(
    *lhs_device_view, *rhs_device_view, op, output_type, rhs.size(), true, false, stream, mr);
}

std::unique_ptr<column> string_null_min_max(column_view const& lhs,
                                            scalar const& rhs,
                                            binary_operator op,
                                            data_type output_type,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(op == binary_operator::NULL_MAX or op == binary_operator::NULL_MIN,
               "Unsupported binary operation");
  if (lhs.is_empty()) return cudf::make_empty_column(output_type);
  auto lhs_device_view = cudf::column_device_view::create(lhs, stream);
  auto rhs_view        = rhs.as_column_view();
  auto rhs_device_view = cudf::column_device_view::create(rhs_view.as_column_view(), stream);
  return null_considering_binop{}(
    *lhs_device_view, *rhs_device_view, op, output_type, lhs.size(), false, true, stream, mr);
}

std::unique_ptr<column> string_null_min_max(column_view const& lhs,
                                            column_view const& rhs,
                                            binary_operator op,
                                            data_type output_type,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(op == binary_operator::NULL_MAX or op == binary_operator::NULL_MIN,
               "Unsupported binary operation");
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Column sizes do not match");
  if (lhs.is_empty()) return cudf::make_empty_column(output_type);
  auto lhs_device_view = cudf::column_device_view::create(lhs, stream);
  auto rhs_device_view = cudf::column_device_view::create(rhs, stream);
  return null_considering_binop{}(
    *lhs_device_view, *rhs_device_view, op, output_type, lhs.size(), false, false, stream, mr);
}

void operator_dispatcher(mutable_column_view& out,
                         column_view const& lhs,
                         column_view const& rhs,
                         bool is_lhs_scalar,
                         bool is_rhs_scalar,
                         binary_operator op,
                         cuda::stream_ref stream)
{
  // clang-format off
switch (op) {
case binary_operator::ADD:                  apply_binary_op<ops::Add>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::SUB:                  apply_binary_op<ops::Sub>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::MUL:                  apply_binary_op<ops::Mul>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::DIV:                  apply_binary_op<ops::Div>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::TRUE_DIV:             apply_binary_op<ops::TrueDiv>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::FLOOR_DIV:            apply_binary_op<ops::FloorDiv>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::MOD:                  apply_binary_op<ops::Mod>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::PYMOD:                apply_binary_op<ops::PyMod>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::POW:                  apply_binary_op<ops::Pow>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::INT_POW:               apply_binary_op<ops::IntPow>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::EQUAL:
case binary_operator::NOT_EQUAL:
if(out.type().id() != type_id::BOOL8) CUDF_FAIL("Output type of Comparison operator should be bool type");
dispatch_equality_op(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, op, stream); break;
case binary_operator::LESS:                 apply_binary_op<ops::Less>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::GREATER:              apply_binary_op<ops::Greater>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::LESS_EQUAL:           apply_binary_op<ops::LessEqual>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::GREATER_EQUAL:        apply_binary_op<ops::GreaterEqual>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::BITWISE_AND:          apply_binary_op<ops::BitwiseAnd>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::BITWISE_OR:           apply_binary_op<ops::BitwiseOr>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::BITWISE_XOR:          apply_binary_op<ops::BitwiseXor>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::LOGICAL_AND:          apply_binary_op<ops::LogicalAnd>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::LOGICAL_OR:           apply_binary_op<ops::LogicalOr>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
/*
case binary_operator::GENERIC_BINARY:      // Cannot be compiled, should be called by jit::binary_operation
*/
case binary_operator::SHIFT_LEFT:           apply_binary_op<ops::ShiftLeft>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::SHIFT_RIGHT:          apply_binary_op<ops::ShiftRight>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::SHIFT_RIGHT_UNSIGNED: apply_binary_op<ops::ShiftRightUnsigned>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::LOG_BASE:             apply_binary_op<ops::LogBase>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::ATAN2:                apply_binary_op<ops::ATan2>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::PMOD:                 apply_binary_op<ops::PMod>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::NULL_EQUALS:          apply_binary_op<ops::NullEquals>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::NULL_NOT_EQUALS:      apply_binary_op<ops::NullNotEquals>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::NULL_MAX:             apply_binary_op<ops::NullMax>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::NULL_MIN:             apply_binary_op<ops::NullMin>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::NULL_LOGICAL_AND:     apply_binary_op<ops::NullLogicalAnd>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
case binary_operator::NULL_LOGICAL_OR:      apply_binary_op<ops::NullLogicalOr>(out, lhs, rhs, is_lhs_scalar, is_rhs_scalar, stream); break;
default:;
}
  // clang-format on
}

// vector_vector
void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      column_view const& rhs,
                      binary_operator op,
                      cuda::stream_ref stream)
{
  operator_dispatcher(out, lhs, rhs, false, false, op, stream);
}
// scalar_vector
void binary_operation(mutable_column_view& out,
                      scalar const& lhs,
                      column_view const& rhs,
                      binary_operator op,
                      cuda::stream_ref stream)
{
  auto const lhs_view = lhs.as_column_view();
  operator_dispatcher(out, lhs_view.as_column_view(), rhs, true, false, op, stream);
}
// vector_scalar
void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      scalar const& rhs,
                      binary_operator op,
                      cuda::stream_ref stream)
{
  auto const rhs_view = rhs.as_column_view();
  operator_dispatcher(out, lhs, rhs_view.as_column_view(), false, true, op, stream);
}

namespace detail {
void apply_sorting_struct_binary_op(mutable_column_view& out,
                                    column_view const& lhs,
                                    column_view const& rhs,
                                    bool is_lhs_scalar,
                                    bool is_rhs_scalar,
                                    binary_operator op,
                                    cuda::stream_ref stream)
{
  CUDF_EXPECTS(lhs.type().id() == type_id::STRUCT && rhs.type().id() == type_id::STRUCT,
               "Both columns must be struct columns");
  CUDF_EXPECTS(!cudf::structs::detail::is_or_has_nested_lists(lhs) and
                 !cudf::structs::detail::is_or_has_nested_lists(rhs),
               "List type is not supported");
  // Struct child column type and structure mismatches are caught within the two_table_comparator
  switch (op) {
    case binary_operator::EQUAL: [[fallthrough]];
    case binary_operator::NOT_EQUAL: [[fallthrough]];
    case binary_operator::NULL_EQUALS: [[fallthrough]];
    case binary_operator::NULL_NOT_EQUALS:
      detail::apply_struct_equality_op(
        out,
        lhs,
        rhs,
        is_lhs_scalar,
        is_rhs_scalar,
        op,
        cudf::detail::row::equality::nan_equal_physical_equality_comparator{},
        stream);
      break;
    case binary_operator::LESS:
      detail::apply_struct_binary_op<ops::Less>(
        out,
        lhs,
        rhs,
        is_lhs_scalar,
        is_rhs_scalar,
        cudf::detail::row::lexicographic::sorting_physical_element_comparator{},
        stream);
      break;
    case binary_operator::GREATER:
      detail::apply_struct_binary_op<ops::Greater>(
        out,
        lhs,
        rhs,
        is_lhs_scalar,
        is_rhs_scalar,
        cudf::detail::row::lexicographic::sorting_physical_element_comparator{},
        stream);
      break;
    case binary_operator::LESS_EQUAL:
      detail::apply_struct_binary_op<ops::LessEqual>(
        out,
        lhs,
        rhs,
        is_lhs_scalar,
        is_rhs_scalar,
        cudf::detail::row::lexicographic::sorting_physical_element_comparator{},
        stream);
      break;
    case binary_operator::GREATER_EQUAL:
      detail::apply_struct_binary_op<ops::GreaterEqual>(
        out,
        lhs,
        rhs,
        is_lhs_scalar,
        is_rhs_scalar,
        cudf::detail::row::lexicographic::sorting_physical_element_comparator{},
        stream);
      break;
    default: CUDF_FAIL("Unsupported operator for structs");
  }
}
}  // namespace detail
}  // namespace compiled
}  // namespace binops
}  // namespace cudf

/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "binary_ops.hpp"
#include "operation.cuh"
#include "struct_binary_ops.cuh"

#include <cudf/binaryop.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/structs/utilities.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/transform.h>

namespace cudf {
namespace binops {
namespace compiled {

namespace {
/**
 * @brief Converts scalar to column_view with single element.
 *
 * @return pair with column_view and column containing any auxiliary data to create column_view from
 * scalar
 */
struct scalar_as_column_view {
  using return_type = typename std::pair<column_view, std::unique_ptr<column>>;
  template <typename T, CUDF_ENABLE_IF(is_fixed_width<T>())>
  return_type operator()(scalar const& s,
                         rmm::cuda_stream_view stream,
                         rmm::device_async_resource_ref mr)
  {
    auto& h_scalar_type_view = static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(s));

    // Valid scalar needs no null mask
    if (s.is_valid(stream)) {
      auto col_v = column_view(s.type(), 1, h_scalar_type_view.data(), nullptr, 0);
      return std::pair{col_v, nullptr};
    }

    // Null scalar needs a single-element null mask (kept alive by an auxiliary column) as its
    // validity (bool) cannot be reinterpreted as `bitmask_type` without reading out of bounds.
    auto null_mask = cudf::detail::create_null_mask(1, cudf::mask_state::ALL_NULL, stream, mr);
    auto const* null_mask_ptr = static_cast<bitmask_type const*>(null_mask.data());
    auto aux_col              = std::make_unique<column>(
      data_type{type_id::INT8}, 0, rmm::device_buffer{}, std::move(null_mask), 1);
    auto col_v = column_view(s.type(), 1, h_scalar_type_view.data(), null_mask_ptr, 1);
    return std::pair{col_v, std::move(aux_col)};
  }
  template <typename T, CUDF_ENABLE_IF(!is_fixed_width<T>())>
  return_type operator()(scalar const&, rmm::cuda_stream_view, rmm::device_async_resource_ref)
  {
    CUDF_FAIL("Unsupported type");
  }
};
// specialization for cudf::string_view
template <>
scalar_as_column_view::return_type scalar_as_column_view::operator()<cudf::string_view>(
  scalar const& s, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  using T                  = cudf::string_view;
  auto& h_scalar_type_view = static_cast<cudf::scalar_type_t<T>&>(const_cast<scalar&>(s));

  // build offsets column from the string size
  auto offsets_transformer_itr = cuda::make_constant_iterator<size_type>(h_scalar_type_view.size());
  auto offsets_column          = std::get<0>(cudf::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + 1, stream, mr));

  // Valid scalar needs no null mask. The offsets child column is kept alive to back the returned
  // string column_view.
  if (s.is_valid(stream)) {
    auto col_v = cudf::column_view(
      s.type(), 1, h_scalar_type_view.data(), nullptr, 0, 0, {offsets_column->view()});
    return std::pair{col_v, std::move(offsets_column)};
  }

  // Null scalar needs a single-element null mask and offsets (kepy alive by an auxiliary column) as
  // its validity (bool) cannot be reinterpreted as `bitmask_type` without reading out of bounds.
  auto null_mask = cudf::detail::create_null_mask(1, cudf::mask_state::ALL_NULL, stream, mr);
  auto const* null_mask_ptr = static_cast<bitmask_type const*>(null_mask.data());
  auto col_v                = cudf::column_view(
    s.type(), 1, h_scalar_type_view.data(), null_mask_ptr, 1, 0, {offsets_column->view()});
  std::vector<std::unique_ptr<column>> children;
  children.push_back(std::move(offsets_column));
  auto aux_col = std::make_unique<column>(data_type{type_id::INT8},
                                          0,
                                          rmm::device_buffer{},
                                          std::move(null_mask),
                                          1,
                                          std::move(children));
  return std::pair{col_v, std::move(aux_col)};
}

// specializing for struct column
template <>
scalar_as_column_view::return_type scalar_as_column_view::operator()<cudf::struct_view>(
  scalar const& s, rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr)
{
  auto col = make_column_from_scalar(s, 1, stream, mr);
  return std::pair{col->view(), std::move(col)};
}

/**
 * @brief Converts scalar to column_view with single element.
 *
 * @param scal    scalar to convert
 * @param stream  CUDA stream used for device memory operations and kernel launches.
 * @param mr      Device memory resource used to allocate the returned column's device memory
 * @return        pair with column_view and column containing any auxiliary data to create
 * column_view from scalar
 */
auto scalar_to_column_view(
  scalar const& scal,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref())
{
  return type_dispatcher(scal.type(), scalar_as_column_view{}, scal, stream, mr);
}

// This functor does the actual comparison between string column value and a scalar string
// or between two string column values using a comparator
template <typename LhsDeviceViewT, typename RhsDeviceViewT, typename OutT, typename CompareFunc>
struct compare_functor {
  LhsDeviceViewT const lhs_dev_view_;  // Scalar or a column device view - lhs
  RhsDeviceViewT const rhs_dev_view_;  // Scalar or a column device view - rhs
  CompareFunc const cfunc_;            // Comparison function

  compare_functor(LhsDeviceViewT const& lhs_dev_view,
                  RhsDeviceViewT const& rhs_dev_view,
                  CompareFunc cf)
    : lhs_dev_view_(lhs_dev_view), rhs_dev_view_(rhs_dev_view), cfunc_(cf)
  {
  }

  // This is used to compare a scalar and a column value
  template <typename LhsViewT = LhsDeviceViewT, typename RhsViewT = RhsDeviceViewT>
  __device__ inline OutT operator()(cudf::size_type i) const
    requires(std::is_same_v<LhsViewT, column_device_view> &&
             !std::is_same_v<RhsViewT, column_device_view>)
  {
    return cfunc_(lhs_dev_view_.is_valid(i),
                  rhs_dev_view_.is_valid(),
                  lhs_dev_view_.is_valid(i) ? lhs_dev_view_.template element<cudf::string_view>(i)
                                            : cudf::string_view{},
                  rhs_dev_view_.is_valid() ? rhs_dev_view_.value() : cudf::string_view{});
  }

  // This is used to compare a scalar and a column value
  template <typename LhsViewT = LhsDeviceViewT, typename RhsViewT = RhsDeviceViewT>
  __device__ inline OutT operator()(cudf::size_type i) const
    requires(!std::is_same_v<LhsViewT, column_device_view> &&
             std::is_same_v<RhsViewT, column_device_view>)
  {
    return cfunc_(lhs_dev_view_.is_valid(),
                  rhs_dev_view_.is_valid(i),
                  lhs_dev_view_.is_valid() ? lhs_dev_view_.value() : cudf::string_view{},
                  rhs_dev_view_.is_valid(i) ? rhs_dev_view_.template element<cudf::string_view>(i)
                                            : cudf::string_view{});
  }

  // This is used to compare 2 column values
  template <typename LhsViewT = LhsDeviceViewT, typename RhsViewT = RhsDeviceViewT>
  __device__ inline OutT operator()(cudf::size_type i) const
    requires(std::is_same_v<LhsViewT, column_device_view> &&
             std::is_same_v<RhsViewT, column_device_view>)
  {
    return cfunc_(lhs_dev_view_.is_valid(i),
                  rhs_dev_view_.is_valid(i),
                  lhs_dev_view_.is_valid(i) ? lhs_dev_view_.template element<cudf::string_view>(i)
                                            : cudf::string_view{},
                  rhs_dev_view_.is_valid(i) ? rhs_dev_view_.template element<cudf::string_view>(i)
                                            : cudf::string_view{});
  }
};

// This functor performs null aware binop between two columns or a column and a scalar by
// iterating over them on the device
struct null_considering_binop {
  [[nodiscard]] auto get_device_view(cudf::scalar const& scalar_item) const
  {
    return get_scalar_device_view(
      static_cast<cudf::scalar_type_t<cudf::string_view>&>(const_cast<scalar&>(scalar_item)));
  }

  [[nodiscard]] auto get_device_view(column_device_view const& col_item) const { return col_item; }

  template <typename LhsViewT, typename RhsViewT, typename OutT, typename CompareFunc>
  void populate_out_col(LhsViewT const& lhsv,
                        RhsViewT const& rhsv,
                        cudf::size_type col_size,
                        rmm::cuda_stream_view stream,
                        CompareFunc cfunc,
                        OutT* out_col) const
  {
    // Create binop functor instance
    compare_functor<LhsViewT, RhsViewT, OutT, CompareFunc> binop_func{lhsv, rhsv, cfunc};

    // Execute it on every element
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{col_size},
                      out_col,
                      binop_func);
  }

  // This is invoked to perform comparison between cudf string types
  template <typename LhsT, typename RhsT>
  std::unique_ptr<column> operator()(LhsT const& lhs,
                                     RhsT const& rhs,
                                     binary_operator op,
                                     data_type output_type,
                                     cudf::size_type col_size,
                                     rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
  {
    // Create device views for inputs
    auto const lhs_dev_view = get_device_view(lhs);
    auto const rhs_dev_view = get_device_view(rhs);
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
    populate_out_col(
      lhs_dev_view, rhs_dev_view, col_size, stream, minmax_func, out_col_strings.data());

    // Create an output column with the resultant strings
    return cudf::make_strings_column(out_col_strings, invalid_str, stream, mr);
  }
};

}  // namespace

std::unique_ptr<column> string_null_min_max(scalar const& lhs,
                                            column_view const& rhs,
                                            binary_operator op,
                                            data_type output_type,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(op == binary_operator::NULL_MAX or op == binary_operator::NULL_MIN,
               "Unsupported binary operation");
  if (rhs.is_empty()) return cudf::make_empty_column(output_type);
  auto rhs_device_view = cudf::column_device_view::create(rhs, stream);
  return null_considering_binop{}(lhs, *rhs_device_view, op, output_type, rhs.size(), stream, mr);
}

std::unique_ptr<column> string_null_min_max(column_view const& lhs,
                                            scalar const& rhs,
                                            binary_operator op,
                                            data_type output_type,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  // hard-coded to only work with cudf::string_view so we don't explode compile times
  CUDF_EXPECTS(lhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported lhs datatype");
  CUDF_EXPECTS(rhs.type().id() == cudf::type_id::STRING, "Invalid/Unsupported rhs datatype");
  CUDF_EXPECTS(op == binary_operator::NULL_MAX or op == binary_operator::NULL_MIN,
               "Unsupported binary operation");
  if (lhs.is_empty()) return cudf::make_empty_column(output_type);
  auto lhs_device_view = cudf::column_device_view::create(lhs, stream);
  return null_considering_binop{}(*lhs_device_view, rhs, op, output_type, lhs.size(), stream, mr);
}

std::unique_ptr<column> string_null_min_max(column_view const& lhs,
                                            column_view const& rhs,
                                            binary_operator op,
                                            data_type output_type,
                                            rmm::cuda_stream_view stream,
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
    *lhs_device_view, *rhs_device_view, op, output_type, lhs.size(), stream, mr);
}

void operator_dispatcher(mutable_column_view& out,
                         column_view const& lhs,
                         column_view const& rhs,
                         bool is_lhs_scalar,
                         bool is_rhs_scalar,
                         binary_operator op,
                         rmm::cuda_stream_view stream)
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
                      rmm::cuda_stream_view stream)
{
  operator_dispatcher(out, lhs, rhs, false, false, op, stream);
}
// scalar_vector
void binary_operation(mutable_column_view& out,
                      scalar const& lhs,
                      column_view const& rhs,
                      binary_operator op,
                      rmm::cuda_stream_view stream)
{
  auto [lhsv, aux] = scalar_to_column_view(lhs, stream);
  operator_dispatcher(out, lhsv, rhs, true, false, op, stream);
}
// vector_scalar
void binary_operation(mutable_column_view& out,
                      column_view const& lhs,
                      scalar const& rhs,
                      binary_operator op,
                      rmm::cuda_stream_view stream)
{
  auto [rhsv, aux] = scalar_to_column_view(rhs, stream);
  operator_dispatcher(out, lhs, rhsv, false, true, op, stream);
}

namespace detail {
void apply_sorting_struct_binary_op(mutable_column_view& out,
                                    column_view const& lhs,
                                    column_view const& rhs,
                                    bool is_lhs_scalar,
                                    bool is_rhs_scalar,
                                    binary_operator op,
                                    rmm::cuda_stream_view stream)
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

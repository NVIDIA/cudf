/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "checked_arithmetic.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/operators/checked_arithmetic.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/std/expected>
#include <cuda/std/optional>
#include <cuda/std/tuple>

#include <jit/column_accessor.cuh>
#include <jit/transform_kernel.cuh>
#include <jit/type_list.cuh>

#include <string>
#include <type_traits>

namespace cudf::detail::checked_arithmetic {
namespace {

template <typename T>
inline constexpr bool is_supported_type =
  (is_numeric<T>() && !std::is_same_v<T, bool>) || is_fixed_point<T>();

struct checked_binary_row_operation {
  binary_operator op;

  template <typename T>
  __device__ static errc assign(cuda::std::optional<T>* out, cuda::std::expected<T, errc> result)
  {
    if (!result.has_value()) {
      *out = cuda::std::nullopt;
      return result.error();
    }
    *out = result.value();
    return errc::SUCCESS;
  }

  template <typename Args>
  __device__ errc operator()(size_type, Args args) const
  {
    auto* out      = cuda::std::get<0>(args);
    auto const lhs = cuda::std::get<1>(args);
    auto const rhs = cuda::std::get<2>(args);

    if (!lhs.has_value() || !rhs.has_value()) {
      *out = cuda::std::nullopt;
      return errc::SUCCESS;
    }

    switch (op) {
      case binary_operator::ADD_OVERFLOW:
        return assign(out, cudf::detail::ops::add_overflow(*lhs, *rhs));
      case binary_operator::SUB_OVERFLOW:
        return assign(out, cudf::detail::ops::sub_overflow(*lhs, *rhs));
      case binary_operator::MUL_OVERFLOW:
        return assign(out, cudf::detail::ops::mul_overflow(*lhs, *rhs));
      case binary_operator::DIV_OVERFLOW:
        return assign(out, cudf::detail::ops::div_overflow(*lhs, *rhs));
      case binary_operator::MOD_OVERFLOW:
        return assign(out, cudf::detail::ops::mod_overflow(*lhs, *rhs));
      default: return errc::SUCCESS;
    }
  }
};

struct checked_unary_row_operation {
  unary_operator op;

  template <typename T>
  __device__ static errc assign(cuda::std::optional<T>* out, cuda::std::expected<T, errc> result)
  {
    if (!result.has_value()) {
      *out = cuda::std::nullopt;
      return result.error();
    }
    *out = result.value();
    return errc::SUCCESS;
  }

  template <typename Args>
  __device__ errc operator()(size_type, Args args) const
  {
    auto* out        = cuda::std::get<0>(args);
    auto const input = cuda::std::get<1>(args);

    if (!input.has_value()) {
      *out = cuda::std::nullopt;
      return errc::SUCCESS;
    }

    switch (op) {
      case unary_operator::NEG_OVERFLOW:
        return assign(out, cudf::detail::ops::neg_overflow(*input));
      case unary_operator::ABS_OVERFLOW:
        return assign(out, cudf::detail::ops::abs_overflow(*input));
      default: return errc::SUCCESS;
    }
  }
};

template <typename T, bool LhsIsScalar, bool RhsIsScalar>
CUDF_KERNEL void checked_binary_kernel(size_type row_size,
                                       column_device_view lhs,
                                       column_device_view rhs,
                                       mutable_column_device_view out,
                                       binary_operator op,
                                       int32_t* max_error)
{
  using input_accessors =
    jit::type_list<jit::column_accessor<0, column_device_view_core, T, LhsIsScalar, 0>,
                   jit::column_accessor<1, column_device_view_core, T, RhsIsScalar, 0>>;
  using output_accessors =
    jit::type_list<jit::column_accessor<0, mutable_column_device_view_core, T, false, 0>>;

  column_device_view_core const inputs[]          = {lhs, rhs};
  mutable_column_device_view_core const outputs[] = {out};
  cudf::detail::transform_kernel<true, input_accessors, output_accessors>(
    row_size, nullptr, inputs, outputs, max_error, checked_binary_row_operation{op});
}

template <typename T>
CUDF_KERNEL void checked_unary_kernel(size_type row_size,
                                      column_device_view input,
                                      mutable_column_device_view out,
                                      unary_operator op,
                                      int32_t* max_error)
{
  using input_accessors =
    jit::type_list<jit::column_accessor<0, column_device_view_core, T, false, 0>>;
  using output_accessors =
    jit::type_list<jit::column_accessor<0, mutable_column_device_view_core, T, false, 0>>;

  column_device_view_core const inputs[]          = {input};
  mutable_column_device_view_core const outputs[] = {out};
  cudf::detail::transform_kernel<true, input_accessors, output_accessors>(
    row_size, nullptr, inputs, outputs, max_error, checked_unary_row_operation{op});
}

void throw_if_error(errc error, error_policy policy)
{
  if (error == errc::SUCCESS || policy == error_policy::NULLIFY) { return; }
  throw evaluation_error(
    error,
    std::string{"Checked arithmetic evaluation failed with error `"} + to_string(error) + "`");
}

template <bool LhsIsScalar, bool RhsIsScalar>
struct binary_launcher {
  template <typename T>
  void operator()(column_view const& lhs,
                  column_view const& rhs,
                  mutable_column_view& out,
                  binary_operator op,
                  error_policy policy,
                  cuda::stream_ref stream) const
  {
    if constexpr (is_supported_type<T>) {
      auto lhs_device = column_device_view::create(lhs, stream);
      auto rhs_device = column_device_view::create(rhs, stream);
      auto out_device = mutable_column_device_view::create(out, stream);
      cudf::detail::device_scalar<int32_t> max_error{
        static_cast<int32_t>(errc::SUCCESS), stream, cudf::get_current_device_resource_ref()};

      cudf::detail::grid_1d config(out.size(), 256);
      checked_binary_kernel<T, LhsIsScalar, RhsIsScalar>
        <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
          out.size(), *lhs_device, *rhs_device, *out_device, op, max_error.data());
      CUDF_CHECK_CUDA(stream.get());

      throw_if_error(static_cast<errc>(max_error.value(stream)), policy);
    } else {
      CUDF_FAIL("Checked arithmetic requires matching arithmetic or fixed-point types",
                cudf::data_type_error);
    }
  }
};

struct unary_launcher {
  template <typename T>
  void operator()(column_view const& input,
                  mutable_column_view& out,
                  unary_operator op,
                  error_policy policy,
                  cuda::stream_ref stream) const
  {
    if constexpr (is_supported_type<T>) {
      auto input_device = column_device_view::create(input, stream);
      auto out_device   = mutable_column_device_view::create(out, stream);
      cudf::detail::device_scalar<int32_t> max_error{
        static_cast<int32_t>(errc::SUCCESS), stream, cudf::get_current_device_resource_ref()};

      cudf::detail::grid_1d config(out.size(), 256);
      checked_unary_kernel<T><<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
        out.size(), *input_device, *out_device, op, max_error.data());
      CUDF_CHECK_CUDA(stream.get());

      throw_if_error(static_cast<errc>(max_error.value(stream)), policy);
    } else {
      CUDF_FAIL("Checked arithmetic requires an arithmetic or fixed-point type",
                cudf::data_type_error);
    }
  }
};

void validate_binary(column_view const& lhs,
                     column_view const& rhs,
                     binary_operator op,
                     data_type output_type)
{
  CUDF_EXPECTS(is_checked(op),
               "Error policies are only supported for checked arithmetic operators");
  CUDF_EXPECTS(lhs.type().id() == rhs.type().id() && output_type.id() == lhs.type().id(),
               "Checked arithmetic requires matching input and output storage types",
               cudf::data_type_error);
  CUDF_EXPECTS(
    (is_numeric(lhs.type()) && lhs.type().id() != type_id::BOOL8) || is_fixed_point(lhs.type()),
    "Checked arithmetic requires arithmetic or fixed-point inputs",
    cudf::data_type_error);

  if (is_fixed_point(lhs.type())) {
    auto const expected_scale =
      cudf::binary_operation_fixed_point_scale(op, lhs.type().scale(), rhs.type().scale());
    CUDF_EXPECTS(output_type.scale() == expected_scale,
                 "Checked fixed-point output has an invalid scale",
                 cudf::data_type_error);
  } else {
    CUDF_EXPECTS(lhs.type() == rhs.type() && output_type == lhs.type(),
                 "Checked arithmetic requires identical input and output types",
                 cudf::data_type_error);
  }
}

template <bool LhsIsScalar, bool RhsIsScalar>
std::unique_ptr<column> binary_operation_impl(column_view const& lhs,
                                              column_view const& rhs,
                                              size_type size,
                                              binary_operator op,
                                              data_type output_type,
                                              error_policy policy,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  validate_binary(lhs, rhs, op, output_type);

  auto result = make_fixed_width_column(output_type, size, mask_state::ALL_VALID, stream, mr);
  if (size == 0) { return result; }

  auto result_view = result->mutable_view();
  cudf::type_dispatcher(lhs.type(),
                        binary_launcher<LhsIsScalar, RhsIsScalar>{},
                        lhs,
                        rhs,
                        result_view,
                        op,
                        policy,
                        stream);
  result->set_null_count(
    cudf::detail::null_count(result_view.null_mask(), 0, result_view.size(), stream));
  return result;
}

}  // namespace

std::unique_ptr<column> binary_operation(scalar const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         error_policy policy,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  auto lhs_column =
    make_column_from_scalar(lhs, 1, stream, cudf::get_current_device_resource_ref());
  return binary_operation_impl<true, false>(
    lhs_column->view(), rhs, rhs.size(), op, output_type, policy, stream, mr);
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         scalar const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         error_policy policy,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  auto rhs_column =
    make_column_from_scalar(rhs, 1, stream, cudf::get_current_device_resource_ref());
  return binary_operation_impl<false, true>(
    lhs, rhs_column->view(), lhs.size(), op, output_type, policy, stream, mr);
}

std::unique_ptr<column> binary_operation(column_view const& lhs,
                                         column_view const& rhs,
                                         binary_operator op,
                                         data_type output_type,
                                         error_policy policy,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Column sizes do not match", std::invalid_argument);
  return binary_operation_impl<false, false>(
    lhs, rhs, lhs.size(), op, output_type, policy, stream, mr);
}

std::unique_ptr<column> unary_operation(column_view const& input,
                                        unary_operator op,
                                        error_policy policy,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(is_checked(op),
               "Error policies are only supported for checked arithmetic operators");
  CUDF_EXPECTS((is_numeric(input.type()) && input.type().id() != type_id::BOOL8) ||
                 is_fixed_point(input.type()),
               "Checked arithmetic requires an arithmetic or fixed-point input",
               cudf::data_type_error);

  auto result =
    make_fixed_width_column(input.type(), input.size(), mask_state::ALL_VALID, stream, mr);
  if (input.is_empty()) { return result; }

  auto result_view = result->mutable_view();
  cudf::type_dispatcher(input.type(), unary_launcher{}, input, result_view, op, policy, stream);
  result->set_null_count(
    cudf::detail::null_count(result_view.null_mask(), 0, result_view.size(), stream));
  return result;
}

}  // namespace cudf::detail::checked_arithmetic

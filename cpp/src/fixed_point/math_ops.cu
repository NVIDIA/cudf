/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "math_ops.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/fixed_point/math_ops.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

#include <memory>

namespace cudf {
namespace detail {
namespace {

template <typename T>
struct divide_decimal_fn {
  using Rep = typename T::rep;

  numeric::scale_type lhs_scale;
  numeric::scale_type rhs_scale;
  rounding_method method;

  __device__ Rep operator()(Rep lhs, Rep rhs) const
  {
    return divide_decimal(T{numeric::scaled_integer<Rep>{lhs, lhs_scale}},
                          T{numeric::scaled_integer<Rep>{rhs, rhs_scale}},
                          method)
      .value();
  }
};

/**
 * @brief Writes `lhs[i] / rhs[i]` into `output[i]` for every row of `output`.
 *
 * Column operands are passed as column iterators and scalar operands as constant iterators, so
 * all three public overloads share this transform.
 */
template <typename T, typename LhsIterator, typename RhsIterator>
void divide_decimal_transform(LhsIterator lhs,
                              RhsIterator rhs,
                              numeric::scale_type lhs_scale,
                              numeric::scale_type rhs_scale,
                              rounding_method method,
                              mutable_column_view output,
                              cuda::stream_ref stream)
{
  using Rep = typename T::rep;
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    lhs,
                    lhs + output.size(),
                    rhs,
                    output.begin<Rep>(),
                    divide_decimal_fn<T>{lhs_scale, rhs_scale, method});
}

/**
 * @brief Allocates the output of a column-scalar division, with nulls wherever `input` is null,
 * or everywhere if the scalar is invalid.
 */
std::unique_ptr<column> make_output_column(data_type type,
                                           column_view const& input,
                                           bool scalar_is_valid,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  auto const size = input.size();
  if (!scalar_is_valid) {
    return make_fixed_width_column(
      type,
      size,
      cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr),
      size,
      stream,
      mr);
  }
  if (input.has_nulls()) {
    return make_fixed_width_column(
      type, size, cudf::detail::copy_bitmask(input, stream, mr), input.null_count(), stream, mr);
  }
  return make_fixed_width_column(type, size, mask_state::UNALLOCATED, stream, mr);
}

struct divide_decimal_dispatch {
  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...) const
    requires(not cudf::is_fixed_point<T>())
  {
    CUDF_FAIL("divide_decimal only supports decimal types");
  }

  template <typename T>
  std::unique_ptr<column> operator()(column_view const& lhs,
                                     column_view const& rhs,
                                     rounding_method method,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_fixed_point<T>())
  {
    using Rep = typename T::rep;

    std::unique_ptr<column> result;
    if (lhs.has_nulls() || rhs.has_nulls()) {
      auto [null_mask, null_count] = cudf::detail::bitmask_and(table_view{{lhs, rhs}}, stream, mr);
      result                       = make_fixed_width_column(
        lhs.type(), lhs.size(), std::move(null_mask), null_count, stream, mr);
    } else {
      result = make_fixed_width_column(lhs.type(), lhs.size(), mask_state::UNALLOCATED, stream, mr);
    }

    divide_decimal_transform<T>(lhs.begin<Rep>(),
                                rhs.begin<Rep>(),
                                numeric::scale_type{lhs.type().scale()},
                                numeric::scale_type{rhs.type().scale()},
                                method,
                                result->mutable_view(),
                                stream);
    return result;
  }

  template <typename T>
  std::unique_ptr<column> operator()(column_view const& lhs,
                                     scalar const& rhs,
                                     rounding_method method,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_fixed_point<T>())
  {
    auto const rhs_value = static_cast<fixed_point_scalar<T> const&>(rhs).value(stream);
    auto result          = make_output_column(lhs.type(), lhs, rhs.is_valid(stream), stream, mr);

    divide_decimal_transform<T>(lhs.begin<typename T::rep>(),
                                cuda::make_constant_iterator(rhs_value),
                                numeric::scale_type{lhs.type().scale()},
                                numeric::scale_type{rhs.type().scale()},
                                method,
                                result->mutable_view(),
                                stream);
    return result;
  }

  template <typename T>
  std::unique_ptr<column> operator()(scalar const& lhs,
                                     column_view const& rhs,
                                     rounding_method method,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr) const
    requires(cudf::is_fixed_point<T>())
  {
    auto const lhs_value = static_cast<fixed_point_scalar<T> const&>(lhs).value(stream);
    auto result          = make_output_column(rhs.type(), rhs, lhs.is_valid(stream), stream, mr);

    divide_decimal_transform<T>(cuda::make_constant_iterator(lhs_value),
                                rhs.begin<typename T::rep>(),
                                numeric::scale_type{lhs.type().scale()},
                                numeric::scale_type{rhs.type().scale()},
                                method,
                                result->mutable_view(),
                                stream);
    return result;
  }
};

}  // namespace

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       column_view const& rhs,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Column sizes must match");
  CUDF_EXPECTS(lhs.type().id() == rhs.type().id(), "Column base types must match");
  CUDF_EXPECTS(cudf::is_fixed_point(lhs.type()), "Columns must be decimal type");

  if (lhs.is_empty()) { return make_empty_column(lhs.type()); }

  return type_dispatcher(lhs.type(), divide_decimal_dispatch{}, lhs, rhs, method, stream, mr);
}

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       scalar const& rhs,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::is_fixed_point(lhs.type()), "Column must be decimal type");
  CUDF_EXPECTS(rhs.type() == lhs.type(), "Scalar type must match column type");

  if (lhs.is_empty()) { return make_empty_column(lhs.type()); }

  return type_dispatcher(lhs.type(), divide_decimal_dispatch{}, lhs, rhs, method, stream, mr);
}

std::unique_ptr<column> divide_decimal(scalar const& lhs,
                                       column_view const& rhs,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(cudf::is_fixed_point(rhs.type()), "Column must be decimal type");
  CUDF_EXPECTS(lhs.type() == rhs.type(), "Scalar type must match column type");

  if (rhs.is_empty()) { return make_empty_column(rhs.type()); }

  return type_dispatcher(rhs.type(), divide_decimal_dispatch{}, lhs, rhs, method, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       column_view const& rhs,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::divide_decimal(lhs, rhs, method, stream, mr);
}

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       scalar const& rhs,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::divide_decimal(lhs, rhs, method, stream, mr);
}

std::unique_ptr<column> divide_decimal(scalar const& lhs,
                                       column_view const& rhs,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::divide_decimal(lhs, rhs, method, stream, mr);
}

}  // namespace cudf

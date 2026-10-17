/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "math_ops.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/fixed_point/math_ops.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/optional>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <memory>

namespace cudf {
namespace detail {
namespace {

template <typename T>
struct column_operand {
  column_device_view column;

  __device__ bool is_valid(size_type i) const { return column.is_valid(i); }
  __device__ T value(size_type i) const { return column.element<T>(i); }
};

template <typename T>
struct scalar_operand {
  fixed_point_scalar_device_view<T> scalar;

  __device__ bool is_valid(size_type) const { return scalar.is_valid(); }
  __device__ T value(size_type) const { return scalar.value(); }
};

template <typename T>
scalar_operand<T> make_scalar_operand(scalar const& s)
{
  return scalar_operand<T>{
    get_scalar_device_view(static_cast<fixed_point_scalar<T>&>(const_cast<scalar&>(s)))};
}

template <typename T, typename LhsOperand, typename RhsOperand>
struct divide_decimal_fn {
  using Rep = typename T::rep;

  LhsOperand lhs;
  RhsOperand rhs;
  rounding_method method;
  Rep* output;
  bool* output_valid;

  __device__ void operator()(size_type i) const
  {
    cuda::std::optional<T> quotient;
    if (lhs.is_valid(i) && rhs.is_valid(i)) {
      quotient = divide_decimal(lhs.value(i), rhs.value(i), method);
    }
    output[i]       = quotient.has_value() ? quotient->value() : Rep{0};
    output_valid[i] = quotient.has_value();
  }
};

/**
 * @brief Divides `lhs` by `rhs` row by row into a new column of type `output_type`.
 *
 * A row is null if either operand is null or the division has no result.
 */
template <typename T, typename LhsOperand, typename RhsOperand>
std::unique_ptr<column> divide_decimal_rows(LhsOperand lhs,
                                            RhsOperand rhs,
                                            data_type output_type,
                                            size_type size,
                                            rounding_method method,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  using Rep = typename T::rep;

  auto result =
    make_fixed_width_column(output_type, size, mask_state::UNALLOCATED, stream, mr.get_output_mr());
  auto valid = cuda::device_buffer<bool>(stream, mr.get_temporary_mr(), size, cuda::no_init);

  thrust::for_each_n(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                     cuda::counting_iterator<size_type>{0},
                     size,
                     divide_decimal_fn<T, LhsOperand, RhsOperand>{
                       lhs, rhs, method, result->mutable_view().begin<Rep>(), valid.data()});

  auto [null_mask, null_count] = cudf::detail::bools_to_mask(
    column_view{data_type{type_id::BOOL8}, size, valid.data(), nullptr, 0},
    stream,
    mr.get_output_mr());
  if (null_count > 0) { result->set_null_mask(std::move(*null_mask), null_count); }
  return result;
}

struct divide_decimal_dispatch {
  template <typename T, typename... Args>
  std::unique_ptr<column> operator()(Args&&...) const
    requires(not cudf::concepts::fixed_point<T>)
  {
    CUDF_FAIL("divide_decimal only supports decimal types");
  }

  template <cudf::concepts::fixed_point T>
  std::unique_ptr<column> operator()(column_view const& lhs,
                                     column_view const& rhs,
                                     rounding_method method,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
  {
    auto const d_lhs = column_device_view::create(lhs, stream);
    auto const d_rhs = column_device_view::create(rhs, stream);

    return divide_decimal_rows<T>(column_operand<T>{*d_lhs},
                                  column_operand<T>{*d_rhs},
                                  lhs.type(),
                                  lhs.size(),
                                  method,
                                  stream,
                                  mr);
  }

  template <cudf::concepts::fixed_point T>
  std::unique_ptr<column> operator()(column_view const& lhs,
                                     scalar const& rhs,
                                     rounding_method method,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
  {
    auto const d_lhs = column_device_view::create(lhs, stream);

    return divide_decimal_rows<T>(column_operand<T>{*d_lhs},
                                  make_scalar_operand<T>(rhs),
                                  lhs.type(),
                                  lhs.size(),
                                  method,
                                  stream,
                                  mr);
  }

  template <cudf::concepts::fixed_point T>
  std::unique_ptr<column> operator()(scalar const& lhs,
                                     column_view const& rhs,
                                     rounding_method method,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr) const
  {
    auto const d_rhs = column_device_view::create(rhs, stream);

    return divide_decimal_rows<T>(make_scalar_operand<T>(lhs),
                                  column_operand<T>{*d_rhs},
                                  lhs.type(),
                                  rhs.size(),
                                  method,
                                  stream,
                                  mr);
  }
};

}  // namespace

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       column_view const& rhs,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_EXPECTS(cudf::is_fixed_point(lhs.type()) && cudf::is_fixed_point(rhs.type()),
               "Columns must be decimal type");
  CUDF_EXPECTS(lhs.type().id() == rhs.type().id(), "Column base types must match");
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Column sizes must match");

  if (lhs.is_empty()) { return make_empty_column(lhs.type()); }

  return type_dispatcher(lhs.type(), divide_decimal_dispatch{}, lhs, rhs, method, stream, mr);
}

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       scalar const& rhs,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_EXPECTS(cudf::is_fixed_point(lhs.type()) && cudf::is_fixed_point(rhs.type()),
               "Column and scalar must be decimal type");
  CUDF_EXPECTS(rhs.type().id() == lhs.type().id(), "Scalar base type must match column type");

  if (lhs.is_empty()) { return make_empty_column(lhs.type()); }

  return type_dispatcher(lhs.type(), divide_decimal_dispatch{}, lhs, rhs, method, stream, mr);
}

std::unique_ptr<column> divide_decimal(scalar const& lhs,
                                       column_view const& rhs,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_EXPECTS(cudf::is_fixed_point(lhs.type()) && cudf::is_fixed_point(rhs.type()),
               "Column and scalar must be decimal type");
  CUDF_EXPECTS(lhs.type().id() == rhs.type().id(), "Scalar base type must match column type");

  if (rhs.is_empty()) { return make_empty_column(lhs.type()); }

  return type_dispatcher(rhs.type(), divide_decimal_dispatch{}, lhs, rhs, method, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       column_view const& rhs,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::divide_decimal(lhs, rhs, method, stream, mr);
}

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       scalar const& rhs,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::divide_decimal(lhs, rhs, method, stream, mr);
}

std::unique_ptr<column> divide_decimal(scalar const& lhs,
                                       column_view const& rhs,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::divide_decimal(lhs, rhs, method, stream, mr);
}

}  // namespace cudf

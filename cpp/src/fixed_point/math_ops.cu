/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "math_ops.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/transform.hpp>
#include <cudf/fixed_point/math_ops.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/utilities/concepts.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

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

template <cudf::concepts::fixed_point T, typename LhsIterator, typename RhsIterator>
struct divide_decimal_fn {
  using Rep = typename T::rep;

  LhsIterator lhs;
  RhsIterator rhs;
  rounding_method method;
  Rep* output;
  bool* output_valid;

  __device__ void operator()(size_type i) const
  {
    cuda::std::optional<T> quotient;
    if (auto const l = lhs[i]) {
      if (auto const r = rhs[i]) { quotient = divide_and_round(*l, *r, method); }
    }
    output[i]       = quotient.has_value() ? quotient->value() : Rep{0};
    output_valid[i] = quotient.has_value();
  }
};

/**
 * @brief Divides `lhs[i]` by `rhs[i]` for each row into a new column of type `output_type`.
 *
 * The iterators yield `cuda::std::optional<T>`. A row is null if either operand is null or the
 * division has no result.
 */
template <cudf::concepts::fixed_point T, typename LhsIterator, typename RhsIterator>
std::unique_ptr<column> divide_decimal_impl(LhsIterator lhs,
                                            RhsIterator rhs,
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
                     divide_decimal_fn<T, LhsIterator, RhsIterator>{
                       lhs, rhs, method, result->mutable_view().begin<Rep>(), valid.data()});

  auto [null_mask, null_count] = cudf::detail::bools_to_mask(
    column_view{data_type{type_id::BOOL8}, size, valid.data(), nullptr, 0},
    stream,
    mr.get_output_mr());
  if (null_count > 0) { result->set_null_mask(std::move(*null_mask), null_count); }
  return result;
}

/**
 * @brief Returns true if every element of `input` is null.
 */
bool all_null(column_view const& input) { return input.null_count() == input.size(); }

/**
 * @brief Returns a column of `size` null elements of type `type`.
 */
std::unique_ptr<column> make_all_null_column(data_type type,
                                             size_type size,
                                             cuda::stream_ref stream,
                                             cudf::memory_resources mr)
{
  return make_fixed_width_column(type, size, mask_state::ALL_NULL, stream, mr.get_output_mr());
}

/**
 * @brief Calls `fn.template operator()<T>()` with the decimal type `T` matching `type`.
 */
template <typename Fn>
std::unique_ptr<column> dispatch_decimal(data_type type, Fn&& fn)
{
  switch (type.id()) {
    case type_id::DECIMAL32: return fn.template operator()<numeric::decimal32>();
    case type_id::DECIMAL64: return fn.template operator()<numeric::decimal64>();
    case type_id::DECIMAL128: return fn.template operator()<numeric::decimal128>();
    default: CUDF_FAIL("divide_decimal only supports decimal types");
  }
}

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
  if (all_null(lhs) || all_null(rhs)) {
    return make_all_null_column(lhs.type(), lhs.size(), stream, mr);
  }

  return dispatch_decimal(lhs.type(), [&]<typename T>() {
    auto const d_lhs = column_device_view::create(lhs, stream);
    auto const d_rhs = column_device_view::create(rhs, stream);
    return divide_decimal_impl<T>(
      make_optional_iterator<T>(*d_lhs, nullate::DYNAMIC{lhs.has_nulls()}),
      make_optional_iterator<T>(*d_rhs, nullate::DYNAMIC{rhs.has_nulls()}),
      lhs.type(),
      lhs.size(),
      method,
      stream,
      mr);
  });
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
  if (all_null(lhs)) { return make_all_null_column(lhs.type(), lhs.size(), stream, mr); }

  return dispatch_decimal(lhs.type(), [&]<typename T>() {
    auto const d_lhs = column_device_view::create(lhs, stream);
    return divide_decimal_impl<T>(
      make_optional_iterator<T>(*d_lhs, nullate::DYNAMIC{lhs.has_nulls()}),
      make_optional_iterator<T>(rhs, nullate::YES{}),
      lhs.type(),
      lhs.size(),
      method,
      stream,
      mr);
  });
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
  if (all_null(rhs)) { return make_all_null_column(lhs.type(), rhs.size(), stream, mr); }

  return dispatch_decimal(rhs.type(), [&]<typename T>() {
    auto const d_rhs = column_device_view::create(rhs, stream);
    return divide_decimal_impl<T>(
      make_optional_iterator<T>(lhs, nullate::YES{}),
      make_optional_iterator<T>(*d_rhs, nullate::DYNAMIC{rhs.has_nulls()}),
      lhs.type(),
      rhs.size(),
      method,
      stream,
      mr);
  });
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

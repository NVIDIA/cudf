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

#include <algorithm>
#include <cstdint>
#include <memory>
#include <type_traits>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief The wider of the representation types `A` and `B`.
 */
template <typename A, typename B>
using wider_rep_t = std::conditional_t<(sizeof(A) >= sizeof(B)), A, B>;

/**
 * @brief The representation type wide enough for the operands and the output.
 */
template <cudf::concepts::fixed_point LhsT,
          cudf::concepts::fixed_point RhsT,
          cudf::concepts::fixed_point OutT>
using compute_rep_t =
  wider_rep_t<wider_rep_t<typename LhsT::rep, typename RhsT::rep>, typename OutT::rep>;

template <typename OutRep, typename CompRep, typename LhsIterator, typename RhsIterator>
struct divide_decimal_fn {
  LhsIterator lhs;
  RhsIterator rhs;
  int32_t exponent;
  rounding_method method;
  OutRep* output;
  bool* output_valid;

  __device__ void operator()(size_type i) const
  {
    cuda::std::optional<OutRep> quotient;
    if (auto const l = lhs[i]) {
      if (auto const r = rhs[i]) {
        quotient = divide_and_round<OutRep>(
          static_cast<CompRep>(l->value()), static_cast<CompRep>(r->value()), exponent, method);
      }
    }
    output[i]       = quotient.value_or(OutRep{0});
    output_valid[i] = quotient.has_value();
  }
};

/**
 * @brief Divides `lhs[i]` by `rhs[i]` for each row into a new column of type `output_type`.
 *
 * The iterators yield `cuda::std::optional<LhsT>` and `cuda::std::optional<RhsT>`. A row is null
 * if either operand is null or the division has no result.
 */
template <cudf::concepts::fixed_point LhsT,
          cudf::concepts::fixed_point RhsT,
          cudf::concepts::fixed_point OutT,
          typename LhsIterator,
          typename RhsIterator>
std::unique_ptr<column> divide_decimal_impl(LhsIterator lhs,
                                            RhsIterator rhs,
                                            int32_t exponent,
                                            data_type output_type,
                                            size_type size,
                                            rounding_method method,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  using OutRep  = typename OutT::rep;
  using CompRep = compute_rep_t<LhsT, RhsT, OutT>;

  auto result =
    make_fixed_width_column(output_type, size, mask_state::UNALLOCATED, stream, mr.get_output_mr());
  auto valid = cuda::device_buffer<bool>(stream, mr.get_temporary_mr(), size, cuda::no_init);

  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
    cuda::counting_iterator<size_type>{0},
    size,
    divide_decimal_fn<OutRep, CompRep, LhsIterator, RhsIterator>{
      lhs, rhs, exponent, method, result->mutable_view().begin<OutRep>(), valid.data()});

  auto [null_mask, null_count] = cudf::detail::bools_to_mask(
    column_view{data_type{type_id::BOOL8}, size, valid.data(), nullptr, 0},
    stream,
    mr.get_output_mr());
  if (null_count > 0) { result->set_null_mask(std::move(*null_mask), null_count); }
  return result;
}

/**
 * @brief Returns the power of 10 that rescales `lhs / rhs` to the scale of `output_type`.
 */
int32_t rescale_exponent(data_type lhs, data_type rhs, data_type output_type)
{
  auto const exponent = static_cast<int64_t>(lhs.scale()) - rhs.scale() - output_type.scale();
  // Any exponent beyond +/-64 already overflows every magnitude type, so clamping keeps results
  // unchanged while avoiding overflow of int32_t.
  return static_cast<int32_t>(std::clamp<int64_t>(exponent, -64, 64));
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

/**
 * @brief Calls `fn.template operator()<LhsT, RhsT, OutT>()` with the decimal types matching
 * `lhs`, `rhs` and `output_type`.
 */
template <typename Fn>
std::unique_ptr<column> dispatch_decimals(data_type lhs,
                                          data_type rhs,
                                          data_type output_type,
                                          Fn&& fn)
{
  return dispatch_decimal(lhs, [&]<typename LhsT>() {
    return dispatch_decimal(rhs, [&]<typename RhsT>() {
      return dispatch_decimal(
        output_type, [&]<typename OutT>() { return fn.template operator()<LhsT, RhsT, OutT>(); });
    });
  });
}

}  // namespace

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       column_view const& rhs,
                                       data_type output_type,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_EXPECTS(cudf::is_fixed_point(lhs.type()) && cudf::is_fixed_point(rhs.type()),
               "Columns must be decimal type");
  CUDF_EXPECTS(cudf::is_fixed_point(output_type), "Output type must be a decimal type");
  CUDF_EXPECTS(lhs.size() == rhs.size(), "Column sizes must match");

  if (lhs.is_empty()) { return make_empty_column(output_type); }
  if (all_null(lhs) || all_null(rhs)) {
    return make_all_null_column(output_type, lhs.size(), stream, mr);
  }

  auto const exponent = rescale_exponent(lhs.type(), rhs.type(), output_type);
  return dispatch_decimals(
    lhs.type(), rhs.type(), output_type, [&]<typename LhsT, typename RhsT, typename OutT>() {
      auto const d_lhs = column_device_view::create(lhs, stream);
      auto const d_rhs = column_device_view::create(rhs, stream);
      return divide_decimal_impl<LhsT, RhsT, OutT>(
        make_optional_iterator<LhsT>(*d_lhs, nullate::DYNAMIC{lhs.has_nulls()}),
        make_optional_iterator<RhsT>(*d_rhs, nullate::DYNAMIC{rhs.has_nulls()}),
        exponent,
        output_type,
        lhs.size(),
        method,
        stream,
        mr);
    });
}

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       scalar const& rhs,
                                       data_type output_type,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_EXPECTS(cudf::is_fixed_point(lhs.type()) && cudf::is_fixed_point(rhs.type()),
               "Column and scalar must be decimal type");
  CUDF_EXPECTS(cudf::is_fixed_point(output_type), "Output type must be a decimal type");

  if (lhs.is_empty()) { return make_empty_column(output_type); }
  if (all_null(lhs)) { return make_all_null_column(output_type, lhs.size(), stream, mr); }

  auto const exponent = rescale_exponent(lhs.type(), rhs.type(), output_type);
  return dispatch_decimals(
    lhs.type(), rhs.type(), output_type, [&]<typename LhsT, typename RhsT, typename OutT>() {
      auto const d_lhs = column_device_view::create(lhs, stream);
      return divide_decimal_impl<LhsT, RhsT, OutT>(
        make_optional_iterator<LhsT>(*d_lhs, nullate::DYNAMIC{lhs.has_nulls()}),
        make_optional_iterator<RhsT>(rhs, nullate::YES{}),
        exponent,
        output_type,
        lhs.size(),
        method,
        stream,
        mr);
    });
}

std::unique_ptr<column> divide_decimal(scalar const& lhs,
                                       column_view const& rhs,
                                       data_type output_type,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_EXPECTS(cudf::is_fixed_point(lhs.type()) && cudf::is_fixed_point(rhs.type()),
               "Column and scalar must be decimal type");
  CUDF_EXPECTS(cudf::is_fixed_point(output_type), "Output type must be a decimal type");

  if (rhs.is_empty()) { return make_empty_column(output_type); }
  if (all_null(rhs)) { return make_all_null_column(output_type, rhs.size(), stream, mr); }

  auto const exponent = rescale_exponent(lhs.type(), rhs.type(), output_type);
  return dispatch_decimals(
    lhs.type(), rhs.type(), output_type, [&]<typename LhsT, typename RhsT, typename OutT>() {
      auto const d_rhs = column_device_view::create(rhs, stream);
      return divide_decimal_impl<LhsT, RhsT, OutT>(
        make_optional_iterator<LhsT>(lhs, nullate::YES{}),
        make_optional_iterator<RhsT>(*d_rhs, nullate::DYNAMIC{rhs.has_nulls()}),
        exponent,
        output_type,
        rhs.size(),
        method,
        stream,
        mr);
    });
}

}  // namespace detail

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       column_view const& rhs,
                                       data_type output_type,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::divide_decimal(lhs, rhs, output_type, method, stream, mr);
}

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       scalar const& rhs,
                                       data_type output_type,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::divide_decimal(lhs, rhs, output_type, method, stream, mr);
}

std::unique_ptr<column> divide_decimal(scalar const& lhs,
                                       column_view const& rhs,
                                       data_type output_type,
                                       rounding_method method,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::divide_decimal(lhs, rhs, output_type, method, stream, mr);
}

}  // namespace cudf

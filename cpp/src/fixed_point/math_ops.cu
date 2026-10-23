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

/**
 * @brief Functor that divides `lhs[i]` by `rhs[i]` and writes the quotient to `output[i]`.
 *
 * The quotient is `lhs * 10^exponent / rhs`, rounded using `method`. A row is null, and its
 * output value is 0, if either operand is null or the division has no result.
 *
 * @tparam OutRep The representation type of the output
 * @tparam CompRep The representation type both operands are widened to for the division
 * @tparam LhsIterator Iterator yielding optional dividends
 * @tparam RhsIterator Iterator yielding optional divisors
 */
template <typename OutRep, typename CompRep, typename LhsIterator, typename RhsIterator>
struct divide_decimal_fn {
  LhsIterator lhs;
  RhsIterator rhs;
  int32_t exponent;
  rounding_method method;
  decimal_intermediate intermediate;
  OutRep* output;
  bool* output_valid;

  __device__ void operator()(size_type i) const
  {
    auto const l        = lhs[i];
    auto const r        = rhs[i];
    auto const quotient = (l and r) ? divide_and_round<OutRep>(static_cast<CompRep>(l->value()),
                                                               static_cast<CompRep>(r->value()),
                                                               exponent,
                                                               method,
                                                               intermediate)
                                    : cuda::std::optional<OutRep>{};
    output[i]           = quotient.value_or(OutRep{0});
    output_valid[i]     = quotient.has_value();
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
                                            decimal_intermediate intermediate,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  using OutRep  = typename OutT::rep;
  using CompRep = compute_rep_t<LhsT, RhsT, OutT>;

  auto result =
    make_fixed_width_column(output_type, size, mask_state::UNALLOCATED, stream, mr.get_output_mr());
  auto valid = cuda::device_buffer<bool>(stream, mr.get_temporary_mr(), size, cuda::no_init);

  thrust::for_each_n(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                     cuda::counting_iterator<size_type>{0},
                     size,
                     divide_decimal_fn<OutRep, CompRep, LhsIterator, RhsIterator>{
                       lhs,
                       rhs,
                       exponent,
                       method,
                       intermediate,
                       result->mutable_view().begin<OutRep>(),
                       valid.data()});

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
[[nodiscard]] int32_t rescale_exponent(data_type lhs, data_type rhs, data_type output_type)
{
  auto const exponent = static_cast<int64_t>(lhs.scale()) - rhs.scale() - output_type.scale();
  // Any exponent beyond +/-80 already overflows even the 256-bit intermediate, so clamping keeps
  // results unchanged while avoiding overflow of int32_t.
  return static_cast<int32_t>(std::clamp<int64_t>(exponent, -80, 80));
}

/**
 * @brief Returns true if every element of `input` is null.
 */
[[nodiscard]] inline bool all_null(column_view const& input)
{
  return input.null_count() == input.size();
}

/**
 * @brief Returns a column of `size` null elements of type `type`.
 */
[[nodiscard]] inline std::unique_ptr<column> make_all_null_column(data_type type,
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

/**
 * @brief Divides column `col` by scalar `s`, or `s` by `col` if `ScalarIsLhs` is true.
 *
 * @tparam ScalarIsLhs Whether `s` is the dividend rather than the divisor
 */
template <bool ScalarIsLhs>
std::unique_ptr<column> divide_decimal_column_scalar(column_view const& col,
                                                     scalar const& s,
                                                     data_type output_type,
                                                     rounding_method method,
                                                     decimal_intermediate intermediate,
                                                     cuda::stream_ref stream,
                                                     cudf::memory_resources mr)
{
  CUDF_EXPECTS(cudf::is_fixed_point(col.type()) && cudf::is_fixed_point(s.type()),
               "Column and scalar must be decimal type");
  CUDF_EXPECTS(cudf::is_fixed_point(output_type), "Output type must be a decimal type");

  if (col.is_empty()) { return make_empty_column(output_type); }
  if (all_null(col)) { return make_all_null_column(output_type, col.size(), stream, mr); }

  auto const lhs_type = ScalarIsLhs ? s.type() : col.type();
  auto const rhs_type = ScalarIsLhs ? col.type() : s.type();
  auto const exponent = rescale_exponent(lhs_type, rhs_type, output_type);
  return dispatch_decimals(
    lhs_type, rhs_type, output_type, [&]<typename LhsT, typename RhsT, typename OutT>() {
      using ColT        = std::conditional_t<ScalarIsLhs, RhsT, LhsT>;
      using ScalarT     = std::conditional_t<ScalarIsLhs, LhsT, RhsT>;
      auto const d_col  = column_device_view::create(col, stream);
      auto const col_it = make_optional_iterator<ColT>(*d_col, nullate::DYNAMIC{col.has_nulls()});
      auto const s_it   = make_optional_iterator<ScalarT>(s, nullate::YES{});
      if constexpr (ScalarIsLhs) {
        return divide_decimal_impl<LhsT, RhsT, OutT>(
          s_it, col_it, exponent, output_type, col.size(), method, intermediate, stream, mr);
      } else {
        return divide_decimal_impl<LhsT, RhsT, OutT>(
          col_it, s_it, exponent, output_type, col.size(), method, intermediate, stream, mr);
      }
    });
}

}  // namespace

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       column_view const& rhs,
                                       data_type output_type,
                                       rounding_method method,
                                       decimal_intermediate intermediate,
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
        intermediate,
        stream,
        mr);
    });
}

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       scalar const& rhs,
                                       data_type output_type,
                                       rounding_method method,
                                       decimal_intermediate intermediate,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  return divide_decimal_column_scalar<false>(
    lhs, rhs, output_type, method, intermediate, stream, mr);
}

std::unique_ptr<column> divide_decimal(scalar const& lhs,
                                       column_view const& rhs,
                                       data_type output_type,
                                       rounding_method method,
                                       decimal_intermediate intermediate,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  return divide_decimal_column_scalar<true>(
    rhs, lhs, output_type, method, intermediate, stream, mr);
}

}  // namespace detail

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       column_view const& rhs,
                                       data_type output_type,
                                       rounding_method method,
                                       decimal_intermediate intermediate,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::divide_decimal(lhs, rhs, output_type, method, intermediate, stream, mr);
}

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       scalar const& rhs,
                                       data_type output_type,
                                       rounding_method method,
                                       decimal_intermediate intermediate,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::divide_decimal(lhs, rhs, output_type, method, intermediate, stream, mr);
}

std::unique_ptr<column> divide_decimal(scalar const& lhs,
                                       column_view const& rhs,
                                       data_type output_type,
                                       rounding_method method,
                                       decimal_intermediate intermediate,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::divide_decimal(lhs, rhs, output_type, method, intermediate, stream, mr);
}

}  // namespace cudf

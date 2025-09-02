/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/decimal/decimal_ops.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_device_view.cuh>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <thrust/transform.h>

namespace cudf {
namespace detail {

template <typename DecimalType>
struct divide_decimal_functor {
  numeric::decimal_rounding_mode rounding_mode;

  __device__ DecimalType operator()(DecimalType const& lhs, DecimalType const& rhs) const
  {
    return numeric::divide_decimal(lhs, rhs, rounding_mode);
  }
};

std::unique_ptr<column> divide_decimal_impl(column_view const& lhs,
                                            column_view const& rhs,
                                            numeric::decimal_rounding_mode rounding_mode,
                                            rmm::cuda_stream_view stream,
                                            rmm::device_async_resource_ref mr)
{
  using namespace numeric;

  auto const size     = lhs.size();
  auto const lhs_type = lhs.type();

  // Create output column with same type as lhs (preserves scale)
  // If there are nulls in inputs, create with null mask; otherwise create without
  std::unique_ptr<column> result;
  if (lhs.has_nulls() || rhs.has_nulls()) {
    auto [null_mask, null_count] = cudf::detail::bitmask_and(table_view{{lhs, rhs}}, stream, mr);
    result =
      cudf::make_fixed_width_column(lhs_type, size, std::move(null_mask), null_count, stream, mr);
  } else {
    // Create non-nullable column when inputs have no nulls
    // Use empty rmm::device_buffer{} directly to ensure column is non-nullable
    result =
      std::make_unique<column>(lhs_type,
                               size,
                               rmm::device_buffer{size * cudf::size_of(lhs_type), stream, mr},
                               rmm::device_buffer{},  // Empty buffer = non-nullable
                               0,                     // null_count = 0
                               std::vector<std::unique_ptr<column>>{});
  }

  auto result_view = result->mutable_view();

  // Get device views
  auto const lhs_dev = column_device_view::create(lhs, stream);
  auto const rhs_dev = column_device_view::create(rhs, stream);

  // Perform element-wise divide_decimal
  if (lhs_type.id() == type_id::DECIMAL32) {
    using Type           = int32_t;
    auto const lhs_scale = lhs_type.scale();
    auto const rhs_scale = rhs.type().scale();
    using DecType        = fixed_point<Type, numeric::Radix::BASE_10>;

    thrust::transform(
      rmm::exec_policy(stream),
      lhs.begin<Type>(),
      lhs.end<Type>(),
      rhs.begin<Type>(),
      result_view.begin<Type>(),
      [lhs_scale, rhs_scale, rounding_mode] __device__(Type lhs_val, Type rhs_val) {
        DecType lhs_fp{numeric::scaled_integer<Type>{lhs_val, numeric::scale_type{lhs_scale}}};
        DecType rhs_fp{numeric::scaled_integer<Type>{rhs_val, numeric::scale_type{rhs_scale}}};
        auto result_fp = numeric::divide_decimal(lhs_fp, rhs_fp, rounding_mode);
        return result_fp.value();
      });
  } else if (lhs_type.id() == type_id::DECIMAL64) {
    using Type           = int64_t;
    auto const lhs_scale = lhs_type.scale();
    auto const rhs_scale = rhs.type().scale();
    using DecType        = fixed_point<Type, numeric::Radix::BASE_10>;

    thrust::transform(
      rmm::exec_policy(stream),
      lhs.begin<Type>(),
      lhs.end<Type>(),
      rhs.begin<Type>(),
      result_view.begin<Type>(),
      [lhs_scale, rhs_scale, rounding_mode] __device__(Type lhs_val, Type rhs_val) {
        DecType lhs_fp{numeric::scaled_integer<Type>{lhs_val, numeric::scale_type{lhs_scale}}};
        DecType rhs_fp{numeric::scaled_integer<Type>{rhs_val, numeric::scale_type{rhs_scale}}};
        auto result_fp = numeric::divide_decimal(lhs_fp, rhs_fp, rounding_mode);
        return result_fp.value();
      });
  } else if (lhs_type.id() == type_id::DECIMAL128) {
    using Type           = __int128_t;
    auto const lhs_scale = lhs_type.scale();
    auto const rhs_scale = rhs.type().scale();
    using DecType        = fixed_point<Type, numeric::Radix::BASE_10>;

    thrust::transform(
      rmm::exec_policy(stream),
      lhs.begin<Type>(),
      lhs.end<Type>(),
      rhs.begin<Type>(),
      result_view.begin<Type>(),
      [lhs_scale, rhs_scale, rounding_mode] __device__(Type lhs_val, Type rhs_val) {
        DecType lhs_fp{numeric::scaled_integer<Type>{lhs_val, numeric::scale_type{lhs_scale}}};
        DecType rhs_fp{numeric::scaled_integer<Type>{rhs_val, numeric::scale_type{rhs_scale}}};
        auto result_fp = numeric::divide_decimal(lhs_fp, rhs_fp, rounding_mode);
        return result_fp.value();
      });
  }

  // Null mask already handled during column creation

  return result;
}

template <typename DecimalType>
std::unique_ptr<column> divide_decimal_scalar_impl(column_view const& lhs,
                                                   scalar const& rhs,
                                                   numeric::decimal_rounding_mode rounding_mode,
                                                   rmm::cuda_stream_view stream,
                                                   rmm::device_async_resource_ref mr)
{
  using namespace numeric;
  using Type = typename DecimalType::rep;

  auto const size      = lhs.size();
  auto const lhs_type  = lhs.type();
  auto const lhs_scale = lhs_type.scale();
  auto const rhs_scale = rhs.type().scale();

  // Get scalar value
  auto const& decimal_scalar = static_cast<fixed_point_scalar<DecimalType> const&>(rhs);
  DecimalType rhs_fp         = decimal_scalar.value(stream);
  Type rhs_val               = rhs_fp.value();

  // Create output column with same type as lhs
  // If there are nulls in inputs, create with null mask; otherwise create without
  std::unique_ptr<column> result;
  if (lhs.has_nulls() || !rhs.is_valid(stream)) {
    if (!rhs.is_valid(stream)) {
      result = cudf::make_fixed_width_column(
        lhs_type,
        size,
        cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr),
        size,
        stream,
        mr);
    } else {
      result = cudf::make_fixed_width_column(
        lhs_type, size, cudf::detail::copy_bitmask(lhs, stream, mr), lhs.null_count(), stream, mr);
    }
  } else {
    // Create non-nullable column when inputs have no nulls
    // Use empty rmm::device_buffer{} directly to ensure column is non-nullable
    result =
      std::make_unique<column>(lhs_type,
                               size,
                               rmm::device_buffer{size * cudf::size_of(lhs_type), stream, mr},
                               rmm::device_buffer{},  // Empty buffer = non-nullable
                               0,                     // null_count = 0
                               std::vector<std::unique_ptr<column>>{});
  }

  auto result_view = result->mutable_view();

  // Perform element-wise divide_decimal
  thrust::transform(
    rmm::exec_policy(stream),
    lhs.begin<Type>(),
    lhs.end<Type>(),
    result_view.begin<Type>(),
    [lhs_scale, rhs_scale, rhs_val, rounding_mode] __device__(Type lhs_val) {
      DecimalType lhs_fp{numeric::scaled_integer<Type>{lhs_val, numeric::scale_type{lhs_scale}}};
      DecimalType rhs_fp{numeric::scaled_integer<Type>{rhs_val, numeric::scale_type{rhs_scale}}};
      auto result_fp = numeric::divide_decimal(lhs_fp, rhs_fp, rounding_mode);
      return result_fp.value();
    });

  // Null mask already handled during column creation

  return result;
}

}  // namespace detail

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       column_view const& rhs,
                                       numeric::decimal_rounding_mode rounding_mode,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(lhs.size() == rhs.size(), "Column sizes must match");
  // For decimal division, we only need the same base type (DECIMAL32/64/128)
  // Different scales are allowed and expected
  CUDF_EXPECTS(lhs.type().id() == rhs.type().id(), "Column base types must match");

  CUDF_EXPECTS(lhs.type().id() == type_id::DECIMAL32 || lhs.type().id() == type_id::DECIMAL64 ||
                 lhs.type().id() == type_id::DECIMAL128,
               "Columns must be decimal type");

  // Note: Zero division check is handled in the kernel
  // GPU kernels cannot throw exceptions, so they produce special values or assert in debug mode

  if (lhs.is_empty()) { return make_empty_column(lhs.type()); }

  return detail::divide_decimal_impl(lhs, rhs, rounding_mode, stream, mr);
}

std::unique_ptr<column> divide_decimal(column_view const& lhs,
                                       scalar const& rhs,
                                       numeric::decimal_rounding_mode rounding_mode,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(lhs.type().id() == type_id::DECIMAL32 || lhs.type().id() == type_id::DECIMAL64 ||
                 lhs.type().id() == type_id::DECIMAL128,
               "Column must be decimal type");
  CUDF_EXPECTS(rhs.type() == lhs.type(), "Scalar type must match column type");

  if (lhs.is_empty()) { return make_empty_column(lhs.type()); }

  using namespace numeric;

  if (lhs.type().id() == type_id::DECIMAL32) {
    using DecType = fixed_point<int32_t, Radix::BASE_10>;
    return detail::divide_decimal_scalar_impl<DecType>(lhs, rhs, rounding_mode, stream, mr);
  } else if (lhs.type().id() == type_id::DECIMAL64) {
    using DecType = fixed_point<int64_t, Radix::BASE_10>;
    return detail::divide_decimal_scalar_impl<DecType>(lhs, rhs, rounding_mode, stream, mr);
  } else {
    using DecType = fixed_point<__int128_t, Radix::BASE_10>;
    return detail::divide_decimal_scalar_impl<DecType>(lhs, rhs, rounding_mode, stream, mr);
  }
}

std::unique_ptr<column> divide_decimal(scalar const& lhs,
                                       column_view const& rhs,
                                       numeric::decimal_rounding_mode rounding_mode,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(rhs.type().id() == type_id::DECIMAL32 || rhs.type().id() == type_id::DECIMAL64 ||
                 rhs.type().id() == type_id::DECIMAL128,
               "Column must be decimal type");
  CUDF_EXPECTS(lhs.type() == rhs.type(), "Scalar type must match column type");

  if (rhs.is_empty()) { return make_empty_column(rhs.type()); }

  using namespace numeric;

  // For scalar-column division, implement directly
  auto const size      = rhs.size();
  auto const rhs_type  = rhs.type();
  auto const lhs_scale = lhs.type().scale();
  auto const rhs_scale = rhs_type.scale();

  // Create output column with same type as rhs
  // If there are nulls in inputs, create with null mask; otherwise create without
  std::unique_ptr<column> result;
  if (!lhs.is_valid(stream) || rhs.has_nulls()) {
    if (!lhs.is_valid(stream)) {
      result = cudf::make_fixed_width_column(
        rhs_type,
        size,
        cudf::detail::create_null_mask(size, mask_state::ALL_NULL, stream, mr),
        size,
        stream,
        mr);
    } else {
      result = cudf::make_fixed_width_column(
        rhs_type, size, cudf::detail::copy_bitmask(rhs, stream, mr), rhs.null_count(), stream, mr);
    }
  } else {
    // Create non-nullable column when inputs have no nulls
    // Use empty rmm::device_buffer{} directly to ensure column is non-nullable
    result =
      std::make_unique<column>(rhs_type,
                               size,
                               rmm::device_buffer{size * cudf::size_of(rhs_type), stream, mr},
                               rmm::device_buffer{},  // Empty buffer = non-nullable
                               0,                     // null_count = 0
                               std::vector<std::unique_ptr<column>>{});
  }

  auto result_view = result->mutable_view();

  // Perform element-wise divide_decimal based on type
  if (rhs_type.id() == type_id::DECIMAL32) {
    using Type    = int32_t;
    using DecType = fixed_point<Type, Radix::BASE_10>;

    auto const& decimal_scalar = static_cast<fixed_point_scalar<DecType> const&>(lhs);
    DecType lhs_fp             = decimal_scalar.value(stream);
    Type lhs_val               = lhs_fp.value();

    thrust::transform(
      rmm::exec_policy(stream),
      rhs.begin<Type>(),
      rhs.end<Type>(),
      result_view.begin<Type>(),
      [lhs_scale, rhs_scale, lhs_val, rounding_mode] __device__(Type rhs_val) {
        DecType lhs_fp{numeric::scaled_integer<Type>{lhs_val, numeric::scale_type{lhs_scale}}};
        DecType rhs_fp{numeric::scaled_integer<Type>{rhs_val, numeric::scale_type{rhs_scale}}};
        auto result_fp = numeric::divide_decimal(lhs_fp, rhs_fp, rounding_mode);
        return result_fp.value();
      });
  } else if (rhs_type.id() == type_id::DECIMAL64) {
    using Type    = int64_t;
    using DecType = fixed_point<Type, Radix::BASE_10>;

    auto const& decimal_scalar = static_cast<fixed_point_scalar<DecType> const&>(lhs);
    DecType lhs_fp             = decimal_scalar.value(stream);
    Type lhs_val               = lhs_fp.value();

    thrust::transform(
      rmm::exec_policy(stream),
      rhs.begin<Type>(),
      rhs.end<Type>(),
      result_view.begin<Type>(),
      [lhs_scale, rhs_scale, lhs_val, rounding_mode] __device__(Type rhs_val) {
        DecType lhs_fp{numeric::scaled_integer<Type>{lhs_val, numeric::scale_type{lhs_scale}}};
        DecType rhs_fp{numeric::scaled_integer<Type>{rhs_val, numeric::scale_type{rhs_scale}}};
        auto result_fp = numeric::divide_decimal(lhs_fp, rhs_fp, rounding_mode);
        return result_fp.value();
      });
  } else {
    using Type    = __int128_t;
    using DecType = fixed_point<Type, Radix::BASE_10>;

    auto const& decimal_scalar = static_cast<fixed_point_scalar<DecType> const&>(lhs);
    DecType lhs_fp             = decimal_scalar.value(stream);
    Type lhs_val               = lhs_fp.value();

    thrust::transform(
      rmm::exec_policy(stream),
      rhs.begin<Type>(),
      rhs.end<Type>(),
      result_view.begin<Type>(),
      [lhs_scale, rhs_scale, lhs_val, rounding_mode] __device__(Type rhs_val) {
        DecType lhs_fp{numeric::scaled_integer<Type>{lhs_val, numeric::scale_type{lhs_scale}}};
        DecType rhs_fp{numeric::scaled_integer<Type>{rhs_val, numeric::scale_type{rhs_scale}}};
        auto result_fp = numeric::divide_decimal(lhs_fp, rhs_fp, rounding_mode);
        return result_fp.value();
      });
  }

  // Null mask already handled during column creation

  return result;
}

}  // namespace cudf

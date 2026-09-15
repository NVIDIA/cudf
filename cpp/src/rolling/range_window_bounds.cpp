/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "detail/range_window_bounds.hpp"

#include <cudf/rolling/range_window_bounds.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf {
namespace {

/**
 * @brief Factory to (copy) construct scalars.
 *
 * Derived types of scalars are cloned, to be adopted by `range_window_bounds`.
 * This makes it possible to copy construct and copy assign `range_window_bounds` objects.
 */
struct range_scalar_constructor {
  template <typename T, CUDF_ENABLE_IF(not detail::is_supported_range_type<T>())>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar_, cuda::stream_ref stream) const
  {
    CUDF_FAIL(
      "Unsupported range type. "
      "Only durations, fixed-point, and non-boolean numeric range types are allowed.");
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_duration<T>())>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar_, cuda::stream_ref stream) const
  {
    return std::make_unique<duration_scalar<T>>(
      static_cast<duration_scalar<T> const&>(range_scalar_),
      stream,
      cudf::get_current_device_resource_ref());
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_numeric<T>() && not cudf::is_boolean<T>())>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar_, cuda::stream_ref stream) const
  {
    return std::make_unique<numeric_scalar<T>>(static_cast<numeric_scalar<T> const&>(range_scalar_),
                                               stream,
                                               cudf::get_current_device_resource_ref());
  }

  template <typename T, CUDF_ENABLE_IF(cudf::is_fixed_point<T>())>
  std::unique_ptr<scalar> operator()(scalar const& range_scalar_, cuda::stream_ref stream) const
  {
    return std::make_unique<fixed_point_scalar<T>>(
      static_cast<fixed_point_scalar<T> const&>(range_scalar_),
      stream,
      cudf::get_current_device_resource_ref());
  }
};
}  // namespace

range_window_bounds::range_window_bounds(extent_type extent_,
                                         std::unique_ptr<scalar> range_scalar_,
                                         cuda::stream_ref stream)
  : _extent{extent_}, _range_scalar{std::move(range_scalar_)}
{
  CUDF_EXPECTS(_range_scalar.get(), "Range window scalar cannot be null.");
  CUDF_EXPECTS(_extent == extent_type::UNBOUNDED || _extent == extent_type::CURRENT_ROW ||
                 _range_scalar->is_valid(stream),
               "Bounded Range window scalar must be valid.");
}

range_window_bounds range_window_bounds::unbounded(data_type type, cuda::stream_ref stream)
{
  auto s = make_default_constructed_scalar(type, stream, cudf::get_current_device_resource_ref());
  s->set_valid_async(true, stream);
  return {extent_type::UNBOUNDED, std::move(s), stream};
}

range_window_bounds range_window_bounds::current_row(data_type type, cuda::stream_ref stream)
{
  auto s = make_default_constructed_scalar(type, stream, cudf::get_current_device_resource_ref());
  s->set_valid_async(true, stream);
  return {extent_type::CURRENT_ROW, std::move(s), stream};
}

range_window_bounds range_window_bounds::get(scalar const& boundary, cuda::stream_ref stream)
{
  return {extent_type::BOUNDED,
          cudf::type_dispatcher(boundary.type(), range_scalar_constructor{}, boundary, stream),
          stream};
}

}  // namespace cudf

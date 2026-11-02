/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/datetime.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <memory>

namespace cudf {
namespace datetime {
namespace detail {
/**
 * @copydoc cudf::extract_datetime_component(cudf::column_view const&, datetime_component,
 * cuda::stream_ref, rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> extract_datetime_component(cudf::column_view const& column,
                                                         datetime_component component,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::last_day_of_month(cudf::column_view const&, cuda::stream_ref,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> last_day_of_month(cudf::column_view const& column,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::day_of_year(cudf::column_view const&, cuda::stream_ref,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> day_of_year(cudf::column_view const& column,
                                          cuda::stream_ref stream,
                                          rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::add_calendrical_months(cudf::column_view const&, cudf::column_view const&,
 * cuda::stream_ref, rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> add_calendrical_months(cudf::column_view const& timestamps,
                                                     cudf::column_view const& months,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::add_calendrical_months(cudf::column_view const&, cudf::scalar const&,
 * cuda::stream_ref, rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> add_calendrical_months(cudf::column_view const& timestamps,
                                                     cudf::scalar const& months,
                                                     cuda::stream_ref stream,
                                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::is_leap_year(cudf::column_view const&, cuda::stream_ref,
 * rmm::device_async_resource_ref)
 *
 */
std::unique_ptr<cudf::column> is_leap_year(cudf::column_view const& column,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr);

std::unique_ptr<cudf::column> extract_quarter(cudf::column_view const& column,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace datetime
}  // namespace cudf

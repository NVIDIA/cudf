/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace strings::detail {

/**
 * @copydoc to_integers(strings_column_view const&,data_type,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> to_integers(strings_column_view const& strings,
                                    data_type output_type,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc from_integers(strings_column_view const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> from_integers(column_view const& integers,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc to_floats(strings_column_view const&,data_type,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> to_floats(strings_column_view const& strings,
                                  data_type output_type,
                                  cuda::stream_ref stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @copydoc from_floats(strings_column_view const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> from_floats(column_view const& floats,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc to_booleans(strings_column_view const&,string_scalar
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> to_booleans(strings_column_view const& strings,
                                    string_scalar const& true_string,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc from_booleans(strings_column_view const&,string_scalar const&,string_scalar
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> from_booleans(column_view const& booleans,
                                      string_scalar const& true_string,
                                      string_scalar const& false_string,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc to_timestamps(strings_column_view const&,data_type,std::string_view,
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<cudf::column> to_timestamps(strings_column_view const& strings,
                                            data_type timestamp_type,
                                            std::string_view format,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @copydoc from_timestamps(strings_column_view const&,std::string_view,
 * strings_column_view const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> from_timestamps(column_view const& timestamps,
                                        std::string_view format,
                                        strings_column_view const& names,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr);

/**
 * @copydoc to_durations(strings_column_view const&,data_type,std::string_view,
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> to_durations(strings_column_view const& strings,
                                     data_type duration_type,
                                     std::string_view format,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc from_durations(strings_column_view const&,std::string_view.
 * rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> from_durations(column_view const& durations,
                                       std::string_view format,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc to_fixed_point(strings_column_view const&,data_type,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> to_fixed_point(strings_column_view const& strings,
                                       data_type output_type,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr);

/**
 * @copydoc from_fixed_point(strings_column_view const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> from_fixed_point(column_view const& integers,
                                         cuda::stream_ref stream,
                                         rmm::device_async_resource_ref mr);

}  // namespace strings::detail
}  // namespace cudf

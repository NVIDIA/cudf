/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/memory_resource.hpp>

namespace cudf::strings::detail {

/**
 * @copydoc cudf::strings::split
 */
std::unique_ptr<table> split(strings_column_view const& strings_column,
                             string_scalar const& delimiter,
                             size_type maxsplit,
                             cuda::stream_ref stream,
                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::strings::rsplit
 */
std::unique_ptr<table> rsplit(strings_column_view const& strings_column,
                              string_scalar const& delimiter,
                              size_type maxsplit,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::strings::split_record
 */
std::unique_ptr<column> split_record(strings_column_view const& strings,
                                     string_scalar const& delimiter,
                                     size_type maxsplit,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::strings::rsplit_record
 */
std::unique_ptr<column> rsplit_record(strings_column_view const& strings,
                                      string_scalar const& delimiter,
                                      size_type maxsplit,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

}  // namespace cudf::strings::detail

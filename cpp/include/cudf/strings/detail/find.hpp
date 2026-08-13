/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/strings/find.hpp>

namespace cudf::strings::detail {

/**
 * @copydoc cudf::strings::contains
 */
std::unique_ptr<column> contains(strings_column_view const& input,
                                 string_scalar const& target,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::strings::starts_with
 */
std::unique_ptr<column> starts_with(strings_column_view const& input,
                                    string_scalar const& target,
                                    rmm::cuda_stream_view stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::strings::ends_with
 */
std::unique_ptr<column> ends_with(strings_column_view const& input,
                                  string_scalar const& target,
                                  rmm::cuda_stream_view stream,
                                  rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::strings::count
 */
std::unique_ptr<column> count(strings_column_view const& input,
                              string_scalar const& target,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr);

}  // namespace cudf::strings::detail

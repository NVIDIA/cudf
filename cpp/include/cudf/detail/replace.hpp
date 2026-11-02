/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/replace.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::replace_nulls(column_view const&, column_view const&,
 * cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                      cudf::column_view const& replacement,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::replace_nulls(column_view const&, scalar const&,
 * cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                      scalar const& replacement,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::replace_nulls(column_view const&, replace_policy const&,
 * cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> replace_nulls(column_view const& input,
                                      replace_policy const& replace_policy,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::replace_nans(column_view const&, column_view const&,
 * cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> replace_nans(column_view const& input,
                                     column_view const& replacement,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::replace_nans(column_view const&, scalar const&,
 * cuda::stream_ref, rmm::device_async_resource_ref)
 */
std::unique_ptr<column> replace_nans(column_view const& input,
                                     scalar const& replacement,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::find_and_replace_all
 */
std::unique_ptr<column> find_and_replace_all(column_view const& input_col,
                                             column_view const& values_to_replace,
                                             column_view const& replacement_values,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::normalize_nans_and_zeros
 */
std::unique_ptr<column> normalize_nans_and_zeros(column_view const& input,
                                                 cuda::stream_ref stream,
                                                 rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf

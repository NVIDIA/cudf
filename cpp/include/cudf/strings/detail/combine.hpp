/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

namespace cudf {
namespace strings::detail {

/**
 * @copydoc concatenate(table_view const&,string_scalar const&,string_scalar
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> concatenate(table_view const& strings_columns,
                                    string_scalar const& separator,
                                    string_scalar const& narep,
                                    separator_on_nulls separate_nulls,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc join_strings(table_view const&,string_scalar const&,string_scalar
 * const&,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> join_strings(strings_column_view const& strings,
                                     string_scalar const& separator,
                                     string_scalar const& narep,
                                     cuda::stream_ref stream,
                                     rmm::device_async_resource_ref mr);

/**
 * @copydoc join_list_elements(table_view const&,string_scalar const&,string_scalar
 * const&,separator_on_nulls,output_if_empty_list,rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> join_list_elements(lists_column_view const& lists_strings_column,
                                           string_scalar const& separator,
                                           string_scalar const& narep,
                                           separator_on_nulls separate_nulls,
                                           output_if_empty_list empty_list_policy,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr);

}  // namespace strings::detail
}  // namespace cudf

/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/memory_resource>
#include <cuda/stream>

#include <memory>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::repeat(table_view const&, column_view const&, bool,
 * cuda::stream_ref, cuda::mr::device_resource_ref)
 */
std::unique_ptr<table> repeat(table_view const& input_table,
                              column_view const& count,
                              bool check_count,
                              cuda::stream_ref stream,
                              cuda::mr::device_resource_ref mr);

/**
 * @copydoc cudf::repeat(table_view const&, size_type,
 * cuda::stream_ref, cuda::mr::device_resource_ref)
 */
std::unique_ptr<table> repeat(table_view const& input_table,
                              size_type count,
                              cuda::stream_ref stream,
                              cuda::mr::device_resource_ref mr);

}  // namespace detail
}  // namespace cudf

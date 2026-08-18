/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <memory>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::gather(table_view const&,column_view const&,table_view
 * const&,cudf::out_of_bounds_policy,cudf::negative_index_policy,cuda::stream_ref,
 * rmm::device_async_resource_ref)
 */
std::unique_ptr<table> gather(table_view const& source_table,
                              column_view const& gather_map,
                              out_of_bounds_policy bounds_policy,
                              negative_index_policy neg_indices,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::detail::gather(table_view const&,column_view const&,table_view
 * const&,cudf::out_of_bounds_policy,cudf::negative_index_policy,cuda::stream_ref,
 * rmm::device_async_resource_ref)
 *
 * @throws cudf::logic_error if `gather_map` span size is larger than max of `size_type`.
 */
std::unique_ptr<table> gather(table_view const& source_table,
                              device_span<size_type const> const gather_map,
                              out_of_bounds_policy bounds_policy,
                              negative_index_policy neg_indices,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf

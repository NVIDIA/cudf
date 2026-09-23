/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

namespace cudf {
namespace detail {

/**
 * @copydoc cudf::lower_bound
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> lower_bound(table_view const& haystack,
                                    table_view const& needles,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::upper_bound
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> upper_bound(table_view const& haystack,
                                    table_view const& needles,
                                    std::vector<order> const& column_order,
                                    std::vector<null_order> const& null_precedence,
                                    cuda::stream_ref stream,
                                    rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::contains(column_view const&, scalar const&, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
bool contains(column_view const& haystack, scalar const& needle, cuda::stream_ref stream);

/**
 * @copydoc cudf::contains(column_view const&, column_view const&, rmm::device_async_resource_ref)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<column> contains(column_view const& haystack,
                                 column_view const& needles,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr);

/**
 * @brief Check if rows in the given `needles` table exist in the `haystack` table.
 *
 * Given two tables, each row in the `needles` table is checked to see if there is any matching row
 * (i.e., compared equal to it) in the `haystack` table. The boolean search results are written into
 * the corresponding rows of the output array.
 *
 * @code{.pseudo}
 * Example:
 *
 * haystack = { { 5, 4, 1, 2, 3 } }
 * needles  = { { 0, 1, 2 } }
 * output   = { false, true, true }
 * @endcode
 *
 * @throws cudf::logic_error If column types of haystack and needles don't match
 *
 * @param haystack The table containing the search space
 * @param needles A table of rows whose existence to check in the search space
 * @param compare_nulls Control whether nulls should be compared as equal or not
 * @param compare_nans Control whether floating-point NaNs values should be compared as equal or not
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A vector of bools indicating if each row in `needles` has matching rows in `haystack`
 */
CUDF_EXPORT
rmm::device_uvector<bool> contains(table_view const& haystack,
                                   table_view const& needles,
                                   null_equality compare_nulls,
                                   nan_equality compare_nans,
                                   cuda::stream_ref stream,
                                   rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace cudf

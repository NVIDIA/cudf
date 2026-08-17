/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @brief Scatters the rows of the source table into a copy of the target table
 * according to a scatter map.
 *
 * Scatters values from the source table into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_map[i]` of the destination table gets row
 * `i` of the source table. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of columns in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 *
 * A negative value `i` in the `scatter_map` is interpreted as `i+n`, where `n`
 * is the number of rows in the `target` table.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 * If any values in `scatter_map` are outside of the interval [-n, n) where `n`
 * is the number of rows in the `target` table, behavior is undefined.
 *
 * @param source The input columns containing values to be scattered into the
 * target columns
 * @param scatter_map A non-nullable column of integral indices that maps the
 * rows in the source table to rows in the target table. The size must be equal
 * to or less than the number of elements in the source columns.
 * @param target The set of columns into which values from the source_table
 * are to be scattered
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return Result of scattering values from source to target
 */
std::unique_ptr<table> scatter(table_view const& source,
                               column_view const& scatter_map,
                               table_view const& target,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::detail::scatter(table_view const&,column_view const&,table_view
 * const&,cuda::stream_ref,rmm::device_async_resource_ref)
 *
 * @throws cudf::logic_error if `scatter_map` span size is larger than max of `size_type`.
 */
std::unique_ptr<table> scatter(table_view const& source,
                               device_span<size_type const> const scatter_map,
                               table_view const& target,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr);

/**
 * @brief Scatters a row of scalar values into a copy of the target table
 * according to a scatter map.
 *
 * Scatters values from the source row into the target table out-of-place,
 * returning a "destination table". The scatter is performed according to a
 * scatter map such that row `scatter_map[i]` of the destination table is
 * replaced by the source row. All other rows of the destination table equal
 * corresponding rows of the target table.
 *
 * The number of elements in source must match the number of columns in target
 * and their corresponding datatypes must be the same.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * If any values in `indices` are outside of the interval [-n, n) where `n`
 * is the number of rows in the `target` table, behavior is undefined.
 *
 * @param source The input scalars containing values to be scattered into the
 * target columns
 * @param indices A non-nullable column of integral indices that indicate
 * the rows in the target table to be replaced by source.
 * @param target The set of columns into which values from the source_table
 * are to be scattered
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned table's device memory
 * @return Result of scattering values from source to target
 */
std::unique_ptr<table> scatter(std::vector<std::reference_wrapper<scalar const>> const& source,
                               column_view const& indices,
                               table_view const& target,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::boolean_mask_scatter(
                      table_view const& source, table_view const& target,
 *                    column_view const& boolean_mask,
 *                    rmm::device_async_resource_ref mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> boolean_mask_scatter(table_view const& source,
                                            table_view const& target,
                                            column_view const& boolean_mask,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr);

/**
 * @copydoc cudf::boolean_mask_scatter(
 *                    std::vector<std::reference_wrapper<scalar>> const& source,
 *                    table_view const& target,
 *                    column_view const& boolean_mask,
 *                    rmm::device_async_resource_ref mr)
 *
 * @param stream CUDA stream used for device memory operations and kernel launches.
 */
std::unique_ptr<table> boolean_mask_scatter(
  std::vector<std::reference_wrapper<scalar const>> const& source,
  table_view const& target,
  column_view const& boolean_mask,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf

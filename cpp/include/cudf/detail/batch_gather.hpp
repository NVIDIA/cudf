/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>

#include <memory>

namespace CUDF_EXPORT cudf {
namespace detail {

/**
 * @brief Gathers the rows of a table into a new table, gathering all fixed-width columns with a
 * single kernel launch.
 *
 * Row `i` of each output column contains row `gather_map[i]` of the corresponding input column.
 * The number of rows in the result table equals `gather_map.size()`.
 *
 * The columns of `source_table` are partitioned host-side: every fixed-width column (including
 * fixed-point, which is gathered through its storage type) is gathered by one fused kernel that
 * processes all such columns in a single launch, while non-fixed-width columns (strings, lists,
 * structs, dictionaries) fall back to the stock per-column gather path. The fallback keeps
 * mixed-type tables correct by construction; a pure fixed-width table pays exactly one kernel
 * launch.
 *
 * A negative value `i` in the gather map is interpreted as `i + n`, where `n` is the number of
 * rows in `source_table`, when `neg_indices` is `negative_index_policy::ALLOWED`. When
 * `bounds_policy` is `out_of_bounds_policy::NULLIFY`, rows whose (wrapped) map entry is outside
 * `[0, n)` become null; under `out_of_bounds_policy::DONT_CHECK` out-of-bounds entries are
 * undefined behavior.
 *
 * The gather map must be a column of integral index types and must not contain nulls.
 *
 * @throws std::invalid_argument if `gather_map` contains nulls
 * @throws cudf::logic_error if `gather_map` is not an index type column
 *
 * @param[in] source_table View of the table whose rows will be gathered
 * @param[in] gather_map View of a non-nullable column of integral indices mapping rows in
 * `source_table` to rows in the result
 * @param[in] bounds_policy Whether out-of-bounds map entries are nullified or unchecked
 * @param[in] neg_indices Whether negative map entries wrap relative to the table end
 * @param[in] stream CUDA stream used for device memory operations and kernel launches
 * @param[in] mr Memory resources used for temporary allocations and the returned table
 * @return The gathered table, with columns in the order of `source_table`
 */
std::unique_ptr<table> batch_gather(table_view const& source_table,
                                    column_view const& gather_map,
                                    out_of_bounds_policy bounds_policy,
                                    negative_index_policy neg_indices,
                                    cuda::stream_ref stream,
                                    memory_resources mr);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf

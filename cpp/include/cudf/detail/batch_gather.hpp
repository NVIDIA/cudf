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
 * fixed-point, which is gathered through its storage type), every struct whose descendants are
 * all fixed-width, and every dictionary with a signed index type is gathered by one fused
 * kernel that processes all of them in a single launch — fully-fixed-width structs flatten into
 * per-leaf data descriptors plus mask-only descriptors for the struct nodes themselves, and
 * dictionaries contribute a data descriptor over their annotated indices with the gathered mask
 * landing on the dictionary parent. String columns join the fused path's scan+payload machinery
 * only below a row-count threshold; at or above it (and all lists, structs with any
 * non-fixed-width descendant, and dictionaries with unsigned indices) they fall back to the
 * stock per-column gather path. The fallback keeps mixed-type tables correct by construction; a
 * pure fixed-width table pays exactly one kernel launch.
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

/**
 * @copydoc cudf::detail::batch_gather(table_view const&, column_view const&,
 * out_of_bounds_policy, negative_index_policy, cuda::stream_ref, memory_resources)
 *
 * @param[in] tile_cols Number of consecutive columns per blockIdx.y tile in the fused kernel;
 * one of {1, 2, 4, 8}. 1 degenerates to one column per y block.
 * @throws std::invalid_argument if `tile_cols` is not one of {1, 2, 4, 8}
 */
std::unique_ptr<table> batch_gather(table_view const& source_table,
                                    column_view const& gather_map,
                                    out_of_bounds_policy bounds_policy,
                                    negative_index_policy neg_indices,
                                    cuda::stream_ref stream,
                                    memory_resources mr,
                                    size_type tile_cols);

/**
 * @brief Classifies the gather map's locality and returns the fused kernel tile width to use.
 *
 * Sampling is skipped entirely — returning the unsampled width without touching the map — when
 * the tile width is measured K-insensitive: fewer than 2 fused descriptors (K1 always), a
 * source/map shape whose sampled work `num_rows x num_fused_cols` is below 4M or with fewer
 * than 262144 rows (K2 while two fused columns plus the map fit half of the device L2, else
 * K1). When sampling runs, a small number of windows is copied device-to-host in one batched
 * copy plus one stream synchronization (no kernel launch) and 8 is reported only when every
 * sampled window spans at most one window length of source rows — near-sorted or contiguous
 * maps, in either direction; anything else returns 1, the safe choice given that misrouting a
 * random map to the wide tile costs far more than the reverse.
 *
 * @param[in] source_table View of the table being gathered (row count feeds the L2 check)
 * @param[in] gather_map View of the gather map column to classify
 * @param[in] num_fused_cols Flattened descriptor count of the fused columns in
 * `source_table` — struct leaves and mask-only struct nodes each count, every dictionary
 * counts one for its index payload — matching the kernel's `num_descriptors` once the build
 * walk flattens the same columns
 * @param[in] max_fused_element_size Largest element size in bytes among the fused columns'
 * data payloads (mask-only struct nodes carry none)
 * @param[in] stream CUDA stream used for the sample copy and its synchronization
 * @return Tile width in {1, 2, 8} for the `batch_gather` tile_cols overload
 */
size_type select_gather_map_tile_cols(table_view const& source_table,
                                      column_view const& gather_map,
                                      size_type num_fused_cols,
                                      size_type max_fused_element_size,
                                      cuda::stream_ref stream);

}  // namespace detail
}  // namespace CUDF_EXPORT cudf

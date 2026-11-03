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

/**
 * @copydoc cudf::detail::batch_gather(table_view const&, column_view const&,
 * out_of_bounds_policy, negative_index_policy, cuda::stream_ref, memory_resources)
 *
 * @param[in] tile_cols Number of consecutive columns per blockIdx.y tile in the fused kernel;
 * one of {1, 2, 4, 8}. 1 degenerates to one column per y block.
 * @param[in] block_size Threads per blockIdx.x row block in the fused kernel; one of
 * {128, 256, 512}. 256 matches the routed entry. This is a measurement axis, not a routing
 * decision: every supported size is a multiple of the warp size, so the destination mask-word
 * exactly-once invariant holds and correctness is identical across the set.
 * @throws std::invalid_argument if `tile_cols` is not one of {1, 2, 4, 8} or `block_size` is
 * not one of {128, 256, 512}
 */
std::unique_ptr<table> batch_gather(table_view const& source_table,
                                    column_view const& gather_map,
                                    out_of_bounds_policy bounds_policy,
                                    negative_index_policy neg_indices,
                                    cuda::stream_ref stream,
                                    memory_resources mr,
                                    size_type tile_cols,
                                    size_type block_size);

/**
 * @brief Classifies the gather map's locality and returns the fused kernel tile width to use.
 *
 * Sampling is skipped entirely — returning the unsampled width without touching the map — when
 * the tile width is measured K-insensitive: fewer than 2 fused columns (K1 always), a
 * source/map shape whose sampled work `num_rows x num_fused_cols` is below 4M or with fewer
 * than 262144 rows (K2 while two fused columns plus the map fit half of the device L2, else
 * K1). When sampling runs, a small number of windows is copied device-to-host in a pitched
 * copy (plus an endpoint-window copy when the window grid is not uniform) and one stream
 * synchronization (no kernel launch); 8 is reported only when every
 * sampled window spans at most one window length of source rows — near-sorted or contiguous
 * maps, in either direction; anything else returns 1, the safe choice given that misrouting a
 * random map to the wide tile costs far more than the reverse.
 *
 * @param[in] source_table View of the table being gathered (row count feeds the L2 check)
 * @param[in] gather_map View of the gather map column to classify
 * @param[in] num_fused_cols Number of fixed-width columns the fused kernel will process
 * @param[in] max_fused_element_size Largest element size in bytes among the fused columns
 * @param[in] stream CUDA stream used for the sample copy and its synchronization
 * @return Tile width in {1, 2, 8} for the `batch_gather` tile_cols parameter
 */
size_type select_gather_map_tile_cols(table_view const& source_table,
                                      column_view const& gather_map,
                                      size_type num_fused_cols,
                                      size_type max_fused_element_size,
                                      cuda::stream_ref stream);

/**
 * @brief Descriptor counts for the two by-value parameter tiers of the fused kernel
 *
 * A table whose fixed-width column count fits one of these caps passes its descriptors as one
 * by-value kernel parameter instead of a device array, which removes the staging allocation,
 * the device allocation, and the host-to-device copy from every call. A descriptor block costs
 * `16 + 48 * kCap` bytes of the 32'764-byte kernel-parameter budget (the small tier 784,
 * the wide tier 24'592); the driver copies the whole fixed-size parameter block on every
 * launch, so the tiers stay below it — the arithmetic ceiling is 680 descriptors.
 * The small tier keeps the launch copy tiny for the short-table/latency-bound shapes; the
 * wide tier pulls the 256/512-column tables off the device-array path, whose staging cost
 * grows with the column count.
 */
constexpr size_type batch_gather_small_by_value_descriptors{16};
constexpr size_type batch_gather_wide_by_value_descriptors{512};

/**
 * @brief Which by-value tier, if any, carries `num_descriptors` fused columns.
 *
 * This is the tier dispatch used by the fused kernel launch; both capacities are inclusive.
 *
 * @param[in] num_descriptors Number of fixed-width columns the fused kernel will process
 * @return 0 for the device-array path, 1 for the small by-value tier, 2 for the wide tier
 */
constexpr int batch_gather_descriptors_by_value_tier(size_type num_descriptors)
{
  if (num_descriptors <= batch_gather_small_by_value_descriptors) { return 1; }
  return num_descriptors <= batch_gather_wide_by_value_descriptors ? 2 : 0;
}

}  // namespace detail
}  // namespace CUDF_EXPORT cudf

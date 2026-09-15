/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace cudf::groupby::detail::hash {

/**
 * @brief Whether the hash groupby can compute the single-pass aggregation `kind` on values of
 * type `values_type` (the keys type for dictionary values).
 */
bool is_single_pass_agg_supported(data_type values_type, aggregation::Kind kind);

/**
 * @brief Input rows reordered so that the rows of every group are contiguous, together with the
 * arrays of the reduction strategy chosen for the group size distribution.
 *
 * Every group is reduced as a segment: groups that are small on average are packed several per
 * block, one thread or one sub-warp each, otherwise a block reduces each group. Groups spanning
 * more than one chunk of rows are first reduced per chunk, so that a few long groups still occupy
 * the whole device and no thread walks a long group alone.
 */
struct grouped_rows {
  device_span<size_type const> rows;             ///< Input row index at each grouped position
  device_span<size_type const> offsets;          ///< `num_groups + 1` offsets delimiting the groups
  rmm::device_uvector<size_type> chunk_offsets;  ///< `num_chunks + 1` chunk boundaries, only when
                                                 ///< some group spans several chunks
  rmm::device_uvector<size_type> group_chunks;   ///< `num_groups + 1` offsets into the chunks, only
                                                 ///< when some group spans several chunks
  size_type packed_rows = 0;  ///< Average rows per group when the segments are packed several per
                              ///< block; 0 when every segment gets a block
};

/**
 * @brief Chooses the reduction strategy for the grouped rows and builds its arrays.
 *
 * @param rows Input row index at each grouped position
 * @param offsets `num_groups + 1` offsets delimiting the groups
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resources used to allocate the returned arrays and temporary storage
 * @return Grouped rows with the arrays required by the chosen reduction strategy
 */
grouped_rows make_grouped_rows(device_span<size_type const> rows,
                               device_span<size_type const> offsets,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr);

/**
 * @brief Computes one single-pass aggregation per values column as a reduction over the grouped
 * rows.
 *
 * Results of aggregations that only feed a compound aggregation are created without a null mask.
 *
 * @param values One values column per aggregation
 * @param agg_kinds The aggregation to compute on each values column
 * @param is_agg_intermediate Whether each aggregation is only an intermediate result
 * @param grouped The input rows grouped by key
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resources used to allocate the result columns and temporary storage
 * @return One result column per aggregation with one row per group
 */
std::vector<std::unique_ptr<column>> compute_single_pass_aggs(
  table_view const& values,
  host_span<aggregation::Kind const> agg_kinds,
  std::span<int8_t const> is_agg_intermediate,
  grouped_rows const& grouped,
  cuda::stream_ref stream,
  cudf::memory_resources mr);

}  // namespace cudf::groupby::detail::hash

/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/column/column.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/array>
#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace cudf::groupby::detail::hash {

/// Supported value reductions. Counts use the separate row-count path.
template <typename T>
constexpr bool is_reduction_supported(aggregation::Kind kind)
{
  switch (kind) {
    case aggregation::SUM: return cudf::detail::is_valid_aggregation<T, aggregation::SUM>();
    case aggregation::PRODUCT: return cudf::detail::is_valid_aggregation<T, aggregation::PRODUCT>();
    case aggregation::SUM_OF_SQUARES:
      return cudf::detail::is_valid_aggregation<T, aggregation::SUM_OF_SQUARES>();
    case aggregation::M2: return cudf::is_numeric<T>() && !cudf::is_fixed_point<T>();
    case aggregation::SUM_OVERFLOW:
      return cudf::detail::is_valid_aggregation<T, aggregation::SUM_OVERFLOW>();
    // Target-type validity alone does not constrain extrema's storage or comparisons.
    case aggregation::MIN:
    case aggregation::MAX: return cudf::is_fixed_width<T>() && is_relationally_comparable<T, T>();
    case aggregation::ARGMIN:
    case aggregation::ARGMAX: return is_relationally_comparable<T, T>() || cudf::is_nested<T>();
    default: return false;
  }
}

/**
 * @brief Whether the hash groupby can compute the single-pass aggregation `kind` on values of
 * type `values_type` (the keys type for dictionary values).
 */
bool is_single_pass_agg_supported(data_type values_type, aggregation::Kind kind);

/**
 * @brief Input rows reordered so that the rows of every group are contiguous, together with the
 * arrays used by the chunked reductions.
 *
 * Small groups use scalar folds, bounded groups use warp reductions, and long groups
 * use block-reduced chunks. Group IDs preserve the original output order.
 */
struct grouped_rows {
  device_span<size_type const> rows;     ///< Input row index at each grouped position
  device_span<size_type const> offsets;  ///< Group boundaries in rows
  rmm::device_uvector<size_type>
    warp_groups;  ///< Group IDs wider than a warp: direct groups, then long groups
  rmm::device_uvector<size_type>
    group_chunks;  ///< Offsets of long groups in chunk_ranges, including the final offset
  rmm::device_uvector<cuda::std::array<size_type, 2>>
    chunk_ranges;                              ///< CSR begin/end positions of each long-group chunk
  rmm::device_uvector<size_type> chunk_order;  ///< Long chunk IDs ordered by first stored input row
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

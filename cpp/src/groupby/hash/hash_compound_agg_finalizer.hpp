/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

namespace cudf::groupby::detail::hash {

/**
 * @brief Functor to finalize compound aggregations in hash groupby.
 */
struct hash_compound_agg_finalizer {
  column_view const col;
  data_type const input_type;
  cudf::detail::result_cache* const cache;
  cuda::stream_ref const stream;
  cudf::memory_resources const mr;

  hash_compound_agg_finalizer(column_view const& col,
                              cudf::detail::result_cache* cache,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr);

  // Default case: no-op
  template <aggregation::Kind k>
  void operator()(aggregation const& agg) const
  {
  }
};

// Declare specializations
template <>
void hash_compound_agg_finalizer::operator()<aggregation::MIN>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer::operator()<aggregation::MAX>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer::operator()<aggregation::VARIANCE>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer::operator()<aggregation::STD>(aggregation const& agg) const;

/**
 * @brief Finalizes each single-pass MEAN in place by dividing its sum by the valid count of the
 * same values column.
 *
 * @param values The values column of each single-pass result, identifying results of one input
 * @param kinds The single-pass aggregation of each result
 * @param results The single-pass results, whose MEAN sums become the means
 * @param cache Cache of earlier results, searched for a count that is not among `results`, or
 * `nullptr` when `results` always contain the counts
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
void finalize_means(table_view const& values,
                    host_span<aggregation::Kind const> kinds,
                    mutable_table_view const& results,
                    cudf::detail::result_cache const* cache,
                    cuda::stream_ref stream);

}  // namespace cudf::groupby::detail::hash

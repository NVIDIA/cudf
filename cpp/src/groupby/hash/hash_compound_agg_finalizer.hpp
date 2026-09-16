/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/types.hpp>

namespace cudf::groupby::detail::hash {

/**
 * @brief Functor to finalize compound aggregations in hash groupby.
 */
struct hash_compound_agg_finalizer {
  column_view const col;
  data_type const input_type;
  cudf::detail::result_cache* const cache;
  bitmask_type const* const d_row_bitmask;
  cuda::stream_ref const stream;
  rmm::device_async_resource_ref const mr;

  hash_compound_agg_finalizer(column_view const& col,
                              cudf::detail::result_cache* cache,
                              bitmask_type const* d_row_bitmask,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr);

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
void hash_compound_agg_finalizer::operator()<aggregation::MEAN>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer::operator()<aggregation::M2>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer::operator()<aggregation::VARIANCE>(aggregation const& agg) const;

template <>
void hash_compound_agg_finalizer::operator()<aggregation::STD>(aggregation const& agg) const;

}  // namespace cudf::groupby::detail::hash

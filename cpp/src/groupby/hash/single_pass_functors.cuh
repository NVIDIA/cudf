/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/device_aggregators.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>

#include <cstdint>

namespace cudf::groupby::detail::hash {
struct compute_single_pass_aggs_base_fn {
  aggregation::Kind const* aggs;
  table_device_view input_values;
  mutable_table_device_view output_values;

  compute_single_pass_aggs_base_fn(aggregation::Kind const* aggs,
                                   table_device_view const& input_values,
                                   mutable_table_device_view const& output_values)
    : aggs(aggs), input_values(input_values), output_values(output_values)
  {
  }
};

/**
 * @brief Functor to compute single-pass aggregations and store the results into an output table,
 * executing for all input rows.
 *
 * This functor writes output to the final dense output table, using the given pre-computed target
 * indices. In addition, all aggregations for all rows are computed concurrently without any order.
 */
struct compute_single_pass_aggs_dense_output_fn : compute_single_pass_aggs_base_fn {
  size_type const* target_indices;

  compute_single_pass_aggs_dense_output_fn(size_type const* target_indices,
                                           aggregation::Kind const* aggs,
                                           table_device_view const& input_values,
                                           mutable_table_device_view const& output_values)
    : compute_single_pass_aggs_base_fn(aggs, input_values, output_values),
      target_indices(target_indices)
  {
  }

  __device__ void operator()(int64_t idx) const
  {
    auto const num_rows       = input_values.num_rows();
    auto const source_row_idx = static_cast<size_type>(idx % num_rows);
    if (auto const target_row_idx = target_indices[source_row_idx];
        target_row_idx != cudf::detail::CUDF_SIZE_TYPE_SENTINEL) {
      auto const col_idx     = static_cast<size_type>(idx / num_rows);
      auto const& source_col = input_values.column(col_idx);
      auto const& target_col = output_values.column(col_idx);
      dispatch_type_and_aggregation(source_col.type(),
                                    aggs[col_idx],
                                    cudf::detail::element_aggregator{},
                                    target_col,
                                    target_row_idx,
                                    source_col,
                                    source_row_idx);
    }
  }
};

}  // namespace cudf::groupby::detail::hash

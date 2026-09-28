/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby_results.hpp"

#include <cudf/aggregation.hpp>
#include <cudf/binaryop.hpp>
#include <cudf/copying.hpp>
#include <cudf/filling.hpp>
#include <cudf/groupby.hpp>
#include <cudf/reduction.hpp>
#include <cudf/reshape.hpp>
#include <cudf/scalar/scalar.hpp>

std::unique_ptr<cudf::table> compute_results(
  cudf::column_view const& cities,
  cudf::column_view const& temperatures,
  std::vector<std::unique_ptr<cudf::groupby_aggregation>>&& aggregations,
  cuda::stream_ref stream)
{
  auto groupby_obj      = cudf::groupby::groupby(cudf::table_view({cities}));
  auto aggregation_reqs = std::vector<cudf::groupby::aggregation_request>{};
  auto& req             = aggregation_reqs.emplace_back();
  req.values            = temperatures;
  req.aggregations      = std::move(aggregations);

  auto result = groupby_obj.aggregate(aggregation_reqs, stream);

  auto rtn = result.first->release();
  for (auto& r : result.second.front().results) {
    rtn.emplace_back(std::move(r));
  }

  return std::make_unique<cudf::table>(std::move(rtn));
}

std::unique_ptr<cudf::table> compute_final_aggregates(
  std::vector<std::unique_ptr<cudf::table>>& agg_data, cuda::stream_ref stream)
{
  // first combine all the results into a vectors of columns
  std::vector<cudf::column_view> min_cols, max_cols, sum_cols, count_cols;
  for (auto& tbl : agg_data) {
    auto const tv = tbl->view();
    min_cols.push_back(tv.column(1));
    max_cols.push_back(tv.column(2));
    sum_cols.push_back(tv.column(3));
    count_cols.push_back(tv.column(4));
  }

  // Create single columns out of the aggregate table results.
  // This relies on every key appearing in every chunk segment.
  // All the values for each key become contiguous within the output column.
  // For example, for N=min_cols.size() (number of unique cities):
  //   All of the mins for city[i] are in row[i] of each column of vector min_cols.
  //   The interleave_columns API transposes these into a single column where
  //   the first N rows are values for city[0],
  //   the next N rows are values for city[1],
  //   ...
  //   the last N rows are values for city[N-1]
  // The final result for each city is computed using segmented_reduce.
  auto mins   = cudf::interleave_columns(cudf::table_view{min_cols});
  auto maxes  = cudf::interleave_columns(cudf::table_view{max_cols});
  auto sums   = cudf::interleave_columns(cudf::table_view{sum_cols});
  auto counts = cudf::interleave_columns(cudf::table_view{count_cols});

  // Build the offsets needed for segmented reduce.
  // These are increasing integer values spaced evenly as per the number of cities (keys).
  auto const num_keys = agg_data.front()->num_rows();
  auto const size     = static_cast<cudf::size_type>(num_keys) + 1;
  auto const start    = cudf::numeric_scalar<cudf::size_type>(0, true, stream);
  auto const step     = cudf::numeric_scalar<cudf::size_type>(agg_data.size(), true, stream);
  auto seg_offsets    = cudf::sequence(size, start, step, stream);
  auto offsets_span   = cudf::device_span<cudf::size_type const>(seg_offsets->view());

  // compute the min/max for each key by using segmented reduce
  auto min_agg = cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
  mins         = cudf::segmented_reduce(
    mins->view(), offsets_span, *min_agg, mins->type(), cudf::null_policy::EXCLUDE, stream);
  auto max_agg = cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
  maxes        = cudf::segmented_reduce(
    maxes->view(), offsets_span, *max_agg, maxes->type(), cudf::null_policy::EXCLUDE, stream);

  // compute the sum and total counts in the same way
  auto sum_agg = cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>();
  sums         = cudf::segmented_reduce(
    sums->view(), offsets_span, *sum_agg, sums->type(), cudf::null_policy::EXCLUDE, stream);
  counts = cudf::segmented_reduce(
    counts->view(), offsets_span, *sum_agg, counts->type(), cudf::null_policy::EXCLUDE, stream);

  // compute the means using binary-operation to divide the individual rows sum/count
  auto means = cudf::binary_operation(
    sums->view(), counts->view(), cudf::binary_operator::DIV, sums->type(), stream);

  std::vector<std::unique_ptr<cudf::column>> results;
  results.emplace_back(std::move(mins));
  results.emplace_back(std::move(maxes));
  results.emplace_back(std::move(means));
  return std::make_unique<cudf::table>(std::move(results));
}

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "groupby/hash/group_reductions.hpp"
#include "groupby/hash/grouped_reductions.cuh"
#include "groupby/hash/merge_m2.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/null_mask.hpp>
#include <cudf/structs/structs_column_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>
#include <cuda/stream>

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

using result_type = double;
static_assert(
  std::is_same_v<cudf::detail::target_type_t<result_type, aggregation::Kind::M2>, result_type>);

/**
 * @brief Functor to read the partial `COUNT_VALID`, `MEAN`, and `M2` results of a grouped row.
 */
template <typename CountType>
struct grouped_m2_state {
  size_type const* rows;
  CountType const* counts;
  result_type const* means;
  result_type const* m2s;

  __device__ hash::m2_state<CountType> operator()(size_type position) const
  {
    auto const row   = rows[position];
    auto const count = counts[row];
    if (count == CountType{0}) { return {CountType{0}, 0.0, 0.0}; }
    return {count, means[row], m2s[row]};
  }
};

template <typename CountType>
struct split_m2_state {
  __device__ cuda::std::tuple<CountType, result_type, result_type> operator()(
    hash::m2_state<CountType> const& state) const
  {
    return {state.count, state.mean, state.m2};
  }
};

template <typename CountType>
std::unique_ptr<column> merge_m2(column_view const& values,
                                 hash::grouped_rows const& grouped,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr)
{
  auto const num_groups = static_cast<size_type>(grouped.offsets.size() - 1);
  auto const output_mr  = mr.get_output_mr();
  auto result_counts    = make_numeric_column(
    data_type(type_to_id<CountType>()), num_groups, mask_state::UNALLOCATED, stream, output_mr);
  auto result_means = make_numeric_column(
    data_type(type_to_id<result_type>()), num_groups, mask_state::UNALLOCATED, stream, output_mr);
  auto result_M2s = make_numeric_column(
    data_type(type_to_id<result_type>()), num_groups, mask_state::UNALLOCATED, stream, output_mr);

  if (num_groups > 0) {
    auto const partials     = structs_column_view{values};
    auto const state_values = cudf::detail::make_counting_transform_iterator(
      0,
      grouped_m2_state<CountType>{grouped.rows.data(),
                                  partials.get_sliced_child(0, stream).begin<CountType>(),
                                  partials.get_sliced_child(1, stream).begin<result_type>(),
                                  partials.get_sliced_child(2, stream).begin<result_type>()});
    auto const output = cuda::transform_output_iterator{
      cuda::make_zip_iterator(result_counts->mutable_view().begin<CountType>(),
                              result_means->mutable_view().begin<result_type>(),
                              result_M2s->mutable_view().begin<result_type>()),
      split_m2_state<CountType>{}};
    hash::reduce_groups(grouped,
                        state_values,
                        output,
                        hash::merge_m2<CountType>{},
                        hash::m2_state<CountType>{CountType{0}, 0.0, 0.0},
                        stream,
                        mr);
  }

  // Output is a structs column containing the merged values of `COUNT_VALID`, `MEAN`, and `M2`.
  std::vector<std::unique_ptr<column>> out_columns;
  out_columns.emplace_back(std::move(result_counts));
  out_columns.emplace_back(std::move(result_means));
  out_columns.emplace_back(std::move(result_M2s));
  return make_structs_column(
    num_groups,
    std::move(out_columns),
    0,
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, output_mr),
    stream,
    output_mr);
}

}  // namespace

std::unique_ptr<column> group_merge_m2(column_view const& values,
                                       hash::grouped_rows const& grouped,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  CUDF_EXPECTS(values.type().id() == type_id::STRUCT,
               "Input to `group_merge_m2` must be a structs column.");
  CUDF_EXPECTS(values.num_children() == 3,
               "Input to `group_merge_m2` must be a structs column having 3 children columns.");

  // The input column stores tuples of values (`COUNT_VALID`, `MEAN`, and `M2`).
  // However, the data type for `COUNT_VALID` must be wide enough such as
  // `INT64` or `FLOAT64` to prevent overflow when summing up.
  // For Apache Spark, the data type used for storing this is `FLOAT64`.
  auto const count_type_id = values.child(0).type().id();
  CUDF_EXPECTS((count_type_id == type_id::INT64 || count_type_id == type_id::FLOAT64) &&
                 values.child(1).type().id() == type_to_id<result_type>() &&
                 values.child(2).type().id() == type_to_id<result_type>(),
               "Input to `group_merge_m2` has invalid children type.");

  return count_type_id == type_id::INT64 ? merge_m2<int64_t>(values, grouped, stream, mr)
                                         : merge_m2<result_type>(values, grouped, stream, mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf

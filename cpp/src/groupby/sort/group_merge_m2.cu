/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/transform.h>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

using result_type = double;
static_assert(
  std::is_same_v<cudf::detail::target_type_t<result_type, aggregation::Kind::M2>, result_type>);

/**
 * @brief Functor to merge partial results of `COUNT_VALID`, `MEAN`, and `M2` aggregations
 * for a given group (key) index.
 */
template <typename count_type>
struct merge_fn {
  size_type const* d_offsets;
  count_type const* d_counts;
  result_type const* d_means;
  result_type const* d_M2s;

  __device__ cuda::std::tuple<count_type, result_type, result_type> operator()(
    size_type const group_idx) const
  {
    count_type n{0};
    result_type avg{0};
    result_type m2{0};

    auto const start_idx = d_offsets[group_idx], end_idx = d_offsets[group_idx + 1];
    for (auto idx = start_idx; idx < end_idx; ++idx) {
      auto const partial_n = d_counts[idx];
      if (partial_n == 0) { continue; }
      auto const partial_avg = d_means[idx];
      auto const partial_m2  = d_M2s[idx];

      // Merging an empty accumulator with a non-empty partial is an identity operation. Running
      // the generic formula for this case can evaluate inf * 0 and turn extreme finite partials
      // into NaN.
      if (n == 0) {
        n   = partial_n;
        avg = partial_avg;
        m2  = partial_m2;
        continue;
      }

      auto const new_n   = n + partial_n;
      auto const delta   = partial_avg - avg;
      auto const delta_n = delta / new_n;
      m2 += partial_m2 + delta * delta_n * n * partial_n;
      avg += delta_n * partial_n;
      n = new_n;
    }

    return {n, avg, m2};
  }
};

template <typename count_type>
std::unique_ptr<column> merge_m2(column_view const& values,
                                 device_span<size_type const> group_offsets,
                                 size_type num_groups,
                                 rmm::cuda_stream_view stream,
                                 rmm::device_async_resource_ref mr)
{
  auto result_counts = make_numeric_column(
    data_type(type_to_id<count_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto result_means = make_numeric_column(
    data_type(type_to_id<result_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);
  auto result_M2s = make_numeric_column(
    data_type(type_to_id<result_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);

  auto const out_iter =
    cuda::make_zip_iterator(result_counts->mutable_view().template data<count_type>(),
                            result_means->mutable_view().template data<result_type>(),
                            result_M2s->mutable_view().template data<result_type>());

  auto const count_valid = values.child(0);
  auto const mean_values = values.child(1);
  auto const M2_values   = values.child(2);
  auto const iter        = cuda::counting_iterator<size_type>{0};

  auto const fn = merge_fn<count_type>{group_offsets.data(),
                                       count_valid.template begin<count_type>(),
                                       mean_values.template begin<result_type>(),
                                       M2_values.template begin<result_type>()};
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    iter,
                    iter + num_groups,
                    out_iter,
                    fn);

  // Output is a structs column containing the merged values of `COUNT_VALID`, `MEAN`, and `M2`.
  std::vector<std::unique_ptr<column>> out_columns;
  out_columns.emplace_back(std::move(result_counts));
  out_columns.emplace_back(std::move(result_means));
  out_columns.emplace_back(std::move(result_M2s));
  return make_structs_column(
    num_groups, std::move(out_columns), 0, rmm::device_buffer{0, stream, mr}, stream, mr);
}

}  // namespace

std::unique_ptr<column> group_merge_m2(column_view const& values,
                                       device_span<size_type const> group_offsets,
                                       size_type num_groups,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
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

  return count_type_id == type_id::INT64
           ? merge_m2<int64_t>(values, group_offsets, num_groups, stream, mr)
           : merge_m2<result_type>(values, group_offsets, num_groups, stream, mr);
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf

/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/aggregation.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>

namespace cudf {
namespace groupby {
namespace detail {
namespace {

template <bool has_nested_columns, typename Nullate>
struct is_unique_iterator_fn {
  using comparator_type =
    typename cudf::detail::row::equality::device_row_comparator<has_nested_columns, Nullate>;

  Nullate nulls;
  column_device_view const v;
  comparator_type equal;
  null_policy null_handling;
  size_type const* group_offsets;
  size_type const* group_labels;

  is_unique_iterator_fn(Nullate nulls,
                        column_device_view const& v,
                        comparator_type const& equal,
                        null_policy null_handling,
                        size_type const* group_offsets,
                        size_type const* group_labels)
    : nulls{nulls},
      v{v},
      equal{equal},
      null_handling{null_handling},
      group_offsets{group_offsets},
      group_labels{group_labels}
  {
  }

  __device__ size_type operator()(size_type i) const
  {
    auto const is_input_countable =
      !nulls || (null_handling == null_policy::INCLUDE || v.is_valid_nocheck(i));
    auto const is_unique =
      is_input_countable && (group_offsets[group_labels[i]] == i ||  // first element or
                             (not equal(i, i - 1)));                 // new unique value in sorted
    return static_cast<size_type>(is_unique);
  }
};
}  // namespace

std::unique_ptr<column> group_nunique(column_view const& values,
                                      cudf::device_span<size_type const> group_labels,
                                      size_type const num_groups,
                                      cudf::device_span<size_type const> group_offsets,
                                      null_policy null_handling,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(num_groups >= 0, "number of groups cannot be negative");
  CUDF_EXPECTS(static_cast<size_t>(values.size()) == group_labels.size(),
               "Size of values column should be same as that of group labels");

  auto result = make_numeric_column(
    data_type(type_to_id<size_type>()), num_groups, mask_state::UNALLOCATED, stream, mr);

  if (num_groups == 0) { return result; }

  auto temp_mr           = cudf::get_current_device_resource_ref();
  auto const values_view = table_view{{values}};
  auto const comparator =
    cudf::detail::row::equality::self_comparator{values_view, stream, temp_mr};

  auto const d_values_view = column_device_view::create(values, stream);

  auto d_result = rmm::device_uvector<size_type>(group_labels.size(), stream);

  auto const comparator_helper = [&](auto const d_equal) {
    auto fn = is_unique_iterator_fn{nullate::DYNAMIC{values.has_nulls()},
                                    *d_values_view,
                                    d_equal,
                                    null_handling,
                                    group_offsets.data(),
                                    group_labels.data()};
    thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{values.size()},
                      d_result.begin(),
                      fn);
  };

  if (cudf::detail::has_nested_columns(values_view)) {
    auto const d_equal = comparator.equal_to<true>(
      cudf::nullate::DYNAMIC{cudf::has_nested_nulls(values_view)}, null_equality::EQUAL);
    comparator_helper(d_equal);
  } else {
    auto const d_equal = comparator.equal_to<false>(
      cudf::nullate::DYNAMIC{cudf::has_nested_nulls(values_view)}, null_equality::EQUAL);
    comparator_helper(d_equal);
  }

  // calling this with a vector instead of a transform iterator is 10x faster to compile;
  // it also helps that we are only calling it once for both conditions
  cudf::detail::reduce_by_key_async(group_labels.begin(),
                                    group_labels.end(),
                                    d_result.begin(),
                                    cuda::make_discard_iterator(),
                                    result->mutable_view().begin<size_type>(),
                                    cuda::std::plus<size_type>(),
                                    stream);

  return result;
}

}  // namespace detail
}  // namespace groupby
}  // namespace cudf

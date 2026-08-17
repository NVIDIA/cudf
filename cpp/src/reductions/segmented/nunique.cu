/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "update_validity.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/labeling/label_segments.cuh>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/reduction/detail/segmented_reduction.cuh>
#include <cudf/reduction/detail/segmented_reduction_functions.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace reduction {
namespace detail {
namespace {
template <typename ComparatorType>
struct is_unique_fn {
  column_device_view const d_col;
  ComparatorType row_equal;
  null_policy null_handling;
  size_type const* offsets;
  size_type const* labels;

  __device__ size_type operator()(size_type idx) const
  {
    if (null_handling == null_policy::EXCLUDE && d_col.is_null(idx)) { return 0; }
    return static_cast<size_type>(offsets[labels[idx]] == idx || (!row_equal(idx, idx - 1)));
  }
};
}  // namespace

std::unique_ptr<cudf::column> segmented_nunique(column_view const& col,
                                                device_span<size_type const> offsets,
                                                null_policy null_handling,
                                                cuda::stream_ref stream,
                                                rmm::device_async_resource_ref mr)
{
  // only support non-nested types
  CUDF_EXPECTS(!cudf::is_nested(col.type()),
               "segmented reduce nunique only supports non-nested column types");

  // compute the unique identifiers within each segment
  auto const identifiers = [&] {
    auto const d_col      = column_device_view::create(col, stream);
    auto const comparator = cudf::detail::row::equality::self_comparator{table_view({col}), stream};
    auto const row_equal =
      comparator.equal_to<false>(cudf::nullate::DYNAMIC{col.has_nulls()}, null_equality::EQUAL);

    auto labels = rmm::device_uvector<size_type>(col.size(), stream);
    cudf::detail::label_segments(
      offsets.begin(), offsets.end(), labels.begin(), labels.end(), stream);
    auto fn = is_unique_fn<decltype(row_equal)>{
      *d_col, row_equal, null_handling, offsets.data(), labels.data()};

    auto identifiers = rmm::device_uvector<size_type>(col.size(), stream);
    thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{col.size()},
                      identifiers.begin(),
                      fn);
    return identifiers;
  }();

  auto result = cudf::make_numeric_column(data_type(type_to_id<size_type>()),
                                          static_cast<size_type>(offsets.size() - 1),
                                          cudf::mask_state::UNALLOCATED,
                                          stream,
                                          mr);

  // Sum the unique identifiers within each segment
  auto add_op = op::sum{};
  cudf::reduction::detail::segmented_reduce(identifiers.begin(),
                                            offsets.begin(),
                                            offsets.end(),
                                            result->mutable_view().data<size_type>(),
                                            add_op.get_binary_op(),
                                            0,
                                            stream);

  // Compute the output null mask
  // - only empty segments are tagged as null
  // - nulls are counted appropriately above per null_handling policy
  auto const bitmask_col = null_handling == null_policy::EXCLUDE ? col : result->view();
  cudf::reduction::detail::segmented_update_validity(
    *result, bitmask_col, offsets, null_policy::EXCLUDE, std::nullopt, stream, mr);

  return result;
}
}  // namespace detail
}  // namespace reduction
}  // namespace cudf

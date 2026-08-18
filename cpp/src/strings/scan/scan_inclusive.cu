/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/utilities/device_operators.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/scan.h>
#include <thrust/scatter.h>

namespace cudf {
namespace strings {
namespace detail {
namespace {

/**
 * @brief Min/Max inclusive scan operator
 *
 * This operator will accept index values, check them and then
 * run the `Op` operation on the individual element objects.
 * The returned result is the appropriate index value.
 *
 * This was specifically created to workaround a thrust issue
 * https://github.com/NVIDIA/thrust/issues/1479
 * where invalid values are passed to the operator.
 */
template <typename Element, typename Op>
struct min_max_scan_operator {
  column_device_view const col;      ///< strings column device view
  Element const null_replacement{};  ///< value used when element is null
  bool const has_nulls;              ///< true if col has null elements

  min_max_scan_operator(column_device_view const& col, bool has_nulls = true)
    : col{col}, null_replacement{Op::template identity<Element>()}, has_nulls{has_nulls}
  {
    // verify validity bitmask is non-null, otherwise, is_null_nocheck() will crash
    if (has_nulls) CUDF_EXPECTS(col.nullable(), "column with nulls must have a validity bitmask");
  }

  __device__ inline size_type operator()(size_type lhs, size_type rhs) const
  {
    // thrust::inclusive_scan may pass us garbage values so we need to protect ourselves;
    // in these cases the return value does not matter since the result is not used
    if (lhs < 0 || rhs < 0 || lhs >= col.size() || rhs >= col.size()) return 0;
    Element d_lhs =
      has_nulls && col.is_null_nocheck(lhs) ? null_replacement : col.element<Element>(lhs);
    Element d_rhs =
      has_nulls && col.is_null_nocheck(rhs) ? null_replacement : col.element<Element>(rhs);
    return Op{}(d_lhs, d_rhs) == d_lhs ? lhs : rhs;
  }
};

struct null_iterator {
  bitmask_type const* mask;
  __device__ bool operator()(size_type idx) const { return !bit_is_set(mask, idx); }
};

}  // namespace

template <typename Op>
std::unique_ptr<column> scan_inclusive(column_view const& input,
                                       bitmask_type const* mask,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  auto d_input = column_device_view::create(input, stream);

  // build indices of the scan operation results
  rmm::device_uvector<size_type> result_map(input.size(), stream);
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         cuda::counting_iterator<size_type>{0},
                         cuda::counting_iterator<size_type>{input.size()},
                         result_map.begin(),
                         min_max_scan_operator<cudf::string_view, Op>{*d_input, input.has_nulls()});

  if (input.has_nulls()) {
    // fill the null rows with out-of-bounds values so gather records them as null;
    // this prevents un-sanitized null entries in the output
    auto null_itr = cudf::detail::make_counting_transform_iterator(0, null_iterator{mask});
    auto oob_val  = cuda::constant_iterator<size_type>(input.size());
    thrust::scatter_if(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       oob_val,
                       oob_val + input.size(),
                       cuda::counting_iterator<size_type>{0},
                       null_itr,
                       result_map.data());
  }

  // call gather using the indices to build the output column
  auto result_table = cudf::detail::gather(cudf::table_view({input}),
                                           result_map,
                                           cudf::out_of_bounds_policy::NULLIFY,
                                           cudf::negative_index_policy::NOT_ALLOWED,
                                           stream,
                                           mr);
  return std::move(result_table->release().front());
}

template std::unique_ptr<column> scan_inclusive<DeviceMin>(column_view const& input,
                                                           bitmask_type const* mask,
                                                           cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr);

template std::unique_ptr<column> scan_inclusive<DeviceMax>(column_view const& input,
                                                           bitmask_type const* mask,
                                                           cuda::stream_ref stream,
                                                           rmm::device_async_resource_ref mr);

}  // namespace detail
}  // namespace strings
}  // namespace cudf

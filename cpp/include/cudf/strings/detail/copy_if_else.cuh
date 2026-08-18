/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/strings_column_factories.cuh>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/optional>
#include <cuda/stream>
#include <thrust/transform.h>

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Returns a new strings column using the specified Filter to select
 * strings from the lhs iterator or the rhs iterator.
 *
 * ```
 * output[i] = filter_fn(i) ? lhs(i) : rhs(i)
 * ```
 *
 * @tparam StringIterLeft A random access iterator whose value_type is
 * `cuda::std::optional<string_view>` where the `optional` has a value iff the element is valid.
 * @tparam StringIterRight A random access iterator whose value_type is
 * `cuda::std::optional<string_view>` where the `optional` has a value iff the element is valid.
 * @tparam Filter Functor that takes an index and returns a boolean.
 *
 * @param lhs_begin Start of first set of data. Used when `filter_fn` returns true.
 * @param lhs_end End of first set of data.
 * @param rhs_begin Strings of second set of data. Used when `filter_fn` returns false.
 * @param filter_fn Called to determine which iterator to use for a specific row.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory.
 * @return New strings column.
 */
template <typename StringIterLeft, typename StringIterRight, typename Filter>
std::unique_ptr<cudf::column> copy_if_else(StringIterLeft lhs_begin,
                                           StringIterLeft lhs_end,
                                           StringIterRight rhs_begin,
                                           Filter filter_fn,
                                           cuda::stream_ref stream,
                                           rmm::device_async_resource_ref mr)
{
  auto strings_count = std::distance(lhs_begin, lhs_end);
  if (strings_count == 0) { return make_empty_column(type_id::STRING); }

  // create null mask
  auto [null_mask, null_count] = cudf::detail::valid_if(
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator{static_cast<size_type>(strings_count)},
    [lhs_begin, rhs_begin, filter_fn] __device__(size_type idx) {
      return filter_fn(idx) ? lhs_begin[idx].has_value() : rhs_begin[idx].has_value();
    },
    stream,
    mr);
  if (null_count == 0) { null_mask = rmm::device_buffer{}; }

  // build vector of strings
  rmm::device_uvector<string_index_pair> indices(strings_count, stream);
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<size_type>{0},
                    cuda::counting_iterator{static_cast<size_type>(strings_count)},
                    indices.begin(),
                    [lhs_begin, rhs_begin, filter_fn] __device__(size_type idx) {
                      auto const result = filter_fn(idx) ? lhs_begin[idx] : rhs_begin[idx];
                      auto const d_str  = result.has_value() ? *result : string_view{"", 0};
                      return string_index_pair{d_str.data(), d_str.size_bytes()};
                    });

  // convert vector into strings column
  auto result = make_strings_column(indices.begin(), indices.end(), stream, mr);
  result->set_null_mask(std::move(null_mask), null_count);
  return result;
}
}  // namespace detail
}  // namespace strings
}  // namespace cudf

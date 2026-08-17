/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/strings/detail/utilities.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/scatter.h>

namespace cudf {
namespace strings {
namespace detail {
/**
 * @brief Scatters strings into a copy of the target column
 * according to a scatter map.
 *
 * The scatter is performed according to the scatter iterator such that row
 * `scatter_map[i]` of the output column is replaced by the source string.
 * All other rows of the output column equal corresponding rows of the target table.
 *
 * If the same index appears more than once in the scatter map, the result is
 * undefined.
 *
 * The caller must update the null mask in the output column.
 *
 * @tparam SourceIterator must produce string_view objects
 * @tparam MapIterator must produce index values within the target column.
 *
 * @param begin Begin iterator of source strings to scatter into the output column.
 * @param end End iterator of source strings to scatter into the output column.
 * @param scatter_map Iterator of indices into the output column.
 * @param target The set of columns into which values from the source column
 *        are to be scattered.
 * @param stream CUDA stream used for device memory operations and kernel launches.
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New strings column.
 */
template <typename SourceIterator, typename MapIterator>
std::unique_ptr<column> scatter(SourceIterator begin,
                                SourceIterator end,
                                MapIterator scatter_map,
                                strings_column_view const& target,
                                cuda::stream_ref stream,
                                rmm::device_async_resource_ref mr)
{
  if (target.is_empty()) return make_empty_column(type_id::STRING);

  // create vector of string_view's to scatter into
  rmm::device_uvector<string_view> target_vector =
    create_string_vector_from_column(target, stream, cudf::get_current_device_resource_ref());

  // this ensures empty strings are not mapped to nulls in the make_strings_column function
  auto const size = cuda::std::distance(begin, end);
  auto itr        = cuda::transform_iterator(
    begin, cuda::proclaim_return_type<string_view>([] __device__(string_view const sv) {
      return sv.empty() ? string_view{} : sv;
    }));

  // do the scatter
  thrust::scatter(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                  itr,
                  itr + size,
                  scatter_map,
                  target_vector.begin());

  // build the output column
  auto sv_span = cudf::device_span<string_view const>(target_vector);
  return make_strings_column(sv_span, string_view{nullptr, 0}, stream, mr);
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf

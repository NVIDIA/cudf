/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/lists/count_elements.hpp>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/transform.h>
#include <thrust/transform_scan.h>

namespace cudf {
namespace lists {
namespace detail {
/**
 * @brief Returns a numeric column containing lengths of each element
 *
 * @param input Input lists column
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned column's device memory
 * @return New size_type column with lengths
 */
std::unique_ptr<column> count_elements(lists_column_view const& input,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  auto device_column = cudf::column_device_view::create(input.parent(), stream);
  auto d_column      = *device_column;
  // create output column
  auto output = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                        input.size(),
                                        cudf::detail::copy_bitmask(input.parent(), stream, mr),
                                        input.null_count(),
                                        stream,
                                        mr);

  // fill in the sizes
  thrust::transform(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{input.size()},
                    output->mutable_view().begin<size_type>(),
                    list_size_functor{d_column});

  output->set_null_count(input.null_count());  // reset null count
  return output;
}

}  // namespace detail

// external APIS

std::unique_ptr<column> count_elements(lists_column_view const& input,
                                       cuda::stream_ref stream,
                                       rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::count_elements(input, stream, mr);
}

}  // namespace lists
}  // namespace cudf

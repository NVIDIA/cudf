/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/binary/binary_column_factories.hpp>
#include <cudf/binary/detail/binary_column_factories.cuh>

#include <cuda/functional>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace {

struct binary_view_to_pair {
  binary_view null_placeholder;

  __device__ binary::detail::binary_index_pair operator()(binary_view value) const
  {
    return value.data() == null_placeholder.data()
             ? binary::detail::binary_index_pair{nullptr, 0}
             : binary::detail::binary_index_pair{value.data(), value.size_bytes()};
  }
};

}  // namespace

std::unique_ptr<column> make_binary_column(device_span<binary_view const> binary_views,
                                           binary_view null_placeholder,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  auto begin =
    thrust::make_transform_iterator(binary_views.begin(), binary_view_to_pair{null_placeholder});
  return binary::detail::make_binary_column(begin, begin + binary_views.size(), stream, mr);
}

}  // namespace cudf

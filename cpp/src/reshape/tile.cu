/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/reshape.hpp>
#include <cudf/reshape.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/stream>
#include <thrust/iterator/transform_iterator.h>

#include <memory>

namespace cudf {
namespace {
struct tile_functor {
  size_type count;
  size_type __device__ operator()(size_type i) { return i % count; }
};

}  // anonymous namespace

namespace detail {
std::unique_ptr<table> tile(table_view const& in,
                            size_type count,
                            cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(count >= 0, "Count cannot be negative");

  auto const in_num_rows = in.num_rows();

  if (count == 0 or in_num_rows == 0) { return empty_like(in); }

  auto out_num_rows = in_num_rows * count;
  auto tiled_it     = cudf::detail::make_counting_transform_iterator(0, tile_functor{in_num_rows});

  return detail::gather(
    in, tiled_it, tiled_it + out_num_rows, out_of_bounds_policy::DONT_CHECK, stream, mr);
}
}  // namespace detail

std::unique_ptr<table> tile(table_view const& in,
                            size_type count,
                            cuda::stream_ref stream,
                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::tile(in, count, stream, mr);
}

}  // namespace cudf

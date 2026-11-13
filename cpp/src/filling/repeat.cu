/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/repeat.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/filling.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/scan.h>

#include <limits>
#include <memory>

namespace cudf {
namespace detail {
std::unique_ptr<table> repeat(table_view const& input_table,
                              column_view const& count,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr)
{
  CUDF_EXPECTS(input_table.num_rows() == count.size(), "in and count must have equal size");
  CUDF_EXPECTS(not count.has_nulls(), "count cannot contain nulls");

  if (input_table.num_rows() == 0) { return cudf::empty_like(input_table); }

  auto const temp_mr = mr.get_temporary_mr();
  auto count_iter    = cudf::detail::indexalator_factory::make_input_iterator(count);

  cuda::device_buffer<size_type> offsets(stream, temp_mr, count.size(), cuda::no_init);
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, temp_mr),
                         count_iter,
                         count_iter + count.size(),
                         offsets.begin());

  size_type output_size;
  cudf::detail::cuda_memcpy(
    host_span<size_type>{&output_size, 1}, device_span<size_type const>{offsets}.last(1), stream);
  cuda::device_buffer<size_type> indices(stream, temp_mr, output_size, cuda::no_init);
  thrust::upper_bound(rmm::exec_policy_nosync(stream, temp_mr),
                      offsets.begin(),
                      offsets.end(),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{output_size},
                      indices.begin());

  return detail::gather(
    input_table, indices.begin(), indices.end(), out_of_bounds_policy::DONT_CHECK, stream, mr);
}

std::unique_ptr<table> repeat(table_view const& input_table,
                              size_type count,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr)
{
  if ((input_table.num_rows() == 0) || (count == 0)) { return cudf::empty_like(input_table); }

  CUDF_EXPECTS(count >= 0, "count value should be non-negative");
  CUDF_EXPECTS(input_table.num_rows() <= std::numeric_limits<size_type>::max() / count,
               "The resulting table exceeds the column size limit",
               std::overflow_error);

  auto output_size = input_table.num_rows() * count;
  auto map_begin   = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<size_type>([count] __device__(size_type i) { return i / count; }));
  auto map_end = map_begin + output_size;

  return detail::gather(
    input_table, map_begin, map_end, out_of_bounds_policy::DONT_CHECK, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> repeat(table_view const& input_table,
                              column_view const& count,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat(input_table, count, stream, mr);
}

std::unique_ptr<table> repeat(table_view const& input_table,
                              size_type count,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat(input_table, count, stream, mr);
}

}  // namespace cudf

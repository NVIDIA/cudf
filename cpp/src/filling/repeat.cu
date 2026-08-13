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
#include <cudf/filling.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <thrust/binary_search.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

#include <limits>
#include <memory>

namespace {
struct count_accessor {
  cudf::scalar const* p_scalar = nullptr;

  template <typename T>
  cudf::size_type operator()(rmm::cuda_stream_view stream)
    requires(std::is_integral_v<T>)
  {
    using ScalarType = cudf::scalar_type_t<T>;
    auto p_count     = static_cast<ScalarType const*>(this->p_scalar);
    auto count       = p_count->value(stream);
    // static_cast is necessary due to bool
    CUDF_EXPECTS(static_cast<int64_t>(count) <= std::numeric_limits<cudf::size_type>::max(),
                 "count should not exceed the column size limit",
                 std::overflow_error);
    return static_cast<cudf::size_type>(count);
  }

  template <typename T>
  cudf::size_type operator()(rmm::cuda_stream_view)
    requires(not std::is_integral_v<T>)
  {
    CUDF_FAIL("count value should be a integral type.");
  }
};

struct count_checker {
  cudf::column_view const& count;

  template <typename T>
  void operator()(rmm::cuda_stream_view stream)
    requires(std::is_integral_v<T>)
  {
    // static_cast is necessary due to bool
    if (static_cast<int64_t>(std::numeric_limits<T>::max()) >
        std::numeric_limits<cudf::size_type>::max()) {
      auto max =
        thrust::reduce(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       count.begin<T>(),
                       count.end<T>(),
                       0,
                       cuda::maximum<T>());
      CUDF_EXPECTS(max <= std::numeric_limits<cudf::size_type>::max(),
                   "count exceeds the column size limit",
                   std::overflow_error);
    }
  }

  template <typename T>
  void operator()(rmm::cuda_stream_view)
    requires(not std::is_integral_v<T>)
  {
    CUDF_FAIL("count value type should be integral.");
  }
};

}  // namespace

namespace cudf {
namespace detail {
std::unique_ptr<table> repeat(table_view const& input_table,
                              column_view const& count,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(input_table.num_rows() == count.size(), "in and count must have equal size");
  CUDF_EXPECTS(not count.has_nulls(), "count cannot contain nulls");

  if (input_table.num_rows() == 0) { return cudf::empty_like(input_table); }

  auto count_iter = cudf::detail::indexalator_factory::make_input_iterator(count);

  rmm::device_uvector<cudf::size_type> offsets(count.size(), stream);
  thrust::inclusive_scan(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         count_iter,
                         count_iter + count.size(),
                         offsets.begin());

  size_type output_size{offsets.back_element(stream)};
  rmm::device_uvector<size_type> indices(output_size, stream);
  thrust::upper_bound(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      offsets.begin(),
                      offsets.end(),
                      cuda::counting_iterator<cudf::size_type>{0},
                      cuda::counting_iterator{output_size},
                      indices.begin());

  return gather(
    input_table, indices.begin(), indices.end(), out_of_bounds_policy::DONT_CHECK, stream, mr);
}

std::unique_ptr<table> repeat(table_view const& input_table,
                              size_type count,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
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

  return gather(input_table, map_begin, map_end, out_of_bounds_policy::DONT_CHECK, stream, mr);
}

}  // namespace detail

std::unique_ptr<table> repeat(table_view const& input_table,
                              column_view const& count,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat(input_table, count, stream, mr);
}

std::unique_ptr<table> repeat(table_view const& input_table,
                              size_type count,
                              rmm::cuda_stream_view stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::repeat(input_table, count, stream, mr);
}

}  // namespace cudf

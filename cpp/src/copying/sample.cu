/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/random>
#include <cuda/stream>
#include <thrust/shuffle.h>

namespace cudf {
namespace detail {

std::unique_ptr<table> sample(table_view const& input,
                              size_type const n,
                              sample_with_replacement replacement,
                              int64_t const seed,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(n >= 0, "expected number of samples should be non-negative", std::invalid_argument);
  auto const num_rows = input.num_rows();

  if (n > num_rows) {
    CUDF_EXPECTS(num_rows > 0,
                 "A nonzero number of samples was requested but the input table has zero rows",
                 std::invalid_argument);
    CUDF_EXPECTS(replacement == sample_with_replacement::TRUE,
                 "If n > number of rows, then multiple sampling of the same row should be allowed",
                 std::invalid_argument);
  }

  if (n == 0) return cudf::empty_like(input);

  if (replacement == sample_with_replacement::TRUE) {
    auto RandomGen = cuda::proclaim_return_type<size_type>([seed, num_rows] __device__(auto i) {
      cuda::std::philox4x32 rng(seed);
      cuda::std::uniform_int_distribution<size_type> dist{0, num_rows - 1};
      rng.discard(i);
      return dist(rng);
    });

    auto begin = cudf::detail::make_counting_transform_iterator(0, RandomGen);

    return detail::gather(input, begin, begin + n, out_of_bounds_policy::DONT_CHECK, stream, mr);
  } else {
    auto gather_map              = make_numeric_column(data_type{type_id::INT32},
                                          num_rows,
                                          mask_state::UNALLOCATED,
                                          stream,
                                          cudf::get_current_device_resource_ref());
    auto gather_map_mutable_view = gather_map->mutable_view();
    // Shuffle all the row indices
    thrust::shuffle_copy(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                         cuda::counting_iterator<size_type>{0},
                         cuda::counting_iterator<size_type>{num_rows},
                         gather_map_mutable_view.begin<size_type>(),
                         cuda::std::philox4x32(seed));

    auto gather_map_view = (n == num_rows)
                             ? gather_map->view()
                             : cudf::detail::slice(gather_map->view(), {0, n}, stream)[0];
    return detail::gather(input,
                          gather_map_view.begin<size_type>(),
                          gather_map_view.end<size_type>(),
                          out_of_bounds_policy::DONT_CHECK,
                          stream,
                          mr);
  }
}

}  // namespace detail

std::unique_ptr<table> sample(table_view const& input,
                              size_type const n,
                              sample_with_replacement replacement,
                              int64_t const seed,
                              cuda::stream_ref stream,
                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::sample(input, n, replacement, seed, stream, mr);
}
}  // namespace cudf

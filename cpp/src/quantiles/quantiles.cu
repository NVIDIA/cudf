/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "quantiles/quantiles_util.hpp"

#include <cudf/copying.hpp>
#include <cudf/detail/gather.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/quantiles.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/iterator/transform_iterator.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace cudf {
namespace detail {
template <typename SortMapIterator>
std::unique_ptr<table> quantiles(table_view const& input,
                                 SortMapIterator sortmap,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  auto quantile_idx_lookup = cuda::proclaim_return_type<size_type>(
    [sortmap, interp, size = input.num_rows()] __device__(double q) {
      auto selector = [sortmap] __device__(auto idx) { return sortmap[idx]; };
      return detail::select_quantile<size_type>(selector, size, q, interp);
    });

  auto const q_device =
    cudf::detail::make_device_uvector_async(q, stream, cudf::get_current_device_resource_ref());

  auto quantile_idx_iter = cuda::transform_iterator(q_device.begin(), quantile_idx_lookup);

  return detail::gather(input,
                        quantile_idx_iter,
                        quantile_idx_iter + q.size(),
                        out_of_bounds_policy::DONT_CHECK,
                        stream,
                        mr);
}

std::unique_ptr<table> quantiles(table_view const& input,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 cudf::sorted is_input_sorted,
                                 std::vector<order> const& column_order,
                                 std::vector<null_order> const& null_precedence,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  if (q.empty()) { return empty_like(input); }

  CUDF_EXPECTS(interp == interpolation::HIGHER || interp == interpolation::LOWER ||
                 interp == interpolation::NEAREST || interp == interpolation::NEAREST_HALF_UP,
               "multi-column quantiles require a non-arithmetic interpolation strategy.",
               std::invalid_argument);

  CUDF_EXPECTS(input.num_rows() > 0, "multi-column quantiles require at least one input row.");

  if (is_input_sorted == sorted::YES) {
    return detail::quantiles(input, cuda::counting_iterator<size_type>{0}, q, interp, stream, mr);
  } else {
    auto sorted_idx = detail::sorted_order(
      input, column_order, null_precedence, stream, cudf::get_current_device_resource_ref());
    return detail::quantiles(input, sorted_idx->view().data<size_type>(), q, interp, stream, mr);
  }
}

}  // namespace detail

std::unique_ptr<table> quantiles(table_view const& input,
                                 std::vector<double> const& q,
                                 interpolation interp,
                                 cudf::sorted is_input_sorted,
                                 std::vector<order> const& column_order,
                                 std::vector<null_order> const& null_precedence,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::quantiles(
    input, q, interp, is_input_sorted, column_order, null_precedence, stream, mr);
}

}  // namespace cudf

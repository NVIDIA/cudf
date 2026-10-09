/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort.hpp"
#include "top_k_dispatch.cuh"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/sorting.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_topk.cuh>
#include <cuda/iterator>
#include <cuda/std/execution>
#include <cuda/stream>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Computes the top-k indices of `col` with `cub::DeviceTopK`
 */
template <typename T>
std::unique_ptr<column> cub_top_k_order(column_view const& input,
                                        size_type k,
                                        order topk_order,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  auto const requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                     cuda::execution::output_ordering::unsorted);
  auto const mr_prop =
    cuda::std::execution::prop{cuda::mr::get_memory_resource, mr.get_temporary_mr()};
  auto const env = cuda::std::execution::env{cuda::stream_ref{stream.get()}, requirements, mr_prop};

  auto keys_in  = input.begin<T>();
  auto keys_out = cuda::make_discard_iterator();
  auto indices  = rmm::device_uvector<size_type>(k, stream, mr.get_output_mr());
  auto vals_in  = cuda::counting_iterator<size_type>();
  auto vals_out = indices.begin();

  if (topk_order == order::ASCENDING) {
    CUDF_CUDA_TRY(
      cub::DeviceTopK::MinPairs(keys_in, keys_out, vals_in, vals_out, input.size(), k, env));
  } else {
    CUDF_CUDA_TRY(
      cub::DeviceTopK::MaxPairs(keys_in, keys_out, vals_in, vals_out, input.size(), k, env));
  }

  return std::make_unique<column>(
    std::move(indices), cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED), 0);
}

}  // namespace

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order topk_order,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr)
{
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
  if (k == 0 || col.is_empty()) { return empty_like(col); }
  if (k >= col.size()) { return std::make_unique<column>(col, stream, mr.get_output_mr()); }

  auto const indices = [&] {
    auto const temp_mr = mr.get_temporary_mr();
    if (is_cub_top_k_supported(col)) {
      return type_dispatcher<dispatch_storage_type>(
        col.type(), dispatch_cub_top_k_key{[&]<typename T>() {
          return cub_top_k_order<T>(col, k, topk_order, stream, memory_resources{temp_mr, temp_mr});
        }});
    }
    auto const nulls = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
    return sorted_order<sort_method::STABLE>(col, topk_order, nulls, stream, temp_mr);
  }();

  auto const k_indices = cudf::detail::split(indices->view(), {k}, stream).front();

  auto const dont_check  = out_of_bounds_policy::DONT_CHECK;
  auto const not_allowed = negative_index_policy::NOT_ALLOWED;
  auto result =
    cudf::detail::gather(cudf::table_view({col}), k_indices, dont_check, not_allowed, stream, mr);
  return std::move(result->release().front());
}

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order topk_order,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr)
{
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
  if (k == 0 || col.is_empty()) { return make_empty_column(cudf::type_to_id<size_type>()); }
  if (k >= col.size()) {
    return cudf::detail::sequence(col.size(),
                                  numeric_scalar<size_type>(0, true, stream, mr.get_temporary_mr()),
                                  stream,
                                  mr.get_output_mr());
  }

  if (is_cub_top_k_supported(col)) {
    return type_dispatcher<dispatch_storage_type>(
      col.type(), dispatch_cub_top_k_key{[&]<typename T>() {
        return cub_top_k_order<T>(col, k, topk_order, stream, mr);
      }});
  }
  auto const nulls = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto indices =
    sorted_order<sort_method::STABLE>(col, topk_order, nulls, stream, mr.get_temporary_mr());

  return std::make_unique<column>(
    cudf::detail::split(indices->view(), {k}, stream).front(), stream, mr.get_output_mr());
}

}  // namespace detail

std::unique_ptr<column> top_k(column_view const& col,
                              size_type k,
                              order topk_order,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k(col, k, topk_order, stream, mr);
}

std::unique_ptr<column> top_k_order(column_view const& col,
                                    size_type k,
                                    order topk_order,
                                    cuda::stream_ref stream,
                                    cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::top_k_order(col, k, topk_order, stream, mr);
}

}  // namespace cudf

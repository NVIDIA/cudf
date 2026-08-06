/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "sort.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/sorting.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/sorting.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/type_dispatcher.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_topk.cuh>
#include <cuda/iterator>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <vector>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Resolves the k indices per segment
 *
 * Marks values outside the k range to -1 to be removed in a separate step.
 * Rows not covered by any segment are also marked to be removed.
 * Also computes the total number of valid indices for each segment.
 * All elements are used in a segment if it has less than k total elements.
 *
 * @param d_offsets Offsets for each segment
 * @param k Number of values to keep in each segment
 * @param d_indices Mark these indices to be removed
 * @param d_segment_sizes Store actual sizes of each segment
 */
CUDF_KERNEL void resolve_segment_indices(device_span<size_type const> d_offsets,
                                         size_type k,
                                         device_span<size_type> d_indices,
                                         size_type* d_segment_sizes)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  if (tid >= d_indices.size()) { return; }

  auto const sitr = thrust::upper_bound(thrust::seq, d_offsets.begin(), d_offsets.end(), tid);
  // Mark rows outside all segments for removal (offsets need not cover all rows).
  if (sitr == d_offsets.begin() || sitr == d_offsets.end()) {
    d_indices[tid] = -1;
    return;
  }
  auto const segment_start = *(sitr - 1);
  auto const segment_end   = *sitr;
  auto const index         = tid - segment_start;
  if (index >= k) { d_indices[tid] = -1; }  // mark values outside of top k

  if (index == 0) {
    auto const segment_size  = segment_end - segment_start;
    auto const segment_index = cuda::std::distance(d_offsets.begin(), sitr) - 1;
    // segment is k or less elements
    d_segment_sizes[segment_index] = cuda::std::min(k, segment_size);
  }
}

/** @brief Computes top-k indices per segment using a full segmented sort. */
std::unique_ptr<column> sort_based_segmented_top_k_order(column_view const& col,
                                                         column_view const& segment_offsets,
                                                         size_type k,
                                                         order topk_order,
                                                         cuda::stream_ref stream,
                                                         rmm::device_async_resource_ref mr)
{
  auto const size_data_type = data_type{type_to_id<size_type>()};

  auto const nulls   = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto const indices = cudf::detail::segmented_sorted_order(
    cudf::table_view({col}), segment_offsets, {topk_order}, {nulls}, stream, temp_mr);
  auto const d_indices = indices->mutable_view().begin<size_type>();

  // Zero-initialized because resolve_segment_indices writes a segment's size only from its
  // first element; an empty segment has none, so its slot must remain 0, not uninitialized.
  auto segment_sizes = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    segment_offsets.size() - 1, stream, temp_mr);
  auto span_indices = device_span<size_type>{d_indices, static_cast<std::size_t>(indices->size())};
  auto const grid   = cudf::detail::grid_1d(indices->size(), 256);
  resolve_segment_indices<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
    segment_offsets, k, span_indices, segment_sizes.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  auto [offsets, total_elements] =
    cudf::detail::make_offsets_child_column(segment_sizes.begin(), segment_sizes.end(), stream, mr);

  auto result = cudf::make_fixed_width_column(
    size_data_type, total_elements, mask_state::UNALLOCATED, stream, mr);
  auto d_result = result->mutable_view().begin<size_type>();
  // remove the indices marked by resolve_segment_indices
  thrust::remove_copy(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                      d_indices,
                      d_indices + indices->size(),
                      d_result,
                      -1);

  auto const num_rows = static_cast<size_type>(offsets->size() - 1);
  return make_lists_column(
    num_rows, std::move(offsets), std::move(result), 0, rmm::device_buffer{});
}

// Limit the number of host-launched DeviceTopK calls.
constexpr size_type cub_max_segments = 64;

// Below this average segment size the segmented sort is faster.
constexpr size_type cub_min_avg_segment_size = 16384;

// Largest selected fraction; as k approaches the segment size the post-selection
// sort approaches the full sort this path avoids.
constexpr size_type cub_max_k_fraction = 8;

bool is_fast_path(column_view const& column)
{
  return !column.has_nulls() && cudf::is_fixed_width(column.type()) &&
         !cudf::is_floating_point(column.type());  // requires NaN-aware ordering
}

/** @brief Selects top-k indices with one DeviceTopK call per segment. */
template <typename T>
std::unique_ptr<column> cub_segmented_top_k_order(column_view const& col,
                                                  host_span<size_type const> h_offsets,
                                                  size_type k,
                                                  order topk_order,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  auto const num_segments = static_cast<size_type>(h_offsets.size()) - 1;

  auto h_out_offsets = std::vector<size_type>(num_segments + 1);
  h_out_offsets[0]   = 0;
  for (size_type i = 0; i < num_segments; ++i) {
    auto const size      = h_offsets[i + 1] - h_offsets[i];
    h_out_offsets[i + 1] = h_out_offsets[i] + cuda::std::min(size, k);
  }

  // Synchronous copy before any CUB work is queued: h_out_offsets is stack-local and an
  // async copy would defer the read.
  auto offsets = std::make_unique<column>(
    cudf::detail::make_device_uvector(h_out_offsets, stream, mr), rmm::device_buffer{}, 0);

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto indices       = rmm::device_uvector<size_type>(h_out_offsets[num_segments], stream, temp_mr);
  auto const in      = col.begin<T>();
  auto keys_out      = cuda::make_discard_iterator();

  auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                               cuda::execution::output_ordering::unsorted);
  auto env          = cuda::std::execution::env{stream, requirements};

  auto run = [&](void* tmp, std::size_t& tmp_size, size_type i) {
    auto const begin    = h_offsets[i];
    auto const size     = h_offsets[i + 1] - begin;
    auto const keys_in  = in + begin;
    auto const vals_in  = cuda::counting_iterator<size_type>{begin};
    auto const vals_out = indices.data() + h_out_offsets[i];
    return topk_order == order::ASCENDING
             ? cub::DeviceTopK::MinPairs(
                 tmp, tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env)
             : cub::DeviceTopK::MaxPairs(
                 tmp, tmp_size, keys_in, keys_out, vals_in, vals_out, size, k, env);
  };

  // Reuse temporary storage across segments.
  auto max_tmp_size = std::size_t{0};
  for (size_type i = 0; i < num_segments; ++i) {
    if (h_offsets[i + 1] - h_offsets[i] <= k) { continue; }
    auto tmp_size = std::size_t{0};
    CUDF_CUDA_TRY(run(nullptr, tmp_size, i));
    max_tmp_size = cuda::std::max(max_tmp_size, tmp_size);
  }
  auto tmp = rmm::device_buffer(max_tmp_size, stream);

  for (size_type i = 0; i < num_segments; ++i) {
    auto const size = h_offsets[i + 1] - h_offsets[i];
    if (size <= 0) { continue; }
    if (size <= k) {
      // Select the entire segment without CUB.
      thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                       indices.begin() + h_out_offsets[i],
                       indices.begin() + h_out_offsets[i + 1],
                       h_offsets[i]);
      continue;
    }
    auto tmp_size = max_tmp_size;
    CUDF_CUDA_TRY(run(tmp.data(), tmp_size, i));
  }

  auto child = std::make_unique<column>(std::move(indices), rmm::device_buffer{}, 0);

  // Order each segment's selection by value like the sort-based path does. Which rows are
  // selected among equal values at the k boundary, and their relative order, stay unspecified.
  auto const nulls = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto const keys  = cudf::detail::gather(cudf::table_view({col}),
                                         child->view(),
                                         out_of_bounds_policy::DONT_CHECK,
                                         negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         temp_mr);
  auto sorted      = cudf::detail::stable_segmented_sort_by_key(cudf::table_view({child->view()}),
                                                           keys->view(),
                                                           offsets->view(),
                                                                {topk_order},
                                                                {nulls},
                                                           stream,
                                                           mr);

  return make_lists_column(num_segments,
                           std::move(offsets),
                           std::move(sorted->release().front()),
                           0,
                           rmm::device_buffer{});
}

struct dispatch_segmented_topk_fn {
  column_view col;
  host_span<size_type const> h_offsets;
  size_type k;
  order topk_order;
  cuda::stream_ref stream;
  rmm::device_async_resource_ref mr;

  template <typename T>
    requires(cudf::is_fixed_width<T>() and !cudf::is_floating_point<T>() and !cudf::is_chrono<T>())
  std::unique_ptr<column> operator()()
  {
    return cub_segmented_top_k_order<T>(col, h_offsets, k, topk_order, stream, mr);
  }

  template <typename T>
    requires(cudf::is_chrono<T>())
  std::unique_ptr<column> operator()()
  {
    using rep_type = typename T::rep;
    return cub_segmented_top_k_order<rep_type>(col, h_offsets, k, topk_order, stream, mr);
  }

  template <typename T>
    requires(not cudf::is_fixed_width<T>() or cudf::is_floating_point<T>())
  std::unique_ptr<column> operator()()
  {
    CUDF_UNREACHABLE("unexpected type for segmented_top_k fast path");
  }
};
}  // namespace

std::unique_ptr<column> segmented_top_k_order(column_view const& col,
                                              column_view const& segment_offsets,
                                              size_type k,
                                              order topk_order,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(k >= 0, "k must be greater than or equal to 0", std::invalid_argument);

  auto const size_data_type = data_type{type_to_id<size_type>()};
  if (k == 0 || col.is_empty()) { return cudf::make_empty_lists_column(size_data_type); }

  CUDF_EXPECTS(segment_offsets.size() > 0,
               "segment_offsets must have at least one element",
               std::invalid_argument);

  CUDF_EXPECTS(segment_offsets.type() == size_data_type,
               "segment_offsets must be of type INT32",
               cudf::data_type_error);
  CUDF_EXPECTS(segment_offsets.null_count() == 0,
               "segment_offsets must not have nulls",
               std::invalid_argument);

  if (auto const num_segments = segment_offsets.size() - 1;
      is_fast_path(col) && num_segments > 0 && num_segments <= cub_max_segments) {
    auto const h_offsets = cudf::detail::make_host_vector(
      device_span<size_type const>{segment_offsets.begin<size_type>(),
                                   static_cast<std::size_t>(num_segments) + 1},
      stream);
    // Malformed offsets keep the sort-based path's failure behavior.
    auto const valid_offsets = h_offsets.front() >= 0 && h_offsets.back() <= col.size() &&
                               std::is_sorted(h_offsets.begin(), h_offsets.end());
    auto const avg_segment_size = (h_offsets.back() - h_offsets.front()) / num_segments;
    if (valid_offsets && avg_segment_size >= cub_min_avg_segment_size &&
        k <= avg_segment_size / cub_max_k_fraction) {
      return type_dispatcher<dispatch_storage_type>(
        col.type(), dispatch_segmented_topk_fn{col, h_offsets, k, topk_order, stream, mr});
    }
  }

  return sort_based_segmented_top_k_order(col, segment_offsets, k, topk_order, stream, mr);
}

std::unique_ptr<column> segmented_top_k(column_view const& col,
                                        column_view const& segment_offsets,
                                        size_type k,
                                        order topk_order,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  if (col.is_empty()) { return cudf::make_empty_column(col.type()); }

  auto ordered =
    cudf::detail::segmented_top_k_order(col, segment_offsets, k, topk_order, stream, mr);
  auto lv = cudf::lists_column_view(ordered->view());
  if (lv.is_empty()) { return cudf::make_empty_lists_column(col.type()); }

  auto result         = cudf::detail::gather(cudf::table_view({col}),
                                     lv.child(),
                                     out_of_bounds_policy::DONT_CHECK,
                                     negative_index_policy::NOT_ALLOWED,
                                     stream,
                                     mr);
  auto offsets        = std::move(ordered->release().children.front());
  auto const num_rows = static_cast<size_type>(offsets->size() - 1);
  return make_lists_column(
    num_rows, std::move(offsets), std::move(result->release().front()), 0, rmm::device_buffer{});
}

}  // namespace detail

std::unique_ptr<column> segmented_top_k(column_view const& col,
                                        column_view const& segment_offsets,
                                        size_type k,
                                        order topk_order,
                                        cuda::stream_ref stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_top_k(col, segment_offsets, k, topk_order, stream, mr);
}

std::unique_ptr<column> segmented_top_k_order(column_view const& col,
                                              column_view const& segment_offsets,
                                              size_type k,
                                              order topk_order,
                                              cuda::stream_ref stream,
                                              rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_top_k_order(col, segment_offsets, k, topk_order, stream, mr);
}
}  // namespace cudf

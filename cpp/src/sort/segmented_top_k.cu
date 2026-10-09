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

#include <cub/device/device_batched_topk.cuh>
#include <cuda/argument>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/transform_reduce.h>

#include <vector>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Static upper bound on the segment size passed to `cub::DeviceBatchedTopK` for keys of
 * `key_size` bytes
 *
 * Segments larger than this are not supported.
 */
constexpr size_type max_segment_size(std::size_t key_size)
{
  return key_size <= sizeof(int32_t) ? 1 << 12 : 1 << 11;
}

/**
 * @brief Returns the size of segment `i`
 *
 * Needed to supply a random-access iterator to `cuda::args::deferred_sequence`
 */
struct segment_size_fn {
  size_type const* d_offsets;
  __device__ size_type operator()(size_type i) const { return d_offsets[i + 1] - d_offsets[i]; }
};

/**
 * @brief Computes whether `cub::DeviceBatchedTopK` can be used to select top-k of the input
 *
 * @param col Column to select from
 * @param segment_offsets Offsets of the segments in `col`
 * @param stream CUDA stream used to inspect the offsets
 * @param temp_mr Memory resource for temporary allocations
 * @return True if `cub::DeviceBatchedTopK` can be used.
 */
bool can_use_cub_batched_top_k(column_view const& col,
                               column_view const& segment_offsets,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref temp_mr)
{
  auto const num_segments = segment_offsets.size() - 1;
  if (not is_cub_top_k_supported(col) or num_segments <= 0) { return false; }

  auto const key_size = cudf::size_of(col.type());
  if (key_size > sizeof(int64_t)) { return false; }

  auto const max_size =
    thrust::transform_reduce(rmm::exec_policy_nosync(stream, temp_mr),
                             cuda::counting_iterator<size_type>{0},
                             cuda::counting_iterator<size_type>{num_segments},
                             segment_size_fn{segment_offsets.begin<size_type>()},
                             size_type{0},
                             cuda::maximum<size_type>{});
  return max_size <= max_segment_size(key_size);
}

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

  auto const sitr = cuda::std::upper_bound(d_offsets.begin(), d_offsets.end(), tid);
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

/**
 * @brief Computes top-k indices per segment with a full segmented sort
 */
std::unique_ptr<column> sort_segmented_top_k_order(column_view const& col,
                                                   column_view const& segment_offsets,
                                                   size_type k,
                                                   order topk_order,
                                                   cuda::stream_ref stream,
                                                   cudf::memory_resources mr)
{
  auto const size_data_type = data_type{type_to_id<size_type>()};

  auto const nulls   = topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE;
  auto const temp_mr = mr.get_temporary_mr();
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
    size_data_type, total_elements, mask_state::UNALLOCATED, stream, mr.get_output_mr());
  auto d_result = result->mutable_view().begin<size_type>();
  // remove the indices marked by resolve_segment_indices
  thrust::remove_copy(
    rmm::exec_policy_nosync(stream, temp_mr), d_indices, d_indices + indices->size(), d_result, -1);

  auto const num_rows = static_cast<size_type>(offsets->size() - 1);
  return make_lists_column(num_rows,
                           std::move(offsets),
                           std::move(result),
                           0,
                           cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

/**
 * @brief Computes top-k indices of all segments with single `cub::DeviceBatchedTopK` call
 */
template <typename T>
std::unique_ptr<column> cub_batched_segmented_top_k_order(column_view const& col,
                                                          column_view const& segment_offsets,
                                                          size_type k,
                                                          order topk_order,
                                                          cuda::stream_ref stream,
                                                          cudf::memory_resources mr)
{
  auto const temp_mr      = mr.get_temporary_mr();
  auto const num_segments = segment_offsets.size() - 1;
  auto const d_offsets    = segment_offsets.begin<size_type>();
  auto const segments     = cuda::counting_iterator<size_type>{0};

  // Output offsets: each segment keeps min(size, k) rows
  auto const out_sizes =
    cuda::make_transform_iterator(segments, [d_offsets, k] __device__(size_type i) -> size_type {
      return cuda::std::min(d_offsets[i + 1] - d_offsets[i], k);
    });
  auto [offsets, num_selected] =
    cudf::detail::make_offsets_child_column(out_sizes, out_sizes + num_segments, stream, mr);
  auto const d_out_offsets = offsets->view().template begin<size_type>();

  auto keys    = rmm::device_uvector<T>(num_selected, stream, temp_mr);
  auto indices = rmm::device_uvector<size_type>(num_selected, stream, temp_mr);

  // Per-segment input and output iterators
  auto const keys_in = cuda::make_transform_iterator(
    segments, [in = col.begin<T>(), d_offsets] __device__(size_type i) -> T const* {
      return in + d_offsets[i];
    });
  auto const vals_in = cuda::make_transform_iterator(
    segments, [d_offsets] __device__(size_type i) -> cuda::counting_iterator<size_type> {
      return cuda::counting_iterator<size_type>{d_offsets[i]};
    });
  auto const keys_out = cuda::make_transform_iterator(
    segments, [out = keys.data(), d_out_offsets] __device__(size_type i) -> T* {
      return out + d_out_offsets[i];
    });
  auto const vals_out = cuda::make_transform_iterator(
    segments, [out = indices.data(), d_out_offsets] __device__(size_type i) -> size_type* {
      return out + d_out_offsets[i];
    });
  auto const sizes = cuda::make_transform_iterator(segments, segment_size_fn{d_offsets});

  auto const segment_sizes =
    cuda::args::deferred_sequence{sizes, cuda::args::bounds<0, max_segment_size(sizeof(T))>()};
  auto const k_arg        = cuda::args::immediate{k};
  auto const num_segs     = cuda::args::immediate{static_cast<cuda::std::int64_t>(num_segments)};
  auto const requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                     cuda::execution::tie_break::unspecified,
                                                     cuda::execution::output_ordering::unsorted);
  auto const mr_prop      = cuda::std::execution::prop{cuda::mr::get_memory_resource, temp_mr};
  auto const env          = cuda::std::execution::env{stream, requirements, mr_prop};

  if (topk_order == order::ASCENDING) {
    CUDF_CUDA_TRY(cub::DeviceBatchedTopK::MinPairs(
      keys_in, keys_out, vals_in, vals_out, segment_sizes, k_arg, num_segs, env));
  } else {
    CUDF_CUDA_TRY(cub::DeviceBatchedTopK::MaxPairs(
      keys_in, keys_out, vals_in, vals_out, segment_sizes, k_arg, num_segs, env));
  }

  // Sort each segment's selected rows by value to match the sort-based path
  auto const indices_view    = column_view(device_span<size_type const>{indices});
  auto const keys_view       = column_view(col.type(), num_selected, keys.data(), nullptr, 0);
  auto const column_order    = std::vector<order>{topk_order};
  auto const null_precedence = std::vector<null_order>{
    topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE};
  auto sorted = cudf::detail::segmented_sort_by_key(cudf::table_view({indices_view}),
                                                    cudf::table_view({keys_view}),
                                                    offsets->view(),
                                                    column_order,
                                                    null_precedence,
                                                    stream,
                                                    mr.get_output_mr());
  return make_lists_column(num_segments,
                           std::move(offsets),
                           std::move(sorted->release().front()),
                           0,
                           cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

}  // namespace

std::unique_ptr<column> segmented_top_k_order(column_view const& col,
                                              column_view const& segment_offsets,
                                              size_type k,
                                              order topk_order,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources mr)
{
  CUDF_EXPECTS(k >= 0, "k must be non-negative", std::invalid_argument);
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

  if (can_use_cub_batched_top_k(col, segment_offsets, stream, mr.get_temporary_mr())) {
    return type_dispatcher<dispatch_storage_type>(
      col.type(), dispatch_cub_top_k_key{[&]<typename T>() {
        return cub_batched_segmented_top_k_order<T>(
          col, segment_offsets, k, topk_order, stream, mr);
      }});
  }
  return sort_segmented_top_k_order(col, segment_offsets, k, topk_order, stream, mr);
}

std::unique_ptr<column> segmented_top_k(column_view const& col,
                                        column_view const& segment_offsets,
                                        size_type k,
                                        order topk_order,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
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
  return make_lists_column(num_rows,
                           std::move(offsets),
                           std::move(result->release().front()),
                           0,
                           cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED));
}

}  // namespace detail

std::unique_ptr<column> segmented_top_k(column_view const& col,
                                        column_view const& segment_offsets,
                                        size_type k,
                                        order topk_order,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_top_k(col, segment_offsets, k, topk_order, stream, mr);
}

std::unique_ptr<column> segmented_top_k_order(column_view const& col,
                                              column_view const& segment_offsets,
                                              size_type k,
                                              order topk_order,
                                              cuda::stream_ref stream,
                                              cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::segmented_top_k_order(col, segment_offsets, k, topk_order, stream, mr);
}
}  // namespace cudf

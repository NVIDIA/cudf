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
#include <cub/device/device_topk.cuh>
#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/stream>
#include <thrust/execution_policy.h>
#include <thrust/remove.h>
#include <thrust/sequence.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <vector>

namespace cudf {
namespace detail {
namespace {

/**
 * @brief Top-K selection method used by `segmented_top_k_order`
 */
enum class top_k_method : int8_t {
  SORT         = 0,  ///< Sort-based path
  CUB_BATCHED  = 1,  ///< Single `cub::DeviceBatchedTopK` call over all segments
  CUB_SEGMENTS = 2,  ///< Per-segment `cub::DeviceTopK` calls
};

/**
 * @brief Limits for using CUB instead of the segmented sort
 *
 * @note These limits are tuned for H100 GPU. See discussion on PR #23602.
 *
 * TODO: Replace the three way method selection with cub::DeviceSegmentedTopK
 * (https://github.com/NVIDIA/cccl/issues/6391) when it is available
 */
struct cub_limits {
  /**
   * @brief Constructs the limits for keys of `key_size` bytes
   */
  explicit constexpr cub_limits(std::size_t key_size)
    : min_num_rows{key_size <= sizeof(int32_t) ? 1 << 18 : 1 << 22},
      max_segment_size{key_size <= sizeof(int32_t) ? 1 << 12 : 1 << 11},
      segments_min_avg_size{key_size <= sizeof(int32_t) ? 1 << 17 : 1 << 20}
  {
  }

  size_type min_num_rows;      ///< `CUB_BATCHED`: Minimum number of rows in all segments
  size_type max_segment_size;  ///< `CUB_BATCHED`: Maximum segment size (one thread block each)
  size_type min_avg_segment_size{1 << 8};  ///< `CUB_BATCHED`: Minimum average segment size
  size_type k_divisor{8};  ///< `CUB_BATCHED`: `k` at most the average segment size divided by this
  size_type segments_min_avg_size;   ///< `CUB_SEGMENTS`: Minimum average segment size
  size_type segments_k_divisor{32};  ///< `CUB_SEGMENTS`: `k` at most the average size / this
};

/**
 * @brief Offset and size ranges of the segments
 */
struct segment_stats {
  size_type min_offset;
  size_type max_offset;
  size_type min_size;
  size_type max_size;
};

/**
 * @brief Returns the optimal top-k selection method for the input
 *
 * @param col Column to select from
 * @param segment_offsets Offsets of the segments in `col`
 * @param k Number of rows to select from each segment
 * @param stream CUDA stream used to inspect the offsets
 * @param mr Memory resources for output and temporary allocations
 * @return Top-k selection method
 */
top_k_method select_top_k_method(column_view const& col,
                                 column_view const& segment_offsets,
                                 size_type k,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr)
{
  auto const num_segments = segment_offsets.size() - 1;
  if (not is_cub_top_k_supported(col) or num_segments <= 0) { return top_k_method::SORT; }

  auto const key_size = cudf::size_of(col.type());
  if (key_size > sizeof(int64_t)) { return top_k_method::SORT; }

  auto const limits     = cub_limits{key_size};
  auto const get_method = [&](size_type num_rows, size_type max_segment_size) {
    auto const avg_segment_size = num_rows / num_segments;
    // Each DeviceTopK call has a fixed cost, so the segments must be large to amortize it
    if (avg_segment_size >= limits.segments_min_avg_size and
        k <= avg_segment_size / limits.segments_k_divisor) {
      return top_k_method::CUB_SEGMENTS;
    }
    // DeviceBatchedTopK selects each segment with one thread block
    if (num_rows >= limits.min_num_rows and max_segment_size <= limits.max_segment_size and
        avg_segment_size >= limits.min_avg_segment_size and
        k <= avg_segment_size / limits.k_divisor) {
      return top_k_method::CUB_BATCHED;
    }
    return top_k_method::SORT;
  };

  // `col.size()` bounds the covered rows and the largest segment is not known yet (0), so inputs
  // that can only sort skip inspecting the offsets
  if (get_method(col.size(), 0) == top_k_method::SORT) { return top_k_method::SORT; }

  auto const d_offsets = segment_offsets.begin<size_type>();
  auto const stats     = thrust::transform_reduce(
    rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
    cuda::counting_iterator<size_type>{0},
    cuda::counting_iterator<size_type>{num_segments},
    [d_offsets] __device__(size_type i) -> segment_stats {
      auto const size = d_offsets[i + 1] - d_offsets[i];
      return {.min_offset = d_offsets[i],
                  .max_offset = d_offsets[i + 1],
                  .min_size   = size,
                  .max_size   = size};
    },
    segment_stats{.min_offset = cuda::std::numeric_limits<size_type>::max(),
                      .max_offset = cuda::std::numeric_limits<size_type>::min(),
                      .min_size   = cuda::std::numeric_limits<size_type>::max(),
                      .max_size   = cuda::std::numeric_limits<size_type>::min()},
    [] __device__(segment_stats const& lhs, segment_stats const& rhs) -> segment_stats {
      return {.min_offset = cuda::std::min(lhs.min_offset, rhs.min_offset),
                  .max_offset = cuda::std::max(lhs.max_offset, rhs.max_offset),
                  .min_size   = cuda::std::min(lhs.min_size, rhs.min_size),
                  .max_size   = cuda::std::max(lhs.max_size, rhs.max_size)};
    });

  // Malformed offsets keep the sort-based path's failure behavior
  if (stats.min_offset < 0 or stats.max_offset > col.size() or stats.min_size < 0) {
    return top_k_method::SORT;
  }
  return get_method(stats.max_offset - stats.min_offset, stats.max_size);
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
 * @brief Computes top-k indices using per-segment `cub::DeviceTopK` calls
 */
template <typename T>
std::unique_ptr<column> cub_per_segment_top_k_order(column_view const& col,
                                                    column_view const& segment_offsets,
                                                    size_type k,
                                                    order topk_order,
                                                    cuda::stream_ref stream,
                                                    cudf::memory_resources mr)
{
  auto const temp_mr      = mr.get_temporary_mr();
  auto const num_segments = segment_offsets.size() - 1;
  auto const h_offsets    = cudf::detail::make_pinned_vector(
    device_span<size_type const>{segment_offsets.begin<size_type>(),
                                    static_cast<std::size_t>(segment_offsets.size())},
    stream);

  // Output offsets: each segment keeps min(size, k) rows
  auto h_out_offsets = cudf::detail::make_pinned_vector_async<size_type>(num_segments + 1, stream);
  h_out_offsets.front() = 0;
  for (size_type i = 0; i < num_segments; ++i) {
    h_out_offsets[i + 1] = h_out_offsets[i] + std::min(h_offsets[i + 1] - h_offsets[i], k);
  }
  auto offsets = std::make_unique<column>(
    cudf::detail::make_device_uvector(h_out_offsets, stream, mr.get_output_mr()),
    cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED),
    0);
  auto indices = rmm::device_uvector<size_type>(h_out_offsets.back(), stream, temp_mr);

  auto const requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                     cuda::execution::output_ordering::unsorted);
  auto const mr_prop      = cuda::std::execution::prop{cuda::mr::get_memory_resource, temp_mr};
  auto const env          = cuda::std::execution::env{stream, requirements, mr_prop};

  for (size_type i = 0; i < num_segments; ++i) {
    auto const begin = h_offsets[i];
    auto const size  = h_offsets[i + 1] - begin;
    auto const out   = indices.data() + h_out_offsets[i];
    if (size == 0) { continue; }
    // Segments of at most k rows are selected whole
    if (size <= k) {
      thrust::sequence(rmm::exec_policy_nosync(stream, temp_mr), out, out + size, begin);
      continue;
    }
    auto const keys_in  = col.begin<T>() + begin;
    auto const keys_out = cuda::make_discard_iterator();
    auto const vals_in  = cuda::counting_iterator<size_type>{begin};
    if (topk_order == order::ASCENDING) {
      CUDF_CUDA_TRY(cub::DeviceTopK::MinPairs(keys_in, keys_out, vals_in, out, size, k, env));
    } else {
      CUDF_CUDA_TRY(cub::DeviceTopK::MaxPairs(keys_in, keys_out, vals_in, out, size, k, env));
    }
  }

  // Sort each segment's selected rows by value to match the sort-based path
  auto const indices_view    = column_view(device_span<size_type const>{indices});
  auto const keys            = cudf::detail::gather(cudf::table_view({col}),
                                         indices_view,
                                         out_of_bounds_policy::DONT_CHECK,
                                         negative_index_policy::NOT_ALLOWED,
                                         stream,
                                         memory_resources{temp_mr, temp_mr});
  auto const column_order    = std::vector<order>{topk_order};
  auto const null_precedence = std::vector<null_order>{
    topk_order == order::ASCENDING ? null_order::AFTER : null_order::BEFORE};
  auto sorted = cudf::detail::segmented_sort_by_key(cudf::table_view({indices_view}),
                                                    keys->view(),
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

  auto const segment_sizes = cuda::args::deferred_sequence{
    sizes, cuda::args::bounds<0, cub_limits{sizeof(T)}.max_segment_size>()};
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

  switch (select_top_k_method(col, segment_offsets, k, stream, mr)) {
    case top_k_method::CUB_SEGMENTS:
      return type_dispatcher<dispatch_storage_type>(
        col.type(), dispatch_cub_top_k_key{[&]<typename T>() {
          return cub_per_segment_top_k_order<T>(col, segment_offsets, k, topk_order, stream, mr);
        }});
    case top_k_method::CUB_BATCHED:
      return type_dispatcher<dispatch_storage_type>(
        col.type(), dispatch_cub_top_k_key{[&]<typename T>() {
          return cub_batched_segmented_top_k_order<T>(
            col, segment_offsets, k, topk_order, stream, mr);
        }});
    default: return sort_segmented_top_k_order(col, segment_offsets, k, topk_order, stream, mr);
  }
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

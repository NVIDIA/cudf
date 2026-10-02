/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "compute_single_pass_aggs.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <cub/block/block_reduce.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>

namespace cudf::groupby::detail::hash {

/// One value/output iterator pair; single-column calls need no descriptor allocation.
template <typename ValueIterator, typename OutputIterator>
struct column_reduction {
  ValueIterator values;
  OutputIterator output;

  CUDF_HOST_DEVICE static constexpr size_type size() { return 1; }
  CUDF_HOST_DEVICE column_reduction operator[](size_type) const { return *this; }
};

/// Runtime columns share segment scheduling, with independent values and outputs.
template <typename Iterator>
struct reduction_columns {
  Iterator columns;
  size_type num_columns;

  CUDF_HOST_DEVICE size_type size() const { return num_columns; }
  CUDF_HOST_DEVICE auto operator[](size_type column) const { return columns[column]; }
};

/// Maximum rows per chunk, independent of the distribution of group sizes.
constexpr size_type rows_per_chunk               = 1 << 10;
constexpr thread_index_type reduction_block_size = 256;

/// Bind inputs that depend on a group (such as centered deviations) once per
/// segment. Ordinary iterators do not evaluate the group lookup.
template <typename Values, typename GroupIterator>
__device__ auto values_for_group(Values values, GroupIterator groups, size_type segment)
{
  if constexpr (requires { values.for_group(size_type{}); }) {
    return values.for_group(groups[segment]);
  } else {
    return values;
  }
}

/// Fold groups no wider than a warp in one thread, using only their actual
/// values.
template <typename Columns, typename Op, typename T>
CUDF_KERNEL void reduce_small_groups_kernel(device_span<size_type const> offsets,
                                            Columns columns,
                                            Op op,
                                            T init)
{
  auto const num_groups  = static_cast<thread_index_type>(offsets.size() - 1);
  constexpr bool batched = !cuda::std::is_same_v<Columns, decltype(columns[size_type{0}])>;
  for (auto linear = cudf::detail::grid_1d::global_thread_id();
       linear < num_groups * columns.size();
       linear += cudf::detail::grid_1d::grid_stride()) {
    // Adjacent threads write adjacent groups of one column.
    auto const group  = batched ? linear % num_groups : linear;
    auto const column = columns[static_cast<size_type>(batched ? linear / num_groups : 0)];
    auto const output = column.output;
    auto const begin  = static_cast<cuda::std::int64_t>(offsets[group]);
    auto const end    = static_cast<cuda::std::int64_t>(offsets[group + 1]);
    if (end - begin <= cudf::detail::warp_size) {
      auto const values =
        values_for_group(column.values, cuda::counting_iterator<size_type>{0}, group);
      T partial = values[begin];
      for (auto position = begin + 1; position < end; ++position) {
        partial = op(partial, values[position]);
      }
      output[group] = op(init, partial);
    }
  }
}

/// Reduces each range with one warp or block and writes its original output index.
template <int threads_per_segment,
          typename BeginIterator,
          typename EndIterator,
          typename Columns,
          typename OutputIndexIterator,
          typename GroupIterator,
          typename Op,
          typename T>
CUDF_KERNEL void reduce_segments_kernel(size_type num_segments,
                                        BeginIterator begins,
                                        EndIterator ends,
                                        Columns columns,
                                        OutputIndexIterator output_indices,
                                        GroupIterator group_indices,
                                        Op op,
                                        T init)
{
  static_assert(threads_per_segment == cudf::detail::warp_size ||
                threads_per_segment == reduction_block_size);
  using segment_reduce = cuda::std::conditional_t<threads_per_segment == cudf::detail::warp_size,
                                                  cub::WarpReduce<T, cudf::detail::warp_size>,
                                                  cub::BlockReduce<T, reduction_block_size>>;
  __shared__
    typename segment_reduce::TempStorage storage[reduction_block_size / threads_per_segment];
  // Direct warps stay within a column; blocks retain interleaved chunk scheduling.
  constexpr bool column_major = threads_per_segment == cudf::detail::warp_size &&
                                !cuda::std::is_same_v<Columns, decltype(columns[size_type{0}])>;
  auto const linear     = cudf::detail::grid_1d::global_thread_id() / threads_per_segment;
  auto const segment    = column_major ? linear % num_segments : linear / columns.size();
  auto const lane       = static_cast<size_type>(threadIdx.x % threads_per_segment);
  auto const collective = threadIdx.x / threads_per_segment;
  if (column_major ? linear >= static_cast<thread_index_type>(num_segments) * columns.size()
                   : segment >= num_segments) {
    return;
  }
  auto const column_index = column_major ? linear / num_segments : linear % columns.size();
  auto const column       = columns[static_cast<size_type>(column_index)];
  auto const values       = values_for_group(column.values, group_indices, segment);
  auto const output       = column.output;
  auto const begin        = static_cast<cuda::std::int64_t>(begins[segment]);
  auto const end          = static_cast<cuda::std::int64_t>(ends[segment]);
  if (begin == end) {
    if (lane == 0) { output[output_indices[segment]] = init; }
    return;
  }
  T partial     = init;
  auto position = begin + lane;
  if (position < end) {
    partial = values[position];
    for (position += threads_per_segment; position < end; position += threads_per_segment) {
      partial = op(partial, values[position]);
    }
  }
  auto const valid_lanes =
    static_cast<int>(cuda::std::min<cuda::std::int64_t>(end - begin, threads_per_segment));
  auto const result = segment_reduce{storage[collective]}.Reduce(partial, op, valid_lanes);
  if (lane == 0) { output[output_indices[segment]] = op(init, result); }
}

/// Launches ranges without allocating storage; iterators encode selection and scheduling.
template <int threads_per_segment,
          typename BeginIterator,
          typename EndIterator,
          typename Columns,
          typename OutputIndexIterator,
          typename GroupIterator,
          typename Op,
          typename T>
void reduce_segments(size_type num_segments,
                     BeginIterator begins,
                     EndIterator ends,
                     Columns columns,
                     OutputIndexIterator output_indices,
                     GroupIterator group_indices,
                     Op op,
                     T init,
                     cuda::stream_ref stream)
{
  if (num_segments == 0) { return; }
  auto const num_collectives = static_cast<thread_index_type>(num_segments) * columns.size();
  CUDF_EXPECTS(
    num_collectives <= std::numeric_limits<thread_index_type>::max() / threads_per_segment,
    "Too many batched reduction segments");
  auto const config =
    cudf::detail::grid_1d{num_collectives * threads_per_segment, reduction_block_size};
  reduce_segments_kernel<threads_per_segment>
    <<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
      num_segments, begins, ends, columns, output_indices, group_indices, op, init);
  CUDF_CUDA_TRY(cudaGetLastError());
}

/// Small and bounded groups write final outputs; only long groups need chunk partials.
template <typename Columns, typename Op, typename T>
void reduce_group_columns(grouped_rows const& grouped,
                          Columns columns,
                          Op op,
                          T init,
                          cuda::stream_ref stream,
                          cudf::memory_resources mr,
                          bool long_groups_only = false)
{
  auto const num_groups        = static_cast<size_type>(grouped.offsets.size() - 1);
  auto const num_warp_groups   = static_cast<size_type>(grouped.warp_groups.size());
  auto const num_long_groups   = grouped.group_chunks.is_empty()
                                   ? size_type{0}
                                   : static_cast<size_type>(grouped.group_chunks.size() - 1);
  auto const num_direct_groups = num_warp_groups - num_long_groups;
  auto const num_chunks        = static_cast<size_type>(grouped.chunk_ranges.size());
  if (long_groups_only && num_long_groups == 0) { return; }
  using Column = decltype(columns[size_type{0}]);
  if constexpr (!cuda::std::is_same_v<Columns, Column>) {
    // Split only when a stage would exceed CUDA's maximum one-dimensional grid size.
    constexpr thread_index_type max_grid_x = std::numeric_limits<size_type>::max();
    auto max_columns                       = static_cast<thread_index_type>(columns.size());
    auto const limit_columns = [&](size_type segments, thread_index_type segments_per_block) {
      if (segments > 0) {
        max_columns = std::min(max_columns, max_grid_x * segments_per_block / segments);
      }
    };
    if (!long_groups_only) {
      limit_columns(num_warp_groups < num_groups ? num_groups : 0, reduction_block_size);
      limit_columns(num_direct_groups, reduction_block_size / cudf::detail::warp_size);
    }
    limit_columns(num_long_groups > 0 ? num_chunks : 0, 1);
    limit_columns(num_long_groups, 1);
    if (columns.size() > max_columns) {
      for (size_type first = 0; first < columns.size();) {
        auto const count =
          static_cast<size_type>(std::min(max_columns, thread_index_type{columns.size() - first}));
        reduce_group_columns(grouped,
                             reduction_columns{columns.columns + first, count},
                             op,
                             init,
                             stream,
                             mr,
                             long_groups_only);
        first += count;
      }
      return;
    }
  }
  if (!long_groups_only && num_warp_groups < num_groups) {
    auto const config = cudf::detail::grid_1d{
      static_cast<thread_index_type>(num_groups) * columns.size(), reduction_block_size};
    reduce_small_groups_kernel<<<config.num_blocks,
                                 config.num_threads_per_block,
                                 0,
                                 stream.get()>>>(grouped.offsets, columns, op, init);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  auto const group_ids = grouped.warp_groups.begin();
  if (!long_groups_only && num_direct_groups > 0) {
    reduce_segments<cudf::detail::warp_size>(
      num_direct_groups,
      cuda::make_permutation_iterator(grouped.offsets.begin(), group_ids),
      cuda::make_permutation_iterator(grouped.offsets.begin() + 1, group_ids),
      columns,
      group_ids,
      group_ids,
      op,
      init,
      stream);
  }
  if (num_long_groups == 0) { return; }
  rmm::device_uvector<T> partials(
    static_cast<std::size_t>(num_chunks) * columns.size(), stream, mr.get_temporary_mr());
  auto const begins = cuda::transform_iterator{
    grouped.chunk_ranges.begin(),
    [] __device__(cuda::std::array<size_type, 2> const& range) -> size_type { return range[0]; }};
  auto const ends = cuda::transform_iterator{
    grouped.chunk_ranges.begin(),
    [] __device__(cuda::std::array<size_type, 2> const& range) -> size_type { return range[1]; }};
  auto const chunk_groups = cuda::transform_iterator{
    grouped.chunk_order.begin(),
    [group_chunks     = grouped.group_chunks.begin(),
     group_chunks_end = grouped.group_chunks.end(),
     long_group_ids   = group_ids + num_direct_groups] __device__(size_type chunk) -> size_type {
      auto const index =
        cuda::std::upper_bound(group_chunks, group_chunks_end, chunk) - group_chunks - 1;
      return long_group_ids[index];
    }};
  auto const reduce_partials = [&](auto first_columns, auto final_columns) {
    reduce_segments<reduction_block_size>(
      num_chunks,
      cuda::make_permutation_iterator(begins, grouped.chunk_order.begin()),
      cuda::make_permutation_iterator(ends, grouped.chunk_order.begin()),
      first_columns,
      grouped.chunk_order.begin(),
      chunk_groups,
      op,
      init,
      stream);
    reduce_segments<reduction_block_size>(num_long_groups,
                                          grouped.group_chunks.begin(),
                                          grouped.group_chunks.begin() + 1,
                                          final_columns,
                                          group_ids + num_direct_groups,
                                          group_ids + num_direct_groups,
                                          op,
                                          init,
                                          stream);
  };
  using ValueIterator  = decltype(std::declval<Column>().values);
  using OutputIterator = decltype(std::declval<Column>().output);
  if constexpr (cuda::std::is_same_v<Columns, Column>) {
    reduce_partials(column_reduction{columns.values, partials.begin()},
                    column_reduction{partials.begin(), columns.output});
  } else {
    // Block passes interleave columns within a batch; partials remain column-major.
    auto const first_columns = cudf::detail::make_counting_transform_iterator(
      0,
      [columns, data = partials.data(), num_chunks] __device__(
        size_type column) -> column_reduction<ValueIterator, T*> {
        return {columns[column].values, data + static_cast<std::size_t>(column) * num_chunks};
      });
    auto const final_columns = cudf::detail::make_counting_transform_iterator(
      0,
      [columns, data = partials.data(), num_chunks] __device__(
        size_type column) -> column_reduction<T*, OutputIterator> {
        return {data + static_cast<std::size_t>(column) * num_chunks, columns[column].output};
      });
    reduce_partials(reduction_columns{first_columns, columns.size()},
                    reduction_columns{final_columns, columns.size()});
  }
}

template <typename ValueIterator, typename OutputIterator, typename Op, typename T>
void reduce_groups(grouped_rows const& grouped,
                   ValueIterator values,
                   OutputIterator output,
                   Op op,
                   T init,
                   cuda::stream_ref stream,
                   cudf::memory_resources mr)
{
  reduce_group_columns(grouped, column_reduction{values, output}, op, init, stream, mr);
}

}  // namespace cudf::groupby::detail::hash

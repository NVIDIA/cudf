/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_single_pass_aggs.hpp"
#include "grouped_reductions.cuh"
#include "single_pass_reductions.hpp"

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
#include <cuda/stream>
#include <thrust/adjacent_difference.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/partition.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/tabulate.h>
#include <thrust/transform_reduce.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

namespace cudf::groupby::detail::hash {

// Count only the group bits; padding remains valid after atomic mask updates.
size_type count_group_nulls(bitmask_type const* mask,
                            size_type num_groups,
                            cuda::stream_ref stream,
                            cudf::memory_resources mr)
{
  auto const temp_mr = mr.get_temporary_mr();
  return thrust::transform_reduce(
    rmm::exec_policy_nosync(stream, temp_mr),
    mask,
    mask + cudf::num_bitmask_words(num_groups),
    [] __device__(bitmask_type word) -> size_type { return __popc(~word); },
    size_type{0},
    cuda::std::plus<size_type>{});
}

std::unique_ptr<column> count_groups(reduction_context const& ctx,
                                     bool valid_only,
                                     cuda::stream_ref stream,
                                     cudf::memory_resources mr)
{
  auto result = make_fixed_width_column(data_type{type_to_id<size_type>()},
                                        ctx.num_groups,
                                        mask_state::UNALLOCATED,
                                        stream,
                                        mr.get_output_mr());
  if (ctx.num_groups == 0) { return result; }

  if (valid_only && ctx.values.has_nulls()) {
    auto const valid_counts = cuda::transform_iterator{
      cuda::make_permutation_iterator(cudf::detail::make_validity_iterator(ctx.d_values),
                                      ctx.grouped.rows.begin()),
      [] __device__(bool valid) -> size_type { return static_cast<size_type>(valid); }};
    reduce_groups(ctx.grouped,
                  valid_counts,
                  result->mutable_view().begin<size_type>(),
                  cuda::std::plus<size_type>{},
                  size_type{0},
                  stream,
                  mr);
  } else {
    thrust::adjacent_difference(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                                ctx.grouped.offsets.begin() + 1,
                                ctx.grouped.offsets.end(),
                                result->mutable_view().begin<size_type>());
  }
  return result;
}

grouped_rows make_grouped_rows(device_span<size_type const> rows,
                               device_span<size_type const> offsets,
                               cuda::stream_ref stream,
                               cudf::memory_resources mr)
{
  auto const temp_mr    = mr.get_temporary_mr();
  auto const num_rows   = static_cast<size_type>(rows.size());
  auto const num_groups = static_cast<size_type>(offsets.size() - 1);
  grouped_rows grouped{
    rows,
    offsets,
    cuda::device_buffer<size_type>{stream, mr.get_output_mr()},
    cuda::device_buffer<size_type>{stream, mr.get_output_mr()},
    cuda::device_buffer<cuda::std::array<size_type, 2>>{stream, mr.get_output_mr()},
    cuda::device_buffer<size_type>{stream, mr.get_output_mr()}};
  // Included groups are nonempty, so equality proves every included group is a singleton.
  if (num_groups == 0 || num_groups == num_rows) { return grouped; }

  auto const policy     = rmm::exec_policy_nosync(stream, temp_mr);
  auto const group_ids  = cuda::counting_iterator<size_type>{0};
  auto const needs_warp = [offsets = offsets.begin()] __device__(size_type group) {
    return offsets[group + 1] - offsets[group] > cudf::detail::warp_size;
  };
  // Only groups wider than a warp need compact IDs and warp work.
  auto const num_warp_groups =
    static_cast<size_type>(thrust::count_if(policy, group_ids, group_ids + num_groups, needs_warp));
  if (num_warp_groups == 0) { return grouped; }
  grouped.warp_groups = cuda::device_buffer<size_type>{
    stream, mr.get_output_mr(), static_cast<std::size_t>(num_warp_groups), cuda::no_init};
  thrust::copy_if(
    policy, group_ids, group_ids + num_groups, grouped.warp_groups.data(), needs_warp);
  auto const long_groups =
    thrust::partition(policy,
                      grouped.warp_groups.data(),
                      grouped.warp_groups.data() + grouped.warp_groups.size(),
                      [offsets = offsets.begin()] __device__(size_type group) {
                        return offsets[group + 1] - offsets[group] <= rows_per_chunk;
                      });
  auto const num_long_groups =
    static_cast<size_type>(grouped.warp_groups.data() + grouped.warp_groups.size() - long_groups);
  if (num_long_groups == 0) { return grouped; }

  auto const chunk_counts = cudf::detail::make_counting_transform_iterator(
    0, [offsets = offsets.begin(), long_groups] __device__(size_type index) -> size_type {
      auto const group = long_groups[index];
      return cudf::util::div_rounding_up_safe(offsets[group + 1] - offsets[group], rows_per_chunk);
    });
  grouped.group_chunks = cuda::device_buffer<size_type>{
    stream, mr.get_output_mr(), static_cast<std::size_t>(num_long_groups) + 1, cuda::no_init};
  CUDF_CUDA_TRY(cudaMemsetAsync(grouped.group_chunks.data(), 0, sizeof(size_type), stream.get()));
  thrust::inclusive_scan(
    policy, chunk_counts, chunk_counts + num_long_groups, grouped.group_chunks.data() + 1);
  size_type num_chunks;
  cudf::detail::cuda_memcpy<size_type>(
    host_span<size_type>{&num_chunks, 1},
    device_span<size_type const>{grouped.group_chunks.data() + num_long_groups, 1},
    stream);

  // Short groups leave gaps in CSR positions; store both endpoints of each long chunk.
  grouped.chunk_ranges = cuda::device_buffer<cuda::std::array<size_type, 2>>{
    stream, mr.get_output_mr(), static_cast<std::size_t>(num_chunks), cuda::no_init};
  thrust::tabulate(
    policy,
    grouped.chunk_ranges.data(),
    grouped.chunk_ranges.data() + grouped.chunk_ranges.size(),
    [offsets = offsets.begin(),
     long_groups,
     group_chunks = grouped.group_chunks.data(),
     group_chunks_end =
       grouped.group_chunks.data() +
       grouped.group_chunks.size()] __device__(size_type chunk) -> cuda::std::array<size_type, 2> {
      auto const index = static_cast<size_type>(
        cuda::std::upper_bound(group_chunks, group_chunks_end, chunk) - group_chunks - 1);
      auto const group = long_groups[index];
      auto const begin =
        static_cast<cuda::std::int64_t>(offsets[group]) +
        static_cast<cuda::std::int64_t>(chunk - group_chunks[index]) * rows_per_chunk;
      auto const end =
        cuda::std::min(begin + rows_per_chunk, static_cast<cuda::std::int64_t>(offsets[group + 1]));
      return {static_cast<size_type>(begin), static_cast<size_type>(end)};
    });

  // Reuse a first-stored-row scheduling hint for long chunks across all value columns.
  cuda::device_buffer<size_type> first_rows(stream, temp_mr, num_chunks, cuda::no_init);
  auto const chunk_begins = cuda::transform_iterator{
    grouped.chunk_ranges.data(),
    [] __device__(cuda::std::array<size_type, 2> const& range) -> size_type { return range[0]; }};
  thrust::gather(policy, chunk_begins, chunk_begins + num_chunks, rows.begin(), first_rows.data());
  grouped.chunk_order = cuda::device_buffer<size_type>{
    stream, mr.get_output_mr(), static_cast<std::size_t>(num_chunks), cuda::no_init};
  thrust::sequence(
    policy, grouped.chunk_order.data(), grouped.chunk_order.data() + grouped.chunk_order.size());
  thrust::sort_by_key(
    policy, first_rows.data(), first_rows.data() + first_rows.size(), grouped.chunk_order.data());
  return grouped;
}

}  // namespace cudf::groupby::detail::hash

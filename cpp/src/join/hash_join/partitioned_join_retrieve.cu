/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "hash_csr_kernels.cuh"
#include "join/join_common_utils.hpp"
#include "join/join_match_state.hpp"

#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/cstdint>

namespace cudf::detail {
namespace {
/// Match counts followed by a zero, so that the last exclusive-scan output is the total
struct padded_counts {
  size_type const* counts;
  size_type size;
  __device__ size_type operator()(size_type idx) const { return idx < size ? counts[idx] : 0; }
};
}  // namespace

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::partitioned_join_retrieve(join_kind join,
                                             cudf::join_partition_context const& context,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  CUDF_EXPECTS(
    join == join_kind::INNER_JOIN || join == join_kind::LEFT_JOIN || join == join_kind::FULL_JOIN,
    "Unsupported join kind for partitioned retrieve");

  CUDF_EXPECTS(context.left_table_context != nullptr,
               "join_partition_context is missing left_table_context",
               std::invalid_argument);

  auto const& match_ctx     = *context.left_table_context;
  auto const left_start_idx = context.left_start_idx;
  auto const left_end_idx   = context.left_end_idx;
  auto const match_counts   = match_ctx.match_counts();
  auto const* state         = join_match_state::get<hash_join_match_state>(match_ctx, this, join);

  CUDF_EXPECTS(left_start_idx >= 0 && left_end_idx >= left_start_idx &&
                 left_end_idx <= match_ctx.left_table().num_rows(),
               "Invalid partition bounds",
               std::invalid_argument);

  // Empty partition
  if (left_start_idx >= left_end_idx) {
    return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                     std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
  }

  auto const partition_size = left_end_idx - left_start_idx;

  // Trivial case: build table is empty
  if (_is_empty) {
    if (join == join_kind::INNER_JOIN) {
      return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                       std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    }
    return get_trivial_left_join_indices(
      cudf::slice(match_ctx.left_table(), {left_start_idx, left_end_idx})[0],
      left_start_idx,
      stream,
      mr);
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();

  // Contexts created by a factory cache the matching groups; others probe the partition again.
  cuda::device_buffer<size_type> probed_groups(
    stream, temp_mr, state == nullptr ? partition_size : 0, cuda::no_init);
  if (state == nullptr) {
    probe(join,
          cudf::slice(match_ctx.left_table(), {left_start_idx, left_end_idx})[0],
          probed_groups.data(),
          nullptr,
          nullptr,
          stream);
  }
  auto const probe_groups =
    state == nullptr ? probed_groups.data() : state->groups.data() + left_start_idx;

  auto const counts = match_counts.data() + left_start_idx;
  auto const sizes  = cuda::transform_iterator(cuda::counting_iterator<size_type>{0},
                                              padded_counts{counts, partition_size});
  cuda::device_buffer<cuda::std::int64_t> offsets(
    stream, temp_mr, static_cast<std::size_t>(partition_size) + 1, cuda::no_init);
  auto const output_size = cudf::detail::sizes_to_offsets(
    sizes, sizes + partition_size + 1, offsets.begin(), 0, stream, temp_mr);

  auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(
    static_cast<std::size_t>(output_size), stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(
    static_cast<std::size_t>(output_size), stream, mr);
  cudf::prefetch::detail::prefetch(*left_indices, stream);
  cudf::prefetch::detail::prefetch(*right_indices, stream);

  if (join == join_kind::INNER_JOIN) {
    launch_hash_csr_inner_retrieve_kernel(output_size,
                                          partition_size,
                                          offsets.data(),
                                          probe_groups,
                                          _impl->csr(),
                                          left_start_idx,
                                          left_indices->data(),
                                          right_indices->data(),
                                          stream);
  } else {
    launch_hash_csr_outer_retrieve_kernel(output_size,
                                          partition_size,
                                          offsets.data(),
                                          probe_groups,
                                          _impl->csr(),
                                          left_start_idx,
                                          left_indices->data(),
                                          right_indices->data(),
                                          stream);
  }

  return {std::move(left_indices), std::move(right_indices)};
}

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::partitioned_join_retrieve(join_kind,
                                                       cudf::join_partition_context const&,
                                                       cuda::stream_ref,
                                                       rmm::device_async_resource_ref) const;

}  // namespace cudf::detail

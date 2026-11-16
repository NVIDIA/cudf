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
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/buffer>

namespace cudf::detail {

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::partitioned_join_retrieve(join_kind join,
                                             cudf::join_partition_context const& context,
                                             cuda::stream_ref stream,
                                             cudf::memory_resources mr) const
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
    return std::pair(
      std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr()),
      std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr()));
  }

  auto const partition_size = left_end_idx - left_start_idx;

  // Trivial case: build table is empty
  if (_is_empty) {
    if (join == join_kind::INNER_JOIN) {
      return std::pair(
        std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr()),
        std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr()));
    }
    return get_trivial_left_join_indices(
      cudf::slice(match_ctx.left_table(), {left_start_idx, left_end_idx})[0],
      left_start_idx,
      stream,
      mr);
  }

  // Contexts created by a factory cache the matching groups; others probe the partition again.
  // TODO: Remove the probe when the deprecated join_match_context constructor is removed in 27.02
  // (#20958).
  cuda::device_buffer<size_type> probed_groups(stream,
                                               cudf::get_current_device_resource_ref(),
                                               state == nullptr ? partition_size : 0,
                                               cuda::no_init);
  if (state == nullptr) {
    auto const left_partition =
      cudf::slice(match_ctx.left_table(), {left_start_idx, left_end_idx})[0];
    validate_hash_join_probe(_right, left_partition, _has_nulls);
    probe(join, left_partition, probed_groups, {}, nullptr, {}, nullptr, stream);
  }
  auto const probe_groups = state == nullptr ? device_span<size_type const>{probed_groups}
                                             : device_span<size_type const>{state->groups}.subspan(
                                                 left_start_idx, partition_size);

  auto const [offsets, output_size] =
    hash_csr_match_offsets(match_counts.subspan(left_start_idx, partition_size), stream);
  return hash_csr_retrieve(join != join_kind::INNER_JOIN,
                           output_size,
                           offsets,
                           probe_groups,
                           _impl->csr(),
                           left_start_idx,
                           0,
                           stream,
                           mr);
}

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::partitioned_join_retrieve(join_kind,
                                                       cudf::join_partition_context const&,
                                                       cuda::stream_ref,
                                                       cudf::memory_resources) const;

}  // namespace cudf::detail

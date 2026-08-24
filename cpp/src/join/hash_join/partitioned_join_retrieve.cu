/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "dispatch.cuh"
#include "hash_csr_kernels.cuh"
#include "join/join_common_utils.hpp"

#include <cudf/copying.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/std/cstdint>

namespace cudf::detail {
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

  CUDF_EXPECTS(match_ctx._match_counts != nullptr,
               "join_match_context is missing match counts",
               std::invalid_argument);
  CUDF_EXPECTS(left_start_idx >= 0 && left_end_idx >= left_start_idx &&
                 left_end_idx <= match_ctx._left_table.num_rows(),
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
      cudf::slice(match_ctx._left_table, {left_start_idx, left_end_idx})[0],
      left_start_idx,
      stream,
      mr);
  }

  // Slice the left table to the partition range
  auto const left_partition_view =
    cudf::slice(match_ctx._left_table, {left_start_idx, left_end_idx})[0];

  validate_hash_join_probe(_right, left_partition_view, _has_nulls);

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left_partition_view, stream, temp_mr);

  auto counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    static_cast<std::size_t>(partition_size) + 1, stream, temp_mr);
  CUDF_CUDA_TRY(
    cudf::detail::memcpy_async(counts.data(),
                               match_ctx._match_counts->data() + left_start_idx,
                               static_cast<std::size_t>(partition_size) * sizeof(size_type),
                               stream));
  auto offsets = cudf::detail::make_zeroed_device_uvector_async<cuda::std::int64_t>(
    static_cast<std::size_t>(partition_size) + 1, stream, temp_mr);
  auto const output_size =
    cudf::detail::sizes_to_offsets(counts.begin(), counts.end(), offsets.begin(), 0, stream);
  CUDF_EXPECTS(output_size >= 0, "Join output size overflowed", std::overflow_error);

  rmm::device_uvector<size_type> probe_slots(partition_size, stream, temp_mr);
  auto const row_bitmask = cudf::detail::bitmask_and(left_partition_view, stream, temp_mr).first;
  auto const valid_rows  = _nulls_equal == null_equality::UNEQUAL
                             ? static_cast<bitmask_type const*>(row_bitmask.data())
                             : nullptr;
  auto save_slots        = [&](auto equality, auto hasher) {
    if (join == join_kind::INNER_JOIN) {
      launch_hash_csr_probe_count_kernel<false>(partition_size,
                                                valid_rows,
                                                probe_slots.data(),
                                                nullptr,
                                                nullptr,
                                                nullptr,
                                                _impl->hash_table(),
                                                _impl->csr(),
                                                equality,
                                                hasher,
                                                stream);
    } else {
      launch_hash_csr_probe_count_kernel<true>(partition_size,
                                               valid_rows,
                                               probe_slots.data(),
                                               nullptr,
                                               nullptr,
                                               nullptr,
                                               _impl->hash_table(),
                                               _impl->csr(),
                                               equality,
                                               hasher,
                                               stream);
    }
  };
  dispatch_join_comparator(_right,
                           left_partition_view,
                           _preprocessed_right,
                           preprocessed_left,
                           _has_nulls,
                           _nulls_equal,
                           save_slots);

  auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(
    static_cast<std::size_t>(output_size), stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(
    static_cast<std::size_t>(output_size), stream, mr);
  cudf::prefetch::detail::prefetch(*left_indices, stream);
  cudf::prefetch::detail::prefetch(*right_indices, stream);

  if (join == join_kind::INNER_JOIN) {
    launch_hash_csr_retrieve_kernel<false>(output_size,
                                           partition_size,
                                           offsets.data(),
                                           probe_slots.data(),
                                           _impl->csr(),
                                           left_start_idx,
                                           left_indices->data(),
                                           right_indices->data(),
                                           stream);
  } else {
    launch_hash_csr_retrieve_kernel<true>(output_size,
                                          partition_size,
                                          offsets.data(),
                                          probe_slots.data(),
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

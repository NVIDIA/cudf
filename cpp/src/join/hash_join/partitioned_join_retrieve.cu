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
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/tuple>
#include <thrust/tabulate.h>

namespace cudf::detail {
namespace {

/**
 * @brief Returns trivial left/right index pairs for an outer join when the build side is empty.
 */
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
make_trivial_outer_indices(size_type left_start_idx,
                           size_type partition_size,
                           rmm::cuda_stream_view stream,
                           rmm::device_async_resource_ref mr)
{
  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(partition_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(partition_size, stream, mr);
  auto out           = cuda::zip_iterator(left_indices->begin(), right_indices->begin());
  thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                   out,
                   out + partition_size,
                   cuda::proclaim_return_type<cuda::std::tuple<size_type, size_type>>(
                     [left_start_idx] __device__(auto i) {
                       return cuda::std::tuple{static_cast<size_type>(left_start_idx + i),
                                               JoinNoMatch};
                     }));
  return std::pair(std::move(left_indices), std::move(right_indices));
}

}  // namespace

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::partitioned_join_retrieve(join_kind join,
                                             cudf::join_partition_context const& context,
                                             rmm::cuda_stream_view stream,
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
    } else {
      return make_trivial_outer_indices(left_start_idx, partition_size, stream, mr);
    }
  }

  // Slice the left table to the partition range
  auto const left_partition_view =
    cudf::slice(match_ctx._left_table, {left_start_idx, left_end_idx})[0];

  validate_hash_join_probe(_right, left_partition_view, _has_nulls);

  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left_partition_view, stream);

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto counts        = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    static_cast<std::size_t>(partition_size) + 1, stream, temp_mr);
  CUDF_CUDA_TRY(
    cudf::detail::memcpy_async(counts.data(),
                               match_ctx._match_counts->data() + left_start_idx,
                               static_cast<std::size_t>(partition_size) * sizeof(size_type),
                               stream));
  auto offsets = cudf::detail::make_zeroed_device_uvector_async<std::int64_t>(
    static_cast<std::size_t>(partition_size) + 1, stream, temp_mr);
  auto const output_size =
    hash_csr_scan_counts(counts.data(), partition_size, offsets.data(), stream);
  CUDF_EXPECTS(output_size >= 0, "Join output size overflowed", std::overflow_error);

  rmm::device_uvector<size_type> probe_slots(partition_size, stream, temp_mr);
  auto const row_bitmask = cudf::detail::bitmask_and(left_partition_view, stream, temp_mr).first;
  auto const valid_rows  = _nulls_equal == null_equality::UNEQUAL
                             ? static_cast<bitmask_type const*>(row_bitmask.data())
                             : nullptr;
  auto save_slots        = [&](auto equality, auto hasher) {
    if (join == join_kind::INNER_JOIN) {
      launch_hash_csr_probe_count<false>(partition_size,
                                         valid_rows,
                                         probe_slots.data(),
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         _impl->map_view(),
                                         _impl->csr_view(),
                                         equality,
                                         hasher,
                                         stream);
    } else {
      launch_hash_csr_probe_count<true>(partition_size,
                                        valid_rows,
                                        probe_slots.data(),
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        _impl->map_view(),
                                        _impl->csr_view(),
                                        equality,
                                        hasher,
                                        stream);
    }
  };
  dispatch_hash_csr_comparator(_right,
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
    launch_hash_csr_retrieve<false>(output_size,
                                    partition_size,
                                    offsets.data(),
                                    probe_slots.data(),
                                    _impl->csr_view(),
                                    left_start_idx,
                                    left_indices->data(),
                                    right_indices->data(),
                                    stream);
  } else {
    launch_hash_csr_retrieve<true>(output_size,
                                   partition_size,
                                   offsets.data(),
                                   probe_slots.data(),
                                   _impl->csr_view(),
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
                                                       rmm::cuda_stream_view,
                                                       rmm::device_async_resource_ref) const;

}  // namespace cudf::detail

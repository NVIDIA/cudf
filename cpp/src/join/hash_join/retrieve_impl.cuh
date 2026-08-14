/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "common.cuh"
#include "dispatch.cuh"
#include "hash_csr_kernels.cuh"
#include "join/join_common_utils.hpp"

#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/device_uvector.hpp>

namespace cudf::detail {

template <typename Hasher>
template <join_kind Join>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::join_retrieve(cudf::table_view const& left,
                                 std::optional<std::size_t> output_size,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr) const
{
  CUDF_FUNC_RANGE();

  validate_hash_join_probe(_right, left, _has_nulls);

  if constexpr (Join == join_kind::INNER_JOIN) {
    if (is_trivial_join(left, _right, Join)) {
      return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                       std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    }
  } else {
    if (_is_empty) { return get_trivial_left_join_indices(left, stream, mr); }

    if (is_trivial_join(left, _right, Join)) {
      return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                       std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    }
  }

  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left, stream);

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto match_counts  = cudf::detail::make_zeroed_device_uvector_async<size_type>(
    static_cast<std::size_t>(left.num_rows()) + 1, stream, temp_mr);
  rmm::device_uvector<size_type> probe_slots(left.num_rows(), stream, temp_mr);
  auto const row_bitmask = cudf::detail::bitmask_and(left, stream, temp_mr).first;
  auto const valid_rows  = _nulls_equal == null_equality::UNEQUAL
                             ? static_cast<bitmask_type const*>(row_bitmask.data())
                             : nullptr;

  auto count_matches = [&](auto equality, auto hasher) {
    launch_hash_csr_probe_count<Join != join_kind::INNER_JOIN>(left.num_rows(),
                                                               valid_rows,
                                                               probe_slots.data(),
                                                               match_counts.data(),
                                                               nullptr,
                                                               nullptr,
                                                               _impl->map_view(),
                                                               _impl->csr_view(),
                                                               equality,
                                                               hasher,
                                                               stream);
  };
  dispatch_join_comparator(
    _right, left, _preprocessed_right, preprocessed_left, _has_nulls, _nulls_equal, count_matches);

  auto offsets = cudf::detail::make_zeroed_device_uvector_async<std::int64_t>(
    static_cast<std::size_t>(left.num_rows()) + 1, stream, temp_mr);
  auto const actual_size = cudf::detail::sizes_to_offsets(
    match_counts.begin(), match_counts.end(), offsets.begin(), 0, stream);
  CUDF_EXPECTS(actual_size >= 0, "Join output size overflowed", std::overflow_error);
  auto const join_size = Join != join_kind::FULL_JOIN && output_size.has_value()
                           ? *output_size
                           : static_cast<std::size_t>(actual_size);
  CUDF_EXPECTS(join_size == static_cast<std::size_t>(actual_size),
               "The provided join output size is incorrect");

  auto left_indices  = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(join_size, stream, mr);
  cudf::prefetch::detail::prefetch(*left_indices, stream);
  cudf::prefetch::detail::prefetch(*right_indices, stream);

  if constexpr (Join == join_kind::INNER_JOIN) {
    launch_hash_csr_retrieve<false>(actual_size,
                                    left.num_rows(),
                                    offsets.data(),
                                    probe_slots.data(),
                                    _impl->csr_view(),
                                    0,
                                    left_indices->data(),
                                    right_indices->data(),
                                    stream);
  } else {
    launch_hash_csr_retrieve<true>(actual_size,
                                   left.num_rows(),
                                   offsets.data(),
                                   probe_slots.data(),
                                   _impl->csr_view(),
                                   0,
                                   left_indices->data(),
                                   right_indices->data(),
                                   stream);
  }

  auto join_indices = std::pair(std::move(left_indices), std::move(right_indices));

  if constexpr (Join == join_kind::FULL_JOIN) {
    return detail::finalize_full_join(
      std::move(join_indices), left.num_rows(), _right.num_rows(), stream, mr);
  } else {
    return join_indices;
  }
}

}  // namespace cudf::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "common.cuh"
#include "dispatch.cuh"
#include "hash_csr_kernels.cuh"
#include "join/join_common_utils.hpp"

#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/std/cstdint>

#include <memory>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <utility>

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

  // The output size is always computed internally; a caller-supplied size is only checked
  // against it, on every path including the trivial ones.
  auto const validate_output_size = [&](std::size_t actual) {
    CUDF_EXPECTS(!output_size.has_value() || *output_size == actual,
                 "The provided join output size is incorrect");
  };

  if constexpr (Join == join_kind::INNER_JOIN) {
    if (is_trivial_join(left, _right, Join)) {
      validate_output_size(0);
      return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                       std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    }
  } else {
    if (_is_empty) {
      validate_output_size(static_cast<std::size_t>(left.num_rows()));
      return get_trivial_left_join_indices(left, 0, stream, mr);
    }

    if (is_trivial_join(left, _right, Join)) {
      validate_output_size(0);
      return std::pair(std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr),
                       std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr));
    }
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();
  std::optional<size_type> unmatched_right_rows;
  // Release retrieval scratch before full-join finalization allocates its match flags.
  auto join_indices = [&] {
    // Only offsets and group IDs are needed to emit the output. Keep counting scratch out
    // of the output allocation's lifetime.
    auto [offsets, probe_groups, actual_size] = [&] {
      auto const preprocessed_left = cudf::detail::row::equality::preprocessed_table::create(
        left, stream, cudf::get_current_device_resource_ref());
      auto match_counts = cudf::detail::make_zeroed_device_uvector_async<size_type>(
        static_cast<std::size_t>(left.num_rows()) + 1, stream, temp_mr);
      rmm::device_uvector<size_type> probe_groups(left.num_rows(), stream, temp_mr);
      // A full join appends the unmatched right rows, so track which build rows the probe matched
      // to size the output exactly. Other join kinds skip these flags and their atomics.
      auto matched_groups = Join == join_kind::FULL_JOIN
                              ? cudf::detail::make_zeroed_device_uvector_async<cuda::std::uint32_t>(
                                  _right.num_rows(), stream, temp_mr)
                              : rmm::device_uvector<cuda::std::uint32_t>{0, stream, temp_mr};
      auto matched_build_rows =
        cudf::detail::device_scalar<cuda::std::uint64_t>(0, stream, temp_mr);
      auto const row_bitmask = cudf::detail::bitmask_and(left, stream, temp_mr).first;
      auto const valid_rows  = _nulls_equal == null_equality::UNEQUAL
                                 ? static_cast<bitmask_type const*>(row_bitmask.data())
                                 : nullptr;

      auto count_matches = [&](auto equality, auto hasher) {
        launch_hash_csr_probe_count_kernel<Join != join_kind::INNER_JOIN>(
          left.num_rows(),
          valid_rows,
          probe_groups.data(),
          match_counts.data(),
          Join == join_kind::FULL_JOIN ? matched_groups.data() : nullptr,
          matched_build_rows.data(),
          _impl->hash_table(),
          _impl->csr(),
          equality,
          hasher,
          stream);
      };
      dispatch_join_comparator(_right,
                               left,
                               _preprocessed_right,
                               preprocessed_left,
                               _has_nulls,
                               _nulls_equal,
                               count_matches);

      auto offsets = cudf::detail::make_zeroed_device_uvector_async<cuda::std::int64_t>(
        static_cast<std::size_t>(left.num_rows()) + 1, stream, temp_mr);
      auto const actual_size = cudf::detail::sizes_to_offsets(
        match_counts.begin(), match_counts.end(), offsets.begin(), 0, stream, temp_mr);
      CUDF_EXPECTS(actual_size >= 0, "Join output size overflowed", std::overflow_error);

      // The count pass already tallied matched build rows. Preserve the exact complement size
      // for output allocation and full-join finalization before releasing its device scalar.
      if constexpr (Join == join_kind::FULL_JOIN) {
        // Each build row is tallied at most once, so narrowing to size_type is safe.
        auto const matched = matched_build_rows.value(stream);
        unmatched_right_rows =
          static_cast<size_type>(static_cast<cuda::std::uint64_t>(_right.num_rows()) - matched);
      }
      return std::tuple(std::move(offsets), std::move(probe_groups), actual_size);
    }();

    auto const join_size = static_cast<std::size_t>(actual_size);
    auto const allocation_size =
      join_size + static_cast<std::size_t>(unmatched_right_rows.value_or(size_type{0}));
    // For a full join the final size includes the unmatched right rows, so validate only now.
    validate_output_size(allocation_size);

    auto left_indices =
      std::make_unique<rmm::device_uvector<size_type>>(allocation_size, stream, mr);
    auto right_indices =
      std::make_unique<rmm::device_uvector<size_type>>(allocation_size, stream, mr);
    left_indices->resize(join_size, stream);
    right_indices->resize(join_size, stream);
    cudf::prefetch::detail::prefetch(*left_indices, stream);
    cudf::prefetch::detail::prefetch(*right_indices, stream);

    if constexpr (Join == join_kind::INNER_JOIN) {
      launch_hash_csr_inner_retrieve_kernel(actual_size,
                                            left.num_rows(),
                                            offsets.data(),
                                            probe_groups.data(),
                                            _impl->csr(),
                                            0,
                                            left_indices->data(),
                                            right_indices->data(),
                                            stream);
    } else {
      launch_hash_csr_outer_retrieve_kernel(actual_size,
                                            left.num_rows(),
                                            offsets.data(),
                                            probe_groups.data(),
                                            _impl->csr(),
                                            0,
                                            left_indices->data(),
                                            right_indices->data(),
                                            stream);
    }

    return std::pair(std::move(left_indices), std::move(right_indices));
  }();

  if constexpr (Join == join_kind::FULL_JOIN) {
    // The HashCSR retrieve kernels do not mark matched right rows, so let `finalize_full_join`
    // derive the match flags from the emitted right indices.
    return detail::finalize_full_join(std::move(join_indices),
                                      left.num_rows(),
                                      _right.num_rows(),
                                      std::nullopt,
                                      stream,
                                      mr,
                                      unmatched_right_rows);
  } else {
    return join_indices;
  }
}

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::join_retrieve<join_kind::INNER_JOIN>(
  cudf::table_view const&,
  std::optional<std::size_t>,
  cuda::stream_ref,
  rmm::device_async_resource_ref) const;
template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::join_retrieve<join_kind::LEFT_JOIN>(
  cudf::table_view const&,
  std::optional<std::size_t>,
  cuda::stream_ref,
  rmm::device_async_resource_ref) const;
template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::join_retrieve<join_kind::FULL_JOIN>(
  cudf::table_view const&,
  std::optional<std::size_t>,
  cuda::stream_ref,
  rmm::device_async_resource_ref) const;

}  // namespace cudf::detail

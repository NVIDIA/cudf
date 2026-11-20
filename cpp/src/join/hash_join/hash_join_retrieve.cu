/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "common.cuh"
#include "dispatch.cuh"
#include "hash_graph_kernels.cuh"
#include "join/join_common_utils.hpp"

#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/dispatchers.hpp>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>

#include <rmm/device_uvector.hpp>

#include <cuda/buffer>
#include <cuda/std/cstdint>

#include <memory>
#include <optional>
#include <utility>

namespace cudf::detail {

template <typename Hasher>
template <join_kind Join>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::join_retrieve(cudf::table_view const& left,
                                 std::optional<std::size_t> output_size,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr) const
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
      return std::pair(
        std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr()),
        std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr()));
    }
  } else {
    if (_is_empty) {
      validate_output_size(static_cast<std::size_t>(left.num_rows()));
      return get_trivial_left_join_indices(left, 0, stream, mr);
    }

    if (is_trivial_join(left, _right, Join)) {
      validate_output_size(0);
      return std::pair(
        std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr()),
        std::make_unique<rmm::device_uvector<size_type>>(0, stream, mr.get_output_mr()));
    }
  }

  auto const temp_mr = cudf::get_current_device_resource_ref();
  auto const preprocessed_left =
    cudf::detail::row::equality::preprocessed_table::create(left, stream, temp_mr);
  std::optional<size_type> unmatched_right_rows;
  // Release retrieval scratch before full-join finalization allocates its match flags.
  auto join_indices = [&] {
    cuda::device_buffer<hash_value_type> hashes(stream, temp_mr, left.num_rows(), cuda::no_init);
    // Only offsets and hashes are needed to emit the output. Keep counting scratch out of the
    // output allocation's lifetime.
    auto [offsets, join_size] = [&] {
      cuda::device_buffer<size_type> match_counts(stream, temp_mr, left.num_rows(), cuda::no_init);
      // A full join appends the unmatched right rows, so track which build rows the probe matched
      // to size the output exactly. Other join kinds skip these flags and their atomics.
      auto matched_rows =
        Join == join_kind::FULL_JOIN
          ? cudf::detail::make_zeroed_device_buffer_async<cuda::std::uint32_t>(
              _right.num_rows(), stream, temp_mr)
          : cuda::device_buffer<cuda::std::uint32_t>{stream, temp_mr, 0, cuda::no_init};
      auto matched_build_rows =
        cudf::detail::device_scalar<cuda::std::uint64_t>(0, stream, temp_mr);
      probe(Join,
            left,
            preprocessed_left,
            hashes,
            match_counts,
            nullptr,
            matched_rows,
            matched_build_rows.data(),
            stream);

      // The count pass already tallied matched build rows. Preserve the exact complement size
      // for output allocation and full-join finalization before releasing its device scalar.
      if constexpr (Join == join_kind::FULL_JOIN) {
        // Each build row is tallied at most once, so narrowing to size_type is safe.
        auto const matched = matched_build_rows.value(stream);
        unmatched_right_rows =
          static_cast<size_type>(static_cast<cuda::std::uint64_t>(_right.num_rows()) - matched);
      }
      return hash_graph_match_offsets(match_counts, stream);
    }();

    auto const extra_capacity =
      static_cast<std::size_t>(unmatched_right_rows.value_or(size_type{0}));
    // For a full join the final size includes the unmatched right rows, so validate only now.
    validate_output_size(join_size + extra_capacity);

    return retrieve(
      Join, left, preprocessed_left, hashes, offsets, join_size, 0, extra_capacity, stream, mr);
  }();

  if constexpr (Join == join_kind::FULL_JOIN) {
    // The retrieve kernels do not mark matched right rows, so let `finalize_full_join`
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

template <typename Hasher>
std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<Hasher>::retrieve(
  join_kind join,
  cudf::table_view const& left,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_left,
  device_span<hash_value_type const> hashes,
  device_span<cuda::std::int64_t const> offsets,
  std::size_t output_size,
  size_type left_index_offset,
  std::size_t extra_capacity,
  cuda::stream_ref stream,
  cudf::memory_resources mr) const
{
  CUDF_EXPECTS(hashes.size() == static_cast<std::size_t>(left.num_rows()) &&
                 offsets.size() == hashes.size() + 1,
               "Need one hash per probe row and one offset per probe row plus the end offset");
  auto left_indices = std::make_unique<rmm::device_uvector<size_type>>(
    output_size + extra_capacity, stream, mr.get_output_mr());
  auto right_indices = std::make_unique<rmm::device_uvector<size_type>>(
    output_size + extra_capacity, stream, mr.get_output_mr());
  left_indices->resize(output_size, stream);
  right_indices->resize(output_size, stream);
  cudf::prefetch::detail::prefetch(*left_indices, stream);
  cudf::prefetch::detail::prefetch(*right_indices, stream);
  if (output_size == 0) { return {std::move(left_indices), std::move(right_indices)}; }

  auto emit = [&](auto equality, auto) {
    dispatch_bool(join != join_kind::INNER_JOIN, [&](auto is_outer) {
      launch_hash_graph_retrieve_kernel<decltype(is_outer)::value>(left.num_rows(),
                                                                   hashes.data(),
                                                                   offsets.data(),
                                                                   _impl->hash_graph(),
                                                                   left_index_offset,
                                                                   left_indices->data(),
                                                                   right_indices->data(),
                                                                   equality,
                                                                   stream);
    });
  };
  dispatch_join_comparator(
    _right, left, _preprocessed_right, preprocessed_left, _has_nulls, _nulls_equal, emit);

  return {std::move(left_indices), std::move(right_indices)};
}

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::retrieve(
  join_kind,
  cudf::table_view const&,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const&,
  device_span<hash_value_type const>,
  device_span<cuda::std::int64_t const>,
  std::size_t,
  size_type,
  std::size_t,
  cuda::stream_ref,
  cudf::memory_resources) const;

template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::join_retrieve<join_kind::INNER_JOIN>(cudf::table_view const&,
                                                                  std::optional<std::size_t>,
                                                                  cuda::stream_ref,
                                                                  cudf::memory_resources) const;
template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::join_retrieve<join_kind::LEFT_JOIN>(cudf::table_view const&,
                                                                 std::optional<std::size_t>,
                                                                 cuda::stream_ref,
                                                                 cudf::memory_resources) const;
template std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                   std::unique_ptr<rmm::device_uvector<size_type>>>
hash_join<hash_join_hasher>::join_retrieve<join_kind::FULL_JOIN>(cudf::table_view const&,
                                                                 std::optional<std::size_t>,
                                                                 cuda::stream_ref,
                                                                 cudf::memory_resources) const;

}  // namespace cudf::detail

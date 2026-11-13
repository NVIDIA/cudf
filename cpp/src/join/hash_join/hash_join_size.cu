/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"

#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/std/cstdint>

#include <limits>
#include <stdexcept>

namespace cudf::detail {

template <typename Hasher>
template <join_kind Join>
std::size_t hash_join<Hasher>::join_size(cudf::table_view const& left,
                                         cuda::stream_ref stream) const
{
  cudf::memory_resources mr = cudf::get_current_device_resource_ref();
  static_assert(Join == join_kind::INNER_JOIN || Join == join_kind::LEFT_JOIN);

  CUDF_FUNC_RANGE();
  validate_hash_join_probe(_right, left, _has_nulls);

  if constexpr (Join == join_kind::INNER_JOIN) {
    if (_is_empty) { return 0; }
  } else {
    if (_is_empty) { return left.num_rows(); }
  }

  auto total_matches =
    cudf::detail::device_scalar<cuda::std::uint64_t>(0, stream, mr.get_temporary_mr());
  probe(Join, left, {}, {}, total_matches.data(), {}, nullptr, stream);
  return static_cast<std::size_t>(total_matches.value(stream));
}

template <typename Hasher>
template <join_kind Join>
std::size_t hash_join<Hasher>::join_size(cudf::table_view const& left,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr) const
{
  static_assert(Join == join_kind::FULL_JOIN);

  CUDF_FUNC_RANGE();
  validate_hash_join_probe(_right, left, _has_nulls);
  if (_is_empty) { return left.num_rows(); }

  auto total_matches =
    cudf::detail::device_scalar<cuda::std::uint64_t>(0, stream, mr.get_output_mr());
  auto matched_groups = cudf::detail::make_zeroed_device_buffer_async<cuda::std::uint32_t>(
    _right.num_rows(), stream, mr.get_output_mr());
  auto matched_build_rows =
    cudf::detail::device_scalar<cuda::std::uint64_t>(0, stream, mr.get_output_mr());
  probe(join_kind::FULL_JOIN,
        left,
        {},
        {},
        total_matches.data(),
        matched_groups,
        matched_build_rows.data(),
        stream);

  auto const left_output_size   = total_matches.value(stream);
  auto const matched_right_rows = matched_build_rows.value(stream);
  auto const right_rows         = static_cast<cuda::std::uint64_t>(_right.num_rows());
  auto const output_size        = left_output_size + right_rows - matched_right_rows;
  CUDF_EXPECTS(output_size <= std::numeric_limits<std::size_t>::max(),
               "Join output size overflowed",
               std::overflow_error);
  return static_cast<std::size_t>(output_size);
}

template std::size_t hash_join<hash_join_hasher>::join_size<join_kind::INNER_JOIN>(
  cudf::table_view const&, cuda::stream_ref) const;
template std::size_t hash_join<hash_join_hasher>::join_size<join_kind::LEFT_JOIN>(
  cudf::table_view const&, cuda::stream_ref) const;
template std::size_t hash_join<hash_join_hasher>::join_size<join_kind::FULL_JOIN>(
  cudf::table_view const&, cuda::stream_ref, cudf::memory_resources) const;

}  // namespace cudf::detail

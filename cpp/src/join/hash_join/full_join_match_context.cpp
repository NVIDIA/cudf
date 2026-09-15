/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/join/hash_join.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/join/hash_join.hpp>
#include <cudf/join/join.hpp>

namespace cudf::detail {

template <typename Hasher>
cudf::join_match_context hash_join<Hasher>::full_join_match_context(
  cudf::table_view const& left, cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
{
  cudf::scoped_range range{"hash_join::full_join_match_context"};
  return cudf::join_match_context{left, make_match_counts(join_kind::FULL_JOIN, left, stream, mr)};
}

template cudf::join_match_context cudf::hash_join::impl_type::full_join_match_context(
  cudf::table_view const& left, cuda::stream_ref stream, rmm::device_async_resource_ref mr) const;

}  // namespace cudf::detail

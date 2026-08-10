/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "hash_join_impl.cuh"
#include "join/join_common_utils.hpp"

#include <cudf/detail/join/hash_join.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <memory>

namespace cudf::detail {

using hash_join_hasher = cudf::hashing::detail::MurmurHash3_x86_32<cudf::hash_value_type>;
using hash_table_t     = typename cudf::detail::hash_join<hash_join_hasher>::impl::hash_table_t;

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type);

std::unique_ptr<rmm::device_uvector<size_type>> make_join_match_counts(
  table_view const& right,
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_right,
  cudf::detail::hash_table_t const& hash_table,
  bool is_empty,
  bool has_nulls,
  null_equality compare_nulls,
  join_kind join,
  table_view const& left,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

}  // namespace cudf::detail

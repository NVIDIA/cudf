/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "hash_join_impl.cuh"

#include <cudf/detail/join/hash_join.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/join/join.hpp>
#include <cudf/table/table_view.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <cuda/stream_ref>

#include <memory>

namespace cudf::detail {

using hash_join_hasher = cudf::hashing::detail::MurmurHash3_x86_32<cudf::hash_value_type>;

bool is_trivial_join(table_view const& left, table_view const& right, join_kind join_type);

void validate_hash_join_probe(table_view const& right, table_view const& left, bool has_nulls);

}  // namespace cudf::detail

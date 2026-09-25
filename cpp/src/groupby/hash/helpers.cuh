/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/types.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

namespace cudf::groupby::detail::hash {
/// Number of threads to handle each input element
CUDF_HOST_DEVICE auto constexpr GROUPBY_CG_SIZE = 1;

/// Number of slots per thread
CUDF_HOST_DEVICE auto constexpr GROUPBY_BUCKET_SIZE = 1;

using row_hash_t = cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                              cudf::nullate::DYNAMIC>;

using row_comparator_t = cudf::detail::row::equality::device_row_comparator<
  false,
  cudf::nullate::DYNAMIC,
  cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

using nullable_row_comparator_t = cudf::detail::row::equality::device_row_comparator<
  true,
  cudf::nullate::DYNAMIC,
  cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

}  // namespace cudf::groupby::detail::hash

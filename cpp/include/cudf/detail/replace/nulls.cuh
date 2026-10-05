/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/types.hpp>

#include <cuda/std/tuple>

namespace cudf {
namespace detail {

using idx_valid_pair_t = cuda::std::tuple<cudf::size_type, bool>;

/**
 * @brief Functor used by `replace_nulls(replace_policy)` to determine the index to gather from in
 * the result column.
 *
 * Binary functor passed to `inclusive_scan` or `inclusive_scan_by_key`. Arguments are a tuple of
 * index and validity of a row. Returns a tuple of current index and a discarded boolean if current
 * row is valid, otherwise a tuple of the nearest non-null row index and a discarded boolean.
 */
struct replace_policy_functor {
  __device__ idx_valid_pair_t operator()(idx_valid_pair_t const& lhs, idx_valid_pair_t const& rhs)
  { return cuda::std::get<1>(rhs) ? rhs : lhs; }
};

}  // namespace detail
}  // namespace cudf

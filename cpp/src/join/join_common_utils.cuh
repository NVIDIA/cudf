/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "join_common_utils.hpp"

#include <cudf/detail/iterator.cuh>
#include <cudf/detail/join/join_key.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/cuda.cuh>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf::detail {
template <typename Hasher>
struct pair_fn {
  CUDF_HOST_DEVICE pair_fn(Hasher hash) : _hash{std::move(hash)} {}

  __device__ join_key<> operator()(size_type i) const noexcept
  {
    return join_key<>{to_join_hash(_hash(i)), to_cuco_index(i)};
  }

 private:
  Hasher _hash;
};

/**
 * @brief Device functor to determine if a row is valid.
 */
class row_is_valid {
 public:
  row_is_valid(bitmask_type const* row_bitmask) : _row_bitmask{row_bitmask} {}

  __device__ bool operator()(size_type const& i) const noexcept
  {
    return cudf::bit_is_set(_row_bitmask, i);
  }

 private:
  bitmask_type const* _row_bitmask;
};

/**
 * @brief Device functor to determine if an index is contained in a range.
 */
template <typename T>
struct valid_range {
  T start, stop;
  __host__ __device__ valid_range(T const begin, T const end) : start(begin), stop(end) {}

  __host__ __device__ __forceinline__ bool operator()(T const index)
  {
    return ((index >= start) && (index < stop));
  }
};

}  // namespace cudf::detail

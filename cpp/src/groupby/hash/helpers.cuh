/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/types.hpp>

#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>

namespace cudf::groupby::detail::hash {
/// Number of threads to handle each input element
auto constexpr GROUPBY_CG_SIZE = 1;

/// Number of slots per thread
auto constexpr GROUPBY_BUCKET_SIZE = 1;

/// Thread block size
auto constexpr GROUPBY_BLOCK_SIZE = 128;

/// Threshold cardinality to switch between shared memory aggregations and global memory
/// aggregations
auto constexpr GROUPBY_CARDINALITY_THRESHOLD = 128;

/// Threshold to switch between two strategies: one is to output the aggregation results directly to
/// the final dense output columns, the other is to output the results to sparse intermediate
/// buffers then gather to the final dense output columns.
auto constexpr GROUPBY_DENSE_OUTPUT_THRESHOLD = 2;

// We add additional `block_size`, because after the number of elements in the local hash set
// exceeds the threshold, all threads in the thread block can still insert one more element.
/// The maximum number of elements handled per block
auto constexpr GROUPBY_SHM_MAX_ELEMENTS = GROUPBY_CARDINALITY_THRESHOLD + GROUPBY_BLOCK_SIZE;

// GROUPBY_SHM_MAX_ELEMENTS with 0.7 occupancy
/// Shared memory hash set extent type
using shmem_extent_t =
  cuco::extent<cudf::size_type,
               static_cast<cudf::size_type>(static_cast<double>(GROUPBY_SHM_MAX_ELEMENTS) * 1.43)>;

/// Number of slots needed by each shared memory hash set
auto constexpr valid_extent =
  cuco::make_valid_extent<GROUPBY_CG_SIZE, GROUPBY_BUCKET_SIZE>(shmem_extent_t{});

using row_hash_t = cudf::detail::row::hash::device_row_hasher<cudf::hashing::detail::default_hash,
                                                              cudf::nullate::DYNAMIC>;

/// Adapter to cudf row hasher with caching support.
class row_hasher_with_cache_t {
  row_hash_t hasher;
  hash_value_type const* values;

 public:
  row_hasher_with_cache_t(row_hash_t const& hasher,
                          hash_value_type const* values = nullptr) noexcept
    : hasher(hasher), values(values)
  {
  }

  __device__ hash_value_type operator()(size_type const idx) const noexcept
  {
    if (values) { return values[idx]; }
    return hasher(idx);
  }
};

/// Probing scheme type used by groupby hash table
using probing_scheme_t = cuco::linear_probing<GROUPBY_CG_SIZE, row_hasher_with_cache_t>;

using row_comparator_t = cudf::detail::row::equality::device_row_comparator<
  false,
  cudf::nullate::DYNAMIC,
  cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

using nullable_row_comparator_t = cudf::detail::row::equality::device_row_comparator<
  true,
  cudf::nullate::DYNAMIC,
  cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

using global_set_t = cuco::static_set<cudf::size_type,
                                      cuco::extent<int64_t>,
                                      cuda::thread_scope_device,
                                      row_comparator_t,
                                      probing_scheme_t,
                                      rmm::mr::polymorphic_allocator<char>,
                                      cuco::storage<GROUPBY_BUCKET_SIZE>>;

using nullable_global_set_t = cuco::static_set<cudf::size_type,
                                               cuco::extent<int64_t>,
                                               cuda::thread_scope_device,
                                               nullable_row_comparator_t,
                                               probing_scheme_t,
                                               rmm::mr::polymorphic_allocator<char>,
                                               cuco::storage<GROUPBY_BUCKET_SIZE>>;

template <typename Op>
using hash_set_ref_t =
  cuco::static_set_ref<cudf::size_type,
                       cuda::thread_scope_device,
                       row_comparator_t,
                       probing_scheme_t,
                       cuco::bucket_storage_ref<cudf::size_type,
                                                GROUPBY_BUCKET_SIZE,
                                                cuco::valid_extent<int64_t, cuco::dynamic_extent>>,
                       Op>;

template <typename Op>
using nullable_hash_set_ref_t =
  cuco::static_set_ref<cudf::size_type,
                       cuda::thread_scope_device,
                       nullable_row_comparator_t,
                       probing_scheme_t,
                       cuco::bucket_storage_ref<cudf::size_type,
                                                GROUPBY_BUCKET_SIZE,
                                                cuco::valid_extent<int64_t, cuco::dynamic_extent>>,
                       Op>;
}  // namespace cudf::groupby::detail::hash

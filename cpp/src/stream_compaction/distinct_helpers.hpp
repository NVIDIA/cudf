/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/hashing.hpp>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/stream_compaction.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/export.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>
#include <rmm/resource_ref.hpp>

#include <cuco/extent.cuh>
#include <cuco/probing_scheme.cuh>
#include <cuco/static_set.cuh>
#include <cuco/storage.cuh>
#include <cuda/atomic>
#include <cuda/stream>

#include <cstdint>
#include <limits>

namespace cudf::detail {

/**
 * @brief Return the reduction identity used to initialize results of `hash_reduce_by_row`.
 *
 * @param keep A value of `duplicate_keep_option` type, must not be `KEEP_ANY`.
 * @return The initial reduction value.
 */
auto constexpr reduction_init_value(duplicate_keep_option keep)
{
  switch (keep) {
    case duplicate_keep_option::KEEP_FIRST: return std::numeric_limits<size_type>::max();
    case duplicate_keep_option::KEEP_LAST: return std::numeric_limits<size_type>::min();
    case duplicate_keep_option::KEEP_NONE: return size_type{0};
    default: CUDF_UNREACHABLE("This function should not be called with KEEP_ANY");
  }
}

CUDF_HIDDEN void initialize_reduction_results(size_type* results,
                                              size_type num_rows,
                                              duplicate_keep_option keep,
                                              cuda::stream_ref stream);

CUDF_HIDDEN size_type copy_reduction_results(size_type const* results,
                                             size_type num_rows,
                                             size_type* output,
                                             duplicate_keep_option keep,
                                             cuda::stream_ref stream);

struct distinct_precomputed_hash {
  CUDF_HOST_DEVICE constexpr distinct_precomputed_hash(hash_value_type const* hashes)
    : _hashes{hashes}
  {
  }

  __device__ __forceinline__ hash_value_type operator()(size_type i) const noexcept
  {
    return _hashes[i];
  }

 private:
  hash_value_type const* _hashes;
};

template <typename RowEqual,
          typename RowHash = cudf::detail::row::hash::
            device_row_hasher<cudf::hashing::detail::default_hash, cudf::nullate::DYNAMIC>>
using distinct_set_t = cuco::static_set<size_type,
                                        cuco::extent<int64_t>,
                                        cuda::thread_scope_device,
                                        RowEqual,
                                        cuco::linear_probing<1, RowHash>,
                                        rmm::mr::polymorphic_allocator<char>,
                                        cuco::storage<1>>;

/**
 * @brief Returns one unspecified row index from each group of equal rows.
 *
 * @tparam Set The type of the auxiliary set
 * @param set The auxiliary set used to identify groups of equal rows
 * @param num_rows The number of input rows
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A device vector containing one row index from each group
 */
template <typename Set>
rmm::device_uvector<size_type> reduce_by_row_keep_any(Set& set,
                                                      size_type num_rows,
                                                      cuda::stream_ref stream,
                                                      rmm::device_async_resource_ref mr);

/**
 * @brief Returns row indices selected from groups of equal rows according to `keep`.
 *
 * `KEEP_FIRST` returns the smallest row index in each group, `KEEP_LAST` returns the largest, and
 * `KEEP_NONE` returns indices only for singleton groups.
 *
 * @tparam Set The type of the auxiliary set
 * @param set The auxiliary set used to identify groups of equal rows
 * @param num_rows The number of input rows
 * @param keep The duplicate selection mode; must not be `KEEP_ANY`
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A device vector containing the selected row indices
 */
template <typename Set>
rmm::device_uvector<size_type> reduce_by_row_keep_first_last_none(
  Set& set,
  size_type num_rows,
  duplicate_keep_option keep,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr);

/**
 * @brief Returns row indices selected from groups of equal rows according to `keep`.
 *
 * @tparam Set The type of the auxiliary set
 * @param set The auxiliary set used to identify groups of equal rows
 * @param num_rows The number of input rows
 * @param keep The duplicate selection mode
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Device memory resource used to allocate the returned vector
 * @return A device vector containing the selected row indices
 */
template <typename Set>
rmm::device_uvector<size_type> reduce_by_row(Set& set,
                                             size_type num_rows,
                                             duplicate_keep_option keep,
                                             cuda::stream_ref stream,
                                             rmm::device_async_resource_ref mr)
{
  if (keep == duplicate_keep_option::KEEP_ANY) {
    return reduce_by_row_keep_any(set, num_rows, stream, mr);
  }
  return reduce_by_row_keep_first_last_none(set, num_rows, keep, stream, mr);
}
}  // namespace cudf::detail

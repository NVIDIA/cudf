/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"

#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/copy.h>

#include <cstddef>
#include <cstring>

namespace cudf::groupby {

/**
 * @brief Invokes a row-equality comparator stored in device memory indirectly.
 *
 * Prevents the underlying comparator's template graph from being inlined into callers.
 *
 * @tparam RowEquality Device-callable row-equality comparator type
 */
template <typename RowEquality>
struct indirect_row_equality {
  RowEquality const* row_equal;  ///< Non-owning device pointer to the row-equality comparator

  __attribute__((noinline)) __device__ bool operator()(size_type lhs, size_type rhs) const noexcept
  {
    return (*row_equal)(lhs, rhs);
  }
};

template <bool has_nested>
size_type streaming_groupby::impl::probe_and_insert_first_batch(
  std::shared_ptr<cudf::detail::row::hash::preprocessed_table> const& preprocessed_batch,
  cudf::nullate::DYNAMIC has_null,
  bitmask_type const* batch_bitmask,
  hash_value_type const* batch_hash_cache,
  size_type batch_size,
  size_type* target_indices,
  size_type* slot_offsets,
  size_type* batch_local_indices,
  cuda::stream_ref stream)
{
  auto const temp_mr        = cudf::get_current_device_resource_ref();
  auto const batch_self_cmp = cudf::detail::row::equality::self_comparator{preprocessed_batch};
  auto batch_self_eq        = batch_self_cmp.equal_to<has_nested>(has_null, null_equality::EQUAL);

  // This device buffer is intentional: invoking the primitive row comparator indirectly prevents
  // NVCC from inlining its expensive template graph into the CUB kernel, reducing build time and
  // binary size.
  auto h_batch_self_eq =
    cudf::detail::make_pinned_vector_async<std::byte>(sizeof(batch_self_eq), stream);
  std::memcpy(h_batch_self_eq.data(), &batch_self_eq, sizeof(batch_self_eq));
  rmm::device_buffer d_batch_self_eq(sizeof(batch_self_eq), stream, temp_mr);
  auto* const d_batch_self_eq_ptr = static_cast<decltype(batch_self_eq)*>(d_batch_self_eq.data());
  cudf::host_span<std::byte const> const h_batch_self_eq_span = h_batch_self_eq;
  cudf::detail::cuda_memcpy_async(
    cudf::device_span<std::byte>{static_cast<std::byte*>(d_batch_self_eq.data()),
                                 sizeof(batch_self_eq)},
    h_batch_self_eq_span,
    stream);
  auto const hasher       = offset_cache_hasher{batch_hash_cache, _max_distinct_keys};
  auto const set_ref_base = _key_set->ref(cuco::op::insert_and_find).rebind_hash_function(hasher);
  auto const first_batch_cmp = first_batch_comparator{
    indirect_row_equality<decltype(batch_self_eq)>{d_batch_self_eq_ptr}, _max_distinct_keys};
  auto* const base = _key_set->data();

  auto const out_end =
    thrust::copy_if(rmm::exec_policy_nosync(stream, temp_mr),
                    cuda::counting_iterator<size_type>(0),
                    cuda::counting_iterator<size_type>(batch_size),
                    batch_local_indices,
                    insert_and_check_fn{set_ref_base.rebind_key_eq(first_batch_cmp),
                                        batch_bitmask,
                                        _max_distinct_keys,
                                        base,
                                        target_indices,
                                        slot_offsets});
  return static_cast<size_type>(out_end - batch_local_indices);
}

}  // namespace cudf::groupby

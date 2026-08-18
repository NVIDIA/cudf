/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"

#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/copy.h>

namespace cudf::groupby {

template <bool has_nested>
size_type streaming_groupby::impl::probe_and_insert_subsequent(
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
  auto const batch_self_eq  = batch_self_cmp.equal_to<has_nested>(has_null, null_equality::EQUAL);
  auto const hasher         = offset_cache_hasher{batch_hash_cache, _max_distinct_keys};
  auto const set_ref_base   = _key_set->ref(cuco::op::insert_and_find).rebind_hash_function(hasher);
  auto* const base          = _key_set->data();

  auto d_cross_eqs = build_cross_comparators<has_nested>(
    preprocessed_batch, _preprocessed_batches, has_null, stream);
  auto const comparator =
    n_table_comparator{batch_self_eq, d_cross_eqs.data(), _key_loc->data(), _max_distinct_keys};

  auto const out_end = thrust::copy_if(rmm::exec_policy_nosync(stream, temp_mr),
                                       cuda::counting_iterator<size_type>(0),
                                       cuda::counting_iterator<size_type>(batch_size),
                                       batch_local_indices,
                                       insert_and_check_fn{set_ref_base.rebind_key_eq(comparator),
                                                           batch_bitmask,
                                                           _max_distinct_keys,
                                                           base,
                                                           target_indices,
                                                           slot_offsets});
  return static_cast<size_type>(out_end - batch_local_indices);
}

}  // namespace cudf::groupby

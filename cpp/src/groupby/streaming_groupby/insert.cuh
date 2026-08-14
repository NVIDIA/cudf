/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "common.cuh"
#include "groupby/common/utils.hpp"

#include <cudf/detail/gather.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>
#include <thrust/transform.h>

#include <string>

namespace cudf::groupby {

template <bool has_nested>
streaming_groupby::impl::batch_insert_result streaming_groupby::impl::probe_and_insert_impl(
  table_view const& batch_keys, cuda::stream_ref stream)
{
  auto const batch_size = batch_keys.num_rows();
  auto const temp_mr    = cudf::get_current_device_resource_ref();
  auto const has_null   = cudf::nullate::DYNAMIC{_has_nullable_keys};

  // Preprocess batch for row operators.
  auto preprocessed_batch = cudf::detail::row::hash::preprocessed_table::create(batch_keys, stream);
  auto const batch_hasher_obj = cudf::detail::row::hash::row_hasher{preprocessed_batch};
  auto const d_batch_hash     = batch_hasher_obj.device_hasher(has_null);

  // Compute the null-exclusion bitmask first so the hash cache pass can skip
  // hashing rows that will be excluded
  auto const skip_rows_with_nulls = _has_nullable_keys && _null_handling == null_policy::EXCLUDE;
  auto [bitmask_buffer, batch_bitmask] =
    skip_rows_with_nulls
      ? detail::compute_row_bitmask(batch_keys, stream)
      : std::pair<rmm::device_buffer, bitmask_type const*>{rmm::device_buffer{0, stream}, nullptr};

  // Precompute batch hash values.  Caching is faster than inlining the row hasher
  rmm::device_uvector<hash_value_type> batch_hash_cache(batch_size, stream, temp_mr);
  thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                    cuda::counting_iterator<size_type>(0),
                    cuda::counting_iterator<size_type>(batch_size),
                    batch_hash_cache.begin(),
                    conditional_hash_fn<decltype(d_batch_hash)>{d_batch_hash, batch_bitmask});

  // Pass 1 — fused insert_and_find + compact via `thrust::copy_if`.
  // Per-row work in the predicate: writes target_indices and slot_offsets.
  // Output: batch row indices of newly inserted keys, compacted into batch_local_indices.
  // Count: iterator distance returned by copy_if inside the helper.
  // slot_offsets stores 4-byte slot offsets (vs. 8-byte raw pointers) to halve temp memory.
  rmm::device_uvector<size_type> target_indices(batch_size, stream, temp_mr);
  rmm::device_uvector<size_type> slot_offsets(batch_size, stream, temp_mr);
  rmm::device_uvector<size_type> batch_local_indices(batch_size, stream, temp_mr);

  // First batch has no compacted batches yet, so all slot values are transient (>=
  // _max_distinct_keys) and only the batch-self equality branch of n_table_comparator
  // can fire.  Dispatch to a dedicated helper that uses first_batch_comparator to skip
  // the cross-table dispatch, the cross-comparator build, and the dense-ID branches.
  // The two helpers live in separate TUs to parallelize the heavy cuco/thrust template
  // instantiations.
  size_type const new_distinct_keys =
    _compacted_batches.empty()
      ? probe_and_insert_first_batch<has_nested>(preprocessed_batch,
                                                 has_null,
                                                 batch_bitmask,
                                                 batch_hash_cache.data(),
                                                 batch_size,
                                                 target_indices.data(),
                                                 slot_offsets.data(),
                                                 batch_local_indices.data(),
                                                 stream)
      : probe_and_insert_subsequent<has_nested>(preprocessed_batch,
                                                has_null,
                                                batch_bitmask,
                                                batch_hash_cache.data(),
                                                batch_size,
                                                target_indices.data(),
                                                slot_offsets.data(),
                                                batch_local_indices.data(),
                                                stream);
  batch_local_indices.resize(new_distinct_keys, stream);

  if (new_distinct_keys > 0) {
    // Bound check: the hash set has already been written above (transient slot values),
    // so on failure the object is left invalidated; further aggregate()/merge() calls
    // will throw immediately while finalize() can still recover partial results.
    if (_distinct_keys + new_distinct_keys > _max_distinct_keys) {
      _invalidated = true;
      CUDF_FAIL("Distinct key count (" + std::to_string(_distinct_keys + new_distinct_keys) +
                ") would exceed max_distinct_keys (" + std::to_string(_max_distinct_keys) + ").");
    }

    // Gather compacted distinct keys from the batch.
    auto compacted = cudf::detail::gather(batch_keys,
                                          batch_local_indices,
                                          out_of_bounds_policy::DONT_CHECK,
                                          cudf::negative_index_policy::NOT_ALLOWED,
                                          stream,
                                          temp_mr);

    auto preprocessed_compacted =
      cudf::detail::row::hash::preprocessed_table::create(compacted->view(), stream);

    // Store the compacted batch.
    auto const new_batch_id    = static_cast<size_type>(_compacted_batches.size());
    auto const dense_id_offset = _distinct_keys;
    _compacted_batches.push_back(std::move(compacted));
    _preprocessed_batches.push_back(preprocessed_compacted);

    // Pass 2 — fused: rewrites slots from transient encoding to dense IDs and
    // writes the {batch_id, row} companion entry.
    auto* const base = _key_set->data();
    thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                       cuda::counting_iterator<size_type>(0),
                       new_distinct_keys,
                       finalize_new_key_fn{batch_local_indices.data(),
                                           base,
                                           slot_offsets.data(),
                                           _key_loc->data(),
                                           new_batch_id,
                                           dense_id_offset});

    thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                       cuda::counting_iterator<size_type>(0),
                       batch_size,
                       update_transient_target_indices_fn{
                         base, slot_offsets.data(), _max_distinct_keys, target_indices.data()});

    _distinct_keys += new_distinct_keys;
  }
  // If new_distinct_keys == 0, target_indices is already final from Pass 1 — every
  // slot held a dense ID at probe time, so *iter was already the correct dense ID.

  return batch_insert_result{
    std::move(target_indices), new_distinct_keys, std::move(bitmask_buffer)};
}

}  // namespace cudf::groupby

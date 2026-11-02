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

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/for_each.h>

#include <string>

namespace cudf::groupby {

template <bool has_nested>
cuda::device_buffer<size_type> streaming_groupby::impl::probe_and_insert_impl(
  table_view const& batch_keys, bool use_partials, cuda::stream_ref stream)
{
  auto const batch_size = batch_keys.num_rows();
  auto const temp_mr    = cudf::get_current_device_resource_ref();
  auto const has_null   = cudf::nullate::DYNAMIC{_has_nullable_keys};

  // Preprocess batch for row operators.
  auto preprocessed_batch =
    cudf::detail::row::hash::preprocessed_table::create(batch_keys, stream, temp_mr);

  // Skip hashing and insertion for rows excluded by the null policy.
  auto const skip_rows_with_nulls = _has_nullable_keys && _null_handling == null_policy::EXCLUDE;
  auto [bitmask_buffer, batch_bitmask] =
    skip_rows_with_nulls
      ? detail::compute_row_bitmask(batch_keys, stream)
      : std::pair<cuda::device_buffer<cudf::bitmask_type>, bitmask_type const*>{
          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream), nullptr};

  // Slots are stored in target_indices until dense IDs have been assigned. Only
  // the representative row list needs separate batch-sized scratch.
  cuda::device_buffer<size_type> target_indices(stream, temp_mr, batch_size, cuda::no_init);
  cuda::device_buffer<size_type> batch_local_indices(stream, temp_mr, batch_size, cuda::no_init);

  auto* const base = _key_set->data();
  size_type new_distinct_keys{};
  {  // Release the cross comparators before aggregate state can grow.
    auto const capacity       = static_cast<cuda::std::uint32_t>(_key_set->size());
    auto const batch_self_cmp = cudf::detail::row::equality::self_comparator{preprocessed_batch};
    auto const batch_self_eq  = batch_self_cmp.equal_to<has_nested>(has_null, null_equality::EQUAL);
    // Nested hashes include structural validity when null checking is enabled. Keep it enabled
    // from the first batch so later nulls cannot change the hashes of already stored valid keys.
    auto const hasher = cudf::detail::row::hash::row_hasher{preprocessed_batch}.device_hasher(
      cudf::nullate::DYNAMIC{has_nested || _has_nullable_keys});
    auto const d_cross_eqs = build_cross_comparators<has_nested>(
      preprocessed_batch, _preprocessed_batches, has_null, stream, cudf::memory_resources{temp_mr});
    auto const comparator = n_table_comparator{
      batch_self_eq, d_cross_eqs.data(), _key_batch_ids->data(), _max_distinct_keys};
    thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                       cuda::counting_iterator<size_type>(0),
                       batch_size,
                       insert_fn{detail::hash::hash_set_ref{base, capacity, capacity},
                                 comparator,
                                 hasher,
                                 batch_bitmask,
                                 _max_distinct_keys,
                                 target_indices.data()});
    new_distinct_keys = static_cast<size_type>(
      thrust::copy_if(rmm::exec_policy_nosync(stream, temp_mr),
                      cuda::counting_iterator<size_type>(0),
                      cuda::counting_iterator<size_type>(batch_size),
                      batch_local_indices.begin(),
                      is_new_key_fn{target_indices.data(), base, _max_distinct_keys}) -
      batch_local_indices.begin());
  }

  // Check capacity before publishing new dense IDs. Even without new keys, a larger
  // batch budget can require more partials for the existing groups.
  auto const distinct_so_far = _distinct_keys.load(std::memory_order_relaxed);
  if (distinct_so_far + new_distinct_keys > _max_distinct_keys) {
    _invalidated = true;
    CUDF_FAIL("Distinct key count (" + std::to_string(distinct_so_far + new_distinct_keys) +
              ") would exceed max_distinct_keys (" + std::to_string(_max_distinct_keys) + ").");
  }
  // Aggregate storage and compacted keys outlive this call and belong to the object.
  auto const state_mr = cudf::memory_resources{_mr, temp_mr};
  ensure_aggregate_capacity(distinct_so_far + new_distinct_keys, stream, state_mr);

  if (new_distinct_keys > 0) {
    // Gather compacted distinct keys from the batch.
    auto compacted = cudf::detail::gather(
      batch_keys,
      cudf::device_span<size_type const>{batch_local_indices.data(),
                                         static_cast<std::size_t>(new_distinct_keys)},
      out_of_bounds_policy::DONT_CHECK,
      cudf::negative_index_policy::NOT_ALLOWED,
      stream,
      state_mr);

    auto preprocessed_compacted = cudf::detail::row::hash::preprocessed_table::create(
      compacted->view(), stream, state_mr.get_output_mr());

    // Reserve both vectors before enqueueing/publishing the new persistent group IDs.
    _compacted_batches.reserve(_compacted_batches.size() + 1);
    _preprocessed_batches.reserve(_preprocessed_batches.size() + 1);
    auto const new_batch_id    = static_cast<size_type>(_compacted_batches.size());
    auto const dense_id_offset = distinct_so_far;
    thrust::for_each_n(rmm::exec_policy_nosync(stream, temp_mr),
                       cuda::counting_iterator<size_type>(0),
                       new_distinct_keys,
                       finalize_new_key_fn{batch_local_indices.data(),
                                           base,
                                           target_indices.data(),
                                           _key_batch_ids->data(),
                                           new_batch_id,
                                           dense_id_offset});

    _compacted_batches.push_back(std::move(compacted));
    _preprocessed_batches.push_back(std::move(preprocessed_compacted));
    _distinct_keys.fetch_add(new_distinct_keys, std::memory_order_relaxed);
  }

  if (!_agg_kinds.empty()) {
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream, temp_mr),
      cuda::counting_iterator<size_type>(0),
      batch_size,
      resolve_aggregate_targets{
        base, target_indices.data(), _aggregate_capacity, use_partials ? _partials_per_group : 1});
  }
  return target_indices;
}

}  // namespace cudf::groupby

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "groupby/hash/helpers.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/hashing/detail/default_hash.cuh>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <cuco/static_set.cuh>
#include <cuda/std/functional>
#include <cuda/std/utility>

#include <memory>
#include <vector>

namespace cudf::groupby {

/*
 * Companion location for a stored dense ID: which compacted batch table the key
 * lives in (`first`) and the row index within that table (`second`).  Packed into
 * one 8-byte pair so the comparator does a single load instead of two.
 */
using key_location_t = cuda::std::pair<size_type, size_type>;

using streaming_probing_scheme_t =
  cuco::linear_probing<detail::hash::GROUPBY_CG_SIZE,
                       cudf::hashing::detail::default_hash<size_type>>;

using streaming_set_t = cuco::static_set<cudf::size_type,
                                         cuco::extent<int64_t>,
                                         cuda::thread_scope_device,
                                         cuda::std::equal_to<size_type>,
                                         streaming_probing_scheme_t,
                                         rmm::mr::polymorphic_allocator<char>,
                                         cuco::storage<detail::hash::GROUPBY_BUCKET_SIZE>>;

/*
 * Comparator for the first batch only.  All slot values are transient (encoded as
 * `max_distinct_keys + row_idx`) since no dense IDs exist yet; this wrapper subtracts
 * the offset and delegates to the batch self-equality.
 */
template <typename RowEqT>
struct first_batch_comparator {
  RowEqT batch_self_eq;
  size_type max_distinct_keys;

  __device__ bool operator()(size_type lhs, size_type rhs) const noexcept
  {
    return batch_self_eq(lhs - max_distinct_keys, rhs - max_distinct_keys);
  }
};

/*
 * N-table comparator for the persistent hash set.
 *
 * Slot values < max_distinct_keys are "stored" dense IDs resolved via the companion
 * vector key_loc[id] = {batch_id, row_within_batch} to a (compacted_batch_table,
 * row) location.  Slot values >= max_distinct_keys are transient batch values: row index
 * = value - max_distinct_keys in the current batch table.  The transient encoding lives
 * only for the duration of one probe_and_insert call; new keys are rewritten to
 * dense IDs before the next batch's insertion.
 *
 * Cross-comparators are pre-built as device_row_comparator(batch, compacted[k])
 * and stored in a device array.  Self-comparisons use batch_self_eq.
 */
template <typename RowEqT>
struct n_table_comparator {
  RowEqT batch_self_eq;           ///< Self-comparator on the current batch table
  RowEqT const* cross_eqs;        ///< Device array [num_compacted_batches]: batch vs compacted[k]
  key_location_t const* key_loc;  ///< {batch_id, row_in_compacted} per dense ID
  size_type max_distinct_keys;  ///< Threshold: idx >= max_distinct_keys is a transient batch value

  __attribute__((noinline)) __device__ bool operator()(size_type lhs, size_type rhs) const noexcept
  {
    bool const lhs_is_batch = (lhs >= max_distinct_keys);
    bool const rhs_is_batch = (rhs >= max_distinct_keys);

    if (lhs_is_batch && rhs_is_batch) {
      return batch_self_eq(lhs - max_distinct_keys, rhs - max_distinct_keys);
    }
    if (lhs_is_batch) {
      auto const loc = key_loc[rhs];
      return cross_eqs[loc.first](lhs - max_distinct_keys, loc.second);
    }
    if (rhs_is_batch) {
      auto const loc = key_loc[lhs];
      return cross_eqs[loc.first](rhs - max_distinct_keys, loc.second);
    }
    // During probe_and_insert, at least one operand is always the batch row being
    // inserted (transient-encoded), so two dense IDs cannot be compared here.
    CUDF_UNREACHABLE("n_table_comparator received two dense-ID operands");
  }
};

/*
 * Predicate used by `thrust::copy_if` to compact the batch row indices of newly
 * inserted keys in a single pass.  Each invocation calls `set_ref.insert_and_find`
 * for one batch row, records the resident slot's value in `target_indices[row_idx]`
 * and the slot offset in `slot_offsets[row_idx]` (or `CUDF_SIZE_TYPE_SENTINEL` for
 * null/excluded rows), and returns true iff this row inserted a new key.  When
 * the return is true `copy_if` writes `row_idx` to `batch_local_indices`; the
 * total insert count is the iterator distance returned by `copy_if`.
 *
 * If the batch produces no new keys, target_indices is already final.  If new
 * keys exist, Pass 2 rewrites the affected slots and the caller re-reads
 * target_indices via slot_offsets in a single transform.
 */
template <typename SetRef>
struct insert_and_check_fn {
  mutable SetRef set_ref;
  bitmask_type const* row_bitmask;
  size_type max_distinct_keys;
  size_type const* base;
  size_type* target_indices;
  size_type* slot_offsets;

  __device__ bool operator()(size_type row_idx) const
  {
    if (row_bitmask && !cudf::bit_is_set(row_bitmask, row_idx)) {
      target_indices[row_idx] = cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
      slot_offsets[row_idx]   = cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
      return false;
    }
    auto const [iter, inserted] = set_ref.insert_and_find(max_distinct_keys + row_idx);
    target_indices[row_idx]     = *iter;
    slot_offsets[row_idx]       = static_cast<size_type>(iter - base);
    return inserted;
  }
};

/*
 * Per-row hash producer used to populate the precomputed batch hash cache.
 * Returns 0 (a dummy value never read) for rows excluded by the null bitmask
 * under `null_policy::EXCLUDE`, avoiding wasted hash work for rows that won't
 * probe the set.
 */
template <typename RowHasher>
struct conditional_hash_fn {
  RowHasher row_hasher;
  bitmask_type const* row_bitmask;

  __device__ hash_value_type operator()(size_type i) const noexcept
  {
    if (row_bitmask && !cudf::bit_is_set(row_bitmask, i)) { return hash_value_type{0}; }
    return row_hasher(i);
  }
};

/*
 * Hasher backed by a precomputed cache, indexed by `idx - offset`.
 * In streaming_groupby `offset = max_distinct_keys`, so transient batch values
 * `max_distinct_keys + row_idx` resolve to `cache[row_idx]`.
 */
struct offset_cache_hasher {
  hash_value_type const* cache;
  size_type offset;
  __device__ hash_value_type operator()(size_type idx) const noexcept
  {
    return cache[idx - offset];
  }
};

/*
 * For each newly discovered key (dense rank `r` within this batch):
 *   1. Rewrite its slot from transient encoding to its dense ID.  Pass 2 runs
 *      after Pass 1's kernel completes and each slot has a single writer, so a
 *      plain store is sufficient.
 *   2. Write the (batch_id, row) pair to the companion vector at the dense ID.
 */
struct finalize_new_key_fn {
  size_type const*
    batch_local_indices;          ///< batch-local row indices of new keys [new_distinct_keys]
  size_type* base;                ///< hash set storage base
  size_type const* slot_offsets;  ///< slot offset per batch row [batch_size]
  key_location_t* key_loc;        ///< {batch_id, row_in_compacted} per dense ID
  size_type batch_id;             ///< the index of this batch in _compacted_batches
  size_type dense_id_offset;      ///< first dense ID assigned to this batch's new keys

  __device__ void operator()(size_type r) const
  {
    auto const dense_id    = dense_id_offset + r;
    auto const batch_local = batch_local_indices[r];

    *(base + slot_offsets[batch_local]) = dense_id;
    *(key_loc + dense_id)               = key_location_t{batch_id, r};
  }
};

struct update_transient_target_indices_fn {
  size_type const* base;
  size_type const* slot_offsets;
  size_type max_distinct_keys;
  size_type* target_indices;

  __device__ void operator()(size_type i) const
  {
    if (target_indices[i] >= max_distinct_keys) { target_indices[i] = *(base + slot_offsets[i]); }
  }
};

template <bool has_nested_columns>
auto build_cross_comparators(
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_batch,
  std::vector<std::shared_ptr<cudf::detail::row::equality::preprocessed_table>> const&
    preprocessed_batches,
  cudf::nullate::DYNAMIC has_null,
  rmm::cuda_stream_view stream)
{
  using eq_t = cudf::detail::row::equality::device_row_comparator<
    has_nested_columns,
    cudf::nullate::DYNAMIC,
    cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

  auto const n       = static_cast<size_type>(preprocessed_batches.size());
  auto const temp_mr = cudf::get_current_device_resource_ref();

  std::vector<eq_t> h_eqs;
  h_eqs.reserve(n);
  for (size_type k = 0; k < n; ++k) {
    auto const cross_cmp = cudf::detail::row::equality::two_table_comparator{
      preprocessed_batch, preprocessed_batches[k]};
    auto const adapter = cross_cmp.equal_to<has_nested_columns>(has_null, null_equality::EQUAL);
    h_eqs.push_back(adapter.comparator);
  }

  return cudf::detail::make_device_uvector_async(h_eqs, stream, temp_mr);
}

/// The impl struct for streaming_groupby. Defined in impl.cu.
struct streaming_groupby::impl {
  std::vector<size_type> _key_indices;
  std::vector<streaming_aggregation_request> _requests_clone;
  size_type _max_distinct_keys;
  null_policy _null_handling;
  cuda::mr::any_resource<cuda::mr::device_accessible> _mr;

  bool _initialized{false};
  /// Set true once an `aggregate()` / `merge()` call has thrown after touching the
  /// hash set.  Subsequent `aggregate()` / `merge()` calls fail fast; only
  /// `finalize()` may still be called to recover partial results.
  bool _invalidated{false};
  /*
   * Number of distinct keys accumulated so far.  Also serves as the high-water
   * mark of dense IDs in the persistent hash set: stored slot values are in
   * [0, _distinct_keys).
   */
  size_type _distinct_keys{0};
  bool _has_nullable_keys{false};
  bool _has_nested_keys{false};

  // -- Compacted batch storage --
  std::vector<std::unique_ptr<table>> _compacted_batches;
  std::vector<std::shared_ptr<cudf::detail::row::equality::preprocessed_table>>
    _preprocessed_batches;

  /// Empty (0-row) table preserving the key schema, used by gather_distinct_keys when
  /// no batches produced any groups (e.g. first batch empty or fully null-excluded).
  std::unique_ptr<table> _empty_key_schema;

  /// Companion vector indexed by dense ID, sized to max_distinct_keys.
  /// Each entry is {batch_id, row_in_compacted_batch}.
  std::unique_ptr<rmm::device_uvector<key_location_t>> _key_loc;

  std::vector<size_type> _request_first_agg_offset;
  std::vector<aggregation::Kind> _agg_kinds;
  std::vector<std::unique_ptr<aggregation>> _agg_objects;
  std::vector<int8_t> _is_agg_intermediate;
  bool _has_compound_aggs{false};

  /*
   * Aggregation results table, pre-allocated to max_distinct_keys rows.
   * Indexed by dense ID (== row index).
   */
  std::unique_ptr<table> _agg_results;
  /*
   * Cached mutable_table_device_view of `_agg_results`.  `_agg_results` is allocated
   * once at initialize() and never resized, so this device-side descriptor can be
   * built once and reused on every aggregate() / merge() call rather than rebuilt
   * (which requires a host-to-device copy of the column metadata).
   */
  std::unique_ptr<mutable_table_device_view, void (*)(mutable_table_device_view*)> _d_agg_results;
  std::vector<size_type> _value_col_indices;
  rmm::device_uvector<aggregation::Kind> _d_agg_kinds;

  std::unique_ptr<streaming_set_t> _key_set;

  [[nodiscard]] size_type num_keys() const { return static_cast<size_type>(_key_indices.size()); }
  [[nodiscard]] bool has_state() const { return _initialized && _distinct_keys > 0; }

  impl(host_span<size_type const> key_indices,
       host_span<streaming_aggregation_request const> requests,
       size_type max_distinct_keys,
       null_policy null_handling,
       cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  void initialize(table_view const& data, rmm::cuda_stream_view stream);
  void create_key_set(rmm::cuda_stream_view stream);
  void update_nullable_state(table_view const& batch_keys);

  struct batch_insert_result {
    rmm::device_uvector<size_type> target_indices;
    size_type new_insertions;
    rmm::device_buffer bitmask_buffer;
  };

  batch_insert_result probe_and_insert(table_view const& batch_keys, rmm::cuda_stream_view stream);

  /*
   * Template implementation of probe_and_insert, split by has_nested.
   * Defined in insert.cuh, instantiated in insert.cu and insert_nested.cu.
   */
  template <bool has_nested>
  batch_insert_result probe_and_insert_impl(table_view const& batch_keys,
                                            rmm::cuda_stream_view stream);

  /*
   * Two helpers split off probe_and_insert_impl for compile-time parallelism.
   * Each builds the cuco set_ref + comparator and runs the fused insert/compact
   * via thrust::copy_if.  Split into separate TUs because both cuco and thrust
   * algorithm template instantiations dominate compile time.
   *
   * Defined in insert_first.cuh / insert_subsequent.cuh, instantiated in
   * insert_first{,_nested}.cu and insert_subsequent{,_nested}.cu respectively.
   */
  template <bool has_nested>
  size_type probe_and_insert_first_batch(
    std::shared_ptr<cudf::detail::row::hash::preprocessed_table> const& preprocessed_batch,
    cudf::nullate::DYNAMIC has_null,
    bitmask_type const* batch_bitmask,
    hash_value_type const* batch_hash_cache,
    size_type batch_size,
    size_type* target_indices,
    size_type* slot_offsets,
    size_type* batch_local_indices,
    rmm::cuda_stream_view stream);

  template <bool has_nested>
  size_type probe_and_insert_subsequent(
    std::shared_ptr<cudf::detail::row::hash::preprocessed_table> const& preprocessed_batch,
    cudf::nullate::DYNAMIC has_null,
    bitmask_type const* batch_bitmask,
    hash_value_type const* batch_hash_cache,
    size_type batch_size,
    size_type* target_indices,
    size_type* slot_offsets,
    size_type* batch_local_indices,
    rmm::cuda_stream_view stream);

  void do_aggregate(table_view const& data, rmm::cuda_stream_view stream);

  [[nodiscard]] std::unique_ptr<table> gather_agg_results(rmm::cuda_stream_view stream,
                                                          rmm::device_async_resource_ref mr) const;
  [[nodiscard]] std::unique_ptr<table> gather_distinct_keys(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const;

  [[nodiscard]] std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> do_finalize(
    rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const;

  void do_merge(impl const& other, rmm::cuda_stream_view stream);
};

}  // namespace cudf::groupby

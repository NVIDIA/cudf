/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "groupby/hash/hash_csr_kernels.cuh"
#include "warp_aggregate.hpp"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_device_view.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <cuda/buffer>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <atomic>
#include <memory>
#include <mutex>
#include <vector>

namespace cudf::groupby {

/*
 * Minimal owning wrapper around a CUDA event, used to order the insertion phase of
 * `aggregate()` / `merge()` calls that overlap on different streams.  Only the insertion
 * phase needs this ordering; the aggregation phase updates each group with atomics and is
 * safe to overlap.
 */
class insert_order_event {
 public:
  insert_order_event() { CUDF_CUDA_TRY(cudaEventCreateWithFlags(&_event, cudaEventDisableTiming)); }
  ~insert_order_event() { cudaEventDestroy(_event); }
  insert_order_event(insert_order_event const&)            = delete;
  insert_order_event& operator=(insert_order_event const&) = delete;

  /// Makes `stream` wait for the most recently recorded insertion.  No-op before the first
  /// `record()`, which is exactly the behavior the first call needs.
  void wait(cuda::stream_ref stream) const
  {
    CUDF_CUDA_TRY(cudaStreamWaitEvent(stream.get(), _event));
  }

  /// Records completion of the insertion just enqueued on `stream`.
  void record(cuda::stream_ref stream) { CUDF_CUDA_TRY(cudaEventRecord(_event, stream.get())); }

 private:
  cudaEvent_t _event{};
};

/// Finish untracked work before releasing bookkeeping locks, preserving the original exception.
inline void synchronize_on_error(cuda::stream_ref stream) noexcept
{
  try {
    stream.sync();
  } catch (...) {
    // The caller is already handling an error; a synchronization failure must not replace it.
  }
}

// Dense IDs within a compacted batch are contiguous and follow its gathered row order.
// Keep their base once per comparator, so each group's companion entry needs only a batch ID.
template <typename RowEqT>
struct stored_batch_comparator {
  RowEqT equal;
  size_type dense_id_offset;

  __device__ bool operator()(size_type batch_row, size_type dense_id) const noexcept
  {
    return equal(batch_row, dense_id - dense_id_offset);
  }
};

/*
 * N-table comparator for the persistent hash set.
 *
 * Slot values < max_distinct_keys are "stored" dense IDs resolved via the companion
 * vector key_batch_ids[id] to a compacted batch. The comparator subtracts that
 * batch's first dense ID to obtain its row.  Slot values >= max_distinct_keys are transient batch
 * values: row index = value - max_distinct_keys in the current batch table.  The transient encoding
 * lives only for the duration of one probe_and_insert call; new keys are rewritten to dense IDs
 * before the next batch's insertion.
 *
 * Cross-comparators are pre-built as device_row_comparator(batch, compacted[k])
 * and stored in a device array.  Self-comparisons use batch_self_eq.
 */
template <typename RowEqT>
struct n_table_comparator {
  RowEqT batch_self_eq;                              ///< Self-comparator on the current batch table
  stored_batch_comparator<RowEqT> const* cross_eqs;  ///< Batch vs compacted[k], with dense-ID base
  size_type const* key_batch_ids;                    ///< Compacted batch ID per dense ID
  size_type max_distinct_keys;  ///< Threshold: idx >= max_distinct_keys is a transient batch value

  __attribute__((noinline)) __device__ bool operator()(size_type lhs, size_type rhs) const noexcept
  {
    // hash_set_ref::insert always compares its transient input row with a stored slot.
    auto const batch_row = lhs - max_distinct_keys;
    if (rhs >= max_distinct_keys) { return batch_self_eq(batch_row, rhs - max_distinct_keys); }
    return cross_eqs[key_batch_ids[rhs]](batch_row, rhs);
  }
};

/*
 * Inserts rows into the open-addressing slot table. Until dense IDs are assigned, target_indices
 * holds each row's slot bits; excluded rows hold the UINT32_MAX (no slot) sentinel.
 */
template <typename Equal, typename Hasher>
struct insert_fn {
  detail::hash::hash_set_ref set;
  Equal equal;
  Hasher hasher;
  bitmask_type const* row_bitmask;
  size_type max_distinct_keys;
  size_type* target_indices;

  __device__ void operator()(size_type row_idx) const
  {
    if (row_bitmask && !cudf::bit_is_set(row_bitmask, row_idx)) {
      target_indices[row_idx] = cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
      return;
    }
    size_type representative{};
    auto const slot =
      set.insert(max_distinct_keys + row_idx, hasher(row_idx), equal, representative).first;
    target_indices[row_idx] = cuda::std::bit_cast<size_type>(slot);
  }
};

struct is_new_key_fn {
  size_type const* target_indices;
  size_type const* slots;
  size_type max_distinct_keys;

  __device__ bool operator()(size_type row_idx) const noexcept
  {
    auto const slot = cuda::std::bit_cast<cuda::std::uint32_t>(target_indices[row_idx]);
    return slot != detail::hash::hash_csr_no_slot && slots[slot] == max_distinct_keys + row_idx;
  }
};

/*
 * For each newly discovered key (dense rank `r` within this batch), rewrites its slot from the
 * transient encoding to its dense ID and records its compacted batch ID. Each slot has a single
 * writer, so plain stores suffice.
 */
struct finalize_new_key_fn {
  size_type const*
    batch_local_indices;            ///< batch-local row indices of new keys [new_distinct_keys]
  size_type* base;                  ///< hash set storage base
  size_type const* target_indices;  ///< slot bits per batch row until finalization
  size_type* key_batch_ids;         ///< Compacted batch ID per dense ID
  size_type batch_id;               ///< the index of this batch in _compacted_batches
  size_type dense_id_offset;        ///< first dense ID assigned to this batch's new keys

  __device__ void operator()(size_type r) const
  {
    auto const dense_id    = dense_id_offset + r;
    auto const batch_local = batch_local_indices[r];

    auto const slot         = cuda::std::bit_cast<cuda::std::uint32_t>(target_indices[batch_local]);
    base[slot]              = dense_id;
    key_batch_ids[dense_id] = batch_id;
  }
};

template <bool has_nested_columns>
auto build_cross_comparators(
  std::shared_ptr<cudf::detail::row::equality::preprocessed_table> const& preprocessed_batch,
  std::vector<std::shared_ptr<cudf::detail::row::equality::preprocessed_table>> const&
    preprocessed_batches,
  cudf::nullate::DYNAMIC has_null,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  using eq_t = cudf::detail::row::equality::device_row_comparator<
    has_nested_columns,
    cudf::nullate::DYNAMIC,
    cudf::detail::row::equality::nan_equal_physical_equality_comparator>;

  auto const num_batches = static_cast<size_type>(preprocessed_batches.size());

  std::vector<stored_batch_comparator<eq_t>> h_eqs;
  size_type dense_id_offset = 0;
  h_eqs.reserve(num_batches);
  for (size_type k = 0; k < num_batches; ++k) {
    auto const cross_cmp = cudf::detail::row::equality::two_table_comparator{
      preprocessed_batch, preprocessed_batches[k]};
    auto const adapter = cross_cmp.equal_to<has_nested_columns>(has_null, null_equality::EQUAL);
    h_eqs.push_back({adapter.comparator, dense_id_offset});
    dense_id_offset += static_cast<table_device_view>(*preprocessed_batches[k]).num_rows();
  }

  return cudf::detail::make_device_buffer(h_eqs, stream, mr.get_output_mr());
}

/*
 * Maps slot bits to aggregate rows. With several partials, rows of the same thread block share
 * one partial so concurrent blocks spread their atomics over different copies of each group.
 */
struct resolve_aggregate_targets {
  size_type const* slots;
  size_type* targets;
  size_type capacity;
  size_type partials_per_group;

  __device__ void operator()(size_type row) const
  {
    auto const slot = cuda::std::bit_cast<cuda::std::uint32_t>(targets[row]);
    if (slot == detail::hash::hash_csr_no_slot) { return; }
    auto const partial = (row / partial_aggregation_block_size) % partials_per_group;
    targets[row]       = partial * capacity + slots[slot];
  }
};

/// The impl struct for streaming_groupby. Defined in impl.cu.
struct streaming_groupby::impl {
  std::vector<size_type> _key_indices;
  std::vector<streaming_aggregation_request> _requests_clone;
  size_type _max_distinct_keys;
  null_policy _null_handling;
  cuda::mr::any_resource<cuda::mr::device_accessible> _mr;

  /*
   * Serializes host insertion, result-capacity changes, and completion bookkeeping.
   * The lock lasts until each call has enqueued its kernels and completion event;
   * result kernels on different streams may still overlap when no growth/read requires a wait.
   */
  mutable std::mutex _insert_mutex;
  /// Orders the insertion phase across calls that supply different streams.
  insert_order_event _insert_done;
  // Fan-in of every queued result use. Bookkeeping is serialized by _insert_mutex.
  mutable insert_order_event _aggregate_done;
  // Writers may overlap one another, but must follow previous snapshot/source reads.
  mutable insert_order_event _results_read_done;

  bool _initialized{false};
  /// Set true once an `aggregate()` / `merge()` call has thrown after touching the
  /// hash set.  Subsequent `aggregate()` / `merge()` calls fail fast; only
  /// `finalize()` may still be called to recover partial results.  Atomic so the
  /// fail-fast check in `do_aggregate` can run ahead of `_insert_mutex`.
  std::atomic<bool> _invalidated{false};
  /*
   * Number of distinct keys accumulated so far.  Also serves as the high-water
   * mark of dense IDs in the persistent hash set: stored slot values are in
   * [0, _distinct_keys).
   */
  std::atomic<size_type> _distinct_keys{0};
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
  /// Each entry identifies the compacted batch containing this dense ID.
  std::unique_ptr<cuda::device_buffer<size_type>> _key_batch_ids;

  std::vector<size_type> _request_first_agg_offset;
  std::vector<aggregation::Kind> _agg_kinds;
  std::vector<std::unique_ptr<aggregation>> _agg_objects;
  std::vector<int8_t> _is_agg_intermediate;
  bool _has_compound_aggs{false};

  // Result columns grow geometrically with observed groups; lookup capacity stays fixed.
  size_type _aggregate_capacity{0};
  // Physical partial row = partial * _aggregate_capacity + group. Budget counts partial rows,
  // growing only when an observed batch budget at least doubles the previous one.
  size_type _partial_budget{0};
  size_type _partials_per_group{1};
  std::vector<data_type> _aggregate_input_types;
  std::unique_ptr<table> _agg_results;
  // Own columns during incremental growth. A failed allocation can leave a grown prefix and
  // an unchanged suffix; finalize() recovers both without assuming a uniform partial layout.
  std::vector<std::unique_ptr<column>> _growing_agg_columns;
  size_type _num_grown_agg_columns{0};
  std::vector<size_type> _value_col_indices;
  std::unique_ptr<cuda::device_buffer<aggregation::Kind>> _d_agg_kinds;

  std::unique_ptr<cuda::device_buffer<size_type>> _key_set;

  [[nodiscard]] size_type num_keys() const { return static_cast<size_type>(_key_indices.size()); }
  void ensure_not_invalidated() const
  {
    CUDF_EXPECTS(!_invalidated.load(std::memory_order_relaxed),
                 "streaming_groupby is in an invalidated state from a prior failure; "
                 "no further aggregate()/merge() is allowed.  finalize() may still be called.");
  }

  impl(host_span<size_type const> key_indices,
       host_span<streaming_aggregation_request const> requests,
       size_type max_distinct_keys,
       null_policy null_handling,
       cuda::mr::any_resource<cuda::mr::device_accessible> mr);

  void initialize(table_view const& data, cuda::stream_ref stream);
  void create_key_set(cuda::stream_ref stream);
  void ensure_aggregate_capacity(size_type required,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr);
  [[nodiscard]] std::unique_ptr<table> make_aggregate_results(size_type capacity,
                                                              size_type first_column,
                                                              size_type num_columns,
                                                              cuda::stream_ref stream,
                                                              cudf::memory_resources mr) const;
  // Call after enqueueing this operation, while holding _insert_mutex.
  void record_aggregate_completion(cuda::stream_ref stream) const
  {
    _aggregate_done.wait(stream);
    _aggregate_done.record(stream);
  }
  void record_result_read(cuda::stream_ref stream) const
  {
    record_aggregate_completion(stream);
    _results_read_done.record(stream);
  }
  void update_nullable_state(table_view const& batch_keys);

  /*
   * Inserts the batch keys and returns each row's aggregate row, or the no-slot sentinel for
   * excluded rows. `use_partials` spreads rows over the per-group partials (aggregate);
   * otherwise every row targets the group's first partial (merge).
   */
  cuda::device_buffer<size_type> probe_and_insert(table_view const& batch_keys,
                                                  bool use_partials,
                                                  cuda::stream_ref stream);

  /*
   * Template implementation of probe_and_insert, split by has_nested.
   * Defined in insert.cuh, instantiated in insert.cu and insert_nested.cu.
   */
  template <bool has_nested>
  cuda::device_buffer<size_type> probe_and_insert_impl(table_view const& batch_keys,
                                                       bool use_partials,
                                                       cuda::stream_ref stream);

  void do_aggregate(table_view const& data, cuda::stream_ref stream);

  [[nodiscard]] std::unique_ptr<table> gather_agg_results(cuda::stream_ref stream,
                                                          cudf::memory_resources mr) const;
  [[nodiscard]] std::unique_ptr<table> gather_distinct_keys(
    cuda::stream_ref stream, rmm::device_async_resource_ref mr) const;

  [[nodiscard]] std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
  finalize_gathered(std::unique_ptr<table> keys,
                    std::unique_ptr<table> agg_gathered,
                    cuda::stream_ref stream,
                    cudf::memory_resources mr) const;
  [[nodiscard]] std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> do_finalize(
    cuda::stream_ref stream, cudf::memory_resources mr) const;
  [[nodiscard]] std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
  do_finalize_and_release(cuda::stream_ref stream, cudf::memory_resources mr);

  void do_merge(impl const& other, cuda::stream_ref stream);
};

}  // namespace cudf::groupby

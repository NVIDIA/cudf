/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "groupby/common/utils.hpp"
#include "groupby/hash/extract_single_pass_aggs.hpp"
#include "groupby/hash/hash_compound_agg_finalizer.hpp"
#include "groupby/hash/output_utils.hpp"
#include "partial_aggregates.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/column/column_stream.hpp>
#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/fixed_point/fixed_point.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/stream>
#include <thrust/for_each.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudf::groupby {

namespace {

// Streaming still uses element_aggregator, whose atomic requirements are independent of the
// reductions used by ordinary hash groupby.
struct is_atomic_aggregation_supported {
  template <typename T, aggregation::Kind K>
  bool operator()() const
  {
    if constexpr (cudf::is_nested<T>()) {
      return false;
    } else if constexpr (std::is_same_v<T, numeric::decimal128> && K == aggregation::SUM) {
      // The existing decimal128 SUM implementation provides its own atomic addition.
      return true;
    } else {
      using Target = cudf::detail::target_type_t<T, K>;
      constexpr auto uses_storage =
        cudf::is_fixed_point<T>() &&
        (K == aggregation::MIN || K == aggregation::MAX || K == aggregation::SUM);
      using DeviceTarget =
        std::conditional_t<uses_storage, cudf::device_storage_type_t<Target>, Target>;
      if constexpr (!std::is_void_v<DeviceTarget>) {
        return cudf::has_atomic_support<DeviceTarget>();
      }
      return false;
    }
  }
};

void validate_requests(host_span<streaming_aggregation_request const> requests)
{
  for (auto const& req : requests) {
    CUDF_EXPECTS(req.aggregation != nullptr,
                 "streaming_aggregation_request must have a non-null aggregation.",
                 std::invalid_argument);
    CUDF_EXPECTS(detail::is_hash_aggregation(req.aggregation->kind) &&
                   req.aggregation->kind != aggregation::ARGMIN &&
                   req.aggregation->kind != aggregation::ARGMAX,
                 "Unsupported aggregation kind for streaming groupby. "
                 "ARGMIN/ARGMAX are not supported because row indices are batch-local.",
                 std::invalid_argument);
  }
}

// Group streaming requests by `column_index` so multiple aggregations on the same
// column become a single `aggregation_request{values, [aggs...]}`.  `extract_single_pass_aggs`
// then dedups repeated simple kinds within the group (e.g. SUM and MEAN both want SUM).
std::vector<aggregation_request> build_aggregation_requests(
  host_span<streaming_aggregation_request const> requests, table_view const& data)
{
  std::vector<aggregation_request> result;
  std::unordered_map<size_type, size_type> col_to_idx;
  col_to_idx.reserve(requests.size());
  for (auto const& req : requests) {
    auto cloned = std::unique_ptr<groupby_aggregation>{
      dynamic_cast<groupby_aggregation*>(req.aggregation->clone().release())};
    auto const [it, inserted] =
      col_to_idx.try_emplace(req.column_index, static_cast<size_type>(result.size()));
    if (inserted) {
      aggregation_request ar;
      ar.values = data.column(req.column_index);
      ar.aggregations.push_back(std::move(cloned));
      result.push_back(std::move(ar));
    } else {
      result[it->second].aggregations.push_back(std::move(cloned));
    }
  }
  return result;
}

}  // namespace

streaming_groupby::impl::impl(host_span<size_type const> key_indices,
                              host_span<streaming_aggregation_request const> requests,
                              size_type max_distinct_keys,
                              null_policy null_handling,
                              cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _max_distinct_keys{max_distinct_keys}, _null_handling{null_handling}, _mr{std::move(mr)}
{
  CUDF_EXPECTS(max_distinct_keys > 0, "max_distinct_keys must be positive.", std::invalid_argument);
  if (!key_indices.empty()) { _key_indices.assign(key_indices.begin(), key_indices.end()); }
  validate_requests(requests);

  for (auto const& req : requests) {
    streaming_aggregation_request clone;
    clone.column_index = req.column_index;
    clone.aggregation  = std::unique_ptr<groupby_aggregation>{
      dynamic_cast<groupby_aggregation*>(req.aggregation->clone().release())};
    _requests_clone.push_back(std::move(clone));
  }
}

void streaming_groupby::impl::initialize(table_view const& data, cuda::stream_ref stream)
{
  auto const mr = cudf::memory_resources{_mr, cudf::get_current_device_resource_ref()};

  // Detect nested key columns (struct, list) for comparator template dispatch.
  std::vector<column_view> key_cols;
  key_cols.reserve(_key_indices.size());
  for (auto idx : _key_indices) {
    key_cols.push_back(data.column(idx));
  }
  _has_nested_keys = cudf::detail::has_nested_columns(table_view{key_cols});

  std::vector<std::unique_ptr<column>> empty_key_cols;
  empty_key_cols.reserve(key_cols.size());
  for (auto const& kc : key_cols) {
    empty_key_cols.push_back(cudf::empty_like(kc));
  }
  _empty_key_schema = std::make_unique<table>(std::move(empty_key_cols));

  auto agg_requests = build_aggregation_requests(_requests_clone, data);

  auto [values_view, agg_kinds_hv, agg_objects, is_intermediate, has_compound] =
    detail::hash::extract_single_pass_aggs(agg_requests, stream);

  _agg_kinds.assign(agg_kinds_hv.begin(), agg_kinds_hv.end());
  _agg_objects         = std::move(agg_objects);
  _is_agg_intermediate = std::move(is_intermediate);
  _has_compound_aggs   = has_compound;

  // Reject aggregation kinds that are unsupported in streaming after decomposition.
  for (auto k : _agg_kinds) {
    CUDF_EXPECTS(k != aggregation::ARGMIN && k != aggregation::ARGMAX,
                 "Streaming groupby does not support MIN/MAX on variable-width types "
                 "(internally decomposed to ARGMIN/ARGMAX).",
                 std::invalid_argument);
    CUDF_EXPECTS(k != aggregation::SUM_OVERFLOW,
                 "Streaming groupby does not support SUM_OVERFLOW "
                 "(struct intermediate cannot be merged across batches).",
                 std::invalid_argument);
  }

  CUDF_EXPECTS(
    std::all_of(cuda::counting_iterator<size_type>{0},
                cuda::counting_iterator<size_type>{values_view.num_columns()},
                [&](auto i) {
                  auto const& values     = values_view.column(i);
                  auto const values_type = cudf::is_dictionary(values.type())
                                             ? cudf::dictionary_column_view(values).keys().type()
                                             : values.type();
                  return cudf::detail::dispatch_type_and_aggregation(
                    values_type, _agg_kinds[i], is_atomic_aggregation_supported{});
                }),
    "streaming_groupby does not support this combination of value type and aggregation kind.",
    std::invalid_argument);

  // Keep only type metadata, never a view or owner of the caller's value buffers.
  _aggregate_input_types.reserve(values_view.num_columns());
  for (auto const& values : values_view) {
    _aggregate_input_types.push_back(cudf::is_dictionary(values.type())
                                       ? cudf::dictionary_column_view(values).keys().type()
                                       : values.type());
  }
  _agg_results = make_aggregate_results(0, 0, _agg_kinds.size(), stream, mr);

  _d_agg_kinds = std::make_unique<cuda::device_buffer<aggregation::Kind>>(
    cudf::detail::make_device_buffer_async(_agg_kinds, stream, mr));

  // Map each column in `values_view` back to its index in `data`.
  _value_col_indices.reserve(values_view.num_columns());
  for (size_type i = 0; i < values_view.num_columns(); ++i) {
    auto const& col = values_view.column(i);
    bool found      = false;
    for (size_type c = 0; c < data.num_columns(); ++c) {
      if (cudf::detail::is_shallow_equivalent(data.column(c), col)) {
        _value_col_indices.push_back(c);
        found = true;
        break;
      }
    }
    CUDF_EXPECTS(found, "Internal error: agg column not found in input data.");
  }

  // For each user streaming request, locate the offset in the dedup'd `_agg_kinds`
  // where its first decomposed simple agg lives (matched by kind + column identity).
  _request_first_agg_offset.reserve(_requests_clone.size());
  for (auto const& req : _requests_clone) {
    auto const& target_col = data.column(req.column_index);
    auto const first_kind =
      detail::hash::get_simple_aggregations(*req.aggregation, target_col.type()).front();
    bool found = false;
    for (size_type k = 0; k < static_cast<size_type>(_agg_kinds.size()); ++k) {
      if (_agg_kinds[k] == first_kind &&
          cudf::detail::is_shallow_equivalent(values_view.column(k), target_col)) {
        _request_first_agg_offset.push_back(k);
        found = true;
        break;
      }
    }
    CUDF_EXPECTS(found, "Internal error: request's first simple agg not found.");
  }

  // Companion vector: one compacted batch ID per dense group.
  _key_batch_ids = std::make_unique<cuda::device_buffer<size_type>>(
    stream, mr.get_output_mr(), _max_distinct_keys, cuda::no_init);

  _initialized = true;
}

std::unique_ptr<table> streaming_groupby::impl::make_aggregate_results(
  size_type capacity,
  size_type first_column,
  size_type num_columns,
  cuda::stream_ref stream,
  cudf::memory_resources mr) const
{
  auto const kinds =
    host_span<aggregation::Kind const>{_agg_kinds}.subspan(first_column, num_columns);
  auto const intermediates =
    std::span<int8_t const>{_is_agg_intermediate}.subspan(first_column, num_columns);
  std::vector<std::unique_ptr<column>> schema_columns;
  schema_columns.reserve(num_columns);
  for (auto const type :
       std::span<data_type const>{_aggregate_input_types}.subspan(first_column, num_columns)) {
    schema_columns.push_back(cudf::make_empty_column(type));
  }
  auto const schema = table{std::move(schema_columns)};
  auto results      = detail::hash::create_results_table(
    capacity, schema.view(), kinds, intermediates, stream, mr.get_output_mr());
  // Future batches may introduce all-null groups even when the first values were valid.
  for (size_type i = 0; i < results->num_columns(); ++i) {
    auto& result = results->get_column(i);
    if (!intermediates[i] && kinds[i] != aggregation::COUNT_VALID &&
        kinds[i] != aggregation::COUNT_ALL) {
      result.set_null_mask(
        cudf::create_null_mask(capacity, mask_state::ALL_NULL, stream, mr.get_output_mr()),
        capacity);
    }
  }
  return results;
}

void streaming_groupby::impl::ensure_aggregate_capacity(size_type required,
                                                        cuda::stream_ref stream,
                                                        cudf::memory_resources mr)
{
  // Without aggregations there is no aggregate state to size.
  if (required == 0 || _agg_kinds.empty()) { return; }
  auto const next_capacity =
    required <= _aggregate_capacity
      ? _aggregate_capacity
      : static_cast<size_type>(std::min<int64_t>(
          _max_distinct_keys, std::max<int64_t>(required, 2LL * _aggregate_capacity)));
  // Multiple partials stay within the budget; a single partial uses capacity rows.
  auto const next_partials_per_group = std::max<size_type>(1, _partial_budget / next_capacity);
  if (next_capacity == _aggregate_capacity && next_partials_per_group == _partials_per_group) {
    return;
  }

  _aggregate_done.wait(stream);
  if (_aggregate_capacity == 0) {
    // There is no old device storage to overlap on the first allocation. Initialize all
    // columns together; incremental replacement only saves memory on subsequent growth.
    _agg_results = make_aggregate_results(
      next_capacity * next_partials_per_group, 0, _agg_kinds.size(), stream, mr);
    _aggregate_capacity = next_capacity;
    _partials_per_group = next_partials_per_group;
    return;
  }
  auto const old_groups  = _distinct_keys.load(std::memory_order_relaxed);
  _growing_agg_columns   = _agg_results->release();
  _num_grown_agg_columns = 0;
  for (size_type i = 0; i < static_cast<size_type>(_growing_agg_columns.size()); ++i) {
    auto& previous = _growing_agg_columns[i];
    auto next = make_aggregate_results(next_capacity * next_partials_per_group, i, 1, stream, mr);
    if (old_groups > 0) {
      auto const source =
        table_device_view::create(table_view{{previous->view()}}, stream, mr.get_temporary_mr());
      auto const target =
        mutable_table_device_view::create(next->mutable_view(), stream, mr.get_temporary_mr());
      // The pitch changes with capacity. Collapse live groups into the first new partial;
      // every other partial and padding row retains its identity.
      thrust::for_each_n(rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
                         cuda::counting_iterator<int64_t>{0},
                         static_cast<int64_t>(old_groups) * _partials_per_group,
                         merge_partial_aggs_fn{nullptr,
                                               _aggregate_capacity,
                                               old_groups,
                                               _partials_per_group,
                                               _d_agg_kinds->data() + i,
                                               *source,
                                               *target});
    }
    auto retired = std::move(previous);
    previous     = std::move(next->release().front());
    ++_num_grown_agg_columns;
    // Aggregate states are fixed-width columns. Prior uses precede this stream, so retire
    // their buffers after the transfer here, before allocating the next replacement column.
    auto contents = retired->release();
    contents.data->set_stream(stream);
    contents.null_mask->set_stream(stream);
  }
  _agg_results = std::make_unique<table>(std::move(_growing_agg_columns));
  _growing_agg_columns.clear();
  _aggregate_capacity = next_capacity;
  _partials_per_group = next_partials_per_group;
}

void streaming_groupby::impl::create_key_set(cuda::stream_ref stream)
{
  // Space for up to twice the bound also lets an over-capacity batch finish
  // insertion before the existing distinct-key check invalidates the object.
  auto const capacity = static_cast<std::size_t>(
    std::ceil(_max_distinct_keys / cudf::detail::CUCO_DESIRED_LOAD_FACTOR));
  _key_set = std::make_unique<cuda::device_buffer<size_type>>(stream, _mr, capacity, cuda::no_init);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(_key_set->data(), 0xff, capacity * sizeof(size_type), stream.get()));
}

void streaming_groupby::impl::update_nullable_state(table_view const& batch_keys)
{
  // Nested members can be null even when the top-level key columns are not.
  _has_nullable_keys = _has_nullable_keys || cudf::has_nested_nullable_columns(batch_keys);
}

std::unique_ptr<table> streaming_groupby::impl::gather_agg_results(cuda::stream_ref stream,
                                                                   cudf::memory_resources mr) const
{
  auto const size = _distinct_keys.load(std::memory_order_relaxed);
  if (!_growing_agg_columns.empty()) {
    // A failed grow leaves completed columns with their old groups in the first partial,
    // while the untouched suffix still uses the old pitch. Preserve finalize() recovery.
    auto result = make_aggregate_results(size, 0, _agg_kinds.size(), stream, mr);
    if (size == 0) { return result; }
    auto const kinds = cudf::detail::make_device_buffer(_agg_kinds, stream, mr.as_temporary());
    for (size_type i = 0; i < static_cast<size_type>(_growing_agg_columns.size()); ++i) {
      auto const source = table_device_view::create(
        table_view{{_growing_agg_columns[i]->view()}}, stream, mr.get_temporary_mr());
      auto const target = mutable_table_device_view::create(
        mutable_table_view{{result->get_column(i).mutable_view()}}, stream, mr.get_temporary_mr());
      auto const partials = i < _num_grown_agg_columns ? 1 : _partials_per_group;
      thrust::for_each_n(
        rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
        cuda::counting_iterator<int64_t>{0},
        static_cast<int64_t>(size) * partials,
        merge_partial_aggs_fn{
          nullptr, _aggregate_capacity, size, partials, kinds.data() + i, *source, *target});
    }
    return result;
  }
  if (_agg_results->num_columns() == 0) { return std::make_unique<table>(); }
  if (_partials_per_group == 1 || size == 0) {
    auto const sliced = cudf::detail::slice(_agg_results->view(), {0, size}, stream).front();
    return std::make_unique<table>(sliced, stream, mr.get_output_mr());
  }

  auto result = make_aggregate_results(size, 0, _agg_kinds.size(), stream, mr);
  auto const source =
    table_device_view::create(_agg_results->view(), stream, mr.get_temporary_mr());
  auto const target =
    mutable_table_device_view::create(result->mutable_view(), stream, mr.get_temporary_mr());
  // finalize_and_release may destroy the object's kinds buffer on its original stream.
  // Own this kernel's kinds on the snapshot stream so its lifetime follows the queued read.
  // Finish the host copy as well: destructive finalization also retires the host vector.
  auto const kinds = cudf::detail::make_device_buffer(_agg_kinds, stream, mr.as_temporary());
  thrust::for_each_n(
    rmm::exec_policy_nosync(stream, mr.get_temporary_mr()),
    cuda::counting_iterator<int64_t>{0},
    static_cast<int64_t>(size) * _partials_per_group * _agg_kinds.size(),
    merge_partial_aggs_fn{
      nullptr, _aggregate_capacity, size, _partials_per_group, kinds.data(), *source, *target});
  return result;
}

std::unique_ptr<table> streaming_groupby::impl::gather_distinct_keys(
  cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
{
  if (_compacted_batches.empty()) {
    return std::make_unique<table>(_empty_key_schema->view(), stream, mr);
  }
  if (_compacted_batches.size() == 1) {
    return std::make_unique<table>(_compacted_batches[0]->view(), stream, mr);
  }
  std::vector<table_view> distinct_keys(_compacted_batches.size());
  std::transform(_compacted_batches.begin(),
                 _compacted_batches.end(),
                 distinct_keys.begin(),
                 [](auto const& batch) { return batch->view(); });
  return cudf::concatenate(distinct_keys, stream, mr);
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
streaming_groupby::impl::finalize_gathered(std::unique_ptr<table> keys,
                                           std::unique_ptr<table> agg_gathered,
                                           cuda::stream_ref stream,
                                           cudf::memory_resources mr) const
{
  // Each input column needs one cache identity shared by all of its simple and compound
  // aggregations. Prefer a result retaining the input type (e.g. SUM for decimal/duration
  // MEAN), even when COUNT was requested first. For promoted numeric results, the compound
  // output types are independent of the original integer width.
  auto const agg_gathered_view = agg_gathered->view();
  std::unordered_map<size_type, size_type> input_offsets;
  for (size_type i = 0; i < agg_gathered_view.num_columns(); ++i) {
    auto const [it, inserted] = input_offsets.try_emplace(_value_col_indices[i], i);
    if (!inserted && agg_gathered_view.column(i).type() == _aggregate_input_types[i]) {
      it->second = i;
    }
  }
  std::vector<column_view> cache_keys;
  cache_keys.reserve(_agg_kinds.size());
  for (auto const input_column : _value_col_indices) {
    cache_keys.push_back(agg_gathered_view.column(input_offsets.at(input_column)));
  }

  // The stored aggregation objects already match the physical result order. Re-extracting
  // them here can reorder results or duplicate intermediates shared with explicit requests.
  cudf::detail::result_cache cache(_agg_kinds.size());
  detail::hash::finalize_output(table_view{cache_keys}, _agg_objects, agg_gathered, &cache, stream);

  // Preserve the user's request order while sharing cached simple and compound results.
  std::vector<aggregation_request> user_requests;
  user_requests.reserve(_requests_clone.size());
  for (size_t i = 0; i < _requests_clone.size(); ++i) {
    aggregation_request ar;
    ar.values = cache_keys[_request_first_agg_offset[i]];
    ar.aggregations.push_back(std::unique_ptr<groupby_aggregation>{
      dynamic_cast<groupby_aggregation*>(_requests_clone[i].aggregation->clone().release())});
    auto const& agg = *ar.aggregations.front();
    if (_has_compound_aggs && !cache.has_result(ar.values, agg)) {
      // Gathered intermediates may lack null masks, so MEAN derives nulls from the counts.
      auto const finalizer =
        detail::hash::hash_compound_agg_finalizer(ar.values, &cache, true, stream, mr);
      cudf::detail::aggregation_dispatcher(agg.kind, finalizer, agg);
    }
    user_requests.push_back(std::move(ar));
  }

  return {
    std::move(keys),
    detail::extract_results(
      std::span<aggregation_request const>{user_requests}, cache, stream, mr.get_output_mr())};
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
streaming_groupby::impl::do_finalize(cuda::stream_ref stream, cudf::memory_resources mr) const
{
  std::lock_guard const lock{_insert_mutex};
  CUDF_EXPECTS(_initialized, "Cannot finalize streaming_groupby with no accumulated data.");
  _aggregate_done.wait(stream);
  try {
    auto result = finalize_gathered(
      gather_distinct_keys(stream, mr.get_output_mr()), gather_agg_results(stream, mr), stream, mr);
    record_result_read(stream);
    return result;
  } catch (...) {
    // A failed read must not leave untracked accesses before a subsequent resize.
    synchronize_on_error(stream);
    throw;
  }
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
streaming_groupby::impl::do_finalize_and_release(cuda::stream_ref stream, cudf::memory_resources mr)
{
  auto keys = gather_distinct_keys(stream, mr.get_output_mr());
  _compacted_batches.clear();
  _empty_key_schema.reset();

  auto agg_gathered = gather_agg_results(stream, mr);
  _agg_results.reset();
  _growing_agg_columns.clear();

  return finalize_gathered(std::move(keys), std::move(agg_gathered), stream, mr);
}

cuda::device_buffer<size_type> streaming_groupby::impl::probe_and_insert(
  table_view const& batch_keys, bool use_partials, cuda::stream_ref stream)
{
  if (_has_nested_keys) { return probe_and_insert_impl<true>(batch_keys, use_partials, stream); }
  return probe_and_insert_impl<false>(batch_keys, use_partials, stream);
}

// Constructor, destructor, and move ops require full impl definition.
streaming_groupby::streaming_groupby(host_span<size_type const> key_indices,
                                     host_span<streaming_aggregation_request const> requests,
                                     size_type max_distinct_keys,
                                     null_policy null_handling,
                                     cuda::mr::any_resource<cuda::mr::device_accessible> mr)
  : _impl{std::make_unique<impl>(
      key_indices, requests, max_distinct_keys, null_handling, std::move(mr))}
{
}

streaming_groupby::~streaming_groupby() = default;

streaming_groupby::streaming_groupby(streaming_groupby&&) noexcept = default;

streaming_groupby& streaming_groupby::operator=(streaming_groupby&&) noexcept = default;

// Private member functions defined here (requires full impl definition).
// The public API wrappers in streaming_groupby.cpp call these.
void streaming_groupby::do_aggregate(table_view const& data, cuda::stream_ref stream)
{
  CUDF_EXPECTS(_impl != nullptr, "streaming_groupby has been consumed or moved from.");
  _impl->do_aggregate(data, stream);
}

void streaming_groupby::do_merge(streaming_groupby const& other, cuda::stream_ref stream)
{
  CUDF_EXPECTS(_impl != nullptr && other._impl != nullptr,
               "streaming_groupby has been consumed or moved from.");
  _impl->do_merge(*other._impl, stream);
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> streaming_groupby::do_finalize(
  cuda::stream_ref stream, rmm::device_async_resource_ref mr) const
{
  CUDF_EXPECTS(_impl != nullptr, "streaming_groupby has been consumed or moved from.");
  return _impl->do_finalize(stream, cudf::memory_resources{mr});
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>>
streaming_groupby::do_finalize_and_release(cuda::stream_ref stream, cudf::memory_resources mr)
{
  CUDF_EXPECTS(_impl != nullptr, "streaming_groupby has been consumed or moved from.");
  CUDF_EXPECTS(_impl->_initialized, "Cannot finalize streaming_groupby with no accumulated data.");
  // The caller must order all prior uses before stream. Wait for those uses
  // before destroying lookup structures on their original allocation streams.
  _impl->_aggregate_done.wait(stream);
  stream.sync();
  auto state = std::move(_impl);
  state->_key_set.reset();
  state->_key_batch_ids.reset();
  state->_preprocessed_batches.clear();
  // Finalization reads these columns asynchronously. Rebind before enqueueing any
  // copies so destruction is stream-ordered even if finalization throws.
  auto rebind_table = [stream](std::unique_ptr<table>& source) {
    if (!source) return;
    auto columns = source->release();
    for (auto& col : columns) {
      col = cudf::rebind_stream(std::move(*col), stream);
    }
    source = std::make_unique<table>(std::move(columns));
  };
  for (auto& batch : state->_compacted_batches) {
    rebind_table(batch);
  }
  rebind_table(state->_empty_key_schema);
  rebind_table(state->_agg_results);
  for (auto& col : state->_growing_agg_columns) {
    col = cudf::rebind_stream(std::move(*col), stream);
  }

  return state->do_finalize_and_release(stream, mr);
}

size_type streaming_groupby::distinct_keys() const noexcept
{
  return _impl ? _impl->_distinct_keys.load(std::memory_order_relaxed) : 0;
}

bool is_streaming_groupby_supported(data_type values_type, aggregation::Kind kind)
{
  switch (kind) {
    case aggregation::SUM:
    case aggregation::PRODUCT:
    case aggregation::COUNT_VALID:
    case aggregation::COUNT_ALL:
    case aggregation::MEAN:
    case aggregation::M2:
    case aggregation::VARIANCE:
    case aggregation::STD:
    case aggregation::SUM_OF_SQUARES: break;
    case aggregation::MIN:
    case aggregation::MAX:
      // Variable-width / compound types decompose to ARGMIN/ARGMAX (unsupported).
      if (!cudf::is_fixed_width(values_type)) { return false; }
      break;
    default: return false;
  }
  // DECIMAL128 SUM uses 128-bit atomic addition, but MIN/MAX still lack atomic support.
  if ((kind == aggregation::MIN || kind == aggregation::MAX) &&
      values_type.id() == type_id::DECIMAL128) {
    return false;
  }
  return true;
}

}  // namespace cudf::groupby

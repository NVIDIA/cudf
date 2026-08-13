/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "common.cuh"
#include "groupby/common/utils.hpp"
#include "groupby/hash/extract_single_pass_aggs.hpp"
#include "groupby/hash/hash_compound_agg_finalizer.hpp"
#include "groupby/hash/output_utils.hpp"

#include <cudf/column/column_view.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/aggregation/result_cache.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/groupby.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace cudf::groupby {

namespace {

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
  : _max_distinct_keys{max_distinct_keys},
    _null_handling{null_handling},
    _mr{std::move(mr)},
    _d_agg_kinds{0, rmm::cuda_stream_default, cudf::get_current_device_resource_ref()},
    _d_agg_results{nullptr, +[](mutable_table_device_view*) {}}
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

void streaming_groupby::impl::initialize(table_view const& data, rmm::cuda_stream_view stream)
{
  auto const mr = cudf::get_current_device_resource_ref();

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

  // TODO: streaming aggregation reuses the cudf hash-groupby element_aggregator,
  // so it inherits the same atomic-support requirement.  In particular, decimal128
  // MIN/MAX/SUM falls through to CUDF_UNREACHABLE because __int128 is not
  // lock-free atomic.  Stateless cudf::groupby falls back to sort-based groupby
  // in that case; streaming has no such fallback.  Until streaming has a
  // non-atomic aggregator path (or 128-bit atomics gain hardware support), gate
  // by the same predicate to fail loudly instead of silently producing garbage.
  CUDF_EXPECTS(detail::hash::can_use_hash_groupby(agg_requests),
               "streaming_groupby does not support this combination of value type and "
               "aggregation kind (e.g. decimal128 MIN/MAX/SUM require 128-bit atomics).",
               std::invalid_argument);

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

  _agg_results = detail::hash::create_results_table(
    _max_distinct_keys, values_view, _agg_kinds, _is_agg_intermediate, stream, mr);

  // Cache the mutable_table_device_view once; the underlying table is fixed-size and
  // never reallocated, so the device-side descriptor stays valid for the whole
  // lifetime of this impl.
  {
    auto raii = mutable_table_device_view::create(*_agg_results, stream);
    _d_agg_results =
      decltype(_d_agg_results){raii.release(), +[](mutable_table_device_view* t) { t->destroy(); }};
  }

  _d_agg_kinds = cudf::detail::make_device_uvector_async(_agg_kinds, stream, mr);

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

  // Companion vector: indexed by dense ID, one {batch_id, row} entry per distinct key.
  _key_loc = std::make_unique<rmm::device_uvector<key_location_t>>(_max_distinct_keys, stream, mr);

  _initialized = true;
}

void streaming_groupby::impl::create_key_set(rmm::cuda_stream_view stream)
{
  _key_set = std::make_unique<streaming_set_t>(
    cuco::extent<int64_t>{static_cast<int64_t>(_max_distinct_keys)},
    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
    cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
    cuda::std::equal_to<size_type>{},
    streaming_probing_scheme_t{cudf::hashing::detail::default_hash<size_type>{}},
    cuco::thread_scope_device,
    cuco::storage<detail::hash::GROUPBY_BUCKET_SIZE>{},
    rmm::mr::polymorphic_allocator<char>{_mr},
    stream.value());
}

void streaming_groupby::impl::update_nullable_state(table_view const& batch_keys)
{
  if (_has_nullable_keys) return;
  for (size_type c = 0; c < num_keys(); ++c) {
    if (batch_keys.column(c).nullable()) {
      _has_nullable_keys = true;
      return;
    }
  }
}

std::unique_ptr<table> streaming_groupby::impl::gather_agg_results(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  // The results we care about are dense in `[0, _distinct_keys)` and can be extracted by
  // slice+copy.
  auto const sliced =
    cudf::detail::slice(_agg_results->view(), {0, _distinct_keys}, stream).front();
  return std::make_unique<table>(sliced, stream, mr);
}

std::unique_ptr<table> streaming_groupby::impl::gather_distinct_keys(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
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
streaming_groupby::impl::do_finalize(rmm::cuda_stream_view stream,
                                     rmm::device_async_resource_ref mr) const
{
  CUDF_EXPECTS(_initialized, "Cannot finalize streaming_groupby with no accumulated data.");

  auto keys         = gather_distinct_keys(stream, mr);
  auto agg_gathered = gather_agg_results(stream, mr);

  // Group user requests by their target column in `agg_gathered` so the cache layout
  // produced by `extract_single_pass_aggs` matches the dedup'd `agg_gathered`.  Uses
  // linear search on a small `group_offsets` vector since the number of distinct
  // columns is typically small.
  auto const agg_gathered_view = agg_gathered->view();
  std::vector<aggregation_request> column_grouped;
  std::vector<size_type> group_offsets;
  for (size_t i = 0; i < _requests_clone.size(); ++i) {
    auto const offset = _request_first_agg_offset[i];
    auto cloned       = std::unique_ptr<groupby_aggregation>{
      dynamic_cast<groupby_aggregation*>(_requests_clone[i].aggregation->clone().release())};
    auto const it = std::find(group_offsets.begin(), group_offsets.end(), offset);
    if (it == group_offsets.end()) {
      aggregation_request ar;
      ar.values = agg_gathered_view.column(offset);
      ar.aggregations.push_back(std::move(cloned));
      column_grouped.push_back(std::move(ar));
      group_offsets.push_back(offset);
    } else {
      column_grouped[std::distance(group_offsets.begin(), it)].aggregations.push_back(
        std::move(cloned));
    }
  }

  auto [values_view_fin, agg_kinds_fin, agg_objects_fin, is_intermediate_fin, has_compound_fin] =
    detail::hash::extract_single_pass_aggs(column_grouped, stream);

  cudf::detail::result_cache cache(_agg_kinds.size());
  detail::hash::finalize_output(values_view_fin, agg_objects_fin, agg_gathered, &cache, stream);

  if (_has_compound_aggs) {
    // Compute compound aggs (MEAN/STD/VARIANCE/M2/...) into the cache.  The cache itself
    // dedupes: skip if (column, kind) is already there from a prior agg in the group.
    for (auto const& req : column_grouped) {
      auto const finalizer =
        detail::hash::hash_compound_agg_finalizer(req.values, &cache, nullptr, stream, mr);
      for (auto const& agg : req.aggregations) {
        if (cache.has_result(req.values, *agg)) continue;
        cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
      }
    }
  }

  // User-1:1 lookup keys so `extract_results` returns results in the order of the
  // original user requests, sharing cache entries across requests on the same column.
  std::vector<aggregation_request> user_requests;
  user_requests.reserve(_requests_clone.size());
  for (size_t i = 0; i < _requests_clone.size(); ++i) {
    aggregation_request ar;
    ar.values = agg_gathered_view.column(_request_first_agg_offset[i]);
    ar.aggregations.push_back(std::unique_ptr<groupby_aggregation>{
      dynamic_cast<groupby_aggregation*>(_requests_clone[i].aggregation->clone().release())});
    user_requests.push_back(std::move(ar));
  }

  return {std::move(keys),
          detail::extract_results(
            std::span<aggregation_request const>{user_requests}, cache, stream, mr)};
}

streaming_groupby::impl::batch_insert_result streaming_groupby::impl::probe_and_insert(
  table_view const& batch_keys, rmm::cuda_stream_view stream)
{
  if (_has_nested_keys) {
    return probe_and_insert_impl<true>(batch_keys, stream);
  } else {
    return probe_and_insert_impl<false>(batch_keys, stream);
  }
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
void streaming_groupby::do_aggregate(table_view const& data, rmm::cuda_stream_view stream)
{
  _impl->do_aggregate(data, stream);
}

void streaming_groupby::do_merge(streaming_groupby const& other, rmm::cuda_stream_view stream)
{
  _impl->do_merge(*other._impl, stream);
}

std::pair<std::unique_ptr<table>, std::vector<aggregation_result>> streaming_groupby::do_finalize(
  rmm::cuda_stream_view stream, rmm::device_async_resource_ref mr) const
{
  return _impl->do_finalize(stream, mr);
}

size_type streaming_groupby::distinct_keys() const noexcept { return _impl->_distinct_keys; }

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
  // decimal128 SUM/MIN/MAX needs 128-bit atomics, which aren't supported.
  if ((kind == aggregation::SUM || kind == aggregation::MIN || kind == aggregation::MAX) &&
      values_type.id() == type_id::DECIMAL128) {
    return false;
  }
  return true;
}

}  // namespace cudf::groupby

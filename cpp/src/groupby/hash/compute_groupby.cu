/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_groupby.hpp"
#include "compute_single_pass_aggs.hpp"
#include "groupby/common/utils.hpp"
#include "hash_compound_agg_finalizer.hpp"
#include "helpers.cuh"
#include "output_utils.hpp"

#include <cudf/detail/aggregation/aggregation.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_set.cuh>
#include <cuda/iterator>
#include <cuda/std/iterator>
#include <thrust/tabulate.h>

namespace cudf::groupby::detail::hash {

namespace {

// The number of columns in the keys table that will trigger caching of row hashes.
// This is a heuristic to reduce memory read when the keys table is hashed twice.
constexpr int HASH_CACHING_THRESHOLD = 4;

int count_nested_columns(column_view const& input)
{
  if (!is_nested(input.type())) { return 1; }

  // Count the current column too.
  return 1 + std::accumulate(
               input.child_begin(), input.child_end(), 0, [](int count, column_view const& child) {
                 return count + count_nested_columns(child);
               });
}

}  // namespace

template <typename Equal, typename Hash>
std::unique_ptr<table> compute_groupby(table_view const& keys,
                                       std::span<aggregation_request const> requests,
                                       bool skip_rows_with_nulls,
                                       Equal const& d_row_equal,
                                       Hash const& d_row_hash,
                                       cudf::detail::result_cache* cache,
                                       rmm::cuda_stream_view stream,
                                       rmm::device_async_resource_ref mr)
{
  auto const num_keys = keys.num_rows();

  [[maybe_unused]] auto [row_bitmask_data, row_bitmask] =
    skip_rows_with_nulls
      ? cudf::groupby::detail::compute_row_bitmask(keys, stream)
      : std::pair<rmm::device_buffer, bitmask_type const*>{
          rmm::device_buffer{0, stream, cudf::get_current_device_resource_ref()}, nullptr};

  auto const cached_hashes = [&]() -> rmm::device_uvector<hash_value_type> {
    auto const num_columns =
      std::accumulate(keys.begin(), keys.end(), 0, [](int count, column_view const& col) {
        return count + count_nested_columns(col);
      });

    if (num_columns <= HASH_CACHING_THRESHOLD) {
      return rmm::device_uvector<hash_value_type>{
        0, stream, cudf::get_current_device_resource_ref()};
    }

    rmm::device_uvector<hash_value_type> hashes(
      num_keys, stream, cudf::get_current_device_resource_ref());
    thrust::tabulate(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     hashes.begin(),
                     hashes.end(),
                     [d_row_hash, row_bitmask] __device__(size_type const idx) {
                       if (!row_bitmask || cudf::bit_is_set(row_bitmask, idx)) {
                         return d_row_hash(idx);
                       }
                       return hash_value_type{0};  // dummy value, as it will be unused
                     });
    return hashes;
  }();

  auto set =
    cuco::static_set{cuco::extent<int64_t>{static_cast<int64_t>(num_keys)},
                     cudf::detail::CUCO_DESIRED_LOAD_FACTOR,  // 50% load factor
                     cuco::empty_key{cudf::detail::CUDF_SIZE_TYPE_SENTINEL},
                     d_row_equal,
                     probing_scheme_t{row_hasher_with_cache_t{d_row_hash, cached_hashes.data()}},
                     cuco::thread_scope_device,
                     cuco::storage<GROUPBY_BUCKET_SIZE>{},
                     rmm::mr::polymorphic_allocator<char>{},
                     stream.value()};

  auto const gather_keys = [&](auto const& gather_map) {
    return cudf::detail::gather(keys,
                                gather_map,
                                out_of_bounds_policy::DONT_CHECK,
                                cudf::negative_index_policy::NOT_ALLOWED,
                                stream,
                                mr);
  };

  // In case of no requests, we still need to generate a set of unique keys.
  if (requests.empty()) {
    thrust::for_each_n(
      rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
      cuda::counting_iterator<cudf::size_type>{0},
      num_keys,
      [set_ref = set.ref(cuco::op::insert), row_bitmask] __device__(size_type const idx) mutable {
        if (!row_bitmask || cudf::bit_is_set(row_bitmask, idx)) { set_ref.insert(idx); }
      });

    rmm::device_uvector<size_type> unique_key_indices(
      num_keys, stream, cudf::get_current_device_resource_ref());
    auto const keys_end       = set.retrieve_all(unique_key_indices.begin(), stream.value());
    auto const key_gather_map = device_span<size_type const>{
      unique_key_indices.data(),
      static_cast<std::size_t>(cuda::std::distance(unique_key_indices.begin(), keys_end))};
    return gather_keys(key_gather_map);
  }

  // Compute all single pass aggs first.
  auto const [key_gather_map, has_compound_aggs] =
    compute_single_pass_aggs(set, row_bitmask, requests, cache, stream, mr);

  if (has_compound_aggs) {
    for (auto const& request : requests) {
      auto const& agg_v = request.aggregations;
      auto const& col   = request.values;

      // The map to find the target output index for each input row is not always available due to
      // minimizing overhead. As such, there is no way for the finalizers to perform additional
      // aggregation operations. They can only compute their output using the previously computed
      // single-pass aggregations with linear transformations such as addition/multiplication (e.g.
      // for variance/stddev). In the future, if there are more compound aggregations that require
      // additional aggregation steps, we can revisit this design.
      auto const finalizer = hash_compound_agg_finalizer(col, cache, row_bitmask, stream, mr);
      for (auto&& agg : agg_v) {
        cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
      }
    }
  }

  return gather_keys(key_gather_map);
}

template std::unique_ptr<table> compute_groupby<row_comparator_t, row_hash_t>(
  table_view const& keys,
  std::span<aggregation_request const> requests,
  bool skip_rows_with_nulls,
  row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);

template std::unique_ptr<table> compute_groupby<nullable_row_comparator_t, row_hash_t>(
  table_view const& keys,
  std::span<aggregation_request const> requests,
  bool skip_rows_with_nulls,
  nullable_row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  cudf::detail::result_cache* cache,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr);
}  // namespace cudf::groupby::detail::hash

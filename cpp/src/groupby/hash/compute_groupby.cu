/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "compute_groupby.hpp"
#include "compute_single_pass_aggs.hpp"
#include "extract_single_pass_aggs.hpp"
#include "groupby/common/utils.hpp"
#include "hash_compound_agg_finalizer.hpp"
#include "hash_csr_kernels.cuh"
#include "helpers.cuh"

#include <cudf/detail/aggregation/aggregation.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/gather.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_buffer.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cuda/iterator>
#include <cuda/std/cstdint>
#include <cuda/std/iterator>
#include <cuda/stream>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cudf::groupby::detail::hash {

namespace {

// The result cache shares results across requests on the same values column. Normalize the
// extracted reductions with the same equality so an earlier compound request cannot choose the
// resource or nullability of a later requested result, or allocate a result that the cache drops.
auto extract_hash_groupby_aggs(std::span<aggregation_request const> requests,
                               cuda::stream_ref stream)
{
  if (requests.size() <= 1) { return extract_single_pass_aggs(requests, stream); }

  using aggregation_set =
    std::unordered_set<std::pair<column_view, std::reference_wrapper<aggregation const>>,
                       cudf::detail::pair_column_aggregation_hash,
                       cudf::detail::pair_column_aggregation_equal_to>;
  aggregation_set requested;
  for (auto const& request : requests) {
    for (auto const& agg : request.aggregations) {
      requested.emplace(request.values, *agg);
    }
  }

  auto [values, kinds, aggs, is_intermediate, has_compound] =
    extract_single_pass_aggs(requests, stream);
  aggregation_set extracted;
  std::vector<column_view> unique_values;
  unique_values.reserve(aggs.size());
  for (std::size_t i = 0; i < aggs.size(); ++i) {
    auto const key =
      std::pair<column_view, std::reference_wrapper<aggregation const>>{values.column(i), *aggs[i]};
    if (!extracted.insert(key).second) { continue; }
    auto const output       = unique_values.size();
    kinds[output]           = kinds[i];
    is_intermediate[output] = !requested.contains(key);
    if (output != i) { aggs[output] = std::move(aggs[i]); }
    unique_values.push_back(values.column(i));
  }
  kinds.resize(unique_values.size());
  aggs.resize(unique_values.size());
  is_intermediate.resize(unique_values.size());
  return std::tuple{table_view{unique_values},
                    std::move(kinds),
                    std::move(aggs),
                    std::move(is_intermediate),
                    has_compound};
}

/// The keys grouped by the HashCSR build.
struct grouped_keys {
  size_type num_groups;
  size_type num_grouped_rows;
  rmm::device_uvector<size_type> key_rows;       ///< One representative input row per group
  rmm::device_uvector<size_type> group_offsets;  ///< `num_groups + 1` offsets into `grouped_rows`
  rmm::device_uvector<size_type> grouped_rows;   ///< Input rows reordered so groups are contiguous
};

cuda::std::uint32_t hash_csr_capacity(size_type num_rows)
{
  auto const requested =
    std::max(static_cast<double>(num_rows) + 1,
             std::ceil(static_cast<double>(num_rows) / cudf::detail::CUCO_DESIRED_LOAD_FACTOR));
  CUDF_EXPECTS(requested <= std::numeric_limits<cuda::std::uint32_t>::max(),
               "HashCSR table capacity is not representable",
               std::overflow_error);
  return static_cast<cuda::std::uint32_t>(requested);
}

struct is_occupied_fn {
  __device__ bool operator()(hash_table_entry_type entry) const
  {
    return entry != cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
  }
};

/**
 * @brief Estimates the table capacity that fits the distinct keys of a large input.
 *
 * Every `stride`-th row is inserted into a small table, and the number of distinct keys among
 * those rows is corrected for the keys the sample missed: `D` distinct keys show
 * `D * (1 - exp(-s / D))` of themselves in a sample of `s` rows, which is solved for `D`.
 *
 * @return Four slots per estimated distinct key, or the maximum capacity when the estimate
 * exceeds the representable capacity
 */
template <typename Equal, typename Hash>
cuda::std::uint32_t estimate_capacity(size_type num_rows,
                                      bitmask_type const* row_bitmask,
                                      Equal const& d_row_equal,
                                      Hash const& d_row_hash,
                                      cuda::stream_ref stream,
                                      rmm::device_async_resource_ref mr)
{
  rmm::device_uvector<hash_table_entry_type> entries(hash_csr_sample_capacity, stream, mr);
  rmm::device_uvector<size_type> counts(2, stream, mr);
  // Counts the valid rows among every `stride`-th row and the distinct keys among them.
  auto const sample = [&](size_type stride) {
    CUDF_CUDA_TRY(cudaMemsetAsync(
      entries.data(), 0xff, entries.size() * sizeof(hash_table_entry_type), stream.get()));
    CUDF_CUDA_TRY(
      cudaMemsetAsync(counts.data(), 0, counts.size() * sizeof(size_type), stream.get()));
    launch_hash_csr_sample_kernel(
      num_rows,
      stride,
      row_bitmask,
      hash_csr_table_ref{entries.data(), hash_csr_sample_capacity, hash_csr_sample_capacity},
      d_row_equal,
      d_row_hash,
      counts.data(),
      stream);
    auto const h_counts =
      cudf::detail::make_pinned_vector(device_span<size_type const>{counts}, stream);
    return std::pair{static_cast<double>(h_counts[0]), static_cast<double>(h_counts[1])};
  };
  // One row in 64 is sampled, fewer when that would fill more than half of the sample table.
  auto const stride = std::max<size_type>(
    64, cudf::util::div_rounding_up_safe<size_type>(num_rows, hash_csr_sample_capacity / 2));
  auto const max_capacity  = std::numeric_limits<cuda::std::uint32_t>::max();
  auto [sampled, distinct] = sample(stride);
  if (sampled == 0) { return hash_csr_min_estimated_capacity; }

  // The expected number of distinct keys seen grows with the population, so bisect on it.
  auto const seen = [sampled](double population) {
    return population * (1.0 - std::exp(-sampled / population));
  };
  auto low  = distinct;
  auto high = static_cast<double>(num_rows);
  for (int i = 0; i < 64; ++i) {
    auto const mid                      = 0.5 * (low + high);
    (seen(mid) < distinct ? low : high) = mid;
  }
  // Four slots per distinct key keep the probes short while the table stays small.
  auto const estimate = 4.0 * high;
  if (estimate >= static_cast<double>(max_capacity)) { return max_capacity; }
  return std::max(hash_csr_min_estimated_capacity, static_cast<cuda::std::uint32_t>(estimate));
}

/**
 * @brief Groups the input rows by key with a HashCSR build.
 *
 * Every valid row inserts its key into an open-addressed table and takes a rank within the slot
 * it lands in. The occupied slots become the groups, a scan of their row counts gives the group
 * offsets, and a scatter of the rows by slot offset plus rank yields the grouped row order.
 */
template <typename Equal, typename Hash>
grouped_keys group_keys(size_type num_rows,
                        size_type key_bytes,
                        bitmask_type const* row_bitmask,
                        Equal const& d_row_equal,
                        Hash const& d_row_hash,
                        bool need_grouped_rows,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  auto const temp_mr = mr.get_temporary_mr();
  auto const policy  = rmm::exec_policy_nosync(stream, temp_mr);

  // A table with a slot for every row would spread a few distinct keys over a table too large for
  // the cache and make clearing and compacting its slots the dominant cost of low-cardinality
  // inputs, so large inputs get a table sized from an estimate of their number of distinct keys.
  // Should the estimate fall short, the build restarts with the table sized for every row.
  auto const full_capacity = hash_csr_capacity(num_rows);
  auto capacity =
    num_rows < hash_csr_min_rows_to_estimate || key_bytes > hash_csr_max_estimated_key_bytes
      ? full_capacity
      : std::min(
          full_capacity,
          estimate_capacity(num_rows, row_bitmask, d_row_equal, d_row_hash, stream, temp_mr));
  rmm::device_uvector<hash_table_entry_type> entries(0, stream, temp_mr);
  rmm::device_uvector<size_type> slot_counts(0, stream, temp_mr);
  rmm::device_uvector<build_position_type> positions(
    need_grouped_rows ? num_rows : 0, stream, temp_mr);
  // Set by the build when the estimated table turns out to be too small.
  std::optional<cudf::detail::device_scalar<int>> overflow;
  if (capacity < full_capacity) { overflow.emplace(0, stream, temp_mr); }
  // The occupied slots, in slot order, are the groups: without aggregations the slots hold the
  // one row wanted for each group, otherwise the slot indices lead to the counts and rows.
  rmm::device_uvector<size_type> key_rows(0, stream, mr.get_output_mr());
  rmm::device_uvector<cuda::std::uint32_t> group_slots(0, stream, temp_mr);
  bool count_by_representative{};
  while (true) {
    auto const is_full_size = capacity == full_capacity;
    count_by_representative =
      need_grouped_rows && static_cast<cuda::std::uint32_t>(num_rows) < capacity;
    auto const count_capacity =
      count_by_representative ? static_cast<cuda::std::uint32_t>(num_rows) : capacity;
    entries.resize(capacity, stream);
    CUDF_CUDA_TRY(cudaMemsetAsync(
      entries.data(), 0xff, entries.size() * sizeof(hash_table_entry_type), stream.get()));
    if (need_grouped_rows) {
      slot_counts.resize(count_capacity, stream);
      if (count_capacity != 0) {
        CUDF_CUDA_TRY(cudaMemsetAsync(
          slot_counts.data(), 0, slot_counts.size() * sizeof(size_type), stream.get()));
      }
    }
    auto const table =
      hash_csr_table_ref{entries.data(), capacity, is_full_size ? capacity : hash_csr_max_probes};
    launch_hash_csr_build_kernel(num_rows,
                                 row_bitmask,
                                 need_grouped_rows ? positions.data() : nullptr,
                                 need_grouped_rows ? slot_counts.data() : nullptr,
                                 count_by_representative,
                                 table,
                                 d_row_equal,
                                 d_row_hash,
                                 is_full_size ? nullptr : overflow->data(),
                                 stream);
    if (count_by_representative) {
      // Count indices identify representative rows, so selection no longer needs the table.
      entries.resize(0, stream);
      entries.shrink_to_fit(stream);
    }
    if (!need_grouped_rows) {
      key_rows.resize(std::min<std::size_t>(num_rows, capacity), stream);
      auto const key_rows_end =
        thrust::copy_if(policy, entries.begin(), entries.end(), key_rows.begin(), is_occupied_fn{});
      key_rows.resize(cuda::std::distance(key_rows.begin(), key_rows_end), stream);
    } else {
      group_slots.resize(std::min<std::size_t>(num_rows, capacity), stream);
      auto const group_slots_end =
        thrust::copy_if(policy,
                        cuda::counting_iterator<cuda::std::uint32_t>{0},
                        cuda::counting_iterator<cuda::std::uint32_t>{count_capacity},
                        slot_counts.begin(),
                        group_slots.begin(),
                        [] __device__(size_type count) -> bool { return count > 0; });
      group_slots.resize(cuda::std::distance(group_slots.begin(), group_slots_end), stream);
    }
    // The compaction has just synchronized the stream, so reading the flag is cheap here.
    if (is_full_size || overflow->value(stream) == 0) { break; }
    // The retry overwrites the table, so release it instead of copying it while growing.
    entries.resize(0, stream);
    entries.shrink_to_fit(stream);
    slot_counts.resize(0, stream);
    slot_counts.shrink_to_fit(stream);
    key_rows.resize(0, stream);
    key_rows.shrink_to_fit(stream);
    group_slots.resize(0, stream);
    group_slots.shrink_to_fit(stream);
    overflow.reset();
    capacity = full_capacity;
  }

  if (!need_grouped_rows) {
    auto const num_groups = static_cast<size_type>(key_rows.size());
    return {num_groups,
            0,
            std::move(key_rows),
            rmm::device_uvector<size_type>{0, stream, mr.get_output_mr()},
            rmm::device_uvector<size_type>{0, stream, mr.get_output_mr()}};
  }

  auto const num_groups = static_cast<size_type>(group_slots.size());
  // Every row is an included singleton group, so input order already forms a valid grouping.
  if (num_groups == num_rows) {
    entries.resize(0, stream);
    entries.shrink_to_fit(stream);
    slot_counts.resize(0, stream);
    slot_counts.shrink_to_fit(stream);
    positions.resize(0, stream);
    positions.shrink_to_fit(stream);
    group_slots.resize(0, stream);
    group_slots.shrink_to_fit(stream);
    overflow.reset();

    key_rows.resize(num_rows, stream);
    rmm::device_uvector<size_type> group_offsets(
      static_cast<std::size_t>(num_rows) + 1, stream, mr.get_output_mr());
    rmm::device_uvector<size_type> grouped_rows(num_rows, stream, mr.get_output_mr());
    thrust::sequence(policy, key_rows.begin(), key_rows.end(), size_type{0});
    thrust::sequence(policy, group_offsets.begin(), group_offsets.end(), size_type{0});
    thrust::sequence(policy, grouped_rows.begin(), grouped_rows.end(), size_type{0});
    return {
      num_groups, num_rows, std::move(key_rows), std::move(group_offsets), std::move(grouped_rows)};
  }

  auto const entry_rows = entries.begin();
  key_rows.resize(num_groups, stream);
  if (count_by_representative) {
    thrust::copy(policy, group_slots.begin(), group_slots.end(), key_rows.begin());
  } else {
    thrust::gather(policy, group_slots.begin(), group_slots.end(), entry_rows, key_rows.begin());
  }
  entries.resize(0, stream);
  entries.shrink_to_fit(stream);

  rmm::device_uvector<size_type> group_offsets(group_slots.size() + 1, stream, mr.get_output_mr());
  group_offsets.set_element_to_zero_async(0, stream);
  auto const group_counts =
    cuda::make_permutation_iterator(slot_counts.begin(), group_slots.begin());
  thrust::inclusive_scan(
    policy, group_counts, group_counts + num_groups, group_offsets.begin() + 1);
  auto const num_grouped_rows =
    row_bitmask == nullptr ? num_rows : group_offsets.back_element(stream);

  // Reuse the slot counts to hold the start offset of the group of each occupied slot, then
  // scatter every row to its group.
  thrust::scatter(policy,
                  group_offsets.begin(),
                  group_offsets.begin() + num_groups,
                  group_slots.begin(),
                  slot_counts.begin());
  group_slots.resize(0, stream);
  group_slots.shrink_to_fit(stream);
  rmm::device_uvector<size_type> grouped_rows(num_grouped_rows, stream, mr.get_output_mr());
  launch_hash_csr_fill_kernel(
    num_rows, positions.data(), slot_counts.data(), grouped_rows.data(), stream);

  return {num_groups,
          num_grouped_rows,
          std::move(key_rows),
          std::move(group_offsets),
          std::move(grouped_rows)};
}

}  // namespace

template <typename Equal, typename Hash>
std::unique_ptr<table> compute_groupby(table_view const& keys,
                                       std::span<aggregation_request const> requests,
                                       bool skip_rows_with_nulls,
                                       Equal const& d_row_equal,
                                       Hash const& d_row_hash,
                                       cudf::detail::result_cache* cache,
                                       cuda::stream_ref stream,
                                       cudf::memory_resources mr)
{
  auto const num_rows            = keys.num_rows();
  auto const temp_mr             = mr.get_temporary_mr();
  auto const temporary_resources = cudf::memory_resources{temp_mr, temp_mr};

  [[maybe_unused]] auto [row_bitmask_data, row_bitmask] =
    skip_rows_with_nulls ? cudf::groupby::detail::compute_row_bitmask(keys, stream, temp_mr)
                         : std::pair<rmm::device_buffer, bitmask_type const*>{
                             rmm::device_buffer{0, stream, temp_mr}, nullptr};

  // Bytes of one key row, with variable-width and nested columns counted as wide.
  auto const key_bytes = std::accumulate(
    keys.begin(), keys.end(), size_type{0}, [](size_type bytes, column_view const& col) {
      return bytes + (cudf::is_fixed_width(col.type()) ? cudf::size_of(col.type()) : 64);
    });
  auto const groups = group_keys(num_rows,
                                 key_bytes,
                                 row_bitmask,
                                 d_row_equal,
                                 d_row_hash,
                                 !requests.empty(),
                                 stream,
                                 temporary_resources);

  auto const gather_keys = [&] {
    return cudf::detail::gather(keys,
                                groups.key_rows,
                                out_of_bounds_policy::DONT_CHECK,
                                cudf::negative_index_policy::NOT_ALLOWED,
                                stream,
                                mr);
  };

  // In case of no requests, we still need to generate a set of unique keys.
  if (requests.empty()) { return gather_keys(); }

  // Compute all single pass aggs first.
  auto const [values, agg_kinds, aggs, is_agg_intermediate, has_compound_aggs] =
    extract_hash_groupby_aggs(requests, stream);

  auto const grouped =
    make_grouped_rows(groups.grouped_rows, groups.group_offsets, stream, temporary_resources);
  auto results =
    compute_single_pass_aggs(values, agg_kinds, is_agg_intermediate, grouped, stream, mr);
  for (std::size_t i = 0; i < results.size(); ++i) {
    cache->add_result(values.column(i), *aggs[i], std::move(results[i]));
  }

  if (has_compound_aggs) {
    // Requested M2 results must be cached on the output resource before VARIANCE or STD asks
    // for an intermediate M2, regardless of the order of requests on a shared values column.
    for (auto const& request : requests) {
      auto const finalizer =
        hash_compound_agg_finalizer(request.values, cache, row_bitmask, stream, mr);
      for (auto const& agg : request.aggregations) {
        if (agg->kind == aggregation::M2) {
          cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
        }
      }
    }
    for (auto const& request : requests) {
      auto const& agg_v = request.aggregations;
      auto const& col   = request.values;

      // The finalizers only combine the single-pass results with linear transformations such as
      // addition/multiplication (e.g. for variance/stddev); they do not aggregate further.
      auto const finalizer = hash_compound_agg_finalizer(col, cache, row_bitmask, stream, mr);
      for (auto&& agg : agg_v) {
        if (agg->kind == aggregation::VARIANCE || agg->kind == aggregation::STD) {
          // Explicit M2 outputs were finalized above. Any missing M2 is only an intermediate
          // for this ordinary groupby; the shared finalizer also serves streaming groupby.
          auto const m2_agg = make_m2_aggregation();
          auto const m2_finalizer =
            hash_compound_agg_finalizer(col, cache, row_bitmask, stream, temporary_resources);
          cudf::detail::aggregation_dispatcher(m2_agg->kind, m2_finalizer, *m2_agg);
        }
        cudf::detail::aggregation_dispatcher(agg->kind, finalizer, *agg);
      }
    }
  }

  return gather_keys();
}

template std::unique_ptr<table> compute_groupby<row_comparator_t, row_hash_t>(
  table_view const& keys,
  std::span<aggregation_request const> requests,
  bool skip_rows_with_nulls,
  row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  cudf::detail::result_cache* cache,
  cuda::stream_ref stream,
  cudf::memory_resources mr);

template std::unique_ptr<table> compute_groupby<nullable_row_comparator_t, row_hash_t>(
  table_view const& keys,
  std::span<aggregation_request const> requests,
  bool skip_rows_with_nulls,
  nullable_row_comparator_t const& d_row_equal,
  row_hash_t const& d_row_hash,
  cudf::detail::result_cache* cache,
  cuda::stream_ref stream,
  cudf::memory_resources mr);

}  // namespace cudf::groupby::detail::hash

/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_keys.hpp"
#include "groupby/common/utils.hpp"
#include "hash_csr_kernels.cuh"
#include "helpers.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/iterator>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/uninitialized_fill.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

namespace cudf::groupby::detail::hash {
namespace {

std::size_t hash_csr_capacity(size_type num_rows)
{
  auto const requested =
    std::max(static_cast<std::size_t>(num_rows) + 1,
             static_cast<std::size_t>(
               std::ceil(static_cast<double>(num_rows) / cudf::detail::CUCO_DESIRED_LOAD_FACTOR)));
  CUDF_EXPECTS(requested <= std::numeric_limits<cuda::std::uint32_t>::max(),
               "HashCSR table capacity is not representable",
               std::overflow_error);
  return requested;
}

// Bound only small scalar domains whose non-null combinations fit the existing capacity floor.
// Null states are independent per column when null keys are included. Stop before either product
// can overflow or make a sparse observed domain allocate a much larger table than the sampler.
std::optional<std::size_t> small_key_domain_capacity(table_view const& keys,
                                                     bool skip_rows_with_nulls)
{
  constexpr std::size_t max_value_domain    = hash_csr_min_estimated_capacity / 4;
  constexpr std::size_t max_nullable_domain = 2 * max_value_domain;
  std::size_t value_domain                  = 1;
  std::size_t domain                        = 1;
  for (auto const& key : keys) {
    std::size_t column_domain;
    switch (key.type().id()) {
      case type_id::BOOL8: column_domain = 2; break;
      case type_id::INT8:
      case type_id::UINT8: column_domain = 1u << 8; break;
      case type_id::INT16:
      case type_id::UINT16: column_domain = 1u << 16; break;
      default: return std::nullopt;
    }
    if (value_domain > max_value_domain / column_domain) { return std::nullopt; }
    value_domain *= column_domain;
    if (!skip_rows_with_nulls && key.nullable()) { ++column_domain; }
    if (domain > max_nullable_domain / column_domain) { return std::nullopt; }
    domain *= column_domain;
  }
  return std::max<std::size_t>(hash_csr_min_estimated_capacity, 4 * domain);
}

struct is_occupied_fn {
  __device__ bool operator()(slot_type slot) const
  {
    return slot != cudf::detail::CUDF_SIZE_TYPE_SENTINEL;
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
std::size_t estimate_capacity(size_type num_rows,
                              bitmask_type const* row_bitmask,
                              Equal const& d_row_equal,
                              Hash const& d_row_hash,
                              cuda::stream_ref stream,
                              cudf::memory_resources mr)
{
  auto const temp_mr = mr.get_temporary_mr();
  auto const policy  = rmm::exec_policy_nosync(stream, temp_mr);
  cuda::device_buffer<slot_type> slots(stream, temp_mr, hash_csr_sample_capacity, cuda::no_init);
  cuda::device_buffer<size_type> counts(stream, temp_mr, 2, cuda::no_init);
  // Counts the valid rows among every `stride`-th row and the distinct keys among them.
  auto const sample = [&](size_type stride) {
    thrust::uninitialized_fill(
      policy, slots.data(), slots.data() + slots.size(), cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
    CUDF_CUDA_TRY(
      cudaMemsetAsync(counts.data(), 0, counts.size() * sizeof(size_type), stream.get()));
    launch_hash_csr_sample_kernel(
      num_rows,
      stride,
      row_bitmask,
      hash_set_ref{slots.data(), hash_csr_sample_capacity, hash_csr_sample_capacity},
      d_row_equal,
      d_row_hash,
      counts.data(),
      stream);
    auto const h_counts = cudf::detail::make_pinned_vector(
      device_span<size_type const>{counts.data(), counts.size()}, stream);
    return std::pair{static_cast<double>(h_counts[0]), static_cast<double>(h_counts[1])};
  };
  // Sample with a prime stride of 67, or a larger odd stride to keep the sample table at most
  // half full. Odd strides avoid repeatedly sampling the same phase of power-of-two patterns.
  auto const stride =
    std::max<size_type>(
      67, cudf::util::div_rounding_up_safe<size_type>(num_rows, hash_csr_sample_capacity / 2)) |
    1;
  auto const max_capacity =
    static_cast<std::size_t>(std::numeric_limits<cuda::std::uint32_t>::max());
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
  return std::max<std::size_t>(hash_csr_min_estimated_capacity, static_cast<std::size_t>(estimate));
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
                        bitmask_type const* row_bitmask,
                        Equal const& d_row_equal,
                        Hash const& d_row_hash,
                        bool need_group_offsets,
                        bool need_grouped_rows,
                        std::optional<std::size_t> domain_capacity,
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
  auto capacity            = full_capacity;
  if (num_rows >= hash_csr_min_rows_to_estimate) {
    // A small finite key domain gives an upper bound without allocating or sampling on device.
    // Keep the same capacity floor and four slots per possible key as the sampled estimate.
    auto const estimated_capacity =
      domain_capacity
        ? *domain_capacity
        : estimate_capacity(num_rows, row_bitmask, d_row_equal, d_row_hash, stream, mr);
    capacity = std::min(full_capacity, estimated_capacity);
  }

  cuda::device_buffer<slot_type> slots(stream, temp_mr);
  cuda::device_buffer<size_type> slot_counts(stream, temp_mr);
  cuda::device_buffer<build_position_type> positions(
    stream, temp_mr, need_grouped_rows ? num_rows : 0, cuda::no_init);

  // Set by the build when the estimated table turns out to be too small.
  std::optional<cudf::detail::device_scalar<cuda::std::int32_t>> overflow;
  if (capacity < full_capacity) { overflow.emplace(0, stream, temp_mr); }

  // The occupied slots, in slot order, are the groups: without aggregations the slots hold the
  // one row wanted for each group, otherwise the slot indices lead to the counts and rows.
  cuda::device_buffer<size_type> key_indices(stream, mr.get_output_mr());
  cuda::device_buffer<cuda::std::uint32_t> group_slots(stream, temp_mr);
  size_type num_groups{};
  bool count_by_representative{};

  while (true) {
    auto const is_full_size = capacity == full_capacity;
    count_by_representative = need_group_offsets && static_cast<std::size_t>(num_rows) < capacity;
    auto const count_capacity =
      count_by_representative ? static_cast<std::size_t>(num_rows) : capacity;

    slots = cuda::device_buffer<slot_type>{stream, temp_mr, capacity, cuda::no_init};
    thrust::uninitialized_fill(
      policy, slots.data(), slots.data() + slots.size(), cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
    if (need_group_offsets) {
      slot_counts = cuda::device_buffer<size_type>{stream, temp_mr, count_capacity, cuda::no_init};
      if (count_capacity != 0) {
        CUDF_CUDA_TRY(cudaMemsetAsync(
          slot_counts.data(), 0, slot_counts.size() * sizeof(size_type), stream.get()));
      }
    }

    // Both capacities are bounded by the checked full capacity.
    auto const device_capacity = static_cast<cuda::std::uint32_t>(capacity);
    auto const set             = hash_set_ref{
      slots.data(), device_capacity, is_full_size ? device_capacity : hash_csr_max_probes};
    launch_hash_csr_build_kernel(num_rows,
                                 row_bitmask,
                                 need_grouped_rows ? positions.data() : nullptr,
                                 need_group_offsets ? slot_counts.data() : nullptr,
                                 count_by_representative,
                                 set,
                                 d_row_equal,
                                 d_row_hash,
                                 is_full_size ? nullptr : overflow->data(),
                                 stream);

    if (count_by_representative) {
      // Count indices identify representative rows, so selection no longer needs the table.
      slots = cuda::device_buffer<slot_type>{stream, temp_mr};
    }

    if (!need_group_offsets) {
      key_indices = cuda::device_buffer<size_type>{
        stream, mr.get_output_mr(), std::min<std::size_t>(num_rows, capacity), cuda::no_init};
      auto const key_indices_end = thrust::copy_if(
        policy, slots.data(), slots.data() + slots.size(), key_indices.data(), is_occupied_fn{});
      num_groups = static_cast<size_type>(cuda::std::distance(key_indices.data(), key_indices_end));
    } else {
      group_slots = cuda::device_buffer<cuda::std::uint32_t>{
        stream, temp_mr, std::min<std::size_t>(num_rows, capacity), cuda::no_init};
      auto const group_slots_end =
        thrust::copy_if(policy,
                        cuda::counting_iterator<cuda::std::uint32_t>{0},
                        cuda::counting_iterator<cuda::std::uint32_t>{
                          static_cast<cuda::std::uint32_t>(count_capacity)},
                        slot_counts.data(),
                        group_slots.data(),
                        [] __device__(size_type count) -> bool { return count > 0; });
      num_groups = static_cast<size_type>(cuda::std::distance(group_slots.data(), group_slots_end));
    }

    // The compaction has just synchronized the stream, so reading the flag is cheap here.
    if (is_full_size || overflow->value(stream) == 0) { break; }

    // The retry overwrites the table, so release it instead of copying it while growing.
    slots       = cuda::device_buffer<slot_type>{stream, temp_mr};
    slot_counts = cuda::device_buffer<size_type>{stream, temp_mr};
    key_indices = cuda::device_buffer<size_type>{stream, mr.get_output_mr()};
    group_slots = cuda::device_buffer<cuda::std::uint32_t>{stream, temp_mr};
    overflow.reset();
    capacity = full_capacity;
  }

  if (!need_group_offsets) {
    return {std::move(key_indices), std::nullopt, std::nullopt, num_groups};
  }

  // Every row is an included singleton group, so input order already forms a valid grouping.
  if (num_groups == num_rows) {
    slots       = cuda::device_buffer<slot_type>{stream, temp_mr};
    slot_counts = cuda::device_buffer<size_type>{stream, temp_mr};
    positions   = cuda::device_buffer<build_position_type>{stream, temp_mr};
    group_slots = cuda::device_buffer<cuda::std::uint32_t>{stream, temp_mr};
    overflow.reset();

    key_indices = cuda::device_buffer<size_type>{
      stream, mr.get_output_mr(), static_cast<std::size_t>(num_rows), cuda::no_init};
    cuda::device_buffer<size_type> group_offsets(
      stream, mr.get_output_mr(), static_cast<std::size_t>(num_rows) + 1, cuda::no_init);
    std::optional<cuda::device_buffer<size_type>> grouped_rows;
    if (need_grouped_rows) {
      grouped_rows.emplace(stream, mr.get_output_mr(), num_rows, cuda::no_init);
    }
    auto const singleton_outputs = cuda::tabulate_output_iterator{
      [key_indices   = key_indices.data(),
       group_offsets = group_offsets.data(),
       grouped_rows  = grouped_rows ? grouped_rows->data() : nullptr,
       num_rows] __device__(cuda::std::ptrdiff_t index, size_type value) -> void {
        group_offsets[index] = value;
        if (index < num_rows) {
          key_indices[index] = value;
          if (grouped_rows != nullptr) { grouped_rows[index] = value; }
        }
      }};
    thrust::sequence(
      policy, singleton_outputs, singleton_outputs + group_offsets.size(), size_type{0});
    return {std::move(key_indices), std::move(group_offsets), std::move(grouped_rows), num_groups};
  }

  auto const slot_rows = slots.data();
  key_indices          = cuda::device_buffer<size_type>{
    stream, mr.get_output_mr(), static_cast<std::size_t>(num_groups), cuda::no_init};
  if (count_by_representative) {
    thrust::copy(policy, group_slots.data(), group_slots.data() + num_groups, key_indices.data());
  } else {
    thrust::gather(
      policy, group_slots.data(), group_slots.data() + num_groups, slot_rows, key_indices.data());
  }
  slots = cuda::device_buffer<slot_type>{stream, temp_mr};

  cuda::device_buffer<size_type> group_offsets(
    stream, mr.get_output_mr(), static_cast<std::size_t>(num_groups) + 1, cuda::no_init);
  CUDF_CUDA_TRY(cudaMemsetAsync(group_offsets.data(), 0, sizeof(size_type), stream.get()));
  auto const group_counts = cuda::make_permutation_iterator(slot_counts.data(), group_slots.data());
  thrust::inclusive_scan(policy, group_counts, group_counts + num_groups, group_offsets.data() + 1);
  if (!need_grouped_rows) {
    return {std::move(key_indices), std::move(group_offsets), std::nullopt, num_groups};
  }

  auto num_grouped_rows = num_rows;
  if (row_bitmask != nullptr) {
    cudf::detail::cuda_memcpy(host_span<size_type>{&num_grouped_rows, 1},
                              device_span<size_type const>{group_offsets.data() + num_groups, 1},
                              stream);
  }

  // Reuse the slot counts to hold the start offset of the group of each occupied slot, then
  // scatter every row to its group.
  thrust::scatter(policy,
                  group_offsets.data(),
                  group_offsets.data() + num_groups,
                  group_slots.data(),
                  slot_counts.data());
  group_slots = cuda::device_buffer<cuda::std::uint32_t>{stream, temp_mr};
  cuda::device_buffer<size_type> grouped_rows(
    stream, mr.get_output_mr(), num_grouped_rows, cuda::no_init);
  launch_hash_csr_fill_kernel(
    num_rows, positions.data(), slot_counts.data(), grouped_rows.data(), stream);

  return {std::move(key_indices), std::move(group_offsets), std::move(grouped_rows), num_groups};
}

}  // namespace

grouped_keys group_keys(table_view const& keys,
                        null_policy include_null_keys,
                        bool need_group_offsets,
                        bool need_grouped_rows,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  CUDF_EXPECTS(need_group_offsets || !need_grouped_rows,
               "Grouped rows require group offsets",
               std::invalid_argument);
  auto const num_rows = keys.num_rows();
  auto const skip_rows_with_nulls =
    cudf::has_nulls(keys) && include_null_keys == null_policy::EXCLUDE;
  auto const has_null = nullate::DYNAMIC{cudf::has_nested_nulls(keys)};
  auto preprocessed_keys =
    cudf::detail::row::hash::preprocessed_table::create(keys, stream, mr.get_temporary_mr());
  auto const comparator = cudf::detail::row::equality::self_comparator{preprocessed_keys};
  auto const row_hash   = cudf::detail::row::hash::row_hasher{std::move(preprocessed_keys)};
  auto const d_row_hash = row_hash.device_hasher(has_null);
  auto [row_bitmask_data, row_bitmask] =
    skip_rows_with_nulls
      ? cudf::groupby::detail::compute_row_bitmask(keys, stream)
      : std::pair<cuda::device_buffer<cudf::bitmask_type>, bitmask_type const*>{
          cudf::create_null_mask(0, mask_state::UNALLOCATED, stream, mr.get_temporary_mr()),
          nullptr};
  std::optional<std::size_t> domain_capacity;
  if (num_rows >= hash_csr_min_rows_to_estimate) {
    domain_capacity = small_key_domain_capacity(keys, skip_rows_with_nulls);
  }
  if (cudf::detail::has_nested_columns(keys)) {
    return group_keys(num_rows,
                      row_bitmask,
                      comparator.equal_to<true>(has_null, null_equality::EQUAL),
                      d_row_hash,
                      need_group_offsets,
                      need_grouped_rows,
                      domain_capacity,
                      stream,
                      mr);
  }
  return group_keys(num_rows,
                    row_bitmask,
                    comparator.equal_to<false>(has_null, null_equality::EQUAL),
                    d_row_hash,
                    need_group_offsets,
                    need_grouped_rows,
                    domain_capacity,
                    stream,
                    mr);
}

}  // namespace cudf::groupby::detail::hash

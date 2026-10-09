/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "group_keys.hpp"
#include "groupby/common/utils.hpp"
#include "hash_csr_kernels.cuh"
#include "helpers.cuh"

#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/functional>
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

struct singleton_group_output {
  size_type* rows;
  size_type* offsets;
  size_type num_rows;

  __device__ void operator()(cuda::std::ptrdiff_t index, size_type value) const
  {
    offsets[index] = value;
    if (index < num_rows) { rows[index] = value; }
  }
};

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
  rmm::device_uvector<slot_type> slots(hash_csr_sample_capacity, stream, temp_mr);
  rmm::device_uvector<size_type> counts(2, stream, temp_mr);
  // Counts the valid rows among every `stride`-th row and the distinct keys among them.
  auto const sample = [&](size_type stride) {
    thrust::uninitialized_fill(
      policy, slots.begin(), slots.end(), cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
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
    auto const h_counts =
      cudf::detail::make_pinned_vector(device_span<size_type const>{counts}, stream);
    return std::pair{static_cast<double>(h_counts[0]), static_cast<double>(h_counts[1])};
  };
  // Sample with a prime stride of 67, or a larger odd stride to keep the sample table at most
  // half full. Odd strides avoid repeatedly sampling the same phase of power-of-two patterns.
  auto const stride =
    std::max<size_type>(67, cuda::ceil_div(num_rows, hash_csr_sample_capacity / 2)) | 1;
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

/// Stable grouped rows and, when kept, the group of each grouped row.
struct stable_rows {
  rmm::device_uvector<size_type> rows;
  rmm::device_uvector<size_type> labels;
};

/**
 * @brief Materializes stable grouped rows directly from the HashCSR build positions.
 *
 * Sorting the rows by group label also sorts the labels, so `keep_labels` returns them instead of
 * discarding them. The singleton and single-group shortcuts do not sort and return no labels.
 */
stable_rows stable_grouped_rows(size_type num_rows,
                                size_type num_grouped_rows,
                                device_span<size_type const> key_rows,
                                rmm::device_uvector<build_position_type> positions,
                                rmm::device_uvector<size_type> slot_counts,
                                rmm::device_uvector<cuda::std::uint32_t> group_slots,
                                bool keep_labels,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr)
{
  auto const temp_mr    = mr.get_temporary_mr();
  auto const output_mr  = mr.get_output_mr();
  auto const policy     = rmm::exec_policy_nosync(stream, temp_mr);
  auto const num_groups = static_cast<size_type>(key_rows.size());
  auto const release    = [stream](auto& buffer) {
    buffer.resize(0, stream);
    buffer.shrink_to_fit(stream);
  };

  if (num_groups == num_grouped_rows) {
    // Every included group is a singleton, including the empty grouping.
    release(positions);
    release(slot_counts);
    release(group_slots);
    rmm::device_uvector<size_type> rows(num_grouped_rows, stream, output_mr);
    thrust::copy(policy, key_rows.begin(), key_rows.end(), rows.begin());
    return {std::move(rows), rmm::device_uvector<size_type>{0, stream, output_mr}};
  }
  if (num_groups == 1) {
    release(slot_counts);
    release(group_slots);
    rmm::device_uvector<size_type> rows(num_grouped_rows, stream, output_mr);
    if (num_grouped_rows == num_rows) {
      release(positions);
      thrust::sequence(policy, rows.begin(), rows.end(), size_type{0});
    } else {
      thrust::copy_if(policy,
                      cuda::counting_iterator<size_type>{0},
                      cuda::counting_iterator<size_type>{num_rows},
                      rows.begin(),
                      [positions = positions.data()] __device__(size_type row) {
                        return positions[row].first != hash_csr_no_slot;
                      });
    }
    return {std::move(rows), rmm::device_uvector<size_type>{0, stream, output_mr}};
  }

  // The count indices can be hash slots or representative rows. Reuse their counts for
  // compact group IDs after the group offsets have consumed the counts.
  thrust::scatter(policy,
                  cuda::counting_iterator<size_type>{0},
                  cuda::counting_iterator<size_type>{num_groups},
                  group_slots.begin(),
                  slot_counts.begin());
  release(group_slots);
  auto const labels_mr = keep_labels ? output_mr : temp_mr;
  rmm::device_uvector<size_type> input_labels(num_rows, stream, labels_mr);
  rmm::device_uvector<size_type> input_rows(num_rows, stream, output_mr);
  auto const initialize_rows = cuda::tabulate_output_iterator{
    [positions = positions.data(),
     group_ids = slot_counts.data(),
     labels    = input_labels.data(),
     rows      = input_rows.data(),
     num_groups] __device__(cuda::std::ptrdiff_t index, size_type row) {
      auto const slot = positions[index].first;
      labels[index]   = slot == hash_csr_no_slot ? num_groups : group_ids[slot];
      rows[index]     = row;
    }};
  thrust::sequence(policy, initialize_rows, initialize_rows + num_rows, size_type{0});
  release(positions);
  release(slot_counts);

  rmm::device_uvector<size_type> ordered_labels(num_rows, stream, labels_mr);
  rmm::device_uvector<size_type> alternate_rows(num_rows, stream, output_mr);
  auto const end_bit = cuda::std::bit_width(
    static_cast<cuda::std::uint32_t>(num_grouped_rows == num_rows ? num_groups - 1 : num_groups));
  auto const sorted = stable_sort_group_rows({input_labels.data(), input_rows.data()},
                                             {ordered_labels.data(), alternate_rows.data()},
                                             num_rows,
                                             end_bit,
                                             stream,
                                             mr);
  auto grouped_rows =
    sorted.rows == input_rows.data() ? std::move(input_rows) : std::move(alternate_rows);
  // Excluded rows sort into a sentinel tail. Retain exactly the included prefix's storage.
  grouped_rows.resize(num_grouped_rows, stream);
  grouped_rows.shrink_to_fit(stream);
  if (!keep_labels) {
    return {std::move(grouped_rows), rmm::device_uvector<size_type>{0, stream, output_mr}};
  }
  auto grouped_labels =
    sorted.labels == input_labels.data() ? std::move(input_labels) : std::move(ordered_labels);
  grouped_labels.resize(num_grouped_rows, stream);
  grouped_labels.shrink_to_fit(stream);
  return {std::move(grouped_rows), std::move(grouped_labels)};
}

}  // namespace

group_row_buffers stable_sort_group_rows(group_row_buffers input,
                                         group_row_buffers alternate,
                                         size_type num_rows,
                                         int end_bit,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr)
{
  auto labels            = cub::DoubleBuffer<size_type>{input.labels, alternate.labels};
  auto rows              = cub::DoubleBuffer<size_type>{input.rows, alternate.rows};
  std::size_t temp_bytes = 0;
  CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
    nullptr, temp_bytes, labels, rows, num_rows, 0, end_bit, stream.get()));
  cuda::device_buffer<std::byte> temp(stream, mr.get_temporary_mr(), temp_bytes, cuda::no_init);
  CUDF_CUDA_TRY(cub::DeviceRadixSort::SortPairs(
    temp.data(), temp_bytes, labels, rows, num_rows, 0, end_bit, stream.get()));
  return {labels.Current(), rows.Current()};
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
                        bool stable_rows,
                        bool keep_labels,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  CUDF_EXPECTS(need_group_offsets || !need_grouped_rows, "Grouped rows require group offsets");
  auto const temp_mr   = mr.get_temporary_mr();
  auto const output_mr = mr.get_output_mr();
  auto const policy    = rmm::exec_policy_nosync(stream, temp_mr);
  auto const release   = [stream](auto& buffer) {
    buffer.resize(0, stream);
    buffer.shrink_to_fit(stream);
  };

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

  rmm::device_uvector<slot_type> slots(0, stream, temp_mr);
  rmm::device_uvector<size_type> slot_counts(0, stream, temp_mr);
  rmm::device_uvector<build_position_type> positions(
    need_grouped_rows ? num_rows : 0, stream, temp_mr);

  // Set by the build when the estimated table turns out to be too small.
  std::optional<cudf::detail::device_scalar<cuda::std::int32_t>> overflow;
  if (capacity < full_capacity) { overflow.emplace(0, stream, temp_mr); }

  // The occupied slots, in slot order, are the groups: without aggregations the slots hold the
  // one row wanted for each group, otherwise the slot indices lead to the counts and rows.
  rmm::device_uvector<size_type> key_rows(0, stream, output_mr);
  rmm::device_uvector<cuda::std::uint32_t> group_slots(0, stream, temp_mr);
  bool count_by_representative{};

  while (true) {
    auto const is_full_size = capacity == full_capacity;
    count_by_representative = need_group_offsets && static_cast<std::size_t>(num_rows) < capacity;
    auto const count_capacity =
      count_by_representative ? static_cast<std::size_t>(num_rows) : capacity;

    slots.resize(capacity, stream);
    thrust::uninitialized_fill(
      policy, slots.begin(), slots.end(), cudf::detail::CUDF_SIZE_TYPE_SENTINEL);
    if (need_group_offsets) {
      slot_counts.resize(count_capacity, stream);
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
      release(slots);
    }

    if (!need_group_offsets) {
      key_rows.resize(std::min<std::size_t>(num_rows, capacity), stream);
      auto const key_rows_end =
        thrust::copy_if(policy, slots.begin(), slots.end(), key_rows.begin(), is_occupied_fn{});
      key_rows.resize(cuda::std::distance(key_rows.begin(), key_rows_end), stream);
    } else {
      group_slots.resize(std::min<std::size_t>(num_rows, capacity), stream);
      auto const group_slots_end =
        thrust::copy_if(policy,
                        cuda::counting_iterator<cuda::std::uint32_t>{0},
                        cuda::counting_iterator<cuda::std::uint32_t>{
                          static_cast<cuda::std::uint32_t>(count_capacity)},
                        slot_counts.begin(),
                        group_slots.begin(),
                        cuda::std::identity{});
      group_slots.resize(cuda::std::distance(group_slots.begin(), group_slots_end), stream);
    }

    // The compaction has just synchronized the stream, so reading the flag is cheap here.
    if (is_full_size || overflow->value(stream) == 0) { break; }

    // The retry overwrites the table, so release it instead of copying it while growing.
    release(slots);
    release(slot_counts);
    release(key_rows);
    release(group_slots);
    overflow.reset();
    capacity = full_capacity;
  }

  auto const no_rows = [&] { return rmm::device_uvector<size_type>{0, stream, output_mr}; };
  if (!need_group_offsets) {
    auto const num_groups = static_cast<size_type>(key_rows.size());
    return {num_groups, 0, std::move(key_rows), no_rows(), no_rows(), no_rows()};
  }

  auto const num_groups = static_cast<size_type>(group_slots.size());
  // Every row is an included singleton group, so input order already forms a valid grouping.
  if (num_groups == num_rows) {
    release(slots);
    release(slot_counts);
    release(positions);
    release(group_slots);
    overflow.reset();

    // Singleton group offsets already identify each representative input row.
    key_rows.resize(need_grouped_rows ? 0 : num_rows, stream);
    rmm::device_uvector<size_type> group_offsets(
      static_cast<std::size_t>(num_rows) + 1, stream, output_mr);
    rmm::device_uvector<size_type> grouped_rows(
      need_grouped_rows ? num_rows : 0, stream, output_mr);
    auto const singleton_outputs = cuda::tabulate_output_iterator{singleton_group_output{
      need_grouped_rows ? grouped_rows.data() : key_rows.data(), group_offsets.data(), num_rows}};
    thrust::sequence(
      policy, singleton_outputs, singleton_outputs + group_offsets.size(), size_type{0});
    return {num_groups,
            num_rows,
            std::move(key_rows),
            std::move(group_offsets),
            std::move(grouped_rows),
            no_rows()};
  }

  auto const slot_rows = slots.begin();
  key_rows.resize(num_groups, stream);
  if (count_by_representative) {
    thrust::copy(policy, group_slots.begin(), group_slots.end(), key_rows.begin());
  } else {
    thrust::gather(policy, group_slots.begin(), group_slots.end(), slot_rows, key_rows.begin());
  }
  release(slots);

  rmm::device_uvector<size_type> group_offsets(
    static_cast<std::size_t>(num_groups) + 1, stream, output_mr);
  group_offsets.set_element_to_zero_async(0, stream);
  auto const group_counts =
    cuda::make_permutation_iterator(slot_counts.begin(), group_slots.begin());
  thrust::inclusive_scan(
    policy, group_counts, group_counts + num_groups, group_offsets.begin() + 1);
  auto num_grouped_rows = num_rows;
  if (row_bitmask != nullptr) {
    cudf::detail::cuda_memcpy(host_span<size_type>{&num_grouped_rows, 1},
                              device_span<size_type const>{group_offsets.data() + num_groups, 1},
                              stream);
  }
  if (!need_grouped_rows) {
    return {num_groups,
            num_grouped_rows,
            std::move(key_rows),
            std::move(group_offsets),
            no_rows(),
            no_rows()};
  }

  if (stable_rows) {
    auto stable = stable_grouped_rows(num_rows,
                                      num_grouped_rows,
                                      key_rows,
                                      std::move(positions),
                                      std::move(slot_counts),
                                      std::move(group_slots),
                                      keep_labels,
                                      stream,
                                      mr);
    return {num_groups,
            num_grouped_rows,
            std::move(key_rows),
            std::move(group_offsets),
            std::move(stable.rows),
            std::move(stable.labels)};
  }

  // Reuse the slot counts to hold the start offset of the group of each occupied slot, then
  // scatter every row to its group.
  thrust::scatter(policy,
                  group_offsets.begin(),
                  group_offsets.begin() + num_groups,
                  group_slots.begin(),
                  slot_counts.begin());
  release(group_slots);
  rmm::device_uvector<size_type> grouped_rows(num_grouped_rows, stream, output_mr);
  launch_hash_csr_fill_kernel(
    num_rows, positions.data(), slot_counts.data(), grouped_rows.data(), stream);

  return {num_groups,
          num_grouped_rows,
          std::move(key_rows),
          std::move(group_offsets),
          std::move(grouped_rows),
          no_rows()};
}

grouped_keys group_keys(table_view const& keys,
                        null_policy include_null_keys,
                        bool need_group_offsets,
                        bool need_grouped_rows,
                        bool stable_rows,
                        bool keep_labels,
                        cuda::stream_ref stream,
                        cudf::memory_resources mr)
{
  auto const num_rows             = keys.num_rows();
  auto const skip_rows_with_nulls = include_null_keys == null_policy::EXCLUDE;
  auto [row_bitmask_data, row_bitmask] =
    skip_rows_with_nulls
      ? compute_row_bitmask(
          keys, stream, cudf::memory_resources{mr.get_temporary_mr(), mr.get_temporary_mr()})
      : std::pair<cuda::device_buffer<std::byte>, bitmask_type const*>{
          cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr.get_temporary_mr()),
          nullptr};
  std::optional<std::size_t> domain_capacity;
  if (num_rows >= hash_csr_min_rows_to_estimate) {
    domain_capacity = small_key_domain_capacity(keys, skip_rows_with_nulls);
  }
  auto preprocessed_keys =
    cudf::detail::row::hash::preprocessed_table::create(keys, stream, mr.get_temporary_mr());
  auto const comparator = cudf::detail::row::equality::self_comparator{preprocessed_keys};
  auto const row_hash   = cudf::detail::row::hash::row_hasher{std::move(preprocessed_keys)};
  auto const has_null   = nullate::DYNAMIC{cudf::has_nested_nulls(keys)};
  auto const d_row_hash = row_hash.device_hasher(has_null);
  if (cudf::detail::has_nested_columns(keys)) {
    return group_keys(num_rows,
                      row_bitmask,
                      comparator.equal_to<true>(has_null, null_equality::EQUAL),
                      d_row_hash,
                      need_group_offsets,
                      need_grouped_rows,
                      domain_capacity,
                      stable_rows,
                      keep_labels,
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
                    stable_rows,
                    keep_labels,
                    stream,
                    mr);
}

}  // namespace cudf::groupby::detail::hash

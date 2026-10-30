/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "join/join_common_utils.cuh"

#include <cudf/copying.hpp>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/detail/join/join.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/row_operator/equality.cuh>
#include <cudf/detail/row_operator/hashing.cuh>
#include <cudf/detail/row_operator/primitive_equality.cuh>
#include <cudf/detail/row_operator/primitive_hashing.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/join/streaming_hash_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/prefetch.hpp>
#include <cudf/utilities/span.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/device/device_scan.cuh>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/cmath>
#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace cudf::detail {
namespace {

/// The low bits of a slot hold a global right row and the high bits a hash fingerprint.
using slot_type = cuda::std::uint32_t;

constexpr thread_index_type block_size      = 256;
constexpr thread_index_type warps_per_block = block_size / cudf::detail::warp_size;
constexpr slot_type empty_slot              = cuda::std::numeric_limits<slot_type>::max();
/// A bucket fills one 32-byte memory sector, so each probe step examines several slots at once.
constexpr cuda::std::uint32_t bucket_size = 8;
/// Slots per 16-byte vector load.
constexpr cuda::std::uint32_t vector_slots = 4;
/**
 * @brief Lanes that probe one left row together, each examining its share of every bucket.
 *
 * Nested row comparisons are long and branchy. Lanes that compare the same left row against
 * duplicate keys follow the same path, so nested keys probe each row with two lanes. Other keys
 * compare cheaply and gain more from probing more rows at once.
 */
template <bool has_nested>
constexpr int probe_group_size = has_nested ? 2 : 1;

template <typename Equality, typename Factory>
auto make_device_comparators(
  std::span<std::shared_ptr<row::equality::preprocessed_table> const> preprocessed_right,
  Factory factory,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref temp_mr)
{
  using allocator_type  = cudf::detail::rmm_host_allocator<Equality>;
  auto host_comparators = std::vector<Equality, allocator_type>{
    allocator_type{cudf::get_pinned_memory_resource(), stream}};
  host_comparators.reserve(preprocessed_right.size());
  std::transform(preprocessed_right.begin(),
                 preprocessed_right.end(),
                 std::back_inserter(host_comparators),
                 factory);
  auto d_comparators = cudf::detail::make_device_uvector_async(
    cudf::host_span<Equality const>{
      host_comparators.data(), host_comparators.size(), /*is_device_accessible=*/true},
    stream,
    temp_mr);
  stream.sync();  // wait for host_comparators to finish copying before it goes out of scope
  return d_comparators;
}

template <bool has_nested>
auto make_device_row_comparators(
  std::shared_ptr<row::equality::preprocessed_table> const& preprocessed_left,
  std::span<std::shared_ptr<row::equality::preprocessed_table> const> preprocessed_right,
  nullate::DYNAMIC has_nulls,
  null_equality compare_nulls,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref temp_mr)
{
  using equality_type =
    row::equality::device_row_comparator<has_nested,
                                         nullate::DYNAMIC,
                                         row::equality::nan_equal_physical_equality_comparator>;

  return make_device_comparators<equality_type>(
    preprocessed_right,
    [&](auto const& right) {
      auto const comparator = row::equality::two_table_comparator{preprocessed_left, right};
      return comparator.equal_to<has_nested>(has_nulls, compare_nulls).comparator;
    },
    stream,
    temp_mr);
}

auto make_device_primitive_row_comparators(
  std::shared_ptr<row::equality::preprocessed_table> const& preprocessed_left,
  std::span<std::shared_ptr<row::equality::preprocessed_table> const> preprocessed_right,
  nullate::DYNAMIC has_nulls,
  null_equality compare_nulls,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref temp_mr)
{
  using equality_type = row::primitive::row_equality_comparator;

  return make_device_comparators<equality_type>(
    preprocessed_right,
    [&](auto const& right) {
      return equality_type{has_nulls, preprocessed_left, right, compare_nulls};
    },
    stream,
    temp_mr);
}

size_type checked_batch_count(size_type batches)
{
  CUDF_EXPECTS(
    batches > 0, "streaming_hash_join requires max_num_batches > 0.", std::invalid_argument);
  return batches;
}

bool is_prime(std::uint64_t value)
{
  if (value < 4) { return value > 1; }
  if (value % 2 == 0 || value % 3 == 0) { return false; }
  for (std::uint64_t divisor = 5; divisor * divisor <= value; divisor += 6) {
    if (value % divisor == 0 || value % (divisor + 2) == 0) { return false; }
  }
  return true;
}

/// Returns a prime number of buckets that holds `rows` at `load_factor`.
cuda::std::uint32_t bucket_count(size_type rows, double load_factor)
{
  CUDF_EXPECTS(
    rows >= 0, "streaming_hash_join requires total_right_rows >= 0.", std::invalid_argument);
  // Keep one empty slot even at load_factor == 1 so that every probe sequence terminates.
  auto const slots =
    std::max(static_cast<long double>(rows) + 1,
             std::ceil(static_cast<long double>(rows) / checked_load_factor(load_factor)));
  // Double hashing over a prime number of buckets visits every bucket.
  auto buckets =
    std::max(std::uint64_t{2}, static_cast<std::uint64_t>(std::ceil(slots / bucket_size)));
  while (!is_prime(buckets)) {
    ++buckets;
  }
  CUDF_EXPECTS(buckets <= std::numeric_limits<cuda::std::uint32_t>::max() / bucket_size,
               "streaming_hash_join slot table capacity is not representable",
               std::overflow_error);
  return static_cast<cuda::std::uint32_t>(buckets);
}

/// Device view of the slot table: a prime number of buckets probed with double hashing.
struct slot_table_ref {
  slot_type* slots;
  cuda::std::uint32_t num_buckets;
  slot_type row_mask;

  /// Returns the first bucket and the bucket step of the probe sequence of `hash`.
  __device__ cuda::std::pair<cuda::std::uint32_t, cuda::std::uint32_t> probe_sequence(
    hash_value_type hash) const
  {
    // Multiply-high maps a hash into a range without division. The first bucket depends on the
    // low hash bits, leaving the high bits to the fingerprint, and the step on the rehashed hash.
    auto const rehash = cudf::hashing::detail::MurmurHash3_x86_32<hash_value_type>{}(hash);
    return {__umulhi(__brev(hash), num_buckets), 1 + __umulhi(rehash, num_buckets - 1)};
  }

  [[nodiscard]] __device__ cuda::std::uint32_t next(cuda::std::uint32_t bucket,
                                                    cuda::std::uint32_t step) const
  {
    bucket += step;
    return bucket >= num_buckets ? bucket - num_buckets : bucket;
  }

  [[nodiscard]] __device__ slot_type* bucket_slots(cuda::std::uint32_t bucket) const
  {
    return slots + static_cast<std::size_t>(bucket) * bucket_size;
  }

  /// Loads the `N` slots of `bucket` that start at slot `first`.
  template <cuda::std::uint32_t N>
  [[nodiscard]] __device__ cuda::std::array<slot_type, N> load(cuda::std::uint32_t bucket,
                                                               cuda::std::uint32_t first) const
  {
    static_assert(N % vector_slots == 0);
    auto const* vectors = reinterpret_cast<uint4 const*>(bucket_slots(bucket) + first);
    cuda::std::array<slot_type, N> window;
#pragma unroll
    for (cuda::std::uint32_t i = 0; i < N / vector_slots; ++i) {
      auto const vector            = vectors[i];
      window[i * vector_slots]     = vector.x;
      window[i * vector_slots + 1] = vector.y;
      window[i * vector_slots + 2] = vector.z;
      window[i * vector_slots + 3] = vector.w;
    }
    return window;
  }

  /// Returns a bit for each slot of `window` whose fingerprint matches `hash`.
  template <cuda::std::size_t N>
  [[nodiscard]] __device__ cuda::std::uint32_t candidates(
    cuda::std::array<slot_type, N> const& window, hash_value_type hash) const
  {
    cuda::std::uint32_t mask = 0;
#pragma unroll
    for (cuda::std::uint32_t slot = 0; slot < N; ++slot) {
      if (window[slot] != empty_slot && ((hash ^ window[slot]) & ~row_mask) == 0) {
        mask |= cuda::std::uint32_t{1} << slot;
      }
    }
    return mask;
  }
};

/// Slots fill each bucket from the front, so a probe sequence ends at a bucket with an empty slot.
template <cuda::std::size_t N>
__device__ bool has_empty_slot(cuda::std::array<slot_type, N> const& window)
{
  return window[N - 1] == empty_slot;
}

/**
 * @brief Compares a left row with the right row stored in a slot.
 *
 * Slots store global right rows. Batches occupy consecutive global rows in batch ID order, so the
 * batch of a row is the last batch that starts at or before it.
 */
template <typename RowEqual>
struct slot_equal {
  RowEqual const* comparators;    ///< Left table against each inserted batch
  size_type const* batch_starts;  ///< First global row of each batch, in batch ID order
  size_type num_batches;
  slot_type row_mask;

  /// Returns the batch and batch-local row of the slot's right row when it equals `left_row`,
  /// or a sentinel batch otherwise.
  __device__ cuda::std::pair<size_type, size_type> operator()(size_type left_row,
                                                              slot_type slot) const
  {
    auto const right_row = static_cast<size_type>(slot & row_mask);
    // Every lane takes the same number of steps, so this search does not diverge.
    size_type batch = 0;
    for (auto stride = cuda::std::bit_floor(static_cast<cuda::std::uint32_t>(num_batches));
         stride > 0;
         stride >>= 1) {
      auto const next = batch + static_cast<size_type>(stride);
      if (next < num_batches && batch_starts[next] <= right_row) { batch = next; }
    }
    auto const batch_row = right_row - batch_starts[batch];
    if (!comparators[batch](left_row, batch_row)) { return {CUDF_SIZE_TYPE_SENTINEL, 0}; }
    return {batch, batch_row};
  }
};

/**
 * @brief Inserts every valid row into its own slot.
 *
 * A row claims the first empty slot of the first bucket along its probe sequence that has one.
 * Slots are claimed in order, so each bucket fills from the front and a probe can stop at the
 * first empty slot.
 */
template <typename Hasher>
CUDF_KERNEL void insert_kernel(size_type num_rows,
                               size_type first_row,
                               bitmask_type const* valid_rows,
                               slot_table_ref table,
                               Hasher hasher)
{
  auto const stride = grid_1d::grid_stride();
  for (auto i = grid_1d::global_thread_id(); i < num_rows; i += stride) {
    auto const row = static_cast<size_type>(i);
    if (valid_rows != nullptr && !cudf::bit_is_set(valid_rows, row)) { continue; }
    auto const hash     = hasher(row);
    auto const desired  = (hash & ~table.row_mask) | static_cast<slot_type>(first_row + row);
    auto [bucket, step] = table.probe_sequence(hash);
    // The table has more slots than rows, so a bucket with an empty slot is always found.
    for (auto inserted = false; !inserted; bucket = table.next(bucket, step)) {
      auto const window = table.load<bucket_size>(bucket, 0);
      auto* const slots = table.bucket_slots(bucket);
      for (cuda::std::uint32_t slot = 0; slot < bucket_size && !inserted; ++slot) {
        auto expected = empty_slot;
        inserted      = window[slot] == empty_slot &&
                   cuda::atomic_ref<slot_type, cuda::thread_scope_device>{slots[slot]}
                     .compare_exchange_strong(expected, desired, cuda::memory_order_relaxed);
      }
    }
  }
}

using warp_tile = cooperative_groups::thread_block_tile<cudf::detail::warp_size>;

/// Returns the index of the calling thread's warp in the grid.
__device__ thread_index_type warp_index()
{
  return static_cast<thread_index_type>(blockIdx.x) * warps_per_block +
         static_cast<thread_index_type>(threadIdx.x) / cudf::detail::warp_size;
}

/**
 * @brief Probes the left rows of `warp` one bucket per step, calling
 * `on_bucket(row, candidates, slots)` on every lane per step.
 *
 * Each group of `GroupSize` lanes probes one left row, examining its share of each bucket, and each
 * warp probes tiles of rows in the same order in every kernel, so the count and retrieve kernels
 * agree on which warp owns each match. `candidates` has a bit for each slot in `slots` whose
 * fingerprint matches the row. The lanes of a warp advance their buckets together, so they also
 * compare their candidates together.
 */
template <int GroupSize, typename Hasher, typename OnBucket>
__device__ void probe_buckets(warp_tile const& warp,
                              size_type num_rows,
                              slot_table_ref table,
                              Hasher const& hasher,
                              OnBucket&& on_bucket)
{
  constexpr auto lane_slots = bucket_size / GroupSize;
  auto const group          = cooperative_groups::tiled_partition<GroupSize>(warp);
  auto const first_slot     = static_cast<cuda::std::uint32_t>(group.thread_rank()) * lane_slots;
  auto const num_warps      = static_cast<thread_index_type>(gridDim.x) * warps_per_block;
  auto constexpr tile_rows  = cudf::detail::warp_size / GroupSize;
  for (auto tile = warp_index(); tile * tile_rows < num_rows; tile += num_warps) {
    auto const row   = tile * tile_rows + static_cast<thread_index_type>(group.meta_group_rank());
    auto searching   = row < num_rows;
    auto const index = static_cast<size_type>(row);
    auto const hash  = searching ? hasher(index) : hash_value_type{0};
    auto [bucket, step] = searching ? table.probe_sequence(hash)
                                    : cuda::std::pair<cuda::std::uint32_t, cuda::std::uint32_t>{};
    while (warp.any(searching)) {
      cuda::std::uint32_t candidates = 0;
      slot_type const* slots         = nullptr;
      if (searching) {
        auto const window = table.load<lane_slots>(bucket, first_slot);
        candidates        = table.candidates(window, hash);
        searching         = !group.any(has_empty_slot(window));
        slots             = table.bucket_slots(bucket) + first_slot;
        bucket            = table.next(bucket, step);
      }
      on_bucket(index, candidates, slots);
    }
  }
}

/// Counts the matches of each warp.
template <int GroupSize, typename Equal, typename Hasher>
CUDF_KERNEL void count_kernel(size_type num_rows,
                              slot_table_ref table,
                              Equal equal,
                              Hasher hasher,
                              cuda::std::int64_t* warp_counts)
{
  auto const warp = cooperative_groups::tiled_partition<cudf::detail::warp_size>(
    cooperative_groups::this_thread_block());
  cuda::std::int64_t count = 0;
  probe_buckets<GroupSize>(
    warp, num_rows, table, hasher, [&](size_type row, auto candidates, auto const* slots) {
      for (; candidates != 0; candidates &= candidates - 1) {
        if (equal(row, slots[__ffs(candidates) - 1]).first != CUDF_SIZE_TYPE_SENTINEL) { ++count; }
      }
    });
  count = cooperative_groups::reduce(warp, count, cooperative_groups::plus<cuda::std::int64_t>{});
  if (warp.thread_rank() == 0) { warp_counts[warp_index()] = count; }
}

/// Writes the matches of each warp starting at its offset from the scanned `count_kernel` counts.
template <int GroupSize, typename Equal, typename Hasher>
CUDF_KERNEL void retrieve_kernel(size_type num_rows,
                                 slot_table_ref table,
                                 Equal equal,
                                 Hasher hasher,
                                 cuda::std::int64_t const* warp_offsets,
                                 size_type* left_indices,
                                 size_type* batch_indices,
                                 size_type* row_indices)
{
  auto const warp = cooperative_groups::tiled_partition<cudf::detail::warp_size>(
    cooperative_groups::this_thread_block());
  auto const lower_lanes = (cuda::std::uint32_t{1} << warp.thread_rank()) - 1;
  auto output            = warp_offsets[warp_index()];
  probe_buckets<GroupSize>(
    warp, num_rows, table, hasher, [&](size_type row, auto candidates, auto const* slots) {
      // Lanes compare one candidate per step so that the matches of a step are written
      // contiguously.
      while (warp.any(candidates != 0)) {
        auto match = cuda::std::pair<size_type, size_type>{CUDF_SIZE_TYPE_SENTINEL, 0};
        if (candidates != 0) {
          match = equal(row, slots[__ffs(candidates) - 1]);
          candidates &= candidates - 1;
        }
        auto const found       = match.first != CUDF_SIZE_TYPE_SENTINEL;
        auto const found_lanes = warp.ballot(found);
        if (found) {
          auto const position     = output + cuda::std::popcount(found_lanes & lower_lanes);
          left_indices[position]  = row;
          batch_indices[position] = match.first;
          row_indices[position]   = match.second;
        }
        output += cuda::std::popcount(found_lanes);
      }
    });
}

using join_indices = std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                               std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                                         std::unique_ptr<rmm::device_uvector<size_type>>>>;

join_indices make_join_indices(std::size_t size,
                               cuda::stream_ref stream,
                               rmm::device_async_resource_ref mr)
{
  auto make_indices = [&] {
    return std::make_unique<rmm::device_uvector<size_type>>(size, stream, mr);
  };
  auto left_indices  = make_indices();
  auto batch_indices = make_indices();
  auto row_indices   = make_indices();
  return std::pair{std::move(left_indices),
                   std::pair{std::move(batch_indices), std::move(row_indices)}};
}

template <int GroupSize, typename Equal, typename Hasher>
join_indices retrieve_matches(size_type num_rows,
                              slot_table_ref table,
                              Equal equal,
                              Hasher hasher,
                              std::optional<std::size_t> output_size,
                              cuda::stream_ref stream,
                              cudf::memory_resources resources)
{
  // Count and retrieval share one grid of resident blocks, so each warp's count is the size of
  // its output range.
  int blocks_per_sm{};
  CUDF_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks_per_sm, retrieve_kernel<GroupSize, Equal, Hasher>, block_size, 0));
  auto const num_blocks = static_cast<cuda::std::uint32_t>(std::max<thread_index_type>(
    1,
    std::min<thread_index_type>(
      cuda::ceil_div(static_cast<thread_index_type>(num_rows), block_size),
      static_cast<thread_index_type>(blocks_per_sm) * cudf::detail::num_multiprocessors())));
  auto const num_warps  = static_cast<thread_index_type>(num_blocks) * warps_per_block;

  auto const temp_mr = resources.get_temporary_mr();
  // One extra element receives the total from the exclusive scan.
  cuda::device_buffer<cuda::std::int64_t> warp_offsets(
    stream, temp_mr, static_cast<std::size_t>(num_warps) + 1, cuda::no_init);
  CUDF_CUDA_TRY(
    cudaMemsetAsync(warp_offsets.data() + num_warps, 0, sizeof(cuda::std::int64_t), stream.get()));
  count_kernel<GroupSize><<<num_blocks, block_size, 0, stream.get()>>>(
    num_rows, table, equal, hasher, warp_offsets.data());
  CUDF_CUDA_TRY(cudaGetLastError());
  {
    std::size_t temp_storage_bytes{};
    CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr,
                                                temp_storage_bytes,
                                                warp_offsets.data(),
                                                warp_offsets.data(),
                                                warp_offsets.size(),
                                                stream.get()));
    cuda::device_buffer<std::byte> temp_storage(stream, temp_mr, temp_storage_bytes, cuda::no_init);
    CUDF_CUDA_TRY(cub::DeviceScan::ExclusiveSum(temp_storage.data(),
                                                temp_storage_bytes,
                                                warp_offsets.data(),
                                                warp_offsets.data(),
                                                warp_offsets.size(),
                                                stream.get()));
  }
  auto const join_size = static_cast<std::size_t>(
    cudf::detail::make_host_vector(
      cudf::device_span<cuda::std::int64_t const>{warp_offsets.data() + num_warps, 1}, stream)
      .front());
  CUDF_EXPECTS(!output_size.has_value() || *output_size == join_size,
               "The provided join output size is incorrect");

  auto result = make_join_indices(join_size, stream, resources.get_output_mr());
  if (join_size > 0) {
    auto& [left_indices, right_indices] = result;
    auto& [batch_indices, row_indices]  = right_indices;
    cudf::prefetch::detail::prefetch(*left_indices, stream);
    cudf::prefetch::detail::prefetch(*batch_indices, stream);
    cudf::prefetch::detail::prefetch(*row_indices, stream);
    retrieve_kernel<GroupSize><<<num_blocks, block_size, 0, stream.get()>>>(num_rows,
                                                                            table,
                                                                            equal,
                                                                            hasher,
                                                                            warp_offsets.data(),
                                                                            left_indices->data(),
                                                                            batch_indices->data(),
                                                                            row_indices->data());
    CUDF_CUDA_TRY(cudaGetLastError());
  }
  return result;
}

}  // namespace

class streaming_hash_join_impl {
 public:
  // Empty copy of the right-side schema, owned so validation never depends on caller lifetimes.
  std::unique_ptr<table> right_schema;
  std::vector<size_type> right_key_indices;
  size_type total_right_rows;
  size_type max_num_batches;
  bool has_nulls;
  null_equality compare_nulls;
  // Assigns batch IDs and global rows together so that batch IDs follow global row order.
  std::mutex insert_mutex;
  std::atomic<size_type> inserted_rows{0};
  std::atomic<size_type> inserted_batches{0};
  bool has_nested_keys{false};

  // Declared before the slot table so the resource outlives allocations that refer to it.
  cudf::memory_resources mr;
  // One slot per right row: the low bits hold the global right row and the high bits a hash
  // fingerprint. For N right rows, bit_width(N) bits encode the row, so the all-ones slot is never
  // a valid entry and marks an empty slot.
  cuda::device_buffer<slot_type> slots;
  slot_type row_mask;

  std::vector<std::shared_ptr<row::equality::preprocessed_table>> preprocessed_right;
  // Views of the inserted partitions, indexed by batch ID so callers can resolve the batch IDs
  // reported by `inner_join()` without tracking concurrent insert ordering themselves.
  std::vector<table_view> right_partitions;
  // First global right row of each batch, indexed by batch ID.
  std::vector<size_type> batch_starts;

  streaming_hash_join_impl(table_view const& schema,
                           std::span<size_type const> key_indices,
                           size_type total_rows,
                           size_type maximum_batches,
                           nullable_join nullable,
                           null_equality nulls_equal,
                           double load_factor,
                           cuda::stream_ref stream,
                           cudf::memory_resources resource)
    // Deep copy a zero-row slice rather than `empty_like`, which drops children such as a
    // dictionary's keys column and would make `have_same_types` reject every partition.
    : right_schema{std::make_unique<table>(
        cudf::slice(schema, {0, 0}).front(), stream, resource.get_output_mr())},
      right_key_indices{key_indices.begin(), key_indices.end()},
      total_right_rows{total_rows},
      max_num_batches{checked_batch_count(maximum_batches)},
      has_nulls{nullable == nullable_join::YES},
      compare_nulls{nulls_equal},
      mr{resource},
      slots{stream,
            mr.get_output_mr(),
            static_cast<std::size_t>(bucket_count(total_rows, load_factor)) * bucket_size,
            cuda::no_init},
      row_mask{(slot_type{1} << cuda::std::bit_width(static_cast<slot_type>(total_rows))) - 1},
      preprocessed_right(static_cast<std::size_t>(max_num_batches)),
      right_partitions(static_cast<std::size_t>(max_num_batches)),
      batch_starts(static_cast<std::size_t>(max_num_batches))
  {
    CUDF_EXPECTS(right_schema->num_columns() > 0,
                 "streaming_hash_join requires at least one right-side column.",
                 std::invalid_argument);
    CUDF_EXPECTS(!right_key_indices.empty(),
                 "streaming_hash_join requires at least one right-side key column.",
                 std::invalid_argument);
    auto const schema_size = right_schema->num_columns();
    for (auto const index : right_key_indices) {
      CUDF_EXPECTS(index >= 0 && index < schema_size,
                   "streaming_hash_join key index is out of range for the provided schema.",
                   std::invalid_argument);
    }
    // The schema exemplar carries nesting, so this is settled once here rather than on insert.
    has_nested_keys =
      cudf::detail::has_nested_columns(right_schema->view().select(right_key_indices));
    CUDF_CUDA_TRY(
      cudaMemsetAsync(slots.data(), 0xff, slots.size() * sizeof(slot_type), stream.get()));
  }

  [[nodiscard]] slot_table_ref slot_table() const
  {
    return {const_cast<slot_type*>(slots.data()),
            static_cast<cuda::std::uint32_t>(slots.size() / bucket_size),
            row_mask};
  }

  void insert(table_view const& right_partition, cuda::stream_ref stream)
  {
    CUDF_EXPECTS(right_partition.num_columns() == right_schema->num_columns(),
                 "streaming_hash_join: inserted partition column count does not match schema.",
                 std::invalid_argument);
    CUDF_EXPECTS(cudf::have_same_types(right_schema->view(), right_partition),
                 "streaming_hash_join: inserted partition schema does not match the schema passed "
                 "to the constructor.",
                 cudf::data_type_error);
    auto const keys = right_partition.select(right_key_indices);

    auto preprocessed = row::equality::preprocessed_table::create(keys, stream, mr.get_output_mr());
    auto const batch_rows = keys.num_rows();
    auto row_bitmask      = [&]() -> std::optional<cuda::device_buffer<cudf::bitmask_type>> {
      if (batch_rows > 0 && compare_nulls == null_equality::UNEQUAL && cudf::has_nulls(keys)) {
        return cudf::detail::bitmask_and(keys, stream, mr.get_temporary_mr()).first;
      }
      return std::nullopt;
    }();

    auto const [batch_id, first_row] = [&] {
      std::lock_guard const lock{insert_mutex};
      auto const rows    = inserted_rows.load(std::memory_order_relaxed);
      auto const batches = inserted_batches.load(std::memory_order_relaxed);
      CUDF_EXPECTS(batch_rows <= total_right_rows - rows,
                   "streaming_hash_join: cumulative inserted rows would exceed total_right_rows.",
                   std::invalid_argument);
      CUDF_EXPECTS(batches < max_num_batches,
                   "streaming_hash_join: number of inserted batches would exceed max_num_batches.",
                   std::invalid_argument);
      batch_starts[batches] = rows;
      inserted_rows.store(rows + batch_rows, std::memory_order_relaxed);
      inserted_batches.store(batches + 1, std::memory_order_relaxed);
      return std::pair{batches, rows};
    }();

    preprocessed_right[batch_id] = std::move(preprocessed);
    right_partitions[batch_id]   = right_partition;

    if (batch_rows == 0) { return; }
    auto const valid_rows = row_bitmask.has_value() ? row_bitmask->data() : nullptr;
    auto const nulls      = nullate::DYNAMIC{has_nulls};
    auto const config     = grid_1d{batch_rows, block_size};
    auto insert_rows      = [&](auto const& hasher) {
      insert_kernel<<<config.num_blocks, config.num_threads_per_block, 0, stream.get()>>>(
        batch_rows, first_row, valid_rows, slot_table(), hasher);
      CUDF_CUDA_TRY(cudaGetLastError());
    };

    if (cudf::detail::is_primitive_row_op_compatible(keys)) {
      insert_rows(row::primitive::row_hasher{nulls, preprocessed_right[batch_id]});
    } else {
      insert_rows(row::hash::row_hasher{preprocessed_right[batch_id]}.device_hasher(nulls));
    }
  }

  template <bool has_nested, bool use_primitive>
  auto probe(table_view const& left,
             std::optional<std::size_t> output_size,
             cuda::stream_ref stream,
             cudf::memory_resources resources) const
  {
    static_assert(!(has_nested && use_primitive),
                  "primitive row comparators are never used for nested keys");
    auto preprocessed_left =
      row::equality::preprocessed_table::create(left, stream, resources.get_temporary_mr());
    auto const num_batches = inserted_batches.load(std::memory_order_relaxed);
    auto const right_comparators =
      std::span{preprocessed_right}.first(static_cast<std::size_t>(num_batches));
    auto comparators = [&] {
      if constexpr (use_primitive) {
        return make_device_primitive_row_comparators(preprocessed_left,
                                                     right_comparators,
                                                     nullate::DYNAMIC{has_nulls},
                                                     compare_nulls,
                                                     stream,
                                                     resources.get_temporary_mr());
      } else {
        return make_device_row_comparators<has_nested>(preprocessed_left,
                                                       right_comparators,
                                                       nullate::DYNAMIC{has_nulls},
                                                       compare_nulls,
                                                       stream,
                                                       resources.get_temporary_mr());
      }
    }();
    auto const d_batch_starts = cudf::detail::make_device_uvector_async(
      cudf::host_span<size_type const>{batch_starts.data(), static_cast<std::size_t>(num_batches)},
      stream,
      resources.get_temporary_mr());
    auto const equality =
      slot_equal{comparators.data(), d_batch_starts.data(), num_batches, row_mask};
    auto const row_hasher = [&] {
      if constexpr (use_primitive) {
        return row::primitive::row_hasher{nullate::DYNAMIC{has_nulls}, preprocessed_left};
      } else {
        return row::hash::row_hasher{preprocessed_left}.device_hasher(nullate::DYNAMIC{has_nulls});
      }
    }();

    return retrieve_matches<probe_group_size<has_nested>>(
      left.num_rows(), slot_table(), equality, row_hasher, output_size, stream, resources);
  }

  [[nodiscard]] table_view get_partition(size_type batch_id) const
  {
    CUDF_EXPECTS(batch_id >= 0 and batch_id < inserted_batches.load(std::memory_order_relaxed),
                 "streaming_hash_join: no partition has been inserted with this batch ID.",
                 std::out_of_range);
    return right_partitions[batch_id];
  }

  join_indices inner_join(table_view const& left,
                          std::optional<std::size_t> output_size,
                          cuda::stream_ref stream,
                          cudf::memory_resources resources) const
  {
    CUDF_EXPECTS(inserted_batches.load(std::memory_order_relaxed) > 0,
                 "streaming_hash_join: inner_join called before any insert().",
                 std::logic_error);
    validate_hash_join_probe(right_schema->view().select(right_key_indices), left, has_nulls);

    if (left.num_rows() == 0 || inserted_rows.load(std::memory_order_relaxed) == 0) {
      CUDF_EXPECTS(!output_size.has_value() || *output_size == 0,
                   "The provided join output size is incorrect");
      return make_join_indices(0, stream, resources.get_output_mr());
    }
    if (cudf::detail::is_primitive_row_op_compatible(left)) {
      return probe<false, true>(left, output_size, stream, resources);
    }
    return has_nested_keys ? probe<true, false>(left, output_size, stream, resources)
                           : probe<false, false>(left, output_size, stream, resources);
  }
};

}  // namespace cudf::detail

namespace cudf {

streaming_hash_join::streaming_hash_join(table_view const& right_schema,
                                         std::span<size_type const> right_key_indices,
                                         size_type total_right_rows,
                                         size_type max_num_batches,
                                         nullable_join has_nulls,
                                         null_equality compare_nulls,
                                         double load_factor,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr)
  : _impl{std::make_unique<cudf::detail::streaming_hash_join_impl>(right_schema,
                                                                   right_key_indices,
                                                                   total_right_rows,
                                                                   max_num_batches,
                                                                   has_nulls,
                                                                   compare_nulls,
                                                                   load_factor,
                                                                   stream,
                                                                   mr)}
{
}

streaming_hash_join::~streaming_hash_join()                                         = default;
streaming_hash_join::streaming_hash_join(streaming_hash_join&&) noexcept            = default;
streaming_hash_join& streaming_hash_join::operator=(streaming_hash_join&&) noexcept = default;

void streaming_hash_join::insert(table_view const& right_partition, cuda::stream_ref stream)
{
  CUDF_FUNC_RANGE();
  _impl->insert(right_partition, stream);
}

table_view streaming_hash_join::get_partition(size_type batch_id) const
{
  return _impl->get_partition(batch_id);
}

std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
          std::pair<std::unique_ptr<rmm::device_uvector<size_type>>,
                    std::unique_ptr<rmm::device_uvector<size_type>>>>
streaming_hash_join::inner_join(table_view const& left,
                                std::optional<std::size_t> output_size,
                                cuda::stream_ref stream,
                                cudf::memory_resources mr) const
{
  CUDF_FUNC_RANGE();
  return _impl->inner_join(left, output_size, stream, mr);
}

}  // namespace cudf

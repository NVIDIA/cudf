/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/jaccard.hpp>

#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cuco/bucket_storage.cuh>
#include <cuco/hash_functions.cuh>
#include <cuco/static_set_ref.cuh>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/std/algorithm>
#include <cuda/std/bit>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/stream>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

constexpr cudf::thread_index_type block_size       = 256;
constexpr cudf::thread_index_type bytes_per_thread = 4;

using hash_value_type = uint32_t;

// Number of bits in a hash value
constexpr int hash_bits = cuda::std::numeric_limits<hash_value_type>::digits;

// Hash value 0 is reserved as the empty-slot sentinel for the sets
constexpr hash_value_type empty_hash = 0;

// The hash values are already well distributed so they are used directly as the probe hash
using probe_type       = cuco::linear_probing<1, cuco::identity_hash<hash_value_type>>;
using storage_ref_type = cuco::bucket_storage_ref<hash_value_type, 1, cuco::extent<int32_t>>;
using set_ref_type     = cuco::static_set_ref<hash_value_type,
                                              cuda::thread_scope_block,
                                              cuda::std::equal_to<hash_value_type>,
                                              probe_type,
                                              storage_ref_type,
                                              cuco::op::insert_tag,
                                              cuco::op::contains_tag>;

// Shared-memory slots for each of the 2 sets used per row by the warp-per-row kernel
constexpr int32_t warp_slots_per_set = 768;  // 8 warps * 2 sets * 768 * 4 bytes = 48KB
// Shared-memory slots for each of the 2 sets used per row partition by the partition kernel
constexpr int32_t block_slots_per_set = 2048;  // 2 sets * 2048 * 4 bytes = 16KB
// The sets are sized to keep the load factor at or below 0.5
constexpr int64_t warp_set_capacity  = warp_slots_per_set / 2;
constexpr int64_t block_set_capacity = block_slots_per_set / 2;
// Maximum number of partitions per row; rows needing more rely on sub-partitioning
constexpr int64_t max_partitions = 4096;

/**
 * @brief Counts the number of substrings in each row of the given strings column
 *
 * Each warp processes a single string.
 * Formula is `count = max(1, str.length() - width + 1)`
 * If a string has less than width characters (but not empty), the count is 1
 * since the entire string is still hashed.
 *
 * @param d_strings Input column of strings
 * @param width Substring size in characters
 * @param d_counts Output number of substring per row of input
 */
CUDF_KERNEL void count_substrings_kernel(cudf::column_device_view const d_strings,
                                         cudf::size_type width,
                                         cuda::std::span<int64_t> d_counts)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  if (idx >= (static_cast<cudf::thread_index_type>(d_strings.size()) * cudf::detail::warp_size)) {
    return;
  }

  auto const str_idx = static_cast<cudf::size_type>(idx / cudf::detail::warp_size);
  if (d_strings.is_null(str_idx)) {
    d_counts[str_idx] = 0;
    return;
  }

  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) {
    d_counts[str_idx] = 0;
    return;
  }

  using warp_reduce = cub::WarpReduce<cudf::size_type>;
  __shared__ typename warp_reduce::TempStorage temp_storage;

  auto const end        = d_str.data() + d_str.size_bytes();
  auto const lane_idx   = idx % cudf::detail::warp_size;
  cudf::size_type count = 0;
  for (auto itr = d_str.data() + (lane_idx * bytes_per_thread); itr < end;
       itr += cudf::detail::warp_size * bytes_per_thread) {
    for (auto s = itr; (s < (itr + bytes_per_thread)) && (s < end); ++s) {
      count += static_cast<cudf::size_type>(cudf::strings::detail::is_begin_utf8_char(*s));
    }
  }
  auto const char_count = warp_reduce(temp_storage).Sum(count);
  if (lane_idx == 0) { d_counts[str_idx] = cuda::std::max(1, char_count - width + 1); }
}

/**
 * @brief Kernel to hash the substrings for each input row
 *
 * Each warp processes a single string.
 * Substrings of string "hello world" with width=4 produce:
 *   "hell", "ello", "llo ", "lo w", "o wo", " wor", "worl", "orld"
 * Each of these substrings is hashed and the hash stored in d_results.
 * A hash value of 0 is stored as 1 since 0 is reserved as the empty-slot sentinel.
 *
 * @param d_strings Input column of strings
 * @param width Substring size in characters
 * @param d_output_offsets Offsets into d_results
 * @param d_results Hash values for each substring
 */
CUDF_KERNEL void substring_hash_kernel(cudf::column_device_view const d_strings,
                                       cudf::size_type width,
                                       cuda::std::span<int64_t const> d_output_offsets,
                                       cuda::std::span<hash_value_type> d_results)
{
  auto const idx     = cudf::detail::grid_1d::global_thread_id();
  auto const str_idx = idx / cudf::detail::warp_size;
  if (str_idx >= d_strings.size() or d_strings.is_null(str_idx)) { return; }
  auto const d_str = d_strings.element<cudf::string_view>(str_idx);
  if (d_str.empty()) { return; }

  namespace cg        = cooperative_groups;
  auto const warp     = cg::tiled_partition<cudf::detail::warp_size>(cg::this_thread_block());
  auto const lane_idx = warp.thread_rank();
  auto const hasher   = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>{0};
  auto const begin    = d_str.data();
  auto const end      = begin + d_str.size_bytes();

  auto d_hashes = d_results.begin() + d_output_offsets[str_idx];
  for (auto itr = begin; itr < end; itr += cudf::detail::warp_size) {
    auto const ptr       = itr + lane_idx;
    auto valid           = false;
    hash_value_type hash = 0;
    if (ptr < end && cudf::strings::detail::is_begin_utf8_char(*ptr)) {
      // resolve substring
      auto const sub_str =
        cudf::string_view(ptr, static_cast<cudf::size_type>(cuda::std::distance(ptr, end)));
      auto const [bytes, left] = cudf::strings::detail::bytes_to_character_position(sub_str, width);
      // hash only if we have the full width of characters or this is the beginning of the string
      if ((left == 0) || (ptr == begin)) {
        hash  = cuda::std::max(hasher(cudf::string_view(ptr, bytes)), hash_value_type{1});
        valid = true;
      }
    }
    // compact the valid hash values for this warp into d_hashes
    auto const mask = warp.ballot(valid);
    if (valid) { d_hashes[cuda::std::popcount(mask & ((1u << lane_idx) - 1u))] = hash; }
    d_hashes += cuda::std::popcount(mask);
  }
}

/**
 * @brief Create hashes for each substring
 *
 * @param d_strings Input strings column to hash
 * @param width Substring width in characters
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations and the returned buffers
 * @return The hash values and offsets to each row
 */
std::pair<cuda::device_buffer<hash_value_type>, cuda::device_buffer<int64_t>> hash_substrings(
  cudf::column_device_view const& d_strings,
  cudf::size_type width,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  auto const output_mr = mr.get_output_mr();

  // count substrings
  auto offsets = cuda::device_buffer<int64_t>(
    stream, output_mr, static_cast<std::size_t>(d_strings.size()) + 1, cuda::no_init);
  auto const num_blocks = cudf::util::div_rounding_up_safe(
    static_cast<cudf::thread_index_type>(d_strings.size()) * cudf::detail::warp_size, block_size);
  count_substrings_kernel<<<num_blocks, block_size, 0, stream.get()>>>(d_strings, width, offsets);
  CUDF_CUDA_TRY(cudaGetLastError());
  auto const total_hashes = cudf::detail::sizes_to_offsets(
    offsets.begin(), offsets.end(), offsets.begin(), 0, stream, mr.as_temporary());

  // hash substrings
  auto hashes =
    cuda::device_buffer<hash_value_type>(stream, output_mr, total_hashes, cuda::no_init);
  substring_hash_kernel<<<num_blocks, block_size, 0, stream.get()>>>(
    d_strings, width, offsets, hashes);
  CUDF_CUDA_TRY(cudaGetLastError());

  return std::make_pair(std::move(hashes), std::move(offsets));
}

/**
 * @brief Retrieve the row data (span) for the given column/row-index
 *
 * @param values Flat vector of all values
 * @param offsets Offsets identifying rows within values
 * @param row_idx Row index to retrieve
 * @return A device-span of the row values
 */
__device__ cuda::std::span<hash_value_type const> get_row(
  cuda::std::span<hash_value_type const> values,
  cuda::std::span<int64_t const> offsets,
  cudf::size_type row_idx)
{
  auto const offset = offsets[row_idx];
  return values.subspan(offset, offsets[row_idx + 1] - offset);
}

/**
 * @brief Creates a set over the given shared-memory slots
 *
 * @param slots Shared-memory slots for the set; must be initialized to `empty_hash`
 * @param num_slots Number of slots
 * @return A set reference over the slots
 */
__device__ set_ref_type make_set(hash_value_type* slots, int32_t num_slots)
{
  return set_ref_type{cuco::empty_key<hash_value_type>{empty_hash},
                      {},
                      {},
                      {},
                      storage_ref_type{cuco::extent<int32_t>{num_slots}, slots}};
}

/**
 * @brief Returns the partition for the given hash value
 *
 * The partition is computed from the high bits of the hash since
 * the set slot is computed from the low bits.
 * Scaling the hash by the number of partitions and keeping only the high bits
 * maps the hash range evenly onto `[0, partitions)`.
 *
 * @param hash Hash value to partition
 * @param partitions Number of partitions
 * @return Partition index in `[0, partitions)`
 */
__device__ uint32_t partition_of(hash_value_type hash, uint64_t partitions)
{
  return static_cast<uint32_t>((static_cast<uint64_t>(hash) * partitions) >> hash_bits);
}

/**
 * @brief Inserts the hashes into set1 and set2 and counts the results
 *
 * The values in hashes1 are inserted into set1 and the values in hashes2
 * are inserted into set2. Each value successfully inserted into set2 is
 * checked against set1 to count the intersection.
 *
 * Only values for which `filter(value)==true` are processed.
 * The group must be synchronized before calling this function so the sets are initialized.
 *
 * @tparam Group Cooperative group type processing the hashes
 * @tparam Filter Predicate type for selecting the hashes to process
 * @param group Cooperative group processing the hashes
 * @param hashes1 Values to insert into set1
 * @param hashes2 Values to insert into set2
 * @param set1 Set for the first values
 * @param set2 Set for the second values
 * @param filter Selects the values to process
 * @return The unique counts of set1 and set2 and the intersection count for this thread
 */
template <typename Group, typename Filter>
__device__ cuda::std::tuple<cudf::size_type, cudf::size_type, cudf::size_type> count_hashes(
  Group const& group,
  cuda::std::span<hash_value_type const> hashes1,
  cuda::std::span<hash_value_type const> hashes2,
  set_ref_type& set1,
  set_ref_type& set2,
  Filter filter)
{
  cudf::size_type count1 = 0;
  for (auto i = static_cast<int64_t>(group.thread_rank()); i < static_cast<int64_t>(hashes1.size());
       i += group.size()) {
    auto const hash = hashes1[i];
    if (filter(hash)) { count1 += set1.insert(hash); }
  }
  group.sync();  // set1 must be complete before it is probed
  cudf::size_type count2     = 0;
  cudf::size_type intersects = 0;
  for (auto i = static_cast<int64_t>(group.thread_rank()); i < static_cast<int64_t>(hashes2.size());
       i += group.size()) {
    auto const hash = hashes2[i];
    if (filter(hash) && set2.insert(hash)) {
      ++count2;
      intersects += set1.contains(hash);
    }
  }
  return {count1, count2, intersects};
}

/**
 * @brief Counts the unique and intersect values for rows that fit within a warp's sets
 *
 * Each warp processes a single row using 2 shared-memory sets.
 * Rows too large for the sets are skipped and processed by `count_partitions_kernel`.
 *
 * @param d_hashes1 Hash values for the first column
 * @param d_offsets1 Offsets to each row's values in d_hashes1
 * @param d_hashes2 Hash values for the second column
 * @param d_offsets2 Offsets to each row's values in d_hashes2
 * @param d_counts1 Output unique counts for the first column
 * @param d_counts2 Output unique counts for the second column
 * @param d_intersects Output intersect counts
 */
CUDF_KERNEL void count_rows_kernel(cuda::std::span<hash_value_type const> d_hashes1,
                                   cuda::std::span<int64_t const> d_offsets1,
                                   cuda::std::span<hash_value_type const> d_hashes2,
                                   cuda::std::span<int64_t const> d_offsets2,
                                   cuda::std::span<cudf::size_type> d_counts1,
                                   cuda::std::span<cudf::size_type> d_counts2,
                                   cuda::std::span<cudf::size_type> d_intersects)
{
  auto const idx = cudf::detail::grid_1d::global_thread_id();
  auto const row = static_cast<cudf::size_type>(idx / cudf::detail::warp_size);
  if (row >= static_cast<cudf::size_type>(d_counts1.size())) { return; }

  auto const hashes1 = get_row(d_hashes1, d_offsets1, row);
  auto const hashes2 = get_row(d_hashes2, d_offsets2, row);
  if (static_cast<int64_t>(cuda::std::max(hashes1.size(), hashes2.size())) > warp_set_capacity) {
    return;
  }

  namespace cg     = cooperative_groups;
  auto const block = cg::this_thread_block();
  auto const warp  = cg::tiled_partition<cudf::detail::warp_size>(block);

  constexpr auto warps_per_block = block_size / cudf::detail::warp_size;
  __shared__ hash_value_type slots[warps_per_block][2 * warp_slots_per_set];
  auto const warp_slots = slots[threadIdx.x / cudf::detail::warp_size];
  for (auto i = static_cast<int32_t>(warp.thread_rank()); i < 2 * warp_slots_per_set;
       i += cudf::detail::warp_size) {
    warp_slots[i] = empty_hash;
  }
  auto set1 = make_set(warp_slots, warp_slots_per_set);
  auto set2 = make_set(warp_slots + warp_slots_per_set, warp_slots_per_set);
  warp.sync();

  auto const [count1, count2, intersects] =
    count_hashes(warp, hashes1, hashes2, set1, set2, [](auto) { return true; });

  auto const sum1           = cg::reduce(warp, count1, cg::plus<cudf::size_type>());
  auto const sum2           = cg::reduce(warp, count2, cg::plus<cudf::size_type>());
  auto const sum_intersects = cg::reduce(warp, intersects, cg::plus<cudf::size_type>());
  if (warp.thread_rank() == 0) {
    d_counts1[row]    = sum1;
    d_counts2[row]    = sum2;
    d_intersects[row] = sum_intersects;
  }
}

/**
 * @brief Scatters each row's hashes into contiguous partition buckets
 *
 * Each block processes a single row. Rows with fewer than 2 partitions are skipped.
 * The start of each bucket is written to `d_buckets[d_work_offsets[row] + partition]`.
 *
 * @param d_hashes Hash values to partition
 * @param d_offsets Offsets to each row's values in d_hashes
 * @param d_work_offsets Offsets to each row's partitions
 * @param d_output Partitioned hash values
 * @param d_buckets Offsets to each partition's values in d_output
 */
CUDF_KERNEL void partition_hashes_kernel(cuda::std::span<hash_value_type const> d_hashes,
                                         cuda::std::span<int64_t const> d_offsets,
                                         cuda::std::span<int64_t const> d_work_offsets,
                                         cuda::std::span<hash_value_type> d_output,
                                         cuda::std::span<int64_t> d_buckets)
{
  auto const row        = static_cast<cudf::size_type>(blockIdx.x);
  auto const partitions = static_cast<int32_t>(d_work_offsets[row + 1] - d_work_offsets[row]);
  if (partitions < 2) { return; }

  __shared__ int32_t bins[max_partitions];
  for (auto i = static_cast<int32_t>(threadIdx.x); i < partitions; i += block_size) {
    bins[i] = 0;
  }
  __syncthreads();

  // build a histogram of the partitions
  auto const offset = d_offsets[row];
  auto const hashes = get_row(d_hashes, d_offsets, row);
  auto const size   = static_cast<int64_t>(hashes.size());
  for (auto i = static_cast<int64_t>(threadIdx.x); i < size; i += block_size) {
    cuda::atomic_ref<int32_t, cuda::thread_scope_block>{bins[partition_of(hashes[i], partitions)]}
      .fetch_add(1, cuda::memory_order_relaxed);
  }
  __syncthreads();

  // exclusive scan of the histogram in block-sized chunks to get each bucket's start;
  // cub is used for block-wide collectives since a block-sized cooperative_groups tile
  // requires cg::block_tile_memory scratch on architectures older than sm80
  using block_scan = cub::BlockScan<int32_t, block_size>;
  __shared__ typename block_scan::TempStorage temp_storage;
  int32_t carry = 0;
  for (int32_t base = 0; base < partitions; base += block_size) {
    auto const i     = base + static_cast<int32_t>(threadIdx.x);
    auto const value = i < partitions ? bins[i] : 0;
    int32_t scanned  = 0;
    int32_t total    = 0;
    block_scan(temp_storage).ExclusiveSum(value, scanned, total);
    if (i < partitions) {
      bins[i]                            = carry + scanned;
      d_buckets[d_work_offsets[row] + i] = offset + carry + scanned;
    }
    carry += total;
    __syncthreads();
  }

  // scatter the values into their buckets
  auto output = d_output.begin() + offset;
  for (auto i = static_cast<int64_t>(threadIdx.x); i < size; i += block_size) {
    auto const hash = hashes[i];
    auto const pos =
      cuda::atomic_ref<int32_t, cuda::thread_scope_block>{bins[partition_of(hash, partitions)]}
        .fetch_add(1, cuda::memory_order_relaxed);
    output[pos] = hash;
  }
}

/**
 * @brief Counts the unique and intersect values for one partition of a row
 *
 * Each block processes a single (row, partition) work item using 2 shared-memory sets.
 * The counts are accumulated atomically into the output for each row.
 *
 * In the unlikely event a partition overflows a set, the partition is processed again
 * in multiple passes using sub-partitions of its hash range.
 *
 * @param d_hashes1 Hash values for the first column
 * @param d_offsets1 Offsets to each row's values in d_hashes1
 * @param d_parted1 Partitioned hash values for the first column
 * @param d_buckets1 Offsets to each partition's values in d_parted1
 * @param d_hashes2 Hash values for the second column
 * @param d_offsets2 Offsets to each row's values in d_hashes2
 * @param d_parted2 Partitioned hash values for the second column
 * @param d_buckets2 Offsets to each partition's values in d_parted2
 * @param d_work_offsets Offsets to each row's partitions
 * @param d_counts1 Output unique counts for the first column
 * @param d_counts2 Output unique counts for the second column
 * @param d_intersects Output intersect counts
 */
CUDF_KERNEL void count_partitions_kernel(cuda::std::span<hash_value_type const> d_hashes1,
                                         cuda::std::span<int64_t const> d_offsets1,
                                         cuda::std::span<hash_value_type const> d_parted1,
                                         cuda::std::span<int64_t const> d_buckets1,
                                         cuda::std::span<hash_value_type const> d_hashes2,
                                         cuda::std::span<int64_t const> d_offsets2,
                                         cuda::std::span<hash_value_type const> d_parted2,
                                         cuda::std::span<int64_t const> d_buckets2,
                                         cuda::std::span<int64_t const> d_work_offsets,
                                         cuda::std::span<cudf::size_type> d_counts1,
                                         cuda::std::span<cudf::size_type> d_counts2,
                                         cuda::std::span<cudf::size_type> d_intersects)
{
  auto const work = static_cast<int64_t>(blockIdx.x);
  // rows with no work items have empty ranges so upper_bound finds the row owning this item
  auto const row = static_cast<cudf::size_type>(
    cuda::std::distance(
      d_work_offsets.begin(),
      cuda::std::upper_bound(d_work_offsets.begin(), d_work_offsets.end(), work)) -
    1);
  auto const partitions = static_cast<uint64_t>(d_work_offsets[row + 1] - d_work_offsets[row]);
  auto const partition  = static_cast<uint64_t>(work - d_work_offsets[row]);

  // locate the values for this partition
  auto get_bucket = [&](cuda::std::span<hash_value_type const> d_hashes,
                        cuda::std::span<int64_t const> d_offsets,
                        cuda::std::span<hash_value_type const> d_parted,
                        cuda::std::span<int64_t const> d_buckets) {
    if (partitions == 1) { return get_row(d_hashes, d_offsets, row); }
    auto const begin = d_buckets[work];
    auto const end   = (partition + 1 < partitions) ? d_buckets[work + 1] : d_offsets[row + 1];
    return cuda::std::span<hash_value_type const>(d_parted.begin() + begin, d_parted.begin() + end);
  };
  auto const hashes1 = get_bucket(d_hashes1, d_offsets1, d_parted1, d_buckets1);
  auto const hashes2 = get_bucket(d_hashes2, d_offsets2, d_parted2, d_buckets2);

  __shared__ hash_value_type slots[2 * block_slots_per_set];
  auto set1 = make_set(slots, block_slots_per_set);
  auto set2 = make_set(slots + block_slots_per_set, block_slots_per_set);

  // cub is used for block-wide collectives since a block-sized cooperative_groups tile
  // requires cg::block_tile_memory scratch on architectures older than sm80
  using block_reduce = cub::BlockReduce<cudf::size_type, block_size>;
  __shared__ typename block_reduce::TempStorage temp_storage;
  __shared__ cudf::size_type totals[3];

  // the block group is needed to call count_hashes() which is shared with count_rows_kernel
  auto const block = cooperative_groups::this_thread_block();

  uint64_t sub_partitions    = 1;
  cudf::size_type count1     = 0;
  cudf::size_type count2     = 0;
  cudf::size_type intersects = 0;
  for (uint64_t sub = 0; sub < sub_partitions;) {
    for (auto i = static_cast<int32_t>(threadIdx.x); i < 2 * block_slots_per_set; i += block_size) {
      slots[i] = empty_hash;
    }
    block.sync();

    auto const total_partitions = partitions * sub_partitions;
    auto const target           = partition * sub_partitions + sub;
    auto filter                 = [total_partitions, target](hash_value_type hash) {
      return total_partitions == 1 || partition_of(hash, total_partitions) == target;
    };
    auto const [c1, c2, c_intersects] = count_hashes(block, hashes1, hashes2, set1, set2, filter);

    auto const sum1 = block_reduce(temp_storage).Sum(c1);
    block.sync();
    auto const sum2 = block_reduce(temp_storage).Sum(c2);
    block.sync();
    auto const sum_intersects = block_reduce(temp_storage).Sum(c_intersects);
    if (threadIdx.x == 0) {
      totals[0] = sum1;
      totals[1] = sum2;
      totals[2] = sum_intersects;
    }
    block.sync();
    auto const total1           = totals[0];
    auto const total2           = totals[1];
    auto const total_intersects = totals[2];
    block.sync();

    if (total1 >= block_slots_per_set || total2 >= block_slots_per_set) {
      // a set is full so some inserts may have failed: restart with more sub-partitions
      sub_partitions *= 2;
      sub        = 0;
      count1     = 0;
      count2     = 0;
      intersects = 0;
      continue;
    }
    count1 += total1;
    count2 += total2;
    intersects += total_intersects;
    ++sub;
  }

  if (threadIdx.x == 0) {
    using atomic_ref = cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device>;
    atomic_ref{d_counts1[row]}.fetch_add(count1, cuda::memory_order_relaxed);
    atomic_ref{d_counts2[row]}.fetch_add(count2, cuda::memory_order_relaxed);
    atomic_ref{d_intersects[row]}.fetch_add(intersects, cuda::memory_order_relaxed);
  }
}

/**
 * @brief Partition the values of rows too large for count_rows_kernel and count them
 *
 * @param hashes1 Hash values for the first column
 * @param offsets1 Offsets to each row's values in hashes1
 * @param hashes2 Hash values for the second column
 * @param offsets2 Offsets to each row's values in hashes2
 * @param d_counts1 Output unique counts for the first column
 * @param d_counts2 Output unique counts for the second column
 * @param d_intersects Output intersect counts
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations
 */
void count_partitions(cuda::std::span<hash_value_type const> hashes1,
                      cuda::std::span<int64_t const> offsets1,
                      cuda::std::span<hash_value_type const> hashes2,
                      cuda::std::span<int64_t const> offsets2,
                      cuda::std::span<cudf::size_type> d_counts1,
                      cuda::std::span<cudf::size_type> d_counts2,
                      cuda::std::span<cudf::size_type> d_intersects,
                      cuda::stream_ref stream,
                      cudf::memory_resources mr)
{
  auto const rows    = static_cast<cudf::size_type>(d_counts1.size());
  auto const temp_mr = mr.get_temporary_mr();

  // compute the number of partitions for each row: rows handled by count_rows_kernel get 0
  auto work_offsets = cuda::device_buffer<int64_t>(
    stream, temp_mr, static_cast<std::size_t>(rows) + 1, cuda::no_init);
  thrust::transform(
    rmm::exec_policy_nosync(stream, temp_mr),
    cuda::counting_iterator<cudf::size_type>{0},
    cuda::counting_iterator<cudf::size_type>{rows},
    work_offsets.begin(),
    [offsets1, offsets2] __device__(auto row) -> int64_t {
      auto const size =
        cuda::std::max(offsets1[row + 1] - offsets1[row], offsets2[row + 1] - offsets2[row]);
      if (size <= warp_set_capacity) { return 0; }
      return cuda::std::min(cudf::util::div_rounding_up_safe(size, int64_t{block_set_capacity}),
                            int64_t{max_partitions});
    });
  auto const total_work = cudf::detail::sizes_to_offsets(
    work_offsets.begin(), work_offsets.end(), work_offsets.begin(), 0, stream, mr.as_temporary());
  if (total_work == 0) { return; }

  // scatter the hashes of each row into their partitions
  auto parted1 =
    cuda::device_buffer<hash_value_type>(stream, temp_mr, hashes1.size(), cuda::no_init);
  auto parted2 =
    cuda::device_buffer<hash_value_type>(stream, temp_mr, hashes2.size(), cuda::no_init);
  auto buckets1 = cuda::device_buffer<int64_t>(stream, temp_mr, total_work, cuda::no_init);
  auto buckets2 = cuda::device_buffer<int64_t>(stream, temp_mr, total_work, cuda::no_init);
  partition_hashes_kernel<<<rows, block_size, 0, stream.get()>>>(
    hashes1, offsets1, work_offsets, parted1, buckets1);
  CUDF_CUDA_TRY(cudaGetLastError());
  partition_hashes_kernel<<<rows, block_size, 0, stream.get()>>>(
    hashes2, offsets2, work_offsets, parted2, buckets2);
  CUDF_CUDA_TRY(cudaGetLastError());

  // count each (row, partition) work item
  count_partitions_kernel<<<total_work, block_size, 0, stream.get()>>>(hashes1,
                                                                       offsets1,
                                                                       parted1,
                                                                       buckets1,
                                                                       hashes2,
                                                                       offsets2,
                                                                       parted2,
                                                                       buckets2,
                                                                       work_offsets,
                                                                       d_counts1,
                                                                       d_counts2,
                                                                       d_intersects);
  CUDF_CUDA_TRY(cudaGetLastError());
}

/**
 * @brief Compute the jaccard distance for each row
 *
 * Formula is J = |A ∩ B| / |A ∪ B|
 *              = |A ∩ B| / (|A| + |B| - |A ∩ B|)
 *
 * where |A ∩ B| is number of common values between A and B
 * and |x| is the number of unique values in x.
 */
struct jaccard_fn {
  cuda::std::span<cudf::size_type const> d_uniques1;
  cuda::std::span<cudf::size_type const> d_uniques2;
  cuda::std::span<cudf::size_type const> d_intersects;

  __device__ float operator()(cudf::size_type idx) const
  {
    auto const count1     = d_uniques1[idx];
    auto const count2     = d_uniques2[idx];
    auto const intersects = d_intersects[idx];
    // the intersect values are in both sets so a union count
    // would need to subtract the intersect count from one set
    // (see formula in comment above)
    auto const unions = count1 + count2 - intersects;
    return unions ? (static_cast<float>(intersects) / static_cast<float>(unions)) : 0.f;
  }
};

}  // namespace

std::unique_ptr<cudf::column> jaccard_index(cudf::strings_column_view const& input1,
                                            cudf::strings_column_view const& input2,
                                            cudf::size_type width,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  CUDF_EXPECTS(
    input1.size() == input2.size(), "input columns must be the same size", std::invalid_argument);
  CUDF_EXPECTS(width >= 2,
               "Parameter width should be an integer value of 2 or greater",
               std::invalid_argument);

  constexpr auto output_type = cudf::data_type{cudf::type_id::FLOAT32};
  if (input1.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const rows           = input1.size();
  auto const temp_mr        = mr.get_temporary_mr();
  auto const temp_resources = mr.as_temporary();

  // unique counts for each column and the intersect counts
  auto counts             = cuda::make_buffer(stream, temp_mr, 3L * rows, cudf::size_type{0});
  auto const d_counts     = cuda::std::span<cudf::size_type>(counts);
  auto const d_counts1    = d_counts.subspan(0, rows);
  auto const d_counts2    = d_counts.subspan(rows, rows);
  auto const d_intersects = d_counts.subspan(2L * rows, rows);

  {
    // build hashes of the substrings
    auto const d_strings1 = cudf::column_device_view::create(input1.parent(), stream, temp_mr);
    auto const d_strings2 = cudf::column_device_view::create(input2.parent(), stream, temp_mr);
    auto const [hash1, offsets1] = hash_substrings(*d_strings1, width, stream, temp_resources);
    auto const [hash2, offsets2] = hash_substrings(*d_strings2, width, stream, temp_resources);

    // count rows small enough to be processed by a single warp
    auto const num_blocks = cudf::util::div_rounding_up_safe(
      static_cast<cudf::thread_index_type>(rows) * cudf::detail::warp_size, block_size);
    count_rows_kernel<<<num_blocks, block_size, 0, stream.get()>>>(
      hash1, offsets1, hash2, offsets2, d_counts1, d_counts2, d_intersects);
    CUDF_CUDA_TRY(cudaGetLastError());

    // count the remaining rows by partitioning their values
    count_partitions(
      hash1, offsets1, hash2, offsets2, d_counts1, d_counts2, d_intersects, stream, temp_resources);
  }

  auto results = cudf::make_numeric_column(
    output_type, rows, cudf::mask_state::UNALLOCATED, stream, mr.get_output_mr());
  auto d_results = results->mutable_view().data<float>();

  // compute the jaccard using the unique counts and the intersect counts
  thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                    cuda::counting_iterator<cudf::size_type>{0},
                    cuda::counting_iterator<cudf::size_type>{rows},
                    d_results,
                    jaccard_fn{d_counts1, d_counts2, d_intersects});

  if (input1.null_count() || input2.null_count()) {
    auto [null_mask, null_count] =
      cudf::detail::bitmask_and(cudf::table_view({input1.parent(), input2.parent()}), stream, mr);
    results->set_null_mask(std::move(null_mask), null_count);
  }

  return results;
}

}  // namespace detail

std::unique_ptr<cudf::column> jaccard_index(cudf::strings_column_view const& input1,
                                            cudf::strings_column_view const& input2,
                                            cudf::size_type width,
                                            cuda::stream_ref stream,
                                            cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::jaccard_index(input1, input2, width, stream, mr);
}

}  // namespace nvtext

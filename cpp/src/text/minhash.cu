/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/device_scalar.hpp>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/offsets_iterator_factory.cuh>
#include <cudf/detail/sequence.hpp>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/detail/utilities/grid_1d.cuh>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/hashing/detail/hashing.hpp>
#include <cudf/hashing/detail/murmurhash3_x64_128.cuh>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/lists/list_device_view.cuh>
#include <cudf/lists/lists_column_device_view.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/minhash.hpp>

#include <rmm/exec_policy.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/transform.h>

namespace nvtext {
namespace detail {
namespace {

constexpr cudf::thread_index_type block_size = 256;
// for potentially tuning minhash_ngrams_kernel independently from block_size
constexpr cudf::thread_index_type tile_size = block_size;

// Number of a/b parameter values to process per thread.
// The parameters and intermediate values are stored in registers and therefore limits this count.
template <typename hash_value_type>
constexpr cuda::std::size_t params_per_thread = sizeof(hash_value_type) == 4 ? 32 : 16;
// Prefetching the next hash value helps hide the load latency for the 64-bit hash type only.
// The extra registers needed reduce the performance of the 32-bit hash type.
template <typename hash_value_type>
constexpr bool prefetch_hash = sizeof(hash_value_type) == 8;

// Separate kernels are used by minhash_ngrams to process rows above and below
// this value (in number of list elements).
constexpr cudf::size_type wide_row_threshold = 1 << 18;  // 256K
// The number of blocks per row for the above-threshold kernel processing.
constexpr cudf::size_type blocks_per_row = 64;
// The above values were determined using the redpajama and books_sample datasets

// Number of character positions hashed by each block per iteration in minhash_fused_kernel
constexpr cudf::size_type seed_tile_bytes = block_size * 4;
// Maximum number of character positions of a row processed by a single block.
// Rows larger than this are processed by multiple blocks (must be a multiple of seed_tile_bytes).
constexpr cudf::size_type section_bytes = 16 * 1024;
// Number of a/b parameter values processed by each thread in minhash_fused_kernel.
// The parameters and minimum values are held in registers for the life of the kernel
// and so this value limits the occupancy.
template <typename hash_value_type>
constexpr cudf::size_type fused_params_per_thread = sizeof(hash_value_type) == 4 ? 16 : 8;
// Minimum blocks per SM for minhash_fused_kernel (limits the registers per thread)
template <typename hash_value_type>
constexpr int fused_min_blocks = sizeof(hash_value_type) == 4 ? 3 : 1;
// The above values were determined using the nvbench benchmark on an RTX A6000
// Maximum number of extra bytes staged in shared memory beyond each seed tile
// to accommodate the substrings that start near the end of the tile
constexpr cudf::size_type max_halo_bytes = 4096;

/**
 * @brief Returns the number of bytes needed beyond a seed tile to include
 * all the bytes accessed when hashing `width` characters
 *
 * Each character is at most 4 bytes and the 4-byte reads used for counting characters
 * may extend a few bytes beyond the last byte. The result is capped at max_halo_bytes.
 */
CUDF_HOST_DEVICE constexpr cudf::size_type halo_size(cudf::size_type width)
{
  return width <= (max_halo_bytes - 16) / 4 ? 4 * width + 16 : max_halo_bytes;
}

/**
 * @brief Accessor for 4-byte little-endian words at any byte position of a string
 *
 * Bytes are read from a tile of the string staged in shared memory when possible
 * and from global memory otherwise. Bytes past the end of the string are returned as 0.
 * The `data()` function similarly returns a pointer to contiguous bytes for hashing.
 * The `pos` must be less than the string size so position arithmetic cannot overflow.
 *
 * @tparam fully_staged True if all the bytes accessed are known to be in shared memory
 */
template <bool fully_staged>
struct staged_string {
  uint32_t const* words;   // tile bytes staged in shared memory (zero-padded)
  cudf::size_type staged;  // number of bytes available in words relative to base
  char const* d_str;       // string data in global memory
  cudf::size_type base;    // byte position in d_str of the first staged byte
  cudf::size_type size;    // number of bytes in d_str

  __device__ inline uint32_t operator()(cudf::size_type pos) const
  {
    auto const rpos = pos - base;
    if (fully_staged || (rpos + 8 <= staged)) {
      auto const idx = rpos / 4;
      return __funnelshift_r(words[idx], words[idx + 1], (rpos % 4) * 8);
    }
    // w holds the raw bytes at [pos, pos+4) with the byte at pos in the lowest 8 bits
    // (same layout as the funnel-shift above)
    uint32_t w = 0;
    for (auto i = 0; i < 4 && i < (size - pos); ++i) {
      w |= static_cast<uint32_t>(static_cast<uint8_t>(d_str[pos + i])) << (i * 8);
    }
    return w;
  }

  // returns a pointer to the `bytes` contiguous bytes starting at `pos`
  __device__ inline char const* data(cudf::size_type pos, cudf::size_type bytes) const
  {
    auto const rpos = pos - base;
    if (fully_staged || (rpos + bytes <= staged)) {
      return reinterpret_cast<char const*>(words) + rpos;
    }
    return d_str + pos;
  }
};

/**
 * @brief Computes the number of bytes for `width` characters starting at `pos`
 *
 * This is equivalent to `cudf::strings::detail::bytes_to_character_position`
 * for valid UTF-8 but processes 4 bytes at a time.
 *
 * Valid UTF-8 requires at most `4 * width` bytes for `width` characters plus one word
 * to locate the next character. The scan is limited to this so that invalid UTF-8
 * (e.g. long runs of continuation bytes) cannot read beyond the staged bytes.
 *
 * @return Number of bytes and the number of characters not found before the end of the string
 */
template <typename Accessor>
__device__ inline cuda::std::pair<cudf::size_type, cudf::size_type> bytes_for_width(
  Accessor const& str, cudf::size_type pos, cudf::size_type width)
{
  cudf::size_type count = 0;
  auto const remaining  = str.size - pos;
  // the scan stops when the bytes left reach this value
  auto const stop = static_cast<cudf::size_type>(
    cuda::std::max(int64_t{0}, remaining - 4 * static_cast<int64_t>(width) - 4));
  // iterating on the remaining bytes (rather than the position) avoids overflow near the max size
  for (auto left = remaining; left > stop; left -= 4) {
    auto const itr = str.size - left;
    auto const w   = str(itr);
    // high bit of each byte is set if the byte is not a UTF-8 continuation byte
    auto starts = (~w | (w << 1)) & 0x8080'8080u;
    if (left < 4) { starts &= (1u << (left * 8)) - 1u; }
    auto const n = cuda::std::popcount(starts);
    if (count + n > width) {
      // the character after the last one in the substring starts within this word
      for (auto i = count; i < width; ++i) {
        starts &= starts - 1;
      }
      return {itr + cuda::std::countr_zero(starts) / 8 - pos, 0};
    }
    count += n;
  }
  return {remaining - stop, width - count};
}

/**
 * @brief Hashing kernel launched as a thread per tile-size (block or warp)
 * for a lists column
 *
 * This kernel computes the hashes for each row using the seed and the specified
 * hash function. The ngrams identifies consecutive strings to hash over in
 * sliding window formation. The hashes are stored in d_hashes and used as input
 * to the minhash_kernel.
 *
 * This kernel also counts the number of rows above the wide_row_threshold
 * and proactively initializes the output values for those rows.
 *
 * @tparam HashFunction The hash function to use for this kernel
 * @tparam hash_value_type Derived from HashFunction result_type
 *
 * @param d_input The input column to hash
 * @param seed The seed used for the hash function
 * @param ngrams Number of strings in each row to hash
 * @param d_hashes The resulting hash values are stored here
 * @param threshold_count Stores the number of rows above wide_row_threshold
 * @param param_count Number of parameters (used for the proactive initialize)
 * @param d_results Final results vector (used for the proactive initialize)
 */
template <typename HashFunction, typename hash_value_type = typename HashFunction::result_type>
CUDF_KERNEL void minhash_ngrams_kernel(cudf::lists_column_device_view const d_input,
                                       hash_value_type seed,
                                       cudf::size_type ngrams,
                                       hash_value_type* d_hashes,
                                       cudf::size_type* threshold_count,
                                       cudf::size_type param_count,
                                       hash_value_type* d_results)
{
  auto const tid     = cudf::detail::grid_1d::global_thread_id();
  auto const row_idx = tid / tile_size;
  if (row_idx >= d_input.size()) { return; }
  if (d_input.is_null(row_idx)) { return; }

  // retrieve this row's offset to locate the output position in d_hashes
  auto const offsets_itr = d_input.offsets().data<int32_t>() + d_input.offset();
  auto const offset      = offsets_itr[row_idx];
  auto const size_row    = offsets_itr[row_idx + 1] - offset;
  if (size_row == 0) { return; }

  auto const d_row    = cudf::list_device_view(d_input, row_idx);
  auto const lane_idx = static_cast<cudf::size_type>(tid % tile_size);

  // hashes for this row/thread are stored here
  auto seed_hashes  = d_hashes + offset - offsets_itr[0] + lane_idx;
  auto const hasher = HashFunction(seed);

  for (auto idx = lane_idx; idx < size_row; idx += tile_size, seed_hashes += tile_size) {
    if (d_row.is_null(idx)) {
      *seed_hashes = 0;
      continue;
    }

    auto next_idx = cuda::std::min(idx + ngrams, size_row - 1);
    if ((idx != 0) && ((next_idx - idx) < ngrams)) {
      *seed_hashes = 0;
      continue;
    }

    auto const first_str = d_row.element<cudf::string_view>(idx);
    auto const last_str  = d_row.element<cudf::string_view>(next_idx);
    // build super-string since adjacent strings are contiguous in memory
    auto const size = static_cast<cudf::size_type>(
      cuda::std::distance(first_str.data(), last_str.data()) + last_str.size_bytes());
    auto const hash_str = cudf::string_view(first_str.data(), size);
    hash_value_type hv;
    if constexpr (cuda::std::is_same_v<hash_value_type, uint32_t>) {
      hv = hasher(hash_str);
    } else {
      hv = cuda::std::get<0>(hasher(hash_str));
    }
    // disallowing hash to zero case
    *seed_hashes = cuda::std::max(hv, hash_value_type{1});
  }

  // logic appended here to count long rows so an extra kernel is not required
  if (size_row >= wide_row_threshold) {
    if (lane_idx == 0) {
      // count the number of wide rows
      cuda::atomic_ref<cudf::size_type, cuda::thread_scope_device> ref{*threshold_count};
      ref.fetch_add(1, cuda::std::memory_order_relaxed);
    }
    // initialize the output -- only needed for wider rows
    auto d_output = d_results + (row_idx * param_count);
    for (auto i = lane_idx; i < param_count; i += tile_size) {
      d_output[i] = cuda::std::numeric_limits<hash_value_type>::max();
    }
  }
}

/**
 * @brief Computes the permutation formula used by datatrove
 *
 * `((hv * a + b) % mersenne_prime) & hash_max`
 *
 * The modulo of the Mersenne prime 2^61-1 is computed using shifts and adds
 * which is significantly faster than the generic 64-bit modulo.
 */
template <typename hash_value_type>
__device__ inline hash_value_type permute(hash_value_type hv, hash_value_type a, hash_value_type b)
{
  constexpr uint64_t mersenne_prime  = (1UL << 61) - 1;
  constexpr hash_value_type hash_max = cuda::std::numeric_limits<hash_value_type>::max();
  auto const x = static_cast<uint64_t>(static_cast<hash_value_type>(hv * a + b));
  // x % (2^61-1) == (x & (2^61-1)) + (x >> 61) with a single conditional subtract
  auto const r = (x & mersenne_prime) + (x >> 61);
  return static_cast<hash_value_type>((r >= mersenne_prime ? r - mersenne_prime : r) & hash_max);
}

/**
 * @brief Permutation calculation kernel
 *
 * This kernel uses the hashes from the minhash_ngrams_kernel
 * and the 'parameter_a' and 'parameter_b' values to compute the final output.
 * The output is the number of input rows (N) by the number of parameter values (M).
 * Each row output[i] is the calculated result for parameter_a/b[0:M].
 *
 * This kernel is launched with either blocks per row of 1 for rows
 * below the wide_row_threshold or blocks per row = blocks_per_rows
 * for rows above wide_row_threshold.
 *
 * Note that this was refactored to accommodate lists of strings which is possible
 * since there is no need here to access the characters, only the hash values.
 * The offsets and width are used to locate and count the hash values produced by
 * kernels above for each input row.
 *
 * @tparam offsets_type Type for the offsets iterator for the input column
 * @tparam hash_value_type Derived from HashFunction result_type
 * @tparam blocks_per_row Number of blocks used to process each row
 *
 * @param offsets_itr The offsets are used to address the d_hashes
 * @param indices The indices of the rows in the input column
 * @param parameter_a 1st set of parameters for the calculation result
 * @param parameter_b 2nd set of parameters for the calculation result
 * @param width Used for calculating the number of available hashes in each row
 * @param d_hashes The hash values computed in one of the hash kernels
 * @param d_results Final results vector of calculate values
 */
template <typename offsets_type, typename hash_value_type, int blocks_per_row>
CUDF_KERNEL void minhash_kernel(offsets_type offsets_itr,
                                cudf::device_span<cudf::size_type const> indices,
                                cudf::device_span<hash_value_type const> parameter_a,
                                cudf::device_span<hash_value_type const> parameter_b,
                                cudf::size_type width,
                                hash_value_type const* d_hashes,
                                hash_value_type* d_results)
{
  auto const tid = cudf::detail::grid_1d::global_thread_id();
  auto const idx = (tid / blocks_per_row) / block_size;
  if (idx >= indices.size()) { return; }
  auto const row_idx = indices[idx];

  auto const block      = cooperative_groups::this_thread_block();
  int const section_idx = block.group_index().x % blocks_per_row;

  auto const offset   = offsets_itr[row_idx];
  auto const row_size = static_cast<cudf::size_type>(offsets_itr[row_idx + 1] - offset);

  // number of items to process in this block;
  // last block also includes any remainder values from the row_size/blocks_per_row truncation
  // example:
  //  each section_size for string with size 588090 and blocks_per_row=64 is 9188
  //  except the last section which is 9188 + (588090 % 64) = 9246
  auto const section_size = (row_size / blocks_per_row) +
                            (section_idx < (blocks_per_row - 1) ? 0 : row_size % blocks_per_row);
  auto const section_offset = section_idx * (row_size / blocks_per_row);

  // hash values for this block/section
  auto const seed_hashes = d_hashes + offset - offsets_itr[0] + section_offset;
  // width used here as a max value since a string's char-count <= byte-count
  auto const hashes_size =
    section_idx < (blocks_per_row - 1)
      ? section_size
      : cuda::std::max(static_cast<cudf::size_type>(row_size > 0), section_size - width + 1);

  auto const init     = row_size == 0 ? 0 : cuda::std::numeric_limits<hash_value_type>::max();
  auto const lane_idx = block.thread_rank();
  auto const d_output = d_results + (row_idx * parameter_a.size());

  auto const begin = seed_hashes + lane_idx;
  auto const end   = seed_hashes + hashes_size;

  auto const warp     = cooperative_groups::tiled_partition<cudf::detail::warp_size>(block);
  auto const warp_idx = warp.meta_group_rank();

  // holds the per-warp minimums for the block-level reduction
  constexpr auto num_warps         = block_size / cudf::detail::warp_size;
  constexpr auto params_per_thread = detail::params_per_thread<hash_value_type>;
  __shared__ hash_value_type warp_values[num_warps * params_per_thread];

  for (std::size_t i = 0; i < parameter_a.size(); i += params_per_thread) {
    auto const param_count =
      cuda::std::min(static_cast<cuda::std::size_t>(params_per_thread), parameter_a.size() - i);

    // parameters and the running minimums are kept in registers;
    // the last chunk may be partial so the extra slots just repeat the last parameter
    hash_value_type pa[params_per_thread];
    hash_value_type pb[params_per_thread];
    hash_value_type mins[params_per_thread];
#pragma unroll
    for (std::size_t k = 0; k < params_per_thread; ++k) {
      auto const param_idx = i + cuda::std::min(k, param_count - 1);
      pa[k]                = parameter_a[param_idx];
      pb[k]                = parameter_b[param_idx];
      mins[k]              = init;
    }

    // each lane accumulates min hashes in its registers
    if constexpr (prefetch_hash<hash_value_type>) {
      // the next hash value is loaded before processing the current one to hide its latency
      auto hv = begin < end ? *begin : hash_value_type{0};
      for (auto itr = begin; itr < end; itr += block_size) {
        auto const next    = itr + block_size;
        auto const hv_next = next < end ? *next : hash_value_type{0};
        // 0 is used as a skip sentinel for UTF-8 and trailing bytes
        if (hv != 0) {
#pragma unroll
          for (std::size_t k = 0; k < params_per_thread; ++k) {
            mins[k] = cuda::std::min(permute(hv, pa[k], pb[k]), mins[k]);
          }
        }
        hv = hv_next;
      }
    } else {
      for (auto itr = begin; itr < end; itr += block_size) {
        auto const hv = *itr;
        // 0 is used as a skip sentinel for UTF-8 and trailing bytes
        if (hv == 0) { continue; }
#pragma unroll
        for (std::size_t k = 0; k < params_per_thread; ++k) {
          mins[k] = cuda::std::min(permute(hv, pa[k], pb[k]), mins[k]);
        }
      }
    }

    // reduce each parameter's values to a single min value:
    // first within each warp and then across the warps using shared memory
#pragma unroll
    for (std::size_t k = 0; k < params_per_thread; ++k) {
      auto const minv =
        cooperative_groups::reduce(warp, mins[k], cooperative_groups::less<hash_value_type>{});
      if (warp.thread_rank() == 0) { warp_values[warp_idx * params_per_thread + k] = minv; }
    }
    block.sync();

    // each thread reduces the warp values for a single parameter (thread per parameter)
    if (lane_idx < param_count) {
      auto minv = warp_values[lane_idx];
      for (auto w = 1; w < num_warps; ++w) {
        minv = cuda::std::min(minv, warp_values[w * params_per_thread + lane_idx]);
      }
      if constexpr (blocks_per_row > 1) {
        // accumulates mins for each block into d_output
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_device> ref{d_output[lane_idx + i]};
        ref.fetch_min(minv, cuda::std::memory_order_relaxed);
      } else {
        d_output[lane_idx + i] = minv;
      }
    }
    block.sync();
  }
}

/**
 * @brief Computes the hash of the `width` characters starting at `pos`
 *
 * Returns 0 if `pos` is not at the beginning of the string and there are fewer than
 * `width` characters remaining. The 0 value is reserved as a skip sentinel so a computed
 * hash of 0 is returned as 1.
 */
template <typename HashFunction, typename hash_value_type, typename Accessor>
__device__ inline hash_value_type hash_substring(Accessor const& str,
                                                 cudf::size_type pos,
                                                 cudf::size_type width,
                                                 hash_value_type seed)
{
  auto const [bytes, left] = bytes_for_width(str, pos, width);
  if ((pos != 0) && (left > 0)) { return 0; }  // true pos+width is past the end of the string
  auto const hasher = HashFunction(seed);
  auto const sv     = cudf::string_view(str.data(pos, bytes), bytes);
  hash_value_type hv;
  if constexpr (cuda::std::is_same_v<hash_value_type, uint32_t>) {
    hv = hasher(sv);
  } else {
    hv = cuda::std::get<0>(hasher(sv));
  }
  return cuda::std::max(hv, hash_value_type{1});
}

/**
 * @brief Returns the number of parameter groups used by each block in minhash_fused_kernel
 *
 * Each group is fused_params_per_thread parameters.
 */
template <typename hash_value_type>
CUDF_HOST_DEVICE constexpr cudf::size_type max_param_groups(cudf::size_type param_count)
{
  constexpr auto ppt = fused_params_per_thread<hash_value_type>;
  return cuda::std::min(cudf::util::div_rounding_up_safe(param_count, ppt),
                        static_cast<cudf::size_type>(block_size));
}

/**
 * @brief Returns the dynamic shared memory size in bytes used by minhash_fused_kernel
 *
 * @param max_groups Maximum number of parameter groups (see max_param_groups)
 * @param halo Number of bytes staged beyond each tile (see halo_size)
 */
template <typename hash_value_type>
constexpr std::size_t fused_shmem_size(cudf::size_type max_groups, cudf::size_type halo)
{
  auto const hash_count = seed_tile_bytes + max_groups * fused_params_per_thread<hash_value_type>;
  return static_cast<std::size_t>(hash_count) * sizeof(hash_value_type) + seed_tile_bytes + halo;
}

/**
 * @brief Computes the minhash values for a strings column
 *
 * Each block processes up to section_bytes character positions of a single row.
 * The block_offsets identify the row and section for each block.
 *
 * The character positions are processed in tiles of seed_tile_bytes.
 * For each tile, the characters (plus a halo for the substrings that extend past the tile)
 * are staged in shared memory and the substring hashes are computed into shared memory.
 * Each thread then applies its fused_params_per_thread parameters to a subset of the
 * tile's hashes keeping a running minimum per parameter in registers.
 * The threads are arranged so that the threads in a warp read the same hash values.
 *
 * Once all tiles are processed, the minimums are reduced across the threads and
 * written to the output. Rows processed by more than one block use atomic-min
 * on the output which must be initialized to the max hash value.
 *
 * If there are more than block_size * fused_params_per_thread parameters,
 * the parameters are processed in chunks with the hashes recomputed for each chunk.
 *
 * @tparam HashFunction The hash function to use for the substrings
 * @tparam hash_value_type The hash type (uint32_t or uint64_t)
 * @tparam fully_staged True if the halo includes all bytes for the `width` characters
 *
 * @param d_strings The input strings
 * @param block_offsets The first block for each row (size = number of rows + 1)
 * @param seed The seed used for the hash function
 * @param width Width in characters used for determining substrings to hash
 * @param parameter_a 1st set of parameters for the calculation result
 * @param parameter_b 2nd set of parameters for the calculation result
 * @param d_results Final results vector of calculated values
 */
template <typename HashFunction, typename hash_value_type, bool fully_staged>
CUDF_KERNEL void __launch_bounds__(block_size, fused_min_blocks<hash_value_type>)
  minhash_fused_kernel(cudf::column_device_view const d_strings,
                       cudf::device_span<cudf::size_type const> block_offsets,
                       hash_value_type seed,
                       cudf::size_type width,
                       cudf::device_span<hash_value_type const> parameter_a,
                       cudf::device_span<hash_value_type const> parameter_b,
                       hash_value_type* d_results)
{
  constexpr auto ppt      = fused_params_per_thread<hash_value_type>;
  constexpr auto hash_max = cuda::std::numeric_limits<hash_value_type>::max();

  auto const block    = cooperative_groups::this_thread_block();
  auto const warp     = cooperative_groups::tiled_partition<cudf::detail::warp_size>(block);
  auto const lane_idx = static_cast<cudf::size_type>(block.thread_rank());

  // locate the row and section for this block
  auto const block_idx = static_cast<cudf::size_type>(block.group_index().x);
  auto const row_idx =
    static_cast<cudf::size_type>(cuda::std::distance(
      block_offsets.begin(),
      thrust::upper_bound(thrust::seq, block_offsets.begin(), block_offsets.end(), block_idx))) -
    1;
  auto const section_idx  = block_idx - block_offsets[row_idx];
  auto const num_sections = block_offsets[row_idx + 1] - block_offsets[row_idx];

  auto const param_count = static_cast<cudf::size_type>(parameter_a.size());
  auto const d_output    = d_results + (static_cast<int64_t>(row_idx) * param_count);

  auto const offsets = d_strings.child(cudf::strings_column_view::offsets_column_index);
  auto const offsets_itr =
    cudf::detail::input_offsetalator(offsets.head(), offsets.type(), d_strings.offset());
  auto const offset     = offsets_itr[row_idx];
  auto const size_bytes = d_strings.is_null(row_idx)
                            ? 0
                            : static_cast<cudf::size_type>(offsets_itr[row_idx + 1] - offset);
  if (size_bytes == 0) {
    for (auto i = lane_idx; i < param_count; i += block_size) {
      d_output[i] = 0;
    }
    return;
  }
  auto const d_str = d_strings.head<char>() + offset;

  // character positions processed by this block;
  // positions after (size_bytes - width) can only produce a 0 hash and so are skipped
  // (computed so that no intermediate value exceeds size_bytes to avoid overflow)
  auto const begin = section_idx * section_bytes;
  auto const limit = cuda::std::max(size_bytes - width + 1, 1);
  auto const end   = (limit - begin) > section_bytes ? begin + section_bytes : limit;

  // shared memory: hash values tile, block minimums, staged characters
  extern __shared__ __align__(8) char shmem[];
  auto const hash_tile    = reinterpret_cast<hash_value_type*>(shmem);
  auto const block_mins   = hash_tile + seed_tile_bytes;
  auto const max_groups   = max_param_groups<hash_value_type>(param_count);
  auto const staged_words = reinterpret_cast<uint32_t*>(block_mins + (max_groups * ppt));
  auto const staged_bytes = reinterpret_cast<char*>(staged_words);
  auto const stage_size   = seed_tile_bytes + halo_size(width);

  // iterating on the chunk index (rather than the position) avoids overflow
  auto const chunk_size = max_groups * ppt;
  auto const num_chunks = cudf::util::div_rounding_up_safe(param_count, chunk_size);
  for (cudf::size_type chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
    auto const chunk       = chunk_idx * chunk_size;
    auto const chunk_count = cuda::std::min(param_count - chunk, chunk_size);
    // each thread is assigned a group of parameters and a hash lane;
    // all the threads in a hash lane read the same hash values
    auto const groups    = cudf::util::div_rounding_up_safe(chunk_count, ppt);
    auto const lanes     = static_cast<cudf::size_type>(block_size) / groups;
    auto const group_idx = lane_idx % groups;
    auto const hash_lane = lane_idx / groups;

    // the last group may be partial so the extra slots just repeat the last parameter
    hash_value_type pa[ppt];
    hash_value_type pb[ppt];
    hash_value_type mins[ppt];
#pragma unroll
    for (cudf::size_type k = 0; k < ppt; ++k) {
      auto const param_idx = chunk + cuda::std::min(group_idx * ppt + k, chunk_count - 1);
      pa[k]                = parameter_a[param_idx];
      pb[k]                = parameter_b[param_idx];
      mins[k]              = hash_max;
    }
    for (auto i = lane_idx; i < groups * ppt; i += block_size) {
      block_mins[i] = hash_max;
    }

    for (cudf::size_type tile_offset = 0; tile_offset < end - begin;
         tile_offset += seed_tile_bytes) {
      auto const tile = begin + tile_offset;
      block.sync();  // protect the shared memory from the previous iteration
      for (auto i = lane_idx; i < stage_size; i += block_size) {
        staged_bytes[i] = i < (size_bytes - tile) ? d_str[tile + i] : 0;
      }
      block.sync();

      // compute the hashes for this tile
      auto const str =
        staged_string<fully_staged>{staged_words, stage_size, d_str, tile, size_bytes};
      auto const tile_count = cuda::std::min(cudf::size_type{seed_tile_bytes}, end - tile);
      for (auto i = lane_idx; i < tile_count; i += block_size) {
        hash_tile[i] = cudf::strings::detail::is_utf8_continuation_char(staged_bytes[i])
                         ? hash_value_type{0}
                         : hash_substring<HashFunction>(str, tile + i, width, seed);
      }
      block.sync();

      // each thread accumulates the min values for its parameters
      if (hash_lane < lanes) {
        for (auto i = hash_lane; i < tile_count; i += lanes) {
          auto const hv = hash_tile[i];
          // 0 is used as a skip sentinel for UTF-8 and trailing bytes
          if (hv == 0) { continue; }
#pragma unroll
          for (cudf::size_type k = 0; k < ppt; ++k) {
            mins[k] = cuda::std::min(permute(hv, pa[k], pb[k]), mins[k]);
          }
        }
      }
    }
    block.sync();  // block_mins initialized before reducing into it

    // reduce the min values across the hash lanes:
    // if the groups divide the warp evenly then the warp lanes with the same group are
    // reduced with shuffles first and only the first lanes update the shared memory
    auto reducer = hash_lane < lanes;
    if (cudf::detail::warp_size % groups == 0) {
#pragma unroll
      for (cudf::size_type k = 0; k < ppt; ++k) {
        for (auto delta = cudf::detail::warp_size / 2; delta >= groups; delta /= 2) {
          mins[k] = cuda::std::min(mins[k], warp.shfl_xor(mins[k], delta));
        }
      }
      reducer = static_cast<cudf::size_type>(warp.thread_rank()) < groups;
    }
    if (reducer) {
#pragma unroll
      for (cudf::size_type k = 0; k < ppt; ++k) {
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_block> ref{
          block_mins[group_idx * ppt + k]};
        ref.fetch_min(mins[k], cuda::std::memory_order_relaxed);
      }
    }
    block.sync();

    for (auto i = lane_idx; i < chunk_count; i += block_size) {
      if (num_sections > 1) {
        cuda::atomic_ref<hash_value_type, cuda::thread_scope_device> ref{d_output[chunk + i]};
        ref.fetch_min(block_mins[i], cuda::std::memory_order_relaxed);
      } else {
        d_output[chunk + i] = block_mins[i];
      }
    }
    block.sync();  // protect block_mins for the next chunk
  }
}

/**
 * @brief Partition input rows by row size
 *
 * The returned index is the first row above the wide_row_threshold size.
 * The returned vector are the indices partitioned above and below the
 * wide_row_threshold size.
 *
 * @param size Number of rows in the input column
 * @param threshold_count Number of rows above wide_row_threshold
 * @param tfn Transform function returns the size of each row
 * @param stream Stream used for allocation and kernel launches
 * @param mr Memory resources used for temporary allocations and the returned indices
 */
template <typename transform_fn>
std::pair<cudf::size_type, cuda::device_buffer<cudf::size_type>> partition_input(
  cudf::size_type size,
  cudf::size_type threshold_count,
  transform_fn tfn,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  auto const temp_mr = mr.get_temporary_mr();
  auto indices =
    cuda::device_buffer<cudf::size_type>(stream, mr.get_output_mr(), size, cuda::no_init);
  thrust::sequence(rmm::exec_policy_nosync(stream, temp_mr), indices.begin(), indices.end());
  cudf::size_type threshold_index = threshold_count < size ? size : 0;

  // if we counted a split of above/below threshold then
  // compute partitions based on the size of each string
  if ((threshold_count > 0) && (threshold_count < size)) {
    auto sizes = cuda::device_buffer<cudf::size_type>(stream, temp_mr, size, cuda::no_init);
    auto begin = cuda::counting_iterator<cudf::size_type>{0};
    auto end   = begin + size;
    thrust::transform(rmm::exec_policy_nosync(stream, temp_mr), begin, end, sizes.data(), tfn);
    // these 2 are slightly faster than using partition()
    thrust::sort_by_key(
      rmm::exec_policy_nosync(stream, temp_mr), sizes.begin(), sizes.end(), indices.begin());
    auto const lb = thrust::lower_bound(
      rmm::exec_policy_nosync(stream, temp_mr), sizes.begin(), sizes.end(), wide_row_threshold);
    threshold_index = static_cast<cudf::size_type>(cuda::std::distance(sizes.begin(), lb));
  }
  return {threshold_index, std::move(indices)};
}

template <typename HashFunction, typename hash_value_type = typename HashFunction::result_type>
std::unique_ptr<cudf::column> minhash_fn(cudf::strings_column_view const& input,
                                         hash_value_type seed,
                                         cudf::device_span<hash_value_type const> parameter_a,
                                         cudf::device_span<hash_value_type const> parameter_b,
                                         cudf::size_type width,
                                         cuda::stream_ref stream,
                                         cudf::memory_resources mr)
{
  CUDF_EXPECTS(width >= 2,
               "Parameter width should be an integer value of 2 or greater",
               std::invalid_argument);
  CUDF_EXPECTS(!parameter_a.empty(), "Parameters A and B cannot be empty", std::invalid_argument);
  CUDF_EXPECTS(parameter_a.size() == parameter_b.size(),
               "Parameters A and B should have the same number of elements",
               std::invalid_argument);
  CUDF_EXPECTS(
    (static_cast<std::size_t>(input.size()) * parameter_a.size()) <
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "The number of parameters times the number of input rows exceeds the column size limit",
    std::overflow_error);

  auto const output_type = cudf::data_type{cudf::type_to_id<hash_value_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const temp_mr   = mr.get_temporary_mr();
  auto const d_strings = cudf::column_device_view::create(input.parent(), stream, temp_mr);

  auto results =
    cudf::make_numeric_column(output_type,
                              input.size() * static_cast<cudf::size_type>(parameter_a.size()),
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr.get_output_mr());
  auto d_results = results->mutable_view().data<hash_value_type>();

  // compute the first block (section) for each row
  auto const num_rows = input.size();
  auto block_offsets =
    cuda::device_buffer<cudf::size_type>(stream, temp_mr, num_rows + 1, cuda::no_init);
  auto sections_fn = cuda::proclaim_return_type<cudf::size_type>(
    [d_strings = *d_strings, num_rows] __device__(cudf::size_type idx) {
      // the value for the last element (idx == num_rows) is ignored by the exclusive scan
      if (idx == num_rows || d_strings.is_null(idx)) { return 1; }
      auto const size = d_strings.element<cudf::string_view>(idx).size_bytes();
      return cuda::std::max(cudf::util::div_rounding_up_safe(size, section_bytes), 1);
    });
  auto const sections_itr =
    cuda::transform_iterator(cuda::counting_iterator<cudf::size_type>{0}, sections_fn);
  auto const total_blocks = cudf::detail::sizes_to_offsets(
    sections_itr, sections_itr + num_rows + 1, block_offsets.begin(), 0, stream, mr);
  CUDF_EXPECTS(total_blocks < static_cast<int64_t>(std::numeric_limits<cudf::size_type>::max()),
               "The input is too large for the minhash kernel",
               std::overflow_error);
  auto const num_blocks = static_cast<cudf::size_type>(total_blocks);
  auto const policy     = rmm::exec_policy_nosync(stream, temp_mr);

  // rows processed by multiple blocks use atomic-min and so require initializing to max
  if (num_blocks > num_rows) {
    thrust::fill(policy,
                 d_results,
                 d_results + results->size(),
                 cuda::std::numeric_limits<hash_value_type>::max());
  }

  // if the halo can hold all the bytes for `width` characters, the global memory
  // fallback in staged_string is not needed
  auto const fully_staged = width <= (max_halo_bytes - 16) / 4;
  auto const param_count  = static_cast<cudf::size_type>(parameter_a.size());
  auto const shmem_size   = fused_shmem_size<hash_value_type>(
    max_param_groups<hash_value_type>(param_count), halo_size(width));
  static_assert(fused_shmem_size<hash_value_type>(block_size, max_halo_bytes) <= 48 * 1024,
                "minhash_fused_kernel shared memory exceeds the default 48KB limit");
  auto fused_kernel = fully_staged ? minhash_fused_kernel<HashFunction, hash_value_type, true>
                                   : minhash_fused_kernel<HashFunction, hash_value_type, false>;
  fused_kernel<<<num_blocks, block_size, shmem_size, stream.get()>>>(
    *d_strings,
    cudf::device_span<cudf::size_type const>(block_offsets.data(), num_rows + 1),
    seed,
    width,
    parameter_a,
    parameter_b,
    d_results);
  CUDF_CUDA_TRY(cudaGetLastError());

  return results;
}

template <typename HashFunction, typename hash_value_type = typename HashFunction::result_type>
std::unique_ptr<cudf::column> minhash_ngrams_fn(
  cudf::lists_column_view const& input,
  cudf::size_type ngrams,
  hash_value_type seed,
  cudf::device_span<hash_value_type const> parameter_a,
  cudf::device_span<hash_value_type const> parameter_b,
  cuda::stream_ref stream,
  cudf::memory_resources mr)
{
  CUDF_EXPECTS(ngrams >= 2,
               "Parameter ngrams should be an integer value of 2 or greater",
               std::invalid_argument);
  CUDF_EXPECTS(!parameter_a.empty(), "Parameters A and B cannot be empty", std::invalid_argument);
  CUDF_EXPECTS(parameter_a.size() == parameter_b.size(),
               "Parameters A and B should have the same number of elements",
               std::invalid_argument);
  CUDF_EXPECTS(
    (static_cast<std::size_t>(input.size()) * parameter_a.size()) <
      static_cast<std::size_t>(std::numeric_limits<cudf::size_type>::max()),
    "The number of parameters times the number of input rows exceeds the column size limit",
    std::overflow_error);

  auto const output_type = cudf::data_type{cudf::type_to_id<hash_value_type>()};
  if (input.is_empty()) { return cudf::make_empty_column(output_type); }

  auto const temp_mr = mr.get_temporary_mr();
  auto const d_input = cudf::column_device_view::create(input.parent(), stream, temp_mr);

  auto results =
    cudf::make_numeric_column(output_type,
                              input.size() * static_cast<cudf::size_type>(parameter_a.size()),
                              cudf::mask_state::UNALLOCATED,
                              stream,
                              mr.get_output_mr());
  auto d_results = results->mutable_view().data<hash_value_type>();

  cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(input.size()) * block_size,
                             block_size};
  auto const hashes_size = input.child().size();
  auto d_hashes = cuda::device_buffer<hash_value_type>(stream, temp_mr, hashes_size, cuda::no_init);
  auto d_threshold_count = cudf::detail::device_scalar<cudf::size_type>(0, stream, temp_mr);

  auto d_list = cudf::lists_column_device_view(*d_input);
  minhash_ngrams_kernel<HashFunction>
    <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(d_list,
                                                                       seed,
                                                                       ngrams,
                                                                       d_hashes.data(),
                                                                       d_threshold_count.data(),
                                                                       parameter_a.size(),
                                                                       d_results);
  CUDF_CUDA_TRY(cudaGetLastError());

  auto sizes_fn = [d_list] __device__(auto idx) -> cudf::size_type {
    if (d_list.is_null(idx)) { return 0; }
    return cudf::list_device_view(d_list, idx).size();
  };
  auto [threshold_index, indices] = partition_input(
    input.size(), d_threshold_count.value(stream), sizes_fn, stream, mr.as_temporary());

  auto input_offsets = input.offsets_begin();  // already includes input.offset()
  using offset_type  = decltype(input_offsets);

  // handle the strings below the threshold width
  if (threshold_index > 0) {
    auto d_indices = cudf::device_span<cudf::size_type const>(indices.data(), threshold_index);
    cudf::detail::grid_1d grid{static_cast<cudf::thread_index_type>(d_indices.size()) * block_size,
                               block_size};
    minhash_kernel<offset_type, hash_value_type, 1>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
        input_offsets, d_indices, parameter_a, parameter_b, ngrams, d_hashes.data(), d_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  // handle the strings above the threshold width
  if (threshold_index < input.size()) {
    auto const count = static_cast<cudf::thread_index_type>(input.size() - threshold_index);
    auto d_indices =
      cudf::device_span<cudf::size_type const>(indices.data() + threshold_index, count);
    cudf::detail::grid_1d grid{count * block_size * blocks_per_row, block_size};
    minhash_kernel<offset_type, hash_value_type, blocks_per_row>
      <<<grid.num_blocks, grid.num_threads_per_block, 0, stream.get()>>>(
        input_offsets, d_indices, parameter_a, parameter_b, ngrams, d_hashes.data(), d_results);
    CUDF_CUDA_TRY(cudaGetLastError());
  }

  return results;
}

std::unique_ptr<cudf::column> build_list_result(cudf::column_view const& input,
                                                std::unique_ptr<cudf::column>&& hashes,
                                                cudf::size_type seeds_size,
                                                cuda::stream_ref stream,
                                                cudf::memory_resources mr)
{
  // build the offsets for the output lists column
  auto const temp_mr   = mr.get_temporary_mr();
  auto const output_mr = mr.get_output_mr();
  auto const zero      = cudf::numeric_scalar<cudf::size_type>(0, true, stream, temp_mr);
  auto const size      = cudf::numeric_scalar<cudf::size_type>(seeds_size, true, stream, temp_mr);
  auto offsets         = cudf::detail::sequence(input.size() + 1, zero, size, stream, output_mr);
  hashes->set_null_mask(cudf::create_null_mask(0, cudf::mask_state::UNALLOCATED, stream, mr),
                        0);  // children have no nulls

  // build the lists column from the offsets and the hashes
  auto result = make_lists_column(input.size(),
                                  std::move(offsets),
                                  std::move(hashes),
                                  input.null_count(),
                                  cudf::detail::copy_bitmask(input, stream, mr));
  // expect this condition to be very rare
  if (input.null_count() > 0) {
    result = cudf::detail::purge_nonempty_nulls(result->view(), stream, output_mr);
  }
  return result;
}
}  // namespace

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      uint32_t seed,
                                      cudf::device_span<uint32_t const> parameter_a,
                                      cudf::device_span<uint32_t const> parameter_b,
                                      cudf::size_type width,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
  auto hashes =
    detail::minhash_fn<HashFunction>(input, seed, parameter_a, parameter_b, width, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), parameter_a.size(), stream, mr);
}

std::unique_ptr<cudf::column> minhash_ngrams(cudf::lists_column_view const& input,
                                             cudf::size_type ngrams,
                                             uint32_t seed,
                                             cudf::device_span<uint32_t const> parameter_a,
                                             cudf::device_span<uint32_t const> parameter_b,
                                             cuda::stream_ref stream,
                                             cudf::memory_resources mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x86_32<cudf::string_view>;
  auto hashes        = detail::minhash_ngrams_fn<HashFunction>(
    input, ngrams, seed, parameter_a, parameter_b, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), parameter_a.size(), stream, mr);
}

std::unique_ptr<cudf::column> minhash64(cudf::strings_column_view const& input,
                                        uint64_t seed,
                                        cudf::device_span<uint64_t const> parameter_a,
                                        cudf::device_span<uint64_t const> parameter_b,
                                        cudf::size_type width,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x64_128<cudf::string_view>;
  auto hashes =
    detail::minhash_fn<HashFunction>(input, seed, parameter_a, parameter_b, width, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), parameter_a.size(), stream, mr);
}

std::unique_ptr<cudf::column> minhash64_ngrams(cudf::lists_column_view const& input,
                                               cudf::size_type ngrams,
                                               uint64_t seed,
                                               cudf::device_span<uint64_t const> parameter_a,
                                               cudf::device_span<uint64_t const> parameter_b,
                                               cuda::stream_ref stream,
                                               cudf::memory_resources mr)
{
  using HashFunction = cudf::hashing::detail::MurmurHash3_x64_128<cudf::string_view>;
  auto hashes        = detail::minhash_ngrams_fn<HashFunction>(
    input, ngrams, seed, parameter_a, parameter_b, stream, mr);
  return build_list_result(input.parent(), std::move(hashes), parameter_a.size(), stream, mr);
}

}  // namespace detail

std::unique_ptr<cudf::column> minhash(cudf::strings_column_view const& input,
                                      uint32_t seed,
                                      cudf::device_span<uint32_t const> parameter_a,
                                      cudf::device_span<uint32_t const> parameter_b,
                                      cudf::size_type width,
                                      cuda::stream_ref stream,
                                      cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash(input, seed, parameter_a, parameter_b, width, stream, mr);
}

std::unique_ptr<cudf::column> minhash_ngrams(cudf::lists_column_view const& input,
                                             cudf::size_type ngrams,
                                             uint32_t seed,
                                             cudf::device_span<uint32_t const> parameter_a,
                                             cudf::device_span<uint32_t const> parameter_b,
                                             cuda::stream_ref stream,
                                             cudf::memory_resources mr)

{
  CUDF_FUNC_RANGE();
  return detail::minhash_ngrams(input, ngrams, seed, parameter_a, parameter_b, stream, mr);
}

std::unique_ptr<cudf::column> minhash64(cudf::strings_column_view const& input,
                                        uint64_t seed,
                                        cudf::device_span<uint64_t const> parameter_a,
                                        cudf::device_span<uint64_t const> parameter_b,
                                        cudf::size_type width,
                                        cuda::stream_ref stream,
                                        cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::minhash64(input, seed, parameter_a, parameter_b, width, stream, mr);
}

std::unique_ptr<cudf::column> minhash64_ngrams(cudf::lists_column_view const& input,
                                               cudf::size_type ngrams,
                                               uint64_t seed,
                                               cudf::device_span<uint64_t const> parameter_a,
                                               cudf::device_span<uint64_t const> parameter_b,
                                               cuda::stream_ref stream,
                                               cudf::memory_resources mr)

{
  CUDF_FUNC_RANGE();
  return detail::minhash64_ngrams(input, ngrams, seed, parameter_a, parameter_b, stream, mr);
}

}  // namespace nvtext

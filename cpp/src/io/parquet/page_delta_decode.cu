/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "delta_binary.cuh"
#include "io/utilities/block_utils.cuh"
#include "page_state_composed.cuh"
#include "page_string_utils.cuh"
#include "parquet_gpu.hpp"

#include <cudf/detail/utilities/cuda.cuh>

#include <rmm/exec_policy.hpp>

#include <cuda/std/type_traits>
#include <thrust/transform_scan.h>

namespace cudf::io::parquet::detail {

namespace {

namespace cg = cooperative_groups;
namespace delta_fastpath {

// Returns whether the fast-path kernel supports the given page.
CUDF_HOST_DEVICE __forceinline__ bool supported(PageInfo const& page, ColumnChunkDesc const& chunk)
{
  return BitAnd(static_cast<uint32_t>(decode_kernel_mask::DELTA_BINARY), page.kernel_mask) != 0 &&
         chunk.physical_type == Type::INT64 && chunk.max_level[level_type::DEFINITION] == 0 &&
         chunk.max_level[level_type::REPETITION] == 0 && chunk.max_nesting_depth == 1;
}

constexpr unsigned full_mask              = 0xffffffffu;
constexpr int warp_size                   = 32;
constexpr int vals_per_block_multiple     = 128;
constexpr int vals_per_miniblock_multiple = 32;
constexpr int bits_per_word               = 32;
constexpr int block_size                  = 128;
constexpr int word_alignment              = 4;
constexpr int max_fast_bit_width          = 27;

// Perform arithmetic through uint64_t to preserve wrapping semantics without signed-overflow UB.
__device__ __forceinline__ int64_t wrapping_i64_add(int64_t a, int64_t b)
{
  return static_cast<int64_t>(static_cast<uint64_t>(a) + static_cast<uint64_t>(b));
}

__device__ __forceinline__ int64_t wrapping_i64_multiply(int64_t a, int64_t b)
{
  return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}

__device__ __forceinline__ int64_t decode_zigzag(uint64_t encoded)
{
  int64_t const magnitude = encoded >> 1;
  int const sign_bit      = (encoded & 1);
  // XOR with 0 or -1 returns magnitude or ~magnitude, respectively.
  return magnitude ^ (-sign_bit);
}

template <bool fast>
using decode_result_t = cuda::std::conditional_t<fast, uint32_t, int64_t>;

// Parquet stores header integers as variable-length ULEB128 values. Each byte contains a
// continuation bit (MSB) and 7 data bits.
//
// The slow path reads one byte at a time, replicated across all lanes. The fast path handles values
// that fit in four bytes. ASSUME_FAST uses format guarantees to omit the slow-path branch.
//
template <bool ASSUME_FAST>
__device__ __forceinline__ static int read_uleb128(uint8_t const* input,
                                                   int lane_id,
                                                   decode_result_t<ASSUME_FAST>& decoded)
{
  // Each group of four lanes collectively loads the first four bytes and shifts them into position.
  // __clz then determines the encoded length from the first cleared continuation bit. Values longer
  // than four bytes fall through to the slow path.
  int const reversed_byte_lane = (~lane_id & 3);
  uint8_t const lane_byte      = __ldg(&input[reversed_byte_lane]);
  uint32_t const lane_bits     = (lane_byte & 0x7f) << (7 * reversed_byte_lane);

  // Reversing the lane order makes __clz return the encoded byte count.
  uint32_t const terminator_mask = __ballot_sync(full_mask, static_cast<int8_t>(lane_byte) >= 0);
  int const byte_count           = __clz(terminator_mask) + 1;
  uint32_t const result          = __reduce_or_sync(full_mask, lane_bits);
  uint32_t const mask            = (1u << (7 * byte_count)) - 1;

  if constexpr (ASSUME_FAST) {
    decoded = result & mask;
    return byte_count;
  }
  if (__builtin_expect(byte_count <= 4, true)) {
    decoded = uint64_t{result & mask};
    return byte_count;
  }

  decoded = 0ull;
  int i   = 0;
  uint8_t byte;
  do {
    byte = input[i];
    // Mask off the continuation bit and shift the payload into position.
    decoded |= (uint64_t{byte} & 0x7f) << (7 * i);
    i += 1;
    // Test the continuation bit.
  } while (byte & 0x80);
  return i;
}

// Extracts one bit-packed value. word_input is word-aligned, and bit_alignment specifies the
// initial bit offset.
template <bool FAST>
__device__ __forceinline__ static void extract_bitpack(uint32_t const* word_input,
                                                       int bit_width,
                                                       int value_index,
                                                       int bit_alignment,
                                                       decode_result_t<FAST>& decoded_value)
{
  if (bit_width < 0 || value_index < 0) { __builtin_unreachable(); }

  // Only the initial misalignment is needed because each warp advances by 32 values, making
  // value_index * bit_width advance in four-byte multiples.
  int const bit_idx  = bit_width * value_index + bit_alignment;
  int const word_idx = bit_idx / bits_per_word;

  // Buffer padding makes the __ldg calls safe unconditionally. A misaligned value of at most 32
  // bits spans no more than two words; a value of at most 64 bits spans no more than three.
  // funnelshift_r implicitly applies bit_idx % 32.
  if constexpr (FAST) {
    if (bit_width > max_fast_bit_width) { __builtin_unreachable(); }
    int const first_word_index  = word_idx;
    int const second_word_index = word_idx + 1;
    uint32_t const first_word   = __ldg(&word_input[first_word_index]);
    uint32_t const second_word  = __ldg(&word_input[second_word_index]);
    uint32_t const mask         = (1u << bit_width) - 1;
    decoded_value               = __funnelshift_r(first_word, second_word, bit_idx) & mask;
  } else {
    int const first_word_index  = word_idx;
    int const second_word_index = word_idx + 1;
    int const third_word_index  = word_idx + 2;
    uint32_t const first_word   = __ldg(&word_input[first_word_index]);
    uint32_t const second_word  = __ldg(&word_input[second_word_index]);
    uint32_t const third_word   = __ldg(&word_input[third_word_index]);

    uint64_t const lower_bits = __funnelshift_r(first_word, second_word, bit_idx);
    uint64_t const upper_bits = __funnelshift_r(second_word, third_word, bit_idx);
    uint64_t const mask       = bit_width == 64 ? ~uint64_t{0} : ((1ull << bit_width) - 1);

    decoded_value = (lower_bits | (upper_bits << 32)) & mask;
  }
}

// Decodes 32 values and writes them to output_ptr. word_input is aligned, bit_alignment specifies
// its initial bit offset, min_delta is the Parquet block's delta offset, and offset_value is the
// running value. decoded_count, skip_count, and decode_end control the output range.
//
// Returns the number of input words consumed.
template <bool FAST>
__device__ __forceinline__ static int handle_32_elements(uint32_t const* word_input,
                                                         int bit_width,
                                                         int bit_alignment,
                                                         int lane_id,
                                                         int64_t min_delta,
                                                         int64_t& offset_value,
                                                         int decoded_count,
                                                         int skip_count,
                                                         int decode_end,
                                                         int64_t* __restrict__ output_ptr)
{
  using value_type = decode_result_t<FAST>;
  value_type decoded;
  extract_bitpack<FAST>(word_input, bit_width, lane_id, bit_alignment, decoded);

  // The decoded values are int64_t, but shuffle instructions operate on 32-bit words. When
  // bit_width is at most max_fast_bit_width, the sum of 32 elements fits in 32 bits, requiring one
  // shuffle per scan stage.
  value_type prefix_sum = decoded;
#pragma unroll
  for (int jump = 1; jump < warp_size; jump *= 2) {
    if constexpr (FAST) {
      // Taken from cub/cub/warp/specializations/warp_scan_shfl.cuh.
      asm volatile(
        "{"
        ".reg .pred p;"
        ".reg .b32 y;"
        "shfl.sync.up.b32 y|p, %0, %1, 0, 0xffffffff;"
        "@p add.u32 %0, %0, y;"
        "}"
        : "+r"(prefix_sum)
        : "r"(jump));
    } else {
      int64_t const partial = __shfl_up_sync(full_mask, prefix_sum, jump);
      if (lane_id >= jump) { prefix_sum = wrapping_i64_add(prefix_sum, partial); }
    }
  }

  // A separate warp-wide reduction avoids the dependency on lane 31's final prefix sum, keeping the
  // offset adjustment off the prefix-scan critical path.
  value_type full_sum;
  if constexpr (FAST) {
    full_sum = __reduce_add_sync(full_mask, decoded);
  } else {
    full_sum = __shfl_sync(full_mask, prefix_sum, warp_size - 1);
  }

  // Applying the int64_t minimum delta and offset after the scan preserves the 32-bit shuffle
  // optimization.
  int64_t const min_delta_lane =
    wrapping_i64_multiply(static_cast<int64_t>(1 + lane_id), min_delta);
  int64_t const delta = wrapping_i64_add(static_cast<int64_t>(prefix_sum), min_delta_lane);
  int64_t const value = wrapping_i64_add(delta, offset_value);

  int64_t const min_delta_total = wrapping_i64_multiply(static_cast<int64_t>(warp_size), min_delta);
  int64_t const new_offset_value =
    wrapping_i64_add(static_cast<int64_t>(full_sum), min_delta_total);
  offset_value = wrapping_i64_add(offset_value, new_offset_value);

  // Bounds are checked once per fast-path 128-element chunk, to avoid unnecessary checks here.
  if (FAST || (decoded_count + lane_id >= skip_count && decoded_count + lane_id < decode_end)) {
    __stcs(&output_ptr[decoded_count - skip_count + lane_id], value);
  }
  return vals_per_miniblock_multiple * bit_width / bits_per_word;
}

// Decodes supported pages using one warp per page.
CUDF_KERNEL void __launch_bounds__(block_size) decode(PageInfo const* __restrict pages,
                                                      int n_pages,
                                                      ColumnChunkDesc const* __restrict chunks,
                                                      size_t min_row,
                                                      size_t num_rows,
                                                      bool const* __restrict page_mask)
{
  // The shuffle is semantically unnecessary but proves to the compiler that warp_id is
  // warp-uniform.
  int const warp_id = __shfl_sync(full_mask, threadIdx.x / warp_size, 0);
  int const lane_id = threadIdx.x % warp_size;

  int const page_idx = blockIdx.x * (block_size / warp_size) + warp_id;
  if (page_idx >= n_pages) { return; }

  PageInfo const page_info         = pages[page_idx];
  bool const page_selected         = page_mask == nullptr || page_mask[page_idx];
  ColumnChunkDesc const chunk_info = chunks[page_info.chunk_idx];
  if (!supported(page_info, chunk_info)) { return; }

  // Skip pages outside [min_row, min_row + num_rows). Because each CTA contains multiple warps,
  // interleaving selected and unselected pages may increase the number of active CTAs.
  int64_t const abs_page_base_row = chunk_info.start_row + page_info.chunk_row;
  if (!page_selected || abs_page_base_row + page_info.num_input_values <= min_row ||
      abs_page_base_row >= min_row + num_rows) {
    return;
  }

  int64_t* __restrict__ const output_ptr =
    static_cast<int64_t*>(chunk_info.column_data_base[0]) +
    max(int64_t{0}, static_cast<int64_t>(abs_page_base_row - min_row));
  uint8_t const* __restrict__ input_ptr =
    page_info.page_data +
    ((page_info.flags & PAGEINFO_FLAGS_V2)
       ? page_info.lvl_bytes[level_type::REPETITION] + page_info.lvl_bytes[level_type::DEFINITION]
       : 0);
  int const skip_count  = max(int64_t{0}, static_cast<int64_t>(min_row - abs_page_base_row));
  int const valid_count = min(static_cast<int64_t>(page_info.num_input_values),
                              static_cast<int64_t>(min_row + num_rows - abs_page_base_row));

  // Parse the page header.
  int64_t offset_value;
  uint32_t encoded_values_per_block, encoded_miniblocks_per_block, total_value_count;
  int64_t encoded_first_value;

  input_ptr += read_uleb128<true>(input_ptr, lane_id, encoded_values_per_block);
  int const values_per_block = static_cast<int>(encoded_values_per_block);
  __builtin_assume(values_per_block % vals_per_block_multiple == 0);
  input_ptr += read_uleb128<true>(input_ptr, lane_id, encoded_miniblocks_per_block);
  int const miniblocks_per_block = static_cast<int>(encoded_miniblocks_per_block);
  input_ptr += read_uleb128<true>(input_ptr, lane_id, total_value_count);
  input_ptr += read_uleb128<false>(input_ptr, lane_id, encoded_first_value);
  offset_value = decode_zigzag(encoded_first_value);

  int const values_per_miniblock = values_per_block / miniblocks_per_block;
  __builtin_assume(values_per_miniblock % vals_per_miniblock_multiple == 0);

  // The first value requires no delta decoding; it is the initial offset value.
  int decoded_count = 0;
  if (skip_count == 0 && lane_id == 0) { __stcs(&output_ptr[0], offset_value); }
  decoded_count += 1;

  int const decode_end = valid_count;
  // Main block decoding loop.
  while (decoded_count < decode_end) {
    int64_t encoded_min_delta = 0;
    input_ptr += read_uleb128<false>(input_ptr, lane_id, encoded_min_delta);

    int64_t const min_delta         = decode_zigzag(encoded_min_delta);
    uint8_t const* const bit_widths = input_ptr;
    uint8_t const* const data       = input_ptr + miniblocks_per_block;

    int const align_bytes     = reinterpret_cast<uintptr_t>(data) & (word_alignment - 1);
    uint32_t const* word_data = reinterpret_cast<uint32_t const*>(data - align_bytes);

    // Collect the bit width for each 32-value group and determine whether the fast path applies:
    // (1) all 128 values are within the output range;
    // (2) all corresponding bit widths are at most max_fast_bit_width.
    int miniblock_group_count = 0;
    int miniblock_index       = 0;
    for (int block_value_offset = 0; block_value_offset < values_per_block;
         block_value_offset += vals_per_block_multiple) {
      int group_bit_widths[vals_per_block_multiple / vals_per_miniblock_multiple];
      bool fast_path =
        decoded_count >= skip_count && decoded_count + vals_per_block_multiple <= decode_end;

      // A 128-value group may cross miniblock boundaries, for example when values_per_miniblock is
      // 96 and values_per_block is 384. Track the miniblock with counters to avoid division and
      // modulo.
#pragma unroll
      for (int group_index = 0; group_index < vals_per_block_multiple / vals_per_miniblock_multiple;
           ++group_index) {
        int const current_miniblock = miniblock_index;
        if (++miniblock_group_count == values_per_miniblock / vals_per_miniblock_multiple) {
          ++miniblock_index;
          miniblock_group_count = 0;
        }

        // Many one-byte bit widths fit in one cache sector, so these loads should frequently hit
        // cache.
        group_bit_widths[group_index] = __ldg(&bit_widths[current_miniblock]);

        fast_path &= group_bit_widths[group_index] <= max_fast_bit_width;
      }

      // Dispatch at coarse granularity to reduce branching.
      if (__builtin_expect(fast_path, true)) {
#pragma unroll
        for (int group_index = 0;
             group_index < vals_per_block_multiple / vals_per_miniblock_multiple;
             ++group_index) {
          word_data +=
            handle_32_elements<true>(word_data,
                                     group_bit_widths[group_index],
                                     align_bytes * 8,
                                     lane_id,
                                     min_delta,
                                     offset_value,
                                     decoded_count + group_index * vals_per_miniblock_multiple,
                                     skip_count,
                                     decode_end,
                                     output_ptr);
        }
        decoded_count += vals_per_block_multiple;
      } else {
        for (int group_index = 0;
             group_index < vals_per_block_multiple / vals_per_miniblock_multiple &&
             decoded_count < decode_end;
             ++group_index) {
          word_data += handle_32_elements<false>(word_data,
                                                 group_bit_widths[group_index],
                                                 align_bytes * 8,
                                                 lane_id,
                                                 min_delta,
                                                 offset_value,
                                                 decoded_count,
                                                 skip_count,
                                                 decode_end,
                                                 output_ptr);
          decoded_count += vals_per_miniblock_multiple;
        }
      }
      input_ptr = reinterpret_cast<uint8_t const*>(word_data) + align_bytes;
    }
  }
}

}  // namespace delta_fastpath

constexpr int decode_block_size              = 128;
constexpr int decode_delta_binary_block_size = 96;

// Size of the ring buffer that maps leaf-value ordinals to output rows (nz_idx). The level
// decoder runs up to two batches ahead of the value consumer and, on nested pages, overshoots
// its target by up to a warp of values, so this needs to exceed 3 * delta_max_batch_size +
// warp_size; anything smaller lets the level decoder wrap onto entries the consumer is reading.
constexpr int delta_nz_buf_size = 4 * delta_max_batch_size;

// DELTA_BYTE_ARRAY encoding (incremental encoding or front compression), is used for BYTE_ARRAY
// columns. For each element in a sequence of strings, a prefix length from the preceding string
// and a suffix is stored. The prefix lengths are DELTA_BINARY_PACKED encoded. The suffixes are
// encoded with DELTA_LENGTH_BYTE_ARRAY encoding, which is a DELTA_BINARY_PACKED list of suffix
// lengths, followed by the concatenated suffix data.
struct delta_byte_array_decoder {
  uint8_t const* last_string;       // pointer to last decoded string...needed for its prefix
  uint8_t const* suffix_char_data;  // pointer to the start of character data

  uint8_t* temp_buf;         // scratch for strings skipped over by a leading row range; the next
                             // batch overwrites it from its start each round
  uint8_t* prefix_seed;      // one reserved slot ahead of temp_buf holding a durable copy of the
                             // last decoded string, used to seed the next batch's first prefix
  uint32_t start_val;        // decoded strings up to this index will be dumped to temp_buf
  uint32_t last_string_len;  // length of the last decoded string

  delta_binary_decoder prefixes;  // state of decoder for prefix lengths
  delta_binary_decoder suffixes;  // state of decoder for suffix lengths

  // initialize the prefixes and suffixes blocks
  __device__ void init(
    uint8_t const* start, uint8_t const* end, uint32_t start_idx, uint8_t* temp, size_t temp_size)
  {
    auto const* suffix_start = prefixes.find_end_of_block(start, end);
    suffix_char_data         = suffixes.find_end_of_block(suffix_start, end);
    last_string              = nullptr;
    // the temp allocation holds one leading string slot (see the string-size prepass) followed by
    // delta_max_batch_size scratch slots. reserve the leading slot for the last decoded string so
    // it stays clear of the scratch, which each round overwrites from its start.
    prefix_seed = temp;
    temp_buf    = temp + temp_size / (delta_max_batch_size + 1);
    start_val   = start_idx;
  }

  // kind of like an inclusive scan for strings. takes prefix_len bytes from preceding
  // string and prepends to the suffix we've already copied into place. called from
  // within loop over values_in_mb, so this only needs to handle a single warp worth of data
  // at a time.
  __device__ void string_scan(uint8_t* strings_out,
                              uint8_t const* last_string,
                              uint32_t start_idx,
                              uint32_t end_idx,
                              uint32_t offset,
                              uint32_t lane_id)
  {
    using cudf::detail::warp_size;

    // let p(n) === length(prefix(string_n))
    //
    // if p(n-1) > p(n), then string_n can be completed when string_n-2 is completed. likewise if
    // p(m) > p(n), then string_n can be completed with string_m-1. however, if p(m) < p(n), then m
    // is a "blocker" for string_n; string_n can be completed only after string_m is.
    //
    // we will calculate the nearest blocking position for each lane, and then fill in string_0. we
    // then iterate, finding all lanes that have had their "blocker" filled in and completing them.
    // when all lanes are filled in, we return. this will still hit the worst case if p(n-1) < p(n)
    // for all n
    __shared__ __align__(8) int64_t prefix_lens[warp_size];
    __shared__ __align__(8) uint8_t const* offsets[warp_size];

    uint32_t const ln_idx   = start_idx + lane_id;
    uint64_t prefix_len     = ln_idx < end_idx ? prefixes.value_at(ln_idx) : 0;
    uint8_t* const lane_out = ln_idx < end_idx ? strings_out + offset : nullptr;

    // if all prefix_len's are zero, then there's nothing to do
    if (__all_sync(0xffff'ffff, prefix_len == 0)) { return; }

    prefix_lens[lane_id] = prefix_len;
    offsets[lane_id]     = lane_out;
    __syncwarp();

    // find a neighbor to the left that has a prefix length less than this lane. once that
    // neighbor is complete, this lane can be completed.
    int blocker = lane_id - 1;
    while (blocker > 0 && prefix_lens[blocker] != 0 && prefix_len <= prefix_lens[blocker]) {
      blocker--;
    }

    // fill in lane 0 (if necessary)
    if (lane_id == 0 && prefix_len > 0) {
      memcpy(lane_out, last_string, prefix_len);
      prefix_lens[0] = prefix_len = 0;
    }
    __syncwarp();

    // now fill in blockers until done
    for (uint32_t i = 1; i < warp_size && i + start_idx < end_idx; i++) {
      // record whether this lane's blocker is complete before any lane writes below
      // prevents race condition accessing prefix_lens
      bool const completed = prefix_len != 0 && prefix_lens[blocker] == 0 && lane_out != nullptr;
      __syncwarp();
      if (completed) {
        memcpy(lane_out, offsets[blocker], prefix_len);
        prefix_lens[lane_id] = prefix_len = 0;
      }
      __syncwarp();

      // check for finished
      if (__all_sync(0xffff'ffff, prefix_len == 0)) { return; }
    }
  }

  // calculate a mini-batch of string values, writing the results to
  // `strings_out`. starting at global index `start_idx` and decoding
  // up to `num_values` strings.
  // called by all threads in a warp. used for strings <= 32 chars.
  // returns number of bytes written
  __device__ size_t calculate_string_values(uint8_t* strings_out,
                                            uint32_t start_idx,
                                            uint32_t num_values,
                                            uint32_t lane_id)
  {
    using cudf::detail::warp_size;
    using WarpScan = cub::WarpScan<uint64_t>;
    __shared__ WarpScan::TempStorage scan_temp;

    if (start_idx >= suffixes.value_count) { return 0; }
    auto end_idx = start_idx + min(suffixes.values_per_mb, num_values);
    end_idx      = min(end_idx, static_cast<uint32_t>(suffixes.value_count));

    auto p_strings_out = strings_out;
    auto p_temp_out    = temp_buf;

    auto copy_batch = [&](uint8_t* out, uint32_t idx, uint32_t end) {
      uint32_t const ln_idx = idx + lane_id;

      // calculate offsets into suffix data
      uint64_t const suffix_len = ln_idx < end ? suffixes.value_at(ln_idx) : 0;
      uint64_t suffix_off       = 0;
      WarpScan(scan_temp).ExclusiveSum(suffix_len, suffix_off);

      // calculate offsets into string data
      uint64_t const prefix_len = ln_idx < end ? prefixes.value_at(ln_idx) : 0;
      uint64_t const string_len = prefix_len + suffix_len;

      // get offset into output for each lane
      uint64_t string_off, warp_total;
      WarpScan(scan_temp).ExclusiveSum(string_len, string_off, warp_total);
      auto const so_ptr = out + string_off;

      // copy suffixes into string data
      if (ln_idx < end) { memcpy(so_ptr + prefix_len, suffix_char_data + suffix_off, suffix_len); }
      __syncwarp();

      // copy prefixes into string data.
      string_scan(out, last_string, idx, end, string_off, lane_id);
      __syncwarp();

      // save the position of the last computed string. this will be used in
      // the next iteration to reconstruct the string in lane 0.
      if (ln_idx == end - 1 || (ln_idx < end && lane_id == 31)) {
        // set last_string to this lane's string
        last_string     = out + string_off;
        last_string_len = string_len;
        // and consume used suffix_char_data
        suffix_char_data += suffix_off + suffix_len;
      }

      return warp_total;
    };

    uint64_t string_total = 0;
    for (int idx = start_idx; idx < end_idx; idx += warp_size) {
      auto const n_in_batch = min(warp_size, end_idx - idx);
      // account for the case where start_val occurs in the middle of this batch
      if (idx < start_val && idx + n_in_batch > start_val) {
        // dump idx...start_val into temp_buf
        copy_batch(p_temp_out, idx, start_val);
        __syncwarp();

        // start_val...idx + n_in_batch into strings_out
        auto nbytes = copy_batch(p_strings_out, start_val, idx + n_in_batch);
        p_strings_out += nbytes;
        string_total = nbytes;
      } else {
        if (idx < start_val) {
          p_temp_out += copy_batch(p_temp_out, idx, end_idx);
        } else {
          auto nbytes = copy_batch(p_strings_out, idx, end_idx);
          p_strings_out += nbytes;
          string_total += nbytes;
        }
      }
      __syncwarp();
    }

    // the next batch overwrites the temp scratch from its start, so if the last decoded string
    // lives there, preserve it in the reserved seed slot ahead of the scratch
    if (end_idx <= start_val && last_string != prefix_seed) {
      // snapshot before lane 0 overwrites last_string below; without this sync, other lanes'
      // (unused but still issued) reads of last_string can race with lane 0's write of it.
      uint8_t const* const last_string_snapshot = last_string;
      uint32_t const last_string_len_snapshot   = last_string_len;
      __syncwarp();
      if (lane_id == 0) {
        memcpy(prefix_seed, last_string_snapshot, last_string_len_snapshot);
        last_string = prefix_seed;
      }
      __syncwarp();
    }

    return string_total;
  }

  // character parallel version of CalculateStringValues(). This is faster for strings longer than
  // 32 chars.
  __device__ size_t calculate_string_values_cp(uint8_t* strings_out,
                                               uint32_t start_idx,
                                               uint32_t num_values,
                                               uint32_t lane_id)
  {
    using cudf::detail::warp_size;
    __shared__ __align__(8) uint8_t* so_ptr;

    if (start_idx >= suffixes.value_count) { return 0; }
    auto end_idx = start_idx + min(suffixes.values_per_mb, num_values);
    end_idx      = min(end_idx, static_cast<uint32_t>(suffixes.value_count));

    if (lane_id == 0) { so_ptr = start_idx < start_val ? temp_buf : strings_out; }
    __syncwarp();

    uint64_t string_total = 0;
    for (int idx = start_idx; idx < end_idx; idx++) {
      uint64_t const suffix_len = suffixes.value_at(idx);
      uint64_t const prefix_len = prefixes.value_at(idx);
      uint64_t const string_len = prefix_len + suffix_len;

      // copy prefix and suffix data into current strings_out position
      // for longer strings use a 4-byte version stolen from gather_chars_fn_string_parallel.
      if (string_len > 64) {
        if (prefix_len > 0) { wideStrcpy(so_ptr, last_string, prefix_len, lane_id); }
        if (suffix_len > 0) {
          wideStrcpy(so_ptr + prefix_len, suffix_char_data, suffix_len, lane_id);
        }
      } else {
        for (int i = lane_id; i < string_len; i += warp_size) {
          so_ptr[i] = i < prefix_len ? last_string[i] : suffix_char_data[i - prefix_len];
        }
      }
      __syncwarp();

      if (idx >= start_val) { string_total += string_len; }

      if (lane_id == 0) {
        last_string     = so_ptr;
        last_string_len = string_len;
        suffix_char_data += suffix_len;
        if (idx == start_val - 1) {
          so_ptr = strings_out;
        } else {
          so_ptr += string_len;
        }
      }
      __syncwarp();
    }

    // the next batch overwrites the temp scratch from its start, so if the last decoded string
    // lives there, preserve it in the reserved seed slot ahead of the scratch
    if (end_idx <= start_val && last_string != prefix_seed) {
      // snapshot before lane 0 overwrites last_string below; without this sync, other lanes'
      // (unused but still issued) reads of last_string can race with lane 0's write of it.
      uint8_t const* const last_string_snapshot = last_string;
      uint32_t const last_string_len_snapshot   = last_string_len;
      __syncwarp();
      if (lane_id == 0) {
        memcpy(prefix_seed, last_string_snapshot, last_string_len_snapshot);
        last_string = prefix_seed;
      }
      __syncwarp();
    }

    return string_total;
  }

  // dump strings before start_val to temp buf. decodes one warp_size-wide pass per round, so
  // any mini-block size is supported. called by all threads in a thread block.
  __device__ void skip(bool use_char_ll,
                       cg::thread_block const& block,
                       cg::thread_block_tile<cudf::detail::warp_size, cg::thread_block> const& warp)
  {
    using cudf::detail::warp_size;

    // is this even necessary? return if asking to skip the whole block.
    if (start_val >= prefixes.num_encoded_values(true)) { return; }

    uint32_t skip_pos = 0;
    while (skip_pos < start_val) {
      // warp 0 decodes a pass of prefixes and warp 1 a pass of suffixes. this will potentially
      // decode past start_val, and those values stay resident in the rolling buffers for the
      // decode loop that follows. dispatch on a compile-time-constant decoder per warp (as the
      // main decode loop does) rather than a runtime-selected `db` pointer: selecting the object
      // at runtime makes the compiler speculatively load both prefixes and suffixes on every
      // `db->` access, so the suffix warp reads prefix state (and vice versa) while the other
      // warp writes it, which compute-sanitizer racecheck reports as a cross-warp hazard.
      if (warp.meta_group_rank() == 0) {
        prefixes.decode_next_pass(warp);
      } else if (warp.meta_group_rank() == 1) {
        suffixes.decode_next_pass(warp);
      }
      block.sync();

      // warp 0 reconstructs this round's skipped strings into the temp scratch (the helpers
      // preserve the round's last string past the scratch area for the next round's prefixes)
      if (warp.meta_group_rank() == 0) {
        auto const num_to_decode = min(static_cast<uint32_t>(warp_size), start_val - skip_pos);
        if (use_char_ll) {
          calculate_string_values_cp(temp_buf, skip_pos, num_to_decode, warp.thread_rank());
        } else {
          calculate_string_values(temp_buf, skip_pos, num_to_decode, warp.thread_rank());
        }
      }
      skip_pos += warp_size;
      block.sync();
    }
  }
};

// Decode page data that is DELTA_BINARY_PACKED encoded. This encoding is
// only used for int32 and int64 physical types (and appears to only be used
// with V2 page headers; see https://www.mail-archive.com/dev@parquet.apache.org/msg11826.html).
// this kernel only needs 96 threads (3 warps)(for now).
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_delta_binary_block_size)
  decode_delta_binary_kernel(PageInfo* pages,
                             device_span<ColumnChunkDesc const> chunks,
                             size_t min_row,
                             size_t num_rows,
                             cudf::device_span<bool const> page_mask,
                             kernel_error::pointer error_code)
{
  __shared__ __align__(16) delta_binary_decoder db_state;
  __shared__ __align__(16) full_page_decode_state state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_nz_buf_size, 1, 1> state_buffers;

  auto* const s      = &state_g;
  auto* const sb     = &state_buffers;
  int const page_idx = cg::this_grid().block_rank();
  auto const block   = cg::this_thread_block();
  auto const warp    = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto* const db     = &db_state;

  // Exit early if the page is pruned
  if (page_mask.size() > 0 and not page_mask[page_idx]) { return; }
  if (delta_fastpath::supported(pages[page_idx], chunks[pages[page_idx].chunk_idx])) {
    return;
  }

  [[maybe_unused]] null_count_back_copier _{s, static_cast<int>(block.thread_rank())};

  // Setup local page info
  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::DELTA_BINARY},
                             page_processing_stage::DECODE)) {
    return;
  }

  // Must be evaluated after setup_local_page_info
  bool const has_repetition = s->setup.col.max_level[level_type::REPETITION] > 0;
  bool const process_nulls  = should_process_nulls(s);

  // Capture initial valid_map_offset before any processing that might modify it
  int const init_valid_map_offset =
    s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  // copying logic from gpuDecodePageData.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting.nesting_info;

  // Get the level decode buffers for this page
  PageInfo* pp       = &pages[page_idx];
  level_t* const def = !process_nulls
                         ? nullptr
                         : reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  auto* const rep    = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t const skipped_leaf_values = s->setup.page.skipped_leaf_values;

  // initialize delta state
  if (block.thread_rank() == 0) { db->init_binary_block(s->stream.data_start, s->stream.data_end); }
  block.sync();

  if (db->error) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    }
    return;
  }

  bool const is_skip_resume = skipped_leaf_values > 0;

  // Number of values produced per main-loop iteration: up to two warp_size passes, so pages whose
  // mini-blocks hold at least two passes keep the schedule of the whole-mini-block decoder. When
  // resuming after skip_values() the producer emits a single pass per iteration: the skip leaves
  // up to warp_size not-yet-consumed values in the rolling buffer, and a larger batch could wrap
  // around and overwrite them before the consumer reads them.
  uint32_t const batch_size =
    is_skip_resume ? cudf::detail::warp_size
                   : min(db->values_per_mb, static_cast<uint32_t>(delta_max_batch_size));
  uint32_t const passes_per_batch = batch_size / cudf::detail::warp_size;

  // if skipped_leaf_values is non-zero, then we need to decode up to the first mini-block
  // that has a value we need.
  if (is_skip_resume) { db->skip_values(skipped_leaf_values, block, warp); }

  while (s->setup.error == 0 && (s->progress.input_value_count < s->setup.num_input_values ||
                                 s->progress.src_pos < s->progress.nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->progress.src_pos;

    if (warp.meta_group_rank() < 2) {  // warp0..1
      target_pos = min(src_pos + 2 * batch_size, s->progress.nz_count + batch_size);
    } else {  // warp2
      target_pos = min(s->progress.nz_count, src_pos + batch_size);
    }
    // This needs to be here to prevent warp 2 modifying src_pos before all threads have read it
    block.sync();

    // warp0 will decode the rep/def levels, warp1 will unpack a mini-batch of deltas.
    // warp2 waits one cycle for warps 0/1 to produce a batch, and then stuffs values
    // into the proper location in the output.
    if (warp.meta_group_rank() == 0) {
      // warp 0
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<delta_nz_buf_size, level_t>(s, sb, target_pos, rep, def, warp);
    } else if (warp.meta_group_rank() == 1) {
      // warp 1
      for (uint32_t i = 0; i < passes_per_batch; i++) {
        // make lane 0's state updates from the previous pass visible to the whole warp; the
        // block-wide sync below covers the last pass of the iteration
        if (i > 0) { warp.sync(); }
        db->decode_next_pass(warp);
      }
    } else if (src_pos < target_pos) {
      // warp 2
      // nesting level that is storing actual leaf values
      int const leaf_level_index = s->setup.col.max_nesting_depth - 1;

      // process the mini-block using warps
      for (uint32_t sp = src_pos + warp.thread_rank(); sp < src_pos + batch_size;
           sp += warp.size()) {
        // the position in the output column/buffer
        int32_t dst_pos = sb->nz_idx[rolling_index<delta_nz_buf_size>(sp)];

        // handle skip_rows here. flat hierarchies can just skip up to first_row.
        if (!has_repetition) { dst_pos -= s->setup.first_row; }

        // place value for this thread
        if (dst_pos >= 0 && sp < target_pos) {
          void* const dst =
            nesting_info_base[leaf_level_index].data_out + dst_pos * s->output_cvt.dtype_len;
          auto const val = db->value_at(sp + skipped_leaf_values);
          switch (s->output_cvt.dtype_len) {
            case 1: *static_cast<int8_t*>(dst) = val; break;
            case 2: *static_cast<int16_t*>(dst) = val; break;
            case 4: *static_cast<int32_t*>(dst) = val; break;
            case 8: *static_cast<int64_t*>(dst) = val; break;
          }
        }
      }
      if (warp.thread_rank() == 0) { s->progress.src_pos = src_pos + batch_size; }
    }

    block.sync();
  }

  if (has_repetition) {
    // Zero-fill null positions after decoding valid values
    auto const& ni = s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1];
    if (ni.valid_map != nullptr) {
      int const num_values = ni.valid_map_offset - init_valid_map_offset;
      zero_fill_null_positions_shared<decode_delta_binary_block_size>(
        s,
        s->output_cvt.dtype_len,
        init_valid_map_offset,
        num_values,
        static_cast<int>(block.thread_rank()));
    }
  }

  if (block.thread_rank() == 0 and s->setup.error != 0) { set_error(s->setup.error, error_code); }
}

// Decode page data that is DELTA_BYTE_ARRAY packed. This encoding consists of a DELTA_BINARY_PACKED
// array of prefix lengths, followed by a DELTA_BINARY_PACKED array of suffix lengths, followed by
// the suffixes (technically the suffixes are DELTA_LENGTH_BYTE_ARRAY encoded). The latter two can
// be used to create an offsets array for the suffix data, but then this needs to be combined with
// the prefix lengths to do the final decode for each value. Because the lengths of the prefixes and
// suffixes are not encoded in the header, we're going to have to first do a quick pass through them
// to find the start/end of each structure.
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  decode_delta_byte_array_kernel(PageInfo* pages,
                                 device_span<ColumnChunkDesc const> chunks,
                                 size_t min_row,
                                 size_t num_rows,
                                 cudf::device_span<bool const> page_mask,
                                 cudf::device_span<size_t> initial_str_offsets,
                                 kernel_error::pointer error_code)
{
  __shared__ __align__(16) delta_byte_array_decoder db_state;
  __shared__ __align__(16) full_page_decode_state state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_nz_buf_size, 1, 1> state_buffers;

  auto* const s         = &state_g;
  auto* const sb        = &state_buffers;
  int const page_idx    = cg::this_grid().block_rank();
  auto const block      = cg::this_thread_block();
  auto const warp       = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto* const prefix_db = &db_state.prefixes;
  auto* const suffix_db = &db_state.suffixes;
  auto* const dba       = &db_state;
  if (page_mask.size() > 0 and not page_mask[page_idx]) { return; }
  [[maybe_unused]] null_count_back_copier _{s, static_cast<int>(block.thread_rank())};

  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{decode_kernel_mask::DELTA_BYTE_ARRAY},
                             page_processing_stage::DECODE)) {
    return;
  }

  if (s->setup.col.logical_type.has_value() &&
      s->setup.col.logical_type->type == LogicalType::DECIMAL) {
    // we cannot read decimal encoded with DELTA_BYTE_ARRAY yet
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_DATA_TYPE), error_code);
    }
    return;
  }

  bool const has_repetition = s->setup.col.max_level[level_type::REPETITION] > 0;
  bool const process_nulls  = should_process_nulls(s);

  // Capture initial valid_map_offset before any processing that might modify it
  int const init_valid_map_offset =
    s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  // choose a character parallel string copy when the average string is longer than a warp
  auto const use_char_ll =
    s->setup.page.num_valids > 0 &&
    (s->setup.page.str_bytes / s->setup.page.num_valids) > cudf::detail::warp_size;

  // copying logic from decode_page_data.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting.nesting_info;

  // Get the level decode buffers for this page
  PageInfo* pp       = &pages[page_idx];
  level_t* const def = !process_nulls
                         ? nullptr
                         : reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  auto* const rep    = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t const skipped_leaf_values = s->setup.page.skipped_leaf_values;

  if (block.thread_rank() == 0) {
    // initialize the prefixes and suffixes blocks
    dba->init(s->stream.data_start,
              s->stream.data_end,
              s->setup.page.start_val,
              s->setup.page.temp_string_buf,
              s->setup.page.temp_string_size);
  }
  block.sync();

  // Propagate malformed-header errors from either underlying DELTA_BINARY_PACKED decoder.
  if (prefix_db->error or suffix_db->error) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    }
    return;
  }

  // assert that prefix and suffix have same mini-block size
  if (prefix_db->values_per_mb != suffix_db->values_per_mb or
      prefix_db->block_size != suffix_db->block_size or
      prefix_db->value_count != suffix_db->value_count) {
    set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAM_MISMATCH),
              error_code);
    return;
  }

  // pointer to location to output final strings
  int const leaf_level_index = s->setup.col.max_nesting_depth - 1;
  auto strings_data          = nesting_info_base[leaf_level_index].string_out;

  // if this is a bounds page and nested, then we need to skip up front. non-nested will work
  // its way through the page.
  int string_pos = has_repetition ? s->setup.page.start_val : 0;
  auto const is_bounds_pg =
    is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition);
  bool const is_skip_resume = is_bounds_pg and string_pos > 0;

  // Number of values produced per main-loop iteration (see decode_delta_binary_kernel for why
  // skip-resume pages must produce a single warp_size pass per iteration).
  uint32_t const batch_size =
    is_skip_resume ? cudf::detail::warp_size
                   : min(prefix_db->values_per_mb, static_cast<uint32_t>(delta_max_batch_size));
  uint32_t const passes_per_batch = batch_size / cudf::detail::warp_size;

  if (is_skip_resume) { dba->skip(use_char_ll, block, warp); }

  while (!s->setup.error && (s->progress.input_value_count < s->setup.num_input_values ||
                             s->progress.src_pos < s->progress.nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->progress.src_pos;

    if (warp.meta_group_rank() < 3) {  // warp 0..2
      target_pos =
        min(src_pos + 2 * batch_size, s->progress.nz_count + s->setup.first_row + batch_size);
    } else {  // warp 3
      target_pos = min(s->progress.nz_count, src_pos + batch_size);
    }
    // this needs to be here to prevent warp 3 modifying src_pos before all threads have read it
    block.sync();

    // warp0 will decode the rep/def levels, warp1 will unpack a mini-batch of prefixes, warp 2 will
    // unpack a mini-batch of suffixes. warp3 waits one cycle for warps 0-2 to produce a batch, and
    // then stuffs values into the proper location in the output.
    if (warp.meta_group_rank() == 0) {
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<delta_nz_buf_size, level_t>(s, sb, target_pos, rep, def, warp);
    } else if (warp.meta_group_rank() == 1) {
      // warp 1
      for (uint32_t i = 0; i < passes_per_batch; i++) {
        // make lane 0's state updates from the previous pass visible to the whole warp; the
        // block-wide sync below covers the last pass of the iteration
        if (i > 0) { warp.sync(); }
        prefix_db->decode_next_pass(warp);
      }
    } else if (warp.meta_group_rank() == 2) {
      // warp 2
      for (uint32_t i = 0; i < passes_per_batch; i++) {
        if (i > 0) { warp.sync(); }
        suffix_db->decode_next_pass(warp);
      }
    } else if (warp.meta_group_rank() == 3 and src_pos < target_pos) {
      // warp 3
      int const nproc = min(batch_size, s->setup.page.end_val - string_pos);
      strings_data +=
        use_char_ll
          ? dba->calculate_string_values_cp(strings_data, string_pos, nproc, warp.thread_rank())
          : dba->calculate_string_values(strings_data, string_pos, nproc, warp.thread_rank());
      string_pos += nproc;

      // Process the mini-block using warp 3
      for (uint32_t sp = src_pos + warp.thread_rank(); sp < src_pos + batch_size;
           sp += warp.size()) {
        // the position in the output column/buffer
        int dst_pos = sb->nz_idx[rolling_index<delta_nz_buf_size>(sp)];

        // handle skip_rows here. flat hierarchies can just skip up to first_row.
        if (!has_repetition) { dst_pos -= s->setup.first_row; }

        if (dst_pos >= 0 && sp < target_pos) {
          auto const offptr =
            reinterpret_cast<size_type*>(nesting_info_base[leaf_level_index].data_out) + dst_pos;
          auto const src_idx = sp + skipped_leaf_values;
          *offptr            = prefix_db->value_at(src_idx) + suffix_db->value_at(src_idx);
        }
        warp.sync();
      }

      if (warp.thread_rank() == 0) { s->progress.src_pos = src_pos + batch_size; }
    }

    block.sync();
  }

  // Zero-fill null positions after decoding valid values
  auto const& ni = s->nesting.nesting_info[leaf_level_index];
  if (ni.valid_map != nullptr) {
    int const num_values = ni.valid_map_offset - init_valid_map_offset;
    zero_fill_null_positions_shared<decode_block_size>(s,
                                                       sizeof(size_type),
                                                       init_valid_map_offset,
                                                       num_values,
                                                       static_cast<int>(block.thread_rank()));
  }

  // For large strings, update the initial string buffer offset to be used during large string
  // column construction. Otherwise, convert string sizes to final offsets.
  if (s->setup.col.is_large_string_col) {
    // page.chunk_idx are ordered by input_col_idx and row_group_idx respectively.
    auto const chunks_per_rowgroup = initial_str_offsets.size();
    auto const input_col_idx       = pages[page_idx].chunk_idx % chunks_per_rowgroup;
    if (has_repetition) {
      compute_initial_large_strings_offset<true>(s, initial_str_offsets[input_col_idx]);
    } else {
      compute_initial_large_strings_offset<false>(s, initial_str_offsets[input_col_idx]);
    }
  } else {
    if (has_repetition) {
      convert_small_string_lengths_to_offsets<decode_block_size, true>(s);
    } else {
      convert_small_string_lengths_to_offsets<decode_block_size, false>(s);
    }
  }

  if (block.thread_rank() == 0 and s->setup.error != 0) { set_error(s->setup.error, error_code); }
}

// Decode page data that is DELTA_LENGTH_BYTE_ARRAY packed. This encoding consists of a
// DELTA_BINARY_PACKED array of string lengths, followed by the string data.
template <typename level_t>
CUDF_KERNEL void __launch_bounds__(decode_block_size)
  decode_delta_length_byte_array_kernel(PageInfo* pages,
                                        device_span<ColumnChunkDesc const> chunks,
                                        size_t min_row,
                                        size_t num_rows,
                                        cudf::device_span<bool const> page_mask,
                                        cudf::device_span<size_t> initial_str_offsets,
                                        kernel_error::pointer error_code)
{
  __shared__ __align__(16) delta_binary_decoder db_state;
  __shared__ __align__(16) full_page_decode_state state_g;
  __shared__ __align__(16) page_state_buffers_s<delta_nz_buf_size, 1, 1> state_buffers;
  __shared__ __align__(8) uint8_t const* page_string_data;
  __shared__ size_t string_offset;

  auto* const s      = &state_g;
  auto* const sb     = &state_buffers;
  int const page_idx = cg::this_grid().block_rank();
  auto const block   = cg::this_thread_block();
  auto const warp    = cg::tiled_partition<cudf::detail::warp_size>(block);
  auto* const db     = &db_state;
  if (page_mask.size() > 0 and not page_mask[page_idx]) { return; }
  [[maybe_unused]] null_count_back_copier _{s, static_cast<int>(block.thread_rank())};

  auto const mask = decode_kernel_mask::DELTA_LENGTH_BA;
  if (!setup_local_page_info(s,
                             &pages[page_idx],
                             chunks,
                             min_row,
                             num_rows,
                             mask_filter{mask},
                             page_processing_stage::DECODE)) {
    return;
  }

  if (s->setup.col.logical_type.has_value() &&
      s->setup.col.logical_type->type == LogicalType::DECIMAL) {
    // we cannot read decimal encoded with DELTA_LENGTH_BYTE_ARRAY yet
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::INVALID_DATA_TYPE), error_code);
    }
    return;
  }

  bool const has_repetition = s->setup.col.max_level[level_type::REPETITION] > 0;
  bool const process_nulls  = should_process_nulls(s);

  // Capture initial valid_map_offset before any processing that might modify it
  int const init_valid_map_offset =
    s->nesting.nesting_info[s->setup.col.max_nesting_depth - 1].valid_map_offset;

  // copying logic from gpuDecodePageData.
  PageNestingDecodeInfo const* nesting_info_base = s->nesting.nesting_info;

  // Get the level decode buffers for this page
  PageInfo* pp       = &pages[page_idx];
  level_t* const def = !process_nulls
                         ? nullptr
                         : reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::DEFINITION]);
  auto* const rep    = reinterpret_cast<level_t*>(pp->lvl_decode_buf[level_type::REPETITION]);

  // skipped_leaf_values will always be 0 for flat hierarchies.
  uint32_t const skipped_leaf_values = s->setup.page.skipped_leaf_values;

  // initialize delta state
  if (block.thread_rank() == 0) {
    string_offset    = 0;
    page_string_data = db->find_end_of_block(s->stream.data_start, s->stream.data_end);
  }
  block.sync();

  // The decode loop below sub-batches each mini-block into warp_size-wide passes, so any mini-block
  // size is supported (see decode_next_pass).
  if (db->error) {
    if (block.thread_rank() == 0) {
      set_error(static_cast<kernel_error::value_type>(decode_error::DELTA_PARAMS_UNSUPPORTED),
                error_code);
    }
    return;
  }

  int const leaf_level_index = s->setup.col.max_nesting_depth - 1;

  // db->init_binary_block below resets db->values_per_mb
  block.sync();
  // if this is a bounds page, then we need to decode up to the first mini-block
  // that has a value we need, and set string_offset to the position of the first value in the
  // string data block.
  auto const is_bounds_pg =
    is_bounds_page(s->setup.page, s->setup.col.start_row, min_row, num_rows, has_repetition);
  bool const is_skip_resume = is_bounds_pg and s->setup.page.start_val > 0;

  // Only nested pages resume the decoder mid-page; flat pages re-init it below and can keep the
  // full batch. Mid-page resumption must produce a single warp_size pass per iteration (see
  // decode_delta_binary_kernel for why).
  bool const resumes_mid_page = is_skip_resume and has_repetition;
  uint32_t const batch_size =
    resumes_mid_page ? cudf::detail::warp_size
                     : min(db->values_per_mb, static_cast<uint32_t>(delta_max_batch_size));
  uint32_t const passes_per_batch = batch_size / cudf::detail::warp_size;

  if (is_skip_resume) {
    if (warp.meta_group_rank() == 0) {
      // string_off is only valid on thread 0
      auto const string_off = db->skip_values_and_sum(s->setup.page.start_val, warp);
      // Threads in the warp might diverge and read in skip_values_and_sum
      // after lane 0 reinits below.
      warp.sync();
      if (warp.thread_rank() == 0) {
        string_offset = string_off;

        // if there is no repetition, then we need to work through the whole page, so reset the
        // delta decoder to the beginning of the page
        if (not has_repetition) { db->init_binary_block(s->stream.data_start, s->stream.data_end); }
      }
    }
    block.sync();
  }

  int string_pos = has_repetition ? s->setup.page.start_val : 0;

  while (!s->setup.error && (s->progress.input_value_count < s->setup.num_input_values ||
                             s->progress.src_pos < s->progress.nz_count)) {
    uint32_t target_pos;
    uint32_t const src_pos = s->progress.src_pos;

    if (warp.meta_group_rank() < 2) {  // warp0..1
      target_pos = min(src_pos + 2 * batch_size, s->progress.nz_count + batch_size);
    } else {  // warp2
      target_pos = min(s->progress.nz_count, src_pos + batch_size);
    }
    // this needs to be here to prevent warp 2 modifying src_pos before all threads have read it
    __syncthreads();

    // warp0 will decode the rep/def levels, warp1 will unpack a mini-batch of deltas.
    // warp2 waits one cycle for warps 0/1 to produce a batch, and then stuffs string sizes
    // into the proper location in the output. warp 3 does nothing until it's time to copy
    // string data.
    if (warp.meta_group_rank() == 0) {
      // warp 0
      // decode repetition and definition levels.
      // - update validity vectors
      // - updates offsets (for nested columns)
      // - produces non-NULL value indices in s->nz_idx for subsequent decoding
      gpuDecodeLevels<delta_nz_buf_size, level_t>(s, sb, target_pos, rep, def, warp);
    } else if (warp.meta_group_rank() == 1) {
      // warp 1
      for (uint32_t i = 0; i < passes_per_batch; i++) {
        // make lane 0's state updates from the previous pass visible to the whole warp; the
        // block-wide sync below covers the last pass of the iteration
        if (i > 0) { warp.sync(); }
        db->decode_next_pass(warp);
      }
    } else if (warp.meta_group_rank() == 2 && src_pos < target_pos) {
      // warp 2
      int const nproc = min(batch_size, s->setup.page.end_val - string_pos);
      string_pos += nproc;

      // process the mini-block in batches of 32
      for (uint32_t sp = src_pos + warp.thread_rank(); sp < src_pos + batch_size;
           sp += warp.size()) {
        // the position in the output column/buffer
        int dst_pos = sb->nz_idx[rolling_index<delta_nz_buf_size>(sp)];

        // handle skip_rows here. flat hierarchies can just skip up to first_row.
        if (!has_repetition) { dst_pos -= s->setup.first_row; }

        // fill in offsets array
        if (dst_pos >= 0 && sp < target_pos) {
          auto const offptr =
            reinterpret_cast<size_type*>(nesting_info_base[leaf_level_index].data_out) + dst_pos;
          *offptr = db->value_at(sp + skipped_leaf_values);
        }
        warp.sync();
      }

      if (warp.thread_rank() == 0) { s->progress.src_pos = src_pos + batch_size; }
    }
    block.sync();
  }

  // Zero-fill null positions after decoding valid values
  auto const& ni = nesting_info_base[leaf_level_index];
  if (ni.valid_map != nullptr) {
    int const num_values = ni.valid_map_offset - init_valid_map_offset;
    zero_fill_null_positions_shared<decode_block_size>(s,
                                                       sizeof(size_type),
                                                       init_valid_map_offset,
                                                       num_values,
                                                       static_cast<int>(block.thread_rank()));
  }

  // For large strings, update the initial string buffer offset to be used during large string
  // column construction. Otherwise, convert string sizes to final offsets.
  if (s->setup.col.is_large_string_col) {
    // page.chunk_idx are ordered by input_col_idx and row_group_idx respectively.
    auto const chunks_per_rowgroup = initial_str_offsets.size();
    auto const input_col_idx       = pages[page_idx].chunk_idx % chunks_per_rowgroup;
    if (has_repetition) {
      compute_initial_large_strings_offset<true>(s, initial_str_offsets[input_col_idx]);
    } else {
      compute_initial_large_strings_offset<false>(s, initial_str_offsets[input_col_idx]);
    }
  } else {
    // convert string sizes to offsets
    if (has_repetition) {
      convert_small_string_lengths_to_offsets<decode_block_size, true>(s);
    } else {
      convert_small_string_lengths_to_offsets<decode_block_size, false>(s);
    }
  }

  // finally, copy the string data into place
  auto const dst = nesting_info_base[leaf_level_index].string_out;
  auto const src = page_string_data + string_offset;
  memcpy_block<decode_block_size, true>(dst, src, s->setup.page.str_bytes, block);

  if (block.thread_rank() == 0 and s->setup.error != 0) { set_error(s->setup.error, error_code); }
}

}  // anonymous namespace

/**
 * @copydoc cudf::io::parquet::detail::decode_delta_binary
 */
void decode_delta_binary(cudf::detail::hostdevice_span<PageInfo> pages,
                         cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                         size_t num_rows,
                         size_t min_row,
                         int level_type_size,
                         cudf::device_span<bool const> page_mask,
                         kernel_error::pointer error_code,
                         cuda::stream_ref stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const v3_block(delta_fastpath::block_size, 1);
  dim3 const v3_grid((pages.size() + (delta_fastpath::block_size / delta_fastpath::warp_size) - 1) /
                       (delta_fastpath::block_size / delta_fastpath::warp_size),
                     1);
  // Dispatch the fast path kernel first, which only processes pages it can handle. The regular kernel
  // runs on the remaining pages.
  delta_fastpath::decode<<<v3_grid, v3_block, 0, stream.get()>>>(
    pages.device_ptr(), static_cast<int>(pages.size()), chunks.device_ptr(), min_row, num_rows,
    page_mask.empty() ? nullptr : page_mask.data());
  CUDF_CUDA_TRY(cudaGetLastError());

  dim3 dim_block(decode_delta_binary_block_size, 1);
  dim3 dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    decode_delta_binary_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_delta_binary_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

/**
 * @copydoc cudf::io::parquet::gpu::decode_delta_byte_array
 */
void decode_delta_byte_array(cudf::detail::hostdevice_span<PageInfo> pages,
                             cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                             size_t num_rows,
                             size_t min_row,
                             int level_type_size,
                             cudf::device_span<bool const> page_mask,
                             cudf::device_span<size_t> initial_str_offsets,
                             kernel_error::pointer error_code,
                             cuda::stream_ref stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    decode_delta_byte_array_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_delta_byte_array_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

/**
 * @copydoc cudf::io::parquet::gpu::decode_delta_length_byte_array
 */
void decode_delta_length_byte_array(cudf::detail::hostdevice_span<PageInfo> pages,
                                    cudf::detail::hostdevice_span<ColumnChunkDesc const> chunks,
                                    size_t num_rows,
                                    size_t min_row,
                                    int level_type_size,
                                    cudf::device_span<bool const> page_mask,
                                    cudf::device_span<size_t> initial_str_offsets,
                                    kernel_error::pointer error_code,
                                    cuda::stream_ref stream)
{
  CUDF_EXPECTS(pages.size() > 0, "There is no page to decode");

  dim3 const dim_block(decode_block_size, 1);
  dim3 const dim_grid(pages.size(), 1);  // 1 threadblock per page

  if (level_type_size == 1) {
    decode_delta_length_byte_array_kernel<uint8_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  } else {
    decode_delta_length_byte_array_kernel<uint16_t><<<dim_grid, dim_block, 0, stream.get()>>>(
      pages.device_ptr(), chunks, min_row, num_rows, page_mask, initial_str_offsets, error_code);
    CUDF_CUDA_TRY(cudaGetLastError());
  }
}

}  // namespace cudf::io::parquet::detail

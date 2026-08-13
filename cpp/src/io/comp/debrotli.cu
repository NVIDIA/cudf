/*
 * SPDX-FileCopyrightText: Copyright 2013 Google Inc. All Rights Reserved.
 * SPDX-FileCopyrightText: Copyright(c) 2009, 2010, 2013 - 2016 by the Brotli Authors.
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 AND MIT
 */

/** @file debrotli.cu
 *
 * CUDA-based brotli decompression
 *
 * Brotli Compressed Data Format
 * https://datatracker.ietf.org/doc/html/rfc7932
 *
 * Portions of this file are derived from Google's Brotli project at
 * https://github.com/google/brotli, original license text below.
 */

/* Copyright 2013 Google Inc. All Rights Reserved.

Distributed under MIT license.
See file LICENSE for detail or copy at https://opensource.org/license/MIT
*/

/*
Copyright(c) 2009, 2010, 2013 - 2016 by the Brotli Authors.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

#include "brotli_dict.hpp"
#include "gpuinflate.hpp"
#include "io/utilities/block_utils.cuh"

#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

namespace cudf::io::detail {

constexpr uint32_t huffman_lookup_table_width      = 8;
constexpr int8_t brotli_code_length_codes          = 18;
constexpr uint32_t brotli_num_distance_short_codes = 16;
constexpr uint32_t brotli_max_allowed_distance     = 0x7FFF'FFFC;
constexpr int block_size                           = 256;

template <typename T0, typename T1>
inline __device__ uint16_t huffcode(T0 len, T1 sym)
{
  return (uint16_t)(((sym) << 4) + (len));
}

inline __device__ uint32_t brotli_distance_alphabet_size(uint8_t npostfix,
                                                         uint32_t ndirect,
                                                         uint32_t maxnbits)
{
  return brotli_num_distance_short_codes + ndirect + (maxnbits << (npostfix + 1));
}

inline __device__ uint32_t brev8(uint32_t x)
{
  return (__brev(x) >> 24u);  // kReverseBits[x]
}

#define CONSTANT static const __device__ __constant__
#include "brotli_tables.hpp"

/* typeof(MODE) == ContextType; returns ContextLut */
__inline__ __device__ int brotli_context_lut(int mode) { return (mode << 9); }

inline __device__ uint8_t brotli_transform_type(int idx) { return kTransformsData[(idx * 3) + 1]; }

inline __device__ uint8_t const* brotli_transform_prefix(int idx)
{
  return &kPrefixSuffix[kPrefixSuffixMap[kTransformsData[(idx * 3)]]];
}

inline __device__ uint8_t const* brotli_transform_suffix(int idx)
{
  return &kPrefixSuffix[kPrefixSuffixMap[kTransformsData[(idx * 3) + 2]]];
}

/* typeof(LUT) == const uint8_t* */
__inline__ __device__ int brotli_need_context_lut(int mode) { return (mode < (4 << 9)); }

__inline__ __device__ int brotli_context(int p1, int p2, int lut)
{
  return kContextLookup[lut + p1] | kContextLookup[lut + p2 + 256];
}

/**
 * @brief Various local scratch arrays
 */
struct huff_scratch_s {
  uint16_t code_length_histo[16];
  uint8_t code_length_code_lengths[brotli_code_length_codes];
  int8_t offset[6];  // offsets in sorted table for each length
  uint16_t lenvlctab[32];
  uint16_t sorted[brotli_code_length_codes];  // symbols sorted by code length
  int16_t next_symbol[32];
  uint16_t symbols_lists_array[720];
};

/**
 * @brief Contains a collection of Huffman trees with the same alphabet size.
 * max_symbol is needed due to simple codes since log2(alphabet_size) could be
 * greater than log2(max_symbol).
 */
struct debrotli_huff_tree_group_s {
  uint16_t alphabet_size;
  uint16_t max_symbol;
  uint16_t num_htrees;
  uint16_t pad;
  uint16_t* htrees[1];
};

// Must be able to at least hold worst-case context maps, tree groups and context modes
constexpr int local_heap_size =
  (256 * 64 + 256 * 4 + 3 * (sizeof(debrotli_huff_tree_group_s) + 255 * sizeof(uint16_t*)) + 256 +
   3 * brotli_huffman_max_size_258 * sizeof(uint16_t) +
   3 * brotli_huffman_max_size_26 * sizeof(uint16_t));

/**
 * Brotli decoder state
 */
struct debrotli_state_s {
  // Bitstream
  uint8_t const* cur;
  uint8_t const* end;
  uint8_t const* base;
  uint2 bitbuf;
  uint32_t bitpos;
  int32_t error;
  // Output
  uint8_t* outbase;
  uint8_t* out;
  size_t bytes_left;
  // Decoded symbols
  uint8_t window_bits;
  uint8_t is_last;
  uint8_t is_uncompressed;
  uint8_t distance_postfix_bits;
  uint8_t distance_postfix_mask;
  uint8_t mtf_upper_bound;
  uint8_t p1;
  uint8_t p2;
  int32_t max_backward_distance;
  uint32_t num_block_types[3];
  uint32_t block_length[3];
  uint32_t num_direct_distance_codes;
  uint32_t meta_block_len;
  uint16_t heap_used;
  uint16_t heap_limit;
  uint8_t* context_map;
  uint8_t* dist_context_map;
  uint8_t* context_modes;
  uint8_t* fb_base;
  uint32_t fb_size;
  uint8_t block_type_rb[6];
  uint8_t pad[2];
  int dist_rb_idx;
  int dist_rb[4];
  debrotli_huff_tree_group_s* literal_hgroup;
  debrotli_huff_tree_group_s* insert_copy_hgroup;
  debrotli_huff_tree_group_s* distance_hgroup;
  uint16_t* block_type_vlc[3];
  huff_scratch_s hs;
  uint32_t mtf[65];
  __align__(8) char heap[local_heap_size];
};

inline __device__ uint32_t Log2Floor(uint32_t value) { return 32 - __clz(value); }

/// @brief Safely read up to 4 bytes with bounds checking, handling end-of-buffer cases
inline __device__ uint32_t safe_load_u32(uint8_t const* p, uint8_t const* end)
{
  if (p >= end) { return 0; }

  uint32_t result = 0;
  cuda::std::memcpy(
    &result, p, cuda::std::min<size_t>(cuda::std::distance(p, end), sizeof(uint32_t)));
  return result;
}

/// @brief initializes the bit reader
__device__ void initbits(debrotli_state_s* s, uint8_t const* base, size_t len, size_t pos = 0)
{
  uint8_t const* p  = base + pos;
  auto prefix_bytes = (uint32_t)(((size_t)p) & 3);
  p -= prefix_bytes;
  s->base     = base;
  s->end      = base + len;
  s->cur      = p;
  s->bitbuf.x = safe_load_u32(p, s->end);
  p += 4;
  s->bitbuf.y = safe_load_u32(p, s->end);
  s->bitpos   = prefix_bytes * 8;
}

// return next 32 bits
inline __device__ uint32_t next32bits(debrotli_state_s const* s)
{
  return __funnelshift_rc(s->bitbuf.x, s->bitbuf.y, s->bitpos);
}

/// return next n bits
inline __device__ uint32_t showbits(debrotli_state_s const* s, uint32_t n)
{
  uint32_t next32 = __funnelshift_rc(s->bitbuf.x, s->bitbuf.y, s->bitpos);
  return (next32 & ((1 << n) - 1));
}

inline __device__ void skipbits(debrotli_state_s* s, uint32_t n)
{
  uint32_t bitpos = s->bitpos + n;
  if (bitpos >= 32) {
    uint8_t const* cur = s->cur + 8;
    s->bitbuf.x        = s->bitbuf.y;
    s->bitbuf.y        = safe_load_u32(cur, s->end);
    s->cur             = cur - 4;
    bitpos &= 0x1f;
  }
  s->bitpos = bitpos;
}

inline __device__ uint32_t getbits(debrotli_state_s* s, uint32_t n)
{
  uint32_t bits = showbits(s, n);
  skipbits(s, n);
  return bits;
}

inline __device__ uint32_t getbits_bytealign(debrotli_state_s* s)
{
  auto n        = (uint32_t)((-(int32_t)s->bitpos) & 7);
  uint32_t bits = showbits(s, n);
  skipbits(s, n);
  return bits;
}

/** @brief Variable-length coding for 8-bit variable (1..11 bits)
 * encoded with the following variable-length code (as it appears in the
 * compressed data, where the bits are parsed from right to left,
 * so 0110111 has the value 12):
 *
 * Value    Bit Pattern
 * -----    -----------
 * 1                0
 * 2             0001
 * 3..4           x0011
 * 5..8          xx0101
 * 9..16        xxx0111
 * 17..32       xxxx1001
 * 33..64      xxxxx1011
 * 65..128    xxxxxx1101
 * 129..256   xxxxxxx1111
 */
static __device__ uint32_t getbits_u8vlc(debrotli_state_s* s)
{
  uint32_t next32 = next32bits(s);
  uint32_t v, len;
  if (!(next32 & 1)) {
    v   = 0;
    len = 1;
  } else {
    len = (next32 >> 1) & 7;
    v   = (1 << len) + ((next32 >> 4) & ((1 << len) - 1));
    len += 4;
  }
  skipbits(s, len);
  return v;
}

/// Decode a Huffman code with 8-bit initial lookup
static __device__ uint32_t getvlc(debrotli_state_s* s, uint16_t const* lut)
{
  uint32_t next32 = next32bits(s);
  uint32_t vlc, len;
  lut += next32 & ((1 << huffman_lookup_table_width) - 1);
  vlc = lut[0];
  len = vlc & 0x0f;
  vlc >>= 4;
  if (len > huffman_lookup_table_width) {
    len -= huffman_lookup_table_width;
    lut += vlc + ((next32 >> huffman_lookup_table_width) & ((1 << len) - 1));
    vlc = lut[0];
    len = huffman_lookup_table_width + (vlc & 0xf);
    vlc >>= 4;
  }
  skipbits(s, len);
  return vlc;
}

static auto __device__ allocation_size(uint32_t bytes) { return (bytes + 7) & ~7; }

/// Alloc bytes from the local (shared mem) heap
static __device__ uint8_t* local_alloc(debrotli_state_s* s, uint32_t bytes)
{
  int heap_used  = s->heap_used;
  auto const len = allocation_size(bytes);
  if (heap_used + len <= s->heap_limit) {
    auto* ptr    = reinterpret_cast<uint8_t*>(&s->heap[heap_used]);
    s->heap_used = (uint16_t)(heap_used + len);
    return ptr;
  } else {
    return nullptr;
  }
}

/// Shrink the size of the local heap, returns ptr to end (used for stack-like intermediate
/// allocations at the end of the heap)
static __device__ uint8_t* local_heap_shrink(debrotli_state_s* s, uint32_t bytes)
{
  int heap_used  = s->heap_used;
  int heap_limit = s->heap_limit;
  auto const len = allocation_size(bytes);
  if (heap_limit - len >= heap_used) {
    heap_limit -= len;
    s->heap_limit = (uint16_t)heap_limit;
    return reinterpret_cast<uint8_t*>(&s->heap[heap_limit]);
  } else {
    return nullptr;
  }
}

static __device__ void local_heap_grow(debrotli_state_s* s, uint32_t bytes)
{
  auto const len = allocation_size(bytes);
  int heap_limit = s->heap_limit + len;
  s->heap_limit  = (uint16_t)heap_limit;
}

/// Alloc memory from the fixed-size heap shared between all blocks (thread0-only)
static __device__ uint8_t* ext_heap_alloc(uint32_t bytes,
                                          uint8_t* ext_heap_base,
                                          uint32_t ext_heap_size)
{
  uint32_t len              = (bytes + 0xf) & ~0xf;
  auto volatile* heap_ptr   = reinterpret_cast<uint32_t volatile*>(ext_heap_base);
  uint32_t first_free_block = ~0;
  for (;;) {
    uint32_t blk_next, blk_prev;
    first_free_block = atomicExch((unsigned int*)heap_ptr, first_free_block);
    if (first_free_block == ~0 || first_free_block >= ext_heap_size) {
      // Some other block is holding the heap or there are no free blocks: try again later
      continue;
    }
    if (first_free_block == 0) {
      // Heap is uninitialized
      first_free_block = 4 * sizeof(uint32_t);
      heap_ptr[4]      = ext_heap_size;
      heap_ptr[5]      = ext_heap_size - first_free_block;
      __threadfence();
      continue;
    }
    blk_prev = 0;
    blk_next = first_free_block;
    do {
      uint32_t next, blksz;
      if (((blk_next & 3) != 0) || (blk_next >= ext_heap_size)) {
        // Corrupted heap
        atomicExch((unsigned int*)heap_ptr, first_free_block);
        return nullptr;
      }
      next  = heap_ptr[(blk_next >> 2) + 0];
      blksz = heap_ptr[(blk_next >> 2) + 1];
      if (blksz >= len) {
        uint32_t blk_new = blk_next + len;
        uint32_t sz_new  = blksz - len;
        if (sz_new >= 16) {
          // Reduce the size of the current block
          if (blk_prev == 0)
            first_free_block = blk_new;
          else
            heap_ptr[(blk_prev >> 2) + 0] = blk_new;
          heap_ptr[(blk_new >> 2) + 0] = next;
          heap_ptr[(blk_new >> 2) + 1] = sz_new;
        } else {
          // Re-use this block
          if (blk_prev == 0)
            first_free_block = next;
          else
            heap_ptr[(blk_prev >> 2) + 0] = next;
        }
        __threadfence();
        // Restore the list head
        atomicExch((unsigned int*)heap_ptr, first_free_block);
        return ext_heap_base + blk_next;
      } else {
        blk_prev = blk_next;
        blk_next = next;
      }
    } while (blk_next != 0 && blk_next < ext_heap_size);
    first_free_block = atomicExch((unsigned int*)heap_ptr, first_free_block);
    // Reaching here means the heap is full
    // Just in case we're trying to allocate more than the entire heap
    if (len > ext_heap_size - 4 * sizeof(uint32_t)) { break; }
  }
  return nullptr;
}

/// Free a memory block (thread0-only)
static __device__ void ext_heap_free(void* ptr,
                                     uint32_t bytes,
                                     uint8_t* ext_heap_base,
                                     uint32_t ext_heap_size)
{
  uint32_t len              = (bytes + 0xf) & ~0xf;
  auto volatile* heap_ptr   = (uint32_t volatile*)ext_heap_base;
  uint32_t first_free_block = ~0;
  auto cur_blk              = static_cast<uint32_t>(static_cast<uint8_t*>(ptr) - ext_heap_base);
  for (;;) {
    first_free_block = atomicExch((unsigned int*)heap_ptr, first_free_block);
    if (first_free_block != ~0) { break; }
    // Some other block is holding the heap
  }
  if (first_free_block >= ext_heap_size) {
    // Heap is currently empty
    first_free_block             = cur_blk;
    heap_ptr[(cur_blk >> 2) + 0] = first_free_block;
    heap_ptr[(cur_blk >> 2) + 1] = len;
  } else {
    uint32_t blk_prev = 0;
    uint32_t blk_next = first_free_block;
    for (;;) {
      uint32_t next  = heap_ptr[(blk_next >> 2) + 0];
      uint32_t blksz = heap_ptr[(blk_next >> 2) + 1];
      if (cur_blk + len < blk_next) {
        // Insert new block
        heap_ptr[(cur_blk >> 2) + 0] = blk_next;
        heap_ptr[(cur_blk >> 2) + 1] = len;
        if (blk_prev == 0) {
          first_free_block = cur_blk;
        } else if (blk_prev + heap_ptr[(blk_prev >> 2) + 1] == cur_blk) {
          // Merge with previous block
          heap_ptr[(blk_prev >> 2) + 1] = heap_ptr[(blk_prev >> 2) + 1] + len;
        } else {
          heap_ptr[(blk_prev >> 2) + 0] = cur_blk;
        }
        break;
      } else if (cur_blk + len == blk_next) {
        // Merge with next block
        heap_ptr[(cur_blk >> 2) + 0] = next;
        heap_ptr[(cur_blk >> 2) + 1] = len + blksz;
        if (blk_prev == 0) {
          first_free_block = cur_blk;
        } else if (blk_prev + heap_ptr[(blk_prev >> 2) + 1] == cur_blk) {
          // Also merge with previous block
          heap_ptr[(blk_prev >> 2) + 0] = next;
          heap_ptr[(blk_prev >> 2) + 1] = heap_ptr[(blk_prev >> 2) + 1] + len + blksz;
        } else {
          heap_ptr[(blk_prev >> 2) + 0] = cur_blk;
        }
        break;
      } else if (next < ext_heap_size) {
        // Move to the next block
        blk_prev = blk_next;
        blk_next = next;
      } else  // Insert this block at the tail
      {
        heap_ptr[(cur_blk >> 2) + 0] = next;
        heap_ptr[(cur_blk >> 2) + 1] = len;
        if (blk_next + blksz == cur_blk) {
          // Merge with last block
          heap_ptr[(blk_next >> 2) + 1] = heap_ptr[(blk_next >> 2) + 1] + len;
        } else {
          heap_ptr[(blk_next >> 2) + 0] = cur_blk;
        }
        break;
      }
    }
  }
  __threadfence();
  atomicExch((unsigned int*)heap_ptr, first_free_block);
}

static __device__ uint32_t BuildSimpleHuffmanTable(uint16_t* lut,
                                                   int root_bits,
                                                   uint16_t* val,
                                                   uint32_t num_symbols)
{
  uint32_t table_size      = 1;
  uint32_t const goal_size = 1U << root_bits;
  switch (num_symbols) {
    case 0: lut[0] = huffcode(0, val[0]); break;
    case 1:
      if (val[1] > val[0]) {
        lut[0] = huffcode(1, val[0]);
        lut[1] = huffcode(1, val[1]);
      } else {
        lut[0] = huffcode(1, val[1]);
        lut[1] = huffcode(1, val[0]);
      }
      table_size = 2;
      break;
    case 2:
      lut[0] = huffcode(1, val[0]);
      lut[2] = huffcode(1, val[0]);
      if (val[2] > val[1]) {
        lut[1] = huffcode(2, val[1]);
        lut[3] = huffcode(2, val[2]);
      } else {
        lut[1] = huffcode(2, val[2]);
        lut[3] = huffcode(2, val[1]);
      }
      table_size = 4;
      break;
    case 3: {
      int i, k;
      for (i = 0; i < 3; ++i) {
        for (k = i + 1; k < 4; ++k) {
          if (val[k] < val[i]) {
            uint16_t t = val[k];
            val[k]     = val[i];
            val[i]     = t;
          }
        }
      }
      lut[0]     = huffcode(2, val[0]);
      lut[2]     = huffcode(2, val[1]);
      lut[1]     = huffcode(2, val[2]);
      lut[3]     = huffcode(2, val[3]);
      table_size = 4;
      break;
    }
    case 4: {
      if (val[3] < val[2]) {
        uint16_t t = val[3];
        val[3]     = val[2];
        val[2]     = t;
      }
      lut[0]     = huffcode(1, val[0]);
      lut[1]     = huffcode(2, val[1]);
      lut[2]     = huffcode(1, val[0]);
      lut[3]     = huffcode(3, val[2]);
      lut[4]     = huffcode(1, val[0]);
      lut[5]     = huffcode(2, val[1]);
      lut[6]     = huffcode(1, val[0]);
      lut[7]     = huffcode(3, val[3]);
      table_size = 8;
      break;
    }
  }
  while (table_size != goal_size) {
    memcpy(&lut[table_size], &lut[0], table_size * sizeof(lut[0]));
    table_size <<= 1;
  }
  return goal_size;
}

static __device__ void BuildCodeLengthsHuffmanTable(huff_scratch_s* hs)
{
  uint32_t code;   // current table entry
  int symbol;      // symbol index in original or sorted table
  int key;         // prefix code
  int key_step;    // prefix code addend
  int step;        // step size to replicate values in current table
  int table_size;  // size of current table
  int bits;

  // Generate offsets into sorted symbol table by code length.
  symbol = -1;
  for (bits = 1; bits <= 5; bits++) {
    symbol += hs->code_length_histo[bits];
    hs->offset[bits] = (int8_t)symbol;
  }
  // Symbols with code length 0 are placed after all other symbols.
  hs->offset[0] = brotli_code_length_codes - 1;
  // Sort symbols by length, by symbol order within each length.
  symbol = brotli_code_length_codes;
  do {
    symbol--;
    hs->sorted[hs->offset[hs->code_length_code_lengths[symbol]]--] = symbol;
  } while (symbol != 0);

  table_size = 1 << 5;

  // Special case: all symbols but one have 0 code length.
  if (hs->offset[0] == 0) {
    code = huffcode(0, hs->sorted[0]);
    for (key = 0; key < table_size; ++key) {
      hs->lenvlctab[key] = code;
    }
    return;
  }

  // Fill in table.
  key      = 0;
  key_step = 1 << 7;
  symbol   = 0;
  bits     = 1;
  step     = 2;
  do {
    for (int bits_count = hs->code_length_histo[bits]; bits_count != 0; --bits_count) {
      int end     = table_size;
      code        = huffcode(bits, hs->sorted[symbol++]);
      uint16_t* p = &hs->lenvlctab[brev8(key)];
      do {
        end -= step;
        p[end] = code;
      } while (end > 0);
      key += key_step;
    }
    step <<= 1;
    key_step >>= 1;
  } while (++bits <= 5);
}

// Returns the table width of the next 2nd level table. |count| is the histogram
// of bit lengths for the remaining symbols, |len| is the code length of the
// next processed symbol.
static __device__ int NextTableBitSize(uint16_t const* const count, int len, int root_bits)
{
  int left = 1 << (len - root_bits);
  while (len < 15) {
    left -= count[len];
    if (left <= 0) break;
    ++len;
    left <<= 1;
  }
  return len - root_bits;
}

// Build a huffman lookup table (currently thread0-only)
static __device__ uint32_t BuildHuffmanTable(uint16_t* root_lut,
                                             int root_bits,
                                             uint16_t const* const symbol_lists,
                                             uint16_t* count)
{
  uint32_t code;     // current table entry
  uint16_t* lut;     // next available space in table
  int len;           // current code length
  int symbol;        // symbol index in original or sorted table
  int key;           // prefix code
  int key_step;      // prefix code addend
  int sub_key;       // 2nd level table prefix code
  int sub_key_step;  // 2nd level table prefix code addend
  int step;          // step size to replicate values in current table
  int table_bits;    // key length of current table
  int table_size;    // size of current table
  int total_size;    // sum of root table size and 2nd level table sizes
  int max_length = -1;
  int bits;

  while (symbol_lists[max_length] == 0xFFFF)
    max_length--;
  max_length += 16;

  lut        = root_lut;
  table_bits = root_bits;
  table_size = 1 << table_bits;
  total_size = table_size;

  // Fill in the root table. Reduce the table size to if possible, and create the repetitions by
  // memcpy.
  if (table_bits > max_length) {
    table_bits = max_length;
    table_size = 1 << table_bits;
  }
  key      = 0;
  key_step = 1 << 7;
  bits     = 1;
  step     = 2;
  do {
    symbol = bits - 16;
    for (int bits_count = count[bits]; bits_count != 0; --bits_count) {
      symbol      = symbol_lists[symbol];
      code        = huffcode(bits, symbol);
      uint16_t* p = &lut[brev8(key)];
      int end     = table_size;
      do {
        end -= step;
        p[end] = code;
      } while (end > 0);
      key += key_step;
    }
    step <<= 1;
    key_step >>= 1;
  } while (++bits <= table_bits);

  // If root_bits != table_bits then replicate to fill the remaining slots.
  while (total_size != table_size) {
    memcpy(&lut[table_size], &lut[0], table_size * sizeof(lut[0]));
    table_size <<= 1;
  }

  // Fill in 2nd level tables and add pointers to root table.
  key_step     = (1 << 7) >> (root_bits - 1);
  sub_key      = (1 << 8);
  sub_key_step = (1 << 7);
  for (len = root_bits + 1, step = 2; len <= max_length; ++len) {
    symbol = len - 16;
    for (; count[len] != 0; --count[len]) {
      if (sub_key == (1 << 8)) {
        lut += table_size;
        table_bits = NextTableBitSize(count, len, root_bits);
        table_size = 1 << table_bits;
        total_size += table_size;
        sub_key = brev8(key);
        key += key_step;
        root_lut[sub_key] =
          huffcode(table_bits + root_bits, (((size_t)(lut - root_lut)) - sub_key));
        sub_key = 0;
      }
      symbol      = symbol_lists[symbol];
      code        = huffcode(len - root_bits, symbol);
      uint16_t* p = &lut[brev8(sub_key)];
      int end     = table_size;
      do {
        end -= step;
        p[end] = code;
      } while (end > 0);
      sub_key += sub_key_step;
    }
    step <<= 1;
    sub_key_step >>= 1;
  }
  return (uint32_t)total_size;
}

/**
3.4.  Simple Prefix Codes

The first two bits of the compressed representation of each prefix
code distinguish between simple and complex prefix codes.  If this
value is 1, then a simple prefix code follows as described in this
section.  Otherwise, a complex prefix code follows as described in
Section 3.5.

A simple prefix code can have up to four symbols with non-zero code
length.  The format of the simple prefix code is as follows:

2 bits: value of 1 indicates a simple prefix code
2 bits: NSYM - 1, where NSYM = number of symbols coded
NSYM symbols, each encoded using ALPHABET_BITS bits
1 bit:  tree-select, present only for NSYM = 4

The value of ALPHABET_BITS depends on the alphabet of the prefix
code: it is the smallest number of bits that can represent all
symbols in the alphabet.  For example, for the alphabet of literal
bytes, ALPHABET_BITS is 8.  The value of each of the NSYM symbols
above is the value of the ALPHABET_BITS width integer value.  If the
integer value is greater than or equal to the alphabet size, or the
value is identical to a previous value, then the stream should be
rejected as invalid.

Note that the NSYM symbols may not be presented in sorted order.
Prefix codes of the same bit length must be assigned to the symbols
in sorted order.

The (non-zero) code lengths of the symbols can be reconstructed as
follows:

*  if NSYM = 1, the code length for the one symbol is zero -- when
encoding this symbol in the compressed data stream using this
prefix code, no actual bits are emitted.  Similarly, when
decoding a symbol using this prefix code, no bits are read and
the one symbol is returned.

*  if NSYM = 2, both symbols have code length 1.

*  if NSYM = 3, the code lengths for the symbols are 1, 2, 2 in
the order they appear in the representation of the simple
prefix code.

*  if NSYM = 4, the code lengths (in order of symbols decoded)
depend on the tree-select bit: 2, 2, 2, 2 (tree-select bit 0),
or 1, 2, 3, 3 (tree-select bit 1).

3.5.  Complex Prefix Codes

A complex prefix code is a canonical prefix code, defined by the
sequence of code lengths, as discussed in Section 3.2.  For even
greater compactness, the code length sequences themselves are
compressed using a prefix code.  The alphabet for code lengths is as
follows:

0..15: Represent code lengths of 0..15
16: Copy the previous non-zero code length 3..6 times.
The next 2 bits indicate repeat length (0 = 3, ... , 3 = 6)
If this is the first code length, or all previous
code lengths are zero, a code length of 8 is
repeated 3..6 times.
A repeated code length code of 16 modifies the
repeat count of the previous one as follows:
repeat count = (4 * (repeat count - 2)) + (3..6 on the next 2 bits)
Example:  Codes 7, 16 (+2 bits 11), 16 (+2 bits 10)
will expand to 22 code lengths of 7 (1 + 4 * (6 - 2) + 5)
17: Repeat a code length of 0 for 3..10 times.
The next 3 bits indicate repeat length (0 = 3, ... , 7 = 10)
A repeated code length code of 17 modifies the
repeat count of the previous one as follows:
repeat count = (8 * (repeat count - 2)) + (3..10 on the next 3 bits)

Note that a code of 16 that follows an immediately preceding 16
modifies the previous repeat count, which becomes the new repeat
count.  The same is true for a 17 following a 17.  A sequence of
three or more 16 codes in a row or three of more 17 codes in a row is
possible, modifying the count each time.  Only the final repeat count
is used.  The modification only applies if the same code follows.  A
16 repeat does not modify an immediately preceding 17 count nor vice
versa.

A code length of 0 indicates that the corresponding symbol in the
alphabet will not occur in the compressed data, and it should not
participate in the prefix code construction algorithm given earlier.
A complex prefix code must have at least two non-zero code lengths.

The bit lengths of the prefix code over the code length alphabet are
compressed with the following variable-length code (as it appears in
the compressed data, where the bits are parsed from right to left):

Symbol   Code
------   ----
0          00
1        0111
2         011
3          10
4          01
5        1111

We can now define the format of the complex prefix code as follows:

o  2 bits: HSKIP, the number of skipped code lengths, can have values
of 0, 2, or 3.  The skipped lengths are taken to be zero.  (An
HSKIP of 1 indicates a Simple prefix code.)

o  Code lengths for symbols in the code length alphabet given just
above, in the order: 1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11,
12, 13, 14, 15.  If HSKIP is 2, then the code lengths for symbols
1 and 2 are zero, and the first code length is for symbol 3.  If
HSKIP is 3, then the code length for symbol 3 is also zero, and
the first code length is for symbol 4.

The code lengths of code length symbols are between 0 and 5, and
they are represented with 2..4 bits according to the variable-
length code above.  A code length of 0 means the corresponding
code length symbol is not used.

If HSKIP is 2 or 3, a respective number of leading code lengths
are implicit zeros and are not present in the code length sequence
above.

If there are at least two non-zero code lengths, any trailing zero
code lengths are omitted, i.e., the last code length in the
sequence must be non-zero.  In this case, the sum of (32 >> code
length) over all the non-zero code lengths must equal to 32.

If the lengths have been read for the entire code length alphabet
and there was only one non-zero code length, then the prefix code
has one symbol whose code has zero length.  In this case, that
symbol results in no bits being emitted by the compressor and no
bits consumed by the decompressor.  That single symbol is
immediately returned when this code is decoded.  An example of
where this occurs is if the entire code to be represented has
symbols of length 8.  For example, a literal code that represents
all literal values with equal probability.  In this case the
single symbol is 16, which repeats the previous length.  The
previous length is taken to be 8 before any code length code
lengths are read.

o  Sequence of code length symbols, which is at most the size of the
alphabet, encoded using the code length prefix code.  Any trailing
0 or 17 must be omitted, i.e., the last encoded code length symbol
must be between 1 and 16.  The sum of (32768 >> code length) over
all the non-zero code lengths in the alphabet, including those
encoded using repeat code(s) of 16, must be equal to 32768.  If
the number of times to repeat the previous length or repeat a zero
length would result in more lengths in total than the number of
symbols in the alphabet, then the stream should be rejected as
invalid.
*/

// Decode Huffman tree (thread0-only)
static __device__ uint32_t DecodeHuffmanTree(debrotli_state_s* s,
                                             uint32_t alphabet_size,
                                             uint32_t max_symbol,
                                             uint16_t* vlctab)
{
  uint32_t prefix_code_type;

  prefix_code_type = getbits(s, 2);
  if (prefix_code_type == 1) {
    // Simple prefix Code
    uint32_t nsym          = getbits(s, 2);
    uint32_t alphabet_bits = Log2Floor(alphabet_size - 1);
    for (uint32_t n = 0; n <= nsym; n++) {
      uint32_t v = getbits(s, alphabet_bits);
      if (v >= max_symbol) {
        s->error = -1;
        return 0;
      }
      for (uint32_t i = 0; i < n; i++) {
        if (v == s->hs.symbols_lists_array[i]) {
          s->error = -1;  // Duplicate symbol
          return 0;
        }
      }
      s->hs.symbols_lists_array[n] = (uint16_t)v;
    }
    if (nsym == 3) {
      nsym += getbits(s, 1);  // tree_select;
    }
    return BuildSimpleHuffmanTable(
      vlctab, huffman_lookup_table_width, s->hs.symbols_lists_array, nsym);
  } else {
    // Complex prefix code
    huff_scratch_s* const hs = &s->hs;
    uint16_t* symbol_lists =
      &s->hs.symbols_lists_array[16];  // Make small negative indexes addressable
    uint32_t space = 32, num_codes = 0, i, prev_code_len, symbol, repeat, repeat_code_len;

    memset(&hs->code_length_histo[0], 0, 6 * sizeof(hs->code_length_histo));
    memset(&hs->code_length_code_lengths[0], 0, sizeof(hs->code_length_code_lengths));
    for (i = prefix_code_type; i < 18; i++) {
      uint8_t const code_len_idx = kCodeLengthCodeOrder[i];
      uint32_t ix, v;

      ix = showbits(s, 4);
      v  = kCodeLengthPrefixValue[ix];
      skipbits(s, kCodeLengthPrefixLength[ix]);
      hs->code_length_code_lengths[code_len_idx] = (uint8_t)v;
      if (v != 0) {
        space = space - (32u >> v);
        ++num_codes;
        ++hs->code_length_histo[v];
        if (space - 1u >= 32u) {
          // space is 0 or wrapped around.
          break;
        }
      }
    }
    if (!(num_codes == 1 || space == 0)) {
      s->error = -1;
      return 0;
    }
    BuildCodeLengthsHuffmanTable(&s->hs);
    for (i = 0; i <= 15; ++i) {
      hs->code_length_histo[i]         = 0;
      hs->next_symbol[i]               = (int16_t)(i - 16);
      symbol_lists[hs->next_symbol[i]] = 0xFFFF;
    }
    symbol          = 0;
    prev_code_len   = 8;
    repeat          = 0;
    repeat_code_len = 0;
    space           = 32768;
    while (symbol < max_symbol && space > 0) {
      uint32_t next32   = next32bits(s);
      uint32_t code_len = hs->lenvlctab[next32 & 0x1f];
      uint32_t vlc_len  = code_len & 0xf;  // Use 1..5 bits
      code_len >>= 4;                      // code_len = 0..17
      if (code_len < 16) {
        // Process single decoded symbol code length :
        // A) reset the repeat variable
        // B) remember code length(if it is not 0)
        // C) extend corresponding index - chain
        // D) reduce the Huffman space
        // E) update the histogram
        repeat = 0;
        if (code_len != 0) {  // code_len == 1..15
          symbol_lists[hs->next_symbol[code_len]] = (uint16_t)symbol;
          hs->next_symbol[code_len]               = (int)symbol;
          prev_code_len                           = code_len;
          space -= 32768u >> code_len;
          hs->code_length_histo[code_len]++;
        }
        symbol++;
      } else {
        // Process repeated symbol code length.
        // A) Check if it is the extension of previous repeat sequence; if the decoded value is not
        // 16, then it is a new symbol-skip B) Update repeat variable C) Check if operation is
        // feasible (fits alphabet) D) For each symbol do the same operations as in single symbol
        uint32_t extra_bits, repeat_delta, new_len, old_repeat;

        if (code_len == 16) {
          extra_bits   = 2;
          repeat_delta = (next32 >> vlc_len) & 3;
          new_len      = prev_code_len;
        } else {
          extra_bits   = 3;
          repeat_delta = (next32 >> vlc_len) & 7;
          new_len      = 0;
        }
        vlc_len += extra_bits;
        if (repeat_code_len != new_len) {
          repeat          = 0;
          repeat_code_len = new_len;
        }
        old_repeat = repeat;
        if (repeat > 0) { repeat = (repeat - 2) << extra_bits; }
        repeat += repeat_delta + 3u;
        repeat_delta = repeat - old_repeat;
        if (symbol + repeat_delta > max_symbol) {
          s->error = -1;
          return 0;
        }
        if (repeat_code_len != 0) {
          uint32_t last = symbol + repeat_delta;
          int next      = hs->next_symbol[repeat_code_len];
          do {
            symbol_lists[next] = (uint16_t)symbol;
            next               = (int)symbol;
          } while (++symbol != last);
          hs->next_symbol[repeat_code_len] = next;
          space -= repeat_delta << (15 - repeat_code_len);
          hs->code_length_histo[repeat_code_len] =
            (uint16_t)(hs->code_length_histo[repeat_code_len] + repeat_delta);
        } else {
          symbol += repeat_delta;
        }
      }
      skipbits(s, vlc_len);
    }
    if (space != 0) {
      s->error = -1;
      return 0;
    }
    return BuildHuffmanTable(
      vlctab, huffman_lookup_table_width, symbol_lists, hs->code_length_histo);
  }
}

/**
9.1.  Format of the Stream Header

The stream header has only the following one field:

1..7 bits: WBITS, a value in the range 10..24, encoded with the
following variable - length code(as it appears in the
compressed data, where the bits are parsed from right
to left) :

Value    Bit Pattern
---- - ---------- -
10        0100001
11        0110001
12        1000001
13        1010001
14        1100001
15        1110001
16              0
17        0000001
18           0011
19           0101
20           0111
21           1001
22           1011
23           1101
24           1111

Note that bit pattern 0010001 is invalid and must not
be used.

The size of the sliding window, which is the maximum value of any
non - dictionary reference backward distance, is given by the following
formula :

window size = (1 << WBITS) - 16
*/
static __device__ void DecodeStreamHeader(debrotli_state_s* s)
{
  uint32_t next32 = next32bits(s);
  uint32_t wbits = 0, len = 0;
  if ((next32 & 1) == 0) {
    // 0
    wbits = 16;
    len   = 1;
  } else {
    uint32_t n = (next32 >> 1) & 7;
    if (n != 0) {
      // xxx1
      wbits = 17 + n;
      len   = 4;
    } else {
      n = (next32 >> 4) & 7;
      if (n != 1) {
        wbits = (n) ? 8 + n : 17;  // xxx0001
        len   = 7;
      } else {
        // Large window (not supported) or invalid, bail
        s->error = -1;
      }
    }
  }
  s->window_bits           = (uint8_t)wbits;
  s->max_backward_distance = (1 << s->window_bits) - 16;
  skipbits(s, len);
}

/**
9.2.Format of the Meta - Block Header

A compliant compressed data set has at least one meta - block.Each
meta - block contains a header with information about the uncompressed
length of the meta - block, and a bit signaling if the meta - block is
the last one.The format of the meta - block header is the following :

1 bit : ISLAST, set to 1 if this is the last meta - block

1 bit : ISLASTEMPTY, if set to 1, the meta - block is empty; this
field is only present if ISLAST bit is set-- if it is 1, then the
meta - block and the brotli stream ends at that bit, with any remaining
bits in the last byte of the compressed stream filled with zeros(if the
fill bits are not zero, then the stream should be rejected as invalid)

2 bits: MNIBBLES, number of nibbles to represent the uncompressed
length, encoded with the following fixed - length code :

Value    Bit Pattern
---- - ---------- -
0             11
4             00
5             01
6             10

If MNIBBLES is 0, the meta - block is empty, i.e., it does not generate
any uncompressed data.In this case, the rest of the meta - block has
the following format :

1 bit : reserved, must be zero

2 bits : MSKIPBYTES, number of bytes to represent metadata length

MSKIPBYTES * 8 bits : MSKIPLEN - 1, where MSKIPLEN is the number of
metadata bytes; this field is only present if MSKIPBYTES is positive;
otherwise, MSKIPLEN is 0 (if MSKIPBYTES is greater than 1, and the last
byte is all zeros, then the stream should be rejected as invalid)

0..7 bits: fill bits until the next byte boundary, must be all zeros

MSKIPLEN bytes of metadata, not part of the uncompressed data
or the sliding window

MNIBBLES * 4 bits: MLEN - 1, where MLEN is the length of the meta -
block uncompressed data in bytes(if MNIBBLES is greater than 4,
and the last nibble is all zeros, then the stream should be rejected
as invalid)

1 bit : ISUNCOMPRESSED, if set to 1, any bits of compressed data up to the
next byte boundary are ignored, and the rest of the meta - block contains
MLEN bytes of literal data; this field is only present if the ISLAST bit is
not set(if the ignored bits are not all zeros, the stream should be rejected
as invalid)
*/

static __device__ void DecodeMetaBlockHeader(debrotli_state_s* s)
{
  uint32_t next32 = next32bits(s);
  uint32_t len = 1, is_empty = 0;
  s->is_last = (uint8_t)(next32 & 1);
  if (s->is_last) {
    is_empty = (uint8_t)((next32 >> 1) & 1);
    len++;
  }
  s->meta_block_len  = 0;
  s->is_uncompressed = 0;
  if (!is_empty) {
    uint32_t mnibbles = 4 + ((next32 >> len) & 3);
    len += 2;
    if (mnibbles < 7) {
      s->meta_block_len = 1 + ((next32 >> len) & ((1u << (mnibbles * 4)) - 1));
      len += mnibbles * 4;
      if (mnibbles > 4u && s->meta_block_len <= (1u << (mnibbles * 4 - 4))) { s->error = -1; }
      if (!s->is_last) {
        s->is_uncompressed = (uint8_t)((next32 >> len) & 1);
        len++;
      }
    } else {
      uint32_t reserved, mskipbytes, mskiplen;
      reserved = (next32 >> len) & 1;
      if (reserved != 0) { s->error = -1; }
      len += 1;
      mskipbytes = (next32 >> len) & 3;
      len += 2;
      if (mskipbytes > 0) {
        mskiplen = 1 + (next32 >> len) & ((1u << (mskipbytes * 8)) - 1);
        if (mskiplen <= ((1u << (mskipbytes * 8)) >> 8)) {
          s->error = -1;  // Last byte is all zeros
        }
        len += mskipbytes * 8;
      } else {
        mskiplen = 0;
      }
      skipbits(s, len);
      if (getbits_bytealign(s) != 0) { s->error = 1; }
      for (len = mskiplen; len >= 32; len -= 32) {
        skipbits(s, 32);
      }
    }
  }
  skipbits(s, len);
}

/**
1..11 bits: NBLTYPESL, number of literal block types

Prefix code over the block type code alphabet for literal block
types, appears only if NBLTYPESL >= 2

Prefix code over the block count code alphabet for literal
block counts, appears only if NBLTYPESL >= 2

Block count code + extra bits for first literal block count,
appears only if NBLTYPESL >= 2

1..11 bits: NBLTYPESI, number of insert-and-copy block types,
encoded with the same variable-length code as above

Prefix code over the block type code alphabet for insert-and-
copy block types, appears only if NBLTYPESI >= 2

Prefix code over the block count code alphabet for insert-and-
copy block counts, appears only if NBLTYPESI >= 2

Block count code + extra bits for first insert-and-copy block
count, appears only if NBLTYPESI >= 2

1..11 bits: NBLTYPESD, number of distance block types, encoded
with the same variable-length code as above

Prefix code over the block type code alphabet for distance
block types, appears only if NBLTYPESD >= 2

Prefix code over the block count code alphabet for distance
block counts, appears only if NBLTYPESD >= 2

Block count code + extra bits for first distance block count,
appears only if NBLTYPESD >= 2
*/

static __device__ void DecodeHuffmanTables(debrotli_state_s* s)
{
  for (int b = 0; b < 3; b++) {
    uint32_t nbltypes     = 1 + getbits_u8vlc(s);
    s->num_block_types[b] = nbltypes;
    if (nbltypes >= 2) {
      uint32_t alphabet_size = nbltypes + 2, index, nbits, maxtblsz;
      uint16_t* vlctab;
      maxtblsz = kMaxHuffmanTableSize[(alphabet_size + 31) >> 5];
      maxtblsz = (maxtblsz > brotli_huffman_max_size_258) ? brotli_huffman_max_size_258 : maxtblsz;
      vlctab   = reinterpret_cast<uint16_t*>(
        local_alloc(s, (brotli_huffman_max_size_26 + maxtblsz) * sizeof(uint16_t)));
      s->block_type_vlc[b] = vlctab;
      DecodeHuffmanTree(s, alphabet_size, alphabet_size, vlctab + brotli_huffman_max_size_26);
      alphabet_size = 26;
      DecodeHuffmanTree(s, alphabet_size, alphabet_size, vlctab);
      if (s->error) { break; }
      index              = getvlc(s, vlctab);
      nbits              = kBlockLengthPrefixCodeBits[index];  // nbits == 2..24
      s->block_length[b] = kBlockLengthPrefixCodeOffset[index] + getbits(s, nbits);

    } else {
      s->block_length[b] = 1 << 24;
    }
  }
  s->block_type_rb[0] = 1;
  s->block_type_rb[1] = 0;
  s->block_type_rb[2] = 1;
  s->block_type_rb[3] = 0;
  s->block_type_rb[4] = 1;
  s->block_type_rb[5] = 0;
}

/** @brief Transform:
 * 1) initialize list L with values 0, 1,... 255
 * 2) For each input element X:
 * 2.1) let Y = L[X]
 * 2.2) remove X-th element from L
 * 2.3) prepend Y to L
 * 2.4) append Y to output
 *
 * In most cases max(Y) <= 7, so most of L remains intact.
 * To reduce the cost of initialization, we reuse L, remember the upper bound
 * of Y values, and reinitialize only first elements in L.
 *
 * Most of input values are 0 and 1. To reduce number of branches, we replace
 * inner for loop with do-while.
 */
static __device__ void InverseMoveToFrontTransform(debrotli_state_s* s, uint8_t* v, uint32_t v_len)
{
  // Reinitialize elements that could have been changed.
  uint32_t i           = 1;
  uint32_t upper_bound = s->mtf_upper_bound;
  uint32_t* mtf        = &s->mtf[1];  // Make mtf[-1] addressable.
  auto* mtf_u8         = reinterpret_cast<uint8_t*>(mtf);
  uint32_t pattern     = 0x0302'0100;  // Little-endian

  // Initialize list using 4 consequent values pattern.
  mtf[0] = pattern;
  do {
    pattern += 0x0404'0404;  // Advance all 4 values by 4.
    mtf[i] = pattern;
    i++;
  } while (i <= upper_bound);

  // Transform the input.
  upper_bound = 0;
  for (i = 0; i < v_len; ++i) {
    int index     = v[i];
    uint8_t value = mtf_u8[index];
    upper_bound |= v[i];
    v[i]       = value;
    mtf_u8[-1] = value;
    do {
      index--;
      mtf_u8[index + 1] = mtf_u8[index];
    } while (index >= 0);
  }
  // Remember amount of elements to be reinitialized.
  s->mtf_upper_bound = upper_bound >> 2;
}

static __device__ uint32_t DecodeContextMap(debrotli_state_s* s,
                                            uint8_t* context_map,
                                            uint32_t context_map_size,
                                            uint16_t* context_map_vlc)
{
  uint32_t num_htrees = getbits_u8vlc(s) + 1;
  uint32_t bits, context_index, max_run_length_prefix, alphabet_size;

  if (num_htrees <= 1) {
    memset(context_map, 0, context_map_size);
    return num_htrees;
  }
  bits = showbits(s, 5);
  if ((bits & 1) != 0) {  // Use RLE for zeros.
    max_run_length_prefix = (bits >> 1) + 1;
    skipbits(s, 5);
  } else {
    max_run_length_prefix = 0;
    skipbits(s, 1);
  }
  alphabet_size = num_htrees + max_run_length_prefix;
  DecodeHuffmanTree(s, alphabet_size, alphabet_size, context_map_vlc);
  if (s->error) { return num_htrees; }
  context_index = 0;
  while (context_index < context_map_size) {
    uint32_t code = getvlc(s, context_map_vlc);
    if (code == 0) {
      context_map[context_index++] = 0;
    } else if (code > max_run_length_prefix) {
      context_map[context_index++] = (uint8_t)(code - max_run_length_prefix);
    } else {
      // RLE sub-stage.
      uint32_t reps = getbits(s, code) + (1u << code);
      if (context_index + reps > context_map_size) {
        s->error = -1;
        break;
      }
      do {
        context_map[context_index++] = 0;
      } while (--reps);
    }
  }
  bits = getbits(s, 1);
  if (bits != 0) { InverseMoveToFrontTransform(s, context_map, context_map_size); }
  return num_htrees;
}

static __device__ void DetectTrivialLiteralBlockTypes(debrotli_state_s* s)
{
  uint32_t i;
  for (i = 0; i < s->num_block_types[0]; i++) {
    uint32_t offset = i << 6;
    uint32_t error  = 0;
    uint32_t sample = s->context_map[offset];
    uint32_t j;
    for (j = 0; j < (1u << 6); ++j) {
      error |= s->context_map[offset + j] ^ sample;
    }
    if (error == 0) { s->context_modes[i] |= 4u; }
  }
}

/**

2 bits: NPOSTFIX, parameter used in the distance coding

4 bits: four most significant bits of NDIRECT, to get the actual
value of the parameter NDIRECT, left-shift this four-bit
number by NPOSTFIX bits

NBLTYPESL * 2 bits: context mode for each literal block type

1..11 bits: NTREESL, number of literal prefix trees, encoded with
the same variable-length code as NBLTYPESL

Literal context map, encoded as described in Section 7.3,
appears only if NTREESL >= 2; otherwise, the context map has
only zero values

1..11 bits: NTREESD, number of distance prefix trees, encoded with
the same variable-length code as NBLTYPESD

Distance context map, encoded as described in Section 7.3,
appears only if NTREESD >= 2; otherwise, the context map has
only zero values
*/

static __device__ debrotli_huff_tree_group_s* HuffmanTreeGroupInit(debrotli_state_s* s,
                                                                   uint32_t alphabet_size,
                                                                   uint32_t max_symbol,
                                                                   uint32_t ntrees)
{
  auto* group          = reinterpret_cast<debrotli_huff_tree_group_s*>(local_alloc(
    s, sizeof(debrotli_huff_tree_group_s) + ntrees * sizeof(uint16_t*) - sizeof(uint16_t*)));
  group->alphabet_size = (uint16_t)alphabet_size;
  group->max_symbol    = (uint16_t)max_symbol;
  group->num_htrees    = (uint16_t)ntrees;
  group->htrees[0]     = nullptr;
  return group;
}

static __device__ void HuffmanTreeGroupAlloc(debrotli_state_s* s, debrotli_huff_tree_group_s* group)
{
  if (!group->htrees[0]) {
    uint32_t alphabet_size  = group->alphabet_size;
    uint32_t ntrees         = group->num_htrees;
    uint32_t max_table_size = kMaxHuffmanTableSize[(alphabet_size + 31) >> 5];
    uint32_t code_size      = sizeof(uint16_t) * ntrees * max_table_size;
    group->htrees[0]        = reinterpret_cast<uint16_t*>(local_alloc(s, code_size));
    if (!group->htrees[0]) {
      if (s->fb_base) { group->htrees[0] = reinterpret_cast<uint16_t*>(s->fb_base + s->fb_size); }
      s->fb_size += (code_size + 3) & ~3;
    }
  }
}

// Decodes a series of Huffman table using ReadHuffmanCode function.
static __device__ void HuffmanTreeGroupDecode(debrotli_state_s* s,
                                              debrotli_huff_tree_group_s* group)
{
  uint16_t* next = group->htrees[0];

  for (int htree_index = 0; htree_index < group->num_htrees; htree_index++) {
    uint32_t table_size = DecodeHuffmanTree(s, group->alphabet_size, group->max_symbol, next);
    if (s->error) break;
    group->htrees[htree_index] = next;
    next += table_size;
  }
}

static __device__ void DecodeHuffmanTreeGroups(debrotli_state_s* s,
                                               uint8_t* fb_heap_base,
                                               uint32_t fb_heap_size)
{
  uint32_t bits, npostfix, ndirect, nbltypesl;
  uint32_t context_map_size;
  uint16_t* context_map_vlc;
  uint32_t num_direct_codes, num_distance_codes, num_literal_htrees, num_dist_htrees;

  // Decode context maps
  bits                         = getbits(s, 6);
  npostfix                     = bits & 3;
  ndirect                      = (bits >> 2) << npostfix;
  s->distance_postfix_bits     = (uint8_t)npostfix;
  s->num_direct_distance_codes = brotli_num_distance_short_codes + ndirect;
  s->distance_postfix_mask     = (1 << npostfix) - 1;
  nbltypesl                    = s->num_block_types[0];
  s->context_modes             = local_alloc(s, nbltypesl);
  for (uint32_t i = 0; i < nbltypesl; i++) {
    s->context_modes[i] = getbits(s, 2);
  }
  context_map_vlc = reinterpret_cast<uint16_t*>(
    local_heap_shrink(s, brotli_huffman_max_size_272 * sizeof(uint16_t)));
  context_map_size   = nbltypesl << 6;
  s->context_map     = local_alloc(s, context_map_size);
  num_literal_htrees = DecodeContextMap(s, s->context_map, context_map_size, context_map_vlc);
  if (s->error) return;
  DetectTrivialLiteralBlockTypes(s);
  context_map_size    = s->num_block_types[2] << 2;
  s->dist_context_map = local_alloc(s, context_map_size);
  num_dist_htrees     = DecodeContextMap(s, s->dist_context_map, context_map_size, context_map_vlc);
  if (s->error) return;
  local_heap_grow(s, brotli_huffman_max_size_272 * sizeof(uint16_t));  // free context map vlc
  num_direct_codes = s->num_direct_distance_codes - brotli_num_distance_short_codes;
  num_distance_codes =
    brotli_distance_alphabet_size(s->distance_postfix_bits, num_direct_codes, 24u);
  s->literal_hgroup = HuffmanTreeGroupInit(
    s, brotli_num_literal_symbols, brotli_num_literal_symbols, num_literal_htrees);
  s->insert_copy_hgroup = HuffmanTreeGroupInit(
    s, brotli_num_command_symbols, brotli_num_command_symbols, s->num_block_types[1]);
  s->distance_hgroup =
    HuffmanTreeGroupInit(s, num_distance_codes, num_distance_codes, num_dist_htrees);
  // Attempt to allocate local memory first, before going to fb
  s->fb_size = 0;
  HuffmanTreeGroupAlloc(s, s->literal_hgroup);
  HuffmanTreeGroupAlloc(s, s->insert_copy_hgroup);
  HuffmanTreeGroupAlloc(s, s->distance_hgroup);
  if (s->fb_size != 0) {
    // Did not fit in local memory -> allocate fb
    s->fb_base = ext_heap_alloc(s->fb_size, fb_heap_base, fb_heap_size);
    if (!s->fb_base) {
      s->error   = -2;
      s->fb_size = 0;
      return;
    }
    // Repeat allocation falling back to fb
    s->fb_size = 0;
    HuffmanTreeGroupAlloc(s, s->literal_hgroup);
    HuffmanTreeGroupAlloc(s, s->insert_copy_hgroup);
    HuffmanTreeGroupAlloc(s, s->distance_hgroup);
  }
  HuffmanTreeGroupDecode(s, s->literal_hgroup);
  if (s->error) return;
  HuffmanTreeGroupDecode(s, s->insert_copy_hgroup);
  if (s->error) return;
  HuffmanTreeGroupDecode(s, s->distance_hgroup);
}

static __device__ int PrepareLiteralDecoding(debrotli_state_s* s, uint8_t const*& context_map_slice)
{
  int context_mode;
  uint32_t block_type     = s->block_type_rb[1];
  uint32_t context_offset = block_type << 6;
  context_map_slice       = s->context_map + context_offset;
  context_mode            = s->context_modes[block_type];
  return brotli_context_lut(context_mode);
}

/// Decodes a command or literal and updates block type ring-buffer. Reads 3..54 bits.
static __device__ uint32_t DecodeBlockTypeAndLength(debrotli_state_s* s, int tree_type)
{
  uint32_t max_block_type = s->num_block_types[tree_type];
  if (max_block_type > 1) {
    uint16_t const* len_tree  = s->block_type_vlc[tree_type];
    uint16_t const* type_tree = len_tree + brotli_huffman_max_size_26;
    uint8_t* ringbuffer       = &s->block_type_rb[tree_type * 2];
    // Read 0..15 + 3..39 bits.
    uint32_t block_type = getvlc(s, type_tree);
    uint32_t block_len  = getvlc(s, len_tree);
    block_len =
      kBlockLengthPrefixCodeOffset[block_len] + getbits(s, kBlockLengthPrefixCodeBits[block_len]);
    if (block_type == 1) {
      block_type = ringbuffer[1] + 1;
    } else if (block_type == 0) {
      block_type = ringbuffer[0];
    } else {
      block_type -= 2;
    }
    if (block_type >= max_block_type) { block_type -= max_block_type; }
    ringbuffer[0] = ringbuffer[1];
    ringbuffer[1] = (uint8_t)block_type;
    return block_len;
  } else {
    return 0;  // Can only get here because of bitstream error
  }
}

inline __device__ int ToUpperCase(uint8_t* p)
{
  if (p[0] < 0xC0) {
    if (p[0] >= 'a' && p[0] <= 'z') { p[0] ^= 32; }
    return 1;
  }
  // An overly simplified uppercasing model for UTF-8.
  if (p[0] < 0xE0) {
    p[1] ^= 32;
    return 2;
  }
  // An arbitrary transform for three byte characters.
  p[2] ^= 5;
  return 3;
}

static __device__ int TransformDictionaryWord(uint8_t* dst,
                                              uint8_t const* word,
                                              int len,
                                              int transform_idx)
{
  int idx               = 0;
  uint8_t const* prefix = brotli_transform_prefix(transform_idx);
  uint8_t type          = brotli_transform_type(transform_idx);
  uint8_t const* suffix = brotli_transform_suffix(transform_idx);
  {
    int prefix_len = *prefix++;
    while (prefix_len--) {
      dst[idx++] = *prefix++;
    }
  }
  {
    int const t = type;
    int i       = 0;
    if (t <= BROTLI_TRANSFORM_OMIT_LAST_9) {
      len -= t;
    } else if (t >= BROTLI_TRANSFORM_OMIT_FIRST_1 && t <= BROTLI_TRANSFORM_OMIT_FIRST_9) {
      int skip = t - (BROTLI_TRANSFORM_OMIT_FIRST_1 - 1);
      word += skip;
      len -= skip;
    }
    while (i < len) {
      dst[idx++] = word[i++];
    }
    if (t == BROTLI_TRANSFORM_UPPERCASE_FIRST) {
      ToUpperCase(&dst[idx - len]);
    } else if (t == BROTLI_TRANSFORM_UPPERCASE_ALL) {
      uint8_t* uppercase = &dst[idx - len];
      while (len > 0) {
        int step = ToUpperCase(uppercase);
        uppercase += step;
        len -= step;
      }
    }
  }
  {
    int suffix_len = *suffix++;
    while (suffix_len--) {
      dst[idx++] = *suffix++;
    }
    return idx;
  }
}

/// ProcessCommands, actual decoding: 1 warp, most work done by thread0
static __device__ void ProcessCommands(debrotli_state_s* s, brotli_dictionary_s const* words, int t)
{
  int32_t meta_block_len = s->meta_block_len;
  uint8_t* out           = s->out;
  int32_t pos            = 0;
  int p1                 = s->p1;
  int p2                 = s->p2;
  uint16_t const* htree_command;
  uint8_t const *context_map_slice, *dist_context_map_slice;
  int dist_rb_idx;
  uint32_t blen_L, blen_I, blen_D;
  auto* const dict_scratch = reinterpret_cast<uint8_t*>(
    &s->hs);  // 24+13 bytes (max length of a dictionary word including prefix & suffix)
  int context_mode;

  if (!t) {
    context_mode           = PrepareLiteralDecoding(s, context_map_slice);
    dist_context_map_slice = s->dist_context_map;
    htree_command          = s->insert_copy_hgroup->htrees[0];
    dist_rb_idx            = s->dist_rb_idx;
    blen_L                 = s->block_length[0];
    blen_I                 = s->block_length[1];
    blen_D                 = s->block_length[2];
  }
  while (pos < meta_block_len) {
    uint32_t copy_length;
    int32_t distance_code;

    if (!t) {
      if (blen_I == 0) {
        blen_I        = DecodeBlockTypeAndLength(s, 1);
        htree_command = s->insert_copy_hgroup->htrees[s->block_type_rb[3]];
        if (s->cur >= s->end) {
          s->error = 1;
          pos      = meta_block_len;
        }
      }
      // Read the insert/copy length in the command.
      {
        uint32_t cmd_code        = getvlc(s, htree_command);
        CmdLutElement v          = kCmdLut[cmd_code];
        uint8_t distance_context = v.context;
        uint32_t insert_length   = v.insert_len_offset;
        int32_t max_distance;
        distance_code = v.distance_code;
        if (v.insert_len_extra_bits) { insert_length += getbits(s, v.insert_len_extra_bits); }
        copy_length = v.copy_len_offset;
        if (v.copy_len_extra_bits) { copy_length += getbits(s, v.copy_len_extra_bits); }
        --blen_I;
        if (insert_length != 0) {
          if (pos + insert_length > meta_block_len) {
            s->error = -2;
            pos      = meta_block_len;
          }
          // Read the literals in the command.
          else
            do {
              int len;
              if (blen_L == 0) {
                blen_L       = DecodeBlockTypeAndLength(s, 0);
                context_mode = PrepareLiteralDecoding(s, context_map_slice);
              }
              len = min(blen_L, insert_length);
              insert_length -= len;
              blen_L -= len;
              if (brotli_need_context_lut(context_mode)) {
                debrotli_huff_tree_group_s const* literal_hgroup = s->literal_hgroup;
                do {
                  int context = brotli_context(p1, p2, context_mode);
                  p2          = p1;
                  p1          = getvlc(s, literal_hgroup->htrees[context_map_slice[context]]);
                  out[pos++]  = p1;
                } while (--len);
              } else {
                uint16_t const* literal_htree = s->literal_hgroup->htrees[context_map_slice[0]];
                do {
                  p2         = p1;
                  p1         = getvlc(s, literal_htree);
                  out[pos++] = p1;
                } while (--len);
              }
            } while (insert_length);
          if (pos == meta_block_len) { copy_length = 0; }
        }
        // Non-literal symbol
        if (pos < meta_block_len) {
          if (distance_code >= 0) {
            // Implicit distance case.
            --dist_rb_idx;
            distance_code    = s->dist_rb[dist_rb_idx & 3];
            distance_context = 1;
          } else {
            uint16_t const* distance_tree;
            int distval;
            // Read distance code in the command, unless it was implicitly zero.
            if (blen_D == 0) {
              blen_D                 = DecodeBlockTypeAndLength(s, 2);
              dist_context_map_slice = s->dist_context_map + (s->block_type_rb[5] << 2);
            }
            distance_tree = s->distance_hgroup->htrees[dist_context_map_slice[distance_context]];
            distance_code = getvlc(s, distance_tree);
            // Convert the distance code to the actual distance by possibly looking up past
            // distances from the s->ringbuffer.
            distance_context = 0;
            if ((distance_code & ~0xF) == 0) {
              // Take distance from ring buffer
              if (distance_code == 0) {
                --dist_rb_idx;
                distance_code = s->dist_rb[dist_rb_idx & 3];
                // Compensate double distance-ring-buffer roll for dictionary items.
                distance_context = 1;
              } else {
                int dist = distance_code << 1;
                // kDistanceShortCodeIndexOffset has 2-bit values from LSB: 3, 2, 1, 0, 3, 3, 3, 3,
                // 3, 3, 2, 2, 2, 2, 2, 2
                uint32_t const kDistanceShortCodeIndexOffset = 0xAAAF'FF1B;
                // kDistanceShortCodeValueOffset has 2-bit values from LSB: -0, 0,-0, 0,-1, 1,-2,
                // 2,-3, 3,-1, 1,-2, 2,-3, 3
                uint32_t const kDistanceShortCodeValueOffset = 0xFA5F'A500;
                int v         = (dist_rb_idx + (int)(kDistanceShortCodeIndexOffset >> dist)) & 0x3;
                distance_code = s->dist_rb[v];
                v             = (int)(kDistanceShortCodeValueOffset >> dist) & 0x3;
                if ((dist & 0x3) != 0) {
                  distance_code += v;
                } else {
                  distance_code -= v;
                  if (distance_code <= 0) {
                    // A huge distance will cause a failure later on. This is a little faster than
                    // failing here.
                    distance_code = 0x7FFF'FFFF;
                  }
                }
              }
            } else {
              distval = distance_code - (int)s->num_direct_distance_codes;
              if (distval >= 0) {
                uint32_t nbits;
                int postfix;
                int offset;
                if (s->distance_postfix_bits == 0) {
                  nbits  = ((uint32_t)distval >> 1) + 1;
                  offset = ((2 + (distval & 1)) << nbits) - 4;
                  distance_code =
                    (int)s->num_direct_distance_codes + offset + (int)getbits(s, nbits);
                } else {
                  // This branch also works well when s->distance_postfix_bits == 0.
                  uint32_t bits;
                  postfix = distval & s->distance_postfix_mask;
                  distval >>= s->distance_postfix_bits;
                  nbits         = ((uint32_t)distval >> 1) + 1;
                  bits          = getbits(s, nbits);
                  offset        = ((2 + (distval & 1)) << nbits) - 4;
                  distance_code = (int)s->num_direct_distance_codes +
                                  ((offset + (int)bits) << s->distance_postfix_bits) + postfix;
                }
              }
              distance_code = distance_code - brotli_num_distance_short_codes + 1;
            }
            --blen_D;
          }
          max_distance = s->max_backward_distance;
          if (max_distance > (out + pos - s->outbase)) {
            max_distance = (int32_t)(out + pos - s->outbase);
          }
          // Apply copy of LZ77 back-reference, or static dictionary reference if the distance is
          // larger than the max LZ77 distance
          if (distance_code > max_distance) {
            // The maximum allowed distance is brotli_max_allowed_distance = 0x7FFF'FFFC.
            // With this choice, no signed overflow can occur after decoding
            // a special distance code (e.g., after adding 3 to the last distance).
            if (distance_code > brotli_max_allowed_distance ||
                copy_length < brotli_min_dictionary_word_length ||
                copy_length > brotli_max_dictionary_word_length) {
              // printf("distance_code = %d/%d, copy_length = %d\n", distance_code, (int)(out -
              // s->outbase), copy_length);
              s->error    = -1;
              pos         = meta_block_len;
              copy_length = 0;
            } else {
              auto offset            = (int32_t)words->offsets_by_length[copy_length];
              uint32_t shift         = words->size_bits_by_length[copy_length];
              uint32_t address       = distance_code - max_distance - 1;
              int32_t word_idx       = address & ((1 << shift) - 1);
              uint32_t transform_idx = address >> shift;
              // Compensate double distance-ring-buffer roll.
              dist_rb_idx += distance_context;
              offset += word_idx * copy_length;
              if (transform_idx == 0) {
                distance_code = -offset;
              } else if (transform_idx < kNumTransforms) {
                copy_length = TransformDictionaryWord(
                  dict_scratch, &words->data[offset], copy_length, transform_idx);
                distance_code = 0;
                if (copy_length == 1) {
                  // Special case for single byte output
                  p2          = p1;
                  p1          = dict_scratch[0];
                  out[pos++]  = p1;
                  copy_length = 0;
                }
              } else {
                // printf("transform_idx=%d/%d, distance_code = %d/%d, copy_length = %d\n",
                // transform_idx, kNumTransforms, distance_code, (int)(out - s->outbase),
                // copy_length);
                s->error    = -1;
                pos         = meta_block_len;
                copy_length = 0;
              }
              if (pos + copy_length > meta_block_len) {
                s->error    = -1;
                pos         = meta_block_len;
                copy_length = 0;
              }
            }
          } else {
            // Update the recent distances cache.
            s->dist_rb[dist_rb_idx & 3] = distance_code;
            ++dist_rb_idx;
            if (pos + copy_length > meta_block_len) {
              s->error    = -1;
              pos         = meta_block_len;
              copy_length = 0;
            }
          }
        }
      }
    }
    pos         = shuffle(pos);
    copy_length = shuffle(copy_length);
    if (copy_length > 0) {
      uint8_t b;
      distance_code = shuffle(distance_code);
      if (distance_code > 0) {
        // Copy
        for (uint32_t i = t; i < copy_length; i += 32) {
          uint8_t const* src =
            out + pos + ((i >= (uint32_t)distance_code) ? (i % (uint32_t)distance_code) : i) -
            distance_code;
          b            = *src;
          out[pos + i] = b;
        }
      } else {
        // Dictionary
        uint8_t const* src = (distance_code < 0) ? &words->data[-distance_code] : dict_scratch;
        if (t < copy_length) {
          b            = src[t];
          out[pos + t] = b;
          if (32 + t < copy_length) {
            b                 = src[32 + t];
            out[pos + 32 + t] = b;
          }
        }
      }
      p1 = shuffle((uint32_t)b, (copy_length - 1) & 0x1f);
      p2 = shuffle((uint32_t)b, (copy_length - 2) & 0x1f);
      pos += copy_length;
    }
  }

  // Ensure all other threads have observed prior state of p1 & p2 before overwriting
  __syncwarp();

  if (!t) {
    s->p1          = (uint8_t)p1;
    s->p2          = (uint8_t)p2;
    s->dist_rb_idx = dist_rb_idx;
  }
}

/**
 * @brief Brotli decoding kernel
 * See https://datatracker.ietf.org/doc/html/rfc7932
 *
 * blockDim = {block_size,1,1}
 *
 * @param[in] inputs Source buffer per block
 * @param[out] outputs Destination buffer per block
 * @param[out] results Decompressor status per block
 * @param scratch Intermediate device memory heap space (will be dynamically shared between blocks)
 * @param scratch_size Size of scratch heap space (smaller sizes may result in serialization between
 * blocks)
 */
CUDF_KERNEL void __launch_bounds__(block_size, 2)
  gpu_debrotli_kernel(device_span<device_span<uint8_t const> const> inputs,
                      device_span<device_span<uint8_t> const> outputs,
                      device_span<codec_exec_result> results,
                      uint8_t* scratch,
                      uint32_t scratch_size)
{
  __shared__ __align__(16) debrotli_state_s state_g;

  int t                     = threadIdx.x;
  auto const block_id       = blockIdx.x;
  debrotli_state_s* const s = &state_g;

  if (block_id >= inputs.size()) { return; }
  // Thread0: initializes shared state and decode stream header
  if (!t) {
    auto const src      = inputs[block_id].data();
    auto const src_size = inputs[block_id].size();
    if (src && src_size >= 8) {
      s->error           = 0;
      s->out             = outputs[block_id].data();
      s->outbase         = s->out;
      s->bytes_left      = outputs[block_id].size();
      s->mtf_upper_bound = 63;
      s->dist_rb[0]      = 16;
      s->dist_rb[1]      = 15;
      s->dist_rb[2]      = 11;
      s->dist_rb[3]      = 4;
      s->dist_rb_idx     = 0;
      s->p1 = s->p2 = 0;
      initbits(s, src, src_size);
      DecodeStreamHeader(s);
    } else {
      s->error = 1;
      s->out = s->outbase = nullptr;
    }
  }
  __syncthreads();
  if (!s->error) {
    // Main loop: decode meta-blocks
    do {
      __syncthreads();
      if (!t) {
        // Thread0: Decode meta-block header
        DecodeMetaBlockHeader(s);
        if (!s->error && s->meta_block_len > s->bytes_left) { s->error = 2; }
      }
      __syncthreads();
      if (!s->error && s->meta_block_len != 0) {
        if (s->is_uncompressed) {
          // Uncompressed block
          uint8_t const* src = s->cur + ((s->bitpos + 7) >> 3);
          __syncthreads();
          uint8_t* dst = s->out;
          if (!t) {
            if (getbits_bytealign(s) != 0) {
              s->error = -1;
            } else if (src + s->meta_block_len > s->end) {
              s->error = 1;
            } else {
              initbits(s, s->base, s->end - s->base, src - s->base);
            }
          }
          __syncthreads();
          if (!s->error) {
            // Simple block-wide memcpy
            for (int32_t i = t; i < s->meta_block_len; i += block_size) {
              dst[i] = src[i];
            }
          }
        } else {
          // Compressed block
          if (!t) {
            // Thread0: Reset local heap, decode huffman tables
            s->heap_used  = 0;
            s->heap_limit = (uint16_t)(sizeof(s->heap) / sizeof(s->heap[0]));
            s->fb_base    = nullptr;
            s->fb_size    = 0;
            DecodeHuffmanTables(s);
            if (!s->error) { DecodeHuffmanTreeGroups(s, scratch, scratch_size); }
          }
          __syncthreads();
          if (!s->error) {
            // Warp0: Decode compressed block, warps 1..7 are all idle (!)
            if (t < 32)
              ProcessCommands(s, reinterpret_cast<brotli_dictionary_s*>(scratch + scratch_size), t);
            __syncthreads();
          }
          // Free any allocated memory
          if (s->fb_base) {
            if (!t) { ext_heap_free(s->fb_base, s->fb_size, scratch, scratch_size); }
            __syncthreads();
          }
        }
        // Update output byte count and position
        if (!t) {
          s->bytes_left -= s->meta_block_len;
          s->out += s->meta_block_len;
        }
      }
      __syncthreads();
    } while (!s->error && !s->is_last && s->bytes_left != 0);
  }
  __syncthreads();
  // Output decompression status
  if (!t) {
    results[block_id].bytes_written = s->out - s->outbase;
    results[block_id].status = (s->error == 0) ? codec_status::SUCCESS : codec_status::FAILURE;
    // Return ext heap used by last block (statistics)
  }
}

/**
 * @brief Computes the size of temporary memory for Brotli decompression
 *
 * In most case, a brotli metablock will require in the order of ~10KB
 * to ~40KB of scratch space for various lookup tables (mainly context maps
 * and Huffman lookup tables), as well as temporary scratch space to decode
 * the header. However, because the syntax allows for a huge number of unique
 * tables, the theoretical worst case is quite large at ~1.3MB per threadblock,
 * which would scale with gpu occupancy.
 *
 * This is solved by a custom memory allocator that first allocates from a local
 * heap in shared mem (with the end of the heap being used as a stack for
 * intermediate small allocations). Once this is exhausted, the 'external'
 * heap is used, allocating from a single scratch surface shared between all
 * the threadblocks, such that allocation can't fail, but may cause serialization
 * between threadblocks should more than one threadblock ever allocate the worst
 * case size.
 *
 * @param[in] max_num_inputs The maximum number of compressed input chunks
 *
 * @return The size in bytes of required temporary memory
 */
size_t get_gpu_debrotli_scratch_size(int max_num_inputs)
{
  uint32_t max_fb_size, min_fb_size, fb_size;
  auto const sm_count = cudf::detail::num_multiprocessors();
  // no more than 3 blocks/sm at most due to 32KB smem use
  max_num_inputs = std::min(max_num_inputs, sm_count * 3);
  if (max_num_inputs <= 0) {
    max_num_inputs = sm_count * 2;  // Target 2 blocks/SM by default for scratch mem computation
  }
  max_num_inputs = std::min(std::max(max_num_inputs, 1), 512);
  // Max fb size per block occurs if all huffman tables for all 3 group types fail local_alloc()
  // with num_htrees=256 (See HuffmanTreeGroupAlloc)
  max_fb_size = 256 * (630 + 1080 + 920) * 2;  // 1.3MB
  // Min avg fb size needed per block (typical)
  min_fb_size = 10 * 1024;  // TODO: Gather some statistics for typical meta-block size
  // Allocate at least two worst-case metablocks or 1 metablock plus typical size for every other
  // block
  fb_size =
    std::max(max_fb_size * std::min(max_num_inputs, 2), max_fb_size + max_num_inputs * min_fb_size);
  // Add some room for alignment
  auto const aligned_size = fb_size + 16 + sizeof(brotli_dictionary_s);
  auto const clamped_size = std::min(aligned_size, static_cast<size_t>(0xffff'ffffu));
  CUDF_EXPECTS(clamped_size >= sizeof(brotli_dictionary_s),
               "Insufficient scratch space for debrotli");
  return clamped_size;
}

#define DUMP_FB_HEAP 0
#if DUMP_FB_HEAP
#include <stdio.h>
#endif

void gpu_debrotli(device_span<device_span<uint8_t const> const> inputs,
                  device_span<device_span<uint8_t> const> outputs,
                  device_span<codec_exec_result> results,
                  rmm::cuda_stream_view stream)
{
  // Scratch memory for decompressing
  rmm::device_uvector<uint8_t> scratch(
    cudf::io::detail::get_gpu_debrotli_scratch_size(inputs.size()), stream);

  auto const count = inputs.size();
  dim3 dim_block(block_size, 1);
  dim3 dim_grid(count, 1);  // TODO: Check max grid dimensions vs max expected count

  auto const fb_heap_size = (uint32_t)((scratch.size() - sizeof(brotli_dictionary_s)) & ~0xf);

  CUDF_CUDA_TRY(cudaMemsetAsync(scratch.data(), 0, 2 * sizeof(uint32_t), stream.value()));
  // NOTE: The 128KB dictionary copy can have a relatively large overhead since source isn't
  // page-locked
  CUDF_CUDA_TRY(cudf::detail::memcpy_async(
    scratch.data() + fb_heap_size, get_brotli_dictionary(), sizeof(brotli_dictionary_s), stream));
  gpu_debrotli_kernel<<<dim_grid, dim_block, 0, stream.value()>>>(
    inputs, outputs, results, scratch.data(), fb_heap_size);
  CUDF_CUDA_TRY(cudaGetLastError());
#if DUMP_FB_HEAP
  uint32_t dump[2];
  uint32_t cur = 0;
  printf("heap dump (%d bytes)\n", fb_heap_size);
  while (cur < fb_heap_size && !(cur & 3)) {
    CUDF_CUDA_TRY(
      cudf::detail::memcpy_async(&dump[0], scratch.data() + cur, 2 * sizeof(uint32_t), stream));
    stream.synchronize();
    printf("@%d: next = %d, size = %d\n", cur, dump[0], dump[1]);
    cur = (dump[0] > cur) ? dump[0] : 0xffff'ffffu;
  }
#endif
}

}  // namespace cudf::io::detail

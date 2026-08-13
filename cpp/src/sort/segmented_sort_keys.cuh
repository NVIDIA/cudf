/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_buffer.hpp>

#include <cub/device/device_select.cuh>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>

namespace cudf {
namespace detail {

// Element class within the tiered ordering key. The ranks are load-bearing: `element_class` maps
// valid/null to 0/1 per the polarity, and `tier_pad` MUST rank strictly above both --
// `cub::BlockMergeSortStrategy::Sort` fills the slots past a segment's real elements with the pad
// key and requires it ordered after every valid item, so a tie could let the merge displace a real
// element past the valid-item boundary and drop it.
enum class tiered_element_class : cuda::std::uint32_t {
  tier_valid = 0,
  tier_null  = 1,
  tier_pad   = 2
};

/// Block size hosting the register/warp tiered virtual warps and the graduated-string warp bands.
constexpr int TIERED_BLOCK_THREADS = 128;

/**
 * @brief Runtime key polarity realizing one explicit (order, null_order) on the segmented-sort
 * fast paths
 *
 * Every engine orders by an unsigned comparison of a packed key, so the requested configuration is
 * folded into the key bits -- one XOR/class operation per element -- rather than into per-engine
 * comparators, which would multiply kernel instantiations. `descending` complements the encoded
 * value field: an order-reversing bijection confined to the field's exact width, leaving the
 * class/segment bits above it untouched. `nulls_first` picks the null class bit. The default
 * `{false, false}` reproduces the shipped ascending / nulls-last keys bit for bit.
 */
struct sort_polarity {
  bool descending  = false;
  bool nulls_first = false;

  /// Class bit for a valid (0/1) or null (1/0) element: unsigned key order then places nulls on
  /// the requested side of every valid element; the tiered pad class (2) stays strictly above both.
  __host__ __device__ cuda::std::uint32_t element_class(bool is_null) const
  {
    return static_cast<cuda::std::uint32_t>(is_null != nulls_first);
  }
  /// XOR mask reversing a 32-bit encoded value's unsigned order when descending.
  __host__ __device__ cuda::std::uint32_t value_mask32() const
  {
    return descending ? ~cuda::std::uint32_t{0} : cuda::std::uint32_t{0};
  }
  /// XOR mask reversing a 64-bit encoded value's unsigned order when descending.
  __host__ __device__ cuda::std::uint64_t value_mask64() const
  {
    return descending ? ~cuda::std::uint64_t{0} : cuda::std::uint64_t{0};
  }
};

/// null_order is comparator-level -- a descending sort swaps the comparison operands, inverting
/// null placement -- so nulls land first exactly when (BEFORE) != (DESCENDING); a zero-null column
/// relaxes to nulls-last, keeping its keys bit-identical to the shipped configuration.
inline sort_polarity resolve_sort_polarity(bool has_nulls,
                                           order column_order,
                                           null_order null_precedence)
{
  auto const descending  = column_order == order::DESCENDING;
  auto const nulls_first = has_nulls and ((null_precedence == null_order::BEFORE) != descending);
  return sort_polarity{descending, nulls_first};
}

/**
 * @brief Fixed-width radix key ordering runs still tied after the single-`uint64` first pass
 *
 * Twelve bytes with no padding decompose to exactly 96 radix bits (12 passes) while carrying a
 * full eight-byte window; key bytes drive this sort's cost, so shedding a 16-byte key's four pad
 * bytes is a direct data-movement win. Fields are most- to least-significant. `seg_null` holds
 * `(rank << 1) | is_null` -- the dense run rank dominates, so a pass preserves every order an
 * earlier pass resolved and reorders only within a tied run, and within a rank a null sorts after
 * every non-null. `prefix_hi`/`prefix_lo` hold the element's next eight window bytes packed
 * big-endian (byte 0 most significant, exhausted bytes zero-filled), so comparing them in order
 * reproduces unsigned-byte lexicographic order; a null's window words are zero, never its string
 * bytes.
 */
struct prefix_key96 {
  cuda::std::uint32_t seg_null;
  cuda::std::uint32_t prefix_hi;
  cuda::std::uint32_t prefix_lo;
};

static_assert(sizeof(prefix_key96) == 12 and alignof(prefix_key96) == 4,
              "prefix_key96 must stay 12 bytes with 4-byte alignment");

/// Decomposes a `prefix_key96` for `cub::DeviceRadixSort`; the leftmost tuple element is the most
/// significant, ordering by run rank and null flag, then the window words.
struct prefix_decomposer {
  __device__ cuda::std::tuple<cuda::std::uint32_t&, cuda::std::uint32_t&, cuda::std::uint32_t&>
  operator()(prefix_key96& key) const
  {
    return {key.seg_null, key.prefix_hi, key.prefix_lo};
  }
};

/// True when two keys are bit-for-bit equal, i.e. in the same run of an unresolved tie.
__device__ inline bool keys_equal(prefix_key96 const& a, prefix_key96 const& b)
{
  return a.seg_null == b.seg_null && a.prefix_hi == b.prefix_hi && a.prefix_lo == b.prefix_lo;
}

/**
 * @brief Packs a dense run rank (bits 1..31) and a null flag (bit 0) into one 32-bit field
 *
 * The rank comes from an inclusive scan over run-head flags, so it is bounded by the element count
 * (a `size_type`, <= 2^31 - 1) and `(rank << 1) | flag` never overflows; a `static_assert` in the
 * caller proves it.
 */
__device__ inline cuda::std::uint32_t pack_seg_null(cuda::std::uint32_t label,
                                                    cuda::std::uint32_t flag)
{
  return (label << 1) | flag;
}

/// Splits a big-endian-packed eight-byte window into a `prefix_key96`'s two window words.
__device__ inline void split_prefix(cuda::std::uint64_t packed,
                                    cuda::std::uint32_t& prefix_hi,
                                    cuda::std::uint32_t& prefix_lo)
{
  prefix_hi = static_cast<cuda::std::uint32_t>(packed >> 32);
  prefix_lo = static_cast<cuda::std::uint32_t>(packed & 0xFFFF'FFFFu);
}

/**
 * @brief Flags the first position of each maximal run of equal keys (1 = head, 0 = continuation)
 *
 * An inclusive sum over these flags yields a dense one-based run rank shared exactly by equal
 * keys; window keys carry the prior pass's rank in their leading field, so each new rank refines
 * the old grouping without merging elements an earlier pass separated.
 */
struct key_head_flag {
  prefix_key96 const* d_keys;
  __device__ cuda::std::uint32_t operator()(size_type i) const
  {
    if (i == 0) { return 1u; }
    return keys_equal(d_keys[i - 1], d_keys[i]) ? 0u : 1u;
  }
};

/**
 * @brief Flags positions whose key equals a neighbor, i.e. those in a run of two or more
 *
 * Used to compact the sorted order to just the still-tied positions later windows must re-sort;
 * singletons are already final. A null-classed position never counts as tied: nulls are
 * position-final after the first pass. `null_flag` is the `seg_null` bit-0 value marking a null --
 * 1 for the strings path and the numeric nulls-last polarity, 0 under numeric nulls-first, whose
 * valid elements carry bit 1 and must stay tie-detectable.
 */
struct key_tied_flag {
  prefix_key96 const* d_keys;
  size_type const num_elements;
  cuda::std::uint32_t const null_flag = 1u;
  __device__ bool operator()(size_type i) const
  {
    auto const cur = d_keys[i];
    if ((cur.seg_null & 1u) == null_flag) { return false; }
    auto const eq_prev = i > 0 && keys_equal(d_keys[i - 1], cur);
    auto const eq_next = i + 1 < num_elements && keys_equal(d_keys[i + 1], cur);
    return eq_prev || eq_next;
  }
};

/**
 * @brief Runs `cub::DeviceSelect::Flagged` to completion: the temp-storage query pass, then the
 * execute pass
 *
 * Shared by every call site that compacts positions down to their `tied_flags`-true subset --
 * both the numeric two-phase DECIMAL128 sort and the strings prefix-radix loop repeat this exact
 * two-call CUB idiom (size the scratch buffer, then execute) with only the iterator types and the
 * element count differing.
 */
template <typename InputIteratorT, typename OutputIteratorT>
void cub_select_flagged(InputIteratorT d_in,
                        bool const* tied_flags,
                        OutputIteratorT d_out,
                        size_type* d_num_selected,
                        size_type count,
                        rmm::cuda_stream_view stream)
{
  rmm::device_buffer d_temp_storage;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::Flagged(d_temp_storage.data(),
                             temp_storage_bytes,
                             d_in,
                             tied_flags,
                             d_out,
                             d_num_selected,
                             count,
                             stream.value());
  d_temp_storage = rmm::device_buffer{temp_storage_bytes, stream};
  cub::DeviceSelect::Flagged(d_temp_storage.data(),
                             temp_storage_bytes,
                             d_in,
                             tied_flags,
                             d_out,
                             d_num_selected,
                             count,
                             stream.value());
}

struct segment_exceeds_size {
  size_type const* d_offsets;
  size_type limit;
  __device__ bool operator()(size_type i) const
  {
    return (d_offsets[i + 1] - d_offsets[i]) > limit;
  }
};

}  // namespace detail
}  // namespace cudf

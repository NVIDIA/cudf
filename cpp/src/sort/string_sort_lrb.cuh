/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sort_string_loads.cuh"
#include "sort_string_radix_refine.cuh"
#include "string_sort_config.hpp"

#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/indexalator.cuh>
#include <cudf/detail/row_operator/common_utils.cuh>
#include <cudf/dictionary/dictionary_column_view.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/traits.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <cub/device/device_merge_sort.cuh>
#include <cuda/iterator>
#include <cuda/std/bit>
#include <cuda/std/execution>
#include <cuda/stream>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>

namespace cudf::detail::radix_lrb_string_sort {

struct string_character_bounds {
  std::uintptr_t begin;
  std::uintptr_t end;
};

template <typename Keys>
__device__ string_character_bounds string_sort_character_bounds(Keys const& keys)
{
  if constexpr (std::is_same_v<Keys, column_device_view>) {
    auto const offsets = keys.child(0);
    auto const i       = keys.offset() + keys.size();
    auto const end     = offsets.type().id() == type_id::INT64
                           ? offsets.template head<int64_t>()[i]
                           : static_cast<int64_t>(offsets.template head<size_type>()[i]);
    auto const begin   = reinterpret_cast<std::uintptr_t>(keys.template head<char>());
    return {begin, begin + end};
  } else {
    return keys.character_bounds();
  }
}

template <typename Integer, typename Keys>
__device__ Integer
load_column_prefix(Keys const& keys, string_view str, size_type offset, bool funnel)
{
  if (!funnel) return load_big_endian<Integer>(str, offset, false);
  auto const bounds = string_sort_character_bounds(keys);
  return load_big_endian_pool<Integer>(str, offset, bounds.begin, bounds.end);
}

template <int Bytes>
__device__ int string_word_difference(std::conditional_t<Bytes == 8, uint64_t, uint32_t> left,
                                      std::conditional_t<Bytes == 8, uint64_t, uint32_t> right)
{
  auto const difference = left ^ right;
  if (difference == 0) return 0;
  int first_bit;
  if constexpr (Bytes == 8)
    first_bit = __ffsll(static_cast<long long>(difference));
  else
    first_bit = __ffs(static_cast<int>(difference));
  auto const shift = (first_bit - 1) & ~7;
  return static_cast<int>((left >> shift) & 255) - static_cast<int>((right >> shift) & 255);
}

template <int Bytes, bool Funnel = false, int CachedBytes = 0>
__device__ int compare_string_suffix_words(string_view lhs, string_view rhs)
{
  static_assert(cuda::std::endian::native == cuda::std::endian::little);
  using word_type = std::conditional_t<Bytes == 8, uint64_t, uint32_t>;
  auto const n    = min(lhs.size_bytes(), rhs.size_bytes());
  if (lhs.data() == rhs.data() && lhs.size_bytes() == rhs.size_bytes()) return 0;
  size_type i = 0;
  if constexpr (Funnel) {
    if (n >= Bytes) {
      string_funnel_suffix_reader<Bytes, CachedBytes> left{lhs}, right{rhs};
      for (; n - i >= Bytes; i += Bytes) {
        auto const difference = string_word_difference<Bytes>(left.next_full(), right.next_full());
        if (difference != 0) return difference;
      }
    }
  } else {
    for (; n - i >= Bytes; i += Bytes) {
      // memcpy permits unaligned addresses and reads only bytes inside both strings.
      word_type left, right;
      memcpy(&left, lhs.data() + i, Bytes);
      memcpy(&right, rhs.data() + i, Bytes);
      auto const difference = string_word_difference<Bytes>(left, right);
      if (difference != 0) return difference;
    }
  }
  for (; i < n; ++i) {
    auto const left  = static_cast<uint8_t>(lhs.data()[i]);
    auto const right = static_cast<uint8_t>(rhs.data()[i]);
    if (left != right) return static_cast<int>(left) - static_cast<int>((right));
  }
  return (lhs.size_bytes() > rhs.size_bytes()) - (lhs.size_bytes() < rhs.size_bytes());
}

template <int CachedBytes>
__device__ inline bool string_suffix_less(string_view lhs,
                                          string_view rhs,
                                          bool ascending,
                                          int word_bytes)
{
  if (word_bytes == 0) return ascending ? lhs < rhs : rhs < lhs;
  auto const result =
    word_bytes < 0
      ? (word_bytes == -8 ? compare_string_suffix_words<8, true, CachedBytes>(lhs, rhs)
                          : compare_string_suffix_words<4, true, CachedBytes>(lhs, rhs))
      : (word_bytes == 8 ? compare_string_suffix_words<8>(lhs, rhs)
                         : compare_string_suffix_words<4>(lhs, rhs));
  return ascending ? result < 0 : result > 0;
}

template <size_type CachedBytes, typename Keys>
__device__ bool cached_prefix_tie_break(
  Keys const& d_column, size_type lhs, size_type rhs, bool ascending, int suffix_word_bytes = 0)
{
  auto const left_element  = d_column.template element<string_view>(lhs);
  auto const right_element = d_column.template element<string_view>(rhs);
  auto const left_size     = left_element.size_bytes();
  auto const right_size    = right_element.size_bytes();
  if (left_size <= CachedBytes or right_size <= CachedBytes) {
    return ascending ? left_size < right_size : right_size < left_size;
  }
  auto const left_suffix = string_view{left_element.data() + CachedBytes, left_size - CachedBytes};
  auto const right_suffix =
    string_view{right_element.data() + CachedBytes, right_size - CachedBytes};
  return string_suffix_less<CachedBytes>(left_suffix, right_suffix, ascending, suffix_word_bytes);
}

// Typed offset storage removes a per-comparison runtime offset-width dispatch.
template <typename Offset>
struct typed_radix_string_storage {
  char const* chars;
  Offset const* offsets;
  bitmask_type const* null_mask;
  size_type row_offset;
  size_type rows;

  __device__ string_character_bounds character_bounds() const
  {
    auto const begin = reinterpret_cast<std::uintptr_t>(chars);
    return {begin, begin + static_cast<int64_t>(offsets[row_offset + rows])};
  }
  __device__ bool is_null(size_type row) const
  {
    return null_mask != nullptr && !cudf::bit_is_set(null_mask, row + row_offset);
  }
  template <typename T>
  __device__ T element(size_type row) const
  {
    static_assert(std::is_same_v<T, string_view>);
    auto const i     = row + row_offset;
    auto const begin = offsets[i];
    auto const end   = offsets[i + 1];
    return string_view{chars == nullptr ? nullptr : chars + begin,
                       static_cast<size_type>(end - begin)};
  }
};

template <int Bytes, bool has_nulls, typename Keys = column_device_view>
struct radix_string_prefix_extractor {
  using key_type =
    std::conditional_t<has_nulls,
                       string_radix_prefix_key,
                       std::conditional_t<Bytes == 8, uint64_t, string_radix_prefix12>>;
  __device__ key_type operator()(size_type row) const
  {
    if constexpr (has_nulls) {
      if (keys.is_null(row)) return string_radix_prefix_key{0, 0, null_rank};
    }
    auto const str = keys.template element<string_view>(row);
    auto hi        = load_column_prefix<uint64_t>(keys, str, 0, prefix_funnel);
    uint32_t lo    = 0;
    if constexpr (Bytes == 12) lo = load_column_prefix<uint32_t>(keys, str, 8, prefix_funnel);
    if (!ascending) {
      hi = ~hi;
      if constexpr (Bytes == 12) lo = ~lo;
    }
    if constexpr (!has_nulls && Bytes == 8)
      return hi;
    else if constexpr (!has_nulls && Bytes == 12)
      return string_radix_prefix12{static_cast<uint32_t>(hi >> 32), static_cast<uint32_t>(hi), lo};
    else
      return string_radix_prefix_key{hi, lo, 1u - null_rank};
  }
  Keys keys;
  bool ascending;
  uint32_t null_rank;
  bool prefix_funnel;
};

template <typename Comparator>
struct radix_proven_suffix_comparator : Comparator {
  radix_lrb_proof const* proofs;
  radix_lrb_metadata const* metadata;
  size_type base{};
  size_type skip{8};
  bool complete{};

  __device__ Comparator base_comparator() const { return static_cast<Comparator const&>(*this); }

  __device__ auto bind_bin(size_type segment_base) const
  {
    auto result = *this;
    result.base = segment_base;
    return result;
  }
  __device__ auto bind_segment(size_type segment) const
  {
    auto result        = *this;
    auto const ordinal = base + segment - metadata->segment_base[this->proof_first_bin()];
    if (ordinal >= 0) {
      auto const proof = proofs[ordinal];
      result.skip      = max(size_type{8}, static_cast<size_type>(proof.prefix));
      result.complete  = (proof.flags & 1u) == 0;
    }
    return result;
  }
  __device__ bool operator()(size_type lhs, size_type rhs) const
  {
    if (lhs == -1 || rhs == -1) return rhs == -1 && lhs != -1;
    auto const left  = this->keys.template element<string_view>(lhs);
    auto const right = this->keys.template element<string_view>(rhs);
    auto const l = left.size_bytes(), r = right.size_bytes();
    int result;
    if (l <= skip || r <= skip)
      result = (l > r) - (l < r);
    else
      result = compare_string_suffix_words<8, true, 8>(string_view{left.data() + skip, l - skip},
                                                       string_view{right.data() + skip, r - skip});
    return this->ascending ? result < 0 : result > 0;
  }
};

template <int Bytes, typename Keys = column_device_view>
struct radix_string_suffix_comparator {
  __device__ bool operator()(size_type lhs, size_type rhs) const
  {
    if (lhs == -1 || rhs == -1) return rhs == -1 && lhs != -1;
    return cached_prefix_tie_break<Bytes>(keys, lhs, rhs, ascending, suffix_word_bytes);
  }
  __host__ __device__ int proof_first_bin() const { return 16; }

  // Common bytes after the already equal R8 prefix, proved against one reference row.
  __device__ unsigned int common_prefix(size_type lhs, size_type rhs) const
  {
    auto const left  = keys.template element<string_view>(lhs);
    auto const right = keys.template element<string_view>(rhs);
    auto const n     = min(left.size_bytes(), right.size_bytes());
    if (left.data() == right.data()) return n;
    size_type i = min(size_type{8}, n);
    if (n - i >= 8) {
      string_funnel_suffix_reader<8, 8> a{string_view{left.data() + i, n - i}};
      string_funnel_suffix_reader<8, 8> b{string_view{right.data() + i, n - i}};
      for (; n - i >= 8; i += 8) {
        auto const difference = a.next_full() ^ b.next_full();
        if (difference != 0) return i + (__ffsll(static_cast<long long>(difference)) - 1) / 8;
      }
    }
    for (; i < n; ++i)
      if (left.data()[i] != right.data()[i]) break;
    return i;
  }
  __device__ bool probe_run(size_type const* output,
                            radix_segment seg,
                            size_type size,
                            bool extra_radix) const
  {
    auto const reference = output[seg.begin];
    auto const length    = keys.template element<string_view>(reference).size_bytes();
    bool duplicate       = true;
    unsigned int common  = length;
    size_type max_length = length;
    size_type min_length = length;
    for (int q = 1; q <= 3; ++q) {
      auto const row    = output[seg.begin + static_cast<int64_t>(seg.end - seg.begin - 1) * q / 3];
      auto const n      = keys.template element<string_view>(row).size_bytes();
      auto const prefix = common_prefix(reference, row);
      common            = min(common, prefix);
      max_length        = max(max_length, n);
      min_length        = min(min_length, n);
      duplicate         = duplicate && n == length && prefix == static_cast<unsigned int>(length);
    }
    bool const full = seg.begin == 0 && seg.end == size;
    return common >= 24 || (duplicate && length <= 128) ||
           (extra_radix && full && min_length == max_length &&
            max_length <= static_cast<size_type>(max(8u, common)) + 8);
  }
  auto make_proof_comparator(radix_lrb_proof const* proofs,
                             radix_lrb_metadata const* metadata) const
  {
    return radix_proven_suffix_comparator<radix_string_suffix_comparator>{*this, proofs, metadata};
  }
  bool try_suffix_radix(radix_lrb_proof proof,
                        size_type size,
                        size_type* output,
                        size_type* scratch,
                        uint64_t* sorted_keys,
                        uint64_t* temporary_keys,
                        cuda::stream_ref stream) const
  {
    auto const skip      = max(size_type{8}, static_cast<size_type>(proof.prefix));
    auto const remaining = proof.max_length - skip;
    if ((proof.flags & 2u) == 0 || proof.min_length != proof.max_length || remaining <= 0 ||
        remaining > 8)
      return false;
    auto const mr      = cudf::get_current_device_resource_ref();
    auto const storage = keys;
    auto const forward = ascending;
    thrust::transform(rmm::exec_policy_nosync(stream, mr),
                      output,
                      output + size,
                      sorted_keys,
                      [storage, forward, skip] __device__(size_type row) {
                        auto const str = storage.template element<string_view>(row);
                        auto const key = load_column_prefix<uint64_t>(storage, str, skip, true);
                        return forward ? key : ~key;
                      });
    cub::DoubleBuffer<uint64_t> key_buffers{sorted_keys, temporary_keys};
    cub::DoubleBuffer<size_type> row_buffers{output, scratch};
    std::size_t bytes = 0;
    auto sort         = [&](void* temp) {
      return cub::DeviceRadixSort::SortPairs(
        temp, bytes, key_buffers, row_buffers, size, 64 - 8 * remaining, 64, stream.get());
    };
    CUDF_CUDA_TRY(sort(nullptr));
    rmm::device_buffer temporary(bytes, stream, mr);
    CUDF_CUDA_TRY(sort(temporary.data()));
    if (row_buffers.Current() != output)
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(
        output, row_buffers.Current(), sizeof(size_type) * size, stream));
    return true;
  }
  Keys keys;
  bool ascending;
  int suffix_word_bytes{};
  bool contiguous_merge{};
  static constexpr bool enable_proof        = true;
  static constexpr bool enable_suffix_radix = true;
};

template <typename Keys>
struct three_way_radix_string_comparator : radix_string_suffix_comparator<8, Keys> {
  __device__ int compare_three_way(size_type lhs, size_type rhs) const
  {
    auto const left       = this->keys.template element<string_view>(lhs);
    auto const right      = this->keys.template element<string_view>(rhs);
    auto const left_size  = left.size_bytes();
    auto const right_size = right.size_bytes();
    int result;
    if (left_size <= 8 || right_size <= 8) {
      result = (left_size > right_size) - (left_size < right_size);
    } else {
      result = compare_string_suffix_words<8, true, 8>(
        string_view{left.data() + 8, left_size - 8}, string_view{right.data() + 8, right_size - 8});
    }
    return this->ascending ? result : -result;
  }
};

/**
 * @brief The tuned R8 recipe: 8-byte radix prefix + row index, LRB refinement, funnel loads.
 *
 * Null ranks participate in the radix key; row IDs remain payloads. Equal prefixes are refined
 * using hybrid warp sorts up to 32 rows, tiered block sorts, and hierarchical stable merges.
 * Native scheduling reads bin metadata once, with a second small read only for a promising
 * full-span suffix-radix candidate. Schedule 2 keeps metadata on device
 * for capture and graph replay. Legacy CUDF_STRING_SORT_* tuning settings do not affect this path.
 */
template <bool Stable, bool HasNulls, typename Keys>
void refined_order(Keys const& keys,
                   mutable_column_view& indices,
                   bool ascending,
                   null_order null_precedence,
                   cuda::stream_ref stream)
{
  auto const null_rank =
    HasNulls ? (ascending == (null_precedence == null_order::BEFORE) ? 0u : 1u) : 2u;
  auto const extractor =
    radix_string_prefix_extractor<8, HasNulls, Keys>{keys, ascending, null_rank, true};
  auto const base_comparator = radix_string_suffix_comparator<8, Keys>{keys, ascending, -8, true};
  auto const comparator      = three_way_radix_string_comparator<Keys>{base_comparator};
  radix_prefix_refine<8, 256, 4, Stable>(indices.size(),
                                         indices.begin<size_type>(),
                                         extractor,
                                         comparator,
                                         null_rank,
                                         true,  // run-length encoding
                                         true,  // device metadata
                                         true,  // compact nontrivial runs
                                         configured_radix_lrb_schedule(),
                                         2,   // tiered logarithmic bins
                                         3,   // hybrid warp sorting
                                         32,  // persistent warps per SM
                                         stream);
}

template <bool Stable>
void sorted_order(column_view const& input,
                  mutable_column_view& indices,
                  bool ascending,
                  null_order null_precedence,
                  cuda::stream_ref stream)
{
  auto const strings   = strings_column_view{input};
  auto const all_equal = !input.has_nulls() && strings.chars_begin(stream) == nullptr;
  if (input.size() < 2 || input.null_count() == input.size() || all_equal) {
    thrust::sequence(rmm::exec_policy_nosync(stream, cudf::get_current_device_resource_ref()),
                     indices.begin<size_type>(),
                     indices.end<size_type>(),
                     size_type{0});
    return;
  }
  auto const offsets  = strings.offsets();
  bool const large    = offsets.type().id() == type_id::INT64;
  auto const data     = large ? static_cast<void const*>(offsets.data<int64_t>())
                              : static_cast<void const*>(offsets.data<size_type>());
  auto const dispatch = [&]<typename Storage>(Storage const& storage) {
    if (input.has_nulls())
      refined_order<Stable, true>(storage, indices, ascending, null_precedence, stream);
    else
      refined_order<Stable, false>(storage, indices, ascending, null_precedence, stream);
  };
  if (large) {
    dispatch(typed_radix_string_storage<int64_t>{strings.chars_begin(stream),
                                                 static_cast<int64_t const*>(data),
                                                 input.null_mask(),
                                                 input.offset(),
                                                 input.size()});
  } else {
    dispatch(typed_radix_string_storage<size_type>{strings.chars_begin(stream),
                                                   static_cast<size_type const*>(data),
                                                   input.null_mask(),
                                                   input.offset(),
                                                   input.size()});
  }
}

}  // namespace cudf::detail::radix_lrb_string_sort

/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "sort_string_loads.cuh"
#include "sort_string_radix_refine.cuh"

#include <cudf/column/column_view.hpp>
#include <cudf/detail/utilities/cuda_memcpy.hpp>
#include <cudf/detail/utilities/getenv_or.hpp>
#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cub/device/device_radix_sort.cuh>
#include <cuda/buffer>
#include <cuda/std/bit>
#include <cuda/stream>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cudf::detail::radix_lrb_string_sort {

/**
 * @brief Whether `LIBCUDF_STRING_SORT_RADIX_LRB` opts single string columns into this sort.
 *
 * The environment is read once, before the first libcudf string sort.
 */
[[nodiscard]] inline bool enabled()
{
  static bool const enabled = get_bool_env_or("LIBCUDF_STRING_SORT_RADIX_LRB", false);
  return enabled;
}

struct string_character_bounds {
  std::uintptr_t begin;
  std::uintptr_t end;
};

__device__ inline int string_word_difference(uint64_t left, uint64_t right)
{
  auto const difference = left ^ right;
  if (difference == 0) return 0;
  auto const shift = (__ffsll(static_cast<long long>(difference)) - 1) & ~7;
  return static_cast<int>((left >> shift) & 255) - static_cast<int>((right >> shift) & 255);
}

/**
 * @brief Three-way bytewise comparison of suffixes that follow at least eight equal bytes.
 */
__device__ inline int compare_string_suffix_words(string_view lhs, string_view rhs)
{
  static_assert(cuda::std::endian::native == cuda::std::endian::little);
  auto const n = min(lhs.size_bytes(), rhs.size_bytes());
  if (lhs.data() == rhs.data() && lhs.size_bytes() == rhs.size_bytes()) return 0;
  size_type i = 0;
  if (n >= 8) {
    string_funnel_suffix_reader left{lhs}, right{rhs};
    for (; n - i >= 8; i += 8) {
      auto const difference = string_word_difference(left.next_full(), right.next_full());
      if (difference != 0) return difference;
    }
  }
  for (; i < n; ++i) {
    auto const left  = static_cast<uint8_t>(lhs.data()[i]);
    auto const right = static_cast<uint8_t>(rhs.data()[i]);
    if (left != right) return static_cast<int>(left) - static_cast<int>(right);
  }
  return (lhs.size_bytes() > rhs.size_bytes()) - (lhs.size_bytes() < rhs.size_bytes());
}

/**
 * @brief Three-way comparison after the equal eight-byte radix prefix.
 */
template <typename Keys>
__device__ int compare_after_prefix(Keys const& keys, size_type lhs, size_type rhs, size_type skip)
{
  auto const left  = keys.template element<string_view>(lhs);
  auto const right = keys.template element<string_view>(rhs);
  auto const l = left.size_bytes(), r = right.size_bytes();
  if (l <= skip || r <= skip) return (l > r) - (l < r);
  return compare_string_suffix_words(string_view{left.data() + skip, l - skip},
                                     string_view{right.data() + skip, r - skip});
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

/**
 * @brief Extracts an order-preserving eight-byte prefix; nullable keys add a null rank.
 */
template <bool has_nulls, typename Keys>
struct radix_string_prefix_extractor {
  using key_type = std::conditional_t<has_nulls, string_radix_prefix_key, uint64_t>;
  __device__ key_type operator()(size_type row) const
  {
    if constexpr (has_nulls) {
      if (keys.is_null(row)) return string_radix_prefix_key{0, null_rank};
    }
    auto const str    = keys.template element<string_view>(row);
    auto const bounds = keys.character_bounds();
    auto hi           = load_big_endian_pool<uint64_t>(str, 0, bounds.begin, bounds.end);
    if (!ascending) hi = ~hi;
    if constexpr (has_nulls)
      return string_radix_prefix_key{hi, 1u - null_rank};
    else
      return hi;
  }
  Keys keys;
  bool ascending;
  uint32_t null_rank;
};

/**
 * @brief Compares strings after the proven common bytes of their run.
 */
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
    auto const result = compare_after_prefix(this->keys, lhs, rhs, skip);
    return this->ascending ? result < 0 : result > 0;
  }
};

/**
 * @brief Orders rows whose eight-byte prefixes are equal by their remaining bytes.
 */
template <typename Keys>
struct radix_string_suffix_comparator {
  __device__ bool operator()(size_type lhs, size_type rhs) const
  {
    if (lhs == -1 || rhs == -1) return rhs == -1 && lhs != -1;
    return compare_three_way(lhs, rhs) < 0;
  }
  __device__ int compare_three_way(size_type lhs, size_type rhs) const
  {
    auto const result = compare_after_prefix(keys, lhs, rhs, 8);
    return ascending ? result : -result;
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
      string_funnel_suffix_reader a{string_view{left.data() + i, n - i}};
      string_funnel_suffix_reader b{string_view{right.data() + i, n - i}};
      for (; n - i >= 8; i += 8) {
        auto const difference = a.next_full() ^ b.next_full();
        if (difference != 0) return i + (__ffsll(static_cast<long long>(difference)) - 1) / 8;
      }
    }
    for (; i < n; ++i)
      if (left.data()[i] != right.data()[i]) break;
    return i;
  }
  __device__ bool probe_run(size_type const* output, radix_segment seg, size_type size) const
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
           (full && min_length == max_length &&
            max_length <= static_cast<size_type>(max(8u, common)) + 8);
  }
  auto make_proof_comparator(radix_lrb_proof const* proofs,
                             radix_lrb_metadata const* metadata) const
  {
    return radix_proven_suffix_comparator<radix_string_suffix_comparator>{*this, proofs, metadata};
  }
  // A single full-input run of equal-length strings with 1-8 unproven bytes is radix sorted.
  bool try_suffix_radix(radix_lrb_proof proof,
                        size_type size,
                        size_type* output,
                        size_type* scratch,
                        uint64_t* sorted_keys,
                        uint64_t* temporary_keys,
                        cuda::stream_ref stream,
                        rmm::device_async_resource_ref temp_mr) const
  {
    auto const skip      = max(size_type{8}, static_cast<size_type>(proof.prefix));
    auto const remaining = proof.max_length - skip;
    if ((proof.flags & 2u) == 0 || proof.min_length != proof.max_length || remaining <= 0 ||
        remaining > 8)
      return false;
    auto const storage = keys;
    auto const forward = ascending;
    thrust::transform(rmm::exec_policy_nosync(stream, temp_mr),
                      output,
                      output + size,
                      sorted_keys,
                      [storage, forward, skip] __device__(size_type row) {
                        auto const str    = storage.template element<string_view>(row);
                        auto const bounds = storage.character_bounds();
                        auto const key =
                          load_big_endian_pool<uint64_t>(str, skip, bounds.begin, bounds.end);
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
    auto temporary = cuda::device_buffer<std::byte>(stream, temp_mr, bytes, cuda::no_init);
    CUDF_CUDA_TRY(sort(temporary.data()));
    if (row_buffers.Current() != output)
      CUDF_CUDA_TRY(cudf::detail::memcpy_async(
        output, row_buffers.Current(), sizeof(size_type) * size, stream));
    return true;
  }
  Keys keys;
  bool ascending;
};

/**
 * @brief Eight-byte radix prefix + row index, then LRB refinement of equal-prefix runs.
 *
 * Null ranks participate in the radix key; row IDs remain payloads. Equal prefixes are refined
 * using hybrid warp sorts up to 32 rows, tiered block sorts, and hierarchical stable merges.
 */
template <bool Stable, bool HasNulls, typename Keys>
void refined_order(Keys const& keys,
                   mutable_column_view& indices,
                   bool ascending,
                   null_order null_precedence,
                   cuda::stream_ref stream,
                   rmm::device_async_resource_ref temp_mr)
{
  auto const null_rank =
    HasNulls ? (ascending == (null_precedence == null_order::BEFORE) ? 0u : 1u) : 2u;
  auto const extractor  = radix_string_prefix_extractor<HasNulls, Keys>{keys, ascending, null_rank};
  auto const comparator = radix_string_suffix_comparator<Keys>{keys, ascending};
  radix_prefix_refine<256, 4, Stable>(
    indices.size(), indices.begin<size_type>(), extractor, comparator, null_rank, stream, temp_mr);
}

template <bool Stable>
void sorted_order(column_view const& input,
                  mutable_column_view& indices,
                  bool ascending,
                  null_order null_precedence,
                  cuda::stream_ref stream,
                  rmm::device_async_resource_ref temp_mr)
{
  auto const strings   = strings_column_view{input};
  auto const all_equal = !input.has_nulls() && strings.chars_begin(stream) == nullptr;
  if (input.size() < 2 || input.null_count() == input.size() || all_equal) {
    thrust::sequence(rmm::exec_policy_nosync(stream, temp_mr),
                     indices.begin<size_type>(),
                     indices.end<size_type>(),
                     size_type{0});
    return;
  }
  auto const dispatch = [&]<typename Offset>() {
    auto const storage = typed_radix_string_storage<Offset>{strings.chars_begin(stream),
                                                            strings.offsets().data<Offset>(),
                                                            input.null_mask(),
                                                            input.offset(),
                                                            input.size()};
    if (input.has_nulls())
      refined_order<Stable, true>(storage, indices, ascending, null_precedence, stream, temp_mr);
    else
      refined_order<Stable, false>(storage, indices, ascending, null_precedence, stream, temp_mr);
  };
  if (strings.offsets().type().id() == type_id::INT64)
    dispatch.template operator()<int64_t>();
  else
    dispatch.template operator()<size_type>();
}

}  // namespace cudf::detail::radix_lrb_string_sort

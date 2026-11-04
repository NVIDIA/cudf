/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_device_view.cuh>
#include <cudf/detail/cuco_helpers.hpp>
#include <cudf/hashing/detail/murmurhash3_x86_32.cuh>
#include <cudf/hashing/detail/xxhash_64.cuh>
#include <cudf/strings/detail/utf8.hpp>
#include <cudf/strings/string_view.cuh>

#include <nvtext/byte_pair_encoding.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/static_map.cuh>
#include <cuco/static_set.cuh>
#include <cuda/std/functional>
#include <cuda/std/iterator>
#include <cuda/std/limits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cstdint>
#include <type_traits>

namespace nvtext {
namespace detail {

using cuco_storage = cuco::storage<1>;

/**
 * @brief 64-bit fingerprint identifying a merge pair
 *
 * The two halves of a pair are always contiguous in memory: adjacent tokens in the
 * input and adjacent rows (2i, 2i+1) in the merge-pairs strings column.
 * So a pair is identified by hashing its bytes together with the size of the
 * left half folded into the seed.
 */
using fingerprint_type                       = uint64_t;
constexpr fingerprint_type empty_fingerprint = ~fingerprint_type{0};

__device__ inline fingerprint_type pair_fingerprint(char const* data,
                                                    cudf::size_type lhs_size,
                                                    cudf::size_type size,
                                                    uint64_t salt)
{
  auto const fp =
    cudf::hashing::detail::XXHash_64<cudf::string_view>{salt ^ static_cast<uint64_t>(lhs_size)}
      .compute_bytes(reinterpret_cast<cuda::std::byte const*>(data), size);
  return fp == empty_fingerprint ? fp - 1 : fp;  // reserved for empty slots
}

/**
 * @brief Hasher for the fingerprint map; the fingerprint is already a hash
 */
struct fingerprint_hasher {
  __device__ uint32_t operator()(fingerprint_type fp) const
  {
    return static_cast<uint32_t>(fp >> 32) ^ static_cast<uint32_t>(fp);
  }
};

/**
 * @brief Maps merge pair fingerprints to their rank (the pair's row in the table)
 */
using merge_pairs_map_type = cuco::static_map<fingerprint_type,
                                              cudf::size_type,
                                              cuco::extent<std::size_t>,
                                              cuda::thread_scope_device,
                                              cuda::std::equal_to<fingerprint_type>,
                                              cuco::linear_probing<1, fingerprint_hasher>,
                                              rmm::mr::polymorphic_allocator<char>,
                                              cuco_storage>;

/**
 * @brief Device functor for looking up the rank of a pair of adjacent tokens
 *
 * The fingerprint is used to locate the candidate entry which is then verified
 * against the merge-pairs strings so the result is exact.
 *
 * @tparam MapRefType The type of the fingerprint map finder object
 */
template <typename MapRefType>
struct rank_finder {
  MapRefType const d_map;
  cudf::column_device_view const d_merge_pairs;
  uint64_t const salt;

  static constexpr cudf::size_type no_rank = cuda::std::numeric_limits<cudf::size_type>::max();

  /**
   * @brief Returns the rank of the pair or `no_rank` if it is not in the table
   *
   * @param data Start of the left token; the right token immediately follows it
   * @param lhs_size Size of the left token in bytes
   * @param size Size of both tokens in bytes
   */
  __device__ cudf::size_type find(char const* data,
                                  cudf::size_type lhs_size,
                                  cudf::size_type size) const
  {
    auto const itr = d_map.find(pair_fingerprint(data, lhs_size, size, salt));
    if (itr == d_map.end()) { return no_rank; }
    auto const rank  = itr->second;
    auto const left  = d_merge_pairs.element<cudf::string_view>(rank * 2);
    auto const right = d_merge_pairs.element<cudf::string_view>(rank * 2 + 1);
    auto const match = left.size_bytes() == lhs_size &&
                       left.size_bytes() + right.size_bytes() == size &&
                       cudf::string_view(left.data(), size) == cudf::string_view(data, size);
    return match ? rank : no_rank;
  }
};

/**
 * @brief Key identifying the characters on either side of a merge point
 *
 * Built from the last character of a merge pair's left half and the first
 * character of its right half. Two adjacent characters `a|b` in the input can
 * only ever be merged if `(a,b)` is one of these keys since any merge across
 * that position must have a left half ending in `a` and a right half starting with `b`.
 */
using cross_key_type = uint64_t;

__device__ inline cross_key_type make_cross_key(char const* lhs_last, char const* rhs_first)
{
  cudf::char_utf8 lhs = 0;
  cudf::char_utf8 rhs = 0;
  cudf::strings::detail::to_char_utf8(lhs_last, lhs);
  cudf::strings::detail::to_char_utf8(rhs_first, rhs);
  return (static_cast<cross_key_type>(lhs) << 32) | static_cast<cross_key_type>(rhs);
}

/**
 * @brief Hasher for the cross-character set
 */
struct cross_hasher {
  using hasher_type = cudf::hashing::detail::MurmurHash3_x86_32<cross_key_type>;
  __device__ hasher_type::result_type operator()(cross_key_type key) const
  {
    return hasher_type{}(key);
  }
};

/**
 * @brief Set of cross-character keys built from the merge pairs table
 */
using cross_set_type = cuco::static_set<cross_key_type,
                                        cuco::extent<std::size_t>,
                                        cuda::thread_scope_device,
                                        cuda::std::equal_to<cross_key_type>,
                                        cuco::linear_probing<1, cross_hasher>,
                                        rmm::mr::polymorphic_allocator<char>,
                                        cuco_storage>;

}  // namespace detail

// since column_device_view::create() returns is a little more than
// std::unique_ptr<column_device_view> this helper simplifies the return type for us
using col_device_view = std::invoke_result_t<decltype(&cudf::column_device_view::create),
                                             cudf::column_view,
                                             cuda::stream_ref,
                                             rmm::device_async_resource_ref>;

struct bpe_merge_pairs::bpe_merge_pairs_impl {
  std::unique_ptr<cudf::column> const merge_pairs;
  col_device_view const d_merge_pairs;
  std::unique_ptr<detail::merge_pairs_map_type> merge_pairs_map;  // for BPE
  std::unique_ptr<detail::cross_set_type> cross_set;              // for locating unpairables

  uint64_t const salt;  // seed adjustment used to create unique fingerprints

  bpe_merge_pairs_impl(std::unique_ptr<cudf::column>&& merge_pairs,
                       col_device_view&& d_merge_pairs,
                       std::unique_ptr<detail::merge_pairs_map_type>&& merge_pairs_map,
                       std::unique_ptr<detail::cross_set_type>&& cross_set,
                       uint64_t salt);

  auto const get_merge_pairs() const { return *d_merge_pairs; }
  auto get_rank_finder() const
  {
    auto map_ref = merge_pairs_map->ref(cuco::op::find);
    return detail::rank_finder<decltype(map_ref)>{map_ref, *d_merge_pairs, salt};
  }
  auto get_cross_set_ref() const { return cross_set->ref(cuco::op::contains); }
};

}  // namespace nvtext

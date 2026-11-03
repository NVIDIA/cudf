/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/bpe/byte_pair_encoding.cuh"

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/algorithms/reduce.cuh>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <nvtext/byte_pair_encoding.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/mr/polymorphic_allocator.hpp>

#include <cuco/utility/reduction_functors.cuh>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/stream>

#include <fstream>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

namespace nvtext {
namespace detail {
namespace {

/**
 * @brief Builds the map of merge pair fingerprints to ranks
 *
 * Duplicate pairs are allowed and the lowest rank (first occurrence) is kept.
 * In the very unlikely event two different pairs produce the same fingerprint,
 * the map is rebuilt using a different salt.
 *
 * @param input Merge pairs with each pair's halves in adjacent rows
 * @param stream CUDA stream used for device memory operations and kernel launches
 * @param mr Memory resources used for temporary allocations and the returned map
 * @return The map and the salt used to create the fingerprints
 */
std::pair<std::unique_ptr<detail::merge_pairs_map_type>, uint64_t> initialize_merge_pairs_map(
  cudf::column_device_view const& input, cuda::stream_ref stream, cudf::memory_resources mr)
{
  auto const elements         = input.size() / 2;
  auto constexpr max_attempts = 4;
  // the table is small and most lookups are for pairs that are not in it;
  // a lower load factor keeps the probe sequences for these misses short
  auto constexpr merge_pairs_load_factor = 0.25;

  std::unique_ptr<detail::merge_pairs_map_type> merge_pairs_map;
  uint64_t salt = 0;
  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    salt            = static_cast<uint64_t>(attempt) * 0x9E37'79B9'7F4A'7C15ul;
    merge_pairs_map = std::make_unique<merge_pairs_map_type>(
      static_cast<size_t>(elements),
      merge_pairs_load_factor,
      cuco::empty_key{empty_fingerprint},
      cuco::empty_value{-1},
      cuda::std::equal_to<fingerprint_type>{},
      cuco::linear_probing<1, fingerprint_hasher>{},
      cuco::thread_scope_device,
      cuco_storage{},
      rmm::mr::polymorphic_allocator<char>{mr.get_output_mr()},
      stream.get());

    // the two halves of each pair are adjacent rows and therefore contiguous in memory
    auto iter = cudf::detail::make_counting_transform_iterator(
      0,
      cuda::proclaim_return_type<cuco::pair<fingerprint_type, cudf::size_type>>(
        [input, salt] __device__(cudf::size_type idx) {
          auto const lhs = input.element<cudf::string_view>(idx * 2);
          auto const rhs = input.element<cudf::string_view>(idx * 2 + 1);
          auto const fp  = pair_fingerprint(
            lhs.data(), lhs.size_bytes(), lhs.size_bytes() + rhs.size_bytes(), salt);
          return cuco::make_pair(fp, idx);
        }));

    // duplicate pairs keep the lowest rank
    merge_pairs_map->insert_or_apply(iter, iter + elements, cuco::reduce::min{}, stream);

    // each pair finds the lowest ranked pair with the same fingerprint;
    // a collision is when that pair has different strings
    auto const map_ref      = merge_pairs_map->ref(cuco::op::find);
    auto const is_collision = cuda::transform_iterator(
      cuda::counting_iterator<cudf::size_type>{0},
      cuda::proclaim_return_type<cudf::size_type>(
        [input, map_ref, iter] __device__(cudf::size_type idx) -> cudf::size_type {
          auto const other = map_ref.find(iter[idx].first)->second;
          if (other == idx) { return 0; }
          return input.element<cudf::string_view>(idx * 2) !=
                   input.element<cudf::string_view>(other * 2) ||
                 input.element<cudf::string_view>(idx * 2 + 1) !=
                   input.element<cudf::string_view>(other * 2 + 1);
        }));
    auto const collisions = cudf::detail::reduce(is_collision,
                                                 is_collision + elements,
                                                 cudf::size_type{0},
                                                 cuda::std::plus<cudf::size_type>{},
                                                 stream);
    if (collisions == 0) { return {std::move(merge_pairs_map), salt}; }
  }

  CUDF_FAIL("Unable to create unique merge pair fingerprints");
}

/**
 * @brief Builds the set of cross-character keys for locating unpairable boundaries
 *
 * See `cross_key_type` for details.
 */
std::unique_ptr<detail::cross_set_type> initialize_cross_set(cudf::column_device_view const& input,
                                                             cuda::stream_ref stream,
                                                             cudf::memory_resources mr)
{
  auto const elements = input.size() / 2;
  // keys hold UTF-8 bytes which are never 0xFF so the all-ones key is never a valid key
  auto cross_set =
    std::make_unique<cross_set_type>(static_cast<size_t>(elements),
                                     cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                                     cuco::empty_key{~cross_key_type{0}},
                                     cuda::std::equal_to<cross_key_type>{},
                                     cuco::linear_probing<1, cross_hasher>{},
                                     cuco::thread_scope_device,
                                     cuco_storage{},
                                     rmm::mr::polymorphic_allocator<char>{mr.get_output_mr()},
                                     stream.get());

  // key is the last character of the left half and the first character of the right half
  auto iter = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<cross_key_type>([input] __device__(cudf::size_type idx) {
      auto const lhs = input.element<cudf::string_view>(idx * 2);
      auto const rhs = input.element<cudf::string_view>(idx * 2 + 1);
      // a never-matching key (not the empty key) for a pair with an empty half
      if (lhs.empty() || rhs.empty()) { return ~cross_key_type{0} - 1; }
      auto ptr = lhs.data() + lhs.size_bytes() - 1;
      while (ptr > lhs.data() && !cudf::strings::detail::is_begin_utf8_char(*ptr)) {
        --ptr;
      }
      return make_cross_key(ptr, rhs.data());
    }));

  cross_set->insert_async(iter, iter + elements, stream.get());

  return cross_set;
}

std::unique_ptr<bpe_merge_pairs::bpe_merge_pairs_impl> create_bpe_merge_pairs_impl(
  std::unique_ptr<cudf::column>&& input, cuda::stream_ref stream, cudf::memory_resources mr)
{
  auto d_input = cudf::column_device_view::create(input->view(), stream, mr.get_output_mr());
  auto [merge_pairs, salt] = initialize_merge_pairs_map(*d_input, stream, mr);
  auto cross_set           = initialize_cross_set(*d_input, stream, mr);
  return std::make_unique<nvtext::bpe_merge_pairs::bpe_merge_pairs_impl>(
    std::move(input), std::move(d_input), std::move(merge_pairs), std::move(cross_set), salt);
}

std::unique_ptr<bpe_merge_pairs::bpe_merge_pairs_impl> create_bpe_merge_pairs_impl(
  cudf::strings_column_view const& input, cuda::stream_ref stream, cudf::memory_resources mr)
{
  auto const space = std::string(" ");  // workaround to ARM issue
  auto pairs =
    cudf::strings::split_record(input,
                                cudf::string_scalar(space, true, stream, mr.get_temporary_mr()),
                                1,
                                stream,
                                mr.get_output_mr());
  auto content = pairs->release();
  return create_bpe_merge_pairs_impl(std::move(content.children.back()), stream, mr);
}

}  // namespace

std::unique_ptr<bpe_merge_pairs> load_merge_pairs(cudf::strings_column_view const& merge_pairs,
                                                  cuda::stream_ref stream,
                                                  cudf::memory_resources mr)
{
  CUDF_EXPECTS(!merge_pairs.is_empty(), "Merge pairs must not be empty");
  CUDF_EXPECTS(!merge_pairs.has_nulls(), "Merge pairs may not contain nulls");
  return std::make_unique<bpe_merge_pairs>(merge_pairs, stream, mr);
}

}  // namespace detail

std::unique_ptr<bpe_merge_pairs> load_merge_pairs(cudf::strings_column_view const& merge_pairs,
                                                  cuda::stream_ref stream,
                                                  cudf::memory_resources mr)
{
  CUDF_FUNC_RANGE();
  return detail::load_merge_pairs(merge_pairs, stream, mr);
}

bpe_merge_pairs::bpe_merge_pairs_impl::bpe_merge_pairs_impl(
  std::unique_ptr<cudf::column>&& merge_pairs,
  std::unique_ptr<cudf::column_device_view, std::function<void(cudf::column_device_view*)>>&&
    d_merge_pairs,
  std::unique_ptr<detail::merge_pairs_map_type>&& merge_pairs_map,
  std::unique_ptr<detail::cross_set_type>&& cross_set,
  uint64_t salt)
  : merge_pairs(std::move(merge_pairs)),
    d_merge_pairs(std::move(d_merge_pairs)),
    merge_pairs_map(std::move(merge_pairs_map)),
    cross_set(std::move(cross_set)),
    salt(salt)
{
}

bpe_merge_pairs::bpe_merge_pairs(std::unique_ptr<cudf::column>&& input,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr)
  : impl(detail::create_bpe_merge_pairs_impl(std::move(input), stream, mr).release())
{
}

bpe_merge_pairs::bpe_merge_pairs(cudf::strings_column_view const& input,
                                 cuda::stream_ref stream,
                                 cudf::memory_resources mr)
  : impl(detail::create_bpe_merge_pairs_impl(input, stream, mr).release())
{
}

bpe_merge_pairs::bpe_merge_pairs() = default;
bpe_merge_pairs::~bpe_merge_pairs() { delete impl; }

}  // namespace nvtext

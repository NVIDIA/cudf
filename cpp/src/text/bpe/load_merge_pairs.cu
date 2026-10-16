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
 * In the very unlikely event two pairs produce the same fingerprint,
 * the map is rebuilt using a different salt.
 *
 * @throw std::invalid_argument if the merge pairs contain duplicates
 *
 * @return The map and the salt used to create the fingerprints
 */
std::pair<std::unique_ptr<detail::merge_pairs_map_type>, uint64_t> initialize_merge_pairs_map(
  cudf::column_device_view const& input, cuda::stream_ref stream)
{
  auto const elements         = input.size() / 2;
  auto constexpr max_attempts = 4;
  // the table is small and most lookups are for pairs that are not in it;
  // a lower load factor keeps the probe sequences for these misses short
  auto constexpr merge_pairs_load_factor = 0.25;

  std::unique_ptr<detail::merge_pairs_map_type> merge_pairs_map;
  uint64_t salt = 0;
  for (int attempt = 0; attempt < max_attempts; ++attempt) {
    salt = static_cast<uint64_t>(attempt) * 0x9E37'79B9'7F4A'7C15ul;
    merge_pairs_map =
      std::make_unique<merge_pairs_map_type>(static_cast<size_t>(elements),
                                             merge_pairs_load_factor,
                                             cuco::empty_key{empty_fingerprint},
                                             cuco::empty_value{-1},
                                             cuda::std::equal_to<fingerprint_type>{},
                                             cuco::linear_probing<1, fingerprint_hasher>{},
                                             cuco::thread_scope_device,
                                             cuco_storage{},
                                             rmm::mr::polymorphic_allocator<char>{},
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

    auto const inserted = merge_pairs_map->insert(iter, iter + elements, stream.get());
    if (inserted == static_cast<std::size_t>(elements)) {
      return {std::move(merge_pairs_map), salt};
    }

    // fewer entries means the table has duplicate pairs or two pairs produced the
    // same fingerprint; a duplicate finds a different pair with the same strings
    auto const map_ref      = merge_pairs_map->ref(cuco::op::find);
    auto const is_duplicate = cuda::transform_iterator(
      cuda::counting_iterator<cudf::size_type>{0},
      cuda::proclaim_return_type<cudf::size_type>(
        [input, map_ref, iter] __device__(cudf::size_type idx) -> cudf::size_type {
          auto const found = map_ref.find(iter[idx].first);
          if (found == map_ref.end() || found->second == idx) { return 0; }
          auto const other = found->second;
          return input.element<cudf::string_view>(idx * 2) ==
                   input.element<cudf::string_view>(other * 2) &&
                 input.element<cudf::string_view>(idx * 2 + 1) ==
                   input.element<cudf::string_view>(other * 2 + 1);
        }));
    auto const duplicates = cudf::detail::reduce(is_duplicate,
                                                 is_duplicate + elements,
                                                 cudf::size_type{0},
                                                 cuda::std::plus<cudf::size_type>{},
                                                 stream);
    CUDF_EXPECTS(duplicates == 0, "Merge pairs must be unique", std::invalid_argument);
  }

  CUDF_FAIL("Unable to create unique merge pair fingerprints");
}

/**
 * @brief Builds the set of cross-character keys for locating unpairable boundaries
 *
 * See `cross_key_type` for details.
 */
std::unique_ptr<detail::cross_set_type> initialize_cross_set(cudf::column_device_view const& input,
                                                             cuda::stream_ref stream)
{
  auto const elements = input.size() / 2;
  auto cross_set      = std::make_unique<cross_set_type>(static_cast<size_t>(elements),
                                                    cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                                                    cuco::empty_key{~cross_key_type{0}},
                                                    cuda::std::equal_to<cross_key_type>{},
                                                    cuco::linear_probing<1, cross_hasher>{},
                                                    cuco::thread_scope_device,
                                                    cuco_storage{},
                                                    rmm::mr::polymorphic_allocator<char>{},
                                                    stream.get());

  // key is the last character of the left half and the first character of the right half
  auto iter = cudf::detail::make_counting_transform_iterator(
    0, cuda::proclaim_return_type<cross_key_type>([input] __device__(cudf::size_type idx) {
      auto const lhs = input.element<cudf::string_view>(idx * 2);
      auto const rhs = input.element<cudf::string_view>(idx * 2 + 1);
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
  std::unique_ptr<cudf::column>&& input, cuda::stream_ref stream)
{
  auto d_input             = cudf::column_device_view::create(input->view(), stream);
  auto [merge_pairs, salt] = initialize_merge_pairs_map(*d_input, stream);
  auto cross_set           = initialize_cross_set(*d_input, stream);
  return std::make_unique<nvtext::bpe_merge_pairs::bpe_merge_pairs_impl>(
    std::move(input), std::move(d_input), std::move(merge_pairs), std::move(cross_set), salt);
}

std::unique_ptr<bpe_merge_pairs::bpe_merge_pairs_impl> create_bpe_merge_pairs_impl(
  cudf::strings_column_view const& input,
  cuda::stream_ref stream,
  rmm::device_async_resource_ref mr)
{
  auto const space = std::string(" ");  // workaround to ARM issue
  auto pairs =
    cudf::strings::split_record(input, cudf::string_scalar(space, true, stream, mr), 1, stream, mr);
  auto content = pairs->release();
  return create_bpe_merge_pairs_impl(std::move(content.children.back()), stream);
}

}  // namespace

std::unique_ptr<bpe_merge_pairs> load_merge_pairs(cudf::strings_column_view const& merge_pairs,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(!merge_pairs.is_empty(), "Merge pairs must not be empty");
  CUDF_EXPECTS(!merge_pairs.has_nulls(), "Merge pairs may not contain nulls");
  return std::make_unique<bpe_merge_pairs>(merge_pairs, stream, mr);
}

}  // namespace detail

std::unique_ptr<bpe_merge_pairs> load_merge_pairs(cudf::strings_column_view const& merge_pairs,
                                                  cuda::stream_ref stream,
                                                  rmm::device_async_resource_ref mr)
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
                                 rmm::device_async_resource_ref)
  : impl(detail::create_bpe_merge_pairs_impl(std::move(input), stream).release())
{
}

bpe_merge_pairs::bpe_merge_pairs(cudf::strings_column_view const& input,
                                 cuda::stream_ref stream,
                                 rmm::device_async_resource_ref mr)
  : impl(detail::create_bpe_merge_pairs_impl(input, stream, mr).release())
{
}

bpe_merge_pairs::bpe_merge_pairs() = default;
bpe_merge_pairs::~bpe_merge_pairs() { delete impl; }

}  // namespace nvtext

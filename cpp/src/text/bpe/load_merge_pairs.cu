/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "text/bpe/byte_pair_encoding.cuh"

#include <cudf/column/column_factories.hpp>
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
#include <cuda/stream>

#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

namespace nvtext {
namespace detail {
namespace {

std::unique_ptr<detail::merge_pairs_map_type> initialize_merge_pairs_map(
  cudf::column_device_view const& input, cuda::stream_ref stream)
{
  auto const elements = input.size() / 2;
  auto merge_pairs_map =
    std::make_unique<merge_pairs_map_type>(static_cast<size_t>(elements),
                                           cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                                           cuco::empty_key{-1},
                                           cuco::empty_value{-1},
                                           bpe_equal{input},
                                           bpe_probe_scheme{bpe_hasher{input}},
                                           cuco::thread_scope_device,
                                           cuco_storage{},
                                           rmm::mr::polymorphic_allocator<char>{},
                                           stream.get());

  auto iter = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cuco::pair<cudf::size_type, cudf::size_type>>(
      [] __device__(cudf::size_type idx) { return cuco::make_pair(idx, idx); }));

  merge_pairs_map->insert_async(iter, iter + elements, stream.get());

  return merge_pairs_map;
}

std::unique_ptr<detail::mp_table_map_type> initialize_mp_table_map(
  cudf::column_device_view const& input, cuda::stream_ref stream)
{
  auto mp_table_map = std::make_unique<mp_table_map_type>(static_cast<size_t>(input.size()),
                                                          cudf::detail::CUCO_DESIRED_LOAD_FACTOR,
                                                          cuco::empty_key{-1},
                                                          cuco::empty_value{-1},
                                                          mp_equal{input},
                                                          mp_probe_scheme{mp_hasher{input}},
                                                          cuco::thread_scope_device,
                                                          cuco_storage{},
                                                          rmm::mr::polymorphic_allocator<char>{},
                                                          stream.get());

  auto iter = cudf::detail::make_counting_transform_iterator(
    0,
    cuda::proclaim_return_type<cuco::pair<cudf::size_type, cudf::size_type>>(
      [] __device__(cudf::size_type idx) { return cuco::make_pair(idx, idx); }));

  mp_table_map->insert_async(iter, iter + input.size(), stream.get());

  return mp_table_map;
}

std::unique_ptr<bpe_merge_pairs::bpe_merge_pairs_impl> create_bpe_merge_pairs_impl(
  std::unique_ptr<cudf::column>&& input, cuda::stream_ref stream)
{
  auto d_input      = cudf::column_device_view::create(input->view(), stream);
  auto merge_pairs  = initialize_merge_pairs_map(*d_input, stream);
  auto mp_table_map = initialize_mp_table_map(*d_input, stream);
  return std::make_unique<nvtext::bpe_merge_pairs::bpe_merge_pairs_impl>(
    std::move(input), std::move(d_input), std::move(merge_pairs), std::move(mp_table_map));
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
  std::unique_ptr<detail::mp_table_map_type>&& mp_table_map)
  : merge_pairs(std::move(merge_pairs)),
    d_merge_pairs(std::move(d_merge_pairs)),
    merge_pairs_map(std::move(merge_pairs_map)),
    mp_table_map(std::move(mp_table_map))
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

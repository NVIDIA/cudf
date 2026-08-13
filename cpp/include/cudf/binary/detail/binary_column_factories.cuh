/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/binary/binary_column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/strings_children.cuh>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/utility>
#include <thrust/iterator/transform_iterator.h>

namespace cudf::binary::detail {

using binary_index_pair = cuda::std::pair<uint8_t const*, size_type>;

/**
 * @brief Creates a BINARY column from device-accessible pointer/size pairs.
 *
 * A null pointer denotes a null row. The bytes referenced by non-null pointers
 * are copied into the output payload.
 */
template <typename IndexPairIterator>
std::unique_ptr<column> make_binary_column(IndexPairIterator begin,
                                           IndexPairIterator end,
                                           rmm::cuda_stream_view stream,
                                           rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  auto const row_count = static_cast<size_type>(cuda::std::distance(begin, end));
  if (row_count == 0) { return make_empty_binary_column(); }

  auto sizes = thrust::make_transform_iterator(
    begin,
    cuda::proclaim_return_type<size_type>([] __device__(binary_index_pair item) {
      return item.first == nullptr ? size_type{0} : item.second;
    }));
  auto [offsets, bytes] = cudf::strings::detail::make_offsets_child_column(
    sizes, sizes + row_count, stream, mr);

  auto const validator = [] __device__(binary_index_pair item) { return item.first != nullptr; };
  auto [null_mask, null_count] = cudf::detail::valid_if(begin, end, validator, stream, mr);
  if (null_count == 0) { null_mask = rmm::device_buffer{0, stream, mr}; }

  auto char_pairs = thrust::make_transform_iterator(
    begin,
    cuda::proclaim_return_type<cuda::std::pair<char const*, size_type>>(
      [] __device__(binary_index_pair item) {
        return cuda::std::pair<char const*, size_type>{
          reinterpret_cast<char const*>(item.first), item.second};
      }));
  auto payload = cudf::strings::detail::make_chars_buffer(
    offsets->view(), bytes, char_pairs, row_count, stream, mr);

  return cudf::make_binary_column(row_count,
                                  std::move(offsets),
                                  payload.release(),
                                  null_count,
                                  std::move(null_mask));
}

}  // namespace cudf::binary::detail

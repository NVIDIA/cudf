/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/valid_if.cuh>
#include <cudf/strings/detail/strings_children.cuh>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/exec_policy.hpp>

#include <cuda/functional>
#include <cuda/std/iterator>
#include <cuda/std/utility>
#include <cuda/stream>
#include <thrust/iterator/transform_iterator.h>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @brief Basic type expected for iterators passed to `make_strings_column` that represent string
 * data in device memory.
 */
using string_index_pair = cuda::std::pair<char const*, size_type>;

/**
 * @brief Create a strings-type column from iterators of pointer/size pairs
 *
 * @tparam IndexPairIterator iterator over type `pair<char const*,size_type>` values
 *
 * @param begin First string row (inclusive)
 * @param end Last string row (exclusive)
 * @param stream CUDA stream used for device memory operations
 * @param mr  Device memory resource used to allocate the returned column's device memory
 * @return New strings column
 */
template <typename IndexPairIterator>
std::unique_ptr<column> make_strings_column(IndexPairIterator begin,
                                            IndexPairIterator end,
                                            cuda::stream_ref stream,
                                            rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  size_type strings_count = cuda::std::distance(begin, end);
  if (strings_count == 0) return make_empty_column(type_id::STRING);

  // build offsets column from the strings sizes
  auto offsets_transformer =
    cuda::proclaim_return_type<size_type>([] __device__(string_index_pair item) -> size_type {
      return (item.first != nullptr ? static_cast<size_type>(item.second) : size_type{0});
    });
  auto offsets_transformer_itr = thrust::make_transform_iterator(begin, offsets_transformer);
  auto [offsets_column, bytes] = cudf::strings::detail::make_offsets_child_column(
    offsets_transformer_itr, offsets_transformer_itr + strings_count, stream, mr);

  // create null mask
  auto validator = [] __device__(string_index_pair const item) { return item.first != nullptr; };
  auto new_nulls = cudf::detail::valid_if(begin, end, validator, stream, mr);
  auto const null_count = new_nulls.second;
  auto null_mask =
    (null_count > 0) ? std::move(new_nulls.first) : rmm::device_buffer{0, stream, mr};

  // build chars column
  auto chars_data =
    make_chars_buffer(offsets_column->view(), bytes, begin, strings_count, stream, mr);

  return make_strings_column(strings_count,
                             std::move(offsets_column),
                             chars_data.release(),
                             null_count,
                             std::move(null_mask));
}

}  // namespace detail
}  // namespace strings
}  // namespace cudf

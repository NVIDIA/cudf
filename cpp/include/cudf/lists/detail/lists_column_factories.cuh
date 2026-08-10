/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_factories.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/sizes_to_offsets_iterator.cuh>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cuda/functional>

#include <limits>
#include <tuple>
#include <utility>

namespace cudf::lists::detail {

/**
 * @brief Builds a width-aware offsets child for a LIST column.
 *
 * INT32 is used unless INT64 is requested or the resulting child size exceeds
 * the INT32 offset range. INT64 offsets require a 64-bit cudf::size_type
 * because every offset must identify a materializable row in the child column.
 *
 * @tparam InputIterator Iterator over per-row list sizes
 * @param begin Beginning of the list sizes
 * @param end End of the list sizes
 * @param preferred_type Minimum desired offset width (INT32 or INT64)
 * @param stream CUDA stream used for device operations
 * @param mr Device memory resource used for the returned column
 * @return The offsets column and total child row count
 */
template <typename InputIterator>
std::pair<std::unique_ptr<column>, size_type> make_offsets_child_column(
  InputIterator begin,
  InputIterator end,
  data_type preferred_type,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  CUDF_EXPECTS(preferred_type.id() == type_id::INT32 || preferred_type.id() == type_id::INT64,
               "List offsets must be INT32 or INT64");
#if CUDF_SIZE_TYPE_BITS == 32
  CUDF_EXPECTS(preferred_type.id() != type_id::INT64,
               "INT64 list offsets require a 64-bit cudf::size_type");
#endif

  auto const count = static_cast<size_type>(std::distance(begin, end));
  auto map_fn =
    cuda::proclaim_return_type<size_type>([begin, count] __device__(size_type idx) -> size_type {
      return idx < count ? static_cast<size_type>(begin[idx]) : size_type{0};
    });
  auto input = cudf::detail::make_counting_transform_iterator(0, map_fn);

  auto make_offsets = [&](data_type type) {
    auto result =
      make_numeric_column(type, count + 1, mask_state::UNALLOCATED, stream, mr);
    auto const total = type.id() == type_id::INT32
                         ? cudf::detail::sizes_to_offsets(
                             input,
                             input + count + 1,
                             result->mutable_view().template data<int32_t>(),
                             0,
                             stream)
                         : cudf::detail::sizes_to_offsets(
                             input,
                             input + count + 1,
                             result->mutable_view().template data<int64_t>(),
                             0,
                             stream);
    result->set_null_count(0);
    return std::pair{std::move(result), total};
  };

  auto [offsets, total] = make_offsets(preferred_type);
  CUDF_EXPECTS(total <= static_cast<decltype(total)>(std::numeric_limits<size_type>::max()),
               "List child size exceeds the column size limit",
               std::overflow_error);

  if (preferred_type.id() == type_id::INT32 &&
      total > static_cast<decltype(total)>(std::numeric_limits<int32_t>::max())) {
#if CUDF_SIZE_TYPE_BITS == 64
    std::tie(offsets, total) = make_offsets(data_type{type_id::INT64});
#else
    CUDF_FAIL("List child size exceeds the column size limit", std::overflow_error);
#endif
  }

  return {std::move(offsets), static_cast<size_type>(total)};
}

template <typename InputIterator>
std::pair<std::unique_ptr<column>, size_type> make_offsets_child_column(
  InputIterator begin,
  InputIterator end,
  rmm::cuda_stream_view stream,
  rmm::device_async_resource_ref mr)
{
  return make_offsets_child_column(
    begin, end, data_type{type_id::INT32}, stream, std::move(mr));
}

}  // namespace cudf::lists::detail

/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/detail/repeat.hpp>
#include <cudf/detail/reshape.hpp>
#include <cudf/filling.hpp>
#include <cudf/join/join.hpp>
#include <cudf/reshape.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace cudf {
namespace detail {
/**
 * @copydoc cudf::cross_join
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::table> cross_join(cudf::table_view const& left,
                                        cudf::table_view const& right,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  // If left or right table has no rows, return an empty table with all columns
  if ((0 == left.num_rows()) || (0 == right.num_rows())) {
    auto empty_left_columns  = empty_like(left)->release();
    auto empty_right_columns = empty_like(right)->release();
    std::move(empty_right_columns.begin(),
              empty_right_columns.end(),
              std::back_inserter(empty_left_columns));
    return std::make_unique<table>(std::move(empty_left_columns));
  }

  auto const out_num_rows =
    static_cast<int64_t>(left.num_rows()) * static_cast<int64_t>(right.num_rows());
  CUDF_EXPECTS(out_num_rows <= std::numeric_limits<size_type>::max(),
               "Cross join result exceeds the column size limit",
               std::overflow_error);

  // Repeat left table
  auto left_repeated = detail::repeat(left, right.num_rows(), stream, mr);

  // Tile right table
  auto right_tiled = detail::tile(right, left.num_rows(), stream, mr);

  // Concatenate all repeated/tiled columns into one table
  auto left_repeated_columns = left_repeated->release();
  auto right_tiled_columns   = right_tiled->release();
  std::move(right_tiled_columns.begin(),
            right_tiled_columns.end(),
            std::back_inserter(left_repeated_columns));

  return std::make_unique<table>(std::move(left_repeated_columns),
                                 static_cast<size_type>(out_num_rows));
}
}  // namespace detail

std::unique_ptr<cudf::table> cross_join(cudf::table_view const& left,
                                        cudf::table_view const& right,
                                        rmm::cuda_stream_view stream,
                                        rmm::device_async_resource_ref mr)
{
  CUDF_FUNC_RANGE();
  return detail::cross_join(left, right, stream, mr);
}

}  // namespace cudf

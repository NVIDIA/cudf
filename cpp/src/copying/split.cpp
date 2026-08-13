/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/detail/copy.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <stdexcept>

namespace cudf {
namespace detail {
namespace {
template <typename T>
std::vector<T> split(T const& input,
                     size_type column_size,
                     std::span<size_type const> splits,
                     rmm::cuda_stream_view stream)
{
  if (splits.empty() or column_size == 0) { return std::vector<T>{input}; }
  CUDF_EXPECTS(
    splits.back() <= column_size, "splits can't exceed size of input columns", std::out_of_range);

  // If the size is not zero, the split will always start at `0`
  std::vector<size_type> indices{0};
  std::for_each(splits.begin(), splits.end(), [&indices](auto split) {
    indices.push_back(split);  // This for end
    indices.push_back(split);  // This for the start
  });

  indices.push_back(column_size);  // This to include rest of the elements

  return detail::slice(input, indices, stream);
}

};  // anonymous namespace

std::vector<cudf::column_view> split(cudf::column_view const& input,
                                     std::span<size_type const> splits,
                                     rmm::cuda_stream_view stream)
{
  return split(input, input.size(), splits, stream);
}

std::vector<cudf::table_view> split(cudf::table_view const& input,
                                    std::span<size_type const> splits,
                                    rmm::cuda_stream_view stream)
{
  // A genuinely empty table (no columns and no rows) has nothing to split.
  if (input.num_columns() == 0 && input.num_rows() == 0) { return {}; }
  return split(input, input.num_rows(), splits, stream);
}

std::vector<column_view> split(column_view const& input,
                               std::initializer_list<size_type> splits,
                               rmm::cuda_stream_view stream)
{
  return detail::split(input, std::span<size_type const>(splits.begin(), splits.size()), stream);
}

std::vector<table_view> split(table_view const& input,
                              std::initializer_list<size_type> splits,
                              rmm::cuda_stream_view stream)
{
  return detail::split(input, std::span<size_type const>(splits.begin(), splits.size()), stream);
}

}  // namespace detail

std::vector<cudf::column_view> split(cudf::column_view const& input,
                                     std::span<size_type const> splits,
                                     rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::split(input, splits, stream);
}

std::vector<cudf::table_view> split(cudf::table_view const& input,
                                    std::span<size_type const> splits,
                                    rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::split(input, splits, stream);
}

std::vector<column_view> split(column_view const& input,
                               std::initializer_list<size_type> splits,
                               rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::split(input, splits, stream);
}

std::vector<table_view> split(table_view const& input,
                              std::initializer_list<size_type> splits,
                              rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::split(input, splits, stream);
}

}  // namespace cudf

/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_view.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/copy.hpp>
#include <cudf/detail/iterator.cuh>
#include <cudf/detail/null_mask.cuh>
#include <cudf/detail/null_mask.hpp>
#include <cudf/detail/nvtx/ranges.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <thrust/iterator/transform_iterator.h>

#include <algorithm>
#include <stdexcept>

namespace cudf {
namespace detail {

template <typename ColumnView>
ColumnView slice(ColumnView const& input,
                 size_type begin,
                 size_type end,
                 rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(begin >= 0, "Invalid beginning of range.", std::out_of_range);
  CUDF_EXPECTS(end >= begin, "Invalid end of range.", std::invalid_argument);
  CUDF_EXPECTS(end <= input.size(), "Slice range out of bounds.", std::out_of_range);

  std::vector<ColumnView> children{};
  children.reserve(input.num_children());
  for (size_type index = 0; index < input.num_children(); index++) {
    children.emplace_back(input.child(index));
  }

  return ColumnView(
    input.type(),
    end - begin,
    input.head(),
    input.null_mask(),
    input.null_count() ? cudf::detail::null_count(input.null_mask(), begin, end, stream) : 0,
    input.offset() + begin,
    children);
}

template column_view slice<column_view>(column_view const&,
                                        size_type,
                                        size_type,
                                        rmm::cuda_stream_view);
template mutable_column_view slice<mutable_column_view>(mutable_column_view const&,
                                                        size_type,
                                                        size_type,
                                                        rmm::cuda_stream_view);

std::vector<column_view> slice(column_view const& input,
                               std::span<size_type const> indices,
                               rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(indices.size() % 2 == 0, "indices size must be even", std::invalid_argument);

  if (indices.empty()) return {};

  // need to shift incoming indices by the column offset to generate the correct bit ranges
  // to count
  auto indices_iter = cudf::detail::make_counting_transform_iterator(
    0, [offset = input.offset(), &indices](size_type index) { return indices[index] + offset; });
  auto null_counts = cudf::detail::segmented_null_count(
    input.null_mask(), indices_iter, indices_iter + indices.size(), stream);

  auto const children = std::vector<column_view>(input.child_begin(), input.child_end());

  auto op = [&](auto i) {
    auto begin = indices[2 * i];
    auto end   = indices[2 * i + 1];
    CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.", std::out_of_range);
    CUDF_EXPECTS(
      end >= begin, "End index cannot be smaller than the starting index.", std::invalid_argument);
    CUDF_EXPECTS(end <= input.size(), "Slice range out of bounds.", std::out_of_range);
    return column_view{input.type(),
                       end - begin,
                       input.head(),
                       input.null_mask(),
                       null_counts[i],
                       input.offset() + begin,
                       children};
  };
  auto begin = cudf::detail::make_counting_transform_iterator(0, op);
  return std::vector<column_view>{begin, begin + indices.size() / 2};
}

std::vector<table_view> slice(table_view const& input,
                              std::span<size_type const> indices,
                              rmm::cuda_stream_view stream)
{
  CUDF_EXPECTS(indices.size() % 2 == 0, "indices size must be even", std::invalid_argument);
  if (indices.empty()) { return {}; }

  // 2d arrangement of column_views that represent the outgoing table_views sliced_table[i][j]
  // where i is the i'th column of the j'th table_view
  auto op = [&indices, &stream](auto const& c) { return cudf::detail::slice(c, indices, stream); };
  auto f  = thrust::make_transform_iterator(input.begin(), op);

  auto sliced_table = std::vector<std::vector<cudf::column_view>>(f, f + input.num_columns());
  sliced_table.reserve(indices.size() + 1);

  std::vector<cudf::table_view> result{};
  // distribute columns into outgoing table_views
  size_t num_output_tables = indices.size() / 2;
  for (size_t i = 0; i < num_output_tables; i++) {
    std::vector<cudf::column_view> table_columns;
    for (size_type j = 0; j < input.num_columns(); j++) {
      table_columns.emplace_back(sliced_table[j][i]);
    }
    auto const begin = indices[2 * i];
    auto const end   = indices[2 * i + 1];
    // For a zero-column input the per-column slice above never runs, so its bounds checks are
    // skipped. Validate the range here before building the zero-column table_view.
    if (input.num_columns() == 0) {
      if (input.num_rows() == 0) {
        // An empty table (no rows) historically returns empty slices regardless of the indices.
        result.emplace_back(table_view{table_columns});
        continue;
      }
      CUDF_EXPECTS(begin >= 0, "Starting index cannot be negative.", std::out_of_range);
      CUDF_EXPECTS(end >= begin,
                   "End index cannot be smaller than the starting index.",
                   std::invalid_argument);
      CUDF_EXPECTS(end <= input.num_rows(), "Slice range out of bounds.", std::out_of_range);
    }
    result.emplace_back(table_view{table_columns, end - begin});
  }

  return result;
}

std::vector<column_view> slice(column_view const& input,
                               std::initializer_list<size_type> indices,
                               rmm::cuda_stream_view stream)
{
  return detail::slice(input, std::span<size_type const>(indices.begin(), indices.size()), stream);
}

std::vector<table_view> slice(table_view const& input,
                              std::initializer_list<size_type> indices,
                              rmm::cuda_stream_view stream)
{
  return detail::slice(input, std::span<size_type const>(indices.begin(), indices.size()), stream);
};

}  // namespace detail

std::vector<column_view> slice(column_view const& input,
                               std::span<size_type const> indices,
                               rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::slice(input, indices, stream);
}

std::vector<table_view> slice(table_view const& input,
                              std::span<size_type const> indices,
                              rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::slice(input, indices, stream);
};

std::vector<column_view> slice(column_view const& input,
                               std::initializer_list<size_type> indices,
                               rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::slice(input, indices, stream);
}

std::vector<table_view> slice(table_view const& input,
                              std::initializer_list<size_type> indices,
                              rmm::cuda_stream_view stream)
{
  CUDF_FUNC_RANGE();
  return detail::slice(input, indices, stream);
};

}  // namespace cudf

/*
 * SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/type_checks.hpp>

#include <algorithm>
#include <vector>

namespace cudf {
namespace detail {
namespace {

template <typename ViewType>
auto concatenate_column_views(std::vector<ViewType> const& views)
{
  using ColumnView = typename ViewType::ColumnView;
  std::vector<ColumnView> concat_cols;
  for (auto& view : views) {
    concat_cols.insert(concat_cols.end(), view.begin(), view.end());
  }
  return concat_cols;
}

template <typename ViewType>
size_type concatenated_table_num_rows(std::vector<ViewType> const& views)
{
  if (views.empty()) { return 0; }
  auto const num_rows = views.front().num_rows();
  CUDF_EXPECTS(std::all_of(views.begin(),
                           views.end(),
                           [num_rows](auto const& view) { return view.num_rows() == num_rows; }),
               "Mismatch in number of rows");
  return num_rows;
}

}  // namespace

template <typename ColumnView>
table_view_base<ColumnView>::table_view_base(std::vector<ColumnView> const& cols) : _columns{cols}
{
  if (num_columns() > 0) {
    std::for_each(_columns.begin(), _columns.end(), [this](ColumnView col) {
      CUDF_EXPECTS(col.size() == _columns.front().size(), "Column size mismatch.");
    });
    _num_rows = _columns.front().size();
  } else {
    _num_rows = 0;
  }
}

template <typename ColumnView>
table_view_base<ColumnView>::table_view_base(std::vector<ColumnView> const& cols,
                                             size_type num_rows)
  : _columns{cols}, _num_rows{num_rows}
{
  CUDF_EXPECTS(num_rows >= 0, "Number of rows cannot be negative.", std::invalid_argument);
  CUDF_EXPECTS(std::all_of(cols.begin(),
                           cols.end(),
                           [num_rows](ColumnView const& col) { return col.size() == num_rows; }),
               "Column size mismatch",
               std::invalid_argument);
}

// Explicit instantiation for a table of `column_view`s
template class table_view_base<column_view>;

// Explicit instantiation for a table of `mutable_column_view`s
template class table_view_base<mutable_column_view>;
}  // namespace detail

// Returns a table_view with set of specified columns
table_view table_view::select(std::vector<size_type> const& column_indices) const
{
  return select(column_indices.begin(), column_indices.end());
}

// Convert mutable view to immutable view
mutable_table_view::operator table_view()
{
  return table_view{std::vector<column_view>{begin(), end()}, num_rows()};
}

table_view::table_view(std::vector<table_view> const& views)
  : table_view{detail::concatenate_column_views(views), detail::concatenated_table_num_rows(views)}
{
}

mutable_table_view::mutable_table_view(std::vector<mutable_table_view> const& views)
  : mutable_table_view{detail::concatenate_column_views(views),
                       detail::concatenated_table_num_rows(views)}
{
}

table_view scatter_columns(table_view const& source,
                           std::vector<size_type> const& map,
                           table_view const& target)
{
  std::vector<cudf::column_view> updated_columns(target.begin(), target.end());
  // scatter(updated_table.begin(),updated_table.end(),indices.begin(),updated_columns.begin());
  for (size_type idx = 0; idx < source.num_columns(); ++idx)
    updated_columns[map[idx]] = source.column(idx);
  return table_view{updated_columns, target.num_rows()};
}

std::vector<column_view> get_nullable_columns(table_view const& table)
{
  std::vector<column_view> result;
  for (auto const& col : table) {
    if (col.nullable()) { result.push_back(col); }
    for (auto it = col.child_begin(); it != col.child_end(); ++it) {
      auto const& child = *it;
      if (child.size() == col.size()) {
        auto const child_result = get_nullable_columns(table_view{{child}});
        result.insert(result.end(), child_result.begin(), child_result.end());
      }
    }
  }
  return result;
}

bool nullable(table_view const& view)
{
  return std::any_of(view.begin(), view.end(), [](auto const& col) { return col.nullable(); });
}

bool has_nulls(table_view const& view)
{
  return std::any_of(view.begin(), view.end(), [](auto const& col) { return col.has_nulls(); });
}

bool has_nested_nulls(table_view const& input)
{
  return std::any_of(input.begin(), input.end(), [](auto const& col) {
    return col.has_nulls() ||
           std::any_of(col.child_begin(), col.child_end(), [](auto const& child_col) {
             return has_nested_nulls(table_view{{child_col}});
           });
  });
}

bool has_nested_nullable_columns(table_view const& input)
{
  return std::any_of(input.begin(), input.end(), [](auto const& col) {
    return col.nullable() ||
           std::any_of(col.child_begin(), col.child_end(), [](auto const& child_col) {
             return has_nested_nullable_columns(table_view{{child_col}});
           });
  });
}

namespace detail {

template <typename TableView>
bool is_relationally_comparable(TableView const& lhs, TableView const& rhs)
{
  return std::equal(lhs.begin(),
                    lhs.end(),
                    rhs.begin(),
                    rhs.end(),
                    [](column_view const& lcol, column_view const& rcol) {
                      return cudf::is_relationally_comparable(lcol.type()) and
                             cudf::have_same_types(lcol, rcol);
                    });
}

// Explicit template instantiation for a table of immutable views
template bool is_relationally_comparable<table_view>(table_view const& lhs, table_view const& rhs);

// Explicit template instantiation for a table of mutable views
template bool is_relationally_comparable<mutable_table_view>(mutable_table_view const& lhs,
                                                             mutable_table_view const& rhs);

bool has_nested_columns(table_view const& table)
{
  return std::any_of(
    table.begin(), table.end(), [](column_view const& col) { return is_nested(col.type()); });
}
}  // namespace detail

bool has_nested_columns(table_view const& table) { return detail::has_nested_columns(table); }
}  // namespace cudf

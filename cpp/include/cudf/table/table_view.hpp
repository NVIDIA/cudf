/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/export.hpp>

#include <algorithm>
#include <vector>

/**
 * @file
 * @brief Class definitions for `(mutable)_table_view`
 *
 * A `(mutable_)table_view` is a set of `(mutable_)column_view`s of equal
 * size.
 *
 * A `(mutable_)table_view` is non-owning and trivially copyable and should be
 * passed by value.
 */

namespace CUDF_EXPORT cudf {
namespace detail {
/**
 * @brief Base class for a table of `ColumnView`s
 *
 * This class should not be used directly. Instead:
 * - `table_view` should be used for a table of columns of type `column_view`
 * - `mutable_table_view` should be used for a table of columns of type `mutable_column_view`
 *
 * All public constructors and member functions of `table_view_base` are
 * available in both `table_view` and `mutable_table_view`.
 *
 * @tparam ColumnView The type of column view the table contains:
 *                    expects column_view or mutable_column_view
 */
template <typename ColumnView>
class table_view_base {
  static_assert(std::is_same_v<ColumnView, column_view> or
                  std::is_same_v<ColumnView, mutable_column_view>,
                "table_view_base can only be instantiated with column_view or "
                "column_view_base.");

 private:
  std::vector<ColumnView> _columns{};  ///< ColumnViews to columns of equal size
  size_type _num_rows{};               ///< The number of elements in every column

 public:
  using iterator       = decltype(std::begin(_columns));   ///< Iterator type for the table
  using const_iterator = decltype(std::cbegin(_columns));  ///< const iterator type for the table

  /**
   * @brief Construct a table from a vector of column views
   *
   * @note Because a `std::vector` is constructible from a
   * `std::initializer_list`, this constructor also supports the following
   * usage:
   * ```
   * column_view c0, c1, c2;
   * ...
   * table_view t{{c0,c1,c2}}; // Creates a `table` from c0, c1, c2
   * ```
   *
   * @throws cudf::logic_error If all views do not have the same size
   *
   * @param cols The vector of columns to construct the table from
   */
  explicit table_view_base(std::vector<ColumnView> const& cols);

  /**
   * @brief Construct a table from a vector of column views with an explicit row count.
   *
   * This is primarily intended for zero-column tables, which cannot otherwise carry
   * a non-zero row count (the row count is normally derived from the columns). When
   * `cols` is non-empty, `num_rows` must equal the size of every column.
   *
   * @throws std::invalid_argument If `cols` is non-empty and any view's size does not equal
   * `num_rows`
   *
   * @param cols The vector of columns to construct the table from
   * @param num_rows The number of rows in the table
   */
  table_view_base(std::vector<ColumnView> const& cols, size_type num_rows);

  /**
   * @brief Returns an iterator to the first view in the `table`.
   *
   * @return An iterator to the first column_view
   */
  iterator begin() noexcept { return std::begin(_columns); }

  /**
   * @brief Returns an iterator to the first view in the `table`.
   *
   * @return An iterator to the first view in the `table`
   */
  [[nodiscard]] const_iterator begin() const noexcept { return std::begin(_columns); }

  /**
   * @brief Returns an iterator one past the last column view in the `table`.
   *
   * `end()` acts as a place holder. Attempting to dereference it results in
   * undefined behavior.
   *
   * @return An iterator to one past the last column view in the `table`
   */
  iterator end() noexcept { return std::end(_columns); }

  /**
   * @brief Returns an iterator one past the last column view in the `table`.
   *
   * `end()` acts as a place holder. Attempting to dereference it results in
   * undefined behavior.
   *
   * @return An iterator to one past the last column view in the `table`
   */
  [[nodiscard]] const_iterator end() const noexcept { return std::end(_columns); }

  /**
   * @brief Returns a reference to the view of the specified column
   *
   * @throws std::out_of_range
   * If `column_index` is out of the range [0, num_columns)
   *
   * @param column_index The index of the desired column
   * @return A reference to the desired column
   */
  [[nodiscard]] ColumnView const& column(size_type column_index) const
  {
    return _columns.at(column_index);
  }

  /**
   * @brief Returns the number of columns
   *
   * @return The number of columns
   */
  [[nodiscard]] size_type num_columns() const noexcept { return _columns.size(); }

  /**
   * @brief Returns the number of rows
   *
   * @return The number of rows
   */
  [[nodiscard]] size_type num_rows() const noexcept { return _num_rows; }

  /**
   * @brief Returns true if `num_columns()` returns zero, or false otherwise
   *
   * @return True if `num_columns()` returns zero, or false otherwise
   */
  [[nodiscard]] size_type is_empty() const noexcept { return num_columns() == 0; }

  table_view_base() = default;

  ~table_view_base() = default;

  table_view_base(table_view_base const&) = default;  ///< Copy constructor

  table_view_base(table_view_base&&) = default;  ///< Move constructor
  /**
   * @brief Copy assignment operator
   *
   * @return Reference to this object
   */
  table_view_base& operator=(table_view_base const&) = default;
  /**
   * @brief Move assignment operator
   *
   * @return Reference to this object (after transferring ownership)
   */
  table_view_base& operator=(table_view_base&&) = default;
};

/**
 * @brief Determine if any nested columns exist in a given table.
 *
 * @param table The input table
 * @return Whether nested columns exist in the input table
 */
bool has_nested_columns(table_view const& table);

}  // namespace detail

/**
 * @brief Determine if any nested columns exist in a given table.
 *
 * @param table The input table
 * @return Whether nested columns exist in the input table
 */
bool has_nested_columns(table_view const& table);

/**
 * @brief A set of cudf::column_view's of the same size.
 *
 * If the set of columns is empty, the view's row count may still be non-zero.
 *
 * @ingroup table_classes
 *
 * All public member functions and constructors are inherited from
 * `table_view_base<column_view>`.
 */
class table_view : public detail::table_view_base<column_view> {
  using detail::table_view_base<column_view>::table_view_base;

 public:
  using ColumnView = column_view;  ///< The type of column view the table contains

  table_view() = default;

  /**
   * @brief Construct a table from a vector of table views
   *
   * @note Because a `std::vector` is constructible from a
   * `std::initializer_list`, this constructor also supports the following
   * usage:
   * ```
   * table_view t0, t1, t2;
   * ...
   * table_view t{{t0,t1,t2}}; // Creates a `table` from the columns of
   * t0, t1, t2
   * ```
   *
   * @throws cudf::logic_error
   * If number of rows mismatch
   *
   * @param views The vector of table views to construct the table from
   */
  table_view(std::vector<table_view> const& views);

  /**
   * @brief Returns a table_view built from a range of column indices.
   *
   * @throws std::out_of_range
   * If any index is outside [0, num_columns())
   *
   * @param begin Beginning of the range
   * @param end Ending of the range
   * @return A table_view consisting of columns from the original table
   * specified by the elements of `column_indices`
   */
  template <typename InputIterator>
  [[nodiscard]] table_view select(InputIterator begin, InputIterator end) const
  {
    std::vector<column_view> columns(std::distance(begin, end));
    std::transform(begin, end, columns.begin(), [this](auto index) { return this->column(index); });
    return table_view{columns, num_rows()};
  }

  /**
   * @brief Returns a table_view with set of specified columns.
   *
   * @throws std::out_of_range
   * If any element in `column_indices` is outside [0, num_columns())
   *
   * @param column_indices Indices of columns in the table
   * @return A table_view consisting of columns from the original table
   * specified by the elements of `column_indices`
   */
  [[nodiscard]] table_view select(std::vector<size_type> const& column_indices) const;
};

/**
 * @brief A set of `mutable_column_view`s of the same size.
 *
 * @ingroup table_classes
 *
 * All public member functions and constructors are inherited from
 * `table_view_base<mutable_column_view>`.
 */
class mutable_table_view : public detail::table_view_base<mutable_column_view> {
  using detail::table_view_base<mutable_column_view>::table_view_base;

 public:
  using ColumnView = mutable_column_view;  ///< The type of column views in the table

  mutable_table_view() = default;

  /**
   * @brief Returns column at specified index
   *
   * @param column_index The index of the desired column
   * @return A mutable column view reference to the desired column
   */
  [[nodiscard]] mutable_column_view& column(size_type column_index) const
  {
    return const_cast<mutable_column_view&>(table_view_base::column(column_index));
  }
  /**
   * @brief Creates an immutable `table_view` of the columns
   */
  operator table_view();

  /**
   * @brief Construct a table from a vector of table views
   *
   * @note Because a `std::vector` is constructible from a
   * `std::initializer_list`, this constructor also supports the following
   * usage:
   * ```
   * table_view t0, t1, t2;
   * ...
   * table_view t{{t0,t1,t2}}; // Creates a `table` from the columns of
   * t0, t1, t2
   * ```
   *
   * @throws cudf::logic_error
   * If number of rows mismatch
   *
   * @param views The vector of table views to construct the table from
   */
  mutable_table_view(std::vector<mutable_table_view> const& views);
};

/**
 * @brief Returns True if any of the columns in the table is nullable. (not entire hierarchy)
 *
 * @param view The table to check for nullability
 * @return True if any of the columns in the table is nullable, false otherwise
 */
bool nullable(table_view const& view);

/**
 * @brief Returns True if the table has nulls in any of its columns.
 *
 * This checks for nulls in the columns and but not in any of the columns' children.
 *
 * @param view The table to check for nulls
 * @return True if the table has nulls in any of its columns, false otherwise
 */
bool has_nulls(table_view const& view);

/**
 * @brief Returns True if the table has nulls in any of its columns hierarchy
 *
 * @param input The table to check for nulls
 * @return True if the table has nulls in any of its columns hierarchy, false otherwise
 */
bool has_nested_nulls(table_view const& input);

/**
 * @brief Returns True if the table has a nullable column at any level of the column hierarchy
 *
 * @param input The table to check for nullable columns
 * @return True if the table has nullable columns at any level of the column hierarchy, false
 * otherwise
 */
bool has_nested_nullable_columns(table_view const& input);

/**
 * @brief The function to collect all nullable columns at all nested levels in a given table.
 *
 * @param table The input table
 * @return A vector containing all nullable columns in the input table
 */
std::vector<column_view> get_nullable_columns(table_view const& table);

/**
 * @brief Copy column_views from a table_view into another table_view according to
 * a column indices map.
 *
 * The output table view, `out_table` is a copy of the `target` table_view but with
 * elements updated according to `out_table[map[i]] = source[i]` where `i` is
 * `[0,source.size())`
 *
 * @param source Table of new columns to scatter into the output table view.
 * @param map The indices where each new_column should be copied into the output.
 * @param target Table to receive the updated column views.
 * @return New table_view.
 */
table_view scatter_columns(table_view const& source,
                           std::vector<size_type> const& map,
                           table_view const& target);

namespace detail {
/**
 * @brief Indicates whether respective columns in input tables are relationally comparable.
 *
 * @param lhs The first table
 * @param rhs The second table (may be the same table as `lhs`)
 * @return true all of respective columns on `lhs` and 'rhs` tables are comparable.
 * @return false any of respective columns on `lhs` and 'rhs` tables are not comparable.
 */
template <typename TableView>
bool is_relationally_comparable(TableView const& lhs, TableView const& rhs);
// @cond
extern template bool is_relationally_comparable<table_view>(table_view const& lhs,
                                                            table_view const& rhs);
extern template bool is_relationally_comparable<mutable_table_view>(mutable_table_view const& lhs,
                                                                    mutable_table_view const& rhs);
// @endcond
}  // namespace detail
}  // namespace CUDF_EXPORT cudf

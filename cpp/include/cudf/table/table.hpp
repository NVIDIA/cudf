/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cudf/column/column.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>
#include <vector>

/**
 * @file
 * @brief Class definition for cudf::table
 */

namespace CUDF_EXPORT cudf {

/**
 * @brief A set of cudf::column's of the same size.
 *
 * If the set of columns is empty, the table's row count may still be non-zero.
 *
 * @ingroup table_classes
 */
class table {
 public:
  table()                        = default;
  ~table()                       = default;
  table(table&&)                 = default;  ///< Move constructor
  table& operator=(table const&) = delete;
  table& operator=(table&&)      = delete;

  /**
   * @brief Construct a new table by copying the contents of another table.
   *
   * Uses the specified `stream` and device memory resource for all allocations
   * and copies.
   *
   * @param other The table to copy
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource to use for all device memory allocations
   */
  explicit table(table const& other,
                 rmm::cuda_stream_view stream      = cudf::get_default_stream(),
                 rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());
  /**
   * @brief Moves the contents from a vector of `unique_ptr`s to columns to
   * construct a new table.
   *
   * @param columns The vector of `unique_ptr`s to columns whose contents will
   * be moved into the new table.
   */
  table(std::vector<std::unique_ptr<column>>&& columns);

  /**
   * @brief Moves the contents from a vector of `unique_ptr`s to columns to
   * construct a new table with an explicit row count.
   *
   * This is primarily intended for zero-column tables, which cannot otherwise
   * carry a non-zero row count (the row count is normally derived from the
   * columns). It is used, for example, when converting a zero-column Arrow array
   * that has a non-zero length. When `columns` is non-empty, `num_rows` must equal
   * the size of every column.
   *
   * @throws std::invalid_argument if `columns` is non-empty and `num_rows` does not
   * match the size of every column.
   *
   * @param columns The vector of `unique_ptr`s to columns whose contents will
   * be moved into the new table.
   * @param num_rows The number of rows in the table.
   */
  table(std::vector<std::unique_ptr<column>>&& columns, size_type num_rows);

  /**
   * @brief Copy the contents of a `table_view` to construct a new `table`.
   *
   * @param view The view whose contents will be copied to create a new `table`
   * @param stream CUDA stream used for device memory operations.
   * @param mr Device memory resource used for allocating the device memory for the new columns
   */
  table(table_view view,
        rmm::cuda_stream_view stream      = cudf::get_default_stream(),
        rmm::device_async_resource_ref mr = cudf::get_current_device_resource_ref());

  /**
   * @brief Returns the number of columns in the table
   *
   * @return The number of columns in the table
   */
  [[nodiscard]] size_type num_columns() const noexcept { return _columns.size(); }

  /**
   * @brief Returns the number of rows
   *
   * @return  The number of rows
   */
  [[nodiscard]] size_type num_rows() const noexcept { return _num_rows; }

  /**
   * @brief Returns the total device allocation size of the table's columns in bytes
   *
   * @see cudf::column::alloc_size
   *
   * @return The total allocation size in bytes
   */
  [[nodiscard]] std::size_t alloc_size() const;

  /**
   * @brief Returns an immutable, non-owning `table_view` of the contents of
   *this `table`.
   *
   * @return An immutable, non-owning `table_view` of the contents of this `table`
   */
  [[nodiscard]] table_view view() const;

  /**
   * @brief Conversion operator to an immutable, non-owning `table_view` of the
   * contents of this `table`.
   */
  operator table_view() const { return this->view(); };

  /**
   * @brief Returns a mutable, non-owning `mutable_table_view` of the contents
   * of this `table`.
   *
   * @return A mutable, non-owning `mutable_table_view` of the contents of this `table`
   */
  mutable_table_view mutable_view();

  /**
   * @brief Conversion operator to a mutable, non-owning `mutable_table_view` of
   *the contents of this `table`.
   */
  operator mutable_table_view() { return this->mutable_view(); };

  /**
   * @brief Releases ownership of the `column`s by returning a vector of
   * `unique_ptr`s to the constituent columns.
   *
   * After `release()`, `num_columns() == 0` and `num_rows() == 0`
   *
   * @returns A vector of `unique_ptr`s to the constituent columns
   */
  std::vector<std::unique_ptr<column>> release();

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
    std::transform(
      begin, end, columns.begin(), [this](auto index) { return _columns.at(index)->view(); });
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
  [[nodiscard]] table_view select(std::vector<cudf::size_type> const& column_indices) const
  {
    return select(column_indices.begin(), column_indices.end());
  };

  /**
   * @brief Returns a reference to the specified column
   *
   * @throws std::out_of_range
   * If i is out of the range [0, num_columns)
   *
   * @param column_index Index of the desired column
   * @return A reference to the desired column
   */
  column& get_column(cudf::size_type column_index) { return *(_columns.at(column_index)); }

  /**
   * @brief Returns a const reference to the specified column
   *
   * @throws std::out_of_range
   * If i is out of the range [0, num_columns)
   *
   * @param i Index of the desired column
   * @return A const reference to the desired column
   */
  [[nodiscard]] column const& get_column(cudf::size_type i) const { return *(_columns.at(i)); }

 private:
  std::vector<std::unique_ptr<column>> _columns{};
  size_type _num_rows{};
};

}  // namespace CUDF_EXPORT cudf

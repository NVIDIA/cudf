/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/column/column_view.hpp>
#include <cudf/table/table_view.hpp>
#include <cudf/utilities/export.hpp>

#include <memory>
#include <vector>

namespace cudf {
namespace detail {

struct linked_column_view;

using LinkedColPtr    = std::shared_ptr<linked_column_view>;
using LinkedColVector = std::vector<LinkedColPtr>;

/**
 * @brief A column_view class with pointer to parent's column_view
 */
struct linked_column_view : public column_view_base {
  linked_column_view(linked_column_view const&)            = delete;
  linked_column_view& operator=(linked_column_view const&) = delete;

  /**
   * @brief Construct from column_view
   *
   * @param col column_view to wrap
   */
  linked_column_view(column_view const& col);

  /**
   * @brief Construct from column_view with its parent
   *
   * @param parent Pointer to the column_view's parent column_view
   * @param col column_view to wrap
   */
  linked_column_view(linked_column_view* parent, column_view const& col);

  /**
   * @brief Conversion operator to cast this instance to its column_view
   */
  operator column_view() const;

  linked_column_view* parent;  ///< Pointer to parent of this column; nullptr if root
  LinkedColVector children;    ///< Vector of children of this instance
};

/**
 * @brief Converts all column_views of a table into linked_column_views
 *
 * @param table table of columns to convert
 * @return Vector of converted linked_column_views
 */
LinkedColVector table_to_linked_columns(table_view const& table);

}  // namespace detail
}  // namespace cudf
